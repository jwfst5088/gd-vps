//! In-memory tables, per-table transition log, and long-poll signalling.

use crate::domain::{
    Expect, NextStateBody, PlayerPresence, PlayerRecord, PlayerType, Seat, StateTransition,
    TableRuntimeState, TableState, TableStatus, iso_timestamp, snapshot_replace_delta,
};
use crate::error::AppError;
use crate::game::card::{parse_card_symbol, is_wild, level_order_value, RuleContext};
use crate::game::engine::{GameEngine, PlayerAction};
use crate::game::rules::narration::{
    format_big_play, format_game_end_by_leave, format_hand_end, format_hand_open,
    format_hand_open_with_tribute_canceled, format_rank_announce, format_tribute_action,
    is_big_play_combination,
};
use crate::game::rules::scoring::{Level, ScoringService, WinType};
use crate::game::types::{
    GameConfig, GamePhase, HandCommitMeta, HandState, HistoryActionKind, TeamId,
};
use crate::learning::game_logger::{GameLogEntry, GameAction, log_game, seat_to_team_str, team_to_str};
use crate::prompt::prompt_builder::{build_observer_prompt, build_player_prompt};
use serde_json::json;
use std::collections::HashMap;
use std::process::Command as StdCommand;
use std::sync::Arc;
use tokio::sync::{Mutex, Notify};
use tokio::time::{Duration, sleep};

const PLAYER_INACTIVITY_TIMEOUT: Duration = Duration::from_secs(30 * 60);

/// If a bot is the current actor and hasn't acted within this time, auto-pass.
const BOT_TURN_TIMEOUT: Duration = Duration::from_secs(30);

/// First segment of hyphenated v4 UUID (8 hex chars before the first `-`).
fn uuid_short_fragment() -> String {
    let s = uuid::Uuid::new_v4().to_string();
    s.split('-')
        .next()
        .expect("uuid string is hyphenated")
        .to_string()
}

#[derive(Clone)]
pub struct TableStore {
    tables: Arc<Mutex<HashMap<String, Arc<TableMutex>>>>,
}

type TableMutex = Mutex<TableInner>;

struct LogEntry {
    transition: StateTransition,
    /// `expect` after applying this transition (per design: client applies delta then reads expect).
    expect_after: Expect,
}

struct TableInner {
    state: TableRuntimeState,
    /// `log[i]` has `transition.seq == i + 1`.
    log: Vec<LogEntry>,
    /// Shared so `nextstate` can await notifications without holding the table mutex.
    notify: Arc<Notify>,
    /// When the current bot's turn started (separate from last_activity_at so
    /// observer polling doesn't reset the timeout). None when not a bot's turn.
    bot_turn_started_at: Option<std::time::Instant>,
    /// PIDs of bot processes spawned for this table (for reliable cleanup).
    bot_pids: Vec<u32>,
}

impl TableStore {
    pub fn new() -> Self {
        Self {
            tables: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    async fn table_mutex(&self, table_id: &str) -> Result<Arc<TableMutex>, AppError> {
        let g = self.tables.lock().await;
        g.get(table_id)
            .cloned()
            .ok_or_else(|| AppError::NotFound(format!("unknown table_id {}", table_id)))
    }
}

impl Default for TableStore {
    fn default() -> Self {
        Self::new()
    }
}

impl TableStore {
    pub async fn create_table(
        &self,
        table_name: Option<String>,
        start_level: Level,
    ) -> TableRuntimeState {
        let id = format!("t_{}", uuid_short_fragment());
        let state = TableRuntimeState::new_with_level(id.clone(), table_name, start_level);
        let inner = Arc::new(Mutex::new(TableInner {
            state,
            log: Vec::new(),
            notify: Arc::new(Notify::new()),
            bot_turn_started_at: None,
            bot_pids: Vec::new(),
        }));
        let mut g = self.tables.lock().await;
        g.insert(id.clone(), inner);
        drop(g);
        self.get_snapshot(&id).await.expect("just inserted")
    }

    pub async fn get_snapshot(&self, table_id: &str) -> Result<TableRuntimeState, AppError> {
        let t = self.table_mutex(table_id).await?;
        let mut inner = t.lock().await;
        Self::expire_inactive_players_locked(&mut inner)?;
        Ok(inner.state.clone())
    }

    /// All tables (materialized runtime state), sorted by `table_id` for stable output.
        /// Register a bot process PID for this table so it can be reliably killed on delete.
    pub async fn register_bot_pid(&self, table_id: &str, pid: u32) -> Result<(), AppError> {
        let arc = self.table_mutex(table_id).await?;
        let mut inner = arc.lock().await;
        inner.bot_pids.push(pid);
        eprintln!("INFO: register_bot_pid: table={table_id} pid={pid}");
        Ok(())
    }

    pub async fn delete_table(&self, table_id: &str) -> Result<(), AppError> {
        // Step 1: End the game if in progress
        if let Ok(arc) = self.table_mutex(table_id).await {
            let mut inner = arc.lock().await;
            if matches!(inner.state.status, TableStatus::InGame) {
                inner.state.status = TableStatus::Finished;
                inner.state.waiting_next_hand_ready = false;
                inner.state.narration = "table disbanded".to_string();
                if let Some(g) = inner.state.game.as_mut() {
                    g.phase = GamePhase::Completed;
                }
                inner.state.sync_phase_from_game();
                inner.state.seq += 1;
                inner.notify.notify_waiters();
            }

            // Step 2: Kill bots - first try tracked PIDs, then fallback to pkill
            let pids: Vec<u32> = inner.bot_pids.drain(..).collect();
            let table_id_for_kill = inner.state.table_id.clone();
            eprintln!("INFO: delete_table: killing {} tracked bots for table {table_id_for_kill}", pids.len());
            drop(inner);
            drop(arc);

            // Kill tracked PIDs with SIGKILL and wait for them
            for pid in &pids {
                let _ = StdCommand::new("kill")
                    .args(["-9", &pid.to_string()])
                    .output();
                // Wait for the child to be reaped
                #[cfg(unix)]
                unsafe {
                    let mut status: i32 = 0;
                    libc::waitpid(*pid as i32, &mut status, 0);
                }
                eprintln!("INFO: delete_table: killed and reaped bot pid={pid}");
            }

            // Fallback: pkill any remaining bots for this table
            let _ = StdCommand::new("pkill")
                .args(["-9", "-f", &format!("clawguandan.*bot.*-t {}", table_id_for_kill)])
                .output();
        }

        // Step 3: Remove from memory
        let mut g = self.tables.lock().await;
        g.remove(table_id)
            .map(|_| ())
            .ok_or_else(|| AppError::NotFound(format!("unknown table_id {}", table_id)))
    }

    /// Exit game -> kill bots -> destroy table
    pub async fn leave_and_destroy(
        &self,
        table_id: &str,
        player_id: &str,
        player_key: &str,
    ) -> Result<(), AppError> {
        let arc = self.table_mutex(table_id).await?;
        let mut inner = arc.lock().await;

        Self::verify_player_identity_locked(&inner, player_id, player_key)?;

        // Step 1: End game
        if matches!(inner.state.status, TableStatus::InGame) {
            let player_name = inner
                .state
                .seats
                .values()
                .find_map(|s| {
                    s.as_ref().and_then(|p| {
                        if p.player_id == player_id { Some(p.player_name.clone()) }
                        else { None }
                    })
                })
                .unwrap_or_else(|| "unknown".to_string());

            inner.state.status = TableStatus::Finished;
            inner.state.waiting_next_hand_ready = false;
            inner.state.narration = format!("{} left the game", player_name);
            if let Some(g) = inner.state.game.as_mut() {
                g.phase = GamePhase::Completed;
            }
            inner.state.sync_phase_from_game();
            inner.state.seq += 1;
            inner.notify.notify_waiters();
        }

        let bot_pids: Vec<u32> = inner.bot_pids.drain(..).collect();
        let table_id_owned = inner.state.table_id.clone();
        drop(inner);
        drop(arc);

        // Step 2: Kill bots - first tracked PIDs, then pkill fallback
        eprintln!("INFO: leave_and_destroy: killing {} tracked bots for table {}", bot_pids.len(), table_id_owned);
        for pid in &bot_pids {
            let _ = StdCommand::new("kill")
                .args(["-9", &pid.to_string()])
                .output();
            #[cfg(unix)]
            unsafe {
                let mut status: i32 = 0;
                libc::waitpid(*pid as i32, &mut status, 0);
            }
            eprintln!("INFO: leave_and_destroy: killed and reaped bot pid={pid}");
        }
        let _ = StdCommand::new("pkill")
            .args(["-9", "-f", &format!("clawguandan.*bot.*-t {}", table_id_owned)])
            .output();

        // Step 3: Destroy table
        eprintln!("INFO: leave_and_destroy: destroying table {}", table_id_owned);
        self.delete_table(&table_id_owned).await
    }

pub async fn list_table_runtimes(&self) -> Vec<TableRuntimeState> {
        let arcs: Vec<Arc<TableMutex>> = {
            let g = self.tables.lock().await;
            g.values().cloned().collect()
        };
        let mut out = Vec::with_capacity(arcs.len());
        for arc in arcs {
            let mut inner = arc.lock().await;
            let _ = Self::expire_inactive_players_locked(&mut inner);
            out.push(inner.state.clone());
        }
        out.sort_by(|a, b| a.table_id.cmp(&b.table_id));
        out
    }

    fn pick_seat(state: &TableRuntimeState, requested: SeatOrAuto) -> Result<Seat, AppError> {
        match requested {
            SeatOrAuto::Auto => Seat::ALL
                .into_iter()
                .find(|s| state.seats.get(s).and_then(|x| x.as_ref()).is_none())
                .ok_or_else(|| AppError::Conflict {
                    message: "table is full".into(),
                    code: "TABLE_FULL",
                    current_seq: Some(state.seq),
                }),
            SeatOrAuto::Fixed(seat) => {
                if state.seats.get(&seat).and_then(|x| x.as_ref()).is_some() {
                    return Err(AppError::Conflict {
                        message: format!("seat {} is occupied", seat.as_str()),
                        code: "SEAT_TAKEN",
                        current_seq: Some(state.seq),
                    });
                }
                Ok(seat)
            }
        }
    }

    pub async fn join(
        &self,
        table_id: &str,
        player_name: String,
        player_type: Option<PlayerType>,
        player_model: Option<String>,
        seat: SeatOrAuto,
    ) -> Result<(String, String, Seat, PlayerType, Option<String>), AppError> {
        let arc = self.table_mutex(table_id).await?;
        let mut inner = arc.lock().await;
        if !matches!(inner.state.status, TableStatus::Waiting) {
            return Err(AppError::Conflict {
                message: "cannot join: game already started or finished".into(),
                code: "INVALID_TABLE_STATUS",
                current_seq: Some(inner.state.seq),
            });
        }

        let seat = Self::pick_seat(&inner.state, seat)?;
        let pt = player_type.unwrap_or_default();
        let player_model = normalize_player_model(pt.clone(), player_model);
        let pid = format!("p_{}", uuid_short_fragment());
        let pkey = uuid::Uuid::new_v4().to_string();
        let prev_snapshot = inner.state.to_table_state();
        let prev_seq = inner.state.seq;

        inner.state.seats.insert(
            seat,
            Some(PlayerRecord {
                player_id: pid.clone(),
                player_key: pkey.clone(),
                player_name,
                player_type: pt.clone(),
                player_model: player_model.clone(),
                presence: PlayerPresence::Active,
                ready: false,
                last_activity_at: std::time::Instant::now(),
            }),
        );
        inner.state.seq += 1;
        let new_snapshot = inner.state.to_table_state();
        let seq = inner.state.seq;
        let expect_after = new_snapshot.expect.clone();

        let tr = build_transition(
            &prev_snapshot,
            &new_snapshot,
            prev_seq,
            seq,
            "PLAYER_JOINED",
            Some(json!({
                "actionType": "join",
                "actorPlayerId": pid,
                "seat": seat.as_str(),
            })),
        );
        inner.log.push(LogEntry {
            transition: tr,
            expect_after,
        });
        inner.notify.notify_waiters();

        Ok((pid, pkey, seat, pt, player_model))
    }

    pub async fn set_ready(
        &self,
        table_id: &str,
        player_id: &str,
        player_key: &str,
        ready: bool,
    ) -> Result<u64, AppError> {
        let arc = self.table_mutex(table_id).await?;
        let mut inner = arc.lock().await;
        Self::verify_player_identity_locked(&inner, player_id, player_key)?;
        Self::touch_player_activity_locked(&mut inner, player_id);
        Self::expire_inactive_players_locked(&mut inner)?;

        let mut found = None;
        for (seat, slot) in &inner.state.seats {
            if let Some(p) = slot.as_ref()
                && p.player_id == player_id
            {
                found = Some((*seat, p.ready));
                break;
            }
        }

        let Some((_seat, was_ready)) = found else {
            return Err(AppError::Forbidden(
                "player is not seated at this table".into(),
            ));
        };

        // Idempotent: no transition or notify.
        if was_ready == ready {
            return Ok(inner.state.seq);
        }

        for (_seat, slot) in &mut inner.state.seats {
            if let Some(p) = slot.as_mut()
                && p.player_id == player_id
            {
                p.ready = ready;
                break;
            }
        }

        let prev_snapshot = inner.state.to_table_state();
        let prev_seq = inner.state.seq;

        let will_start_first_hand =
            inner.state.all_ready() && matches!(inner.state.status, TableStatus::Waiting);
        let will_start_next_hand = inner.state.all_ready()
            && matches!(inner.state.status, TableStatus::InGame)
            && inner.state.waiting_next_hand_ready;
        if will_start_first_hand {
            inner.state.status = TableStatus::InGame;
            inner.state.game_config = GameConfig {
                rng_seed: TableRuntimeState::hash_table_id_seed(&inner.state.table_id),
                randomize_deals: true, // 房规（用户 2026-09-03）：真实牌桌每局真随机
            };
            let first_hand_level = match inner.state.current_declarer {
                TeamId::Ew => inner.state.team_progress_ew.level,
                TeamId::Sn => inner.state.team_progress_sn.level,
            };
            let engine = GameEngine::new(inner.state.game_config.clone());
            let mut gs = engine.init_table(inner.state.table_id.clone());
            engine
                .start_first_hand(&mut gs, Seat::E, first_hand_level)
                .expect("start_first_hand should not fail");
            inner.state.game = Some(gs);
            inner.state.sync_phase_from_game();
            inner.state.waiting_next_hand_ready = false;
            inner.state.narration =
                format_hand_open(inner.state.current_declarer, first_hand_level.as_api_str());
        } else if will_start_next_hand {
            let seq = inner.state.seq;
            let declarer = inner.state.current_declarer;
            let next_hand_level = match declarer {
                TeamId::Ew => inner.state.team_progress_ew.level,
                TeamId::Sn => inner.state.team_progress_sn.level,
            };
            let finishing_order = inner.state.last_finishing_order.clone();
            let engine = GameEngine::new(inner.state.game_config.clone());
            let canceled_opening_lead = {
                let game = inner
                    .state
                    .game
                    .as_mut()
                    .ok_or_else(|| AppError::Conflict {
                        message: "game state not initialized".into(),
                        code: "INVALID_TABLE_STATUS",
                        current_seq: Some(seq),
                    })?;
                engine
                    .start_next_hand_with_tribute(game, declarer, next_hand_level, &finishing_order)
                    .map_err(|msg| map_engine_error(msg, seq))?;
                game.hand
                    .as_ref()
                    .and_then(|h| h.tribute.as_ref())
                    .and_then(|t| {
                        if t.canceled {
                            Some(game.turn_seat)
                        } else {
                            None
                        }
                    })
            };
            inner.state.sync_phase_from_game();
            inner.state.waiting_next_hand_ready = false;
            let level_s = next_hand_level.as_api_str();
            if let Some(lead) = canceled_opening_lead {
                inner.state.narration = format_hand_open_with_tribute_canceled(
                    declarer,
                    level_s,
                    &player_name_for_seat(&inner.state, lead),
                );
            } else {
                inner.state.narration = format_hand_open(declarer, level_s);
            }
        }

        inner.state.seq += 1;
        let new_snapshot = inner.state.to_table_state();
        let new_seq = inner.state.seq;
        let expect_after = new_snapshot.expect.clone();

        let transition_type = if will_start_first_hand {
            "GAME_STARTED"
        } else if will_start_next_hand {
            "NEXT_HAND_STARTED"
        } else {
            "PLAYER_READY_CHANGED"
        };

        let tr = build_transition(
            &prev_snapshot,
            &new_snapshot,
            prev_seq,
            new_seq,
            transition_type,
            Some(json!({
                "actionType": "ready",
                "actorPlayerId": player_id,
                "ready": ready,
                "gameStarted": will_start_first_hand,
                "nextHandStarted": will_start_next_hand,
            })),
        );
        inner.log.push(LogEntry {
            transition: tr,
            expect_after,
        });

        // If we auto-started game, we still only emitted one transition (merged).
        inner.notify.notify_waiters();

        Ok(new_seq)
    }

    fn apply_action_locked(
        inner: &mut TableInner,
        player_id: &str,
        player_key: &str,
        client_seq: u64,
        action_type: &'static str,
        event_payload: serde_json::Value,
    ) -> Result<u64, AppError> {
        Self::verify_player_identity_locked(inner, player_id, player_key)?;
        Self::touch_player_activity_locked(inner, player_id);
        let _ = Self::expire_bot_turns_locked(inner);
        Self::expire_inactive_players_locked(inner)?;
        if client_seq != inner.state.seq {
            return Err(AppError::Conflict {
                message: format!(
                    "stale seq: expected {}, got {}",
                    inner.state.seq, client_seq
                ),
                code: "STALE_SEQ",
                current_seq: Some(inner.state.seq),
            });
        }
        if !matches!(inner.state.status, TableStatus::InGame) {
            return Err(AppError::Conflict {
                message: "action is only allowed when table is in_game".into(),
                code: "INVALID_TABLE_STATUS",
                current_seq: Some(inner.state.seq),
            });
        }
        let seat = inner
            .state
            .seat_for_player(player_id)
            .ok_or_else(|| AppError::Forbidden("player is not seated at this table".into()))?;

        let action = parse_player_action(action_type, &event_payload)?;

        let prev_snapshot = inner.state.to_table_state();
        let prev_game = inner.state.game.clone();
        let prev_seq = inner.state.seq;
        let seq = inner.state.seq;
        let playing_commit = matches!(
            inner.state.game.as_ref().map(|g| g.phase),
            Some(GamePhase::Playing)
        )
        .then(|| HandCommitMeta {
            seq: seq + 1,
            timestamp: iso_timestamp(),
        });

        let engine = GameEngine::new(inner.state.game_config.clone());
        let game = inner
            .state
            .game
            .as_mut()
            .ok_or_else(|| AppError::Conflict {
                message: "game state not initialized".into(),
                code: "INVALID_TABLE_STATUS",
                current_seq: Some(seq),
            })?;
        engine
            .apply_player_action(game, seat, action, playing_commit)
            .map_err(|msg| map_engine_error(msg, seq))?;
        inner.state.sync_phase_from_game();
        inner.state.narration =
            build_action_narration(&inner.state, prev_game.as_ref(), action_type);

        inner.state.seq += 1;
        let new_snapshot = inner.state.to_table_state();
        let new_seq = inner.state.seq;
        let expect_after = new_snapshot.expect.clone();

        let logged_payload = normalized_event_payload(
            action_type,
            &event_payload,
            inner.state.game.as_ref(),
            new_seq,
        );
        let tr = build_transition(
            &prev_snapshot,
            &new_snapshot,
            prev_seq,
            new_seq,
            "ACTION_APPLIED",
            Some(json!({
                "actionType": action_type,
                "actorPlayerId": player_id,
                "payload": logged_payload
            })),
        );
        inner.log.push(LogEntry {
            transition: tr,
            expect_after,
        });
        inner.notify.notify_waiters();

        // Clear bot turn started marker when action succeeds (turn passes to next player)
        inner.bot_turn_started_at = None;

        // If hand enters scoring, apply scoring and switch to re-ready flow.
        Self::settle_scoring_and_wait_ready(inner)?;

        Ok(inner.state.seq)
    }

    fn settle_scoring_and_wait_ready(inner: &mut TableInner) -> Result<(), AppError> {
        if !matches!(inner.state.status, TableStatus::InGame) {
            return Ok(());
        }

        let seq = inner.state.seq;
        if inner.state.waiting_next_hand_ready {
            return Ok(());
        }
        let game = match inner.state.game.as_ref() {
            Some(g) => g,
            None => {
                return Ok(());
            }
        };
        if game.phase != GamePhase::Scoring {
            return Ok(());
        }
        let winner = game.winner_team.ok_or_else(|| AppError::Conflict {
            message: "winner team missing when entering scoring".into(),
            code: "INVALID_TABLE_STATUS",
            current_seq: Some(seq),
        })?;
        let completed_order = {
            let hand = game.hand.as_ref().ok_or_else(|| AppError::Conflict {
                message: "hand missing when entering scoring".into(),
                code: "INVALID_TABLE_STATUS",
                current_seq: Some(seq),
            })?;
            complete_finishing_order(hand)
        };

        let prev_snapshot = inner.state.to_table_state();
        let prev_seq = inner.state.seq;
        let win_type = infer_win_type(&completed_order, winner);
        let outcome = ScoringService::apply_hand(
            inner.state.team_progress_ew.clone(),
            inner.state.team_progress_sn.clone(),
            inner.state.current_declarer,
            winner,
            win_type,
            true,
        )
        .map_err(|msg| AppError::Conflict {
            message: msg,
            code: "INVALID_SCORING",
            current_seq: Some(seq),
        })?;

        inner.state.team_progress_ew = outcome.progress_ew;
        inner.state.team_progress_sn = outcome.progress_sn;
        inner.state.current_declarer = outcome.next_declarer;
        inner.state.last_finishing_order = completed_order.clone();
        reset_all_players_ready(&mut inner.state);

        let ew_level = inner.state.team_progress_ew.level.as_api_str();
        let sn_level = inner.state.team_progress_sn.level.as_api_str();
        let finish_names = completed_order
            .iter()
            .map(|seat| player_name_for_seat(&inner.state, *seat))
            .collect::<Vec<_>>();
        if outcome.winner_team.is_some() {
            inner.state.status = TableStatus::Finished;
            inner.state.waiting_next_hand_ready = false;
            inner.state.game_winner_team_id = outcome.winner_team;
            inner.state.narration = format_hand_end(
                &finish_names,
                ew_level,
                sn_level,
                false,
                true,
                outcome.winner_team,
            );
            if let Some(g) = inner.state.game.as_mut() {
                g.phase = GamePhase::Completed;
            }
            inner.state.sync_phase_from_game();
            
            Self::log_game_result(inner, &completed_order, winner);
        } else {
            inner.state.waiting_next_hand_ready = true;
            inner.state.narration =
                format_hand_end(&finish_names, ew_level, sn_level, true, false, None);
        }

        inner.state.seq += 1;
        let new_snapshot = inner.state.to_table_state();
        let new_seq = inner.state.seq;
        let expect_after = new_snapshot.expect.clone();

        let transition_type = if matches!(inner.state.status, TableStatus::Finished) {
            "GAME_COMPLETED"
        } else {
            "HAND_ENDED_WAITING_READY"
        };
        let tr = build_transition(
            &prev_snapshot,
            &new_snapshot,
            prev_seq,
            new_seq,
            transition_type,
            Some(json!({
                "actionType": "hand_end",
                "winnerTeamId": winner.as_str(),
                "finishingOrder": completed_order.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            })),
        );
        inner.log.push(LogEntry {
            transition: tr,
            expect_after,
        });
        inner.notify.notify_waiters();
        Ok(())
    }

    pub async fn apply_action(
        &self,
        table_id: &str,
        player_id: &str,
        player_key: &str,
        client_seq: u64,
        action_type: &'static str,
        event_payload: serde_json::Value,
    ) -> Result<u64, AppError> {
        let arc = self.table_mutex(table_id).await?;
        let mut inner = arc.lock().await;
        Self::apply_action_locked(
            &mut inner,
            player_id,
            player_key,
            client_seq,
            action_type,
            event_payload,
        )
    }

    /// Fetch transition `since_seq + 1`, waiting if needed. Returns `None` on timeout (204).
    pub async fn next_state(
        &self,
        table_id: &str,
        since_seq: u64,
        timeout: Option<Duration>,
    ) -> Result<Option<NextStateBody>, AppError> {
        let arc = self.table_mutex(table_id).await?;
        let timeout = timeout.unwrap_or(Duration::from_secs(60));

        loop {
            let notify = {
                let mut inner = arc.lock().await;
                Self::expire_inactive_players_locked(&mut inner)?;
                if since_seq > inner.state.seq {
                    return Err(AppError::BadRequest(format!(
                        "sinceSeq {} is ahead of currentSeq {}",
                        since_seq, inner.state.seq
                    )));
                }
                if since_seq < inner.state.seq {
                    let idx = since_seq as usize;
                    let entry = inner.log.get(idx).ok_or_else(|| {
                        AppError::BadRequest("internal: missing transition for seq".into())
                    })?;
                    let tr = entry.transition.clone();
                    let tr_seq = tr.seq;
                    let expect = entry.expect_after.clone();
                    let prompt = None;
                    let lag = inner.state.seq.saturating_sub(tr_seq);
                    return Ok(Some(NextStateBody {
                        transition: tr,
                        lag,
                        expect,
                        private: None,
                        prompt,
                    }));
                }
                // At head: no further transitions will arrive after the table ends.
                if matches!(inner.state.status, TableStatus::Finished) {
                    return Ok(None);
                }
                // since_seq == current: subscribe then release lock before awaiting.
                inner.notify.clone()
            };

            let wait = notify.notified();

            tokio::select! {
                _ = wait => { continue; }
                _ = sleep(timeout) => { return Ok(None); }
            }
        }
    }

    pub async fn next_state_with_prompt(
        &self,
        table_id: &str,
        since_seq: u64,
        player_id: Option<&str>,
        player_key: Option<&str>,
        timeout: Option<Duration>,
    ) -> Result<Option<NextStateBody>, AppError> {
        if let Some(pid) = player_id {
            let key = player_key.ok_or_else(|| {
                AppError::BadRequest("playerKey is required with playerId".into())
            })?;
            self.verify_player_identity(table_id, pid, key).await?;
            self.touch_player_activity(table_id, pid).await?;
        }
        let body = self.next_state(table_id, since_seq, timeout).await?;
        let Some(mut body) = body else {
            if let Some(pid) = player_id {
                self.touch_player_activity(table_id, pid).await?;
            }
            return Ok(None);
        };

        if let Some(pid) = player_id {
            let snap = self.get_snapshot(table_id).await?;
            let mine_ready = snap
                .seats
                .values()
                .flatten()
                .find(|p| p.player_id == pid)
                .map(|p| p.ready);
            let is_actor = body.expect.actor_player_ids.iter().any(|id| id == pid);
            body.prompt = Some(build_player_prompt(&body.expect, mine_ready, is_actor));
            body.private = snap.private_view_for_player(pid);
        } else {
            // Observer: read-only prompt
            body.prompt = Some(build_observer_prompt(&body.expect));
        }

        if let Some(pid) = player_id {
            self.touch_player_activity(table_id, pid).await?;
        }
        Ok(Some(body))
    }

    pub async fn touch_player_activity(
        &self,
        table_id: &str,
        player_id: &str,
    ) -> Result<(), AppError> {
        let arc = self.table_mutex(table_id).await?;
        let mut inner = arc.lock().await;
        Self::touch_player_activity_locked(&mut inner, player_id);
        Self::expire_inactive_players_locked(&mut inner)?;
        Ok(())
    }

    pub async fn verify_player_identity(
        &self,
        table_id: &str,
        player_id: &str,
        player_key: &str,
    ) -> Result<(), AppError> {
        let arc = self.table_mutex(table_id).await?;
        let mut inner = arc.lock().await;
        Self::expire_inactive_players_locked(&mut inner)?;
        Self::verify_player_identity_locked(&inner, player_id, player_key)
    }

    fn touch_player_activity_locked(inner: &mut TableInner, player_id: &str) -> bool {
        let now = std::time::Instant::now();
        for slot in inner.state.seats.values_mut() {
            if let Some(player) = slot.as_mut()
                && player.player_id == player_id
            {
                player.last_activity_at = now;
                return true;
            }
        }
        false
    }

    fn verify_player_identity_locked(
        inner: &TableInner,
        player_id: &str,
        player_key: &str,
    ) -> Result<(), AppError> {
        let Some(seat) = inner.state.seat_for_player(player_id) else {
            return Err(AppError::Forbidden(
                "playerId is not seated at this table".into(),
            ));
        };
        let Some(player) = inner.state.seats.get(&seat).and_then(|p| p.as_ref()) else {
            return Err(AppError::Forbidden(
                "playerId is not seated at this table".into(),
            ));
        };
        if player.player_key != player_key {
            return Err(AppError::Forbidden("invalid playerKey for playerId".into()));
        }
        Ok(())
    }

    /// Auto-pass for bots that have exceeded their turn deadline.
    /// Uses bot_turn_started_at (NOT last_activity_at) to avoid observer polling resetting the timer.
    fn expire_bot_turns_locked(inner: &mut TableInner) -> Result<bool, AppError> {
        if !matches!(inner.state.status, TableStatus::InGame) {
            return Ok(false);
        }
        let game_phase = inner.state.game.as_ref().map(|g| g.phase);
        // Handle Playing, Tribute, and Exchange phases
        if !matches!(game_phase, Some(GamePhase::Playing | GamePhase::Tribute | GamePhase::Exchange)) {
            return Ok(false);
        }
        // Determine the actor seat for the current phase
        let actor_seat = match game_phase {
            Some(GamePhase::Playing) => inner.state.game.as_ref().map(|g| g.turn_seat),
            Some(GamePhase::Tribute) => inner.state.game.as_ref().and_then(|g| {
                g.hand.as_ref().and_then(|h| h.next_tribute_actor())
            }),
            Some(GamePhase::Exchange) => inner.state.game.as_ref().and_then(|g| {
                g.hand.as_ref().and_then(|h| h.next_exchange_actor())
            }),
            _ => None,
        };
        let actor_seat = match actor_seat {
            Some(s) => s,
            None => return Ok(false),
        };
        let actor_player = match inner.state.seats.get(&actor_seat).and_then(|p| p.as_ref()) {
            Some(p) => p,
            None => return Ok(false),
        };
        if !matches!(actor_player.player_type, PlayerType::Bot) {
            // Not a bot: clear the turn started marker
            inner.bot_turn_started_at = None;
            return Ok(false);
        }
        // Set bot_turn_started_at on first check if not set
        if inner.bot_turn_started_at.is_none() {
            inner.bot_turn_started_at = Some(std::time::Instant::now());
            return Ok(false);
        }
        let now = std::time::Instant::now();
        let started = inner.bot_turn_started_at.unwrap();
        if now.duration_since(started) <= BOT_TURN_TIMEOUT {
            return Ok(false);
        }

        // Bot has timed out: clear the marker and auto-pass / auto-tribute / auto-return
        inner.bot_turn_started_at = None;
        let player_id = actor_player.player_id.clone();
        let player_name = actor_player.player_name.clone();
        let seq = inner.state.seq;

        // Determine the correct action based on game phase
        let game_phase = game_phase; // already captured above
        let (action, action_label) = match game_phase {
            Some(GamePhase::Tribute) => {
                // Auto-tribute: find the highest non-wild single card
                let game = inner.state.game.as_ref().ok_or_else(|| AppError::Conflict {
                    message: "game state not initialized".into(),
                    code: "INVALID_TABLE_STATUS",
                    current_seq: Some(seq),
                })?;
                let hand = game.hand.as_ref().ok_or_else(|| AppError::Conflict {
                    message: "hand state not initialized".into(),
                    code: "INVALID_TABLE_STATUS",
                    current_seq: Some(seq),
                })?;
                let ctx = RuleContext { hand_level: hand.hand_level };
                let cards = hand.hands.get(&actor_seat).ok_or_else(|| AppError::Conflict {
                    message: "actor seat missing from hand".into(),
                    code: "INVALID_TABLE_STATUS",
                    current_seq: Some(seq),
                })?;
                let highest = cards.iter()
                    .filter(|c| {
                        parse_card_symbol(c).ok().map_or(false, |p| !is_wild(p, ctx))
                    })
                    .max_by_key(|c| {
                        parse_card_symbol(c).ok().map(|p| level_order_value(p, ctx)).unwrap_or(0)
                    })
                    .ok_or_else(|| AppError::BadRequest("no valid tribute card in bot hand".into()))?
                    .clone();
                (PlayerAction::Tribute { card: highest }, "tribute")
            }
            Some(GamePhase::Exchange) => {
                // Auto-return: find a card with different rank than the received tribute card
                let game = inner.state.game.as_ref().ok_or_else(|| AppError::Conflict {
                    message: "game state not initialized".into(),
                    code: "INVALID_TABLE_STATUS",
                    current_seq: Some(seq),
                })?;
                let hand = game.hand.as_ref().ok_or_else(|| AppError::Conflict {
                    message: "hand state not initialized".into(),
                    code: "INVALID_TABLE_STATUS",
                    current_seq: Some(seq),
                })?;
                let tribute = hand.tribute.as_ref().ok_or_else(|| AppError::Conflict {
                    message: "tribute plan not found".into(),
                    code: "INVALID_TABLE_STATUS",
                    current_seq: Some(seq),
                })?;
                let pair = tribute.pairs.iter()
                    .find(|p| p.receiver == actor_seat && p.return_card.is_none())
                    .ok_or_else(|| AppError::Conflict {
                        message: "player is not expected to return card".into(),
                        code: "INVALID_TABLE_STATUS",
                        current_seq: Some(seq),
                    })?;
                let paid_card = pair.paid_card.as_ref().ok_or_else(|| AppError::Conflict {
                    message: "tribute not yet paid".into(),
                    code: "INVALID_TABLE_STATUS",
                    current_seq: Some(seq),
                })?;
                let paid_parsed = parse_card_symbol(paid_card).map_err(|e| {
                    AppError::BadRequest(format!("invalid paid_card symbol: {e}"))
                })?;
                let ctx = RuleContext { hand_level: hand.hand_level };
                let cards = hand.hands.get(&actor_seat).ok_or_else(|| AppError::Conflict {
                    message: "actor seat missing from hand".into(),
                    code: "INVALID_TABLE_STATUS",
                    current_seq: Some(seq),
                })?;
                let paid_val = level_order_value(paid_parsed, ctx);
                let return_card = cards.iter()
                    .find(|c| {
                        parse_card_symbol(c).ok().map_or(false, |p| {
                            level_order_value(p, ctx) != paid_val && !is_wild(p, ctx)
                        })
                    })
                    .cloned()
                    .unwrap_or_else(|| cards[0].clone());
                (PlayerAction::ReturnCard { card: return_card }, "return_card")
            }
            _ => {
                // Playing phase: auto-pass
                (PlayerAction::Pass, "pass")
            }
        };

        let playing_commit = matches!(game_phase, Some(GamePhase::Playing))
            .then(|| HandCommitMeta {
                seq: seq + 1,
                timestamp: iso_timestamp(),
            });

        let prev_snapshot = inner.state.to_table_state();
        let prev_seq = inner.state.seq;

        let engine = GameEngine::new(inner.state.game_config.clone());
        let game = inner.state.game.as_mut().ok_or_else(|| AppError::Conflict {
            message: "game state not initialized".into(),
            code: "INVALID_TABLE_STATUS",
            current_seq: Some(seq),
        })?;
        engine
            .apply_player_action(game, actor_seat, action, playing_commit)
            .map_err(|msg| map_engine_error(msg, seq))?;

        inner.state.sync_phase_from_game();
        inner.state.narration = match action_label {
            "tribute" => format!(
                "{} (bot) auto-tributed (timed out after {}s)",
                player_name,
                BOT_TURN_TIMEOUT.as_secs()
            ),
            "return_card" => format!(
                "{} (bot) auto-returned (timed out after {}s)",
                player_name,
                BOT_TURN_TIMEOUT.as_secs()
            ),
            _ => format!(
                "{} (bot) auto-passed (timed out after {}s)",
                player_name,
                BOT_TURN_TIMEOUT.as_secs()
            ),
        };
        inner.state.seq += 1;

        let new_snapshot = inner.state.to_table_state();
        let new_seq = inner.state.seq;
        let expect_after = new_snapshot.expect.clone();

        let tr = build_transition(
            &prev_snapshot,
            &new_snapshot,
            prev_seq,
            new_seq,
            "BOT_AUTO_PASS",
            Some(json!({
                "actionType": "pass",
                "actorPlayerId": player_id,
                "reason": "bot_turn_timeout",
                "payload": {}
            })),
        );
        inner.log.push(LogEntry {
            transition: tr,
            expect_after,
        });
        inner.notify.notify_waiters();

        // If hand enters scoring, apply scoring
        Self::settle_scoring_and_wait_ready(inner)?;

        Ok(true)
    }

    fn expire_inactive_players_locked(inner: &mut TableInner) -> Result<(), AppError> {
        if matches!(inner.state.status, TableStatus::Finished) {
            return Ok(());
        }
        // Check bot turn timeout first (shorter timeout than player inactivity)
        if Self::expire_bot_turns_locked(inner)? {
            if matches!(inner.state.status, TableStatus::Finished) {
                return Ok(());
            }
        }
        let now = std::time::Instant::now();
        let mut away_player_ids = Vec::new();
        let mut away_player_names = Vec::new();
        for slot in inner.state.seats.values_mut() {
            if let Some(player) = slot.as_mut() {
                if player.presence == PlayerPresence::Away {
                    continue;
                }
                if now.duration_since(player.last_activity_at) > PLAYER_INACTIVITY_TIMEOUT {
                    player.presence = PlayerPresence::Away;
                    away_player_ids.push(player.player_id.clone());
                    away_player_names.push(player.player_name.clone());
                }
            }
        }
        if away_player_ids.is_empty() {
            return Ok(());
        }

        let prev_snapshot = inner.state.to_table_state();
        let prev_seq = inner.state.seq;
        let should_finish_game = matches!(inner.state.status, TableStatus::InGame);
        if should_finish_game {
            inner.state.status = TableStatus::Finished;
            inner.state.waiting_next_hand_ready = false;
            inner.state.narration = format_game_end_by_leave(&away_player_names);
            if let Some(g) = inner.state.game.as_mut() {
                g.phase = GamePhase::Completed;
            }
            inner.state.sync_phase_from_game();
        }

        inner.state.seq += 1;
        let new_snapshot = inner.state.to_table_state();
        let new_seq = inner.state.seq;
        let expect_after = new_snapshot.expect.clone();
        let transition_type = if should_finish_game {
            "PLAYER_AWAY_GAME_ENDED"
        } else {
            "PLAYER_MARKED_AWAY"
        };
        let tr = build_transition(
            &prev_snapshot,
            &new_snapshot,
            prev_seq,
            new_seq,
            transition_type,
            Some(json!({
                "actionType": "player_timeout",
                "awayPlayerIds": away_player_ids,
                "gameEnded": should_finish_game,
            })),
        );
        inner.log.push(LogEntry {
            transition: tr,
            expect_after,
        });
        inner.notify.notify_waiters();
        Ok(())
    }
}

fn build_action_narration(
    state: &TableRuntimeState,
    prev_game: Option<&crate::game::types::TableGameState>,
    action_type: &'static str,
) -> String {
    if action_type == "play" {
        return build_play_narration(state, prev_game);
    }
    let Some(game) = state.game.as_ref() else {
        return String::new();
    };
    let Some(hand) = game.hand.as_ref() else {
        return String::new();
    };
    let Some(tribute) = hand.tribute.as_ref() else {
        return String::new();
    };
    let prev_pairs = prev_game
        .and_then(|g| g.hand.as_ref())
        .and_then(|h| h.tribute.as_ref())
        .map(|t| &t.pairs);

    for pair in &tribute.pairs {
        let prev_pair = prev_pairs.and_then(|pairs| {
            pairs
                .iter()
                .find(|x| x.payer == pair.payer && x.receiver == pair.receiver)
        });
        if action_type == "tribute" {
            let changed = pair.paid_card.is_some()
                && prev_pair.and_then(|p| p.paid_card.as_deref()) != pair.paid_card.as_deref();
            if changed && let Some(card) = pair.paid_card.as_deref() {
                return format_tribute_action(
                    &player_name_for_seat(state, pair.payer),
                    card,
                    &player_name_for_seat(state, pair.receiver),
                    false,
                );
            }
        } else if action_type == "return_card" {
            let changed = pair.return_card.is_some()
                && prev_pair.and_then(|p| p.return_card.as_deref()) != pair.return_card.as_deref();
            if changed && let Some(card) = pair.return_card.as_deref() {
                return format_tribute_action(
                    &player_name_for_seat(state, pair.receiver),
                    card,
                    &player_name_for_seat(state, pair.payer),
                    true,
                );
            }
        }
    }
    String::new()
}

fn build_play_narration(
    state: &TableRuntimeState,
    prev_game: Option<&crate::game::types::TableGameState>,
) -> String {
    let Some(game) = state.game.as_ref() else {
        return String::new();
    };
    let Some(hand) = game.hand.as_ref() else {
        return String::new();
    };
    let Some(last) = hand.history.last() else {
        return String::new();
    };
    if last.action_type != HistoryActionKind::Play {
        return String::new();
    }
    let Some(comb) = last.combination_type.as_deref() else {
        return String::new();
    };
    if !is_big_play_combination(comb) {
        let prev_finishing_len = prev_game
            .and_then(|g| g.hand.as_ref())
            .map(|h| h.finishing_order.len())
            .unwrap_or(0);
        if hand.finishing_order.len() > prev_finishing_len {
            let rank = hand.finishing_order.len();
            if rank <= 2
                && let Some(seat) = hand.finishing_order.last().copied()
            {
                return format_rank_announce(&player_name_for_seat(state, seat), rank);
            }
        }
        return String::new();
    }
    format_big_play(&player_name_for_seat(state, last.seat), comb)
}

fn player_name_for_seat(state: &TableRuntimeState, seat: Seat) -> String {
    state
        .seats
        .get(&seat)
        .and_then(|o| o.as_ref())
        .map(|p| p.player_name.clone())
        .unwrap_or_else(|| seat.as_str().to_string())
}

fn reset_all_players_ready(state: &mut TableRuntimeState) {
    for slot in state.seats.values_mut() {
        if let Some(p) = slot.as_mut() {
            p.ready = false;
        }
    }
}

fn complete_finishing_order(hand: &HandState) -> Vec<Seat> {
    let mut order: Vec<Seat> = hand.finishing_order.clone();
    for seat in Seat::ALL {
        if !order.contains(&seat) {
            order.push(seat);
        }
    }
    order
}

fn seat_team(seat: Seat) -> TeamId {
    match seat {
        Seat::E | Seat::W => TeamId::Ew,
        Seat::S | Seat::N => TeamId::Sn,
    }
}

fn infer_win_type(order: &[Seat], winner: TeamId) -> WinType {
    if order.len() >= 2 && seat_team(order[0]) == winner && seat_team(order[1]) == winner {
        return WinType::OneTwo;
    }
    if order.len() >= 3 && seat_team(order[0]) == winner && seat_team(order[2]) == winner {
        return WinType::OneThree;
    }
    WinType::OneFour
}

#[derive(Clone, Copy)]
pub enum SeatOrAuto {
    Auto,
    Fixed(Seat),
}

impl SeatOrAuto {
    pub fn parse(s: &str) -> Result<Self, AppError> {
        match s {
            "auto" => Ok(SeatOrAuto::Auto),
            "E" => Ok(SeatOrAuto::Fixed(Seat::E)),
            "S" => Ok(SeatOrAuto::Fixed(Seat::S)),
            "W" => Ok(SeatOrAuto::Fixed(Seat::W)),
            "N" => Ok(SeatOrAuto::Fixed(Seat::N)),
            _ => Err(AppError::BadRequest(format!("invalid seat {:?}", s))),
        }
    }
}

fn map_engine_error(message: String, current_seq: u64) -> AppError {
    let code: &'static str = if message.contains("wrong turn") {
        "WRONG_TURN"
    } else if message.contains("not allowed in current phase")
        || message.contains("not expected to tribute")
        || message.contains("not expected to return")
    {
        "INVALID_PHASE_ACTION"
    } else {
        "ILLEGAL_ACTION"
    };
    AppError::Unprocessable {
        message,
        code,
        current_seq: Some(current_seq),
    }
}

fn parse_player_action(
    action_type: &'static str,
    payload: &serde_json::Value,
) -> Result<PlayerAction, AppError> {
    match action_type {
        "tribute" => {
            let card = payload
                .get("card")
                .and_then(|v| v.as_str())
                .ok_or_else(|| AppError::BadRequest("missing card".into()))?
                .to_string();
            Ok(PlayerAction::Tribute { card })
        }
        "return_card" => {
            let card = payload
                .get("card")
                .and_then(|v| v.as_str())
                .ok_or_else(|| AppError::BadRequest("missing card".into()))?
                .to_string();
            Ok(PlayerAction::ReturnCard { card })
        }
        "play" => {
            let cards: Vec<String> = serde_json::from_value(
                payload
                    .get("cards")
                    .cloned()
                    .ok_or_else(|| AppError::BadRequest("missing cards".into()))?,
            )
            .map_err(|e| AppError::BadRequest(format!("cards: {}", e)))?;
            let wild_targets = payload
                .get("declaredWildMapping")
                .and_then(|v| v.get("wildTargets"))
                .and_then(|v| serde_json::from_value::<Vec<String>>(v.clone()).ok());
            Ok(PlayerAction::Play {
                cards,
                wild_targets,
            })
        }
        "pass" => Ok(PlayerAction::Pass),
        _ => Err(AppError::BadRequest("unknown action_type".into())),
    }
}

fn normalized_event_payload(
    action_type: &'static str,
    event_payload: &serde_json::Value,
    game: Option<&crate::game::types::TableGameState>,
    new_seq: u64,
) -> serde_json::Value {
    if action_type != "play" {
        return event_payload.clone();
    }
    let Some(g) = game else {
        return event_payload.clone();
    };
    let Some(hand) = g.hand.as_ref() else {
        return event_payload.clone();
    };
    let Some(last) = hand.history.last() else {
        return event_payload.clone();
    };
    if last.seq != new_seq || last.action_type != crate::game::types::HistoryActionKind::Play {
        return event_payload.clone();
    }
    let mut payload = json!({
        "cards": last.cards.clone(),
    });
    if let Some(wt) = &last.wild_targets {
        payload["declaredWildMapping"] = json!({ "wildTargets": wt });
    }
    payload
}

fn build_transition(
    prev: &TableState,
    next: &TableState,
    prev_seq: u64,
    seq: u64,
    transition_type: &str,
    event: Option<serde_json::Value>,
) -> StateTransition {
    let mut delta = snapshot_replace_delta(prev, next);
    delta.event = event.map(|trigger| json!({ "trigger": trigger, "derived": [] }));
    StateTransition {
        seq,
        prev_seq,
        table_id: next.table_id.clone(),
        timestamp: iso_timestamp(),
        transition_type: transition_type.into(),
        delta,
    }
}

fn normalize_player_model(player_type: PlayerType, player_model: Option<String>) -> Option<String> {
    if !matches!(player_type, PlayerType::Bot) {
        return None;
    }
    player_model.and_then(|m| {
        let trimmed = m.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

impl TableStore {
    fn log_game_result(
        inner: &mut TableInner,
        completed_order: &[Seat],
        winner: TeamId,
    ) {
        let game = match inner.state.game.as_ref() {
            Some(g) => g,
            None => return,
        };
        
        let hand = match game.hand.as_ref() {
            Some(h) => h,
            None => return,
        };
        
        let mut bot_seats = Vec::new();
        let mut human_seats = Vec::new();
        
        for seat in Seat::ALL {
            if let Some(slot) = inner.state.seats.get(&seat) {
                if let Some(player) = slot.as_ref() {
                    match player.player_type {
                        PlayerType::Bot => bot_seats.push(seat.as_str().to_string()),
                        PlayerType::Human => human_seats.push(seat.as_str().to_string()),
                        _ => {}
                    }
                }
            }
        }
        
        if bot_seats.is_empty() {
            return;
        }
        
        let mut actions = Vec::new();
        for entry in &hand.history {
            actions.push(GameAction {
                seat: entry.seat.as_str().to_string(),
                seat_team: seat_to_team_str(entry.seat),
                action_type: match entry.action_type {
                    HistoryActionKind::Play => "play".to_string(),
                    HistoryActionKind::Pass => "pass".to_string(),
                },
                cards: if !entry.cards.is_empty() { Some(entry.cards.clone()) } else { None },
                seq: entry.seq,
            });
        }
        
        let finishing_order: Vec<String> = completed_order.iter()
            .map(|s| s.as_str().to_string())
            .collect();
        
        let entry = GameLogEntry {
            game_id: uuid::Uuid::new_v4().to_string(),
            table_id: game.table_id.clone(),
            start_time: iso_timestamp(),
            end_time: iso_timestamp(),
            finishing_order,
            winner_team: team_to_str(winner),
            bot_seats,
            human_seats,
            bot_params: None,
            actions,
            hand_level: hand.hand_level.as_api_str().to_string(),
        };
        
        log_game(entry);
    }
}

#[cfg(feature = "test-utils")]
impl TableStore {
    /// Replace in-memory engine state (preserves `seq` and transition log).
    /// Hidden hook for integration tests; not part of the public HTTP contract.
    #[doc(hidden)]
    pub async fn test_set_game_state(
        &self,
        table_id: &str,
        game: crate::game::types::TableGameState,
        game_config: GameConfig,
    ) -> Result<(), AppError> {
        let arc = self.table_mutex(table_id).await?;
        let mut inner = arc.lock().await;
        inner.state.game_config = game_config;
        inner.state.game = Some(game);
        inner.state.sync_phase_from_game();
        inner.state.status = TableStatus::InGame;
        inner.state.waiting_next_hand_ready = false;
        inner.state.narration.clear();
        Ok(())
    }

    /// Rewind one seated player's activity timestamp by `ago`.
    #[doc(hidden)]
    pub async fn test_rewind_player_activity(
        &self,
        table_id: &str,
        player_id: &str,
        ago: Duration,
    ) -> Result<(), AppError> {
        let arc = self.table_mutex(table_id).await?;
        let mut inner = arc.lock().await;
        let when = std::time::Instant::now() - ago;
        for slot in inner.state.seats.values_mut() {
            if let Some(player) = slot.as_mut()
                && player.player_id == player_id
            {
                player.last_activity_at = when;
                return Ok(());
            }
        }
        Err(AppError::NotFound(format!(
            "unknown player_id {} at table {}",
            player_id, table_id
        )))
    }
}
