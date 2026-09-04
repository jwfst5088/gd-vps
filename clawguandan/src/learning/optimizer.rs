//! Hill-climbing optimizer for `AdvancedBotParams`.
//!
//! Evaluates parameter sets via self-play using the simulation engine,
//! then uses hill-climbing to find better parameters.

use std::thread;
use std::time::Duration;

use crate::bot::plugins::AdvancedBotParams;
use crate::domain::Seat;
use crate::game::card::HandLevel;
use crate::game::engine::GameEngine;
use crate::game::types::{GameConfig, GamePhase, TeamId};
use crate::simulation::engine::run_match_engine;
use crate::strategy::suggest::set_learn_params_for_teams;

fn seat_team(seat: Seat) -> TeamId {
    match seat {
        Seat::E | Seat::W => TeamId::Ew,
        Seat::S | Seat::N => TeamId::Sn,
    }
}

/// Configuration for self-play evaluation.
#[derive(Clone, Debug)]
pub struct SelfPlayConfig {
    /// Number of matches to play per evaluation.
    pub matches_per_eval: u32,
    /// Maximum plies per match (safety limit).
    pub max_plies: usize,
}

/// Result of a single self-play evaluation.
#[derive(Clone, Debug)]
pub struct EvalResult {
    pub params: AdvancedBotParams,
    pub win_rate: f32,
    /// 平均升级行（0..1）：每局按掼蛋记级结算——赢双上+3 / 赢单上+2 / 输单上−2 / 被双上−3，
    /// 归一化 (delta+3)/6。直接优化"每局升级期望"而非裸胜率：
    /// 同胜率下，被双上多的参数组会显式变差（路线图①，用户 2026-09-03）。
    pub level_ev: f32,
    pub first_out_rate: f32,
    /// NS队结束时平均剩余手牌总数(S+N剩余之和)。越少越好。
    /// 直接衡量"残局剩牌"表现，缓解"最后剩小牌和单张"问题。
    pub avg_endgame_residual: f32,
    pub matches_played: u32,
}

pub fn eval_to_score(eval: &EvalResult) -> f32 {
    // 残局剩牌越少越好：clear_rate = 1 - avg_residual/27
    // (27 ≈ NS两人初始手牌总数上界，用作归一化)
    // 权重：升级期望0.5 + 头游0.2 + 残局清牌0.3
    // ① 升级期望替代裸胜率：双上+3/单上+2/输单上−2/被双上−3——同胜率下，
    //    输局常被打双上的参数组被显式惩罚（"输了也要少输"，掼蛋核心）。
    // ② 头游/清牌项保留：鼓励抢头游与残局少剩牌。
    let clear_rate = (1.0 - (eval.avg_endgame_residual / 27.0).clamp(0.0, 1.0)).max(0.0);
    eval.level_ev * 0.5 + eval.first_out_rate * 0.2 + clear_rate * 0.3
}

/// Configuration for hill-climbing optimization.
#[derive(Clone, Debug)]
pub struct HillClimbConfig {
    pub iterations: u32,
    pub step_size: f32,
    pub eval_config: SelfPlayConfig,
}

/// Evaluate a parameter set by running self-play matches.
/// Each match: 4 bots play with the same params, NS vs EW.
/// Returns the NS team's win rate.
/// Calls `progress_cb(completed, total)` after each match for progress reporting.
pub fn evaluate_params(
    params: &AdvancedBotParams,
    config: &SelfPlayConfig,
) -> EvalResult {
    evaluate_params_with_progress::<fn(u32, u32)>(params, config, None)
}

/// Same as evaluate_params but with an optional progress callback.
pub fn evaluate_params_with_progress<F: Fn(u32, u32)>(
    params: &AdvancedBotParams,
    config: &SelfPlayConfig,
    progress_cb: Option<F>,
) -> EvalResult {
    // 打破对称性：NS 用候选参数，EW 用固定基线参数(js_trained_params 房规基线)。
    // 这样 NS 胜率能真实反映候选参数相对基准的优劣，避免自对弈对称导致的~50%胜率随机游走。
    // （房规：基线从 default_balanced 改为 js_trained_params，与线上实际回退一致）
    let baseline = crate::strategy::suggest::js_trained_params();
    // 房规隔离（2026-09-03）：LEARN_PARAMS 只对本训练线程生效，线上桌面永远房规基线
    let _training_scope = crate::strategy::suggest::TrainingGuard::new();
    set_learn_params_for_teams(Some(params.clone()), Some(baseline));

    // 记录当前训练 generation,用于检测是否有新训练启动
    let my_gen = crate::learning::current_generation();

    let mut ns_wins = 0u32;
    let mut ns_first_out = 0u32;
    let mut ns_residual_sum: usize = 0;
    let mut ns_level_delta_sum: i32 = 0;
    let mut played = 0u32;

    let total = config.matches_per_eval;
    for i in 0..total {
        // 检查是否被新训练取代或被用户停止
        if !crate::learning::is_running_generation(my_gen) {
            break;
        }
        let engine = GameEngine::new(GameConfig { rng_seed: rand::random(), randomize_deals: false });
        match run_single_match(&engine, config.max_plies) {
            Ok(Some((winner, first_out_team, ns_level_delta, ns_residual))) => {
                played += 1;
                if winner == TeamId::Sn {
                    ns_wins += 1;
                }
                if first_out_team == TeamId::Sn {
                    ns_first_out += 1;
                }
                ns_level_delta_sum += ns_level_delta;
                ns_residual_sum += ns_residual;
            }
            Ok(None) => {
                played += 1;
                // 失败比赛按最差情况计：NS两人满手牌未清(27)、升级行按被双上(−3)计
                // 避免失败比赛被当作0张残牌而人为抬高clear_rate评分
                ns_residual_sum += 27;
                ns_level_delta_sum -= 3;
            }
            Err(e) => {
                eprintln!("[learn] match error: {e}");
                played += 1;
                ns_residual_sum += 27; // 同上：失败按最差情况计
                ns_level_delta_sum -= 3;
            }
        }
        if let Some(ref cb) = progress_cb {
            cb(i + 1, total);
        }
        if crate::learning::is_running_generation(my_gen) {
            thread::sleep(Duration::from_millis(5));
        }
    }

    // 只有当 generation 仍匹配时才 reset
    // 如果 generation 不匹配(被新训练取代),不 reset,避免清除新训练的参数
    if crate::learning::is_running_generation(my_gen) {
        set_learn_params_for_teams(None, None);
    }

    let win_rate = if played > 0 {
        ns_wins as f32 / played as f32
    } else {
        0.0
    };
    let first_out_rate = if played > 0 {
        ns_first_out as f32 / played as f32
    } else {
        0.0
    };
    let avg_endgame_residual = if played > 0 {
        ns_residual_sum as f32 / played as f32
    } else {
        27.0
    };

    EvalResult {
        params: params.clone(),
        win_rate,
        level_ev: if played > 0 {
            ((ns_level_delta_sum as f32 / played as f32) + 3.0) / 6.0
        } else {
            0.0
        },
        first_out_rate,
        avg_endgame_residual,
        matches_played: played,
    }
}

/// Run a single match: create a new game state, deal cards, and play until completion.
/// Returns (winner_team, first_out_team, ns_level_delta, ns_residual):
/// - ns_level_delta: NS 队本局升级行（掼蛋记级：赢双上+3/赢单上+2/输单上−2/被双上−3）
/// - ns_residual: NS (S+N) 剩余手牌总数 — 用于惩罚残局剩牌。
fn run_single_match(
    engine: &GameEngine,
    max_plies: usize,
) -> Result<Option<(TeamId, TeamId, i32, usize)>, String> {
    let mut state = engine.init_table(format!("learn_{}", uuid::Uuid::new_v4()));
    let first_drawer = Seat::S;
    engine
        .start_first_hand(&mut state, first_drawer, HandLevel::Two)
        .map_err(|e| format!("start_first_hand: {e}"))?;

    let outcome = run_match_engine(engine, &mut state, 1, max_plies)
        .map_err(|e| format!("run_match: {e}"))?;

    if outcome.final_phase == GamePhase::Scoring {
        if let Some(winner) = state.winner_team {
            let finishing = state
                .hand
                .as_ref()
                .map(|h| h.finishing_order.clone())
                .unwrap_or_default();
            let first_out_team = finishing.first().copied().map(seat_team).unwrap_or(winner);
            // 升级行：头游队为胜方；胜方二游也是本队 → 双上+3，否则单上+2；负方取负。
            let ns_level_delta = match (finishing.first(), finishing.get(1)) {
                (Some(f1), Some(f2)) => {
                    let gain = if seat_team(*f2) == seat_team(*f1) { 3i32 } else { 2 };
                    if seat_team(*f1) == TeamId::Sn { gain } else { -gain }
                }
                _ => 0,
            };
            // NS队结束时剩余手牌总数(S+N)：衡量残局剩牌表现
            let ns_residual = state
                .hand
                .as_ref()
                .map(|h| {
                    h.hands.get(&Seat::S).map(|v| v.len()).unwrap_or(0)
                        + h.hands.get(&Seat::N).map(|v| v.len()).unwrap_or(0)
                })
                .unwrap_or(0);
            Ok(Some((winner, first_out_team, ns_level_delta, ns_residual)))
        } else {
            Ok(None)
        }
    } else {
        Ok(None)
    }
}

/// Hill-climbing optimization: start with initial params, mutate and evaluate.
/// Keeps the better params and continues for the configured number of iterations.
pub fn optimize(start: &AdvancedBotParams, config: &HillClimbConfig) -> AdvancedBotParams {    let mut best = start.clone();
    let mut best_eval = evaluate_params(&best, &config.eval_config);
    println!(
        "[learn] iter 0: level_ev={:.3} win_rate={:.3} first_out={:.3} residual={:.1} score={:.4}",
        best_eval.level_ev, best_eval.win_rate, best_eval.first_out_rate, best_eval.avg_endgame_residual,
        eval_to_score(&best_eval)
    );

    for i in 1..=config.iterations {
        let candidate = best.mutate_random(config.step_size);
        let eval = evaluate_params(&candidate, &config.eval_config);

        println!(
            "[learn] iter {i}: level_ev={:.3} win_rate={:.3} first_out={:.3} residual={:.1} (best_score={:.4})",
            eval.level_ev, eval.win_rate, eval.first_out_rate, eval.avg_endgame_residual, eval_to_score(&best_eval)
        );

        // 用综合评分(eval_to_score)比较：升级期望0.5 + 头游0.2 + 清牌0.3。
        if eval_to_score(&eval) > eval_to_score(&best_eval) {
            best = candidate;
            best_eval = eval;
            println!(
                "[learn]   -> improved! new best score={:.4} (level_ev={:.3} win={:.3} residual={:.1})",
                eval_to_score(&best_eval), best_eval.level_ev, best_eval.win_rate, best_eval.avg_endgame_residual
            );
        }
    }

    best
}

#[cfg(test)]
mod trainer_repro_tests {
    use super::*;

    /// 排查训练器全部对局 0 分问题：单局自对弈能否正常完成并产生胜负。
    #[test]
    fn single_selfplay_match_completes() {
        let mut completed = 0;
        let mut none_count = 0;
        let mut err_count = 0;
        for seed in 1..=10u64 {
            let t0 = std::time::Instant::now();
            let engine = GameEngine::new(GameConfig { rng_seed: seed, randomize_deals: false });
            match run_single_match(&engine, 2000) {
                Ok(Some((w, f, d, res))) => {
                    completed += 1;
                    println!("seed {seed}: OK winner={w:?} first_out={f:?} delta={d} residual={res} ({:.1}s)", t0.elapsed().as_secs_f32());
                }
                Ok(None) => {
                    none_count += 1;
                    println!("seed {seed}: Ok(None) ({:.1}s) —— 相位Scoring但无winner，或非Scoring相位", t0.elapsed().as_secs_f32());
                }
                Err(e) => {
                    err_count += 1;
                    println!("seed {seed}: Err: {e} ({:.1}s)", t0.elapsed().as_secs_f32());
                }
            }
        }
        println!("completed={completed} none={none_count} err={err_count}");
        assert!(completed > 0, "没有任何一局正常完成: none={none_count} err={err_count}");
    }
}