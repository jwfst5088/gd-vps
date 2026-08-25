//! Deterministic choice among [`super::enumerate_legal_actions`] for a single actor.
//!
//! Strategy improvements over the baseline:
//! - Hand composition analysis to avoid splitting bombs/straights/triples
//! - Endgame mode (≤6 cards): prioritize clearing more cards when leading
//! - Teammate sprint support: play supportive card types when teammate is close to winning
//! - Opponent sprint interrupt: prefer bombs/intercepts when opponent is close to winning
//! - Bomb conservation: 1 bomb → save for midgame; endgame 1 bomb → use to win
//! - Joker respect: early game only (my_remaining > 10), don't bomb jokers/level cards when bombs < 3
//! - Endgame bomb: 1 bomb in endgame → play small cards first, use bomb to win (not hold forever)
//! - Straight priority: prefer straights that consume ≥3 single cards, even if it means splitting combos
//! - Pass as strategic choice: Pass competes with plays in scoring, enabling bomb conservation
//! - Endgame cleanup: clear small singles when following in endgame, keep big cards + bombs
//! - Bomb sizing: prefer smaller bombs (4-card) over bigger bombs, straight flush, or four joker

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Mutex;

use crate::domain::Seat;
use crate::game::card::{
    is_wild, level_order_value, natural_rank_value, parse_card_symbol, RuleContext,
};
use crate::game::engine::PlayerAction;
use crate::game::rules::combination_parser::{
    BombKind, CombinationClass, CombinationKind, CombinationParser, OrdinaryKind,
};
use crate::game::types::{GamePhase, HandState, TableGameState};

use super::enumerate_legal_actions;

// ── Learning params (global store for AI self-learning) ──
// When set, the strategy uses these weights instead of hardcoded values.
// Set by the learning module before running self-play evaluations.
use crate::bot::plugins::advanced_bot::AdvancedBotParams;
static LEARN_PARAMS: Mutex<Option<AdvancedBotParams>> = Mutex::new(None);

/// Set the global learning parameters. Pass `None` to reset to defaults.
pub fn set_learn_params(params: Option<AdvancedBotParams>) {
    if let Ok(mut lock) = LEARN_PARAMS.lock() {
        *lock = params;
    }
}

/// Get the current learning parameters (or None if using defaults).
fn get_learn_params() -> Option<AdvancedBotParams> {
    LEARN_PARAMS.lock().ok().and_then(|lock| lock.clone())
}

// ── Team-specific learning params (asymmetric training) ──
// When set, NS team (S/N seats) uses LEARN_PARAMS_NS, EW team (E/W) uses LEARN_PARAMS_EW.
// Falls back to the generic LEARN_PARAMS when the team-specific slot is None.
// This breaks NS/EW symmetry in self-play so the optimizer can actually tell whether a
// candidate param set plays better than the baseline, instead of seeing ~50% win rate
// regardless of params.
static LEARN_PARAMS_NS: Mutex<Option<AdvancedBotParams>> = Mutex::new(None);
static LEARN_PARAMS_EW: Mutex<Option<AdvancedBotParams>> = Mutex::new(None);

/// Set team-specific learning parameters for asymmetric self-play evaluation.
/// Pass `(Some(candidate), Some(baseline))` to evaluate candidate (NS) vs baseline (EW).
/// Pass `None` for both to reset to symmetric mode (uses generic LEARN_PARAMS).
pub fn set_learn_params_for_teams(
    ns: Option<AdvancedBotParams>,
    ew: Option<AdvancedBotParams>,
) {
    if let Ok(mut lock) = LEARN_PARAMS_NS.lock() {
        *lock = ns;
    }
    if let Ok(mut lock) = LEARN_PARAMS_EW.lock() {
        *lock = ew;
    }
}

/// Get learning parameters for a specific seat.
/// Priority: team-specific (NS/EW) > generic LEARN_PARAMS > None.
fn get_learn_params_for_seat(seat: Seat) -> Option<AdvancedBotParams> {
    let team_lock = match seat {
        Seat::S | Seat::N => LEARN_PARAMS_NS.lock(),
        _ => LEARN_PARAMS_EW.lock(),
    };
    team_lock
        .ok()
        .and_then(|l| l.clone())
        .or_else(get_learn_params)
}

// ── Hand composition analysis ──────────────────────────────────────────

/// Records which cards in hand belong to "good combos" that should not be split.
#[derive(Clone, Debug, Default)]
#[allow(dead_code)]
struct HandCombos {
    /// For each card symbol, the natural rank value (0 if not computed).
    card_to_rank: HashMap<String, u8>,
    /// How many cards of each natural rank exist in hand.
    rank_to_count: HashMap<u8, usize>,
    /// Ranks that have >= 4 cards (potential bomb).
    bomb_ranks: Vec<u8>,
    /// Ranks that have >= 3 cards (potential triple / plate part).
    triple_ranks: Vec<u8>,
    /// Ranks that have >= 2 cards (potential pair / tube part).
    pair_ranks: Vec<u8>,
    /// Consecutive triple ranks (plate candidates): pairs of (rank, rank+1).
    plate_pairs: Vec<(u8, u8)>,
    /// Consecutive pair triples (tube candidates): triples of (rank, rank+1, rank+2).
    tube_triples: Vec<(u8, u8, u8)>,
    /// Total estimated bomb count in hand (same-rank 4+ + straight flush candidates).
    /// Used for bomb conservation strategy.
    bomb_count: usize,
    /// Number of straight flush candidates (cached to avoid recomputation).
    straight_flush_count: usize,
}

fn analyze_hand_combos(hand: &[String], ctx: RuleContext) -> HandCombos {
    let mut card_to_rank: HashMap<String, u8> = HashMap::new();
    let mut rank_to_count: HashMap<u8, usize> = HashMap::new();
    let mut wild_count = 0usize;

    for card in hand {
        if let Ok(c) = parse_card_symbol(card) {
            if is_wild(c, ctx) {
                wild_count += 1;
                continue; // wildcards are flexible; no penalty for using them
            }
            if let Ok(nat_val) = natural_rank_value(c.rank) {
                *rank_to_count.entry(nat_val).or_default() += 1;
                card_to_rank.insert(card.clone(), nat_val);
            }
        }
    }

    let bomb_ranks: Vec<u8> = rank_to_count
        .iter()
        .filter(|(_, c)| **c >= 4)
        .map(|(r, _)| *r)
        .collect();
    let triple_ranks: Vec<u8> = rank_to_count
        .iter()
        .filter(|(_, c)| **c >= 3)
        .map(|(r, _)| *r)
        .collect();
    let pair_ranks: Vec<u8> = rank_to_count
        .iter()
        .filter(|(_, c)| **c >= 2)
        .map(|(r, _)| *r)
        .collect();

    // Find consecutive triple ranks (plate candidates)
    let mut plate_pairs = Vec::new();
    let mut sorted_triples = triple_ranks.clone();
    sorted_triples.sort();
    for w in sorted_triples.windows(2) {
        if w[1] - w[0] == 1 {
            plate_pairs.push((w[0], w[1]));
        }
    }

    // Find consecutive pair triples (tube candidates)
    let mut tube_triples = Vec::new();
    let mut sorted_pairs = pair_ranks.clone();
    sorted_pairs.sort();
    for w in sorted_pairs.windows(3) {
        if w[1] - w[0] == 1 && w[2] - w[1] == 1 {
            tube_triples.push((w[0], w[1], w[2]));
        }
    }

    // Count total bombs: same-rank 4+ + wildcard-assisted + straight flush candidates
    let straight_flush_count = count_straight_flush_candidates(hand, ctx);

    // Wildcard-assisted bombs: 3 same-rank + 1 wildcard = a bomb
    let wild_assisted_bombs = if wild_count >= 1 {
        rank_to_count.values().filter(|&&c| c == 3).count().min(wild_count)
    } else {
        0
    };

    let bomb_count = bomb_ranks.len() + wild_assisted_bombs + straight_flush_count;

    HandCombos {
        card_to_rank,
        rank_to_count,
        bomb_ranks,
        triple_ranks,
        pair_ranks,
        plate_pairs,
        tube_triples,
        bomb_count,
        straight_flush_count,
    }
}

/// Count potential straight flush combinations in hand.
/// A straight flush is 5+ consecutive cards of the same suit.
fn count_straight_flush_candidates(hand: &[String], ctx: RuleContext) -> usize {
    let mut suit_to_ranks: HashMap<String, Vec<u8>> = HashMap::new();
    for card in hand {
        if let Ok(c) = parse_card_symbol(card) {
            if is_wild(c, ctx) {
                continue;
            }
            if let Ok(nat_val) = natural_rank_value(c.rank) {
                let suit = card.chars().next().map(|ch| ch.to_string()).unwrap_or_default();
                if !suit.is_empty() {
                    suit_to_ranks.entry(suit).or_default().push(nat_val);
                }
            }
        }
    }

    let mut count = 0;
    for (_, mut ranks) in suit_to_ranks {
        ranks.sort();
        ranks.dedup();
        if ranks.len() < 5 {
            continue;
        }
        // Find consecutive sequences of 5+
        let mut run_len = 1;
        for i in 1..ranks.len() {
            if ranks[i] - ranks[i - 1] == 1 {
                run_len += 1;
                if run_len >= 5 {
                    count += 1;
                    run_len = 0; // reset to avoid double-counting overlapping sequences
                }
            } else {
                run_len = 1;
            }
        }
    }

    count
}

/// Returns a penalty score for a play based on how much it "splits" good combos.
/// Higher = worse (more splitting).
///
/// Penalty is scaled by total bomb count in hand:
/// - 1 bomb: breaking it = 10 (never break the only bomb)
/// - 2 bombs: breaking = 5 (strongly discourage)
/// - 3+ bombs: breaking = 3 (normal penalty)
///
/// Plate/Tube protection:
/// - Playing a single/pair from a triple that's part of a plate: penalty 3
/// - Playing the triple (3 cards) but not the full plate: penalty 1
/// - Playing a single from a pair that's part of a tube: penalty 2
/// - Playing the pair (2 cards) but not the full tube: penalty 1
fn split_penalty(cards: &[String], combos: &HandCombos, ctx: RuleContext) -> u8 {
    // Count how many cards of each rank are in this play
    let mut play_rank_counts: HashMap<u8, usize> = HashMap::new();
    for card in cards {
        if let Some(&rank) = combos.card_to_rank.get(card) {
            *play_rank_counts.entry(rank).or_default() += 1;
        }
    }

    let _bomb_count = combos.bomb_count;
    let mut penalty = 0u8;

    // Level card rank for bomb-break encouragement
    let level_rank = natural_rank_value(ctx.hand_level.to_rank()).unwrap_or(0);

    for (rank, play_count) in &play_rank_counts {
        let hand_count = combos.rank_to_count.get(rank).copied().unwrap_or(0);
        let is_level_rank = *rank == level_rank;

        // ── Bomb protection ──
        if hand_count >= 4 && *play_count < hand_count {
            if is_level_rank {
                // 级牌炸弹应拆开：不惩罚拆级牌炸弹
                continue;
            }
            // 炸弹不能拆开出：4张同rank不能拆成3+1、2+2等
            // 惩罚极高，确保任何情况下都不会拆炸弹
            penalty = penalty.saturating_add(50);
            continue; // bomb check done, skip plate/tube checks
        }

        // ── Plate protection (钢板: consecutive triples) ──
        if hand_count >= 3 {
            let is_plate_part = combos
                .plate_pairs
                .iter()
                .any(|(a, b)| *rank == *a || *rank == *b);
            if is_plate_part {
                let plays_full_plate = combos
                    .plate_pairs
                    .iter()
                    .filter(|(a, b)| *rank == *a || *rank == *b)
                    .any(|(a, b)| {
                        let other = if *rank == *a { *b } else { *a };
                        play_rank_counts.get(&other).copied().unwrap_or(0) >= 3
                            && *play_count >= 3
                    });
                if !plays_full_plate {
                    if *play_count < 3 {
                        penalty += 3;
                    } else {
                        penalty += 1;
                    }
                }
                continue;
            }
            if *play_count < 3 && *play_count < hand_count {
                penalty += 1;
            }
            continue;
        }

        // ── Tube protection (木板: consecutive pairs) ──
        if hand_count >= 2 {
            let is_tube_part = combos
                .tube_triples
                .iter()
                .any(|(a, b, c)| *rank == *a || *rank == *b || *rank == *c);
            if is_tube_part {
                let plays_full_tube = combos
                    .tube_triples
                    .iter()
                    .filter(|(a, b, c)| *rank == *a || *rank == *b || *rank == *c)
                    .any(|(a, b, c)| {
                        [*a, *b, *c].iter().all(|r| {
                            play_rank_counts.get(r).copied().unwrap_or(0) >= 2
                        })
                    });
                if !plays_full_tube {
                    if *play_count < 2 {
                        penalty += 2;
                    } else {
                        penalty += 1;
                    }
                }
            }
        }
    }

    penalty
}

// ── Context-aware play selection ───────────────────────────────────────

/// Pre-computed card info to avoid repeated parse_card_symbol calls in score_play.
#[derive(Clone, Debug)]
struct CardInfo {
    /// natural_rank_value
    rank: u8,
    is_wild: bool,
    is_level: bool,
    is_joker: bool,
}

/// Game context extracted once for scoring all candidate plays.
struct PlayContext {
    is_leading: bool,
    /// Whether the current top card was played by the teammate.
    partner_leading: bool,
    /// Primary value of the current top play (if any), used to judge if teammate's card is "big".
    top_play_value: Option<u8>,
    /// Debug representation of the current top play's combination kind.
    /// E.g. "Ordinary(Single)", "Ordinary(Pair)", "Ordinary(Triple)".
    top_play_kind: Option<String>,
    my_remaining: usize,
    /// The seat of the actor (used to pick team-specific learn params).
    actor: Seat,
    teammate_remaining: usize,
    min_opp_remaining: usize,
    combos: HandCombos,
    /// Total bomb count in hand (same-rank 4+ + straight flush candidates).
    bomb_count: usize,
    // ── 精准残局判断变量 ──
    /// 对手剩1张
    opponent_1: bool,
    /// 对手剩2张
    opponent_2: bool,
    /// 对手剩3-5张（危险区间）
    opponent_3_5: bool,
    /// 队友剩1张
    teammate_1: bool,
    /// 队友剩2张
    teammate_2: bool,
    /// 队友剩3张
    teammate_3: bool,
    // ── 炸弹使用原则：对手剩牌数细分 ──
    /// 对手剩5张（大概率4炸+单，必炸）
    opponent_5: bool,
    /// 对手剩7张（多为一炸+两手牌，必炸）
    opponent_7: bool,
    /// 对手剩4张（极可能本身就是4张炸，慎炸）
    opponent_4: bool,
    /// 对手剩8张（可能是两炸或炸+牌，缓观）
    opponent_8: bool,
    // ── 顺风判断 ──
    /// 我方是否顺风大优（我+队友剩余牌数都少于对手）
    is_wind_advantage: bool,
    // ── 队友牌质量 ──
    /// 队友牌烂（队友剩余牌≥15张，全程无上手机会）
    teammate_weak: bool,
    // ── 当前牌权是否被队友炸弹控制 ──
    /// 队友刚才出了炸弹并拿到牌权
    partner_just_bombed: bool,
    // ── 手牌散牌统计 ──
    /// 手牌中单张的数量（用于判断炸完后是否全是散牌）
    singles_count: usize,
    // ── 炸弹大小统计 ──
    /// 4张小炸弹数量
    small_bomb_count: usize,
    /// 5张中炸弹+同花顺数量
    mid_bomb_count: usize,
    /// 6张+大炸弹+王炸数量
    big_bomb_count: usize,
    /// 当前打几的级牌rank值（如打2则level_rank=2）
    level_rank: u8,
    /// Pre-computed card info for actor's hand (avoids repeated parse_card_symbol).
    card_info: HashMap<String, CardInfo>,
}

fn build_play_context(hand: &HandState, actor: Seat) -> PlayContext {
    let my_cards = hand.hands.get(&actor).map(|v| v.as_slice()).unwrap_or(&[]);
    let my_remaining = my_cards.len();
    let top = hand.trick.top_play.as_ref();
    let is_leading = top.is_none();

    let teammate_seat = actor.teammate();
    let partner_leading = top.map(|t| t.seat == teammate_seat).unwrap_or(false);
    let teammate_remaining = hand
        .hands
        .get(&teammate_seat)
        .map(|v| v.len())
        .unwrap_or(27);

    // Collect opponent card counts once instead of traversing 8 times
    let opp_counts: Vec<usize> = Seat::ALL
        .iter()
        .filter(|s| **s != actor && **s != teammate_seat)
        .filter_map(|s| hand.hands.get(s).map(|v| v.len()))
        .collect();
    let min_opp_remaining = opp_counts.iter().min().copied().unwrap_or(27);

    let opponent_1 = opp_counts.iter().any(|&n| n == 1);
    let opponent_2 = !opponent_1 && opp_counts.iter().any(|&n| n == 2);
    let opponent_3_5 = !opponent_1 && !opponent_2 && opp_counts.iter().any(|&n| n >= 3 && n <= 5);

    let teammate_1 = teammate_remaining == 1;
    let teammate_2 = teammate_remaining == 2;
    let teammate_3 = teammate_remaining == 3;

    let opponent_5 = !opponent_1 && !opponent_2 && !opponent_3_5 && opp_counts.iter().any(|&n| n == 5);
    let opponent_7 = !opponent_1 && !opponent_2 && !opponent_3_5 && !opponent_5 && opp_counts.iter().any(|&n| n == 7);
    let opponent_4 = !opponent_1 && !opponent_2 && !opponent_3_5 && opp_counts.iter().any(|&n| n == 4);
    let opponent_8 = !opponent_1 && !opponent_2 && !opponent_3_5 && !opponent_5 && !opponent_7 && opp_counts.iter().any(|&n| n == 8);

    // ── 顺风判断：我+队友剩余牌数都少于对手 ──
    let opp_max_remaining = opp_counts.iter().max().copied().unwrap_or(27);
    let is_wind_advantage = my_remaining < opp_max_remaining && teammate_remaining < opp_max_remaining;

    // ── 队友牌质量 ──
    let teammate_weak = teammate_remaining >= 15;

    // ── 队友刚才是否出了炸弹 ──
    let partner_just_bombed = top.as_ref().map(|t| {
        t.seat == teammate_seat && format!("{:?}", t.combination.kind).starts_with("Bomb")
    }).unwrap_or(false);

    let ctx = RuleContext {
        hand_level: hand.hand_level,
    };
    let combos = analyze_hand_combos(my_cards, ctx);
    let bomb_count = combos.bomb_count;

    // ── 手牌散牌统计（依赖combos.rank_to_count）──
    let singles_count = my_cards.iter().filter(|card| {
        if let Ok(c) = parse_card_symbol(card) {
            if let Ok(nat) = natural_rank_value(c.rank) {
                let count = combos.rank_to_count.get(&nat).copied().unwrap_or(0);
                return count == 1;
            }
        }
        false
    }).count();

    // ── 炸弹大小统计 ──
    let mut small_bomb_count = 0usize;
    let mut mid_bomb_count = 0usize;
    let mut big_bomb_count = 0usize;
    for rank in &combos.bomb_ranks {
        let count = combos.rank_to_count.get(rank).copied().unwrap_or(0);
        if count == 4 {
            small_bomb_count += 1;
        } else if count == 5 {
            mid_bomb_count += 1;
        } else {
            big_bomb_count += 1;
        }
    }
    // 同花顺算中等炸弹
    mid_bomb_count += combos.straight_flush_count;
    // 王炸算大炸弹（wild_assisted_bombs按最坏情况估算）

    let top_play_value = top.map(|t| t.combination.primary);

    let top_play_kind = top.map(|t| format!("{:?}", t.combination.kind));

    let level_rank = natural_rank_value(hand.hand_level.to_rank()).unwrap_or(2) as u8;

    // Pre-compute card info for score_play to avoid repeated parse_card_symbol
    let card_info: HashMap<String, CardInfo> = my_cards.iter().map(|card| {
        let c = parse_card_symbol(card).unwrap();
        let rank = natural_rank_value(c.rank).unwrap_or(0);
        (card.clone(), CardInfo {
            rank,
            is_wild: is_wild(c, ctx),
            is_level: rank == level_rank,
            is_joker: card.starts_with("🃏") || card.starts_with("👑"),
        })
    }).collect();

    PlayContext {
        is_leading,
        partner_leading,
        top_play_value,
        top_play_kind,
        my_remaining,
        actor,
        teammate_remaining,
        min_opp_remaining,
        combos,
        bomb_count,
        opponent_1,
        opponent_2,
        opponent_3_5,
        teammate_1,
        teammate_2,
        teammate_3,
        opponent_5,
        opponent_7,
        opponent_4,
        opponent_8,
        is_wind_advantage,
        teammate_weak,
        partner_just_bombed,
        singles_count,
        small_bomb_count,
        mid_bomb_count,
        big_bomb_count,
        level_rank,
        card_info,
    }
}

// ── Public API ─────────────────────────────────────────────────────────

/// Pick one legal action:
/// - playing: context-aware strategy (small first, avoid splitting, endgame, teammate support)
/// - tribute/return: smaller card value first
pub fn suggest_next_action(
    state: &TableGameState,
    actor: Seat,
) -> Result<PlayerAction, String> {
    let legal = enumerate_legal_actions(state, actor)?;
    if legal.is_empty() {
        return Err("no legal actions".into());
    }

    let hand = state.hand.as_ref().ok_or_else(|| "no hand".to_string())?;
    let ctx = RuleContext {
        hand_level: hand.hand_level,
    };

    match state.phase {
        GamePhase::Playing => pick_playing_with_context(&legal, ctx, hand, actor),
        GamePhase::Tribute => pick_tribute(&legal, ctx),
        GamePhase::Exchange => pick_return(&legal, ctx),
        _ => Err("suggest_next_action: not in tribute, exchange, or playing".into()),
    }
}

fn pick_tribute(legal: &[PlayerAction], ctx: RuleContext) -> Result<PlayerAction, String> {
    pick_lowest_card_action(
        legal,
        ctx,
        |a| match a {
            PlayerAction::Tribute { card } => Some(card.as_str()),
            _ => None,
        },
        "suggest: no tribute action",
    )
}

fn pick_return(legal: &[PlayerAction], ctx: RuleContext) -> Result<PlayerAction, String> {
    pick_lowest_card_action(
        legal,
        ctx,
        |a| match a {
            PlayerAction::ReturnCard { card } => Some(card.as_str()),
            _ => None,
        },
        "suggest: no return_card action",
    )
}

fn pick_lowest_card_action<'a>(
    legal: &'a [PlayerAction],
    ctx: RuleContext,
    card_for_action: impl Fn(&'a PlayerAction) -> Option<&'a str>,
    no_action_error: &'static str,
) -> Result<PlayerAction, String> {
    let mut items: Vec<(u8, String, PlayerAction)> = Vec::new();
    for a in legal {
        let Some(card) = card_for_action(a) else {
            continue;
        };
        let c = parse_card_symbol(card)?;
        let v = level_order_value(c, ctx);
        items.push((v, card.to_string(), a.clone()));
    }
    items.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    items
        .into_iter()
        .next()
        .map(|(_, _, act)| act)
        .ok_or_else(|| no_action_error.into())
}

// ── Playing phase: context-aware selection ─────────────────────────────

fn pick_playing_with_context(
    legal: &[PlayerAction],
    ctx: RuleContext,
    hand: &HandState,
    actor: Seat,
) -> Result<PlayerAction, String> {
    let plays: Vec<&PlayerAction> = legal
        .iter()
        .filter(|a| matches!(a, PlayerAction::Play { .. }))
        .collect();

    let pass_action = legal
        .iter()
        .find(|a| matches!(a, PlayerAction::Pass));

    if plays.is_empty() {
        return pass_action
            .cloned()
            .ok_or_else(|| "suggest: no pass in legal".into());
    }

    let pctx = build_play_context(hand, actor);

    // ══ 房规：只剩最后1张时，能压就立即打出清空夺头游，任何启发式不得拦截 ══
    if pass_action.is_some() && pctx.my_remaining == 1 && !plays.is_empty() {
        return Ok(plays[0].clone());
    }

    // Score each play with context, then pick the best (lowest score).
    let mut scored: Vec<(PlayScore, &PlayerAction)> = Vec::with_capacity(plays.len() + 1);
    for a in &plays {
        let score = score_play(a, ctx, &pctx);
        scored.push((score, *a));
    }

    // Include Pass as a scored option so bomb conservation can actually trigger passing.
    // Pass tier 5: beats tier 6+ (bomb conservation, partner big card, joker respect)
    // but loses to tier 0-4 (normal plays, partner following, endgame bomb with 2+ bombs).
    if let Some(pass) = pass_action {
        let mut strategic_tier = 5u8;
        
        // 应用学习参数对 Pass 的调整（使用 seat-specific 参数打破对称性）
        if let Some(lp) = get_learn_params_for_seat(actor) {
            let scale = lp.team_win_weight.clamp(0.1, 10.0);
            let pass_penalty = lp.pass_stall_penalty.clamp(0.1, 10.0);
            strategic_tier = ((strategic_tier as f32 * pass_penalty * scale).round() as u8).min(255);
        }
        
        let pass_score = PlayScore {
            strategic_tier,
            split_penalty: 0,
            wild_count: 0,
            is_bomb: false,
            is_non_level: true,
            primary: 0,
            cards_len: std::cmp::Reverse(0),
            sorted_cards: vec![],
        };
        scored.push((pass_score, pass));
    }

    scored.sort_by(|a, b| a.0.cmp(&b.0));

    // ══ 房规硬守卫：残局保留最后一个炸弹——最优解若为「非清空用最后炸弹」且对手未冲刺，强制过牌 ══
    if let Some((_, best_act)) = scored.first() {
        if let PlayerAction::Play { cards: best_cards, .. } = best_act {
            if best_cards.len() < pctx.my_remaining
                && pctx.combos.bomb_count == 1
                && pctx.my_remaining <= 6
                && pctx.min_opp_remaining > 3
                && pctx.my_remaining - best_cards.len() > 2
            {
                let looks_bomb = CombinationParser::parse(best_cards, None, ctx)
                    .map(|c| matches!(c.class(), CombinationClass::Bomb))
                    .unwrap_or(false);
                if looks_bomb {
                    if let Some(pass) = pass_action {
                        return Ok(pass.clone());
                    }
                }
            }
        }
    }

    scored
        .first()
        .map(|(_, a)| (*a).clone())
        .ok_or_else(|| "suggest: empty play list".into())
}

// ── Play scoring ───────────────────────────────────────────────────────

/// Composite sort key for a play candidate. Lower = better.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct PlayScore {
    /// Strategic tier: 0=normal, 1=teammate support, 2=endgame clear, 3=opponent intercept
    strategic_tier: u8,
    /// Split penalty: 0=no split, 1=minor, 2=medium, 3=heavy
    split_penalty: u8,
    /// Fewer wildcards = better
    wild_count: usize,
    /// Non-bomb (false) before bomb (true)
    is_bomb: bool,
    /// Level card combo (false) before non-level (true): level cards should be preferred
    is_non_level: bool,
    /// Smaller primary value = better
    primary: u8,
    /// More cards = better (Reverse makes larger sort first)
    cards_len: std::cmp::Reverse<usize>,
    /// Lexicographic tie-break
    sorted_cards: Vec<String>,
}

fn score_play(a: &PlayerAction, ctx: RuleContext, pctx: &PlayContext) -> PlayScore {
    let (cards, wild_targets) = match a {
        PlayerAction::Play {
            cards,
            wild_targets,
        } => (cards, wild_targets),
        _ => {
            return PlayScore {
                strategic_tier: 255,
                split_penalty: 0,
                wild_count: 0,
                is_bomb: true,
                is_non_level: true,
                primary: 255,
                cards_len: std::cmp::Reverse(0),
                sorted_cards: vec![],
            }
        }
    };

    let combo = CombinationParser::parse(cards, wild_targets.as_deref(), ctx).unwrap_or_else(|_| {
        // Fallback: treat as single with max value
        CombinationParser::parse(&[cards[0].clone()], None, ctx).unwrap()
    });

    let wild_count = cards
        .iter()
        .filter_map(|s| pctx.card_info.get(s).map(|ci| usize::from(ci.is_wild)))
        .sum::<usize>();

    let is_bomb = matches!(combo.class(), CombinationClass::Bomb);
    let penalty = split_penalty(cards, &pctx.combos, ctx);

    // ── 炸弹拆分兜底检查（直接解析卡片rank，不依赖card_to_rank映射）──
    // 如果手牌中有4+张同rank且本次出牌只用了其中部分 → 判定为拆炸弹
    let mut direct_bomb_penalty = 0u8;
    {
        let level_rank_val = pctx.level_rank;
        let mut play_rank_counts: HashMap<u8, usize> = HashMap::new();
        for card in cards {
            if let Some(ci) = pctx.card_info.get(card) {
                if !ci.is_wild {
                    *play_rank_counts.entry(ci.rank).or_default() += 1;
                }
            }
        }
        // 房规三豁免（mirror JS classifyBombSplit）：①清空手牌 ②中盘带出≥3张单牌
        // ③中盘同花顺且剩余单张≤2。残局一律禁止。
        let split_clearing = cards.len() >= pctx.my_remaining;
        let split_endgame = pctx.my_remaining <= 6;
        let mut carried_singles = 0usize;
        for (&r, &pc) in &play_rank_counts {
            if pctx.combos.rank_to_count.get(&r).copied().unwrap_or(0) == 1 {
                carried_singles += pc;
            }
        }
        let is_sf_play = matches!(combo.kind, CombinationKind::Bomb(BombKind::StraightFlush));
        let mut singles_after_sf = usize::MAX;
        if is_sf_play {
            let mut s = pctx.singles_count;
            for c in cards {
                if let Some(ci) = pctx.card_info.get(c) {
                    if pctx.combos.rank_to_count.get(&ci.rank).copied().unwrap_or(0) == 1 {
                        s = s.saturating_sub(1);
                    }
                }
            }
            singles_after_sf = s;
        }
        for (rank, play_count) in &play_rank_counts {
            if *rank == level_rank_val {
                continue; // 级牌炸弹允许拆
            }
            let hand_total = pctx.combos.rank_to_count.get(rank).copied().unwrap_or(0);
            if hand_total >= 4 && *play_count < hand_total {
                let exempt = split_clearing
                    || (!split_endgame && carried_singles >= 3)
                    || (!split_endgame && is_sf_play && singles_after_sf <= 2);
                if !exempt {
                    direct_bomb_penalty = 100; // 拆炸弹，极高惩罚
                }
                break;
            }
        }
    }

    // ── 房规公共量：是否清空手牌 ──
    let clearing = cards.len() >= pctx.my_remaining;

    let mut sorted = cards.clone();
    sorted.sort();

    // Determine strategic tier based on game context
    let mut strategic_tier = determine_strategic_tier(is_bomb, pctx);

    // 级牌和大小王仅领牌时禁止空出（除非出到最后）
    // 炸弹领牌时不能空炸，跟牌时炸弹可以夺牌权
    // 这些牌应该留着压牌，绝对不能空出浪费
    // 只有最后出牌（cards.len() == my_remaining）时才允许
    if cards.len() < pctx.my_remaining {
        let has_level = cards.iter().any(|c| {
            pctx.card_info.get(c).map(|ci| ci.is_level).unwrap_or(false)
        });
        let has_joker = cards.iter().any(|c| c.starts_with("🃏") || c.starts_with("👑"));
        // 级牌和大小王：仅领牌时禁止空出
        if pctx.is_leading && (has_level || has_joker) {
            strategic_tier = strategic_tier.saturating_add(100); // 绝对不能空出
        }
        // 炸弹：领牌时不能空炸，跟牌时炸弹可以夺牌权
        // 房规豁免（mirror JS「空出炸弹重罚」例外2）：剩余手牌全是炸弹（各组≥4张，
        // 王/百搭按其rank键单独成组也需≥4张）→ 领炸连出不受罚
        if is_bomb && pctx.is_leading {
            let mut rest_counts: HashMap<u8, usize> = HashMap::new();
            for (card_str, ci) in &pctx.card_info {
                if cards.contains(card_str) {
                    continue;
                }
                *rest_counts.entry(ci.rank).or_default() += 1;
            }
            let rest_all_bombs = rest_counts.values().all(|&n| n >= 4);
            if !rest_all_bombs {
                strategic_tier = strategic_tier.saturating_add(100); // 领牌空炸，绝对禁止
            }
        }
    }

    // 逢人配不能单出，尽量配成炸弹/同花顺/木板/钢板/杂顺等组合
    // 逢人配极其珍贵，单出/对子/三张同中浪费逢人配 → 极高惩罚；用于组合牌型 → 不惩罚
    // 房规豁免（mirror JS finishingPlay）：清空手牌时百搭任意用法全部豁免
    if wild_count > 0 && !is_bomb && !clearing {
        let waste = match combo.kind {
            CombinationKind::Ordinary(OrdinaryKind::Single) => 100, // 逢人配绝不能单出，极高惩罚
            CombinationKind::Ordinary(OrdinaryKind::Pair) => 50,    // 对子中浪费逢人配，极高惩罚
            CombinationKind::Ordinary(OrdinaryKind::Triple) => 30,  // 三张同中浪费逢人配，高惩罚
            // 顺子/木板/钢板/三带二等组合牌型：逢人配用于构成合理牌型 → 不惩罚
            _ => 0,
        };
        strategic_tier = strategic_tier.saturating_add(waste);
    }

    // ══ 房规：百搭分级惩罚（mirror JS DUAL_WILD_* / WILD_ON_LEVEL_* / UPGRADED_BOMB_*）══
    // 清空手牌豁免所有百搭罚分（上面已挡）；此处覆盖双百搭矩阵与升档炸。
    if wild_count >= 2 && !clearing {
        let endgame_hand = pctx.my_remaining <= 6;
        let touches_level_natural = cards.iter().any(|c| {
            pctx.card_info.get(c).map(|ci| ci.is_level && !ci.is_wild).unwrap_or(false)
        });
        let nonwild_count = cards.len() - wild_count;
        // 白名单（mirror JS pushDualWildSanctioned）：残局≤6、非级牌参与、
        // 四头炸(非级牌对+2百搭)/三带二/木板/钢板 → 仅轻罚
        let sanctioned_shape = endgame_hand && !touches_level_natural && match combo.kind {
            CombinationKind::Bomb(BombKind::SameRank { n }) => n == 4 && nonwild_count == 2,
            CombinationKind::Ordinary(OrdinaryKind::FullHouse) => nonwild_count == 3,
            CombinationKind::Ordinary(OrdinaryKind::Plate) => true,
            CombinationKind::Ordinary(OrdinaryKind::Tube) => true,
            _ => false,
        };
        if sanctioned_shape {
            strategic_tier = strategic_tier.saturating_add(1); // 白名单用法：仅 −10 级别
        } else {
            // 其余一律重罚：中盘 −600 / 残局 −60；裸双百搭再叠 −200；沾级牌再叠 −250/−20
            let bare = nonwild_count == 0;
            let mut pen: u8 = if endgame_hand { 6 } else { 40 };
            if bare {
                pen = pen.saturating_add(12);
            }
            if touches_level_natural {
                pen = pen.saturating_add(if endgame_hand { 4 } else { 22 });
            }
            strategic_tier = strategic_tier.saturating_add(pen);
        }
    }

    // 升档炸弹（3自然+1百搭）：轻罚（−150/−10）；对手冲刺≤6豁免；级牌rank走沾级罚
    if wild_count == 1 && is_bomb && cards.len() == 4 && !clearing && pctx.min_opp_remaining > 6 {
        let mut nat_rank_count: HashMap<u8, usize> = HashMap::new();
        for c in cards {
            if let Some(ci) = pctx.card_info.get(c) {
                if !ci.is_wild {
                    *nat_rank_count.entry(ci.rank).or_default() += 1;
                }
            }
        }
        let up_three = nat_rank_count.values().max().copied().unwrap_or(0) == 3;
        let up_level = nat_rank_count.keys().any(|&r| r == pctx.level_rank);
        if up_three && !up_level {
            strategic_tier = strategic_tier.saturating_add(if pctx.my_remaining <= 6 { 2 } else { 12 });
        }
    }

    // 王对子惩罚：不能用一对王（大小王）去打小对子
    if matches!(combo.kind, CombinationKind::Ordinary(OrdinaryKind::Pair)) {
        let has_joker = cards.iter().any(|c| c.starts_with("🃏") || c.starts_with("👑"));
        if has_joker {
            strategic_tier = strategic_tier.saturating_add(5);
        }
    }

    // ══ 房规：先出小牌，不要空出大牌（J以上按点数递增罚；清空豁免）══
    // mirror JS scoreLeadPlay「先出小牌」：J −18 / Q −36 / K −54 / A −72 / 王 −108 / 百搭 −90
    if pctx.is_leading && !clearing {
        let has_joker_card = cards.iter().any(|c| c.starts_with("🃏") || c.starts_with("👑"));
        let has_wild = cards.iter().any(|c| pctx.card_info.get(c).map(|ci| ci.is_wild).unwrap_or(false));
        let max_nat = cards
            .iter()
            .filter_map(|c| pctx.card_info.get(c))
            .filter(|ci| !ci.is_wild && !ci.is_joker)
            .map(|ci| ci.rank)
            .max()
            .unwrap_or(0);
        let max_any = if has_joker_card { 16 } else if has_wild { 15 } else { max_nat };
        if max_any >= 11 {
            strategic_tier = strategic_tier.saturating_add((max_any - 10).saturating_mul(2));
        }
    }

    // ══ 房规：接风重奖——队友已全部出完（mirror JS +120 首出 / +180 压敌）══
    if pctx.teammate_remaining == 0 {
        if pctx.is_leading {
            strategic_tier = strategic_tier.saturating_sub(12);
        } else if !pctx.partner_leading {
            strategic_tier = strategic_tier.saturating_sub(18);
        }
    }

    // 级牌炸弹惩罚：级牌应拆开出，不应形成炸弹
    // 房规（mirror JS 天然级牌炸弹规则）：天然级牌四炸中盘近乎禁绝(+50)，残局留作万不得已(+8)
    if is_bomb {
        let all_level_natural = cards.iter().all(|c| {
            pctx.card_info.get(c).map(|ci| ci.is_level && !ci.is_wild).unwrap_or(false)
        });
        if all_level_natural {
            let tier_add = if !clearing && pctx.my_remaining > 6 { 50 } else { 8 };
            strategic_tier = strategic_tier.saturating_add(tier_add);
        }
    }

    // 三带二带级牌对或王对惩罚：一般不这样用，除非万不得已
    if matches!(combo.kind, CombinationKind::Ordinary(OrdinaryKind::FullHouse)) {
        let has_level_pair = cards.iter().filter(|c| {
            pctx.card_info.get(c.as_str()).map(|ci| ci.is_level).unwrap_or(false)
        }).count() >= 2;
        let has_joker_pair = cards.iter().filter(|c| c.starts_with("🃏") || c.starts_with("👑")).count() >= 2;

        if has_level_pair || has_joker_pair {
            let mut penalty = 20;

            let is_last_play = cards.len() >= pctx.my_remaining;
            let opponent_will_win = pctx.opponent_1 && !pctx.is_leading && !is_last_play;

            if is_last_play {
                penalty = 2;
            } else if opponent_will_win {
                penalty = 5;
            }

            strategic_tier = strategic_tier.saturating_add(penalty);
        }

        // 三带二中使用逢人配：逢人配默认应配在大对子上形成更大的三张同
        // 例如：两张9和两张6，逢人配红桃5应配在9上形成3张9的三带二，而非配在6上
        // PlayScore排序中primary越小越好，会导致逢人配配小对子反而优先。
        // 通过给primary更低的FullHouse加惩罚（逆反排序），确保逢人配优先配大对子。
        if wild_count > 0 {
            // primary越小（配小对子）→ penalty越大 → 越不被优先
            // primary越大（配大对子）→ penalty越小 → 越被优先
            let penalty = 20u8.saturating_sub(combo.primary);
            strategic_tier = strategic_tier.saturating_add(penalty);
        }
    }

    // 逢人配不要配在已经是炸弹的牌上：非逢人配已构成炸弹时，逢人配浪费
    if is_bomb && wild_count > 0 {
        let non_wild_count = cards.len() - wild_count;
        if non_wild_count >= 4 {
            let mut rank_counts: HashMap<u8, usize> = HashMap::new();
            for c in cards {
                if let Some(ci) = pctx.card_info.get(c) {
                    if !ci.is_wild {
                        *rank_counts.entry(ci.rank).or_default() += 1;
                    }
                }
            }
            if rank_counts.values().any(|&c| c >= 4) {
                strategic_tier = strategic_tier.saturating_add(3);
            }
        }
    }

    // ══════════════════════════════════════════════════════════════
    // 逢人配使用原则：优先组合多数量炸弹或同花顺
    // 红桃级牌优先组合多数量炸弹（4张+逢人配变5炸）或同花顺
    // 手里仅有1张逢人配，不要为凑4张炸浪费，留到残局凑同花顺更大收益
    // 逢人配组成的同花顺＞普通5炸，但低于6张纯炸弹
    // ══════════════════════════════════════════════════════════════
    if wild_count > 0 && is_bomb {
        // 逢人配用于炸弹：如果是4张+逢人配→5炸，这是好的使用方式
        let non_wild = cards.len() - wild_count;
        if non_wild == 3 && wild_count == 1 {
            // 3张+1逢人配=4炸，如果只有1张逢人配，浪费了
            // 只有1张逢人配时，优先留到残局凑同花顺
            if pctx.combos.bomb_count <= 2 {
                strategic_tier = strategic_tier.saturating_add(2); // 仅1张逢人配，不凑4炸
            }
        }
        // 逢人配组成的同花顺：优先级高于普通5炸但低于6张纯炸弹
        if matches!(combo.kind, CombinationKind::Bomb(BombKind::StraightFlush)) {
            // 同花顺是好的逢人配使用方式，给予奖励
            strategic_tier = strategic_tier.saturating_sub(1); // 略微降低tier，鼓励逢人配组同花顺
        }

        // 三同张带2张逢人配：3张同rank + 2张逢人配 = 5张炸弹
        // 不推荐同时使用2张逢人配（但允许，非禁止）：有更好选择时优先不用
        if matches!(combo.kind, CombinationKind::Bomb(BombKind::SameRank { n: 5 })) && wild_count == 2 {
            let non_wild = cards.len() - wild_count;
            if non_wild == 3 {
                // 惩罚值:轻微不推荐(而非禁止)
                // 原值过重(15/8/5/2)，等同禁止；现改为"不推荐"级别
                let mut penalty = 6;

                let is_last_play = cards.len() >= pctx.my_remaining;
                let opponent_will_win = pctx.opponent_1 && !pctx.is_leading;

                if is_last_play {
                    penalty = 0; // 最后一手完全允许(清牌优先)
                } else if opponent_will_win {
                    penalty = 2; // 关键时刻允许
                } else if pctx.combos.bomb_count == 0 && pctx.min_opp_remaining <= 3 {
                    penalty = 4; // 中等不推荐
                }

                strategic_tier = strategic_tier.saturating_add(penalty);
            }
        }
    }

    // 炸弹绝不能空出，能用小炸弹压牌绝不能用大炸弹压牌
    // 小炸弹（4张）优先于大炸弹（5张+）、同花顺、王炸
    if is_bomb {
        let bomb_size_penalty: u8 = match combo.kind {
            CombinationKind::Bomb(BombKind::SameRank { n }) => {
                n.saturating_sub(4) // 4-card=0, 5-card=1, 6-card=2, ...
            }
            CombinationKind::Bomb(BombKind::StraightFlush) => 4, // 同花顺优先级低于4-7张炸
            CombinationKind::Bomb(BombKind::FourJoker) => 6,      // 王炸最后才用
            _ => 0,
        };
        strategic_tier = strategic_tier.saturating_add(bomb_size_penalty);
    }

    // ══════════════════════════════════════════════════════════════
    // 炸弹使用原则：红线2 - 炸完后全是单张小散，无法连续出牌
    // 炸完后自己全是单张小散，无法连续出牌，不炸，放过本轮
    // ══════════════════════════════════════════════════════════════
    let is_last_play = cards.len() >= pctx.my_remaining;

    // ══ 房规：避免把自己打到「只剩小单张」——非清空出牌后剩余≥3张且全为≤10散单张时递增罚 ══
    // mirror JS −22/张。仅中盘生效（>6张）：残局以多清牌为先，不与此冲突。
    if !is_last_play && pctx.my_remaining > 6 {
        let remaining_after = pctx.my_remaining - cards.len();
        if remaining_after >= 3 {
            let played_set: std::collections::HashSet<&String> = cards.iter().collect();
            let mut rest_cnt: HashMap<u8, usize> = HashMap::new();
            let mut bad_remainder = false;
            for (card_str, ci) in &pctx.card_info {
                if played_set.contains(card_str) {
                    continue;
                }
                if ci.is_wild || ci.is_joker {
                    bad_remainder = true;
                    break;
                }
                *rest_cnt.entry(ci.rank).or_default() += 1;
            }
            if !bad_remainder {
                let mut distinct_singles = 0usize;
                for (&r, &n) in &rest_cnt {
                    if n == 0 {
                        continue;
                    }
                    if n != 1 || r == pctx.level_rank || r > 10 {
                        bad_remainder = true;
                        break;
                    }
                    distinct_singles += 1;
                }
                if !bad_remainder && distinct_singles >= 3 {
                    strategic_tier = strategic_tier.saturating_add(distinct_singles.min(8) as u8);
                }
            }
        }
    }

    // ══ 房规：残局保留最后一个炸弹（mirror JS −400）：清空/对手冲刺≤3/打完剩≤2张 豁免 ══
    if is_bomb && !clearing && !is_last_play
        && pctx.combos.bomb_count == 1
        && pctx.my_remaining <= 6
        && pctx.min_opp_remaining > 3
        && pctx.my_remaining - cards.len() > 2
    {
        strategic_tier = strategic_tier.saturating_add(40);
    }

    // 红线2：炸完后全是单张小散，无法连续出牌，不炸放过本轮
    if is_bomb && !is_last_play {
        let remaining_after = pctx.my_remaining - cards.len();
        // 计算炸完后手牌中单张的数量
        let mut remaining_singles = pctx.singles_count;
        // 减去本次出牌中消耗的单张
        for card in cards {
            if let Some(ci) = pctx.card_info.get(card) {
                let count = pctx.combos.rank_to_count.get(&ci.rank).copied().unwrap_or(0);
                if count == 1 {
                    remaining_singles = remaining_singles.saturating_sub(1);
                }
            }
        }
        // 如果炸完后剩余牌中单张占比超过60%，说明全是散牌，不应炸
        // 剩余1张时不罚（1张牌永远是单张，且是最后一张牌）
        if remaining_after > 1 && remaining_singles as f64 / remaining_after as f64 > 0.6 {
            strategic_tier = strategic_tier.saturating_add(20); // 炸完全是散牌，重罚
        }
    }

    // ══════════════════════════════════════════════════════════════
    // 通用残局散牌惩罚：残局出组合牌型后，剩余手牌全是散牌，应避免
    // 解决"牌局最后剩小牌和单张"问题：防止残局拆组合出大牌后留下无法组合的小散牌。
    // 仅对非炸弹、非单张、非最后出牌生效：单张是在消化散牌（不罚），组合拆散才罚。
    // ══════════════════════════════════════════════════════════════
    if !is_bomb && !is_last_play && pctx.my_remaining <= 8 {
        let remaining_after = pctx.my_remaining - cards.len();
        if remaining_after > 1 {
            let mut remaining_singles = pctx.singles_count;
            for card in cards {
                if let Some(ci) = pctx.card_info.get(card) {
                    let count = pctx.combos.rank_to_count.get(&ci.rank).copied().unwrap_or(0);
                    if count == 1 {
                        remaining_singles = remaining_singles.saturating_sub(1);
                    }
                }
            }
            let ratio_after = remaining_singles as f64 / remaining_after as f64;
            let is_single_play = matches!(combo.kind, CombinationKind::Ordinary(OrdinaryKind::Single));
            // 出非单张组合后剩余散牌占比 > 60%，说明拆了组合留散牌，惩罚
            if ratio_after > 0.6 && !is_single_play {
                strategic_tier = strategic_tier.saturating_add(5);
            }
        }
    }

    // ══════════════════════════════════════════════════════════════
    // 炸弹使用原则：红线3 - 对手只剩1张，炸完不能一手清完
    // 对手只剩1张牌，你炸完不能一手清完，不要开炸，留给队友拦截
    // ══════════════════════════════════════════════════════════════
    if is_bomb && pctx.opponent_1 && !is_last_play {
        strategic_tier = strategic_tier.saturating_add(25); // 对手1张，炸不完不炸
    }

    // ── 精准残局送牌/卡牌：根据敌我剩牌数决定出牌型偏好 ──

    // 对手剩1张 + 领牌：绝不放单张，重罚单张
    if pctx.opponent_1 && pctx.is_leading && !is_last_play {
        if matches!(combo.kind, CombinationKind::Ordinary(OrdinaryKind::Single)) {
            strategic_tier = strategic_tier.saturating_add(30); // 绝不放单
        }
    }

    // 对手剩2张 + 领牌：少放对子，多打单/三带/顺子
    if pctx.opponent_2 && pctx.is_leading && !is_last_play {
        if matches!(combo.kind, CombinationKind::Ordinary(OrdinaryKind::Pair)) {
            strategic_tier = strategic_tier.saturating_add(15); // 少放对子
        }
    }

    // 队友剩1张 + 领牌：全程出单，拆对子、拆三带也要送单
    if pctx.teammate_1 && pctx.is_leading && !is_last_play {
        if matches!(combo.kind, CombinationKind::Ordinary(OrdinaryKind::Single)) {
            strategic_tier = strategic_tier.saturating_sub(10); // 强烈偏好单张
        }
    }

    // 队友剩2张 + 领牌：只打对子
    if pctx.teammate_2 && pctx.is_leading && !is_last_play {
        if matches!(combo.kind, CombinationKind::Ordinary(OrdinaryKind::Pair)) {
            strategic_tier = strategic_tier.saturating_sub(10); // 强烈偏好对子
        }
    }

    // 队友剩3张 + 领牌：打三不带/三带二
    if pctx.teammate_3 && pctx.is_leading && !is_last_play {
        if matches!(combo.kind, CombinationKind::Ordinary(OrdinaryKind::Triple)) {
            strategic_tier = strategic_tier.saturating_sub(10); // 强烈偏好三张
        }
    }

    // Merge split_penalty into strategic_tier for correct ranking.
    // Previously split_penalty was a separate field compared AFTER strategic_tier,
    // which meant a play that breaks a bomb (tier 0, penalty 3) could outrank
    // playing the bomb intact (tier 5, penalty 0).
    // Now penalty is added to tier so the ranking is correct:
    // e.g. breaking only bomb: 0+10=10 > playing bomb: 5+0=5 (break is worse)
    strategic_tier = strategic_tier.saturating_add(penalty);
    strategic_tier = strategic_tier.saturating_add(direct_bomb_penalty); // 兜底：拆炸弹惩罚

    // ── 顺子优先消灭单张 ──
    // 组顺子的原则是尽量消灭单张，比如有三个单张4、6、8，就可以组成45678，
    // 此时可以拆牌优先组顺子。顺子包含≥3个手牌单张时，给予奖励（降低tier）。
    if matches!(combo.kind, CombinationKind::Ordinary(OrdinaryKind::Straight)) {
        let mut singles_consumed = 0u8;
        for card in cards {
            if let Some(&rank) = pctx.combos.card_to_rank.get(card) {
                if pctx.combos.rank_to_count.get(&rank).copied().unwrap_or(0) == 1 {
                    singles_consumed += 1;
                }
            }
        }
        if singles_consumed >= 3 {
            let bonus = singles_consumed - 2; // 3单张→bonus=1, 4→2, 5→3
            strategic_tier = strategic_tier.saturating_sub(bonus);
        }
    }

    // ── 学习参数调整：仅在自对弈训练时生效 ──
    // 当学习参数被设置时，使用学习到的权重调整策略评分。
    // 使用 seat-specific 参数（NS/EW 各自一套），打破自对弈对称性，让优化器能区分参数好坏。
    if let Some(lp) = get_learn_params_for_seat(pctx.actor) {
        let mut multiplier = lp.team_win_weight.clamp(0.1, 10.0);

        if is_bomb {
            if pctx.is_leading {
                // 炸弹保守：领牌出炸弹应更保守（tier↑），避免乱炸。
                // 修复原方向错误：原 *= 0.8 会降tier鼓励出炸弹，与"保守"语义相反。
                multiplier /= lp.bomb_conserve_bias.clamp(0.1, 10.0);
            }
            if pctx.min_opp_remaining <= lp.enemy_low_cards_threshold as usize {
                // 对手少牌时激进出炸弹拦截（tier↓）
                multiplier /= lp.bomb_aggression_when_enemy_low.clamp(0.1, 10.0);
            }
        }

        if pctx.my_remaining <= lp.endgame_hand_count_threshold as usize {
            // 残局更谨慎（tier↑）：避免盲目出牌拆散组合导致残局剩散牌。
            // 修复原方向错误：原 /= 1.2 会降tier鼓励出牌，反而导致拆大牌、剩小单张。
            // determine_strategic_tier 残局领牌出非炸弹已是 tier=0（最高优先），
            // 此处 ×1.2 对 tier=0 无影响，但会提高炸弹 tier，避免残局乱炸。
            multiplier *= lp.endgame_clear_hand_bias.clamp(0.1, 10.0);
        }

        if pctx.partner_leading {
            multiplier *= lp.yield_to_partner_bias.clamp(0.1, 10.0);
        }

        if pctx.is_leading {
            multiplier /= lp.proactive_play_bias.clamp(0.1, 10.0);

            if matches!(combo.kind, CombinationKind::Ordinary(OrdinaryKind::Single)) {
                multiplier /= lp.low_card_dump_bias.clamp(0.1, 10.0);
            }
        }

        if pctx.teammate_remaining <= lp.partner_sprint_threshold as usize && pctx.is_leading {
            multiplier /= lp.first_out_weight.clamp(0.1, 10.0);
        }

        if pctx.teammate_remaining <= lp.partner_sprint_threshold as usize + 2 && !pctx.is_leading {
            multiplier *= lp.second_out_weight.clamp(0.1, 10.0);
        }

        // 用 round 替代 as u8 截断，减少低 tier 值的取整信息丢失。
        strategic_tier = ((strategic_tier as f32 * multiplier).round() as u8).min(255);
    }

    let is_non_level = !cards.iter().any(|c| {
        pctx.card_info.get(c).map(|ci| ci.is_level).unwrap_or(false)
    });

    PlayScore {
        strategic_tier,
        split_penalty: 0, // merged into strategic_tier above
        wild_count,
        is_bomb,
        is_non_level,
        primary: combo.primary,
        cards_len: std::cmp::Reverse(cards.len()),
        sorted_cards: sorted,
    }
}

/// Determine the strategic priority tier for a play.
/// Lower tier = higher priority.
///
/// Bomb conservation strategy:
/// - 0 bombs in hand: normal play
/// - 1 bomb in hand: save for endgame; only use when opponent sprinting or in endgame
/// - 2+ bombs in hand: can use one in early/mid, keep at least one for endgame
///
/// Tier 0: Highest priority (always pick)
/// Tier 1-2: Normal priority
/// Tier 3: Low priority (use only when necessary)
/// Tier 5: Very low priority (essentially never use)
fn determine_strategic_tier(is_bomb: bool, pctx: &PlayContext) -> u8 {
    let endgame = pctx.my_remaining <= 6;
    let approaching_endgame = pctx.my_remaining <= 10 && pctx.my_remaining > 6;
    let teammate_sprinting = pctx.teammate_remaining <= 3;
    let teammate_few = pctx.teammate_remaining <= 6;
    let opponent_sprinting = pctx.min_opp_remaining <= 6;
    let opponent_last = pctx.min_opp_remaining <= 2;
    let bombs = pctx.bomb_count;

    // ══════════════════════════════════════════════════════════════
    // 铁律：绝不能压队友的钢板(Plate)、木板(Tube)、杂顺(Straight)
    // 无论队友剩几张牌、牌值大小，这三种牌型绝对不压
    // ══════════════════════════════════════════════════════════════
    if pctx.partner_leading && !pctx.is_leading {
        let is_partner_special = matches!(pctx.top_play_kind.as_deref(),
            Some(k) if k.contains("Plate") || k.contains("Tube") ||
                (k.contains("Straight") && !k.contains("StraightFlush"))
        );
        if is_partner_special {
            return 6; // 绝不压队友的钢板/木板/杂顺
        }
    }

    // ══════════════════════════════════════════════════════════════
    // 铁律：不能压队友的级牌
    // 级牌是当前打几的关键牌，压队友级牌等于浪费我方资源
    // 例外：如果是让队友接牌（队友出小级牌，你出大级牌，队友再出更大的→协同夺牌权）
    //   目前统一禁止压队友级牌，避免误判
    // ══════════════════════════════════════════════════════════════
    if pctx.partner_leading && !pctx.is_leading {
        let top_is_level = pctx.top_play_value.map(|v| v == pctx.level_rank).unwrap_or(false);
        if top_is_level {
            return 6; // 绝不压队友的级牌
        }
    }

    // ══════════════════════════════════════════════════════════════
    // 炸弹使用原则：队友打出炸弹，你有更大炸弹不要随便盖
    // 队友炸完刚拿到牌权，你再盖炸会夺走他的出牌机会
    // 除非对手马上要反炸队友，才续炸控场
    // ══════════════════════════════════════════════════════════════
    if is_bomb && pctx.partner_just_bombed && !pctx.is_leading {
        return 6; // 队友刚出炸弹拿牌权，不要盖
    }

    // ══════════════════════════════════════════════════════════════
    // 铁律：队友出大小王，绝对不能压
    // 大小王是最大单牌，队友出王说明有明确意图，压队友王等于内耗
    // ══════════════════════════════════════════════════════════════
    if pctx.partner_leading && !pctx.is_leading {
        let top_is_joker = pctx.top_play_value.map(|v| v >= 15).unwrap_or(false);
        if top_is_joker {
            return 6; // 绝不压队友的大小王
        }
    }

    // ══════════════════════════════════════════════════════════════
    // 炸弹使用原则：对手出炸弹压制队友，必须反炸
    // 队友被敌方炸弹压住，你持有更大炸弹时立刻跟炸，夺回牌权给队友跑牌
    // ══════════════════════════════════════════════════════════════
    let top_is_opponent_bomb = !pctx.is_leading && !pctx.partner_leading &&
        pctx.top_play_kind.as_deref().map(|k| k.starts_with("Bomb")).unwrap_or(false);
    if is_bomb && top_is_opponent_bomb {
        return 0; // 对手炸弹压队友，必须反炸
    }

    // ══════════════════════════════════════════════════════════════
    // 残局拦截：对方发小牌（小单张/小对子），最后一家必须压牌
    // 对手剩≤6张且牌值≤8时，无条件压牌拦截，绝不让对方顺牌出完
    // ══════════════════════════════════════════════════════════════
    if opponent_sprinting && !pctx.is_leading && !pctx.partner_leading {
        if pctx.top_play_value.unwrap_or(15) <= 8 {
            return 0; // 残局拦截对手小牌，必须压
        }
    }

    // ══════════════════════════════════════════════════════════════
    // 最高优先级：对手只剩1张 → 必须全力拦截，绝不放单
    // 但队友领牌时不能压队友的牌
    // ══════════════════════════════════════════════════════════════
    if pctx.opponent_1 && !pctx.is_leading && !pctx.partner_leading {
        // 对手剩1张，跟牌时必须出牌压住
        if is_bomb {
            return 0; // 炸弹拦截，最高优先级
        } else {
            return 0; // 任何牌都出，绝不让对手走单
        }
    }

    // ══════════════════════════════════════════════════════════════
    // 对手剩1张 + 领牌：绝不放单张，打对子/三带/顺子逼对手拆牌
    // ══════════════════════════════════════════════════════════════
    if pctx.opponent_1 && pctx.is_leading {
        // 领牌时绝不放单，出对子/三带/顺子等牌型
        if is_bomb {
            // 如果队友牌差（≥6张），直接开炸控场
            if pctx.teammate_remaining >= 6 {
                return 0; // 开炸控场，打对子/三带
            }
            return 3; // 保留炸弹，不出炸弹
        } else {
            return 0; // 出非炸弹牌型（对子/三带/顺子优先，单张会受惩罚）
        }
    }

    // ══════════════════════════════════════════════════════════════
    // 对手剩2张 → 少放对子，多打单/三带/顺子，逼对手拆牌
    // 但队友领牌时不能压队友的牌
    // ══════════════════════════════════════════════════════════════
    if pctx.opponent_2 && !pctx.is_leading && !pctx.partner_leading {
        if is_bomb {
            return 0; // 炸弹拦截
        } else {
            return 0; // 出牌拦截
        }
    }

    // ══════════════════════════════════════════════════════════════
    // 对手剩3-5张（危险区间）→ 手里留炸弹防守，不放小牌型
    // 但队友领牌时不能压队友的牌
    // ══════════════════════════════════════════════════════════════
    if pctx.opponent_3_5 && !pctx.is_leading && !pctx.partner_leading {
        if is_bomb {
            return 0; // 炸弹拦截
        } else {
            return 0; // 出牌拦截
        }
    }
    if pctx.opponent_3_5 && pctx.is_leading {
        // 对手3-5张，领牌时保留炸弹防守
        if is_bomb {
            return 3; // 保留炸弹
        } else {
            return 0; // 出非炸弹
        }
    }

    // ══════════════════════════════════════════════════════════════
    // 炸弹使用原则：炸五不炸四，炸七不炸八
    // 对手剩5张（大概率4炸+单）→ 必炸拦截
    // 对手剩7张（多为一炸+两手牌）→ 必炸拦截
    // 对手剩4张（极可能本身就是4张炸）→ 慎炸，炸完无牌收尾等于白送
    // 对手剩8张（可能是两炸或炸+牌）→ 缓观，不急炸
    // ══════════════════════════════════════════════════════════════
    if is_bomb && !pctx.is_leading && !pctx.partner_leading {
        if pctx.opponent_5 || pctx.opponent_7 {
            return 0; // 必炸：5/7张大概率一炸+散牌，不炸直接头游
        }
        if pctx.opponent_4 || pctx.opponent_8 {
            return 4; // 慎炸/缓观：4张可能是炸弹，8张可能是两炸
        }
    }

    // ══════════════════════════════════════════════════════════════
    // 队友剩1张 → 全程出单，拆对子、拆三带也要送单
    // ══════════════════════════════════════════════════════════════
    if pctx.teammate_1 && pctx.is_leading {
        // 队友剩1张，领牌时优先出单张送队友
        if is_bomb {
            return 5; // 不出炸弹，送单张
        } else {
            return 0; // 出单张送队友（单张会获得bonus）
        }
    }
    if pctx.teammate_1 && !pctx.is_leading {
        if pctx.partner_leading {
            return 6; // 队友领牌，不压
        } else {
            // 对手领牌，用最小牌压，然后出单张送队友
            if is_bomb {
                return 4; // 不乱用炸弹
            } else {
                return 0; // 压牌后出单送队友
            }
        }
    }

    // ══════════════════════════════════════════════════════════════
    // 队友剩2张 → 只打对子
    // ══════════════════════════════════════════════════════════════
    if pctx.teammate_2 && pctx.is_leading {
        if is_bomb {
            return 5; // 不出炸弹，送对子
        } else {
            return 0; // 出对子送队友（对子会获得bonus）
        }
    }
    if pctx.teammate_2 && !pctx.is_leading {
        if pctx.partner_leading {
            return 6; // 队友领牌，不压
        } else {
            if is_bomb {
                return 4;
            } else {
                return 0; // 压牌后出对子送队友
            }
        }
    }

    // ══════════════════════════════════════════════════════════════
    // 队友剩3张 → 打三不带/三带二
    // ══════════════════════════════════════════════════════════════
    if pctx.teammate_3 && pctx.is_leading {
        if is_bomb {
            return 5; // 不出炸弹，送三张
        } else {
            return 0; // 出三张送队友（三张会获得bonus）
        }
    }
    if pctx.teammate_3 && !pctx.is_leading {
        if pctx.partner_leading {
            return 6;
        } else {
            if is_bomb {
                return 4;
            } else {
                return 0;
            }
        }
    }

    // ══════════════════════════════════════════════════════════════
    // 通用残局逻辑（兜底）
    // ══════════════════════════════════════════════════════════════

    // 残局对手冲刺：必须拦截
    if opponent_last && !pctx.is_leading {
        if is_bomb { return 0; } else { return 0; }
    }
    if opponent_sprinting && !pctx.is_leading {
        if is_bomb { return 0; } else { return 0; }
    }

    // 残局队友冲刺：领牌时送队友需要的牌型
    if teammate_sprinting && pctx.is_leading {
        if is_bomb { return 3; } else { return 0; }
    }

    // 残局队友冲刺：跟牌时判断队友要什么
    if teammate_sprinting && !pctx.is_leading {
        if pctx.partner_leading {
            return 6;
        } else {
            if is_bomb { return 4; } else { return 0; }
        }
    }

    // 残局领牌：对手冲刺时出大牌，队友冲刺时出小牌
    if endgame && pctx.is_leading {
        if opponent_sprinting {
            if is_bomb { return 0; } else { return 0; }
        }
        if teammate_sprinting {
            if is_bomb { return 3; } else { return 0; }
        }
        if is_bomb { return 1; } else { return 0; }
    }

    // 铁律：队友领牌时，大牌不压，小牌可顺
    if pctx.partner_leading && !pctx.is_leading {
        let top_is_bomb = pctx.top_play_kind.as_deref()
            .map(|k| k.starts_with("Bomb"))
            .unwrap_or(false);
        if top_is_bomb {
            return 6; // 绝不压队友的炸弹
        }
        let big_threshold: u8 = match pctx.top_play_kind.as_deref() {
            Some(kind) if kind.contains("Single") => 13,
            _ => 10,
        };
        let top_is_big = pctx.top_play_value.unwrap_or(0) >= big_threshold;
        // 队友出了大牌 → 绝不压，即使是残局对手冲刺也不压（队友有能力处理）
        if top_is_big {
            return 6;
        }
        // 队友出了小牌且对手冲刺 → 需要拦截，但要先判定是否压队友
        if opponent_sprinting {
            if is_bomb { return 6; } else { return 0; }
        }
        if is_bomb { return 6; } else { return 0; }
    }

    // 开局炸弹少时不压王和级牌
    let top_is_joker_or_level = matches!(pctx.top_play_value, Some(14 | 15 | 16));
    let early_game = pctx.my_remaining > 10;
    if is_bomb && top_is_joker_or_level && bombs < 3 && early_game {
        if opponent_sprinting { return 0; }
        return 7;
    }

    // ══════════════════════════════════════════════════════════════
    // 炸弹使用原则：红线1 - 顺风大优不浪炸
    // 仅当炸弹稀缺（1个）时严格保留；2+个炸弹时即使顺风也要积极夺牌权
    // ══════════════════════════════════════════════════════════════
    if is_bomb && pctx.is_wind_advantage && !opponent_sprinting && bombs == 1 {
        return 7; // 仅1个炸弹+顺风大优，保留防突袭
    }

    // ══════════════════════════════════════════════════════════════
    // 炸弹使用原则：少炸不主动开花
    // 全场仅持有1颗炸弹，绝不主动先出炸弹，留作残局拦截用
    // ══════════════════════════════════════════════════════════════
    if is_bomb && bombs == 1 && pctx.is_leading {
        return 8; // 仅1颗炸弹，绝不主动开花
    }

    // ══════════════════════════════════════════════════════════════
    // 炸弹使用原则：队友牌烂，主动开炸铺路
    // 队友全程过小牌、无上手机会，主动用炸弹拿权，持续打队友适配牌型
    // ══════════════════════════════════════════════════════════════
    if is_bomb && pctx.teammate_weak && !pctx.is_leading && !pctx.partner_leading {
        if bombs >= 2 {
            return 0; // 队友牌烂，主动开炸拿牌权铺路
        }
    }

    // ══════════════════════════════════════════════════════════════
    // 炸弹使用原则：大炸封头（炸弹分级留存）
    // 6张及以上大炸、天王炸：全程保留，只用来拦截对手最后冲刺
    // 注意：具体炸弹大小惩罚在score_play中通过bomb_size_penalty实现
    // 仅当只有大炸没有小炸时保留；有小炸时优先用小炸
    // ══════════════════════════════════════════════════════════════
    if is_bomb && !endgame && !opponent_sprinting {
        if pctx.big_bomb_count > 0 && pctx.small_bomb_count == 0 {
            return 6; // 只有大炸没有小炸，非残局非冲刺时保留
        }
    }

    // Bomb conservation: 2+ bombs（优先判断，确保积极夺牌权）
    // 有2个及以上炸弹时，只保留1个到残局控牌，其他炸弹及时夺取出牌权
    if is_bomb && bombs >= 2 {
        if opponent_sprinting { return 0; }
        if endgame { return 2; }          // 残局保留1个炸弹控牌
        if approaching_endgame { return 2; } // 接近残局时更积极使用炸弹
        return 1;                          // 非残局积极使用炸弹夺取出牌权
    }

    // Bomb conservation: only 1 bomb
    if is_bomb && bombs == 1 {
        if opponent_sprinting { return 0; }
        if endgame { return 2; }
        if approaching_endgame { return 6; }
        // 非残局非冲刺，1个炸弹谨慎使用
        if pctx.is_leading {
            return 5; // 领牌时1个炸弹保留
        } else {
            return 3; // 跟牌时1个炸弹可适度使用夺牌权
        }
    }

    // Normal (non-bomb or 0 bombs)
    if opponent_sprinting {
        if is_bomb { 0 } else { 1 }
    } else if endgame && !pctx.is_leading {
        if is_bomb { 2 } else { 0 }
    } else if teammate_few && pctx.is_leading {
        if is_bomb { 2 } else { 0 }
    } else {
        if is_bomb { 1 } else { 0 }
    }
}

// ── Legacy API (kept for backward compatibility with tests) ────────────

/// Legacy pick_playing without game context. Used by tests.
/// Prefer `pick_playing_with_context` for production use.
#[allow(dead_code)]
fn pick_playing(legal: &[PlayerAction], ctx: RuleContext) -> Result<PlayerAction, String> {
    let plays: Vec<&PlayerAction> = legal
        .iter()
        .filter(|a| matches!(a, PlayerAction::Play { .. }))
        .collect();

    if plays.is_empty() {
        return legal
            .iter()
            .find(|a| matches!(a, PlayerAction::Pass))
            .cloned()
            .ok_or_else(|| "suggest: no pass in legal".into());
    }

    let mut best: Option<&PlayerAction> = None;
    for a in plays {
        if playing_cmp(a, best, ctx)? == Ordering::Less {
            best = Some(a);
        }
    }
    best.cloned()
        .ok_or_else(|| "suggest: empty play list".into())
}

/// Prefer `a` over `b` if Ordering::Less.
fn playing_cmp(
    a: &PlayerAction,
    b: Option<&PlayerAction>,
    ctx: RuleContext,
) -> Result<Ordering, String> {
    let Some(b) = b else {
        return Ok(Ordering::Less);
    };
    Ok(play_key(a, ctx)?.cmp(&play_key(b, ctx)?))
}

/// Sort key for preferred playing suggestion:
/// 1) fewer wildcard cards first (0 < 1 < 2 ...)
/// 2) non-bomb before bomb
/// 3) smaller combination primary value first
/// 4) if same primary, more cards first
/// 5) lexicographic card symbols for deterministic tie-break
fn play_key(
    a: &PlayerAction,
    ctx: RuleContext,
) -> Result<(usize, bool, u8, std::cmp::Reverse<usize>, Vec<String>), String> {
    match a {
        PlayerAction::Play {
            cards,
            wild_targets,
        } => {
            let combo = CombinationParser::parse(cards, wild_targets.as_deref(), ctx)?;
            let wild_count = cards.iter().try_fold(0usize, |acc, s| {
                let c = parse_card_symbol(s)?;
                Ok::<usize, String>(acc + usize::from(is_wild(c, ctx)))
            })?;
            let is_bomb = matches!(combo.class(), CombinationClass::Bomb);
            let mut sorted = cards.clone();
            sorted.sort();
            Ok((
                wild_count,
                is_bomb,
                combo.primary,
                std::cmp::Reverse(cards.len()),
                sorted,
            ))
        }
        _ => Err("suggest: play_key expects Play action".into()),
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::card::HandLevel;
    use crate::game::rules::combination_parser::CombinationParser;
    use crate::game::types::{HandState, PlayState};

    fn ctx() -> RuleContext {
        RuleContext {
            hand_level: HandLevel::Two,
        }
    }

    fn mk_playing_state(
        actor: Seat,
        actor_hand: Vec<&str>,
        top_cards: Option<(Seat, Vec<&str>)>,
    ) -> TableGameState {
        let mut s = TableGameState::new("t_suggest".into());
        s.phase = GamePhase::Playing;
        s.turn_seat = actor;
        s.leader_seat = actor;

        let mut hand = HandState::new(HandLevel::Two);
        hand.hands.insert(
            actor,
            actor_hand.into_iter().map(ToString::to_string).collect(),
        );
        for seat in Seat::ALL {
            hand.hands.entry(seat).or_insert_with(Vec::new);
        }

        if let Some((seat, cards)) = top_cards {
            let cards: Vec<String> = cards.into_iter().map(ToString::to_string).collect();
            let combo = CombinationParser::parse(&cards, None, ctx()).unwrap();
            hand.trick.top_play = Some(PlayState {
                seat,
                cards: cards.clone(),
                wild_targets: None,
                combination: combo,
            });
            hand.trick.last_play_seat = Some(seat);
        }

        s.hand = Some(hand);
        s
    }

    #[test]
    fn prefers_non_bomb_over_bomb() {
        let legal = vec![
            PlayerAction::Play {
                cards: vec!["♠3".into()],
                wild_targets: None,
            },
            PlayerAction::Play {
                cards: vec!["♠4".into(), "♥4".into(), "♦4".into(), "♣4".into()],
                wild_targets: None,
            },
        ];
        let picked = pick_playing(&legal, ctx()).unwrap();
        assert_eq!(
            picked,
            PlayerAction::Play {
                cards: vec!["♠3".into()],
                wild_targets: None,
            }
        );
    }

    #[test]
    fn prefers_smaller_primary_value() {
        let legal = vec![
            PlayerAction::Play {
                cards: vec!["♠7".into()],
                wild_targets: None,
            },
            PlayerAction::Play {
                cards: vec!["♠9".into()],
                wild_targets: None,
            },
        ];
        let picked = pick_playing(&legal, ctx()).unwrap();
        assert_eq!(
            picked,
            PlayerAction::Play {
                cards: vec!["♠7".into()],
                wild_targets: None,
            }
        );
    }

    #[test]
    fn prefers_more_cards_when_primary_is_same() {
        let legal = vec![
            PlayerAction::Play {
                cards: vec!["♠7".into()],
                wild_targets: None,
            },
            PlayerAction::Play {
                cards: vec!["♠7".into(), "♥7".into()],
                wild_targets: None,
            },
        ];
        let picked = pick_playing(&legal, ctx()).unwrap();
        assert_eq!(
            picked,
            PlayerAction::Play {
                cards: vec!["♠7".into(), "♥7".into()],
                wild_targets: None,
            }
        );
    }

    #[test]
    fn suggest_follow_play_prefers_non_bomb_when_both_legal() {
        let state = mk_playing_state(
            Seat::E,
            vec!["♠7", "♠8", "♥8", "♦8", "♣8"],
            Some((Seat::N, vec!["♠6"])),
        );
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        assert_eq!(
            picked,
            PlayerAction::Play {
                cards: vec!["♠7".into()],
                wild_targets: None,
            }
        );
    }

    #[test]
    fn suggest_returns_pass_when_no_play_can_beat_top() {
        let state = mk_playing_state(Seat::E, vec!["♠3"], Some((Seat::N, vec!["♠A"])));
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        assert_eq!(picked, PlayerAction::Pass);
    }

    #[test]
    fn prefers_level_cards_before_non_level() {
        let state = mk_playing_state(Seat::E, vec!["♠2", "♠7"], Some((Seat::N, vec!["♠6"])));
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        assert_eq!(
            picked,
            PlayerAction::Play {
                cards: vec!["♠2".into()],
                wild_targets: None,
            }
        );
    }

    #[test]
    fn avoids_splitting_bomb_when_leading() {
        // Hand has ♠3,♥3,♦3,♣3 (bomb) + ♠5 (single)
        // When leading, should prefer ♠5 over ♠3 (breaking bomb)
        let state = mk_playing_state(
            Seat::E,
            vec!["♠3", "♥3", "♦3", "♣3", "♠5"],
            None, // leading
        );
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        assert_eq!(
            picked,
            PlayerAction::Play {
                cards: vec!["♠5".into()],
                wild_targets: None,
            }
        );
    }

    #[test]
    fn endgame_leading_prefers_larger_non_bomb_combos() {
        // Endgame: 5 cards, can play pair (3,3) or single (5)
        // Should prefer pair to clear more cards
        let state = mk_playing_state(
            Seat::E,
            vec!["♠3", "♥3", "♠5", "♠6", "♠7"],
            None, // leading
        );
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        // Should pick the pair (♠3,♥3) over single ♠5
        match &picked {
            PlayerAction::Play { cards, .. } => {
                assert_eq!(cards.len(), 2, "Endgame should prefer pair to clear more cards");
            }
            _ => panic!("Expected Play action"),
        }
    }

    #[test]
    fn opponent_sprinting_prefers_bomb_intercept() {
        // Hand: bomb + single. Opponent has 2 cards.
        let mut state = mk_playing_state(
            Seat::E,
            vec!["♠3", "♥3", "♦3", "♣3", "♠5"],
            Some((Seat::N, vec!["♠K"])), // opponent leading with K
        );
        // Set opponent to 2 cards
        if let Some(ref mut hand) = state.hand {
            hand.hands.insert(Seat::N, vec!["♠A".to_string(), "♠K".to_string()]);
        }
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        // Should prefer bomb to intercept
        match &picked {
            PlayerAction::Play { cards, .. } => {
                assert!(
                    cards.len() >= 4,
                    "Opponent sprinting should prefer bomb, got {} cards",
                    cards.len()
                );
            }
            _ => panic!("Expected Play action, got {:?}", picked),
        }
    }
}