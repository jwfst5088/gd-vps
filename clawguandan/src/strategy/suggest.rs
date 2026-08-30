//! Deterministic choice among [`super::enumerate_legal_actions`] for a single actor.
//!
//! **1:1 port of the JavaScript rule standard** (`gd-cloudflare-main/src/bot-advanced.js`):
//! - [`decide_follow`]      ⇔ JS `decideAdvancedPlay` (pass/play heuristic, JS 460-611)
//! - [`find_best_play_follow`] ⇔ JS `findBestPlay` (sort + hard guards, JS 622-725)
//! - [`score_follow`]       ⇔ JS `scorePlay` (follow scoring, base 100, JS 732-1157)
//! - [`score_lead`]         ⇔ JS `scoreLeadPlay` (lead scoring, base 50, JS 1465-1992)
//! - [`analyze_hand_combos`] ⇔ JS `analyzeHandCombos` (JS 1215-1320)
//! - [`classify_bomb_split`] ⇔ JS `classifyBombSplit` (JS 1330-1363)
//! - [`split_penalty`]      ⇔ JS `splitPenalty` (JS 1381-1457)
//!
//! Scoring contract (mirrors JS exactly):
//! - `f32` scores, **higher is better**.
//! - Probability-dependent terms (`probOpponentCanFollow`, `probOpponentHasBomb`,
//!   `calculateGameWinProb`) are ACTIVE: 牌踪器（rank 级剩余牌统计）已激活，
//!   公式与 JS HandTracker/ProbabilisticReasoner 1:1；受 `hand_tracker_enabled` 开关控制。
//! - Seat remaining counts are read directly from `state.hand.hands` (`len`);
//!   `min_opp_remaining` = min of the two opponents; teammate = `Seat::teammate`.
//!
//! The old Rust-only rules (tier 0/1 wild slots, stall penalty, endgame tidiness bonus,
//! level-empty-single penalty, redundant-wild penalty, dedicated clearing fast path, …)
//! were removed: JS is the single source of truth. Clearing emerges from the `+10000`
//! in-score bonus plus the "last card → forced play" exception (JS 488-497).

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use crate::bot::plugins::advanced_bot::AdvancedBotParams;
use crate::domain::Seat;
use crate::game::card::{
    is_wild, level_order_value, natural_rank_value, parse_card_symbol, Rank, RuleContext, Suit,
};
use crate::game::engine::PlayerAction;
use crate::game::rules::combination_parser::{
    BombKind, Combination, CombinationClass, CombinationKind, CombinationParser, OrdinaryKind,
};
use crate::game::types::{
    GamePhase, HandHistoryEntry, HandState, HistoryActionKind, PlayState, TableGameState,
};

use super::enumerate_legal_actions;

// ── Learning params (global store for AI self-learning) ────────────────
// Interface kept for `crate::learning` (set_learn_params_for_teams). When set, the
// heuristic weights come from training; otherwise the JS `TRAINED_PARAMS` defaults
// (the rule standard) are used.
static LEARN_PARAMS: Mutex<Option<AdvancedBotParams>> = Mutex::new(None);
static LEARN_PARAMS_NS: Mutex<Option<AdvancedBotParams>> = Mutex::new(None);
static LEARN_PARAMS_EW: Mutex<Option<AdvancedBotParams>> = Mutex::new(None);

/// Set the global learning parameters. Pass `None` to reset to defaults.
pub fn set_learn_params(params: Option<AdvancedBotParams>) {
    if let Ok(mut lock) = LEARN_PARAMS.lock() {
        *lock = params;
    }
}

/// Set team-specific learning parameters for asymmetric self-play evaluation.
pub fn set_learn_params_for_teams(ns: Option<AdvancedBotParams>, ew: Option<AdvancedBotParams>) {
    if let Ok(mut lock) = LEARN_PARAMS_NS.lock() {
        *lock = ns;
    }
    if let Ok(mut lock) = LEARN_PARAMS_EW.lock() {
        *lock = ew;
    }
}

fn get_learn_params() -> Option<AdvancedBotParams> {
    LEARN_PARAMS.lock().ok().and_then(|lock| lock.clone())
}

fn get_params_for_seat(seat: Seat) -> AdvancedBotParams {
    let team_lock = match seat {
        Seat::S | Seat::N => LEARN_PARAMS_NS.lock(),
        _ => LEARN_PARAMS_EW.lock(),
    };
    team_lock
        .ok()
        .and_then(|l| l.clone())
        .or_else(get_learn_params)
        .unwrap_or_else(js_trained_params)
}

/// JS `TRAINED_PARAMS` (bot-advanced.js L49-69) — the rule-standard defaults.
/// 房规基线：训练器(learning)起点/评估基线也以此为准（含冲刺=6 等用户房规）。
pub(crate) fn js_trained_params() -> AdvancedBotParams {
    AdvancedBotParams {
        team_win_weight: 1.0,
        first_out_weight: 0.7657717,
        second_out_weight: 0.9,
        yield_to_partner_bias: 1.4,
        partner_sprint_threshold: 2,
        bomb_conserve_bias: 0.8,
        bomb_aggression_when_enemy_low: 2.2148905,
        enemy_low_cards_threshold: 6, // 用户房规：冲刺 = 任一对手剩 ≤6 张（JS 原值 3）
        endgame_hand_count_threshold: 6,
        endgame_clear_hand_bias: 1.2,
        proactive_play_bias: 1.1,
        low_card_dump_bias: 1.4,
        pass_stall_penalty: 0.9,
        hand_tracker_enabled: true,
        prob_threshold_for_bomb: 0.6,
        prob_threshold_for_intercept: 0.4,
        enable_reason_trace: true,
    }
}

// ── JS 房规常量 (bot-advanced.js L14-42), 按值移植 ──────────────────────

/// JS `DUAL_WILD_HAND_ENDGAME`: 手牌 ≤ 此值视为残局（百搭/拆炸房规专用，硬编码 6）
const DUAL_WILD_HAND_ENDGAME: usize = 6;
/// JS `DUAL_WILD_PENALTY_MIDGAME`: 中盘双百搭同出重罚（近乎禁绝）
const DUAL_WILD_PENALTY_MIDGAME: f32 = 600.0;
/// JS `DUAL_WILD_PENALTY_ENDGAME`: 残局双百搭同出罚
const DUAL_WILD_PENALTY_ENDGAME: f32 = 60.0;
/// JS `DUAL_WILD_CANDIDATE_HAND_MAX`: 仅 movegen（JS generatePlaysOfType）使用；
/// 本文件不枚举候选（由 `enumerate_legal_actions` 负责），保留仅为 1:1 对应。
#[allow(dead_code)]
const DUAL_WILD_CANDIDATE_HAND_MAX: usize = 6;
/// JS `UPGRADED_BOMB_WILD_PENALTY_MIDGAME`: 天然炸弹贴百搭升档中盘重罚
const UPGRADED_BOMB_WILD_PENALTY_MIDGAME: f32 = 150.0;
/// JS `UPGRADED_BOMB_WILD_PENALTY_ENDGAME`: 升档残局轻罚
const UPGRADED_BOMB_WILD_PENALTY_ENDGAME: f32 = 10.0;
/// JS `WILD_ON_LEVEL_PENALTY_MIDGAME`: 百搭落级牌中盘重罚
const WILD_ON_LEVEL_PENALTY_MIDGAME: f32 = 250.0;
/// JS `WILD_ON_LEVEL_PENALTY_ENDGAME`: 百搭落级牌残局轻罚
const WILD_ON_LEVEL_PENALTY_ENDGAME: f32 = 20.0;
/// JS `WILD_PLAIN_PAIR_PENALTY_MIDGAME`: 百搭配普通单张成普通对中盘重罚
const WILD_PLAIN_PAIR_PENALTY_MIDGAME: f32 = 300.0;
/// JS `WILD_PAIR_PENALTY_ENDGAME`: 百搭配对子残局轻罚
const WILD_PAIR_PENALTY_ENDGAME: f32 = 15.0;
/// JS `BARE_DUAL_WILD_EXTRA_PENALTY`: 裸出双百搭额外加重
const BARE_DUAL_WILD_EXTRA_PENALTY: f32 = 200.0;

// JS 内联分值常量（scorePlay base 100 / scoreLeadPlay base 50 / penalty×20 / 清空+10000）
const BASE_FOLLOW_SCORE: f32 = 100.0;
const BASE_LEAD_SCORE: f32 = 50.0;
const SPLIT_PENALTY_SCALE: f32 = 20.0;
const CLEAR_HAND_BONUS: f32 = 10000.0;
const BANNED_SCORE: f32 = 99999.0;

// ── Card helpers (cards.js getRank / rankValue / NATURAL_RANK 等价物) ───

/// Pre-parsed card metadata for hot loops.
#[derive(Clone, Copy, Debug)]
struct CardMeta {
    rank: Rank,
    /// `natural_rank_value` (NATURAL_RANK: 2→2 … A→14); jokers → None.
    natural: Option<u8>,
    is_wild: bool,
    is_joker: bool,
    /// JS `rankValue` comparison scale: 3=3 … A=14, 2=15, 小王=16, 大王=17.
    rank_value: u8,
}

fn meta_of(sym: &str, ctx: RuleContext) -> Option<CardMeta> {
    let card = parse_card_symbol(sym).ok()?;
    Some(CardMeta {
        rank: card.rank,
        natural: natural_rank_value(card.rank).ok(),
        is_wild: is_wild(card, ctx),
        is_joker: card.suit == Suit::Joker,
        rank_value: rank_value_js(card.rank),
    })
}

/// JS `rankValue` (bot-advanced.js L116-119): 3=3 … A=14, 2=15, B=16, R=17.
fn rank_value_js(rank: Rank) -> u8 {
    match rank {
        Rank::Two => 15,
        Rank::Three => 3,
        Rank::Four => 4,
        Rank::Five => 5,
        Rank::Six => 6,
        Rank::Seven => 7,
        Rank::Eight => 8,
        Rank::Nine => 9,
        Rank::Ten => 10,
        Rank::J => 11,
        Rank::Q => 12,
        Rank::K => 13,
        Rank::A => 14,
        Rank::BlackJoker => 16,
        Rank::RedJoker => 17,
    }
}

fn is_joker_sym(sym: &str) -> bool {
    sym.starts_with('🃏')
}

// ── Hand combo analysis (JS analyzeHandCombos L1215-1320) ──────────────

#[derive(Clone, Debug, Default)]
#[allow(dead_code)] // straight_flush_count/wild_count 仅参与 bomb_count 推导（JS 字段 1:1 保留）
struct HandCombos {
    /// symbol → natural rank value (non-wild, non-joker only) — JS `cardToRank`.
    card_to_rank: HashMap<String, u8>,
    /// natural rank → count in hand (non-wild, non-joker) — JS `rankToCount`.
    rank_to_count: HashMap<u8, usize>,
    /// consecutive triple pairs (plate candidates) — JS `platePairs`.
    plate_pairs: Vec<(u8, u8)>,
    /// consecutive pair runs of 3 (tube candidates) — JS `tubeTriples`.
    tube_triples: Vec<(u8, u8, u8)>,
    /// bombRanks.len + wild-assisted + straight-flush candidates — JS `bombCount`.
    bomb_count: usize,
    /// JS `straightFlushCount`.
    straight_flush_count: usize,
    /// cards whose rank appears exactly once in hand (wilds/jokers excluded) — JS `singlesCount`.
    singles_count: usize,
    /// JS `wildCount`.
    wild_count: usize,
}

fn analyze_hand_combos(hand: &[String], ctx: RuleContext) -> HandCombos {
    let mut card_to_rank: HashMap<String, u8> = HashMap::new();
    let mut rank_to_count: HashMap<u8, usize> = HashMap::new();
    let mut wild_count = 0usize;
    let mut red_joker_count = 0usize;
    let mut black_joker_count = 0usize;

    for card in hand {
        let Some(c) = parse_card_symbol(card).ok() else {
            continue;
        };
        if is_wild(c, ctx) {
            wild_count += 1;
            continue;
        }
        // 天王炸（2大王+2小王）也是炸弹——此前 jokers 因无 natural_rank 被完全跳过，
        // 导致"手里唯一的炸弹是4王"时 bomb_count=0，守卫①（唯一炸保留）从不生效
        // （与 CF bot-advanced.js 同步修复）。
        if c.suit == Suit::Joker {
            match c.rank {
                Rank::RedJoker => red_joker_count += 1,
                Rank::BlackJoker => black_joker_count += 1,
                _ => {}
            }
            continue;
        }
        if let Ok(nv) = natural_rank_value(c.rank) {
            *rank_to_count.entry(nv).or_default() += 1;
            card_to_rank.insert(card.clone(), nv);
        }
    }

    let mut bomb_ranks: Vec<u8> = rank_to_count
        .iter()
        .filter(|&(_, &c)| c >= 4)
        .map(|(&r, _)| r)
        .collect();
    let mut triple_ranks: Vec<u8> = rank_to_count
        .iter()
        .filter(|&(_, &c)| c >= 3)
        .map(|(&r, _)| r)
        .collect();
    let mut pair_ranks: Vec<u8> = rank_to_count
        .iter()
        .filter(|&(_, &c)| c >= 2)
        .map(|(&r, _)| r)
        .collect();
    bomb_ranks.sort_unstable();
    triple_ranks.sort_unstable();
    pair_ranks.sort_unstable();

    // platePairs: consecutive triples (JS L1246-1251)
    let mut plate_pairs = Vec::new();
    for w in triple_ranks.windows(2) {
        if w[1] - w[0] == 1 {
            plate_pairs.push((w[0], w[1]));
        }
    }
    // tubeTriples: consecutive pair runs of 3 (JS L1255-1260)
    let mut tube_triples = Vec::new();
    for w in pair_ranks.windows(3) {
        if w[1] - w[0] == 1 && w[2] - w[1] == 1 {
            tube_triples.push((w[0], w[1], w[2]));
        }
    }

    // straight flush candidates (JS L1263-1285)
    let mut straight_flush_count = 0usize;
    let mut suit_to_ranks: HashMap<Suit, Vec<u8>> = HashMap::new();
    for card in hand {
        let Some(c) = parse_card_symbol(card).ok() else {
            continue;
        };
        if is_wild(c, ctx) || c.suit == Suit::Joker {
            continue;
        }
        if let Ok(nv) = natural_rank_value(c.rank) {
            suit_to_ranks.entry(c.suit).or_default().push(nv);
        }
    }
    for ranks in suit_to_ranks.values_mut() {
        ranks.sort_unstable();
        ranks.dedup();
        let mut run_len = 1usize;
        for i in 1..ranks.len() {
            if ranks[i] - ranks[i - 1] == 1 {
                run_len += 1;
                if run_len >= 5 {
                    straight_flush_count += 1;
                    run_len = 0;
                }
            } else {
                run_len = 1;
            }
        }
    }

    // wildcard-assisted bombs (JS L1288-1290): ranks with exactly 3 naturals, capped by wilds
    let wild_assisted_bombs = if wild_count >= 1 {
        rank_to_count.values().filter(|&&c| c == 3).count().min(wild_count)
    } else {
        0
    };

    let bomb_count = bomb_ranks.len()
        + wild_assisted_bombs
        + straight_flush_count
        + usize::from(red_joker_count == 2 && black_joker_count == 2);

    // singles count (JS L1295-1302)
    let mut singles_count = 0usize;
    for card in hand {
        let Some(c) = parse_card_symbol(card).ok() else {
            continue;
        };
        if is_wild(c, ctx) || c.suit == Suit::Joker {
            continue;
        }
        if let Ok(nv) = natural_rank_value(c.rank) {
            if rank_to_count.get(&nv).copied().unwrap_or(0) == 1 {
                singles_count += 1;
            }
        }
    }

    HandCombos {
        card_to_rank,
        rank_to_count,
        plate_pairs,
        tube_triples,
        bomb_count,
        straight_flush_count,
        singles_count,
        wild_count,
    }
}

// ── 拆炸弹裁决 (JS classifyBombSplit L1330-1363) ────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BombSplitVerdict {
    /// 未拆炸弹
    NotBreaking,
    /// 放行拆炸（清空 / 中盘带出≥3单牌 / 中盘同花顺剩单≤2）
    Exempt,
    /// 禁止拆炸（含残局一律禁止）
    Banned,
}

fn classify_bomb_split(
    play_cards: &[String],
    hand: &[String],
    play_kind: &CombinationKind,
    my_remaining: usize,
) -> BombSplitVerdict {
    // handCnt: all non-joker cards counted at face rank (wilds included, JS L1332-1337)
    let mut hand_cnt: HashMap<Rank, usize> = HashMap::new();
    for hc in hand {
        if is_joker_sym(hc) {
            continue;
        }
        if let Ok(c) = parse_card_symbol(hc) {
            *hand_cnt.entry(c.rank).or_default() += 1;
        }
    }
    let mut breaks_bomb = false;
    let mut singles_carried = 0usize;
    let mut play_cnt: HashMap<Rank, usize> = HashMap::new();
    for c in play_cards {
        if is_joker_sym(c) {
            continue;
        }
        if let Ok(card) = parse_card_symbol(c) {
            if hand_cnt.get(&card.rank).copied().unwrap_or(0) == 1 {
                singles_carried += 1;
            }
            *play_cnt.entry(card.rank).or_default() += 1;
        }
    }
    // 只有「部分消耗」≥4 的点数才算拆（整组全出不算）(JS L1348-1350)
    for (r, &pc) in &play_cnt {
        let hc = hand_cnt.get(r).copied().unwrap_or(0);
        if hc >= 4 && pc >= 1 && pc < hc {
            breaks_bomb = true;
        }
    }
    if !breaks_bomb {
        return BombSplitVerdict::NotBreaking;
    }
    if play_cards.len() >= my_remaining {
        return BombSplitVerdict::Exempt; // ①清空手牌
    }
    if my_remaining <= DUAL_WILD_HAND_ENDGAME {
        return BombSplitVerdict::Banned; // 残局一律禁止
    }
    if singles_carried >= 3 {
        return BombSplitVerdict::Exempt; // ②同时减少≥3张单牌
    }
    if matches!(play_kind, CombinationKind::Bomb(BombKind::StraightFlush)) {
        // ③同花顺且剩单≤2 (JS L1355-1361)
        let mut remain_singles = 0i32;
        for (r, &cnt) in &hand_cnt {
            let played = play_cnt.get(r).copied().unwrap_or(0) as i32;
            if cnt as i32 - played == 1 {
                remain_singles += 1;
            }
        }
        if remain_singles <= 2 {
            return BombSplitVerdict::Exempt;
        }
    }
    BombSplitVerdict::Banned
}

// ── 拆牌惩罚 (JS splitPenalty L1381-1457) ───────────────────────────────
/// Higher = worse. JS 只会返回 0 或 99999（绝对禁令）。
fn split_penalty(
    play_cards: &[String],
    combos: &HandCombos,
    level_nat: u8,
    has_level_card_or_joker: bool,
    play_kind: &CombinationKind,
    allow_endgame_split: bool,
) -> u32 {
    // playRankCounts via cardToRank: wilds/jokers contribute nothing (JS L1386-1392)
    let mut play_rank_counts: HashMap<u8, usize> = HashMap::new();
    for card in play_cards {
        if let Some(&nv) = combos.card_to_rank.get(card) {
            *play_rank_counts.entry(nv).or_default() += 1;
        }
    }

    for (&rank, &play_count) in &play_rank_counts {
        let hand_count = combos.rank_to_count.get(&rank).copied().unwrap_or(0);
        let is_level_rank = rank == level_nat;

        // Not splitting (playCount >= handCount), skip (JS L1399-1400)
        if play_count >= hand_count {
            continue;
        }

        // ABSOLUTE BAN: never split bombs (≥4 same rank, except level cards)
        if hand_count >= 4 {
            if is_level_rank {
                continue; // level card bombs can be split
            }
            return BANNED_SCORE_U32; // ABSOLUTE BAN
        }

        // Plate protection + triple splitting rules
        if hand_count >= 3 {
            let is_plate_part = combos
                .plate_pairs
                .iter()
                .any(|&(a, b)| rank == a || rank == b);
            if is_plate_part {
                return BANNED_SCORE_U32; // ABSOLUTE BAN: plate
            }
            if play_count < 3 {
                if is_level_rank {
                    continue; // 级牌可以单出、对子、三张
                }
                // 顺子拆对/三同张优先于禁止拆牌规则
                if matches!(play_kind, CombinationKind::Ordinary(OrdinaryKind::Straight)) {
                    continue;
                }
                // 房规豁免（用户 2026-08-30）：残局+手牌无单张+队友本轮已过牌 → 允许拆三张同
                if allow_endgame_split {
                    continue;
                }
                if rank <= 10 {
                    return BANNED_SCORE_U32; // ABSOLUTE BAN: rank ≤ 10
                }
                if has_level_card_or_joker {
                    return BANNED_SCORE_U32; // ABSOLUTE BAN
                }
                // rank > 10 且无级牌/王：允许拆
            }
            continue;
        }

        // Tube protection + pair splitting rules
        if hand_count >= 2 {
            let is_tube_part = combos
                .tube_triples
                .iter()
                .any(|&(a, b, c)| rank == a || rank == b || rank == c);
            if is_tube_part {
                return BANNED_SCORE_U32; // ABSOLUTE BAN: tube
            }
            if play_count < 2 {
                if is_level_rank {
                    continue;
                }
                if matches!(play_kind, CombinationKind::Ordinary(OrdinaryKind::Straight)) {
                    continue;
                }
                // 房规豁免（用户 2026-08-30）：残局+手牌无单张+队友本轮已过牌 → 允许拆对子
                if allow_endgame_split {
                    continue;
                }
                if rank <= 10 {
                    return BANNED_SCORE_U32;
                }
                if has_level_card_or_joker {
                    return BANNED_SCORE_U32;
                }
            }
        }
    }

    0 // No penalty (allowed)
}

const BANNED_SCORE_U32: u32 = 99999;

// ── Play context (预计算, mirror JS build_play_context 参数面) ──────────

// ── 牌踪器（JS HandTracker 的 rank 级统计，1:1 语义）────────────────────
// 双副牌 108 张：点数 3..A、2 各 8 张（4 花色 × 2 副），黑王/红王各 2 张。
// 已见 = 我手牌 + 全部历史出牌；剩余池 = 全部 − 已见。

/// JS `rankValue` 的值域索引：3..=15 为点数（2=15），16=小王，17=大王。
const RANK_SLOT_MAX: usize = 18;

#[derive(Clone, Debug, Default)]
struct PoolStats {
    /// rank_value 值 → 剩余未见张数
    rank_counts: [u16; RANK_SLOT_MAX],
    /// suffix_ge[v] = rank 值 ≥ v 的剩余张数
    suffix_ge: [u16; RANK_SLOT_MAX + 1],
    /// 剩余未见总张数
    total: u16,
}

impl PoolStats {
    fn build(my_hand: &[String], history: &[HandHistoryEntry], ctx: RuleContext) -> Self {
        // 已见按 rank 统计（双王/级牌/百搭都按其符号→rank 计数）
        let mut seen_rank = [0u16; RANK_SLOT_MAX];
        let mut bump = |sym: &str| {
            if let Some(meta) = meta_of(sym, ctx) {
                let v = meta.rank_value as usize;
                if v < RANK_SLOT_MAX {
                    seen_rank[v] += 1;
                }
            }
        };
        for c in my_hand {
            bump(c);
        }
        for e in history {
            if e.action_type == HistoryActionKind::Play {
                for c in &e.cards {
                    bump(c);
                }
            }
        }
        // 双副牌全量 − 已见
        let mut s = PoolStats::default();
        let mut total = 0u16;
        for v in 3..=15usize {
            s.rank_counts[v] = 8 - seen_rank[v].min(8);
            total += s.rank_counts[v];
        }
        for v in 16..=17usize {
            s.rank_counts[v] = 2 - seen_rank[v].min(2);
            total += s.rank_counts[v];
        }
        s.total = total;
        let mut running = 0u16;
        for v in (0..RANK_SLOT_MAX).rev() {
            running += s.rank_counts[v];
            s.suffix_ge[v] = running;
        }
        s
    }
}

struct PlayContext {
    ctx: RuleContext,
    /// 当前级牌 Rank（JS `levelRank`）
    level_rank: Rank,
    /// 级牌的 NATURAL_RANK 值（JS `NATURAL_RANK[lvl]`）
    level_nat: u8,
    params: AdvancedBotParams,
    /// 我的手牌（clone）
    my_hand: Vec<String>,
    my_remaining: usize,
    teammate_seat: Seat,
    teammate_remaining: usize,
    /// 两个对手的最小剩余张数
    min_opp_remaining: usize,
    /// JS `isEndgame = myRemaining <= params.endgame_hand_count_threshold`
    is_endgame: bool,
    combos: HandCombos,
    /// 手中是否有（非百搭）级牌或王 — JS `hasLevelCardOrJoker`
    has_level_card_or_joker: bool,
    /// 我手牌的预解析元数据
    meta: HashMap<String, CardMeta>,
    /// 房规（2026-08-30）：本轮顶牌之后队友是否已 Pass（残局拆对豁免条件之一）
    teammate_passed_top: bool,
    /// 牌踪器：本座（actor）
    actor_seat: Seat,
    /// 牌踪器：各座剩余张数（JS tracker.seatRemainingCounts）
    seat_remaining: HashMap<Seat, usize>,
    /// 牌踪器：两个对手座
    enemy_seats: [Seat; 2],
    /// 牌踪器：剩余未见牌的 rank 级统计
    pool: PoolStats,
}

fn build_play_context(hand: &HandState, actor: Seat, ctx: RuleContext) -> PlayContext {
    let my_hand: Vec<String> = hand.hands.get(&actor).cloned().unwrap_or_default();
    let my_remaining = my_hand.len();
    let teammate_seat = actor.teammate();
    // 各座剩余张数直接取 hands.len()（任务要求；JS 用 tracker 计数，等价）
    let teammate_remaining = hand.remaining_count(teammate_seat);
    let opp_counts: Vec<usize> = Seat::ALL
        .iter()
        .filter(|s| **s != actor && **s != teammate_seat)
        .map(|s| hand.remaining_count(*s))
        .collect();
    let min_opp_remaining = opp_counts.iter().min().copied().unwrap_or(0);

    let params = get_params_for_seat(actor);
    let is_endgame = my_remaining <= params.endgame_hand_count_threshold as usize;
    let combos = analyze_hand_combos(&my_hand, ctx);
    let level_rank = ctx.hand_level.to_rank();
    let level_nat = natural_rank_value(level_rank).unwrap_or(2);

    let meta: HashMap<String, CardMeta> = my_hand
        .iter()
        .filter_map(|s| meta_of(s, ctx).map(|m| (s.clone(), m)))
        .collect();

    let has_level_card_or_joker = my_hand.iter().any(|c| {
        match meta.get(c) {
            Some(m) => m.is_joker || (m.rank == level_rank && !m.is_wild),
            None => is_joker_sym(c),
        }
    });

    // ── 牌踪器：各座剩余 + 剩余未见牌统计（JS HandTracker.init/updatePlay 等价）──
    let actor_seat = actor;
    let mut seat_remaining: HashMap<Seat, usize> = HashMap::new();
    for s in Seat::ALL {
        seat_remaining.insert(s, hand.remaining_count(s));
    }
    let enemy_seats: [Seat; 2] = Seat::ALL
        .iter()
        .copied()
        .filter(|s| *s != actor && *s != teammate_seat)
        .collect::<Vec<_>>()
        .try_into()
        .expect("exactly two enemy seats");
    let pool = PoolStats::build(&my_hand, &hand.history, ctx);

    // 房规（用户 2026-08-30）：残局拆对豁免需要"队友本轮已过牌（不要单张）"——
    // 从 history 末尾向前取 Pass 条目（直到当前顶牌=最后一条 Play）看队友是否 Pass 过。
    let teammate_passed_top = hand
        .history
        .iter()
        .rev()
        .take_while(|e| e.action_type != HistoryActionKind::Play)
        .any(|e| e.action_type == HistoryActionKind::Pass && e.seat == teammate_seat);

    PlayContext {
        ctx,
        level_rank,
        level_nat,
        params,
        my_hand,
        my_remaining,
        teammate_seat,
        teammate_remaining,
        min_opp_remaining,
        is_endgame,
        combos,
        has_level_card_or_joker,
        meta,
        teammate_passed_top,
        actor_seat,
        seat_remaining,
        enemy_seats,
        pool,
    }
}

impl PlayContext {
    fn meta_for(&self, sym: &str) -> Option<CardMeta> {
        self.meta
            .get(sym)
            .copied()
            .or_else(|| meta_of(sym, self.ctx))
    }

    // ── 牌踪器概率接口（JS HandTracker / ProbabilisticReasoner 1:1）─────────

    fn remaining_of(&self, seat: Seat) -> usize {
        self.seat_remaining.get(&seat).copied().unwrap_or(0)
    }

    /// JS `getProbRankInHand`（L228-238）
    fn prob_rank_in_hand(&self, seat: Seat, rank_v: usize) -> f32 {
        if seat == self.actor_seat {
            return 0.0;
        }
        let total_remaining = self.remaining_of(seat);
        if total_remaining == 0 {
            return 0.0;
        }
        let remaining_in_pool = self.pool.rank_counts.get(rank_v).copied().unwrap_or(0) as f32;
        let total_unknown = self.pool.total as f32;
        if total_unknown == 0.0 {
            return 0.0;
        }
        (remaining_in_pool / total_unknown) * (total_remaining as f32 / total_unknown).min(1.0)
    }

    /// JS `getProbHasBomb`（L244-262）
    fn prob_has_bomb(&self, seat: Seat) -> f32 {
        if seat == self.actor_seat {
            return 0.0;
        }
        let total_remaining = self.remaining_of(seat);
        if total_remaining < 4 {
            return 0.0;
        }
        let mut prob = 0.0f32;
        for v in 3..=15usize {
            // JS bombRanks '2'..'A' → rank 值 3..=15
            let count_in_pool = self.pool.rank_counts[v];
            if count_in_pool >= 4 {
                prob += 0.2;
            } else if count_in_pool >= 3 {
                prob += 0.1;
            } else if count_in_pool >= 2 {
                prob += 0.05;
            }
        }
        prob.min(1.0) * (total_remaining as f32 / 20.0).min(1.0)
    }

    /// JS `getProbCanFollow`（L268-278）
    fn prob_can_follow(&self, seat: Seat, min_rank_v: usize) -> f32 {
        if seat == self.actor_seat {
            return 0.0;
        }
        let total_remaining = self.remaining_of(seat);
        if total_remaining == 0 {
            return 0.0;
        }
        let total_unknown = self.pool.total as f32;
        if total_unknown == 0.0 {
            return 0.0;
        }
        let higher_in_pool = self
            .pool
            .suffix_ge
            .get(min_rank_v)
            .copied()
            .unwrap_or(0) as f32;
        (higher_in_pool / total_unknown).min(1.0)
    }

    /// JS `calculateOpponentBombProb`（L320-329）：对手中最大的持炸概率
    fn opponent_bomb_prob(&self) -> f32 {
        self.enemy_seats
            .iter()
            .map(|s| self.prob_has_bomb(*s))
            .fold(0.0f32, f32::max)
    }

    /// JS `calculateOpponentHasRank`（L335-341）：全体（含队友，自身恒 0）概率求和，封顶 1
    fn opponent_has_rank(&self, rank_v: usize) -> f32 {
        Seat::ALL
            .iter()
            .map(|s| self.prob_rank_in_hand(*s, rank_v))
            .sum::<f32>()
            .min(1.0)
    }

    /// JS `calculateGameWinProb`（L375-397）：团队级胜率（不含自对弈噪声）
    fn game_win_prob(&self) -> f32 {
        let my_remaining = self.remaining_of(self.actor_seat);
        let partner_remaining = self.remaining_of(self.teammate_seat);
        let team_remaining = my_remaining + partner_remaining;
        let mut enemy_remaining = 0usize;
        for s in self.enemy_seats {
            enemy_remaining += self.remaining_of(s);
        }
        let total_cards = team_remaining + enemy_remaining;
        if total_cards == 0 {
            return 0.5;
        }
        let progress_ratio = enemy_remaining as f32 / total_cards as f32;
        let advantage = 1.0 - progress_ratio;
        let bomb_factor = self.opponent_bomb_prob();
        let adjusted = advantage * (1.0 - bomb_factor * 0.3);
        adjusted.clamp(0.0, 1.0)
    }
}

// ── Public API ─────────────────────────────────────────────────────────

/// Pick one legal action:
/// - Playing: JS decision flow (lead → score_lead max; follow → pass/play heuristic,
///   then findBestPlay guards).
/// - Tribute/Return: smaller card value first (unchanged, out of JS scope).
pub fn suggest_next_action(state: &TableGameState, actor: Seat) -> Result<PlayerAction, String> {
    let legal = enumerate_legal_actions(state, actor)?;
    if legal.is_empty() {
        return Err("no legal actions".into());
    }

    let hand = state.hand.as_ref().ok_or_else(|| "no hand".to_string())?;
    let ctx = RuleContext {
        hand_level: hand.hand_level,
    };

    match state.phase {
        GamePhase::Playing => pick_playing(hand, actor, &legal),
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

// ── Playing phase: JS decision flow ────────────────────────────────────

/// Parsed candidate: the [`PlayerAction`] plus its resolved [`Combination`].
struct Candidate<'a> {
    action: &'a PlayerAction,
    cards: &'a [String],
    combo: Combination,
}

fn parse_candidates<'a>(legal: &'a [PlayerAction], ctx: RuleContext) -> Vec<Candidate<'a>> {
    let mut out = Vec::new();
    for a in legal {
        if let PlayerAction::Play { cards, wild_targets } = a {
            if let Ok(combo) = CombinationParser::parse(cards, wild_targets.as_deref(), ctx) {
                out.push(Candidate {
                    action: a,
                    cards,
                    combo,
                });
            }
        }
    }
    out
}

fn pick_playing(
    hand: &HandState,
    actor: Seat,
    legal: &[PlayerAction],
) -> Result<PlayerAction, String> {
    let ctx = RuleContext {
        hand_level: hand.hand_level,
    };
    let my_hand = hand
        .hands
        .get(&actor)
        .ok_or_else(|| "missing actor hand".to_string())?;
    let top = hand.trick.top_play.as_ref();
    let pass_action = legal.iter().find(|a| matches!(a, PlayerAction::Pass));
    let candidates = parse_candidates(legal, ctx);

    // ── 领牌（无 lastPlay）：全部候选用 score_lead 打分取最大（JS findBestLeadPlay）──
    let Some(top) = top else {
        if candidates.is_empty() {
            return Err("suggest: no lead play available".into());
        }
        let p = build_play_context(hand, actor, ctx);
        let mut best: Option<(f32, &Candidate)> = None;
        let mut best_non_bomb: Option<(f32, &Candidate)> = None;
        for cand in &candidates {
            let s = score_lead(cand.cards, &cand.combo, &p);
            if best.map_or(true, |(bs, _)| s > bs) {
                best = Some((s, cand));
            }
            if cand.combo.class() != CombinationClass::Bomb
                && best_non_bomb.map_or(true, |(bs, _)| s > bs)
            {
                best_non_bomb = Some((s, cand));
            }
        }
        let (_, best_cand) = best.expect("candidates non-empty");
        // ── 房规（用户）：中盘主动出炸后，剩余手牌重新清点炸弹数（含潜在炸）——
        //    剩 0 → 改出非炸最优牌；无非炸候选（整手皆炸）时维持原炸（打完仍剩其余炸）。
        //    豁免：残局（≤6 张）、对手冲刺（≤6 张）、清空出牌。
        {
            let is_endgame = p.my_remaining <= 6;
            let is_opp_sprinting = p.min_opp_remaining <= 6;
            if best_cand.combo.class() == CombinationClass::Bomb
                && !is_endgame
                && !is_opp_sprinting
                && best_cand.cards.len() < p.my_remaining
            {
                let rest: Vec<String> = p
                    .my_hand
                    .iter()
                    .filter(|c| !best_cand.cards.contains(*c))
                    .cloned()
                    .collect();
                if analyze_hand_combos(&rest, ctx).bomb_count == 0 {
                    if let Some((_, alt)) = best_non_bomb {
                        return Ok(alt.action.clone());
                    }
                }
            }
        }
        return Ok(best_cand.action.clone());
    };

    // 跟牌：Pass 必然在合法动作中（movegen 保证）
    let pass = pass_action.ok_or_else(|| "suggest: no pass in legal".to_string())?;

    // ── 房规：只剩最后 1 张时，只要能合法压过就立即打出清空（JS 488-497）──
    if my_hand.len() == 1 {
        if let Some(cand) = candidates.first() {
            return Ok(cand.action.clone());
        }
    }

    if candidates.is_empty() {
        return Ok(pass.clone());
    }

    let p = build_play_context(hand, actor, ctx);

    // ── JS decideAdvancedPlay：pass/play 启发式（概率项置 0，确定性项全保留）──
    if decide_follow(top, &p, actor) == FollowDecision::Pass {
        return Ok(pass.clone());
    }

    // ── JS findBestPlay：全部候选 score_follow 取最大 + 硬守卫 ──
    find_best_play_follow(&candidates, top, &p)
}

// ── JS decideAdvancedPlay (L460-611): pass/play 启发式 ──────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FollowDecision {
    Pass,
    Play,
}

/// JS `extractTopRank` (L445-448): 取 top 第一张牌的 rank。
fn extract_top_rank(top: &PlayState) -> Option<Rank> {
    top.cards.first().and_then(|c| parse_card_symbol(c).ok()).map(|c| c.rank)
}

fn decide_follow(top: &PlayState, p: &PlayContext, actor: Seat) -> FollowDecision {
    let params = &p.params;
    let top_play_seat = top.seat;
    let partner_leading = top_play_seat == p.teammate_seat;
    let enemy_leading = top_play_seat != p.teammate_seat && top_play_seat != actor;

    let mut pass_score = 0.0f32;
    let mut play_score = 0.0f32;

    if partner_leading {
        // Partner is leading — be conservative about overriding (JS 509-561)
        let top_is_bomb = top.combination.class() == CombinationClass::Bomb;
        if top_is_bomb {
            return FollowDecision::Pass; // NEVER override teammate's bomb
        }

        let top_rank = extract_top_rank(top);
        let top_is_level = top_rank == Some(p.level_rank);
        if top_is_level {
            return FollowDecision::Pass; // NEVER override teammate's level card
        }

        let top_is_joker = top_rank.map_or(false, |r| matches!(r, Rank::RedJoker | Rank::BlackJoker))
            || top.cards.iter().any(|c| is_joker_sym(c));
        if top_is_joker {
            return FollowDecision::Pass; // NEVER override teammate's joker
        }

        // 钢板(Plate)和木板(Tube)绝对不能压队友的牌
        match top.combination.kind {
            CombinationKind::Ordinary(OrdinaryKind::Plate)
            | CombinationKind::Ordinary(OrdinaryKind::Tube) => {
                return FollowDecision::Pass;
            }
            _ => {}
        }

        let trv = top_rank.map(rank_value_js).unwrap_or(0);
        // 队友的对子大于12（K、A、2）不能压
        if matches!(
            top.combination.kind,
            CombinationKind::Ordinary(OrdinaryKind::Pair)
        ) && trv > 12
        {
            return FollowDecision::Pass;
        }
        // 队友的三带二大于等于10（10、J、Q、K、A、2）不能压——用户房规"10以上不压队友"。
        // 三张 rank 取解析后的 combination.primary（修复 extract_top_rank 只看首张牌在
        // 乱序/百搭记录下判错致队友 JJJ 被压）。primary 为 level_order_value 刻度
        // （10=9,J=10,Q=11,K=12,A=13,级牌=14），阈值 9 对齐 JS rankValue>=10（CF 同步）。
        if matches!(
            top.combination.kind,
            CombinationKind::Ordinary(OrdinaryKind::FullHouse)
        ) && top.combination.primary >= 9
        {
            return FollowDecision::Pass;
        }

        let top_is_big = trv >= 12; // Q or higher
        if top_is_big {
            return FollowDecision::Pass; // Do not override teammate's big card
        }

        // 概率项 probOpponentCanFollow（JS 555-561）：对手/队友是否还接得住这个 rank
        if p.params.hand_tracker_enabled {
            let top_rank_v = top_rank.map(rank_value_js).unwrap_or(0) as usize;
            let prob_opponent_can_follow = p.opponent_has_rank(top_rank_v);
            if prob_opponent_can_follow < params.prob_threshold_for_intercept {
                play_score += 2.0 * params.low_card_dump_bias;
            } else {
                pass_score += 1.0 * params.yield_to_partner_bias;
            }
        }
        // （hand_tracker_enabled=false 时概率项贡献 0，维持旧行为）
    } else if enemy_leading {
        // Enemy is leading — try to follow (JS 562-580)
        play_score += 2.0;

        // 概率项 probOpponentHasBomb（JS 566-569）：对手大概率持炸 → 保守
        if p.params.hand_tracker_enabled {
            let prob_opponent_has_bomb = p.opponent_bomb_prob();
            if prob_opponent_has_bomb > params.prob_threshold_for_bomb {
                pass_score += 2.0 * params.bomb_conserve_bias;
            }
        }

        let enemy_low = p.min_opp_remaining <= params.enemy_low_cards_threshold as usize;
        if enemy_low {
            play_score += params.bomb_aggression_when_enemy_low;
        }

        if p.is_endgame {
            play_score += params.endgame_clear_hand_bias;
        }
    } else {
        // Self is leading (shouldn't happen with lastPlay, but fallback)
        play_score += params.proactive_play_bias;
    }

    // Partner sprinting check (JS 587-590)
    let partner_sprinting = p.teammate_remaining <= params.partner_sprint_threshold as usize;
    if partner_sprinting && !enemy_leading {
        pass_score += 2.0 * params.second_out_weight;
    }

    // 概率项 calculateGameWinProb（JS 592-598）：局势占优且非敌领 → 过牌保局面；劣势 → 抢
    if p.params.hand_tracker_enabled {
        let game_win_prob = p.game_win_prob();
        if game_win_prob > 0.7 && !enemy_leading {
            pass_score += 1.0;
        } else if game_win_prob < 0.3 {
            play_score += 1.5;
        }
    }

    // Decision: pass if passScore > playScore, else find best play (JS 605-610)
    if pass_score > play_score {
        FollowDecision::Pass
    } else {
        FollowDecision::Play
    }
}

// ── JS findBestPlay (L622-725): 打分 + 硬守卫 ───────────────────────────

fn find_best_play_follow<'a>(
    candidates: &[Candidate<'a>],
    top: &PlayState,
    p: &PlayContext,
) -> Result<PlayerAction, String> {
    // JS 硬编码：isEndgame = myRemaining <= 6；冲刺 = 任一对手 ≤ 6（用户房规，原 JS 为 3）
    let my_remaining = p.my_remaining;
    let is_endgame = my_remaining <= 6;
    let is_opp_sprinting = p.min_opp_remaining <= 6;

    // 房规 B1（反炸省百搭）：反炸场景下，若存在"不含百搭"的候选炸能压过，
    // 则含百搭的炸（且非清空候选）不参与最优选择——张数/点数 penalty 照旧自然选最小够用。
    // 豁免：残局（≤6）、对手冲刺（≤6）、炸完直接清空（那时烧百搭值）。
    // （注：不存在免百搭候选时含百搭的炸照常可选——用户未批"唯一反炸含百搭则不反"）
    let top_is_bomb = top.combination.class() == CombinationClass::Bomb;
    let cand_has_wild = |c: &Candidate| {
        c.cards
            .iter()
            .any(|s| p.meta_for(s).map(|m| m.is_wild).unwrap_or(false))
    };
    let has_wildfree_bomb = candidates
        .iter()
        .any(|c| c.combo.class() == CombinationClass::Bomb && !cand_has_wild(c));

    // Score each possible play and pick the best one (ties → first in order, JS stable sort)
    let mut best: Option<(f32, &Candidate)> = None;
    for cand in candidates {
        // 房规 B1：反炸 + 有免百搭候选 + 非豁免场景 → 跳过含百搭的炸（非清空）
        if top_is_bomb
            && has_wildfree_bomb
            && !is_endgame
            && !is_opp_sprinting
            && cand.combo.class() == CombinationClass::Bomb
            && cand_has_wild(cand)
            && cand.cards.len() < my_remaining
        {
            continue;
        }
        let s = score_follow(cand.cards, &cand.combo, top, p);
        if best.map_or(true, |(bs, _)| s > bs) {
            best = Some((s, cand));
        }
    }
    let Some((best_score, best)) = best else {
        return Ok(PlayerAction::Pass);
    };
    let best_cards = best.cards;
    let best_is_bomb = best.combo.class() == CombinationClass::Bomb;

    // ① 炸弹保留：非残局、非对手冲刺、手里炸弹总数 = 1 个时，不出炸弹，直接过。
    //    （用户房规改自 JS 647-655：JS 为 bombCount<=2，现为 =1——
    //      2 个炸弹不再拦；"手里 ≥3 个炸可用"的豁免由条件 =1 自然排除）
    if best_is_bomb && !is_endgame && !is_opp_sprinting && p.combos.bomb_count == 1 {
        return Ok(PlayerAction::Pass);
    }

    // ①b 炸弹保留·事后重算（用户房规）：中盘出炸（含反炸）后，对剩余手牌重新清点
    //    炸弹数（天然 4+ 同张、百搭拼 3 同张、同花顺候选全部重算）——剩 0 → 不出，
    //    保炸到残局。解决"潜在炸共用百搭导致账面虚增"（如 ♠6789+♥6789+♥2 账面 2 颗、
    //    打掉一颗后实际 0 颗）。豁免：残局（≤6 张）、对手冲刺（≤6 张）、炸完直接清空。
    if best_is_bomb && !is_endgame && !is_opp_sprinting && best_cards.len() < my_remaining {
        let rest: Vec<String> = p
            .my_hand
            .iter()
            .filter(|c| !best_cards.contains(*c))
            .cloned()
            .collect();
        let rest_ctx = RuleContext {
            hand_level: p.ctx.hand_level,
        };
        if analyze_hand_combos(&rest, rest_ctx).bomb_count == 0 {
            return Ok(PlayerAction::Pass);
        }
    }

    // ③ 绝对禁止：队友出牌时，绝不能使用炸弹压队友的牌 (JS 664-669)
    if top.seat == p.teammate_seat && best_is_bomb {
        return Ok(PlayerAction::Pass);
    }

    // ④ 王和级牌之外的任何单张禁止使用炸弹 (JS 671-685)
    if best_is_bomb
        && matches!(
            top.combination.kind,
            CombinationKind::Ordinary(OrdinaryKind::Single)
        )
        && top.cards.len() == 1
    {
        let last = &top.cards[0];
        let is_joker = is_joker_sym(last);
        let is_level_card = parse_card_symbol(last).map(|c| c.rank) == Ok(p.level_rank);
        if !is_joker && !is_level_card {
            return Ok(PlayerAction::Pass);
        }
    }

    // ⑤ 王和级牌之外的任何对子禁止使用炸弹 (JS 687-699)
    if best_is_bomb
        && matches!(
            top.combination.kind,
            CombinationKind::Ordinary(OrdinaryKind::Pair)
        )
    {
        let is_joker = top.cards.iter().any(|c| is_joker_sym(c));
        let is_level = top
            .cards
            .iter()
            .all(|c| parse_card_symbol(c).map(|c| c.rank) == Ok(p.level_rank));
        if !is_joker && !is_level {
            return Ok(PlayerAction::Pass);
        }
    }

    // ⑥ 绝对禁止：用炸弹压三张（太浪费，除非对手冲刺或残局）(JS 701-711)
    //    房规扩展（用户 2026-08-30）：12 以下（普通点数 <Q，即 J 及以下）的三带二同样不炸；
    //    Q/K/A/级牌 的三带二仍可炸。豁免照旧：残局（≤6）、对手冲刺（≤6）。
    //    注：primary 为 level_order_value 尺度（Q=11, K=12, A=13, 级牌=14），故 <Q ⇔ primary<11。
    if best_is_bomb
        && !is_endgame
        && !is_opp_sprinting
        && match top.combination.kind {
            CombinationKind::Ordinary(OrdinaryKind::Triple) => true,
            CombinationKind::Ordinary(OrdinaryKind::FullHouse) => top.combination.primary < 11,
            _ => false,
        }
    {
        return Ok(PlayerAction::Pass);
    }

    // ⑦ 绝对禁止：拆牌惩罚得分极低时，过牌比出牌更好 (JS 713-718)
    if best_score < -1000.0 {
        return Ok(PlayerAction::Pass);
    }

    Ok(best.action.clone())
}

// ── JS scorePlay (L732-1157): 跟牌打分，base 100，越高越好 ──────────────

fn score_follow(
    play_cards: &[String],
    play_combo: &Combination,
    top: &PlayState,
    p: &PlayContext,
) -> f32 {
    let kind = &play_combo.kind;
    let mut score = BASE_FOLLOW_SCORE; // JS L734 base score

    let my_remaining = p.my_remaining;
    let is_endgame = p.is_endgame;
    let combos = &p.combos;

    // ── Hand combo analysis & split penalty (JS 740-756) ──
    let bomb_split_verdict =
        classify_bomb_split(play_cards, &p.my_hand, kind, my_remaining);
    let play_is_bomb = play_combo.class() == CombinationClass::Bomb;
    // 房规（用户 2026-08-30）：拆对/拆三张豁免 = 残局(≤6) + 手牌无单张 + 队友本轮已过牌。
    let allow_endgame_split =
        p.is_endgame && p.combos.singles_count == 0 && p.teammate_passed_top;
    let mut penalty =
        split_penalty(play_cards, combos, p.level_nat, p.has_level_card_or_joker, kind, allow_endgame_split) as f32;
    if !play_is_bomb {
        if bomb_split_verdict == BombSplitVerdict::Exempt && penalty >= BANNED_SCORE {
            penalty = 0.0; // 房规豁免：放行拆炸
        }
        if bomb_split_verdict == BombSplitVerdict::Banned {
            penalty = penalty.max(BANNED_SCORE); // 双保险
        }
    }
    score -= penalty * SPLIT_PENALTY_SCALE; // Heavy penalty for splitting good combos

    let is_bomb = play_is_bomb;
    let is_last_play = play_cards.len() >= my_remaining; // JS L759

    // ── Bomb conservation (JS 762-807) ──
    if is_bomb {
        // Bomb size: prefer smaller bombs
        match kind {
            CombinationKind::Bomb(BombKind::SameRank { n: 4 }) => score += 5.0,
            CombinationKind::Bomb(BombKind::SameRank { n: 5 }) => score -= 3.0,
            CombinationKind::Bomb(BombKind::StraightFlush) => score -= 12.0,
            CombinationKind::Bomb(BombKind::FourJoker) => score -= 20.0,
            _ => score -= 6.0, // other bombs: slightly penalized
        }

        let min_opp_remaining = p.min_opp_remaining;
        if is_last_play {
            score += 20.0; // Bonus: clearing hand with bomb
        } else if min_opp_remaining <= 6 {
            score += 15.0; // 对手≤6张，炸弹拦截是好选择
        } else if combos.bomb_count >= 3 && !is_endgame {
            score -= 10.0; // 3+炸弹，留至少1个到残局
        }

        // 只有1-2个炸弹时，非残局非对手冲刺不能用炸弹
        if !is_endgame && !is_last_play && min_opp_remaining > 6 {
            if combos.bomb_count < 2 {
                score -= 200.0; // 仅1个炸弹，绝对保留
            } else if combos.bomb_count < 3 {
                score -= 50.0; // 2个炸弹，至少留1个
            }
        }

        // 房规：残局保留最后一个炸弹——非清空不轻出 (JS 800-806)
        let rest_cards_after_bomb = my_remaining.saturating_sub(play_cards.len());
        if is_endgame
            && !is_last_play
            && combos.bomb_count == 1
            && min_opp_remaining > 3
            && rest_cards_after_bomb > 2
        {
            score -= 400.0; // 留炸保底
        }
    }

    // ── 禁止出级牌炸弹 (JS 809-818) ──
    if is_bomb {
        let level_cards = play_cards
            .iter()
            .filter(|c| {
                p.meta_for(c)
                    .map(|m| m.rank == p.level_rank && !m.is_wild)
                    .unwrap_or(false)
            })
            .count();
        if level_cards >= 4 {
            score -= BANNED_SCORE; // 级牌炸弹，直接禁止
        }
    }

    // 手牌≤6张时必须保留至少1个炸弹 → 非炸弹出牌重奖 (JS 822-827)
    if !is_bomb && !is_last_play {
        let remaining_after = my_remaining.saturating_sub(play_cards.len());
        if remaining_after <= 6 && combos.bomb_count >= 1 {
            score += 500.0; // 保留炸弹到残局，重奖！
        }
    }

    // ── 炸弹压非炸弹牌型扣分 (JS 831-855) ──
    if is_bomb {
        let last_is_bomb = top.combination.class() == CombinationClass::Bomb;
        if !last_is_bomb {
            let min_opp_rem = p.min_opp_remaining;
            if !is_last_play && min_opp_rem > 6 && !is_endgame {
                match top.combination.kind {
                    CombinationKind::Ordinary(OrdinaryKind::Single) => score -= 300.0,
                    CombinationKind::Ordinary(OrdinaryKind::Pair) => score -= 200.0,
                    CombinationKind::Ordinary(OrdinaryKind::Straight)
                    | CombinationKind::Ordinary(OrdinaryKind::Tube)
                    | CombinationKind::Ordinary(OrdinaryKind::Plate) => score -= 80.0,
                    // 炸弹可以压三张/三带二，不扣分
                    _ => {}
                }
            }
        }
    }

    // ── 逢人配优先组成炸弹/同花顺奖励 (JS 864-881) ──
    let has_wildcard = play_cards
        .iter()
        .any(|c| p.meta_for(c).map(|m| m.is_wild).unwrap_or(false));
    if has_wildcard {
        if is_bomb
            || matches!(kind, CombinationKind::Bomb(BombKind::StraightFlush))
        {
            // 房规 B1（用户 2026-08-30）：反炸场景取消"逢人配配炸 +100"奖励——
            // 该奖励在反炸时主动鼓励烧百搭（如用 4个5+逢人配 反 4K）。豁免照旧：
            // 残局（≤6）、对手冲刺（≤6）、炸完直接清空（那时烧百搭值）。
            let counter_bomb = top.combination.class() == CombinationClass::Bomb;
            let not_clearing = play_cards.len() < my_remaining;
            let b1_exempt = is_endgame || p.min_opp_remaining <= 6 || !not_clearing;
            if !(counter_bomb && !b1_exempt) {
                score += 100.0; // 逢人配配炸弹/同花顺：重奖！
            }
        } else {
            match kind {
                CombinationKind::Ordinary(OrdinaryKind::Straight)
                | CombinationKind::Ordinary(OrdinaryKind::Plate)
                | CombinationKind::Ordinary(OrdinaryKind::Tube)
                | CombinationKind::Ordinary(OrdinaryKind::FullHouse) => score += 30.0,
                CombinationKind::Ordinary(OrdinaryKind::Triple) => score += 20.0,
                _ => {}
            }
        }
    }

    // ── 房规：已是天然炸弹再贴百搭升档 = 浪费 (JS 886-908) ──
    if has_wildcard && play_cards.len() < my_remaining {
        let wild_cnt = play_cards
            .iter()
            .filter(|c| p.meta_for(c).map(|m| m.is_wild).unwrap_or(false))
            .count();
        let naturals: Vec<CardMeta> = play_cards
            .iter()
            .filter_map(|c| p.meta_for(c))
            .filter(|m| !m.is_wild && !m.is_joker)
            .collect();
        let uniq: HashSet<Rank> = naturals.iter().map(|m| m.rank).collect();
        let naturals_already_bomb = naturals.len() >= 4
            && uniq.len() == 1
            && naturals.len() == play_cards.len() - wild_cnt;
        if naturals_already_bomb && naturals[0].rank != p.level_rank {
            if p.min_opp_remaining > DUAL_WILD_HAND_ENDGAME {
                if my_remaining <= DUAL_WILD_HAND_ENDGAME {
                    score -= UPGRADED_BOMB_WILD_PENALTY_ENDGAME; // 残局升档：轻罚
                } else {
                    score -= UPGRADED_BOMB_WILD_PENALTY_MIDGAME; // 中盘无场景升档：浪费重罚
                }
            }
        }
    }

    // Prefer playing the minimum needed to beat (smallest margin) (JS 911-914)
    let margin = play_combo.primary as i32 - top.combination.primary as i32;
    score -= margin as f32 * 2.0;

    // Prefer fewer cards when not endgame (JS 917-919)
    if !is_endgame || !is_bomb {
        score -= play_cards.len() as f32 * 3.0;
    }

    // Endgame: prefer clearing hand (JS 922-924)
    if is_endgame {
        score += p.params.endgame_clear_hand_bias * 10.0;
    }

    // ── 残局散牌惩罚: bomb leaves scattered singles (JS 928-943) ──
    if is_bomb && play_cards.len() < my_remaining {
        let remaining_after = my_remaining - play_cards.len();
        if remaining_after > 1 {
            let mut remaining_singles = combos.singles_count;
            for card in play_cards {
                if let Some(&nv) = combos.card_to_rank.get(card) {
                    if combos.rank_to_count.get(&nv).copied().unwrap_or(0) == 1 {
                        remaining_singles = remaining_singles.saturating_sub(1);
                    }
                }
            }
            let ratio_after = remaining_singles as f32 / remaining_after as f32;
            if ratio_after > 0.6 {
                score -= 30.0; // Heavy penalty: bomb leaves scattered singles
            }
        }
    }

    // ── Team awareness: 联邦接风重奖 (JS 947-957) ──
    let top_is_teammate = top.seat == p.teammate_seat;
    if top_is_teammate && !is_bomb {
        score += 300.0; // 给联邦接风，重奖！
    }
    if p.teammate_remaining == 1 && !is_bomb {
        score += 10.0;
    } else if p.teammate_remaining <= 6 && !is_bomb {
        score += 5.0;
    }

    // ── 残局移除单张奖励 (JS 960-984) ──
    if my_remaining <= 6 && !is_bomb {
        let mut singles_removed = 0usize;
        let mut small_singles_removed = 0usize;
        for card in play_cards {
            let Some(m) = p.meta_for(card) else { continue };
            if m.is_joker || m.is_wild || m.rank == p.level_rank {
                continue;
            }
            let Some(nv) = m.natural else { continue };
            if combos.rank_to_count.get(&nv).copied().unwrap_or(0) == 1 {
                singles_removed += 1;
                if m.rank_value <= 10 {
                    small_singles_removed += 1;
                }
            }
        }
        if singles_removed > 0 {
            score += singles_removed as f32 * 400.0; // 残局跟牌移除单张，重奖！
        }
        if small_singles_removed > 0 {
            score += small_singles_removed as f32 * 300.0; // 残局跟牌移除小单张，额外重奖！
        }
    }

    // ── 对手≤6张时强制拦截 (JS 987-1013) ──
    let min_opp_remaining = p.min_opp_remaining;
    if min_opp_remaining <= 6 && !is_bomb {
        match top.combination.kind {
            CombinationKind::Ordinary(OrdinaryKind::Single)
            | CombinationKind::Ordinary(OrdinaryKind::Pair) => {
                if play_combo.primary > 10 {
                    score += 15.0; // 出大牌阻止对手送牌
                }
            }
            _ => {
                score += 10.0; // 对手出其他牌型，跟牌压住
            }
        }
    } else if min_opp_remaining <= 6 && is_bomb {
        score += 10.0; // 对手≤6张，用炸弹拦截也是好选择
    }

    // ── 逢人配不能浪费（惩罚性检查）(JS 1018-1057) ──
    if has_wildcard {
        let finishing_play = play_cards.len() >= my_remaining; // 清空手牌：全部豁免
        let endgame_hand = my_remaining <= DUAL_WILD_HAND_ENDGAME;

        // 房规：百搭落在级牌上 = 不合理，重罚
        let same_rank_type = matches!(
            kind,
            CombinationKind::Ordinary(OrdinaryKind::Pair)
                | CombinationKind::Ordinary(OrdinaryKind::Triple)
        ) || is_bomb;
        let touches_level_natural = play_cards.iter().any(|c| {
            p.meta_for(c)
                .map(|m| m.rank == p.level_rank && !m.is_wild)
                .unwrap_or(false)
        });
        if !finishing_play && same_rank_type && touches_level_natural {
            score -= if endgame_hand {
                WILD_ON_LEVEL_PENALTY_ENDGAME
            } else {
                WILD_ON_LEVEL_PENALTY_MIDGAME
            };
        }

        // 房规：天然级牌炸弹（百搭当级牌面值凑炸）= 严重浪费
        let lvl_face_bomb = is_bomb
            && play_cards.iter().all(|c| {
                p.meta_for(c)
                    .map(|m| m.is_wild || m.rank == p.level_rank)
                    .unwrap_or(false)
            });
        if !finishing_play && lvl_face_bomb {
            score -= if endgame_hand {
                DUAL_WILD_PENALTY_ENDGAME
            } else {
                DUAL_WILD_PENALTY_MIDGAME
            };
        }

        if matches!(kind, CombinationKind::Ordinary(OrdinaryKind::Single)) && !finishing_play {
            score -= BANNED_SCORE; // 逢人配绝不能单出——清空手牌绝对豁免
        } else if is_bomb || matches!(kind, CombinationKind::Bomb(BombKind::StraightFlush)) {
            // 逢人配配非级牌炸弹、同花顺：最优使用，不罚
        } else if matches!(
            kind,
            CombinationKind::Ordinary(OrdinaryKind::Plate)
                | CombinationKind::Ordinary(OrdinaryKind::Tube)
                | CombinationKind::Ordinary(OrdinaryKind::FullHouse)
                | CombinationKind::Ordinary(OrdinaryKind::Triple)
                | CombinationKind::Ordinary(OrdinaryKind::Straight)
        ) {
            // 合理使用，不罚
        } else if matches!(kind, CombinationKind::Ordinary(OrdinaryKind::Pair)) {
            // 房规：百搭配普通单张成普通对 = 又弱又废，重罚
            if !finishing_play {
                let pair_naturals: Vec<CardMeta> = play_cards
                    .iter()
                    .filter_map(|c| p.meta_for(c))
                    .filter(|m| !m.is_wild && !m.is_joker)
                    .collect();
                let pair_rank = pair_naturals.first().map(|m| m.rank).unwrap_or(p.level_rank);
                if pair_rank != p.level_rank {
                    score -= if endgame_hand {
                        WILD_PAIR_PENALTY_ENDGAME
                    } else {
                        WILD_PLAIN_PAIR_PENALTY_MIDGAME
                    };
                }
            }
        } else {
            score -= 10.0; // 其他非最优使用
        }
    }

    // ── 双百搭同出 (JS 1061-1086) ──
    let dw_finishing = play_cards.len() >= my_remaining;
    let dw_endgame = my_remaining <= DUAL_WILD_HAND_ENDGAME;

    // 房规：拆炸弹兜底补罚 (JS 1065-1067)
    if !is_bomb && bomb_split_verdict == BombSplitVerdict::Banned {
        score -= BANNED_SCORE; // 拆炸弹绝对禁止
    }

    let wild_count_in_play = play_cards
        .iter()
        .filter(|c| p.meta_for(c).map(|m| m.is_wild).unwrap_or(false))
        .count();
    if !dw_finishing && wild_count_in_play >= 2 {
        let dw_naturals: Vec<CardMeta> = play_cards
            .iter()
            .filter_map(|c| p.meta_for(c))
            .filter(|m| !m.is_wild && !m.is_joker)
            .collect();
        let dw_ranks: HashSet<Rank> = dw_naturals.iter().map(|m| m.rank).collect();
        let bare_dual = dw_naturals.is_empty();
        let sanctioned_endgame = dw_endgame
            && !bare_dual
            && !dw_ranks.is_empty()
            && !dw_ranks.contains(&p.level_rank)
            && (is_bomb
                || matches!(
                    kind,
                    CombinationKind::Ordinary(OrdinaryKind::FullHouse)
                        | CombinationKind::Ordinary(OrdinaryKind::Plate)
                        | CombinationKind::Ordinary(OrdinaryKind::Tube)
                ));
        if sanctioned_endgame {
            score -= 10.0; // 残局唯一合法用法：轻微不鼓励
        } else {
            score -= if dw_endgame {
                DUAL_WILD_PENALTY_ENDGAME
            } else {
                DUAL_WILD_PENALTY_MIDGAME
            };
            if bare_dual {
                score -= BARE_DUAL_WILD_EXTRA_PENALTY; // 裸双百搭：额外重罚
            }
        }
    }

    // ── 三带二不能带两张级牌 (JS 1089-1096) ──
    if matches!(kind, CombinationKind::Ordinary(OrdinaryKind::FullHouse))
        && play_cards.len() >= 5
    {
        let mut rank_counts: HashMap<Rank, usize> = HashMap::new();
        for c in play_cards {
            if is_joker_sym(c) {
                continue;
            }
            if let Ok(card) = parse_card_symbol(c) {
                *rank_counts.entry(card.rank).or_default() += 1;
            }
        }
        let pair_part = rank_counts.iter().find(|&(_, &n)| n == 2).map(|(r, _)| *r);
        if pair_part == Some(p.level_rank) {
            score -= BANNED_SCORE; // 三带二不能带两张级牌，直接禁止
        }
    }

    // ── 4张级牌不能同时出 (JS 1099-1104) ──
    if play_cards.len() < my_remaining {
        let level_card_count = play_cards
            .iter()
            .filter(|c| {
                p.meta_for(c)
                    .map(|m| m.rank == p.level_rank && !m.is_wild)
                    .unwrap_or(false)
            })
            .count();
        if level_card_count >= 4 {
            score -= BANNED_SCORE; // 4张级牌不能同时出，直接禁止
        }
    }

    // ── 出炸弹要先小后大 (JS 1107-1117) ──
    if is_bomb {
        match kind {
            CombinationKind::Bomb(BombKind::SameRank { n: 5 }) => score -= 5.0,
            CombinationKind::Bomb(BombKind::SameRank { n: 6..=10 }) => score -= 15.0,
            _ => {}
        }
    }

    // ── 房规：接风重奖——队友已全部出完 (JS 1120-1125) ──
    if !is_last_play && p.teammate_remaining == 0 {
        score += 180.0; // 为队友接风：压制敌人拿回出牌权，重奖
    }

    // ── 房规：避免把自己打到「只剩小单张」(JS 1128-1148) ──
    if play_cards.len() < my_remaining {
        let mut used: HashMap<Rank, i32> = HashMap::new();
        for c in play_cards {
            if let Ok(card) = parse_card_symbol(c) {
                *used.entry(card.rank).or_default() += 1;
            }
        }
        let mut rest: HashMap<Rank, usize> = HashMap::new();
        let mut sm_has_bad = false;
        for hc in &p.my_hand {
            let Ok(card) = parse_card_symbol(hc) else { continue };
            let cnt = used.entry(card.rank).or_default();
            if *cnt > 0 {
                *cnt -= 1;
                continue;
            }
            let m = p.meta_for(hc);
            if m.map(|m| m.is_joker || m.is_wild).unwrap_or(false)
                || m.and_then(|m| m.natural).map_or(true, |nv| nv > 10)
            {
                sm_has_bad = true;
                break;
            }
            *rest.entry(card.rank).or_default() += 1;
        }
        let sm_vals: Vec<usize> = rest.values().copied().collect();
        if !sm_has_bad && sm_vals.len() >= 3 && sm_vals.iter().all(|&n| n == 1) {
            score -= sm_vals.len() as f32 * 22.0;
        }
    }

    // ── 清空手牌重奖 (JS 1152-1154) ──
    if play_cards.len() >= my_remaining {
        score += CLEAR_HAND_BONUS; // 清空手牌！重奖！
    }

    score
}

// ── JS scoreLeadPlay (L1465-1992): 领牌打分，base 50 ────────────────────

fn score_lead(play_cards: &[String], play_combo: &Combination, p: &PlayContext) -> f32 {
    let kind = &play_combo.kind;
    let mut score = BASE_LEAD_SCORE; // JS L1466 base score

    let my_remaining = p.my_remaining;
    let is_endgame = p.is_endgame;
    let combos = &p.combos;

    // ── Hand combo analysis & split penalty (JS 1473-1489) ──
    let bomb_split_verdict =
        classify_bomb_split(play_cards, &p.my_hand, kind, my_remaining);
    let play_is_bomb = play_combo.class() == CombinationClass::Bomb;
    // 领出路径：拆对豁免不适用（房规豁免仅限跟牌场景，用户 2026-08-30）
    let mut penalty =
        split_penalty(play_cards, combos, p.level_nat, p.has_level_card_or_joker, kind, false) as f32;
    if !play_is_bomb {
        if bomb_split_verdict == BombSplitVerdict::Exempt && penalty >= BANNED_SCORE {
            penalty = 0.0; // 房规豁免：放行拆炸
        }
        if bomb_split_verdict == BombSplitVerdict::Banned {
            penalty = penalty.max(BANNED_SCORE); // 双保险
        }
    }
    score -= penalty * SPLIT_PENALTY_SCALE;

    let is_bomb = play_is_bomb;

    // ── 手牌大于6张禁止空出王和级牌 (JS 1492-1502) ──
    if my_remaining > 6 && matches!(kind, CombinationKind::Ordinary(OrdinaryKind::Single)) {
        if let Some(first) = play_cards.first() {
            let m = p.meta_for(first);
            if m.map(|m| m.is_joker).unwrap_or(false) {
                score -= BANNED_SCORE; // 空出王：绝对禁止
            } else if m.map(|m| m.rank == p.level_rank).unwrap_or(false) {
                score -= BANNED_SCORE; // 空出级牌：绝对禁止
            }
        }
    }

    // ── Bomb conservation (JS 1505-1528) ──
    if is_bomb {
        if combos.bomb_count < 2 {
            score -= 30.0; // 必须留到残局，不能先出炸弹
        } else {
            score -= 40.0; // 留至少1个到残局控牌
        }

        if play_cards.len() >= my_remaining {
            score += 30.0; // Bonus: last card(s) played with bomb is ideal
        }

        match kind {
            CombinationKind::Bomb(BombKind::SameRank { n: 4 }) => score += 10.0,
            CombinationKind::Bomb(BombKind::StraightFlush) => score -= 15.0,
            CombinationKind::Bomb(BombKind::FourJoker) => score -= 30.0,
            _ => {}
        }
    }

    // ── 禁止出级牌炸弹 (JS 1531-1538) ──
    if is_bomb {
        let level_cards = play_cards
            .iter()
            .filter(|c| {
                p.meta_for(c)
                    .map(|m| m.rank == p.level_rank && !m.is_wild)
                    .unwrap_or(false)
            })
            .count();
        if level_cards >= 4 {
            score -= BANNED_SCORE; // 级牌炸弹，直接禁止
        }
    }

    // 手牌≤6张时必须保留至少1个炸弹 → 非炸弹出牌重奖 (JS 1541-1546)
    if !is_bomb && play_cards.len() < my_remaining {
        let remaining_after = my_remaining - play_cards.len();
        if remaining_after <= 6 && combos.bomb_count >= 1 {
            score += 500.0; // 保留炸弹到残局，重奖！
        }
    }

    // ── Team awareness: teammate sprinting (JS 1550-1571) ──
    let teammate_remaining = p.teammate_remaining;
    if teammate_remaining == 1 {
        if matches!(kind, CombinationKind::Ordinary(OrdinaryKind::Single)) {
            score += 40.0; // Strongly prefer singles to feed teammate
        } else if !is_bomb {
            score -= 20.0; // Discourage non-single plays when teammate has 1 card
        }
    } else if teammate_remaining == 2 {
        if matches!(kind, CombinationKind::Ordinary(OrdinaryKind::Pair)) {
            score += 40.0; // Strongly prefer pairs to feed teammate
        }
    } else if teammate_remaining == 3 {
        if matches!(kind, CombinationKind::Ordinary(OrdinaryKind::Triple)) {
            score += 40.0; // Strongly prefer triples to feed teammate
        }
    } else if teammate_remaining <= 6 {
        if matches!(
            kind,
            CombinationKind::Ordinary(OrdinaryKind::Pair)
                | CombinationKind::Ordinary(OrdinaryKind::Straight)
                | CombinationKind::Ordinary(OrdinaryKind::Tube)
                | CombinationKind::Ordinary(OrdinaryKind::Plate)
                | CombinationKind::Ordinary(OrdinaryKind::FullHouse)
        ) {
            score += 15.0; // Prefer combos to help teammate
        }
    }

    // ── Opponent interception: opponent sprinting (JS 1575-1619) ──
    let min_opp_remaining = p.min_opp_remaining;
    if min_opp_remaining == 1 {
        if matches!(kind, CombinationKind::Ordinary(OrdinaryKind::Single)) {
            score -= 50.0; // NEVER lead singles when opponent has 1 card
        }
        if !is_bomb {
            score += 10.0; // Prefer non-bomb plays to intercept
        }
    } else if min_opp_remaining == 2 {
        if matches!(kind, CombinationKind::Ordinary(OrdinaryKind::Pair)) {
            score -= 20.0; // Avoid leading pairs when opponent has 2 cards
        }
    } else if min_opp_remaining <= 6 {
        if matches!(kind, CombinationKind::Ordinary(OrdinaryKind::Single)) {
            if play_combo.primary <= 10 {
                score -= 30.0; // 不出小单张，对手可能吃单张
            } else {
                score += 15.0; // 出大单张阻止对手
            }
        }
        if matches!(kind, CombinationKind::Ordinary(OrdinaryKind::Pair)) {
            if play_combo.primary <= 10 {
                score -= 15.0; // 不出小对子
            } else {
                score += 15.0; // 出大对子阻止对手
            }
        }
        if matches!(
            kind,
            CombinationKind::Ordinary(OrdinaryKind::Straight)
                | CombinationKind::Ordinary(OrdinaryKind::Tube)
                | CombinationKind::Ordinary(OrdinaryKind::Plate)
                | CombinationKind::Ordinary(OrdinaryKind::FullHouse)
        ) && !is_bomb
        {
            score += 20.0; // 出组合牌型让对手拆牌，更难接
        }
    }

    // ── Endgame: play small cards first, keep big cards (JS 1623-1634) ──
    if is_endgame && !is_bomb {
        score += play_cards.len() as f32 * 5.0;
        score -= play_combo.primary as f32 * 1.5;
    } else {
        score += play_cards.len() as f32 * 8.0;
        score -= play_combo.primary as f32 * 0.5;
    }

    // ── 残局散牌处理：重奖移除单张的出牌 (JS 1638-1686) ──
    if my_remaining <= 6 && !is_bomb {
        let mut singles_removed = 0usize;
        let mut small_singles_removed = 0usize;
        for card in play_cards {
            let Some(m) = p.meta_for(card) else { continue };
            if m.is_joker || m.is_wild || m.rank == p.level_rank {
                continue;
            }
            let Some(nv) = m.natural else { continue };
            if combos.rank_to_count.get(&nv).copied().unwrap_or(0) == 1 {
                singles_removed += 1;
                if m.rank_value <= 10 {
                    small_singles_removed += 1;
                }
            }
        }
        if singles_removed > 0 {
            score += singles_removed as f32 * 400.0; // 残局移除单张，重奖！
        }
        if small_singles_removed > 0 {
            score += small_singles_removed as f32 * 300.0; // 残局移除小单张，额外重奖！
        }
        // 基础惩罚，让移除单张的出牌有净正收益
        let mut bad_singles = 0usize;
        let mut small_cards = 0usize;
        for card in &p.my_hand {
            let Some(m) = p.meta_for(card) else { continue };
            if m.is_joker || m.is_wild || m.rank == p.level_rank {
                continue;
            }
            let Some(nv) = m.natural else { continue };
            if combos.rank_to_count.get(&nv).copied().unwrap_or(0) == 1 {
                bad_singles += 1;
                if m.rank_value <= 10 {
                    small_cards += 1;
                }
            }
        }
        if bad_singles >= 1 {
            score -= 100.0; // 基础惩罚（远小于移除奖励）
        }
        if small_cards >= 1 {
            score -= 150.0; // 基础惩罚（远小于移除奖励）
        }
    }

    // ── 手牌有≥3张单牌能通过拆牌组成顺子奖励 (JS 1690-1715) ──
    if combos.singles_count >= 3 && !is_bomb {
        let mut single_natural_ranks: Vec<u8> = Vec::new();
        for card in &p.my_hand {
            if is_joker_sym(card) {
                continue;
            }
            if let Ok(c) = parse_card_symbol(card) {
                if let Ok(nv) = natural_rank_value(c.rank) {
                    if combos.rank_to_count.get(&nv).copied().unwrap_or(0) == 1 {
                        single_natural_ranks.push(nv);
                    }
                }
            }
        }
        single_natural_ranks.sort_unstable();
        let mut consecutive_count = 1usize;
        let mut max_consecutive = 1usize;
        for i in 1..single_natural_ranks.len() {
            if single_natural_ranks[i] - single_natural_ranks[i - 1] == 1 {
                consecutive_count += 1;
                max_consecutive = max_consecutive.max(consecutive_count);
            } else {
                consecutive_count = 1;
            }
        }
        if max_consecutive >= 3 {
            score += 30.0; // 单牌能组成顺子，奖励
        }
    }

    // ── 残局散牌惩罚: plays that leave scattered singles (JS 1719-1737) ──
    let remaining_after = my_remaining.saturating_sub(play_cards.len());
    if remaining_after > 1 && !is_bomb {
        let mut remaining_singles = combos.singles_count;
        for card in play_cards {
            if let Some(&nv) = combos.card_to_rank.get(card) {
                if combos.rank_to_count.get(&nv).copied().unwrap_or(0) == 1 {
                    remaining_singles = remaining_singles.saturating_sub(1);
                }
            }
        }
        let ratio_after = remaining_singles as f32 / remaining_after as f32;
        if ratio_after > 0.6 {
            score -= 150.0; // Heavy penalty: scattered singles
        } else if ratio_after > 0.4 {
            score -= 80.0; // Medium penalty
        } else if ratio_after > 0.2 {
            score -= 20.0; // Light penalty
        }
    }

    // ── 主动出牌：先出单张和小牌 (JS 1741-1754) ──
    if !is_bomb {
        let primary = play_combo.primary;
        if matches!(kind, CombinationKind::Ordinary(OrdinaryKind::Single)) {
            score += 20.0; // 单张优先出
            if primary <= 10 {
                score += 15.0; // 小单张更优先
            }
        }
        if primary > 10 {
            score -= 30.0; // 大牌绝不能先出
        } else {
            score += 10.0; // 小牌奖励
        }
    }

    // ── 逢人配优先组成炸弹、同花顺、顺子、钢板、木板 (JS 1759-1774) ──
    let has_wildcard = play_cards
        .iter()
        .any(|c| p.meta_for(c).map(|m| m.is_wild).unwrap_or(false));
    if has_wildcard {
        if is_bomb || matches!(kind, CombinationKind::Bomb(BombKind::StraightFlush)) {
            score += 100.0; // 逢人配配炸弹/同花顺：重奖！
        } else {
            match kind {
                CombinationKind::Ordinary(OrdinaryKind::Straight)
                | CombinationKind::Ordinary(OrdinaryKind::Plate)
                | CombinationKind::Ordinary(OrdinaryKind::Tube)
                | CombinationKind::Ordinary(OrdinaryKind::FullHouse) => score += 30.0,
                CombinationKind::Ordinary(OrdinaryKind::Triple) => score += 20.0,
                _ => {}
            }
        }
    }

    // ── 房规：已是天然炸弹再贴百搭升档 = 浪费 (JS 1779-1801) ──
    if has_wildcard && play_cards.len() < my_remaining {
        let wild_cnt = play_cards
            .iter()
            .filter(|c| p.meta_for(c).map(|m| m.is_wild).unwrap_or(false))
            .count();
        let naturals: Vec<CardMeta> = play_cards
            .iter()
            .filter_map(|c| p.meta_for(c))
            .filter(|m| !m.is_wild && !m.is_joker)
            .collect();
        let uniq: HashSet<Rank> = naturals.iter().map(|m| m.rank).collect();
        let naturals_already_bomb = naturals.len() >= 4
            && uniq.len() == 1
            && naturals.len() == play_cards.len() - wild_cnt;
        if naturals_already_bomb && naturals[0].rank != p.level_rank {
            if p.min_opp_remaining > DUAL_WILD_HAND_ENDGAME {
                if my_remaining <= DUAL_WILD_HAND_ENDGAME {
                    score -= UPGRADED_BOMB_WILD_PENALTY_ENDGAME;
                } else {
                    score -= UPGRADED_BOMB_WILD_PENALTY_MIDGAME;
                }
            }
        }
    }

    // ── 逢人配不能浪费（惩罚性检查）(JS 1804-1843) ──
    if has_wildcard {
        let finishing_play = play_cards.len() >= my_remaining;
        let endgame_hand = my_remaining <= DUAL_WILD_HAND_ENDGAME;

        let same_rank_type = matches!(
            kind,
            CombinationKind::Ordinary(OrdinaryKind::Pair)
                | CombinationKind::Ordinary(OrdinaryKind::Triple)
        ) || is_bomb;
        let touches_level_natural = play_cards.iter().any(|c| {
            p.meta_for(c)
                .map(|m| m.rank == p.level_rank && !m.is_wild)
                .unwrap_or(false)
        });
        if !finishing_play && same_rank_type && touches_level_natural {
            score -= if endgame_hand {
                WILD_ON_LEVEL_PENALTY_ENDGAME
            } else {
                WILD_ON_LEVEL_PENALTY_MIDGAME
            };
        }

        let lvl_face_bomb = is_bomb
            && play_cards.iter().all(|c| {
                p.meta_for(c)
                    .map(|m| m.is_wild || m.rank == p.level_rank)
                    .unwrap_or(false)
            });
        if !finishing_play && lvl_face_bomb {
            score -= if endgame_hand {
                DUAL_WILD_PENALTY_ENDGAME
            } else {
                DUAL_WILD_PENALTY_MIDGAME
            };
        }

        if matches!(kind, CombinationKind::Ordinary(OrdinaryKind::Single)) && !finishing_play {
            score -= BANNED_SCORE; // 逢人配绝不能单出——清空手牌绝对豁免
        } else if is_bomb || matches!(kind, CombinationKind::Bomb(BombKind::StraightFlush)) {
            // 最优使用，不罚
        } else if matches!(
            kind,
            CombinationKind::Ordinary(OrdinaryKind::Plate)
                | CombinationKind::Ordinary(OrdinaryKind::Tube)
                | CombinationKind::Ordinary(OrdinaryKind::FullHouse)
                | CombinationKind::Ordinary(OrdinaryKind::Triple)
                | CombinationKind::Ordinary(OrdinaryKind::Straight)
        ) {
            // 合理使用，不罚
        } else if matches!(kind, CombinationKind::Ordinary(OrdinaryKind::Pair)) {
            if !finishing_play {
                let pair_naturals: Vec<CardMeta> = play_cards
                    .iter()
                    .filter_map(|c| p.meta_for(c))
                    .filter(|m| !m.is_wild && !m.is_joker)
                    .collect();
                let pair_rank = pair_naturals.first().map(|m| m.rank).unwrap_or(p.level_rank);
                if pair_rank != p.level_rank {
                    score -= if endgame_hand {
                        WILD_PAIR_PENALTY_ENDGAME
                    } else {
                        WILD_PLAIN_PAIR_PENALTY_MIDGAME
                    };
                }
            }
        } else {
            score -= 10.0; // 其他非最优使用
        }
    }

    // ── 双百搭同出 (JS 1847-1872) ──
    let dw_finishing = play_cards.len() >= my_remaining;
    let dw_endgame = my_remaining <= DUAL_WILD_HAND_ENDGAME;

    if !is_bomb && bomb_split_verdict == BombSplitVerdict::Banned {
        score -= BANNED_SCORE; // 拆炸弹绝对禁止
    }

    let wild_count_in_play = play_cards
        .iter()
        .filter(|c| p.meta_for(c).map(|m| m.is_wild).unwrap_or(false))
        .count();
    if !dw_finishing && wild_count_in_play >= 2 {
        let dw_naturals: Vec<CardMeta> = play_cards
            .iter()
            .filter_map(|c| p.meta_for(c))
            .filter(|m| !m.is_wild && !m.is_joker)
            .collect();
        let dw_ranks: HashSet<Rank> = dw_naturals.iter().map(|m| m.rank).collect();
        let bare_dual = dw_naturals.is_empty();
        let sanctioned_endgame = dw_endgame
            && !bare_dual
            && !dw_ranks.is_empty()
            && !dw_ranks.contains(&p.level_rank)
            && (is_bomb
                || matches!(
                    kind,
                    CombinationKind::Ordinary(OrdinaryKind::FullHouse)
                        | CombinationKind::Ordinary(OrdinaryKind::Plate)
                        | CombinationKind::Ordinary(OrdinaryKind::Tube)
                ));
        if sanctioned_endgame {
            score -= 10.0; // 残局唯一合法用法：轻微不鼓励
        } else {
            score -= if dw_endgame {
                DUAL_WILD_PENALTY_ENDGAME
            } else {
                DUAL_WILD_PENALTY_MIDGAME
            };
            if bare_dual {
                score -= BARE_DUAL_WILD_EXTRA_PENALTY; // 裸双百搭：额外重罚
            }
        }
    }

    // ── 三带二不能带两张级牌 (JS 1875-1884) ──
    if matches!(kind, CombinationKind::Ordinary(OrdinaryKind::FullHouse))
        && play_cards.len() >= 5
    {
        let mut rank_counts: HashMap<Rank, usize> = HashMap::new();
        for c in play_cards {
            if is_joker_sym(c) {
                continue;
            }
            if let Ok(card) = parse_card_symbol(c) {
                *rank_counts.entry(card.rank).or_default() += 1;
            }
        }
        let pair_part = rank_counts.iter().find(|&(_, &n)| n == 2).map(|(r, _)| *r);
        if pair_part == Some(p.level_rank) {
            score -= BANNED_SCORE;
        }
    }

    // ── 4张级牌不能同时出 (JS 1887-1892) ──
    if play_cards.len() < my_remaining {
        let level_card_count = play_cards
            .iter()
            .filter(|c| {
                p.meta_for(c)
                    .map(|m| m.rank == p.level_rank && !m.is_wild)
                    .unwrap_or(false)
            })
            .count();
        if level_card_count >= 4 {
            score -= BANNED_SCORE;
        }
    }

    // ── 出炸弹要先小后大 (JS 1895-1908) ──
    if is_bomb {
        match kind {
            CombinationKind::Bomb(BombKind::SameRank { n: 5 }) => score -= 5.0,
            CombinationKind::Bomb(BombKind::SameRank { n: 6..=10 }) => score -= 15.0,
            _ => {}
        }
    }

    // Don't lead with level cards or jokers (save for intercepting) (JS 1911-1917)
    if play_cards.len() < my_remaining {
        let has_level = play_cards.iter().any(|c| {
            p.meta_for(c)
                .map(|m| m.rank == p.level_rank && !m.is_wild)
                .unwrap_or(false)
        });
        let has_joker = play_cards.iter().any(|c| is_joker_sym(c));
        if has_level || has_joker {
            score -= 40.0; // Never lead with level cards or jokers
        }
    }

    // ── 房规：先出小牌，不要空出大牌 (JS 1920-1930) ──
    if play_cards.len() < my_remaining {
        let mut lead_max_nv = 0u8;
        for c in play_cards {
            let m = p.meta_for(c);
            let nv = if m.map(|m| m.is_joker).unwrap_or(false) {
                16
            } else if m.map(|m| m.is_wild).unwrap_or(false) {
                15
            } else {
                m.and_then(|m| m.natural).unwrap_or(0)
            };
            lead_max_nv = lead_max_nv.max(nv);
        }
        if lead_max_nv >= 11 {
            score -= (lead_max_nv - 10) as f32 * 18.0; // J −18 / Q −36 / K −54 / A −72 / 王 −108
        }
    }

    // ── 房规：接风重奖——队友已全部出完，本圈由我接风先出 (JS 1933-1937) ──
    if p.teammate_remaining == 0 {
        score += 120.0; // 接风首出权重奖
    }

    // ── 房规：空出炸弹重罚 (JS 1940-1960) ──
    if is_bomb && play_cards.len() < my_remaining {
        let mut used: HashMap<Rank, i32> = HashMap::new();
        for c in play_cards {
            if let Ok(card) = parse_card_symbol(c) {
                *used.entry(card.rank).or_default() += 1;
            }
        }
        let mut rest_groups: HashMap<Rank, usize> = HashMap::new();
        let mut rest_has_joker = false;
        for hc in &p.my_hand {
            let Ok(card) = parse_card_symbol(hc) else { continue };
            let cnt = used.entry(card.rank).or_default();
            if *cnt > 0 {
                *cnt -= 1;
                continue;
            }
            if card.suit == Suit::Joker {
                rest_has_joker = true;
                continue;
            }
            *rest_groups.entry(card.rank).or_default() += 1;
        }
        let rest_vals: Vec<usize> = rest_groups.values().copied().collect();
        let all_bombs_rest = !rest_has_joker
            && (rest_vals.is_empty() || rest_vals.iter().all(|&n| n >= 4));
        if !all_bombs_rest {
            score -= 450.0; // 空出炸弹：手里还有非炸弹牌却主动领炸，严重浪费
        }
    }

    // ── 房规：避免把自己打到「只剩小单张」(JS 1963-1983) ──
    if play_cards.len() < my_remaining {
        let mut used: HashMap<Rank, i32> = HashMap::new();
        for c in play_cards {
            if let Ok(card) = parse_card_symbol(c) {
                *used.entry(card.rank).or_default() += 1;
            }
        }
        let mut rest: HashMap<Rank, usize> = HashMap::new();
        let mut sm_has_bad = false;
        for hc in &p.my_hand {
            let Ok(card) = parse_card_symbol(hc) else { continue };
            let cnt = used.entry(card.rank).or_default();
            if *cnt > 0 {
                *cnt -= 1;
                continue;
            }
            let m = p.meta_for(hc);
            if m.map(|m| m.is_joker || m.is_wild).unwrap_or(false)
                || m.and_then(|m| m.natural).map_or(true, |nv| nv > 10)
            {
                sm_has_bad = true;
                break;
            }
            *rest.entry(card.rank).or_default() += 1;
        }
        let sm_vals: Vec<usize> = rest.values().copied().collect();
        if !sm_has_bad && sm_vals.len() >= 3 && sm_vals.iter().all(|&n| n == 1) {
            score -= sm_vals.len() as f32 * 22.0; // 剩3张-66 … 剩5张-110
        }
    }

    // ── 清空手牌重奖 (JS 1987-1989) ──
    if play_cards.len() == my_remaining {
        score += CLEAR_HAND_BONUS; // 清空手牌！重奖！
    }

    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::card::HandLevel;
    use crate::game::types::PlayState;

    fn ctx() -> RuleContext {
        RuleContext {
            hand_level: HandLevel::Two,
        }
    }

    // ══ 房规回归测试（2026-08-30）：残局拆对/拆三张豁免三条件（残局+无单张+队友过牌）══

    #[test]
    fn endgame_split_pair_requires_allow_flag() {
        // [♠5,♥5,♠8,♥8]（残局 4 张、无单张）跟单张 3：
        // 三条件齐（flag=true）→ 允许拆对（penalty 0）；缺任一（flag=false）→ 99999 禁止。
        let hand: Vec<String> = vec!["♠5", "♥5", "♠8", "♥8"].into_iter().map(String::from).collect();
        let combos = analyze_hand_combos(&hand, ctx());
        assert_eq!(combos.singles_count, 0, "test hand must have no singles");
        let play = vec!["♠5".to_string()];
        let kind = CombinationKind::Ordinary(OrdinaryKind::Single);
        assert_eq!(
            split_penalty(&play, &combos, 2, false, &kind, true),
            0,
            "flag=true (残局+无单张+队友过牌) must allow pair split"
        );
        assert_eq!(
            split_penalty(&play, &combos, 2, false, &kind, false),
            BANNED_SCORE_U32,
            "flag=false must keep the absolute ban"
        );
        // 拆三张同理：[♠7,♥7,♦7,♠8,♥8,♠9,♥9]（无单张）拆 7 出单
        let hand3: Vec<String> = vec!["♠7", "♥7", "♦7", "♠8", "♥8", "♠9", "♥9"]
            .into_iter().map(String::from).collect();
        let combos3 = analyze_hand_combos(&hand3, ctx());
        let play3 = vec!["♠7".to_string()];
        assert_eq!(
            split_penalty(&play3, &combos3, 2, false, &kind, true),
            0,
            "flag=true must allow triple split"
        );
        assert_eq!(
            split_penalty(&play3, &combos3, 2, false, &kind, false),
            BANNED_SCORE_U32,
            "flag=false must keep the triple-split ban"
        );
    }

    #[test]
    fn never_override_teammate_plate_and_tube() {
        // 房规：队友的钢板/木板绝对不能压——有更大同型也不行，持炸也不行。
        // 队友 S 领出木板 334455；N 持更大木板 667788 + 炸 KKKK → 必须过。
        let state = mk_playing_state(
            Seat::N,
            vec!["♠6", "♥6", "♠7", "♥7", "♠8", "♥8", "♠K", "♥K", "♦K", "♣K"],
            Some((Seat::S, vec!["♦3", "♣3", "♦4", "♣4", "♦5", "♣5"])),
        );
        let act = suggest_next_action(&state, Seat::N).unwrap();
        assert!(
            matches!(act, PlayerAction::Pass),
            "must pass teammate's tube, got {act:?}"
        );

        // 队友 S 领出钢板 333444；N 持更大钢板 555666 + 炸 KKKK → 必须过。
        let state = mk_playing_state(
            Seat::N,
            vec!["♠5", "♥5", "♦5", "♠6", "♥6", "♦6", "♠K", "♥K", "♦K", "♣K"],
            Some((Seat::S, vec!["♦3", "♣3", "♥3", "♦4", "♣4", "♥4"])),
        );
        let act = suggest_next_action(&state, Seat::N).unwrap();
        assert!(
            matches!(act, PlayerAction::Pass),
            "must pass teammate's plate, got {act:?}"
        );
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

    fn fill_seats(
        state: &mut TableGameState,
        n_cards: Vec<&str>,
        s_cards: Vec<&str>,
        w_cards: Vec<&str>,
    ) {
        if let Some(hand) = state.hand.as_mut() {
            hand.hands.insert(Seat::N, n_cards.into_iter().map(ToString::to_string).collect());
            hand.hands.insert(Seat::S, s_cards.into_iter().map(ToString::to_string).collect());
            hand.hands.insert(Seat::W, w_cards.into_iter().map(ToString::to_string).collect());
        }
    }

    fn mk_top(seat: Seat, cards: Vec<&str>) -> PlayState {
        let cards: Vec<String> = cards.into_iter().map(ToString::to_string).collect();
        let combo = CombinationParser::parse(&cards, None, ctx()).unwrap();
        PlayState {
            seat,
            cards,
            wild_targets: None,
            combination: combo,
        }
    }

    fn pctx_of(state: &TableGameState, actor: Seat) -> PlayContext {
        let hand = state.hand.as_ref().unwrap();
        build_play_context(hand, actor, ctx())
    }

    fn combo_of(cards: Vec<&str>, targets: Vec<&str>) -> Combination {
        let cards: Vec<String> = cards.into_iter().map(ToString::to_string).collect();
        let targets: Vec<String> = targets.into_iter().map(ToString::to_string).collect();
        CombinationParser::parse(&cards, Some(&targets), ctx()).unwrap()
    }

    // ══ 牌踪器（rank 级剩余牌统计）测试 ══

    use crate::game::types::{HandHistoryEntry, HistoryActionKind};

    fn history_entry(seat: Seat, cards: Vec<&str>) -> HandHistoryEntry {
        HandHistoryEntry {
            seq: 0,
            action_id: "t".into(),
            seat,
            timestamp: String::new(),
            action_type: HistoryActionKind::Play,
            cards: cards.into_iter().map(ToString::to_string).collect(),
            combination_type: None,
            wild_targets: None,
        }
    }

    #[test]
    fn pool_stats_counts_double_deck_correctly() {
        // 我手 3 张 + N 已出 2 张 → 108−5=103；rank3 见 4 张余 4；红王见 1 余 1
        let mut state = mk_playing_state(Seat::E, vec!["♠3", "♥3", "🃏R"], None);
        fill_seats(
            &mut state,
            vec!["♦3", "♣3"],
            vec!["♠4", "♥4", "♦4", "♣4", "♠5", "♥5", "♦5", "♣5", "♠6"],
            vec!["♥6", "♦6", "♣6", "♠7", "♥7", "♦7", "♣7", "♠8", "♥8"],
        );
        state.hand.as_mut().unwrap().history = vec![history_entry(Seat::N, vec!["♦3", "♣3"])];
        let p = pctx_of(&state, Seat::E);
        assert_eq!(p.pool.total, 103, "双副牌108 − 已见5");
        assert_eq!(p.pool.rank_counts[3], 4, "rank3: 8−4=4");
        assert_eq!(p.pool.rank_counts[17], 1, "红王: 2−1=1");
        assert_eq!(
            p.pool.suffix_ge[15],
            p.pool.rank_counts[15] + p.pool.rank_counts[16] + p.pool.rank_counts[17],
            "≥2 = 点数2 + 双王"
        );
    }

    #[test]
    fn prob_functions_sanity() {
        let mut state = mk_playing_state(Seat::E, vec!["♠3", "♥3", "🃏R"], None);
        fill_seats(
            &mut state,
            vec!["♦3", "♣3", "♠4", "♥4", "♦4", "♣4", "♠5", "♥5", "♦5"],
            vec!["♣5", "♠6", "♥6", "♦6", "♣6", "♠7", "♥7", "♦7", "♣7"],
            vec!["♠8", "♥8", "♦8", "♣8", "♠9", "♥9", "♦9", "♣9", "♠10"],
        );
        state.hand.as_mut().unwrap().history = vec![history_entry(Seat::N, vec!["♦3", "♣3"])];
        let p = pctx_of(&state, Seat::E);
        assert_eq!(p.prob_rank_in_hand(Seat::E, 3), 0.0, "自己恒 0");
        assert_eq!(p.prob_has_bomb(Seat::E), 0.0, "自己恒 0");
        let has_rank3 = p.opponent_has_rank(3);
        assert!(has_rank3 > 0.0 && has_rank3 <= 1.0, "rank3 概率 (0,1]，got {has_rank3}");
        let bomb = p.opponent_bomb_prob();
        assert!((0.0..=1.0).contains(&bomb), "炸弹概率 ∈[0,1], got {bomb}");
        let win = p.game_win_prob();
        assert!((0.0..=1.0).contains(&win), "胜率 ∈[0,1], got {win}");
    }

    #[test]
    fn mutation_never_moves_house_thresholds() {
        // 房规锁：mutate_random 只动 10 个 f32 权重 + 2 个概率阈值，3 个 u8 阈值不可动
        let base = crate::strategy::suggest::js_trained_params();
        for _ in 0..300 {
            let m = base.mutate_random(0.05);
            assert_eq!(m.enemy_low_cards_threshold, 6, "冲刺阈值必须锁死为 6");
            assert_eq!(m.endgame_hand_count_threshold, 6, "残局阈值必须锁死为 6");
            assert_eq!(m.partner_sprint_threshold, 2, "队友冲刺阈值必须锁死为 2");
        }
    }

    // ══ JS 兼容存量测试（行为与 JS 规则一致，保持原样）══

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
    fn avoids_splitting_bomb_when_leading() {
        // 手 [♠3,♥3,♦3,♣3(炸弹),♠5] 领出：JS 领牌分下 ♠5（单张+45 且不拆炸）
        // 胜过拆/出炸弹（空出炸弹 −450）。
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

    // ══ 存量失败测试：按任务要求原样保留、必须继续失败 ══

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

    // ══ 房规回归测试：不得把百搭（逢人配）留成最后一张孤牌 ══

    #[test]
    fn two_card_wild_hand_pairs_out_instead_of_stranding() {
        // [♦9, ♥2(级2百搭)] 领出：必须出对 9（百搭当 9）一手清空获胜
        //（清空 +10000 自然产生，JS 语义）。
        let state = mk_playing_state(Seat::E, vec!["♦9", "♥2"], None);
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        match &picked {
            PlayerAction::Play { cards, wild_targets } => {
                assert_eq!(
                    cards.len(), 2,
                    "must pair out with wild instead of stranding it: {:?}",
                    picked
                );
                assert!(wild_targets.is_some(), "wild pair must declare targets");
            }
            _ => panic!("Expected Play action, got {:?}", picked),
        }
    }

    #[test]
    fn clearing_triple_with_wild_beats_natural_pair() {
        // [♠9, ♥9, ♥2(百搭)] 领出：三张 999（清空即胜 +10000）必须压过对 99。
        let state = mk_playing_state(Seat::E, vec!["♠9", "♥9", "♥2"], None);
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        match &picked {
            PlayerAction::Play { cards, .. } => {
                assert_eq!(
                    cards.len(), 3,
                    "clearing triple (instant win) must beat non-clearing pair: {:?}",
                    picked
                );
            }
            _ => panic!("Expected Play action, got {:?}", picked),
        }
    }

    #[test]
    fn never_leaves_wild_as_lone_leftover() {
        // [♠3,♥3,♦3,♣3,♥2(百搭)] 领出：JS 清空 +10000 → 3333+百搭成 5 炸清空
        //（对手剩 0 张不触发升档罚），绝不留 [♥2] 孤张。
        let state = mk_playing_state(Seat::E, vec!["♠3", "♥3", "♦3", "♣3", "♥2"], None);
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        if let PlayerAction::Play { cards, .. } = &picked {
            let mut left: Vec<&str> = vec!["♠3", "♥3", "♦3", "♣3", "♥2"]
                .into_iter()
                .filter(|c| !cards.iter().any(|s| s.as_str() == *c))
                .collect();
            left.sort();
            assert_ne!(
                left,
                vec!["♥2"],
                "must not strand the wild as the lone leftover: picked {:?}",
                cards
            );
        } else {
            panic!("Expected Play action, got {:?}", picked);
        }
    }

    #[test]
    fn endgame_wild_converts_to_straight_or_straight_flush() {
        // [♠5,♠6,♠7,♠8,♠9,♥2(级2百搭)] 残局（6张）领出：JS 百搭并入杂顺
        //（+30 且 +400/张 移除单张奖励）远胜同花顺（空出炸弹 −450）。
        let state = mk_playing_state(Seat::E, vec!["♠5", "♠6", "♠7", "♠8", "♠9", "♥2"], None);
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        match &picked {
            PlayerAction::Play { cards, wild_targets } => {
                let combo = CombinationParser::parse(cards, wild_targets.as_deref(), ctx())
                    .expect("candidate must parse");
                let premium = matches!(
                    combo.kind,
                    CombinationKind::Bomb(BombKind::StraightFlush)
                        | CombinationKind::Ordinary(OrdinaryKind::Straight)
                );
                assert!(
                    premium && cards.iter().any(|s| s.as_str() == "♥2"),
                    "endgame wild must convert to a straight / straight flush, got {:?}",
                    picked
                );
            }
            _ => panic!("Expected Play action, got {:?}", picked),
        }
    }

    // ══ JS 行为测试 ①：百搭配炸弹 +100 应胜过百搭配顺子 +30（同手牌两候选）══

    #[test]
    fn wild_bomb_bonus_beats_wild_straight_bonus() {
        // JS scorePlay：百搭进炸弹 +100 vs 百搭进顺子 +30。
        // 同手牌 [♠5,♥5,♦5,♠6..♠A,♠3,♥2]（14张中盘），对单张4：
        // 555+百搭（炸弹4）得分必须高于 百搭补4 的 4-8 顺子。
        // （手牌 14 张 → 两候选出后均剩 >6 张，+500 保留炸弹奖励对二者皆不触发。）
        let mut state = mk_playing_state(
            Seat::E,
            vec![
                "♠5", "♥5", "♦5", "♠6", "♠7", "♠8", "♠9", "♠10", "♠J", "♠Q", "♠K", "♠A", "♠3",
                "♥2",
            ],
            Some((Seat::N, vec!["♠4"])),
        );
        fill_seats(
            &mut state,
            vec!["♦4", "♣4", "♦5", "♣5", "♦6"], // N 5 (minOpp ≤ 6 → 无 −200 保留罚)
            vec!["♥6", "♣6", "♦7", "♣7", "♦8"], // S 5
            vec!["♥9", "♦9", "♣9", "♥10", "♦10", "♣10", "♦J", "♣J", "♣Q"], // W 9
        );
        let p = pctx_of(&state, Seat::E);
        let top = mk_top(Seat::N, vec!["♠4"]);

        let bomb_cards = vec!["♠5".to_string(), "♥5".to_string(), "♦5".to_string(), "♥2".to_string()];
        let bomb_combo = combo_of(vec!["♠5", "♥5", "♦5", "♥2"], vec!["♣5"]);
        let straight_cards = vec![
            "♥2".to_string(),
            "♠5".to_string(),
            "♠6".to_string(),
            "♠7".to_string(),
            "♠8".to_string(),
        ];
        let straight_combo = combo_of(vec!["♥2", "♠5", "♠6", "♠7", "♠8"], vec!["♦4"]);

        let bomb_score = score_follow(&bomb_cards, &bomb_combo, &top, &p);
        let straight_score = score_follow(&straight_cards, &straight_combo, &top, &p);
        assert!(
            bomb_score > straight_score,
            "wild-in-bomb (+100) must outscore wild-in-straight (+30): bomb={bomb_score} straight={straight_score}"
        );
    }

    // ══ JS 行为测试 ②：百搭单出 −99999 禁止（清空豁免）══

    #[test]
    fn wild_single_banned_unless_clearing() {
        // 跟牌单元：百搭单出（非清空）→ −99999 绝对禁止。
        let mut state = mk_playing_state(
            Seat::E,
            vec!["♥2", "♠3", "♠4"],
            Some((Seat::N, vec!["♠5"])),
        );
        fill_seats(&mut state, vec!["♦5"], vec!["♥5"], vec!["♦6"]);
        let p = pctx_of(&state, Seat::E);
        let top = mk_top(Seat::N, vec!["♠5"]);

        let single_cards = vec!["♥2".to_string()];
        let single_combo = combo_of(vec!["♥2"], vec!["♠K"]);
        let s = score_follow(&single_cards, &single_combo, &top, &p);
        assert!(s < -50000.0, "wild single (non-clearing) must be banned: {s}");

        // 清空豁免：手中只剩 1 张百搭时，单出 = 清空 → +10000。
        let mut clear_state = mk_playing_state(Seat::E, vec!["♥2"], Some((Seat::N, vec!["♠3"])));
        fill_seats(&mut clear_state, vec!["♦5"], vec!["♥5"], vec!["♦6"]);
        let cp = pctx_of(&clear_state, Seat::E);
        let ctop = mk_top(Seat::N, vec!["♠3"]);
        let clear_cards = vec!["♥2".to_string()];
        let clear_combo = combo_of(vec!["♥2"], vec!["♠K"]);
        let cs = score_follow(&clear_cards, &clear_combo, &ctop, &cp);
        assert!(cs > 9000.0, "clearing wild single must be exempt: {cs}");

        // 端到端：全百搭两手牌领出 → 必须出对子清空（单出 −99999 被自然排除）。
        let wild_pair_state = mk_playing_state(Seat::E, vec!["♥2", "♥2"], None);
        let picked = suggest_next_action(&wild_pair_state, Seat::E).unwrap();
        match &picked {
            PlayerAction::Play { cards, .. } => {
                assert_eq!(cards.len(), 2, "all-wild hand must pair out, got {:?}", picked);
            }
            other => panic!("Expected Play (pair), got {:?}", other),
        }
    }

    // ══ JS 行为测试 ③：天然炸弹+百搭升档 中盘 −150 / 残局 −10（非级牌）══

    #[test]
    fn upgraded_bomb_wild_penalty_midgame_vs_endgame() {
        let play_cards = vec!["♠5", "♥5", "♦5", "♣5", "♥2"];
        let bomb5 = combo_of(play_cards.clone(), vec!["♦5"]);
        let top = mk_top(Seat::N, vec!["♦4", "♣4", "♥4", "♠9", "♦9"]); // FH 444+99

        // 中盘（7 张，minOpp 9 > 6）：升档 −150 + 炸弹保留 −200
        let mut mid = mk_playing_state(
            Seat::E,
            vec!["♠5", "♥5", "♦5", "♣5", "♥2", "♠8", "♠9"],
            None,
        );
        fill_seats(
            &mut mid,
            vec!["♠4", "♥4", "♣5", "♥5", "♦6", "♣6", "♥6", "♦7", "♣7"],
            vec!["♠6", "♥7", "♦8", "♣8", "♠10", "♥10", "♦10", "♣10", "♠J"],
            vec!["♥J", "♦J", "♣J", "♠Q", "♥Q", "♦Q", "♣Q", "♠K", "♦K"],
        );
        let mid_p = pctx_of(&mid, Seat::E);
        let mid_cards: Vec<String> = play_cards.iter().map(|s| s.to_string()).collect();
        let mid_score = score_follow(&mid_cards, &bomb5, &top, &mid_p);

        // 残局（6 张，minOpp 9 > 3 → 守卫②豁免：打完剩 1 张 ≤ 2）：升档仅 −10
        let mut endg = mk_playing_state(
            Seat::E,
            vec!["♠5", "♥5", "♦5", "♣5", "♥2", "♠8"],
            None,
        );
        fill_seats(
            &mut endg,
            vec!["♠4", "♥4", "♣5", "♥5", "♦6", "♣6", "♥6", "♦7", "♣7"],
            vec!["♠6", "♥7", "♦8", "♣8", "♠10", "♥10", "♦10", "♣10", "♠J"],
            vec!["♥J", "♦J", "♣J", "♠Q", "♥Q", "♦Q", "♣Q", "♠K", "♦K"],
        );
        let end_p = pctx_of(&endg, Seat::E);
        let end_score = score_follow(&mid_cards, &bomb5, &top, &end_p);

        assert!(mid_score < 0.0, "midgame upgraded bomb must be heavily penalized: {mid_score}");
        assert!(end_score > 0.0, "endgame upgraded bomb must stay viable: {end_score}");
        assert!(
            end_score - mid_score > 100.0,
            "endgame (−10) must beat midgame (−150) by >100: {end_score} vs {mid_score}"
        );
    }

    // ══ JS 行为测试 ④：残局移除单张 +400/张（小单张另 +300/张）══

    #[test]
    fn endgame_lead_single_removal_reward() {
        // 残局（6张）领出，手 [♠3,♥3,♠7,♠K,♠8,♠9]：JS +400/+300 使小单张
        // （♠7/♠8/♠9）远胜对子/大牌单张；♠7（primary 最小）胜出。
        let mut state = mk_playing_state(
            Seat::E,
            vec!["♠3", "♥3", "♠7", "♠K", "♠8", "♠9"],
            None,
        );
        fill_seats(
            &mut state,
            vec!["♦3", "♣3", "♦7", "♣7", "♦8", "♣8", "♦9", "♣9", "♦4"],
            vec!["♠4", "♥4", "♠5", "♥5", "♦5", "♣5", "♠6", "♥6", "♦6"],
            vec!["♠10", "♥10", "♦10", "♣10", "♠J", "♥J", "♦J", "♣J", "♠Q"],
        );
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        assert_eq!(
            picked,
            PlayerAction::Play {
                cards: vec!["♠7".into()],
                wild_targets: None,
            },
            "endgame removal reward must prefer the small single, got {:?}",
            picked
        );
    }

    // ══ JS 行为测试 ⑤：拆炸弹 banned —— 残局拆炸必须禁止（Pass 或换牌）══

    #[test]
    fn endgame_split_bomb_banned_prefers_alternative() {
        // 残局（6张）跟对6：拆 5555 出对 55 被 classify_bomb_split=Banned 绝对禁止；
        // 结果只能是 Pass（炸弹压对子被守卫⑤拦截）或换对 77。
        let mut state = mk_playing_state(
            Seat::E,
            vec!["♠5", "♥5", "♦5", "♣5", "♠7", "♥7"],
            Some((Seat::N, vec!["♠6", "♥6"])),
        );
        fill_seats(
            &mut state,
            vec!["♦6", "♣6", "♦8", "♣8", "♦9", "♣9", "♦10", "♣10", "♦J"],
            vec!["♠8", "♥8", "♠9", "♥9", "♠10", "♥10", "♣10", "♠J", "♥J"],
            vec!["♠Q", "♥Q", "♦Q", "♣Q", "♠K", "♥K", "♦K", "♣K", "♠A"],
        );
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        match &picked {
            PlayerAction::Pass => {}
            PlayerAction::Play { cards, .. } => {
                let is_banned_split = cards.len() == 2
                    && cards.iter().all(|c| c.ends_with('5'));
                assert!(
                    !is_banned_split,
                    "splitting the bomb into a pair is banned in endgame, got {:?}",
                    picked
                );
            }
            other => panic!("Unexpected action, got {:?}", other),
        }
    }

    #[test]
    fn endgame_split_bomb_banned_forces_pass() {
        // 残局（6张）跟对6：手 [5555,K,A] 无合法替牌 → 炸弹被守卫⑤拦截 → 必须 Pass。
        let mut state = mk_playing_state(
            Seat::E,
            vec!["♠5", "♥5", "♦5", "♣5", "♠K", "♠A"],
            Some((Seat::N, vec!["♠6", "♥6"])),
        );
        fill_seats(
            &mut state,
            vec!["♦6", "♣6", "♦8", "♣8", "♦9", "♣9", "♦10", "♣10", "♦J"],
            vec!["♠8", "♥8", "♠9", "♥9", "♠10", "♥10", "♣10", "♠J", "♥J"],
            vec!["♠Q", "♥Q", "♦Q", "♣Q", "♥K", "♦K", "♣K", "♠A", "♥A"],
        );
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        assert_eq!(picked, PlayerAction::Pass, "no legal alternative → must Pass");
    }

    // ══ JS 行为测试 ⑥：双百搭矩阵（残局 −10 轻罚 / 其余 −60、−600、裸出 −200）══

    #[test]
    fn dual_wild_penalty_matrix() {
        let top = mk_top(Seat::N, vec!["♦4", "♣4", "♥4", "♠9", "♦9"]); // FH 444+99
        let fills = |st: &mut TableGameState| {
            fill_seats(
                st,
                vec!["♠3", "♥3", "♦3", "♣3", "♠4", "♥4", "♠6", "♥6", "♦6"],
                vec!["♣6", "♠7", "♥7", "♦7", "♣7", "♠10", "♥10", "♦10", "♣10"],
                vec!["♠J", "♥J", "♦J", "♣J", "♠Q", "♥Q", "♦Q", "♣Q", "♠A"],
            );
        };

        // 残局 sanctioned：非级牌对 + 双百搭 = 四头炸（−10 轻罚）
        let mut end_state = mk_playing_state(Seat::E, vec!["♠5", "♦5", "♥2", "♥2", "♠9", "♠8"], None);
        fills(&mut end_state);
        let end_p = pctx_of(&end_state, Seat::E);
        let bomb4_cards = vec!["♠5".to_string(), "♦5".to_string(), "♥2".to_string(), "♥2".to_string()];
        let bomb4 = combo_of(vec!["♠5", "♦5", "♥2", "♥2"], vec!["♣5", "♥5"]);
        let sanctioned_end = score_follow(&bomb4_cards, &bomb4, &top, &end_p);

        // 中盘同型（8 张）：双百搭同出 −600 重罚
        let mut mid_state = mk_playing_state(
            Seat::E,
            vec!["♠5", "♦5", "♥2", "♥2", "♠9", "♠8", "♠7", "♠6"],
            None,
        );
        fills(&mut mid_state);
        let mid_p = pctx_of(&mid_state, Seat::E);
        let sanctioned_mid = score_follow(&bomb4_cards, &bomb4, &top, &mid_p);

        // 残局落级牌：百搭+级牌同点炸 → 不 sanctioned（−60）+ 落级牌（−20）+ 级牌面值炸（−60）
        let mut lvl_state = mk_playing_state(Seat::E, vec!["♠2", "♦2", "♥2", "♥2", "♠9", "♠8"], None);
        fills(&mut lvl_state);
        let lvl_p = pctx_of(&lvl_state, Seat::E);
        let lvl_bomb_cards = vec!["♠2".to_string(), "♦2".to_string(), "♥2".to_string(), "♥2".to_string()];
        let lvl_bomb = combo_of(vec!["♠2", "♦2", "♥2", "♥2"], vec!["♣2", "♦2"]);
        let level_end = score_follow(&lvl_bomb_cards, &lvl_bomb, &top, &lvl_p);

        // 残局裸双百搭成普通对：−60 −200，必然低于 sanctioned
        let mut bare_state = mk_playing_state(Seat::E, vec!["♥2", "♥2", "♠9", "♠8", "♠7", "♠6"], None);
        fills(&mut bare_state);
        let bare_p = pctx_of(&bare_state, Seat::E);
        let bare_cards = vec!["♥2".to_string(), "♥2".to_string()];
        let bare_pair = combo_of(vec!["♥2", "♥2"], vec!["♠3", "♥3"]);
        let bare_end = score_follow(&bare_cards, &bare_pair, &top, &bare_p);

        // 中盘裸双百搭：−600 −200，更差（避免 5 连同花 → 不产生 bombCount）
        let mut bare_mid_state = mk_playing_state(
            Seat::E,
            vec!["♥2", "♥2", "♠9", "♠8", "♠7", "♠6", "♦5", "♣4"],
            None,
        );
        fills(&mut bare_mid_state);
        let bare_mid_p = pctx_of(&bare_mid_state, Seat::E);
        let bare_mid = score_follow(&bare_cards, &bare_pair, &top, &bare_mid_p);

        assert!(
            sanctioned_end - sanctioned_mid > 500.0,
            "endgame sanctioned dual-wild bomb (−10) must beat midgame (−600) by >500: {sanctioned_end} vs {sanctioned_mid}"
        );
        assert!(
            sanctioned_end > level_end,
            "non-level dual-wild bomb (−10) must beat level-touching dual-wild bomb: {sanctioned_end} vs {level_end}"
        );
        assert!(
            bare_end < sanctioned_end - 200.0,
            "bare dual wild (−60−200) must be worse than sanctioned: {bare_end} vs {sanctioned_end}"
        );
        assert!(
            bare_mid < bare_end - 400.0,
            "midgame bare dual wild (−600−200) must be far worse: {bare_mid} vs {bare_end}"
        );
    }

    // ══ −99999 禁令：级牌炸弹 / 三带二带级牌对 ══

    #[test]
    fn level_bomb_and_four_level_cards_are_banned() {
        let mut state = mk_playing_state(
            Seat::E,
            vec!["♠2", "♠2", "♦2", "♣2", "♠8", "♠9"],
            None,
        );
        fill_seats(
            &mut state,
            vec!["♠4", "♥4", "♦4", "♣4", "♠6", "♥6", "♦6", "♣6", "♠7"],
            vec!["♥7", "♦7", "♣7", "♠8", "♥8", "♦8", "♣8", "♠10", "♥10"],
            vec!["♦10", "♣10", "♠J", "♥J", "♦J", "♣J", "♠Q", "♥Q", "♦Q"],
        );
        let p = pctx_of(&state, Seat::E);
        let top = mk_top(Seat::N, vec!["♦4", "♣4", "♥4", "♠9", "♦9"]);
        let lvl_bomb_cards = vec!["♠2".to_string(), "♠2".to_string(), "♦2".to_string(), "♣2".to_string()];
        let lvl_bomb = CombinationParser::parse(&lvl_bomb_cards, None, ctx()).unwrap();
        let s = score_follow(&lvl_bomb_cards, &lvl_bomb, &top, &p);
        assert!(s < -50000.0, "level bomb (4 level cards) must be banned: {s}");
    }

    #[test]
    fn fullhouse_with_level_pair_banned() {
        let mut state = mk_playing_state(
            Seat::E,
            vec!["♠5", "♥5", "♦5", "♠2", "♦2", "♠8"],
            None,
        );
        fill_seats(
            &mut state,
            vec!["♠4", "♥4", "♦4", "♣4", "♣5", "♥5", "♦5", "♣5", "♠6"],
            vec!["♥6", "♦6", "♣6", "♠7", "♥7", "♦7", "♣7", "♠10", "♥10"],
            vec!["♦10", "♣10", "♠J", "♥J", "♦J", "♣J", "♠Q", "♥Q", "♦Q"],
        );
        let p = pctx_of(&state, Seat::E);
        let top = mk_top(Seat::N, vec!["♦4", "♣4", "♥4", "♠9", "♦9"]);
        let fh_cards = vec![
            "♠5".to_string(),
            "♥5".to_string(),
            "♦5".to_string(),
            "♠2".to_string(),
            "♦2".to_string(),
        ];
        let fh = CombinationParser::parse(&fh_cards, None, ctx()).unwrap();
        let s = score_follow(&fh_cards, &fh, &top, &p);
        assert!(s < -50000.0, "full house carrying a level pair must be banned: {s}");
    }

    // ══ 房规：剩 1 张强制打出（JS 488-497）══

    #[test]
    fn last_card_forced_play_when_beatable() {
        let state = mk_playing_state(Seat::E, vec!["♠9"], Some((Seat::N, vec!["♠3"])));
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        assert_eq!(
            picked,
            PlayerAction::Play {
                cards: vec!["♠9".into()],
                wild_targets: None,
            },
            "last card must be forced out when it beats the top"
        );
    }

    // ══ 队友硬禁压（JS decideAdvancedPlay 确定性项）══

    #[test]
    fn partner_big_pair_cannot_be_overridden() {
        // 队友领出大对 KK（rankValue 13 > 12）→ 绝对不能压，即使我有 AA。
        let state = mk_playing_state(
            Seat::E,
            vec!["♠A", "♥A", "♠5", "♠6"],
            Some((Seat::W, vec!["♠K", "♥K"])),
        );
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        assert_eq!(picked, PlayerAction::Pass, "never override teammate's big pair");
    }

    #[test]
    fn enemy_big_pair_can_be_taken() {
        // 敌家领出大对 KK → 可以用 AA 接（JS 无敌家硬禁压）。
        let mut state = mk_playing_state(
            Seat::E,
            vec!["♠A", "♥A", "♠5", "♠6"],
            Some((Seat::N, vec!["♠K", "♥K"])),
        );
        fill_seats(
            &mut state,
            vec!["♦K", "♣K", "♦A", "♣A", "♦5", "♣5", "♦6", "♣6", "♦7"],
            vec!["♠7", "♥7", "♦7", "♣7", "♠8", "♥8", "♦8", "♣8", "♠9"],
            vec!["♥9", "♦9", "♣9", "♠10", "♥10", "♦10", "♣10", "♠J", "♥J"],
        );
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        match &picked {
            PlayerAction::Play { cards, .. } => {
                assert_eq!(
                    cards.len(),
                    2,
                    "enemy big pair may be overtaken with AA, got {:?}",
                    picked
                );
            }
            other => panic!("Expected Play (pair AA), got {:?}", other),
        }
    }

    // ══ JS 硬守卫回归（改写为 JS 语义）══

    #[test]
    fn midgame_follow_small_pair_prefers_pass_over_bomb() {
        // 中盘（我11张，对手各9张无人冲刺）：对手领出小对44，
        // 我方唯一能压的是炸弹3333 → JS 守卫①（bombCount≤2 非残局非冲刺）→ Pass。
        let mut state = mk_playing_state(
            Seat::E,
            vec!["♠3", "♥3", "♦3", "♣3", "♠8", "♠9", "♠10", "♠J", "♠Q", "♠K", "♠A"],
            Some((Seat::N, vec!["♠4", "♥4"])),
        );
        fill_seats(
            &mut state,
            vec!["♦4", "♣4", "♦5", "♥5", "♦6", "♣6", "♦7", "♣7", "♦8"],
            vec!["♠5", "♥6", "♦7", "♣8", "♠9", "♥10", "♦J", "♣Q", "♦K"],
            vec!["♥7", "♦9", "♣10", "♠J", "♥Q", "♦K", "♣A", "♥8", "♠6"],
        );
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        assert_eq!(
            picked,
            PlayerAction::Pass,
            "mid-game must pass instead of bombing a small pair"
        );
    }

    #[test]
    fn opponent_sprinting_prefers_bomb_intercept() {
        // JS：绝不炸单张/对子，但对手冲刺（S 剩2张）时允许炸三张拦截。
        let mut state = mk_playing_state(
            Seat::E,
            vec!["♠3", "♥3", "♦3", "♣3", "♠5"],
            Some((Seat::N, vec!["♠4", "♥4", "♦4"])),
        );
        fill_seats(
            &mut state,
            vec!["♣4", "♦5", "♣5", "♥5", "♦6", "♣6", "♥6", "♦7", "♣7"],
            vec!["♠6", "♥7"], // S 冲刺（2 张）
            vec!["♥8", "♦9", "♣10", "♠J", "♥Q", "♦K", "♣A", "♥9", "♠8"],
        );
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        match &picked {
            PlayerAction::Play { cards, .. } => {
                assert_eq!(
                    cards.len(), 4,
                    "opponent sprinting exempts the bomb over a triple, got {:?}",
                    picked
                );
            }
            other => panic!("Expected Play (bomb intercept), got {:?}", other),
        }
    }

    #[test]
    fn last_bomb_used_when_opponent_sprinting() {
        // JS：中盘 + 仅 1 颗炸弹 + 对手冲刺（S 剩2）→ 守卫①/⑥ 豁免 → 可炸三张。
        let mut state = mk_playing_state(
            Seat::E,
            vec!["♠3", "♥3", "♦3", "♣3", "♠8", "♠9", "♠10", "♠J", "♠Q", "♠K", "♠A"],
            Some((Seat::N, vec!["♠4", "♥4", "♦4"])),
        );
        fill_seats(
            &mut state,
            vec!["♣4", "♦5", "♣5", "♥5", "♦6", "♣6", "♥6", "♦7", "♣7"],
            vec!["♠6", "♥7"], // S 冲刺
            vec!["♥8", "♦9", "♣10", "♥J", "♥Q", "♦K", "♣A", "♥9", "♣9"],
        );
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        match &picked {
            PlayerAction::Play { cards, wild_targets } => {
                let combo = CombinationParser::parse(cards, wild_targets.as_deref(), ctx())
                    .expect("intercept candidate must parse");
                assert!(
                    matches!(combo.class(), CombinationClass::Bomb),
                    "opponent sprinting exempts the last bomb for intercept, got {:?}",
                    picked
                );
            }
            _ => panic!("Expected Play (bomb intercept), got {:?}", picked),
        }
    }

    #[test]
    fn two_bombs_allow_using_one_vs_sprinting() {
        // JS：2 颗炸弹 + 对手冲刺（S 剩2）→ 可用一颗炸三张拦截。
        let mut state = mk_playing_state(
            Seat::E,
            vec!["♠3", "♥3", "♦3", "♣3", "♠5", "♥5", "♦5", "♣5", "♠8", "♠9", "♠10"],
            Some((Seat::N, vec!["♠4", "♥4", "♦4"])),
        );
        fill_seats(
            &mut state,
            vec!["♣4", "♦6", "♣6", "♥6", "♦7", "♣7", "♥7", "♦8", "♣8"],
            vec!["♠6", "♥9"], // S 冲刺
            vec!["♥8", "♦9", "♣10", "♥J", "♥Q", "♦K", "♣A", "♣9", "♦10"],
        );
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        match &picked {
            PlayerAction::Play { cards, wild_targets } => {
                let combo = CombinationParser::parse(cards, wild_targets.as_deref(), ctx())
                    .expect("intercept candidate must parse");
                assert!(
                    matches!(combo.class(), CombinationClass::Bomb),
                    "with 2 bombs one may be spent vs sprinting opponents, got {:?}",
                    picked
                );
            }
            _ => panic!("Expected Play (bomb intercept), got {:?}", picked),
        }
    }

    #[test]
    fn midgame_still_allows_bomb_on_big_top() {
        // JS 守卫①：中盘 bombCount≤2 一律保留；3 颗炸弹时对三带二可炸。
        let mut state = mk_playing_state(
            Seat::E,
            vec![
                "♠3", "♥3", "♦3", "♣3", "♠5", "♥5", "♦5", "♣5", "♠6", "♥6", "♦6", "♣6", "♠8",
            ],
            Some((Seat::N, vec!["♠K", "♥K", "♦K", "♠9", "♦9"])),
        );
        fill_seats(
            &mut state,
            vec!["♣4", "♥4", "♦4", "♠4", "♣7", "♥7", "♦7", "♠7", "♣8"],
            vec!["♠10", "♥10", "♦10", "♣10", "♠J", "♥J", "♦J", "♣J", "♠Q"],
            vec!["♣Q", "♥Q", "♦Q", "♠K", "♣K", "♦K", "♠A", "♥A", "♦A"],
        );
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        match &picked {
            PlayerAction::Play { cards, .. } => {
                assert_eq!(
                    cards.len(), 4,
                    "with 3 bombs a mid-game bomb over a full house stays allowed"
                );
            }
            _ => panic!("Expected Play (bomb on big top), got {:?}", picked),
        }
    }

    #[test]
    fn counter_bomb_blocked_by_post_play_recount_shared_wild() {
        // 用户房规：中盘反炸后剩余炸弹数（含潜在炸，重算）= 0 → 不出。
        // 手 [♠5,♥5,♦5,♥2,♠6,♠7,♠8,♠9]（8 张中盘）：账面 2 颗（百搭拼 555 炸 + ♠6789 拼同花顺），
        // 但两颗潜在炸共用 ♥2——反炸打掉任意一颗后实际剩 0 → 必须 Pass 保炸。
        let mut state = mk_playing_state(
            Seat::E,
            vec!["♠5", "♥5", "♦5", "♥2", "♠6", "♠7", "♠8", "♠9"],
            Some((Seat::N, vec!["♣4", "♦4", "♥4", "♠4"])),
        );
        fill_seats(
            &mut state,
            vec!["♣5", "♦5", "♣6", "♥6", "♦6", "♣7", "♥7", "♦7", "♣8"],
            vec!["♠10", "♥10", "♦10", "♣10", "♠J", "♥J", "♦J", "♣J", "♠Q"],
            vec!["♣Q", "♥Q", "♦Q", "♠K", "♣K", "♦K", "♠A", "♥A", "♦A"],
        );
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        assert_eq!(
            picked,
            PlayerAction::Pass,
            "反炸会把最后一颗（潜在）炸耗尽（共用百搭），中盘必须 Pass 保炸"
        );
    }

    #[test]
    fn two_bombs_counter_allowed_when_post_play_bombs_remain() {
        // 用户房规对照：两颗独立炸（天然 5555 + 百搭拼 666）中盘反炸 4 炸后仍剩 1 颗 → 允许出。
        let mut state = mk_playing_state(
            Seat::E,
            vec!["♠5", "♥5", "♦5", "♣5", "♠6", "♥6", "♦6", "♥2"],
            Some((Seat::N, vec!["♣4", "♦4", "♥4", "♠4"])),
        );
        fill_seats(
            &mut state,
            vec!["♣7", "♥7", "♦7", "♣8", "♥8", "♦8", "♣9", "♥9", "♦9"],
            vec!["♠10", "♥10", "♦10", "♣10", "♠J", "♥J", "♦J", "♣J", "♠Q"],
            vec!["♣Q", "♥Q", "♦Q", "♠K", "♣K", "♦K", "♠A", "♥A", "♦A"],
        );
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        match &picked {
            PlayerAction::Play { .. } => {}
            other => panic!("两颗独立炸中盘反炸（打完剩 1 颗）应允许，got {:?}", other),
        }
    }

    #[test]
    fn small_full_house_never_bombed_midgame() {
        // 用户房规 A（2026-08-30）：12 以下（<Q）的三带二中盘一律不炸。
        // 敌领 777+88，我 3 颗天然炸、无人冲刺 → Pass（旧 JS 语义为可炸，已按房规收紧）。
        let mut state = mk_playing_state(
            Seat::E,
            vec![
                "♠3", "♥3", "♦3", "♣3", "♠5", "♥5", "♦5", "♣5", "♠6", "♥6", "♦6", "♣6", "♠8",
            ],
            Some((Seat::N, vec!["♥7", "♠7", "♦7", "♣8", "♥8"])),
        );
        fill_seats(
            &mut state,
            vec!["♣4", "♥4", "♦4", "♠4", "♣7", "♠9", "♥9", "♦9", "♣9"],
            vec!["♠10", "♥10", "♦10", "♣10", "♠J", "♥J", "♦J", "♣J", "♠Q"],
            vec!["♣Q", "♦Q", "♥Q", "♠K", "♥K", "♦K", "♣K", "♠A", "♥A"],
        );
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        assert_eq!(
            picked,
            PlayerAction::Pass,
            "12 以下的三带二中盘不许用炸（房规 A），got {:?}",
            picked
        );
    }

    #[test]
    fn counter_bomb_prefers_wildfree_candidate() {
        // 用户房规 B1（2026-08-30）：反炸有免百搭候选时，不选含百搭的炸。
        // 敌领 4K；我有 4A（免百搭）与 5555+♥2（五张含百搭）都能压 → 必须选 4A。
        let mut state = mk_playing_state(
            Seat::E,
            vec![
                "♠A", "♥A", "♦A", "♣A", "♠5", "♥5", "♦5", "♣5", "♥2", "♠9", "♥9", "♦9",
            ],
            Some((Seat::N, vec!["♠K", "♥K", "♦K", "♣K"])),
        );
        fill_seats(
            &mut state,
            vec!["♣4", "♥4", "♦4", "♠4", "♣6", "♥6", "♦6", "♣6", "♠7"],
            vec!["♥7", "♦7", "♣7", "♠8", "♥8", "♦8", "♣8", "♠10", "♥10"],
            vec!["♣10", "♦10", "♠J", "♥J", "♦J", "♣J", "♠Q", "♥Q", "♦Q"],
        );
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        match &picked {
            PlayerAction::Play { cards, .. } => {
                let mut got: Vec<String> = cards.clone();
                got.sort();
                assert_eq!(
                    got,
                    vec!["♠A", "♣A", "♥A", "♦A"],
                    "反炸必须选免百搭的 4A，不许烧百搭（房规 B1），got {:?}",
                    picked
                );
            }
            other => panic!("Expected Play (4A counter), got {:?}", other),
        }
    }

    #[test]
    fn counter_bomb_wild_allowed_when_sprinting() {
        // 房规 B1 豁免：对手冲刺（W 剩 2 张）时反炸可烧百搭（+100 奖励恢复）。
        let mut state = mk_playing_state(
            Seat::E,
            vec![
                "♠A", "♥A", "♦A", "♣A", "♠5", "♥5", "♦5", "♣5", "♥2", "♠9", "♥9", "♦9",
            ],
            Some((Seat::N, vec!["♠K", "♥K", "♦K", "♣K"])),
        );
        fill_seats(
            &mut state,
            vec!["♣4", "♥4", "♦4", "♠4", "♣6", "♥6", "♦6", "♣6", "♠7"],
            vec!["♥7", "♦7"],
            vec!["♣10", "♦10", "♠J", "♥J", "♦J", "♣J", "♠Q", "♥Q", "♦Q"],
        );
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        match &picked {
            PlayerAction::Play { cards, .. } => {
                assert_eq!(
                    cards.len(),
                    5,
                    "冲刺豁免：含百搭的五张炸反炸应被允许，got {:?}",
                    picked
                );
            }
            other => panic!("Expected Play (wild bomb counter under sprint), got {:?}", other),
        }
    }

    #[test]
    fn counter_bomb_only_wild_option_still_plays_when_bombs_remain() {
        // 房规 B1 边界（用户未批 B2）：唯一能反炸的候选含百搭时，只要 ①b 重算后仍剩炸
        // （这里手里还有 3333），照常出炸——不新增"含百搭一律不反"的 Pass 规则。
        let mut state = mk_playing_state(
            Seat::E,
            vec!["♠5", "♥5", "♦5", "♣5", "♥2", "♠3", "♥3", "♦3", "♣3", "♠9"],
            Some((Seat::N, vec!["♠K", "♥K", "♦K", "♣K"])),
        );
        fill_seats(
            &mut state,
            vec!["♣4", "♥4", "♦4", "♠4", "♣6", "♥6", "♦6", "♣6", "♠7"],
            vec!["♥7", "♦7", "♣7", "♠8", "♥8", "♦8", "♣8", "♠10", "♥10"],
            vec!["♣10", "♦10", "♠J", "♥J", "♦J", "♣J", "♠Q", "♥Q", "♦Q"],
        );
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        match &picked {
            PlayerAction::Play { cards, wild_targets } => {
                assert!(
                    cards.len() == 5,
                    "唯一反炸手段（含百搭五张炸）在打完仍有炸时应照常打出，got {:?}",
                    picked
                );
                assert!(
                    wild_targets.is_some() || cards.iter().any(|c| c == "♥2"),
                    "应打出含百搭的炸，got {:?}",
                    picked
                );
            }
            other => panic!("Expected Play (only wild counter), got {:?}", other),
        }
    }

    #[test]
    fn endgame_upgraded_wild_bomb_beats_natural_bomb() {
        // JS 语义：残局 +100 百搭进炸弹奖励压过 −10 升档轻罚 → 5555+百搭(5炸)
        // 胜过纯天然 5555（对手三张顶、无更优替牌）。
        let mut state = mk_playing_state(
            Seat::E,
            vec!["♠5", "♥5", "♦5", "♣5", "♥2", "♠K"],
            Some((Seat::N, vec!["♠9", "♥9", "♦9"])),
        );
        fill_seats(
            &mut state,
            vec!["♣9", "♦9", "♣4", "♦4", "♠6", "♥6", "♦6", "♣6", "♠7"],
            vec!["♥7", "♦7", "♣7", "♠8", "♥8", "♦8", "♣8", "♠10", "♥10"],
            vec!["♦10", "♣10", "♠J", "♥J", "♦J", "♣J", "♠Q", "♥Q", "♦Q"],
        );
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        match &picked {
            PlayerAction::Play { cards, wild_targets } => {
                let combo = CombinationParser::parse(cards, wild_targets.as_deref(), ctx())
                    .expect("candidate must parse");
                let uses_wild = wild_targets.as_deref().map_or(false, |t| !t.is_empty());
                assert!(
                    matches!(combo.class(), CombinationClass::Bomb)
                        && cards.len() == 5
                        && uses_wild,
                    "endgame must prefer the upgraded wild bomb (wild bonus +100 vs −10), got {:?}",
                    picked
                );
            }
            other => panic!("Expected Play (upgraded wild bomb), got {:?}", other),
        }
    }

    #[test]
    fn endgame_follow_single_removal_reward_beats_wild_bomb() {
        // JS 语义：残局 +400/+300 单张移除奖励 → 百搭三带二 555+8(百搭)
        //（移除单张8）胜过百搭炸弹 555+百搭（不移除单张）。
        let mut state = mk_playing_state(
            Seat::E,
            vec!["♠5", "♥5", "♦5", "♠8", "♠K", "♥2"],
            Some((Seat::N, vec!["♦4", "♣4", "♥4", "♠9", "♦9"])),
        );
        fill_seats(
            &mut state,
            vec!["♠3", "♥3", "♦3", "♣3", "♠4", "♥4", "♠6", "♥6", "♦6"],
            vec!["♣6", "♠7", "♥7", "♦7", "♣7", "♠10", "♥10", "♦10", "♣10"],
            vec!["♠J", "♥J", "♦J", "♣J", "♠Q", "♥Q", "♦Q", "♣Q", "♠A"],
        );
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        match &picked {
            PlayerAction::Play { cards, wild_targets } => {
                let combo = CombinationParser::parse(cards, wild_targets.as_deref(), ctx())
                    .expect("candidate must parse");
                assert!(
                    matches!(combo.kind, CombinationKind::Ordinary(OrdinaryKind::FullHouse)),
                    "endgame single-removal reward (+400/+300) must favor the full house, got {:?}",
                    picked
                );
            }
            other => panic!("Expected Play (full house), got {:?}", other),
        }
    }

    #[test]
    fn endgame_follow_prefers_small_single_removal_over_level_card() {
        // JS 语义：残局跟单6，手 [♠2(级牌),♠7]：♠7 移除单张 +700 胜过级牌单张。
        let state = mk_playing_state(Seat::E, vec!["♠2", "♠7"], Some((Seat::N, vec!["♠6"])));
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
    fn leading_level_single_banned_when_hand_large() {
        // JS：手牌 > 6 张时空出级牌单张 −99999。手 [♥2,♠2,♦2,♣2,♠3..♠6] 领出
        // → 必须出小单张 ♠3，绝不能领 2 的单张。
        let state = mk_playing_state(
            Seat::E,
            vec!["♥2", "♠2", "♦2", "♣2", "♠3", "♠4", "♠5", "♠6"],
            None,
        );
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        match &picked {
            PlayerAction::Play { cards, wild_targets } => {
                let combo = CombinationParser::parse(cards, wild_targets.as_deref(), ctx())
                    .expect("candidate must parse");
                let is_level_single = matches!(
                    combo.kind,
                    CombinationKind::Ordinary(OrdinaryKind::Single)
                ) && cards.iter().any(|c| c.ends_with('2'));
                assert!(
                    !is_level_single,
                    "must not lead a level-card single with a large hand, got {:?}",
                    picked
                );
            }
            other => panic!("Expected Play, got {:?}", other),
        }
    }
}
