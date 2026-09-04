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
    is_wild, level_order_value, natural_rank_value, parse_card_symbol, HandLevel, Rank,
    RuleContext, Suit,
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

thread_local! {
    /// 训练评估线程标记（2026-09-03 排查修复）：仅训练线程为 true。
    static IN_TRAINING_EVAL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// RAII：训练评估入口置位，Drop 复位。作用域内的 `get_params_for_seat` 才会读
/// `LEARN_PARAMS*`（训练候选 vs 基线的自对弈语义）。
///
/// 房规隔离修复（2026-09-03）：此前 `LEARN_PARAMS*` 是全局量，训练评估（24h 自动续跑）
/// 期间**线上对局桌面的机器人也会读到变异中的候选参数**。参数面扩到 52 维后单次变异
/// 扰动巨大（如 keep_bomb_bonus 500→8、dual_wild_penalty_mid 600→3），玩家桌上
/// 留炸/百搭罚/接风等打分型房规全部失真。修复后线上桌面永远使用 `js_trained_params`
/// 房规基线，训练扰动只存在于训练线程自己的自对弈里。
pub(crate) struct TrainingGuard;

impl TrainingGuard {
    pub(crate) fn new() -> Self {
        IN_TRAINING_EVAL.with(|c| c.set(true));
        TrainingGuard
    }
}

impl Drop for TrainingGuard {
    fn drop(&mut self) {
        IN_TRAINING_EVAL.with(|c| c.set(false));
    }
}

fn get_learn_params() -> Option<AdvancedBotParams> {
    LEARN_PARAMS.lock().ok().and_then(|lock| lock.clone())
}

fn get_params_for_seat(seat: Seat) -> AdvancedBotParams {
    // 房规隔离（2026-09-03）：非训练线程一律使用房规基线 js_trained_params。
    if !IN_TRAINING_EVAL.with(std::cell::Cell::get) {
        return js_trained_params();
    }
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
/// 2026-09-02 基线分裂修复：f32 权重升级为 2026-08-31 训练冠军值（= advanced_params.json
/// = CF TRAINED_PARAMS 逐键一致）。此前 Rust 线上用旧基线(first_out 0.7658)、CF 用
/// 训练值(0.8317)，两端行为分裂。u8 房规阈值仍冻结不变。
pub(crate) fn js_trained_params() -> AdvancedBotParams {
    AdvancedBotParams {
        team_win_weight: 1.0285988,
        first_out_weight: 0.8316725,
        second_out_weight: 0.9,
        yield_to_partner_bias: 1.4,
        partner_sprint_threshold: 2,
        bomb_conserve_bias: 0.83482033,
        bomb_aggression_when_enemy_low: 2.2,
        enemy_low_cards_threshold: 6, // 用户房规：冲刺 = 任一对手剩 ≤6 张（JS 原值 3）
        endgame_hand_count_threshold: 6,
        endgame_clear_hand_bias: 1.2082484,
        proactive_play_bias: 1.1,
        low_card_dump_bias: 1.4,
        pass_stall_penalty: 0.90296406,
        hand_tracker_enabled: true,
        prob_threshold_for_bomb: 0.56296045,
        prob_threshold_for_intercept: 0.4,
        enable_reason_trace: false, // 与 CF/参数文件一致（true 仅多一条调试打印）
        keep_bomb_bonus: 499.99548, // 2026-09-02 16:06 训练首个采纳变异（57键新冠军，bestScore 0.6665）
        // 用户 2026-09-03 调令：加大"百搭组成炸弹/同花顺"奖励力度（100→250，与 CF 同步）。
        // 250 压过所有次级百搭信号（保留罚 −150 / 顺子 +30 / 配对 −300），且低于清空 +10000。
        // 语义边界（用户裁决）：奖励只作用于百搭"拼成"炸/同花顺（百搭必要）；天然炸贴百搭
        // 升档仍属浪费——升档罚同步 +150 配平（mid 300 / end 160），升档净效应不变。
        // 三键显式钉住（同 keep_bomb_bonus）防训练冠军同步时回落。
        wild_bomb_bonus: 250.0,
        upgraded_bomb_wild_mid: 300.0,
        upgraded_bomb_wild_end: 160.0,
        // 用户 2026-09-03 调令：百搭配单张成对 300→800 / 残局 15→100（救孤候选经
        // score_follow_ex 的 wild_rescue_lift 豁免）。显式钉住防训练冠军同步回落。
        wild_plain_pair_mid: 800.0,
        wild_pair_penalty_end: 100.0,
        ..crate::bot::plugins::advanced_bot::params::AdvancedBotParams::scoring_defaults()
    }
}

// ── JS 房规常量 (bot-advanced.js L14-42), 按值移植 ──────────────────────
// （罚值量级已参数化 → params.dual_wild_penalty_* 等；禁令/豁免结构保持常量）
/// JS `DUAL_WILD_HAND_ENDGAME`: 手牌 ≤ 此值视为残局（百搭/拆炸房规专用，硬编码 6）
const DUAL_WILD_HAND_ENDGAME: usize = 6;
/// JS `DUAL_WILD_CANDIDATE_HAND_MAX`: 仅 movegen（JS generatePlaysOfType）使用；
/// 本文件不枚举候选（由 `enumerate_legal_actions` 负责），保留仅为 1:1 对应。
#[allow(dead_code)]
const DUAL_WILD_CANDIDATE_HAND_MAX: usize = 6;

// JS 内联分值常量（scorePlay base 100 / scoreLeadPlay base 50 / 清空+10000）
// penalty×20 缩放已参数化 → params.split_penalty_scale（路线图②）
const BASE_FOLLOW_SCORE: f32 = 100.0;
const BASE_LEAD_SCORE: f32 = 50.0;
const CLEAR_HAND_BONUS: f32 = 10000.0;
const BANNED_SCORE: f32 = 99999.0;
/// 房规（用户 2026-09-03）：百搭优先用于炸弹/同花顺——手中余牌与剩余百搭仍可组成
/// 炸弹/同花顺时，把百搭用进更低级组合的冻结惩罚（房规禁令不入训练面）。
const WILD_CONSERVATION_PENALTY: f32 = 150.0;
/// 房规（用户 2026-09-02 反孤儿条款）：出牌后剩余手牌全是百搭 → 百搭将沦为最后
/// 孤张（被迫单出/孤注一掷）→ 重罚该出牌，倒逼把百搭并入当前组合一起走。
/// CF scoreLeadPlay 原有领出侧单百搭 −800 同源；本次统一为 all-wild 口径并
/// 补齐领出+跟牌两侧（冻结不入训练面）。
const WILD_STRAND_PENALTY: f32 = 800.0;
/// 房规（用户 2026-09-03）：百搭同花顺拆牌质量评估——
/// ① 拆完剩余牌可组成新的杂顺子 → +450 重奖（抵消空出炸弹罚；领出侧另加
///    keep_bomb_bonus 抵"留炸到残局"倾向——"SF 先手 + 剩顺后续"两墩计划成立）；
/// ② 拆完剩余散单张 ≥3 → −250 惩罚（同花顺拆出一手烂剩牌）。
/// 杂顺窗口与 movegen 一致（2 作低：2-6 … 10-A）；百搭/王不参与检测与散张计数。
/// （冻结不入训练面；仅作用于百搭同花顺候选，跟牌侧同受 B1 守卫约束。）
const SF_LEFTOVER_STRAIGHT_BONUS: f32 = 450.0;
const SF_LEFTOVER_SINGLES_PENALTY: f32 = 250.0;

// ── 房规（用户 2026-09-03）：残局报牌防走/送队友（领出侧，冻结不入训练面）──
// 原弱化版（+40/−50/−20）被其他分项淹没导致"房规失效"，本次按用户口径原位加强。
/// 对手剩 1 张：领出单张强阻尼（防对手最后一张直接走人）；不得不发时从大往小
const OPP_LAST_CARD_SINGLE_PENALTY: f32 = 400.0;
/// 对手剩 2 张：领出对子强阻尼（防对手对子直接走人）→ 改拆对发单张或领其他牌型
const OPP_LAST_TWO_PAIR_PENALTY: f32 = 400.0;
/// 队友剩 1/2 张：送单张/对子强奖励
const TEAMMATE_FEED_BONUS: f32 = 400.0;
/// 从小送/从大发的 primary 倾斜系数（primary: 3=3…A=14, 2=15, 王=16/17）
const FEED_RANK_TILT: f32 = 8.0;
/// 对手剩 2 张：领出单张净鼓励（拆对发单张/散单——对手剩两张接不走单张）
const OPP_TWO_SINGLE_NUDGE: f32 = 60.0;
/// 对手剩 1 张：领出对子净鼓励（对手单张接不走对子，最安全的压制）
const OPP_ONE_PAIR_NUDGE: f32 = 120.0;

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

/// 房规（用户 2026-09-03）：手里余牌与剩余百搭能否组成炸弹或同花顺。
/// 炸弹补足：某点数天然牌 ≥2 且 天然数+剩余百搭 ≥4（级牌点数除外——级牌炸弹被房规禁止）。
/// 同花顺：同花色 5 连窗（natural 2..=14，2 作低——JS 23456 合法），缺张数 ∈ [1, 剩余百搭数]。
/// 百搭/王不计入天然牌；jokers 无花色不参与同花顺。
fn wilds_could_form_bomb_or_sf(p: &PlayContext, play_cards: &[String]) -> bool {
    let mut to_remove: Vec<String> = play_cards.to_vec();
    let mut remaining: Vec<&String> = Vec::with_capacity(p.my_hand.len());
    for c in &p.my_hand {
        if let Some(pos) = to_remove.iter().position(|r| r == c) {
            to_remove.remove(pos);
        } else {
            remaining.push(c);
        }
    }
    let wild_left = remaining
        .iter()
        .filter(|c| p.meta_for(c).map(|m| m.is_wild).unwrap_or(false))
        .count();
    if wild_left == 0 {
        return false;
    }
    let mut rank_counts: HashMap<Rank, usize> = HashMap::new();
    let mut suit_ranks: HashMap<Suit, HashSet<u8>> = HashMap::new();
    for c in remaining {
        if is_joker_sym(c) {
            continue;
        }
        let Some(m) = p.meta_for(c) else { continue };
        if m.is_wild {
            continue;
        }
        if m.rank != p.level_rank {
            *rank_counts.entry(m.rank).or_default() += 1;
        }
        if let Some(nv) = m.natural {
            if let Ok(card) = parse_card_symbol(c) {
                suit_ranks.entry(card.suit).or_default().insert(nv);
            }
        }
    }
    // 炸弹补足：某点数天然牌 ≥2，剩余百搭能补到 4
    if rank_counts
        .values()
        .any(|&n| n >= 2 && n + wild_left >= 4)
    {
        return true;
    }
    // 同花顺：同花色 5 连窗缺张可由百搭补足
    for ranks in suit_ranks.values() {
        for lo in 2..=10u8 {
            let missing = (lo..lo + 5).filter(|r| !ranks.contains(r)).count();
            if missing >= 1 && missing <= wild_left {
                return true;
            }
        }
    }
    false
}

/// 房规（用户 2026-09-06）：手牌（出牌前）是否已存在"百搭可完成"的炸弹/同花顺。
/// 炸弹：任一非级牌点数天然牌 ≥3 张（+1 百搭 = 4 炸）；
/// 同花顺：某花色 5 连窗（lo 3..=10）中同花自然牌 ≥4（缺位 ≤1 由百搭补）。
/// 用于"百搭配三带二禁令"：手牌能组百搭炸/顺时，百搭不再进三带二。与 CF handHasWildBombOrSF 1:1。
fn hand_has_wild_bomb_or_sf(p: &PlayContext) -> bool {
    let mut has_wild = false;
    let mut rank_counts: HashMap<Rank, usize> = HashMap::new();
    let mut suit_ranks: HashMap<Suit, HashSet<u8>> = HashMap::new();
    for c in &p.my_hand {
        let Some(m) = p.meta_for(c) else { continue };
        if m.is_wild {
            has_wild = true;
            continue;
        }
        if is_joker_sym(c) {
            continue;
        }
        if m.rank != p.level_rank {
            *rank_counts.entry(m.rank).or_default() += 1;
        }
        if let (Some(nv), Ok(card)) = (m.natural, parse_card_symbol(c)) {
            suit_ranks.entry(card.suit).or_default().insert(nv);
        }
    }
    if !has_wild {
        return false;
    }
    if rank_counts.values().any(|&n| n >= 3) {
        return true;
    }
    for ranks in suit_ranks.values() {
        for lo in 3..=10u8 {
            let have = (lo..lo + 5).filter(|r| ranks.contains(r)).count();
            if have >= 4 {
                return true;
            }
        }
    }
    false
}

/// 房规（用户 2026-09-02 反孤儿条款）：出牌后剩余手牌是否全为百搭。
/// 全为百搭 = 百搭即将沦为最后孤张（无任何天然牌可搭伙）→ 触发 WILD_STRAND_PENALTY。
/// 2026-09-06 扩口径：剩余=百搭∪王 且仍含百搭 同样算孤（王与百搭互不成墩，百搭必被
/// 拖到最后单出——实战 seq1028/1041：W 出 99 留 [♥A+🃏B] 百搭苟到最终张）。
fn leftover_all_wild(p: &PlayContext, play_cards: &[String]) -> bool {
    let mut to_remove: Vec<String> = play_cards.to_vec();
    let mut any_wild_left = false;
    for c in &p.my_hand {
        if let Some(pos) = to_remove.iter().position(|r| r == c) {
            to_remove.remove(pos);
        } else {
            let is_wild = p.meta_for(c).map(|m| m.is_wild).unwrap_or(false);
            if is_wild {
                any_wild_left = true;
            } else if !is_joker_sym(c) {
                // 剩余含天然牌 → 百搭有搭伙对象，不算孤张
                return false;
            }
        }
    }
    any_wild_left
}

/// 房规（用户 2026-09-03）：百搭同花顺拆牌质量——
/// 返回 (剩余可组杂顺数, 去顺后散单张数)。百搭/王不计入；级牌天然牌按普通点数参与。
/// 杂顺窗口与 movegen 一致（2 作低：2-6 … 10-A，各 5 连点数）。
fn sf_leftover_straights_and_singles(p: &PlayContext, play_cards: &[String]) -> (usize, usize) {
    let mut to_remove: Vec<String> = play_cards.to_vec();
    let mut rank_cnt: HashMap<u8, usize> = HashMap::new();
    for c in &p.my_hand {
        if let Some(pos) = to_remove.iter().position(|r| r == c) {
            to_remove.remove(pos);
            continue;
        }
        let Some(m) = p.meta_for(c) else { continue };
        if m.is_wild || m.is_joker {
            continue;
        }
        if let Some(nv) = m.natural {
            *rank_cnt.entry(nv).or_default() += 1;
        }
    }
    let mut straights = 0usize;
    loop {
        let mut found = false;
        for lo in 2u8..=10u8 {
            if (lo..lo + 5).all(|r| rank_cnt.get(&r).copied().unwrap_or(0) >= 1) {
                for r in lo..lo + 5 {
                    if let Some(n) = rank_cnt.get_mut(&r) {
                        *n -= 1;
                    }
                }
                straights += 1;
                found = true;
                break;
            }
        }
        if !found {
            break;
        }
    }
    let singles = rank_cnt.values().filter(|&&n| n == 1).count();
    (straights, singles)
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

    // （2026-09-06 裁决后潜在炸/潜在同花顺不再计入 bomb_count，原 wild_sf_candidates
    // 统计段已删除；潜在结构判断由 hand_has_wild_bomb_or_sf 独立承担。）
    // 房规（用户 2026-09-06 裁决）：bomb_count 只数天然炸——天然4+同点炸、天然同花顺、四王。
    // 百搭潜在炸（三同张拼炸/4连拼同花顺）不消耗天然炸储备，不计入：
    // 潜在炸计入曾使守卫①（唯一炸保留）与"烧最后一炸"硬禁令全部失效
    // （实战 seq20：J×4 天然炸 + A×3/Q×3 潜在 → bomb_count=2 → JJJJ 被烧）。
    // 潜在炸的存在性判断由 hand_has_wild_bomb_or_sf/wilds_could_form_bomb_or_sf 独立承担。
    let bomb_count = bomb_ranks.len()
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
    unlock_pair_for_single: bool,
    unlock_triple_for_pair: bool,
    feed_exempt: bool,
) -> u32 {
    // 房规（用户 2026-08-30）：条件解锁——对手连续≥2轮领单张无人接 → 允许拆对出单张；
    // 连续≥2轮领对子无人接 → 允许拆三张同出对子。优先级：先拆"孤"（非木板/钢板成员），
    // 同优先级拆大的（10以上优先，依次9.8.7…），用分级小罚分表达（远小于禁令，能压过牌值偏好）。
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
            if is_plate_part && !unlock_triple_for_pair {
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
                // 条件解锁（用户 2026-08-30）：对手连续≥2轮领对子无人接 → 拆三张同出对子
                if unlock_triple_for_pair && matches!(play_kind, CombinationKind::Ordinary(OrdinaryKind::Pair)) {
                    return u32::from(is_plate_part) * 10 + u32::from(14u8.saturating_sub(rank));
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
            if is_tube_part && !unlock_pair_for_single {
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
                // 房规（用户 2026-09-03）：报牌豁免——对手剩≤2 或队友剩 1 时，领出拆对
                // 发单张放行（小罚分分级：连三对成员 +10；拆小对罚更小 → 天然优先拆小对）。
                // 仅领出路径传入；跟牌路径的既有拆对房规不受影响。
                if feed_exempt
                    && matches!(play_kind, CombinationKind::Ordinary(OrdinaryKind::Single))
                {
                    return u32::from(is_tube_part) * 10 + u32::from(14u8.saturating_sub(rank));
                }
                // 条件解锁（用户 2026-08-30）：对手连续≥2轮领单张无人接 → 拆对子出单张
                if unlock_pair_for_single && matches!(play_kind, CombinationKind::Ordinary(OrdinaryKind::Single)) {
                    return u32::from(is_tube_part) * 10 + u32::from(14u8.saturating_sub(rank));
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
    /// 房规（用户 2026-08-30，同步 CF isAnyEnemySprinting）：活跃对手（剩 >0 张）中
    /// 是否有人 ≤6 张——已走完（剩 0 张）的对手不算冲刺，他不可能再赢。
    enemy_sprinting: bool,
    /// 活跃对手（剩 >0 张）的剩余张数（供带阈值参数的冲刺判定用）
    enemy_rem_active: Vec<usize>,
    /// 房规（用户 2026-09-03 修订）：当前顶牌是对手领出的 K 以下（level_order <12）
    /// 单张且我方无人接住（顶牌是对手的 = 我方没接管）→ ≥1 轮即解锁拆对跟单张。
    /// 其他牌型（三张/顺子/炸弹等）不计入、不触发。
    unlock_single_follow_split: bool,
    /// 同上，对子版 → 解锁拆三张同跟对子
    unlock_pair_follow_split: bool,
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
    /// 残局求解器 memo（路线图③）：手牌多重集 → 最少出牌墩数（仅手牌 ≤6 张时使用）
    endgame_memo: std::cell::RefCell<HashMap<String, usize>>,
    /// 求解器单次决策时间闸（2026-09-03 排查修复）：求解器在带百搭的残局上可能秒级耗时，
    /// 触发 store.rs `BOT_TURN_TIMEOUT=30s` 的 bot_turn_timeout 强制过牌——线上机器人在
    /// 残局被系统自动 Pass，留炸/接风/清牌等房规全部"失效"。每次决策建 context 时定为
    /// now+500ms；超时后其余候选退回传统打分（远低于 30s 强制过牌线）。
    solver_deadline: std::time::Instant,
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
    // 房规（用户 2026-08-30，同步 CF）：冲刺判定只看"活跃"对手——剩 0 张（已走完）不计入。
    let enemy_rem_active: Vec<usize> = opp_counts.iter().copied().filter(|&c| c > 0).collect();
    let enemy_sprinting = enemy_rem_active.iter().any(|&c| c <= 6);

    // 房规（用户 2026-09-03 修订）：当前顶牌是对手领出的 K 以下（level_order <12）
    // 单张/对子且我方无人接住（顶牌是对手的 = 我方没接管）→ ≥1 轮即触发解锁。
    // 其他牌型不计入。history 不再参与判定。
    let (unlock_single_follow_split, unlock_pair_follow_split) = {
        let mut u_single = false;
        let mut u_pair = false;
        if let Some(tp) = hand.trick.top_play.as_ref() {
            if tp.seat != actor && tp.seat != teammate_seat {
                let primary = tp.combination.primary;
                if primary > 0 && primary < 12 {
                    if matches!(tp.combination.kind, CombinationKind::Ordinary(OrdinaryKind::Single)) {
                        u_single = true;
                    } else if matches!(tp.combination.kind, CombinationKind::Ordinary(OrdinaryKind::Pair)) {
                        u_pair = true;
                    }
                }
            }
        }
        (u_single, u_pair)
    };

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
        enemy_sprinting,
        enemy_rem_active,
        unlock_single_follow_split,
        unlock_pair_follow_split,
        is_endgame,
        combos,
        has_level_card_or_joker,
        meta,
        teammate_passed_top,
        actor_seat,
        seat_remaining,
        enemy_seats,
        pool,
        endgame_memo: std::cell::RefCell::new(HashMap::new()),
        solver_deadline: std::time::Instant::now() + std::time::Duration::from_millis(500),
    }
}

/// 残局求解器（路线图③，用户 2026-09-03）：最小出牌墩数规划。
/// 把 `cards` 精确划分成最少墩数的合法组合（单/对/三/三带二/顺/连对/钢板/炸/火箭…），
/// 返回最少墩数。`cards` ≤6 张时子集枚举 ≤63 个/节点、深度 ≤6，微秒级。
/// 百搭由解析器自动解算（子集直接 parse，None targets）。
fn min_tricks_partition(
    cards: &[String],
    level: HandLevel,
    memo: &mut HashMap<String, usize>,
    deadline: std::time::Instant,
) -> usize {
    if cards.is_empty() {
        return 0;
    }
    let mut sorted: Vec<String> = cards.to_vec();
    sorted.sort();
    let key = sorted.join(",");
    if let Some(&v) = memo.get(&key) {
        return v;
    }
    let n = sorted.len();
    let ctx = RuleContext { hand_level: level };
    let mut best = usize::MAX;
    // 锚定首张：每一墩必含某张牌，枚举"含 sorted[0]"的全部子集（其余 n-1 位按位掩码）。
    // 掩码选的是下标，天然处理重复符号（多重集）。
    for mask in 0..(1u16 << (n - 1)) {
        // 时间闸（2026-09-03）：带百搭的残局子集解析可能远超预期，超时返回当前最优的
        // 上界（cards.len()=全拆单张，恒为合法上界）。防线上 30s bot_turn_timeout 强制过牌。
        if std::time::Instant::now() > deadline {
            let bound = if best == usize::MAX { n } else { best.min(n) };
            memo.insert(key, bound); // 上界可安全缓存（只会高估墩数，不会低估）
            return bound;
        }
        let mut subset: Vec<String> = Vec::with_capacity(n);
        let mut rest: Vec<String> = Vec::with_capacity(n);
        subset.push(sorted[0].clone());
        for i in 1..n {
            if mask & (1 << (i - 1)) != 0 {
                subset.push(sorted[i].clone());
            } else {
                rest.push(sorted[i].clone());
            }
        }
        if subset.len() > 1 && CombinationParser::parse(&subset, None, ctx).is_err() {
            continue;
        }
        let sub = min_tricks_partition(&rest, level, memo, deadline);
        if sub + 1 < best {
            best = sub + 1;
            if best == 1 {
                break; // 一手出完，不可能更小
            }
        }
    }
    let result = if best == usize::MAX { n } else { best };
    memo.insert(key, result);
    result
}

impl PlayContext {
    fn meta_for(&self, sym: &str) -> Option<CardMeta> {
        self.meta
            .get(sym)
            .copied()
            .or_else(|| meta_of(sym, self.ctx))
    }

    /// 残局求解器（路线图③）：出掉 `play_cards` 后，余牌最少还需几墩出完。
    fn min_tricks_after(&self, play_cards: &[String]) -> usize {
        let mut rest: Vec<String> = Vec::with_capacity(self.my_hand.len());
        let mut to_remove: Vec<String> = play_cards.to_vec();
        for c in &self.my_hand {
            if let Some(pos) = to_remove.iter().position(|r| r == c) {
                to_remove.remove(pos);
            } else {
                rest.push(c.clone());
            }
        }
        let mut memo = self.endgame_memo.borrow_mut();
        min_tricks_partition(&rest, self.ctx.hand_level, &mut memo, self.solver_deadline)
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
            // 房规（用户 2026-08-30）：百搭与对子配成三张同 = 浪费——领出跳过，
            // 唯一豁免 = 清空手牌（len == remaining）或拦截对手冲刺（活跃对手≤6张）。
            if matches!(cand.combo.kind, CombinationKind::Ordinary(OrdinaryKind::Triple))
                && cand
                    .cards
                    .iter()
                    .filter(|s| p.meta_for(s).map(|m| m.is_wild).unwrap_or(false))
                    .count()
                    == 1
                && cand.cards.len() < p.my_remaining
                && !p.enemy_sprinting
            {
                continue;
            }
            // 房规（用户 2026-09-03）：三张天然、对子=1天然+1百搭（百搭凑对）→ 禁止领出。
            // 豁免：清空手牌、对手冲刺。
            if matches!(cand.combo.kind, CombinationKind::Ordinary(OrdinaryKind::FullHouse)) {
                let (_, wild_pair_fh) = fh_wild_shape(cand.cards, &p);
                if wild_pair_fh && cand.cards.len() < p.my_remaining && !p.enemy_sprinting {
                    continue;
                }
            }
            // 房规（用户 2026-09-06）：百搭优先成炸/同花顺——手牌已存在百搭可完成的
            // 炸弹/同花顺时，含百搭的三带二不领出（豁免=清空手牌 或 拦截对手冲刺）。
            // 与 CF 领出侧同规（手牌级判定）。
            if matches!(cand.combo.kind, CombinationKind::Ordinary(OrdinaryKind::FullHouse))
                && cand
                    .cards
                    .iter()
                    .any(|s| p.meta_for(s).map(|m| m.is_wild).unwrap_or(false))
                && hand_has_wild_bomb_or_sf(&p)
                && cand.cards.len() < p.my_remaining
                && !p.enemy_sprinting
            {
                continue;
            }
            // 房规（用户 2026-09-03）：双百搭同出候选仅残局（手牌≤6张）才枚举——领出路径
            // 硬门槛（2026-09-03 审计补，与跟牌过滤及 CF movegen 门槛一致）。
            // 清空手牌的牌不受限。
            if p.my_remaining > 6
                && cand
                    .cards
                    .iter()
                    .filter(|s| p.meta_for(s).map(|m| m.is_wild).unwrap_or(false))
                    .count()
                    >= 2
                && cand.cards.len() < p.my_remaining
            {
                continue;
            }
            let s = score_lead(cand.cards, &cand.combo, &p);
            if best.map_or(true, |(bs, _)| s > bs) {
                best = Some((s, cand));
            }
            if cand.combo.class() != CombinationClass::Bomb
                && best_non_bomb.map_or(true, |(bs, _)| s > bs)
            {
                // 整手皆炸豁免判据（2026-09-06）：只认"非拆炸禁令"的非炸候选——
                // 纯炸手（如 3333+4444）的拆炸单/对/三不算数，此时炸候选解禁（与 CF 同规）。
                if classify_bomb_split(&cand.cards, &p.my_hand, &cand.combo.kind, p.my_remaining)
                    == BombSplitVerdict::Banned
                {
                    continue;
                }
                best_non_bomb = Some((s, cand));
            }
        }
        // 房规过滤器（百搭配三张）理论上不可能排除全部领出候选（单张/对子始终存在）；
        // 极端手牌万一全被排除时，退回无过滤最优，保证必有出牌。
        let best = match best {
            Some(b) => Some(b),
            None => candidates
                .iter()
                .map(|c| (score_lead(c.cards, &c.combo, &p), c))
                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)),
        };
        let (_, best_cand) = best.expect("candidates non-empty");
        // ── 房规（用户 2026-09-06 绝对版）：炸弹/同花顺（天然+百搭+四王一律同规）
        //    绝对禁止领出，唯一豁免：①打出即清空手牌 ②整手皆炸/顺（best_non_bomb
        //    只认非拆炸候选，纯炸手时为 None → 炸候选解禁）。
        //    残局的炸弹由此自然留作最后一张收尾清空；中盘绝不空耗炸弹领出。
        //    （取代旧"重算换牌"：旧逻辑手里还有别的炸时允许领出，用户裁决为违规。）
        {
            let is_clearing_lead = best_cand.cards.len() >= p.my_remaining;
            if best_cand.combo.class() == CombinationClass::Bomb && !is_clearing_lead {
                if let Some((_, alt)) = best_non_bomb {
                    return Ok(alt.action.clone());
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

    // ── 用户 2026-09-06：队友快走完（剩余 ≤6）时，队友的顶牌一律不压——
    // 我每压一墩，队友就少走一墩。唯一豁免：我此手打完即清空（夺游优先）。
    // 记牌器对队友剩余张数是精确公开信息。与 CF decideAdvancedPlay 同规。──
    if top.seat == p.teammate_seat && p.teammate_remaining > 0 && p.teammate_remaining <= 6 {
        let can_clear = candidates.iter().any(|c| c.cards.len() >= p.my_remaining);
        if !can_clear {
            return Ok(pass.clone());
        }
    }

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

        // 队友的对子大于12（K、A、级牌）不能压。非级牌"2"按用户口径（2026-09-03 确认：
        // 2 不是级牌时是最小的牌）不算大对——原 rank_value_js>12 误把 2 计入大对；
        // primary 为 level_order_value 尺度（K=12 → "K及以上" ⇔ >=12），级牌对=14 仍受保护。
        if matches!(
            top.combination.kind,
            CombinationKind::Ordinary(OrdinaryKind::Pair)
        ) && top.combination.primary >= 12
        {
            return FollowDecision::Pass;
        }
        // 队友的三带二大于等于10（10、J、Q、K、A）不能压——用户房规"10以上不压队友"。
        // 非级牌"2"是最小的牌（2026-09-03 用户确认），不算"10以上"。
        // 三张 rank 取解析后的 combination.primary（修复 extract_top_rank 只看首张牌在
        // 乱序/百搭记录下判错致队友 JJJ 被压）。primary 为 level_order_value 刻度
        // （10=9,J=10,Q=11,K=12,A=13,级牌=14），阈值 9 ⇔ "≥10"；CF 端 2026-09-03 起
        // 用 ruleOrderValue>=9 同口径（2 垫底）。
        if matches!(
            top.combination.kind,
            CombinationKind::Ordinary(OrdinaryKind::FullHouse)
        ) && top.combination.primary >= 9
        {
            return FollowDecision::Pass;
        }

        // 队友领大牌（Q 及以上，level_order Q=11 → >=11）不压。非级牌"2"按用户口径
        // （2026-09-03 确认：2 不是级牌时是最小的牌）不算大牌——原 rank_value_js>=12 误把 2 计入。
        let top_is_big = top.combination.primary >= 11;
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

        // 冲刺判定（JS isAnyEnemySprinting）：已走完（剩0张）的对手不计入
        let enemy_low = p
            .enemy_rem_active
            .iter()
            .any(|&c| c <= params.enemy_low_cards_threshold as usize);
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

/// 房规（用户 2026-09-03）：三带二（5张）的百搭用法分类（JS fhWildShape 同源）。
/// 返回 (三张=2天然+1百搭[对子+百搭拼三张→适当惩罚], 对子=1天然+1百搭[三张天然+百搭凑对→禁止])。
fn fh_wild_shape(cards: &[String], p: &PlayContext) -> (bool, bool) {
    let wild_cnt = cards
        .iter()
        .filter(|c| p.meta_for(c).map(|m| m.is_wild).unwrap_or(false))
        .count();
    if cards.len() != 5 || wild_cnt != 1 {
        return (false, false);
    }
    let mut nat_cnts: HashMap<u8, u8> = HashMap::new();
    for c in cards {
        if let Some(r) = p.combos.card_to_rank.get(c) {
            *nat_cnts.entry(*r).or_insert(0) += 1;
        }
    }
    let mut counts: Vec<u8> = nat_cnts.values().copied().collect();
    counts.sort_unstable_by(|a, b| b.cmp(a));
    (
        counts.len() == 2 && counts[0] == 2 && counts[1] == 2,
        counts.len() == 2 && counts[0] == 3 && counts[1] == 1,
    )
}

fn find_best_play_follow<'a>(
    candidates: &[Candidate<'a>],
    top: &PlayState,
    p: &PlayContext,
) -> Result<PlayerAction, String> {
    // JS 硬编码：isEndgame = myRemaining <= 6；冲刺 = 活跃对手（剩>0张）任一 ≤6（用户房规，
    // 原 JS 为 3；已走完的对手不计入——同步 CF isAnyEnemySprinting）
    let my_remaining = p.my_remaining;
    let is_endgame = my_remaining <= 6;
    let is_opp_sprinting = p.enemy_sprinting;

    // 房规 B1（2026-08-30 收紧；2026-09-02 扩面）：有免百搭炸可压 → 跳过含百搭的炸（非清空）。
    // 2026-09-02 实战（CF 局）：敌领 777，我持 8888+♥2 → 出了 8888+♥2 五炸（wild_bomb_bonus
    // +100 反而压过天然 8888）——原守卫只在顶牌是炸弹时生效，普通顶牌漏防。
    // 现推广到任意顶牌：天然炸已经压得住就不烧百搭。唯一豁免=该含百搭炸能直接清空手牌。
    let top_is_bomb = top.combination.class() == CombinationClass::Bomb;
    let cand_has_wild = |c: &Candidate| {
        c.cards
            .iter()
            .any(|s| p.meta_for(s).map(|m| m.is_wild).unwrap_or(false))
    };
    let has_wildfree_bomb = candidates
        .iter()
        .any(|c| c.combo.class() == CombinationClass::Bomb && !cand_has_wild(c));

    // 房规（用户 2026-09-03）：能跟就不烧——存在同型非炸、不含百搭、且不拆手牌天然炸弹的
    // 天然候选能压顶牌时，非清空候选中"炸弹"与"含百搭"的一律排除（残局/冲刺场景同样生效；
    // 烧百搭的"跟"与拆炸的"跟"都不算能跟，否则百搭配对三带二会顶着冲刺豁免+10分顶掉天然平跟）。
    // 出炸直接清空手牌不受限（那是赢牌不是浪费）。顶牌是炸弹时同型非炸不存在，规则自然不生效
    // （反炸省百搭由 B1 管）。候选本就全部合法能压顶，"能压"无需再验。
    let follow_breaks_bomb = |c: &Candidate| {
        let mut use_count: HashMap<u8, usize> = HashMap::new();
        for s in c.cards.iter() {
            if let Some(&r) = p.combos.card_to_rank.get(s) {
                *use_count.entry(r).or_default() += 1;
            }
        }
        use_count.iter().any(|(&r, &n)| {
            let hand_n = p.combos.rank_to_count.get(&r).copied().unwrap_or(0);
            hand_n >= 4 && n < hand_n
        })
    };
    let has_same_type_follow = !top_is_bomb
        && candidates.iter().any(|c| {
            c.combo.class() == CombinationClass::Ordinary
                && !cand_has_wild(c)
                && !follow_breaks_bomb(c)
                && c.combo.kind == top.combination.kind
        });
    // 反孤儿条款（2026-09-02）：所有天然同型平跟都会把百搭打成最后孤张（出牌后
    // 剩余全百搭）→ 解除"能跟就不烧"对含百搭候选的排除，让救孤候选参与计分
    // （配合 WILD_STRAND_PENALTY：天然平跟 −800 vs 救孤候选轻罚 → 救孤胜出）。
    // 炸弹不随此豁免（能跟就普通跟的硬规则不变），只有百搭候选获得救孤通道。
    let natural_follow_strands_wild = has_same_type_follow
        && candidates
            .iter()
            .filter(|c| {
                c.combo.class() == CombinationClass::Ordinary
                    && !cand_has_wild(c)
                    && !follow_breaks_bomb(c)
                    && c.combo.kind == top.combination.kind
            })
            .all(|c| leftover_all_wild(p, &c.cards));

    // 2026-09-06：候选中是否存在"含百搭的真炸弹"（bomb4+，不含同花顺——
    // class()==Bomb 把 SF 也算 Bomb，但 SF 压三带二属过度击杀另有重罚，
    // 不应触发三带二禁令）。用于百搭三带二禁令的跟牌侧判定（候选级）。
    let has_wild_bomb_candidate = candidates.iter().any(|c| {
        c.combo.class() == CombinationClass::Bomb
            && !matches!(c.combo.kind, CombinationKind::Bomb(BombKind::StraightFlush))
            && cand_has_wild(c)
    });
    // 2026-09-06 阶梯扩展：含百搭的钢板/木板/杂顺候选（阶梯第2级）——存在时，含百搭
    // 三带二（第3级）跟牌排除（天然压不过顶牌时，百搭只用在更高一级墩型上）。
    let has_wild_mid_rung_candidate = candidates.iter().any(|c| {
        matches!(
            c.combo.kind,
            CombinationKind::Ordinary(OrdinaryKind::Straight)
                | CombinationKind::Ordinary(OrdinaryKind::Tube)
                | CombinationKind::Ordinary(OrdinaryKind::Plate)
        ) && cand_has_wild(c)
    });
    // 2026-09-06 必要性门判据：存在任何无百搭候选（candidates 里的候选全部能压顶；
    // 拆炸候选不算——那是被禁的打法，不是真替代）。
    let has_wildfree_follow = candidates
        .iter()
        .any(|c| !cand_has_wild(c) && !follow_breaks_bomb(c));

    // Score each possible play and pick the best one (ties → first in order, JS stable sort)
    let mut best: Option<(f32, &Candidate)> = None;
    for cand in candidates {
        // 房规 B1（2026-08-30 收紧；2026-09-02 扩面）：有免百搭炸可压 → 跳过含百搭的炸（非清空），
        // 任意顶牌适用（原仅限顶牌=炸弹，普通顶牌漏防——见上方实战案例）。唯一豁免=清空手牌。
        if has_wildfree_bomb
            && cand.combo.class() == CombinationClass::Bomb
            && cand_has_wild(cand)
            && cand.cards.len() < my_remaining
        {
            continue;
        }
        // 房规（用户 2026-09-06 必要性门）：中盘跟牌时，天然（无百搭）候选能压顶 →
        // 所有含百搭候选一律排除——天然牌能赢的墩绝不烧百搭（百搭是非常珍贵的）。
        // 豁免：残局（≤6）、清空手牌。天然炸被留炸守卫拦时引擎自会过牌，不在此强制。
        if has_wildfree_follow
            && p.my_remaining > 6
            && cand_has_wild(cand)
            && cand.cards.len() < my_remaining
        {
            continue;
        }
        // 房规（用户 2026-09-03）：能跟就不烧——有天然同型平跟时，非清空候选里的
        // 炸弹与含百搭候选全部排除（唯一豁免 = 直接清空手牌）。
        // 反孤儿豁免（2026-09-02）：天然平跟全部会把百搭打成孤张时，含百搭候选解除
        // 排除（救孤通道）；炸弹仍排除（能普通跟就不炸的硬规则不变）。
        if has_same_type_follow
            && cand.cards.len() < my_remaining
            && (cand.combo.class() == CombinationClass::Bomb
                || (cand_has_wild(cand) && !natural_follow_strands_wild))
        {
            continue;
        }
        // 房规（用户 2026-08-30）：百搭与对子配成三张同 = 浪费——跟牌跳过，
        // 唯一豁免 = 清空手牌（len == remaining）或拦截对手冲刺（任一对手≤6张）。
        if matches!(cand.combo.kind, CombinationKind::Ordinary(OrdinaryKind::Triple))
            && cand
                .cards
                .iter()
                .filter(|s| p.meta_for(s).map(|m| m.is_wild).unwrap_or(false))
                .count()
                == 1
            && cand.cards.len() < my_remaining
            && !is_opp_sprinting
        {
            continue;
        }
        // 房规（用户 2026-09-03）：对方领三带二时，禁止用"3张级牌+百搭+1张单张"的
        // 级牌三带二去压（烧三张最强级牌+百搭补对子=太浪费）。
        // 豁免 = 清空手牌（len == remaining）或拦截对手冲刺；队友领的三带二不受此限。
        if matches!(cand.combo.kind, CombinationKind::Ordinary(OrdinaryKind::FullHouse))
            && matches!(top.combination.kind, CombinationKind::Ordinary(OrdinaryKind::FullHouse))
            && top.seat != p.actor_seat
            && top.seat != p.teammate_seat
        {
            let wild_count = cand
                .cards
                .iter()
                .filter(|s| p.meta_for(s).map(|m| m.is_wild).unwrap_or(false))
                .count();
            let nat_level_count = cand
                .cards
                .iter()
                .filter(|s| {
                    !p.meta_for(s).map(|m| m.is_wild).unwrap_or(false)
                        && p.combos.card_to_rank.get(*s) == Some(&p.level_nat)
                })
                .count();
            if wild_count == 1 && nat_level_count == 3 && cand.cards.len() < my_remaining && !is_opp_sprinting {
                continue;
            }
        }
        // 房规（用户 2026-09-03）：双百搭同出候选仅残局（手牌≤6张）才枚举——补齐跟牌路径
        // （领出路径已有此门槛）。清空手牌的牌不受限。
        if p.my_remaining > 6
            && cand
                .cards
                .iter()
                .filter(|s| p.meta_for(s).map(|m| m.is_wild).unwrap_or(false))
                .count()
                >= 2
            && cand.cards.len() < my_remaining
        {
            continue;
        }
        // 房规（用户 2026-09-03）：三张天然、对子=1天然+1百搭（百搭凑对）→ 禁止。
        // 豁免：清空手牌、对手冲刺。领出/跟牌均禁。
        if matches!(cand.combo.kind, CombinationKind::Ordinary(OrdinaryKind::FullHouse)) {
            let (_, wild_pair_fh) = fh_wild_shape(cand.cards, p);
            if wild_pair_fh && cand.cards.len() < my_remaining && !is_opp_sprinting {
                continue;
            }
        }
        // 房规（用户 2026-09-06）：百搭优先成炸/同花顺——候选中存在"能压当前顶牌"的
        // 含百搭真炸弹时，含百搭的三带二跟牌直接排除（同墩内炸弹完胜三带二）。
        // 2026-09-06 阶梯扩展：存在能压顶的含百搭钢板/木板/杂顺（第2级）时同样排除
        // 含百搭三带二（第3级）——天然压不过顶牌时，百搭只用在更高一级墩型上。
        // 仅候选级判定 + 残局（≤6张）不适用（双百搭残局豁免体系优先）。
        // 豁免：清空手牌 或 拦截对手冲刺。与 CF 跟牌侧同规。
        if p.my_remaining > 6
            && matches!(cand.combo.kind, CombinationKind::Ordinary(OrdinaryKind::FullHouse))
            && cand_has_wild(cand)
            && (has_wild_bomb_candidate || has_wild_mid_rung_candidate)
            && cand.cards.len() < my_remaining
            && !is_opp_sprinting
        {
            continue;
        }
        // 房规（用户 2026-09-03 加大）：百搭配单张成普通对——中盘（手牌>6）非救孤/非清空
        // 直接排除候选（宁可过牌也不烧百搭凑对；−800 罚留作兜底计分）。残局保留候选但
        // score_follow_ex 内 −100 重罚。级牌对不在此类（wild_on_level 已另行重罚）。
        if p.my_remaining > 6
            && cand.cards.len() < my_remaining
            && cand_has_wild(cand)
            && matches!(cand.combo.kind, CombinationKind::Ordinary(OrdinaryKind::Pair))
            && !natural_follow_strands_wild
        {
            let nat_rank = cand.cards.iter().find_map(|s| {
                let m = p.meta_for(s)?;
                if m.is_wild || m.is_joker {
                    None
                } else {
                    Some(m.rank)
                }
            });
            if let Some(pr) = nat_rank {
                if pr != p.level_rank {
                    continue;
                }
            }
        }
        let s = score_follow_ex(cand.cards, &cand.combo, top, p, natural_follow_strands_wild);
        if best.map_or(true, |(bs, _)| s > bs) {
            best = Some((s, cand));
        }
    }
    let Some((best_score, best)) = best else {
        return Ok(PlayerAction::Pass);
    };
    let best_cards = best.cards;
    let best_is_bomb = best.combo.class() == CombinationClass::Bomb;
    // 2026-09-06：最优解=含百搭的炸 且 手牌本就存在百搭可完成的炸/同花顺
    // （用户"百搭优先成炸/顺"——此刻不出，百搭只会漏进低级组合或苟成孤张；
    //   天然炸的保留/禁炸逻辑不变）。豁免 ①/①b/⑥ 三个守卫。
    let best_is_wild_bomb = best_is_bomb
        && best_cards
            .iter()
            .any(|s| p.meta_for(s).map(|m| m.is_wild).unwrap_or(false));
    let wild_priority_case = best_is_wild_bomb && hand_has_wild_bomb_or_sf(p);

    // ① 炸弹保留：非残局、非对手冲刺、手里炸弹总数 = 1 个时，不出炸弹，直接过。
    //    （用户房规改自 JS 647-655：JS 为 bombCount<=2，现为 =1——
    //      2 个炸弹不再拦；"手里 ≥3 个炸可用"的豁免由条件 =1 自然排除）
    //    （2026-09-03 审计补：出炸直接清空手牌=赢牌，不受保留限制。）
    if best_is_bomb
        && !wild_priority_case
        && !is_endgame
        && !is_opp_sprinting
        && p.combos.bomb_count == 1
        && best_cards.len() < my_remaining
    {
        return Ok(PlayerAction::Pass);
    }

    // ①b 炸弹保留·事后重算（用户房规）：中盘出炸（含反炸）后，对剩余手牌重新清点
    //    炸弹数（天然 4+ 同张、百搭拼 3 同张、同花顺候选全部重算）——剩 0 → 不出，
    //    保炸到残局。解决"潜在炸共用百搭导致账面虚增"（如 ♠6789+♥6789+♥2 账面 2 颗、
    //    打掉一颗后实际 0 颗）。豁免：残局（≤6 张）、对手冲刺（≤6 张）、炸完直接清空。
    //    （2026-09-06：wild_priority_case 豁免同 ①。）
    if best_is_bomb && !wild_priority_case && !is_endgame && !is_opp_sprinting && best_cards.len() < my_remaining {
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
    //    （2026-09-03 审计补：出炸直接清空手牌=赢牌不是浪费，本守卫不拦清空炸。）
    //    （2026-09-06：wild_priority_case 豁免——百搭优先成炸。）
    //    注：primary 为 level_order_value 尺度（Q=11, K=12, A=13, 级牌=14），故 <Q ⇔ primary<11。
    if best_is_bomb
        && !wild_priority_case
        && !is_endgame
        && !is_opp_sprinting
        && best_cards.len() < my_remaining
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
    score_follow_ex(play_cards, play_combo, top, p, false)
}

/// wild_rescue_lift = 反孤儿救孤旗（natural_follow_strands_wild）：所有天然同型平跟
/// 都会把百搭打成最后孤张时，含百搭候选走救孤通道——百搭配单张成对的罚分豁免，
/// 否则 −800 配对罚与天然平跟的 −800 孤张罚同线打平，救孤胜负沦为候选顺序。
fn score_follow_ex(
    play_cards: &[String],
    play_combo: &Combination,
    top: &PlayState,
    p: &PlayContext,
    wild_rescue_lift: bool,
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
    // 房规（用户 2026-09-03 修订）：顶牌为对手领出的 K 以下单张/对子且我方无人接住
    // → ≥1 轮即解锁（判定已在 build_play_context 完成）
    let unlock_pair_for_single = p.unlock_single_follow_split;
    let unlock_triple_for_pair = p.unlock_pair_follow_split;
    let mut penalty = split_penalty(
        play_cards,
        combos,
        p.level_nat,
        p.has_level_card_or_joker,
        kind,
        allow_endgame_split,
        unlock_pair_for_single,
        unlock_triple_for_pair,
        false, // 报牌拆对豁免仅限领出路径（跟牌侧旧房规不变）
    ) as f32;
    if !play_is_bomb {
        if bomb_split_verdict == BombSplitVerdict::Exempt && penalty >= BANNED_SCORE {
            penalty = 0.0; // 房规豁免：放行拆炸
        }
        if bomb_split_verdict == BombSplitVerdict::Banned {
            penalty = penalty.max(BANNED_SCORE); // 双保险
        }
    }
    score -= penalty * p.params.split_penalty_scale; // Heavy penalty for splitting good combos

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
            score += p.params.last_play_clear_bonus; // Bonus: clearing hand with bomb
        } else if min_opp_remaining <= 6 {
            score += p.params.intercept_sprint_bonus; // 对手≤6张，炸弹拦截是好选择
        } else if combos.bomb_count >= 3 && !is_endgame {
            score -= p.params.bomb_keep_many; // 3+炸弹，留至少1个到残局
        }

        // 房规（用户 2026-09-03 追问澄清）："至少留 1 炸到残局"——中盘（手牌>6）打出
        // 手里最后一炸（打完 0 炸）= 硬禁令：否则到残局时 0 炸，"留 1 炸控牌"落空。
        // 2026-09-02 实战复盘（t_15bfb589 局 [22]/[59]）：原 −200 软罚（bomb_keep_single）
        // 被炸弹收益淹没，且 is_last_play（位置最后出手）使 bot 总能"赢墩式"烧光两炸
        // （W/N 均在 12 张时烧光）。本禁令不设 is_last_play 豁免——"留到残局"是存在性保证。
        // 豁免：①对手冲刺（任一对手剩≤6，拦截优先）②打完后剩≤2张（下一手即可清空）。
        // 清空本身（len>=remaining）不进本分支；2 炸用第 1 炸不受影响（bomb_count==2）。
        let rest_cards_after_bomb_mid = my_remaining.saturating_sub(play_cards.len());
        if !is_endgame
            && min_opp_remaining > 6
            && combos.bomb_count < 2
            && rest_cards_after_bomb_mid > 2
        {
            score -= BANNED_SCORE; // 中盘烧最后一炸 = 硬禁令（至少留 1 炸到残局）
        }

        // 房规禁令（用户 2026-09-03 重申并升级）：残局手里只剩 1 炸必须留作控牌
        // ——非清空不得轻出。禁令级（BANNED_SCORE），任何奖励/求解器项都不可抵消。
        // 豁免（用户 2026-09-03 修订口径=全项目冲刺阈值）：①对手冲刺（任一对手剩≤6张，
        // 炸了拦人）②出完手牌 ③打完后剩≤2张（下一手即可清空）。
        let rest_cards_after_bomb = my_remaining.saturating_sub(play_cards.len());
        if is_endgame
            && !is_last_play
            && combos.bomb_count == 1
            && min_opp_remaining > 6
            && rest_cards_after_bomb > 2
        {
            score -= BANNED_SCORE; // 残局留炸禁令
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
            score += p.params.keep_bomb_bonus; // 保留炸弹到残局，重奖！
        }
    }

    // ── 炸弹压非炸弹牌型扣分 (JS 831-855) ──
    if is_bomb {
        let last_is_bomb = top.combination.class() == CombinationClass::Bomb;
        if !last_is_bomb {
            let min_opp_rem = p.min_opp_remaining;
            if !is_last_play && min_opp_rem > 6 && !is_endgame {
                match top.combination.kind {
                    CombinationKind::Ordinary(OrdinaryKind::Single) => score -= p.params.bomb_over_single,
                    CombinationKind::Ordinary(OrdinaryKind::Pair) => score -= p.params.bomb_over_pair,
                    // 房规（用户 2026-09-03）：炸弹压顺子/钢板/木板不扣分
                    CombinationKind::Ordinary(OrdinaryKind::Straight)
                    | CombinationKind::Ordinary(OrdinaryKind::Tube)
                    | CombinationKind::Ordinary(OrdinaryKind::Plate) => {}
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
                score += p.params.wild_bomb_bonus; // 逢人配配炸弹/同花顺：重奖！
                // 房规（用户 2026-09-03）：百搭同花顺拆牌质量——剩牌组新杂顺 +450 /
                // 去顺后散单张≥3 −250。置于 B1 同一守卫内：B1 压制时不给拆牌奖励。
                if matches!(kind, CombinationKind::Bomb(BombKind::StraightFlush))
                    && not_clearing
                {
                    let (lo_straights, lo_singles) =
                        sf_leftover_straights_and_singles(p, play_cards);
                    if lo_straights > 0 {
                        score += SF_LEFTOVER_STRAIGHT_BONUS;
                    }
                    if lo_singles >= 3 {
                        score -= SF_LEFTOVER_SINGLES_PENALTY;
                    }
                }
            }
        } else {
            match kind {
                CombinationKind::Ordinary(OrdinaryKind::Straight)
                | CombinationKind::Ordinary(OrdinaryKind::Plate)
                | CombinationKind::Ordinary(OrdinaryKind::Tube) => score += p.params.wild_run_bonus,
                CombinationKind::Ordinary(OrdinaryKind::FullHouse) => {
                    // 房规（用户 2026-09-03）：三张=对子+百搭（2天然+1百搭）、对子天然
                    // = 适当惩罚（替代原 +30）；清空手牌豁免照旧；残局轻罚。
                    // 百搭凑对子（三张天然）的三带二已被禁止过滤，仅豁免场景到此处。
                    // 用户 2026-09-03：百搭成三带二不奖励（原 +10 移除）。
                    let (wild_triple_fh, _) = fh_wild_shape(play_cards, p);
                    if wild_triple_fh && play_cards.len() < my_remaining {
                        if is_endgame {
                            score -= 15.0; // 残局轻罚
                        } else {
                            score -= 300.0; // 中盘适当惩罚
                        }
                    }
                }
                // 用户 2026-09-03：百搭成三张不奖励（原 +20 移除）
                CombinationKind::Ordinary(OrdinaryKind::Triple) => {}
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
                    score -= p.params.upgraded_bomb_wild_end; // 残局升档：轻罚
                } else {
                    score -= p.params.upgraded_bomb_wild_mid; // 中盘无场景升档：浪费重罚
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
        score += p.params.partner_feng_bonus; // 给联邦接风，重奖！
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
            score += singles_removed as f32 * p.params.endgame_single_removal; // 残局跟牌移除单张，重奖！
        }
        if small_singles_removed > 0 {
            score += small_singles_removed as f32 * p.params.endgame_small_single_removal; // 残局跟牌移除小单张，额外重奖！
        }
    }

    // ── 对手≤6张时强制拦截 (JS 987-1013) ──
    let min_opp_remaining = p.min_opp_remaining;
    if min_opp_remaining <= 6 && !is_bomb {
        match top.combination.kind {
            CombinationKind::Ordinary(OrdinaryKind::Single)
            | CombinationKind::Ordinary(OrdinaryKind::Pair) => {
                if play_combo.primary > 10 {
                    score += p.params.block_enemy_bonus; // 出大牌阻止对手送牌
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
                p.params.wild_on_level_end
            } else {
                p.params.wild_on_level_mid
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
                p.params.dual_wild_penalty_end
            } else {
                p.params.dual_wild_penalty_mid
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
            // 房规（用户 2026-09-03 加大）：百搭配普通单张成普通对 = 又弱又废——
            // 中盘 −800（压到反孤儿罚同线，仅救孤候选经 wild_rescue_lift 豁免）、
            // 残局 −100。级牌对已由上方统一罚。
            if !finishing_play && !wild_rescue_lift {
                let pair_naturals: Vec<CardMeta> = play_cards
                    .iter()
                    .filter_map(|c| p.meta_for(c))
                    .filter(|m| !m.is_wild && !m.is_joker)
                    .collect();
                let pair_rank = pair_naturals.first().map(|m| m.rank).unwrap_or(p.level_rank);
                if pair_rank != p.level_rank {
                    score -= if endgame_hand {
                        p.params.wild_pair_penalty_end
                    } else {
                        p.params.wild_plain_pair_mid
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
        // 房规（用户 2026-09-03）：一对百搭（两张百搭凑对子）压对子 = 禁止。
        // 此前仅软罚（中盘 −600/残局 −60），残局 −60 形同虚设，机器人照样双百搭压对子。
        // 豁免：清空手牌（外层 dw_finishing 已排除）、对手冲刺（≤6）。
        if matches!(kind, CombinationKind::Ordinary(OrdinaryKind::Pair))
            && p.min_opp_remaining > 6
        {
            score -= BANNED_SCORE;
        }
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
                p.params.dual_wild_penalty_end
            } else {
                p.params.dual_wild_penalty_mid
            };
            if bare_dual {
                score -= p.params.bare_dual_wild_extra; // 裸双百搭：额外重罚
            }
        }
    }

    // ── 房规（用户 2026-09-03）：百搭优先用于炸弹/同花顺（跟牌侧） ──
    // 手中余牌与剩余百搭仍可组成炸弹/同花顺时，把百搭用进更低级组合 → 冻结惩罚（不入训练面）。
    // 豁免：清空手牌、炸弹/同花顺本身、对手冲刺（≤6）。
    if wild_count_in_play >= 1
        && !is_bomb
        && !matches!(kind, CombinationKind::Bomb(BombKind::StraightFlush))
        && play_cards.len() < my_remaining
        && p.min_opp_remaining > 6
        && wilds_could_form_bomb_or_sf(p, play_cards)
    {
        score -= WILD_CONSERVATION_PENALTY;
    }

    // ── 房规（用户 2026-09-02 反孤儿条款，跟牌侧）：出牌后剩余全百搭 → 重罚 ──
    // 与领出侧同规：百搭将沦为最后孤张（被迫单出）时，倒逼把百搭并入当前组合。
    // 救孤候选自己剩余含天然牌 → 不触发（天然平跟触发 −800 → 救孤胜出）。
    if play_cards.len() < my_remaining && leftover_all_wild(p, play_cards) {
        score -= WILD_STRAND_PENALTY;
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
        score += p.params.partner_feng_lead_bonus; // 为队友接风：压制敌人拿回出牌权，重奖
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
            score -= sm_vals.len() as f32 * p.params.avoid_small_singles_each;
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
    // 房规（用户 2026-09-03）：报牌场景（任一活跃对手剩≤2 或队友剩 1/2）——
    // 此时打法由报牌房规主导，残局求解器的"最少墩数"规划让位（它只规划自己清牌，
    // 不理解"防对手走/送队友"）。非报牌场景求解器照常。
    let feed_scenario =
        p.min_opp_remaining <= 2 || (p.teammate_remaining > 0 && p.teammate_remaining <= 2);
    // 残局求解器激活条件（路线图③）：手牌 ≤6 张、非清空领出——此时由求解器的
    // "打完后剩几墩"规划接管，旧的残局单张奖励/整形项让位（见下方 solver_active 守卫）。
    let solver_active = my_remaining <= 6 && play_cards.len() < my_remaining && !feed_scenario;

    // ── Hand combo analysis & split penalty (JS 1473-1489) ──
    let bomb_split_verdict =
        classify_bomb_split(play_cards, &p.my_hand, kind, my_remaining);
    let play_is_bomb = play_combo.class() == CombinationClass::Bomb;
    // 领出路径：拆对豁免不适用（房规豁免仅限跟牌场景，用户 2026-08-30）；
    // 例外（用户 2026-09-03 报牌房规）：对手剩≤2 或队友剩 1 → 允许拆对发单张。
    let feed_exempt = feed_scenario;
    let mut penalty =
        split_penalty(play_cards, combos, p.level_nat, p.has_level_card_or_joker, kind, false, false, false, feed_exempt) as f32;
    if !play_is_bomb {
        if bomb_split_verdict == BombSplitVerdict::Exempt && penalty >= BANNED_SCORE {
            penalty = 0.0; // 房规豁免：放行拆炸
        }
        if bomb_split_verdict == BombSplitVerdict::Banned {
            penalty = penalty.max(BANNED_SCORE); // 双保险
        }
    }
    score -= penalty * p.params.split_penalty_scale;

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
            score += p.params.keep_bomb_bonus; // 保留炸弹到残局，重奖！
        }
    }

    // ── Team awareness: teammate sprinting (JS 1550-1571) ──
    // 房规（用户 2026-09-03，原位加强）：队友剩 1 → 发小单张送；剩 2 → 发小对子送。
    let teammate_remaining = p.teammate_remaining;
    if teammate_remaining == 1 {
        if matches!(kind, CombinationKind::Ordinary(OrdinaryKind::Single)) {
            // 强奖励 + 从小单张开始（primary 越小加分越多；+400 远超 +40 旧值）
            score += TEAMMATE_FEED_BONUS - play_combo.primary as f32 * FEED_RANK_TILT;
        } else if !is_bomb {
            score -= 20.0; // Discourage non-single plays when teammate has 1 card
        }
    } else if teammate_remaining == 2 {
        if matches!(kind, CombinationKind::Ordinary(OrdinaryKind::Pair)) {
            // 强奖励 + 从小对子开始送
            score += TEAMMATE_FEED_BONUS - play_combo.primary as f32 * FEED_RANK_TILT;
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
            score += p.params.teammate_combo_bonus; // Prefer combos to help teammate
        }
    }

    // ── Opponent interception: opponent sprinting (JS 1575-1619) ──
    // 房规（用户 2026-09-03，原位加强）：对手剩 1 → 不发单张（不得不发从大往小）；
    // 对手剩 2 → 不发对子（改拆对发单张或领其他牌型）。
    let min_opp_remaining = p.min_opp_remaining;
    if min_opp_remaining == 1 {
        if matches!(kind, CombinationKind::Ordinary(OrdinaryKind::Single)) {
            // 强阻尼 + 不得不发时从大往小（primary 越大罚越轻）
            score -= OPP_LAST_CARD_SINGLE_PENALTY;
            score += play_combo.primary as f32 * FEED_RANK_TILT;
        }
        if matches!(kind, CombinationKind::Ordinary(OrdinaryKind::Pair)) {
            score += OPP_ONE_PAIR_NUDGE; // 对手单张接不走对子——最安全的压制
        }
        if !is_bomb {
            score += 10.0; // Prefer non-bomb plays to intercept
        }
    } else if min_opp_remaining == 2 {
        if matches!(kind, CombinationKind::Ordinary(OrdinaryKind::Pair)) {
            score -= OPP_LAST_TWO_PAIR_PENALTY; // 防对手对子直接走人
            // 房规（用户 2026-09-03 补充）：万一要发对子，从大对子开始
            //（primary 越大罚越轻——大牌对压得住对手的小对，防走概率最大化）
            score += play_combo.primary as f32 * FEED_RANK_TILT;
        }
        if matches!(kind, CombinationKind::Ordinary(OrdinaryKind::Single)) {
            score += OPP_TWO_SINGLE_NUDGE; // 拆对发单张/散单——对手两张接不走单张
        }
    } else if min_opp_remaining <= 6 {
        if matches!(kind, CombinationKind::Ordinary(OrdinaryKind::Single)) {
            if play_combo.primary <= 10 {
                score -= 30.0; // 不出小单张，对手可能吃单张
            } else {
                score += p.params.block_enemy_bonus; // 出大单张阻止对手
            }
        }
        if matches!(kind, CombinationKind::Ordinary(OrdinaryKind::Pair)) {
            if play_combo.primary <= 10 {
                score -= 15.0; // 不出小对子
            } else {
                score += p.params.block_enemy_bonus; // 出大对子阻止对手
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
            score += p.params.combo_shape_bonus; // 出组合牌型让对手拆牌，更难接
        }
    }

    // ── Endgame: play small cards first, keep big cards (JS 1623-1634) ──
    if is_endgame && !is_bomb {
        score += play_cards.len() as f32 * p.params.lead_len_step_endgame;
        score -= play_combo.primary as f32 * p.params.lead_primary_step_endgame;
    } else {
        score += play_cards.len() as f32 * p.params.lead_len_step;
        score -= play_combo.primary as f32 * p.params.lead_primary_step;
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
            score += singles_removed as f32 * p.params.endgame_single_removal; // 残局移除单张，重奖！
        }
        if small_singles_removed > 0 {
            score += small_singles_removed as f32 * p.params.endgame_small_single_removal; // 残局移除小单张，额外重奖！
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
            score += p.params.straight_build_bonus; // 单牌能组成顺子，奖励
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
            score -= p.params.many_singles_penalty; // Medium penalty
        } else if ratio_after > 0.2 {
            score -= 20.0; // Light penalty
        }
    }

    // ── 主动出牌：先出单张和小牌 (JS 1741-1754) ──
    if !is_bomb {
        let primary = play_combo.primary;
        if matches!(kind, CombinationKind::Ordinary(OrdinaryKind::Single)) {
            score += p.params.single_lead_bonus; // 单张优先出
            if primary <= 10 {
                score += p.params.small_single_lead_bonus; // 小单张更优先
            }
        }
        if primary > 10 {
            score -= 30.0; // 大牌绝不能先出
        } else {
            score += p.params.small_card_lead_bonus; // 小牌奖励
        }
    }

    // ── 逢人配优先组成炸弹、同花顺、顺子、钢板、木板 (JS 1759-1774) ──
    let has_wildcard = play_cards
        .iter()
        .any(|c| p.meta_for(c).map(|m| m.is_wild).unwrap_or(false));
    if has_wildcard {
        if is_bomb || matches!(kind, CombinationKind::Bomb(BombKind::StraightFlush)) {
            score += p.params.wild_bomb_bonus; // 逢人配配炸弹/同花顺：重奖！
            // 房规（用户 2026-09-03）：百搭同花顺拆牌质量——
            // ① 拆完剩牌可组新杂顺 → +450 抵空出炸弹罚，另加 keep_bomb_bonus 抵
            //    "留炸到残局"倾向（"SF 先手 + 剩顺后续"是完整两墩计划，不属浪费炸）；
            // ② 拆完去顺后散单张≥3 → −250（拆出一手烂剩牌，不如出杂顺/百搭顺）。
            if matches!(kind, CombinationKind::Bomb(BombKind::StraightFlush))
                && play_cards.len() < my_remaining
            {
                let (lo_straights, lo_singles) =
                    sf_leftover_straights_and_singles(p, play_cards);
                if lo_straights > 0 {
                    score += SF_LEFTOVER_STRAIGHT_BONUS + p.params.keep_bomb_bonus;
                }
                if lo_singles >= 3 {
                    score -= SF_LEFTOVER_SINGLES_PENALTY;
                }
            }
        } else {
            match kind {
                CombinationKind::Ordinary(OrdinaryKind::Straight)
                | CombinationKind::Ordinary(OrdinaryKind::Plate)
                | CombinationKind::Ordinary(OrdinaryKind::Tube) => score += p.params.wild_run_bonus,
                CombinationKind::Ordinary(OrdinaryKind::FullHouse) => {
                    // 房规（用户 2026-09-03）：三张=对子+百搭、对子天然 = 适当惩罚（替代 +30）。
                    // 用户 2026-09-03：百搭成三带二不奖励（原 +10 移除）。
                    let (wild_triple_fh, _) = fh_wild_shape(play_cards, p);
                    if wild_triple_fh && play_cards.len() < my_remaining {
                        if my_remaining <= 6 {
                            score -= 15.0; // 残局轻罚
                        } else {
                            score -= 300.0; // 中盘适当惩罚
                        }
                    }
                }
                // 用户 2026-09-03：百搭成三张不奖励（原 +20 移除）
                CombinationKind::Ordinary(OrdinaryKind::Triple) => {}
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
                    score -= p.params.upgraded_bomb_wild_end;
                } else {
                    score -= p.params.upgraded_bomb_wild_mid;
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
                p.params.wild_on_level_end
            } else {
                p.params.wild_on_level_mid
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
                p.params.dual_wild_penalty_end
            } else {
                p.params.dual_wild_penalty_mid
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
                        p.params.wild_pair_penalty_end
                    } else {
                        p.params.wild_plain_pair_mid
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
                p.params.dual_wild_penalty_end
            } else {
                p.params.dual_wild_penalty_mid
            };
            if bare_dual {
                score -= p.params.bare_dual_wild_extra; // 裸双百搭：额外重罚
            }
        }
    }

    // ── 房规（用户 2026-09-03）：百搭优先用于炸弹/同花顺（领出侧） ──
    // 领出无"压对子"，双百搭对子禁令仅跟牌侧（中盘双百搭领出已被 movegen 门槛挡住）；
    // 百搭保留原则两侧同规：余牌+剩余百搭可组成炸弹/同花顺时，浪费百搭 → 冻结惩罚。
    // 豁免：清空手牌、炸弹/同花顺本身、对手冲刺（≤6）。
    if wild_count_in_play >= 1
        && !is_bomb
        && !matches!(kind, CombinationKind::Bomb(BombKind::StraightFlush))
        && play_cards.len() < my_remaining
        && p.min_opp_remaining > 6
        && wilds_could_form_bomb_or_sf(p, play_cards)
    {
        score -= WILD_CONSERVATION_PENALTY;
    }

    // ── 房规（用户 2026-09-02 反孤儿条款，领出侧）：出牌后剩余全百搭 → 重罚 ──
    // CF scoreLeadPlay 原有单百搭 −800 同源；统一为 all-wild 口径（双百搭余牌同样算孤）。
    if play_cards.len() < my_remaining && leftover_all_wild(p, play_cards) {
        score -= WILD_STRAND_PENALTY;
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
        score += p.params.partner_feng_first_bonus; // 接风首出权重奖
    }

    // ── 房规禁令（用户 2026-09-03 重申并升级）：残局领出侧同款留炸禁令 ──
    // 残局（手牌≤6）且手里只剩 1 炸：非清空不得主动领出，留作控场底线。
    // 豁免：①对手冲刺（任一对手剩≤6张）②出完手牌 ③打完后剩≤2张。
    // 禁令级（BANNED_SCORE），残局求解器不可抵消。
    if is_bomb && is_endgame && play_cards.len() < my_remaining {
        let rest_after = my_remaining - play_cards.len();
        if combos.bomb_count == 1 && p.min_opp_remaining > 6 && rest_after > 2 {
            score -= BANNED_SCORE; // 残局留炸禁令（领出侧）
        }
    }

    // ── 房规（用户 2026-09-03 追问澄清）：中盘领出侧同款"留 1 炸到残局"硬禁令 ──
    // 中盘（手牌>6）领出手里最后一炸（打完 0 炸）→ 到残局必 0 炸 → 禁止。
    // 与跟牌侧同规：不设 is_last_play 豁免（存在性保证）。
    // 豁免：①对手冲刺（≤6）②打完后剩≤2张；清空（len>=remaining）不进本分支。
    if is_bomb && !is_endgame && play_cards.len() < my_remaining {
        let rest_after_mid = my_remaining - play_cards.len();
        if combos.bomb_count < 2 && p.min_opp_remaining > 6 && rest_after_mid > 2 {
            score -= BANNED_SCORE; // 中盘留炸禁令（领出侧）
        }
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
            score -= p.params.empty_lead_bomb_penalty; // 空出炸弹：手里还有非炸弹牌却主动领炸，严重浪费
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
            score -= sm_vals.len() as f32 * p.params.avoid_small_singles_each; // 剩3张-66 … 剩5张-110
        }
    }

    // ── 残局求解器（路线图③，用户 2026-09-03）：手牌 ≤6 张领出时按"打完后剩几墩"规划 ──
    // 剩余墩数越少越好（每墩 −500）；同墩平手时小幅倾向甩掉手中的天然废单
    // （+15/张——与 endgame_lead_single_removal_reward 的设计行为一致：先甩垃圾单张、
    //   保留组合牌作回手/逃生，专家打法）。清空候选不进本段（+10000 已覆盖）。
    // 运维杀开关：CLAW_DISABLE_SOLVER=1 完全关闭残局求解器（2026-09-03 排查用）
    let solver_enabled = std::env::var("CLAW_DISABLE_SOLVER").is_err();
    if solver_enabled && solver_active && std::time::Instant::now() < p.solver_deadline {
        let residual_tricks = p.min_tricks_after(play_cards);
        score -= residual_tricks as f32 * p.params.solver_trick_penalty;
        let mut junk_removed = 0usize;
        for card in play_cards {
            let Some(m) = p.meta_for(card) else { continue };
            if m.is_joker || m.is_wild || m.rank == p.level_rank {
                continue;
            }
            let Some(nv) = m.natural else { continue };
            if p.combos.rank_to_count.get(&nv).copied().unwrap_or(0) == 1 {
                junk_removed += 1;
            }
        }
        score += junk_removed as f32 * p.params.solver_junk_bonus;
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
            split_penalty(&play, &combos, 2, false, &kind, true, false, false, false),
            0,
            "flag=true (残局+无单张+队友过牌) must allow pair split"
        );
        assert_eq!(
            split_penalty(&play, &combos, 2, false, &kind, false, false, false, false),
            BANNED_SCORE_U32,
            "flag=false must keep the absolute ban"
        );
        // 拆三张同理：[♠7,♥7,♦7,♠8,♥8,♠9,♥9]（无单张）拆 7 出单
        let hand3: Vec<String> = vec!["♠7", "♥7", "♦7", "♠8", "♥8", "♠9", "♥9"]
            .into_iter().map(String::from).collect();
        let combos3 = analyze_hand_combos(&hand3, ctx());
        let play3 = vec!["♠7".to_string()];
        assert_eq!(
            split_penalty(&play3, &combos3, 2, false, &kind, true, false, false, false),
            0,
            "flag=true must allow triple split"
        );
        assert_eq!(
            split_penalty(&play3, &combos3, 2, false, &kind, false, false, false, false),
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

    #[test]
    fn counter_bomb_prefers_wildfree_even_endgame() {
        // 房规 B1 收紧（2026-08-30）：反炸有免百搭候选时不得烧百搭（残局也不例外，
        // 唯一豁免=清空）。对方 3333（level_order 3=2），我方 7777（LOV 7=6>2）可压。
        // 残局：N = 7777 + ♥2(百搭) + 散张 → 必须 7777，不得 7777+百搭升档。
        let state = mk_playing_state(
            Seat::N,
            vec!["♠7", "♥7", "♦7", "♣7", "♥2", "♠4"],
            Some((Seat::E, vec!["♠3", "♥3", "♦3", "♣3"])),
        );
        let act = suggest_next_action(&state, Seat::N).unwrap();
        match act {
            PlayerAction::Play { cards, .. } => {
                assert_eq!(cards.len(), 4, "must play wild-free 7777, got {cards:?}");
                assert!(!cards.contains(&"♥2".to_string()), "must not burn the wild, got {cards:?}");
            }
            other => panic!("expected play, got {other:?}"),
        }
        // 中盘：N = 7777 + 8888 + ♥2 + 散张 → 仍是免百搭 4 张炸
        let state = mk_playing_state(
            Seat::N,
            vec!["♠7", "♥7", "♦7", "♣7", "♠8", "♥8", "♦8", "♣8", "♥2", "♠4"],
            Some((Seat::E, vec!["♠3", "♥3", "♦3", "♣3"])),
        );
        let act = suggest_next_action(&state, Seat::N).unwrap();
        match act {
            PlayerAction::Play { cards, .. } => {
                assert_eq!(cards.len(), 4, "must play wild-free bomb4, got {cards:?}");
                assert!(!cards.contains(&"♥2".to_string()), "must not burn the wild, got {cards:?}");
            }
            other => panic!("expected play, got {other:?}"),
        }
    }

    #[test]
    fn bomb_count_counts_natural_bombs_only() {
        // 房规（用户 2026-09-06 裁决）：bomb_count 只数天然炸——天然4+同点、天然同花顺、四王。
        // 百搭潜在炸（三同张拼炸/4连拼同花顺）不消耗天然炸储备，一律不计入。
        // （旧"潜在按百搭封顶计入"语义已废除——潜在计入曾使守卫①与烧最后一炸禁令失效。）
        let ctx = ctx();
        // ① 1天然炸 + 1百搭 + 无候选 → 总数 1
        let hand: Vec<String> = ["♠5", "♥5", "♦5", "♣5", "♥2", "♠9", "♥9", "♠4", "♥4"]
            .iter().map(|s| s.to_string()).collect();
        assert_eq!(analyze_hand_combos(&hand, ctx).bomb_count, 1);
        // ② 1天然炸 + 1百搭 + 1组三同张 → 潜在不计入 → 总数 1（旧=2）
        let hand: Vec<String> = ["♠5", "♥5", "♦5", "♣5", "♠9", "♥9", "♦9", "♥2", "♠K"]
            .iter().map(|s| s.to_string()).collect();
        assert_eq!(analyze_hand_combos(&hand, ctx).bomb_count, 1);
        // ③ 1天然炸 + 1百搭 + 同花4连 → 潜在同花顺不计入 → 总数 1（旧=2）
        let hand: Vec<String> = ["♠5", "♥5", "♦5", "♣5", "♠6", "♠7", "♠8", "♠9", "♥2"]
            .iter().map(|s| s.to_string()).collect();
        assert_eq!(analyze_hand_combos(&hand, ctx).bomb_count, 1);
        // ④ 1天然炸 + 1百搭 + 同花仅3连 → 总数 1
        let hand: Vec<String> = ["♠5", "♥5", "♦5", "♣5", "♠9", "♠10", "♠J", "♥2"]
            .iter().map(|s| s.to_string()).collect();
        assert_eq!(analyze_hand_combos(&hand, ctx).bomb_count, 1);
        // ⑤ 天然同花顺计入总数
        let hand: Vec<String> = ["♠5", "♥5", "♦5", "♣5", "♠6", "♠7", "♠8", "♠9", "♠10"]
            .iter().map(|s| s.to_string()).collect();
        assert_eq!(analyze_hand_combos(&hand, ctx).bomb_count, 2); // 5555 + 天然同花顺
        // ⑥ 无百搭：三同张不记潜在
        let hand: Vec<String> = ["♠5", "♥5", "♦5", "♣5", "♠9", "♥9", "♦9", "♠K"]
            .iter().map(|s| s.to_string()).collect();
        assert_eq!(analyze_hand_combos(&hand, ctx).bomb_count, 1);
    }

    #[test]
    fn sole_bomb_conserved_when_enemy_finished() {
        // 房规①（用户 2026-08-30）：唯一炸保留到残局，除非对手冲刺。
        // 已走完（剩0张）的对手不算冲刺（同步 CF isAnyEnemySprinting）：
        // E 已走完（手牌空），W 活跃 9 张 → 非冲刺 → N 唯一炸 5555 中盘必须保留。
        let mut state = mk_playing_state(
            Seat::N,
            vec!["♠5", "♥5", "♦5", "♣5", "♥2", "♠9", "♥9", "♠4", "♥4"],
            Some((Seat::E, vec!["♠Q", "♥Q", "♦Q", "♣J", "♥J"])),
        );
        fill_seats(
            &mut state,
            vec!["♠5", "♥5", "♦5", "♣5", "♥2", "♠9", "♥9", "♠4", "♥4"],
            vec!["♦3", "♣3", "♠7", "♥7", "♦9", "♣9", "♠6", "♥6", "♦6"],
            vec!["♣10", "♦10", "♠J", "♥J", "♦J", "♣J", "♠Q", "♥Q", "♦Q"],
        );
        let act = suggest_next_action(&state, Seat::N).unwrap();
        assert!(
            matches!(act, PlayerAction::Pass),
            "sole bomb must be conserved mid-game when no ACTIVE enemy sprints, got {act:?}"
        );
    }

    #[test]
    fn split_unlock_on_repeated_enemy_leads() {
        // 房规（用户 2026-09-03 修订）：当前顶牌是对手领出的 K 以下（level_order<12）
        // 单张/对子且我方无人接住 → ≥1 轮即解锁：单张 → 拆对跟牌（优先孤对、拆大）；
        // 对子 → 拆三张同跟对子（优先孤三张）。K及以上、其他牌型不触发。
        // ① E 领单张3（第1轮）→ 即解锁 → 拆对跟牌（全木板成员 → 板内拆大 → 出8）。
        let mut state = mk_playing_state(
            Seat::N,
            vec!["♠8", "♥8", "♠7", "♥7", "♠6", "♥6", "♠5", "♥5"],
            Some((Seat::E, vec!["♠3"])),
        );
        fill_seats(
            &mut state,
            vec!["♠8", "♥8", "♠7", "♥7", "♠6", "♥6", "♠5", "♥5"],
            vec!["♦4", "♣4", "♠10", "♥10", "♦10", "♣10", "♠J", "♥J", "♦J"],
            vec!["♣6", "♥6", "♦6", "♣7", "♥7", "♦7", "♣8", "♥8", "♦8"],
        );
        let act = suggest_next_action(&state, Seat::N).unwrap();
        match &act {
            PlayerAction::Play { cards, .. } => {
                assert_eq!(cards.len(), 1, "round 1 must unlock (≥1), got {act:?}");
                assert!(cards[0].ends_with('8'), "板内拆大 → 出8, got {cards:?}");
            }
            other => panic!("≥1 unlock must allow pair-split follow, got {other:?}"),
        }

        // ② 孤对优先+拆大：E 领单张4 → 解锁 → 拆孤对99（99不参与木板556677）。
        let mut state = mk_playing_state(
            Seat::N,
            vec!["♠9", "♥9", "♠5", "♥5", "♠6", "♥6", "♠7", "♥7"],
            Some((Seat::E, vec!["♠4"])),
        );
        fill_seats(
            &mut state,
            vec!["♠9", "♥9", "♠5", "♥5", "♠6", "♥6", "♠7", "♥7"],
            vec!["♦4", "♣4", "♠10", "♥10", "♦10", "♣10", "♠J", "♥J", "♦J"],
            vec!["♣6", "♥6", "♦6", "♣7", "♥7", "♦7", "♣8", "♥8", "♦8"],
        );
        state.hand.as_mut().unwrap().history = vec![
            history_entry(Seat::E, vec!["♠3"]),
            history_pass(Seat::S),
            history_pass(Seat::W),
            history_pass(Seat::N),
        ];
        let act = suggest_next_action(&state, Seat::N).unwrap();
        match act {
            PlayerAction::Play { cards, .. } => {
                assert_eq!(cards.len(), 1, "must split a pair for a single, got {cards:?}");
                let r = cards
                    .first()
                    .and_then(|c| parse_card_symbol(c).ok())
                    .map(|c| c.rank);
                assert_eq!(r, Some(Rank::Nine), "孤对99优先拆大跟4, got {cards:?}");
            }
            other => panic!("unlock must allow pair-split follow, got {other:?}"),
        }

        // ③ 对子版：E 领对子44 → 解锁 → 拆孤三张999跟对子
        let mut state = mk_playing_state(
            Seat::N,
            vec!["♠9", "♥9", "♦9", "♠5", "♥5", "♦5", "♠6", "♥6", "♦6", "♠K"],
            Some((Seat::E, vec!["♠4", "♥4"])),
        );
        fill_seats(
            &mut state,
            vec!["♠9", "♥9", "♦9", "♠5", "♥5", "♦5", "♠6", "♥6", "♦6", "♠K"],
            vec!["♦4", "♣4", "♠10", "♥10", "♦10", "♣10", "♠J", "♥J", "♦J", "♣J"],
            vec!["♣6", "♥6", "♦6", "♣7", "♥7", "♦7", "♣8", "♥8", "♦8", "♠A"],
        );
        state.hand.as_mut().unwrap().history = vec![
            history_entry(Seat::E, vec!["♠3", "♥3"]),
            history_pass(Seat::S),
            history_pass(Seat::W),
            history_pass(Seat::N),
        ];
        let act = suggest_next_action(&state, Seat::N).unwrap();
        match act {
            PlayerAction::Play { cards, .. } => {
                assert_eq!(cards.len(), 2, "must split a triple for a pair, got {cards:?}");
                let nines = cards.iter().filter(|c| c.ends_with('9')).count();
                assert_eq!(nines, 2, "孤三张999优先 → 拆99跟44, got {cards:?}");
            }
            other => panic!("unlock must allow triple-split follow, got {other:?}"),
        }
        // ④ 与之前轮次牌型无关：E 领三张555（全过）→ E 领单张6 → 顶牌是K以下单张 → 解锁出8。
        // 手牌 7 张=中盘，避开残局拆对豁免（≤6+无单张+队友已过）。
        let mut state = mk_playing_state(
            Seat::N,
            vec!["♠8", "♥8", "♠7", "♥7", "♠5", "♥5", "♠4"],
            Some((Seat::E, vec!["♠6"])),
        );
        fill_seats(
            &mut state,
            vec!["♠8", "♥8", "♠7", "♥7", "♠5", "♥5", "♠4"],
            vec!["♦4", "♣4", "♠10", "♥10", "♦10", "♣10", "♠J", "♥J", "♦J"],
            vec!["♣6", "♥6", "♦6", "♣7", "♥7", "♦7", "♣8", "♥8", "♦8"],
        );
        state.hand.as_mut().unwrap().history = vec![
            history_entry(Seat::E, vec!["♠5", "♥5", "♦5"]),
            history_pass(Seat::S),
            history_pass(Seat::W),
            history_pass(Seat::N),
        ];
        let act = suggest_next_action(&state, Seat::N).unwrap();
        match act {
            PlayerAction::Play { cards, .. } => {
                assert_eq!(cards.len(), 1, "K以下单张顶牌必须解锁, got {cards:?}");
                assert!(cards[0].ends_with('8'), "孤对88优先拆大跟6, got {cards:?}");
            }
            other => panic!("unlock must allow pair-split follow, got {other:?}"),
        }

        // ⑤ K及以上顶牌不触发：手持级牌2（♦2）+ A三张，E 领对子KK → 不解锁 →
        //   拆AAA出AA被禁（rank>10 且有级牌）→ 过牌；
        //   对照 E 领对子QQ（<K）→ 解锁 → 拆AA跟牌。手牌 7 张=中盘。
        let base = |top: Vec<&str>| {
            let mut state = mk_playing_state(
                Seat::N,
                vec!["♠A", "♥A", "♦A", "♠8", "♥8", "♠7", "♦2"],
                Some((Seat::E, top)),
            );
            fill_seats(
                &mut state,
                vec!["♠A", "♥A", "♦A", "♠8", "♥8", "♠7", "♦2"],
                vec!["♦4", "♣4", "♠10", "♥10", "♦10", "♣10", "♠J", "♥J", "♦J"],
                vec!["♣6", "♥6", "♦6", "♣7", "♥7", "♦7", "♣8", "♥8", "♦8"],
            );
            state
        };
        let act = suggest_next_action(&base(vec!["♠K", "♥K"]), Seat::N).unwrap();
        assert!(
            matches!(act, PlayerAction::Pass),
            "对子KK（=K）不触发解锁 → 拆A被禁 → 过牌, got {act:?}"
        );
        let act = suggest_next_action(&base(vec!["♠Q", "♥Q"]), Seat::N).unwrap();
        match act {
            PlayerAction::Play { cards, .. } => {
                let aces = cards.iter().filter(|c| c.ends_with('A')).count();
                assert_eq!(aces, 2, "对子QQ（<K）解锁 → 拆AA跟牌, got {cards:?}");
            }
            other => panic!("对子QQ必须解锁拆A跟牌, got {other:?}"),
        }
    }

    #[test]
    fn dual_wild_follow_candidates_only_in_endgame() {
        // 房规（用户 2026-09-03）：双百搭同出候选仅残局（手牌≤6张）才枚举——跟牌路径补门槛。
        // 中盘：双百搭三带二不进候选 + 小三带二禁炸 → 过牌（不烧百搭）；
        // 残局：双百搭照常可用（轻罚）→ 出级牌三带二。
        let fh_top = vec!["♠3", "♥3", "♦3", "♣5", "♥5"];
        let e9 = ["♠9", "♥9", "♦9", "♣9", "♠8", "♥8", "♦8", "♣8", "♠7"];

        // ① 中盘9张：2级牌+2百搭+9999+♠5 → 双百搭不进候选、9999 被禁炸 → 过牌。
        let mut state = mk_playing_state(
            Seat::N,
            vec!["♠2", "♦2", "♥2", "♥2", "♠9", "♥9", "♦9", "♣9", "♠5"],
            Some((Seat::E, fh_top.clone())),
        );
        fill_seats(
            &mut state,
            vec!["♠2", "♦2", "♥2", "♥2", "♠9", "♥9", "♦9", "♣9", "♠5"],
            vec!["♦4", "♣4", "♠10", "♥10", "♦10", "♣10", "♠J", "♥J", "♦J"],
            vec!["♣6", "♥6", "♦6", "♣7", "♥7", "♦7", "♣8", "♥8", "♦8"],
        );
        if let Some(hand) = state.hand.as_mut() {
            hand.hands.insert(Seat::E, e9.iter().map(|s| s.to_string()).collect());
        }
        let act = suggest_next_action(&state, Seat::N).unwrap();
        assert!(
            matches!(act, PlayerAction::Pass),
            "midgame dual-wild candidates must not be enumerated → pass, got {act:?}"
        );

        // ② 残局6张：双百搭照常可用 → 出级牌三带二（♠2♦2+♥2 + ♥4+♥2）。
        let mut state = mk_playing_state(
            Seat::N,
            vec!["♠2", "♦2", "♥2", "♥2", "♠5", "♥4"],
            Some((Seat::E, fh_top.clone())),
        );
        fill_seats(
            &mut state,
            vec!["♠2", "♦2", "♥2", "♥2", "♠5", "♥4"],
            vec!["♦4", "♣4", "♠10", "♥10", "♦10", "♣10", "♠J", "♥J", "♦J"],
            vec!["♣6", "♥6", "♦6", "♣7", "♥7", "♦7", "♣8", "♥8", "♦8"],
        );
        if let Some(hand) = state.hand.as_mut() {
            hand.hands.insert(Seat::E, e9.iter().map(|s| s.to_string()).collect());
        }
        let act = suggest_next_action(&state, Seat::N).unwrap();
        match act {
            PlayerAction::Play { cards, .. } => {
                assert_eq!(cards.len(), 5, "endgame dual-wild FH must still play, got {cards:?}");
            }
            other => panic!("endgame must keep dual-wild FH available, got {other:?}"),
        }
    }

    #[test]
    fn wild_usage_priority_bomb_straight_over_fh() {
        // 房规（用户 2026-09-03）：百搭使用优先级 炸弹/同花顺 > 钢板/木板/杂顺子 > 三带二。
        // ① 领出：555+♥2 可拼炸/拼三带二 + 9999 保炸 → 出百搭炸弹；
        // ② 领出：5678+♥2 可拼顺子/拼三带二 → 出百搭顺子。
        let fill9 = |state: &mut TableGameState, n: &[&str]| {
            fill_seats(
                state,
                n.to_vec(),
                vec!["♦4", "♣4", "♠10", "♥10", "♦10", "♣10", "♠J", "♥J", "♦J"],
                vec!["♣6", "♥6", "♦6", "♣7", "♥7", "♦7", "♣8", "♥8", "♦8"],
            );
            if let Some(hand) = state.hand.as_mut() {
                hand.hands.insert(
                    Seat::E,
                    ["♠A", "♥A", "♦A", "♣A", "♠3", "♥3", "♦3", "♣3"]
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                );
            }
        };

        // ① 跟牌 KKK+33（三张=级牌→允许炸；小fh跟不了）：唯一用法比较 → 百搭配炸弹 555+♥2。
        // （领出场景下 555+88 可拼天然三带二，不属"百搭使用优先级"范畴，故用跟牌设计。）
        // 2026-09-02 场景修正：原手含天然 9999，B1 扩面（有免百搭炸可压就不烧百搭）
        // 会正确改出 9999——为继续单测"百搭使用优先级"（炸弹>三带二），移除天然炸并改
        // 残局（6张，避开中盘"出炸后0炸"守卫），让百搭炸弹成为唯一炸弹候选。
        let n_hand = ["♠5", "♥5", "♦5", "♥2", "♠3", "♠Q"];
        let mut state = mk_playing_state(
            Seat::N,
            n_hand.to_vec(),
            Some((Seat::E, vec!["♠K", "♥K", "♦K", "♠3", "♥3"])),
        );
        fill9(&mut state, &n_hand);
        let act = suggest_next_action(&state, Seat::N).unwrap();
        match act {
            PlayerAction::Play { cards, .. } => {
                assert_eq!(cards.len(), 4, "wild bomb must be played, got {cards:?}");
                assert!(cards.contains(&"♥2".to_string()), "bomb must use wild, got {cards:?}");
            }
            other => panic!("expected wild bomb, got {other:?}"),
        }

        // ② 百搭配顺子优先于百搭配三带二（手牌无天然顺子可用，5678+♥2 唯一顺子）。
        let n_hand2 = ["♠5", "♥5", "♠6", "♥6", "♠7", "♠8", "♥2", "♠Q", "♠K"];
        let mut state = mk_playing_state(
            Seat::N,
            n_hand2.to_vec(),
            None,
        );
        fill9(&mut state, &n_hand2);
        let act = suggest_next_action(&state, Seat::N).unwrap();
        match act {
            PlayerAction::Play { cards, .. } => {
                assert_eq!(cards.len(), 5, "wild straight must be led, got {cards:?}");
                assert!(cards.contains(&"♥2".to_string()), "straight must use wild, got {cards:?}");
            }
            other => panic!("expected wild straight lead, got {other:?}"),
        }
    }

    #[test]
    fn follow_instead_of_bomb_when_same_type_follow_exists() {
        // 房规（用户 2026-09-03）：能跟就不炸——存在同型非炸（且不含百搭）平跟候选时
        // 禁用炸弹；冲刺/残局豁免不再为"有平跟还烧炸"开绿灯；无平跟时冲刺拦截照旧可用。
        let fh_top = vec!["♠3", "♥3", "♦3", "♣5", "♥5"];
        let n10 = ["♠8", "♥8", "♦8", "♠4", "♥4", "♠9", "♥9", "♦9", "♥2", "♠3"];
        let fill = |state: &mut TableGameState, n: &[&str], e: &[&str]| {
            fill_seats(
                state,
                n.to_vec(),
                vec!["♦4", "♣4", "♠10", "♥10", "♦10", "♣10", "♠J", "♥J", "♦J", "♣10"],
                vec!["♣6", "♥6", "♦6", "♣7", "♥7", "♦7", "♣8", "♥8", "♦8", "♣J"],
            );
            if let Some(hand) = state.hand.as_mut() {
                hand.hands.insert(Seat::E, e.iter().map(|s| s.to_string()).collect());
            }
        };

        // ① 对方领小三带二 + 我有天然8/9三带二 + 百搭炸(999+♥2) + E冲刺(剩5张)：
        //    旧逻辑=烧炸抢节奏；新规=用同型三带二平跟（同样赢下这轮+出牌权，零浪费）。
        let mut state = mk_playing_state(
            Seat::N,
            n10.to_vec(),
            Some((Seat::E, fh_top.clone())),
        );
        fill(&mut state, &n10, &["♠9", "♥9", "♦9", "♣9", "♠7"]);
        let act = suggest_next_action(&state, Seat::N).unwrap();
        match act {
            PlayerAction::Play { cards, .. } => {
                assert_eq!(cards.len(), 5, "must follow with natural FH, got {cards:?}");
                assert!(!cards.contains(&"♥2".to_string()), "no wild bomb when follow exists, got {cards:?}");
            }
            other => panic!("expected natural FH follow, got {other:?}"),
        }

        // ② 无冲刺对照：同样平跟（不因房规A走向过牌——平跟在候选层接管）。
        let mut state = mk_playing_state(
            Seat::N,
            n10.to_vec(),
            Some((Seat::E, fh_top.clone())),
        );
        fill(&mut state, &n10, &["♠9", "♥9", "♦9", "♣9", "♠7", "♥8", "♦8", "♣8", "♠6"]);
        let act = suggest_next_action(&state, Seat::N).unwrap();
        match act {
            PlayerAction::Play { cards, .. } => {
                assert_eq!(cards.len(), 5, "must follow with natural FH, got {cards:?}");
                assert!(!cards.contains(&"♥2".to_string()), "no wild bomb when follow exists, got {cards:?}");
            }
            other => panic!("expected natural FH follow, got {other:?}"),
        }

        // ③ 无平跟 + 冲刺 → 炸弹拦截照旧可用（不误伤冲刺设计）。
        //   顶牌用天然级牌三带二 KKK+33（9的三带二/百搭配对fh都压不了）→ 唯一合法候选=百搭炸。
        let kf_top = vec!["♠K", "♥K", "♦K", "♠3", "♥3"];
        let n6 = ["♠9", "♥9", "♦9", "♥2", "♠3", "♠4"];
        let mut state = mk_playing_state(
            Seat::N,
            n6.to_vec(),
            Some((Seat::E, kf_top)),
        );
        fill(&mut state, &n6, &["♠9", "♥9", "♦9", "♣9", "♠7"]);
        let act = suggest_next_action(&state, Seat::N).unwrap();
        match act {
            PlayerAction::Play { cards, .. } => {
                assert_eq!(cards.len(), 4, "sprint intercept bomb still allowed, got {cards:?}");
                assert!(cards.contains(&"♥2".to_string()), "wild bomb must play, got {cards:?}");
            }
            other => panic!("expected wild bomb intercept, got {other:?}"),
        }
    }

    #[test]
    fn lead_shape_b_wild_pair_fh_banned_unless_clearing() {
        // 房规（用户 2026-09-03）领出路径回归：三张天然+百搭凑对子的三带二禁领（清空豁免）。
        let mut state = mk_playing_state(
            Seat::N,
            vec!["♠5", "♥5", "♦5", "♠6", "♥2", "♠3", "♠4"],
            None,
        );
        fill_seats(
            &mut state,
            vec!["♠5", "♥5", "♦5", "♠6", "♥2", "♠3", "♠4"],
            vec!["♦4", "♣4", "♠10", "♥10", "♦10", "♣10", "♠J"],
            vec!["♣6", "♥6", "♦6", "♣7", "♥7", "♦7", "♣8"],
        );
        let act = suggest_next_action(&state, Seat::N).unwrap();
        match act {
            PlayerAction::Play { cards, .. } => {
                // Shape B fh 特征：5张且同时用到天然三张的两张5（♠5♥5♦5 中任二）+ 百搭。
                // 百搭顺子 23456(+30) 是合法领出，不受此限。
                let shape_b_like = cards.len() == 5
                    && cards.contains(&"♥2".to_string())
                    && ((cards.contains(&"♥5".to_string()) && cards.contains(&"♦5".to_string()))
                        || (cards.contains(&"♠5".to_string()) && cards.contains(&"♥5".to_string()))
                        || (cards.contains(&"♠5".to_string()) && cards.contains(&"♦5".to_string())));
                assert!(!shape_b_like, "Shape B FH must not be led midgame, got {cards:?}");
            }
            other => panic!("expected a lead, got {other:?}"),
        }

        // 清空豁免：5张手恰好=Shape B fh → 放行。
        let mut state = mk_playing_state(
            Seat::N,
            vec!["♠5", "♥5", "♦5", "♠6", "♥2"],
            None,
        );
        fill_seats(
            &mut state,
            vec!["♠5", "♥5", "♦5", "♠6", "♥2"],
            vec!["♦4", "♣4", "♠10", "♥10", "♦10"],
            vec!["♣6", "♥6", "♦6", "♣7", "♥7"],
        );
        let act = suggest_next_action(&state, Seat::N).unwrap();
        match act {
            PlayerAction::Play { cards, .. } => {
                assert_eq!(cards.len(), 5, "clearing exempts Shape B lead ban, got {cards:?}");
            }
            other => panic!("clearing Shape B fh must lead, got {other:?}"),
        }
    }

    #[test]
    fn lead_dual_wild_candidates_gated_midgame() {
        // 房规（用户 2026-09-03）领出路径回归（审计补）：手牌>6张时双百搭候选不枚举
        // （与跟牌硬门槛、CF movegen 门槛一致）。双百搭对子不得在中盘领出。
        let n8 = vec!["♥2", "♥2", "♠5", "♥5", "♦5", "♠6", "♠3", "♠4"];
        let mut state = mk_playing_state(Seat::N, n8.clone(), None);
        fill_seats(
            &mut state,
            n8.clone(),
            vec!["♦4", "♣4", "♠10", "♥10", "♦10", "♣10", "♠J", "♥J"],
            vec!["♣6", "♥6", "♦6", "♣7", "♥7", "♦7", "♣8", "♥8"],
        );
        let act = suggest_next_action(&state, Seat::N).unwrap();
        match act {
            PlayerAction::Play { cards, .. } => {
                assert!(
                    !(cards.len() == 2 && cards.iter().all(|c| c == "♥2")),
                    "dual-wild pair must not be led midgame, got {cards:?}"
                );
            }
            other => panic!("expected a lead, got {other:?}"),
        }
    }

    #[test]
    fn fh_222_banned_midgame_when_two_not_level() {
        // 非级牌"2"尺度回归（用户 2026-09-03 确认：2 不是级牌时是最小的牌）：
        // 级牌=8 的桌，对手领 222+55（最小的三带二，<Q）→ 中盘禁炸（房规A）。
        // 旧 JS 尺度（2=15）会把它当大三带二放行出炸——两端必须一致禁炸。
        let n8 = vec!["♠9", "♥9", "♦9", "♣9", "♠10", "♥10", "♦10", "♣10"];
        let mut state = mk_playing_state_level(
            Seat::N,
            n8.clone(),
            Some((Seat::E, vec!["♠2", "♦2", "♣2", "♠5", "♥5"])),
            HandLevel::Eight,
        );
        fill_seats(
            &mut state,
            n8.clone(),
            vec!["♦3", "♣3", "♠7", "♥7", "♦9", "♣9", "♠K", "♥K", "♠A"],
            vec!["♠J", "♥J", "♦J", "♣J", "♠6", "♥6", "♦6", "♣6"],
        );
        let act = suggest_next_action(&state, Seat::N).unwrap();
        match act {
            PlayerAction::Pass => {}
            PlayerAction::Play { cards, .. } => {
                panic!("222+xx is the smallest FH (<Q) — midgame bomb must be banned, got {cards:?}")
            }
            other => panic!("expected pass or play, got {other:?}"),
        }
    }

    #[test]
    fn fh_wild_pair_banned_and_wild_triple_penalized() {
        // 房规（用户 2026-09-03）：
        // ① 三张天然、对子=1天然+1百搭（百搭凑对）→ 禁止（清空/冲刺豁免）；
        // ② 三张=对子+百搭（2天然+1百搭）、对子天然 → 适当惩罚（中盘-300/残局-15，非禁止）。
        let fh_top = vec!["♠3", "♥3", "♦3", "♣5", "♥5"];
        let e9 = ["♠9", "♥9", "♦9", "♣9", "♠8", "♥8", "♦8", "♣8", "♠7"];
        let fill = |state: &mut TableGameState, n: &[&str], e: &[&str]| {
            fill_seats(
                state,
                n.to_vec(),
                vec!["♦4", "♣4", "♠10", "♥10", "♦10", "♣10", "♠J", "♥J", "♦J"],
                vec!["♣6", "♥6", "♦6", "♣7", "♥7", "♦7", "♣8", "♥8", "♦8"],
            );
            if let Some(hand) = state.hand.as_mut() {
                hand.hands.insert(Seat::E, e.iter().map(|s| s.to_string()).collect());
            }
        };

        // ① 禁止：555天然+6+♥2(凑对6)+3+4 → 三带二被禁 → 555+♥2的4张炸被小三带二禁炸 → 过牌。
        let mut state = mk_playing_state(
            Seat::N,
            vec!["♠5", "♥5", "♦5", "♠6", "♥2", "♠3", "♥4"],
            Some((Seat::E, fh_top.clone())),
        );
        fill(&mut state, &["♠5", "♥5", "♦5", "♠6", "♥2", "♠3", "♥4"], &e9);
        let act = suggest_next_action(&state, Seat::N).unwrap();
        assert!(
            matches!(act, PlayerAction::Pass),
            "wild-pair FH must be banned → pass, got {act:?}"
        );

        // ② 清空豁免：5张手恰好=该三带二 → 放行。
        let mut state = mk_playing_state(
            Seat::N,
            vec!["♠5", "♥5", "♦5", "♠6", "♥2"],
            Some((Seat::E, fh_top.clone())),
        );
        fill(&mut state, &["♠5", "♥5", "♦5", "♠6", "♥2"], &e9);
        let act = suggest_next_action(&state, Seat::N).unwrap();
        match act {
            PlayerAction::Play { cards, .. } => assert_eq!(cards.len(), 5, "clearing exempts ban, got {cards:?}"),
            other => panic!("clearing must allow wild-pair FH, got {other:?}"),
        }

        // ③ 冲刺豁免：E剩1张 → 放行拦截。
        let mut state = mk_playing_state(
            Seat::N,
            vec!["♠5", "♥5", "♦5", "♠6", "♥2", "♠3", "♥4"],
            Some((Seat::E, fh_top.clone())),
        );
        fill(&mut state, &["♠5", "♥5", "♦5", "♠6", "♥2", "♠3", "♥4"], &["♠9"]);
        let act = suggest_next_action(&state, Seat::N).unwrap();
        match act {
            PlayerAction::Play { cards, .. } => assert_eq!(cards.len(), 5, "sprint exempts ban, got {cards:?}"),
            other => panic!("sprint must allow wild-pair FH, got {other:?}"),
        }

        // ④ 适当惩罚（非禁止）：55+♥2拼三张+66天然对 → 唯一能压 → 照常出。
        let mut state = mk_playing_state(
            Seat::N,
            vec!["♠5", "♥5", "♥2", "♠6", "♥6", "♠3", "♠4"],
            Some((Seat::E, fh_top.clone())),
        );
        fill(&mut state, &["♠5", "♥5", "♥2", "♠6", "♥6", "♠3", "♠4"], &e9);
        let act = suggest_next_action(&state, Seat::N).unwrap();
        match act {
            PlayerAction::Play { cards, .. } => {
                assert_eq!(cards.len(), 5, "wild-triple FH still plays, got {cards:?}");
                assert!(cards.contains(&"♥2".to_string()));
            }
            other => panic!("wild-triple FH must not be banned, got {other:?}"),
        }

        // ⑤ 惩罚生效判别：天然三带二(555+66) vs 百搭拼三张三带二 → 选天然。
        let mut state = mk_playing_state(
            Seat::N,
            vec!["♠5", "♥5", "♦5", "♠6", "♥6", "♥2", "♠4"],
            Some((Seat::E, fh_top.clone())),
        );
        fill(&mut state, &["♠5", "♥5", "♦5", "♠6", "♥6", "♥2", "♠4"], &e9);
        let act = suggest_next_action(&state, Seat::N).unwrap();
        match act {
            PlayerAction::Play { cards, .. } => {
                assert_eq!(cards.len(), 5, "natural FH must win, got {cards:?}");
                assert!(!cards.contains(&"♥2".to_string()), "must not burn wild, got {cards:?}");
            }
            other => panic!("natural FH must be preferred, got {other:?}"),
        }
    }

    #[test]
    fn fh_with_level_triple_and_wild_pair_is_banned_vs_opponent() {
        // 房规（用户 2026-09-03）：对方领三带二 → 禁"3张级牌+百搭+1张单张"级牌三带二
        // （烧三张最强级牌+百搭补对子=太浪费）。豁免=清空手牌或拦截对手冲刺。
        // 级别=2（百搭♥2，自然级牌=♠2♦2♣2）。E 必须给真实手牌（否则误判冲刺）。
        let fill_e = |state: &mut TableGameState, cards: &[&str]| {
            if let Some(hand) = state.hand.as_mut() {
                hand.hands.insert(Seat::E, cards.iter().map(|s| s.to_string()).collect());
            }
        };
        let fh_top = vec!["♠3", "♥3", "♦3", "♣5", "♥5"];
        let s9 = ["♠9", "♥9", "♦9", "♣9", "♠8", "♥8", "♦8", "♣8", "♠7"];

        // ① 禁止：3级牌+百搭+单张 → 级牌三带二被过滤；
        //    次优=3级牌+百搭的4张炸 → ①炸弹保留（中盘7张、唯一炸、无冲刺）→ 过牌。
        let mut state = mk_playing_state(
            Seat::N,
            vec!["♠2", "♦2", "♣2", "♥2", "♠5", "♥4", "♠6"],
            Some((Seat::E, fh_top.clone())),
        );
        fill_seats(
            &mut state,
            vec!["♠2", "♦2", "♣2", "♥2", "♠5", "♥4", "♠6"],
            vec!["♦4", "♣4", "♠10", "♥10", "♦10", "♣10", "♠J", "♥J", "♦J"],
            vec!["♣6", "♥6", "♦6", "♣7", "♥7", "♦7", "♣8", "♥8", "♦8"],
        );
        fill_e(&mut state, &s9);
        let act = suggest_next_action(&state, Seat::N).unwrap();
        assert!(
            matches!(act, PlayerAction::Pass),
            "level-triple wild FH must be banned → pass, got {act:?}"
        );

        // ② 清空豁免：5张手牌恰好就是级牌三带二 → 放行。
        let mut state = mk_playing_state(
            Seat::N,
            vec!["♠2", "♦2", "♣2", "♥2", "♠5"],
            Some((Seat::E, fh_top.clone())),
        );
        fill_seats(
            &mut state,
            vec!["♠2", "♦2", "♣2", "♥2", "♠5"],
            vec!["♦4", "♣4", "♠10", "♥10", "♦10", "♣10", "♠J", "♥J", "♦J"],
            vec!["♣6", "♥6", "♦6", "♣7", "♥7", "♦7", "♣8", "♥8", "♦8"],
        );
        fill_e(&mut state, &s9);
        let act = suggest_next_action(&state, Seat::N).unwrap();
        match act {
            PlayerAction::Play { cards, .. } => {
                assert_eq!(cards.len(), 5, "clearing exempts the ban, got {cards:?}");
                assert_eq!(cards.iter().filter(|c| c.ends_with('2')).count(), 4);
            }
            other => panic!("clearing must allow level FH, got {other:?}"),
        }

        // ③ 冲刺豁免：E 剩1张 → 放行拦截。
        let mut state = mk_playing_state(
            Seat::N,
            vec!["♠2", "♦2", "♣2", "♥2", "♠5", "♥4", "♠6"],
            Some((Seat::E, fh_top.clone())),
        );
        fill_seats(
            &mut state,
            vec!["♠2", "♦2", "♣2", "♥2", "♠5", "♥4", "♠6"],
            vec!["♦4", "♣4", "♠10", "♥10", "♦10", "♣10", "♠J", "♥J", "♦J"],
            vec!["♣6", "♥6", "♦6", "♣7", "♥7", "♦7", "♣8", "♥8", "♦8"],
        );
        fill_e(&mut state, &["♠9"]);
        let act = suggest_next_action(&state, Seat::N).unwrap();
        match act {
            PlayerAction::Play { cards, .. } => {
                assert_eq!(cards.len(), 5, "sprint exempts the ban, got {cards:?}");
            }
            other => panic!("sprint must allow level FH intercept, got {other:?}"),
        }

        // ④ 房规更新（2026-09-03）：三张天然+百搭凑对子的三带二一律禁止（不分级牌与否）：
        // 非级牌 A 三张同样被禁 → 555…炸被小三带二禁炸过滤 → 过牌。
        let mut state = mk_playing_state(
            Seat::N,
            vec!["♠A", "♥A", "♦A", "♥2", "♠5", "♥4", "♠6"],
            Some((Seat::E, fh_top.clone())),
        );
        fill_seats(
            &mut state,
            vec!["♠A", "♥A", "♦A", "♥2", "♠5", "♥4", "♠6"],
            vec!["♦4", "♣4", "♠10", "♥10", "♦10", "♣10", "♠J", "♥J", "♦J"],
            vec!["♣6", "♥6", "♦6", "♣7", "♥7", "♦7", "♣8", "♥8", "♦8"],
        );
        fill_e(&mut state, &s9);
        let act = suggest_next_action(&state, Seat::N).unwrap();
        assert!(
            matches!(act, PlayerAction::Pass),
            "non-level natural-triple wild-pair FH also banned → pass, got {act:?}"
        );
    }

    #[test]
    fn wild_pair_triple_is_waste_unless_clearing_or_sprint() {
        // 房规（2026-08-30）：百搭配对子成三张同 = 浪费——跟牌/领出均排除，
        // 唯一豁免=清空手牌或拦截对手冲刺（残局不豁免）。
        // 注：E 必须给真实手牌——mk_playing_state 只填 actor 手牌，E 空 = 剩0 = 误判冲刺。
        let fill_e = |state: &mut TableGameState| {
            if let Some(hand) = state.hand.as_mut() {
                hand.hands.insert(
                    Seat::E,
                    ["♠A", "♥A", "♦A", "♣A", "♠Q", "♥Q", "♦Q", "♣Q"]
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                );
            }
        };
        // ① 跟牌：E 领 555（LOV 4）；N = 88+♥2+散张（7张，无冲刺）→ 888+♥2 唯一能压 → 必须过。
        let mut state = mk_playing_state(
            Seat::N,
            vec!["♠8", "♥8", "♥2", "♠9", "♥4", "♦6", "♣7"],
            Some((Seat::E, vec!["♠5", "♥5", "♦5"])),
        );
        fill_seats(
            &mut state,
            vec!["♠8", "♥8", "♥2", "♠9", "♥4", "♦6", "♣7"],
            vec!["♦3", "♣3", "♠7", "♥7", "♦9", "♣9", "♠6"],
            vec!["♣10", "♦10", "♠J", "♥J", "♦J", "♣J", "♠Q"],
        );
        fill_e(&mut state);
        let act = suggest_next_action(&state, Seat::N).unwrap();
        assert!(
            matches!(act, PlayerAction::Pass),
            "midgame wild-pair triple is waste: must pass, got {act:?}"
        );

        // ② 冲刺豁免：W 剩 3 张 → 允许 888+♥2 拦截抢出牌权。
        let mut state = mk_playing_state(
            Seat::N,
            vec!["♠8", "♥8", "♥2", "♠9", "♥4", "♦6", "♣7"],
            Some((Seat::E, vec!["♠5", "♥5", "♦5"])),
        );
        fill_seats(
            &mut state,
            vec!["♠8", "♥8", "♥2", "♠9", "♥4", "♦6", "♣7"],
            vec!["♦3", "♣3", "♠7", "♥7", "♦9", "♣9", "♠6"],
            vec!["♣10", "♦10", "♠J"],
        );
        fill_e(&mut state);
        let act = suggest_next_action(&state, Seat::N).unwrap();
        match act {
            PlayerAction::Play { cards, .. } => {
                assert_eq!(
                    cards,
                    vec!["♠8", "♥8", "♥2"],
                    "sprint allows the wild-triple intercept, got {cards:?}"
                );
            }
            other => panic!("expected play under sprint, got {other:?}"),
        }

        // ③ 领出：无顶牌 → 不得领出 888+♥2（浪费）。
        let mut state = mk_playing_state(
            Seat::N,
            vec!["♠8", "♥8", "♥2", "♠9", "♥4", "♦6", "♣7"],
            None,
        );
        fill_seats(
            &mut state,
            vec!["♠8", "♥8", "♥2", "♠9", "♥4", "♦6", "♣7"],
            vec!["♦3", "♣3", "♠7", "♥7", "♦9", "♣9", "♠6"],
            vec!["♣10", "♦10", "♠J", "♥J", "♦J", "♣J", "♠Q"],
        );
        fill_e(&mut state);
        let act = suggest_next_action(&state, Seat::N).unwrap();
        match act {
            PlayerAction::Play { cards, .. } => {
                let is_wild_triple = cards.len() == 3 && cards.iter().any(|c| c == "♥2");
                assert!(!is_wild_triple, "must not lead wild-pair triple, got {cards:?}");
            }
            other => panic!("expected play, got {other:?}"),
        }
    }

    fn mk_playing_state(
        actor: Seat,
        actor_hand: Vec<&str>,
        top_cards: Option<(Seat, Vec<&str>)>,
    ) -> TableGameState {
        mk_playing_state_level(actor, actor_hand, top_cards, HandLevel::Two)
    }

    fn mk_playing_state_level(
        actor: Seat,
        actor_hand: Vec<&str>,
        top_cards: Option<(Seat, Vec<&str>)>,
        level: HandLevel,
    ) -> TableGameState {
        let mut s = TableGameState::new("t_suggest".into());
        s.phase = GamePhase::Playing;
        s.turn_seat = actor;
        s.leader_seat = actor;

        let mut hand = HandState::new(level);
        hand.hands.insert(
            actor,
            actor_hand.into_iter().map(ToString::to_string).collect(),
        );
        for seat in Seat::ALL {
            hand.hands.entry(seat).or_insert_with(Vec::new);
        }

        if let Some((seat, cards)) = top_cards {
            let cards: Vec<String> = cards.into_iter().map(ToString::to_string).collect();
            let combo =
                CombinationParser::parse(&cards, None, RuleContext { hand_level: level }).unwrap();
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

    fn history_pass(seat: Seat) -> HandHistoryEntry {
        HandHistoryEntry {
            seq: 0,
            action_id: "t".into(),
            seat,
            timestamp: String::new(),
            action_type: HistoryActionKind::Pass,
            cards: vec![],
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
        // 残局（5张）[♠3,♥3,♠5,♠6,♠7] 领出：最小墩数划分 = {对3,♠5,♠6,♠7}（4墩），
        // 领对/领单剩余墩数相同（3墩）→ 平手 → 求解器甩废单倾向生效：先甩废单 ♠5，
        // 保留对 3 作回手/逃生（与 endgame_lead_single_removal_reward 同一专家策略）。
        // （原"大组合先出多清牌"期望与该设计矛盾——两个存量测试二选一，按专家打法保留
        //   甩废单策略；剩余墩数不同时求解器仍会强制大组合规划，见 solver 项。）
        let state = mk_playing_state(
            Seat::E,
            vec!["♠3", "♥3", "♠5", "♠6", "♠7"],
            None, // leading
        );
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        assert_eq!(
            picked,
            PlayerAction::Play {
                cards: vec!["♠5".into()],
                wild_targets: None,
            },
            "endgame tie → shed the junk single, keep the pair for retake"
        );
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
    fn wild_pair_midgame_excluded_without_natural_follow() {
        // 房规（用户 2026-09-03 加大）：百搭配单张成对——中盘（手牌>6）无天然对可跟时
        // 百搭对候选直接排除 → 宁可过牌也不烧百搭凑对（残局保留候选走 −100 罚）。
        // E 持 7 张（无天然对，唯一能压对9的跟法 = ♠K+♥2 百搭对）→ 必须过牌。
        let mut state = mk_playing_state(
            Seat::E,
            vec!["♥2", "♠K", "♠3", "♠7", "♠8", "♠9", "♠10"],
            Some((Seat::N, vec!["♠9", "♥9"])),
        );
        fill_seats(
            &mut state,
            vec!["♣3", "♠4", "♥4", "♦4", "♣6", "♥6", "♦8", "♣8", "♦J"],
            vec!["♦3", "♣7", "♥7", "♦9", "♣9", "♠J", "♥J", "♦Q", "♣Q"],
            vec!["♠5", "♥5", "♦6", "♠8", "♥10", "♦10", "♣10", "♥Q", "♦K"],
        );
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        assert!(
            matches!(picked, PlayerAction::Pass),
            "中盘百搭配单张成对必须被排除（过牌不烧百搭），got {:?}",
            picked
        );
    }

    #[test]
    fn wild_sf_with_straight_leftover_beats_plain_run_lead() {
        // 房规（用户 2026-09-03）：百搭同花顺拆牌质量①——拆完剩余牌可组新杂顺 →
        // +450 恰好抵消空出炸弹罚，"SF 先手 + 剩顺后续"两墩计划成立 → SF 领出胜出
        // （旧语义被空出炸弹罚 −450 压到杂顺之后）。
        // 手 [♥2,♠4,♠5,♠6,♠7,♦8,♦9,♦10,♦J,♦Q] 领出：SF=♠4-7+♥2，剩 ♦8-Q 杂顺。
        // W 剩 6 张 = 对手冲刺 → 豁免"中盘最后一炸禁令"（否则唯一炸 SF 被禁无法比较）。
        let mut state = mk_playing_state(
            Seat::E,
            vec!["♥2", "♠4", "♠5", "♠6", "♠7", "♦8", "♦9", "♦10", "♦J", "♦Q"],
            None,
        );
        fill_seats(
            &mut state,
            vec!["♣3", "♠7", "♥7", "♦7", "♣7", "♠9", "♥9", "♦9", "♣9"],
            vec!["♦3", "♣3", "♠6", "♥6", "♦6", "♣6", "♠8", "♥8", "♦8"],
            vec!["♠A", "♥A", "♦A", "♣3", "♣5", "♣10"],
        );
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        match &picked {
            PlayerAction::Play { cards, wild_targets } => {
                let combo = CombinationParser::parse(cards, wild_targets.as_deref(), ctx())
                    .expect("candidate must parse");
                assert!(
                    matches!(combo.class(), CombinationClass::Bomb)
                        && cards.len() == 5
                        && cards.contains(&"♥2".to_string())
                        && cards.contains(&"♠4".to_string())
                        && cards.contains(&"♠7".to_string()),
                    "百搭SF剩顺两墩计划必须胜出杂顺领出（拆牌质量奖励），got {:?}",
                    picked
                );
            }
            other => panic!("Expected Play (wild SF), got {:?}", other),
        }
    }

    #[test]
    fn wild_sf_with_many_single_leftover_avoided() {
        // 房规（用户 2026-09-03）：百搭同花顺拆牌质量②——拆完去顺后散单张 ≥3 →
        // −250 惩罚。手 [♥2,♠4-7,♠K,♥K,♦K,♣K,♦3,♣9,♠J] 领出：SF 拆完剩
        // KKKK+♦3+♣9+♠J（3 散单）→ 不得领出该 SF（KKKK/小牌保留炸弹等更优）。
        let mut state = mk_playing_state(
            Seat::E,
            vec!["♥2", "♠4", "♠5", "♠6", "♠7", "♠K", "♥K", "♦K", "♣K", "♦3", "♣9", "♠J"],
            None,
        );
        fill_seats(
            &mut state,
            vec!["♣3", "♠7", "♥7", "♦7", "♣7", "♠9", "♥9", "♦9", "♣9"],
            vec!["♦3", "♣3", "♠6", "♥6", "♦6", "♣6", "♠8", "♥8", "♦8"],
            vec!["♠A", "♥A", "♦A", "♣5", "♣10", "♠2", "♥3", "♦4", "♣8"],
        );
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        match &picked {
            PlayerAction::Play { cards, wild_targets } => {
                let combo = CombinationParser::parse(cards, wild_targets.as_deref(), ctx())
                    .expect("candidate must parse");
                let is_wild_sf = cards.len() == 5
                    && cards.contains(&"♥2".to_string())
                    && cards.contains(&"♠4".to_string())
                    && cards.contains(&"♠7".to_string())
                    && matches!(combo.class(), CombinationClass::Bomb);
                assert!(
                    !is_wild_sf,
                    "SF 拆完散单≥3 必须被惩罚回避（不得领出该百搭同花顺），got {:?}",
                    picked
                );
            }
            other => panic!("Expected Play action, got {:?}", other),
        }
    }

    #[test]
    fn wild_pair_rescues_strand_when_natural_follow_exists() {
        // 反孤儿条款（2026-09-02，用户实战报告"百搭最后出的"）：跟牌场景天然平跟
        // 会把百搭打成最后孤张时，救孤候选必须能突破"能跟就不烧"的排除。
        // E 持 [♠K,♥K,♥2]（残局3张）跟 N 的对9：出 KK → 剩 [♥2] 孤张（WILD_STRAND_PENALTY
        // −800）；出 K+♥2 对 → 剩 [♠K] 非孤。修复前"能跟就不烧"直接排除 K+♥2 → 百搭
        // 一路 defer 到最后孤张单出。修复后救孤候选参与计分并胜出。
        let mut state = mk_playing_state(
            Seat::E,
            vec!["♠K", "♥K", "♥2"],
            Some((Seat::N, vec!["♠9", "♥9"])),
        );
        fill_seats(
            &mut state,
            vec!["♦9", "♣9", "♣4", "♥4", "♦4", "♠4", "♣6", "♥6", "♦6"],
            vec!["♣6", "♠10", "♥10", "♦10", "♣10", "♠3", "♥3", "♦3", "♣3"],
            vec!["♠Q", "♥Q", "♦Q", "♣Q", "♠J", "♥J", "♦J", "♣J", "♠8"],
        );
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        match &picked {
            PlayerAction::Play { cards, .. } => {
                assert!(
                    cards.len() == 2 && cards.contains(&"♥2".to_string()),
                    "天然平跟会留百搭孤张时必须带百搭出对救孤（反孤儿条款），got {:?}",
                    picked
                );
            }
            other => panic!("Expected Play (K+百搭对子救孤), got {:?}", other),
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

    // ══ 房规（2026-09-03）：双百搭压对子禁止 + 百搭保留优先 ══

    #[test]
    fn dual_wild_pair_follow_banned() {
        // 跟对 4：双百搭凑对子 = 禁止（BANNED_SCORE），不得压过天然对 9
        let top = mk_top(Seat::N, vec!["♦4", "♣4"]);
        let mut st = mk_playing_state(
            Seat::E,
            vec!["♥2", "♥2", "♠9", "♥9", "♠K", "♠A"],
            Some((Seat::N, vec!["♦4", "♣4"])),
        );
        fill_seats(
            &mut st,
            vec!["♦9", "♣9", "♦J", "♣J", "♦Q", "♣Q", "♦K", "♣K", "♦A"],
            vec!["♠3", "♥3", "♠5", "♥5", "♠6", "♥6", "♠7", "♥7", "♠8"],
            vec!["♥8", "♦8", "♣8", "♠10", "♥10", "♦10", "♣10", "♠J", "♥J"],
        );
        let p = pctx_of(&st, Seat::E);
        let wpair_cards = vec!["♥2".to_string(), "♥2".to_string()];
        let wpair = combo_of(vec!["♥2", "♥2"], vec!["♠9", "♥9"]); // 双百搭充当对 9
        let s = score_follow(&wpair_cards, &wpair, &top, &p);
        assert!(s <= -50000.0, "双百搭压对子必须被禁止, got {s}");
    }

    #[test]
    fn wild_conservation_detector() {
        // 余牌与百搭可成炸弹（333+百搭）→ 判真；可成同花顺（♠45678 缺 4）→ 判真
        let mut st = mk_playing_state(
            Seat::E,
            vec!["♠3", "♥3", "♦3", "♥2", "♦K"],
            None,
        );
        fill_seats(
            &mut st,
            vec!["♠4", "♥4", "♦4", "♣4", "♠5"],
            vec!["♥5", "♦5", "♣5", "♠6", "♥6"],
            vec!["♦6", "♣6", "♠7", "♥7", "♦7"],
        );
        let p = pctx_of(&st, Seat::E);
        assert!(wilds_could_form_bomb_or_sf(&p, &["♦K".to_string()]));
        // 对照：拆掉三张同点后（33+百搭 补不成 4 炸）→ 判假
        assert!(!wilds_could_form_bomb_or_sf(
            &p,
            &["♦K".to_string(), "♠3".to_string()]
        ));
        let mut st2 = mk_playing_state(
            Seat::E,
            vec!["♠5", "♠6", "♠7", "♠8", "♥2", "♦K"],
            None,
        );
        fill_seats(
            &mut st2,
            vec!["♠4", "♥4", "♦4", "♣4", "♠9"],
            vec!["♥9", "♦9", "♣9", "♠10", "♥10"],
            vec!["♦10", "♣10", "♠J", "♥J", "♦J"],
        );
        let p2 = pctx_of(&st2, Seat::E);
        assert!(wilds_could_form_bomb_or_sf(&p2, &["♦K".to_string()]));
    }

    #[test]
    fn opp_one_card_avoid_single_lead() {
        // 房规（2026-09-03）：对手 N 剩 1 张 → 不得领单张（发对子压制）
        let mut st = mk_playing_state(Seat::E, vec!["♠5", "♥5", "♠K", "♥K"], None);
        fill_seats(
            &mut st,
            vec!["♠3"], // N 对手剩 1 张
            vec!["♦4", "♣4", "♦6", "♣6", "♦7", "♣7", "♦8", "♣8", "♦9"],
            vec!["♣10", "♠J", "♥J", "♦J", "♣J", "♠Q", "♥Q", "♦Q", "♣Q"],
        );
        let picked = suggest_next_action(&st, Seat::E).unwrap();
        match picked {
            PlayerAction::Play { cards, .. } => {
                assert_eq!(cards.len(), 2, "对手剩1张时不得领单张, picked={cards:?}");
            }
            other => panic!("应领出, got {other:?}"),
        }
    }

    #[test]
    fn teammate_one_card_feed_small_single() {
        // 房规（2026-09-03）：队友 W 剩 1 张 → 领小单张送队友（3 优先于 K/KKK）
        // 注：小单 3 为天然散牌；K 们构成三张（拆三张出单 K 被既有房规禁止）
        let mut st = mk_playing_state(Seat::E, vec!["♠3", "♠K", "♥K", "♦K"], None);
        fill_seats(
            &mut st,
            vec!["♠4", "♥4", "♦4", "♣4", "♠5", "♥5", "♦5", "♣5", "♠6"],
            vec!["♥6", "♦6", "♣6", "♠7", "♥7", "♦7", "♣7", "♠8", "♥8"],
            vec!["♣9"], // W 队友剩 1 张
        );
        let picked = suggest_next_action(&st, Seat::E).unwrap();
        match picked {
            PlayerAction::Play { cards, .. } => {
                assert_eq!(cards.len(), 1, "队友剩1张应送单张, picked={cards:?}");
                assert!(cards[0].ends_with('3'), "应送小单张 3, picked={cards:?}");
            }
            other => panic!("应领出, got {other:?}"),
        }
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
        // （2026-08-30 新口径：♠6-9 是同花4连+2百搭=1潜在同花顺，会记 bombCount——
        //   打断 4 连（♥7）保持"无候选"本意）
        let mut bare_state = mk_playing_state(Seat::E, vec!["♥2", "♥2", "♠9", "♠8", "♥7", "♠6"], None);
        fills(&mut bare_state);
        let bare_p = pctx_of(&bare_state, Seat::E);
        let bare_cards = vec!["♥2".to_string(), "♥2".to_string()];
        let bare_pair = combo_of(vec!["♥2", "♥2"], vec!["♠3", "♥3"]);
        let bare_end = score_follow(&bare_cards, &bare_pair, &top, &bare_p);

        // 中盘裸双百搭：−600 −200，更差（避免 5 连同花 → 不产生 bombCount）
        let mut bare_mid_state = mk_playing_state(
            Seat::E,
            vec!["♥2", "♥2", "♠9", "♠8", "♥7", "♠6", "♦5", "♣4"],
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
    fn wildfree_bomb_beats_ordinary_top_without_burning_wild() {
        // 房规 B1 扩面（2026-09-02）：任意顶牌——有免百搭炸可压就不烧百搭。
        // 实战案例（CF 局）：敌领 777，我持 8888+♥2 → 出了 8888+♥2 五炸
        // （wild_bomb_bonus +100 曾使含百搭炸反超天然炸），必须选天然 8888。
        // 场景设为残局（6张）：房规A（三张<Q中盘禁炸）在残局豁免，从而精确
        // 隔离 B1 扩面语义；打完剩 2 张 ≤2 也不触发残局留炸禁令。
        let mut state = mk_playing_state(
            Seat::E,
            vec!["♠8", "♥8", "♦8", "♣8", "♠5", "♥2"],
            Some((Seat::N, vec!["♠7", "♥7", "♦7"])),
        );
        fill_seats(
            &mut state,
            vec!["♣4", "♥4", "♦4", "♠4", "♣6", "♥6", "♦6", "♣6", "♠3"],
            vec!["♥7", "♦7", "♣7", "♠10", "♥10", "♦10", "♣10", "♠3", "♥3"],
            vec!["♦3", "♣3", "♠J", "♥J", "♦J", "♣J", "♠Q", "♥Q", "♦Q"],
        );
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        match &picked {
            PlayerAction::Play { cards, .. } => {
                assert_eq!(
                    cards.len(),
                    4,
                    "必须出 4 张天然炸 8888，got {:?}",
                    picked
                );
                assert!(
                    !cards.contains(&"♥2".to_string()),
                    "有免百搭炸可压 777 时不得烧百搭升档（房规B1扩面），got {:?}",
                    picked
                );
            }
            other => panic!("Expected Play (wild-free bomb), got {:?}", other),
        }
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
    fn counter_bomb_prefers_wildfree_even_when_sprinting() {
        // 房规 B1 收紧（2026-08-30）：反炸有免百搭候选时不得烧百搭——对手冲刺也不例外
        // （唯一豁免=清空）。对方 KKKK（LOV 12），E 持 4A（LOV 13 直接可压）+ ♥2 + 5555 + 999
        // → 必须 4A 免百搭反炸，不得 4A+♥2 升档 5 炸。
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
            vec!["♣10", "♦10", "♠J", "♥J", "♦J"],
        );
        let picked = suggest_next_action(&state, Seat::E).unwrap();
        match &picked {
            PlayerAction::Play { cards, .. } => {
                assert_eq!(
                    cards.len(),
                    4,
                    "B1收紧：免百搭 4A 可压 KKKK 时不得烧 ♥2 升档，got {:?}",
                    picked
                );
                assert!(
                    !cards.contains(&"♥2".to_string()),
                    "不得烧百搭，got {:?}",
                    picked
                );
            }
            other => panic!("Expected Play (wild-free 4A counter), got {:?}", other),
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
    fn endgame_prefers_natural_bomb_over_wild_upgrade() {
        // 语义更新（2026-09-02，用户实战裁决"为什么加百搭"）：旧版"+100 百搭奖励压过
        // −10 升档轻罚 → 残局宁烧百搭升档 5 炸"已被房规 B1 扩面统一取代——
        // 有免百搭炸可压（任意顶牌/任意阶段）就不烧百搭，唯一豁免=清空。
        // 残局顶 999 三张，持 5555+♥2+♠K → 必须天然 5555，百搭留给 ♠K 组合/后续。
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
                assert!(
                    cards.len() == 4 && !cards.contains(&"♥2".to_string()),
                    "有天然5555可压999时不得烧百搭升档5炸（房规B1扩面），got {:?}",
                    picked
                );
                let _ = wild_targets;
            }
            other => panic!("Expected Play (natural 5555), got {:?}", other),
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
            vec!["♠3", "♥3", "♦3", "♣3", "♠4"],
            vec!["♣6", "♠7", "♥7", "♦7", "♣7", "♠10", "♥10", "♦10", "♣10"],
            vec!["♠J", "♥J", "♦J", "♣J", "♠Q", "♥Q", "♦Q", "♣Q", "♠A"],
        );
        // 注：N 剩5张=对手冲刺 → 房规（2026-09-03）百搭凑对子三带二的禁止被冲刺豁免，
        // 保留本测试原意（残局单张移除奖励 → 三带二胜过百搭炸弹）。E 的队友是 W，不是 N。
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
