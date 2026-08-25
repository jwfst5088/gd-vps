//! Enumerate legal [`PlayerAction`](crate::game::engine::PlayerAction)s for the current actor.

use std::collections::HashSet;
use std::collections::HashMap;

use crate::domain::Seat;
use crate::game::card::{
    Rank, RuleContext, Suit, is_wild, level_order_value, natural_rank_value, parse_card_symbol,
};
use crate::game::engine::PlayerAction;
use crate::game::rules::beat_comparator::BeatComparator;
use crate::game::rules::combination_parser::CombinationParser;
use crate::game::types::{GamePhase, HandState, TableGameState};

/// Hard cap on combination-index iterations per `enumerate_legal_actions` call.
const MAX_COMBO_TRIES: usize = 240_000;

/// Max Cartesian products for wildcard target enumeration per card subset.
const MAX_WILD_PRODUCT: usize = 256;

/// All 52 suit/rank symbols (for wildcard target enumeration).
fn all_non_joker_symbols() -> Vec<String> {
    let suits = ["♠", "♥", "♦", "♣"];
    let ranks = [
        "A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2",
    ];
    let mut v = Vec::with_capacity(52);
    for s in suits {
        for r in ranks {
            v.push(format!("{}{}", s, r));
        }
    }
    v
}

/// Candidate symbols that may appear as wildcard targets (bounded).
fn wild_target_pool(hand: &[String], ctx: RuleContext) -> Vec<String> {
    let mut set: HashSet<String> = all_non_joker_symbols().into_iter().collect();
    for s in hand {
        if let Ok(c) = parse_card_symbol(s) {
            if !is_wild(c, ctx) {
                set.insert(s.clone());
            }
        }
    }
    let mut v: Vec<String> = set.into_iter().collect();
    
    let level_rank = ctx.hand_level.to_rank();
    v.sort_by(|a, b| {
        let a_card = parse_card_symbol(a).ok();
        let b_card = parse_card_symbol(b).ok();
        
        let a_is_level = a_card.map(|c| c.rank == level_rank).unwrap_or(false);
        let b_is_level = b_card.map(|c| c.rank == level_rank).unwrap_or(false);
        
        if a_is_level && !b_is_level {
            return std::cmp::Ordering::Less;
        }
        if !a_is_level && b_is_level {
            return std::cmp::Ordering::Greater;
        }
        
        let a_val = a_card.map(|c| level_order_value(c, ctx)).unwrap_or(0);
        let b_val = b_card.map(|c| level_order_value(c, ctx)).unwrap_or(0);
        
        b_val.cmp(&a_val).then_with(|| a.cmp(b))
    });
    
    v
}

fn combinations_of_indices(n: usize, k: usize, f: &mut impl FnMut(&[usize]) -> bool) {
    if k == 0 || k > n {
        return;
    }
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        if !f(&idx) {
            return;
        }
        let mut i = k;
        while i > 0 && idx[i - 1] == n - k + i - 1 {
            i -= 1;
        }
        if i == 0 {
            return;
        }
        i -= 1;
        idx[i] += 1;
        for j in i + 1..k {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

fn wild_positions_in_cards(cards: &[String], ctx: RuleContext) -> Vec<usize> {
    cards
        .iter()
        .enumerate()
        .filter_map(|(i, s)| {
            parse_card_symbol(s)
                .ok()
                .filter(|&c| is_wild(c, ctx))
                .map(|_| i)
        })
        .collect()
}

fn try_play(
    cards: &[String],
    wild_targets: Option<&[String]>,
    ctx: RuleContext,
    top: Option<&crate::game::rules::combination_parser::Combination>,
) -> Option<crate::game::rules::combination_parser::Combination> {
    let combo = CombinationParser::parse(cards, wild_targets, ctx).ok()?;
    if let Some(t) = top {
        if !BeatComparator::can_beat(t, &combo) {
            return None;
        }
    }
    Some(combo)
}

fn push_play_unique(
    out: &mut Vec<PlayerAction>,
    seen: &mut HashSet<(Vec<String>, Option<Vec<String>>)>,
    cards: Vec<String>,
    wild_targets: Option<Vec<String>>,
    ctx: RuleContext,
    top: Option<&crate::game::rules::combination_parser::Combination>,
) {
    let wt = wild_targets.clone();
    if try_play(&cards, wt.as_deref(), ctx, top).is_none() {
        return;
    }
    let key = (cards.clone(), wild_targets.clone());
    if seen.insert(key) {
        out.push(PlayerAction::Play {
            cards,
            wild_targets,
        });
    }
}

fn enumerate_wild_products(
    pool: &[String],
    wild_count: usize,
    mut f: impl FnMut(&[String]) -> bool,
) {
    if wild_count == 0 {
        let empty: [String; 0] = [];
        f(&empty);
        return;
    }
    let mut buf = vec![pool.first().cloned().unwrap_or_default(); wild_count];
    let mut count = 0usize;
    fn rec(
        pool: &[String],
        buf: &mut [String],
        pos: usize,
        count: &mut usize,
        max: usize,
        f: &mut impl FnMut(&[String]) -> bool,
    ) -> bool {
        if *count >= max {
            return false;
        }
        if pos == buf.len() {
            *count += 1;
            return f(buf);
        }
        for t in pool {
            buf[pos] = t.clone();
            if !rec(pool, buf, pos + 1, count, max, f) {
                return false;
            }
        }
        true
    }
    let mut fmut = f;
    rec(pool, &mut buf, 0, &mut count, MAX_WILD_PRODUCT, &mut fmut);
}

/// Ascending natural-rank windows of length `k` over `vals` (deduped in place),
/// including the A-low wrap (e.g. straight A2345, tube AA2233, plate AAA222).
fn natural_windows(mut vals: Vec<u8>, k: usize) -> Vec<Vec<u8>> {
    vals.sort_unstable();
    vals.dedup();
    let mut out = Vec::new();
    if k > 0 && vals.len() >= k {
        for i in 0..=(vals.len() - k) {
            let w = &vals[i..i + k];
            if w.windows(2).all(|p| p[1] == p[0] + 1) {
                out.push(w.to_vec());
            }
        }
    }
    if k >= 2 {
        // A-low: A(14) treated as lowest — {A,2} / {A,2,3} / {A,2,3,4,5}.
        let low: Vec<u8> = (2u8..2u8 + (k as u8 - 1))
            .chain(std::iter::once(14u8))
            .collect();
        if low.iter().all(|v| vals.contains(v)) && !out.contains(&low) {
            out.push(low);
        }
    }
    out
}

/// 房规（对齐 bot-advanced.js `findBestLeadPlay` 的 allTypes —— 现包含全部炸弹
/// 类型，以及 `generatePlaysOfType` 的结构化生成）：领出/跟牌候选必须覆盖
/// 同 rank 炸弹 4..=10、同花顺、四王炸与木板/钢板。
///
/// 通用 k 子集枚举受 [`MAX_COMBO_TRIES`] 预算限制：满手 27 张时 k>=6 的子集会
/// 被截断（k=7..10 完全不会执行），导致大炸弹从领出候选中消失。这里按牌组
/// 直接生成补齐；仅用纯自然牌，含百搭的变体仍由通用路径枚举。小手牌下产生的
/// 动作与通用枚举重复，由 `seen` 去重，不改变行为。
fn generate_structural_bombs_and_sequences(
    h: &[String],
    ctx: RuleContext,
    top: Option<&crate::game::rules::combination_parser::Combination>,
    out: &mut Vec<PlayerAction>,
    seen: &mut HashSet<(Vec<String>, Option<Vec<String>>)>,
) {
    let mut rank_to_indices: HashMap<u8, Vec<usize>> = HashMap::new();
    let mut suit_rank_index: HashMap<Suit, HashMap<u8, usize>> = HashMap::new();
    let mut reds: Vec<usize> = Vec::new();
    let mut blacks: Vec<usize> = Vec::new();
    for (i, s) in h.iter().enumerate() {
        let Ok(c) = parse_card_symbol(s) else {
            continue;
        };
        if is_wild(c, ctx) {
            continue;
        }
        match c.suit {
            Suit::Joker => match c.rank {
                Rank::RedJoker => reds.push(i),
                Rank::BlackJoker => blacks.push(i),
                _ => {}
            },
            _ => {
                if let Ok(nat) = natural_rank_value(c.rank) {
                    rank_to_indices.entry(nat).or_default().push(i);
                    suit_rank_index
                        .entry(c.suit)
                        .or_default()
                        .entry(nat)
                        .or_insert(i);
                }
            }
        }
    }

    // 同 rank 炸弹 n = 4..=min(count, 10)。
    let mut ranks: Vec<u8> = rank_to_indices.keys().copied().collect();
    ranks.sort_unstable();
    for r in ranks {
        let idxs = &rank_to_indices[&r];
        let max_n = idxs.len().min(10);
        for n in 4..=max_n {
            let cards: Vec<String> = idxs.iter().take(n).map(|&i| h[i].clone()).collect();
            push_play_unique(out, seen, cards, None, ctx, top);
        }
    }

    // 四王炸：双红双黑。
    if reds.len() >= 2 && blacks.len() >= 2 {
        let cards = vec![
            h[reds[0]].clone(),
            h[reds[1]].clone(),
            h[blacks[0]].clone(),
            h[blacks[1]].clone(),
        ];
        push_play_unique(out, seen, cards, None, ctx, top);
    }

    // 同花顺：同花色 5 连自然 rank（含 A 低顺）。
    for by_rank in suit_rank_index.values() {
        let vals: Vec<u8> = by_rank.keys().copied().collect();
        for w in natural_windows(vals, 5) {
            let cards: Vec<String> = w.iter().map(|v| h[by_rank[v]].clone()).collect();
            push_play_unique(out, seen, cards, None, ctx, top);
        }
    }

    // 木板：三连对（含 A 低 AA2233）。
    let pair_ranks: Vec<u8> = rank_to_indices
        .iter()
        .filter(|(_, v)| v.len() >= 2)
        .map(|(&r, _)| r)
        .collect();
    for w in natural_windows(pair_ranks, 3) {
        let cards: Vec<String> = w
            .iter()
            .flat_map(|v| rank_to_indices[v].iter().take(2))
            .map(|&i| h[i].clone())
            .collect();
        if cards.len() == 6 {
            push_play_unique(out, seen, cards, None, ctx, top);
        }
    }

    // 钢板：两连三张（含 A 低 AAA222）。
    let triple_ranks: Vec<u8> = rank_to_indices
        .iter()
        .filter(|(_, v)| v.len() >= 3)
        .map(|(&r, _)| r)
        .collect();
    for w in natural_windows(triple_ranks, 2) {
        let cards: Vec<String> = w
            .iter()
            .flat_map(|v| rank_to_indices[v].iter().take(3))
            .map(|&i| h[i].clone())
            .collect();
        if cards.len() == 6 {
            push_play_unique(out, seen, cards, None, ctx, top);
        }
    }
}

fn enumerate_playing(
    hand_state: &HandState,
    actor: Seat,
    ctx: RuleContext,
) -> Result<Vec<PlayerAction>, String> {
    let h = hand_state
        .hands
        .get(&actor)
        .ok_or_else(|| "missing actor hand".to_string())?;
    let top = hand_state.trick.top_play.as_ref().map(|p| &p.combination);

    // No cards: must pass (engine).
    if h.is_empty() {
        return Ok(vec![PlayerAction::Pass]);
    }

    let mut out = Vec::new();
    let mut seen: HashSet<(Vec<String>, Option<Vec<String>>)> = HashSet::new();
    let n = h.len();
    let max_k = n.min(10);

    // Leading: cannot pass while holding cards.
    let may_pass = top.is_some();

    if may_pass {
        out.push(PlayerAction::Pass);
        seen.insert((vec![], None)); // not used for Pass dedup
    }

    let pool = wild_target_pool(h, ctx);
    let mut tries = 0usize;

    // Pre-compute wild positions in hand to avoid repeated wild_positions_in_cards calls
    let hand_wild_positions: Vec<bool> = h.iter().map(|s| {
        parse_card_symbol(s).ok().map(|c| is_wild(c, ctx)).unwrap_or(false)
    }).collect();

    // Generate k=1 (singles) directly: each card is a unique play.
    // 逢人配（红桃级牌）极其珍贵，绝大多数情况不能单出。
    // 只有当手牌只剩逢人配（没有其他选择）时，才允许逢人配单出。
    for i in 0..n {
        let cards = vec![h[i].clone()];
        if hand_wild_positions[i] {
            // 逢人配单出：仅当手牌全是逢人配（无其他牌可选）时才允许
            if n == 1 || h.iter().all(|s| {
                parse_card_symbol(s).ok().map(|c| is_wild(c, ctx)).unwrap_or(false)
            }) {
                enumerate_wild_products(&pool, 1, |targets| {
                    push_play_unique(&mut out, &mut seen, cards.clone(), Some(targets.to_vec()), ctx, top);
                    true
                });
            }
            // 否则跳过逢人配单出，不生成该动作
        } else {
            push_play_unique(&mut out, &mut seen, cards, None, ctx, top);
        }
    }

    // Generate k=2 (pairs) directly: group by rank to avoid duplicates
    if n >= 2 {
        let mut rank_to_indices: HashMap<u8, Vec<usize>> = HashMap::new();
        for i in 0..n {
            if hand_wild_positions[i] { continue; }
            if let Ok(c) = parse_card_symbol(&h[i]) {
                if let Ok(nat) = natural_rank_value(c.rank) {
                    rank_to_indices.entry(nat).or_default().push(i);
                }
            }
        }
        let mut seen_pairs: HashSet<Vec<String>> = HashSet::new();
        for indices in rank_to_indices.values() {
            if indices.len() < 2 { continue; }
            for a in 0..indices.len() {
                for b in (a+1)..indices.len() {
                    let mut cards = vec![h[indices[a]].clone(), h[indices[b]].clone()];
                    cards.sort();
                    if seen_pairs.insert(cards.clone()) {
                        // Check wildcard interactions
                        let wilds_in_card: Vec<usize> = (0..2).filter(|&j| {
                            parse_card_symbol(&cards[j]).ok().map(|c| is_wild(c, ctx)).unwrap_or(false)
                        }).collect();
                        if wilds_in_card.is_empty() {
                            push_play_unique(&mut out, &mut seen, cards, None, ctx, top);
                        } else {
                            let wn = wilds_in_card.len();
                            enumerate_wild_products(&pool, wn, |targets| {
                                push_play_unique(&mut out, &mut seen, cards.clone(), Some(targets.to_vec()), ctx, top);
                                true
                            });
                        }
                    }
                }
            }
        }
    }

    // 房规：候选必须包含全部炸弹类型（同 rank 4..=10、同花顺、四王）与
    // 木板/钢板 —— 领出不得被限制在炸弹之外。置于预算受限的通用枚举之前，
    // 保证大手牌时也不因 MAX_COMBO_TRIES 截断而缺失。
    generate_structural_bombs_and_sequences(h, ctx, top, &mut out, &mut seen);

    // Generate k=3..=max_k via combinations
    for k in 3..=max_k {
        combinations_of_indices(n, k, &mut |idxs| {
            if tries >= MAX_COMBO_TRIES {
                return false;
            }
            tries += 1;
            let cards: Vec<String> = idxs.iter().map(|&i| h[i].clone()).collect();
            let wild_count = idxs.iter().filter(|&&i| hand_wild_positions[i]).count();
            if wild_count == 0 {
                push_play_unique(&mut out, &mut seen, cards, None, ctx, top);
            } else {
                enumerate_wild_products(&pool, wild_count, |targets| {
                    push_play_unique(
                        &mut out,
                        &mut seen,
                        cards.clone(),
                        Some(targets.to_vec()),
                        ctx,
                        top,
                    );
                    true
                });
            }
            true
        });
        if tries >= MAX_COMBO_TRIES {
            break;
        }
    }

    // If leading and nothing parsed, fall back to playing the lowest single card.
    if top.is_none() && !h.is_empty() {
        let has_play = out.iter().any(|a| matches!(a, PlayerAction::Play { .. }));
        if !has_play {
            // Fallback: play the lowest non-wild single card
            let mut best_single: Option<String> = None;
            let mut best_val: u8 = 255;
            for card in h {
                if let Ok(c) = parse_card_symbol(card) {
                    if !is_wild(c, ctx) {
                        let v = level_order_value(c, ctx);
                        if v < best_val {
                            best_val = v;
                            best_single = Some(card.clone());
                        }
                    }
                }
            }
            // If all cards are wild, pick any non-wild or just the first card
            if best_single.is_none() {
                best_single = h.first().cloned();
            }
            if let Some(card) = best_single {
                out.push(PlayerAction::Play {
                    cards: vec![card],
                    wild_targets: None,
                });
            } else {
                return Err("movegen: no legal lead play found (budget or rules)".into());
            }
        }
    }

    Ok(out)
}

fn enumerate_tribute(hand_state: &HandState, actor: Seat, ctx: RuleContext) -> Vec<PlayerAction> {
    let h = match hand_state.hands.get(&actor) {
        Some(x) => x,
        None => return vec![],
    };
    let mut best = 0u8;
    for s in h {
        if let Ok(c) = parse_card_symbol(s) {
            if !is_wild(c, ctx) {
                best = best.max(level_order_value(c, ctx));
            }
        }
    }
    let mut out = Vec::new();
    for s in h {
        if let Ok(c) = parse_card_symbol(s) {
            if !is_wild(c, ctx) && level_order_value(c, ctx) == best {
                out.push(PlayerAction::Tribute { card: s.clone() });
            }
        }
    }
    out
}

fn enumerate_return(hand_state: &HandState, actor: Seat) -> Result<Vec<PlayerAction>, String> {
    let tribute = hand_state
        .tribute
        .as_ref()
        .ok_or_else(|| "missing tribute".to_string())?;
    let pair = tribute
        .pairs
        .iter()
        .find(|p| p.receiver == actor && p.return_card.is_none())
        .ok_or_else(|| "not return actor".to_string())?;
    let paid = pair
        .paid_card
        .as_ref()
        .ok_or_else(|| "tribute not paid".to_string())?;
    let paid_rank = parse_card_symbol(paid)?.rank;
    let level_rank = hand_state.hand_level.to_rank();
    let h = hand_state
        .hands
        .get(&actor)
        .ok_or_else(|| "missing hand".to_string())?;
    let mut out = Vec::new();
    for s in h {
        let r = parse_card_symbol(s)?.rank;
        if r != paid_rank && r != level_rank {
            out.push(PlayerAction::ReturnCard { card: s.clone() });
        }
    }
    Ok(out)
}

/// Seat that must act next (differs from [`TableGameState::turn_seat`] during tribute / exchange).
pub fn current_actor_seat(state: &TableGameState) -> Option<Seat> {
    let h = state.hand.as_ref()?;
    match state.phase {
        GamePhase::Tribute => {
            if let Some(a) = h.next_tribute_actor() {
                return Some(a);
            }
            let t = h.tribute.as_ref()?;
            if t.canceled {
                return t.opening_lead_candidates.first().copied();
            }
            None
        }
        GamePhase::Exchange => h.next_exchange_actor(),
        GamePhase::Playing => Some(state.turn_seat),
        GamePhase::Scoring | GamePhase::Dealing | GamePhase::TableSetup | GamePhase::Completed => {
            None
        }
    }
}

/// All legal actions for `actor` when they are the current actor ([`current_actor_seat`]).
pub fn enumerate_legal_actions(
    state: &TableGameState,
    actor: Seat,
) -> Result<Vec<PlayerAction>, String> {
    if current_actor_seat(state) != Some(actor) {
        return Err("not actor turn".into());
    }
    let hand = state.hand.as_ref().ok_or_else(|| "no hand".to_string())?;
    let ctx = RuleContext {
        hand_level: hand.hand_level,
    };

    match state.phase {
        GamePhase::Playing => enumerate_playing(hand, actor, ctx),
        GamePhase::Tribute => Ok(enumerate_tribute(hand, actor, ctx)),
        GamePhase::Exchange => enumerate_return(hand, actor),
        GamePhase::Scoring | GamePhase::Dealing | GamePhase::TableSetup | GamePhase::Completed => {
            Ok(vec![])
        }
    }
}
