use serde_json::Value;

#[derive(Clone, Debug, Default)]
pub struct RuleFeatures {
    pub legal_actions: Vec<String>,
    pub can_pass: bool,
    pub can_play: bool,
    pub my_seat: Option<String>,
    pub teammate_seat: Option<String>,
    pub top_play_seat: Option<String>,
    pub my_hand_count: usize,
    pub low_card_count: usize,
    pub low_card_ratio: f32,
    pub teammate_remaining: Option<u8>,
    pub enemy_min_remaining: Option<u8>,
    pub enemy_low_cards_urgent: bool,
    pub endgame_mode: bool,
    pub leading_new_trick: bool,
    /// Estimated bomb count in hand (same-rank 4+ cards).
    /// Used for bomb conservation strategy.
    pub bomb_count: usize,
    /// Kind of the current top play combination (e.g. "Ordinary(Single)", "Ordinary(Pair)").
    pub top_play_kind: Option<String>,
    /// Count of single cards with rank 2-12 (excluding level), for teammate follow strategy.
    pub medium_single_count: usize,
    /// Count of ranks with 2+ cards of rank 2-9 (excluding level), for teammate pair follow.
    pub small_pair_rank_count: usize,
    /// Count of ranks with 3+ cards of rank 2-9 (excluding level), for teammate triple follow.
    pub small_triple_rank_count: usize,
    /// Primary value of the current top play (if any), used to detect level card suppression.
    pub top_play_value: Option<u8>,
    /// Current hand level rank value (e.g., 2 for level 2).
    pub level_rank: Option<u8>,
}

pub fn extract_rule_features(
    state: &Value,
    enemy_low_cards_threshold: u8,
    endgame_hand_count_threshold: u8,
) -> RuleFeatures {
    let legal_actions: Vec<String> = state
        .get("expect")
        .and_then(|x| x.get("legalActions"))
        .and_then(|x| x.as_array())
        .map(|xs| {
            xs.iter()
                .filter_map(|v| v.as_str().map(ToString::to_string))
                .collect()
        })
        .unwrap_or_default();
    let can_pass = legal_actions.iter().any(|s| s == "pass");
    let can_play = legal_actions.iter().any(|s| s == "play");

    let my_seat = state
        .get("private")
        .and_then(|x| x.get("seat"))
        .and_then(|x| x.as_str())
        .map(ToString::to_string);
    let teammate_seat = state
        .get("private")
        .and_then(|x| x.get("teammateSeat"))
        .and_then(|x| x.as_str())
        .map(ToString::to_string)
        .filter(|s| !s.is_empty());
    let top_play_seat = state
        .get("hand")
        .and_then(|h| h.get("topPlay"))
        .and_then(|tp| tp.get("seat"))
        .and_then(|x| x.as_str())
        .map(ToString::to_string);

    let hand_cards: Vec<String> = state
        .get("private")
        .and_then(|x| x.get("handCards"))
        .and_then(|x| x.as_array())
        .map(|cards| {
            cards
                .iter()
                .filter_map(|v| v.as_str().map(ToString::to_string))
                .collect()
        })
        .unwrap_or_default();
    let hand_level = state
        .get("hand")
        .and_then(|h| h.get("handLevel"))
        .and_then(|v| v.as_str());

    let my_hand_count = hand_cards.len();
    let low_card_count = hand_cards
        .iter()
        .filter(|s| is_small_card_symbol(s, hand_level))
        .count();
    let low_card_ratio = if my_hand_count == 0 {
        0.0
    } else {
        low_card_count as f32 / my_hand_count as f32
    };

    let teammate_remaining = teammate_seat
        .as_ref()
        .and_then(|seat| remaining_count_by_seat(state, seat));
    let enemy_min_remaining =
        min_enemy_remaining(state, my_seat.as_deref(), teammate_seat.as_deref());
    let enemy_low_cards_urgent = enemy_min_remaining
        .map(|r| r <= enemy_low_cards_threshold)
        .unwrap_or(false);
    let endgame_mode = my_hand_count <= endgame_hand_count_threshold as usize;
    let leading_new_trick = top_play_seat.is_none();

    let bomb_count = count_bombs_in_hand(&hand_cards, hand_level);

    // Extract top play kind from state JSON (topPlay is inside hand)
    let top_play_kind = state
        .get("hand")
        .and_then(|h| h.get("topPlay"))
        .and_then(|tp| tp.get("combinationKind"))
        .and_then(|v| v.as_str())
        .map(ToString::to_string);

    // Count medium singles (rank 2-12, excluding level) for teammate follow strategy
    let medium_single_count = count_medium_singles(&hand_cards, hand_level);

    // Count small pair ranks (2+ cards of rank 2-9, excluding level)
    let small_pair_rank_count = count_small_pair_ranks(&hand_cards, hand_level);

    // Count small triple ranks (3+ cards of rank 2-9, excluding level)
    let small_triple_rank_count = count_small_triple_ranks(&hand_cards, hand_level);

    // Extract top play primary value for level card detection
    let top_play_value = state
        .get("hand")
        .and_then(|h| h.get("topPlay"))
        .and_then(|tp| tp.get("primary"))
        .and_then(|v| v.as_u64())
        .map(|v| v as u8);

    // Compute current level rank
    let level_rank = hand_level.and_then(|hl| {
        let rank_str = hl.chars().next().map(|c| c.to_string()).unwrap_or_default();
        rank_str.parse::<u8>().ok()
    });

    RuleFeatures {
        legal_actions,
        can_pass,
        can_play,
        my_seat,
        teammate_seat,
        top_play_seat,
        my_hand_count,
        low_card_count,
        low_card_ratio,
        teammate_remaining,
        enemy_min_remaining,
        enemy_low_cards_urgent,
        endgame_mode,
        leading_new_trick,
        bomb_count,
        top_play_kind,
        medium_single_count,
        small_pair_rank_count,
        small_triple_rank_count,
        top_play_value,
        level_rank,
    }
}

fn remaining_count_by_seat(state: &Value, seat: &str) -> Option<u8> {
    state
        .get("seats")
        .and_then(|x| x.get(seat))
        .and_then(|x| x.get("remainingCount"))
        .and_then(|x| x.as_u64())
        .map(|x| x as u8)
}

fn min_enemy_remaining(
    state: &Value,
    my_seat: Option<&str>,
    teammate_seat: Option<&str>,
) -> Option<u8> {
    let seats = state.get("seats").and_then(|x| x.as_object())?;
    seats
        .iter()
        .filter_map(|(seat, _)| {
            if my_seat == Some(seat.as_str()) || teammate_seat == Some(seat.as_str()) {
                None
            } else {
                remaining_count_by_seat(state, seat)
            }
        })
        .min()
}

fn is_small_card_symbol(card: &str, hand_level: Option<&str>) -> bool {
    let Some(rank) = card_rank_token(card) else {
        return false;
    };
    let is_small = matches!(rank, "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "10");
    if !is_small {
        return false;
    }
    hand_level != Some(rank)
}

fn card_rank_token(card: &str) -> Option<&str> {
    let t = card.trim();
    if t.starts_with("🃏") {
        return None;
    }
    for suit in ["♠", "♥", "♦", "♣"] {
        if let Some(rest) = t.strip_prefix(suit) {
            return Some(rest);
        }
    }
    Some(t)
}

/// Count bombs in hand: same-rank 4+ cards, plus wildcard-assisted bombs.
/// Wildcard (逢人配) is ONLY ♥[level] (e.g. ♥2 when level is 2).
/// Other suits of the same level are NOT wildcards – they are regular level cards.
/// This also counts straight flush candidates approximately.
fn count_bombs_in_hand(hand_cards: &[String], hand_level: Option<&str>) -> usize {
    use std::collections::HashMap;

    let mut rank_counts: HashMap<&str, usize> = HashMap::new();
    let mut wildcard_count = 0usize;
    for card in hand_cards {
        // Only ♥[level] is the wildcard, not all level cards
        if is_wildcard_symbol(card, hand_level) {
            wildcard_count += 1;
            continue;
        }
        if let Some(rank) = card_rank_token(card) {
            *rank_counts.entry(rank).or_default() += 1;
        }
    }
    // Same-rank bombs: 4+ of the same rank
    let mut bomb_count = rank_counts.values().filter(|&&c| c >= 4).count();

    // Wildcard-assisted bombs: 3 same-rank + 1 wildcard = a bomb
    if wildcard_count >= 1 {
        let assisted = rank_counts.values().filter(|&&c| c == 3).count();
        bomb_count += assisted.min(wildcard_count);
    }

    // Approximate straight flush candidates (same suit, 5+ consecutive)
    bomb_count += count_straight_flush_approx(hand_cards, hand_level);

    bomb_count
}

/// Check if a card symbol is the wildcard (逢人配): ♥[level].
fn is_wildcard_symbol(card: &str, hand_level: Option<&str>) -> bool {
    let level = match hand_level {
        Some(l) => l,
        None => return false,
    };
    card.trim() == format!("♥{}", level)
}

/// Rough estimate of straight flush candidates in hand.
/// A straight flush = 5+ consecutive cards of the same suit.
fn count_straight_flush_approx(hand_cards: &[String], hand_level: Option<&str>) -> usize {
    use std::collections::HashMap;

    let mut suit_to_ranks: HashMap<char, Vec<u8>> = HashMap::new();
    for card in hand_cards {
        // Skip wildcards
        if is_wildcard_symbol(card, hand_level) {
            continue;
        }
        let t = card.trim();
        if t.len() < 2 {
            continue;
        }
        let suit_char = t.chars().next().unwrap();
        let rank_str = &t[suit_char.len_utf8()..];
        let rank_val = match rank_str {
            "A" => 14, "K" => 13, "Q" => 12, "J" => 11, "10" => 9,
            "9" => 8, "8" => 7, "7" => 6, "6" => 5, "5" => 4, "4" => 3, "3" => 2, "2" => 1,
            _ => continue,
        };
        suit_to_ranks.entry(suit_char).or_default().push(rank_val);
    }

    let mut count = 0;
    for (_, mut ranks) in suit_to_ranks {
        ranks.sort();
        ranks.dedup();
        if ranks.len() < 5 {
            continue;
        }
        let mut run_len = 1;
        for i in 1..ranks.len() {
            if ranks[i] == ranks[i - 1] + 1 {
                run_len += 1;
                if run_len >= 5 {
                    count += 1;
                    run_len = 0;
                }
            } else {
                run_len = 1;
            }
        }
    }
    count
}

/// Count single cards with rank 2-12 (excluding level cards).
/// Used for teammate follow strategy: when teammate plays a small single,
/// play your larger single (up to Q) to reduce hand count.
fn count_medium_singles(hand_cards: &[String], hand_level: Option<&str>) -> usize {
    use std::collections::HashMap;
    let mut rank_counts: HashMap<&str, usize> = HashMap::new();
    for card in hand_cards {
        if let Some(rank) = card_rank_token(card) {
            let is_medium = matches!(rank, "2"|"3"|"4"|"5"|"6"|"7"|"8"|"9"|"10"|"J"|"Q");
            if !is_medium {
                continue;
            }
            if hand_level == Some(rank) {
                continue; // exclude level cards
            }
            *rank_counts.entry(rank).or_default() += 1;
        }
    }
    // Count ranks that have at least one card (can play as single)
    rank_counts.values().filter(|&&c| c >= 1).count()
}

/// Count ranks that have 2+ cards of rank 2-9 (excluding level cards).
/// Used for teammate pair follow strategy.
fn count_small_pair_ranks(hand_cards: &[String], hand_level: Option<&str>) -> usize {
    use std::collections::HashMap;
    let mut rank_counts: HashMap<&str, usize> = HashMap::new();
    for card in hand_cards {
        if let Some(rank) = card_rank_token(card) {
            let is_small = matches!(rank, "2"|"3"|"4"|"5"|"6"|"7"|"8"|"9");
            if !is_small {
                continue;
            }
            if hand_level == Some(rank) {
                continue;
            }
            *rank_counts.entry(rank).or_default() += 1;
        }
    }
    rank_counts.values().filter(|&&c| c >= 2).count()
}

/// Count ranks that have 3+ cards of rank 2-9 (excluding level cards).
/// Used for teammate triple follow strategy.
fn count_small_triple_ranks(hand_cards: &[String], hand_level: Option<&str>) -> usize {
    use std::collections::HashMap;
    let mut rank_counts: HashMap<&str, usize> = HashMap::new();
    for card in hand_cards {
        if let Some(rank) = card_rank_token(card) {
            let is_small = matches!(rank, "2"|"3"|"4"|"5"|"6"|"7"|"8"|"9");
            if !is_small {
                continue;
            }
            if hand_level == Some(rank) {
                continue;
            }
            *rank_counts.entry(rank).or_default() += 1;
        }
    }
    rank_counts.values().filter(|&&c| c >= 3).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_card_range_is_two_to_ten() {
        assert!(is_small_card_symbol("♠2", Some("A")));
        assert!(is_small_card_symbol("♥10", Some("A")));
        assert!(!is_small_card_symbol("♣J", Some("A")));
        assert!(!is_small_card_symbol("🃏R", Some("A")));
    }

    #[test]
    fn small_card_excludes_current_hand_level() {
        assert!(!is_small_card_symbol("♠2", Some("2")));
        assert!(!is_small_card_symbol("♥10", Some("10")));
        assert!(is_small_card_symbol("♦9", Some("10")));
    }

    #[test]
    fn medium_singles_count_excludes_level_and_high_cards() {
        let hand = vec![
            "♠3".to_string(), "♥5".to_string(), "♣J".to_string(), "♦Q".to_string(),
            "♠K".to_string(), "♥A".to_string(), "♠2".to_string(),
        ];
        // 3,5,J,Q are medium (2-12), K and A excluded, 2 is level
        assert_eq!(count_medium_singles(&hand, Some("2")), 4);
    }

    #[test]
    fn small_pair_ranks_count() {
        let hand = vec![
            "♠3", "♥3", "♠5", "♥5", "♣5", "♠7", "♠9", "♥9", "♠J", "♠K",
        ];
        let hand: Vec<String> = hand.into_iter().map(|s| s.to_string()).collect();
        // Ranks 3,5,9 have 2+ cards; 5 has 3 cards (still counts as 1 pair rank); 7, J, K have 1
        assert_eq!(count_small_pair_ranks(&hand, Some("2")), 3);
    }

    #[test]
    fn small_triple_ranks_count() {
        let hand = vec![
            "♠3", "♥3", "♦3", "♠5", "♥5", "♠7", "♠9", "♥9", "♦9", "♠K",
        ];
        let hand: Vec<String> = hand.into_iter().map(|s| s.to_string()).collect();
        // Ranks 3 and 9 have 3+ cards; 5 has 2; 7, K have 1
        assert_eq!(count_small_triple_ranks(&hand, Some("2")), 2);
    }
}
