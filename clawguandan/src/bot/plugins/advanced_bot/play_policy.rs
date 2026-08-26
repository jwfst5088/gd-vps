use std::sync::{Arc, Mutex};

use crate::bot::plugin::{BotDecision, BotTurnContext};
use crate::bot::policies::PlayPolicy;
use crate::domain::Seat;
use crate::game::card::{Card, Rank};
use crate::game::engine::PlayerAction;

use super::hand_tracker::HandTracker;
use super::params::AdvancedBotParams;
use super::prob_reasoner::ProbabilisticReasoner;

#[derive(Debug)]
pub struct AdvancedPlayPolicy {
    pub params: Arc<AdvancedBotParams>,
    tracker: Arc<Mutex<HandTracker>>,
}

impl AdvancedPlayPolicy {
    pub fn new(params: Arc<AdvancedBotParams>) -> Self {
        Self {
            params,
            tracker: Arc::new(Mutex::new(HandTracker::new(Seat::E))),
        }
    }
}

impl PlayPolicy for AdvancedPlayPolicy {
    fn decide_play(&self, ctx: &BotTurnContext) -> Result<BotDecision, String> {
        let state = &ctx.state;
        let legal_actions: Vec<String> = state["expect"]["legalActions"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
            .unwrap_or_default();

        let can_pass = legal_actions.contains(&"pass".to_string());
        let can_play = legal_actions.contains(&"play".to_string());

        if can_pass && !can_play {
            return Ok(BotDecision::Action(PlayerAction::Pass));
        }
        if can_play && !can_pass {
            return Ok(BotDecision::UseSuggest);
        }

        let my_seat: Seat = state["private"]["seat"]
            .as_str()
            .and_then(|s| match s {
                "E" => Some(Seat::E),
                "S" => Some(Seat::S),
                "W" => Some(Seat::W),
                "N" => Some(Seat::N),
                _ => None,
            })
            .unwrap_or(Seat::E);

        let teammate_seat: Seat = state["private"]["teammateSeat"]
            .as_str()
            .and_then(|s| match s {
                "E" => Some(Seat::E),
                "S" => Some(Seat::S),
                "W" => Some(Seat::W),
                "N" => Some(Seat::N),
                _ => None,
            })
            .unwrap_or(Seat::W);

        let top_play_seat: Option<Seat> = state["hand"]["topPlay"]["seat"]
            .as_str()
            .and_then(|s| match s {
                "E" => Some(Seat::E),
                "S" => Some(Seat::S),
                "W" => Some(Seat::W),
                "N" => Some(Seat::N),
                _ => None,
            });

        let my_hand: Vec<Card> = state["private"]["handCards"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| parse_card(v.as_str())).collect())
            .unwrap_or_default();

        let level_str = state["hand"]["handLevel"].as_str().unwrap_or("2");
        let level_rank = parse_rank(level_str);

        let mut tracker = self.tracker.lock().unwrap();

        if !tracker.is_initialized() || tracker.get_remaining_count(my_seat) != my_hand.len() {
            tracker.reset();
            tracker.set_my_seat(my_seat);
            tracker.init(&my_hand, level_rank);
        }

        for seat in Seat::ALL {
            if let Some(count) = state["seats"][seat.as_str()]["remainingCount"].as_u64() {
                tracker.update_seat_count(seat, count as usize);
            }
        }

        let top_play_cards: Vec<Card> = state["hand"]["topPlay"]["cards"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| parse_card(v.as_str())).collect())
            .unwrap_or_default();

        if let Some(tps) = top_play_seat {
            let already_recorded = tracker.played_cards.iter()
                .any(|(seat, cards)| *seat == tps && *cards == top_play_cards);
            if !already_recorded && !top_play_cards.is_empty() {
                tracker.update_play(tps, &top_play_cards);
            }
        }

        let tracker_clone = (*tracker).clone();
        drop(tracker);
        let reasoner = ProbabilisticReasoner::new(tracker_clone);

        // ── 房规（mirror gd.rmyy.nyc.mn decideAdvancedPlay「只剩1张必打」）──────
        // 只剩最后1张且能出牌时，立即交由 suggest 强制打出清空夺头游；
        // 任何让牌启发式（压队友大牌/疑似炸弹/队友冲刺/胜率保守等）不得拦截。
        if can_pass && can_play && my_hand.len() == 1 {
            return Ok(BotDecision::UseSuggest);
        }

        let partner_leading = top_play_seat == Some(teammate_seat);
        let enemy_leading = top_play_seat.is_some() && !partner_leading;

        let mut pass_score = 0.0;
        let mut play_score = 0.0;
        let mut reasons = Vec::new();

        if partner_leading {
            let top_is_bomb = is_bomb_play(&state["hand"]["topPlay"]);
            if top_is_bomb {
                pass_score += 10.0 * self.params.team_win_weight;
                reasons.push("pass: NEVER override teammate's bomb".to_string());
                return Ok(BotDecision::Action(PlayerAction::Pass));
            }

            let top_rank = extract_top_rank(&state["hand"]["topPlay"]);
            let top_is_level = top_rank == level_rank;
            if top_is_level {
                pass_score += 10.0 * self.params.team_win_weight;
                reasons.push("pass: NEVER override teammate's level card".to_string());
                return Ok(BotDecision::Action(PlayerAction::Pass));
            }

            let top_is_joker = top_rank.map(|r| matches!(r, Rank::BlackJoker | Rank::RedJoker)).unwrap_or(false)
                || is_joker_play(&state["hand"]["topPlay"]);
            if top_is_joker {
                pass_score += 10.0 * self.params.team_win_weight;
                reasons.push("pass: NEVER override teammate's joker".to_string());
                return Ok(BotDecision::Action(PlayerAction::Pass));
            }

            let top_is_big = top_rank.map(|r| rank_value(r) >= 12).unwrap_or(false);
            if top_is_big {
                pass_score += 5.0 * self.params.team_win_weight;
                reasons.push("pass: do not override teammate's big card".to_string());
                return Ok(BotDecision::Action(PlayerAction::Pass));
            }

            let prob_opponent_can_follow = if let Some(tr) = top_rank {
                reasoner.calculate_opponent_has_rank(tr)
            } else {
                0.0
            };

            if prob_opponent_can_follow < self.params.prob_threshold_for_intercept {
                play_score += 2.0 * self.params.low_card_dump_bias;
                reasons.push("play: safe to follow partner".to_string());
            } else {
                pass_score += 1.0 * self.params.yield_to_partner_bias;
                reasons.push("pass: opponent likely can follow".to_string());
            }
        } else if enemy_leading {
            // Default incentive to follow opponent's lead (try to play rather than pass)
            // 2.0 beats bomb_conserve_bias (2.0*0.8=1.6) and other pass incentives
            play_score += 2.0;

            let prob_opponent_has_bomb = reasoner.calculate_opponent_bomb_prob(my_seat);

            if prob_opponent_has_bomb > self.params.prob_threshold_for_bomb {
                pass_score += 2.0 * self.params.bomb_conserve_bias;
                reasons.push("pass: opponent likely has bomb".to_string());
            }

            let enemy_low = reasoner.is_any_enemy_sprinting(my_seat, self.params.enemy_low_cards_threshold);
            if enemy_low {
                play_score += self.params.bomb_aggression_when_enemy_low;
                reasons.push("play: enemy sprinting, need to intercept".to_string());
            }

            let my_remaining = reasoner.tracker().get_remaining_count(my_seat);
            let is_endgame = my_remaining <= self.params.endgame_hand_count_threshold as usize;
            if is_endgame {
                play_score += self.params.endgame_clear_hand_bias;
                reasons.push("play: endgame, need to clear".to_string());
            }
        } else {
            play_score += self.params.proactive_play_bias;
            reasons.push("play: leading, be proactive".to_string());
        }

        let partner_sprinting = reasoner.is_partner_sprinting(my_seat, self.params.partner_sprint_threshold);
        if partner_sprinting && !enemy_leading {
            pass_score += 2.0 * self.params.second_out_weight;
            reasons.push("pass: teammate sprinting, yield".to_string());
        }

        let game_win_prob = reasoner.calculate_game_win_prob(my_seat);
        if game_win_prob > 0.7 && !enemy_leading {
            pass_score += 1.0;
            reasons.push("pass: team advantage, conserve cards".to_string());
        } else if game_win_prob < 0.3 {
            play_score += 1.5;
            reasons.push("play: team disadvantage, need to take control".to_string());
        }

        if self.params.enable_reason_trace {
            eprintln!(
                "[advanced-bot] pass_score={:.2} play_score={:.2} reasons={:?}",
                pass_score, play_score, reasons
            );
        }

        if pass_score > play_score {
            Ok(BotDecision::Action(PlayerAction::Pass))
        } else {
            Ok(BotDecision::UseSuggest)
        }
    }
}

fn parse_card(s: Option<&str>) -> Option<Card> {
    let s = s?;
    let chars: Vec<char> = s.chars().collect();
    if chars.is_empty() {
        return None;
    }

    let suit = match chars[0] {
        '♠' => crate::game::card::Suit::Spades,
        '♥' => crate::game::card::Suit::Hearts,
        '♦' => crate::game::card::Suit::Diamonds,
        '♣' => crate::game::card::Suit::Clubs,
        '🃏' => crate::game::card::Suit::Joker,
        _ => return None,
    };

    let rank_str: String = chars[1..].iter().collect();
    let rank = parse_rank(&rank_str)?;

    Some(Card { suit, rank })
}

fn parse_rank(s: &str) -> Option<Rank> {
    match s {
        "3" => Some(Rank::Three),
        "4" => Some(Rank::Four),
        "5" => Some(Rank::Five),
        "6" => Some(Rank::Six),
        "7" => Some(Rank::Seven),
        "8" => Some(Rank::Eight),
        "9" => Some(Rank::Nine),
        "10" => Some(Rank::Ten),
        "J" => Some(Rank::J),
        "Q" => Some(Rank::Q),
        "K" => Some(Rank::K),
        "A" => Some(Rank::A),
        "2" => Some(Rank::Two),
        "BJ" => Some(Rank::BlackJoker),
        "RJ" => Some(Rank::RedJoker),
        "R" => Some(Rank::RedJoker),   // Game engine format: 🃏R
        "b" => Some(Rank::BlackJoker), // Game engine format: 🃏b
        _ => None,
    }
}

fn is_bomb_play(top_play: &serde_json::Value) -> bool {
    top_play["kind"].as_str().map(|k| k.starts_with("Bomb")).unwrap_or(false)
}

fn is_joker_play(top_play: &serde_json::Value) -> bool {
    top_play["cards"].as_array().map(|cards| {
        cards.iter().any(|c| c.as_str().map(|s| s.contains('🃏')).unwrap_or(false))
    }).unwrap_or(false)
}

fn extract_top_rank(top_play: &serde_json::Value) -> Option<Rank> {
    top_play["cards"].as_array().and_then(|cards| {
        cards.first().and_then(|c| c.as_str().and_then(|s| parse_card(Some(s)))).map(|c| c.rank)
    })
}

fn rank_value(r: Rank) -> u8 {
    match r {
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
        Rank::Two => 15,
        Rank::BlackJoker => 16,
        Rank::RedJoker => 17,
    }
}
