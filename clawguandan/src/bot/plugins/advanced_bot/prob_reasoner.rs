use crate::domain::Seat;
use crate::game::card::{Card, Rank};
use super::hand_tracker::HandTracker;

#[derive(Clone, Debug)]
pub struct ProbabilisticReasoner {
    tracker: HandTracker,
}

impl ProbabilisticReasoner {
    pub fn new(tracker: HandTracker) -> Self {
        Self { tracker }
    }

    pub fn tracker(&self) -> &HandTracker {
        &self.tracker
    }

    pub fn tracker_mut(&mut self) -> &mut HandTracker {
        &mut self.tracker
    }

    pub fn calculate_opponent_bomb_prob(&self, my_seat: Seat) -> f32 {
        let mut max_prob: f32 = 0.0;
        for seat in Seat::ALL {
            if seat != my_seat && seat != my_seat.teammate() {
                let prob = self.tracker.get_prob_has_bomb(seat);
                max_prob = max_prob.max(prob);
            }
        }
        max_prob
    }

    pub fn calculate_partner_needs(&self, my_seat: Seat) -> Vec<(Rank, f32)> {
        let partner = my_seat.teammate();
        let partner_remaining = self.tracker.get_remaining_count(partner);
        if partner_remaining == 0 {
            return Vec::new();
        }

        let mut needs = Vec::new();
        let ranks = [Rank::Three, Rank::Four, Rank::Five, Rank::Six, Rank::Seven,
                     Rank::Eight, Rank::Nine, Rank::Ten, Rank::J, Rank::Q, Rank::K, Rank::A,
                     Rank::Two];

        for rank in ranks {
            let prob_partner_has = self.tracker.get_prob_rank_in_hand(partner, rank);
            let prob_opponent_has = self.calculate_opponent_has_rank(rank);
            let need_score = (1.0 - prob_partner_has) * (1.0 - prob_opponent_has);
            if need_score > 0.05 {
                needs.push((rank, need_score));
            }
        }

        needs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        needs
    }

    pub fn calculate_opponent_has_rank(&self, rank: Rank) -> f32 {
        let mut total_prob = 0.0;
        for seat in Seat::ALL {
            total_prob += self.tracker.get_prob_rank_in_hand(seat, rank);
        }
        total_prob.min(1.0)
    }

    pub fn calculate_win_prob(&self, my_seat: Seat, my_cards: &[Card]) -> f32 {
        if my_cards.is_empty() {
            return 0.0;
        }

        let max_rank = my_cards.iter().map(|c| rank_value(c.rank)).max().unwrap_or(0);
        let mut prob_no_opponent_can_follow = 1.0;

        for seat in Seat::ALL {
            if seat != my_seat && seat != my_seat.teammate() {
                let min_rank = rank_from_value(max_rank);
                let prob_can_follow = self.tracker.get_prob_can_follow(seat, min_rank);
                prob_no_opponent_can_follow *= (1.0 - prob_can_follow);
            }
        }

        let my_remaining = self.tracker.get_remaining_count(my_seat);
        let team_remaining = self.tracker.get_team_remaining(my_seat);
        let enemy_remaining = self.tracker.get_enemy_remaining(my_seat);

        let hand_progress = 1.0 - (my_remaining as f32 / 25.0);
        let team_advantage = (team_remaining as f32 / (team_remaining + enemy_remaining) as f32).max(0.5);

        prob_no_opponent_can_follow * 0.6 + hand_progress * 0.2 + team_advantage * 0.2
    }

    pub fn calculate_game_win_prob(&self, my_seat: Seat) -> f32 {
        let my_remaining = self.tracker.get_remaining_count(my_seat);
        let partner_remaining = self.tracker.get_remaining_count(my_seat.teammate());
        let team_remaining = my_remaining + partner_remaining;

        let mut enemy_remaining = 0;
        for seat in Seat::ALL {
            if seat != my_seat && seat != my_seat.teammate() {
                enemy_remaining += self.tracker.get_remaining_count(seat);
            }
        }

        let total_cards = team_remaining + enemy_remaining;
        if total_cards == 0 {
            return 0.5;
        }

        let progress_ratio = enemy_remaining as f32 / total_cards as f32;
        let advantage = 1.0 - progress_ratio;

        let bomb_factor = self.calculate_opponent_bomb_prob(my_seat);
        let adjusted = advantage * (1.0 - bomb_factor * 0.3);

        adjusted.clamp(0.0, 1.0)
    }

    pub fn is_partner_sprinting(&self, my_seat: Seat, threshold: u8) -> bool {
        let partner = my_seat.teammate();
        self.tracker.get_remaining_count(partner) <= threshold as usize
    }

    pub fn is_any_enemy_sprinting(&self, my_seat: Seat, threshold: u8) -> bool {
        for seat in Seat::ALL {
            if seat != my_seat && seat != my_seat.teammate() {
                if self.tracker.get_remaining_count(seat) <= threshold as usize {
                    return true;
                }
            }
        }
        false
    }

    pub fn get_strongest_card_prob(&self, seat: Seat) -> Option<(Rank, f32)> {
        let ranks = [Rank::RedJoker, Rank::BlackJoker, Rank::Two, Rank::A, Rank::K, Rank::Q];
        for rank in ranks {
            let prob = self.tracker.get_prob_rank_in_hand(seat, rank);
            if prob > 0.1 {
                return Some((rank, prob));
            }
        }
        None
    }
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

fn rank_from_value(v: u8) -> Rank {
    match v {
        3 => Rank::Three,
        4 => Rank::Four,
        5 => Rank::Five,
        6 => Rank::Six,
        7 => Rank::Seven,
        8 => Rank::Eight,
        9 => Rank::Nine,
        10 => Rank::Ten,
        11 => Rank::J,
        12 => Rank::Q,
        13 => Rank::K,
        14 => Rank::A,
        15 => Rank::Two,
        16 => Rank::BlackJoker,
        17 => Rank::RedJoker,
        _ => Rank::Three,
    }
}
