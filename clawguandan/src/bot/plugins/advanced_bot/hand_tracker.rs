use crate::domain::Seat;
use crate::game::card::{Card, Rank, Suit};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
pub struct HandTracker {
    pub played_cards: Vec<(Seat, Vec<Card>)>,
    pub known_cards: HashSet<Card>,
    pub remaining_card_pool: Vec<Card>,
    pub hand_level: Option<Rank>,
    pub seat_remaining_counts: HashMap<Seat, usize>,
    my_seat: Seat,
}

impl HandTracker {
    pub fn new(my_seat: Seat) -> Self {
        Self {
            played_cards: Vec::new(),
            known_cards: HashSet::new(),
            remaining_card_pool: Vec::new(),
            hand_level: None,
            seat_remaining_counts: Seat::ALL.iter().map(|s| (*s, 25)).collect(),
            my_seat,
        }
    }

    pub fn is_initialized(&self) -> bool {
        !self.known_cards.is_empty()
    }

    pub fn reset(&mut self) {
        self.played_cards.clear();
        self.known_cards.clear();
        self.remaining_card_pool.clear();
        self.hand_level = None;
        self.seat_remaining_counts.clear();
        for seat in Seat::ALL {
            self.seat_remaining_counts.insert(seat, 25);
        }
    }

    pub fn set_my_seat(&mut self, seat: Seat) {
        self.my_seat = seat;
    }

    pub fn init(&mut self, my_hand: &[Card], level_rank: Option<Rank>) {
        self.hand_level = level_rank;
        for card in my_hand {
            self.known_cards.insert(*card);
        }
        let all_cards = generate_all_cards();
        self.remaining_card_pool = all_cards
            .into_iter()
            .filter(|c| !self.known_cards.contains(c))
            .collect();
    }

    pub fn update_play(&mut self, seat: Seat, cards: &[Card]) {
        self.played_cards.push((seat, cards.to_vec()));
        for card in cards {
            self.known_cards.insert(*card);
            if let Some(idx) = self.remaining_card_pool.iter().position(|c| c == card) {
                self.remaining_card_pool.remove(idx);
            }
        }
        if let Some(count) = self.seat_remaining_counts.get_mut(&seat) {
            *count = (*count).saturating_sub(cards.len());
        }
    }

    pub fn update_pass(&mut self, seat: Seat) {
        if let Some(count) = self.seat_remaining_counts.get_mut(&seat) {
            *count = (*count).saturating_sub(0);
        }
    }

    pub fn update_seat_count(&mut self, seat: Seat, count: usize) {
        self.seat_remaining_counts.insert(seat, count);
    }

    pub fn get_remaining_count(&self, seat: Seat) -> usize {
        *self.seat_remaining_counts.get(&seat).unwrap_or(&0)
    }

    pub fn get_prob_rank_in_hand(&self, seat: Seat, rank: Rank) -> f32 {
        if seat == self.my_seat {
            return 0.0;
        }
        let total_remaining = self.get_remaining_count(seat);
        if total_remaining == 0 {
            return 0.0;
        }
        let remaining_in_pool = self.remaining_card_pool
            .iter()
            .filter(|c| c.rank == rank)
            .count();
        let total_unknown = self.remaining_card_pool.len();
        if total_unknown == 0 {
            return 0.0;
        }
        (remaining_in_pool as f32 / total_unknown as f32) * (total_remaining as f32 / total_unknown as f32).min(1.0)
    }

    pub fn get_prob_has_bomb(&self, seat: Seat) -> f32 {
        if seat == self.my_seat {
            return 0.0;
        }
        let total_remaining = self.get_remaining_count(seat);
        if total_remaining < 4 {
            return 0.0;
        }
        let bomb_ranks = [Rank::Two, Rank::Three, Rank::Four, Rank::Five,
                          Rank::Six, Rank::Seven, Rank::Eight, Rank::Nine,
                          Rank::Ten, Rank::J, Rank::Q, Rank::K, Rank::A];
        let mut prob: f32 = 0.0;
        for rank in bomb_ranks {
            let count_in_pool = self.remaining_card_pool
                .iter()
                .filter(|c| c.rank == rank)
                .count();
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

    pub fn get_prob_can_follow(&self, seat: Seat, min_rank: Rank) -> f32 {
        if seat == self.my_seat {
            return 0.0;
        }
        let total_remaining = self.get_remaining_count(seat);
        if total_remaining == 0 {
            return 0.0;
        }
        let higher_in_pool = self.remaining_card_pool
            .iter()
            .filter(|c| rank_ge(c.rank, min_rank))
            .count();
        let total_unknown = self.remaining_card_pool.len();
        if total_unknown == 0 {
            return 0.0;
        }
        (higher_in_pool as f32 / total_unknown as f32).min(1.0)
    }

    pub fn get_team_remaining(&self, seat: Seat) -> usize {
        let teammate = seat.teammate();
        self.get_remaining_count(seat) + self.get_remaining_count(teammate)
    }

    pub fn get_enemy_remaining(&self, seat: Seat) -> usize {
        Seat::ALL.iter()
            .filter(|s| **s != seat && **s != seat.teammate())
            .map(|s| self.get_remaining_count(*s))
            .sum()
    }

    pub fn get_remaining_pool_size(&self) -> usize {
        self.remaining_card_pool.len()
    }

    pub fn is_endgame(&self, threshold: u8) -> bool {
        self.seat_remaining_counts.values().any(|&c| c <= threshold as usize)
    }
}

fn generate_all_cards() -> Vec<Card> {
    let mut cards = Vec::new();
    let suits = [Suit::Spades, Suit::Hearts, Suit::Diamonds, Suit::Clubs];
    let ranks = [Rank::Three, Rank::Four, Rank::Five, Rank::Six, Rank::Seven,
                 Rank::Eight, Rank::Nine, Rank::Ten, Rank::J, Rank::Q, Rank::K, Rank::A,
                 Rank::Two];
    for suit in suits.iter() {
        for rank in ranks.iter() {
            cards.push(Card { suit: *suit, rank: *rank });
        }
    }
    cards.push(Card { suit: Suit::Joker, rank: Rank::BlackJoker });
    cards.push(Card { suit: Suit::Joker, rank: Rank::RedJoker });
    cards
}

fn rank_ge(a: Rank, b: Rank) -> bool {
    rank_value(a) >= rank_value(b)
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
