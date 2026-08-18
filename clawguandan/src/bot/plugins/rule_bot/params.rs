use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuleBotParams {
    pub team_win_weight: f32,
    pub first_out_weight: f32,
    pub second_out_weight: f32,
    pub yield_to_partner_bias: f32,
    pub partner_support_threshold: u8,
    /// Teammate hand count ≤ this → "sprinting" mode (aggressive support).
    pub partner_sprint_threshold: u8,
    pub bomb_conserve_bias: f32,
    pub bomb_aggression_when_enemy_low_cards: f32,
    pub enemy_low_cards_threshold: u8,
    pub endgame_hand_count_threshold: u8,
    pub endgame_clear_hand_bias: f32,
    pub proactive_play_bias: f32,
    pub low_card_dump_bias: f32,
    pub pass_stall_penalty: f32,
    pub use_suggest_fallback: bool,
    pub enable_reason_trace: bool,
}

impl RuleBotParams {
    pub fn default_balanced() -> Self {
        Self {
            team_win_weight: 1.0,
            first_out_weight: 0.8,
            second_out_weight: 0.9,
            yield_to_partner_bias: 1.4,
            partner_support_threshold: 3,
            partner_sprint_threshold: 3,
            bomb_conserve_bias: 0.8,
            bomb_aggression_when_enemy_low_cards: 2.2,
            enemy_low_cards_threshold: 3,
            endgame_hand_count_threshold: 6,
            endgame_clear_hand_bias: 1.2,
            proactive_play_bias: 1.1,
            low_card_dump_bias: 1.4,
            pass_stall_penalty: 0.9,
            use_suggest_fallback: true,
            enable_reason_trace: false,
        }
    }

    pub fn default_aggressive() -> Self {
        Self {
            team_win_weight: 0.9,
            first_out_weight: 1.4,
            second_out_weight: 0.7,
            yield_to_partner_bias: 0.6,
            partner_support_threshold: 2,
            partner_sprint_threshold: 2,
            bomb_conserve_bias: 0.3,
            bomb_aggression_when_enemy_low_cards: 2.8,
            enemy_low_cards_threshold: 4,
            endgame_hand_count_threshold: 8,
            endgame_clear_hand_bias: 2.0,
            proactive_play_bias: 1.6,
            low_card_dump_bias: 1.1,
            pass_stall_penalty: 1.2,
            use_suggest_fallback: true,
            enable_reason_trace: false,
        }
    }

    pub fn default_supportive() -> Self {
        Self {
            team_win_weight: 1.4,
            first_out_weight: 0.7,
            second_out_weight: 1.3,
            yield_to_partner_bias: 2.2,
            partner_support_threshold: 4,
            partner_sprint_threshold: 4,
            bomb_conserve_bias: 1.1,
            bomb_aggression_when_enemy_low_cards: 1.6,
            enemy_low_cards_threshold: 3,
            endgame_hand_count_threshold: 6,
            endgame_clear_hand_bias: 1.0,
            proactive_play_bias: 0.6,
            low_card_dump_bias: 1.2,
            pass_stall_penalty: 0.5,
            use_suggest_fallback: true,
            enable_reason_trace: false,
        }
    }

    /// Load parameters from a JSON file.
    pub fn load(path: &str) -> Result<Self, String> {
        let raw = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str(&raw).map_err(|e| format!("parse params: {e}"))
    }

    /// Save parameters to a JSON file.
    pub fn save(&self, path: &str) -> Result<(), String> {
        let dir = std::path::Path::new(path)
            .parent()
            .unwrap_or(std::path::Path::new("."));
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        let s = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, s).map_err(|e| e.to_string())
    }

    /// Randomly mutate one float parameter by ±step_size (clamped to [0.1, 10.0]).
    /// Returns a new mutated copy.
    pub fn mutate_random(&self, step_size: f32) -> Self {
        let mut rng = rand::rng();
        let mut cloned = self.clone();
        // Pick a random float field to mutate
        let idx = rng.random_range(0..12);
        let val = match idx {
            0 => &mut cloned.team_win_weight,
            1 => &mut cloned.first_out_weight,
            2 => &mut cloned.second_out_weight,
            3 => &mut cloned.yield_to_partner_bias,
            4 => &mut cloned.bomb_conserve_bias,
            5 => &mut cloned.bomb_aggression_when_enemy_low_cards,
            6 => &mut cloned.endgame_clear_hand_bias,
            7 => &mut cloned.proactive_play_bias,
            8 => &mut cloned.low_card_dump_bias,
            9 => &mut cloned.pass_stall_penalty,
            10 => {
                // Mutate an integer field
                let ii = rng.random_range(0..3);
                match ii {
                    0 => cloned.partner_support_threshold = mutate_u8(cloned.partner_support_threshold, 1, 8),
                    1 => cloned.partner_sprint_threshold = mutate_u8(cloned.partner_sprint_threshold, 1, 6),
                    _ => cloned.enemy_low_cards_threshold = mutate_u8(cloned.enemy_low_cards_threshold, 1, 8),
                }
                return cloned;
            }
            _ => {
                cloned.endgame_hand_count_threshold = mutate_u8(cloned.endgame_hand_count_threshold, 3, 12);
                return cloned;
            }
        };
        let delta = rng.random_range(-step_size..step_size);
        *val = (*val + delta).clamp(0.1, 10.0);
        cloned
    }
}

fn mutate_u8(current: u8, min: u8, max: u8) -> u8 {
    let mut rng = rand::rng();
    let delta: i8 = if rng.random_bool(0.5) { 1 } else { -1 };
    let new = current as i16 + delta as i16;
    new.clamp(min as i16, max as i16) as u8
}

impl Default for RuleBotParams {
    fn default() -> Self {
        Self::default_balanced()
    }
}