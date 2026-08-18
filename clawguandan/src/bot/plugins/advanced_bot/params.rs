use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdvancedBotParams {
    pub team_win_weight: f32,
    pub first_out_weight: f32,
    pub second_out_weight: f32,
    pub yield_to_partner_bias: f32,
    pub partner_sprint_threshold: u8,
    pub bomb_conserve_bias: f32,
    pub bomb_aggression_when_enemy_low: f32,
    pub enemy_low_cards_threshold: u8,
    pub endgame_hand_count_threshold: u8,
    pub endgame_clear_hand_bias: f32,
    pub proactive_play_bias: f32,
    pub low_card_dump_bias: f32,
    pub pass_stall_penalty: f32,
    pub hand_tracker_enabled: bool,
    pub prob_threshold_for_bomb: f32,
    pub prob_threshold_for_intercept: f32,
    pub enable_reason_trace: bool,
}

impl AdvancedBotParams {
    pub fn default_balanced() -> Self {
        Self {
            team_win_weight: 1.0,
            first_out_weight: 0.8,
            second_out_weight: 0.9,
            yield_to_partner_bias: 1.4,
            partner_sprint_threshold: 3,
            bomb_conserve_bias: 0.8,
            bomb_aggression_when_enemy_low: 2.2,
            enemy_low_cards_threshold: 3,
            endgame_hand_count_threshold: 6,
            endgame_clear_hand_bias: 1.2,
            proactive_play_bias: 1.1,
            low_card_dump_bias: 1.4,
            pass_stall_penalty: 0.9,
            hand_tracker_enabled: true,
            prob_threshold_for_bomb: 0.6,
            prob_threshold_for_intercept: 0.4,
            enable_reason_trace: false,
        }
    }

    pub fn default_aggressive() -> Self {
        Self {
            team_win_weight: 0.9,
            first_out_weight: 1.4,
            second_out_weight: 0.7,
            yield_to_partner_bias: 0.6,
            partner_sprint_threshold: 2,
            bomb_conserve_bias: 0.3,
            bomb_aggression_when_enemy_low: 2.8,
            enemy_low_cards_threshold: 4,
            endgame_hand_count_threshold: 8,
            endgame_clear_hand_bias: 2.0,
            proactive_play_bias: 1.6,
            low_card_dump_bias: 1.1,
            pass_stall_penalty: 1.2,
            hand_tracker_enabled: true,
            prob_threshold_for_bomb: 0.5,
            prob_threshold_for_intercept: 0.3,
            enable_reason_trace: false,
        }
    }

    pub fn default_supportive() -> Self {
        Self {
            team_win_weight: 1.4,
            first_out_weight: 0.7,
            second_out_weight: 1.3,
            yield_to_partner_bias: 2.2,
            partner_sprint_threshold: 4,
            bomb_conserve_bias: 1.1,
            bomb_aggression_when_enemy_low: 1.6,
            enemy_low_cards_threshold: 3,
            endgame_hand_count_threshold: 6,
            endgame_clear_hand_bias: 1.0,
            proactive_play_bias: 0.6,
            low_card_dump_bias: 1.2,
            pass_stall_penalty: 0.5,
            hand_tracker_enabled: true,
            prob_threshold_for_bomb: 0.7,
            prob_threshold_for_intercept: 0.5,
            enable_reason_trace: false,
        }
    }

    pub fn load(path: &str) -> Result<Self, String> {
        let raw = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str(&raw).map_err(|e| format!("parse params: {e}"))
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        let dir = std::path::Path::new(path)
            .parent()
            .unwrap_or(std::path::Path::new("."));
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        let s = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, s).map_err(|e| e.to_string())
    }

    pub fn mutate_random(&self, step_size: f32) -> Self {
        let mut rng = rand::rng();
        let mut cloned = self.clone();
        // 15 个可变异参数（排除 hand_tracker_enabled 和 enable_reason_trace 两个布尔开关）
        let idx = rng.random_range(0..15);
        match idx {
            // f32 参数：加随机浮点扰动，clamp 到合理范围
            0 => cloned.team_win_weight = (cloned.team_win_weight + rng.random_range(-step_size..step_size)).clamp(0.1, 10.0),
            1 => cloned.first_out_weight = (cloned.first_out_weight + rng.random_range(-step_size..step_size)).clamp(0.1, 10.0),
            2 => cloned.second_out_weight = (cloned.second_out_weight + rng.random_range(-step_size..step_size)).clamp(0.1, 10.0),
            3 => cloned.yield_to_partner_bias = (cloned.yield_to_partner_bias + rng.random_range(-step_size..step_size)).clamp(0.1, 10.0),
            4 => cloned.bomb_conserve_bias = (cloned.bomb_conserve_bias + rng.random_range(-step_size..step_size)).clamp(0.1, 10.0),
            5 => cloned.bomb_aggression_when_enemy_low = (cloned.bomb_aggression_when_enemy_low + rng.random_range(-step_size..step_size)).clamp(0.1, 10.0),
            6 => cloned.endgame_clear_hand_bias = (cloned.endgame_clear_hand_bias + rng.random_range(-step_size..step_size)).clamp(0.1, 10.0),
            7 => cloned.proactive_play_bias = (cloned.proactive_play_bias + rng.random_range(-step_size..step_size)).clamp(0.1, 10.0),
            8 => cloned.low_card_dump_bias = (cloned.low_card_dump_bias + rng.random_range(-step_size..step_size)).clamp(0.1, 10.0),
            9 => cloned.pass_stall_penalty = (cloned.pass_stall_penalty + rng.random_range(-step_size..step_size)).clamp(0.1, 10.0),
            // 概率阈值参数：clamp 到 [0.1, 1.0]
            10 => cloned.prob_threshold_for_bomb = (cloned.prob_threshold_for_bomb + rng.random_range(-step_size..step_size)).clamp(0.1, 1.0),
            11 => cloned.prob_threshold_for_intercept = (cloned.prob_threshold_for_intercept + rng.random_range(-step_size..step_size)).clamp(0.1, 1.0),
            // u8 参数：整数步长扰动，clamp 到合理范围
            12 => cloned.partner_sprint_threshold = (cloned.partner_sprint_threshold as i32 + rng.random_range(-2..=2)).clamp(1, 8) as u8,
            13 => cloned.enemy_low_cards_threshold = (cloned.enemy_low_cards_threshold as i32 + rng.random_range(-2..=2)).clamp(1, 8) as u8,
            14 => cloned.endgame_hand_count_threshold = (cloned.endgame_hand_count_threshold as i32 + rng.random_range(-2..=2)).clamp(1, 10) as u8,
            _ => unreachable!(),
        }
        cloned
    }
}

impl Default for AdvancedBotParams {
    fn default() -> Self {
        Self::default_balanced()
    }
}
