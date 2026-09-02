use rand::Rng;
use serde::{Deserialize, Serialize};

// ── 扩大参数面（路线图②，用户 2026-09-03）：打分常数可训练化 ──
// 默认值 = 引擎现值（零行为变化）。存"量级"（正数），打分时按语义加减。
// 旧 advanced_params.json 缺这些键 → serde default 兜底，加载不失败。
pub const D_BOMB_KEEP_SINGLE: f32 = 200.0; // 炸弹保留：手里只剩1炸 打分减
pub const D_BOMB_KEEP_DOUBLE: f32 = 50.0; // 炸弹保留：2炸
pub const D_LAST_BOMB_PENALTY: f32 = 400.0; // 残局末炸过早出
pub const D_BOMB_OVER_SINGLE: f32 = 300.0; // 炸压单张
pub const D_BOMB_OVER_PAIR: f32 = 200.0; // 炸压对子
pub const D_BOMB_OVER_RUN: f32 = 80.0; // 炸压顺/连对
pub const D_WILD_BOMB_BONUS: f32 = 100.0; // 百搭成炸/同花顺
pub const D_WILD_RUN_BONUS: f32 = 30.0; // 百搭成顺/钢板/木板
pub const D_WILD_TRIPLE_BONUS: f32 = 20.0; // 百搭成三张
pub const D_WILD_FH_BONUS: f32 = 10.0; // 百搭成三带二
pub const D_ENDGAME_SINGLE_REMOVAL: f32 = 400.0; // 残局移除单张
pub const D_ENDGAME_SMALL_SINGLE_REMOVAL: f32 = 300.0; // 残局移除小单张
pub const D_EMPTY_LEAD_BOMB_PENALTY: f32 = 450.0; // 空出炸弹领出
pub const D_SPLIT_PENALTY_SCALE: f32 = 20.0; // 拆牌罚缩放（含百搭孤张保护）
pub const D_KEEP_BOMB_BONUS: f32 = 500.0; // 留炸到残局控牌
pub const D_SOLVER_TRICK_PENALTY: f32 = 500.0; // 残局求解器每墩

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
    // ── 打分常数（默认=引擎现值，见上方 D_* 常量）──
    #[serde(default = "default_bomb_keep_single")]
    pub bomb_keep_single: f32,
    #[serde(default = "default_bomb_keep_double")]
    pub bomb_keep_double: f32,
    #[serde(default = "default_last_bomb_penalty")]
    pub last_bomb_penalty: f32,
    #[serde(default = "default_bomb_over_single")]
    pub bomb_over_single: f32,
    #[serde(default = "default_bomb_over_pair")]
    pub bomb_over_pair: f32,
    #[serde(default = "default_bomb_over_run")]
    pub bomb_over_run: f32,
    #[serde(default = "default_wild_bomb_bonus")]
    pub wild_bomb_bonus: f32,
    #[serde(default = "default_wild_run_bonus")]
    pub wild_run_bonus: f32,
    #[serde(default = "default_wild_triple_bonus")]
    pub wild_triple_bonus: f32,
    #[serde(default = "default_wild_fh_bonus")]
    pub wild_fh_bonus: f32,
    #[serde(default = "default_endgame_single_removal")]
    pub endgame_single_removal: f32,
    #[serde(default = "default_endgame_small_single_removal")]
    pub endgame_small_single_removal: f32,
    #[serde(default = "default_empty_lead_bomb_penalty")]
    pub empty_lead_bomb_penalty: f32,
    #[serde(default = "default_split_penalty_scale")]
    pub split_penalty_scale: f32,
    #[serde(default = "default_keep_bomb_bonus")]
    pub keep_bomb_bonus: f32,
    #[serde(default = "default_solver_trick_penalty")]
    pub solver_trick_penalty: f32,
}

macro_rules! serde_default_fn {
    ($fn_name:ident, $const_name:ident) => {
        fn $fn_name() -> f32 {
            $const_name
        }
    };
}

serde_default_fn!(default_bomb_keep_single, D_BOMB_KEEP_SINGLE);
serde_default_fn!(default_bomb_keep_double, D_BOMB_KEEP_DOUBLE);
serde_default_fn!(default_last_bomb_penalty, D_LAST_BOMB_PENALTY);
serde_default_fn!(default_bomb_over_single, D_BOMB_OVER_SINGLE);
serde_default_fn!(default_bomb_over_pair, D_BOMB_OVER_PAIR);
serde_default_fn!(default_bomb_over_run, D_BOMB_OVER_RUN);
serde_default_fn!(default_wild_bomb_bonus, D_WILD_BOMB_BONUS);
serde_default_fn!(default_wild_run_bonus, D_WILD_RUN_BONUS);
serde_default_fn!(default_wild_triple_bonus, D_WILD_TRIPLE_BONUS);
serde_default_fn!(default_wild_fh_bonus, D_WILD_FH_BONUS);
serde_default_fn!(default_endgame_single_removal, D_ENDGAME_SINGLE_REMOVAL);
serde_default_fn!(default_endgame_small_single_removal, D_ENDGAME_SMALL_SINGLE_REMOVAL);
serde_default_fn!(default_empty_lead_bomb_penalty, D_EMPTY_LEAD_BOMB_PENALTY);
serde_default_fn!(default_split_penalty_scale, D_SPLIT_PENALTY_SCALE);
serde_default_fn!(default_keep_bomb_bonus, D_KEEP_BOMB_BONUS);
serde_default_fn!(default_solver_trick_penalty, D_SOLVER_TRICK_PENALTY);

impl AdvancedBotParams {
    /// 打分常数组（默认=引擎现值），供各 preset / js_trained_params 复用。
    pub(crate) fn scoring_defaults() -> Self {
        Self {
            bomb_keep_single: D_BOMB_KEEP_SINGLE,
            bomb_keep_double: D_BOMB_KEEP_DOUBLE,
            last_bomb_penalty: D_LAST_BOMB_PENALTY,
            bomb_over_single: D_BOMB_OVER_SINGLE,
            bomb_over_pair: D_BOMB_OVER_PAIR,
            bomb_over_run: D_BOMB_OVER_RUN,
            wild_bomb_bonus: D_WILD_BOMB_BONUS,
            wild_run_bonus: D_WILD_RUN_BONUS,
            wild_triple_bonus: D_WILD_TRIPLE_BONUS,
            wild_fh_bonus: D_WILD_FH_BONUS,
            endgame_single_removal: D_ENDGAME_SINGLE_REMOVAL,
            endgame_small_single_removal: D_ENDGAME_SMALL_SINGLE_REMOVAL,
            empty_lead_bomb_penalty: D_EMPTY_LEAD_BOMB_PENALTY,
            split_penalty_scale: D_SPLIT_PENALTY_SCALE,
            keep_bomb_bonus: D_KEEP_BOMB_BONUS,
            solver_trick_penalty: D_SOLVER_TRICK_PENALTY,
            team_win_weight: 0.0,
            first_out_weight: 0.0,
            second_out_weight: 0.0,
            yield_to_partner_bias: 0.0,
            partner_sprint_threshold: 3,
            bomb_conserve_bias: 0.0,
            bomb_aggression_when_enemy_low: 0.0,
            enemy_low_cards_threshold: 3,
            endgame_hand_count_threshold: 6,
            endgame_clear_hand_bias: 0.0,
            proactive_play_bias: 0.0,
            low_card_dump_bias: 0.0,
            pass_stall_penalty: 0.0,
            hand_tracker_enabled: true,
            prob_threshold_for_bomb: 0.6,
            prob_threshold_for_intercept: 0.4,
            enable_reason_trace: false,
        }
    }

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
            ..Self::scoring_defaults()
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
            ..Self::scoring_defaults()
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
            ..Self::scoring_defaults()
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
        // 房规锁：可变异参数为 10 个 f32 权重（0-9）+ 2 个概率阈值（10-11）+
        // 16 个打分常数（12-27，路线图②扩大参数面；默认=引擎现值）。
        // 3 个 u8 阈值（冲刺/残局/队友冲刺）不参与变异——训练不可漂移房规。
        let idx = rng.random_range(0..28);
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
            // 概率阈值参数（牌踪器已激活）：clamp 到 [0.1, 1.0]
            10 => cloned.prob_threshold_for_bomb = (cloned.prob_threshold_for_bomb + rng.random_range(-step_size..step_size)).clamp(0.1, 1.0),
            11 => cloned.prob_threshold_for_intercept = (cloned.prob_threshold_for_intercept + rng.random_range(-step_size..step_size)).clamp(0.1, 1.0),
            // 打分常数（量级，正数）：clamp 到 [1, 2000]
            12 => cloned.bomb_keep_single = (cloned.bomb_keep_single + rng.random_range(-step_size..step_size)).clamp(1.0, 2000.0),
            13 => cloned.bomb_keep_double = (cloned.bomb_keep_double + rng.random_range(-step_size..step_size)).clamp(1.0, 2000.0),
            14 => cloned.last_bomb_penalty = (cloned.last_bomb_penalty + rng.random_range(-step_size..step_size)).clamp(1.0, 2000.0),
            15 => cloned.bomb_over_single = (cloned.bomb_over_single + rng.random_range(-step_size..step_size)).clamp(1.0, 2000.0),
            16 => cloned.bomb_over_pair = (cloned.bomb_over_pair + rng.random_range(-step_size..step_size)).clamp(1.0, 2000.0),
            17 => cloned.bomb_over_run = (cloned.bomb_over_run + rng.random_range(-step_size..step_size)).clamp(1.0, 2000.0),
            18 => cloned.wild_bomb_bonus = (cloned.wild_bomb_bonus + rng.random_range(-step_size..step_size)).clamp(1.0, 2000.0),
            19 => cloned.wild_run_bonus = (cloned.wild_run_bonus + rng.random_range(-step_size..step_size)).clamp(1.0, 2000.0),
            20 => cloned.wild_triple_bonus = (cloned.wild_triple_bonus + rng.random_range(-step_size..step_size)).clamp(1.0, 2000.0),
            21 => cloned.wild_fh_bonus = (cloned.wild_fh_bonus + rng.random_range(-step_size..step_size)).clamp(1.0, 2000.0),
            22 => cloned.endgame_single_removal = (cloned.endgame_single_removal + rng.random_range(-step_size..step_size)).clamp(1.0, 2000.0),
            23 => cloned.endgame_small_single_removal = (cloned.endgame_small_single_removal + rng.random_range(-step_size..step_size)).clamp(1.0, 2000.0),
            24 => cloned.empty_lead_bomb_penalty = (cloned.empty_lead_bomb_penalty + rng.random_range(-step_size..step_size)).clamp(1.0, 2000.0),
            25 => cloned.split_penalty_scale = (cloned.split_penalty_scale + rng.random_range(-step_size..step_size)).clamp(5.0, 100.0),
            26 => cloned.keep_bomb_bonus = (cloned.keep_bomb_bonus + rng.random_range(-step_size..step_size)).clamp(1.0, 2000.0),
            27 => cloned.solver_trick_penalty = (cloned.solver_trick_penalty + rng.random_range(-step_size..step_size)).clamp(1.0, 2000.0),
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
