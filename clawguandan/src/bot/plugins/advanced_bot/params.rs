use rand::Rng;
use serde::{Deserialize, Serialize};

// ── 扩大参数面（路线图②，用户 2026-09-03）：打分常数可训练化 ──
// 默认值 = 引擎现值（除用户明令调整的四项外零行为变化）。存"量级"（正数），打分时按语义加减。
// 旧 advanced_params.json 缺这些键 → serde default 兜底，加载不失败。
//
// 用户房规调整（2026-09-03，双端生效）：
//   · 逢人配组成三张：不奖励（原 +20 移除）
//   · 逢人配组成三带二：不奖励（原 +10 移除；"百搭凑对三带二"罚侧房规保留）
//   · 炸弹压顺子/钢板/木板：不扣分（原 −80 移除）
//   · 手里有 2 炸时用炸：不扣分（原 −50 移除；1 炸的保留罚不变）
pub const D_BOMB_KEEP_SINGLE: f32 = 200.0; // 炸弹保留：手里只剩1炸 打分减
pub const D_LAST_BOMB_PENALTY: f32 = 400.0; // 残局末炸过早出
pub const D_BOMB_OVER_SINGLE: f32 = 300.0; // 炸压单张
pub const D_BOMB_OVER_PAIR: f32 = 200.0; // 炸压对子
pub const D_WILD_BOMB_BONUS: f32 = 100.0; // 百搭成炸/同花顺
pub const D_WILD_RUN_BONUS: f32 = 30.0; // 百搭成顺/钢板/木板
pub const D_ENDGAME_SINGLE_REMOVAL: f32 = 400.0; // 残局移除单张
pub const D_ENDGAME_SMALL_SINGLE_REMOVAL: f32 = 300.0; // 残局移除小单张
pub const D_EMPTY_LEAD_BOMB_PENALTY: f32 = 450.0; // 空出炸弹领出
pub const D_SPLIT_PENALTY_SCALE: f32 = 20.0; // 拆牌罚缩放（含百搭孤张保护）
pub const D_KEEP_BOMB_BONUS: f32 = 500.0; // 留炸到残局控牌
pub const D_SOLVER_TRICK_PENALTY: f32 = 500.0; // 残局求解器每墩
pub const D_BOMB_KEEP_MANY: f32 = 10.0; // 3+炸时用炸轻罚
pub const D_INTERCEPT_SPRINT_BONUS: f32 = 15.0; // 对手≤6张 炸弹拦截是好选择
pub const D_LAST_PLAY_CLEAR_BONUS: f32 = 20.0; // 末手出炸清牌
pub const D_COMBO_SHAPE_BONUS: f32 = 20.0; // 出组合牌型让对手难接
pub const D_PARTNER_FENG_BONUS: f32 = 300.0; // 给联邦接风（跟牌侧）
pub const D_PARTNER_FENG_LEAD_BONUS: f32 = 180.0; // 为队友接风压制敌人
pub const D_PARTNER_FENG_FIRST_BONUS: f32 = 120.0; // 接风首出权
pub const D_TEAMMATE_COMBO_BONUS: f32 = 15.0; // 优先出帮队友的组合
pub const D_BLOCK_ENEMY_BONUS: f32 = 15.0; // 出大牌阻止对手送牌
pub const D_STRAIGHT_BUILD_BONUS: f32 = 30.0; // 单牌能组成顺子
pub const D_MANY_SINGLES_PENALTY: f32 = 80.0; // 打完后剩太多散单
pub const D_SINGLE_LEAD_BONUS: f32 = 20.0; // 单张优先出
pub const D_SMALL_SINGLE_LEAD_BONUS: f32 = 15.0; // 小单张更优先
pub const D_SMALL_CARD_LEAD_BONUS: f32 = 10.0; // 小牌奖励
pub const D_AVOID_SMALL_SINGLES_EACH: f32 = 22.0; // 残局避免剩小单张 每张
pub const D_LEAD_LEN_STEP: f32 = 8.0; // 非残局领出每张加分
pub const D_LEAD_LEN_STEP_ENDGAME: f32 = 5.0; // 残局领出每张加分
pub const D_LEAD_PRIMARY_STEP: f32 = 0.5; // 非残局 primary 每点罚
pub const D_LEAD_PRIMARY_STEP_ENDGAME: f32 = 1.5; // 残局 primary 每点罚
pub const D_SOLVER_JUNK_BONUS: f32 = 15.0; // 求解器同墩甩废单倾向 每张
pub const D_DUAL_WILD_PENALTY_MID: f32 = 600.0; // 中盘双百搭同出重罚
pub const D_DUAL_WILD_PENALTY_END: f32 = 60.0; // 残局双百搭同出罚
pub const D_UPGRADED_BOMB_WILD_MID: f32 = 150.0; // 天然炸弹贴百搭升档中盘重罚
pub const D_UPGRADED_BOMB_WILD_END: f32 = 10.0; // 升档残局轻罚
pub const D_WILD_ON_LEVEL_MID: f32 = 250.0; // 百搭落级牌中盘重罚
pub const D_WILD_ON_LEVEL_END: f32 = 20.0; // 百搭落级牌残局轻罚
pub const D_WILD_PLAIN_PAIR_MID: f32 = 300.0; // 百搭配普通单张成普通对中盘重罚
pub const D_WILD_PAIR_PENALTY_END: f32 = 15.0; // 百搭配对子残局轻罚
pub const D_BARE_DUAL_WILD_EXTRA: f32 = 200.0; // 裸出双百搭额外加重

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
    #[serde(default = "default_last_bomb_penalty")]
    pub last_bomb_penalty: f32,
    #[serde(default = "default_bomb_over_single")]
    pub bomb_over_single: f32,
    #[serde(default = "default_bomb_over_pair")]
    pub bomb_over_pair: f32,
    #[serde(default = "default_wild_bomb_bonus")]
    pub wild_bomb_bonus: f32,
    #[serde(default = "default_wild_run_bonus")]
    pub wild_run_bonus: f32,
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
    #[serde(default = "default_bomb_keep_many")]
    pub bomb_keep_many: f32,
    #[serde(default = "default_intercept_sprint_bonus")]
    pub intercept_sprint_bonus: f32,
    #[serde(default = "default_last_play_clear_bonus")]
    pub last_play_clear_bonus: f32,
    #[serde(default = "default_combo_shape_bonus")]
    pub combo_shape_bonus: f32,
    #[serde(default = "default_partner_feng_bonus")]
    pub partner_feng_bonus: f32,
    #[serde(default = "default_partner_feng_lead_bonus")]
    pub partner_feng_lead_bonus: f32,
    #[serde(default = "default_partner_feng_first_bonus")]
    pub partner_feng_first_bonus: f32,
    #[serde(default = "default_teammate_combo_bonus")]
    pub teammate_combo_bonus: f32,
    #[serde(default = "default_block_enemy_bonus")]
    pub block_enemy_bonus: f32,
    #[serde(default = "default_straight_build_bonus")]
    pub straight_build_bonus: f32,
    #[serde(default = "default_many_singles_penalty")]
    pub many_singles_penalty: f32,
    #[serde(default = "default_single_lead_bonus")]
    pub single_lead_bonus: f32,
    #[serde(default = "default_small_single_lead_bonus")]
    pub small_single_lead_bonus: f32,
    #[serde(default = "default_small_card_lead_bonus")]
    pub small_card_lead_bonus: f32,
    #[serde(default = "default_avoid_small_singles_each")]
    pub avoid_small_singles_each: f32,
    #[serde(default = "default_lead_len_step")]
    pub lead_len_step: f32,
    #[serde(default = "default_lead_len_step_endgame")]
    pub lead_len_step_endgame: f32,
    #[serde(default = "default_lead_primary_step")]
    pub lead_primary_step: f32,
    #[serde(default = "default_lead_primary_step_endgame")]
    pub lead_primary_step_endgame: f32,
    #[serde(default = "default_solver_junk_bonus")]
    pub solver_junk_bonus: f32,
    #[serde(default = "default_dual_wild_penalty_mid")]
    pub dual_wild_penalty_mid: f32,
    #[serde(default = "default_dual_wild_penalty_end")]
    pub dual_wild_penalty_end: f32,
    #[serde(default = "default_upgraded_bomb_wild_mid")]
    pub upgraded_bomb_wild_mid: f32,
    #[serde(default = "default_upgraded_bomb_wild_end")]
    pub upgraded_bomb_wild_end: f32,
    #[serde(default = "default_wild_on_level_mid")]
    pub wild_on_level_mid: f32,
    #[serde(default = "default_wild_on_level_end")]
    pub wild_on_level_end: f32,
    #[serde(default = "default_wild_plain_pair_mid")]
    pub wild_plain_pair_mid: f32,
    #[serde(default = "default_wild_pair_penalty_end")]
    pub wild_pair_penalty_end: f32,
    #[serde(default = "default_bare_dual_wild_extra")]
    pub bare_dual_wild_extra: f32,
}

macro_rules! serde_default_fn {
    ($fn_name:ident, $const_name:ident) => {
        fn $fn_name() -> f32 {
            $const_name
        }
    };
}

serde_default_fn!(default_bomb_keep_single, D_BOMB_KEEP_SINGLE);
serde_default_fn!(default_last_bomb_penalty, D_LAST_BOMB_PENALTY);
serde_default_fn!(default_bomb_over_single, D_BOMB_OVER_SINGLE);
serde_default_fn!(default_bomb_over_pair, D_BOMB_OVER_PAIR);
serde_default_fn!(default_wild_bomb_bonus, D_WILD_BOMB_BONUS);
serde_default_fn!(default_wild_run_bonus, D_WILD_RUN_BONUS);
serde_default_fn!(default_endgame_single_removal, D_ENDGAME_SINGLE_REMOVAL);
serde_default_fn!(default_endgame_small_single_removal, D_ENDGAME_SMALL_SINGLE_REMOVAL);
serde_default_fn!(default_empty_lead_bomb_penalty, D_EMPTY_LEAD_BOMB_PENALTY);
serde_default_fn!(default_split_penalty_scale, D_SPLIT_PENALTY_SCALE);
serde_default_fn!(default_keep_bomb_bonus, D_KEEP_BOMB_BONUS);
serde_default_fn!(default_solver_trick_penalty, D_SOLVER_TRICK_PENALTY);
serde_default_fn!(default_bomb_keep_many, D_BOMB_KEEP_MANY);
serde_default_fn!(default_intercept_sprint_bonus, D_INTERCEPT_SPRINT_BONUS);
serde_default_fn!(default_last_play_clear_bonus, D_LAST_PLAY_CLEAR_BONUS);
serde_default_fn!(default_combo_shape_bonus, D_COMBO_SHAPE_BONUS);
serde_default_fn!(default_partner_feng_bonus, D_PARTNER_FENG_BONUS);
serde_default_fn!(default_partner_feng_lead_bonus, D_PARTNER_FENG_LEAD_BONUS);
serde_default_fn!(default_partner_feng_first_bonus, D_PARTNER_FENG_FIRST_BONUS);
serde_default_fn!(default_teammate_combo_bonus, D_TEAMMATE_COMBO_BONUS);
serde_default_fn!(default_block_enemy_bonus, D_BLOCK_ENEMY_BONUS);
serde_default_fn!(default_straight_build_bonus, D_STRAIGHT_BUILD_BONUS);
serde_default_fn!(default_many_singles_penalty, D_MANY_SINGLES_PENALTY);
serde_default_fn!(default_single_lead_bonus, D_SINGLE_LEAD_BONUS);
serde_default_fn!(default_small_single_lead_bonus, D_SMALL_SINGLE_LEAD_BONUS);
serde_default_fn!(default_small_card_lead_bonus, D_SMALL_CARD_LEAD_BONUS);
serde_default_fn!(default_avoid_small_singles_each, D_AVOID_SMALL_SINGLES_EACH);
serde_default_fn!(default_lead_len_step, D_LEAD_LEN_STEP);
serde_default_fn!(default_lead_len_step_endgame, D_LEAD_LEN_STEP_ENDGAME);
serde_default_fn!(default_lead_primary_step, D_LEAD_PRIMARY_STEP);
serde_default_fn!(default_lead_primary_step_endgame, D_LEAD_PRIMARY_STEP_ENDGAME);
serde_default_fn!(default_solver_junk_bonus, D_SOLVER_JUNK_BONUS);
serde_default_fn!(default_dual_wild_penalty_mid, D_DUAL_WILD_PENALTY_MID);
serde_default_fn!(default_dual_wild_penalty_end, D_DUAL_WILD_PENALTY_END);
serde_default_fn!(default_upgraded_bomb_wild_mid, D_UPGRADED_BOMB_WILD_MID);
serde_default_fn!(default_upgraded_bomb_wild_end, D_UPGRADED_BOMB_WILD_END);
serde_default_fn!(default_wild_on_level_mid, D_WILD_ON_LEVEL_MID);
serde_default_fn!(default_wild_on_level_end, D_WILD_ON_LEVEL_END);
serde_default_fn!(default_wild_plain_pair_mid, D_WILD_PLAIN_PAIR_MID);
serde_default_fn!(default_wild_pair_penalty_end, D_WILD_PAIR_PENALTY_END);
serde_default_fn!(default_bare_dual_wild_extra, D_BARE_DUAL_WILD_EXTRA);

impl AdvancedBotParams {
    /// 打分常数组（默认=引擎现值），供各 preset / js_trained_params 复用。
    pub(crate) fn scoring_defaults() -> Self {
        Self {
            bomb_keep_single: D_BOMB_KEEP_SINGLE,
            last_bomb_penalty: D_LAST_BOMB_PENALTY,
            bomb_over_single: D_BOMB_OVER_SINGLE,
            bomb_over_pair: D_BOMB_OVER_PAIR,
            wild_bomb_bonus: D_WILD_BOMB_BONUS,
            wild_run_bonus: D_WILD_RUN_BONUS,
            endgame_single_removal: D_ENDGAME_SINGLE_REMOVAL,
            endgame_small_single_removal: D_ENDGAME_SMALL_SINGLE_REMOVAL,
            empty_lead_bomb_penalty: D_EMPTY_LEAD_BOMB_PENALTY,
            split_penalty_scale: D_SPLIT_PENALTY_SCALE,
            keep_bomb_bonus: D_KEEP_BOMB_BONUS,
            solver_trick_penalty: D_SOLVER_TRICK_PENALTY,
            bomb_keep_many: D_BOMB_KEEP_MANY,
            intercept_sprint_bonus: D_INTERCEPT_SPRINT_BONUS,
            last_play_clear_bonus: D_LAST_PLAY_CLEAR_BONUS,
            combo_shape_bonus: D_COMBO_SHAPE_BONUS,
            partner_feng_bonus: D_PARTNER_FENG_BONUS,
            partner_feng_lead_bonus: D_PARTNER_FENG_LEAD_BONUS,
            partner_feng_first_bonus: D_PARTNER_FENG_FIRST_BONUS,
            teammate_combo_bonus: D_TEAMMATE_COMBO_BONUS,
            block_enemy_bonus: D_BLOCK_ENEMY_BONUS,
            straight_build_bonus: D_STRAIGHT_BUILD_BONUS,
            many_singles_penalty: D_MANY_SINGLES_PENALTY,
            single_lead_bonus: D_SINGLE_LEAD_BONUS,
            small_single_lead_bonus: D_SMALL_SINGLE_LEAD_BONUS,
            small_card_lead_bonus: D_SMALL_CARD_LEAD_BONUS,
            avoid_small_singles_each: D_AVOID_SMALL_SINGLES_EACH,
            lead_len_step: D_LEAD_LEN_STEP,
            lead_len_step_endgame: D_LEAD_LEN_STEP_ENDGAME,
            lead_primary_step: D_LEAD_PRIMARY_STEP,
            lead_primary_step_endgame: D_LEAD_PRIMARY_STEP_ENDGAME,
            solver_junk_bonus: D_SOLVER_JUNK_BONUS,
            dual_wild_penalty_mid: D_DUAL_WILD_PENALTY_MID,
            dual_wild_penalty_end: D_DUAL_WILD_PENALTY_END,
            upgraded_bomb_wild_mid: D_UPGRADED_BOMB_WILD_MID,
            upgraded_bomb_wild_end: D_UPGRADED_BOMB_WILD_END,
            wild_on_level_mid: D_WILD_ON_LEVEL_MID,
            wild_on_level_end: D_WILD_ON_LEVEL_END,
            wild_plain_pair_mid: D_WILD_PLAIN_PAIR_MID,
            wild_pair_penalty_end: D_WILD_PAIR_PENALTY_END,
            bare_dual_wild_extra: D_BARE_DUAL_WILD_EXTRA,
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
        // 41 个打分常数（12-52，路线图②扩大参数面；默认=引擎现值）。
        // 3 个 u8 阈值（冲刺/残局/队友冲刺）不参与变异——训练不可漂移房规。
        // 禁令/豁免/阈值全部冻结，可训练的只有打分量级。
        let idx = rng.random_range(0..53);
        macro_rules! m {
            ($field:ident, $lo:expr, $hi:expr) => {
                cloned.$field = (cloned.$field + rng.random_range(-step_size..step_size)).clamp($lo, $hi)
            };
        }
        match idx {
            // f32 参数：加随机浮点扰动，clamp 到合理范围
            0 => m!(team_win_weight, 0.1, 10.0),
            1 => m!(first_out_weight, 0.1, 10.0),
            2 => m!(second_out_weight, 0.1, 10.0),
            3 => m!(yield_to_partner_bias, 0.1, 10.0),
            4 => m!(bomb_conserve_bias, 0.1, 10.0),
            5 => m!(bomb_aggression_when_enemy_low, 0.1, 10.0),
            6 => m!(endgame_clear_hand_bias, 0.1, 10.0),
            7 => m!(proactive_play_bias, 0.1, 10.0),
            8 => m!(low_card_dump_bias, 0.1, 10.0),
            9 => m!(pass_stall_penalty, 0.1, 10.0),
            // 概率阈值参数（牌踪器已激活）：clamp 到 [0.1, 1.0]
            10 => m!(prob_threshold_for_bomb, 0.1, 1.0),
            11 => m!(prob_threshold_for_intercept, 0.1, 1.0),
            // 打分常数（量级，正数）：clamp 到 [1, 2000]
            12 => m!(bomb_keep_single, 1.0, 2000.0),
            13 => m!(last_bomb_penalty, 1.0, 2000.0),
            14 => m!(bomb_over_single, 1.0, 2000.0),
            15 => m!(bomb_over_pair, 1.0, 2000.0),
            16 => m!(wild_bomb_bonus, 1.0, 2000.0),
            17 => m!(wild_run_bonus, 1.0, 2000.0),
            18 => m!(endgame_single_removal, 1.0, 2000.0),
            19 => m!(endgame_small_single_removal, 1.0, 2000.0),
            20 => m!(empty_lead_bomb_penalty, 1.0, 2000.0),
            21 => m!(split_penalty_scale, 5.0, 100.0),
            22 => m!(keep_bomb_bonus, 1.0, 2000.0),
            23 => m!(solver_trick_penalty, 1.0, 2000.0),
            24 => m!(bomb_keep_many, 1.0, 2000.0),
            25 => m!(intercept_sprint_bonus, 1.0, 2000.0),
            26 => m!(last_play_clear_bonus, 1.0, 2000.0),
            27 => m!(combo_shape_bonus, 1.0, 2000.0),
            28 => m!(partner_feng_bonus, 1.0, 2000.0),
            29 => m!(partner_feng_lead_bonus, 1.0, 2000.0),
            30 => m!(partner_feng_first_bonus, 1.0, 2000.0),
            31 => m!(teammate_combo_bonus, 1.0, 2000.0),
            32 => m!(block_enemy_bonus, 1.0, 2000.0),
            33 => m!(straight_build_bonus, 1.0, 2000.0),
            34 => m!(many_singles_penalty, 1.0, 2000.0),
            35 => m!(single_lead_bonus, 1.0, 2000.0),
            36 => m!(small_single_lead_bonus, 1.0, 2000.0),
            37 => m!(small_card_lead_bonus, 1.0, 2000.0),
            38 => m!(avoid_small_singles_each, 1.0, 2000.0),
            // 领出整形/求解器倾向：允许训练到 0（关闭该项）
            39 => m!(lead_len_step, 0.0, 100.0),
            40 => m!(lead_len_step_endgame, 0.0, 100.0),
            41 => m!(lead_primary_step, 0.0, 50.0),
            42 => m!(lead_primary_step_endgame, 0.0, 50.0),
            43 => m!(solver_junk_bonus, 0.0, 500.0),
            // 双百搭/升档/落级牌罚家族（JS 房规口径，量级可训练）
            44 => m!(dual_wild_penalty_mid, 1.0, 2000.0),
            45 => m!(dual_wild_penalty_end, 1.0, 2000.0),
            46 => m!(upgraded_bomb_wild_mid, 1.0, 2000.0),
            47 => m!(upgraded_bomb_wild_end, 1.0, 2000.0),
            48 => m!(wild_on_level_mid, 1.0, 2000.0),
            49 => m!(wild_on_level_end, 1.0, 2000.0),
            50 => m!(wild_plain_pair_mid, 1.0, 2000.0),
            51 => m!(wild_pair_penalty_end, 1.0, 2000.0),
            52 => m!(bare_dual_wild_extra, 1.0, 2000.0),
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
