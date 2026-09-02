use std::thread;
use std::time::Duration;

use crate::bot::plugins::AdvancedBotParams;
use crate::game::card::HandLevel;
use crate::game::engine::GameEngine;
use crate::game::types::{GameConfig, GamePhase, TeamId};
use crate::domain::Seat;
use crate::simulation::engine::run_match_engine;
use crate::strategy::suggest::set_learn_params_for_teams;
use rand::Rng;

#[derive(Clone, Debug)]
pub struct GeneticConfig {
    pub population_size: usize,
    pub generations: u32,
    pub matches_per_eval: u32,
    pub crossover_rate: f32,
    pub mutation_rate: f32,
    pub mutation_step_size: f32,
    pub elitism_count: usize,
}

impl Default for GeneticConfig {
    fn default() -> Self {
        Self {
            population_size: 20,
            generations: 50,
            matches_per_eval: 50,
            crossover_rate: 0.7,
            mutation_rate: 0.2,
            mutation_step_size: 0.15,
            elitism_count: 2,
        }
    }
}

#[derive(Clone, Debug)]
struct Individual {
    params: AdvancedBotParams,
    score: f32,
    eval_result: EvalResult,
}

impl Ord for Individual {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score.partial_cmp(&other.score).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl PartialOrd for Individual {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Individual {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for Individual {}

#[derive(Clone, Debug)]
pub struct EvalResult {
    pub win_rate: f32,
    /// 平均升级行（0..1）：赢双上+3/赢单上+2/输单上−2/被双上−3，归一化 (delta+3)/6。
    /// 与 optimizer.rs 保持一致（路线图①，用户 2026-09-03）。
    pub level_ev: f32,
    pub first_out_rate: f32,
    /// NS队结束时平均剩余手牌总数(S+N剩余之和)。越少越好。
    /// 直接衡量"残局剩牌"表现，缓解"最后剩小牌和单张"问题。
    pub avg_endgame_residual: f32,
    pub matches_played: u32,
}

pub fn eval_to_score(result: &EvalResult) -> f32 {
    // 残局剩牌越少越好：clear_rate = 1 - avg_residual/27
    // (27 ≈ NS两人初始手牌总数上界，用作归一化)
    // 权重：升级期望0.5 + 头游0.2 + 残局清牌0.3
    // 与 optimizer.rs 保持一致，避免遗传算法训练目标与爬山算法不一致。
    let clear_rate = (1.0 - (result.avg_endgame_residual / 27.0).clamp(0.0, 1.0)).max(0.0);
    result.level_ev * 0.5 + result.first_out_rate * 0.2 + clear_rate * 0.3
}

fn seat_team(seat: Seat) -> TeamId {
    match seat {
        Seat::E | Seat::W => TeamId::Ew,
        Seat::S | Seat::N => TeamId::Sn,
    }
}

fn run_single_match(engine: &GameEngine) -> Result<Option<(TeamId, TeamId, i32, usize)>, String> {
    let mut state = engine.init_table(format!("ga_{}", uuid::Uuid::new_v4()));
    let first_drawer = Seat::S;
    engine
        .start_first_hand(&mut state, first_drawer, HandLevel::Two)
        .map_err(|e| format!("start_first_hand: {e}"))?;

    let outcome = run_match_engine(engine, &mut state, 1, 2000)
        .map_err(|e| format!("run_match: {e}"))?;

    if outcome.final_phase == GamePhase::Scoring {
        if let Some(winner) = state.winner_team {
            let finishing = state
                .hand
                .as_ref()
                .map(|h| h.finishing_order.clone())
                .unwrap_or_default();
            let first_out_team = finishing.first().copied().map(seat_team).unwrap_or(winner);
            // 升级行：头游队为胜方；胜方二游也是本队 → 双上+3，否则单上+2；负方取负。
            let ns_level_delta = match (finishing.first(), finishing.get(1)) {
                (Some(f1), Some(f2)) => {
                    let gain = if seat_team(*f2) == seat_team(*f1) { 3i32 } else { 2 };
                    if seat_team(*f1) == TeamId::Sn { gain } else { -gain }
                }
                _ => 0,
            };
            // NS队结束时剩余手牌总数(S+N)：衡量残局剩牌表现
            let ns_residual = state
                .hand
                .as_ref()
                .map(|h| {
                    h.hands.get(&Seat::S).map(|v| v.len()).unwrap_or(0)
                        + h.hands.get(&Seat::N).map(|v| v.len()).unwrap_or(0)
                })
                .unwrap_or(0);
            Ok(Some((winner, first_out_team, ns_level_delta, ns_residual)))
        } else {
            Ok(None)
        }
    } else {
        Ok(None)
    }
}

fn evaluate_individual(params: &AdvancedBotParams, matches: u32) -> EvalResult {
    // 打破对称性：NS 用候选参数，EW 用固定基线参数(js_trained_params 房规基线)。
    // 这样 NS 胜率能真实反映候选参数相对基准的优劣，避免自对弈对称导致的~50%胜率随机游走。
    // 与 optimizer.rs 保持一致。（房规：基线从 default_balanced 改为 js_trained_params）
    let baseline = crate::strategy::suggest::js_trained_params();
    // 房规隔离（2026-09-03）：LEARN_PARAMS 只对本训练线程生效，线上桌面永远房规基线
    let _training_scope = crate::strategy::suggest::TrainingGuard::new();
    set_learn_params_for_teams(Some(params.clone()), Some(baseline));

    // 记录当前训练 generation,用于检测是否有新训练启动
    let my_gen = crate::learning::current_generation();

    let mut ns_wins = 0u32;
    let mut ns_first_out = 0u32;
    let mut ns_residual_sum: usize = 0;
    let mut ns_level_delta_sum: i32 = 0;
    let mut played = 0u32;

    for _ in 0..matches {
        // 检查是否被新训练取代或被用户停止
        if !crate::learning::is_running_generation(my_gen) {
            break;
        }
        let engine = GameEngine::new(GameConfig { rng_seed: rand::random(), randomize_deals: false });
        match run_single_match(&engine) {
            Ok(Some((winner, first_out_team, ns_level_delta, ns_residual))) => {
                played += 1;
                if winner == TeamId::Sn {
                    ns_wins += 1;
                }
                if first_out_team == TeamId::Sn {
                    ns_first_out += 1;
                }
                ns_level_delta_sum += ns_level_delta;
                ns_residual_sum += ns_residual;
            }
            _ => {
                played += 1;
                // 失败比赛按最差情况计：NS两人满手牌未清(27)、升级行按被双上(−3)计
                // 避免失败比赛被当作0张残牌而人为抬高clear_rate评分
                ns_residual_sum += 27;
                ns_level_delta_sum -= 3;
            }
        }
        if crate::learning::is_running_generation(my_gen) {
            thread::sleep(Duration::from_millis(5));
        }
    }

    // 只有当 generation 仍匹配时才 reset
    // 如果 generation 不匹配(被新训练取代),不 reset,避免清除新训练的参数
    if crate::learning::is_running_generation(my_gen) {
        set_learn_params_for_teams(None, None);
    }

    let win_rate = if played > 0 {
        ns_wins as f32 / played as f32
    } else {
        0.5
    };
    let first_out_rate = if played > 0 {
        ns_first_out as f32 / played as f32
    } else {
        0.5
    };
    let avg_endgame_residual = if played > 0 {
        ns_residual_sum as f32 / played as f32
    } else {
        27.0
    };

    EvalResult {
        win_rate,
        level_ev: if played > 0 {
            ((ns_level_delta_sum as f32 / played as f32) + 3.0) / 6.0
        } else {
            0.5
        },
        first_out_rate,
        avg_endgame_residual,
        matches_played: played,
    }
}

fn crossover(parent1: &AdvancedBotParams, parent2: &AdvancedBotParams) -> AdvancedBotParams {
    let mut rng = rand::rng();
    let mut child = parent1.clone();

    if rng.random_bool(0.5) { child.team_win_weight = parent2.team_win_weight; }
    if rng.random_bool(0.5) { child.first_out_weight = parent2.first_out_weight; }
    if rng.random_bool(0.5) { child.second_out_weight = parent2.second_out_weight; }
    if rng.random_bool(0.5) { child.yield_to_partner_bias = parent2.yield_to_partner_bias; }
    if rng.random_bool(0.5) { child.bomb_conserve_bias = parent2.bomb_conserve_bias; }
    if rng.random_bool(0.5) { child.bomb_aggression_when_enemy_low = parent2.bomb_aggression_when_enemy_low; }
    if rng.random_bool(0.5) { child.endgame_clear_hand_bias = parent2.endgame_clear_hand_bias; }
    if rng.random_bool(0.5) { child.proactive_play_bias = parent2.proactive_play_bias; }
    if rng.random_bool(0.5) { child.low_card_dump_bias = parent2.low_card_dump_bias; }
    if rng.random_bool(0.5) { child.pass_stall_penalty = parent2.pass_stall_penalty; }

    if rng.random_bool(0.5) { child.partner_sprint_threshold = parent2.partner_sprint_threshold; }
    if rng.random_bool(0.5) { child.enemy_low_cards_threshold = parent2.enemy_low_cards_threshold; }
    if rng.random_bool(0.5) { child.endgame_hand_count_threshold = parent2.endgame_hand_count_threshold; }

    // 打分常数（路线图②扩大参数面）：与 optimizer.rs 变异表对齐
    // （用户 2026-09-03：bomb_keep_double/bomb_over_run/wild_triple_bonus/wild_fh_bonus
    //   已按房规调整移除——不奖不罚，不再入参）
    if rng.random_bool(0.5) { child.bomb_keep_single = parent2.bomb_keep_single; }
    if rng.random_bool(0.5) { child.bomb_over_single = parent2.bomb_over_single; }
    if rng.random_bool(0.5) { child.bomb_over_pair = parent2.bomb_over_pair; }
    if rng.random_bool(0.5) { child.wild_bomb_bonus = parent2.wild_bomb_bonus; }
    if rng.random_bool(0.5) { child.wild_run_bonus = parent2.wild_run_bonus; }
    if rng.random_bool(0.5) { child.endgame_single_removal = parent2.endgame_single_removal; }
    if rng.random_bool(0.5) { child.endgame_small_single_removal = parent2.endgame_small_single_removal; }
    if rng.random_bool(0.5) { child.empty_lead_bomb_penalty = parent2.empty_lead_bomb_penalty; }
    if rng.random_bool(0.5) { child.split_penalty_scale = parent2.split_penalty_scale; }
    if rng.random_bool(0.5) { child.keep_bomb_bonus = parent2.keep_bomb_bonus; }
    if rng.random_bool(0.5) { child.solver_trick_penalty = parent2.solver_trick_penalty; }
    if rng.random_bool(0.5) { child.bomb_keep_many = parent2.bomb_keep_many; }
    if rng.random_bool(0.5) { child.intercept_sprint_bonus = parent2.intercept_sprint_bonus; }
    if rng.random_bool(0.5) { child.last_play_clear_bonus = parent2.last_play_clear_bonus; }
    if rng.random_bool(0.5) { child.combo_shape_bonus = parent2.combo_shape_bonus; }
    if rng.random_bool(0.5) { child.partner_feng_bonus = parent2.partner_feng_bonus; }
    if rng.random_bool(0.5) { child.partner_feng_lead_bonus = parent2.partner_feng_lead_bonus; }
    if rng.random_bool(0.5) { child.partner_feng_first_bonus = parent2.partner_feng_first_bonus; }
    if rng.random_bool(0.5) { child.teammate_combo_bonus = parent2.teammate_combo_bonus; }
    if rng.random_bool(0.5) { child.block_enemy_bonus = parent2.block_enemy_bonus; }
    if rng.random_bool(0.5) { child.straight_build_bonus = parent2.straight_build_bonus; }
    if rng.random_bool(0.5) { child.many_singles_penalty = parent2.many_singles_penalty; }
    if rng.random_bool(0.5) { child.single_lead_bonus = parent2.single_lead_bonus; }
    if rng.random_bool(0.5) { child.small_single_lead_bonus = parent2.small_single_lead_bonus; }
    if rng.random_bool(0.5) { child.small_card_lead_bonus = parent2.small_card_lead_bonus; }
    if rng.random_bool(0.5) { child.avoid_small_singles_each = parent2.avoid_small_singles_each; }
    if rng.random_bool(0.5) { child.lead_len_step = parent2.lead_len_step; }
    if rng.random_bool(0.5) { child.lead_len_step_endgame = parent2.lead_len_step_endgame; }
    if rng.random_bool(0.5) { child.lead_primary_step = parent2.lead_primary_step; }
    if rng.random_bool(0.5) { child.lead_primary_step_endgame = parent2.lead_primary_step_endgame; }
    if rng.random_bool(0.5) { child.solver_junk_bonus = parent2.solver_junk_bonus; }
    if rng.random_bool(0.5) { child.dual_wild_penalty_mid = parent2.dual_wild_penalty_mid; }
    if rng.random_bool(0.5) { child.dual_wild_penalty_end = parent2.dual_wild_penalty_end; }
    if rng.random_bool(0.5) { child.upgraded_bomb_wild_mid = parent2.upgraded_bomb_wild_mid; }
    if rng.random_bool(0.5) { child.upgraded_bomb_wild_end = parent2.upgraded_bomb_wild_end; }
    if rng.random_bool(0.5) { child.wild_on_level_mid = parent2.wild_on_level_mid; }
    if rng.random_bool(0.5) { child.wild_on_level_end = parent2.wild_on_level_end; }
    if rng.random_bool(0.5) { child.wild_plain_pair_mid = parent2.wild_plain_pair_mid; }
    if rng.random_bool(0.5) { child.wild_pair_penalty_end = parent2.wild_pair_penalty_end; }
    if rng.random_bool(0.5) { child.bare_dual_wild_extra = parent2.bare_dual_wild_extra; }

    child
}

fn mutate(params: &AdvancedBotParams, rate: f32, step_size: f32) -> AdvancedBotParams {
    let mut rng = rand::rng();
    let mut mutated = params.clone();

    let r = rate as f64;

    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.team_win_weight = (mutated.team_win_weight + delta).clamp(0.1, 10.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.first_out_weight = (mutated.first_out_weight + delta).clamp(0.1, 10.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.second_out_weight = (mutated.second_out_weight + delta).clamp(0.1, 10.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.yield_to_partner_bias = (mutated.yield_to_partner_bias + delta).clamp(0.1, 10.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.bomb_conserve_bias = (mutated.bomb_conserve_bias + delta).clamp(0.1, 10.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.bomb_aggression_when_enemy_low = (mutated.bomb_aggression_when_enemy_low + delta).clamp(0.1, 10.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.endgame_clear_hand_bias = (mutated.endgame_clear_hand_bias + delta).clamp(0.1, 10.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.proactive_play_bias = (mutated.proactive_play_bias + delta).clamp(0.1, 10.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.low_card_dump_bias = (mutated.low_card_dump_bias + delta).clamp(0.1, 10.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.pass_stall_penalty = (mutated.pass_stall_penalty + delta).clamp(0.1, 10.0);
    }

    // 打分常数（路线图②扩大参数面）：clamp [1, 2000]
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.bomb_keep_single = (mutated.bomb_keep_single + delta).clamp(1.0, 2000.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.bomb_over_single = (mutated.bomb_over_single + delta).clamp(1.0, 2000.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.bomb_over_pair = (mutated.bomb_over_pair + delta).clamp(1.0, 2000.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.wild_bomb_bonus = (mutated.wild_bomb_bonus + delta).clamp(1.0, 2000.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.wild_run_bonus = (mutated.wild_run_bonus + delta).clamp(1.0, 2000.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.endgame_single_removal = (mutated.endgame_single_removal + delta).clamp(1.0, 2000.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.endgame_small_single_removal = (mutated.endgame_small_single_removal + delta).clamp(1.0, 2000.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.empty_lead_bomb_penalty = (mutated.empty_lead_bomb_penalty + delta).clamp(1.0, 2000.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.split_penalty_scale = (mutated.split_penalty_scale + delta).clamp(5.0, 100.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.bomb_keep_many = (mutated.bomb_keep_many + delta).clamp(1.0, 2000.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.intercept_sprint_bonus = (mutated.intercept_sprint_bonus + delta).clamp(1.0, 2000.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.last_play_clear_bonus = (mutated.last_play_clear_bonus + delta).clamp(1.0, 2000.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.combo_shape_bonus = (mutated.combo_shape_bonus + delta).clamp(1.0, 2000.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.partner_feng_bonus = (mutated.partner_feng_bonus + delta).clamp(1.0, 2000.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.partner_feng_lead_bonus = (mutated.partner_feng_lead_bonus + delta).clamp(1.0, 2000.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.partner_feng_first_bonus = (mutated.partner_feng_first_bonus + delta).clamp(1.0, 2000.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.teammate_combo_bonus = (mutated.teammate_combo_bonus + delta).clamp(1.0, 2000.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.block_enemy_bonus = (mutated.block_enemy_bonus + delta).clamp(1.0, 2000.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.straight_build_bonus = (mutated.straight_build_bonus + delta).clamp(1.0, 2000.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.many_singles_penalty = (mutated.many_singles_penalty + delta).clamp(1.0, 2000.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.single_lead_bonus = (mutated.single_lead_bonus + delta).clamp(1.0, 2000.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.small_single_lead_bonus = (mutated.small_single_lead_bonus + delta).clamp(1.0, 2000.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.small_card_lead_bonus = (mutated.small_card_lead_bonus + delta).clamp(1.0, 2000.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.avoid_small_singles_each = (mutated.avoid_small_singles_each + delta).clamp(1.0, 2000.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.lead_len_step = (mutated.lead_len_step + delta).clamp(0.0, 100.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.lead_len_step_endgame = (mutated.lead_len_step_endgame + delta).clamp(0.0, 100.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.lead_primary_step = (mutated.lead_primary_step + delta).clamp(0.0, 50.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.lead_primary_step_endgame = (mutated.lead_primary_step_endgame + delta).clamp(0.0, 50.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.solver_junk_bonus = (mutated.solver_junk_bonus + delta).clamp(0.0, 500.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.dual_wild_penalty_mid = (mutated.dual_wild_penalty_mid + delta).clamp(1.0, 2000.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.dual_wild_penalty_end = (mutated.dual_wild_penalty_end + delta).clamp(1.0, 2000.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.upgraded_bomb_wild_mid = (mutated.upgraded_bomb_wild_mid + delta).clamp(1.0, 2000.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.upgraded_bomb_wild_end = (mutated.upgraded_bomb_wild_end + delta).clamp(1.0, 2000.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.wild_on_level_mid = (mutated.wild_on_level_mid + delta).clamp(1.0, 2000.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.wild_on_level_end = (mutated.wild_on_level_end + delta).clamp(1.0, 2000.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.wild_plain_pair_mid = (mutated.wild_plain_pair_mid + delta).clamp(1.0, 2000.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.wild_pair_penalty_end = (mutated.wild_pair_penalty_end + delta).clamp(1.0, 2000.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.bare_dual_wild_extra = (mutated.bare_dual_wild_extra + delta).clamp(1.0, 2000.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.keep_bomb_bonus = (mutated.keep_bomb_bonus + delta).clamp(1.0, 2000.0);
    }
    if rng.random_bool(r) {
        let delta = rng.random_range(-step_size..step_size);
        mutated.solver_trick_penalty = (mutated.solver_trick_penalty + delta).clamp(1.0, 2000.0);
    }

    if rng.random_bool(r) {
        let delta: i8 = if rng.random_bool(0.5) { 1 } else { -1 };
        mutated.partner_sprint_threshold = (mutated.partner_sprint_threshold as i16 + delta as i16).clamp(1, 6) as u8;
    }
    if rng.random_bool(r) {
        let delta: i8 = if rng.random_bool(0.5) { 1 } else { -1 };
        mutated.enemy_low_cards_threshold = (mutated.enemy_low_cards_threshold as i16 + delta as i16).clamp(1, 8) as u8;
    }
    if rng.random_bool(r) {
        let delta: i8 = if rng.random_bool(0.5) { 1 } else { -1 };
        mutated.endgame_hand_count_threshold = (mutated.endgame_hand_count_threshold as i16 + delta as i16).clamp(3, 12) as u8;
    }

    mutated
}

fn select_tournament(population: &[Individual], tournament_size: usize) -> &Individual {
    let mut rng = rand::rng();
    let mut best: Option<&Individual> = None;
    
    for _ in 0..tournament_size {
        let idx = rng.random_range(0..population.len());
        let candidate = &population[idx];
        if best.is_none() || candidate.score > best.as_ref().unwrap().score {
            best = Some(candidate);
        }
    }
    
    best.unwrap()
}

pub fn genetic_optimize(config: &GeneticConfig, base_params_path: Option<&str>, mut progress_cb: Option<Box<dyn FnMut(u32, u32, f32, f32, &str)>>) -> AdvancedBotParams {
    let mut rng = rand::rng();

    // 记录当前训练 generation,用于检测是否有新训练启动
    let my_gen = crate::learning::current_generation();

    let mut population: Vec<Individual> = Vec::with_capacity(config.population_size);
    
    let existing_params = base_params_path.and_then(|path| {
        AdvancedBotParams::load(path).ok().inspect(|_| {
            println!("[GA] Loaded existing params from {}", path);
        })
    });

    for i in 0..config.population_size {
        let base = if i == 0 && existing_params.is_some() {
            existing_params.clone().unwrap()
        } else if i == 0 || i == 1 {
            AdvancedBotParams::default_balanced()
        } else if i == 2 {
            AdvancedBotParams::default_aggressive()
        } else if i == 3 {
            AdvancedBotParams::default_supportive()
        } else if rng.random_bool(0.5) {
            AdvancedBotParams::default_balanced()
        } else {
            AdvancedBotParams::default_aggressive()
        };
        let mutation_rate = if i == 0 && existing_params.is_some() { 0.1 } else { 0.8 };
        let mutation_step = if i == 0 && existing_params.is_some() { 0.1 } else { 0.3 };
        let params = mutate(&base, mutation_rate, mutation_step);
        let eval_result = evaluate_individual(&params, config.matches_per_eval);
        let score = eval_to_score(&eval_result);
        population.push(Individual { params, score, eval_result });
    }
    
    population.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let mut best_score = population[0].score;
    let mut best_params = population[0].params.clone();
    
    if let Some(ref mut cb) = progress_cb {
        cb(0, 0, population[0].score, best_score, "Initial population evaluated");
    }
    
    println!("[GA] Generation 0: best score={:.4} (win={:.3}, first={:.3}, residual={:.1})", 
        best_score, population[0].eval_result.win_rate, population[0].eval_result.first_out_rate,
        population[0].eval_result.avg_endgame_residual);
    
    for generation in 1..=config.generations {
        // 检查是否被新训练取代或被用户停止
        if !crate::learning::is_running_generation(my_gen) {
            println!("[GA] Optimization stopped by user or superseded");
            break;
        }

        let mut new_population: Vec<Individual> = Vec::with_capacity(config.population_size);
        
        for i in 0..config.elitism_count {
            new_population.push(population[i].clone());
        }
        
        while new_population.len() < config.population_size {
            let parent1 = select_tournament(&population, 3);
            let parent2 = select_tournament(&population, 3);
            
            let child_params = if rand::rng().random_bool(config.crossover_rate as f64) {
                crossover(&parent1.params, &parent2.params)
            } else {
                parent1.params.clone()
            };
            
            let child_params = mutate(&child_params, config.mutation_rate, config.mutation_step_size);
            let eval_result = evaluate_individual(&child_params, config.matches_per_eval);
            let score = eval_to_score(&eval_result);
            
            new_population.push(Individual { 
                params: child_params, 
                score, 
                eval_result 
            });
        }
        
        new_population.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        population = new_population;
        
        if population[0].score > best_score {
            best_score = population[0].score;
            best_params = population[0].params.clone();
            println!("[GA] Generation {}: improved! best={:.4} (win={:.3}, first={:.3}, residual={:.1})", 
                generation, best_score, population[0].eval_result.win_rate, population[0].eval_result.first_out_rate,
                population[0].eval_result.avg_endgame_residual);
            if let Some(path) = base_params_path {
                if best_params.save(path).is_ok() {
                    println!("[GA]   -> checkpoint saved");
                }
            }
        } else {
            println!("[GA] Generation {}: best={:.4}", generation, best_score);
        }
        
        if let Some(ref mut cb) = progress_cb {
            let total_matches = (config.population_size as u32 * config.matches_per_eval)  // initial
                + generation as u32 * ((config.population_size - config.elitism_count) as u32) * config.matches_per_eval;
            cb(generation, total_matches, population[0].score, best_score, 
                &format!("Gen {}/{} - evaluating", generation, config.generations));
        }
    }
    
    println!("[GA] Optimization complete. Best score={:.4}", best_score);
    println!("[GA] Best params: {:?}", best_params);
    
    best_params
}
