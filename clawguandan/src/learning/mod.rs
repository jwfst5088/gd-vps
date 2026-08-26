//! AI self-learning module: parameter optimization via self-play and human-game analysis.
//!
//! Supports multiple learning methods to tune `AdvancedBotParams`:
//! 1. Hill-climbing: simple local search via self-play
//! 2. Genetic algorithm: population-based optimization for global search
//! 3. Human-game analysis: learning from recorded matches with human players
//!
//! Optimized params are saved to `advanced_params.json` and loaded by the bot plugin at runtime.

mod optimizer;
mod genetic_optimizer;
mod record_learner;
pub mod game_logger;
pub mod human_stats;
mod task_manager;
pub use optimizer::{HillClimbConfig, optimize, EvalResult, SelfPlayConfig};
pub use genetic_optimizer::{GeneticConfig, genetic_optimize};
pub use record_learner::{analyze_logs, patterns_to_params, learn_from_logs, LogAnalysis};
pub use game_logger::{GameLogger, GameLogEntry, GameAction, log_game};
pub use task_manager::{
    init_task_manager, start_learning, update_progress, finish_learning, 
    get_status, stop_learning, is_running, is_running_generation, current_generation,
    auto_resume, LearningStatus
};

use crate::domain::Seat;
use crate::game::types::{GamePhase, TeamId};
use crate::game::engine::GameEngine;
use crate::bot::plugins::AdvancedBotParams;
use crate::simulation::run_match_engine;
use crate::strategy::suggest::set_learn_params_for_teams;

/// Run the full learning pipeline: optimize parameters via self-play and save to file.
pub fn run_learning(
    matches_per_eval: u32,
    iterations: u32,
    output_path: &str,
) -> Result<(), String> {
    run_learning_with_progress(matches_per_eval, iterations, output_path)
}

/// Run learning with progress updates for web UI.
pub fn run_learning_with_progress(
    matches_per_eval: u32,
    iterations: u32,
    output_path: &str,
) -> Result<(), String> {
    let eval_config = SelfPlayConfig {
        matches_per_eval,
        max_plies: 2000,
    };

    let climb_config = HillClimbConfig {
        iterations,
        step_size: 0.05,
        eval_config: eval_config.clone(),
    };

    let total_matches = matches_per_eval * iterations * 2; // current + candidate per iteration

    let start = AdvancedBotParams::load(output_path).unwrap_or_else(|_| {
        println!("[learn] No existing params found at {}, starting from default balanced", output_path);
        AdvancedBotParams::default_balanced()
    });
    println!("[learn] Starting hill-climb optimization via self-play...");
    println!("[learn] Matches per evaluation: {matches_per_eval}");
    println!("[learn] Iterations: {iterations}");
    println!("[learn] Total matches: ~{total_matches}");

    let mut start_iter = 0u32;
    let mut total_matches_done = 0u32;
    let mut resumed_best_score = 0.0f32;
    let mut has_resumed_state = false;

    let state_file = std::path::Path::new("./learning_state.json");
    if state_file.exists() {
        if let Ok(content) = std::fs::read_to_string(state_file) {
            if let Ok(saved_state) = serde_json::from_str::<crate::learning::task_manager::SavedLearningState>(&content) {
                start_iter = saved_state.current_iteration;
                total_matches_done = saved_state.matches_completed;
                resumed_best_score = saved_state.best_score;
                has_resumed_state = true;
                if start_iter > 0 {
                    println!("[learn] Resuming from iteration {}, matches done: {}, best_score: {:.4}",
                        start_iter, total_matches_done, resumed_best_score);
                }
            }
        }
    }

    update_progress(start_iter, total_matches_done, 0.0, resumed_best_score, "Starting optimization...");

    // 续传时使用已保存的 best_score,而不是重置为 0.0
    // 这样可以避免续传后第一轮 current_score 与 0 比较,导致保存退步的参数。
    // best 参数从 advanced_params.json 加载(上次训练的最优 checkpoint),
    // 如果文件不存在(尚未产生 checkpoint),则用 default_balanced。
    let mut best = start.clone();
    let mut best_score = if has_resumed_state {
        resumed_best_score
    } else {
        0.0
    };

    // 记录当前训练 generation,用于检测是否有新训练启动(用户重新开始训练)
    // 如果 generation 不匹配,说明 start_learning 被调用过(新训练启动),本线程应退出
    let my_generation = current_generation();

    for i in start_iter..iterations {
        if !is_running_generation(my_generation) {
            println!("[learn] Learning stopped by user or superseded by new training");
            update_progress(i, total_matches_done, 0.0, best_score, "Stopped");
            return Err("Stopped by user".to_string());
        }
        
        let iter_num = i + 1;
        let current_eval = optimizer::evaluate_params_with_progress(
            &best, &eval_config,
            Some(|done: u32, batch_total: u32| {
                let global_done = total_matches_done + done;
                let progress = (global_done as f32) / (total_matches as f32) * 100.0;
                update_progress(
                    iter_num,
                    global_done,
                    0.0,
                    best_score,
                    &format!("Iter {}/{} [1/2] evaluating ({}/{})", iter_num, iterations, done, batch_total),
                );
            }),
        );
        let current_score = optimizer::eval_to_score(&current_eval);
        total_matches_done += matches_per_eval;
        
        // 修复:续传时不应无条件接受第一轮的分数作为 best_score。
        // 原代码 `i == 0 || current_score > best_score` 在续传场景下(start_iter > 0)
        // 不会触发(i 从 start_iter 开始),但如果是从头开始且 best_score 已恢复,
        // 也应该用严格的 > 比较而非无条件接受。
        // 只有在全新训练(best_score 仍为 0.0 且非续传)时,第一轮才无条件接受。
        if (!has_resumed_state && i == 0) || current_score > best_score {
            best_score = current_score;
        }
        
        update_progress(
            iter_num,
            total_matches_done,
            current_score,
            best_score,
            &format!("Iter {}/{} - score: {:.4}", iter_num, iterations, current_score),
        );

        println!("[learn] iter {}: score={:.4} (best={:.4})", iter_num, current_score, best_score);

        if i < iterations - 1 && is_running_generation(my_generation) {
            let candidate = best.mutate_random(climb_config.step_size);
            let candidate_eval = optimizer::evaluate_params_with_progress(
                &candidate, &eval_config,
                Some(|done: u32, batch_total: u32| {
                    let global_done = total_matches_done + done;
                    update_progress(
                        iter_num,
                        global_done,
                        0.0,
                        best_score,
                        &format!("Iter {}/{} [2/2] candidate eval ({}/{})", iter_num, iterations, done, batch_total),
                    );
                }),
            );
            let candidate_score = optimizer::eval_to_score(&candidate_eval);
            total_matches_done += matches_per_eval;
            
            if candidate_score > best_score {
                best = candidate;
                best_score = candidate_score;
                println!("[learn]   -> improved! new best score={:.4}", best_score);
                update_progress(
                    iter_num,
                    total_matches_done,
                    candidate_score,
                    best_score,
                    &format!("Iter {}/{} - improved! best={:.4}", iter_num, iterations, best_score),
                );
                if best.save(output_path).is_ok() {
                    println!("[learn]   -> checkpoint saved");
                }
            }
        }
    }

    update_progress(
        iterations,
        total_matches_done,
        best_score,
        best_score,
        "Saving params...",
    );

    println!("\n[learn] Self-play optimization complete. Saving to {output_path}");
    best.save(output_path)?;

    println!("[learn] Best params: {:?}", best);
    println!("[learn] Done.");

    Ok(())
}

/// Run learning from recorded game logs (human vs bot matches).
pub fn run_learning_from_logs(
    log_path: &str,
    iterations: u32,
    output_path: &str,
) -> Result<(), String> {

    let logs = game_logger::GameLogger::read_logs(log_path)?;
    if logs.is_empty() {
        return Err("No game logs found".to_string());
    }

    println!("[learn] Starting optimization from game logs...");
    println!("[learn] Logs loaded: {}", logs.len());
    println!("[learn] Iterations: {}", iterations);

    let start = AdvancedBotParams::default_balanced();
    let best = optimize_from_logs(&start, iterations, &logs);

    println!("\n[learn] Log-based optimization complete. Saving to {output_path}");
    best.save(output_path)?;

    println!("[learn] Best params: {:?}", best);
    println!("[learn] Done.");

    Ok(())
}

fn optimize_from_logs(
    start: &AdvancedBotParams,
    iterations: u32,
    logs: &[GameLogEntry],
) -> AdvancedBotParams {
    let mut best = start.clone();
    let mut best_score = evaluate_params_from_logs(&best, logs);
    println!("[learn] iter 0: score={:.4}", best_score);

    for i in 1..=iterations {
        let candidate = best.mutate_random(0.15);
        let score = evaluate_params_from_logs(&candidate, logs);

        println!("[learn] iter {i}: score={:.4} (best={:.4})", score, best_score);

        if score > best_score {
            best = candidate;
            best_score = score;
            println!("[learn]   -> improved! new best score={:.4}", best_score);
        }
    }

    best
}

fn evaluate_params_from_logs(params: &AdvancedBotParams, logs: &[GameLogEntry]) -> f32 {
    use crate::game::types::GameConfig;
    
    // 打破对称性：NS 用候选参数，EW 用固定基线参数(default_balanced)。
    // 与 optimizer.rs / genetic_optimizer.rs 保持一致。
    let baseline = AdvancedBotParams::default_balanced();
    set_learn_params_for_teams(Some(params.clone()), Some(baseline));

    let mut ns_wins = 0u32;
    let mut ns_first_out = 0u32;
    let mut ns_residual_sum: usize = 0;
    let mut played = 0u32;

    let matches_to_run = std::cmp::min(logs.len() as u32, 50);
    for _ in 0..matches_to_run {
        let engine = GameEngine::new(GameConfig { rng_seed: rand::random() });
        match run_single_match_from_log(&engine, params) {
            Ok(Some((winner, first_out_team, ns_residual))) => {
                played += 1;
                if winner == TeamId::Sn {
                    ns_wins += 1;
                }
                if first_out_team == TeamId::Sn {
                    ns_first_out += 1;
                }
                ns_residual_sum += ns_residual;
            }
            _ => {
                played += 1;
                // 失败比赛按最差情况计：NS两人满手牌未清(27)
                // 避免失败比赛被当作0张残牌而人为抬高clear_rate评分
                ns_residual_sum += 27;
            }
        }
    }

    set_learn_params_for_teams(None, None);

    if played == 0 {
        return 0.5;
    }

    let win_rate = ns_wins as f32 / played as f32;
    let first_out_rate = ns_first_out as f32 / played as f32;
    let avg_endgame_residual = ns_residual_sum as f32 / played as f32;
    // 与 optimizer.rs::eval_to_score 保持一致：胜率0.5 + 头游0.2 + 残局清牌0.3
    let clear_rate = (1.0 - (avg_endgame_residual / 27.0).clamp(0.0, 1.0)).max(0.0);
    win_rate * 0.5 + first_out_rate * 0.2 + clear_rate * 0.3
}

fn run_single_match_from_log(
    engine: &GameEngine,
    _params: &AdvancedBotParams,
) -> Result<Option<(TeamId, TeamId, usize)>, String> {
    use crate::game::card::HandLevel;
    
    let mut state = engine.init_table(format!("learn_{}", uuid::Uuid::new_v4()));
    let first_drawer = Seat::S;
    engine
        .start_first_hand(&mut state, first_drawer, HandLevel::Two)
        .map_err(|e| format!("start_first_hand: {e}"))?;

    let outcome = run_match_engine(engine, &mut state, 1, 2000)
        .map_err(|e| format!("run_match: {e}"))?;

    if outcome.final_phase == GamePhase::Scoring {
        if let Some(winner) = state.winner_team {
            let first_out_team = state.hand
                .as_ref()
                .and_then(|h| h.finishing_order.first().copied())
                .map(|seat| match seat {
                    Seat::E | Seat::W => TeamId::Ew,
                    Seat::S | Seat::N => TeamId::Sn,
                })
                .unwrap_or(winner);
            // NS队结束时剩余手牌总数(S+N)：衡量残局剩牌表现
            let ns_residual = state
                .hand
                .as_ref()
                .map(|h| {
                    h.hands.get(&Seat::S).map(|v| v.len()).unwrap_or(0)
                        + h.hands.get(&Seat::N).map(|v| v.len()).unwrap_or(0)
                })
                .unwrap_or(0);
            Ok(Some((winner, first_out_team, ns_residual)))
        } else {
            Ok(None)
        }
    } else {
        Ok(None)
    }
}

/// Run genetic algorithm optimization via self-play.
pub fn run_genetic_learning(
    population_size: usize,
    generations: u32,
    matches_per_eval: u32,
    output_path: &str,
) -> Result<(), String> {
    run_genetic_learning_with_progress(population_size, generations, matches_per_eval, output_path)
}

/// Run genetic algorithm optimization with progress updates.
pub fn run_genetic_learning_with_progress(
    population_size: usize,
    generations: u32,
    matches_per_eval: u32,
    output_path: &str,
) -> Result<(), String> {
    let config = GeneticConfig {
        population_size,
        generations,
        matches_per_eval,
        ..GeneticConfig::default()
    };

    let total_matches = (population_size as u32 * matches_per_eval)  // initial population
        + generations * ((population_size - config.elitism_count) as u32) * matches_per_eval;  // per generation
    
    update_progress(0, 0, 0.0, 0.0, "Starting genetic algorithm...");
    
    println!("[GA] Starting genetic algorithm optimization...");
    println!("[GA] Population size: {}", population_size);
    println!("[GA] Generations: {}", generations);
    println!("[GA] Matches per evaluation: {}", matches_per_eval);
    println!("[GA] Total matches: ~{}", total_matches);

    let best = genetic_optimize(&config, Some(output_path), Some(Box::new(|generation, matches_done, current_score, best_score, msg| {
        update_progress(generation, matches_done, current_score, best_score, msg);
    })));

    update_progress(generations, total_matches, 0.0, 0.0, "Saving params...");
    
    println!("\n[GA] Genetic optimization complete. Saving to {output_path}");
    best.save(output_path)?;
    
    println!("[GA] Best params: {:?}", best);
    println!("[GA] Done.");
    
    Ok(())
}

/// Run learning from recorded game logs with progress updates.
pub fn run_record_learning_with_progress(
    log_path: &str,
    iterations: u32,
    output_path: &str,
) -> Result<(), String> {
    let logs = game_logger::GameLogger::read_logs(log_path)?;
    if logs.is_empty() {
        return Err("No game logs found".to_string());
    }

    update_progress(0, 0, 0.0, 0.0, "Analyzing game logs...");
    
    println!("[record_learner] Starting learning from game logs...");
    println!("[record_learner] Logs loaded: {}", logs.len());
    println!("[record_learner] Iterations: {}", iterations);

    let analysis = analyze_logs(&logs);
    
    println!("[record_learner] Human wins: {}, losses: {}", 
        analysis.human_wins, analysis.human_losses);

    let pattern_params = patterns_to_params(&analysis);
    
    let best_params = AdvancedBotParams::load(output_path).unwrap_or_else(|_| {
        println!("[record_learner] No existing params found, using pattern-based params");
        pattern_params
    });
    
    update_progress(1, logs.len() as u32, 0.0, 0.0, "Optimizing from patterns...");

    let eval_config = SelfPlayConfig {
        matches_per_eval: 30,
        max_plies: 2000,
    };
    
    let mut best = best_params;
    let mut best_score = optimizer::eval_to_score(&optimizer::evaluate_params(&best, &eval_config));
    
    let my_generation = current_generation();

    for i in 1..=iterations {
        if !is_running_generation(my_generation) {
            update_progress(i, logs.len() as u32, best_score, best_score, "Stopped");
            return Err("Stopped by user".to_string());
        }
        
        let candidate = best.mutate_random(0.1);
        let eval = optimizer::evaluate_params(&candidate, &eval_config);
        let score = optimizer::eval_to_score(&eval);
        
        if score > best_score {
            best = candidate;
            best_score = score;
        }
        
        update_progress(i, logs.len() as u32, score, best_score, 
            &format!("Iter {}/{}", i, iterations));
    }

    update_progress(iterations, logs.len() as u32, best_score, best_score, "Saving params...");
    
    println!("\n[record_learner] Record-based learning complete. Saving to {output_path}");
    best.save(output_path)?;
    
    println!("[record_learner] Best params: {:?}", best);
    println!("[record_learner] Done.");
    
    Ok(())
}