use std::sync::{Mutex, OnceLock};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use std::fs;
use serde::{Deserialize, Serialize};
use uuid;

/// 全局训练 generation 计数器。
/// 每次 start_learning 递增,用于区分不同的训练任务。
/// 旧训练线程通过检查 generation 是否匹配来判断自己是否应该退出。
/// 解决"用户停止后立即重新开始训练,旧线程不退出"的问题:
///   - stop_learning 设 is_running=false(但不清除 guard)
///   - start_learning 递增 generation,旧线程发现 generation 不匹配即退出
static GENERATION: AtomicU64 = AtomicU64::new(0);

/// 获取当前训练 generation。
pub fn current_generation() -> u64 {
    GENERATION.load(Ordering::SeqCst)
}

/// 检查指定 generation 的训练是否仍在运行。
/// 如果 generation 不匹配(已有新的训练启动),返回 false。
/// 如果 generation 匹配,返回 is_running 状态(用户是否按了停止)。
pub fn is_running_generation(generation: u64) -> bool {
    if generation != current_generation() {
        return false; // 新训练已启动,旧训练应退出
    }
    is_running()
}

const STATE_FILE: &str = "/home/Cooki/domains/gg.meaigo.eu.org/clawguandan/learning_state.json";

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LearningStatus {
    pub is_running: bool,
    pub job_id: Option<String>,
    pub progress: f32,
    pub current_iteration: u32,
    pub total_iterations: u32,
    pub current_eval_score: f32,
    pub best_score: f32,
    pub matches_completed: u32,
    pub matches_total: u32,
    pub start_time: Option<String>,
    pub elapsed_seconds: u64,
    pub status_message: String,
    pub params_path: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SavedLearningState {
    pub job_id: String,
    pub matches_per_eval: u32,
    pub iterations: u32,
    pub output_path: String,
    pub current_iteration: u32,
    pub matches_completed: u32,
    pub current_eval_score: f32,
    pub best_score: f32,
    pub status_message: String,
    /// 训练模式:selfplay / genetic / record / fromlogs
    /// 旧状态文件可能没有此字段,默认为 "selfplay"
    #[serde(default = "default_mode")]
    pub mode: String,
    /// 遗传算法的种群大小(仅 genetic 模式用)
    #[serde(default = "default_population")]
    pub population_size: usize,
}

fn default_mode() -> String { "selfplay".to_string() }
fn default_population() -> usize { 8 }

#[derive(Debug)]
pub struct LearningTask {
    pub job_id: String,
    pub matches_per_eval: u32,
    pub iterations: u32,
    pub output_path: String,
    pub start_time: Instant,
    pub current_iteration: u32,
    pub matches_completed: u32,
    pub current_eval_score: f32,
    pub best_score: f32,
    pub status_message: String,
    pub is_running: bool,
    /// 训练模式
    pub mode: String,
    /// 遗传算法种群大小
    pub population_size: usize,
}

static LEARNING_TASK: OnceLock<Mutex<Option<LearningTask>>> = OnceLock::new();

pub fn init_task_manager() {
    LEARNING_TASK.get_or_init(|| Mutex::new(None));
    load_saved_state();
}

fn load_saved_state() {
    let state_file = std::path::Path::new(STATE_FILE);
    if !state_file.exists() {
        return;
    }

    let content = match fs::read_to_string(state_file) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[task_manager] Failed to read state file: {}", e);
            return;
        }
    };

    let saved_state: SavedLearningState = match serde_json::from_str(&content) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[task_manager] Failed to parse state file: {}", e);
            return;
        }
    };

    if let Some(task) = LEARNING_TASK.get() {
        if let Ok(mut guard) = task.lock() {
            *guard = Some(LearningTask {
                job_id: saved_state.job_id,
                matches_per_eval: saved_state.matches_per_eval,
                iterations: saved_state.iterations,
                output_path: saved_state.output_path,
                start_time: Instant::now(),
                current_iteration: saved_state.current_iteration,
                matches_completed: saved_state.matches_completed,
                current_eval_score: saved_state.current_eval_score,
                best_score: saved_state.best_score,
                status_message: format!("Recovered from previous session: {}", saved_state.status_message),
                is_running: false,
                mode: saved_state.mode,
                population_size: saved_state.population_size,
            });
            eprintln!("[task_manager] Loaded saved state: iteration {}, matches {}", 
                saved_state.current_iteration, saved_state.matches_completed);
        }
    }
}

fn save_state(task: &LearningTask) {
    let saved_state = SavedLearningState {
        job_id: task.job_id.clone(),
        matches_per_eval: task.matches_per_eval,
        iterations: task.iterations,
        output_path: task.output_path.clone(),
        current_iteration: task.current_iteration,
        matches_completed: task.matches_completed,
        current_eval_score: task.current_eval_score,
        best_score: task.best_score,
        status_message: task.status_message.clone(),
        mode: task.mode.clone(),
        population_size: task.population_size,
    };

    let json = match serde_json::to_string_pretty(&saved_state) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("[task_manager] Failed to serialize state: {}", e);
            return;
        }
    };

    if let Err(e) = fs::write(STATE_FILE, json) {
        eprintln!("[task_manager] Failed to write state file: {}", e);
    }
}

fn clear_state() {
    let state_file = std::path::Path::new(STATE_FILE);
    if state_file.exists() {
        if let Err(e) = fs::remove_file(state_file) {
            eprintln!("[task_manager] Failed to remove state file: {}", e);
        }
    }
}

pub fn start_learning(
    matches_per_eval: u32,
    iterations: u32,
    output_path: &str,
    mode: &str,
    population_size: usize,
) -> Result<String, String> {
    let task = LEARNING_TASK.get().ok_or("task manager not initialized")?;
    let mut guard = task.lock().map_err(|e| e.to_string())?;
    
    if guard.as_ref().map(|t| t.is_running).unwrap_or(false) {
        return Err("A learning task is already running".to_string());
    }
    
    // 递增 generation,使旧的训练线程能检测到自己已被取代
    GENERATION.fetch_add(1, Ordering::SeqCst);
    
    let state_file = std::path::Path::new(STATE_FILE);
    if state_file.exists() {
        let content = match fs::read_to_string(state_file) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[task_manager] Failed to read state file: {}", e);
                return Err(format!("Failed to read saved state: {}", e));
            }
        };

        let saved_state: SavedLearningState = match serde_json::from_str(&content) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[task_manager] Failed to parse state file: {}", e);
                return Err(format!("Failed to parse saved state: {}", e));
            }
        };

        let job_id = saved_state.job_id.clone();

        // 续传：使用 API 传入的新参数(matches_per_eval/iterations/mode/population_size),
        // 但保留已保存的训练进度(current_iteration/matches_completed/best_score)。
        // 这样停止后再开始时,如果用户在 UI 改了参数,会用新参数;服务器崩溃续传时
        // auto_resume 传入的也是旧参数,行为一致。
        let effective_iterations = iterations;
        let effective_matches = matches_per_eval;
        let effective_mode = mode.to_string();
        let effective_population = population_size;

        *guard = Some(LearningTask {
            job_id: saved_state.job_id,
            matches_per_eval: effective_matches,
            iterations: effective_iterations,
            output_path: saved_state.output_path,
            start_time: Instant::now(),
            current_iteration: saved_state.current_iteration,
            matches_completed: saved_state.matches_completed,
            current_eval_score: saved_state.current_eval_score,
            best_score: saved_state.best_score,
            status_message: "Resuming from saved progress...".to_string(),
            is_running: true,
            mode: effective_mode,
            population_size: effective_population,
        });

        eprintln!("[task_manager] Resuming saved task: iteration {}/{}, matches {} (API: {} matches/{} iters)", 
            saved_state.current_iteration, effective_iterations, saved_state.matches_completed,
            effective_matches, effective_iterations);
        
        return Ok(job_id);
    }
    
    let job_id = format!("learn_{}", uuid::Uuid::new_v4());
    *guard = Some(LearningTask {
        job_id: job_id.clone(),
        matches_per_eval,
        iterations,
        output_path: output_path.to_string(),
        start_time: Instant::now(),
        current_iteration: 0,
        matches_completed: 0,
        current_eval_score: 0.0,
        best_score: 0.0,
        status_message: "Starting learning...".to_string(),
        is_running: true,
        mode: mode.to_string(),
        population_size,
    });
    
    Ok(job_id)
}

pub fn update_progress(
    iteration: u32,
    matches_done: u32,
    eval_score: f32,
    best_score: f32,
    message: &str,
) {
    if let Some(task) = LEARNING_TASK.get() {
        if let Ok(mut guard) = task.lock() {
            if let Some(t) = guard.as_mut() {
                if !t.is_running {
                    return;
                }
                t.current_iteration = iteration;
                t.matches_completed = matches_done;
                t.current_eval_score = eval_score;
                t.best_score = best_score;
                t.status_message = message.to_string();
                save_state(t);
            }
        }
    }
}

pub fn finish_learning(success: bool, message: &str) {
    if let Some(task) = LEARNING_TASK.get() {
        if let Ok(mut guard) = task.lock() {
            if let Some(t) = guard.as_mut() {
                t.is_running = false;
                t.status_message = if success {
                    format!("Learning completed: {}", message)
                } else {
                    format!("Learning failed: {}", message)
                };
                save_state(t);
                if success {
                    clear_state();
                }
            }
            // 训练完成(或失败)后清除 LearningTask,允许启动新训练
            *guard = None;
        }
    }
}

pub fn get_status() -> LearningStatus {
    if let Some(task) = LEARNING_TASK.get() {
        if let Ok(guard) = task.lock() {
            if let Some(t) = guard.as_ref() {
                let elapsed = t.start_time.elapsed();
                let total_matches = t.matches_per_eval * t.iterations;
                let progress = if total_matches > 0 {
                    (t.matches_completed as f32) / (total_matches as f32)
                } else {
                    0.0
                };
                
                return LearningStatus {
                    is_running: t.is_running,
                    job_id: Some(t.job_id.clone()),
                    progress,
                    current_iteration: t.current_iteration,
                    total_iterations: t.iterations,
                    current_eval_score: t.current_eval_score,
                    best_score: t.best_score,
                    matches_completed: t.matches_completed,
                    matches_total: total_matches,
                    start_time: Some(format!("{}", t.start_time.elapsed().as_secs())),
                    elapsed_seconds: elapsed.as_secs(),
                    status_message: t.status_message.clone(),
                    params_path: Some(t.output_path.clone()),
                };
            }
        }
    }
    
    LearningStatus {
        is_running: false,
        job_id: None,
        progress: 0.0,
        current_iteration: 0,
        total_iterations: 0,
        current_eval_score: 0.0,
        best_score: 0.0,
        matches_completed: 0,
        matches_total: 0,
        start_time: None,
        elapsed_seconds: 0,
        status_message: "No learning task running".to_string(),
        params_path: None,
    }
}

pub fn stop_learning() -> Result<String, String> {
    let task = LEARNING_TASK.get().ok_or("task manager not initialized")?;
    let mut guard = task.lock().map_err(|e| e.to_string())?;
    
    if guard.is_none() {
        return Err("No learning task is running".to_string());
    }
    
    let job_id = guard.as_ref().unwrap().job_id.clone();
    if let Some(t) = guard.as_mut() {
        t.is_running = false;
        t.status_message = "Learning stopped by user".to_string();
        save_state(t);
    }
    // 不清除 *guard = None,保留 LearningTask 让旧训练线程能检测 is_running=false 并退出。
    // 配合 generation 机制,start_learning 递增 generation,旧线程通过 generation 不匹配退出。
    // 注意: save_state 保留 learning_state.json 用于断点续传。
    // 再次开始训练时,start_learning 会用 API 新参数(matches_per_eval/iterations)
    // 但保留已保存的进度(current_iteration/best_score),实现无缝续传。
    
    Ok(format!("Learning task {} stopped", job_id))
}

pub fn is_running() -> bool {
    if let Some(task) = LEARNING_TASK.get() {
        if let Ok(guard) = task.lock() {
            if let Some(t) = guard.as_ref() {
                return t.is_running;
            }
        }
    }
    false
}

/// 服务器重启后自动续传训练。
/// 检查 learning_state.json 是否存在且训练未完成 (current_iteration < iterations),
/// 如果是,则启动后台线程从断点继续训练。
/// 会在 init_task_manager() 之后被调用。
pub fn auto_resume() {
    let state_file = std::path::Path::new(STATE_FILE);
    if !state_file.exists() {
        return;
    }

    let content = match fs::read_to_string(state_file) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[task_manager] auto_resume: failed to read state file: {}", e);
            return;
        }
    };

    let saved_state: SavedLearningState = match serde_json::from_str(&content) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[task_manager] auto_resume: failed to parse state file: {}", e);
            return;
        }
    };

    // 训练已完成,不续传
    if saved_state.current_iteration >= saved_state.iterations {
        eprintln!("[task_manager] auto_resume: training already completed (iter {}/{})",
            saved_state.current_iteration, saved_state.iterations);
        return;
    }

    let matches_per_eval = saved_state.matches_per_eval;
    let iterations = saved_state.iterations;
    let mode = saved_state.mode.clone();
    let population_size = saved_state.population_size;
    let output_path = saved_state.output_path.clone();

    eprintln!("[task_manager] auto_resume: resuming '{}' training from iteration {}/{}",
        mode, saved_state.current_iteration, iterations);

    // 用 std::thread::spawn 启动训练线程(不依赖 tokio runtime)
    // 需要先调用 start_learning 恢复 task 状态,再启动训练
    std::thread::spawn(move || {
        match start_learning(matches_per_eval, iterations, &output_path, &mode, population_size) {
            Ok(_job_id) => {
                let result = match mode.as_str() {
                    "genetic" => crate::learning::run_genetic_learning_with_progress(
                        population_size, iterations, matches_per_eval, &output_path
                    ),
                    "record" => {
                        let logs_path = format!("{}/game_logs.jsonl",
                            std::path::Path::new(&output_path).parent()
                                .map(|p| p.to_string_lossy().to_string())
                                .unwrap_or_else(|| ".".to_string()));
                        crate::learning::run_record_learning_with_progress(
                            &logs_path, iterations, &output_path
                        )
                    },
                    "fromlogs" => {
                        let logs_path = format!("{}/game_logs.jsonl",
                            std::path::Path::new(&output_path).parent()
                                .map(|p| p.to_string_lossy().to_string())
                                .unwrap_or_else(|| ".".to_string()));
                        crate::learning::run_learning_from_logs(
                            &logs_path, iterations, &output_path
                        )
                    },
                    _ => crate::learning::run_learning_with_progress(
                        matches_per_eval, iterations, &output_path
                    ),
                };
                match result {
                    Ok(_) => {
                        eprintln!("[task_manager] auto_resume: training completed successfully");
                        crate::learning::finish_learning(true, "Training completed successfully (auto-resumed)");
                    }
                    Err(e) => {
                        eprintln!("[task_manager] auto_resume: training failed: {}", e);
                        crate::learning::finish_learning(false, &e);
                    }
                }
            }
            Err(e) => {
                eprintln!("[task_manager] auto_resume: failed to start learning: {}", e);
            }
        }
    });
}