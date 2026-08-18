use crate::bot::plugins::AdvancedBotParams;
use crate::domain::Seat;
use crate::game::types::{TeamId, GamePhase};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GameAction {
    pub seat: String,
    pub seat_team: String,
    pub action_type: String,
    pub cards: Option<Vec<String>>,
    pub seq: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GameLogEntry {
    pub game_id: String,
    pub table_id: String,
    pub start_time: String,
    pub end_time: String,
    pub finishing_order: Vec<String>,
    pub winner_team: String,
    pub bot_seats: Vec<String>,
    pub human_seats: Vec<String>,
    pub bot_params: Option<AdvancedBotParams>,
    pub actions: Vec<GameAction>,
    pub hand_level: String,
}

static GAME_LOG_PATH: &str = "./game_logs.jsonl";

pub struct GameLogger {
    entries: Vec<GameLogEntry>,
    flush_interval: usize,
}

impl Default for GameLogger {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            flush_interval: 10,
        }
    }
}

impl GameLogger {
    pub fn new(flush_interval: usize) -> Self {
        Self {
            entries: Vec::new(),
            flush_interval,
        }
    }

    pub fn log_game(&mut self, entry: GameLogEntry) {
        self.entries.push(entry);
        if self.entries.len() >= self.flush_interval {
            self.flush().ok();
        }
    }

    pub fn flush(&mut self) -> Result<(), String> {
        if self.entries.is_empty() {
            return Ok(());
        }

        let path = Path::new(GAME_LOG_PATH);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| e.to_string())?;

        let mut writer = BufWriter::new(file);
        for entry in &self.entries {
            let line = serde_json::to_string(entry).map_err(|e| e.to_string())?;
            writeln!(writer, "{line}").map_err(|e| e.to_string())?;
        }
        writer.flush().map_err(|e| e.to_string())?;

        self.entries.clear();
        Ok(())
    }

    pub fn read_logs(path: &str) -> Result<Vec<GameLogEntry>, String> {
        let content = fs::read_to_string(path).map_err(|e| e.to_string())?;
        let mut entries = Vec::new();
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let entry: GameLogEntry = serde_json::from_str(line).map_err(|e| e.to_string())?;
            entries.push(entry);
        }
        Ok(entries)
    }
}

pub fn seat_to_team_str(seat: Seat) -> String {
    match seat {
        Seat::E | Seat::W => "EW".to_string(),
        Seat::S | Seat::N => "SN".to_string(),
    }
}

pub fn team_to_str(team: TeamId) -> String {
    match team {
        TeamId::Ew => "EW".to_string(),
        TeamId::Sn => "SN".to_string(),
    }
}

use std::sync::{Mutex, OnceLock};

static GLOBAL_LOGGER: OnceLock<Mutex<GameLogger>> = OnceLock::new();

pub fn init_global_logger(flush_interval: usize) {
    GLOBAL_LOGGER.get_or_init(|| Mutex::new(GameLogger::new(flush_interval)));
}

pub fn log_game(entry: GameLogEntry) {
    if let Some(logger) = GLOBAL_LOGGER.get() {
        if let Ok(mut guard) = logger.lock() {
            guard.log_game(entry);
        }
    }
}

pub fn flush_global_logger() -> Result<(), String> {
    if let Some(logger) = GLOBAL_LOGGER.get() {
        if let Ok(mut guard) = logger.lock() {
            guard.flush()
        } else {
            Ok(())
        }
    } else {
        Ok(())
    }
}