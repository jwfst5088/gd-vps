use crate::bot::plugins::AdvancedBotParams;
use crate::learning::game_logger::GameLogEntry;
use crate::game::card::{parse_card_symbol, RuleContext, HandLevel, natural_rank_value};
use crate::game::rules::combination_parser::{CombinationParser, CombinationClass};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
pub struct PlayerPatterns {
    pub avg_hand_size_at_first_play: f32,
    pub bomb_usage_rate: f32,
    pub pass_when_partner_leads_rate: f32,
    pub follow_partner_small_card_rate: f32,
    pub intercept_enemy_small_card_rate: f32,
    pub bomb_when_enemy_low_cards_rate: f32,
    pub save_bomb_for_endgame_rate: f32,
    pub level_card_usage_rate: f32,
    pub combo_follow_rate: f32,
    pub proactive_play_rate: f32,
}

impl Default for PlayerPatterns {
    fn default() -> Self {
        Self {
            avg_hand_size_at_first_play: 18.0,
            bomb_usage_rate: 0.3,
            pass_when_partner_leads_rate: 0.6,
            follow_partner_small_card_rate: 0.7,
            intercept_enemy_small_card_rate: 0.8,
            bomb_when_enemy_low_cards_rate: 0.9,
            save_bomb_for_endgame_rate: 0.6,
            level_card_usage_rate: 0.4,
            combo_follow_rate: 0.5,
            proactive_play_rate: 0.5,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LogAnalysis {
    pub winning_patterns: PlayerPatterns,
    pub losing_patterns: PlayerPatterns,
    pub human_wins: u32,
    pub human_losses: u32,
    pub total_games: u32,
}

fn is_bomb_action(cards: &[String], hand_level: &str) -> bool {
    if cards.is_empty() {
        return false;
    }
    let ctx = RuleContext { hand_level: HandLevel::from_api_str(hand_level).unwrap_or(HandLevel::Two) };
    match CombinationParser::parse(cards, None, ctx) {
        Ok(combo) => matches!(combo.class(), CombinationClass::Bomb),
        Err(_) => false,
    }
}

fn is_small_card_action(cards: &[String], hand_level: &str) -> bool {
    if cards.is_empty() {
        return false;
    }
    let level_val = HandLevel::from_api_str(hand_level)
        .map(|hl| natural_rank_value(hl.to_rank()).unwrap_or(2))
        .unwrap_or(2);
    
    for card in cards {
        if let Ok(c) = parse_card_symbol(card) {
            let val = natural_rank_value(c.rank).unwrap_or(0) as u32;
            let level_u32 = level_val as u32;
            if val > 10 || val == level_u32 {
                return false;
            }
        }
    }
    true
}

fn seat_team_str(seat: &str) -> &str {
    if seat == "E" || seat == "W" { "EW" } else { "SN" }
}

fn is_partner(seat_a: &str, seat_b: &str) -> bool {
    seat_team_str(seat_a) == seat_team_str(seat_b)
}

fn analyze_game_log(entry: &GameLogEntry) -> (bool, PlayerPatterns) {
    let is_human_win = !entry.human_seats.is_empty() && {
        let human_teams: HashSet<String> = entry.human_seats.iter()
            .map(|s| seat_team_str(s).to_string())
            .collect();
        human_teams.contains(&entry.winner_team)
    };

    let mut patterns = PlayerPatterns::default();
    if entry.human_seats.is_empty() || entry.actions.is_empty() {
        return (is_human_win, patterns);
    }

    let human_set: HashSet<&str> = entry.human_seats.iter().map(|s| s.as_str()).collect();

    // Track hand sizes for all seats during iteration
    let mut hand_sizes: HashMap<&str, usize> = HashMap::new();
    for seat in ["E", "S", "W", "N"] {
        hand_sizes.insert(seat, 27);
    }

    let mut bomb_count = 0u32;
    let mut total_play_actions = 0u32;
    let mut level_card_actions = 0u32;
    let mut first_play_hand_sizes: Vec<usize> = Vec::new();
    let mut seat_has_played: HashSet<&str> = HashSet::new();

    // Partner leading patterns
    let mut pass_when_partner = 0u32;
    let mut total_when_partner = 0u32;
    let mut follow_partner_small = 0u32;
    let mut total_partner_small = 0u32;
    let mut combo_follow = 0u32;
    let mut total_follow_chance = 0u32;

    // Enemy leading patterns
    let mut intercept_enemy_small = 0u32;
    let mut total_enemy_small = 0u32;
    let mut bomb_when_enemy_low = 0u32;
    let mut total_enemy_low = 0u32;

    // Endgame bomb saving
    let mut bomb_when_self_low = 0u32;
    let mut total_self_low = 0u32;

    // Track top play (the last non-pass action before current)
    let mut top_play_seat: Option<String> = None;
    let mut top_play_cards: Option<Vec<String>> = None;

    for action in &entry.actions {
        let is_human = human_set.contains(action.seat.as_str());

        // Update hand size when a play happens
        if action.action_type == "play" {
            if let Some(cards) = &action.cards {
                if let Some(sz) = hand_sizes.get_mut(action.seat.as_str()) {
                    *sz = sz.saturating_sub(cards.len());
                }
            }
        }

        if !is_human {
            // Non-human action: if it's a play, update top play
            if action.action_type == "play" {
                top_play_seat = Some(action.seat.clone());
                top_play_cards = action.cards.clone();
            } else if action.action_type == "pass" {
                // pass doesn't change top play
            }
            continue;
        }

        // Human action analysis
        if action.action_type == "play" {
            total_play_actions += 1;

            if !seat_has_played.contains(action.seat.as_str()) {
                if let Some(sz) = hand_sizes.get(action.seat.as_str()) {
                    first_play_hand_sizes.push(*sz);
                }
                seat_has_played.insert(action.seat.as_str());
            }

            if let Some(cards) = &action.cards {
                if is_bomb_action(cards, &entry.hand_level) {
                    bomb_count += 1;
                }

                let level_rank = HandLevel::from_api_str(&entry.hand_level).map(|hl| hl.to_rank());
                let has_level = cards.iter().any(|c| {
                    if let Ok(card) = parse_card_symbol(c) {
                        level_rank.map(|lr| card.rank == lr).unwrap_or(false)
                    } else {
                        false
                    }
                });
                if has_level {
                    level_card_actions += 1;
                }
            }

            // Analyze follow behavior based on top play
            if let (Some(top_seat), Some(top_cards)) = (&top_play_seat, &top_play_cards) {
                let top_is_partner = is_partner(&action.seat, top_seat);

                total_follow_chance += 1;

                if top_is_partner {
                    // Partner is leading
                    total_when_partner += 1;

                    // Did human follow with combo (same kind)?
                    // Simple heuristic: if played same number of cards, consider it a combo follow
                    let my_card_count = action.cards.as_ref().map(|c| c.len()).unwrap_or(0);
                    if top_cards.len() == my_card_count && my_card_count > 0 {
                        combo_follow += 1;
                    }

                    // Check if partner led with small card
                    if is_small_card_action(top_cards, &entry.hand_level) {
                        total_partner_small += 1;
                        // Human followed with a play (not pass) = followed partner's small card
                        follow_partner_small += 1;
                    }
                } else {
                    // Enemy is leading
                    let human_team = seat_team_str(&action.seat);
                    let enemy_hand_low = hand_sizes.iter()
                        .any(|(seat, &sz)| seat_team_str(seat) != human_team && sz > 0 && sz <= 3);

                    if is_small_card_action(top_cards, &entry.hand_level) {
                        total_enemy_small += 1;
                        // Human played (not pass) = intercepted enemy small card
                        intercept_enemy_small += 1;
                    }

                    if enemy_hand_low {
                        total_enemy_low += 1;
                        if is_bomb_action(action.cards.as_deref().unwrap_or(&[]), &entry.hand_level) {
                            bomb_when_enemy_low += 1;
                        }
                    }
                }
            }

            // Update top play to current
            top_play_seat = Some(action.seat.clone());
            top_play_cards = action.cards.clone();
        } else if action.action_type == "pass" {
            // Human passed
            if let Some(top_seat) = &top_play_seat {
                let top_is_partner = is_partner(&action.seat, top_seat);

                if top_is_partner {
                    total_when_partner += 1;
                    pass_when_partner += 1;

                    if let Some(top_cards) = &top_play_cards {
                        if is_small_card_action(top_cards, &entry.hand_level) {
                            total_partner_small += 1;
                            // Passed on partner's small card = did NOT follow
                        }
                    }
                } else {
                    if let Some(top_cards) = &top_play_cards {
                        if is_small_card_action(top_cards, &entry.hand_level) {
                            total_enemy_small += 1;
                            // Passed on enemy small card = did NOT intercept
                        }
                    }
                }
            }
        }

        // Endgame bomb saving: track when human has few cards
        let my_hand = *hand_sizes.get(action.seat.as_str()).unwrap_or(&27);
        if my_hand <= 5 && my_hand > 0 {
            total_self_low += 1;
            if action.action_type == "play" {
                if let Some(cards) = &action.cards {
                    if is_bomb_action(cards, &entry.hand_level) {
                        bomb_when_self_low += 1;
                    }
                }
            }
        }
    }

    // Compute patterns
    patterns.avg_hand_size_at_first_play = if first_play_hand_sizes.is_empty() {
        18.0
    } else {
        first_play_hand_sizes.iter().sum::<usize>() as f32 / first_play_hand_sizes.len() as f32
    };

    patterns.bomb_usage_rate = if total_play_actions > 0 {
        bomb_count as f32 / total_play_actions as f32
    } else {
        0.3
    };

    patterns.level_card_usage_rate = if total_play_actions > 0 {
        level_card_actions as f32 / total_play_actions as f32
    } else {
        0.4
    };

    patterns.pass_when_partner_leads_rate = if total_when_partner > 0 {
        pass_when_partner as f32 / total_when_partner as f32
    } else {
        0.6
    };

    patterns.follow_partner_small_card_rate = if total_partner_small > 0 {
        follow_partner_small as f32 / total_partner_small as f32
    } else {
        0.7
    };

    patterns.intercept_enemy_small_card_rate = if total_enemy_small > 0 {
        intercept_enemy_small as f32 / total_enemy_small as f32
    } else {
        0.8
    };

    patterns.bomb_when_enemy_low_cards_rate = if total_enemy_low > 0 {
        bomb_when_enemy_low as f32 / total_enemy_low as f32
    } else {
        0.9
    };

    // save_bomb_for_endgame: inverse of bomb_when_self_low (lower = more saving)
    patterns.save_bomb_for_endgame_rate = if total_self_low > 0 {
        1.0 - (bomb_when_self_low as f32 / total_self_low as f32)
    } else {
        0.6
    };

    patterns.combo_follow_rate = if total_follow_chance > 0 {
        combo_follow as f32 / total_follow_chance as f32
    } else {
        0.5
    };

    patterns.proactive_play_rate = if total_play_actions > 0 {
        // Proactive = play actions where human was the top_play_seat (leading) vs following
        let mut leading = 0u32;
        let mut top_seat: Option<String> = None;
        for a in &entry.actions {
            if !human_set.contains(a.seat.as_str()) {
                if a.action_type == "play" {
                    top_seat = Some(a.seat.clone());
                }
                continue;
            }
            if a.action_type == "play" {
                if top_seat.is_none() || top_seat.as_deref() == Some(a.seat.as_str()) {
                    leading += 1;
                }
                top_seat = Some(a.seat.clone());
            }
        }
        leading as f32 / total_play_actions as f32
    } else {
        0.5
    };

    (is_human_win, patterns)
}

pub fn analyze_logs(logs: &[GameLogEntry]) -> LogAnalysis {
    let mut winning_patterns_sum = PlayerPatterns::default();
    let mut losing_patterns_sum = PlayerPatterns::default();
    let mut human_wins = 0u32;
    let mut human_losses = 0u32;
    let mut win_count = 0usize;
    let mut loss_count = 0usize;

    for entry in logs {
        if entry.human_seats.is_empty() {
            continue;
        }
        
        let (is_human_win, patterns) = analyze_game_log(entry);
        
        if is_human_win {
            human_wins += 1;
            win_count += 1;
            
            winning_patterns_sum.avg_hand_size_at_first_play += patterns.avg_hand_size_at_first_play;
            winning_patterns_sum.bomb_usage_rate += patterns.bomb_usage_rate;
            winning_patterns_sum.pass_when_partner_leads_rate += patterns.pass_when_partner_leads_rate;
            winning_patterns_sum.follow_partner_small_card_rate += patterns.follow_partner_small_card_rate;
            winning_patterns_sum.intercept_enemy_small_card_rate += patterns.intercept_enemy_small_card_rate;
            winning_patterns_sum.bomb_when_enemy_low_cards_rate += patterns.bomb_when_enemy_low_cards_rate;
            winning_patterns_sum.save_bomb_for_endgame_rate += patterns.save_bomb_for_endgame_rate;
            winning_patterns_sum.level_card_usage_rate += patterns.level_card_usage_rate;
            winning_patterns_sum.combo_follow_rate += patterns.combo_follow_rate;
            winning_patterns_sum.proactive_play_rate += patterns.proactive_play_rate;
        } else {
            human_losses += 1;
            loss_count += 1;
            
            losing_patterns_sum.avg_hand_size_at_first_play += patterns.avg_hand_size_at_first_play;
            losing_patterns_sum.bomb_usage_rate += patterns.bomb_usage_rate;
            losing_patterns_sum.pass_when_partner_leads_rate += patterns.pass_when_partner_leads_rate;
            losing_patterns_sum.follow_partner_small_card_rate += patterns.follow_partner_small_card_rate;
            losing_patterns_sum.intercept_enemy_small_card_rate += patterns.intercept_enemy_small_card_rate;
            losing_patterns_sum.bomb_when_enemy_low_cards_rate += patterns.bomb_when_enemy_low_cards_rate;
            losing_patterns_sum.save_bomb_for_endgame_rate += patterns.save_bomb_for_endgame_rate;
            losing_patterns_sum.level_card_usage_rate += patterns.level_card_usage_rate;
            losing_patterns_sum.combo_follow_rate += patterns.combo_follow_rate;
            losing_patterns_sum.proactive_play_rate += patterns.proactive_play_rate;
        }
    }

    let avg_patterns = |sum: PlayerPatterns, count: usize| -> PlayerPatterns {
        if count == 0 {
            sum
        } else {
            PlayerPatterns {
                avg_hand_size_at_first_play: sum.avg_hand_size_at_first_play / count as f32,
                bomb_usage_rate: sum.bomb_usage_rate / count as f32,
                pass_when_partner_leads_rate: sum.pass_when_partner_leads_rate / count as f32,
                follow_partner_small_card_rate: sum.follow_partner_small_card_rate / count as f32,
                intercept_enemy_small_card_rate: sum.intercept_enemy_small_card_rate / count as f32,
                bomb_when_enemy_low_cards_rate: sum.bomb_when_enemy_low_cards_rate / count as f32,
                save_bomb_for_endgame_rate: sum.save_bomb_for_endgame_rate / count as f32,
                level_card_usage_rate: sum.level_card_usage_rate / count as f32,
                combo_follow_rate: sum.combo_follow_rate / count as f32,
                proactive_play_rate: sum.proactive_play_rate / count as f32,
            }
        }
    };

    LogAnalysis {
        winning_patterns: avg_patterns(winning_patterns_sum, win_count),
        losing_patterns: avg_patterns(losing_patterns_sum, loss_count),
        human_wins,
        human_losses,
        total_games: human_wins + human_losses,
    }
}

pub fn patterns_to_params(analysis: &LogAnalysis) -> AdvancedBotParams {
    let win = &analysis.winning_patterns;
    let loss = &analysis.losing_patterns;

    let mut params = AdvancedBotParams::default_balanced();

    params.team_win_weight = 1.0 + (win.pass_when_partner_leads_rate - loss.pass_when_partner_leads_rate) * 2.0;
    
    params.first_out_weight = 0.8 + (win.proactive_play_rate - loss.proactive_play_rate) * 1.0;
    
    params.second_out_weight = 0.9 + (win.follow_partner_small_card_rate - loss.follow_partner_small_card_rate) * 1.0;
    
    params.yield_to_partner_bias = 1.4 + (win.pass_when_partner_leads_rate - loss.pass_when_partner_leads_rate) * 2.0;
    
    params.bomb_conserve_bias = 0.8 + (win.save_bomb_for_endgame_rate - loss.save_bomb_for_endgame_rate) * 1.5;
    
    params.bomb_aggression_when_enemy_low = 2.2 + (win.bomb_when_enemy_low_cards_rate - loss.bomb_when_enemy_low_cards_rate) * 2.0;
    
    params.endgame_clear_hand_bias = 1.2 + (win.save_bomb_for_endgame_rate - loss.save_bomb_for_endgame_rate) * 1.0;
    
    params.proactive_play_bias = 1.1 + (win.proactive_play_rate - loss.proactive_play_rate) * 1.5;
    
    params.low_card_dump_bias = 1.4 + (win.follow_partner_small_card_rate - loss.follow_partner_small_card_rate) * 1.0;
    
    params.pass_stall_penalty = 0.9 - (win.proactive_play_rate - loss.proactive_play_rate) * 0.5;

    params.endgame_hand_count_threshold = if win.save_bomb_for_endgame_rate > loss.save_bomb_for_endgame_rate {
        5
    } else {
        7
    };
    
    params.partner_sprint_threshold = if win.intercept_enemy_small_card_rate > loss.intercept_enemy_small_card_rate {
        3
    } else {
        4
    };
    
    params.enemy_low_cards_threshold = if win.bomb_when_enemy_low_cards_rate > loss.bomb_when_enemy_low_cards_rate {
        3
    } else {
        4
    };

    params.team_win_weight = params.team_win_weight.clamp(0.5, 2.0);
    params.first_out_weight = params.first_out_weight.clamp(0.5, 2.0);
    params.second_out_weight = params.second_out_weight.clamp(0.5, 2.0);
    params.yield_to_partner_bias = params.yield_to_partner_bias.clamp(0.5, 3.0);
    params.bomb_conserve_bias = params.bomb_conserve_bias.clamp(0.3, 2.0);
    params.bomb_aggression_when_enemy_low = params.bomb_aggression_when_enemy_low.clamp(1.0, 4.0);
    params.endgame_clear_hand_bias = params.endgame_clear_hand_bias.clamp(0.5, 2.0);
    params.proactive_play_bias = params.proactive_play_bias.clamp(0.5, 2.5);
    params.low_card_dump_bias = params.low_card_dump_bias.clamp(0.5, 2.0);
    params.pass_stall_penalty = params.pass_stall_penalty.clamp(0.3, 2.0);

    params
}

pub fn learn_from_logs(logs: &[GameLogEntry], iterations: u32, output_path: &str) -> Result<AdvancedBotParams, String> {
    if logs.is_empty() {
        return Err("No game logs found".to_string());
    }

    println!("[record_learner] Analyzing {} game logs...", logs.len());
    
    let analysis = analyze_logs(logs);
    
    println!("[record_learner] Human wins: {}, losses: {} ({} total)", 
        analysis.human_wins, analysis.human_losses, analysis.total_games);
    
    println!("[record_learner] Winning patterns:");
    println!("[record_learner]   - Avg hand at first play: {:.1}", analysis.winning_patterns.avg_hand_size_at_first_play);
    println!("[record_learner]   - Bomb usage rate: {:.2}", analysis.winning_patterns.bomb_usage_rate);
    println!("[record_learner]   - Pass when partner leads: {:.2}", analysis.winning_patterns.pass_when_partner_leads_rate);
    println!("[record_learner]   - Follow partner small card: {:.2}", analysis.winning_patterns.follow_partner_small_card_rate);
    println!("[record_learner]   - Intercept enemy small card: {:.2}", analysis.winning_patterns.intercept_enemy_small_card_rate);
    println!("[record_learner]   - Proactive play rate: {:.2}", analysis.winning_patterns.proactive_play_rate);
    
    let mut best_params = patterns_to_params(&analysis);
    
    println!("[record_learner] Initial params from logs: {:?}", best_params);
    
    use crate::learning::optimizer::{evaluate_params, SelfPlayConfig, eval_to_score};
    
    let eval_config = SelfPlayConfig {
        matches_per_eval: 30,
        max_plies: 2000,
    };
    
    let mut best_score = eval_to_score(&evaluate_params(&best_params, &eval_config));
    println!("[record_learner] Initial score: {:.4}", best_score);
    
    for i in 1..=iterations {
        let candidate = best_params.mutate_random(0.1);
        let eval = evaluate_params(&candidate, &eval_config);
        let score = eval_to_score(&eval);
        
        if score > best_score {
            best_params = candidate;
            best_score = score;
            println!("[record_learner] Iter {}: improved! score={:.4}", i, best_score);
        } else {
            println!("[record_learner] Iter {}: score={:.4}", i, score);
        }
    }
    
    best_params.save(output_path)?;
    println!("[record_learner] Learned params saved to {}", output_path);
    
    Ok(best_params)
}
