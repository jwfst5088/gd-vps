use super::features::RuleFeatures;
use super::params::RuleBotParams;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlayCandidate {
    Pass,
    SuggestPlay,
}

#[derive(Clone, Debug, Default)]
pub struct ScoreTrace {
    pub pass_score: f32,
    pub suggest_score: f32,
    pub reasons: Vec<String>,
}

pub fn choose_play_candidate(
    params: &RuleBotParams,
    f: &RuleFeatures,
) -> (PlayCandidate, ScoreTrace) {
    let mut trace = ScoreTrace::default();
    let partner_leading = is_partner_leading(f);
    let teammate_sprinting = f
        .teammate_remaining
        .map(|x| x <= params.partner_sprint_threshold)
        .unwrap_or(false);
    let not_urgent = !f.enemy_low_cards_urgent;

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 队友领牌时绝不压队友（即使对手冲刺也不例外）
    // 但队友出小牌时，自己可以顺牌减少手牌——按牌型区分：
    // - 单张：手中有2~Q单张（级牌除外）→ 鼓励顺牌，可以过牌
    // - 对子：手中有2~9对子（级牌除外）→ 鼓励顺牌，可以过牌
    // - 三张同：手中有2~9三张同（级牌除外）→ 鼓励顺牌，可以过牌
    // 注意：顺牌是温和建议，不强制，不拆牌，允许过牌
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if partner_leading {
        let top_is_bomb = f.top_play_kind.as_deref()
            .map(|k| k.starts_with("Bomb"))
            .unwrap_or(false);
        if top_is_bomb {
            // 铁律：绝不压队友的炸弹
            trace.pass_score += 10.0 * params.team_win_weight;
            trace.suggest_score -= 10.0;
            trace.reasons.push("pass: NEVER override teammate's bomb".into());
            return (PlayCandidate::Pass, trace);
        }

        // 铁律：不能压队友的级牌
        // 级牌是当前打几的关键牌，压队友级牌等于浪费我方资源
        let top_is_level = f.top_play_value.zip(f.level_rank)
            .map(|(v, r)| v == r)
            .unwrap_or(false);
        if top_is_level {
            trace.pass_score += 10.0 * params.team_win_weight;
            trace.suggest_score -= 10.0;
            trace.reasons.push("pass: NEVER override teammate's level card".into());
            return (PlayCandidate::Pass, trace);
        }

        // 铁律：队友出大牌（Joker、A、K、Q）时，绝不压
        // 队友出大牌是为了拿牌权或冲刺，压队友大牌等于内耗
        let top_is_big = f.top_play_value
            .map(|v| v >= 12)  // Q=12, K=13, A=14, Joker=15/16
            .unwrap_or(false);
        if top_is_big {
            trace.pass_score += 5.0 * params.team_win_weight;
            trace.suggest_score -= 5.0;
            trace.reasons.push("pass: do not override teammate's big card".into());
            return (PlayCandidate::Pass, trace);
        }

        let has_playable_follow = match f.top_play_kind.as_deref() {
            Some(kind) if kind.contains("Single") => f.medium_single_count > 0,
            Some(kind) if kind.contains("Pair") => f.small_pair_rank_count > 0,
            Some(kind) if kind.contains("Triple") => f.small_triple_rank_count > 0,
            // 其他牌型（顺子、钢板、木板等）：用原有逻辑
            _ => f.low_card_count > 0,
        };

        if has_playable_follow && f.can_play {
            // 手中有可顺的牌 → 积极鼓励出牌，顺牌减少手牌
            trace.suggest_score += 3.0 * params.low_card_dump_bias * params.team_win_weight;
            trace.reasons.push(
                "suggest: follow partner to reduce hand".into()
            );
        } else {
            // 手中没有可顺的牌或不能出牌 → 偏向过牌
            trace.pass_score += 1.0 * params.team_win_weight;
            trace.suggest_score -= 0.5;
            trace.reasons.push("pass: prefer pass when partner leads".into());

            // 原有小牌顺牌逻辑作为兜底
            if f.low_card_count > 0 && f.can_play {
                trace.suggest_score += 2.5 * params.low_card_dump_bias;
                trace.reasons
                    .push("suggest: can follow partner with small card".into());
            }
        }

        // 队友冲刺：强烈偏向让牌
        if teammate_sprinting {
            trace.pass_score += 2.0 * params.second_out_weight;
            trace.reasons
                .push("pass: teammate sprinting, yield completely".into());
        }
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 铁律 2：对手冲刺时必须拦截（但队友领牌时不拦截，已在上面处理）
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if f.enemy_low_cards_urgent && !partner_leading {
        trace.suggest_score += 5.0 * params.team_win_weight;
        trace.pass_score -= 5.0 * params.team_win_weight;
        trace.reasons
            .push("suggest: MUST intercept enemy sprinting".into());
        if f.can_play {
            return (PlayCandidate::SuggestPlay, trace);
        }
    }

    // ── 队友冲刺 + 我领牌: 主动出牌递牌 ──
    if teammate_sprinting && f.leading_new_trick && !partner_leading {
        trace.suggest_score += params.proactive_play_bias * 1.8 * params.second_out_weight;
        if f.low_card_count > 0 {
            trace.suggest_score +=
                params.low_card_dump_bias * 1.5 * params.second_out_weight;
            trace
                .reasons
                .push("suggest: lead small for sprinting teammate".into());
        } else {
            trace.suggest_score += params.proactive_play_bias * 0.8;
            trace
                .reasons
                .push("suggest: lead to support sprinting teammate".into());
        }
        trace.pass_score -= 1.0;
    }

    // ── 炸弹保留策略 ──
    // 开局炸弹少(<3)时偏向保留，不是紧急情况不主动出炸弹
    if f.bomb_count < 3 && not_urgent && !f.leading_new_trick && !f.endgame_mode {
        trace.pass_score += params.bomb_conserve_bias * 1.5 * params.team_win_weight;
        trace.reasons
            .push("pass: bombs < 3, conserve early game".into());
    }

    // 残局1个炸弹：允许出牌（由suggest层决定先出小牌，最后用炸弹冲线）
    if f.bomb_count == 1 && f.endgame_mode && not_urgent {
        trace.suggest_score += 1.0 * params.endgame_clear_hand_bias;
        trace.reasons
            .push("endgame: 1 bomb, let suggest handle (small first, bomb last)".into());
    } else if f.bomb_count == 1 && not_urgent {
        // 只有1个炸弹：轻微偏好pass，suggest层会自动选非炸弹牌
        trace.pass_score += 1.0 * params.team_win_weight;
        trace.reasons.push("pass: only 1 bomb, slight conserve".into());
    } else if f.bomb_count >= 2 && not_urgent && !f.leading_new_trick {
        // ≥2个炸弹：非紧急非领牌时轻微倾向pass
        trace.pass_score += 0.5 * params.team_win_weight;
        trace.reasons.push("pass: conserve bomb for endgame".into());
    } else if f.bomb_count >= 2 && not_urgent && f.leading_new_trick {
        // ≥2个炸弹且领牌：可以主动出牌
        trace.reasons.push("suggest: have spare bombs, can lead".into());
    }

    // ── 中性局面: 鼓励主动出牌清理小牌 ──
    let neutral_table = !f.enemy_low_cards_urgent && !partner_leading && !teammate_sprinting;
    if neutral_table && f.can_play {
        trace.suggest_score += params.proactive_play_bias * params.team_win_weight;
        trace.reasons.push("suggest: proactive tempo".into());
        if f.low_card_count > 0 {
            trace.suggest_score +=
                params.low_card_dump_bias * f.low_card_ratio * params.first_out_weight;
            trace.reasons.push("suggest: dump small cards".into());
        }
        if f.leading_new_trick {
            trace.suggest_score += 0.35 * params.proactive_play_bias;
            trace.reasons.push("suggest: lead and shape trick".into());
        }
    }

    // ── 防挂机: 惩罚连续pass ──
    if f.can_pass && neutral_table && !f.endgame_mode {
        trace.pass_score -= params.pass_stall_penalty * params.team_win_weight;
        trace.reasons.push("pass: stall penalty".into());
    }

    // ── 残局模式 (≤6张): 优先出小牌（单张/对子），保留大牌最后出 ──
    if f.endgame_mode && !partner_leading {
        if f.low_card_count > 0 {
            // 手中有小牌 → 强烈鼓励出小牌，不要过牌
            trace.suggest_score += params.endgame_clear_hand_bias * params.first_out_weight * 1.8;
            trace.reasons.push("suggest: endgame play small cards first".into());
            if f.can_pass {
                trace.pass_score -= 1.0 * params.first_out_weight;
            }
        } else {
            // 只剩大牌 → 正常出牌，可以过牌等机会
            trace.suggest_score += params.endgame_clear_hand_bias * params.first_out_weight * 0.6;
            trace.reasons.push("suggest: endgame clear-hand (big cards only)".into());
        }
    }

    // ── 最终决策 ──
    let picked = if !f.can_play && f.can_pass {
        PlayCandidate::Pass
    } else if f.can_play && !f.can_pass {
        PlayCandidate::SuggestPlay
    } else if trace.pass_score > trace.suggest_score {
        PlayCandidate::Pass
    } else {
        PlayCandidate::SuggestPlay
    };

    (picked, trace)
}

fn is_partner_leading(f: &RuleFeatures) -> bool {
    matches!(
        (f.teammate_seat.as_deref(), f.top_play_seat.as_deref()),
        (Some(t), Some(top)) if t == top
    )
}