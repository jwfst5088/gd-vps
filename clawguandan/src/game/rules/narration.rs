use crate::game::types::TeamId;
use serde_json::{Value, json};

pub fn is_big_play_combination(combination_type: &str) -> bool {
    let ct = combination_type.trim();
    ct.starts_with("bomb") || ct == "straightFlush" || ct == "fourJoker"
}

pub fn format_big_play(player_name: &str, combination_type: &str) -> String {
    let (kind_zh, kind_en) = match combination_type {
        "straightFlush" => ("同花顺炸", "straight flush bomb"),
        "fourJoker" => ("四王炸", "four-joker bomb"),
        x if x.starts_with("bomb") => ("炸弹", "bomb"),
        _ => ("大牌", "big play"),
    };
    bilingual(
        format!("{} 打出{}! 💥", safe_name(player_name), kind_zh),
        format!("{} plays {}! 💥", safe_name(player_name), kind_en),
    )
}

pub fn format_rank_announce(player_name: &str, rank: usize) -> String {
    let (rank_name_zh, rank_name_en) = match rank {
        1 => ("头游", "first out"),
        2 => ("二游", "second out"),
        3 => ("三游", "third out"),
        4 => ("末游", "last out"),
        _ => ("出完", "out"),
    };
    bilingual(
        format!("{} {}! 🏁", safe_name(player_name), rank_name_zh),
        format!("{} {}! 🏁", safe_name(player_name), rank_name_en),
    )
}

pub fn format_tribute_action(
    player_name: &str,
    card: &str,
    target_name: &str,
    is_return: bool,
) -> String {
    if is_return {
        bilingual(
            format!(
                "{}还贡了{}给{}。",
                safe_name(player_name),
                card.trim(),
                safe_name(target_name)
            ),
            format!(
                "{} returned {} to {}.",
                safe_name(player_name),
                card.trim(),
                safe_name(target_name)
            ),
        )
    } else {
        bilingual(
            format!(
                "{}进贡了{}给{}。",
                safe_name(player_name),
                card.trim(),
                safe_name(target_name)
            ),
            format!(
                "{} tributed {} to {}.",
                safe_name(player_name),
                card.trim(),
                safe_name(target_name)
            ),
        )
    }
}

pub fn format_tribute_canceled(opening_player_name: &str) -> String {
    bilingual(
        format!(
            "本局抗贡（免进贡），由{}先出。",
            safe_name(opening_player_name)
        ),
        format!(
            "Tribute canceled for this hand; {} leads first.",
            safe_name(opening_player_name)
        ),
    )
}

fn declarer_phrase(team: TeamId) -> (&'static str, &'static str) {
    match team {
        TeamId::Ew => ("EW（东西组）", "the EW team (East–West)"),
        TeamId::Sn => ("SN（南北组）", "the SN team (South–North)"),
    }
}

/// Opening line: declarer side and hand level (bilingual JSON string).
pub fn format_hand_open(declarer: TeamId, level_api: &str) -> String {
    let (d_zh, d_en) = declarer_phrase(declarer);
    let lv = level_api.trim();
    bilingual(
        format!("本局庄家方为{}，打{}。", d_zh, lv),
        format!("This hand: {} is declarer; hand level {}.", d_en, lv),
    )
}

/// Same as [`format_hand_open`] plus tribute-canceled lead (one combined message).
pub fn format_hand_open_with_tribute_canceled(
    declarer: TeamId,
    level_api: &str,
    opening_player_name: &str,
) -> String {
    let (d_zh, d_en) = declarer_phrase(declarer);
    let lv = level_api.trim();
    let name = safe_name(opening_player_name);
    bilingual(
        format!(
            "本局庄家方为{}，打{}。抗贡（免进贡），由{}先出。",
            d_zh, lv, name
        ),
        format!(
            "This hand: {} is declarer; hand level {}. Tribute canceled; {} leads first.",
            d_en, lv, name
        ),
    )
}

/// Team label used by hand-end extras (matches JS `_formatHandEnd`).
fn hand_end_team_label(team: TeamId) -> (&'static str, &'static str) {
    match team {
        TeamId::Ew => ("EW（东西组）", "EW (East–West)"),
        TeamId::Sn => ("SN（南北组）", "SN (South–North)"),
    }
}

pub fn format_hand_end(
    finishing_names: &[String],
    level_ew: &str,
    level_sn: &str,
    waiting_ready: bool,
    game_over: bool,
    winner_team: Option<TeamId>,
    demoted_from_a: bool,
    declarer_team: TeamId,
    ew_a_fail_count: u32,
    sn_a_fail_count: u32,
) -> String {
    let ranking_zh = if finishing_names.is_empty() {
        "本手结束".to_string()
    } else {
        format!("本手排名: {}", finishing_names.join(" > "))
    };
    let ranking_en = if finishing_names.is_empty() {
        "Hand ended".to_string()
    } else {
        format!("Ranking: {}", finishing_names.join(" > "))
    };
    let levels_zh = format!("当前级别 EW {} / SN {} 📈", level_ew, level_sn);
    let levels_en = format!("Levels EW {} / SN {} 📈", level_ew, level_sn);
    if game_over {
        let final_score_zh = format!("最终成绩 EW {} / SN {}", level_ew, level_sn);
        let final_score_en = format!("Final score EW {} / SN {}", level_ew, level_sn);
        let (winner_zh, winner_en) = match winner_team {
            Some(TeamId::Ew) => ("EW（东西组）", "EW (East–West)"),
            Some(TeamId::Sn) => ("SN（南北组）", "SN (South–North)"),
            None => {
                // 根据最终级别判断获胜队伍：哪个队伍升到A且领先
                let ew_is_a = level_ew == "A";
                let sn_is_a = level_sn == "A";
                if ew_is_a && sn_is_a {
                    // 双方都到A，判断谁先到（最终成绩中级别高的）
                    ("EW（东西组）", "EW (East–West)")
                } else if ew_is_a {
                    ("EW（东西组）", "EW (East–West)")
                } else if sn_is_a {
                    ("SN（南北组）", "SN (South–North)")
                } else {
                    // 根据级别数值判断
                    ("EW（东西组）", "EW (East–West)")
                }
            }
        };
        bilingual(
            format!(
                "{}；{}，恭喜获胜队{}，游戏结束！🎉",
                ranking_zh, final_score_zh, winner_zh
            ),
            format!(
                "{}; {}. Congratulations to the winning team {}. Game over! 🎉",
                ranking_en, final_score_en, winner_en
            ),
        )
    } else if waiting_ready {
        // 房规附加行（mirror JS `_formatHandEnd`）：
        // - demoted_from_a: A级三战失败退回2级，比赛继续
        // - 否则：仍在A的队伍显示本手冲击未成的失败计数（第 n/3 次）
        let (extra_zh, extra_en) = if demoted_from_a {
            let (d_zh, d_en) = hand_end_team_label(declarer_team);
            (
                format!(" 💥 {} 冲击A级三战失败，退回 2 级，游戏继续！", d_zh),
                format!(
                    " 💥 {} failed three challenges at level A and drops back to 2. The game continues!",
                    d_en
                ),
            )
        } else {
            let n = ew_a_fail_count.max(sn_a_fail_count);
            if n > 0 {
                let (t_zh, t_en) = if ew_a_fail_count > 0 {
                    hand_end_team_label(TeamId::Ew)
                } else {
                    hand_end_team_label(TeamId::Sn)
                };
                (
                    format!(
                        " ⚠️ {} 本手冲击A级未成（第 {}/3 次；三败退回2级）",
                        t_zh, n
                    ),
                    format!(
                        " ⚠️ {} failed this level-A challenge ({}/3; three fails drop to 2)",
                        t_en, n
                    ),
                )
            } else {
                (String::new(), String::new())
            }
        };
        bilingual(
            format!(
                "{}; {}。{} 请全员再次准备 ▶️",
                ranking_zh, levels_zh, extra_zh
            ),
            format!(
                "{}; {}.{} Everyone ready again ▶️",
                ranking_en, levels_en, extra_en
            ),
        )
    } else {
        bilingual(
            format!("{}; {}。", ranking_zh, levels_zh),
            format!("{}; {}.", ranking_en, levels_en),
        )
    }
}

fn winning_team_phrase(team: TeamId) -> (&'static str, &'static str) {
    match team {
        TeamId::Ew => ("EW（东西组）", "EW (East-West)"),
        TeamId::Sn => ("SN（南北组）", "SN (South-North)"),
    }
}

/// 房规：A级双上夺冠战报（mirror JS `_formatGameEnd`，结构化双语对象）。
/// 唯一夺冠方式是 A级双上；整场结束不回大厅 —— 原地重开新一场（双方从2级，12秒后自动发牌）。
pub fn format_game_end_champion(
    winner_team: TeamId,
    finishing_names: &[String],
    level_ew: &str,
    level_sn: &str,
) -> String {
    let ranking_zh = if finishing_names.is_empty() {
        "本手结束".to_string()
    } else {
        format!("本手排名: {}", finishing_names.join(" > "))
    };
    let ranking_en = if finishing_names.is_empty() {
        "Hand ended".to_string()
    } else {
        format!("Ranking: {}", finishing_names.join(" > "))
    };
    let (winner_zh, winner_en) = hand_end_team_label(winner_team);
    let headline_zh = format!("🏆 {} A级双上，夺得本场冠军！", winner_zh);
    let headline_en = format!("🏆 {} completes level A and wins the match!", winner_en);
    let final_score_zh = format!("最终成绩 EW {} / SN {}", level_ew, level_sn);
    let final_score_en = format!("Final score EW {} / SN {}", level_ew, level_sn);
    bilingual(
        format!(
            "{} {}；{}。🎉 新一场即将开始：双方从 2 级重新对战！（12秒后自动发牌）",
            headline_zh, ranking_zh, final_score_zh
        ),
        format!(
            "{} {}; {}. 🎉 A new game starts soon: both teams replay from level 2! (dealing automatically in 12 seconds)",
            headline_en, ranking_en, final_score_en
        ),
    )
}

pub fn format_game_end_by_leave(leaving_names: &[String]) -> String {
    let names = if leaving_names.is_empty() {
        "玩家".to_string()
    } else {
        leaving_names.join("、")
    };
    bilingual(
        format!("{} 超时离开，游戏结束，本局不计分。", names),
        format!(
            "{} timed out and left. Game ended with no score settlement.",
            names
        ),
    )
}

pub fn last_narration_from_nextstate_json(v: &Value) -> Option<String> {
    let ops = v.get("delta")?.get("ops")?.as_array()?;
    let mut out: Option<String> = None;
    for op in ops {
        if op.get("op").and_then(|x| x.as_str()) == Some("replace")
            && op.get("path").and_then(|x| x.as_str()) == Some("/narration")
            && let Some(val) = op.get("value")
        {
            out = Some(match val {
                Value::String(s) => s.clone(),
                _ => val.to_string(),
            });
        }
    }
    out
}

pub fn narration_display_en(raw: &str) -> String {
    let t = raw.trim();
    if t.is_empty() {
        return String::new();
    }
    if let Ok(v) = serde_json::from_str::<Value>(t)
        && let Some(en) = v.get("en").and_then(|x| x.as_str())
    {
        return en.trim().to_string();
    }
    t.to_string()
}

fn bilingual(zh: String, en: String) -> String {
    json!({ "zh": zh, "en": en }).to_string()
}

fn safe_name(name: &str) -> &str {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        "玩家"
    } else {
        trimmed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tribute_canceled_narration_mentions_opening_player() {
        let msg = format_tribute_canceled("Alice");
        assert!(msg.contains("抗贡"));
        assert!(msg.contains("Alice"));
        assert!(msg.contains("Tribute canceled"));
    }

    #[test]
    fn hand_open_narration_includes_declarer_and_level() {
        let msg = format_hand_open(TeamId::Ew, "5");
        assert!(msg.contains("庄家方"));
        assert!(msg.contains("打5"));
        assert!(msg.contains("declarer"));
        assert!(msg.contains("hand level 5"));
    }

    #[test]
    fn hand_open_with_tribute_canceled_combines_parts() {
        let msg = format_hand_open_with_tribute_canceled(TeamId::Sn, "A", "Bob");
        assert!(msg.contains("打A"));
        assert!(msg.contains("抗贡"));
        assert!(msg.contains("Bob"));
        assert!(msg.contains("Tribute canceled"));
    }

    #[test]
    fn narration_display_en_prefers_en_field() {
        let raw = r#"{"zh":"中文","en":"Hello"}"#;
        assert_eq!(narration_display_en(raw), "Hello");
        assert_eq!(narration_display_en("plain"), "plain");
    }

    #[test]
    fn last_narration_from_nextstate_json_reads_replace_ops() {
        let v = json!({
            "seq": 3u64,
            "delta": {
                "ops": [
                    { "op": "replace", "path": "/phase", "value": "playing" },
                    { "op": "replace", "path": "/narration", "value": "{\"zh\":\"z\",\"en\":\"e\"}" }
                ]
            }
        });
        assert_eq!(
            last_narration_from_nextstate_json(&v).as_deref(),
            Some("{\"zh\":\"z\",\"en\":\"e\"}")
        );
    }
}
