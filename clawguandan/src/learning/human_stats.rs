//! 房规（c 方案）：真人对局习惯统计。
//!
//! 只读 `game_logs.jsonl`，对比人类玩家与 AI 的出牌习惯，
//! 输出可直接转成规则强度 / 参数初值的对照报告。不修改任何状态。

use std::collections::BTreeMap;

use super::game_logger::{GameLogger, GameLogEntry};

/// 从卡牌符号提取点数值（3..A,2,双王；百搭按级牌近似 16）。
fn rank_value(sym: &str) -> Option<f32> {
    let s = sym
        .strip_prefix('♠')
        .or_else(|| sym.strip_prefix('♥'))
        .or_else(|| sym.strip_prefix('♦'))
        .or_else(|| sym.strip_prefix('♣'))
        .unwrap_or(sym);
    Some(match s {
        "3" => 3.0,
        "4" => 4.0,
        "5" => 5.0,
        "6" => 6.0,
        "7" => 7.0,
        "8" => 8.0,
        "9" => 9.0,
        "10" => 10.0,
        "J" => 11.0,
        "Q" => 12.0,
        "K" => 13.0,
        "A" => 14.0,
        "2" => 16.0,
        "BJ" | "b" => 17.0,
        "R" | "RJ" => 18.0,
        _ => return None,
    })
}

#[derive(Default)]
struct SideStat {
    single_plays: Vec<f32>,
    first_lead_max: Vec<f32>,
    bomb_pos: Vec<f32>,
    finish_ranks: Vec<usize>,
}

impl SideStat {
    fn avg(v: &[f32]) -> Option<f32> {
        if v.is_empty() {
            None
        } else {
            Some(v.iter().sum::<f32>() / v.len() as f32)
        }
    }
}

/// 统计入口：`clawguandan stats --logs game_logs.jsonl`
pub fn run(path: &str) -> Result<(), String> {
    let logs = GameLogger::read_logs(path)?;
    if logs.is_empty() {
        return Err(format!("no log entries found in {path}"));
    }

    let mut sides: BTreeMap<&'static str, SideStat> = BTreeMap::new();
    let mut human_games = 0usize;
    let mut human_seats_total = 0usize;
    let mut with_params = 0usize;

    for e in &logs {
        if !e.human_seats.is_empty() {
            human_games += 1;
            human_seats_total += e.human_seats.len();
        }
        if e.bot_params.is_some() {
            with_params += 1;
        }

        let side_of = |seat: &str| -> Option<&'static str> {
            if e.human_seats.iter().any(|s| s == seat) {
                Some("human")
            } else if e.bot_seats.iter().any(|s| s == seat) {
                Some("bot")
            } else {
                None
            }
        };

        // 完牌位次：finishing_order 第 i 位 → 名次 i+1
        for (i, seat) in e.finishing_order.iter().enumerate() {
            if let Some(side) = side_of(seat) {
                sides.entry(side).or_default().finish_ranks.push(i + 1);
            }
        }

        // 每局首个动作若为出牌 → 必为首出
        if let Some(a) = e.actions.first() {
            if a.action_type == "play" {
                if let Some(cards) = &a.cards {
                    let m = cards
                        .iter()
                        .filter_map(|c| rank_value(c))
                        .fold(f32::MIN, f32::max);
                    if let Some(side) = side_of(&a.seat) {
                        sides.entry(side).or_default().first_lead_max.push(m);
                    }
                }
            }
        }

        let n = e.actions.len().max(1);
        for (idx, a) in e.actions.iter().enumerate() {
            if a.action_type != "play" {
                continue;
            }
            let cards = match &a.cards {
                Some(c) if !c.is_empty() => c,
                _ => continue,
            };
            // 炸弹粗判：4 张以上同点，或双王
            let is_bomb = (cards.len() >= 4
                && cards
                    .iter()
                    .filter_map(|c| rank_value(c).map(|v| v as u32))
                    .collect::<Vec<_>>()
                    .windows(2)
                    .all(|w| w[0] == w[1]))
                || (cards.len() == 2 && cards.iter().all(|c| rank_value(c) == Some(17.0)));
            if is_bomb {
                if let Some(side) = side_of(&a.seat) {
                    sides
                        .entry(side)
                        .or_default()
                        .bomb_pos
                        .push(idx as f32 / (n - 1).max(1) as f32);
                }
            }
            if cards.len() == 1 {
                if let Some(v) = rank_value(&cards[0]) {
                    if let Some(side) = side_of(&a.seat) {
                        sides.entry(side).or_default().single_plays.push(v);
                    }
                }
            }
        }
    }

    println!("=== 对局日志习惯统计：{path} ===");
    println!(
        "样本：{} 局（含真人 {} 局 / {} 人次；带参数快照 {with_params} 条）",
        logs.len(),
        human_games,
        human_seats_total,
        with_params = with_params
    );

    for (side, st) in &sides {
        println!("\n--- {side} ---");
        println!(
            "单张出牌均值      : {:.2}  (n={})",
            SideStat::avg(&st.single_plays).unwrap_or(f32::NAN),
            st.single_plays.len()
        );
        println!(
            "首出最大点数均值  : {:.2}  (n={})",
            SideStat::avg(&st.first_lead_max).unwrap_or(f32::NAN),
            st.first_lead_max.len()
        );
        if !st.bomb_pos.is_empty() {
            let early = st.bomb_pos.iter().filter(|&&p| p < 1.0 / 3.0).count();
            let late = st.bomb_pos.iter().filter(|&&p| p >= 2.0 / 3.0).count();
            println!(
                "炸弹使用位置      : 早段 {} / 中段 {} / 末段 {}  (n={})",
                early,
                st.bomb_pos.len() - early - late,
                late,
                st.bomb_pos.len()
            );
        } else {
            println!("炸弹使用位置      : 无样本");
        }
        let finish_avg = if st.finish_ranks.is_empty() {
            f32::NAN
        } else {
            st.finish_ranks.iter().sum::<usize>() as f32 / st.finish_ranks.len() as f32
        };
        println!(
            "完牌平均名次      : {:.2}  (n={})",
            finish_avg,
            st.finish_ranks.len()
        );
    }

    println!("\n=== 参数初值建议 ===");
    let h_single = sides.get("human").and_then(|s| SideStat::avg(&s.single_plays));
    let b_single = sides.get("bot").and_then(|s| SideStat::avg(&s.single_plays));
    if let (Some(h), Some(b)) = (h_single, b_single)
        && h + 1.0 < b
    {
        println!(
            "人类单张均值({h:.2}) 明显低于 AI({b:.2}) → 「小牌优先」规则方向正确，可保持/加强。"
        );
    }
    let h_lead = sides.get("human").and_then(|s| SideStat::avg(&s.first_lead_max));
    let b_lead = sides.get("bot").and_then(|s| SideStat::avg(&s.first_lead_max));
    if let (Some(h), Some(b)) = (h_lead, b_lead)
        && h > b
    {
        println!(
            "人类首出更凶({h:.2} > AI {b:.2}) → proactive_play_bias 可小幅上调验证。"
        );
    }
    println!("提示：样本建议 ≥30 局再据此调参；当前样本量见上方「样本」行。");
    Ok(())
}
