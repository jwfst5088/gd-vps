use crate::game::types::TeamId;

pub use crate::game::card::HandLevel as Level;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WinType {
    OneTwo,
    OneThree,
    OneFour,
}

impl WinType {
    pub fn promotion_delta(self) -> u8 {
        match self {
            WinType::OneFour => 1,
            WinType::OneThree => 2,
            WinType::OneTwo => 3,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TeamProgress {
    pub team: TeamId,
    pub level: Level,
    /// Count of unsuccessful attempts as A-level declarer (not necessarily consecutive).
    pub ace_failed_attempts: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HandResult {
    pub winner_team: TeamId,
    pub win_type: WinType,
    pub promotion_delta: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GameOutcome {
    pub winner_team: Option<TeamId>,
    pub next_declarer: TeamId,
    pub progress_ew: TeamProgress,
    pub progress_sn: TeamProgress,
}

pub struct ScoringService;

impl ScoringService {
    /// Apply scoring to team progress.
    ///
    /// Inputs:
    /// - `declarer`: which team is declarer in this hand
    /// - `win_type`: 1-2/1-3/1-4
    /// - `winner`: which team won this hand
    /// - `ace_finish_demotes_declarer`: only relevant when declarer is A-level and loses:
    ///   if true, declarer is demoted to level 2 immediately.
    pub fn apply_hand(
        progress_ew: TeamProgress,
        progress_sn: TeamProgress,
        declarer: TeamId,
        winner: TeamId,
        win_type: WinType,
        ace_finish_demotes_declarer: bool,
    ) -> Result<GameOutcome, String> {
        let delta = win_type.promotion_delta();

        let mut ew = progress_ew;
        let mut sn = progress_sn;

        let (mut declarer_prog, opp_prog) = match declarer {
            TeamId::Ew => (ew.clone(), sn.clone()),
            TeamId::Sn => (sn.clone(), ew.clone()),
        };

        let declarer_won = winner == declarer;
        let declarer_is_a = declarer_prog.level == Level::A;

        // A级终结胜利：庄家必须双上（1-2）才算获胜
        if declarer_is_a && declarer_won && matches!(win_type, WinType::OneTwo) {
            return Ok(GameOutcome {
                winner_team: Some(declarer),
                next_declarer: declarer,
                progress_ew: ew,
                progress_sn: sn,
            });
        }

        // 赢家升级（封顶A级）
        match winner {
            TeamId::Ew => ew.level = ew.level.promote_by(delta),
            TeamId::Sn => sn.level = sn.level.promote_by(delta),
        }

        // A级失败追踪：除双上外的任何结果（1-3、1-4、输牌）都算失败。
        // 累计3次失败后：退回2级，对方获胜。
        if declarer_is_a {
            if !declarer_won || !matches!(win_type, WinType::OneTwo) {
                declarer_prog.ace_failed_attempts += 1;
            }

            if declarer_prog.ace_failed_attempts >= 3 {
                declarer_prog.level = Level::Two;
                declarer_prog.ace_failed_attempts = 0;

                // 对方队伍获胜
                let game_winner = match declarer {
                    TeamId::Ew => TeamId::Sn,
                    TeamId::Sn => TeamId::Ew,
                };

                // 回写庄家方的降级
                match declarer {
                    TeamId::Ew => {
                        ew.level = declarer_prog.level;
                        ew.ace_failed_attempts = declarer_prog.ace_failed_attempts;
                    }
                    TeamId::Sn => {
                        sn.level = declarer_prog.level;
                        sn.ace_failed_attempts = declarer_prog.ace_failed_attempts;
                    }
                }

                return Ok(GameOutcome {
                    winner_team: Some(game_winner),
                    next_declarer: winner,
                    progress_ew: ew,
                    progress_sn: sn,
                });
            }
        }

        // Write back: winner promotion already applied to `ew`/`sn`. Only overwrite declarer's
        // level when A-level rules modified `declarer_prog` (never clobber winner's level with a
        // stale `opp_prog` snapshot).
        match declarer {
            TeamId::Ew => {
                ew.ace_failed_attempts = declarer_prog.ace_failed_attempts;
                if declarer_is_a {
                    ew.level = declarer_prog.level;
                }
                sn.ace_failed_attempts = opp_prog.ace_failed_attempts;
            }
            TeamId::Sn => {
                sn.ace_failed_attempts = declarer_prog.ace_failed_attempts;
                if declarer_is_a {
                    sn.level = declarer_prog.level;
                }
                ew.ace_failed_attempts = opp_prog.ace_failed_attempts;
            }
        }

        Ok(GameOutcome {
            winner_team: None,
            next_declarer: winner,
            progress_ew: ew,
            progress_sn: sn,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prog(team: TeamId, level: Level, fails: u32) -> TeamProgress {
        TeamProgress {
            team,
            level,
            ace_failed_attempts: fails,
        }
    }

    #[test]
    fn a_level_declarer_wins_12_game_over() {
        let ew = prog(TeamId::Ew, Level::A, 0);
        let sn = prog(TeamId::Sn, Level::K, 0);
        let out =
            ScoringService::apply_hand(ew, sn, TeamId::Ew, TeamId::Ew, WinType::OneTwo, false)
                .unwrap();
        assert_eq!(out.winner_team, Some(TeamId::Ew));
    }

    #[test]
    fn a_level_declarer_14_increments_fail_and_opponent_wins_on_third() {
        let ew = prog(TeamId::Ew, Level::A, 2);
        let sn = prog(TeamId::Sn, Level::Q, 0);
        let out =
            ScoringService::apply_hand(ew, sn, TeamId::Ew, TeamId::Ew, WinType::OneFour, false)
                .unwrap();
        assert_eq!(out.winner_team, Some(TeamId::Sn));
        assert_eq!(out.progress_ew.level, Level::Two);
        assert_eq!(out.progress_ew.ace_failed_attempts, 0);
    }

    #[test]
    fn a_level_declarer_loses_counts_as_failure_no_demotion_before_three() {
        let ew = prog(TeamId::Ew, Level::A, 1);
        let sn = prog(TeamId::Sn, Level::Ten, 0);
        let out =
            ScoringService::apply_hand(ew, sn, TeamId::Ew, TeamId::Sn, WinType::OneThree, false)
                .unwrap();
        // 输牌1次+原有1次=2次失败，未达3次，不降级
        assert_eq!(out.winner_team, None);
        assert_eq!(out.progress_ew.level, Level::A);
        assert_eq!(out.progress_ew.ace_failed_attempts, 2);
        // SN赢牌升级: Ten -> Q (OneThree delta=2)
        assert_eq!(out.progress_sn.level, Level::Q);
    }

    #[test]
    fn non_a_winner_promotes_by_win_type() {
        let ew = prog(TeamId::Ew, Level::Five, 0);
        let sn = prog(TeamId::Sn, Level::K, 0);
        let out =
            ScoringService::apply_hand(ew, sn, TeamId::Ew, TeamId::Sn, WinType::OneTwo, false)
                .unwrap();
        assert_eq!(out.progress_sn.level, Level::A);
    }

    #[test]
    fn level_promote_capped_at_a() {
        assert_eq!(Level::K.promote_by(4), Level::A);
        assert_eq!(Level::A.promote_by(4), Level::A);
    }
}
