use crate::game::card::{RuleContext, Suit, is_wild, parse_card_symbol};

pub struct WildcardResolver;

impl WildcardResolver {
    pub fn resolve(
        cards: &[String],
        wild_targets: Option<&[String]>,
        ctx: RuleContext,
    ) -> Result<Vec<crate::game::card::Card>, String> {
        let parsed: Vec<_> = cards
            .iter()
            .map(|s| parse_card_symbol(s))
            .collect::<Result<_, _>>()?;
        let wild_count = parsed.iter().filter(|c| is_wild(**c, ctx)).count();

        let targets = wild_targets.unwrap_or(&[]);
        if wild_count != targets.len() {
            return Err(format!(
                "wildTargets length mismatch: expected {}, got {}",
                wild_count,
                targets.len()
            ));
        }

        // 房规（对齐 cards.js resolveWildcards）：两张逢人配允许配在同一个 rank 上
        // （例如自然对 X + 双百搭都当 X，构成四张炸）。旧版「两张百搭不能同 rank」
        // 的限制已按 JS 最终策略移除；纯自然炸弹不受影响。
        // 仍然禁止：百搭表示王（下方逐张校验）。

        let mut resolved = Vec::with_capacity(parsed.len());
        let mut ti = 0usize;
        for c in parsed {
            if is_wild(c, ctx) {
                let t = parse_card_symbol(&targets[ti])?;
                if t.suit == Suit::Joker {
                    return Err("wild card cannot represent joker".into());
                }
                resolved.push(t);
                ti += 1;
            } else {
                resolved.push(c);
            }
        }
        Ok(resolved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::card::HandLevel;

    #[test]
    fn reject_joker_target_for_wild() {
        let cards = vec!["♥2".to_string()];
        let targets = vec!["🃏R".to_string()];
        let ctx = RuleContext {
            hand_level: HandLevel::Two,
        };
        let err = WildcardResolver::resolve(&cards, Some(&targets), ctx).unwrap_err();
        assert!(err.contains("cannot represent joker"));
    }

    #[test]
    fn length_mismatch_errors() {
        let cards = vec!["♥2".into(), "♥2".into()];
        let targets = vec!["♠K".into()];
        let ctx = RuleContext {
            hand_level: HandLevel::Two,
        };
        let err = WildcardResolver::resolve(&cards, Some(&targets), ctx).unwrap_err();
        assert!(err.contains("mismatch"));
    }

    #[test]
    fn resolves_single_wild() {
        let cards = vec!["♥2".into()];
        let targets = vec!["♠K".into()];
        let ctx = RuleContext {
            hand_level: HandLevel::Two,
        };
        let out = WildcardResolver::resolve(&cards, Some(&targets), ctx).unwrap();
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn dual_wilds_may_share_target_rank() {
        // 房规：两张逢人配可配同一 rank（如双百搭都当 5，与自然对成四炸）。
        let cards = vec!["♠5".into(), "♦5".into(), "♥2".into(), "♥2".into()];
        let targets = vec!["♣5".into(), "♥5".into()];
        let ctx = RuleContext {
            hand_level: HandLevel::Two,
        };
        let out = WildcardResolver::resolve(&cards, Some(&targets), ctx).unwrap();
        assert_eq!(out.len(), 4);
    }
}
