use crate::bot::plugin::BotPlugin;
use crate::bot::policies::PlayPolicy;
use std::sync::Arc;

pub use self::params::AdvancedBotParams;
use self::play_policy::AdvancedPlayPolicy;

mod hand_tracker;
mod params;
mod play_policy;
mod prob_reasoner;

#[derive(Clone)]
pub struct AdvancedBotPlugin {
    play: Arc<dyn PlayPolicy>,
}

impl Default for AdvancedBotPlugin {
    fn default() -> Self {
        // 房规可移植性：默认相对 cwd（daemon/bot 均以部署目录为 cwd），可用 CLAW_PARAMS_PATH 覆盖
        let params_path = std::env::var("CLAW_PARAMS_PATH")
            .unwrap_or_else(|_| "./advanced_params.json".to_string());
        let params = AdvancedBotParams::load(&params_path).unwrap_or_else(|_| {
            eprintln!("[advanced-bot] No {} found, using default balanced profile", params_path);
            AdvancedBotParams::default_balanced()
        });
        eprintln!("[advanced-bot] Loaded params: {:?}", params);
        Self::with_params(params)
    }
}

impl AdvancedBotPlugin {
    pub fn with_params(params: AdvancedBotParams) -> Self {
        let params = Arc::new(params);
        Self {
            play: Arc::new(AdvancedPlayPolicy::new(Arc::clone(&params))),
        }
    }
}

impl BotPlugin for AdvancedBotPlugin {
    fn plugin_id(&self) -> &'static str {
        "advanced-bot"
    }

    fn play_policy(&self) -> Arc<dyn PlayPolicy> {
        Arc::clone(&self.play)
    }
}
