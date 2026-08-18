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
        let params_path = "/home/Cooki/domains/gg.meaigo.eu.org/clawguandan/advanced_params.json";
        let params = AdvancedBotParams::load(params_path).unwrap_or_else(|_| {
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
