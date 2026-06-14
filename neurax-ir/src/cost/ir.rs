//! Cost IR structures

/// Cost IR - dialecte économique
#[derive(Debug, Clone)]
pub struct CostIR {
    pub pricing_model: PricingModel,
    pub metrics: CostMetrics,
    pub metrics_done: bool,
}

impl Default for CostIR {
    fn default() -> Self {
        Self {
            pricing_model: PricingModel::default(),
            metrics: CostMetrics::default(),
            metrics_done: false,
        }
    }
}

/// Pricing model
#[derive(Debug, Clone)]
pub struct PricingModel {
    pub gpu_hour_usd: f64,
    pub energy_kwh_usd: f64,
    pub pue_factor: f64,
    pub gpu_tdp_watts: f64,
    pub co2_per_kwh: f64,
}

impl Default for PricingModel {
    fn default() -> Self {
        Self {
            gpu_hour_usd: 3.0,
            energy_kwh_usd: 0.12,
            pue_factor: 1.2,
            gpu_tdp_watts: 400.0,
            co2_per_kwh: 0.233, // kg CO2/kWh (EU average)
        }
    }
}

/// Cost metrics (Métriques 41-44)
#[derive(Debug, Clone, Default)]
pub struct CostMetrics {
    /// Métrique 41: Temps d'entraînement (heures)
    pub training_time_hours: f64,
    /// Métrique 41: Coût d'entraînement (USD)
    pub training_cost_usd: f64,
    /// GPU hours totaux
    pub gpu_hours_total: f64,
    /// Métrique 44: Énergie consommée (kWh)
    pub energy_kwh: f64,
    /// Métrique 44: CO2 émis (kg)
    pub co2_kg: f64,
    /// Coût par token (USD)
    pub cost_per_token_usd: f64,
    /// Coût par million de tokens (USD)
    pub cost_per_million_tokens_usd: f64,
    /// Coût mensuel d'inférence (USD)
    pub monthly_inference_cost_usd: f64,
    /// Coût par step (USD)
    pub cost_per_step_usd: f64,
}

impl CostMetrics {
    pub fn is_valid(&self) -> bool {
        self.training_cost_usd >= 0.0 && self.energy_kwh > 0.0
    }
}
