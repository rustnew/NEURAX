//! Inference IR — structures de données pour la simulation d'inférence.

use serde::{Deserialize, Serialize};

/// Paramètres d'inférence fournis par le frontend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceParams {
    // Sampling Strategy
    pub temperature: f64,
    pub top_k: u32,
    pub top_p: f64,
    pub beam_width: u32,
    pub repetition_penalty: f64,
    pub presence_penalty: f64,
    pub frequency_penalty: f64,
    // Context Configuration
    pub prompt_length: u32,
    pub max_output_tokens: u32,
    pub sliding_window: bool,
    pub kv_cache_reuse: bool,
    // Model Behavior
    pub architecture_family: String,
    pub attention_type: String,
    pub moe_router_mode: Option<String>,
    pub quantization_level: String,
    // Stability Stress Tests
    pub long_context_simulation: bool,
    pub adversarial_prompt: bool,
    pub high_temperature_mode: bool,
    pub low_temperature_mode: bool,
}

impl Default for InferenceParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            beam_width: 1,
            repetition_penalty: 1.1,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            prompt_length: 2048,
            max_output_tokens: 1024,
            sliding_window: true,
            kv_cache_reuse: true,
            architecture_family: "transformer".to_string(),
            attention_type: "standard".to_string(),
            moe_router_mode: None,
            quantization_level: "fp16".to_string(),
            long_context_simulation: false,
            adversarial_prompt: false,
            high_temperature_mode: false,
            low_temperature_mode: false,
        }
    }
}

// ── Output types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StabilityLevel {
    Stable,
    Drift,
    Unstable,
    Chaotic,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RiskLevel {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityIndex {
    /// Score normalisé [0, 1] — 1 = parfaitement stable
    pub score: f64,
    pub level: StabilityLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HallucinationRisk {
    pub risk: RiskLevel,
    /// Confiance estimée [0, 100]
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingVolatility {
    /// Diversité de sortie [0, 1]
    pub diversity: f64,
    /// Déterminisme [0, 1]
    pub determinism: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterStability {
    /// Consistance du routage [0, 1]
    pub stability: f64,
    /// Distribution de charge par expert (N valeurs, somme ≈ 1)
    pub distribution: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskOverview {
    pub coherence: RiskLevel,
    pub overconfidence: RiskLevel,
    pub collapse: RiskLevel,
    pub degeneration: RiskLevel,
}

/// Rapport complet retourné par l'endpoint `/inference/simulate`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceReport {
    /// Widget 1 — Generation Stability Index
    pub stability_index: StabilityIndex,
    /// Widget 2 — Entropy Evolution (20 points)
    pub entropy_evolution: Vec<f64>,
    /// Widget 3 — Noise Schedule Curve (diffusion seulement)
    pub noise_schedule: Option<Vec<f64>>,
    /// Widget 4 — Hallucination Risk
    pub hallucination_risk: HallucinationRisk,
    /// Widget 5 — Attention Focus (12 tokens)
    pub attention_focus: Vec<f64>,
    /// Widget 6 — State Stability / SSM [0, 1]
    pub state_stability: f64,
    /// Widget 7 — Context Degradation : % de fenêtre effective restante
    pub context_degradation: f64,
    /// Widget 8 — Sampling Volatility
    pub sampling_volatility: SamplingVolatility,
    /// Widget 9 — Router Stability (MoE seulement)
    pub router_stability: Option<RouterStability>,
    /// Widget 10 — Inference Risk Overview
    pub risk_overview: RiskOverview,
}
