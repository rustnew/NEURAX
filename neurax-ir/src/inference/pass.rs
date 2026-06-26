//! Inference Pass — calcul analytique de tous les widgets du BehaviorDashboard.
//!
//! Toutes les métriques sont dérivées analytiquement depuis les paramètres
//! d'inférence et les propriétés structurelles du modèle. Aucun modèle ML
//! externe n'est requis.

use super::ir::{
    HallucinationRisk, InferenceParams, InferenceReport, RiskLevel, RiskOverview,
    RouterStability, SamplingVolatility, StabilityIndex, StabilityLevel,
};

pub struct InferencePass;

impl InferencePass {
    /// Point d'entrée principal : calcule l'intégralité du rapport d'inférence.
    pub fn run(params: &InferenceParams) -> InferenceReport {
        let stability = Self::compute_stability(params);
        let entropy = Self::compute_entropy_evolution(params);
        let noise_schedule = if params.architecture_family.to_lowercase().contains("diffusion") {
            Some(Self::compute_noise_schedule(20))
        } else {
            None
        };
        let hallucination = Self::compute_hallucination_risk(params);
        let attention = Self::compute_attention_focus(params);
        let state_stability = Self::compute_state_stability(params);
        let context_deg = Self::compute_context_degradation(params);
        let volatility = Self::compute_sampling_volatility(params);
        let router = if params.architecture_family.to_lowercase() == "moe" {
            Some(Self::compute_router_stability(params))
        } else {
            None
        };
        let risks = Self::compute_risk_overview(params);

        InferenceReport {
            stability_index: stability,
            entropy_evolution: entropy,
            noise_schedule,
            hallucination_risk: hallucination,
            attention_focus: attention,
            state_stability,
            context_degradation: context_deg,
            sampling_volatility: volatility,
            router_stability: router,
            risk_overview: risks,
        }
    }

    // ── Widget 1 : Generation Stability Index ────────────────────────────────

    fn compute_stability(p: &InferenceParams) -> StabilityIndex {
        let effective_temp = if p.high_temperature_mode {
            p.temperature.max(1.6)
        } else if p.low_temperature_mode {
            p.temperature.min(0.3)
        } else {
            p.temperature
        };

        let quant_penalty = match p.quantization_level.as_str() {
            "int4" => 0.30,
            "int8" => 0.15,
            "bf16" => 0.02,
            _ => 0.0,
        };

        let beam_bonus = (p.beam_width as f64 - 1.0) * 0.05;
        let adversarial_penalty = if p.adversarial_prompt { 0.20 } else { 0.0 };

        let base_score = 1.0 - (effective_temp / 2.0).min(1.0);
        let score =
            (base_score + beam_bonus - quant_penalty - adversarial_penalty).clamp(0.0, 1.0);

        let level = if score >= 0.75 {
            StabilityLevel::Stable
        } else if score >= 0.50 {
            StabilityLevel::Drift
        } else if score >= 0.25 {
            StabilityLevel::Unstable
        } else {
            StabilityLevel::Chaotic
        };

        StabilityIndex { score, level }
    }

    // ── Widget 2 : Entropy Evolution ─────────────────────────────────────────

    fn compute_entropy_evolution(p: &InferenceParams) -> Vec<f64> {
        let n = 20usize;
        let max_entropy = p.temperature * 3.5 * (1.0 + (1.0 - p.top_p) * 0.5);
        let decay_rate = p.repetition_penalty * 0.3 + (1.0 - p.top_p) * 0.5;

        (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                let base = max_entropy * (-decay_rate * t).exp();
                let noise = (t * 17.0).sin() * 0.05 * p.temperature;
                (base + noise).max(0.01)
            })
            .collect()
    }

    // ── Widget 3 : Noise Schedule (diffusion uniquement) ─────────────────────

    fn compute_noise_schedule(steps: usize) -> Vec<f64> {
        // Cosine schedule inverse : t=T (bruit max) → t=0 (propre)
        (0..steps)
            .map(|i| {
                let t = (steps - 1 - i) as f64 / (steps - 1) as f64;
                1.0 - (1.0 - t * t).sqrt()
            })
            .collect()
    }

    // ── Widget 4 : Hallucination Risk ────────────────────────────────────────

    fn compute_hallucination_risk(p: &InferenceParams) -> HallucinationRisk {
        let temp_factor = p.temperature / 2.0;
        let beam_factor = 1.0 / p.beam_width as f64;
        let diversity_factor = p.top_p;
        let rep_mitigation = (p.repetition_penalty - 1.0) * 0.3;
        let adversarial_factor = if p.adversarial_prompt { 0.30 } else { 0.0 };

        let raw_risk = (temp_factor * 0.40
            + beam_factor * 0.20
            + diversity_factor * 0.20
            + adversarial_factor
            - rep_mitigation)
            .clamp(0.0, 1.0);

        let confidence = ((1.0 - raw_risk) * 100.0).round();

        let risk = if raw_risk < 0.35 {
            RiskLevel::Low
        } else if raw_risk < 0.65 {
            RiskLevel::Medium
        } else {
            RiskLevel::High
        };

        HallucinationRisk { risk, confidence }
    }

    // ── Widget 5 : Attention Focus ───────────────────────────────────────────

    fn compute_attention_focus(p: &InferenceParams) -> Vec<f64> {
        let n = 12usize;
        match p.attention_type.as_str() {
            "flash" => {
                // Flash attention : pic sur tokens récents + début
                (0..n)
                    .map(|i| {
                        let pos = i as f64 / (n - 1) as f64;
                        let peak_recent = (-4.0 * (pos - 0.85).powi(2)).exp() * 0.90;
                        let peak_start = (-4.0 * (pos - 0.05).powi(2)).exp() * 0.40;
                        (peak_recent + peak_start).clamp(0.05, 1.0)
                    })
                    .collect()
            }
            "linear" => {
                // Linear attention : décroissance douce depuis le début
                (0..n)
                    .map(|i| {
                        let pos = i as f64 / (n - 1) as f64;
                        (0.30 + 0.70 * (1.0 - pos * 0.5)).clamp(0.10, 1.0)
                    })
                    .collect()
            }
            _ => {
                // Standard : localité + contexte global progressif
                (0..n)
                    .map(|i| {
                        let pos = i as f64 / (n - 1) as f64;
                        let local = (-3.0 * (pos - 1.0).powi(2)).exp() * 0.80;
                        let global = 0.15 + pos * 0.20;
                        (local + global).clamp(0.05, 1.0)
                    })
                    .collect()
            }
        }
    }

    // ── Widget 6 : State Stability (SSM) ─────────────────────────────────────

    fn compute_state_stability(p: &InferenceParams) -> f64 {
        let base = match p.architecture_family.to_lowercase().as_str() {
            "ssm" | "mamba" => {
                let seq_ratio = p.prompt_length as f64 / 8192.0;
                1.0 - (seq_ratio.min(0.80)) * 0.40
            }
            "transformer" => {
                let kv_bonus = if p.kv_cache_reuse { 0.10 } else { 0.0 };
                0.85 + kv_bonus
            }
            "moe" => 0.82,
            "diffusion" => 0.90, // diffusion models: state stability not sequential
            _ => 0.80,
        };

        let stress_penalty = if p.long_context_simulation { 0.15 } else { 0.0 };
        (base - stress_penalty).clamp(0.10, 1.0)
    }

    // ── Widget 7 : Context Degradation ───────────────────────────────────────

    fn compute_context_degradation(p: &InferenceParams) -> f64 {
        let total_capacity = 32_768u32;
        let used = p.prompt_length.saturating_add(p.max_output_tokens);
        let sliding_bonus = if p.sliding_window { 0.15 } else { 0.0 };
        let kv_bonus = if p.kv_cache_reuse { 0.10 } else { 0.0 };
        let lc_penalty = if p.long_context_simulation { 0.20 } else { 0.0 };

        let utilization = used as f64 / total_capacity as f64;
        let effective_pct =
            ((1.0 - utilization + sliding_bonus + kv_bonus - lc_penalty) * 100.0)
                .clamp(5.0, 100.0);

        effective_pct.round()
    }

    // ── Widget 8 : Sampling Volatility ───────────────────────────────────────

    fn compute_sampling_volatility(p: &InferenceParams) -> SamplingVolatility {
        let raw_diversity =
            (p.temperature * 0.5 * p.top_p) / (p.repetition_penalty * p.beam_width as f64);
        let diversity = raw_diversity.clamp(0.0, 1.0);

        let determinism = (1.0 / (1.0 + p.temperature * p.top_p))
            * (1.0 + (p.beam_width as f64 - 1.0) * 0.20);
        let determinism = determinism.clamp(0.0, 1.0);

        SamplingVolatility {
            diversity,
            determinism,
        }
    }

    // ── Widget 9 : Router Stability (MoE) ────────────────────────────────────

    fn compute_router_stability(p: &InferenceParams) -> RouterStability {
        let (base_stability, distribution_shape) =
            match p.moe_router_mode.as_deref().unwrap_or("top-k") {
                "top-k" => (0.92f64, "top_k"),
                "expert-choice" => (0.85f64, "balanced"),
                "soft" => (0.70f64, "distributed"),
                _ => (0.88f64, "top_k"),
            };

        let temp_penalty = (p.temperature - 0.7).max(0.0) * 0.05;
        let final_stability = (base_stability - temp_penalty).clamp(0.0, 1.0);

        let n_experts = 8usize;
        let distribution: Vec<f64> = match distribution_shape {
            "top_k" => (0..n_experts)
                .map(|i| if i < 2 { 0.25 } else { 0.50 / (n_experts - 2) as f64 })
                .collect(),
            "balanced" => vec![1.0 / n_experts as f64; n_experts],
            _ => (0..n_experts)
                .map(|i| {
                    let base = 1.0 / n_experts as f64;
                    let var = (i as f64 * 1.7).sin() * 0.05;
                    (base + var).max(0.01)
                })
                .collect(),
        };

        RouterStability {
            stability: final_stability,
            distribution,
        }
    }

    // ── Widget 10 : Inference Risk Overview ──────────────────────────────────

    fn compute_risk_overview(p: &InferenceParams) -> RiskOverview {
        // Coherence : température élevée + beam étroit = sorties incohérentes
        let coherence_score =
            (p.temperature * 0.50 - (p.beam_width as f64 - 1.0) * 0.10).clamp(0.0, 1.0);
        let coherence = Self::score_to_risk(coherence_score);

        // Overconfidence : faible top_p + faible rep_penalty
        let overconf_score = ((1.0 - p.top_p) * 0.50
            + (1.0 - (p.repetition_penalty - 1.0).min(1.0)) * 0.30)
            .clamp(0.0, 1.0);
        let overconfidence = Self::score_to_risk(overconf_score);

        // Collapse (MoE) : risque de convergence vers peu d'experts
        let collapse_score = if p.architecture_family.to_lowercase() == "moe" {
            match p.moe_router_mode.as_deref().unwrap_or("top-k") {
                "top-k" => 0.10,
                "expert-choice" => 0.25,
                "soft" => 0.40,
                _ => 0.15,
            }
        } else {
            0.05
        };
        let collapse = Self::score_to_risk(collapse_score);

        // Degeneration : répétition + température élevée
        let degen_score =
            ((2.0 - p.repetition_penalty).max(0.0) * 0.40 + p.temperature * 0.20
                - p.frequency_penalty * 0.10)
                .clamp(0.0, 1.0);
        let degeneration = Self::score_to_risk(degen_score);

        RiskOverview {
            coherence,
            overconfidence,
            collapse,
            degeneration,
        }
    }

    // ── Utilitaire ────────────────────────────────────────────────────────────

    fn score_to_risk(score: f64) -> RiskLevel {
        if score < 0.33 {
            RiskLevel::Low
        } else if score < 0.66 {
            RiskLevel::Medium
        } else {
            RiskLevel::High
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params() -> InferenceParams {
        InferenceParams::default()
    }

    #[test]
    fn test_stability_low_temp_is_stable() {
        let mut p = default_params();
        p.temperature = 0.2;
        let r = InferencePass::run(&p);
        assert_eq!(r.stability_index.level, StabilityLevel::Stable);
    }

    #[test]
    fn test_stability_high_temp_is_chaotic() {
        let mut p = default_params();
        p.temperature = 2.0;
        p.beam_width = 1;
        let r = InferencePass::run(&p);
        assert!(
            r.stability_index.level == StabilityLevel::Chaotic
                || r.stability_index.level == StabilityLevel::Unstable
        );
    }

    #[test]
    fn test_entropy_evolution_length() {
        let p = default_params();
        let r = InferencePass::run(&p);
        assert_eq!(r.entropy_evolution.len(), 20);
    }

    #[test]
    fn test_noise_schedule_only_for_diffusion() {
        let mut p = default_params();
        p.architecture_family = "transformer".to_string();
        let r = InferencePass::run(&p);
        assert!(r.noise_schedule.is_none());

        p.architecture_family = "diffusion".to_string();
        let r = InferencePass::run(&p);
        assert!(r.noise_schedule.is_some());
        assert_eq!(r.noise_schedule.unwrap().len(), 20);
    }

    #[test]
    fn test_router_stability_only_for_moe() {
        let mut p = default_params();
        p.architecture_family = "transformer".to_string();
        let r = InferencePass::run(&p);
        assert!(r.router_stability.is_none());

        p.architecture_family = "moe".to_string();
        let r = InferencePass::run(&p);
        assert!(r.router_stability.is_some());
        assert_eq!(r.router_stability.unwrap().distribution.len(), 8);
    }

    #[test]
    fn test_attention_focus_length() {
        let p = default_params();
        let r = InferencePass::run(&p);
        assert_eq!(r.attention_focus.len(), 12);
    }

    #[test]
    fn test_context_degradation_range() {
        let p = default_params();
        let r = InferencePass::run(&p);
        assert!((5.0..=100.0).contains(&r.context_degradation));
    }

    #[test]
    fn test_sampling_volatility_range() {
        let p = default_params();
        let r = InferencePass::run(&p);
        assert!((0.0..=1.0).contains(&r.sampling_volatility.diversity));
        assert!((0.0..=1.0).contains(&r.sampling_volatility.determinism));
    }

    #[test]
    fn test_high_beam_reduces_hallucination() {
        let mut p = default_params();
        p.beam_width = 1;
        let r1 = InferencePass::run(&p);
        p.beam_width = 8;
        let r2 = InferencePass::run(&p);
        // More beam = lower hallucination or equal
        let r1_score = match r1.hallucination_risk.risk {
            RiskLevel::Low => 0,
            RiskLevel::Medium => 1,
            RiskLevel::High => 2,
        };
        let r2_score = match r2.hallucination_risk.risk {
            RiskLevel::Low => 0,
            RiskLevel::Medium => 1,
            RiskLevel::High => 2,
        };
        assert!(r2_score <= r1_score);
    }

    #[test]
    fn test_adversarial_increases_risk() {
        let mut p = default_params();
        p.temperature = 0.7;
        p.adversarial_prompt = false;
        let r1 = InferencePass::run(&p);
        p.adversarial_prompt = true;
        let r2 = InferencePass::run(&p);
        assert!(r2.stability_index.score <= r1.stability_index.score);
    }

    #[test]
    fn test_mamba_state_stability_degrades_with_long_context() {
        let mut p = default_params();
        p.architecture_family = "mamba".to_string();
        p.long_context_simulation = false;
        let r1 = InferencePass::run(&p);
        p.long_context_simulation = true;
        let r2 = InferencePass::run(&p);
        assert!(r2.state_stability <= r1.state_stability);
    }
}
