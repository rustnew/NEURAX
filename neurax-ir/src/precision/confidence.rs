//! Confidence scoring per metric per tuning.md
//! 
//! Each metric has its own confidence score calculated independently
//! based on degradation factors.

use serde::Serialize;

/// Confidence score for a single metric
#[derive(Debug, Clone, Serialize)]
pub struct MetricConfidence {
    /// Score 0.0–1.0
    pub score: f64,
    /// Estimated precision (± %)
    pub estimated_error_pct: f64,
    /// Level reached
    pub level: super::PrecisionLevel,
    /// Factors that degrade confidence
    pub degradation_factors: Vec<ConfidenceFactor>,
}

impl Default for MetricConfidence {
    fn default() -> Self {
        Self {
            score: 1.0,
            estimated_error_pct: 0.0,
            level: super::PrecisionLevel::Industrial,
            degradation_factors: Vec::new(),
        }
    }
}

/// Factors that degrade confidence
#[derive(Debug, Clone, Serialize)]
pub enum ConfidenceFactor {
    /// Symbolic dimensions remaining
    SymbolicDimensions { count: u32, impact: f32 },
    /// Custom layer without formula
    CustomLayerNoFormula { layer_id: String, impact: f32 },
    /// GPU not in calibration database
    GpuNotInCalibrationDb { gpu: String, impact: f32 },
    /// Approximated backward ratio
    ApproximatedBackwardRatio { op: String, impact: f32 },
    /// No training steps provided
    NoTrainingStepsProvided { impact: f32 },
    /// Estimated efficiency factor
    EstimatedEfficiencyFactor { op: String, gpu: String, impact: f32 },
    /// Flash Attention not enabled
    FlashAttentionNotEnabled { impact: f32 },
    /// Memory fragmentation estimated
    MemoryFragmentationEstimated { impact: f32 },
}

/// Calculator for confidence scores per metric
pub struct ConfidenceCalculator;

impl ConfidenceCalculator {
    /// Confidence for parameter count metrics
    pub fn for_params(
        has_custom_no_formula: bool,
        symbolic_dims: u32,
    ) -> MetricConfidence {
        let mut score = 1.0f64;
        let mut factors = vec![];

        if has_custom_no_formula {
            score *= 0.70;
            factors.push(ConfidenceFactor::CustomLayerNoFormula {
                layer_id: "unknown".to_string(),
                impact: 0.30,
            });
        }
        if symbolic_dims > 0 {
            // Parameters rarely depend on dynamic shapes
            // Minimal impact for parameters
            score *= 1.0 - (symbolic_dims as f64 * 0.01).min(0.10);
            factors.push(ConfidenceFactor::SymbolicDimensions {
                count: symbolic_dims,
                impact: (symbolic_dims as f32 * 0.01).min(0.10),
            });
        }
        
        MetricConfidence {
            score,
            estimated_error_pct: (1.0 - score) * 100.0,
            level: super::PrecisionLevel::from_confidence(score),
            degradation_factors: factors,
        }
    }

    /// Confidence for FLOPs metrics
    pub fn for_flops(
        symbolic_dims: u32,
        total_dims: u32,
        has_custom_no_formula: bool,
        gpu_in_calibration: bool,
    ) -> MetricConfidence {
        let mut score = 1.0f64;
        let mut factors = vec![];

        let sym_ratio = symbolic_dims as f64 / total_dims.max(1) as f64;
        if sym_ratio > 0.0 {
            score *= 1.0 - sym_ratio * 0.8;
            factors.push(ConfidenceFactor::SymbolicDimensions {
                count: symbolic_dims,
                impact: (sym_ratio * 0.8) as f32,
            });
        }
        if has_custom_no_formula {
            score *= 0.50;
            factors.push(ConfidenceFactor::CustomLayerNoFormula {
                layer_id: "unknown".to_string(),
                impact: 0.50,
            });
        }
        if !gpu_in_calibration {
            score *= 0.85;
            factors.push(ConfidenceFactor::GpuNotInCalibrationDb {
                gpu: "unknown".to_string(),
                impact: 0.15,
            });
        }
        
        MetricConfidence {
            score,
            estimated_error_pct: lerp(1.5, 25.0, 1.0 - score),
            level: super::PrecisionLevel::from_confidence(score),
            degradation_factors: factors,
        }
    }

    /// Confidence for memory metrics
    pub fn for_memory(
        symbolic_dims: u32,
        total_dims: u32,
        has_gradient_checkpointing: bool,
        fragmentation_modeled: bool,
    ) -> MetricConfidence {
        let mut score = 1.0f64;
        let mut factors = vec![];

        let sym_ratio = symbolic_dims as f64 / total_dims.max(1) as f64;
        if sym_ratio > 0.0 {
            score *= 1.0 - sym_ratio * 0.6;
            factors.push(ConfidenceFactor::SymbolicDimensions {
                count: symbolic_dims,
                impact: (sym_ratio * 0.6) as f32,
            });
        }
        if !has_gradient_checkpointing {
            // Without checkpointing, memory estimate is less precise
            score *= 0.90;
        }
        if !fragmentation_modeled {
            score *= 0.88;
            factors.push(ConfidenceFactor::MemoryFragmentationEstimated {
                impact: 0.12,
            });
        }
        
        MetricConfidence {
            score,
            estimated_error_pct: lerp(3.0, 35.0, 1.0 - score),
            level: super::PrecisionLevel::from_confidence(score),
            degradation_factors: factors,
        }
    }

    /// Confidence for latency metrics
    pub fn for_latency(
        flops_confidence: f64,
        gpu_in_calibration: bool,
        ops_have_efficiency_data: f64, // 0.0–1.0 ratio
    ) -> MetricConfidence {
        let mut score = flops_confidence;
        let mut factors = vec![];

        if !gpu_in_calibration {
            score *= 0.65;
            factors.push(ConfidenceFactor::GpuNotInCalibrationDb {
                gpu: "unknown".to_string(),
                impact: 0.35,
            });
        }
        score *= 0.85 + ops_have_efficiency_data * 0.15;

        MetricConfidence {
            score,
            estimated_error_pct: lerp(5.0, 50.0, 1.0 - score),
            level: super::PrecisionLevel::from_confidence(score),
            degradation_factors: factors,
        }
    }

    /// Confidence for cost metrics
    pub fn for_cost(
        latency_confidence: f64,
        has_training_steps: bool,
        cloud_pricing_current: bool,
    ) -> MetricConfidence {
        let mut score = latency_confidence;
        let mut factors = vec![];

        if !has_training_steps {
            score *= 0.75;
            factors.push(ConfidenceFactor::NoTrainingStepsProvided {
                impact: 0.25,
            });
        }
        if !cloud_pricing_current {
            score *= 0.90;
        }
        
        MetricConfidence {
            score,
            estimated_error_pct: lerp(8.0, 60.0, 1.0 - score),
            level: super::PrecisionLevel::from_confidence(score),
            degradation_factors: factors,
        }
    }

    /// Confidence for parallelism metrics
    pub fn for_parallelism(
        has_explicit_config: bool,
        communication_modeled: bool,
    ) -> MetricConfidence {
        let mut score = 1.0f64;
        let factors = vec![];

        if !has_explicit_config {
            score *= 0.70;
        }
        if !communication_modeled {
            score *= 0.80;
        }
        
        MetricConfidence {
            score,
            estimated_error_pct: lerp(5.0, 30.0, 1.0 - score),
            level: super::PrecisionLevel::from_confidence(score),
            degradation_factors: factors,
        }
    }
}

/// Linear interpolation helper
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

/// Overall confidence report for all metrics
#[derive(Debug, Clone, Serialize, Default)]
pub struct ConfidenceReport {
    pub params: MetricConfidence,
    pub flops: MetricConfidence,
    pub memory: MetricConfidence,
    pub latency: MetricConfidence,
    pub cost: MetricConfidence,
    pub parallelism: MetricConfidence,
    /// Overall confidence score (weighted average)
    pub overall_score: f64,
    pub overall_level: super::PrecisionLevel,
}

impl ConfidenceReport {
    /// Calculate overall confidence from individual metrics
    pub fn calculate_overall(&mut self) {
        // Weighted average: latency and memory are most critical
        let weights = [
            ("params", &self.params, 0.10),
            ("flops", &self.flops, 0.20),
            ("memory", &self.memory, 0.25),
            ("latency", &self.latency, 0.25),
            ("cost", &self.cost, 0.10),
            ("parallelism", &self.parallelism, 0.10),
        ];
        
        let total: f64 = weights.iter()
            .map(|(_, conf, w)| conf.score * w)
            .sum();
        
        self.overall_score = total;
        self.overall_level = super::PrecisionLevel::from_confidence(self.overall_score);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PrecisionLevel;

    #[test]
    fn test_params_confidence() {
        let conf = ConfidenceCalculator::for_params(false, 0);
        assert!(conf.score >= 0.95);
        assert_eq!(conf.level, PrecisionLevel::Industrial);
        
        let conf = ConfidenceCalculator::for_params(true, 5);
        assert!(conf.score < 0.95);
        assert!(!conf.degradation_factors.is_empty());
    }

    #[test]
    fn test_flops_confidence() {
        let conf = ConfidenceCalculator::for_flops(0, 10, false, true);
        assert!(conf.score >= 0.85);
        
        let conf = ConfidenceCalculator::for_flops(5, 10, true, false);
        assert!(conf.score < 0.50);
    }

    #[test]
    fn test_latency_confidence() {
        let conf = ConfidenceCalculator::for_latency(0.90, true, 0.95);
        assert!(conf.score >= 0.80);
        
        let conf = ConfidenceCalculator::for_latency(0.90, false, 0.50);
        assert!(conf.score < 0.70);
    }

    #[test]
    fn test_confidence_report() {
        let mut report = ConfidenceReport {
            params: ConfidenceCalculator::for_params(false, 0),
            flops: ConfidenceCalculator::for_flops(0, 10, false, true),
            memory: ConfidenceCalculator::for_memory(0, 10, true, true),
            latency: ConfidenceCalculator::for_latency(0.90, true, 0.90),
            cost: ConfidenceCalculator::for_cost(0.85, true, true),
            parallelism: ConfidenceCalculator::for_parallelism(true, true),
            ..Default::default()
        };
        
        report.calculate_overall();
        assert!(report.overall_score > 0.85);
    }
}
