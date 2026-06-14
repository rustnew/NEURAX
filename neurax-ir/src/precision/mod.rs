//! Precision levels and resolution strategies per tuning.md
//! 
//! Implements the 4-level precision system:
//! - Level 1: Approximative (±20-30%)
//! - Level 2: Estimation (±10-15%)
//! - Level 3: Production (±5-8%)
//! - Level 4: Industrial (±1-3%)

pub mod confidence;
pub mod backward;


/// Precision level for metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PrecisionLevel {
    /// ±20-30% - Heuristics, symbolic shapes
    Approximative,
    /// ±10-15% - Generic formulas, partial shapes
    Estimation,
    /// ±5-8% - Exact formulas, resolved shapes
    Production,
    /// ±1-3% - Calibrated formulas, concrete shapes
    Industrial,
}

impl Default for PrecisionLevel {
    fn default() -> Self {
        Self::Estimation
    }
}

impl PrecisionLevel {
    /// Get estimated error percentage for this level
    pub fn error_range(&self) -> (f64, f64) {
        match self {
            Self::Approximative => (20.0, 30.0),
            Self::Estimation => (10.0, 15.0),
            Self::Production => (5.0, 8.0),
            Self::Industrial => (1.0, 3.0),
        }
    }
    
    /// Convert confidence score (0.0-1.0) to precision level
    pub fn from_confidence(score: f64) -> Self {
        match score {
            s if s >= 0.95 => Self::Industrial,
            s if s >= 0.85 => Self::Production,
            s if s >= 0.70 => Self::Estimation,
            _ => Self::Approximative,
        }
    }
}

/// Dimension value - either known concrete or symbolic
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dim {
    Known(u64),
    Symbolic(String),
}

impl Dim {
    pub fn is_known(&self) -> bool {
        matches!(self, Self::Known(_))
    }
    
    pub fn is_symbolic(&self) -> bool {
        matches!(self, Self::Symbolic(_))
    }
    
    pub fn value(&self) -> Option<u64> {
        match self {
            Self::Known(v) => Some(*v),
            _ => None,
        }
    }
}

impl Default for Dim {
    fn default() -> Self {
        Self::Symbolic("unknown".to_string())
    }
}

/// Source of shape resolution
#[derive(Debug, Clone, serde::Serialize)]
pub enum ShapeSource {
    /// Provided explicitly in JSON
    ExplicitJson,
    /// Inferred from global_params
    GlobalParamInferred,
    /// Propagated from neighbor node
    NeighborPropagated,
    /// Bound to batch_size or seq_len
    TrainingConfigBound,
    /// Deduced from architectural rule
    ArchitectureRule,
    /// Fallback from calibration DB
    CalibrationFallback,
}

/// Resolution context for dimension resolution
#[derive(Debug, Clone)]
pub struct ResolutionContext {
    pub training: TrainingConfig,
    pub global: GlobalParams,
    pub data: DataConfig,
}

#[derive(Debug, Clone, Default)]
pub struct TrainingConfig {
    pub batch_size: Option<u64>,
    pub seq_len: Option<u64>,
}

#[derive(Debug, Clone, Default)]
pub struct GlobalParams {
    pub hidden_size: Option<u64>,
    pub intermediate_size: Option<u64>,
    pub vocab_size: Option<u64>,
    pub num_attention_heads: Option<u32>,
    pub num_kv_heads: Option<u32>,
    pub num_experts: Option<u32>,
    pub ssm_state_size: Option<u64>,
}

#[derive(Debug, Clone, Default)]
pub struct DataConfig {
    pub image_height: Option<u64>,
    pub image_width: Option<u64>,
    pub image_channels: Option<u64>,
}

/// Report of dimension resolution
#[derive(Debug, Clone, Default)]
pub struct ResolutionReport {
    pub resolved: Vec<(usize, String, u64, &'static str)>,
    pub unresolved: Vec<(usize, String)>,
}

impl ResolutionReport {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn resolution_ratio(&self) -> f64 {
        let total = self.resolved.len() + self.unresolved.len();
        if total == 0 {
            1.0
        } else {
            self.resolved.len() as f64 / total as f64
        }
    }
}

/// Strategy for resolving symbolic dimensions
pub trait ResolutionStrategy: Send + Sync {
    fn resolve(&self, sym: &str, ctx: &ResolutionContext) -> Option<u64>;
    fn name(&self) -> &'static str;
}

/// Strategy 1: Concrete value from JSON
pub struct ConcreteShapeStrategy;

impl ResolutionStrategy for ConcreteShapeStrategy {
    fn resolve(&self, sym: &str, ctx: &ResolutionContext) -> Option<u64> {
        match sym {
            "B" | "batch" | "batch_size" => ctx.training.batch_size,
            "S" | "seq" | "seq_len" | "T" => ctx.training.seq_len,
            "D" | "H" | "hidden" | "d_model" => ctx.global.hidden_size,
            "I" | "ff" | "intermediate" => ctx.global.intermediate_size,
            "V" | "vocab" => ctx.global.vocab_size,
            "heads" | "H_heads" => ctx.global.num_attention_heads.map(|h| h as u64),
            "kv_heads" => ctx.global.num_kv_heads.map(|h| h as u64),
            "d_head" | "head_dim" => {
                let h = ctx.global.hidden_size?;
                let n = ctx.global.num_attention_heads? as u64;
                if n > 0 { Some(h / n) } else { None }
            }
            "N_exp" | "num_experts" => ctx.global.num_experts.map(|e| e as u64),
            "d_state" | "ssm_state" => ctx.global.ssm_state_size,
            "H_img" | "W_img" => ctx.data.image_height,
            "C_img" => ctx.data.image_channels,
            _ => None,
        }
    }
    
    fn name(&self) -> &'static str { "concrete_json" }
}

/// Strategy 2: Architecture rule inference
pub struct ArchitectureRuleStrategy;

impl ResolutionStrategy for ArchitectureRuleStrategy {
    fn resolve(&self, sym: &str, ctx: &ResolutionContext) -> Option<u64> {
        match sym {
            // GQA: head_dim = hidden / num_heads
            "d_head" | "head_dim" => {
                let h = ctx.global.hidden_size?;
                let n = ctx.global.num_attention_heads? as u64;
                if n > 0 { Some(h / n) } else { None }
            }
            // FFN intermediate typically 4x hidden for standard, 8/3x for SwiGLU
            "I" | "intermediate" if ctx.global.intermediate_size.is_none() => {
                ctx.global.hidden_size.map(|h| h * 4)
            }
            _ => None,
        }
    }
    
    fn name(&self) -> &'static str { "architecture_rule" }
}

/// Precision resolver with cascade of strategies
pub struct PrecisionResolver {
    pub target_level: PrecisionLevel,
    pub strategies: Vec<Box<dyn ResolutionStrategy>>,
}

impl PrecisionResolver {
    pub fn new_industrial() -> Self {
        Self {
            target_level: PrecisionLevel::Industrial,
            strategies: vec![
                Box::new(ConcreteShapeStrategy),
                Box::new(ArchitectureRuleStrategy),
            ],
        }
    }
    
    pub fn new_production() -> Self {
        Self {
            target_level: PrecisionLevel::Production,
            strategies: vec![
                Box::new(ConcreteShapeStrategy),
                Box::new(ArchitectureRuleStrategy),
            ],
        }
    }
    
    /// Resolve dimensions using cascade of strategies
    pub fn resolve_dims(
        &self,
        dims: &mut Vec<Dim>,
        ctx: &ResolutionContext,
    ) -> ResolutionReport {
        let mut report = ResolutionReport::new();
        
        for (i, dim) in dims.iter_mut().enumerate() {
            if let Dim::Symbolic(s) = dim {
                let sym = s.clone(); // Clone to avoid borrow issues
                for strategy in &self.strategies {
                    if let Some(value) = strategy.resolve(&sym, ctx) {
                        *dim = Dim::Known(value);
                        report.resolved.push((i, sym.clone(), value, strategy.name()));
                        break;
                    }
                }
                if dim.is_symbolic() {
                    report.unresolved.push((i, sym));
                }
            }
        }
        
        report
    }
    
    /// Calculate precision level based on resolution ratio
    pub fn compute_precision_level(&self, report: &ResolutionReport) -> PrecisionLevel {
        let ratio = report.resolution_ratio();
        PrecisionLevel::from_confidence(ratio)
    }
}

/// Target precision by metric type
#[derive(Debug, Clone, serde::Serialize)]
pub struct MetricPrecisionTarget {
    pub metric_name: String,
    pub level_1_error: f64,
    pub level_2_error: f64,
    pub level_3_error: f64,
    pub level_4_error: f64,
}

/// Precision targets per tuning.md Table 1.1
pub fn get_precision_targets() -> Vec<MetricPrecisionTarget> {
    vec![
        MetricPrecisionTarget { metric_name: "total_parameters".into(), level_1_error: 5.0, level_2_error: 1.0, level_3_error: 0.1, level_4_error: 0.01 },
        MetricPrecisionTarget { metric_name: "active_parameters".into(), level_1_error: 10.0, level_2_error: 3.0, level_3_error: 1.0, level_4_error: 0.1 },
        MetricPrecisionTarget { metric_name: "flops_forward".into(), level_1_error: 25.0, level_2_error: 10.0, level_3_error: 5.0, level_4_error: 1.5 },
        MetricPrecisionTarget { metric_name: "flops_backward".into(), level_1_error: 30.0, level_2_error: 15.0, level_3_error: 8.0, level_4_error: 3.0 },
        MetricPrecisionTarget { metric_name: "vram_inference".into(), level_1_error: 30.0, level_2_error: 15.0, level_3_error: 8.0, level_4_error: 3.0 },
        MetricPrecisionTarget { metric_name: "vram_training".into(), level_1_error: 35.0, level_2_error: 18.0, level_3_error: 10.0, level_4_error: 5.0 },
        MetricPrecisionTarget { metric_name: "latency_forward".into(), level_1_error: 50.0, level_2_error: 25.0, level_3_error: 12.0, level_4_error: 5.0 },
        MetricPrecisionTarget { metric_name: "training_cost".into(), level_1_error: 60.0, level_2_error: 30.0, level_3_error: 15.0, level_4_error: 8.0 },
        MetricPrecisionTarget { metric_name: "energy_kwh".into(), level_1_error: 70.0, level_2_error: 35.0, level_3_error: 18.0, level_4_error: 10.0 },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_precision_level_from_confidence() {
        assert_eq!(PrecisionLevel::from_confidence(0.97), PrecisionLevel::Industrial);
        assert_eq!(PrecisionLevel::from_confidence(0.90), PrecisionLevel::Production);
        assert_eq!(PrecisionLevel::from_confidence(0.75), PrecisionLevel::Estimation);
        assert_eq!(PrecisionLevel::from_confidence(0.50), PrecisionLevel::Approximative);
    }
    
    #[test]
    fn test_resolution_strategy() {
        let ctx = ResolutionContext {
            training: TrainingConfig { batch_size: Some(32), seq_len: Some(512) },
            global: GlobalParams { hidden_size: Some(768), ..Default::default() },
            data: DataConfig::default(),
        };
        
        let strategy = ConcreteShapeStrategy;
        assert_eq!(strategy.resolve("B", &ctx), Some(32));
        assert_eq!(strategy.resolve("S", &ctx), Some(512));
        assert_eq!(strategy.resolve("H", &ctx), Some(768));
    }
    
    #[test]
    fn test_precision_resolver() {
        let resolver = PrecisionResolver::new_industrial();
        let ctx = ResolutionContext {
            training: TrainingConfig { batch_size: Some(32), seq_len: Some(512) },
            global: GlobalParams { hidden_size: Some(768), vocab_size: Some(50000), ..Default::default() },
            data: DataConfig::default(),
        };
        
        let mut dims = vec![
            Dim::Symbolic("B".into()),
            Dim::Symbolic("S".into()),
            Dim::Symbolic("H".into()),
            Dim::Symbolic("unknown".into()),
        ];
        
        let report = resolver.resolve_dims(&mut dims, &ctx);
        
        assert_eq!(dims[0], Dim::Known(32));
        assert_eq!(dims[1], Dim::Known(512));
        assert_eq!(dims[2], Dim::Known(768));
        assert!(dims[3].is_symbolic());
        assert_eq!(report.resolved.len(), 3);
        assert_eq!(report.unresolved.len(), 1);
    }
}
