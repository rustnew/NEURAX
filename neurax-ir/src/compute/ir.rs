//! Compute IR structures

use std::collections::HashMap;

/// Compute IR - dialecte du calcul analytique
#[derive(Debug, Clone)]
pub struct ComputeIR {
    /// FLOPs par opération (hérités de OperatorIR)
    pub op_flops: Vec<OpFlops>,
    pub metrics: ComputeMetrics,
    pub metrics_done: bool,
}

impl Default for ComputeIR {
    fn default() -> Self {
        Self {
            op_flops: Vec::new(),
            metrics: ComputeMetrics::default(),
            metrics_done: false,
        }
    }
}

/// FLOPs record for an operation
#[derive(Debug, Clone)]
pub struct OpFlops {
    pub op_id: usize,
    pub layer_id: String,
    pub forward_flops: f64,
    pub backward_flops: f64,
}

/// Compute metrics (Métriques 6-11, 20-24)
#[derive(Debug, Clone, Default)]
pub struct ComputeMetrics {
    /// Métrique 6: FLOPs totaux précis
    pub total_flops: f64,
    /// Métrique 7: MACs = FLOPs / 2
    pub macs: f64,
    /// Métrique 8: FLOPs par layer
    pub flops_per_layer: HashMap<String, f64>,
    /// Métrique 9: FLOPs par token
    /// INVARIANT: flops_per_token × seq_len ≈ forward_flops (±0.1%)
    /// Formule: forward_flops / seq_len (pas divisé par batch)
    pub flops_per_token: f64,
    /// Métrique 10: Intensité arithmétique (FLOPs / bytes_moved)
    pub arithmetic_intensity: f64,
    /// Métrique 10: Complexité asymptotique
    pub complexity_class: ComplexityClass,
    /// Métrique 21: Forward pass FLOPs
    pub forward_flops: f64,
    /// Métrique 22: Backward pass FLOPs
    pub backward_flops: f64,
    /// Coût optimizer step
    pub optimizer_flops: f64,
    /// Métrique 23: Total FLOPs par step
    pub total_step_flops: f64,
    /// FLOPs par batch
    pub flops_per_batch: f64,
    /// Bytes accédés en mémoire
    pub bytes_accessed: u64,
    /// FLOPs pour inférence incrémentale (1 token avec KV cache plein)
    /// ≈ 60% de flops_per_token car attention devient O(S) au lieu de O(S²)
    pub flops_incremental_decode: f64,
}

impl ComputeMetrics {
    pub fn is_valid(&self) -> bool {
        self.forward_flops > 0.0 && self.backward_flops > 0.0
    }
}

/// Complexity class enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplexityClass {
    Linear,      // O(n)
    NLogN,       // O(n log n)
    Quadratic,   // O(n²) — attention standard
    Cubic,       // O(n³)
    Exponential, // O(2^n)
    Custom,
}

impl ComplexityClass {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Linear => "O(n)",
            Self::NLogN => "O(n log n)",
            Self::Quadratic => "O(n²)",
            Self::Cubic => "O(n³)",
            Self::Exponential => "O(2^n)",
            Self::Custom => "Custom",
        }
    }
}

impl Default for ComplexityClass {
    fn default() -> Self {
        Self::Linear
    }
}
