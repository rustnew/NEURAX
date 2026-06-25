//! Report IR structures

use chrono::{DateTime, Utc};
use serde::Serialize;

/// Per-pass timing entry for compilation timeline
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct PhaseTimingEntry {
    pub name: String,
    pub duration_ms: u64,
    pub status: String,
}

/// Per-layer gradient memory breakdown
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct GradientMemoryEntry {
    pub name: String,
    pub forward: u64,
    pub backward: u64,
}

/// KV cache scaling point (seq_len → cache_bytes)
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct KvCacheEntry {
    pub seq: u32,
    pub value: u64,
}

/// Report IR - rapport final consolidé
#[derive(Debug, Clone, Serialize)]
pub struct ReportIR {
    pub metadata: ReportMetadata,
    pub metrics: AllMetrics,
    pub diagnostics: Vec<Diagnostic>,
    pub recommendations: Vec<Recommendation>,
    pub warnings: Vec<String>,
    /// Confidence score [0.0, 1.0] based on shape resolution and custom layers
    pub confidence_score: f64,
    /// Compilation phase timeline (timing per IR pass)
    pub phase_timeline: Vec<PhaseTimingEntry>,
}

impl Default for ReportIR {
    fn default() -> Self {
        Self {
            metadata: ReportMetadata::default(),
            metrics: AllMetrics::default(),
            diagnostics: Vec::new(),
            recommendations: Vec::new(),
            warnings: Vec::new(),
            confidence_score: 1.0,
            phase_timeline: Vec::new(),
        }
    }
}

/// Report metadata
#[derive(Debug, Clone, Serialize)]
pub struct ReportMetadata {
    pub generated_at: DateTime<Utc>,
    pub neurax_version: String,
    pub model_name: String,
    pub model_type: String,
    pub schema_version: String,
    pub analysis_time_ms: u64,
}

impl Default for ReportMetadata {
    fn default() -> Self {
        Self {
            generated_at: Utc::now(),
            neurax_version: env!("CARGO_PKG_VERSION").to_string(),
            model_name: "Unknown".to_string(),
            model_type: "Unknown".to_string(),
            schema_version: "1.0".to_string(),
            analysis_time_ms: 0,
        }
    }
}

/// All metrics consolidated (35+ metrics per impl_2.md)
#[derive(Debug, Clone, Default, Serialize)]
pub struct AllMetrics {
    // === Structure Metrics (1-5) ===
    pub total_parameters: u64,
    pub num_layers: usize,
    pub model_type: String,
    pub hidden_size: usize,
    pub vocab_size: u64,

    // === Graph Metrics (6-8) ===
    pub graph_depth: usize,
    pub total_operations: usize,
    pub critical_path_length: usize,

    // === Tensor Metrics (9-12) ===
    pub tensor_resolution_ratio: f32,
    pub unresolved_dim_count: usize,
    pub total_tensor_count: usize,
    pub largest_tensor_bytes: u64,

    // === Compute Metrics (13-18) ===
    pub total_flops: f64,
    pub forward_flops: f64,
    pub backward_flops: f64,
    pub flops_per_token: f64,
    /// FLOPs pour inférence incrémentale (1 token avec KV cache plein)
    /// ≈ 60% de flops_per_token car attention devient O(S) au lieu de O(S²)
    pub flops_incremental_decode: f64,
    pub arithmetic_intensity: f64,
    pub ops_distribution: std::collections::HashMap<String, usize>,

    // === Memory Metrics (19-25) ===
    pub peak_vram_bytes: u64,
    pub parameter_memory_bytes: u64,
    pub activation_memory_bytes: u64,
    pub gradient_memory_bytes: u64,
    pub optimizer_state_bytes: u64,
    pub max_batch_size_fit: u32,
    pub memory_fragmentation: f64,

    // === Parallelism Metrics (26-30) ===
    pub data_parallel_efficiency: f64,
    pub communication_overhead: f64,
    pub optimal_gpu_count: u32,
    pub pipeline_stages: u32,
    pub tensor_parallel_degree: u32,

    // === Hardware Metrics (31-35) ===
    pub latency_ms: f64,
    pub throughput_tokens_per_s: f64,
    pub gpu_utilization: f64,
    pub bottleneck: String,
    pub roofline_position: f64, // 0.0 = memory-bound, 1.0 = compute-bound

    // === Hardware Config (from JSON) ===
    pub gpu_name: String,
    pub gpu_count: usize,
    pub gpu_memory_gb: f64,
    pub gpu_tflops_fp16: f64,
    pub gpu_memory_bandwidth_gbs: f64,
    pub interconnect: String,
    pub interconnect_bandwidth_gbs: f64,

    // === Cost Metrics (36-40) ===
    pub training_cost_usd: f64,
    pub training_time_hours: f64,
    pub energy_kwh: f64,
    pub co2_kg: f64,
    pub cost_per_million_tokens_usd: f64,

    // === Confidence & Quality (41-43) ===
    pub confidence_score: f64,
    pub custom_layer_count: usize,
    pub diagnostic_count: usize,

    // === Per-Layer Breakdown Maps ===
    pub params_per_layer: std::collections::HashMap<String, u64>,
    pub flops_per_layer: std::collections::HashMap<String, f64>,
    pub latency_per_layer: std::collections::HashMap<String, f64>,

    // === Rich per-layer metrics (optional) ===
    pub gradient_memory_per_layer: Vec<GradientMemoryEntry>,
    pub kv_cache_scaling: Vec<KvCacheEntry>,
}

/// Diagnostic information (re-exported from lib.rs for convenience)
pub use crate::Diagnostic;
pub use crate::DiagnosticCategory;
pub use crate::DiagnosticCode;
pub use crate::Severity;

/// Recommendation
#[derive(Debug, Clone, Serialize)]
pub struct Recommendation {
    pub category: RecommendationCategory,
    pub title: String,
    pub description: String,
    pub impact: String,
    pub priority: Priority,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RecommendationCategory {
    MemoryOptimization,
    ComputeOptimization,
    Parallelism,
    Hardware,
    Cost,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Priority {
    High,
    Medium,
    Low,
}
