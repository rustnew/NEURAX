//! # NEURAX IR
//!
//! **The NEURAX 10-pass intermediate representation — the analytical compiler
//! core.**
//!
//! Part of the [NEURAX](https://github.com/rustnew/NEURAX) analytical compiler
//! for neural network architectures. Defines the IR and the pass pipeline that
//! transforms a parsed model into a complete analytical report:
//! architecture → graph → tensor → operator → compute → memory → parallelism →
//! hardware → cost → report.
//!
//! Every pass implements the [`IrPass`](crate::traits::IrPass) trait
//! (`build` / `compute_metrics` / `validate`), which makes the pipeline
//! composable and testable.
//!
//! ## Quick example
//!
//! ```rust
//! use neurax_ir::{NeuraxContext, ComputeConfig};
//! use neurax_ir::architecture::ArchitecturePass;
//! use neurax_ir::traits::IrPass;
//! use neurax_parser::parse_model_config;
//!
//! let json = r#"{
//!   "schema_version": "1.0",
//!   "model": {
//!     "name": "tiny-gpt",
//!     "type": "transformer",
//!     "layers": [
//!       { "id": "attn_0", "layer_type": "attention",
//!         "input_shape": [128, 768], "output_shape": [128, 768],
//!         "params": { "num_heads": 12 } }
//!     ],
//!     "global_params": { "hidden_size": 768, "num_layers": 1 }
//!   },
//!   "training": { "batch_size": 32, "optimizer": "adamw", "precision": "bf16" },
//!   "hardware": {
//!     "gpus": [
//!       { "name": "A100-80GB", "count": 1, "memory_gb": 80,
//!         "tflops_fp16": 312, "tflops_fp32": 19.5,
//!         "memory_bandwidth_gb_s": 2039, "tensor_cores": true }
//!     ],
//!     "interconnect": "None", "interconnect_bandwidth_gb_s": 0
//!   }
//! }"#;
//!
//! let config = parse_model_config(json).expect("valid NEURAX JSON");
//! let ctx = NeuraxContext::new(config.clone());
//!
//! // Each pass implements IrPass: build -> compute_metrics -> validate
//! let pass = ArchitecturePass;
//! let mut arch = pass.build(&config, &ctx).expect("architecture pass");
//! let metrics = pass.compute_metrics(&mut arch, &ctx).expect("metrics");
//! pass.validate(&arch, &metrics).expect("validation");
//! assert!(!arch.layers.is_empty());
//! ```
//!
//! ## Run the example
//!
//! ```bash
//! cargo run --example pipeline
//! ```

pub mod architecture;
pub mod compute;
pub mod cost;
pub mod dynamic;
pub mod error;
pub mod graph;
pub mod hardware;
pub mod inference;
pub mod memory;
pub mod operator;
pub mod parallelism;
pub mod precision;
pub mod report;
pub mod tensor;
pub mod traits;

pub use architecture::*;
pub use compute::*;
pub use cost::*;
pub use error::*;
pub use graph::*;
pub use hardware::*;
pub use memory::*;
pub use operator::*;
pub use parallelism::*;
#[allow(ambiguous_glob_reexports)]
pub use precision::*;
pub use report::*;
#[allow(ambiguous_glob_reexports)]
pub use tensor::*;
pub use traits::IrPass;

use ahash::AHashMap as HashMap;
use neurax_hardware_db::HardwareDatabase;
use neurax_parser::ModelConfig;
use parking_lot::Mutex;
use serde::Serialize;
use std::sync::Arc;

/// Shared context for IR passes
pub struct NeuraxContext {
    /// Original parsed configuration
    pub config: Arc<ModelConfig>,
    /// Hardware database
    pub gpu_db: Arc<HardwareDatabase>,
    /// Compute configuration
    pub compute_config: ComputeConfig,
    /// Diagnostics collected during analysis
    pub diagnostics: Arc<Mutex<Vec<Diagnostic>>>,
    /// Metrics store for inter-pass communication
    metrics_store: Arc<Mutex<HashMap<String, f64>>>,
}

impl NeuraxContext {
    /// VRAM of the primary GPU, in bytes.
    ///
    /// Resolution order: the value stated in the config, then the hardware
    /// database entry for the named GPU, then a conservative default. Naming a
    /// GPU without restating its specs must yield that GPU's real numbers, not
    /// a placeholder.
    pub fn primary_gpu_vram_bytes(&self) -> u64 {
        const GIB: u64 = 1024 * 1024 * 1024;
        let gpu = self.config.hardware.gpus.first();
        gpu.and_then(|g| g.memory_gb)
            .or_else(|| {
                gpu.and_then(|g| self.gpu_db.get_gpu(&g.name))
                    .map(|spec| spec.memory_gb)
            })
            .unwrap_or(40)
            .saturating_mul(GIB)
    }

    /// Memory bandwidth of the primary GPU, in GB/s. Same resolution order as
    /// [`Self::primary_gpu_vram_bytes`].
    pub fn primary_gpu_bandwidth_gbs(&self) -> f64 {
        let gpu = self.config.hardware.gpus.first();
        gpu.and_then(|g| g.memory_bandwidth_gbs)
            .or_else(|| {
                gpu.and_then(|g| self.gpu_db.get_gpu(&g.name))
                    .map(|spec| spec.memory_bandwidth_gbs)
            })
            .unwrap_or(1000.0)
    }

    pub fn new(config: ModelConfig) -> Self {
        Self {
            config: Arc::new(config),
            gpu_db: Arc::new(HardwareDatabase::new()),
            compute_config: ComputeConfig::default(),
            diagnostics: Arc::new(Mutex::new(Vec::new())),
            metrics_store: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn add_diagnostic(&self, diagnostic: Diagnostic) {
        self.diagnostics.lock().push(diagnostic);
    }

    /// Store a metric value for inter-pass communication
    pub fn set_metric(&self, key: &str, value: f64) {
        self.metrics_store.lock().insert(key.to_string(), value);
    }

    /// Retrieve a stored metric value
    pub fn get_metric(&self, key: &str) -> Option<f64> {
        self.metrics_store.lock().get(key).copied()
    }
}

/// Compute configuration
#[derive(Debug, Clone)]
pub struct ComputeConfig {
    pub num_threads: usize,
    pub ir_timeout_s: u64,
    pub enable_parallelism: bool,
}

impl Default for ComputeConfig {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get(),
            ir_timeout_s: 30,
            enable_parallelism: true,
        }
    }
}

/// Diagnostic information
#[derive(Debug, Clone, Serialize)]
pub struct Diagnostic {
    pub severity: Severity,
    pub category: DiagnosticCategory,
    pub code: DiagnosticCode,
    pub message: String,
    pub layer_id: Option<String>,
    pub suggestion: Option<String>,
    /// Impact on precision (0.0 = no impact, 1.0 = critical)
    pub precision_impact: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Severity {
    Info,
    Warning,
    Critical,
    Hint,
}

/// Standardized diagnostic codes per impl_2.md
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DiagnosticCode {
    // Errors (E001-E005) - Blocking issues
    E001, // OOM Risk
    E002, // Shape gate blocked
    E003, // Custom formula failed
    E004, // Unsupported layer
    E005, // Cycle in graph

    // Warnings (W001-W007) - Precision reduced
    W001, // Custom layer without formula
    W002, // Symbolic dimensions remaining
    W003, // ZeRO not recommended
    W004, // Flash Attention not enabled
    W005, // Memory close to GPU limit
    W006, // Inefficient parallelism
    W007, // Cross-layer shape mismatch
    W008, // Orphaned layer — connected to nothing
    W009, // Embedding table dominates the parameter count

    // Info (I001-I003) - Observations
    I001, // GQA detected
    I002, // MoE detected
    I003, // Flash Attention detected

    // Hints (H001-H008) - Recommendations
    H001, // Enable gradient checkpointing
    H002, // Enable Flash Attention
    H003, // Consider INT8 quantization
    H004, // Increase micro-batches PP
    H005, // ZeRO-3 recommended
    H006, // No LR warmup configured
    H007, // Learning rate exceeds the Lipschitz-based stability bound
    H008, // Tokens-per-parameter ratio far from compute-optimal (Chinchilla)
}

impl DiagnosticCode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::E001 => "E001",
            Self::E002 => "E002",
            Self::E003 => "E003",
            Self::E004 => "E004",
            Self::E005 => "E005",
            Self::W001 => "W001",
            Self::W002 => "W002",
            Self::W003 => "W003",
            Self::W004 => "W004",
            Self::W005 => "W005",
            Self::W006 => "W006",
            Self::W007 => "W007",
            Self::W008 => "W008",
            Self::W009 => "W009",
            Self::I001 => "I001",
            Self::I002 => "I002",
            Self::I003 => "I003",
            Self::H001 => "H001",
            Self::H002 => "H002",
            Self::H003 => "H003",
            Self::H004 => "H004",
            Self::H005 => "H005",
            Self::H006 => "H006",
            Self::H007 => "H007",
            Self::H008 => "H008",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::E001 => "OOM Risk detected",
            Self::E002 => "Shape gate blocked - insufficient dimension resolution",
            Self::E003 => "Custom formula evaluation failed",
            Self::E004 => "Unsupported layer type",
            Self::E005 => "Cycle detected in computation graph",
            Self::W001 => "Custom layer without formula - using estimation",
            Self::W002 => "Symbolic dimensions remaining in shapes",
            Self::W003 => "ZeRO not recommended for this configuration",
            Self::W004 => "Flash Attention not enabled",
            Self::W005 => "Memory usage close to GPU limit",
            Self::W006 => "Inefficient parallelism strategy",
            Self::W007 => {
                "A layer's declared input shape does not match the previous layer's output shape"
            }
            Self::W008 => "Layer is not connected to any other layer, but is still costed",
            Self::W009 => "Most parameters sit in the embedding table rather than in the layers",
            Self::I001 => "Grouped Query Attention (GQA) detected",
            Self::I002 => "Mixture of Experts (MoE) detected",
            Self::I003 => "Flash Attention detected",
            Self::H001 => "Consider enabling gradient checkpointing",
            Self::H002 => "Consider enabling Flash Attention",
            Self::H003 => "Consider INT8 quantization for inference",
            Self::H004 => "Consider increasing micro-batches for pipeline parallelism",
            Self::H005 => "ZeRO-3 recommended for this model size",
            Self::H006 => "No learning-rate warmup configured",
            Self::H007 => "Learning rate exceeds the estimated stability bound",
            Self::H008 => "Tokens-per-parameter ratio far from compute-optimal",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DiagnosticCategory {
    MemoryOverflow,
    BottleneckDetected,
    ParallelismSuboptimal,
    ArchitectureInefficiency,
    CostAlert,
    CustomLayerFallback,
    ShapeInference,
    Configuration,
}
