//! NEURAX IR - 10 dialectes IR pour l'analyse analytique

pub mod traits;
pub mod error;
pub mod precision;
pub mod architecture;
pub mod graph;
pub mod tensor;
pub mod operator;
pub mod compute;
pub mod memory;
pub mod parallelism;
pub mod hardware;
pub mod cost;
pub mod report;
pub mod ir_injector;
pub mod dynamic;
pub mod inference;

pub use traits::IrPass;
pub use error::*;
pub use precision::*;
pub use architecture::*;
pub use graph::*;
pub use tensor::*;
pub use operator::*;
pub use compute::*;
pub use memory::*;
pub use parallelism::*;
pub use hardware::*;
pub use cost::*;
pub use report::*;
pub use ir_injector::{IrInjector, ArchitectureIRInput, MemoryPassConfig, HardwarePassConfig, CostPassConfig};

use std::sync::Arc;
use parking_lot::Mutex;
use ahash::AHashMap as HashMap;
use neurax_parser::ModelConfig;
use neurax_hardware_db::HardwareDatabase;
use serde::Serialize;

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
    
    // Warnings (W001-W006) - Precision reduced
    W001, // Custom layer without formula
    W002, // Symbolic dimensions remaining
    W003, // ZeRO not recommended
    W004, // Flash Attention not enabled
    W005, // Memory close to GPU limit
    W006, // Inefficient parallelism
    
    // Info (I001-I003) - Observations
    I001, // GQA detected
    I002, // MoE detected
    I003, // Flash Attention detected
    
    // Hints (H001-H005) - Recommendations
    H001, // Enable gradient checkpointing
    H002, // Enable Flash Attention
    H003, // Consider INT8 quantization
    H004, // Increase micro-batches PP
    H005, // ZeRO-3 recommended
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
            Self::I001 => "I001",
            Self::I002 => "I002",
            Self::I003 => "I003",
            Self::H001 => "H001",
            Self::H002 => "H002",
            Self::H003 => "H003",
            Self::H004 => "H004",
            Self::H005 => "H005",
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
            Self::I001 => "Grouped Query Attention (GQA) detected",
            Self::I002 => "Mixture of Experts (MoE) detected",
            Self::I003 => "Flash Attention detected",
            Self::H001 => "Consider enabling gradient checkpointing",
            Self::H002 => "Consider enabling Flash Attention",
            Self::H003 => "Consider INT8 quantization for inference",
            Self::H004 => "Consider increasing micro-batches for pipeline parallelism",
            Self::H005 => "ZeRO-3 recommended for this model size",
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
