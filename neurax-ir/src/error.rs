//! Error types for NEURAX IR

use thiserror::Error;

/// Main NEURAX error type
#[derive(Debug, Error)]
pub enum NeuraxError {
    // Parser errors
    #[error("Parser error: {0}")]
    Parser(#[from] neurax_parser::ParserError),
    
    // IR-specific errors
    #[error("Architecture IR error: {0}")]
    Architecture(#[from] ArchitectureError),
    
    #[error("Graph IR error: {0}")]
    Graph(#[from] GraphError),
    
    #[error("Tensor IR error: {0}")]
    Tensor(#[from] TensorError),
    
    #[error("Operator IR error: {0}")]
    Operator(#[from] OperatorError),
    
    #[error("Compute IR error: {0}")]
    Compute(#[from] ComputeError),
    
    #[error("Memory IR error: {0}")]
    Memory(#[from] MemoryError),
    
    #[error("Parallelism IR error: {0}")]
    Parallelism(#[from] ParallelismError),
    
    #[error("Hardware IR error: {0}")]
    Hardware(#[from] HardwareError),
    
    #[error("Cost IR error: {0}")]
    Cost(#[from] CostError),
    
    // General errors
    #[error("Metrics validation failed at pass '{pass}': {reason}")]
    MetricsValidation { pass: String, reason: String },
    
    #[error("Custom expression evaluation error in layer '{layer}': {msg}")]
    CustomExprEval { layer: String, msg: String },
    
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Architecture IR errors
#[derive(Debug, Error)]
pub enum ArchitectureError {
    #[error("No layers defined in model")]
    EmptyLayers,
    
    #[error("Unknown model type: '{0}'")]
    UnknownModelType(String),
    
    #[error("Parameter count is zero for layer '{layer_id}' of type '{layer_type}'")]
    ZeroParameters { layer_id: String, layer_type: String },
    
    #[error("Layer '{id}' has inconsistent shapes: input={input:?}, output={output:?}")]
    ShapeInconsistency { id: String, input: Vec<usize>, output: Vec<usize> },
    
    #[error("Failed to compute parameters: {0}")]
    ParameterComputation(String),
}

/// Graph IR errors
#[derive(Debug, Error)]
pub enum GraphError {
    #[error("Cycle detected in computation graph")]
    CycleDetected,
    
    #[error("Empty graph - no nodes")]
    EmptyGraph,
    
    #[error("Invalid node index: {0}")]
    InvalidNodeIndex(usize),
    
    #[error("Topological sort failed: {0}")]
    TopologicalSortFailed(String),
}

/// Tensor IR errors
#[derive(Debug, Error)]
pub enum TensorError {
    #[error("Shape propagation failed for layer '{layer}': {reason}")]
    ShapePropagationFailed { layer: String, reason: String },
    
    #[error("Unknown dimension: '{0}'")]
    UnknownDimension(String),
    
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },
    
    #[error("Empty tensor")]
    EmptyTensor,
    
    #[error("Shape gate blocked: only {resolved:.1}% dimensions resolved (threshold: {threshold:.0}%). {unresolved} unresolved dimensions.")]
    ShapeGateBlocked { resolved: f32, threshold: f32, unresolved: usize },
}

/// Operator IR errors
#[derive(Debug, Error)]
pub enum OperatorError {
    #[error("Unknown operator type: '{0}'")]
    UnknownOperator(String),
    
    #[error("FLOPs calculation failed for operator '{op}': {reason}")]
    FlopsCalculationFailed { op: String, reason: String },
    
    #[error("Custom equation evaluation failed: {0}")]
    CustomEquationFailed(String),
}

/// Compute IR errors
#[derive(Debug, Error)]
pub enum ComputeError {
    #[error("Total FLOPs is zero")]
    ZeroFlops,
    
    #[error("Invalid arithmetic intensity: {0}")]
    InvalidArithmeticIntensity(String),
    
    #[error("Compute time calculation failed: {0}")]
    ComputeTimeFailed(String),
}

/// Memory IR errors
#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("Peak VRAM ({peak_gb:.1} GB) exceeds GPU memory ({gpu_gb:.1} GB). Consider gradient checkpointing or model parallelism.")]
    VramOverflow { peak_gb: f64, gpu_gb: f64 },
    
    #[error("Activation memory calculation failed: liveness analysis returned empty intervals")]
    EmptyLivenessIntervals,
    
    #[error("Memory calculation failed: {0}")]
    MemoryCalculationFailed(String),
}

/// Parallelism IR errors
#[derive(Debug, Error)]
pub enum ParallelismError {
    #[error("Invalid parallelism configuration: {0}")]
    InvalidConfiguration(String),
    
    #[error("Communication cost calculation failed: {0}")]
    CommunicationCostFailed(String),
}

/// Hardware IR errors
#[derive(Debug, Error)]
pub enum HardwareError {
    #[error("GPU not found in database: '{0}'")]
    GpuNotFound(String),
    
    #[error("Roofline model calculation failed: {0}")]
    RooflineFailed(String),
    
    #[error("Invalid latency calculation: {0}")]
    InvalidLatency(String),
}

/// Cost IR errors
#[derive(Debug, Error)]
pub enum CostError {
    #[error("Pricing calculation failed: {0}")]
    PricingFailed(String),
    
    #[error("Invalid cost configuration: {0}")]
    InvalidConfiguration(String),
}
