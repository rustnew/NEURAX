//! Operator IR structures

use std::collections::HashMap;
use crate::tensor::Shape;

/// Operator IR - dialecte des opérations ML standard
#[derive(Debug, Clone)]
pub struct OperatorIR {
    /// Toutes les opérations élémentaires du modèle
    pub operations: Vec<AtomOp>,
    /// Mapping layer_id → liste d'AtomOps indices
    pub layer_ops: HashMap<String, Vec<usize>>,
    pub metrics: OperatorMetrics,
    pub metrics_done: bool,
}

impl Default for OperatorIR {
    fn default() -> Self {
        Self {
            operations: Vec::new(),
            layer_ops: HashMap::new(),
            metrics: OperatorMetrics::default(),
            metrics_done: false,
        }
    }
}

/// Atomic operation
#[derive(Debug, Clone)]
pub struct AtomOp {
    pub id: usize,
    pub op_type: OpType,
    pub layer_id: String,
    pub input_shapes: Vec<Shape>,
    pub output_shape: Shape,
    /// FLOPs calculés analytiquement
    pub flops: f64,
    /// Paramètres impliqués dans cette opération
    pub param_count: u64,
    /// Mémoire nécessaire en activation
    pub activation_memory: u64,
    /// Si c'est une opération custom
    pub is_custom: bool,
}

/// Operation type - 24+ AtomOps per impl_2.md
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpType {
    // Core compute
    MatMul,
    BatchedMatMul,
    Conv2D,
    DepthwiseConv2D,
    Linear,
    
    // Attention variants
    Attention,              // Standard Multi-Head Attention
    FlashAttention,          // Memory-optimized attention
    GroupedQueryAttention,   // GQA (LLaMA-style)
    MultiQueryAttention,     // MQA
    AttentionScores,
    AttentionOutput,
    Softmax,
    
    // Normalization
    LayerNorm,
    BatchNorm,
    RMSNorm,
    GroupNorm,
    
    // Embeddings
    Embedding,
    TokenEmbedding,
    PositionalEmbedding,
    RotaryEmbedding,
    
    // MoE operations
    MoE,
    MoERouter,               // Expert routing
    MoEExpertGroup,          // Expert computation
    
    // State Space Models (SSM/Mamba)
    SsmStateUpdate,          // Mamba/SSM state
    MambaConv1d,             // Mamba convolution
    S4Block,
    H3Block,
    
    // RNN/LSTM/GRU
    LstmCell,
    GruCell,
    RnnCell,
    
    // Fine-tuning
    LoRALinear,              // LoRA adapter
    
    // Activations
    Add,
    Mul,
    Div,
    ReLU,
    GELU,
    SiLU,
    Tanh,
    Sigmoid,
    
    // Pooling
    Pooling(PoolingType),
    
    // Tensor ops
    Reshape,
    Transpose,
    Concat,
    Split,
    
    // Custom
    Custom,
}

impl OpType {
    pub fn as_str(&self) -> &'static str {
        match self {
            // Core compute
            Self::MatMul => "matmul",
            Self::BatchedMatMul => "batched_matmul",
            Self::Conv2D => "conv2d",
            Self::DepthwiseConv2D => "depthwise_conv2d",
            Self::Linear => "linear",
            
            // Attention variants
            Self::Attention => "attention",
            Self::FlashAttention => "flash_attention",
            Self::GroupedQueryAttention => "grouped_query_attention",
            Self::MultiQueryAttention => "multi_query_attention",
            Self::AttentionScores => "attention_scores",
            Self::AttentionOutput => "attention_output",
            Self::Softmax => "softmax",
            
            // Normalization
            Self::LayerNorm => "layer_norm",
            Self::BatchNorm => "batch_norm",
            Self::RMSNorm => "rms_norm",
            Self::GroupNorm => "group_norm",
            
            // Embeddings
            Self::Embedding => "embedding",
            Self::TokenEmbedding => "token_embedding",
            Self::PositionalEmbedding => "positional_embedding",
            Self::RotaryEmbedding => "rotary_embedding",
            
            // MoE operations
            Self::MoE => "moe",
            Self::MoERouter => "moe_router",
            Self::MoEExpertGroup => "moe_expert_group",
            
            // State Space Models
            Self::SsmStateUpdate => "ssm_state_update",
            Self::MambaConv1d => "mamba_conv1d",
            Self::S4Block => "s4_block",
            Self::H3Block => "h3_block",
            
            // RNN/LSTM/GRU
            Self::LstmCell => "lstm_cell",
            Self::GruCell => "gru_cell",
            Self::RnnCell => "rnn_cell",
            
            // Fine-tuning
            Self::LoRALinear => "lora_linear",
            
            // Activations
            Self::Add => "add",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::ReLU => "relu",
            Self::GELU => "gelu",
            Self::SiLU => "silu",
            Self::Tanh => "tanh",
            Self::Sigmoid => "sigmoid",
            
            // Pooling
            Self::Pooling(t) => match t {
                PoolingType::Max => "max_pool",
                PoolingType::Avg => "avg_pool",
            },
            
            // Tensor ops
            Self::Reshape => "reshape",
            Self::Transpose => "transpose",
            Self::Concat => "concat",
            Self::Split => "split",
            
            // Custom
            Self::Custom => "custom",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PoolingType {
    Max,
    Avg,
}

/// Operator metrics (Métriques 6, 8)
#[derive(Debug, Clone, Default)]
pub struct OperatorMetrics {
    /// FLOPs par layer (Métrique 8)
    pub flops_per_layer: HashMap<String, f64>,
    /// Nombre total d'opérations
    pub total_op_count: usize,
    /// Distribution par type d'opération
    pub op_type_distribution: HashMap<String, usize>,
    /// Nombre d'opérations custom
    pub custom_op_count: usize,
    /// FLOPs totaux approximatifs (Métrique 6)
    pub total_flops_approx: f64,
}

impl OperatorMetrics {
    pub fn is_valid(&self) -> bool {
        self.total_op_count > 0
    }
}
