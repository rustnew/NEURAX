//! Operator dialect for NEURAX
//!
//! Models ML operations (MatMul, Conv, Attention, etc.)

use melior::ir::{Attribute, Identifier, Location, Operation, operation::OperationBuilder};
use melior::Context;
use super::utils::{int_attr, float_attr, int_array_attr, string_attr};

/// Operator dialect name
pub const DIALECT_NAME: &str = "op";

/// Operator dialect
pub struct OperatorDialect;

impl OperatorDialect {
    /// Get the dialect name
    pub fn name() -> &'static str {
        DIALECT_NAME
    }
    
    /// Create a MatMul operation
    pub fn matmul<'c>(
        context: &'c Context,
        param_count: i64,
        flops: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.matmul", location)
            .add_attributes(&[
                (Identifier::new(context, "param_count"), int_attr(context, param_count)),
                (Identifier::new(context, "flops"), float_attr(context, flops)),
            ])
            .build()
    }
    
    /// Create a Conv2D operation
    pub fn conv2d<'c>(
        context: &'c Context,
        in_channels: i64,
        out_channels: i64,
        kernel_size: &[i64],
        param_count: i64,
        flops: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.conv2d", location)
            .add_attributes(&[
                (Identifier::new(context, "in_channels"), int_attr(context, in_channels)),
                (Identifier::new(context, "out_channels"), int_attr(context, out_channels)),
                (Identifier::new(context, "kernel_size"), int_array_attr(context, kernel_size)),
                (Identifier::new(context, "param_count"), int_attr(context, param_count)),
                (Identifier::new(context, "flops"), float_attr(context, flops)),
            ])
            .build()
    }
    
    /// Create a Pooling operation (max or avg)
    pub fn pooling<'c>(
        context: &'c Context,
        pool_type: &str, // "max" or "avg"
        kernel_size: &[i64],
        stride: i64,
        flops: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.pooling", location)
            .add_attributes(&[
                (Identifier::new(context, "pool_type"), string_attr(context, pool_type)),
                (Identifier::new(context, "kernel_size"), int_array_attr(context, kernel_size)),
                (Identifier::new(context, "stride"), int_attr(context, stride)),
                (Identifier::new(context, "flops"), float_attr(context, flops)),
            ])
            .build()
    }
    
    /// Create an Attention operation
    pub fn attention<'c>(
        context: &'c Context,
        hidden_size: i64,
        num_heads: i64,
        param_count: i64,
        flops: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.attention", location)
            .add_attributes(&[
                (Identifier::new(context, "hidden_size"), int_attr(context, hidden_size)),
                (Identifier::new(context, "num_heads"), int_attr(context, num_heads)),
                (Identifier::new(context, "param_count"), int_attr(context, param_count)),
                (Identifier::new(context, "flops"), float_attr(context, flops)),
            ])
            .build()
    }
    
    /// Create a MoE operation
    pub fn moe<'c>(
        context: &'c Context,
        hidden_size: i64,
        num_experts: i64,
        top_k: i64,
        param_count: i64,
        flops: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.moe", location)
            .add_attributes(&[
                (Identifier::new(context, "hidden_size"), int_attr(context, hidden_size)),
                (Identifier::new(context, "num_experts"), int_attr(context, num_experts)),
                (Identifier::new(context, "top_k"), int_attr(context, top_k)),
                (Identifier::new(context, "param_count"), int_attr(context, param_count)),
                (Identifier::new(context, "flops"), float_attr(context, flops)),
            ])
            .build()
    }
    
    /// Create an LSTM operation
    pub fn lstm<'c>(
        context: &'c Context,
        hidden_size: i64,
        num_layers: i64,
        bidirectional: bool,
        param_count: i64,
        flops: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.lstm", location)
            .add_attributes(&[
                (Identifier::new(context, "hidden_size"), int_attr(context, hidden_size)),
                (Identifier::new(context, "num_layers"), int_attr(context, num_layers)),
                (Identifier::new(context, "bidirectional"), Attribute::parse(context, &format!("{}", bidirectional)).unwrap()),
                (Identifier::new(context, "param_count"), int_attr(context, param_count)),
                (Identifier::new(context, "flops"), float_attr(context, flops)),
            ])
            .build()
    }
    
    /// Create a GRU operation
    pub fn gru<'c>(
        context: &'c Context,
        hidden_size: i64,
        num_layers: i64,
        bidirectional: bool,
        param_count: i64,
        flops: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.gru", location)
            .add_attributes(&[
                (Identifier::new(context, "hidden_size"), int_attr(context, hidden_size)),
                (Identifier::new(context, "num_layers"), int_attr(context, num_layers)),
                (Identifier::new(context, "bidirectional"), Attribute::parse(context, &format!("{}", bidirectional)).unwrap()),
                (Identifier::new(context, "param_count"), int_attr(context, param_count)),
                (Identifier::new(context, "flops"), float_attr(context, flops)),
            ])
            .build()
    }
    
    /// Create an SSM (State Space Model) operation for Mamba
    pub fn ssm<'c>(
        context: &'c Context,
        state_dim: i64,
        hidden_size: i64,
        expansion_factor: i64,
        param_count: i64,
        flops: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.ssm", location)
            .add_attributes(&[
                (Identifier::new(context, "state_dim"), int_attr(context, state_dim)),
                (Identifier::new(context, "hidden_size"), int_attr(context, hidden_size)),
                (Identifier::new(context, "expansion_factor"), int_attr(context, expansion_factor)),
                (Identifier::new(context, "param_count"), int_attr(context, param_count)),
                (Identifier::new(context, "flops"), float_attr(context, flops)),
            ])
            .build()
    }
    
    /// Create a Custom operation with custom equations
    pub fn custom<'c>(
        context: &'c Context,
        op_name: &str,
        param_count: i64,
        flops: f64,
        custom_flops_eq: Option<&str>,
        custom_memory_eq: Option<&str>,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        let mut attrs = vec![
            (Identifier::new(context, "op_name"), string_attr(context, op_name)),
            (Identifier::new(context, "param_count"), int_attr(context, param_count)),
            (Identifier::new(context, "flops"), float_attr(context, flops)),
        ];
        
        if let Some(eq) = custom_flops_eq {
            attrs.push((Identifier::new(context, "custom_flops_eq"), string_attr(context, eq)));
        }
        if let Some(eq) = custom_memory_eq {
            attrs.push((Identifier::new(context, "custom_memory_eq"), string_attr(context, eq)));
        }
        
        OperationBuilder::new("op.custom", location)
            .add_attributes(&attrs)
            .build()
    }
    
    // ============================================
    // GAN Operations (Generative Adversarial Networks)
    // ============================================
    
    /// Create a Generator operation for GANs
    pub fn generator<'c>(
        context: &'c Context,
        latent_dim: i64,
        output_dim: i64,
        num_layers: i64,
        param_count: i64,
        flops: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.generator", location)
            .add_attributes(&[
                (Identifier::new(context, "latent_dim"), int_attr(context, latent_dim)),
                (Identifier::new(context, "output_dim"), int_attr(context, output_dim)),
                (Identifier::new(context, "num_layers"), int_attr(context, num_layers)),
                (Identifier::new(context, "param_count"), int_attr(context, param_count)),
                (Identifier::new(context, "flops"), float_attr(context, flops)),
            ])
            .build()
    }
    
    /// Create a Discriminator operation for GANs
    pub fn discriminator<'c>(
        context: &'c Context,
        input_dim: i64,
        num_layers: i64,
        param_count: i64,
        flops: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.discriminator", location)
            .add_attributes(&[
                (Identifier::new(context, "input_dim"), int_attr(context, input_dim)),
                (Identifier::new(context, "num_layers"), int_attr(context, num_layers)),
                (Identifier::new(context, "param_count"), int_attr(context, param_count)),
                (Identifier::new(context, "flops"), float_attr(context, flops)),
            ])
            .build()
    }
    
    /// Create a transposed conv (deconv) for GANs
    pub fn conv_transpose2d<'c>(
        context: &'c Context,
        in_channels: i64,
        out_channels: i64,
        kernel_size: &[i64],
        stride: i64,
        param_count: i64,
        flops: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.conv_transpose2d", location)
            .add_attributes(&[
                (Identifier::new(context, "in_channels"), int_attr(context, in_channels)),
                (Identifier::new(context, "out_channels"), int_attr(context, out_channels)),
                (Identifier::new(context, "kernel_size"), int_array_attr(context, kernel_size)),
                (Identifier::new(context, "stride"), int_attr(context, stride)),
                (Identifier::new(context, "param_count"), int_attr(context, param_count)),
                (Identifier::new(context, "flops"), float_attr(context, flops)),
            ])
            .build()
    }
    
    // ============================================
    // Spiking Neural Network Operations
    // ============================================
    
    /// Create a Spiking Dense layer
    pub fn spiking_dense<'c>(
        context: &'c Context,
        in_features: i64,
        out_features: i64,
        threshold: f64,
        decay: f64,
        param_count: i64,
        flops: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.spiking_dense", location)
            .add_attributes(&[
                (Identifier::new(context, "in_features"), int_attr(context, in_features)),
                (Identifier::new(context, "out_features"), int_attr(context, out_features)),
                (Identifier::new(context, "threshold"), float_attr(context, threshold)),
                (Identifier::new(context, "decay"), float_attr(context, decay)),
                (Identifier::new(context, "param_count"), int_attr(context, param_count)),
                (Identifier::new(context, "flops"), float_attr(context, flops)),
            ])
            .build()
    }
    
    /// Create a LIF (Leaky Integrate-and-Fire) neuron
    pub fn lif_neuron<'c>(
        context: &'c Context,
        threshold: f64,
        decay: f64,
        reset_value: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.lif_neuron", location)
            .add_attributes(&[
                (Identifier::new(context, "threshold"), float_attr(context, threshold)),
                (Identifier::new(context, "decay"), float_attr(context, decay)),
                (Identifier::new(context, "reset_value"), float_attr(context, reset_value)),
            ])
            .build()
    }
    
    /// Create a Spiking Conv2D
    pub fn spiking_conv2d<'c>(
        context: &'c Context,
        in_channels: i64,
        out_channels: i64,
        kernel_size: &[i64],
        threshold: f64,
        param_count: i64,
        flops: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.spiking_conv2d", location)
            .add_attributes(&[
                (Identifier::new(context, "in_channels"), int_attr(context, in_channels)),
                (Identifier::new(context, "out_channels"), int_attr(context, out_channels)),
                (Identifier::new(context, "kernel_size"), int_array_attr(context, kernel_size)),
                (Identifier::new(context, "threshold"), float_attr(context, threshold)),
                (Identifier::new(context, "param_count"), int_attr(context, param_count)),
                (Identifier::new(context, "flops"), float_attr(context, flops)),
            ])
            .build()
    }
    
    /// Create a Surrogate Gradient operation for SNN backprop
    pub fn surrogate_gradient<'c>(
        context: &'c Context,
        gradient_type: &str, // "fast_sigmoid", "piecewise_linear", etc.
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.surrogate_gradient", location)
            .add_attributes(&[
                (Identifier::new(context, "gradient_type"), string_attr(context, gradient_type)),
            ])
            .build()
    }
    
    // ============================================
    // Diffusion Model Operations
    // ============================================
    
    /// Create a Noise Scheduler for Diffusion
    pub fn noise_scheduler<'c>(
        context: &'c Context,
        num_timesteps: i64,
        beta_start: f64,
        beta_end: f64,
        schedule_type: &str, // "linear", "cosine", "quadratic"
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.noise_scheduler", location)
            .add_attributes(&[
                (Identifier::new(context, "num_timesteps"), int_attr(context, num_timesteps)),
                (Identifier::new(context, "beta_start"), float_attr(context, beta_start)),
                (Identifier::new(context, "beta_end"), float_attr(context, beta_end)),
                (Identifier::new(context, "schedule_type"), string_attr(context, schedule_type)),
            ])
            .build()
    }
    
    /// Create a Cross-Attention for Diffusion (conditioning)
    pub fn cross_attention<'c>(
        context: &'c Context,
        query_dim: i64,
        context_dim: i64,
        num_heads: i64,
        param_count: i64,
        flops: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.cross_attention", location)
            .add_attributes(&[
                (Identifier::new(context, "query_dim"), int_attr(context, query_dim)),
                (Identifier::new(context, "context_dim"), int_attr(context, context_dim)),
                (Identifier::new(context, "num_heads"), int_attr(context, num_heads)),
                (Identifier::new(context, "param_count"), int_attr(context, param_count)),
                (Identifier::new(context, "flops"), float_attr(context, flops)),
            ])
            .build()
    }
    
    // ============================================
    // Graph Neural Network Operations
    // ============================================
    
    /// Create a Graph Convolution
    pub fn graph_conv<'c>(
        context: &'c Context,
        in_features: i64,
        out_features: i64,
        num_nodes: i64,
        param_count: i64,
        flops: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.graph_conv", location)
            .add_attributes(&[
                (Identifier::new(context, "in_features"), int_attr(context, in_features)),
                (Identifier::new(context, "out_features"), int_attr(context, out_features)),
                (Identifier::new(context, "num_nodes"), int_attr(context, num_nodes)),
                (Identifier::new(context, "param_count"), int_attr(context, param_count)),
                (Identifier::new(context, "flops"), float_attr(context, flops)),
            ])
            .build()
    }
    
    /// Create a Message Passing operation for GNN
    pub fn message_passing<'c>(
        context: &'c Context,
        node_dim: i64,
        edge_dim: i64,
        aggregation: &str, // "sum", "mean", "max"
        param_count: i64,
        flops: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.message_passing", location)
            .add_attributes(&[
                (Identifier::new(context, "node_dim"), int_attr(context, node_dim)),
                (Identifier::new(context, "edge_dim"), int_attr(context, edge_dim)),
                (Identifier::new(context, "aggregation"), string_attr(context, aggregation)),
                (Identifier::new(context, "param_count"), int_attr(context, param_count)),
                (Identifier::new(context, "flops"), float_attr(context, flops)),
            ])
            .build()
    }
    
    /// Create a LayerNorm operation
    pub fn layer_norm<'c>(
        context: &'c Context,
        hidden_size: i64,
        param_count: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.layer_norm", location)
            .add_attributes(&[
                (Identifier::new(context, "hidden_size"), int_attr(context, hidden_size)),
                (Identifier::new(context, "param_count"), int_attr(context, param_count)),
            ])
            .build()
    }
    
    /// Create a GELU operation
    pub fn gelu<'c>(_context: &'c Context, location: Location<'c>) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.gelu", location).build()
    }
    
    /// Create metrics operation
    pub fn metrics<'c>(
        context: &'c Context,
        total_flops_approx: f64,
        total_op_count: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.metrics", location)
            .add_attributes(&[
                (Identifier::new(context, "total_flops_approx"), float_attr(context, total_flops_approx)),
                (Identifier::new(context, "total_op_count"), int_attr(context, total_op_count)),
            ])
            .build()
    }
    
    // ============================================
    // Embedding and Tokenization Operations
    // ============================================
    
    /// Create an Embedding operation
    pub fn embedding<'c>(
        context: &'c Context,
        vocab_size: i64,
        embedding_dim: i64,
        param_count: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.embedding", location)
            .add_attributes(&[
                (Identifier::new(context, "vocab_size"), int_attr(context, vocab_size)),
                (Identifier::new(context, "embedding_dim"), int_attr(context, embedding_dim)),
                (Identifier::new(context, "param_count"), int_attr(context, param_count)),
            ])
            .build()
    }
    
    /// Create a Positional Embedding operation
    pub fn positional_embedding<'c>(
        context: &'c Context,
        max_seq_len: i64,
        embedding_dim: i64,
        param_count: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.positional_embedding", location)
            .add_attributes(&[
                (Identifier::new(context, "max_seq_len"), int_attr(context, max_seq_len)),
                (Identifier::new(context, "embedding_dim"), int_attr(context, embedding_dim)),
                (Identifier::new(context, "param_count"), int_attr(context, param_count)),
            ])
            .build()
    }
    
    /// Create Rotary Position Embedding (RoPE)
    pub fn rope<'c>(
        context: &'c Context,
        hidden_size: i64,
        head_dim: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.rope", location)
            .add_attributes(&[
                (Identifier::new(context, "hidden_size"), int_attr(context, hidden_size)),
                (Identifier::new(context, "head_dim"), int_attr(context, head_dim)),
            ])
            .build()
    }
    
    // ============================================
    // Regularization and Normalization Operations
    // ============================================
    
    /// Create a Dropout operation
    pub fn dropout<'c>(
        context: &'c Context,
        probability: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.dropout", location)
            .add_attributes(&[
                (Identifier::new(context, "probability"), float_attr(context, probability)),
            ])
            .build()
    }
    
    /// Create a BatchNorm operation
    pub fn batch_norm<'c>(
        context: &'c Context,
        num_features: i64,
        param_count: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.batch_norm", location)
            .add_attributes(&[
                (Identifier::new(context, "num_features"), int_attr(context, num_features)),
                (Identifier::new(context, "param_count"), int_attr(context, param_count)),
            ])
            .build()
    }
    
    /// Create RMSNorm operation
    pub fn rms_norm<'c>(
        context: &'c Context,
        hidden_size: i64,
        epsilon: f64,
        param_count: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.rms_norm", location)
            .add_attributes(&[
                (Identifier::new(context, "hidden_size"), int_attr(context, hidden_size)),
                (Identifier::new(context, "epsilon"), float_attr(context, epsilon)),
                (Identifier::new(context, "param_count"), int_attr(context, param_count)),
            ])
            .build()
    }
    
    // ============================================
    // Activation Functions
    // ============================================
    
    /// Create ReLU operation
    pub fn relu<'c>(_context: &'c Context, location: Location<'c>) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.relu", location).build()
    }
    
    /// Create SiLU (Swish) operation
    pub fn silu<'c>(_context: &'c Context, location: Location<'c>) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.silu", location).build()
    }
    
    /// Create Tanh operation
    pub fn tanh<'c>(_context: &'c Context, location: Location<'c>) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.tanh", location).build()
    }
    
    /// Create Softmax operation
    pub fn softmax<'c>(
        context: &'c Context,
        dim: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.softmax", location)
            .add_attributes(&[
                (Identifier::new(context, "dim"), int_attr(context, dim)),
            ])
            .build()
    }
    
    // ============================================
    // KV-Cache and Inference Operations
    // ============================================
    
    /// Create KV-Cache operation for inference
    pub fn kv_cache<'c>(
        context: &'c Context,
        num_layers: i64,
        num_heads: i64,
        head_dim: i64,
        max_seq_len: i64,
        memory_bytes: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.kv_cache", location)
            .add_attributes(&[
                (Identifier::new(context, "num_layers"), int_attr(context, num_layers)),
                (Identifier::new(context, "num_heads"), int_attr(context, num_heads)),
                (Identifier::new(context, "head_dim"), int_attr(context, head_dim)),
                (Identifier::new(context, "max_seq_len"), int_attr(context, max_seq_len)),
                (Identifier::new(context, "memory_bytes"), int_attr(context, memory_bytes)),
            ])
            .build()
    }
    
    /// Create Flash Attention operation
    pub fn flash_attention<'c>(
        context: &'c Context,
        hidden_size: i64,
        num_heads: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.flash_attention", location)
            .add_attributes(&[
                (Identifier::new(context, "hidden_size"), int_attr(context, hidden_size)),
                (Identifier::new(context, "num_heads"), int_attr(context, num_heads)),
            ])
            .build()
    }
    
    // ============================================
    // Quantization Operations
    // ============================================
    
    /// Create Quantize operation (INT8/INT4)
    pub fn quantize<'c>(
        context: &'c Context,
        bits: i64, // 4, 8
        scale: f64,
        zero_point: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.quantize", location)
            .add_attributes(&[
                (Identifier::new(context, "bits"), int_attr(context, bits)),
                (Identifier::new(context, "scale"), float_attr(context, scale)),
                (Identifier::new(context, "zero_point"), int_attr(context, zero_point)),
            ])
            .build()
    }
    
    /// Create Dequantize operation
    pub fn dequantize<'c>(
        context: &'c Context,
        bits: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("op.dequantize", location)
            .add_attributes(&[
                (Identifier::new(context, "bits"), int_attr(context, bits)),
            ])
            .build()
    }
}
