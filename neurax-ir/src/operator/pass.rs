//! Operator IR pass

use crate::traits::IrPass;
use crate::error::OperatorError;
use crate::NeuraxContext;
use crate::tensor::TensorIR;
use crate::architecture::ArchitectureIR;
use super::{OperatorIR, AtomOp, OpType, OperatorMetrics};
use neurax_parser::LayerType;
use neurax_formulas::{attention, mlp, embedding, normalization, moe};
use crate::tensor::Shape;

/// Operator pass implementation
pub struct OperatorPass;

impl IrPass for OperatorPass {
    type Input = (TensorIR, ArchitectureIR);
    type Output = OperatorIR;
    type Metrics = OperatorMetrics;
    type PassError = OperatorError;

    fn name(&self) -> &'static str {
        "OperatorIR"
    }

    fn build(&self, input: &Self::Input, ctx: &NeuraxContext) -> Result<Self::Output, Self::PassError> {
        let (_tensor_ir, arch_ir) = input;
        let mut op_ir = OperatorIR::default();
        let batch = ctx.config.training.batch_size;
        let seq = arch_ir.global_params.sequence_length.unwrap_or(512);
        let dtype = &ctx.config.training.precision;
        
        // Compute block_scale: same logic as ArchitecturePass
        let json_attention_count = arch_ir.layers.iter()
            .filter(|l| l.layer_type == neurax_parser::LayerType::Attention)
            .count();
        let json_mlp_count = arch_ir.layers.iter()
            .filter(|l| l.layer_type == neurax_parser::LayerType::Mlp)
            .count();
        let json_block_count = json_attention_count.max(json_mlp_count).max(1);
        let global_num_layers = arch_ir.global_params.num_layers
            .unwrap_or(arch_ir.layers.len() as u64) as usize;
        let block_scale = if global_num_layers > json_block_count {
            global_num_layers as f64 / json_block_count as f64
        } else {
            1.0_f64
        };
        
        for layer in &arch_ir.layers {
            let mut layer_ops = Vec::new();
            
            let mut ops = decompose_layer_to_ops(layer, batch, seq, dtype, ctx);
            
            // Scale FLOPs for repeatable layers
            let is_repeatable = matches!(layer.layer_type,
                neurax_parser::LayerType::Attention |
                neurax_parser::LayerType::Mlp |
                neurax_parser::LayerType::Normalization
            );
            if is_repeatable && block_scale > 1.0 {
                for op in &mut ops {
                    op.flops *= block_scale;
                }
            }
            
            for op in ops {
                let op_id = op_ir.operations.len();
                layer_ops.push(op_id);
                op_ir.operations.push(op);
            }
            
            op_ir.layer_ops.insert(layer.id.clone(), layer_ops);
        }
        
        Ok(op_ir)
    }

    fn compute_metrics(&self, output: &mut Self::Output, _ctx: &NeuraxContext) -> Result<Self::Metrics, Self::PassError> {
        let mut metrics = OperatorMetrics {
            total_op_count: output.operations.len(),
            ..Default::default()
        };
        
        for op in &output.operations {
            // FLOPs per layer
            let entry = metrics.flops_per_layer.entry(op.layer_id.clone()).or_insert(0.0);
            *entry += op.flops;
            metrics.total_flops_approx += op.flops;
            
            // Op type distribution
            let type_str = op.op_type.as_str().to_string();
            *metrics.op_type_distribution.entry(type_str).or_insert(0) += 1;
            
            if op.is_custom {
                metrics.custom_op_count += 1;
            }
        }
        
        output.metrics = metrics.clone();
        output.metrics_done = true;
        Ok(metrics)
    }

    fn validate(&self, _output: &Self::Output, metrics: &Self::Metrics) -> Result<(), Self::PassError> {
        if metrics.total_op_count == 0 {
            return Err(OperatorError::UnknownOperator("No operations generated".to_string()));
        }
        Ok(())
    }
}

/// Decompose a layer into atomic operations
fn decompose_layer_to_ops(
    layer: &crate::architecture::LayerDef,
    batch: usize,
    seq: usize,
    dtype: &str,
    ctx: &NeuraxContext,
) -> Vec<AtomOp> {
    let mut ops = Vec::new();
    
    match layer.layer_type {
        LayerType::Embedding => {
            let vocab = layer.params.vocab_size.unwrap_or(50000);
            let dim = layer.params.embedding_dim.unwrap_or(layer.params.hidden_size.unwrap_or(512));
            ops.push(AtomOp {
                id: 0,
                op_type: OpType::Embedding,
                layer_id: layer.id.clone(),
                input_shapes: vec![crate::tensor::Shape::known(vec![batch, seq])],
                output_shape: crate::tensor::Shape::known(vec![batch, seq, dim]),
                flops: embedding::embedding_flops(batch, seq, dim),
                param_count: (vocab * dim) as u64,
                activation_memory: (batch * seq * dim * neurax_formulas::dtype_bytes(dtype)) as u64,
                is_custom: false,
            });
        }
        LayerType::Attention => {
            let hidden = layer.params.hidden_size.unwrap_or(512);
            let heads = layer.params.num_heads.unwrap_or(8);
            let kv_heads = layer.params.num_kv_heads.unwrap_or(heads);
            let causal = layer.params.causal;
            let head_dim = hidden / heads;
            
            // Use GQA formula if kv_heads < heads (Multi-Query or Grouped-Query Attention)
            let attn_flops = if kv_heads < heads {
                attention::gqa_flops(batch, seq, hidden, heads, kv_heads, causal)
            } else {
                attention::attention_flops(batch, seq, hidden, heads, causal)
            };
            
            // QKV projections (scaled for GQA)
            let qkv_param_count = if kv_heads < heads {
                let kv_dim = kv_heads * head_dim;
                hidden * hidden + 2 * hidden * kv_dim + hidden * hidden // Q + K,V + O
            } else {
                4 * hidden * hidden // Standard: Q,K,V,O
            };
            
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Attention,
                layer_id: layer.id.clone(),
                input_shapes: vec![crate::tensor::Shape::known(vec![batch, seq, hidden])],
                output_shape: crate::tensor::Shape::known(vec![batch, seq, hidden]),
                flops: attn_flops,
                param_count: qkv_param_count as u64,
                activation_memory: (batch * seq * hidden * neurax_formulas::dtype_bytes(dtype)) as u64,
                is_custom: false,
            });
            
            // Add RoPE FLOPs for transformer models (applied to Q and K)
            let rope_flops = embedding::rope_flops(batch, seq, heads, head_dim);
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Mul,  // RoPE is element-wise rotation
                layer_id: layer.id.clone(),
                input_shapes: vec![],
                output_shape: crate::tensor::Shape::known(vec![batch, seq, hidden]),
                flops: rope_flops,
                param_count: 0,
                activation_memory: 0,
                is_custom: false,
            });
        }
        LayerType::Mlp => {
            let hidden = layer.params.hidden_size.unwrap_or(512);
            let intermediate = layer.params.intermediate_size.unwrap_or(4 * hidden);
            let activation = layer.params.activation.as_deref().unwrap_or("gelu");
            
            let flops = if layer.params.gated {
                mlp::gated_mlp_flops(batch, seq, hidden, intermediate, activation)
            } else {
                mlp::mlp_flops(batch, seq, hidden, intermediate, activation)
            };
            
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Linear,
                layer_id: layer.id.clone(),
                input_shapes: vec![crate::tensor::Shape::known(vec![batch, seq, hidden])],
                output_shape: crate::tensor::Shape::known(vec![batch, seq, hidden]),
                flops,
                param_count: layer.param_count,
                activation_memory: (batch * seq * intermediate * neurax_formulas::dtype_bytes(dtype)) as u64,
                is_custom: false,
            });
        }
        LayerType::Dense => {
            let in_f = layer.params.in_features
                .or(layer.params.in_channels)
                .or(layer.params.hidden_size)
                .unwrap_or(512);
            let out_f = layer.params.out_features
                .or(layer.params.out_channels)
                .or(layer.params.hidden_size)
                .unwrap_or(512);
            let outer_dims = layer.input_shape
                .iter()
                .take(layer.input_shape.len().saturating_sub(1))
                .copied()
                .product::<usize>()
                .max(1);
            let output_elements = if layer.output_shape.is_empty() {
                outer_dims * out_f
            } else {
                layer.output_shape.iter().copied().product::<usize>()
            };
            
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::MatMul,
                layer_id: layer.id.clone(),
                input_shapes: vec![crate::tensor::Shape::known(layer.input_shape.clone())],
                output_shape: crate::tensor::Shape::known(layer.output_shape.clone()),
                flops: 2.0 * outer_dims as f64 * in_f as f64 * out_f as f64,
                param_count: layer.param_count,
                activation_memory: (output_elements * neurax_formulas::dtype_bytes(dtype)) as u64,
                is_custom: false,
            });
        }
        LayerType::Normalization => {
            let hidden = layer.params.hidden_size.unwrap_or(512);
            let is_rms = layer.params.activation.as_deref() == Some("rms");
            
            let (op_type, flops) = if is_rms {
                (OpType::RMSNorm, normalization::rms_norm_flops(batch, seq, hidden))
            } else {
                (OpType::LayerNorm, normalization::layer_norm_flops(batch, seq, hidden))
            };
            
            ops.push(AtomOp {
                id: ops.len(),
                op_type,
                layer_id: layer.id.clone(),
                input_shapes: vec![crate::tensor::Shape::known(vec![batch, seq, hidden])],
                output_shape: crate::tensor::Shape::known(vec![batch, seq, hidden]),
                flops,
                param_count: if is_rms { hidden as u64 } else { 2 * hidden as u64 },
                activation_memory: 0,
                is_custom: false,
            });
        }
        LayerType::Conv => {
            // Conv2D FLOPs calculation
            let in_ch = layer.params.in_channels
                .or_else(|| ctx.config.data.image_channels)
                .unwrap_or(3);
            let out_ch = layer.params.out_channels.unwrap_or(64);
            let kernel_h = layer.params.kernel_size.unwrap_or(3);
            let kernel_w = layer.params.kernel_w.unwrap_or(kernel_h);
            let stride = layer.params.stride.unwrap_or(1);
            let padding = layer.params.padding.unwrap_or(0);
            let groups = 1; // Standard convolution
            
            // Calculate output dimensions from input_shape or data config
            let (batch, in_h, in_w) = if layer.input_shape.len() >= 4 {
                (layer.input_shape[0], layer.input_shape[2], layer.input_shape[3])
            } else {
                // Use data config for image dimensions
                let h = ctx.config.data.image_height.unwrap_or(224);
                let w = ctx.config.data.image_width.unwrap_or(224);
                (ctx.config.training.batch_size, h, w)
            };
            
            let out_h = (in_h + 2 * padding - kernel_h + stride) / stride;
            let out_w = (in_w + 2 * padding - kernel_w + stride) / stride;
            
            let flops = super::formulas::conv2d_flops(batch, out_h, out_w, out_ch, kernel_h, kernel_w, in_ch, groups);
            
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Conv2D,
                layer_id: layer.id.clone(),
                input_shapes: vec![],
                output_shape: crate::tensor::Shape::default(),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: false,
            });
        }
        LayerType::Pooling => {
            // Pooling FLOPs calculation
            let kernel_size = layer.params.kernel_size.unwrap_or(2);
            
            // Calculate output dimensions from input_shape
            let (batch, channels, in_h, in_w) = if layer.input_shape.len() >= 4 {
                (layer.input_shape[0], layer.input_shape[1], layer.input_shape[2], layer.input_shape[3])
            } else {
                (1, 64, 224, 224)
            };
            
            let stride = layer.params.stride.unwrap_or(2);
            let out_h = (in_h + stride - 1) / stride;
            let out_w = (in_w + stride - 1) / stride;
            
            let flops = super::formulas::pooling_flops(batch, channels, out_h, out_w, kernel_size);
            
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Pooling(super::PoolingType::Max),
                layer_id: layer.id.clone(),
                input_shapes: vec![],
                output_shape: crate::tensor::Shape::default(),
                flops,
                param_count: 0,
                activation_memory: 0,
                is_custom: false,
            });
        }
        LayerType::MoE => {
            // MoE: router + expert computation
            let hidden = layer.params.hidden_size.unwrap_or(512);
            let intermediate = layer.params.intermediate_size.unwrap_or(4 * hidden);
            let num_experts = layer.params.num_experts.unwrap_or(8);
            let top_k = layer.params.top_k.unwrap_or(2);
            let flops = moe::moe_flops(batch, seq, hidden, intermediate, num_experts, top_k as f64);
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::MoE,
                layer_id: layer.id.clone(),
                input_shapes: vec![Shape::known(vec![batch, seq, hidden])],
                output_shape: Shape::known(vec![batch, seq, hidden]),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: false,
            });
        }
        // CNN layer types - use conv-like operations
        LayerType::ResidualBlock | LayerType::Mbconv | LayerType::Inception 
        | LayerType::DenseBlock | LayerType::ConvnextBlock | LayerType::ShuffleUnit 
        | LayerType::C2f | LayerType::Detection | LayerType::Transition => {
            // Treat as conv-like operations
            let hidden = layer.params.hidden_size.unwrap_or(512);
            let flops = (batch * seq * hidden * hidden) as f64;
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Conv2D,
                layer_id: layer.id.clone(),
                input_shapes: vec![],
                output_shape: crate::tensor::Shape::default(),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: false,
            });
        }
        // State Space Model layer types
        LayerType::MambaBlock | LayerType::S4Block | LayerType::H3Block 
        | LayerType::StateSpace | LayerType::RwkvBlock | LayerType::RetentionBlock => {
            let hidden = layer.params.hidden_size.unwrap_or(512);
            let state_dim = layer.params.state_dim.unwrap_or(16);
            let expand = layer.params.expansion_factor.unwrap_or(2);
            
            // Use proper Mamba FLOPs formula from neurax_formulas
            let flops = neurax_formulas::ssm::mamba_flops(batch, seq, hidden, state_dim, expand);
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Linear,
                layer_id: layer.id.clone(),
                input_shapes: vec![Shape::known(vec![batch, seq, hidden])],
                output_shape: Shape::known(vec![batch, seq, hidden]),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: false,
            });
        }
        // GAN layer types
        LayerType::GeneratorBlock | LayerType::DiscriminatorBlock 
        | LayerType::StyleMod | LayerType::AdaIN | LayerType::MinibatchStd 
        | LayerType::PixelNorm | LayerType::SelfAttention | LayerType::SpectralNorm
        | LayerType::ProgressiveBlock => {
            let hidden = layer.params.hidden_size.unwrap_or(512);
            let flops = (batch * seq * hidden * hidden) as f64;
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Conv2D,
                layer_id: layer.id.clone(),
                input_shapes: vec![],
                output_shape: crate::tensor::Shape::default(),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: false,
            });
        }
        // LSTM/RNN layer types
        LayerType::LstmBlock | LayerType::GruBlock | LayerType::RnnCell 
        | LayerType::Bidirectional | LayerType::EncoderBlock | LayerType::DecoderBlock => {
            let hidden = layer.params.rnn_hidden_size.unwrap_or(512);
            // LSTM: 4 gates, GRU: 3 gates
            let gates = if matches!(layer.layer_type, LayerType::GruBlock) { 3 } else { 4 };
            let flops = (batch * seq * hidden * hidden * gates) as f64;
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Linear,
                layer_id: layer.id.clone(),
                input_shapes: vec![Shape::known(vec![batch, seq, hidden])],
                output_shape: Shape::known(vec![batch, seq, hidden]),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: false,
            });
        }
        // Diffusion layer types
        LayerType::UnetBlock | LayerType::TimeEmbedding | LayerType::CrossAttention 
        | LayerType::DownBlock | LayerType::UpBlock | LayerType::MidBlock 
        | LayerType::ResnetBlock | LayerType::TimestepBlock | LayerType::ConditionBlock 
        | LayerType::NoisePredictor | LayerType::VaeEncoder | LayerType::VaeDecoder => {
            let hidden = layer.params.hidden_size.unwrap_or(320);
            let flops = (batch * seq * hidden * hidden * 4) as f64;
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Linear,
                layer_id: layer.id.clone(),
                input_shapes: vec![Shape::known(vec![batch, seq, hidden])],
                output_shape: Shape::known(vec![batch, seq, hidden]),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: false,
            });
        }
        // Custom layer - use custom equations if provided
        LayerType::Custom => {
            let hidden = layer.params.hidden_size.unwrap_or(512);
            // Use custom equations or default FLOPs
            let flops = if let Some(ref eqs) = layer.custom_equations {
                // Parse custom FLOPs equation (simplified)
                eqs.flops_forward.as_ref()
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0)
            } else {
                0.0
            };
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Custom,
                layer_id: layer.id.clone(),
                input_shapes: vec![Shape::known(vec![batch, seq, hidden])],
                output_shape: Shape::known(vec![batch, seq, hidden]),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: true,
            });
        }
    }
    
    ops
}
