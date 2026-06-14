//! Tensor IR pass

use crate::traits::IrPass;
use crate::error::TensorError;
use crate::NeuraxContext;
use crate::graph::GraphIR;
use super::{TensorIR, TensorInfo, TensorMetrics, Shape, LayerTensors};
use neurax_parser::LayerType;

/// Tensor pass implementation
pub struct TensorPass;

impl TensorPass {
    /// ShapeInferenceGate threshold - block analysis if less than 70% dimensions resolved
    pub const SHAPE_GATE_THRESHOLD: f32 = 0.70;
}

impl IrPass for TensorPass {
    type Input = GraphIR;
    type Output = TensorIR;
    type Metrics = TensorMetrics;
    type PassError = TensorError;

    fn name(&self) -> &'static str {
        "TensorIR"
    }

    fn build(&self, input: &Self::Input, ctx: &NeuraxContext) -> Result<Self::Output, Self::PassError> {
        let mut tensor_ir = TensorIR::default();
        let dtype = ctx.config.training.precision.clone();
        
        // Get batch size and sequence length for activation memory calculation
        let batch = ctx.config.training.batch_size;
        let seq = ctx.config.model.global_params.sequence_length.unwrap_or(512);
        let hidden = ctx.config.model.global_params.embedding_dim
            .or_else(|| ctx.config.model.layers.iter().find_map(|l| l.params.hidden_size))
            .unwrap_or(512);
        
        // Process each node in topological order
        for &node_idx in &input.topo_order {
            if let Some(node) = input.dag.node_weight(node_idx) {
                // Create input tensors with proper shapes if not provided
                let mut input_tensors = Vec::new();
                for (i, shape) in node.input_shapes.iter().enumerate() {
                    let (tensor_shape, size_bytes) = if shape.is_empty() {
                        // Default shape based on layer type
                        let default_shape = match node.layer_type {
                            LayerType::Embedding => vec![batch, seq],
                            _ => vec![batch, seq, hidden],
                        };
                        let bytes = default_shape.iter().product::<usize>() 
                            * neurax_formulas::dtype_bytes(&dtype);
                        (default_shape, bytes as u64)
                    } else {
                        let bytes = shape.iter().product::<usize>() 
                            * neurax_formulas::dtype_bytes(&dtype);
                        (shape.clone(), bytes as u64)
                    };
                    
                    let tensor_id = format!("{}_input_{}", node.layer_id, i);
                    let info = TensorInfo {
                        id: tensor_id.clone(),
                        shape: Shape::known(tensor_shape),
                        dtype: dtype.clone(),
                        size_bytes,
                        produced_by: "input".to_string(),
                        consumed_by: vec![node.layer_id.clone()],
                    };
                    tensor_ir.tensors.insert(tensor_id.clone(), info);
                    input_tensors.push(tensor_id);
                }
                
                // Create output tensor with proper shape
                let output_shape = if !node.output_shape.is_empty() {
                    Shape::known(node.output_shape.clone())
                } else {
                    // Use default shape based on layer type
                    let default_shape = match node.layer_type {
                        LayerType::Embedding => vec![batch, seq, hidden],
                        _ => vec![batch, seq, hidden],
                    };
                    Shape::known(default_shape)
                };
                
                let output_size = output_shape.size_bytes(&dtype);
                
                let output_id = format!("{}_output", node.layer_id);
                let output_info = TensorInfo {
                    id: output_id.clone(),
                    shape: output_shape,
                    dtype: dtype.clone(),
                    size_bytes: output_size,
                    produced_by: node.layer_id.clone(),
                    consumed_by: vec![],
                };
                tensor_ir.tensors.insert(output_id.clone(), output_info);
                
                // Record layer tensors
                tensor_ir.layer_tensors.insert(node.layer_id.clone(), LayerTensors {
                    inputs: input_tensors,
                    outputs: vec![output_id],
                });
            }
        }
        
        Ok(tensor_ir)
    }

    fn compute_metrics(&self, output: &mut Self::Output, _ctx: &NeuraxContext) -> Result<Self::Metrics, Self::PassError> {
        let mut metrics = TensorMetrics {
            total_tensor_count: output.tensors.len(),
            resolution_ratio: 1.0,
            unresolved_dim_count: 0,
            total_dim_count: 0,
            ..Default::default()
        };
        
        let mut resolved_dims = 0usize;
        let mut total_dims = 0usize;
        
        for (id, tensor) in &output.tensors {
            metrics.tensor_size_distribution.add(tensor.size_bytes);
            metrics.activation_memory_bytes += tensor.size_bytes;
            
            if tensor.size_bytes > metrics.largest_tensor_bytes {
                metrics.largest_tensor_bytes = tensor.size_bytes;
                metrics.largest_tensor_id = Some(id.clone());
            }
            
            // Count resolved vs unresolved dimensions
            for dim in &tensor.shape.0 {
                total_dims += 1;
                match dim {
                    crate::tensor::Dim::Known(_) => resolved_dims += 1,
                    _ => metrics.unresolved_dim_count += 1,
                }
            }
        }
        
        // Calculate resolution ratio
        metrics.total_dim_count = total_dims;
        metrics.resolution_ratio = if total_dims > 0 {
            resolved_dims as f32 / total_dims as f32
        } else {
            1.0
        };
        
        output.metrics = metrics.clone();
        output.metrics_done = true;
        Ok(metrics)
    }
    
    fn validate(&self, _output: &Self::Output, metrics: &Self::Metrics) -> Result<(), Self::PassError> {
        if metrics.total_tensor_count == 0 {
            return Err(TensorError::EmptyTensor);
        }
        
        // ShapeInferenceGate: block if resolution ratio < 70%
        if metrics.resolution_ratio < Self::SHAPE_GATE_THRESHOLD {
            return Err(TensorError::ShapeGateBlocked {
                resolved: metrics.resolution_ratio * 100.0,
                threshold: Self::SHAPE_GATE_THRESHOLD * 100.0,
                unresolved: metrics.unresolved_dim_count,
            });
        }
        
        Ok(())
    }
}

/// Propagate shape through a layer
fn propagate_shape(layer_type: &LayerType, input_shapes: &[Vec<usize>], config: &neurax_parser::ModelConfig) -> Shape {
    if input_shapes.is_empty() {
        return Shape::default();
    }
    
    let input = &input_shapes[0];
    
    match layer_type {
        LayerType::Embedding => {
            // [batch, seq] -> [batch, seq, embedding_dim]
            let embed_dim = config.model.global_params.embedding_dim
                .or_else(|| config.model.layers.first().and_then(|l| l.params.embedding_dim))
                .unwrap_or(512);
            let mut shape = input.to_vec();
            shape.push(embed_dim);
            Shape::known(shape)
        }
        LayerType::Attention => {
            // Attention preserves shape [batch, seq, hidden]
            Shape::known(input.to_vec())
        }
        LayerType::Mlp | LayerType::Dense => {
            // MLP preserves batch dims, changes last dim
            // For now, preserve input shape
            Shape::known(input.to_vec())
        }
        LayerType::Normalization => {
            // Normalization preserves shape
            Shape::known(input.to_vec())
        }
        LayerType::Conv => {
            // Conv changes shape based on kernel, stride, padding
            // Simplified: preserve input shape
            Shape::known(input.to_vec())
        }
        LayerType::Pooling => {
            // Pooling reduces spatial dims
            Shape::known(input.to_vec())
        }
        LayerType::MoE => {
            // MoE preserves shape
            Shape::known(input.to_vec())
        }
        // CNN layer types - preserve shape for now
        LayerType::ResidualBlock | LayerType::Mbconv | LayerType::Inception 
        | LayerType::DenseBlock | LayerType::ConvnextBlock | LayerType::ShuffleUnit 
        | LayerType::C2f | LayerType::Detection | LayerType::Transition => {
            Shape::known(input.to_vec())
        }
        // State Space Model layer types
        LayerType::MambaBlock | LayerType::S4Block | LayerType::H3Block 
        | LayerType::StateSpace | LayerType::RwkvBlock | LayerType::RetentionBlock => {
            // SSM preserves sequence and hidden dims
            Shape::known(input.to_vec())
        }
        // GAN layer types
        LayerType::GeneratorBlock | LayerType::DiscriminatorBlock 
        | LayerType::StyleMod | LayerType::AdaIN | LayerType::MinibatchStd 
        | LayerType::PixelNorm | LayerType::SelfAttention | LayerType::SpectralNorm
        | LayerType::ProgressiveBlock => {
            Shape::known(input.to_vec())
        }
        // LSTM/RNN layer types
        LayerType::LstmBlock | LayerType::GruBlock | LayerType::RnnCell 
        | LayerType::Bidirectional | LayerType::EncoderBlock | LayerType::DecoderBlock => {
            // RNN preserves [batch, seq, hidden]
            Shape::known(input.to_vec())
        }
        // Diffusion layer types
        LayerType::UnetBlock | LayerType::TimeEmbedding | LayerType::CrossAttention 
        | LayerType::DownBlock | LayerType::UpBlock | LayerType::MidBlock 
        | LayerType::ResnetBlock | LayerType::TimestepBlock | LayerType::ConditionBlock 
        | LayerType::NoisePredictor | LayerType::VaeEncoder | LayerType::VaeDecoder => {
            // Diffusion preserves shape
            Shape::known(input.to_vec())
        }
        // Custom layer - preserve input shape
        LayerType::Custom => {
            Shape::known(input.to_vec())
        }
    }
}
