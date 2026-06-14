//! Memory IR pass

use crate::traits::IrPass;
use crate::error::MemoryError;
use crate::NeuraxContext;
use crate::compute::ComputeIR;
use crate::tensor::TensorIR;
use crate::architecture::ArchitectureIR;
use super::{MemoryIR, MemoryMetrics, LivenessInterval, MemorySnapshot, OomRisk, TensorSizeDist};
use neurax_formulas::dtype_bytes;

/// Memory pass implementation
pub struct MemoryPass;

impl IrPass for MemoryPass {
    type Input = (ComputeIR, TensorIR, ArchitectureIR);
    type Output = MemoryIR;
    type Metrics = MemoryMetrics;
    type PassError = MemoryError;

    fn name(&self) -> &'static str {
        "MemoryIR"
    }

    fn build(&self, input: &Self::Input, ctx: &NeuraxContext) -> Result<Self::Output, Self::PassError> {
        let (_compute_ir, tensor_ir, arch_ir) = input;
        let mut memory_ir = MemoryIR::default();
        
        // Calculate total_parameters with same scaling logic as ArchitecturePass
        let global_num_layers = ctx.config.model.global_params.num_layers
            .unwrap_or(ctx.config.model.layers.len() as u64) as usize;
        let num_dense_layers = ctx.config.model.global_params.num_dense_layers
            .unwrap_or(0) as usize;
        
        let json_moe_count = ctx.config.model.layers.iter()
            .filter(|l| l.layer_type == neurax_parser::LayerType::MoE)
            .count();
        let json_attention_count = ctx.config.model.layers.iter()
            .filter(|l| l.layer_type == neurax_parser::LayerType::Attention)
            .count();
        let json_mamba_count = ctx.config.model.layers.iter()
            .filter(|l| matches!(l.layer_type, 
                neurax_parser::LayerType::MambaBlock |
                neurax_parser::LayerType::S4Block |
                neurax_parser::LayerType::H3Block |
                neurax_parser::LayerType::StateSpace))
            .count();
        
        let total_parameters: u64 = ctx.config.model.layers.iter().map(|l| {
            let raw_params = crate::architecture::calculate_layer_params(&neurax_parser::Layer {
                id: l.id.clone(),
                layer_type: l.layer_type,
                input_shape: l.input_shape.clone(),
                output_shape: l.output_shape.clone(),
                params: l.params.clone(),
                custom_equations: l.custom_equations.clone(),
            });
            
            let is_repeatable = matches!(l.layer_type,
                neurax_parser::LayerType::Attention |
                neurax_parser::LayerType::Mlp |
                neurax_parser::LayerType::Normalization |
                neurax_parser::LayerType::MoE |
                neurax_parser::LayerType::MambaBlock |
                neurax_parser::LayerType::S4Block |
                neurax_parser::LayerType::H3Block |
                neurax_parser::LayerType::StateSpace |
                neurax_parser::LayerType::RwkvBlock |
                neurax_parser::LayerType::RetentionBlock
            );
            
            let scale = if num_dense_layers > 0 && json_moe_count > 0 {
                if l.layer_type == neurax_parser::LayerType::MoE {
                    let num_moe_layers = global_num_layers.saturating_sub(num_dense_layers);
                    num_moe_layers as f64 / json_moe_count.max(1) as f64
                } else {
                    let dense_blocks = json_attention_count.saturating_sub(json_moe_count).max(1);
                    num_dense_layers as f64 / dense_blocks as f64
                }
            } else {
                // Use mamba_count for SSM models, attention_count for transformers
                let json_block_count = json_attention_count.max(json_mamba_count).max(1);
                if global_num_layers > json_block_count {
                    global_num_layers as f64 / json_block_count as f64
                } else {
                    1.0
                }
            };
            
            if is_repeatable { (raw_params as f64 * scale).round() as u64 } else { raw_params }
        }).sum();
        
        memory_ir.total_parameters = total_parameters;
        
        // Calculate liveness intervals
        memory_ir.liveness = calculate_liveness(tensor_ir, arch_ir);
        
        // Build memory timeline
        memory_ir.memory_timeline = build_memory_timeline(&memory_ir.liveness, arch_ir);
        
        Ok(memory_ir)
    }

    fn compute_metrics(&self, output: &mut Self::Output, ctx: &NeuraxContext) -> Result<Self::Metrics, Self::PassError> {
        let dtype = &ctx.config.training.precision;
        let dtype_bytes_val = dtype_bytes(dtype) as u64;
        
        // Quant factor for param storage
        let quant_factor = match dtype.as_str() {
            "fp32"           => 1.0_f64,
            "fp16" | "bfloat16" => 0.5,
            "fp8"            => 0.25,
            "int8"           => 0.25,
            "int4"           => 0.125,
            _                => 1.0,
        };
        
        // Use total_parameters from ArchitectureIR (stored in MemoryIR during build)
        let total_parameters = output.total_parameters;
        
        let global_num_layers = ctx.config.model.global_params.num_layers
            .unwrap_or(ctx.config.model.layers.len() as u64) as usize;
        
        // ── Scaling factor identical to ArchitecturePass ──────────────────────
        let json_attention_count = ctx.config.model.layers.iter()
            .filter(|l| l.layer_type == neurax_parser::LayerType::Attention)
            .count();
        let json_mlp_count = ctx.config.model.layers.iter()
            .filter(|l| l.layer_type == neurax_parser::LayerType::Mlp)
            .count();
        let json_moe_count = ctx.config.model.layers.iter()
            .filter(|l| l.layer_type == neurax_parser::LayerType::MoE)
            .count();
        
        let global_num_layers = ctx.config.model.global_params.num_layers
            .unwrap_or(ctx.config.model.layers.len() as u64) as usize;
        let num_dense_layers = ctx.config.model.global_params.num_dense_layers
            .unwrap_or(0) as usize;
        
        // Use same scaling logic as ArchitecturePass
        let (dense_scale, moe_scale) = if num_dense_layers > 0 && json_moe_count > 0 {
            let dense_blocks = json_attention_count.saturating_sub(json_moe_count).max(1);
            let moe_blocks = json_moe_count.max(1);
            let num_moe_layers = global_num_layers.saturating_sub(num_dense_layers);
            
            let ds = if dense_blocks > 0 { num_dense_layers as f64 / dense_blocks as f64 } else { 1.0 };
            let ms = if moe_blocks > 0 { num_moe_layers as f64 / moe_blocks as f64 } else { 1.0 };
            (ds, ms)
        } else {
            let json_block_count = json_attention_count.max(json_mlp_count).max(1);
            let scale = if global_num_layers > json_block_count {
                global_num_layers as f64 / json_block_count as f64
            } else {
                1.0_f64
            };
            (scale, scale)
        };

        // ── Parameter memory ─────────────────────────────────────────────────
        // Use total_parameters already calculated with correct scaling
        let parameter_memory_bytes = (total_parameters as f64 * dtype_bytes_val as f64 * quant_factor) as u64;

        // ── Activation memory from liveness ──────────────────────────────────
        // Gradient checkpointing reduces activation memory to ~sqrt(L) layers kept
        // Formula: sqrt(num_layers) / num_layers for L layers
        let gradient_checkpointing = ctx.config.training.gradient_checkpointing;
        let num_layers = global_num_layers.max(1);
        let checkpoint_factor = if gradient_checkpointing {
            (num_layers as f64).sqrt() / num_layers as f64
        } else { 
            1.0 
        };
        
        // Use realistic activation memory formula instead of raw liveness
        // Activation memory ≈ batch_size * seq_len * hidden_size * num_layers * dtype_bytes
        // With tensor parallelism, this is divided by tensor_parallel degree
        let batch_size = ctx.config.training.batch_size;
        let seq_len = ctx.config.model.global_params.sequence_length.unwrap_or(2048) as usize;
        let hidden_size = ctx.config.model.global_params.embedding_dim.unwrap_or(512) as usize;
        
        // For MoE models, account for expert routing overhead
        let moe_factor = if ctx.config.model.model_type.as_str() == "moe" {
            1.5 // MoE models have ~50% more activation memory due to expert routing
        } else {
            1.0
        };
        
        // Per-layer activation: batch * seq * hidden * dtype_bytes
        let per_layer_activation = batch_size * seq_len * hidden_size * dtype_bytes_val as usize;
        
        // Total activation across all layers
        let total_activation = per_layer_activation as f64 * num_layers as f64 * moe_factor * checkpoint_factor;
        
        // Divide by tensor parallelism degree (activations are sharded across TP ranks)
        let tensor_parallel = ctx.config.training.parallelism.tensor_parallel.max(1) as f64;
        let activation_memory_bytes = (total_activation / tensor_parallel).round() as u64;

        // ── Determine if this is a training run ───────────────────────────────
        // Training = optimizer is not "none"/"inference" AND there are steps
        let has_optimizer = !ctx.config.training.optimizer.is_empty()
            && ctx.config.training.optimizer.to_lowercase() != "none"
            && ctx.config.training.optimizer.to_lowercase() != "inference";
        let is_training = has_optimizer;

        // ── Gradient memory ─────────────────────────────────────────────────
        // Stored in FP32 regardless of training precision (mixed precision)
        let gradient_memory_bytes = if is_training {
            // FP32 gradients: params * 4 bytes
            total_parameters * 4 // FP32
        } else {
            0
        };

        // ── Optimizer state memory ──────────────────────────────────────────
        // Adam/AdamW: 2 FP32 states (momentum + variance) per param
        // SGD with momentum: 1 FP32 state
        let optimizer_state_bytes = if is_training {
            let total_params_count = gradient_memory_bytes / 4;  // reuse calculation
            match ctx.config.training.optimizer.to_lowercase().as_str() {
                "sgd"   => total_params_count * 4,               // 1 FP32 state
                "lamb"  => total_params_count * 4 * 2,            // same as Adam
                _       => total_params_count * 4 * 2,            // Adam: 2 FP32 states
            }
        } else {
            0
        };

        // ── ZeRO: partition across GPUs ────────────────────────────────────
        let num_gpus = ctx.config.hardware.total_gpu_count().max(1) as u64;
        let zero_stage = ctx.config.training.zero_stage;
        // ZeRO-1: partition optimizer states
        // ZeRO-2: partition optimizer + gradients
        // ZeRO-3: partition optimizer + gradients + parameters
        let (param_factor, grad_factor, optim_factor) = match zero_stage {
            1 => (1.0, 1.0, 1.0 / num_gpus as f64),
            2 => (1.0, 1.0 / num_gpus as f64, 1.0 / num_gpus as f64),
            3 => (1.0 / num_gpus as f64, 1.0 / num_gpus as f64, 1.0 / num_gpus as f64),
            _ => (1.0, 1.0, 1.0),
        };

        let effective_param_mem   = (parameter_memory_bytes as f64 * param_factor) as u64;
        let effective_grad_mem    = (gradient_memory_bytes as f64 * grad_factor) as u64;
        let effective_optim_mem   = (optimizer_state_bytes as f64 * optim_factor) as u64;

        // ── Peak VRAM (correct formula) ───────────────────────────────────
        // Per-GPU peak = params_per_gpu + activations + gradients_per_gpu + optimizer_per_gpu
        // Activations are always local (not sharded by ZeRO)
        let peak_vram_bytes = effective_param_mem
            + activation_memory_bytes
            + effective_grad_mem
            + effective_optim_mem;

        // GPU VRAM capacity
        let gpu_vram_bytes = ctx.config.hardware.gpus.first()
            .map(|g| g.memory_gb as u64 * 1024 * 1024 * 1024)
            .unwrap_or(40 * 1024 * 1024 * 1024);

        // OOM risk
        let oom_risk = OomRisk::from_ratio(peak_vram_bytes, gpu_vram_bytes);

        // Max batch size that fits
        let max_batch_size_fit = calculate_max_batch_size(
            effective_param_mem,
            effective_optim_mem,
            activation_memory_bytes,
            gpu_vram_bytes,
            ctx.config.training.batch_size,
        );

        // Memory bandwidth requirement
        let memory_bandwidth_req = ctx.config.hardware.gpus.first()
            .map(|g| g.memory_bandwidth_gbs)
            .unwrap_or(1000.0);

        let metrics = MemoryMetrics {
            parameter_memory_bytes: effective_param_mem,
            activation_memory_bytes,
            gradient_memory_bytes: effective_grad_mem,
            optimizer_state_bytes: effective_optim_mem,
            peak_vram_bytes,
            memory_bandwidth_req,
            tensor_size_dist: TensorSizeDist::default(),
            fragmentation_estimate: 0.1,
            max_batch_size_fit,
            oom_risk,
            gpu_vram_bytes,
        };

        output.metrics = metrics.clone();
        output.metrics_done = true;
        Ok(metrics)
    }

    fn validate(&self, _output: &Self::Output, metrics: &Self::Metrics) -> Result<(), Self::PassError> {
        if metrics.peak_vram_bytes == 0 {
            return Err(MemoryError::MemoryCalculationFailed("Peak VRAM is zero".to_string()));
        }
        if metrics.parameter_memory_bytes == 0 {
            return Err(MemoryError::MemoryCalculationFailed("Parameter memory is zero".to_string()));
        }
        Ok(())
    }
}

fn calculate_liveness(tensor_ir: &TensorIR, arch_ir: &ArchitectureIR) -> Vec<LivenessInterval> {
    let mut intervals = Vec::new();
    
    for (id, tensor) in &tensor_ir.tensors {
        // Find the step where this tensor is created and last used
        let start_step = arch_ir.layers.iter()
            .position(|l| l.id == tensor.produced_by)
            .unwrap_or(0);
        
        let end_step = tensor.consumed_by.iter()
            .filter_map(|consumer| arch_ir.layers.iter()
                .position(|l| &l.id == consumer))
            .max()
            .unwrap_or(start_step);
        
        intervals.push(LivenessInterval {
            tensor_id: id.clone(),
            size_bytes: tensor.size_bytes,
            start_step,
            end_step,
        });
    }
    
    intervals
}

fn build_memory_timeline(liveness: &[LivenessInterval], arch_ir: &ArchitectureIR) -> Vec<MemorySnapshot> {
    let num_steps = arch_ir.layers.len();
    let mut timeline = Vec::new();
    
    for step in 0..num_steps {
        let live_tensors: Vec<_> = liveness.iter()
            .filter(|l| l.start_step <= step && l.end_step >= step)
            .map(|l| l.tensor_id.clone())
            .collect();
        
        let total_memory: u64 = liveness.iter()
            .filter(|l| l.start_step <= step && l.end_step >= step)
            .map(|l| l.size_bytes)
            .sum();
        
        timeline.push(MemorySnapshot {
            step,
            live_tensors,
            total_memory,
        });
    }
    
    timeline
}

fn calculate_max_batch_size(
    param_bytes: u64,
    optim_bytes: u64,
    activation_bytes: u64,
    gpu_vram: u64,
    current_batch: usize,
) -> u32 {
    // Fixed memory (params + optimizer states)
    let fixed = param_bytes + optim_bytes;
    
    // Memory per sample (activation / batch)
    let per_sample = if current_batch > 0 {
        activation_bytes / current_batch as u64
    } else {
        1024 // Default estimate
    };
    
    if per_sample == 0 {
        return current_batch as u32;
    }
    
    let available = gpu_vram.saturating_sub(fixed);
    (available / per_sample) as u32
}

#[allow(dead_code)]
fn estimate_memory_time(ctx: &NeuraxContext, _bytes_moved: u64) -> f64 {
    // GB/s = bytes / time
    // Estimate based on GPU specs
    let gpu_bw = ctx.config.hardware.gpus.first()
        .map(|g| g.memory_bandwidth_gbs)
        .unwrap_or(1000.0);
    
    // Assume we use ~70% of peak bandwidth
    gpu_bw * 0.7
}

fn calculate_memory_bandwidth(ctx: &NeuraxContext, _peak_vram_bytes: u64) -> f64 {
    // Estimate memory bandwidth requirement based on GPU specs
    ctx.config.hardware.gpus.first()
        .map(|g| g.memory_bandwidth_gbs)
        .unwrap_or(1000.0)
}
