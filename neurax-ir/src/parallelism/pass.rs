//! Parallelism IR pass

use crate::traits::IrPass;
use crate::error::ParallelismError;
use crate::NeuraxContext;
use crate::memory::MemoryIR;
use crate::graph::GraphIR;
use super::{ParallelismIR, ParallelStrategy, ParallelismMetrics};

/// Parallelism pass implementation
pub struct ParallelismPass;

impl IrPass for ParallelismPass {
    type Input = (MemoryIR, GraphIR);
    type Output = ParallelismIR;
    type Metrics = ParallelismMetrics;
    type PassError = ParallelismError;

    fn name(&self) -> &'static str {
        "ParallelismIR"
    }

    fn build(&self, input: &Self::Input, ctx: &NeuraxContext) -> Result<Self::Output, Self::PassError> {
        let (memory_ir, graph_ir) = input;
        let mut parallel_ir = ParallelismIR::default();
        
        // Analyze available strategies
        let num_gpus = ctx.config.hardware.total_gpu_count();
        let gpu_vram = ctx.config.hardware.primary_gpu()
            .map(|g| g.memory_gb * 1024 * 1024 * 1024)
            .unwrap_or(40 * 1024 * 1024 * 1024);
        
        // Data parallel
        let dp_efficiency = calculate_dp_efficiency(ctx, memory_ir.metrics.parameter_memory_bytes);
        parallel_ir.strategies.push(ParallelStrategy::DataParallel {
            num_gpus,
            efficiency: dp_efficiency,
        });
        
        // Model parallel if model doesn't fit on one GPU
        if memory_ir.metrics.peak_vram_bytes > gpu_vram {
            let _num_splits = (memory_ir.metrics.peak_vram_bytes as f64 / gpu_vram as f64).ceil() as u32;
            parallel_ir.strategies.push(ParallelStrategy::ModelParallel {
                splits: vec![], // Would be computed properly
            });
            parallel_ir.metrics.model_parallel_feasible = true;
        } else {
            parallel_ir.metrics.model_parallel_feasible = false;
        }
        
        // Pipeline parallel
        if num_gpus > 1 && graph_ir.metrics.graph_depth > 2 {
            parallel_ir.strategies.push(ParallelStrategy::PipelineParallel {
                stages: num_gpus,
                micro_batches: num_gpus * 2,
                bubble_ratio: 1.0 / (num_gpus * 2) as f64,
            });
            parallel_ir.metrics.pipeline_stages = Some(num_gpus);
        }
        
        // ZeRO
        if ctx.config.training.zero_stage > 0 {
            let zero_memory = calculate_zero_memory(memory_ir, ctx.config.training.zero_stage, num_gpus);
            parallel_ir.strategies.push(ParallelStrategy::ZeRO {
                stage: ctx.config.training.zero_stage,
                memory_per_gpu: zero_memory,
            });
        }
        
        // Hybrid (3D parallelism)
        if num_gpus >= 8 {
            let dp = ctx.config.training.parallelism.data_parallel;
            let tp = ctx.config.training.parallelism.tensor_parallel;
            let pp = ctx.config.training.parallelism.pipeline_parallel;
            if dp > 1 || tp > 1 || pp > 1 {
                parallel_ir.strategies.push(ParallelStrategy::Hybrid { dp, tp, pp });
            }
        }
        
        // Select optimal strategy
        parallel_ir.optimal_strategy = select_optimal_strategy(&parallel_ir.strategies, memory_ir, ctx);
        
        Ok(parallel_ir)
    }

    fn compute_metrics(&self, output: &mut Self::Output, ctx: &NeuraxContext) -> Result<Self::Metrics, Self::PassError> {
        let num_gpus = ctx.config.hardware.total_gpu_count();
        let _gpu_vram = ctx.config.hardware.primary_gpu()
            .map(|g| g.memory_gb * 1024 * 1024 * 1024)
            .unwrap_or(40_000_000_000);
        
        // Get memory metrics from context (stored during build)
        let param_bytes = ctx.get_metric("parameter_memory_bytes").unwrap_or(0.0) as u64;
        let gradient_bytes = ctx.get_metric("gradient_memory_bytes").unwrap_or(0.0) as u64;
        let optimizer_bytes = ctx.get_metric("optimizer_state_bytes").unwrap_or(0.0) as u64;
        
        // Calculate communication overhead
        let interconnect_bw = ctx.config.hardware.interconnect_bandwidth_gbs * 1e9;
        
        let allreduce_time_ms = if num_gpus > 1 && interconnect_bw > 0.0 {
            // Ring All-Reduce: 2 * (N-1)/N * data_size / bandwidth
            let factor = 2.0 * (num_gpus - 1) as f64 / num_gpus as f64;
            (param_bytes as f64 * factor / interconnect_bw) * 1000.0
        } else {
            0.0
        };
        
        // Compute time (from hardware IR, estimated here)
        let compute_time_ms = 100.0; // Placeholder
        
        let communication_overhead = if compute_time_ms > 0.0 {
            allreduce_time_ms / (compute_time_ms + allreduce_time_ms)
        } else {
            0.0
        };
        
        // Data parallel efficiency - use improved model
        let data_parallel_efficiency = calculate_dp_efficiency(ctx, param_bytes);
        
        // Memory per GPU
        let memory_per_gpu = match &output.optimal_strategy {
            ParallelStrategy::ZeRO { stage, .. } => {
                let base = param_bytes + gradient_bytes + optimizer_bytes;
                match stage {
                    1 => base - optimizer_bytes / num_gpus as u64,
                    2 => base / 2,
                    3 => base / num_gpus as u64,
                    _ => base,
                }
            }
            ParallelStrategy::Hybrid { tp, .. } if *tp > 1 => {
                param_bytes / *tp as u64
            }
            _ => param_bytes,
        };
        
        // Scaling efficiency curve
        let scaling_efficiency_curve = calculate_scaling_curve(ctx, param_bytes);
        
        let metrics = ParallelismMetrics {
            data_parallel_efficiency,
            model_parallel_feasible: output.metrics.model_parallel_feasible,
            pipeline_stages: output.metrics.pipeline_stages,
            communication_overhead,
            optimal_gpu_count: num_gpus,
            memory_per_gpu_bytes: memory_per_gpu,
            scaling_efficiency_curve,
            allreduce_time_ms,
            compute_time_ms,
        };
        
        output.metrics = metrics.clone();
        output.metrics_done = true;
        Ok(metrics)
    }

    fn validate(&self, _output: &Self::Output, metrics: &Self::Metrics) -> Result<(), Self::PassError> {
        if metrics.optimal_gpu_count == 0 {
            return Err(ParallelismError::InvalidConfiguration("GPU count is zero".to_string()));
        }
        Ok(())
    }
}

fn calculate_dp_efficiency(ctx: &NeuraxContext, param_bytes: u64) -> f64 {
    let num_gpus = ctx.config.hardware.total_gpu_count();
    if num_gpus <= 1 {
        return 1.0;
    }
    
    let interconnect_bw = ctx.config.hardware.interconnect_bandwidth_gbs * 1e9;
    
    // Real-world scaling efficiency factors (empirical data from large-scale training)
    // Based on: GPT-3, Megatron-LM, DeepSpeed benchmarks
    let base_efficiency: f64 = match num_gpus {
        1 => 1.0,
        2 => 0.95,   // 95% efficiency with 2 GPUs
        4 => 0.92,   // 92% efficiency with 4 GPUs
        8 => 0.88,   // 88% efficiency with 8 GPUs
        16 => 0.85,  // 85% efficiency with 16 GPUs
        32 => 0.82,  // 82% efficiency with 32 GPUs
        64 => 0.78,  // 78% efficiency with 64 GPUs
        128 => 0.75, // 75% efficiency with 128 GPUs
        256 => 0.72, // 72% efficiency with 256 GPUs
        512 => 0.68, // 68% efficiency with 512 GPUs
        _ => 0.65,   // 65% efficiency for 1024+ GPUs
    };
    
    // Adjust for interconnect bandwidth and type
    // Use both bandwidth and interconnect type from config
    let interconnect_type = ctx.config.hardware.interconnect.to_lowercase();
    let interconnect_factor = if interconnect_bw >= 600e9 || interconnect_type.contains("nvlink") {
        1.0   // NVLink
    } else if interconnect_bw >= 200e9 || interconnect_type.contains("infiniband") || interconnect_type.contains("ib") {
        0.90  // InfiniBand HDR
    } else if interconnect_bw >= 100e9 || interconnect_type.contains("ethernet") || interconnect_type.contains("roce") {
        0.80  // InfiniBand EDR or RoCE
    } else if interconnect_bw > 0.0 {
        0.70  // Slower interconnect
    } else {
        0.50  // No fast interconnect specified
    };
    
    // Communication overhead increases with model size
    // Larger models have more gradient synchronization overhead
    let param_gb = param_bytes as f64 / 1e9;
    let size_penalty = if param_gb > 100.0 {
        0.95  // 5% penalty for very large models (>100GB params)
    } else if param_gb > 10.0 {
        0.98  // 2% penalty for large models (>10GB params)
    } else {
        1.0   // No penalty for smaller models
    };
    
    (base_efficiency * interconnect_factor * size_penalty).min(1.0_f64).max(0.5_f64)
}

fn calculate_zero_memory(memory_ir: &MemoryIR, stage: u8, num_gpus: u32) -> u64 {
    let base = memory_ir.metrics.parameter_memory_bytes 
        + memory_ir.metrics.gradient_memory_bytes 
        + memory_ir.metrics.optimizer_state_bytes;
    
    match stage {
        1 => base - memory_ir.metrics.optimizer_state_bytes / num_gpus as u64,
        2 => base / 2,
        3 => base / num_gpus as u64,
        _ => base,
    }
}

fn select_optimal_strategy(
    strategies: &[ParallelStrategy],
    memory_ir: &MemoryIR,
    ctx: &NeuraxContext,
) -> ParallelStrategy {
    let gpu_vram = ctx.config.hardware.primary_gpu()
        .map(|g| g.memory_gb * 1024 * 1024 * 1024)
        .unwrap_or(40_000_000_000);
    
    // If model fits in single GPU, use data parallel
    if memory_ir.metrics.peak_vram_bytes <= gpu_vram {
        return strategies.first().cloned().unwrap_or_default();
    }
    
    // Otherwise, prefer ZeRO-3 or model parallel
    strategies.iter()
        .find(|s| matches!(s, ParallelStrategy::ZeRO { stage: 3, .. }))
        .or_else(|| strategies.iter().find(|s| matches!(s, ParallelStrategy::ModelParallel { .. })))
        .cloned()
        .unwrap_or_default()
}

fn calculate_scaling_curve(ctx: &NeuraxContext, param_bytes: u64) -> Vec<(u32, f64)> {
    let interconnect_bw = ctx.config.hardware.interconnect_bandwidth_gbs * 1e9;
    let mut curve = Vec::new();
    
    for n in [1, 2, 4, 8, 16, 32, 64, 128] {
        if n == 1 {
            curve.push((n, 1.0));
        } else if interconnect_bw > 0.0 {
            let compute_time = 100.0;
            let comm_time = (param_bytes as f64 * 2.0 * (n - 1) as f64 / n as f64 / interconnect_bw) * 1000.0;
            let efficiency = compute_time / (compute_time + comm_time);
            curve.push((n, efficiency));
        }
    }
    
    curve
}
