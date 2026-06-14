//! Parallelism strategies

use crate::parallelism::ParallelStrategy;

/// Determine optimal parallelism configuration
pub fn determine_optimal_parallelism(
    model_memory: u64,
    gpu_memory: u64,
    num_gpus: u32,
    _interconnect_bw_gbs: f64,
) -> ParallelStrategy {
    fn estimate_dp_efficiency(_num_gpus: u32, _interconnect_bw_gbs: f64) -> f64 {
        0.95 // Estimated
    }

    if model_memory <= gpu_memory {
        // Model fits on single GPU
        ParallelStrategy::DataParallel {
            num_gpus,
            efficiency: estimate_dp_efficiency(num_gpus, _interconnect_bw_gbs),
        }
    } else if num_gpus >= 4 {
        // Need model parallelism
        let tp_degree = ((model_memory as f64 / gpu_memory as f64).ceil() as u32).min(num_gpus);
        ParallelStrategy::TensorParallel {
            tp_degree,
        }
    } else {
        ParallelStrategy::ZeRO {
            stage: 3,
            memory_per_gpu: model_memory / num_gpus as u64,
        }
    }
}

/// Calculate bubble ratio for pipeline parallelism
pub fn pipeline_bubble_ratio(num_stages: u32, num_micro_batches: u32) -> f64 {
    if num_stages == 0 || num_micro_batches == 0 {
        return 0.0;
    }
    (num_stages - 1) as f64 / (num_stages - 1 + num_micro_batches) as f64
}

/// Calculate ZeRO memory savings
pub fn zero_memory_savings(stage: u8, num_gpus: u32) -> f64 {
    match stage {
        0 => 1.0,
        1 => 1.0 / num_gpus as f64, // Only optimizer states sharded
        2 => 2.0 / num_gpus as f64, // Gradients + optimizer states
        3 => 4.0 / num_gpus as f64, // Parameters + gradients + optimizer states
        _ => 1.0,
    }
}
