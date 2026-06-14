//! Parallelism metrics utilities

use super::{ParallelismMetrics, ParallelStrategy};

/// Format parallelism strategy for display
pub fn format_strategy(strategy: &ParallelStrategy) -> String {
    match strategy {
        ParallelStrategy::DataParallel { num_gpus, efficiency } => {
            format!("Data Parallel ({} GPUs, {:.1}% efficiency)", num_gpus, efficiency * 100.0)
        }
        ParallelStrategy::ModelParallel { splits } => {
            format!("Model Parallel ({} splits)", splits.len())
        }
        ParallelStrategy::PipelineParallel { stages, micro_batches, bubble_ratio } => {
            format!("Pipeline Parallel ({} stages, {} micro-batches, {:.1}% bubble)", 
                stages, micro_batches, bubble_ratio * 100.0)
        }
        ParallelStrategy::Hybrid { dp, tp, pp } => {
            format!("3D Parallelism (DP={}, TP={}, PP={})", dp, tp, pp)
        }
        ParallelStrategy::ZeRO { stage, memory_per_gpu } => {
            format!("ZeRO-{} ({:.1} GB/GPU)", stage, *memory_per_gpu as f64 / 1e9)
        }
        ParallelStrategy::TensorParallel { tp_degree } => {
            format!("Tensor Parallel (TP={})", tp_degree)
        }
    }
}

/// Get recommended parallelism configuration
pub fn get_recommendation(metrics: &ParallelismMetrics) -> String {
    if metrics.communication_overhead > 0.3 {
        "High communication overhead. Consider tensor parallelism or faster interconnect.".to_string()
    } else if metrics.data_parallel_efficiency < 0.7 {
        "Low scaling efficiency. Consider hybrid parallelism.".to_string()
    } else {
        "Parallelism configuration is optimal.".to_string()
    }
}
