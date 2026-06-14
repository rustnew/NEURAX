//! Tensor metrics

use super::TensorIR;

/// Calculate memory bandwidth requirement
pub fn calculate_memory_bandwidth(tensor_ir: &TensorIR, step_time_ms: f64) -> f64 {
    if step_time_ms <= 0.0 {
        return 0.0;
    }
    
    let total_bytes = tensor_ir.tensors.values()
        .map(|t| t.size_bytes)
        .sum::<u64>();
    
    // GB/s = bytes / (step_time_s * 1e9)
    (total_bytes as f64) / (step_time_ms / 1000.0) / 1e9
}

/// Get tensor size histogram
pub fn get_size_histogram(tensor_ir: &TensorIR) -> Vec<(String, usize)> {
    let dist = &tensor_ir.metrics.tensor_size_distribution;
    vec![
        ("tiny (<1KB)".to_string(), dist.tiny),
        ("small (1KB-1MB)".to_string(), dist.small),
        ("medium (1MB-100MB)".to_string(), dist.medium),
        ("large (100MB-1GB)".to_string(), dist.large),
        ("huge (>1GB)".to_string(), dist.huge),
    ]
}

/// Calculate activation memory with gradient checkpointing
pub fn calculate_checkpointed_memory(tensor_ir: &TensorIR, checkpoint_layers: &[String]) -> u64 {
    tensor_ir.tensors.values()
        .filter(|t| {
            // Only count tensors from checkpointed layers
            checkpoint_layers.contains(&t.produced_by)
        })
        .map(|t| t.size_bytes)
        .sum()
}
