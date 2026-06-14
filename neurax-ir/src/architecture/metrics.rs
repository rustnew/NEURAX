//! Architecture metrics calculation

use super::ArchitectureMetrics;

/// Calculate parameter distribution statistics
pub fn calculate_param_distribution(metrics: &ArchitectureMetrics) -> ParamDistribution {
    let mut dist = ParamDistribution::default();
    
    if metrics.params_per_layer.is_empty() {
        return dist;
    }
    
    let params: Vec<_> = metrics.params_per_layer.values().copied().collect();
    let total = params.iter().sum::<u64>() as f64;
    
    dist.total = metrics.total_parameters;
    dist.mean = total / params.len() as f64;
    dist.max = params.iter().max().copied().unwrap_or(0);
    dist.min = params.iter().min().copied().unwrap_or(0);
    
    // Find largest layer
    for (id, &count) in &metrics.params_per_layer {
        if count == dist.max {
            dist.largest_layer = Some(id.clone());
            break;
        }
    }
    
    dist
}

#[derive(Debug, Clone, Default)]
pub struct ParamDistribution {
    pub total: u64,
    pub mean: f64,
    pub max: u64,
    pub min: u64,
    pub largest_layer: Option<String>,
}

/// Estimate model size in bytes
pub fn estimate_model_size(total_params: u64, precision: &str) -> u64 {
    let bytes_per_param = match precision {
        "fp64" | "float64" => 8,
        "fp32" | "float32" => 4,
        "fp16" | "float16" => 2,
        "bf16" | "bfloat16" => 2,
        "int8" => 1,
        "int4" => 1, // packed
        _ => 4,
    };
    total_params * bytes_per_param
}

/// Format parameter count for display
pub fn format_param_count(count: u64) -> String {
    if count >= 1_000_000_000 {
        format!("{:.2}B", count as f64 / 1e9)
    } else if count >= 1_000_000 {
        format!("{:.2}M", count as f64 / 1e6)
    } else if count >= 1_000 {
        format!("{:.2}K", count as f64 / 1e3)
    } else {
        count.to_string()
    }
}
