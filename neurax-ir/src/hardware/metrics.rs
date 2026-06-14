//! Hardware metrics utilities

use super::Bottleneck;

/// Format latency for display
pub fn format_latency(ms: f64) -> String {
    if ms >= 1000.0 {
        format!("{:.2} s", ms / 1000.0)
    } else if ms >= 1.0 {
        format!("{:.2} ms", ms)
    } else {
        format!("{:.2} µs", ms * 1000.0)
    }
}

/// Format throughput for display
pub fn format_throughput(tokens_per_s: f64) -> String {
    if tokens_per_s >= 1e9 {
        format!("{:.2} B tokens/s", tokens_per_s / 1e9)
    } else if tokens_per_s >= 1e6 {
        format!("{:.2} M tokens/s", tokens_per_s / 1e6)
    } else if tokens_per_s >= 1e3 {
        format!("{:.2} K tokens/s", tokens_per_s / 1e3)
    } else {
        format!("{:.0} tokens/s", tokens_per_s)
    }
}

/// Get bottleneck description
pub fn bottleneck_description(bottleneck: Bottleneck) -> &'static str {
    match bottleneck {
        Bottleneck::ComputeBound => "Model is compute-bound. Consider GPU upgrade or optimization.",
        Bottleneck::MemoryBound => "Model is memory-bound. Consider larger batch size or gradient checkpointing.",
        Bottleneck::Balanced => "Model is well-balanced between compute and memory.",
    }
}

/// Calculate performance headroom
pub fn performance_headroom(utilization: f64) -> &'static str {
    if utilization > 0.9 {
        "GPU is fully utilized. Limited headroom for larger models."
    } else if utilization > 0.7 {
        "Good GPU utilization with some headroom."
    } else if utilization > 0.5 {
        "Moderate GPU utilization. Consider increasing batch size."
    } else {
        "Low GPU utilization. Significant optimization potential."
    }
}
