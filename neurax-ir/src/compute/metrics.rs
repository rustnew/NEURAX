//! Compute metrics utilities


/// Format FLOPs for display
pub fn format_flops(flops: f64) -> String {
    if flops >= 1e15 {
        format!("{:.2} PFLOPs", flops / 1e15)
    } else if flops >= 1e12 {
        format!("{:.2} TFLOPs", flops / 1e12)
    } else if flops >= 1e9 {
        format!("{:.2} GFLOPs", flops / 1e9)
    } else if flops >= 1e6 {
        format!("{:.2} MFLOPs", flops / 1e6)
    } else if flops >= 1e3 {
        format!("{:.2} KFLOPs", flops / 1e3)
    } else {
        format!("{:.0} FLOPs", flops)
    }
}

/// Calculate FLOPs utilization
pub fn calculate_flops_utilization(achieved_tflops: f64, peak_tflops: f64) -> f64 {
    if peak_tflops <= 0.0 {
        return 0.0;
    }
    achieved_tflops / peak_tflops
}

/// Get compute efficiency category
pub fn get_efficiency_category(arithmetic_intensity: f64, ridge_point: f64) -> &'static str {
    if arithmetic_intensity > ridge_point * 2.0 {
        "compute-bound"
    } else if arithmetic_intensity < ridge_point * 0.5 {
        "memory-bound"
    } else {
        "balanced"
    }
}
