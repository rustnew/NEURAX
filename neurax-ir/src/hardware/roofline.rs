//! Roofline model implementation

use super::{GpuProfile, Bottleneck};

/// Roofline analysis result
#[derive(Debug, Clone)]
pub struct RooflineAnalysis {
    pub arithmetic_intensity: f64,
    pub ridge_point: f64,
    pub achievable_tflops: f64,
    pub bottleneck: Bottleneck,
}

impl RooflineAnalysis {
    /// Perform roofline analysis
    pub fn analyze(
        flops: f64,
        bytes_accessed: u64,
        gpu: &GpuProfile,
        precision: &str,
    ) -> Self {
        let arithmetic_intensity = if bytes_accessed > 0 {
            flops / bytes_accessed as f64
        } else {
            0.0
        };
        
        let ridge_point = gpu.peak_tflops * 1e12 / (gpu.memory_bandwidth * 1e9);
        
        // Achievable performance is min of compute and memory roof
        let compute_roof = gpu.effective_tflops(precision);
        let memory_roof = gpu.memory_bandwidth * arithmetic_intensity / 1e3; // Convert to TFLOPS
        
        let achievable_tflops = compute_roof.min(memory_roof);
        
        let bottleneck = if arithmetic_intensity > ridge_point {
            Bottleneck::ComputeBound
        } else if arithmetic_intensity < ridge_point * 0.5 {
            Bottleneck::MemoryBound
        } else {
            Bottleneck::Balanced
        };
        
        Self {
            arithmetic_intensity,
            ridge_point,
            achievable_tflops,
            bottleneck,
        }
    }
}

/// Calculate memory bandwidth requirement
pub fn memory_bandwidth_requirement(
    bytes_accessed: u64,
    time_ms: f64,
) -> f64 {
    if time_ms <= 0.0 {
        return 0.0;
    }
    // GB/s = bytes / (time_s * 1e9)
    bytes_accessed as f64 / (time_ms / 1000.0) / 1e9
}

/// Calculate compute efficiency
pub fn compute_efficiency(
    achieved_tflops: f64,
    peak_tflops: f64,
) -> f64 {
    if peak_tflops <= 0.0 {
        return 0.0;
    }
    achieved_tflops / peak_tflops
}
