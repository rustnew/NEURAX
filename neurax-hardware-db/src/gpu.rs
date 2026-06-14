//! GPU specifications

use serde::{Deserialize, Serialize};

/// GPU specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuSpec {
    /// GPU name (e.g., "A100-SXM")
    pub name: String,
    /// Manufacturer
    pub manufacturer: String,
    /// VRAM in GB
    pub memory_gb: u64,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gbs: f64,
    /// TFLOPS for FP64
    pub tflops_fp64: f64,
    /// TFLOPS for FP32
    pub tflops_fp32: f64,
    /// TFLOPS for FP16 (with Tensor Cores)
    pub tflops_fp16: f64,
    /// TFLOPS for BF16
    pub tflops_bf16: f64,
    /// TFLOPS for INT8
    pub tflops_int8: f64,
    /// TFLOPS for FP8 (Hopper+)
    #[serde(default)]
    pub tflops_fp8: f64,
    /// Has Tensor Cores
    pub tensor_cores: bool,
    /// Supports NVLink
    pub nvlink: bool,
    /// NVLink bandwidth per direction in GB/s
    pub nvlink_bandwidth_gbs: f64,
    /// Thermal Design Power in watts
    pub tdp_watts: u64,
    /// Launch year
    pub launch_year: u32,
    // ── Extended specs for Industrial Roofline ─────────────────────────
    /// L2 cache size in MB
    #[serde(default)]
    pub l2_cache_mb: Option<f64>,
    /// Number of Streaming Multiprocessors
    #[serde(default)]
    pub num_sms: Option<u32>,
}

impl GpuSpec {
    /// Create a generic GPU spec as fallback
    pub fn generic() -> Self {
        Self {
            name: "Generic-GPU".to_string(),
            manufacturer: "Unknown".to_string(),
            memory_gb: 16,
            memory_bandwidth_gbs: 500.0,
            tflops_fp64: 5.0,
            tflops_fp32: 10.0,
            tflops_fp16: 50.0,
            tflops_bf16: 50.0,
            tflops_int8: 100.0,
            tflops_fp8: 200.0,
            tensor_cores: false,
            nvlink: false,
            nvlink_bandwidth_gbs: 0.0,
            tdp_watts: 250,
            launch_year: 2020,
            l2_cache_mb: None,
            num_sms: None,
        }
    }
    
    /// Get TFLOPS for a given precision
    pub fn tflops_for_precision(&self, precision: &str) -> f64 {
        match precision {
            "fp64" | "float64" => self.tflops_fp64,
            "fp32" | "float32" => self.tflops_fp32,
            "fp16" | "float16" => self.tflops_fp16,
            "bf16" | "bfloat16" => self.tflops_bf16,
            "int8" => self.tflops_int8,
            _ => self.tflops_fp32,
        }
    }
    
    /// Get efficiency factor (real-world vs theoretical performance)
    pub fn efficiency_factor(&self) -> f64 {
        // Typical efficiency is 40-70% of theoretical peak
        // Newer GPUs with better software stack tend to achieve higher
        match self.launch_year {
            2022.. => 0.65,
            2020..=2021 => 0.60,
            2018..=2019 => 0.55,
            _ => 0.50,
        }
    }
    
    /// Get effective TFLOPS (theoretical × efficiency)
    pub fn effective_tflops(&self, precision: &str) -> f64 {
        self.tflops_for_precision(precision) * self.efficiency_factor()
    }
    
    /// Calculate ridge point (FLOPs/byte where compute = memory bound)
    pub fn ridge_point(&self, precision: &str) -> f64 {
        let tflops = self.tflops_for_precision(precision);
        let bandwidth_gbs = self.memory_bandwidth_gbs;
        // Ridge point = TFLOPS / Bandwidth (in FLOPs/byte)
        tflops * 1e12 / (bandwidth_gbs * 1e9)
    }
    
    /// Estimate compute time for given FLOPs
    pub fn compute_time_ms(&self, flops: f64, precision: &str) -> f64 {
        let effective_tflops = self.effective_tflops(precision);
        flops / (effective_tflops * 1e12) * 1000.0
    }
    
    /// Estimate memory time for given bytes
    pub fn memory_time_ms(&self, bytes: u64) -> f64 {
        bytes as f64 / (self.memory_bandwidth_gbs * 1e9) * 1000.0
    }
    
    /// Check if model fits in memory
    pub fn fits_in_memory(&self, required_bytes: u64) -> bool {
        required_bytes <= self.memory_gb * 1024 * 1024 * 1024
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generic_gpu() {
        let gpu = GpuSpec::generic();
        assert!(gpu.memory_gb > 0);
        assert!(gpu.tflops_fp32 > 0.0);
    }

    #[test]
    fn test_ridge_point() {
        let gpu = GpuSpec {
            name: "Test".to_string(),
            manufacturer: "NVIDIA".to_string(),
            memory_gb: 40,
            memory_bandwidth_gbs: 1000.0,
            tflops_fp64: 10.0,
            tflops_fp32: 20.0,
            tflops_fp16: 100.0,
            tflops_bf16: 100.0,
            tflops_int8: 200.0,
            tflops_fp8: 0.0,
            tensor_cores: true,
            nvlink: false,
            nvlink_bandwidth_gbs: 0.0,
            tdp_watts: 250,
            launch_year: 2020,
            l2_cache_mb: None,
            num_sms: None,
        };
        
        let ridge = gpu.ridge_point("fp16");
        // 100 TFLOPS / 1000 GB/s = 100 FLOPs/byte
        assert!((ridge - 100.0).abs() < 1.0);
    }
}
