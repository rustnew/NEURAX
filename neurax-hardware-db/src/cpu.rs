//! CPU specifications

use serde::{Deserialize, Serialize};

/// CPU specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuSpec {
    /// CPU name
    pub name: String,
    /// Manufacturer
    pub manufacturer: String,
    /// Number of physical cores
    pub cores: u32,
    /// Number of threads (with hyperthreading)
    pub threads: u32,
    /// Base frequency in GHz
    pub base_freq_ghz: f64,
    /// Boost frequency in GHz
    pub boost_freq_ghz: f64,
    /// Thermal Design Power in watts
    pub tdp_watts: u64,
    /// Number of memory channels
    pub memory_channels: u32,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gbs: f64,
}

impl CpuSpec {
    /// Create a generic CPU spec as fallback
    pub fn generic() -> Self {
        Self {
            name: "Generic-CPU".to_string(),
            manufacturer: "Unknown".to_string(),
            cores: 32,
            threads: 64,
            base_freq_ghz: 2.5,
            boost_freq_ghz: 3.5,
            tdp_watts: 200,
            memory_channels: 8,
            memory_bandwidth_gbs: 200.0,
        }
    }
    
    /// Estimate GFLOPS for CPU
    pub fn gflops_estimate(&self) -> f64 {
        // AVX-512 can do 32 FLOPs per cycle per core
        // AVX2 can do 16 FLOPs per cycle per core
        let flops_per_cycle = 16.0; // Conservative estimate for AVX2
        self.cores as f64 * self.base_freq_ghz * 1e9 * flops_per_cycle / 1e9
    }
}
