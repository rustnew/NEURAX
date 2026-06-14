//! Interconnect specifications

use serde::{Deserialize, Serialize};

/// Interconnect specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterconnectSpec {
    /// Interconnect name
    pub name: String,
    /// Bandwidth in GB/s (per direction)
    pub bandwidth_gbs: f64,
    /// Latency in nanoseconds
    pub latency_ns: f64,
}

impl InterconnectSpec {
    /// Create a generic interconnect spec
    pub fn generic() -> Self {
        Self {
            name: "Generic-Interconnect".to_string(),
            bandwidth_gbs: 64.0,
            latency_ns: 1000.0,
        }
    }
    
    /// Calculate transfer time for given bytes
    pub fn transfer_time_ms(&self, bytes: u64) -> f64 {
        bytes as f64 / (self.bandwidth_gbs * 1e9) * 1000.0 + self.latency_ns / 1e6
    }
    
    /// Calculate All-Reduce time for N GPUs
    /// Ring All-Reduce: 2 × (N-1) / N × data_size / bandwidth
    pub fn allreduce_time_ms(&self, bytes: u64, num_gpus: u32) -> f64 {
        if num_gpus <= 1 {
            return 0.0;
        }
        let factor = 2.0 * (num_gpus - 1) as f64 / num_gpus as f64;
        self.transfer_time_ms(bytes) * factor
    }
}
