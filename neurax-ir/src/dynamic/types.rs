//! Dynamic System Types
//! 
//! Common types used across the dynamic analysis passes.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::virtual_memory::VirtualMemoryMetrics;
use super::stability::StabilityMetrics;
use super::behavioral::BehavioralMetrics;

/// Results from all dynamic passes
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DynamicResults {
    /// Virtual memory analysis results (M36-M42)
    pub virtual_memory: Option<VirtualMemoryMetrics>,
    /// Stability analysis results (M43-M49)
    pub stability: Option<StabilityMetrics>,
    /// Behavioral synthesis results (M50-M55)
    pub behavioral: Option<BehavioralMetrics>,
}

/// Configuration for dynamic analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicConfig {
    /// Enable virtual memory analysis (default: true, cost: ~5ms)
    pub enable_virtual_memory: bool,
    /// Enable stability analysis (default: true, cost: ~10ms)
    pub enable_stability: bool,
    /// Enable behavioral synthesis (default: false, cost: ~50ms with model)
    pub enable_behavioral: bool,
    /// Risk threshold for stability warnings (default: 0.2)
    pub stability_risk_threshold: f64,
    /// Fragmentation warning threshold in % (default: 15.0)
    pub fragmentation_warning_threshold: f64,
    /// Path to BPS model (optional, for V2)
    pub bps_model_path: Option<String>,
}

impl Default for DynamicConfig {
    fn default() -> Self {
        Self {
            enable_virtual_memory: true,
            enable_stability: true,
            enable_behavioral: false,
            stability_risk_threshold: 0.2,
            fragmentation_warning_threshold: 15.0,
            bps_model_path: None,
        }
    }
}

/// Calibration data for dynamic analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CalibrationData {
    /// Fragmentation measurements by model size
    pub fragmentation_by_size: HashMap<String, f64>,
    /// GPU efficiency factors
    pub gpu_efficiency_factors: HashMap<String, f64>,
}

impl CalibrationData {
    /// Check if fragmentation calibration data is available
    pub fn has_fragmentation_data(&self) -> bool {
        !self.fragmentation_by_size.is_empty()
    }
    
    /// Get calibrated fragmentation for a given peak VRAM and activation ratio
    pub fn get_fragmentation(&self, peak_vram_gb: f64, activation_ratio: f64) -> Option<f64> {
        // Find closest match in calibration data
        let key = format!("{:.1}_{:.2}", peak_vram_gb, activation_ratio);
        self.fragmentation_by_size.get(&key).copied()
    }
}
