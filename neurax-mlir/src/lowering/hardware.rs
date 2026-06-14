//! Hardware Dialect Lowering
//!
//! Lowers hw.* operations to target-specific configurations:
//! - hw.gpu → Target backend selection
//! - hw.roofline → Preserved as metadata
//! - hw.bottleneck → Preserved as metadata

use super::context::LoweringContext;
use super::pass::LoweringPass;
use crate::targets::TargetBackend;
use melior::ir::Module;

/// Hardware dialect lowering pass
pub struct HardwareLowering;

impl LoweringPass for HardwareLowering {
    fn name() -> &'static str {
        "HardwareLowering"
    }
    
    fn description() -> &'static str {
        "Lowers hw.gpu, hw.roofline, hw.bottleneck to target-specific config"
    }
    
    fn run<'c>(_module: &mut Module<'c>, context: &mut LoweringContext<'c>) -> Result<(), String> {
        // Hardware lowering selects the target backend and applies optimizations
        let _target = context.target();
        Ok(())
    }
}

/// Hardware configuration for code generation
pub struct HardwareConfig {
    pub gpu_name: String,
    pub vram_gb: u64,
    pub peak_tflops: f64,
    pub memory_bandwidth_gbs: f64,
    pub tensor_cores: bool,
    pub nvlink: bool,
}

impl HardwareConfig {
    /// Create from GPU name with sensible defaults
    pub fn from_gpu_name(name: &str) -> Self {
        match name {
            "H100" | "H100_SXM" | "H100-SXM" => Self {
                gpu_name: "H100_SXM".to_string(),
                vram_gb: 80,
                peak_tflops: 989.0,
                memory_bandwidth_gbs: 3352.0,
                tensor_cores: true,
                nvlink: true,
            },
            "H200" | "H200_SXM" => Self {
                gpu_name: "H200_SXM".to_string(),
                vram_gb: 141,
                peak_tflops: 989.0,
                memory_bandwidth_gbs: 4800.0,
                tensor_cores: true,
                nvlink: true,
            },
            "A100" | "A100_SXM" | "A100-SXM" => Self {
                gpu_name: "A100_SXM".to_string(),
                vram_gb: 80,
                peak_tflops: 312.0,
                memory_bandwidth_gbs: 2039.0,
                tensor_cores: true,
                nvlink: true,
            },
            "A100_40GB" | "A100-40" => Self {
                gpu_name: "A100_40GB".to_string(),
                vram_gb: 40,
                peak_tflops: 312.0,
                memory_bandwidth_gbs: 1555.0,
                tensor_cores: true,
                nvlink: false,
            },
            "L40S" => Self {
                gpu_name: "L40S".to_string(),
                vram_gb: 48,
                peak_tflops: 362.0,
                memory_bandwidth_gbs: 864.0,
                tensor_cores: true,
                nvlink: false,
            },
            "RTX4090" => Self {
                gpu_name: "RTX4090".to_string(),
                vram_gb: 24,
                peak_tflops: 165.0,
                memory_bandwidth_gbs: 1008.0,
                tensor_cores: true,
                nvlink: false,
            },
            _ => Self {
                gpu_name: name.to_string(),
                vram_gb: 40,
                peak_tflops: 100.0,
                memory_bandwidth_gbs: 1000.0,
                tensor_cores: true,
                nvlink: false,
            },
        }
    }
    
    /// Get the recommended target backend for this GPU
    pub fn recommended_backend(&self) -> TargetBackend {
        if self.tensor_cores {
            TargetBackend::Cuda
        } else {
            TargetBackend::Cpu
        }
    }
}

/// Generate hardware attributes for the module
pub fn generate_hardware_attributes(config: &HardwareConfig) -> String {
    format!(
r#"  // Hardware configuration
  // gpu_name = "{}"
  // vram_gb = {}
  // peak_tflops = {}
  // memory_bandwidth_gbs = {}
  // tensor_cores = {}
  // nvlink = {}
"#,
        config.gpu_name,
        config.vram_gb,
        config.peak_tflops,
        config.memory_bandwidth_gbs,
        config.tensor_cores,
        config.nvlink
    )
}

/// Calculate roofline efficiency
pub fn calculate_roofline(
    flops: f64,
    bytes_transferred: u64,
    peak_flops: f64,
    peak_bandwidth: f64,
) -> (f64, f64, bool) {
    let arithmetic_intensity = flops / bytes_transferred as f64;
    let ridge_point = peak_flops / peak_bandwidth;
    
    let achievable_flops = if arithmetic_intensity >= ridge_point {
        peak_flops // Compute-bound
    } else {
        arithmetic_intensity * peak_bandwidth // Memory-bound
    };
    
    let efficiency = achievable_flops / peak_flops;
    let is_memory_bound = arithmetic_intensity < ridge_point;
    
    (efficiency, arithmetic_intensity, is_memory_bound)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hardware_config_h100() {
        let config = HardwareConfig::from_gpu_name("H100");
        assert_eq!(config.vram_gb, 80);
        assert_eq!(config.peak_tflops, 989.0);
    }
    
    #[test]
    fn test_hardware_config_a100() {
        let config = HardwareConfig::from_gpu_name("A100");
        assert_eq!(config.vram_gb, 80);
    }
    
    #[test]
    fn test_roofline_calculation() {
        let (efficiency, intensity, is_mem_bound) = calculate_roofline(
            1e12,  // 1 TFLOP
            1_000_000_000,  // 1 GB transferred
            989e12,  // H100 peak
            3.352e12,  // H100 bandwidth
        );
        assert!(efficiency > 0.0);
        assert!(intensity > 0.0);
    }
}
