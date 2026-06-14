//! Hardware IR structures

use neurax_hardware_db::GpuSpec;

/// Hardware IR - dialecte de simulation hardware
#[derive(Debug, Clone)]
pub struct HardwareIR {
    pub gpu_profile: GpuProfile,
    pub roofline: RooflineModel,
    pub per_layer_timings: Vec<LayerTiming>,
    pub metrics: HardwareMetrics,
    pub metrics_done: bool,
}

impl Default for HardwareIR {
    fn default() -> Self {
        Self {
            gpu_profile: GpuProfile::default(),
            roofline: RooflineModel::default(),
            per_layer_timings: Vec::new(),
            metrics: HardwareMetrics::default(),
            metrics_done: false,
        }
    }
}

/// GPU profile for analysis with extended specs for Industrial Roofline
#[derive(Debug, Clone)]
pub struct GpuProfile {
    pub name: String,
    pub peak_tflops: f64,
    pub memory_bandwidth: f64,
    pub tensor_core_tflops: f64,
    pub efficiency_factor: f64,
    pub vram_gb: u64,
    // ── Extended specs for Industrial Roofline ─────────────────────────
    /// TFLOPS by precision
    pub tflops_fp32: f64,
    pub tflops_fp16: f64,
    pub tflops_bf16: f64,
    pub tflops_int8: f64,
    pub tflops_fp8: f64,
    /// Tensor Core speedup vs CUDA cores
    pub tensor_core_speedup: f64,
    // ── Memory Hierarchy ───────────────────────────────────────────────
    /// HBM bandwidth (GB/s)
    pub hbm_bandwidth_gb_s: f64,
    /// HBM capacity (GB)
    pub hbm_capacity_gb: f64,
    /// L2 cache size (MB)
    pub l2_cache_mb: f64,
    /// L2 bandwidth (TB/s) - typically 5-10× HBM
    pub l2_bandwidth_tb_s: f64,
    /// SRAM per SM (KB)
    pub sram_per_sm_kb: f64,
    /// SRAM bandwidth (TB/s) - typically 100× HBM
    pub sram_bandwidth_tb_s: f64,
    // ── Compute topology ──────────────────────────────────────────────
    pub num_sms: u32,
    pub cuda_cores_per_sm: u32,
    pub tensor_cores_per_sm: u32,
    // ── Interconnect ───────────────────────────────────────────────────
    pub pcie_bandwidth_gb_s: f64,
    pub nvlink_bandwidth_gb_s: Option<f64>,
    // ── Power ───────────────────────────────────────────────────────────
    pub tdp_watts: u32,
    pub boost_clock_mhz: u32,
}

impl Default for GpuProfile {
    fn default() -> Self {
        Self {
            name: "Generic-GPU".to_string(),
            peak_tflops: 100.0,
            memory_bandwidth: 1000.0,
            tensor_core_tflops: 200.0,
            efficiency_factor: 0.6,
            vram_gb: 40,
            // Extended defaults (A100-like)
            tflops_fp32: 19.5,
            tflops_fp16: 312.0,
            tflops_bf16: 312.0,
            tflops_int8: 624.0,
            tflops_fp8: 1248.0,
            tensor_core_speedup: 16.0,
            hbm_bandwidth_gb_s: 2039.0,
            hbm_capacity_gb: 80.0,
            l2_cache_mb: 40.0,
            l2_bandwidth_tb_s: 10.0,
            sram_per_sm_kb: 164.0,
            sram_bandwidth_tb_s: 100.0,
            num_sms: 108,
            cuda_cores_per_sm: 128,
            tensor_cores_per_sm: 4,
            pcie_bandwidth_gb_s: 64.0,
            nvlink_bandwidth_gb_s: Some(600.0),
            tdp_watts: 400,
            boost_clock_mhz: 1410,
        }
    }
}

impl GpuProfile {
    /// Get effective TFLOPS for a given precision
    pub fn effective_tflops(&self, precision: &str) -> f64 {
        let base = match precision {
            "fp32" => self.tflops_fp32,
            "fp16" => self.tflops_fp16,
            "bfloat16" | "bf16" => self.tflops_bf16,
            "int8" => self.tflops_int8,
            "fp8" => self.tflops_fp8,
            _ => self.peak_tflops,
        };
        base * self.efficiency_factor
    }
    
    /// Get bandwidth for a specific cache level
    pub fn bandwidth_for_cache(&self, level: CacheLevel) -> f64 {
        match level {
            CacheLevel::Sram => self.sram_bandwidth_tb_s * 1e12,
            CacheLevel::L2 => self.l2_bandwidth_tb_s * 1e12,
            CacheLevel::Hbm => self.hbm_bandwidth_gb_s * 1e9,
        }
    }
}

impl From<&GpuSpec> for GpuProfile {
    fn from(spec: &GpuSpec) -> Self {
        Self {
            name: spec.name.clone(),
            peak_tflops: spec.tflops_fp16,
            memory_bandwidth: spec.memory_bandwidth_gbs,
            tensor_core_tflops: spec.tflops_fp16,
            efficiency_factor: spec.efficiency_factor(),
            vram_gb: spec.memory_gb,
            tflops_fp32: spec.tflops_fp32,
            tflops_fp16: spec.tflops_fp16,
            tflops_bf16: spec.tflops_bf16,
            tflops_int8: spec.tflops_int8,
            tflops_fp8: spec.tflops_fp8,
            tensor_core_speedup: 16.0,
            hbm_bandwidth_gb_s: spec.memory_bandwidth_gbs,
            hbm_capacity_gb: spec.memory_gb as f64,
            l2_cache_mb: spec.l2_cache_mb.unwrap_or(40.0),
            l2_bandwidth_tb_s: spec.memory_bandwidth_gbs as f64 * 5.0 / 1000.0,
            sram_per_sm_kb: 164.0,
            sram_bandwidth_tb_s: spec.memory_bandwidth_gbs as f64 * 50.0 / 1000.0,
            num_sms: spec.num_sms.unwrap_or(108),
            cuda_cores_per_sm: 128,
            tensor_cores_per_sm: 4,
            pcie_bandwidth_gb_s: 64.0,
            nvlink_bandwidth_gb_s: Some(spec.nvlink_bandwidth_gbs),
            tdp_watts: spec.tdp_watts as u32,
            boost_clock_mhz: 1410,
        }
    }
}

/// Cache level for Industrial Roofline
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheLevel {
    /// SRAM/L1 - fastest, smallest
    Sram,
    /// L2 cache
    L2,
    /// HBM (main GPU memory)
    Hbm,
}

impl Default for CacheLevel {
    fn default() -> Self {
        Self::Hbm
    }
}

/// Roofline model - extended to 4 levels
#[derive(Debug, Clone)]
pub struct RooflineModel {
    /// Classic roofline
    pub compute_roof: f64,
    pub memory_roof: f64,
    pub ridge_point: f64,
    // ── Extended Industrial Roofline ───────────────────────────────────
    /// Model level (1-4)
    pub level: RooflineLevel,
    /// L2 cache roof (if level >= 3)
    pub l2_roof: Option<f64>,
    /// SRAM roof (if level >= 4)
    pub sram_roof: Option<f64>,
    /// Overlap factor (compute + memory can overlap)
    pub overlap_factor: f64,
    /// Kernel launch overhead (µs)
    pub kernel_launch_overhead_us: f64,
}

impl Default for RooflineModel {
    fn default() -> Self {
        Self {
            compute_roof: 100.0,
            memory_roof: 1000.0,
            ridge_point: 100.0,
            level: RooflineLevel::Classic,
            l2_roof: None,
            sram_roof: None,
            overlap_factor: 0.0,
            kernel_launch_overhead_us: 5.0,
        }
    }
}

/// Roofline precision level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RooflineLevel {
    /// Level 1: Classic (peak_flops vs peak_BW)
    Classic,
    /// Level 2: Calibrated (efficiency per op)
    Calibrated,
    /// Level 3: Memory Hierarchy (L1/L2/HBM)
    MemoryHierarchy,
    /// Level 4: Industrial (all effects)
    Industrial,
}

impl Default for RooflineLevel {
    fn default() -> Self {
        Self::Classic
    }
}

impl RooflineModel {
    /// Create Industrial-level roofline
    pub fn industrial(gpu: &GpuProfile) -> Self {
        Self {
            compute_roof: gpu.peak_tflops,
            memory_roof: gpu.hbm_bandwidth_gb_s,
            ridge_point: gpu.peak_tflops / (gpu.hbm_bandwidth_gb_s * 1e9 / 1e12),
            level: RooflineLevel::Industrial,
            l2_roof: Some(gpu.l2_bandwidth_tb_s),
            sram_roof: Some(gpu.sram_bandwidth_tb_s),
            overlap_factor: 0.3, // 30% overlap possible
            kernel_launch_overhead_us: 5.0,
        }
    }
    
    /// Estimate latency for an operation using Industrial Roofline
    pub fn estimate_latency(
        &self,
        flops: f64,
        bytes_hbm: u64,
        bytes_l2: u64,
        cache_level: CacheLevel,
        efficiency: f64,
        gpu: &GpuProfile,
    ) -> LatencyEstimate {
        // 1. Compute time
        let peak_compute = gpu.effective_tflops("bf16") * 1e12 * efficiency;
        let t_compute = flops / peak_compute;
        
        // 2. Memory time based on cache level
        let peak_bw = gpu.bandwidth_for_cache(cache_level);
        let t_memory = match cache_level {
            CacheLevel::Sram => bytes_l2 as f64 / peak_bw,
            CacheLevel::L2 => bytes_l2 as f64 / peak_bw,
            CacheLevel::Hbm => bytes_hbm as f64 / peak_bw,
        };
        
        // 3. Overlap
        let t_effective = if self.overlap_factor > 0.0 {
            let max_t = t_compute.max(t_memory);
            let min_t = t_compute.min(t_memory);
            max_t + min_t * (1.0 - self.overlap_factor)
        } else {
            t_compute.max(t_memory)
        };
        
        // 4. Kernel overhead
        let kernel_overhead = self.kernel_launch_overhead_us * 1e-6;
        
        let total_latency_ms = (t_effective + kernel_overhead) * 1000.0;
        
        // 5. Determine bottleneck
        let arith_intensity = if bytes_hbm > 0 { flops / bytes_hbm as f64 } else { 0.0 };
        let bottleneck = if arith_intensity >= self.ridge_point {
            Bottleneck::ComputeBound
        } else {
            Bottleneck::MemoryBound
        };
        
        LatencyEstimate {
            latency_ms: total_latency_ms,
            compute_time_ms: t_compute * 1000.0,
            memory_time_ms: t_memory * 1000.0,
            bottleneck,
            cache_level,
            arithmetic_intensity: arith_intensity,
            ridge_point: self.ridge_point,
            efficiency_used: efficiency,
        }
    }
}

/// Latency estimate with breakdown
#[derive(Debug, Clone)]
pub struct LatencyEstimate {
    pub latency_ms: f64,
    pub compute_time_ms: f64,
    pub memory_time_ms: f64,
    pub bottleneck: Bottleneck,
    pub cache_level: CacheLevel,
    pub arithmetic_intensity: f64,
    pub ridge_point: f64,
    pub efficiency_used: f64,
}

/// Per-layer timing
#[derive(Debug, Clone)]
pub struct LayerTiming {
    pub layer_id: String,
    pub compute_time_ms: f64,
    pub memory_time_ms: f64,
    pub total_time_ms: f64,
}

/// Hardware metrics (Métriques 23, 29-34, 37-40, 42-43)
#[derive(Debug, Clone, Default)]
pub struct HardwareMetrics {
    /// Métrique 42: Latence par step (ms)
    pub latency_ms: f64,
    /// Métrique 43: Throughput (tokens/s)
    pub throughput_tokens_per_s: f64,
    /// Métrique 37: GPU utilization (0.0–1.0)
    pub gpu_utilization: f64,
    /// Métrique 38: Tensor core utilization
    pub tensor_core_utilization: f64,
    /// Métrique 39: Kernel launch count
    pub kernel_launch_count: usize,
    /// Métrique 40: Bottleneck type
    pub bottleneck: Bottleneck,
    /// Effective TFLOPS achieved
    pub effective_tflops: f64,
    /// Memory bandwidth achieved
    pub memory_bandwidth_achieved: f64,
    /// Samples per second
    pub samples_per_s: f64,
    /// Roofline position (0.0 = memory-bound, 1.0 = compute-bound)
    pub roofline_position: f64,
}

impl HardwareMetrics {
    pub fn is_valid(&self) -> bool {
        self.latency_ms > 0.0 && self.gpu_utilization > 0.0
    }
}

/// Bottleneck type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bottleneck {
    ComputeBound,
    MemoryBound,
    Balanced,
}

impl Bottleneck {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ComputeBound => "compute-bound",
            Self::MemoryBound => "memory-bound",
            Self::Balanced => "balanced",
        }
    }
}

impl Default for Bottleneck {
    fn default() -> Self {
        Self::Balanced
    }
}
