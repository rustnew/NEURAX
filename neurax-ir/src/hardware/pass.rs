//! Hardware IR pass

use crate::traits::IrPass;
use crate::error::HardwareError;
use crate::NeuraxContext;
use crate::compute::ComputeIR;
use crate::memory::MemoryIR;
use crate::parallelism::ParallelismIR;
use super::{HardwareIR, GpuProfile, RooflineModel, HardwareMetrics, Bottleneck, LayerTiming};

/// Hardware pass implementation
pub struct HardwarePass;

impl IrPass for HardwarePass {
    type Input = (ComputeIR, MemoryIR, ParallelismIR);
    type Output = HardwareIR;
    type Metrics = HardwareMetrics;
    type PassError = HardwareError;

    fn name(&self) -> &'static str {
        "HardwareIR"
    }

    fn build(&self, input: &Self::Input, ctx: &NeuraxContext) -> Result<Self::Output, Self::PassError> {
        let (compute_ir, _memory_ir, _parallel_ir) = input;
        let mut hw_ir = HardwareIR::default();
        
        // Get GPU profile from JSON config or fallback to database
        let gpu_config = ctx.config.hardware.gpus.first();
        let gpu_name = gpu_config.map(|g| g.name.as_str()).unwrap_or("Generic-GPU");
        
        // Try to get base profile from database, then override with JSON values
        let mut gpu_profile = ctx.gpu_db.get_gpu(gpu_name)
            .map(GpuProfile::from)
            .unwrap_or_else(|| {
                // GPU not in database - create profile from JSON config
                let mut profile = GpuProfile::default();
                if let Some(gpu) = gpu_config {
                    profile.name = gpu.name.clone();
                    profile.vram_gb = gpu.memory_gb as u64;
                    profile.peak_tflops = if gpu.tflops_fp16 > 0.0 { gpu.tflops_fp16 } else { gpu.tflops_fp32 };
                    profile.memory_bandwidth = gpu.memory_bandwidth_gbs;
                }
                profile
            });
        
        // Override with values from JSON config if provided
        if let Some(gpu) = gpu_config {
            // Use tflops from JSON based on precision
            let precision = &ctx.config.training.precision;
            if precision == "fp8" {
                if gpu.tflops_fp8 > 0.0 {
                    gpu_profile.peak_tflops = gpu.tflops_fp8;
                } else if gpu.tflops_fp16 > 0.0 {
                    gpu_profile.peak_tflops = gpu.tflops_fp16 * 2.0; // fp8 is 2x fp16
                }
            } else if precision == "fp16" || precision == "bfloat16" {
                if gpu.tflops_fp16 > 0.0 {
                    gpu_profile.peak_tflops = gpu.tflops_fp16;
                }
            } else {
                if gpu.tflops_fp32 > 0.0 {
                    gpu_profile.peak_tflops = gpu.tflops_fp32;
                }
            }
            
            // Use memory bandwidth from JSON if specified
            if gpu.memory_bandwidth_gbs > 0.0 {
                gpu_profile.memory_bandwidth = gpu.memory_bandwidth_gbs;
            }
            
            // Tensor cores enable higher TFLOPs for FP16/BF16
            if gpu.tensor_cores && (precision == "fp16" || precision == "bfloat16") {
                gpu_profile.tensor_core_tflops = gpu_profile.peak_tflops * 2.0;
            }
        }
        
        hw_ir.gpu_profile = gpu_profile;
        
        // Build roofline model (Industrial level)
        hw_ir.roofline = RooflineModel {
            compute_roof: hw_ir.gpu_profile.peak_tflops * hw_ir.gpu_profile.efficiency_factor,
            memory_roof: hw_ir.gpu_profile.memory_bandwidth,
            ridge_point: calculate_ridge_point(&hw_ir.gpu_profile, &ctx.config.training.precision),
            level: crate::hardware::RooflineLevel::Industrial,
            l2_roof: Some(hw_ir.gpu_profile.l2_bandwidth_tb_s),
            sram_roof: Some(hw_ir.gpu_profile.sram_bandwidth_tb_s),
            overlap_factor: 0.3,
            kernel_launch_overhead_us: 5.0,
        };
        
        // Calculate per-layer timings
        hw_ir.per_layer_timings = calculate_layer_timings(&compute_ir, &hw_ir.gpu_profile);
        
        Ok(hw_ir)
    }

    fn compute_metrics(&self, output: &mut Self::Output, ctx: &NeuraxContext) -> Result<Self::Metrics, Self::PassError> {
        let batch = ctx.config.training.batch_size;
        let seq = ctx.config.model.global_params.sequence_length.unwrap_or(512);
        let precision = &ctx.config.training.precision;
        
        // Count attention vs MLP layers for efficiency estimation
        let attention_count = ctx.config.model.layers.iter()
            .filter(|l| l.layer_type == neurax_parser::LayerType::Attention)
            .count();
        let mlp_count = ctx.config.model.layers.iter()
            .filter(|l| l.layer_type == neurax_parser::LayerType::Mlp)
            .count();
        let total_layers = (attention_count + mlp_count).max(1);
        
        // Real-world GPU efficiency factors based on operation type
        // Attention is memory-bound (lower efficiency)
        // MLP is compute-bound (higher efficiency)
        let attention_efficiency = match precision.as_str() {
            "fp32" => 0.45,
            "fp16" | "bfloat16" => 0.55,  // FlashAttention helps
            "fp8" => 0.65,
            _ => 0.45,
        };
        let mlp_efficiency = match precision.as_str() {
            "fp32" => 0.75,
            "fp16" | "bfloat16" => 0.85,  // Tensor cores shine
            "fp8" => 0.92,
            _ => 0.75,
        };
        
        // Weighted average efficiency based on layer distribution
        let gpu_efficiency = (attention_count as f64 * attention_efficiency + 
                             mlp_count as f64 * mlp_efficiency) / total_layers as f64;
        
        // FlashAttention reduces memory by ~4x
        let flash_attention_enabled = true; // Could be a config option
        let attention_memory_factor = if flash_attention_enabled { 0.25 } else { 1.0 };
        
        // Compute time with efficiency factor
        let compute_time_ms: f64 = output.per_layer_timings.iter()
            .map(|t| t.compute_time_ms / gpu_efficiency)
            .sum();
        
        // Memory time with FlashAttention optimization
        let memory_time_ms: f64 = output.per_layer_timings.iter()
            .map(|t| t.memory_time_ms * attention_memory_factor)
            .sum();
        
        // Total FLOPs from compute IR with efficiency factor
        let total_flops = output.per_layer_timings.iter()
            .map(|t| t.compute_time_ms * output.gpu_profile.effective_tflops(precision) * gpu_efficiency * 1e9 / 1000.0)
            .sum::<f64>();
        
        // Arithmetic intensity (FLOPs/byte)
        let arithmetic_intensity = if memory_time_ms > 0.0 && output.gpu_profile.memory_bandwidth > 0.0 {
            let bytes = memory_time_ms * output.gpu_profile.memory_bandwidth * 1e6;
            total_flops / bytes.max(1.0)
        } else {
            0.0
        };
        
        // Communication overhead for multi-GPU
        let num_gpus = ctx.config.hardware.total_gpu_count();
        let interconnect_bw = ctx.config.hardware.interconnect_bandwidth_gbs * 1e9; // bytes/s
        let param_bytes = output.per_layer_timings.iter()
            .map(|t| t.memory_time_ms * output.gpu_profile.memory_bandwidth * 1e6)
            .sum::<f64>() as u64;
        
        // AllReduce overhead: 2 * (N-1)/N * params / bandwidth
        let communication_overhead_ms = if num_gpus > 1 && interconnect_bw > 0.0 {
            let factor = 2.0 * (num_gpus - 1) as f64 / num_gpus as f64;
            (param_bytes as f64 * factor / interconnect_bw) * 1000.0
        } else {
            0.0
        };
        
        // Total latency including communication
        // Divide by total GPU count for parallel execution
        let num_gpus_f64 = num_gpus.max(1) as f64;
        let parallel_compute_time_ms = compute_time_ms / num_gpus_f64;
        let parallel_memory_time_ms = memory_time_ms / num_gpus_f64;
        let latency_ms = parallel_compute_time_ms.max(parallel_memory_time_ms) + communication_overhead_ms;
        
        // Determine bottleneck
        let bottleneck = if compute_time_ms > memory_time_ms * 1.5 {
            Bottleneck::ComputeBound
        } else if memory_time_ms > compute_time_ms * 1.5 {
            Bottleneck::MemoryBound
        } else {
            Bottleneck::Balanced
        };
        
        // GPU utilization
        let gpu_utilization = if latency_ms > 0.0 {
            compute_time_ms / latency_ms
        } else {
            0.0
        };
        
        // Throughput
        let samples_per_s = if latency_ms > 0.0 {
            1000.0 / latency_ms * batch as f64
        } else {
            0.0
        };
        
        let throughput_tokens_per_s = samples_per_s * seq as f64;
        
        // Tensor core utilization
        let tensor_core_utilization = calculate_tensor_core_utilization(ctx);
        
        // Kernel launches (estimate)
        let kernel_launch_count = ctx.config.model.layers.len() * 2;
        
        // Effective TFLOPS
        let effective_tflops = if latency_ms > 0.0 && batch > 0 && seq > 0 {
            let flops_per_step = compute_ir_metrics(ctx).total_flops;
            flops_per_step / latency_ms / 1000.0 / 1e9
        } else {
            0.0
        };
        
        let metrics = HardwareMetrics {
            latency_ms,
            throughput_tokens_per_s,
            gpu_utilization,
            tensor_core_utilization,
            kernel_launch_count,
            bottleneck,
            effective_tflops,
            memory_bandwidth_achieved: output.gpu_profile.memory_bandwidth * gpu_utilization,
            samples_per_s,
            roofline_position: calculate_roofline_position(&output.roofline, arithmetic_intensity),
        };
        
        output.metrics = metrics.clone();
        output.metrics_done = true;
        Ok(metrics)
    }

    fn validate(&self, _output: &Self::Output, metrics: &Self::Metrics) -> Result<(), Self::PassError> {
        if metrics.latency_ms <= 0.0 {
            return Err(HardwareError::InvalidLatency("Latency is zero or negative".to_string()));
        }
        if metrics.gpu_utilization <= 0.0 {
            return Err(HardwareError::RooflineFailed("GPU utilization is zero".to_string()));
        }
        Ok(())
    }
}

fn calculate_ridge_point(gpu: &GpuProfile, _precision: &str) -> f64 {
    let tflops = gpu.peak_tflops * gpu.efficiency_factor;
    let bandwidth = gpu.memory_bandwidth;
    // Ridge point = TFLOPS / Bandwidth (FLOPs/byte)
    tflops * 1e12 / (bandwidth * 1e9)
}

fn calculate_layer_timings(compute_ir: &crate::compute::ComputeIR, gpu: &GpuProfile) -> Vec<LayerTiming> {
    compute_ir.op_flops.iter()
        .map(|op| {
            // Compute time: FLOPs / (TFLOPs * efficiency * 1e9) * 1000 (ms)
            let compute_time_ms = op.forward_flops / (gpu.peak_tflops * gpu.efficiency_factor * 1e9) * 1000.0;
            // Simplified memory time
            let memory_time_ms = compute_time_ms * 0.5;
            
            LayerTiming {
                layer_id: op.layer_id.clone(),
                compute_time_ms,
                memory_time_ms,
                total_time_ms: compute_time_ms.max(memory_time_ms),
            }
        })
        .collect()
}

fn calculate_tensor_core_utilization(ctx: &NeuraxContext) -> f64 {
    // Estimate based on layer types
    let total_layers = ctx.config.model.layers.len();
    if total_layers == 0 {
        return 0.0;
    }
    
    let tc_layers = ctx.config.model.layers.iter()
        .filter(|l| matches!(l.layer_type, 
            neurax_parser::LayerType::Attention | 
            neurax_parser::LayerType::Mlp | 
            neurax_parser::LayerType::Dense |
            neurax_parser::LayerType::Conv
        ))
        .count();
    
    tc_layers as f64 / total_layers as f64
}

fn compute_ir_metrics(_ctx: &NeuraxContext) -> crate::compute::ComputeMetrics {
    // Simplified - would normally come from actual ComputeIR
    crate::compute::ComputeMetrics::default()
}

/// Calculate roofline position (0.0 = memory-bound, 1.0 = compute-bound)
fn calculate_roofline_position(roofline: &crate::hardware::RooflineModel, arithmetic_intensity: f64) -> f64 {
    if roofline.ridge_point <= 0.0 {
        return 0.5;
    }
    // Position relative to ridge point
    // < 1.0 = memory-bound, > 1.0 = compute-bound
    let position = arithmetic_intensity / roofline.ridge_point;
    // Clamp to [0.0, 1.0] range
    position.clamp(0.0, 1.0)
}
