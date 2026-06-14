//! Hardware dialect for NEURAX
//!
//! Models GPU hardware simulation

use melior::ir::{Attribute, Identifier, Location, Operation, operation::OperationBuilder};
use melior::Context;
use super::utils::{string_attr, int_attr, float_attr};

/// Hardware dialect name
pub const DIALECT_NAME: &str = "hw";

/// Hardware dialect
pub struct HardwareDialect;

impl HardwareDialect {
    /// Get the dialect name
    pub fn name() -> &'static str {
        DIALECT_NAME
    }
    
    /// Create a GPU hardware profile operation
    pub fn gpu<'c>(
        context: &'c Context,
        name: &str,
        vram_gb: i64,
        peak_tflops: f64,
        memory_bandwidth: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("hw.gpu", location)
            .add_attributes(&[
                (Identifier::new(context, "name"), string_attr(context, name)),
                (Identifier::new(context, "vram_gb"), int_attr(context, vram_gb)),
                (Identifier::new(context, "peak_tflops"), float_attr(context, peak_tflops)),
                (Identifier::new(context, "memory_bandwidth"), float_attr(context, memory_bandwidth)),
            ])
            .build()
    }
    
    /// Create a full GPU hardware profile with all fields
    pub fn gpu_full<'c>(
        context: &'c Context,
        name: &str,
        count: i64,
        vram_gb: i64,
        peak_tflops_fp16: f64,
        peak_tflops_fp32: f64,
        memory_bandwidth: f64,
        tensor_cores: bool,
        nvlink: bool,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("hw.gpu", location)
            .add_attributes(&[
                (Identifier::new(context, "name"), string_attr(context, name)),
                (Identifier::new(context, "count"), int_attr(context, count)),
                (Identifier::new(context, "vram_gb"), int_attr(context, vram_gb)),
                (Identifier::new(context, "peak_tflops_fp16"), float_attr(context, peak_tflops_fp16)),
                (Identifier::new(context, "peak_tflops_fp32"), float_attr(context, peak_tflops_fp32)),
                (Identifier::new(context, "memory_bandwidth"), float_attr(context, memory_bandwidth)),
                (Identifier::new(context, "tensor_cores"), Attribute::parse(context, &format!("{}", tensor_cores)).unwrap()),
                (Identifier::new(context, "nvlink"), Attribute::parse(context, &format!("{}", nvlink)).unwrap()),
            ])
            .build()
    }
    
    /// Create an interconnect operation
    pub fn interconnect<'c>(
        context: &'c Context,
        interconnect_type: &str,
        bandwidth_gbs: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("hw.interconnect", location)
            .add_attributes(&[
                (Identifier::new(context, "type"), string_attr(context, interconnect_type)),
                (Identifier::new(context, "bandwidth_gbs"), float_attr(context, bandwidth_gbs)),
            ])
            .build()
    }
    
    /// Create a roofline operation
    pub fn roofline<'c>(
        context: &'c Context,
        compute_roof: f64,
        memory_roof: f64,
        ridge_point: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("hw.roofline", location)
            .add_attributes(&[
                (Identifier::new(context, "compute_roof"), float_attr(context, compute_roof)),
                (Identifier::new(context, "memory_roof"), float_attr(context, memory_roof)),
                (Identifier::new(context, "ridge_point"), float_attr(context, ridge_point)),
            ])
            .build()
    }
    
    /// Create a timing operation
    pub fn timing<'c>(
        context: &'c Context,
        layer_id: &str,
        compute_time_ms: f64,
        memory_time_ms: f64,
        total_time_ms: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("hw.timing", location)
            .add_attributes(&[
                (Identifier::new(context, "layer_id"), string_attr(context, layer_id)),
                (Identifier::new(context, "compute_time_ms"), float_attr(context, compute_time_ms)),
                (Identifier::new(context, "memory_time_ms"), float_attr(context, memory_time_ms)),
                (Identifier::new(context, "total_time_ms"), float_attr(context, total_time_ms)),
            ])
            .build()
    }
    
    /// Create a bottleneck operation
    pub fn bottleneck<'c>(
        context: &'c Context,
        bottleneck_type: &str,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("hw.bottleneck", location)
            .add_attributes(&[
                (Identifier::new(context, "bottleneck_type"), string_attr(context, bottleneck_type)),
            ])
            .build()
    }
    
    /// Create metrics operation
    pub fn metrics<'c>(
        context: &'c Context,
        latency_ms: f64,
        throughput_tokens_per_s: f64,
        gpu_utilization: f64,
        tensor_core_utilization: f64,
        effective_tflops: f64,
        memory_bandwidth_achieved: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("hw.metrics", location)
            .add_attributes(&[
                (Identifier::new(context, "latency_ms"), float_attr(context, latency_ms)),
                (Identifier::new(context, "throughput_tokens_per_s"), float_attr(context, throughput_tokens_per_s)),
                (Identifier::new(context, "gpu_utilization"), float_attr(context, gpu_utilization)),
                (Identifier::new(context, "tensor_core_utilization"), float_attr(context, tensor_core_utilization)),
                (Identifier::new(context, "effective_tflops"), float_attr(context, effective_tflops)),
                (Identifier::new(context, "memory_bandwidth_achieved"), float_attr(context, memory_bandwidth_achieved)),
            ])
            .build()
    }
}
