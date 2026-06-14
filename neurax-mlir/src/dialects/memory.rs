//! Memory dialect for NEURAX
//!
//! Models memory simulation (liveness, VRAM)

use melior::ir::{Identifier, Location, Operation, operation::OperationBuilder};
use melior::Context;
use super::utils::{string_attr, int_attr, float_attr};

/// Memory dialect name
pub const DIALECT_NAME: &str = "mem";

/// Memory dialect
pub struct MemoryDialect;

impl MemoryDialect {
    /// Get the dialect name
    pub fn name() -> &'static str {
        DIALECT_NAME
    }
    
    /// Create a liveness operation
    pub fn liveness<'c>(
        context: &'c Context,
        tensor_id: &str,
        start_step: i64,
        end_step: i64,
        size_bytes: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("mem.liveness", location)
            .add_attributes(&[
                (Identifier::new(context, "tensor_id"), string_attr(context, tensor_id)),
                (Identifier::new(context, "start_step"), int_attr(context, start_step)),
                (Identifier::new(context, "end_step"), int_attr(context, end_step)),
                (Identifier::new(context, "size_bytes"), int_attr(context, size_bytes)),
            ])
            .build()
    }
    
    /// Create an alloc operation
    pub fn alloc<'c>(
        context: &'c Context,
        size_bytes: i64,
        tensor_id: &str,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("mem.alloc", location)
            .add_attributes(&[
                (Identifier::new(context, "size_bytes"), int_attr(context, size_bytes)),
                (Identifier::new(context, "tensor_id"), string_attr(context, tensor_id)),
            ])
            .build()
    }
    
    /// Create a peak operation
    pub fn peak<'c>(
        context: &'c Context,
        peak_bytes: i64,
        peak_step: Option<i64>,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        let mut attrs = vec![
            (Identifier::new(context, "peak_bytes"), int_attr(context, peak_bytes)),
        ];
        if let Some(step) = peak_step {
            attrs.push((Identifier::new(context, "peak_step"), int_attr(context, step)));
        }
        OperationBuilder::new("mem.peak", location)
            .add_attributes(&attrs)
            .build()
    }
    
    /// Create an OOM risk operation
    pub fn oom_risk<'c>(
        context: &'c Context,
        risk_level: &str,
        utilization_ratio: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("mem.oom_risk", location)
            .add_attributes(&[
                (Identifier::new(context, "risk_level"), string_attr(context, risk_level)),
                (Identifier::new(context, "utilization_ratio"), float_attr(context, utilization_ratio)),
            ])
            .build()
    }
    
    /// Create metrics operation
    pub fn metrics<'c>(
        context: &'c Context,
        parameter_memory_bytes: i64,
        activation_memory_bytes: i64,
        gradient_memory_bytes: i64,
        optimizer_state_bytes: i64,
        peak_vram_bytes: i64,
        max_batch_size_fit: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("mem.metrics", location)
            .add_attributes(&[
                (Identifier::new(context, "parameter_memory_bytes"), int_attr(context, parameter_memory_bytes)),
                (Identifier::new(context, "activation_memory_bytes"), int_attr(context, activation_memory_bytes)),
                (Identifier::new(context, "gradient_memory_bytes"), int_attr(context, gradient_memory_bytes)),
                (Identifier::new(context, "optimizer_state_bytes"), int_attr(context, optimizer_state_bytes)),
                (Identifier::new(context, "peak_vram_bytes"), int_attr(context, peak_vram_bytes)),
                (Identifier::new(context, "max_batch_size_fit"), int_attr(context, max_batch_size_fit)),
            ])
            .build()
    }
    
    /// Create a full memory breakdown operation
    pub fn breakdown<'c>(
        context: &'c Context,
        weights_memory: i64,
        optimizer_states_memory: i64,
        gradients_memory: i64,
        activations_memory: i64,
        temporary_memory: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("mem.breakdown", location)
            .add_attributes(&[
                (Identifier::new(context, "weights_memory"), int_attr(context, weights_memory)),
                (Identifier::new(context, "optimizer_states_memory"), int_attr(context, optimizer_states_memory)),
                (Identifier::new(context, "gradients_memory"), int_attr(context, gradients_memory)),
                (Identifier::new(context, "activations_memory"), int_attr(context, activations_memory)),
                (Identifier::new(context, "temporary_memory"), int_attr(context, temporary_memory)),
            ])
            .build()
    }
    
    /// Create a memory hierarchy operation
    pub fn hierarchy<'c>(
        context: &'c Context,
        hbm_size_gb: f64,
        ddr_size_gb: f64,
        nvme_size_gb: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("mem.hierarchy", location)
            .add_attributes(&[
                (Identifier::new(context, "hbm_size_gb"), float_attr(context, hbm_size_gb)),
                (Identifier::new(context, "ddr_size_gb"), float_attr(context, ddr_size_gb)),
                (Identifier::new(context, "nvme_size_gb"), float_attr(context, nvme_size_gb)),
            ])
            .build()
    }
    
    /// Create a memory offload operation
    pub fn offload<'c>(
        context: &'c Context,
        source: &str,
        destination: &str,
        size_bytes: i64,
        bandwidth_gbs: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("mem.offload", location)
            .add_attributes(&[
                (Identifier::new(context, "source"), string_attr(context, source)),
                (Identifier::new(context, "destination"), string_attr(context, destination)),
                (Identifier::new(context, "size_bytes"), int_attr(context, size_bytes)),
                (Identifier::new(context, "bandwidth_gbs"), float_attr(context, bandwidth_gbs)),
            ])
            .build()
    }
    
    /// Create a memory efficiency operation
    pub fn efficiency<'c>(
        context: &'c Context,
        utilization: f64,
        fragmentation: f64,
        bandwidth_utilization: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("mem.efficiency", location)
            .add_attributes(&[
                (Identifier::new(context, "utilization"), float_attr(context, utilization)),
                (Identifier::new(context, "fragmentation"), float_attr(context, fragmentation)),
                (Identifier::new(context, "bandwidth_utilization"), float_attr(context, bandwidth_utilization)),
            ])
            .build()
    }
}
