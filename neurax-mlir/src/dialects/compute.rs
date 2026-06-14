//! Compute dialect for NEURAX
//!
//! Models analytical computation (FLOPs, intensity)

use melior::ir::{Identifier, Location, Operation, operation::OperationBuilder};
use melior::Context;
use super::utils::{string_attr, int_attr, float_attr};

/// Compute dialect name
pub const DIALECT_NAME: &str = "compute";

/// Compute dialect
pub struct ComputeDialect;

impl ComputeDialect {
    /// Get the dialect name
    pub fn name() -> &'static str {
        DIALECT_NAME
    }
    
    /// Create a FLOPs operation
    pub fn flops<'c>(
        context: &'c Context,
        forward_flops: f64,
        backward_flops: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("compute.flops", location)
            .add_attributes(&[
                (Identifier::new(context, "forward_flops"), float_attr(context, forward_flops)),
                (Identifier::new(context, "backward_flops"), float_attr(context, backward_flops)),
            ])
            .build()
    }
    
    /// Create an intensity operation
    pub fn intensity<'c>(
        context: &'c Context,
        flops: f64,
        bytes_accessed: i64,
        intensity: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("compute.intensity", location)
            .add_attributes(&[
                (Identifier::new(context, "flops"), float_attr(context, flops)),
                (Identifier::new(context, "bytes_accessed"), int_attr(context, bytes_accessed)),
                (Identifier::new(context, "intensity"), float_attr(context, intensity)),
            ])
            .build()
    }
    
    /// Create a complexity operation
    pub fn complexity<'c>(
        context: &'c Context,
        complexity_class: &str,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("compute.complexity", location)
            .add_attributes(&[
                (Identifier::new(context, "complexity_class"), string_attr(context, complexity_class)),
            ])
            .build()
    }
    
    /// Create metrics operation
    pub fn metrics<'c>(
        context: &'c Context,
        total_flops: f64,
        macs: f64,
        flops_per_token: f64,
        arithmetic_intensity: f64,
        forward_flops: f64,
        backward_flops: f64,
        optimizer_flops: f64,
        total_step_flops: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("compute.metrics", location)
            .add_attributes(&[
                (Identifier::new(context, "total_flops"), float_attr(context, total_flops)),
                (Identifier::new(context, "macs"), float_attr(context, macs)),
                (Identifier::new(context, "flops_per_token"), float_attr(context, flops_per_token)),
                (Identifier::new(context, "arithmetic_intensity"), float_attr(context, arithmetic_intensity)),
                (Identifier::new(context, "forward_flops"), float_attr(context, forward_flops)),
                (Identifier::new(context, "backward_flops"), float_attr(context, backward_flops)),
                (Identifier::new(context, "optimizer_flops"), float_attr(context, optimizer_flops)),
                (Identifier::new(context, "total_step_flops"), float_attr(context, total_step_flops)),
            ])
            .build()
    }
}
