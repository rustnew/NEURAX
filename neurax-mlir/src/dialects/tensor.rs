//! Tensor dialect for NEURAX
//!
//! Models tensor shapes and propagation

use melior::ir::{Identifier, Location, Operation, operation::OperationBuilder};
use melior::Context;
use super::utils::{string_attr, int_attr, float_attr, int_array_attr};

/// Tensor dialect name
pub const DIALECT_NAME: &str = "tensor";

/// Tensor dialect
pub struct TensorDialect;

impl TensorDialect {
    /// Get the dialect name
    pub fn name() -> &'static str {
        DIALECT_NAME
    }
    
    /// Create a tensor info operation
    pub fn tensor_info<'c>(
        context: &'c Context,
        tensor_id: &str,
        shape: &[i64],
        dtype: &str,
        size_bytes: i64,
        produced_by: &str,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("tensor.info", location)
            .add_attributes(&[
                (Identifier::new(context, "tensor_id"), string_attr(context, tensor_id)),
                (Identifier::new(context, "shape"), int_array_attr(context, shape)),
                (Identifier::new(context, "dtype"), string_attr(context, dtype)),
                (Identifier::new(context, "size_bytes"), int_attr(context, size_bytes)),
                (Identifier::new(context, "produced_by"), string_attr(context, produced_by)),
            ])
            .build()
    }
    
    /// Create metrics operation
    pub fn metrics<'c>(
        context: &'c Context,
        activation_memory_bytes: i64,
        memory_bandwidth_required: f64,
        total_tensor_count: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("tensor.metrics", location)
            .add_attributes(&[
                (Identifier::new(context, "activation_memory_bytes"), int_attr(context, activation_memory_bytes)),
                (Identifier::new(context, "memory_bandwidth_required"), float_attr(context, memory_bandwidth_required)),
                (Identifier::new(context, "total_tensor_count"), int_attr(context, total_tensor_count)),
            ])
            .build()
    }
}
