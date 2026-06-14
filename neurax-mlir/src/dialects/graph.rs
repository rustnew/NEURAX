//! Graph dialect for NEURAX
//!
//! Models computation graphs as DAGs

use melior::ir::{Identifier, Location, Operation, operation::OperationBuilder};
use melior::Context;
use super::utils::{string_attr, int_attr, float_attr, int_array_attr};

/// Graph dialect name
pub const DIALECT_NAME: &str = "graph";

/// Graph dialect
pub struct GraphDialect;

impl GraphDialect {
    /// Get the dialect name
    pub fn name() -> &'static str {
        DIALECT_NAME
    }
    
    /// Create a node operation
    pub fn node<'c>(
        context: &'c Context,
        layer_id: &str,
        layer_type: &str,
        flops_approx: f64,
        param_count: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("graph.node", location)
            .add_attributes(&[
                (Identifier::new(context, "layer_id"), string_attr(context, layer_id)),
                (Identifier::new(context, "layer_type"), string_attr(context, layer_type)),
                (Identifier::new(context, "flops_approx"), float_attr(context, flops_approx)),
                (Identifier::new(context, "param_count"), int_attr(context, param_count)),
            ])
            .build()
    }
    
    /// Create an edge operation
    pub fn edge<'c>(
        context: &'c Context,
        tensor_shape: &[i64],
        dtype: &str,
        size_bytes: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("graph.edge", location)
            .add_attributes(&[
                (Identifier::new(context, "tensor_shape"), int_array_attr(context, tensor_shape)),
                (Identifier::new(context, "dtype"), string_attr(context, dtype)),
                (Identifier::new(context, "size_bytes"), int_attr(context, size_bytes)),
            ])
            .build()
    }
    
    /// Create a connect operation for SSA data flow
    pub fn connect<'c>(
        context: &'c Context,
        from_layer: &str,
        to_layer: &str,
        tensor_shape: &[i64],
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("graph.connect", location)
            .add_attributes(&[
                (Identifier::new(context, "from"), string_attr(context, from_layer)),
                (Identifier::new(context, "to"), string_attr(context, to_layer)),
                (Identifier::new(context, "tensor_shape"), int_array_attr(context, tensor_shape)),
            ])
            .build()
    }
    
    /// Create metrics operation
    pub fn metrics<'c>(
        context: &'c Context,
        graph_depth: i64,
        total_operations: i64,
        total_intermediate_tensors: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("graph.metrics", location)
            .add_attributes(&[
                (Identifier::new(context, "graph_depth"), int_attr(context, graph_depth)),
                (Identifier::new(context, "total_operations"), int_attr(context, total_operations)),
                (Identifier::new(context, "total_intermediate_tensors"), int_attr(context, total_intermediate_tensors)),
            ])
            .build()
    }
}
