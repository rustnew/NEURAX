//! Optimization dialect for NEURAX
//!
//! Models optimization strategies (quantization, pruning, distillation, fusion)

use melior::ir::{Identifier, Location, Operation, operation::OperationBuilder};
use melior::Context;
use super::utils::{float_attr, int_attr, string_attr, bool_attr};

/// Optimization dialect name
pub const DIALECT_NAME: &str = "opt";

/// Optimization dialect
pub struct OptimizationDialect;

impl OptimizationDialect {
    /// Get the dialect name
    pub fn name() -> &'static str {
        DIALECT_NAME
    }
    
    /// Create a quantization operation
    pub fn quantization<'c>(
        context: &'c Context,
        quant_type: &str,
        bits: i64,
        group_size: i64,
        symmetrical: bool,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("opt.quantization", location)
            .add_attributes(&[
                (Identifier::new(context, "type"), string_attr(context, quant_type)),
                (Identifier::new(context, "bits"), int_attr(context, bits)),
                (Identifier::new(context, "group_size"), int_attr(context, group_size)),
                (Identifier::new(context, "symmetrical"), bool_attr(context, symmetrical)),
            ])
            .build()
    }
    
    /// Create a pruning operation
    pub fn pruning<'c>(
        context: &'c Context,
        pruning_type: &str,
        sparsity: f64,
        granularity: &str,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("opt.pruning", location)
            .add_attributes(&[
                (Identifier::new(context, "type"), string_attr(context, pruning_type)),
                (Identifier::new(context, "sparsity"), float_attr(context, sparsity)),
                (Identifier::new(context, "granularity"), string_attr(context, granularity)),
            ])
            .build()
    }
    
    /// Create a knowledge distillation operation
    pub fn distillation<'c>(
        context: &'c Context,
        teacher_model: &str,
        temperature: f64,
        alpha: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("opt.distillation", location)
            .add_attributes(&[
                (Identifier::new(context, "teacher_model"), string_attr(context, teacher_model)),
                (Identifier::new(context, "temperature"), float_attr(context, temperature)),
                (Identifier::new(context, "alpha"), float_attr(context, alpha)),
            ])
            .build()
    }
    
    /// Create a kernel fusion operation
    pub fn fusion<'c>(
        context: &'c Context,
        fusion_type: &str,
        fused_ops: &[&str],
        speedup: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("opt.fusion", location)
            .add_attributes(&[
                (Identifier::new(context, "type"), string_attr(context, fusion_type)),
                (Identifier::new(context, "fused_ops"), string_attr(context, &fused_ops.join(","))),
                (Identifier::new(context, "speedup"), float_attr(context, speedup)),
            ])
            .build()
    }
    
    /// Create a graph optimization operation
    pub fn graph_optimization<'c>(
        context: &'c Context,
        optimization_level: i64,
        constant_folding: bool,
        dead_code_elimination: bool,
        operator_fusion: bool,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("opt.graph", location)
            .add_attributes(&[
                (Identifier::new(context, "optimization_level"), int_attr(context, optimization_level)),
                (Identifier::new(context, "constant_folding"), bool_attr(context, constant_folding)),
                (Identifier::new(context, "dead_code_elimination"), bool_attr(context, dead_code_elimination)),
                (Identifier::new(context, "operator_fusion"), bool_attr(context, operator_fusion)),
            ])
            .build()
    }
    
    /// Create a memory optimization operation
    pub fn memory_optimization<'c>(
        context: &'c Context,
        activation_offload: bool,
        weight_offload: bool,
        recomputation: bool,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("opt.memory", location)
            .add_attributes(&[
                (Identifier::new(context, "activation_offload"), bool_attr(context, activation_offload)),
                (Identifier::new(context, "weight_offload"), bool_attr(context, weight_offload)),
                (Identifier::new(context, "recomputation"), bool_attr(context, recomputation)),
            ])
            .build()
    }
    
    /// Create a sparsity operation
    pub fn sparsity<'c>(
        context: &'c Context,
        pattern: &str,
        density: f64,
        block_size: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("opt.sparsity", location)
            .add_attributes(&[
                (Identifier::new(context, "pattern"), string_attr(context, pattern)),
                (Identifier::new(context, "density"), float_attr(context, density)),
                (Identifier::new(context, "block_size"), int_attr(context, block_size)),
            ])
            .build()
    }
    
    /// Create optimization metrics operation
    pub fn metrics<'c>(
        context: &'c Context,
        model_size_reduction: f64,
        inference_speedup: f64,
        memory_reduction: f64,
        accuracy_impact: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("opt.metrics", location)
            .add_attributes(&[
                (Identifier::new(context, "model_size_reduction"), float_attr(context, model_size_reduction)),
                (Identifier::new(context, "inference_speedup"), float_attr(context, inference_speedup)),
                (Identifier::new(context, "memory_reduction"), float_attr(context, memory_reduction)),
                (Identifier::new(context, "accuracy_impact"), float_attr(context, accuracy_impact)),
            ])
            .build()
    }
}
