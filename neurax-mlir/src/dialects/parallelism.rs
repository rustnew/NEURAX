//! Parallelism dialect for NEURAX
//!
//! Models parallelism strategies (DP, TP, PP, ZeRO)

use melior::ir::{Identifier, Location, Operation, operation::OperationBuilder};
use melior::Context;
use super::utils::{int_attr, float_attr};

/// Parallelism dialect name
pub const DIALECT_NAME: &str = "par";

/// Parallelism dialect
pub struct ParallelismDialect;

impl ParallelismDialect {
    /// Get the dialect name
    pub fn name() -> &'static str {
        DIALECT_NAME
    }
    
    /// Create a data parallel operation
    pub fn data_parallel<'c>(
        context: &'c Context,
        num_gpus: i64,
        efficiency: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("par.data_parallel", location)
            .add_attributes(&[
                (Identifier::new(context, "num_gpus"), int_attr(context, num_gpus)),
                (Identifier::new(context, "efficiency"), float_attr(context, efficiency)),
            ])
            .build()
    }
    
    /// Create a tensor parallel operation
    pub fn tensor_parallel<'c>(
        context: &'c Context,
        tp_degree: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("par.tensor_parallel", location)
            .add_attributes(&[
                (Identifier::new(context, "tp_degree"), int_attr(context, tp_degree)),
            ])
            .build()
    }
    
    /// Create a pipeline parallel operation
    pub fn pipeline_parallel<'c>(
        context: &'c Context,
        stages: i64,
        micro_batches: i64,
        bubble_ratio: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("par.pipeline_parallel", location)
            .add_attributes(&[
                (Identifier::new(context, "stages"), int_attr(context, stages)),
                (Identifier::new(context, "micro_batches"), int_attr(context, micro_batches)),
                (Identifier::new(context, "bubble_ratio"), float_attr(context, bubble_ratio)),
            ])
            .build()
    }
    
    /// Create a ZeRO operation
    pub fn zero<'c>(
        context: &'c Context,
        stage: i64,
        memory_per_gpu_bytes: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("par.zero", location)
            .add_attributes(&[
                (Identifier::new(context, "stage"), int_attr(context, stage)),
                (Identifier::new(context, "memory_per_gpu_bytes"), int_attr(context, memory_per_gpu_bytes)),
            ])
            .build()
    }
    
    /// Create a hybrid operation
    pub fn hybrid<'c>(
        context: &'c Context,
        dp: i64,
        tp: i64,
        pp: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("par.hybrid", location)
            .add_attributes(&[
                (Identifier::new(context, "dp"), int_attr(context, dp)),
                (Identifier::new(context, "tp"), int_attr(context, tp)),
                (Identifier::new(context, "pp"), int_attr(context, pp)),
            ])
            .build()
    }
    
    /// Create a hybrid operation with expert parallelism
    pub fn hybrid_full<'c>(
        context: &'c Context,
        dp: i64,
        tp: i64,
        pp: i64,
        ep: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("par.hybrid", location)
            .add_attributes(&[
                (Identifier::new(context, "dp"), int_attr(context, dp)),
                (Identifier::new(context, "tp"), int_attr(context, tp)),
                (Identifier::new(context, "pp"), int_attr(context, pp)),
                (Identifier::new(context, "ep"), int_attr(context, ep)),
            ])
            .build()
    }
    
    /// Create an expert parallel operation
    pub fn expert_parallel<'c>(
        context: &'c Context,
        num_experts: i64,
        num_gpus: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("par.expert_parallel", location)
            .add_attributes(&[
                (Identifier::new(context, "num_experts"), int_attr(context, num_experts)),
                (Identifier::new(context, "num_gpus"), int_attr(context, num_gpus)),
            ])
            .build()
    }
    
    /// Create metrics operation
    pub fn metrics<'c>(
        context: &'c Context,
        data_parallel_efficiency: f64,
        communication_overhead: f64,
        optimal_gpu_count: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("par.metrics", location)
            .add_attributes(&[
                (Identifier::new(context, "data_parallel_efficiency"), float_attr(context, data_parallel_efficiency)),
                (Identifier::new(context, "communication_overhead"), float_attr(context, communication_overhead)),
                (Identifier::new(context, "optimal_gpu_count"), int_attr(context, optimal_gpu_count)),
            ])
            .build()
    }
}
