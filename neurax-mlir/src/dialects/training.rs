//! Training dialect for NEURAX
//!
//! Models training configuration (optimizer, scheduler, precision, checkpointing)

use melior::ir::{Identifier, Location, Operation, operation::OperationBuilder};
use melior::Context;
use super::utils::{float_attr, int_attr, string_attr, bool_attr};

/// Training dialect name
pub const DIALECT_NAME: &str = "train";

/// Training dialect
pub struct TrainingDialect;

impl TrainingDialect {
    /// Get the dialect name
    pub fn name() -> &'static str {
        DIALECT_NAME
    }
    
    /// Create a training configuration operation
    pub fn config<'c>(
        context: &'c Context,
        batch_size: i64,
        precision: &str,
        max_steps: i64,
        warmup_steps: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("train.config", location)
            .add_attributes(&[
                (Identifier::new(context, "batch_size"), int_attr(context, batch_size)),
                (Identifier::new(context, "precision"), string_attr(context, precision)),
                (Identifier::new(context, "max_steps"), int_attr(context, max_steps)),
                (Identifier::new(context, "warmup_steps"), int_attr(context, warmup_steps)),
            ])
            .build()
    }
    
    /// Create an optimizer operation
    pub fn optimizer<'c>(
        context: &'c Context,
        optimizer_type: &str,
        learning_rate: f64,
        weight_decay: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("train.optimizer", location)
            .add_attributes(&[
                (Identifier::new(context, "type"), string_attr(context, optimizer_type)),
                (Identifier::new(context, "learning_rate"), float_attr(context, learning_rate)),
                (Identifier::new(context, "weight_decay"), float_attr(context, weight_decay)),
                (Identifier::new(context, "beta1"), float_attr(context, beta1)),
                (Identifier::new(context, "beta2"), float_attr(context, beta2)),
                (Identifier::new(context, "epsilon"), float_attr(context, epsilon)),
            ])
            .build()
    }
    
    /// Create a scheduler operation
    pub fn scheduler<'c>(
        context: &'c Context,
        scheduler_type: &str,
        warmup_steps: i64,
        decay_steps: i64,
        decay_rate: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("train.scheduler", location)
            .add_attributes(&[
                (Identifier::new(context, "type"), string_attr(context, scheduler_type)),
                (Identifier::new(context, "warmup_steps"), int_attr(context, warmup_steps)),
                (Identifier::new(context, "decay_steps"), int_attr(context, decay_steps)),
                (Identifier::new(context, "decay_rate"), float_attr(context, decay_rate)),
            ])
            .build()
    }
    
    /// Create a gradient checkpointing operation
    pub fn gradient_checkpointing<'c>(
        context: &'c Context,
        enabled: bool,
        checkpoint_layers: i64,
        memory_savings_bytes: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("train.gradient_checkpointing", location)
            .add_attributes(&[
                (Identifier::new(context, "enabled"), bool_attr(context, enabled)),
                (Identifier::new(context, "checkpoint_layers"), int_attr(context, checkpoint_layers)),
                (Identifier::new(context, "memory_savings_bytes"), int_attr(context, memory_savings_bytes)),
            ])
            .build()
    }
    
    /// Create a mixed precision operation
    pub fn mixed_precision<'c>(
        context: &'c Context,
        enabled: bool,
        compute_dtype: &str,
        master_weights_dtype: &str,
        loss_scaling: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("train.mixed_precision", location)
            .add_attributes(&[
                (Identifier::new(context, "enabled"), bool_attr(context, enabled)),
                (Identifier::new(context, "compute_dtype"), string_attr(context, compute_dtype)),
                (Identifier::new(context, "master_weights_dtype"), string_attr(context, master_weights_dtype)),
                (Identifier::new(context, "loss_scaling"), float_attr(context, loss_scaling)),
            ])
            .build()
    }
    
    /// Create a ZeRO configuration operation
    pub fn zero_config<'c>(
        context: &'c Context,
        stage: i64,
        offload_optimizer: bool,
        offload_param: bool,
        partition_activations: bool,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("train.zero", location)
            .add_attributes(&[
                (Identifier::new(context, "stage"), int_attr(context, stage)),
                (Identifier::new(context, "offload_optimizer"), bool_attr(context, offload_optimizer)),
                (Identifier::new(context, "offload_param"), bool_attr(context, offload_param)),
                (Identifier::new(context, "partition_activations"), bool_attr(context, partition_activations)),
            ])
            .build()
    }
    
    /// Create a batch accumulation operation
    pub fn gradient_accumulation<'c>(
        context: &'c Context,
        steps: i64,
        effective_batch_size: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("train.gradient_accumulation", location)
            .add_attributes(&[
                (Identifier::new(context, "steps"), int_attr(context, steps)),
                (Identifier::new(context, "effective_batch_size"), int_attr(context, effective_batch_size)),
            ])
            .build()
    }
    
    /// Create a regularization operation
    pub fn regularization<'c>(
        context: &'c Context,
        dropout_rate: f64,
        weight_decay: f64,
        label_smoothing: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("train.regularization", location)
            .add_attributes(&[
                (Identifier::new(context, "dropout_rate"), float_attr(context, dropout_rate)),
                (Identifier::new(context, "weight_decay"), float_attr(context, weight_decay)),
                (Identifier::new(context, "label_smoothing"), float_attr(context, label_smoothing)),
            ])
            .build()
    }
    
    /// Create training metrics operation
    pub fn metrics<'c>(
        context: &'c Context,
        throughput_tokens_per_sec: f64,
        throughput_samples_per_sec: f64,
        memory_efficiency: f64,
        compute_efficiency: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("train.metrics", location)
            .add_attributes(&[
                (Identifier::new(context, "throughput_tokens_per_sec"), float_attr(context, throughput_tokens_per_sec)),
                (Identifier::new(context, "throughput_samples_per_sec"), float_attr(context, throughput_samples_per_sec)),
                (Identifier::new(context, "memory_efficiency"), float_attr(context, memory_efficiency)),
                (Identifier::new(context, "compute_efficiency"), float_attr(context, compute_efficiency)),
            ])
            .build()
    }
}
