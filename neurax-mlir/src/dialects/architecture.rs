//! Architecture dialect for NEURAX
//!
//! Models neural network architecture structure

use melior::ir::{Identifier, Location, Operation, Region, operation::OperationBuilder};
use melior::Context;
use super::utils::{string_attr, int_attr, float_attr};

/// Architecture dialect name
pub const DIALECT_NAME: &str = "arch";

/// Architecture dialect
pub struct ArchitectureDialect;

impl ArchitectureDialect {
    /// Get the dialect name
    pub fn name() -> &'static str {
        DIALECT_NAME
    }
    
    /// Create a model operation
    pub fn model<'c>(
        context: &'c Context,
        name: &str,
        model_type: &str,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("arch.model", location)
            .add_attributes(&[
                (Identifier::new(context, "name"), string_attr(context, name)),
                (Identifier::new(context, "model_type"), string_attr(context, model_type)),
            ])
            .add_regions([Region::new()])
            .build()
    }
    
    /// Create a full model operation with all attributes
    pub fn model_full<'c>(
        context: &'c Context,
        name: &str,
        model_type: &str,
        version: &str,
        description: &str,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("arch.model", location)
            .add_attributes(&[
                (Identifier::new(context, "name"), string_attr(context, name)),
                (Identifier::new(context, "model_type"), string_attr(context, model_type)),
                (Identifier::new(context, "version"), string_attr(context, version)),
                (Identifier::new(context, "description"), string_attr(context, description)),
            ])
            .add_regions([Region::new()])
            .build()
    }
    
    /// Create a layer operation
    pub fn layer<'c>(
        context: &'c Context,
        id: &str,
        layer_type: &str,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("arch.layer", location)
            .add_attributes(&[
                (Identifier::new(context, "id"), string_attr(context, id)),
                (Identifier::new(context, "layer_type"), string_attr(context, layer_type)),
            ])
            .add_regions([Region::new()])
            .build()
    }
    
    /// Create a full layer operation with input/output shapes
    pub fn layer_full<'c>(
        context: &'c Context,
        id: &str,
        layer_type: &str,
        input_shape: &[i64],
        output_shape: &[i64],
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("arch.layer", location)
            .add_attributes(&[
                (Identifier::new(context, "id"), string_attr(context, id)),
                (Identifier::new(context, "layer_type"), string_attr(context, layer_type)),
                (Identifier::new(context, "input_shape"), string_attr(context, &format!("{:?}", input_shape))),
                (Identifier::new(context, "output_shape"), string_attr(context, &format!("{:?}", output_shape))),
            ])
            .add_regions([Region::new()])
            .build()
    }
    
    /// Create global params operation
    pub fn global_params<'c>(
        context: &'c Context,
        params: &[(&str, i64)],
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        let attrs: Vec<(Identifier, _)> = params.iter()
            .map(|(key, value)| (Identifier::new(context, key), int_attr(context, *value)))
            .collect();
        OperationBuilder::new("arch.global_params", location)
            .add_attributes(&attrs)
            .build()
    }
    
    /// Create a full global params operation with all attributes
    pub fn global_params_full<'c>(
        context: &'c Context,
        hidden_size: i64,
        intermediate_size: i64,
        num_attention_heads: i64,
        num_key_value_heads: i64,
        num_layers: i64,
        vocab_size: i64,
        sequence_length: i64,
        embedding_dim: i64,
        dropout_rate: f64,
        layer_norm_eps: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("arch.global_params", location)
            .add_attributes(&[
                (Identifier::new(context, "hidden_size"), int_attr(context, hidden_size)),
                (Identifier::new(context, "intermediate_size"), int_attr(context, intermediate_size)),
                (Identifier::new(context, "num_attention_heads"), int_attr(context, num_attention_heads)),
                (Identifier::new(context, "num_key_value_heads"), int_attr(context, num_key_value_heads)),
                (Identifier::new(context, "num_layers"), int_attr(context, num_layers)),
                (Identifier::new(context, "vocab_size"), int_attr(context, vocab_size)),
                (Identifier::new(context, "sequence_length"), int_attr(context, sequence_length)),
                (Identifier::new(context, "embedding_dim"), int_attr(context, embedding_dim)),
                (Identifier::new(context, "dropout_rate"), float_attr(context, dropout_rate)),
                (Identifier::new(context, "layer_norm_eps"), float_attr(context, layer_norm_eps)),
            ])
            .build()
    }
    
    /// Create metrics operation
    pub fn metrics<'c>(
        context: &'c Context,
        total_parameters: i64,
        num_layers: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("arch.metrics", location)
            .add_attributes(&[
                (Identifier::new(context, "total_parameters"), int_attr(context, total_parameters)),
                (Identifier::new(context, "num_layers"), int_attr(context, num_layers)),
            ])
            .build()
    }
    
    /// Create a full metrics operation with detailed architecture info
    pub fn metrics_full<'c>(
        context: &'c Context,
        total_parameters: i64,
        num_layers: i64,
        num_attention_layers: i64,
        num_mlp_layers: i64,
        num_embedding_layers: i64,
        num_normalization_layers: i64,
        model_size_mb: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("arch.metrics", location)
            .add_attributes(&[
                (Identifier::new(context, "total_parameters"), int_attr(context, total_parameters)),
                (Identifier::new(context, "num_layers"), int_attr(context, num_layers)),
                (Identifier::new(context, "num_attention_layers"), int_attr(context, num_attention_layers)),
                (Identifier::new(context, "num_mlp_layers"), int_attr(context, num_mlp_layers)),
                (Identifier::new(context, "num_embedding_layers"), int_attr(context, num_embedding_layers)),
                (Identifier::new(context, "num_normalization_layers"), int_attr(context, num_normalization_layers)),
                (Identifier::new(context, "model_size_mb"), float_attr(context, model_size_mb)),
            ])
            .build()
    }
    
    /// Create a layer params operation
    pub fn layer_params<'c>(
        context: &'c Context,
        layer_id: &str,
        param_count: i64,
        flops_per_token: f64,
        memory_bytes: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("arch.layer_params", location)
            .add_attributes(&[
                (Identifier::new(context, "layer_id"), string_attr(context, layer_id)),
                (Identifier::new(context, "param_count"), int_attr(context, param_count)),
                (Identifier::new(context, "flops_per_token"), float_attr(context, flops_per_token)),
                (Identifier::new(context, "memory_bytes"), int_attr(context, memory_bytes)),
            ])
            .build()
    }
    
    /// Create a model family operation
    pub fn model_family<'c>(
        context: &'c Context,
        family: &str,
        base_model: &str,
        variant: &str,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("arch.model_family", location)
            .add_attributes(&[
                (Identifier::new(context, "family"), string_attr(context, family)),
                (Identifier::new(context, "base_model"), string_attr(context, base_model)),
                (Identifier::new(context, "variant"), string_attr(context, variant)),
            ])
            .build()
    }
    
    /// Create a repeat operation for factorizing repeated layer patterns
    pub fn repeat<'c>(
        context: &'c Context,
        count: i64,
        body_region: Region<'c>,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("arch.repeat", location)
            .add_attributes(&[
                (Identifier::new(context, "count"), int_attr(context, count)),
            ])
            .add_regions([body_region])
            .build()
    }
    
    /// Create a layer pattern operation for describing repeated structures
    pub fn layer_pattern<'c>(
        context: &'c Context,
        pattern_name: &str,
        layer_types: &[&str],
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        let layers_str = layer_types.join(", ");
        OperationBuilder::new("arch.layer_pattern", location)
            .add_attributes(&[
                (Identifier::new(context, "pattern_name"), string_attr(context, pattern_name)),
                (Identifier::new(context, "layer_types"), string_attr(context, &layers_str)),
            ])
            .build()
    }
}
