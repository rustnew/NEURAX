//! Data dialect for NEURAX
//!
//! Models data configuration (vocab, sequence, tokenization, dataset)

use melior::ir::{Identifier, Location, Operation, operation::OperationBuilder};
use melior::Context;
use super::utils::{float_attr, int_attr, string_attr};

/// Data dialect name
pub const DIALECT_NAME: &str = "data";

/// Data dialect
pub struct DataDialect;

impl DataDialect {
    /// Get the dialect name
    pub fn name() -> &'static str {
        DIALECT_NAME
    }
    
    /// Create a vocabulary operation
    pub fn vocab<'c>(
        context: &'c Context,
        vocab_size: i64,
        vocab_type: &str,
        special_tokens: &[&str],
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("data.vocab", location)
            .add_attributes(&[
                (Identifier::new(context, "vocab_size"), int_attr(context, vocab_size)),
                (Identifier::new(context, "vocab_type"), string_attr(context, vocab_type)),
                (Identifier::new(context, "special_tokens"), string_attr(context, &special_tokens.join(","))),
            ])
            .build()
    }
    
    /// Create a sequence configuration operation
    pub fn sequence<'c>(
        context: &'c Context,
        max_length: i64,
        stride: i64,
        padding_side: &str,
        truncation_side: &str,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("data.sequence", location)
            .add_attributes(&[
                (Identifier::new(context, "max_length"), int_attr(context, max_length)),
                (Identifier::new(context, "stride"), int_attr(context, stride)),
                (Identifier::new(context, "padding_side"), string_attr(context, padding_side)),
                (Identifier::new(context, "truncation_side"), string_attr(context, truncation_side)),
            ])
            .build()
    }
    
    /// Create a tokenization operation
    pub fn tokenization<'c>(
        context: &'c Context,
        tokenizer_type: &str,
        model_max_length: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("data.tokenization", location)
            .add_attributes(&[
                (Identifier::new(context, "tokenizer_type"), string_attr(context, tokenizer_type)),
                (Identifier::new(context, "model_max_length"), int_attr(context, model_max_length)),
            ])
            .build()
    }
    
    /// Create a dataset operation
    pub fn dataset<'c>(
        context: &'c Context,
        name: &str,
        num_samples: i64,
        num_tokens: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("data.dataset", location)
            .add_attributes(&[
                (Identifier::new(context, "name"), string_attr(context, name)),
                (Identifier::new(context, "num_samples"), int_attr(context, num_samples)),
                (Identifier::new(context, "num_tokens"), int_attr(context, num_tokens)),
            ])
            .build()
    }
    
    /// Create a dataloader operation
    pub fn dataloader<'c>(
        context: &'c Context,
        batch_size: i64,
        shuffle: bool,
        num_workers: i64,
        pin_memory: bool,
        drop_last: bool,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("data.dataloader", location)
            .add_attributes(&[
                (Identifier::new(context, "batch_size"), int_attr(context, batch_size)),
                (Identifier::new(context, "shuffle"), Attribute::parse(context, &format!("{}", shuffle)).unwrap()),
                (Identifier::new(context, "num_workers"), int_attr(context, num_workers)),
                (Identifier::new(context, "pin_memory"), Attribute::parse(context, &format!("{}", pin_memory)).unwrap()),
                (Identifier::new(context, "drop_last"), Attribute::parse(context, &format!("{}", drop_last)).unwrap()),
            ])
            .build()
    }
    
    /// Create a data augmentation operation
    pub fn augmentation<'c>(
        context: &'c Context,
        augmentation_type: &str,
        probability: f64,
        params: &str,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("data.augmentation", location)
            .add_attributes(&[
                (Identifier::new(context, "type"), string_attr(context, augmentation_type)),
                (Identifier::new(context, "probability"), float_attr(context, probability)),
                (Identifier::new(context, "params"), string_attr(context, params)),
            ])
            .build()
    }
    
    /// Create data metrics operation
    pub fn metrics<'c>(
        context: &'c Context,
        total_tokens: i64,
        total_samples: i64,
        tokens_per_sample: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("data.metrics", location)
            .add_attributes(&[
                (Identifier::new(context, "total_tokens"), int_attr(context, total_tokens)),
                (Identifier::new(context, "total_samples"), int_attr(context, total_samples)),
                (Identifier::new(context, "tokens_per_sample"), float_attr(context, tokens_per_sample)),
            ])
            .build()
    }
}

use melior::ir::Attribute;
