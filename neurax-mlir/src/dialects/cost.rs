//! Cost dialect for NEURAX
//!
//! Models economic analysis (training cost, energy, CO2)

use melior::ir::{Identifier, Location, Operation, operation::OperationBuilder};
use melior::Context;
use super::utils::{float_attr, string_attr};

/// Cost dialect name
pub const DIALECT_NAME: &str = "cost";

/// Cost dialect
pub struct CostDialect;

impl CostDialect {
    /// Get the dialect name
    pub fn name() -> &'static str {
        DIALECT_NAME
    }
    
    /// Create a pricing model operation
    pub fn pricing_model<'c>(
        context: &'c Context,
        gpu_hour_usd: f64,
        energy_kwh_usd: f64,
        pue_factor: f64,
        gpu_tdp_watts: f64,
        co2_per_kwh: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("cost.pricing", location)
            .add_attributes(&[
                (Identifier::new(context, "gpu_hour_usd"), float_attr(context, gpu_hour_usd)),
                (Identifier::new(context, "energy_kwh_usd"), float_attr(context, energy_kwh_usd)),
                (Identifier::new(context, "pue_factor"), float_attr(context, pue_factor)),
                (Identifier::new(context, "gpu_tdp_watts"), float_attr(context, gpu_tdp_watts)),
                (Identifier::new(context, "co2_per_kwh"), float_attr(context, co2_per_kwh)),
            ])
            .build()
    }
    
    /// Create a full pricing model with provider
    pub fn pricing_full<'c>(
        context: &'c Context,
        provider: &str,
        gpu_hour_usd: f64,
        energy_kwh_usd: f64,
        pue_factor: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("cost.pricing", location)
            .add_attributes(&[
                (Identifier::new(context, "provider"), string_attr(context, provider)),
                (Identifier::new(context, "gpu_hour_usd"), float_attr(context, gpu_hour_usd)),
                (Identifier::new(context, "energy_kwh_usd"), float_attr(context, energy_kwh_usd)),
                (Identifier::new(context, "pue_factor"), float_attr(context, pue_factor)),
            ])
            .build()
    }
    
    /// Create a training cost operation
    pub fn training_cost<'c>(
        context: &'c Context,
        training_time_hours: f64,
        training_cost_usd: f64,
        gpu_hours_total: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("cost.training", location)
            .add_attributes(&[
                (Identifier::new(context, "training_time_hours"), float_attr(context, training_time_hours)),
                (Identifier::new(context, "training_cost_usd"), float_attr(context, training_cost_usd)),
                (Identifier::new(context, "gpu_hours_total"), float_attr(context, gpu_hours_total)),
            ])
            .build()
    }
    
    /// Create an energy operation
    pub fn energy<'c>(
        context: &'c Context,
        energy_kwh: f64,
        co2_kg: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("cost.energy", location)
            .add_attributes(&[
                (Identifier::new(context, "energy_kwh"), float_attr(context, energy_kwh)),
                (Identifier::new(context, "co2_kg"), float_attr(context, co2_kg)),
            ])
            .build()
    }
    
    /// Create a token cost operation
    pub fn token_cost<'c>(
        context: &'c Context,
        cost_per_million_tokens_usd: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("cost.token_cost", location)
            .add_attributes(&[
                (Identifier::new(context, "cost_per_million_tokens_usd"), float_attr(context, cost_per_million_tokens_usd)),
            ])
            .build()
    }
    
    /// Create metrics operation
    pub fn metrics<'c>(
        context: &'c Context,
        training_cost_usd: f64,
        training_time_hours: f64,
        energy_kwh: f64,
        co2_kg: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("cost.metrics", location)
            .add_attributes(&[
                (Identifier::new(context, "training_cost_usd"), float_attr(context, training_cost_usd)),
                (Identifier::new(context, "training_time_hours"), float_attr(context, training_time_hours)),
                (Identifier::new(context, "energy_kwh"), float_attr(context, energy_kwh)),
                (Identifier::new(context, "co2_kg"), float_attr(context, co2_kg)),
            ])
            .build()
    }
}
