//! Report dialect for NEURAX
//!
//! Models final consolidated reports

use melior::ir::{Identifier, Location, Operation, Region, operation::OperationBuilder};
use melior::Context;
use super::utils::{string_attr, int_attr, float_attr};

/// Report dialect name
pub const DIALECT_NAME: &str = "report";

/// Report dialect
pub struct ReportDialect;

impl ReportDialect {
    /// Get the dialect name
    pub fn name() -> &'static str {
        DIALECT_NAME
    }
    
    /// Create a report operation
    pub fn report<'c>(
        context: &'c Context,
        model_name: &str,
        model_type: &str,
        schema_version: &str,
        analysis_time_ms: i64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("report.report", location)
            .add_attributes(&[
                (Identifier::new(context, "model_name"), string_attr(context, model_name)),
                (Identifier::new(context, "model_type"), string_attr(context, model_type)),
                (Identifier::new(context, "schema_version"), string_attr(context, schema_version)),
                (Identifier::new(context, "analysis_time_ms"), int_attr(context, analysis_time_ms)),
            ])
            .add_regions([Region::new(), Region::new(), Region::new()])
            .build()
    }
    
    /// Create an all metrics operation
    pub fn all_metrics<'c>(
        context: &'c Context,
        total_parameters: i64,
        total_flops: f64,
        peak_vram_bytes: i64,
        latency_ms: f64,
        training_cost_usd: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("report.all_metrics", location)
            .add_attributes(&[
                (Identifier::new(context, "total_parameters"), int_attr(context, total_parameters)),
                (Identifier::new(context, "total_flops"), float_attr(context, total_flops)),
                (Identifier::new(context, "peak_vram_bytes"), int_attr(context, peak_vram_bytes)),
                (Identifier::new(context, "latency_ms"), float_attr(context, latency_ms)),
                (Identifier::new(context, "training_cost_usd"), float_attr(context, training_cost_usd)),
            ])
            .build()
    }
    
    /// Create a diagnostic operation
    pub fn diagnostic<'c>(
        context: &'c Context,
        category: &str,
        severity: &str,
        message: &str,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("report.diagnostic", location)
            .add_attributes(&[
                (Identifier::new(context, "category"), string_attr(context, category)),
                (Identifier::new(context, "severity"), string_attr(context, severity)),
                (Identifier::new(context, "message"), string_attr(context, message)),
            ])
            .build()
    }
    
    /// Create a recommendation operation
    pub fn recommendation<'c>(
        context: &'c Context,
        category: &str,
        title: &str,
        description: &str,
        priority: &str,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("report.recommendation", location)
            .add_attributes(&[
                (Identifier::new(context, "category"), string_attr(context, category)),
                (Identifier::new(context, "title"), string_attr(context, title)),
                (Identifier::new(context, "description"), string_attr(context, description)),
                (Identifier::new(context, "priority"), string_attr(context, priority)),
            ])
            .build()
    }
    
    /// Create a hardware config operation
    pub fn hardware_config<'c>(
        context: &'c Context,
        gpu_name: &str,
        gpu_count: i64,
        gpu_memory_gb: f64,
        location: Location<'c>,
    ) -> Result<Operation<'c>, melior::Error> {
        OperationBuilder::new("report.hw_config", location)
            .add_attributes(&[
                (Identifier::new(context, "gpu_name"), string_attr(context, gpu_name)),
                (Identifier::new(context, "gpu_count"), int_attr(context, gpu_count)),
                (Identifier::new(context, "gpu_memory_gb"), float_attr(context, gpu_memory_gb)),
            ])
            .build()
    }
}
