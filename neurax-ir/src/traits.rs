//! Core IR traits

use crate::error::NeuraxError;
use crate::NeuraxContext;

/// Core trait for all IR passes
pub trait IrPass: Send + Sync {
    /// Input type for this pass
    type Input;
    /// Output type for this pass
    type Output;
    /// Metrics type produced by this pass
    type Metrics;
    /// Error type for this pass
    type PassError: Into<NeuraxError> + Send + Sync + 'static;

    /// Build the IR from input
    fn build(&self, input: &Self::Input, ctx: &NeuraxContext) -> Result<Self::Output, Self::PassError>;
    
    /// Compute metrics on the built IR
    fn compute_metrics(&self, output: &mut Self::Output, ctx: &NeuraxContext) -> Result<Self::Metrics, Self::PassError>;
    
    /// Validate that metrics are complete and consistent
    fn validate(&self, output: &Self::Output, metrics: &Self::Metrics) -> Result<(), Self::PassError>;
    
    /// Name of this pass for logging
    fn name(&self) -> &'static str;
}

/// Trait for passes that produce a report
pub trait ReportPass<'a>: Send + Sync {
    type Input: 'a;
    type Output;
    type PassError: Into<NeuraxError> + Send + Sync + 'static;

    fn build_report(&self, input: &Self::Input, ctx: &NeuraxContext) -> Result<Self::Output, Self::PassError>;
    fn name(&self) -> &'static str;
}
