//! Cost IR pass

use crate::traits::IrPass;
use crate::error::CostError;
use crate::NeuraxContext;
use crate::hardware::HardwareIR;
use crate::parallelism::ParallelismIR;
use super::{CostIR, PricingModel, CostMetrics};

/// Cost pass implementation
pub struct CostPass;

impl IrPass for CostPass {
    type Input = (HardwareIR, ParallelismIR);
    type Output = CostIR;
    type Metrics = CostMetrics;
    type PassError = CostError;

    fn name(&self) -> &'static str {
        "CostIR"
    }

    fn build(&self, input: &Self::Input, ctx: &NeuraxContext) -> Result<Self::Output, Self::PassError> {
        let (_hw_ir, _parallel_ir) = input;
        let mut cost_ir = CostIR::default();
        
        // Set up pricing model from config
        cost_ir.pricing_model = PricingModel {
            gpu_hour_usd: ctx.config.cost_config.gpu_hour_usd,
            energy_kwh_usd: ctx.config.cost_config.energy_kwh_usd,
            pue_factor: ctx.config.cost_config.pue_factor,
            gpu_tdp_watts: ctx.config.hardware.gpus.first()
                .map(|g| match g.name.as_str() {
                    "A100-SXM" | "A100-PCIe" => 400.0,
                    "H100-SXM" => 700.0,
                    "H100-PCIe" => 350.0,
                    "RTX4090" => 450.0,
                    "V100" => 300.0,
                    _ => 300.0,
                })
                .unwrap_or(300.0),
            co2_per_kwh: 0.233,
        };
        
        Ok(cost_ir)
    }

    fn compute_metrics(&self, output: &mut Self::Output, ctx: &NeuraxContext) -> Result<Self::Metrics, Self::PassError> {
        let max_steps = ctx.config.training.max_steps;
        let num_gpus = ctx.config.hardware.total_gpu_count();
        let batch = ctx.config.training.batch_size;
        let seq = ctx.config.model.global_params.sequence_length.unwrap_or(512);
        
        // Get latency from hardware IR (would be passed in properly)
        let latency_ms = 100.0; // Placeholder
        
        // Training time
        let training_time_s = max_steps as f64 * latency_ms / 1000.0;
        let training_time_hours = training_time_s / 3600.0;
        
        // GPU hours
        let gpu_hours_total = training_time_hours * num_gpus as f64;
        
        // Training cost
        let training_cost_usd = gpu_hours_total * output.pricing_model.gpu_hour_usd;
        
        // Energy consumption
        let energy_kwh = training_time_hours 
            * output.pricing_model.gpu_tdp_watts 
            * num_gpus as f64 
            / 1000.0 
            * output.pricing_model.pue_factor;
        
        // CO2 emissions
        let co2_kg = energy_kwh * output.pricing_model.co2_per_kwh;
        
        // Cost per token
        let total_tokens = max_steps as f64 * batch as f64 * seq as f64;
        let cost_per_token_usd = if total_tokens > 0.0 {
            training_cost_usd / total_tokens
        } else {
            0.0
        };
        
        let cost_per_million_tokens_usd = cost_per_token_usd * 1e6;
        
        // Cost per step
        let cost_per_step_usd = if max_steps > 0 {
            training_cost_usd / max_steps as f64
        } else {
            0.0
        };
        
        // Monthly inference cost (estimate)
        let monthly_inference_cost_usd = 30.0 * 24.0 * output.pricing_model.gpu_hour_usd * num_gpus as f64;
        
        let metrics = CostMetrics {
            training_time_hours,
            training_cost_usd,
            gpu_hours_total,
            energy_kwh,
            co2_kg,
            cost_per_token_usd,
            cost_per_million_tokens_usd,
            monthly_inference_cost_usd,
            cost_per_step_usd,
        };
        
        output.metrics = metrics.clone();
        output.metrics_done = true;
        Ok(metrics)
    }

    fn validate(&self, _output: &Self::Output, metrics: &Self::Metrics) -> Result<(), Self::PassError> {
        if metrics.training_cost_usd < 0.0 {
            return Err(CostError::PricingFailed("Training cost is negative".to_string()));
        }
        Ok(())
    }
}
