//! Architecture IR pass

use super::{ArchitectureIR, ArchitectureMetrics, LayerDef};
use crate::error::ArchitectureError;
use crate::traits::IrPass;
use crate::NeuraxContext;
use neurax_parser::ModelConfig;
use std::collections::HashMap;

/// Architecture pass implementation
pub struct ArchitecturePass;

impl IrPass for ArchitecturePass {
    type Input = ModelConfig;
    type Output = ArchitectureIR;
    type Metrics = ArchitectureMetrics;
    type PassError = ArchitectureError;

    fn name(&self) -> &'static str {
        "ArchitectureIR"
    }

    fn build(
        &self,
        input: &Self::Input,
        _ctx: &NeuraxContext,
    ) -> Result<Self::Output, Self::PassError> {
        if input.model.layers.is_empty() {
            return Err(ArchitectureError::EmptyLayers);
        }

        let mut layers: Vec<LayerDef> = input.model.layers.iter().map(LayerDef::from).collect();

        // Calculate parameters for each layer
        for layer in &mut layers {
            layer.param_count = super::calculate_layer_params(&neurax_parser::Layer {
                id: layer.id.clone(),
                layer_type: layer.layer_type,
                input_shape: layer.input_shape.clone(),
                output_shape: layer.output_shape.clone(),
                params: layer.params.clone(),
                custom_equations: layer.custom_equations.clone(),
            });
        }

        Ok(ArchitectureIR {
            model_type: input.model.model_type,
            model_name: input.model.name.clone(),
            layers,
            global_params: input.model.global_params.clone(),
            training_config: input.training.clone(),
            hardware_config: input.hardware.clone(),
            metrics: ArchitectureMetrics::default(),
            metrics_done: false,
        })
    }

    fn compute_metrics(
        &self,
        output: &mut Self::Output,
        _ctx: &NeuraxContext,
    ) -> Result<Self::Metrics, Self::PassError> {
        let json_layer_count = output.layers.len();

        // Detect if the JSON is a partial representation of a larger model.
        let global_num_layers = output
            .global_params
            .num_layers
            .unwrap_or(json_layer_count as u64) as usize;

        // Total parameters - sum from individual layer calculations
        // This works for any architecture (Transformer, Mamba, MoE, etc.)
        let total_params: u64 = output.layers.iter().map(|l| l.param_count).sum();

        let mut metrics = ArchitectureMetrics {
            num_layers: global_num_layers.max(json_layer_count),
            model_type_info: output.model_type.as_str().to_string(),
            params_per_layer: HashMap::new(),
            layers_by_type: HashMap::new(),
            total_parameters: total_params,
        };

        // Fill params_per_layer and layers_by_type for reference
        for layer in &output.layers {
            metrics
                .params_per_layer
                .insert(layer.id.clone(), layer.param_count);
            let type_str = layer.layer_type.as_str();
            *metrics
                .layers_by_type
                .entry(type_str.to_string())
                .or_insert(0) += 1;
        }

        output.metrics = metrics.clone();
        output.metrics_done = true;
        Ok(metrics)
    }

    fn validate(
        &self,
        _output: &Self::Output,
        metrics: &Self::Metrics,
    ) -> Result<(), Self::PassError> {
        if metrics.total_parameters == 0 {
            return Err(ArchitectureError::ParameterComputation(
                "Total parameters is zero".to_string(),
            ));
        }
        if metrics.num_layers == 0 {
            return Err(ArchitectureError::EmptyLayers);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neurax_parser::parse_model_config;

    #[test]
    fn test_architecture_pass_transformer() {
        let json = r#"{
            "schema_version": "1.0",
            "model": {
                "name": "TestTransformer",
                "type": "transformer",
                "layers": [
                    {"id": "embed", "layer_type": "embedding", "params": {"vocab_size": 50000, "embedding_dim": 768}},
                    {"id": "attn1", "layer_type": "attention", "params": {"hidden_size": 768, "num_heads": 12}},
                    {"id": "mlp1", "layer_type": "mlp", "params": {"hidden_size": 768, "intermediate_size": 3072}}
                ]
            },
            "training": {"batch_size": 32},
            "hardware": {"gpus": [{"name": "A100", "count": 1}]}
        }"#;

        let config = parse_model_config(json).unwrap();
        let ctx = NeuraxContext::new(config.clone());
        let pass = ArchitecturePass;

        let mut ir = pass.build(&config, &ctx).unwrap();
        let metrics = pass.compute_metrics(&mut ir, &ctx).unwrap();

        assert_eq!(metrics.num_layers, 3);
        assert!(metrics.total_parameters > 0);
        assert!(ir.metrics_done);
    }
}
