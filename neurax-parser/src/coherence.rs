//! Coherence validation rules from mlir_json.md
//!
//! Implements numerical coherence checks for model validation

use serde_json::Value;
use std::collections::HashMap;
use evalexpr::Value as EvalValue;

/// Coherence check result
#[derive(Debug, Clone)]
pub struct CoherenceResult {
    pub is_coherent: bool,
    pub computed_params: u64,
    pub expected_params: Option<u64>,
    pub param_diff_percent: Option<f64>,
    pub flops_forward: f64,
    pub flops_backward: f64,
    pub memory_activation_bytes: u64,
    pub memory_gradient_bytes: u64,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

/// Coherence validator for numerical rules
pub struct CoherenceValidator {
    batch_size: u64,
    seq_length: u64,
}

impl CoherenceValidator {
    pub fn new() -> Self {
        Self { batch_size: 1, seq_length: 1 }
    }
    
    pub fn with_batch_size(mut self, batch: u64) -> Self {
        self.batch_size = batch;
        self
    }
    
    pub fn with_seq_length(mut self, seq: u64) -> Self {
        self.seq_length = seq;
        self
    }
    
    /// Validate coherence rules (Rules 4.1, 4.2, 4.3)
    pub fn validate(&self, value: &Value) -> CoherenceResult {
        let mut result = CoherenceResult {
            is_coherent: true,
            computed_params: 0,
            expected_params: None,
            param_diff_percent: None,
            flops_forward: 0.0,
            flops_backward: 0.0,
            memory_activation_bytes: 0,
            memory_gradient_bytes: 0,
            warnings: Vec::new(),
            errors: Vec::new(),
        };
        
        // Get global params
        let global_params = value.get("model")
            .and_then(|m| m.get("global_params"));
        
        // Get training params - use local variables
        let mut batch_size = self.batch_size;
        let mut seq_length = self.seq_length;
        
        if let Some(training) = value.get("training") {
            if let Some(bs) = training.get("batch_size").and_then(|v| v.as_u64()) {
                batch_size = bs;
            }
        }
        
        if let Some(params) = global_params {
            if let Some(seq) = params.get("sequence_length").and_then(|v| v.as_u64()) {
                seq_length = seq;
            }
        }
        
        // Rule 4.1: Compute and validate total parameters
        if let Some(layers) = value.get("model").and_then(|m| m.get("layers")) {
            self.compute_layer_params(layers, &mut result);
        }
        
        // Check against expected params
        if let Some(expected) = global_params.and_then(|p| p.get("total_parameters")).and_then(|v| v.as_u64()) {
            result.expected_params = Some(expected);
            if result.computed_params > 0 {
                let diff = ((result.computed_params as f64 - expected as f64).abs() / expected as f64) * 100.0;
                result.param_diff_percent = Some(diff);
                
                // Rule 4.1: Alert if diff > 1%
                if diff > 1.0 {
                    result.warnings.push(format!(
                        "Parameter count mismatch: computed {} vs expected {} (diff: {:.2}%)",
                        result.computed_params, expected, diff
                    ));
                }
            }
        }
        
        // Rule 4.2: Compute FLOPs
        if let Some(layers) = value.get("model").and_then(|m| m.get("layers")) {
            self.compute_flops(layers, global_params, batch_size, seq_length, &mut result);
        }
        
        // Rule 4.3: Compute memory
        self.compute_memory(global_params, batch_size, seq_length, &mut result);
        
        result.is_coherent = result.errors.is_empty();
        result
    }
    
    /// Compute parameters from layers (Rule 4.1)
    fn compute_layer_params(&self, layers: &Value, result: &mut CoherenceResult) {
        if let Some(layers_arr) = layers.as_array() {
            for layer in layers_arr {
                let layer_type = layer.get("layer_type").and_then(|v| v.as_str()).unwrap_or("custom");
                let params = layer.get("params");
                
                let layer_params = match layer_type {
                    "embedding" => self.embedding_params(params),
                    "dense" => self.dense_params(params),
                    "attention" => self.attention_params(params),
                    "conv" | "conv2d" => self.conv2d_params(params),
                    "moe" => self.moe_params(params),
                    "lstm" => self.lstm_params(params),
                    "gru" => self.gru_params(params),
                    "layer_norm" | "rms_norm" | "batch_norm" => self.norm_params(params),
                    _ => self.custom_params(layer),
                };
                
                result.computed_params += layer_params;
            }
        }
    }
    
    fn embedding_params(&self, params: Option<&Value>) -> u64 {
        let vocab = params.and_then(|p| p.get("vocab_size")).and_then(|v| v.as_u64()).unwrap_or(0);
        let dim = params.and_then(|p| p.get("embedding_dim")).and_then(|v| v.as_u64()).unwrap_or(0);
        vocab * dim
    }
    
    fn dense_params(&self, params: Option<&Value>) -> u64 {
        let in_f = params.and_then(|p| p.get("in_features")).and_then(|v| v.as_u64()).unwrap_or(0);
        let out_f = params.and_then(|p| p.get("out_features")).and_then(|v| v.as_u64()).unwrap_or(0);
        in_f * out_f + out_f // weights + bias
    }
    
    fn attention_params(&self, params: Option<&Value>) -> u64 {
        let hidden = params.and_then(|p| p.get("hidden_size")).and_then(|v| v.as_u64()).unwrap_or(0);
        let _heads = params.and_then(|p| p.get("num_attention_heads")).and_then(|v| v.as_u64()).unwrap_or(1);
        let layers = params.and_then(|p| p.get("num_layers")).and_then(|v| v.as_u64()).unwrap_or(1);
        
        // Q, K, V, O projections per layer
        let params_per_layer = 4 * hidden * hidden;
        layers * params_per_layer
    }
    
    fn conv2d_params(&self, params: Option<&Value>) -> u64 {
        let in_ch = params.and_then(|p| p.get("in_channels")).and_then(|v| v.as_u64()).unwrap_or(0);
        let out_ch = params.and_then(|p| p.get("out_channels")).and_then(|v| v.as_u64()).unwrap_or(0);
        let kernel = params.and_then(|p| p.get("kernel_size")).and_then(|v| v.as_u64()).unwrap_or(3);
        
        in_ch * out_ch * kernel * kernel + out_ch // weights + bias
    }
    
    fn moe_params(&self, params: Option<&Value>) -> u64 {
        let hidden = params.and_then(|p| p.get("hidden_size")).and_then(|v| v.as_u64()).unwrap_or(0);
        let experts = params.and_then(|p| p.get("num_experts")).and_then(|v| v.as_u64()).unwrap_or(8);
        let inter = params.and_then(|p| p.get("intermediate_size")).and_then(|v| v.as_u64()).unwrap_or(hidden * 4);
        let layers = params.and_then(|p| p.get("num_layers")).and_then(|v| v.as_u64()).unwrap_or(1);
        
        // Router + experts (each expert has FFN)
        let router_params = hidden * experts;
        let expert_params = experts * (hidden * inter * 2 + hidden * inter * 2); // up + down proj
        
        layers * (router_params + expert_params)
    }
    
    fn lstm_params(&self, params: Option<&Value>) -> u64 {
        let hidden = params.and_then(|p| p.get("hidden_size")).and_then(|v| v.as_u64()).unwrap_or(0);
        let layers = params.and_then(|p| p.get("num_layers")).and_then(|v| v.as_u64()).unwrap_or(1);
        // LSTM has 4 gates: input, forget, cell, output
        layers * 4 * hidden * hidden
    }
    
    fn gru_params(&self, params: Option<&Value>) -> u64 {
        let hidden = params.and_then(|p| p.get("hidden_size")).and_then(|v| v.as_u64()).unwrap_or(0);
        let layers = params.and_then(|p| p.get("num_layers")).and_then(|v| v.as_u64()).unwrap_or(1);
        // GRU has 3 gates
        layers * 3 * hidden * hidden
    }
    
    fn norm_params(&self, params: Option<&Value>) -> u64 {
        let hidden = params.and_then(|p| p.get("hidden_size")).and_then(|v| v.as_u64()).unwrap_or(0);
        hidden * 2 // weight + bias (for LayerNorm/RMSNorm)
    }
    
    fn custom_params(&self, layer: &Value) -> u64 {
        // Try to get from params or use 0
        layer.get("params")
            .and_then(|p| p.get("param_count"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    }
    
    /// Compute FLOPs (Rule 4.2)
    fn compute_flops(&self, layers: &Value, global_params: Option<&Value>, batch_size: u64, seq_length: u64, result: &mut CoherenceResult) {
        let hidden = global_params
            .and_then(|p| p.get("hidden_size"))
            .and_then(|v| v.as_u64())
            .unwrap_or(768) as f64;
        
        let seq = seq_length as f64;
        let batch = batch_size as f64;
        
        if let Some(layers_arr) = layers.as_array() {
            for layer in layers_arr {
                let layer_type = layer.get("layer_type").and_then(|v| v.as_str()).unwrap_or("custom");
                
                // Check for custom equations first
                if let Some(eqs) = layer.get("custom_equations") {
                    if let Some(fwd_eq) = eqs.get("flops_forward").and_then(|v| v.as_str()) {
                        result.flops_forward += self.eval_equation(fwd_eq, batch, seq, hidden);
                    }
                    if let Some(bwd_eq) = eqs.get("flops_backward").and_then(|v| v.as_str()) {
                        result.flops_backward += self.eval_equation(bwd_eq, batch, seq, hidden);
                    }
                    continue;
                }
                
                // Standard FLOPs calculations
                match layer_type {
                    "dense" => {
                        let in_f = layer.get("params")
                            .and_then(|p| p.get("in_features"))
                            .and_then(|v| v.as_u64())
                            .unwrap_or(hidden as u64) as f64;
                        let out_f = layer.get("params")
                            .and_then(|p| p.get("out_features"))
                            .and_then(|v| v.as_u64())
                            .unwrap_or(hidden as u64) as f64;
                        
                        // Forward: 2 * batch * seq * in * out
                        result.flops_forward += 2.0 * batch * seq * in_f * out_f;
                        // Backward: 2x forward (weights + input gradients)
                        result.flops_backward += 4.0 * batch * seq * in_f * out_f;
                    }
                    "attention" => {
                        // Attention: 4 * batch * seq * hidden^2 per layer
                        let num_layers = layer.get("params")
                            .and_then(|p| p.get("num_layers"))
                            .and_then(|v| v.as_u64())
                            .unwrap_or(1) as f64;
                        
                        let attn_flops = 4.0 * batch * seq * hidden * hidden * num_layers;
                        result.flops_forward += attn_flops;
                        result.flops_backward += 2.0 * attn_flops;
                    }
                    "embedding" => {
                        // Embedding lookup is negligible, but we count it
                        result.flops_forward += batch * seq;
                    }
                    "mamba" | "mamba_block" | "ssm" => {
                        // Mamba SSM FLOPs calculation
                        let d_model = layer.get("params")
                            .and_then(|p| p.get("d_model"))
                            .and_then(|v| v.as_u64())
                            .unwrap_or(hidden as u64) as f64;
                        let d_state = layer.get("params")
                            .and_then(|p| p.get("d_state"))
                            .and_then(|v| v.as_u64())
                            .unwrap_or(16) as f64;
                        let expand = layer.get("params")
                            .and_then(|p| p.get("expand"))
                            .and_then(|v| v.as_u64())
                            .unwrap_or(2) as f64;
                        
                        let d_inner = d_model * expand;
                        
                        // Input projection: 2 * batch * seq * d_model * (d_inner * 2)
                        let in_proj = 2.0 * batch * seq * d_model * (d_inner * 2.0);
                        // Conv1d: 2 * batch * d_inner * seq * 4
                        let conv1d = 2.0 * batch * d_inner * seq * 4.0;
                        // SSM state: 2 * batch * seq * d_inner * d_state
                        let ssm_state = 2.0 * batch * seq * d_inner * d_state;
                        // SSM output: 2 * batch * seq * d_inner
                        let ssm_output = 2.0 * batch * seq * d_inner;
                        // Output projection: 2 * batch * seq * d_inner * d_model
                        let out_proj = 2.0 * batch * seq * d_inner * d_model;
                        
                        let mamba_flops = in_proj + conv1d + ssm_state + ssm_output + out_proj;
                        result.flops_forward += mamba_flops;
                        result.flops_backward += 2.0 * mamba_flops;
                    }
                    "conv" | "conv2d" => {
                        let in_channels = layer.get("params")
                            .and_then(|p| p.get("in_channels"))
                            .and_then(|v| v.as_u64())
                            .unwrap_or(3) as f64;
                        let out_channels = layer.get("params")
                            .and_then(|p| p.get("out_channels"))
                            .and_then(|v| v.as_u64())
                            .unwrap_or(hidden as u64) as f64;
                        let kernel_size = layer.get("params")
                            .and_then(|p| p.get("kernel_size"))
                            .and_then(|v| v.as_u64())
                            .unwrap_or(3) as f64;
                        
                        // Conv2D FLOPs: 2 * batch * out_channels * H * W * in_channels * kernel^2
                        // Simplified: 2 * batch * out * in * k^2 (per position)
                        let conv_flops = 2.0 * batch * out_channels * in_channels * kernel_size * kernel_size;
                        result.flops_forward += conv_flops;
                        result.flops_backward += 2.0 * conv_flops;
                    }
                    _ => {}
                }
            }
        }
    }
    
    /// Evaluate a custom equation string
    fn eval_equation(&self, eq: &str, batch: f64, seq: f64, hidden: f64) -> f64 {
        // Simple equation parser for common patterns
        let eq = eq.replace(" ", "");
        
        // Replace variables
        let eq = eq
            .replace("batch", &format!("{}", batch))
            .replace("seq", &format!("{}", seq))
            .replace("hidden", &format!("{}", hidden));
        
        // Use evalexpr for evaluation
        match evalexpr::eval(&eq) {
            Ok(EvalValue::Float(f)) => f,
            Ok(EvalValue::Int(i)) => i as f64,
            _ => 0.0,
        }
    }
    
    /// Compute memory estimates (Rule 4.3)
    fn compute_memory(&self, global_params: Option<&Value>, batch_size: u64, seq_length: u64, result: &mut CoherenceResult) {
        let hidden = global_params
            .and_then(|p| p.get("hidden_size"))
            .and_then(|v| v.as_u64())
            .unwrap_or(768);
        
        let num_layers = global_params
            .and_then(|p| p.get("num_layers"))
            .and_then(|v| v.as_u64())
            .unwrap_or(12);
        
        let seq = seq_length;
        let batch = batch_size;
        
        // Activation memory: batch * seq * hidden * num_layers * 4 bytes (fp32)
        result.memory_activation_bytes = batch * seq * hidden * num_layers * 4;
        
        // Gradient memory: same as activation
        result.memory_gradient_bytes = result.memory_activation_bytes;
    }
}

impl Default for CoherenceValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    
    #[test]
    fn test_embedding_params() {
        let validator = CoherenceValidator::new();
        let json = json!({
            "model": {
                "layers": [
                    {"id": "embed", "layer_type": "embedding", "params": {"vocab_size": 50000, "embedding_dim": 768}}
                ]
            }
        });
        
        let result = validator.validate(&json);
        assert_eq!(result.computed_params, 50000 * 768);
    }
    
    #[test]
    fn test_param_coherence() {
        let validator = CoherenceValidator::new();
        let json = json!({
            "model": {
                "global_params": {
                    "hidden_size": 768,
                    "num_layers": 12,
                    "total_parameters": 125000000
                },
                "layers": [
                    {"id": "embed", "layer_type": "embedding", "params": {"vocab_size": 50000, "embedding_dim": 768}}
                ]
            }
        });
        
        let result = validator.validate(&json);
        // Should have warning about param mismatch
        assert!(result.warnings.iter().any(|w| w.contains("mismatch")));
    }
    
    #[test]
    fn test_custom_equations() {
        let validator = CoherenceValidator::new()
            .with_batch_size(32)
            .with_seq_length(1024);
        
        let json = json!({
            "model": {
                "global_params": {"hidden_size": 2560, "num_layers": 32},
                "layers": [
                    {
                        "id": "ssm",
                        "layer_type": "custom",
                        "custom_equations": {
                            "flops_forward": "2 * batch * seq * hidden * hidden * 2",
                            "flops_backward": "4 * batch * seq * hidden * hidden"
                        }
                    }
                ]
            }
        });
        
        let result = validator.validate(&json);
        assert!(result.flops_forward > 0.0);
        assert!(result.flops_backward > 0.0);
    }
}
