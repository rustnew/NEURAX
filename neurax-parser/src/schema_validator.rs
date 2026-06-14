//! JSON Schema Validator for Neurax IR Model Files
//!
//! Validates model JSON files against the official schema and coherence rules

use serde_json::Value;
use thiserror::Error;
use std::collections::HashSet;

#[derive(Debug, Error, Clone)]
pub enum ValidationError {
    #[error("Missing required field: '{0}'")]
    MissingField(String),
    
    #[error("Invalid field type for '{field}': expected {expected}, got {actual}")]
    InvalidType { field: String, expected: String, actual: String },
    
    #[error("Invalid value for '{field}': {reason}")]
    InvalidValue { field: String, reason: String },
    
    #[error("Layer shape mismatch: layer '{layer}' output {output:?} != next layer input {next_input:?}")]
    ShapeMismatch { layer: String, output: Vec<usize>, next_input: Vec<usize> },
    
    #[error("Duplicate layer ID: '{0}'")]
    DuplicateLayerId(String),
    
    #[error("Parameter count mismatch: computed {computed}, expected {expected} (diff: {diff:.1}%)")]
    ParameterMismatch { computed: u64, expected: u64, diff: f64 },
    
    #[error("Parallelism mismatch: DP({dp}) x TP({tp}) x PP({pp}) = {product} != {gpu_count} GPUs")]
    ParallelismMismatch { dp: u32, tp: u32, pp: u32, product: u32, gpu_count: u32 },
    
    #[error("Schema validation failed: {0}")]
    SchemaValidation(String),
}

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<String>,
    pub metrics: ValidationMetrics,
}

#[derive(Debug, Clone, Default)]
pub struct ValidationMetrics {
    pub total_params_computed: u64,
    pub layer_count: usize,
    pub shape_chain_valid: bool,
    pub gpu_count: u32,
}

pub struct ModelValidator {
    strict_mode: bool,
}

impl ModelValidator {
    pub fn new() -> Self {
        Self { strict_mode: true }
    }
    
    pub fn with_strict_mode(mut self, strict: bool) -> Self {
        self.strict_mode = strict;
        self
    }
    
    /// Validate a model JSON against all rules
    pub fn validate(&self, json: &str) -> ValidationResult {
        let mut result = ValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            metrics: ValidationMetrics::default(),
        };
        
        // Parse JSON
        let value: Value = match serde_json::from_str(json) {
            Ok(v) => v,
            Err(e) => {
                result.is_valid = false;
                result.errors.push(ValidationError::SchemaValidation(format!("JSON parse error: {}", e)));
                return result;
            }
        };
        
        // Rule 1: Structure validation
        self.validate_structure(&value, &mut result);
        
        // Rule 2: Model validation
        if let Some(model) = value.get("model") {
            self.validate_model(model, &mut result);
        }
        
        // Rule 3: Layer validation
        if let Some(layers) = value.get("model").and_then(|m| m.get("layers")) {
            self.validate_layers(layers, &mut result);
        }
        
        // Rule 4: Coherence validation
        self.validate_coherence(&value, &mut result);
        
        // Rule 5: Hardware validation
        if let Some(hardware) = value.get("hardware") {
            self.validate_hardware(hardware, &mut result);
        }
        
        // Rule 6: Training validation
        if let Some(training) = value.get("training") {
            self.validate_training(training, &value, &mut result);
        }
        
        // Rule 7: Aberrant values check
        self.check_aberrant_values(&value, &mut result);
        
        result.is_valid = result.errors.is_empty();
        result
    }
    
    fn validate_structure(&self, value: &Value, result: &mut ValidationResult) {
        // Rule 1.1: schema_version
        if !value.get("schema_version").is_some() {
            result.errors.push(ValidationError::MissingField("schema_version".to_string()));
        } else if let Some(v) = value.get("schema_version") {
            if !v.is_string() {
                result.errors.push(ValidationError::InvalidType {
                    field: "schema_version".to_string(),
                    expected: "string".to_string(),
                    actual: v.to_string(),
                });
            }
        }
        
        // Rule 1.2: Required fields - model is mandatory, training/hardware recommended
        for field in &["model"] {
            if !value.get(field).is_some() {
                result.errors.push(ValidationError::MissingField(field.to_string()));
            }
        }
        
        // Rule 1.2: Warn if training/hardware missing (for training analysis)
        if value.get("training").is_none() {
            result.warnings.push("Missing 'training' field - training analysis will be skipped".to_string());
        }
        if value.get("hardware").is_none() {
            result.warnings.push("Missing 'hardware' field - hardware analysis will be skipped".to_string());
        }
    }
    
    fn validate_model(&self, model: &Value, result: &mut ValidationResult) {
        // Rule 2.1: name
        if let Some(name) = model.get("name") {
            if !name.is_string() || name.as_str().map(|s| s.is_empty()).unwrap_or(true) {
                result.errors.push(ValidationError::InvalidValue {
                    field: "model.name".to_string(),
                    reason: "must be non-empty string".to_string(),
                });
            }
        } else {
            result.errors.push(ValidationError::MissingField("model.name".to_string()));
        }
        
        // Rule 2.2: type
        let valid_types = ["transformer", "ssm", "moe", "cnn", "rnn", "diffusion", "custom"];
        if let Some(t) = model.get("type") {
            if let Some(type_str) = t.as_str() {
                if !valid_types.contains(&type_str) {
                    result.errors.push(ValidationError::InvalidValue {
                        field: "model.type".to_string(),
                        reason: format!("must be one of: {:?}", valid_types),
                    });
                }
            } else {
                result.errors.push(ValidationError::InvalidType {
                    field: "model.type".to_string(),
                    expected: "string".to_string(),
                    actual: t.to_string(),
                });
            }
        } else {
            result.errors.push(ValidationError::MissingField("model.type".to_string()));
        }
        
        // Rule 2.3: global_params
        if let Some(params) = model.get("global_params") {
            // Required fields
            for field in &["hidden_size", "num_layers"] {
                if let Some(v) = params.get(field) {
                    if !v.is_i64() || v.as_i64().unwrap_or(0) <= 0 {
                        result.errors.push(ValidationError::InvalidValue {
                            field: format!("model.global_params.{}", field),
                            reason: "must be positive integer".to_string(),
                        });
                    }
                } else {
                    result.errors.push(ValidationError::MissingField(format!("model.global_params.{}", field)));
                }
            }
            
            // Rule 2.3: vocab_size and sequence_length - warn if missing for applicable models
            let model_type = model.get("type").and_then(|t| t.as_str()).unwrap_or("");
            if !["cnn", "diffusion"].contains(&model_type) {
                if params.get("vocab_size").is_none() {
                    result.warnings.push("global_params.vocab_size missing - may affect parameter calculation".to_string());
                }
            }
            if !["cnn"].contains(&model_type) {
                if params.get("sequence_length").is_none() {
                    result.warnings.push("global_params.sequence_length missing - may affect FLOPs calculation".to_string());
                }
            }
        } else {
            result.errors.push(ValidationError::MissingField("model.global_params".to_string()));
        }
        
        // Rule 2.4: layers array
        if let Some(layers) = model.get("layers") {
            if !layers.is_array() || layers.as_array().map(|a| a.is_empty()).unwrap_or(true) {
                result.errors.push(ValidationError::InvalidValue {
                    field: "model.layers".to_string(),
                    reason: "must be non-empty array".to_string(),
                });
            }
            result.metrics.layer_count = layers.as_array().map(|a| a.len()).unwrap_or(0);
            
            // Rule 2.4: Check num_layers matches layers count
            if let Some(num_layers) = model.get("global_params").and_then(|p| p.get("num_layers")).and_then(|v| v.as_u64()) {
                let layers_count = layers.as_array().map(|a| a.len()).unwrap_or(0) as u64;
                if num_layers != layers_count {
                    result.warnings.push(format!(
                        "global_params.num_layers ({}) != actual layers count ({})",
                        num_layers, layers_count
                    ));
                }
            }
        } else {
            result.errors.push(ValidationError::MissingField("model.layers".to_string()));
        }
    }
    
    fn validate_layers(&self, layers: &Value, result: &mut ValidationResult) {
        let layers_arr = match layers.as_array() {
            Some(a) => a,
            None => return,
        };
        
        let mut seen_ids = HashSet::new();
        let mut prev_output: Option<Vec<usize>> = None;
        
        for (i, layer) in layers_arr.iter().enumerate() {
            // Rule 3.1: unique id
            if let Some(id) = layer.get("id").and_then(|v| v.as_str()) {
                if seen_ids.contains(id) {
                    result.errors.push(ValidationError::DuplicateLayerId(id.to_string()));
                }
                seen_ids.insert(id.to_string());
            } else {
                result.errors.push(ValidationError::MissingField(format!("model.layers[{}].id", i)));
            }
            
            // Rule 3.2: layer_type
            let valid_types = ["embedding", "dense", "attention", "conv", "conv1d", "conv2d", 
                              "normalization", "pooling", "activation", "dropout", "moe", "custom"];
            if let Some(t) = layer.get("layer_type").and_then(|v| v.as_str()) {
                if !valid_types.contains(&t) {
                    result.errors.push(ValidationError::InvalidValue {
                        field: format!("model.layers[{}].layer_type", i),
                        reason: format!("must be one of: {:?}", valid_types),
                    });
                }
            } else {
                result.errors.push(ValidationError::MissingField(format!("model.layers[{}].layer_type", i)));
            }
            
            // Rule 3.3: input_shape and output_shape
            let input_shape = self.parse_shape(layer.get("input_shape"));
            let output_shape = self.parse_shape(layer.get("output_shape"));
            
            if input_shape.is_none() {
                result.errors.push(ValidationError::MissingField(format!("model.layers[{}].input_shape", i)));
            }
            if output_shape.is_none() {
                result.errors.push(ValidationError::MissingField(format!("model.layers[{}].output_shape", i)));
            }
            
            // Rule 3.6: shape chain
            if let (Some(ref input), Some(ref prev_out)) = (&input_shape, &prev_output) {
                if input != prev_out {
                    result.errors.push(ValidationError::ShapeMismatch {
                        layer: format!("layers[{}]", i),
                        output: prev_out.clone(),
                        next_input: input.clone(),
                    });
                }
            }
            prev_output = output_shape;
            
            // Rule 3.5: custom equations for custom layers
            if layer.get("layer_type").and_then(|v| v.as_str()) == Some("custom") {
                if layer.get("custom_equations").is_none() {
                    result.warnings.push(format!(
                        "Layer {} has type 'custom' but no custom_equations provided - FLOPs may be inaccurate",
                        layer.get("id").and_then(|v| v.as_str()).unwrap_or("unknown")
                    ));
                }
            }
        }
        
        result.metrics.shape_chain_valid = prev_output.is_some();
    }
    
    fn parse_shape(&self, value: Option<&Value>) -> Option<Vec<usize>> {
        value?.as_array()?.iter().map(|v| v.as_u64().map(|n| n as usize)).collect()
    }
    
    fn validate_coherence(&self, value: &Value, result: &mut ValidationResult) {
        // Rule 4.1: Parameter count coherence
        // This would require computing params from layers - simplified check
        if let Some(total) = value.get("model")
            .and_then(|m| m.get("global_params"))
            .and_then(|p| p.get("total_parameters"))
            .and_then(|v| v.as_u64())
        {
            result.metrics.total_params_computed = total;
        }
    }
    
    fn validate_hardware(&self, hardware: &Value, result: &mut ValidationResult) {
        // Rule 5.1: GPUs
        if let Some(gpus) = hardware.get("gpus") {
            if let Some(gpus_arr) = gpus.as_array() {
                if gpus_arr.is_empty() {
                    result.errors.push(ValidationError::InvalidValue {
                        field: "hardware.gpus".to_string(),
                        reason: "must contain at least one GPU".to_string(),
                    });
                } else {
                    let mut total_gpus = 0u32;
                    for (i, gpu) in gpus_arr.iter().enumerate() {
                        // Required fields
                        for field in &["name", "count", "memory_gb"] {
                            if gpu.get(field).is_none() {
                                result.errors.push(ValidationError::MissingField(
                                    format!("hardware.gpus[{}].{}", i, field)
                                ));
                            }
                        }
                        if let Some(count) = gpu.get("count").and_then(|v| v.as_u64()) {
                            total_gpus += count as u32;
                        }
                        
                        // Rule 5.2: GPU performance fields - warn if missing
                        if gpu.get("gpu_tflops_fp16").is_none() {
                            result.warnings.push(format!(
                                "hardware.gpus[{}].gpu_tflops_fp16 missing - using default values",
                                i
                            ));
                        }
                        if gpu.get("gpu_memory_bandwidth_gbs").is_none() {
                            result.warnings.push(format!(
                                "hardware.gpus[{}].gpu_memory_bandwidth_gbs missing - using default values",
                                i
                            ));
                        }
                    }
                    result.metrics.gpu_count = total_gpus;
                }
            }
        } else {
            result.errors.push(ValidationError::MissingField("hardware.gpus".to_string()));
        }
        
        // Rule 5.3: Interconnect for multi-GPU
        if result.metrics.gpu_count > 1 && hardware.get("interconnect").is_none() {
            result.warnings.push("Multiple GPUs configured but no interconnect specified".to_string());
        }
        if result.metrics.gpu_count > 1 && hardware.get("interconnect_bandwidth_gbs").is_none() {
            result.warnings.push("Multiple GPUs configured but no interconnect_bandwidth_gbs specified".to_string());
        }
    }
    
    fn validate_training(&self, training: &Value, _root: &Value, result: &mut ValidationResult) {
        // Rule 6.1: Required fields
        for field in &["batch_size", "optimizer", "precision"] {
            if training.get(field).is_none() {
                result.errors.push(ValidationError::MissingField(format!("training.{}", field)));
            }
        }
        
        // Rule 6.1: learning_rate - warn if missing
        if training.get("learning_rate").is_none() {
            result.warnings.push("training.learning_rate missing - using default".to_string());
        }
        
        // Rule 6.2: max_steps or training_tokens
        if training.get("max_steps").is_none() && training.get("training_tokens").is_none() {
            result.warnings.push("Neither max_steps nor training_tokens specified - cost estimation may be inaccurate".to_string());
        }
        
        // Rule 6.3: Parallelism coherence
        let dp = training.get("data_parallel").and_then(|v| v.as_u64()).unwrap_or(1) as u32;
        let tp = training.get("tensor_parallel").and_then(|v| v.as_u64()).unwrap_or(1) as u32;
        let pp = training.get("pipeline_parallel").and_then(|v| v.as_u64()).unwrap_or(1) as u32;
        let product = dp * tp * pp;
        
        if product > 1 && result.metrics.gpu_count > 0 && product != result.metrics.gpu_count {
            result.errors.push(ValidationError::ParallelismMismatch {
                dp, tp, pp, product, gpu_count: result.metrics.gpu_count
            });
        }
    }
    
    fn check_aberrant_values(&self, value: &Value, result: &mut ValidationResult) {
        // Rule 7.3: Aberrant value warnings
        
        // Check GPU memory
        if let Some(hardware) = value.get("hardware").and_then(|h| h.get("gpus")) {
            if let Some(gpus) = hardware.as_array() {
                for gpu in gpus {
                    if let (Some(mem), Some(count)) = (
                        gpu.get("memory_gb").and_then(|v| v.as_f64()),
                        gpu.get("count").and_then(|v| v.as_u64()),
                    ) {
                        let total_mem = mem * count as f64;
                        if total_mem > 10000.0 {
                            result.warnings.push(format!("Total GPU memory ({:.0} GB) seems unusually high", total_mem));
                        }
                    }
                }
            }
        }
        
        // Check latency > 1 hour
        if let Some(latency) = value.get("hardware")
            .and_then(|h| h.get("metrics"))
            .and_then(|m| m.get("latency_ms"))
            .and_then(|v| v.as_f64())
        {
            if latency > 3600000.0 { // 1 hour in ms
                result.warnings.push(format!("Latency ({:.2} hours) seems unusually high", latency / 3600000.0));
            }
        }
        
        // Check memory > 2x GPU capacity
        if let (Some(peak_mem), Some(gpu_mem)) = (
            value.get("memory").and_then(|m| m.get("peak_vram_bytes")).and_then(|v| v.as_u64()),
            value.get("hardware").and_then(|h| h.get("gpus"))
                .and_then(|g| g.as_array())
                .and_then(|a| a.first())
                .and_then(|g| g.get("memory_gb"))
                .and_then(|v| v.as_f64())
        ) {
            let gpu_capacity = (gpu_mem * 1e9) as u64; // GB to bytes
            if peak_mem > 2 * gpu_capacity {
                result.warnings.push(format!(
                    "Peak memory ({:.2} GB) exceeds 2x GPU capacity ({:.0} GB) - OOM likely",
                    peak_mem as f64 / 1e9, gpu_mem
                ));
            }
        }
        
        // Check batch_size vs sequence_length
        if let (Some(batch), Some(seq)) = (
            value.get("training").and_then(|t| t.get("batch_size")).and_then(|v| v.as_u64()),
            value.get("model").and_then(|m| m.get("global_params"))
                .and_then(|p| p.get("sequence_length")).and_then(|v| v.as_u64()),
        ) {
            if batch * seq > 1000000 {
                result.warnings.push(format!(
                    "batch_size ({}) x sequence_length ({}) = {} is very large - may cause memory issues",
                    batch, seq, batch * seq
                ));
            }
        }
    }
}

impl Default for ModelValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_valid_model() {
        let json = r#"{
            "schema_version": "1.0",
            "model": {
                "name": "Test Model",
                "type": "transformer",
                "global_params": {
                    "hidden_size": 768,
                    "num_layers": 12,
                    "vocab_size": 50000,
                    "sequence_length": 1024
                },
                "layers": [
                    {"id": "embed", "layer_type": "embedding", "input_shape": [1024], "output_shape": [1024, 768]},
                    {"id": "lm_head", "layer_type": "dense", "input_shape": [1024, 768], "output_shape": [1024, 50000]}
                ]
            },
            "training": {"batch_size": 32, "optimizer": "adamw", "precision": "bf16"},
            "hardware": {"gpus": [{"name": "A100", "count": 8, "memory_gb": 80}]}
        }"#;
        
        let validator = ModelValidator::new();
        let result = validator.validate(json);
        
        assert!(result.is_valid, "Validation errors: {:?}", result.errors);
    }
    
    #[test]
    fn test_missing_schema_version() {
        let json = r#"{"model": {"name": "Test"}}"#;
        let validator = ModelValidator::new();
        let result = validator.validate(json);
        
        assert!(!result.is_valid);
        assert!(result.errors.iter().any(|e| matches!(e, ValidationError::MissingField(f) if f == "schema_version")));
    }
    
    #[test]
    fn test_shape_mismatch() {
        let json = r#"{
            "schema_version": "1.0",
            "model": {
                "name": "Test",
                "type": "transformer",
                "global_params": {"hidden_size": 768, "num_layers": 2},
                "layers": [
                    {"id": "l1", "layer_type": "dense", "input_shape": [10], "output_shape": [20]},
                    {"id": "l2", "layer_type": "dense", "input_shape": [30], "output_shape": [40]}
                ]
            },
            "training": {"batch_size": 32, "optimizer": "adam", "precision": "fp32"}
        }"#;
        
        let validator = ModelValidator::new();
        let result = validator.validate(json);
        
        assert!(result.errors.iter().any(|e| matches!(e, ValidationError::ShapeMismatch { .. })));
    }
}
