//! Runner utilities

use neurax_parser::ModelConfig;
use neurax_ir::NeuraxError;
use crate::{AnalysisResult, run_analysis};

/// Run analysis from JSON string
pub fn analyze_json(json: &str) -> Result<AnalysisResult, NeuraxError> {
    let config = neurax_parser::parse_model_config(json)?;
    run_analysis(config)
}

/// Run analysis from file
pub fn analyze_file(path: &std::path::Path) -> Result<AnalysisResult, NeuraxError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| NeuraxError::Internal(format!("Failed to read file: {}", e)))?;
    analyze_json(&content)
}

/// Validate JSON without full analysis
pub fn validate_json(json: &str) -> Result<ModelConfig, NeuraxError> {
    let config = neurax_parser::parse_model_config(json)?;
    neurax_parser::validate_model_config(&config)?;
    Ok(config)
}

/// Get model summary without full analysis
pub fn get_model_summary(config: &ModelConfig) -> ModelSummary {
    ModelSummary {
        name: config.model.name.clone().unwrap_or("Unknown".to_string()),
        model_type: config.model.model_type.as_str().to_string(),
        num_layers: config.model.layers.len(),
        batch_size: config.training.batch_size,
        precision: config.training.precision.clone(),
        gpu_count: config.hardware.total_gpu_count(),
    }
}

/// Quick model summary
#[derive(Debug, Clone)]
pub struct ModelSummary {
    pub name: String,
    pub model_type: String,
    pub num_layers: usize,
    pub batch_size: usize,
    pub precision: String,
    pub gpu_count: u32,
}
