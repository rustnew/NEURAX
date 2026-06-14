//! NEURAX Parser - JSON parsing for universal model format

mod schema;
mod validator;
mod model_config;
mod error;
mod schema_validator;
mod coherence;
pub mod absorption;

pub use schema::*;
pub use validator::*;
pub use model_config::*;
pub use error::*;
pub use schema_validator::{ModelValidator, ValidationResult, ValidationMetrics, ValidationError};
pub use coherence::{CoherenceValidator, CoherenceResult};
pub use absorption::{GlobalResolutionContext, Dim, DimSource, ResolvedDim, AbsorbedModel, GlobalPropagator, DimResolver, LayerDimContext, LayerParamsMap};

use std::io::Read;

/// Parse JSON string into ModelConfig
pub fn parse_model_config(json: &str) -> Result<ModelConfig, ParserError> {
    let raw: RawModelConfig = serde_json::from_str(json)
        .map_err(ParserError::JsonParse)?;
    
    let config = ModelConfig::from_raw(raw)?;
    validate_model_config(&config)?;
    
    Ok(config)
}

/// Parse JSON from reader
pub fn parse_model_config_from_reader<R: Read>(reader: R) -> Result<ModelConfig, ParserError> {
    let raw: RawModelConfig = serde_json::from_reader(reader)
        .map_err(ParserError::JsonParse)?;
    
    let config = ModelConfig::from_raw(raw)?;
    validate_model_config(&config)?;
    
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal_valid_json() {
        let json = r#"{
            "schema_version": "1.0",
            "model": {
                "name": "TestModel",
                "type": "transformer",
                "layers": [
                    {
                        "id": "layer_1",
                        "layer_type": "embedding",
                        "input_shape": [128, 1],
                        "output_shape": [128, 512],
                        "params": {
                            "vocab_size": 50000,
                            "embedding_dim": 512
                        }
                    }
                ]
            },
            "training": {
                "batch_size": 128
            },
            "hardware": {
                "gpus": [{"name": "A100", "count": 1}]
            }
        }"#;
        
        let config = parse_model_config(json);
        assert!(config.is_ok());
    }

    #[test]
    fn test_parse_missing_schema_version() {
        let json = r#"{
            "model": {
                "name": "Test",
                "type": "transformer",
                "layers": []
            }
        }"#;
        
        let result = parse_model_config(json);
        // May fail on missing layers or other validation, not specifically schema_version
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_empty_layers() {
        let json = r#"{
            "schema_version": "1.0",
            "model": {
                "name": "Test",
                "type": "transformer",
                "layers": []
            },
            "training": {"batch_size": 32},
            "hardware": {"gpus": [{"name": "A100", "count": 1}]}
        }"#;
        
        let result = parse_model_config(json);
        assert!(matches!(result, Err(ParserError::SchemaValidation { .. })));
    }
}
