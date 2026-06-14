//! Parser errors

use thiserror::Error;

/// Parser error types
#[derive(Debug, Error)]
pub enum ParserError {
    #[error("JSON parsing error: {0}")]
    JsonParse(#[from] serde_json::Error),
    
    #[error("Schema validation error: {details}")]
    SchemaValidation {
        details: String,
        field: Option<String>,
    },
    
    #[error("Invalid layer type: '{0}'")]
    InvalidLayerType(String),
    
    #[error("Invalid model type: '{0}'")]
    InvalidModelType(String),
    
    #[error("Shape mismatch: {details}")]
    ShapeMismatch { details: String },
    
    #[error("Missing required field: '{0}'")]
    MissingField(String),
    
    #[error("Invalid value for field '{field}': {reason}")]
    InvalidValue { field: String, reason: String },
}

impl ParserError {
    pub fn schema_validation(details: &str) -> Self {
        Self::SchemaValidation {
            details: details.to_string(),
            field: None,
        }
    }
    
    pub fn schema_validation_with_field(details: &str, field: &str) -> Self {
        Self::SchemaValidation {
            details: details.to_string(),
            field: Some(field.to_string()),
        }
    }
}
