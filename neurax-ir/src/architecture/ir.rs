//! Architecture IR structures

use neurax_parser::{Layer, ModelType, GlobalParams, TrainingConfig, HardwareConfig};
use std::collections::HashMap;

/// Architecture IR - premier dialecte du pipeline NEURAX
#[derive(Debug, Clone)]
pub struct ArchitectureIR {
    /// Type de modèle normalisé
    pub model_type: ModelType,
    /// Nom du modèle
    pub model_name: Option<String>,
    /// Liste normalisée des layers
    pub layers: Vec<LayerDef>,
    /// Paramètres globaux du modèle
    pub global_params: GlobalParams,
    /// Configuration d'entraînement
    pub training_config: TrainingConfig,
    /// Configuration hardware
    pub hardware_config: HardwareConfig,
    /// Métriques calculées par cette IR
    pub metrics: ArchitectureMetrics,
    /// État de complétion des métriques
    pub metrics_done: bool,
}

impl Default for ArchitectureIR {
    fn default() -> Self {
        Self {
            model_type: ModelType::Transformer,
            model_name: None,
            layers: Vec::new(),
            global_params: GlobalParams::default(),
            training_config: TrainingConfig::default(),
            hardware_config: HardwareConfig::default(),
            metrics: ArchitectureMetrics::default(),
            metrics_done: false,
        }
    }
}

/// Layer definition
#[derive(Debug, Clone)]
pub struct LayerDef {
    pub id: String,
    pub layer_type: neurax_parser::LayerType,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub params: neurax_parser::LayerParams,
    pub custom_equations: Option<neurax_parser::CustomEquations>,
    /// Calculated parameter count
    pub param_count: u64,
}

impl From<&Layer> for LayerDef {
    fn from(layer: &Layer) -> Self {
        Self {
            id: layer.id.clone(),
            layer_type: layer.layer_type,
            input_shape: layer.input_shape.clone(),
            output_shape: layer.output_shape.clone(),
            params: layer.params.clone(),
            custom_equations: layer.custom_equations.clone(),
            param_count: 0, // Calculated later
        }
    }
}

/// Architecture metrics (Métriques 1-2)
#[derive(Debug, Clone, Default)]
pub struct ArchitectureMetrics {
    /// Métrique 1: Nombre total de paramètres
    pub total_parameters: u64,
    /// Métrique 2: Nombre de layers
    pub num_layers: usize,
    /// Métrique 2 (type): Type de modèle
    pub model_type_info: String,
    /// Paramètres par layer
    pub params_per_layer: HashMap<String, u64>,
    /// Nombre de layers par type
    pub layers_by_type: HashMap<String, usize>,
}

impl ArchitectureMetrics {
    /// Check if metrics are valid
    pub fn is_valid(&self) -> bool {
        self.total_parameters > 0 && self.num_layers > 0
    }
}
