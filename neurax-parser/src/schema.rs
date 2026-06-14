//! JSON schema structures for serde deserialization

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Raw model configuration from JSON
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct RawModelConfig {
    pub schema_version: String,
    pub model: RawModel,
    #[serde(default)]
    pub training: RawTraining,
    #[serde(default)]
    pub hardware: RawHardware,
    #[serde(default)]
    pub data: RawData,
    #[serde(default)]
    pub metrics_config: RawMetricsConfig,
    #[serde(default)]
    pub cost_config: RawCostConfig,
}

/// Raw model definition
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct RawModel {
    pub name: Option<String>,
    #[serde(rename = "type")]
    pub model_type: String,
    pub layers: Vec<RawLayer>,
    #[serde(default)]
    pub global_params: RawGlobalParams,
}

/// Raw layer definition
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct RawLayer {
    pub id: String,
    #[serde(rename = "layer_type")]
    pub layer_type: String,
    #[serde(default)]
    pub input_shape: Vec<usize>,
    #[serde(default)]
    pub output_shape: Vec<usize>,
    #[serde(default)]
    pub params: RawLayerParams,
    #[serde(default)]
    pub custom_equations: Option<RawCustomEquations>,
}

/// Raw layer parameters (flexible key-value)
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct RawLayerParams {
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl RawLayerParams {
    pub fn get_usize(&self, key: &str) -> Option<usize> {
        self.extra.get(key).and_then(|v| {
            if let Some(n) = v.as_u64() {
                Some(n as usize)
            } else if let Some(s) = v.as_str() {
                s.parse().ok()
            } else {
                None
            }
        })
    }
    
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.extra.get(key).and_then(|v| {
            if let Some(n) = v.as_f64() {
                Some(n)
            } else if let Some(n) = v.as_u64() {
                Some(n as f64)
            } else if let Some(n) = v.as_i64() {
                Some(n as f64)
            } else if let Some(s) = v.as_str() {
                s.parse().ok()
            } else {
                None
            }
        })
    }
    
    pub fn get_string(&self, key: &str) -> Option<String> {
        self.extra.get(key).and_then(|v| v.as_str().map(|s| s.to_string()))
    }
    
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.extra.get(key).and_then(|v| v.as_bool())
    }
    
    pub fn get_string_vec(&self, key: &str) -> Option<Vec<String>> {
        self.extra.get(key).and_then(|v| {
            v.as_array().map(|arr| {
                arr.iter()
                    .filter_map(|item| item.as_str().map(|s| s.to_string()))
                    .collect()
            })
        })
    }
    
    pub fn get_usize_vec(&self, key: &str) -> Option<Vec<usize>> {
        self.extra.get(key).and_then(|v| {
            v.as_array().map(|arr| {
                arr.iter()
                    .filter_map(|item| {
                        if let Some(n) = item.as_u64() {
                            Some(n as usize)
                        } else if let Some(s) = item.as_str() {
                            s.parse().ok()
                        } else {
                            None
                        }
                    })
                    .collect()
            })
        })
    }
    
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.extra.get(key).and_then(|v| {
            if let Some(n) = v.as_u64() {
                Some(n)
            } else if let Some(s) = v.as_str() {
                s.parse().ok()
            } else {
                None
            }
        })
    }
}

/// Raw custom equations
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct RawCustomEquations {
    pub flops_forward: Option<String>,
    pub memory_activation: Option<String>,
    pub gradient: Option<String>,
    #[serde(flatten)]
    pub extra: HashMap<String, String>,
}

/// Raw global parameters
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub struct RawGlobalParams {
    pub sequence_length: Option<usize>,
    pub vocab_size: Option<usize>,
    pub embedding_dim: Option<usize>,
    pub num_experts: Option<usize>,
    pub diffusion_timesteps: Option<usize>,
    pub graph_message_dim: Option<usize>,
    /// Total number of transformer/repeatable layers in the full model.
    /// Use this when the JSON only lists a subset of representative layers.
    pub num_layers: Option<u64>,
    /// Number of dense (non-MoE) layers in MoE models like DeepSeek-V3
    pub num_dense_layers: Option<u64>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Raw training configuration
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub struct RawTraining {
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default)]
    pub optimizer: Option<String>,
    #[serde(default)]
    pub learning_rate: Option<f64>,
    #[serde(default = "default_precision")]
    pub precision: String,
    #[serde(default)]
    pub gradient_checkpointing: bool,
    #[serde(default)]
    pub zero_stage: u8,
    #[serde(default)]
    pub max_steps: usize,
    #[serde(default)]
    pub warmup_steps: usize,
    #[serde(default)]
    pub parallelism: RawParallelism,
}

fn default_batch_size() -> usize { 32 }
fn default_precision() -> String { "fp32".to_string() }

/// Raw parallelism configuration
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub struct RawParallelism {
    #[serde(default = "default_one")]
    pub data_parallel: u32,
    #[serde(default = "default_one")]
    pub tensor_parallel: u32,
    #[serde(default = "default_one")]
    pub pipeline_parallel: u32,
}

fn default_one() -> u32 { 1 }

/// Raw hardware configuration
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub struct RawHardware {
    #[serde(default)]
    pub gpus: Vec<RawGpu>,
    #[serde(default)]
    pub interconnect: Option<String>,
    #[serde(default, alias = "interconnect_bandwidth_gbs")]
    pub interconnect_bandwidth_gb_s: Option<f64>,
}

/// Raw GPU definition
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct RawGpu {
    pub name: String,
    #[serde(default = "default_gpu_count")]
    pub count: u32,
    #[serde(default)]
    pub memory_gb: Option<u64>,
    #[serde(default)]
    pub tflops_fp16: Option<f64>,
    #[serde(default)]
    pub tflops_fp32: Option<f64>,
    #[serde(default)]
    pub tflops_fp8: Option<f64>,
    #[serde(default, alias = "memory_bandwidth_gbs")]
    pub memory_bandwidth_gb_s: Option<f64>,
    #[serde(default)]
    pub tensor_cores: Option<bool>,
    #[serde(default)]
    pub nvlink: Option<bool>,
}

fn default_gpu_count() -> u32 { 1 }

/// Raw data configuration
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub struct RawData {
    #[serde(default)]
    pub input_shape: Vec<usize>,
    #[serde(default = "default_dtype")]
    pub dtype: String,
    #[serde(default)]
    pub vocab_size: Option<usize>,
    #[serde(default)]
    pub num_classes: Option<usize>,
    #[serde(default)]
    pub image_channels: Option<usize>,
    #[serde(default)]
    pub image_height: Option<usize>,
    #[serde(default)]
    pub image_width: Option<usize>,
}

fn default_dtype() -> String { "float32".to_string() }

/// Raw metrics configuration
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub struct RawMetricsConfig {
    #[serde(default = "default_true")]
    pub calculate_all: bool,
    #[serde(default)]
    pub groups: RawMetricGroups,
}

fn default_true() -> bool { true }

/// Raw metric groups
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct RawMetricGroups {
    #[serde(default = "default_true")]
    pub structure: bool,
    #[serde(default = "default_true")]
    pub compute: bool,
    #[serde(default = "default_true")]
    pub memory: bool,
    #[serde(default = "default_true")]
    pub training: bool,
    #[serde(default = "default_true")]
    pub parallelism: bool,
    #[serde(default = "default_true")]
    pub hardware: bool,
    #[serde(default = "default_true")]
    pub performance: bool,
    #[serde(default = "default_true")]
    pub cost: bool,
}

impl Default for RawMetricGroups {
    fn default() -> Self {
        Self {
            structure: true,
            compute: true,
            memory: true,
            training: true,
            parallelism: true,
            hardware: true,
            performance: true,
            cost: true,
        }
    }
}

/// Raw cost configuration
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub struct RawCostConfig {
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub gpu_hour_usd: Option<f64>,
    #[serde(default)]
    pub energy_kwh_usd: Option<f64>,
    #[serde(default = "default_pue")]
    pub pue_factor: f64,
}

fn default_pue() -> f64 { 1.2 }
