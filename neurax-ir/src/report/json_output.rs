//! JSON output serialization with all 35 metrics

use chrono::{DateTime, Utc};
use serde::Serialize;
use std::collections::HashMap;

/// Complete JSON output structure with all 35 metrics
#[derive(Debug, Clone, Serialize)]
pub struct JsonOutput {
    pub schema_version: String,
    pub generated_at: DateTime<Utc>,
    pub neurax_version: String,
    pub analysis_time_ms: u64,
    pub input_file: String,
    pub model: ModelInfo,
    pub metrics: MetricsOutput,
    pub diagnostics: Vec<DiagnosticOutput>,
    pub recommendations: Vec<RecommendationOutput>,
    pub warnings: Vec<String>,
}

/// Model information section
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub model_type: String,
    pub total_parameters: u64,
    pub num_layers: usize,
    pub vocab_size: Option<u64>,
    pub sequence_length: Option<u64>,
    pub hidden_size: Option<u64>,
    pub num_attention_heads: Option<u64>,
    pub num_key_value_heads: Option<u64>,
    pub intermediate_size: Option<u64>,
}

/// All 35 metrics organized by category
#[derive(Debug, Clone, Serialize)]
pub struct MetricsOutput {
    // === STRUCTURE METRICS (5) ===
    pub structure: StructureMetricsOutput,

    // === GRAPH METRICS (2) ===
    pub graph: GraphMetricsOutput,

    // === COMPUTE METRICS (5) ===
    pub compute: ComputeMetricsOutput,

    // === MEMORY METRICS (6) ===
    pub memory: MemoryMetricsOutput,

    // === PARALLELISM METRICS (3) ===
    pub parallelism: ParallelismMetricsOutput,

    // === PERFORMANCE METRICS (4) ===
    pub performance: PerformanceMetricsOutput,

    // === COST METRICS (5) ===
    pub cost: CostMetricsOutput,

    // === HARDWARE METRICS (5) ===
    pub hardware: HardwareMetricsOutput,

    // === DYNAMIC METRICS (20) ===
    pub dynamic: Option<DynamicMetricsOutput>,
}

/// Structure metrics (5 metrics)
#[derive(Debug, Clone, Serialize)]
pub struct StructureMetricsOutput {
    pub total_parameters: u64,
    pub num_layers: usize,
    pub model_type: String,
    pub params_per_layer: HashMap<String, u64>,
    pub layers_by_type: HashMap<String, usize>,
}

/// Graph metrics (2 metrics)
#[derive(Debug, Clone, Serialize)]
pub struct GraphMetricsOutput {
    pub graph_depth: usize,
    pub total_operations: usize,
}

/// Compute metrics (5 metrics)
#[derive(Debug, Clone, Serialize)]
pub struct ComputeMetricsOutput {
    pub total_flops: f64,
    pub forward_flops: f64,
    pub backward_flops: f64,
    pub flops_per_token: f64,
    /// FLOPs pour inférence incrémentale (1 token avec KV cache plein)
    pub flops_incremental_decode: f64,
    pub arithmetic_intensity: f64,
    pub flops_per_layer: HashMap<String, f64>,
    pub op_type_distribution: HashMap<String, usize>,
}

/// Memory metrics (6 metrics)
#[derive(Debug, Clone, Serialize)]
pub struct MemoryMetricsOutput {
    pub peak_vram_bytes: u64,
    pub peak_vram_gb: f64,
    pub parameter_memory_bytes: u64,
    pub parameter_memory_gb: f64,
    pub activation_memory_bytes: u64,
    pub activation_memory_gb: f64,
    pub gradient_memory_bytes: u64,
    pub gradient_memory_gb: f64,
    pub optimizer_state_bytes: u64,
    pub optimizer_state_gb: f64,
    pub max_batch_size_fit: u32,
    pub oom_risk: String,
}

/// Parallelism metrics (3 metrics)
#[derive(Debug, Clone, Serialize)]
pub struct ParallelismMetricsOutput {
    pub data_parallel_efficiency: f64,
    pub communication_overhead: f64,
    pub optimal_gpu_count: u32,
    pub data_parallel: u32,
    pub tensor_parallel: u32,
    pub pipeline_parallel: u32,
}

/// Performance metrics (4 metrics)
#[derive(Debug, Clone, Serialize)]
pub struct PerformanceMetricsOutput {
    pub latency_ms: f64,
    pub latency_seconds: f64,
    pub throughput_tokens_per_s: f64,
    pub gpu_utilization: f64,
    pub gpu_utilization_percent: f64,
    pub bottleneck: String,
    pub latency_per_layer: HashMap<String, f64>,
}

/// Cost metrics (5 metrics)
#[derive(Debug, Clone, Serialize)]
pub struct CostMetricsOutput {
    pub training_cost_usd: f64,
    pub training_time_hours: f64,
    pub energy_kwh: f64,
    pub co2_kg: f64,
    pub cost_per_million_tokens_usd: f64,
    pub gpu_hours: f64,
    pub provider: String,
}

/// Hardware metrics (5 metrics)
#[derive(Debug, Clone, Serialize)]
pub struct HardwareMetricsOutput {
    pub gpu_name: String,
    pub gpu_count: usize,
    pub gpu_memory_gb: f64,
    pub gpu_tflops_fp16: f64,
    pub gpu_memory_bandwidth_gbs: f64,
    pub interconnect: String,
    pub interconnect_bandwidth_gbs: f64,
}

/// Dynamic metrics (20 metrics: M36-M55)
#[derive(Debug, Clone, Serialize)]
pub struct DynamicMetricsOutput {
    // === VIRTUAL MEMORY (M36-M42) ===
    pub virtual_memory: VirtualMemoryMetricsOutput,

    // === STABILITY (M43-M49) ===
    pub stability: StabilityMetricsOutput,

    // === BEHAVIORAL (M50-M55) ===
    pub behavioral: BehavioralMetricsOutput,
}

/// Virtual Memory metrics (M36-M42)
#[derive(Debug, Clone, Serialize)]
pub struct VirtualMemoryMetricsOutput {
    /// M36: Fragmentation overhead (GB)
    pub fragmentation_overhead_gb: f64,
    /// M37: Fragmentation percentage
    pub fragmentation_pct: f64,
    /// M38: Defrag savings (GB)
    pub defrag_savings_gb: f64,
    /// M39: Virtual savings (GB)
    pub virtual_savings_gb: f64,
    /// M40: Virtual savings percentage
    pub virtual_savings_pct: f64,
    /// M41: Peak VRAM with defrag (GB)
    pub peak_vram_with_defrag_gb: f64,
    /// M42: Peak VRAM with virtual (GB)
    pub peak_vram_with_virtual_gb: f64,
    /// Recommended strategy
    pub recommended_strategy: String,
    /// Confidence score
    pub confidence: f64,
}

/// Stability metrics (M43-M49)
#[derive(Debug, Clone, Serialize)]
pub struct StabilityMetricsOutput {
    /// M44: Lyapunov exponent mean
    pub lyapunov_exponent_mean: f64,
    /// M45: Chaos index [0,1]
    pub chaos_index: f64,
    /// M46: High risk layers count
    pub high_risk_layers_count: usize,
    /// M47: FP32 required percentage
    pub fp32_required_pct: f64,
    /// M48: Global robustness score
    pub global_robustness_score: f64,
    /// M49: FP32 fallback memory overhead (GB)
    pub fp32_fallback_memory_overhead_gb: f64,
    /// Confidence score
    pub confidence: f64,
}

/// Behavioral metrics (M50-M55)
#[derive(Debug, Clone, Serialize)]
pub struct BehavioralMetricsOutput {
    /// M50: Expert load imbalance [0,1]
    pub expert_load_imbalance: f64,
    /// M51: Memory contention score [0,1]
    pub memory_contention_score: f64,
    /// M52: Cache locality score [0,1]
    pub cache_locality_score: f64,
    /// M53: Numerical sensitivity [0,1]
    pub numerical_sensitivity: f64,
    /// M54: Load balance efficiency [%]
    pub load_balance_efficiency: f64,
    /// M55: Memory bank conflict rate [%]
    pub memory_bank_conflict_rate: f64,
    /// Confidence score
    pub prediction_confidence: f64,
}

/// Diagnostic output
#[derive(Debug, Clone, Serialize)]
pub struct DiagnosticOutput {
    pub category: String,
    pub severity: String,
    pub message: String,
    pub layer_id: Option<String>,
    pub suggestion: Option<String>,
}

/// Recommendation output
#[derive(Debug, Clone, Serialize)]
pub struct RecommendationOutput {
    pub category: String,
    pub priority: String,
    pub title: String,
    pub description: String,
    pub impact: String,
}

impl JsonOutput {
    /// Create a new JSON output from report IR
    pub fn from_report(
        report: &crate::report::ReportIR,
        input_file: &str,
        analysis_time_ms: u64,
    ) -> Self {
        Self {
            schema_version: report.metadata.schema_version.clone(),
            generated_at: report.metadata.generated_at,
            neurax_version: report.metadata.neurax_version.clone(),
            analysis_time_ms,
            input_file: input_file.to_string(),
            model: ModelInfo {
                name: report.metadata.model_name.clone(),
                model_type: report.metadata.model_type.clone(),
                total_parameters: report.metrics.total_parameters,
                num_layers: report.metrics.num_layers,
                vocab_size: None,
                sequence_length: None,
                hidden_size: None,
                num_attention_heads: None,
                num_key_value_heads: None,
                intermediate_size: None,
            },
            metrics: MetricsOutput::from_all_metrics(&report.metrics),
            diagnostics: report
                .diagnostics
                .iter()
                .map(DiagnosticOutput::from)
                .collect(),
            recommendations: report
                .recommendations
                .iter()
                .map(RecommendationOutput::from)
                .collect(),
            warnings: report.warnings.clone(),
        }
    }

    /// Create JSON output with dynamic metrics (M36-M55)
    pub fn from_report_with_dynamic(
        report: &crate::report::ReportIR,
        input_file: &str,
        analysis_time_ms: u64,
        dynamic: &crate::dynamic::DynamicResults,
    ) -> Self {
        let mut output = Self::from_report(report, input_file, analysis_time_ms);
        output.metrics.dynamic = Some(DynamicMetricsOutput {
            virtual_memory: VirtualMemoryMetricsOutput {
                fragmentation_overhead_gb: dynamic
                    .virtual_memory
                    .as_ref()
                    .map(|v| v.fragmentation_overhead_gb)
                    .unwrap_or(0.0),
                fragmentation_pct: dynamic
                    .virtual_memory
                    .as_ref()
                    .map(|v| v.fragmentation_pct)
                    .unwrap_or(0.0),
                defrag_savings_gb: dynamic
                    .virtual_memory
                    .as_ref()
                    .map(|v| v.defrag_savings_gb)
                    .unwrap_or(0.0),
                virtual_savings_gb: dynamic
                    .virtual_memory
                    .as_ref()
                    .map(|v| v.virtual_savings_gb)
                    .unwrap_or(0.0),
                virtual_savings_pct: dynamic
                    .virtual_memory
                    .as_ref()
                    .map(|v| v.virtual_savings_pct)
                    .unwrap_or(0.0),
                peak_vram_with_defrag_gb: dynamic
                    .virtual_memory
                    .as_ref()
                    .map(|v| v.peak_vram_with_defrag_gb)
                    .unwrap_or(0.0),
                peak_vram_with_virtual_gb: dynamic
                    .virtual_memory
                    .as_ref()
                    .map(|v| v.peak_vram_with_virtual_gb)
                    .unwrap_or(0.0),
                recommended_strategy: dynamic
                    .virtual_memory
                    .as_ref()
                    .map(|v| format!("{:?}", v.recommended_strategy))
                    .unwrap_or("NoAction".to_string()),
                confidence: dynamic
                    .virtual_memory
                    .as_ref()
                    .map(|v| v.confidence)
                    .unwrap_or(0.0),
            },
            stability: StabilityMetricsOutput {
                lyapunov_exponent_mean: dynamic
                    .stability
                    .as_ref()
                    .map(|s| s.lyapunov_exponent_mean)
                    .unwrap_or(0.0),
                chaos_index: dynamic
                    .stability
                    .as_ref()
                    .map(|s| s.chaos_index)
                    .unwrap_or(0.0),
                high_risk_layers_count: dynamic
                    .stability
                    .as_ref()
                    .map(|s| s.high_risk_layers.len())
                    .unwrap_or(0),
                fp32_required_pct: dynamic
                    .stability
                    .as_ref()
                    .map(|s| s.fp32_required_pct)
                    .unwrap_or(0.0),
                global_robustness_score: dynamic
                    .stability
                    .as_ref()
                    .map(|s| s.global_robustness_score)
                    .unwrap_or(1.0),
                fp32_fallback_memory_overhead_gb: dynamic
                    .stability
                    .as_ref()
                    .map(|s| s.fp32_fallback_memory_overhead_gb)
                    .unwrap_or(0.0),
                confidence: dynamic
                    .stability
                    .as_ref()
                    .map(|s| s.confidence)
                    .unwrap_or(0.0),
            },
            behavioral: BehavioralMetricsOutput {
                expert_load_imbalance: dynamic
                    .behavioral
                    .as_ref()
                    .map(|b| b.expert_load_imbalance)
                    .unwrap_or(0.0),
                memory_contention_score: dynamic
                    .behavioral
                    .as_ref()
                    .map(|b| b.memory_contention_score)
                    .unwrap_or(0.0),
                cache_locality_score: dynamic
                    .behavioral
                    .as_ref()
                    .map(|b| b.cache_locality_score)
                    .unwrap_or(0.0),
                numerical_sensitivity: dynamic
                    .behavioral
                    .as_ref()
                    .map(|b| b.numerical_sensitivity)
                    .unwrap_or(0.0),
                load_balance_efficiency: dynamic
                    .behavioral
                    .as_ref()
                    .map(|b| b.load_balance_efficiency)
                    .unwrap_or(100.0),
                memory_bank_conflict_rate: dynamic
                    .behavioral
                    .as_ref()
                    .map(|b| b.memory_bank_conflict_rate)
                    .unwrap_or(0.0),
                prediction_confidence: dynamic
                    .behavioral
                    .as_ref()
                    .map(|b| b.prediction_confidence)
                    .unwrap_or(0.0),
            },
        });
        output
    }

    /// Serialize to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Serialize to JSON bytes
    pub fn to_json_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec_pretty(self)
    }
}

impl MetricsOutput {
    pub fn from_all_metrics(metrics: &crate::report::AllMetrics) -> Self {
        Self {
            structure: StructureMetricsOutput {
                total_parameters: metrics.total_parameters,
                num_layers: metrics.num_layers,
                model_type: metrics.model_type.clone(),
                params_per_layer: metrics.params_per_layer.clone(),
                layers_by_type: HashMap::new(),
            },
            graph: GraphMetricsOutput {
                graph_depth: metrics.graph_depth,
                total_operations: metrics.total_operations,
            },
            compute: ComputeMetricsOutput {
                total_flops: metrics.total_flops,
                forward_flops: metrics.forward_flops,
                backward_flops: metrics.backward_flops,
                flops_per_token: metrics.flops_per_token,
                flops_incremental_decode: metrics.flops_incremental_decode,
                arithmetic_intensity: metrics.arithmetic_intensity,
                flops_per_layer: metrics.flops_per_layer.clone(),
                op_type_distribution: metrics.ops_distribution.clone(),
            },
            memory: MemoryMetricsOutput {
                peak_vram_bytes: metrics.peak_vram_bytes,
                peak_vram_gb: metrics.peak_vram_bytes as f64 / 1e9,
                parameter_memory_bytes: metrics.parameter_memory_bytes,
                parameter_memory_gb: metrics.parameter_memory_bytes as f64 / 1e9,
                activation_memory_bytes: metrics.activation_memory_bytes,
                activation_memory_gb: metrics.activation_memory_bytes as f64 / 1e9,
                gradient_memory_bytes: metrics.gradient_memory_bytes,
                gradient_memory_gb: metrics.gradient_memory_bytes as f64 / 1e9,
                optimizer_state_bytes: metrics.optimizer_state_bytes,
                optimizer_state_gb: metrics.optimizer_state_bytes as f64 / 1e9,
                max_batch_size_fit: metrics.max_batch_size_fit,
                oom_risk: "low".to_string(),
            },
            parallelism: ParallelismMetricsOutput {
                data_parallel_efficiency: metrics.data_parallel_efficiency,
                communication_overhead: metrics.communication_overhead,
                optimal_gpu_count: metrics.optimal_gpu_count,
                data_parallel: 1,
                tensor_parallel: 1,
                pipeline_parallel: 1,
            },
            performance: PerformanceMetricsOutput {
                latency_ms: metrics.latency_ms,
                latency_seconds: metrics.latency_ms / 1000.0,
                throughput_tokens_per_s: metrics.throughput_tokens_per_s,
                gpu_utilization: metrics.gpu_utilization,
                gpu_utilization_percent: metrics.gpu_utilization * 100.0,
                bottleneck: metrics.bottleneck.clone(),
                latency_per_layer: metrics.latency_per_layer.clone(),
            },
            cost: CostMetricsOutput {
                training_cost_usd: metrics.training_cost_usd,
                training_time_hours: metrics.training_time_hours,
                energy_kwh: metrics.energy_kwh,
                co2_kg: metrics.co2_kg,
                cost_per_million_tokens_usd: metrics.cost_per_million_tokens_usd,
                gpu_hours: 0.0,
                provider: "unknown".to_string(),
            },
            hardware: HardwareMetricsOutput {
                gpu_name: metrics.gpu_name.clone(),
                gpu_count: metrics.gpu_count,
                gpu_memory_gb: metrics.gpu_memory_gb,
                gpu_tflops_fp16: metrics.gpu_tflops_fp16,
                gpu_memory_bandwidth_gbs: metrics.gpu_memory_bandwidth_gbs,
                interconnect: metrics.interconnect.clone(),
                interconnect_bandwidth_gbs: metrics.interconnect_bandwidth_gbs,
            },
            dynamic: None,
        }
    }
}

impl From<&crate::report::Diagnostic> for DiagnosticOutput {
    fn from(d: &crate::report::Diagnostic) -> Self {
        Self {
            category: format!("{:?}", d.category).to_lowercase(),
            severity: format!("{:?}", d.severity).to_lowercase(),
            message: d.message.clone(),
            layer_id: d.layer_id.clone(),
            suggestion: d.suggestion.clone(),
        }
    }
}

impl From<&crate::report::Recommendation> for RecommendationOutput {
    fn from(r: &crate::report::Recommendation) -> Self {
        Self {
            category: format!("{:?}", r.category).to_lowercase(),
            priority: format!("{:?}", r.priority).to_lowercase(),
            title: r.title.clone(),
            description: r.description.clone(),
            impact: r.impact.clone(),
        }
    }
}
