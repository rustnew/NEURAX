//! Report IR pass

use super::{
    AllMetrics, Diagnostic, DiagnosticCategory, DiagnosticCode, GradientMemoryEntry,
    KvCacheEntry, Priority, Recommendation, RecommendationCategory, ReportIR, ReportMetadata,
    Severity,
};
use crate::architecture::ArchitectureIR;
use crate::compute::ComputeIR;
use crate::cost::CostIR;
use crate::error::NeuraxError;
use crate::graph::GraphIR;
use crate::hardware::HardwareIR;
use crate::memory::MemoryIR;
use crate::operator::OperatorIR;
use crate::parallelism::ParallelismIR;
use crate::tensor::TensorIR;
use crate::traits::ReportPass as ReportPassTrait;
use crate::NeuraxContext;

/// Input for report generation
pub struct ReportInput<'a> {
    pub arch: &'a ArchitectureIR,
    pub graph: &'a GraphIR,
    pub tensor: &'a TensorIR,
    pub operator: &'a OperatorIR,
    pub compute: &'a ComputeIR,
    pub memory: &'a MemoryIR,
    pub parallelism: &'a ParallelismIR,
    pub hardware: &'a HardwareIR,
    pub cost: &'a CostIR,
}

/// Report pass implementation
pub struct ReportPass;

impl<'a> ReportPassTrait<'a> for ReportPass {
    type Input = ReportInput<'a>;
    type Output = ReportIR;
    type PassError = NeuraxError;

    fn name(&self) -> &'static str {
        "ReportIR"
    }

    fn build_report(
        &self,
        input: &Self::Input,
        ctx: &NeuraxContext,
    ) -> Result<Self::Output, Self::PassError> {
        let mut report = ReportIR::default();

        // Get metric groups config for filtering
        let groups = &ctx.config.metrics_config.groups;

        // Set metadata
        report.metadata = ReportMetadata {
            generated_at: chrono::Utc::now(),
            neurax_version: env!("CARGO_PKG_VERSION").to_string(),
            model_name: input
                .arch
                .model_name
                .clone()
                .unwrap_or("Unknown".to_string()),
            model_type: input.arch.model_type.as_str().to_string(),
            schema_version: ctx.config.schema_version.clone(),
            analysis_time_ms: 0, // Would be measured
        };

        // Consolidate metrics based on enabled groups
        report.metrics = AllMetrics {
            // === Structure Metrics (1-5) ===
            total_parameters: if groups.structure {
                input.arch.metrics.total_parameters
            } else {
                0
            },
            num_layers: if groups.structure {
                input.arch.metrics.num_layers
            } else {
                0
            },
            model_type: input.arch.model_type.as_str().to_string(),
            hidden_size: ctx.config.model.global_params.embedding_dim.unwrap_or(0),
            vocab_size: ctx.config.model.global_params.vocab_size.unwrap_or(0) as u64,

            // === Graph Metrics (6-8) ===
            graph_depth: if groups.structure {
                input.graph.metrics.graph_depth
            } else {
                0
            },
            total_operations: if groups.structure {
                input.graph.metrics.total_operations
            } else {
                0
            },
            critical_path_length: input.graph.metrics.graph_depth, // Simplified

            // === Tensor Metrics (9-12) ===
            tensor_resolution_ratio: input.tensor.metrics.resolution_ratio,
            unresolved_dim_count: input.tensor.metrics.unresolved_dim_count,
            total_tensor_count: input.tensor.metrics.total_tensor_count,
            largest_tensor_bytes: input.tensor.metrics.largest_tensor_bytes,

            // === Compute Metrics (13-18) ===
            total_flops: if groups.compute {
                input.compute.metrics.total_flops
            } else {
                0.0
            },
            forward_flops: if groups.compute {
                input.compute.metrics.forward_flops
            } else {
                0.0
            },
            backward_flops: if groups.compute {
                input.compute.metrics.backward_flops
            } else {
                0.0
            },
            flops_per_token: if groups.compute {
                input.compute.metrics.flops_per_token
            } else {
                0.0
            },
            flops_incremental_decode: if groups.compute {
                input.compute.metrics.flops_incremental_decode
            } else {
                0.0
            },
            arithmetic_intensity: if groups.compute {
                input.compute.metrics.arithmetic_intensity
            } else {
                0.0
            },
            ops_distribution: input.operator.metrics.op_type_distribution.clone(),

            // === Memory Metrics (19-25) ===
            peak_vram_bytes: if groups.memory {
                input.memory.metrics.peak_vram_bytes
            } else {
                0
            },
            parameter_memory_bytes: if groups.memory {
                input.memory.metrics.parameter_memory_bytes
            } else {
                0
            },
            activation_memory_bytes: if groups.memory {
                input.memory.metrics.activation_memory_bytes
            } else {
                0
            },
            gradient_memory_bytes: if groups.memory {
                input.memory.metrics.gradient_memory_bytes
            } else {
                0
            },
            optimizer_state_bytes: if groups.memory {
                input.memory.metrics.optimizer_state_bytes
            } else {
                0
            },
            max_batch_size_fit: if groups.memory {
                input.memory.metrics.max_batch_size_fit
            } else {
                0
            },
            memory_fragmentation: input.memory.metrics.fragmentation_estimate,

            // === Parallelism Metrics (26-30) ===
            data_parallel_efficiency: if groups.parallelism {
                input.parallelism.metrics.data_parallel_efficiency
            } else {
                0.0
            },
            communication_overhead: if groups.parallelism {
                input.parallelism.metrics.communication_overhead
            } else {
                0.0
            },
            optimal_gpu_count: if groups.parallelism {
                input.parallelism.metrics.optimal_gpu_count
            } else {
                0
            },
            pipeline_stages: input.parallelism.metrics.pipeline_stages.unwrap_or(0),
            tensor_parallel_degree: ctx.config.training.parallelism.tensor_parallel,

            // === Hardware Metrics (31-35) ===
            latency_ms: if groups.performance {
                input.hardware.metrics.latency_ms
            } else {
                0.0
            },
            throughput_tokens_per_s: if groups.performance {
                input.hardware.metrics.throughput_tokens_per_s
            } else {
                0.0
            },
            gpu_utilization: if groups.performance {
                input.hardware.metrics.gpu_utilization
            } else {
                0.0
            },
            bottleneck: input.hardware.metrics.bottleneck.as_str().to_string(),
            roofline_position: input.hardware.metrics.roofline_position,

            // === Hardware Config (from JSON) ===
            gpu_name: input.hardware.gpu_profile.name.clone(),
            gpu_count: ctx.config.hardware.total_gpu_count() as usize,
            gpu_memory_gb: input.hardware.gpu_profile.vram_gb as f64,
            gpu_tflops_fp16: input.hardware.gpu_profile.peak_tflops,
            gpu_memory_bandwidth_gbs: input.hardware.gpu_profile.memory_bandwidth,
            interconnect: ctx.config.hardware.interconnect.clone(),
            interconnect_bandwidth_gbs: ctx.config.hardware.interconnect_bandwidth_gbs,

            // === Cost Metrics (36-40) ===
            training_cost_usd: if groups.cost {
                input.cost.metrics.training_cost_usd
            } else {
                0.0
            },
            training_time_hours: if groups.cost {
                input.cost.metrics.training_time_hours
            } else {
                0.0
            },
            energy_kwh: if groups.cost {
                input.cost.metrics.energy_kwh
            } else {
                0.0
            },
            co2_kg: if groups.cost {
                input.cost.metrics.co2_kg
            } else {
                0.0
            },
            cost_per_million_tokens_usd: if groups.cost {
                input.cost.metrics.cost_per_million_tokens_usd
            } else {
                0.0
            },

            // === Confidence & Quality (41-43) ===
            confidence_score: report.confidence_score,
            custom_layer_count: input.operator.metrics.custom_op_count,
            diagnostic_count: 0, // Will be updated after diagnostics generated

            // === Per-Layer Breakdown Maps ===
            params_per_layer: if groups.structure {
                input.arch.metrics.params_per_layer.clone()
            } else {
                std::collections::HashMap::new()
            },
            flops_per_layer: if groups.compute {
                input.compute.metrics.flops_per_layer.clone()
            } else {
                std::collections::HashMap::new()
            },
            latency_per_layer: if groups.performance {
                input
                    .hardware
                    .per_layer_timings
                    .iter()
                    .map(|t| (t.layer_id.clone(), t.total_time_ms))
                    .collect()
            } else {
                std::collections::HashMap::new()
            },

            // === Rich per-layer metrics ===
            gradient_memory_per_layer: build_gradient_memory_per_layer(
                &input.arch.metrics.params_per_layer,
                input.memory.metrics.gradient_memory_bytes,
                input.memory.metrics.activation_memory_bytes,
            ),
            kv_cache_scaling: build_kv_cache_scaling(
                ctx.config.model.global_params.num_layers.unwrap_or(0) as usize,
                ctx.config.model.global_params.embedding_dim.unwrap_or(0),
            ),
        };

        // Generate diagnostics
        report.diagnostics = generate_diagnostics(&report.metrics, &input.memory);
        report.metrics.diagnostic_count = report.diagnostics.len(); // Update count

        // Generate recommendations
        report.recommendations =
            generate_recommendations(&report.metrics, &input.memory, &input.hardware);

        // Collect warnings
        report.warnings = collect_warnings(&input);

        // Compute confidence score
        report.confidence_score = compute_confidence_score(
            input.tensor.metrics.resolution_ratio,
            input.operator.metrics.custom_op_count > 0,
            has_custom_formulas(&input.arch),
            input.tensor.metrics.unresolved_dim_count == 0,
        );

        Ok(report)
    }
}

/// Compute confidence score based on tensor resolution and custom layers
/// Formula from impl_2.md:
/// - Base score = tensor_resolution_ratio
/// - If custom layers without formulas: score *= 0.60
/// - If dimensions not all concrete: score *= 0.80
fn compute_confidence_score(
    tensor_resolution: f32,
    has_custom_layers: bool,
    custom_layers_have_formulas: bool,
    dims_all_concrete: bool,
) -> f64 {
    let mut score = 1.0f64;

    // Factor 1: Tensor shape resolution ratio
    score *= tensor_resolution as f64;

    // Factor 2: Custom layers without formulas reduce confidence
    if has_custom_layers && !custom_layers_have_formulas {
        score *= 0.60;
    }

    // Factor 3: Unresolved symbolic/dynamic dimensions reduce confidence
    if !dims_all_concrete {
        score *= 0.80;
    }

    score.clamp(0.0, 1.0)
}

/// Check if custom layers have custom equations defined
fn has_custom_formulas(arch: &ArchitectureIR) -> bool {
    arch.layers
        .iter()
        .any(|layer| layer.custom_equations.is_some())
}

fn generate_diagnostics(metrics: &AllMetrics, memory: &MemoryIR) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();

    // E001: Memory overflow check
    if metrics.peak_vram_bytes > memory.metrics.gpu_vram_bytes {
        diagnostics.push(Diagnostic {
            category: DiagnosticCategory::MemoryOverflow,
            severity: Severity::Critical,
            code: DiagnosticCode::E001,
            message: format!(
                "Peak VRAM ({:.1} GB) exceeds GPU memory ({:.1} GB)",
                metrics.peak_vram_bytes as f64 / 1e9,
                memory.metrics.gpu_vram_bytes as f64 / 1e9
            ),
            layer_id: None,
            suggestion: Some("Enable gradient checkpointing or use model parallelism".to_string()),
            precision_impact: 0.0,
        });
    }

    // W005: Memory close to limit (80-100%)
    if metrics.peak_vram_bytes > memory.metrics.gpu_vram_bytes * 80 / 100
        && metrics.peak_vram_bytes <= memory.metrics.gpu_vram_bytes
    {
        diagnostics.push(Diagnostic {
            category: DiagnosticCategory::MemoryOverflow,
            severity: Severity::Warning,
            code: DiagnosticCode::W005,
            message: format!(
                "Memory usage ({:.1}%) close to GPU limit",
                metrics.peak_vram_bytes as f64 / memory.metrics.gpu_vram_bytes as f64 * 100.0
            ),
            layer_id: None,
            suggestion: Some("Consider gradient checkpointing to reduce memory".to_string()),
            precision_impact: 0.0,
        });
    }

    // W006: Low GPU utilization
    if metrics.gpu_utilization < 0.5 {
        diagnostics.push(Diagnostic {
            category: DiagnosticCategory::ArchitectureInefficiency,
            severity: Severity::Warning,
            code: DiagnosticCode::W006,
            message: format!(
                "Low GPU utilization ({:.1}%)",
                metrics.gpu_utilization * 100.0
            ),
            layer_id: None,
            suggestion: Some("Increase batch size or consider tensor parallelism".to_string()),
            precision_impact: 0.2,
        });
    }

    // W006: High communication overhead
    if metrics.communication_overhead > 0.3 {
        diagnostics.push(Diagnostic {
            category: DiagnosticCategory::ParallelismSuboptimal,
            severity: Severity::Warning,
            code: DiagnosticCode::W006,
            message: format!(
                "High communication overhead ({:.1}%)",
                metrics.communication_overhead * 100.0
            ),
            layer_id: None,
            suggestion: Some("Consider tensor parallelism or faster interconnect".to_string()),
            precision_impact: 0.1,
        });
    }

    diagnostics
}

fn generate_recommendations(
    metrics: &AllMetrics,
    memory: &MemoryIR,
    _hardware: &HardwareIR,
) -> Vec<Recommendation> {
    let mut recommendations = Vec::new();

    // Memory optimization
    if metrics.peak_vram_bytes > memory.metrics.gpu_vram_bytes * 80 / 100 {
        recommendations.push(Recommendation {
            category: RecommendationCategory::MemoryOptimization,
            title: "Enable Gradient Checkpointing".to_string(),
            description: "Reduce activation memory by recomputing during backward pass".to_string(),
            impact: format!(
                "Save ~{:.1} GB VRAM",
                metrics.activation_memory_bytes as f64 / 1e9 * 0.7
            ),
            priority: Priority::High,
        });
    }

    // Parallelism
    if metrics.optimal_gpu_count > 1 && metrics.data_parallel_efficiency < 0.8 {
        recommendations.push(Recommendation {
            category: RecommendationCategory::Parallelism,
            title: "Use Hybrid Parallelism".to_string(),
            description: "Combine data, tensor, and pipeline parallelism for better scaling"
                .to_string(),
            impact: format!("Improve efficiency to ~90%"),
            priority: Priority::Medium,
        });
    }

    // Hardware
    if metrics.bottleneck == "memory-bound" {
        recommendations.push(Recommendation {
            category: RecommendationCategory::Hardware,
            title: "Consider Higher Bandwidth GPU".to_string(),
            description: "Model is memory-bound; H100 SXM offers 3.35 TB/s bandwidth".to_string(),
            impact: "Potential 2-3x speedup".to_string(),
            priority: Priority::Medium,
        });
    }

    // Cost
    if metrics.training_cost_usd > 10000.0 {
        recommendations.push(Recommendation {
            category: RecommendationCategory::Cost,
            title: "Optimize Training Duration".to_string(),
            description: "Consider spot instances or reserved capacity for cost savings"
                .to_string(),
            impact: format!("Save up to 70% on GPU costs"),
            priority: Priority::Low,
        });
    }

    recommendations
}

fn collect_warnings(input: &ReportInput) -> Vec<String> {
    let mut warnings = Vec::new();

    if input.operator.metrics.custom_op_count > 0 {
        warnings.push(format!(
            "Model contains {} custom operations with estimated FLOPs",
            input.operator.metrics.custom_op_count
        ));
    }

    warnings
}

/// Distribute gradient + activation memory proportionally across layers
fn build_gradient_memory_per_layer(
    params_per_layer: &std::collections::HashMap<String, u64>,
    total_gradient_bytes: u64,
    total_activation_bytes: u64,
) -> Vec<GradientMemoryEntry> {
    if params_per_layer.is_empty() {
        return Vec::new();
    }
    let total_params: u64 = params_per_layer.values().sum();
    if total_params == 0 {
        return Vec::new();
    }
    let mut entries: Vec<GradientMemoryEntry> = params_per_layer
        .iter()
        .map(|(name, &params)| {
            let share = params as f64 / total_params as f64;
            GradientMemoryEntry {
                name: name.clone(),
                forward: (total_activation_bytes as f64 * share) as u64,
                backward: (total_gradient_bytes as f64 * share) as u64,
            }
        })
        .collect();
    entries.sort_by(|a, b| b.backward.cmp(&a.backward));
    entries
}

/// Compute KV cache size for representative sequence lengths (attention models only)
fn build_kv_cache_scaling(num_layers: usize, hidden_size: usize) -> Vec<KvCacheEntry> {
    if num_layers == 0 || hidden_size == 0 {
        return Vec::new();
    }
    let seq_lengths: &[u32] = &[256, 512, 1024, 2048, 4096, 8192, 16384];
    seq_lengths
        .iter()
        .map(|&seq| {
            // KV cache: 2 (K+V) * num_layers * hidden_size * seq_len * 2 bytes (BF16)
            let value = 2u64 * num_layers as u64 * hidden_size as u64 * seq as u64 * 2;
            KvCacheEntry { seq, value }
        })
        .collect()
}
