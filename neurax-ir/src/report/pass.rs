//! Report IR pass

use super::{
    AllMetrics, Diagnostic, DiagnosticCategory, DiagnosticCode, GradientMemoryEntry, KvCacheEntry,
    Priority, Recommendation, RecommendationCategory, ReportIR, ReportMetadata, Severity,
};
use crate::architecture::ArchitectureIR;
use crate::compute::ComputeIR;
use crate::cost::CostIR;
use crate::dynamic::StabilityMetrics;
use crate::error::NeuraxError;
use crate::graph::GraphIR;
use crate::hardware::HardwareIR;
use crate::memory::MemoryIR;
use crate::operator::OperatorIR;
use crate::parallelism::ParallelismIR;
use crate::tensor::TensorIR;
use crate::traits::ReportPass as ReportPassTrait;
use crate::NeuraxContext;
use neurax_parser::{LayerType, ModelType};

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
            active_parameters: if groups.structure {
                input.arch.metrics.active_parameters
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

            // === Compute Metrics (13-22) ===
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
            macs: if groups.compute {
                input.compute.metrics.macs
            } else {
                0.0
            },
            total_step_flops: if groups.compute {
                input.compute.metrics.total_step_flops
            } else {
                0.0
            },
            flops_per_batch: if groups.compute {
                input.compute.metrics.flops_per_batch
            } else {
                0.0
            },
            bytes_accessed: if groups.compute {
                input.compute.metrics.bytes_accessed
            } else {
                0
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
            memory_fragmentation_pct: input.memory.metrics.fragmentation_estimate * 100.0,
            oom_risk: if input.memory.metrics.peak_vram_bytes > input.memory.metrics.gpu_vram_bytes
            {
                "critical".to_string()
            } else if input.memory.metrics.peak_vram_bytes
                > input.memory.metrics.gpu_vram_bytes * 80 / 100
            {
                "high".to_string()
            } else {
                "low".to_string()
            },

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
            tensor_core_utilization: if groups.performance {
                input.hardware.metrics.tensor_core_utilization
            } else {
                0.0
            },
            effective_tflops: if groups.performance {
                input.hardware.metrics.effective_tflops
            } else {
                0.0
            },
            samples_per_s: if groups.performance {
                input.hardware.metrics.samples_per_s
            } else {
                0.0
            },

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
            training_budget_stated: groups.cost && input.cost.effective_steps > 0,

            // === Tensor Metrics (Tensor IR) ===
            activation_memory_bytes_tensor: input.tensor.metrics.activation_memory_bytes,

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
        report.diagnostics = generate_diagnostics(&report.metrics, input.memory);
        report
            .diagnostics
            .extend(check_shape_consistency(input.graph));
        report
            .diagnostics
            .extend(detect_architecture_features(&ctx.config));
        // Diagnostics a pass pushed directly during the pipeline (currently
        // E003 "custom formula failed" and W001 "custom layer without a
        // formula", both from OperatorPass) — collected here into the same
        // Vec everything else lands in. Before this, ctx.diagnostics was
        // written to but never read anywhere, so a model that hit either
        // condition got no warning in its report at all.
        report.diagnostics.extend(ctx.diagnostics.lock().drain(..));
        report.metrics.diagnostic_count = report.diagnostics.len(); // Update count

        // Generate recommendations
        report.recommendations =
            generate_recommendations(&report.metrics, input.memory, input.hardware, &ctx.gpu_db);

        // Collect warnings
        report.warnings = collect_warnings(input);

        // Compute confidence score
        report.confidence_score = compute_confidence_score(
            input.tensor.metrics.resolution_ratio,
            input.operator.metrics.custom_op_count > 0,
            has_custom_formulas(input.arch),
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

/// Flag a layer whose declared input shape does not match the single
/// upstream layer feeding it.
///
/// Restricted to nodes with exactly one incoming edge: a merge point (concat,
/// add from a skip connection) legitimately combines tensors of different
/// shape, so comparing against just one of several predecessors would be a
/// false positive rather than a real inconsistency. Shapes are compared
/// skipping the batch dimension, and only when both sides are non-empty and
/// of equal rank — an omitted shape is a missing-input problem (see
/// `unresolved_dim_count`), not a mismatch.
fn check_shape_consistency(graph: &GraphIR) -> Vec<Diagnostic> {
    use petgraph::visit::EdgeRef;
    use petgraph::Direction;

    let mut diagnostics = Vec::new();

    for &idx in &graph.topo_order {
        let mut incoming = graph.dag.edges_directed(idx, Direction::Incoming);
        let (Some(edge), None) = (incoming.next(), incoming.next()) else {
            continue; // no predecessor, or a merge — not checked here
        };
        let Some(node) = graph.dag.node_weight(idx) else {
            continue;
        };
        let Some(prev) = graph.dag.node_weight(edge.source()) else {
            continue;
        };

        let Some(declared_input) = node.input_shapes.first() else {
            continue;
        };
        if declared_input.is_empty() || prev.output_shape.is_empty() {
            continue;
        }
        if declared_input.len() != prev.output_shape.len() {
            continue;
        }
        let matches = declared_input
            .iter()
            .skip(1)
            .zip(prev.output_shape.iter().skip(1))
            .all(|(a, b)| a == b);
        if matches {
            continue;
        }

        diagnostics.push(Diagnostic {
            category: DiagnosticCategory::ShapeInference,
            severity: Severity::Warning,
            code: DiagnosticCode::W007,
            message: format!(
                "Layer '{}' declares input shape {:?}, but '{}' right before it \
                 produces {:?}. Downstream shapes past this point are unreliable.",
                node.layer_id, declared_input, prev.layer_id, prev.output_shape
            ),
            layer_id: Some(node.layer_id.clone()),
            suggestion: Some(format!(
                "Recompute '{}' from '{}'s actual output rather than a fixed value.",
                node.layer_id, prev.layer_id
            )),
            precision_impact: 0.5,
        });
    }

    diagnostics
}

/// I001/I002: plain observations about architecture features actually
/// present in the config — not warnings or hints, just facts a designer
/// reading the report benefits from having surfaced explicitly rather than
/// inferred from raw layer params.
fn detect_architecture_features(config: &neurax_parser::ModelConfig) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();

    // I001: Grouped Query Attention — fewer KV heads than query heads on at
    // least one attention layer. Only fires when both counts are actually
    // stated; a layer that gives num_kv_heads without num_heads (or vice
    // versa) has nothing to compare, so it's skipped rather than guessed.
    let gqa_layer = config.model.layers.iter().find(|l| {
        matches!(l.layer_type, LayerType::Attention)
            && match (l.params.num_heads, l.params.num_kv_heads) {
                (Some(heads), Some(kv_heads)) => kv_heads < heads,
                _ => false,
            }
    });
    if let Some(layer) = gqa_layer {
        diagnostics.push(Diagnostic {
            category: DiagnosticCategory::ArchitectureInefficiency,
            severity: Severity::Info,
            code: DiagnosticCode::I001,
            message: format!(
                "Grouped Query Attention detected on layer '{}' ({} query heads, {} KV heads).",
                layer.id,
                layer.params.num_heads.unwrap_or(0),
                layer.params.num_kv_heads.unwrap_or(0)
            ),
            layer_id: Some(layer.id.clone()),
            suggestion: None,
            precision_impact: 0.0,
        });
    }

    // I002: Mixture of Experts — at least one layer routes to experts. Kept
    // as a layer-level check (not just `model.model_type == Moe`) since a
    // hybrid architecture can mix MoE layers into an otherwise dense model.
    if config
        .model
        .layers
        .iter()
        .any(|l| matches!(l.layer_type, LayerType::MoE))
    {
        diagnostics.push(Diagnostic {
            category: DiagnosticCategory::ArchitectureInefficiency,
            severity: Severity::Info,
            code: DiagnosticCode::I002,
            message: "Mixture of Experts (MoE) layer(s) detected.".to_string(),
            layer_id: None,
            suggestion: None,
            precision_impact: 0.0,
        });
    }

    diagnostics
}

/// Hyperparameter recommendations grounded in published, family-general or
/// family-specific results — never a fabricated "optimal value". Computed
/// separately from `generate_diagnostics` because it needs `ctx.config` and
/// (for H007) the Dynamic phase's stability output, which runs concurrently
/// with — and so isn't available inside — `build_report` itself; the caller
/// merges this into `report.diagnostics` once both branches have finished.
pub fn generate_hyperparameter_diagnostics(
    ctx: &NeuraxContext,
    total_parameters: u64,
    stability: Option<&StabilityMetrics>,
) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    let training = &ctx.config.training;

    // H006: warmup — near-universal practice across families and
    // optimizers; not specific to any one architecture the way the other
    // two checks below are.
    if training.warmup_steps == 0 && training.max_steps > 0 {
        diagnostics.push(Diagnostic {
            category: DiagnosticCategory::Configuration,
            severity: Severity::Hint,
            code: DiagnosticCode::H006,
            message: format!(
                "No learning-rate warmup configured ({} steps planned, warmup_steps = 0). \
                 A short LR warmup is close to universal practice at this scale — it protects \
                 against early-training instability, particularly with Adam-family optimizers.",
                training.max_steps
            ),
            layer_id: None,
            suggestion: Some("Set warmup_steps to roughly 1-5% of max_steps.".to_string()),
            precision_impact: 0.0,
        });
    }

    // H007: learning rate vs. the Lipschitz-based stability bound
    // (`lr < 2/L`, classical gradient-descent stability theory). L itself is
    // a per-layer heuristic estimate (see StabilityAnalysisPass), so this is
    // directional, not an exact bound — but the composition genuinely
    // depends on this model's own architecture (depth, attention seq_len,
    // MoE routing, SSM state size), not a family-wide constant.
    if let Some(sta) = stability {
        if training.learning_rate > sta.recommended_max_learning_rate {
            diagnostics.push(Diagnostic {
                category: DiagnosticCategory::Configuration,
                severity: Severity::Hint,
                code: DiagnosticCode::H007,
                message: format!(
                    "learning_rate ({:.2e}) exceeds the estimated stability bound (~{:.2e}, \
                     from lr < 2/L with L ≈ {:.2} estimated from this architecture's layer \
                     composition). Directional heuristic, not an exact bound — but a large \
                     excess is a real risk factor for early divergence.",
                    training.learning_rate,
                    sta.recommended_max_learning_rate,
                    sta.network_lipschitz_estimate
                ),
                layer_id: None,
                suggestion: Some(format!(
                    "Consider a learning_rate at or below ~{:.2e}.",
                    sta.recommended_max_learning_rate
                )),
                precision_impact: 0.0,
            });
        }
    }

    // H008: compute-optimal tokens-per-parameter ratio (Hoffmann et al.
    // 2022, "Chinchilla") — established for LLM pretraining (Transformer,
    // MoE); only fires when the client actually states a dataset size, so
    // this never guesses a training-set size that wasn't given.
    if matches!(
        ctx.config.model.model_type,
        ModelType::Transformer | ModelType::Moe
    ) {
        if let Some(dataset_tokens) = ctx.config.data.dataset_size {
            let params = total_parameters as f64;
            if params > 0.0 && dataset_tokens > 0.0 {
                const CHINCHILLA_TOKENS_PER_PARAM: f64 = 20.0;
                let ratio = dataset_tokens / params;
                if !(CHINCHILLA_TOKENS_PER_PARAM / 3.0..=CHINCHILLA_TOKENS_PER_PARAM * 3.0)
                    .contains(&ratio)
                {
                    let verdict = if ratio < CHINCHILLA_TOKENS_PER_PARAM / 3.0 {
                        "under-trained for its size — more data would likely help more than a bigger model"
                    } else {
                        "large relative to its training budget — a smaller model on the same tokens would likely reach similar loss for less compute"
                    };
                    diagnostics.push(Diagnostic {
                        category: DiagnosticCategory::Configuration,
                        severity: Severity::Hint,
                        code: DiagnosticCode::H008,
                        message: format!(
                            "Tokens-per-parameter ratio is {:.1} ({:.2e} tokens / {:.2e} params); \
                             the compute-optimal ratio from Chinchilla scaling laws (Hoffmann et \
                             al. 2022) is ~{:.0}. This model looks {}.",
                            ratio, dataset_tokens, params, CHINCHILLA_TOKENS_PER_PARAM, verdict
                        ),
                        layer_id: None,
                        suggestion: None,
                        precision_impact: 0.0,
                    });
                }
            }
        }
    }

    diagnostics
}

/// Check if custom layers have custom equations defined
fn has_custom_formulas(arch: &ArchitectureIR) -> bool {
    arch.layers
        .iter()
        .any(|layer| layer.custom_equations.is_some())
}

/// Diagnostics the compiler can stand behind.
///
/// Each one states the measurement that triggered it, what that measurement
/// costs, and what to change — a bare "Low GPU utilization (11.6%)" tells a
/// designer a number they can already read off the report. Codes are unique per
/// condition: memory pressure and communication overhead both used to be
/// reported as W006, so filtering or counting by code conflated them.
fn generate_diagnostics(metrics: &AllMetrics, memory: &MemoryIR) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    let vram = memory.metrics.gpu_vram_bytes;

    // ── Memory ───────────────────────────────────────────────────────────
    if vram > 0 && metrics.peak_vram_bytes > vram {
        let over = metrics.peak_vram_bytes as f64 / vram as f64;
        diagnostics.push(Diagnostic {
            category: DiagnosticCategory::MemoryOverflow,
            severity: Severity::Critical,
            code: DiagnosticCode::E001,
            message: format!(
                "This model needs {:.1} GB but the target GPU has {:.1} GB — {:.1}x over. \
                 It will not start.",
                metrics.peak_vram_bytes as f64 / 1e9,
                vram as f64 / 1e9,
                over
            ),
            layer_id: None,
            suggestion: Some(memory_advice(metrics, over)),
            precision_impact: 0.0,
        });
    } else if vram > 0 && metrics.peak_vram_bytes * 100 > vram * 80 {
        diagnostics.push(Diagnostic {
            category: DiagnosticCategory::MemoryOverflow,
            severity: Severity::Warning,
            code: DiagnosticCode::W005,
            message: format!(
                "Peak memory is {:.0}% of the GPU. A longer sequence or a larger batch \
                 will not fit.",
                metrics.peak_vram_bytes as f64 / vram as f64 * 100.0
            ),
            layer_id: None,
            suggestion: Some(
                "Gradient checkpointing trades about 30% more compute for roughly a \
                 square-root reduction in activation memory."
                    .to_string(),
            ),
            precision_impact: 0.0,
        });
    }

    // What dominates the footprint is more actionable than the total.
    let footprint = metrics.parameter_memory_bytes
        + metrics.activation_memory_bytes
        + metrics.gradient_memory_bytes
        + metrics.optimizer_state_bytes;
    if footprint > 0 {
        let optimizer_share = metrics.optimizer_state_bytes as f64 / footprint as f64;
        if optimizer_share > 0.4 {
            diagnostics.push(Diagnostic {
                category: DiagnosticCategory::MemoryOverflow,
                severity: Severity::Hint,
                code: DiagnosticCode::H001,
                message: format!(
                    "Optimizer state is {:.0}% of memory ({:.1} GB) — more than the weights.",
                    optimizer_share * 100.0,
                    metrics.optimizer_state_bytes as f64 / 1e9
                ),
                layer_id: None,
                suggestion: Some(
                    "ZeRO stage 1 shards optimizer state across data-parallel ranks and \
                     leaves the maths unchanged."
                        .to_string(),
                ),
                precision_impact: 0.0,
            });
        }
    }

    // ── Model shape ──────────────────────────────────────────────────────
    if metrics.total_parameters > 0 && metrics.vocab_size > 0 && metrics.hidden_size > 0 {
        let embedding = metrics.vocab_size * metrics.hidden_size as u64;
        let share = embedding as f64 / metrics.total_parameters as f64;
        if share > 0.3 {
            diagnostics.push(Diagnostic {
                category: DiagnosticCategory::ArchitectureInefficiency,
                severity: Severity::Warning,
                code: DiagnosticCode::W009,
                message: format!(
                    "The embedding table is {:.0}% of all parameters ({} x {}). \
                     Most capacity sits in lookup rather than computation.",
                    share * 100.0,
                    metrics.vocab_size,
                    metrics.hidden_size
                ),
                layer_id: None,
                suggestion: Some(
                    "Tie the input and output embeddings, or reduce the vocabulary — \
                     both cut this in half or better without touching the layers."
                        .to_string(),
                ),
                precision_impact: 0.0,
            });
        }
    }

    // ── Hardware fit ─────────────────────────────────────────────────────
    if metrics.gpu_utilization > 0.0 && metrics.gpu_utilization < 0.5 {
        diagnostics.push(Diagnostic {
            category: DiagnosticCategory::ArchitectureInefficiency,
            severity: Severity::Warning,
            code: DiagnosticCode::W006,
            message: format!(
                "The GPU is idle {:.0}% of each step — you are paying for hardware that \
                 is waiting.",
                (1.0 - metrics.gpu_utilization) * 100.0
            ),
            layer_id: None,
            suggestion: Some(
                "A larger batch is the usual fix; if memory does not allow it, the model \
                 is too small for this GPU."
                    .to_string(),
            ),
            precision_impact: 0.2,
        });
    }

    if metrics.communication_overhead > 0.3 {
        diagnostics.push(Diagnostic {
            category: DiagnosticCategory::ParallelismSuboptimal,
            severity: Severity::Warning,
            code: DiagnosticCode::W003,
            message: format!(
                "{:.0}% of each step is spent exchanging gradients rather than computing.",
                metrics.communication_overhead * 100.0
            ),
            layer_id: None,
            suggestion: Some(
                "Fewer, larger data-parallel ranks, or tensor parallelism inside a node \
                 where the interconnect is fast."
                    .to_string(),
            ),
            precision_impact: 0.1,
        });
    }

    // ── Confidence in the analysis itself ────────────────────────────────
    if metrics.unresolved_dim_count > 0 {
        diagnostics.push(Diagnostic {
            category: DiagnosticCategory::ShapeInference,
            severity: Severity::Warning,
            code: DiagnosticCode::W002,
            message: format!(
                "{} tensor dimensions could not be resolved, so the figures below are \
                 lower bounds.",
                metrics.unresolved_dim_count
            ),
            layer_id: None,
            suggestion: Some(
                "State the missing shapes on the affected blocks — hidden size, sequence \
                 length and vocabulary are the usual omissions."
                    .to_string(),
            ),
            precision_impact: 0.4,
        });
    }

    diagnostics
}

fn generate_recommendations(
    metrics: &AllMetrics,
    memory: &MemoryIR,
    hardware: &HardwareIR,
    gpu_db: &neurax_hardware_db::HardwareDatabase,
) -> Vec<Recommendation> {
    let mut recommendations = Vec::new();

    // Memory optimization
    if metrics.peak_vram_bytes > memory.metrics.gpu_vram_bytes * 80 / 100 {
        // Checkpointing every sqrt(L)-th layer (Chen et al. 2016, "Training
        // Deep Nets with Sublinear Memory Cost") retains O(sqrt(L))
        // activations instead of O(L) for an L-layer model — the fraction
        // of activation memory freed is `1 - 1/sqrt(L)`, not a flat
        // constant. This used to be a flat 0.7 regardless of depth,
        // contradicting the W005 diagnostic's own text ("roughly a
        // square-root reduction") right next to it, and giving a 4-layer
        // model and a 96-layer one the identical savings estimate despite
        // the real technique scaling very differently with depth (4
        // layers: sqrt(4)=2, 50% freed; 96 layers: sqrt(96)~9.8, ~90%
        // freed).
        let checkpointing_savings_fraction = if metrics.num_layers > 1 {
            1.0 - 1.0 / (metrics.num_layers as f64).sqrt()
        } else {
            0.0
        };
        recommendations.push(Recommendation {
            category: RecommendationCategory::MemoryOptimization,
            title: "Enable Gradient Checkpointing".to_string(),
            description: "Reduce activation memory by recomputing during backward pass".to_string(),
            impact: format!(
                "Save ~{:.1} GB VRAM (~{:.0}% of activation memory, checkpointing every ~{:.0} of {} layers)",
                metrics.activation_memory_bytes as f64 / 1e9 * checkpointing_savings_fraction,
                checkpointing_savings_fraction * 100.0,
                (metrics.num_layers as f64).sqrt().round().max(1.0),
                metrics.num_layers
            ),
            priority: Priority::High,
        });
    }

    // Parallelism
    if metrics.optimal_gpu_count > 1 && metrics.data_parallel_efficiency < 0.8 {
        // Grounded in Narayanan et al. 2021 ("Efficient Large-Scale Language
        // Model Training on GPU Clusters Using Megatron-LM"), Table 2 /
        // Figure 10: at the same 174.6B-parameter model and 1536 GPUs,
        // pure data parallelism (ZeRO-3, no model parallelism) fell to
        // 30.6% of its own small-scale efficiency (44 of 144 TFLOP/s/GPU),
        // while adding tensor+pipeline parallelism (PTD-P) recovered it to
        // 97.9% (141 of 144) — about 97% of the efficiency pure data
        // parallelism lost at that scale. Used conservatively here (90% of
        // the gap) and applied to *this* model's own current efficiency,
        // not a flat absolute target — a model already at 79% and one at
        // 30% get different, both-realistic ceilings, and the estimate can
        // never claim a target below where the model already is — the flat
        // "~90%" this replaced didn't depend on the model at all.
        let recovered_fraction = 0.90;
        let achievable_efficiency = metrics.data_parallel_efficiency
            + (1.0 - metrics.data_parallel_efficiency) * recovered_fraction;
        recommendations.push(Recommendation {
            category: RecommendationCategory::Parallelism,
            title: "Use Hybrid Parallelism".to_string(),
            description: "Combine data, tensor, and pipeline parallelism for better scaling"
                .to_string(),
            impact: format!(
                "Improve efficiency from ~{:.0}% to ~{:.0}%",
                metrics.data_parallel_efficiency * 100.0,
                achievable_efficiency * 100.0
            ),
            priority: Priority::Medium,
        });
    }

    // Hardware — a memory-bound step's latency scales roughly linearly with
    // 1/bandwidth (bytes moved is fixed; time = bytes / bandwidth), so the
    // achievable speedup from a faster GPU is the real bandwidth ratio, not
    // a flat guess. This used to hardcode "H100 SXM, 2-3x speedup" even for
    // a model already running ON an H100 (or something faster) — a claimed
    // speedup from switching to the same or a slower card.
    let current_bandwidth = hardware.gpu_profile.memory_bandwidth;
    if metrics.bottleneck == "memory-bound" && current_bandwidth > 0.0 {
        if let Some(fastest) = gpu_db
            .list_gpus()
            .into_iter()
            .max_by(|a, b| a.memory_bandwidth_gbs.total_cmp(&b.memory_bandwidth_gbs))
        {
            let speedup = fastest.memory_bandwidth_gbs / current_bandwidth;
            // Only worth surfacing if there's real headroom — the database's
            // fastest entry being the card already in use (or slower) is not
            // a recommendation.
            if speedup > 1.05 {
                recommendations.push(Recommendation {
                    category: RecommendationCategory::Hardware,
                    title: "Consider Higher Bandwidth GPU".to_string(),
                    description: format!(
                        "Model is memory-bound on {} ({:.0} GB/s); {} offers {:.0} GB/s",
                        hardware.gpu_profile.name,
                        current_bandwidth,
                        fastest.name,
                        fastest.memory_bandwidth_gbs
                    ),
                    impact: format!(
                        "Potential ~{:.1}x speedup (memory-bound, scales with bandwidth ratio)",
                        speedup
                    ),
                    priority: Priority::Medium,
                });
            }
        }
    }

    // Cost — the discount percentage itself is a market/region fact (AWS
    // states "up to 90%" off On-Demand for Spot; real GPU instances
    // commonly realize 60-90%, e.g. a documented p5.48xlarge, 8x H100,
    // averaging ~80% off), not something derivable from the model, so a
    // single conservative constant (60%, the low end of that range) is
    // used here as before. What actually used to be wrong is that the
    // dollar figure was never computed at all — "Save up to 70%" told a
    // $10,001 run and a $10,000,000 run the exact same thing.
    if metrics.training_cost_usd > 10000.0 {
        const CONSERVATIVE_SPOT_DISCOUNT: f64 = 0.60;
        let estimated_savings_usd = metrics.training_cost_usd * CONSERVATIVE_SPOT_DISCOUNT;
        recommendations.push(Recommendation {
            category: RecommendationCategory::Cost,
            title: "Optimize Training Duration".to_string(),
            description: "Consider spot instances or reserved capacity for cost savings"
                .to_string(),
            impact: format!(
                "Save ~${:.0} (~{:.0}% of this run's ${:.0} estimated on-demand cost via spot capacity)",
                estimated_savings_usd,
                CONSERVATIVE_SPOT_DISCOUNT * 100.0,
                metrics.training_cost_usd
            ),
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
    entries.sort_by_key(|e| std::cmp::Reverse(e.backward));
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

/// Advice scaled to how far over the memory budget a design is.
fn memory_advice(metrics: &AllMetrics, over: f64) -> String {
    if over > 8.0 {
        "Nothing short of a smaller model closes a gap this size — reduce depth or \
             width, or move to a multi-node parallel strategy."
            .to_string()
    } else if over > 2.0 {
        "Gradient checkpointing and ZeRO stage 2 together typically recover this much; \
             otherwise shard the model across GPUs."
            .to_string()
    } else if metrics.optimizer_state_bytes > metrics.parameter_memory_bytes {
        "Optimizer state is the largest single term here — ZeRO stage 1 alone may be \
             enough."
            .to_string()
    } else {
        "Gradient checkpointing, or a narrower dtype, should be enough to fit.".to_string()
    }
}
