//! Streaming analysis event system
//!
//! Provides real-time event emission during the analysis pipeline,
//! enabling SSE streaming to clients and live progress updates.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::broadcast;
use std::time::Instant;

use crate::AnalysisResult;
use crate::IrPassExt;
use neurax_parser::ModelConfig;
use neurax_ir::NeuraxError;
use neurax_ir::traits::{IrPass, ReportPass as ReportPassTrait};
use neurax_ir::{
    NeuraxContext,
    ArchitectureIR, ArchitecturePass,
    GraphIR, GraphPass,
    TensorIR, TensorPass,
    OperatorIR, OperatorPass,
    ComputeIR, ComputePass,
    MemoryIR, MemoryPass,
    ParallelismIR, ParallelismPass,
    HardwareIR, HardwarePass,
    CostIR, CostPass,
    ReportIR, ReportPass,
    dynamic::{
        VirtualMemoryPass, StabilityAnalysisPass, BehavioralSynthesisPass,
        DynamicResults, DynamicConfig,
    },
};
use neurax_ir::parallelism::ParallelismMetrics;
use neurax_ir::hardware::HardwareMetrics;
use neurax_ir::report::{ReportInput, PhaseTimingEntry};

/// Unique identifier for an analysis job
pub type JobId = String;

/// Events emitted during the analysis pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum AnalysisEvent {
    /// Pipeline started
    Started {
        job_id: JobId,
        model_name: String,
        model_type: String,
        num_layers: usize,
    },
    /// A specific phase has started
    PhaseStarted {
        job_id: JobId,
        phase: String,
        phase_index: usize,
        total_phases: usize,
    },
    /// A specific phase has completed
    PhaseCompleted {
        job_id: JobId,
        phase: String,
        phase_index: usize,
        total_phases: usize,
        duration_ms: u64,
    },
    /// Overall progress update
    Progress {
        job_id: JobId,
        phase: String,
        phase_index: usize,
        total_phases: usize,
        progress_pct: f64,
        elapsed_ms: u64,
    },
    /// A diagnostic was generated during a phase
    Diagnostic {
        job_id: JobId,
        phase: String,
        category: String,
        severity: String,
        code: Option<String>,
        message: String,
        suggestion: Option<String>,
    },
    /// Pipeline completed successfully
    Completed {
        job_id: JobId,
        total_ms: u64,
    },
    /// Pipeline failed
    Failed {
        job_id: JobId,
        error: String,
        phase: String,
    },
}

/// Trait for emitting analysis events
pub trait AnalysisEventEmitter: Send + Sync {
    fn emit(&self, event: AnalysisEvent);
}

/// No-op emitter that discards all events
pub struct NoopEmitter;

impl AnalysisEventEmitter for NoopEmitter {
    fn emit(&self, _event: AnalysisEvent) {}
}

/// Broadcast emitter using tokio broadcast channel
pub struct BroadcastEmitter {
    sender: broadcast::Sender<AnalysisEvent>,
}

impl BroadcastEmitter {
    pub fn new(buffer_capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(buffer_capacity);
        Self { sender }
    }

    /// Create from an existing broadcast sender
    pub fn from_sender(sender: broadcast::Sender<AnalysisEvent>) -> Self {
        Self { sender }
    }

    /// Get a receiver for consuming events
    pub fn subscribe(&self) -> broadcast::Receiver<AnalysisEvent> {
        self.sender.subscribe()
    }

    /// Get the sender for creating additional emitters
    pub fn sender(&self) -> broadcast::Sender<AnalysisEvent> {
        self.sender.clone()
    }
}

impl AnalysisEventEmitter for BroadcastEmitter {
    fn emit(&self, event: AnalysisEvent) {
        // Ignore send errors (no receivers)
        let _ = self.sender.send(event);
    }
}

/// Thread-safe wrapper for sharing an emitter across threads
#[derive(Clone)]
pub struct SharedEmitter {
    inner: Arc<dyn AnalysisEventEmitter>,
}

impl SharedEmitter {
    pub fn new(emitter: impl AnalysisEventEmitter + 'static) -> Self {
        Self {
            inner: Arc::new(emitter),
        }
    }

    pub fn noop() -> Self {
        Self::new(NoopEmitter)
    }

    pub fn emit(&self, event: AnalysisEvent) {
        self.inner.emit(event);
    }
}

impl AnalysisEventEmitter for SharedEmitter {
    fn emit(&self, event: AnalysisEvent) {
        self.inner.emit(event);
    }
}

/// Job state for tracking analysis progress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobState {
    pub job_id: JobId,
    pub status: JobStatus,
    pub current_phase: String,
    pub progress_pct: f64,
    pub started_at_ms: u64,
    pub completed_at_ms: Option<u64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    Queued,
    Running,
    Completed,
    Failed,
}

/// Run the full analysis pipeline with event emission
pub fn run_analysis_streaming(
    config: ModelConfig,
    emitter: &SharedEmitter,
    job_id: &str,
) -> Result<AnalysisResult, NeuraxError> {
    let start = Instant::now();
    let total_phases = 11usize; // 10 dialects + dynamic

    let model_name = config.model.name.clone().unwrap_or_else(|| "Unknown".to_string());
    let model_type = config.model.model_type.as_str().to_string();
    let num_layers = config.model.layers.len();

    // Emit started event
    emitter.emit(AnalysisEvent::Started {
        job_id: job_id.to_string(),
        model_name: model_name.clone(),
        model_type: model_type.clone(),
        num_layers,
    });

    let ctx = neurax_ir::NeuraxContext::new(config.clone());
    let mut phase_timeline: Vec<neurax_ir::report::PhaseTimingEntry> = Vec::new();

    macro_rules! run_phase {
        ($phase_name:expr, $phase_index:expr, $block:expr) => {{
            let phase_label = $phase_name.to_string();
            emitter.emit(AnalysisEvent::PhaseStarted {
                job_id: job_id.to_string(),
                phase: phase_label.clone(),
                phase_index: $phase_index,
                total_phases,
            });

            let t0 = Instant::now();
            let result = $block;
            let duration_ms = t0.elapsed().as_millis() as u64;

            phase_timeline.push(PhaseTimingEntry {
                name: phase_label.clone(),
                duration_ms,
                status: "completed".to_string(),
            });

            // Emit progress
            let progress = (($phase_index + 1) as f64) / (total_phases as f64) * 100.0;
            emitter.emit(AnalysisEvent::Progress {
                job_id: job_id.to_string(),
                phase: phase_label.clone(),
                phase_index: $phase_index,
                total_phases,
                progress_pct: progress,
                elapsed_ms: start.elapsed().as_millis() as u64,
            });

            emitter.emit(AnalysisEvent::PhaseCompleted {
                job_id: job_id.to_string(),
                phase: phase_label,
                phase_index: $phase_index,
                total_phases,
                duration_ms,
            });

            result
        }};
    }

    // Phase 1: Architecture
    let arch_pass = ArchitecturePass;
    let (mut arch, _arch_metrics) = run_phase!("Architecture", 0, {
        let (mut a, _) = arch_pass.run(&config, &ctx)?;
        let m = arch_pass.compute_metrics(&mut a, &ctx)?;
        arch_pass.validate(&a, &m)?;
        (a, m)
    });

    // Phase 2: Graph
    let graph_pass = GraphPass;
    let (mut graph, _graph_metrics) = run_phase!("Graph", 1, {
        let (mut g, _) = graph_pass.run(&arch, &ctx)?;
        let m = graph_pass.compute_metrics(&mut g, &ctx)?;
        graph_pass.validate(&g, &m)?;
        (g, m)
    });

    // Phase 3: Tensor
    let tensor_pass = TensorPass;
    let (mut tensor, _tensor_metrics) = run_phase!("Tensor", 2, {
        let (mut t, _) = tensor_pass.run(&graph, &ctx)?;
        let m = tensor_pass.compute_metrics(&mut t, &ctx)?;
        tensor_pass.validate(&t, &m)?;
        (t, m)
    });

    // Phase 4: Operator
    let operator_pass = OperatorPass;
    let (mut operator, _operator_metrics) = run_phase!("Operator", 3, {
        let (mut o, _) = operator_pass.run(&(tensor.clone(), arch.clone()), &ctx)?;
        let m = operator_pass.compute_metrics(&mut o, &ctx)?;
        operator_pass.validate(&o, &m)?;
        (o, m)
    });

    // Phase 5: Compute
    let compute_pass = ComputePass;
    let (mut compute, _compute_metrics) = run_phase!("Compute", 4, {
        let (mut c, _) = compute_pass.run(&operator, &ctx)?;
        let m = compute_pass.compute_metrics(&mut c, &ctx)?;
        compute_pass.validate(&c, &m)?;
        (c, m)
    });

    // Phase 6: Memory
    let memory_pass = MemoryPass;
    let (mut memory, _memory_metrics) = run_phase!("Memory", 5, {
        let (mut m, _) = memory_pass.run(&(compute.clone(), tensor.clone(), arch.clone()), &ctx)?;
        let metrics = memory_pass.compute_metrics(&mut m, &ctx)?;
        memory_pass.validate(&m, &metrics)?;
        (m, metrics)
    });

    // Phase 7 & 8: Parallelism and Hardware in parallel
    let ((parallelism, _parallelism_metrics), (_hardware_initial, _hardware_metrics_initial)) = rayon::join(
        || {
            let parallelism_pass = ParallelismPass;
            let mut parallelism = parallelism_pass.build(&(memory.clone(), graph.clone()), &ctx)
                .unwrap_or_else(|_| ParallelismIR::default());
            let parallelism_metrics = parallelism_pass.compute_metrics(&mut parallelism, &ctx)
                .unwrap_or_else(|_| ParallelismMetrics::default());
            let _ = parallelism_pass.validate(&parallelism, &parallelism_metrics);
            (parallelism, parallelism_metrics)
        },
        || {
            let hardware_pass = HardwarePass;
            let mut hardware = hardware_pass.build(&(compute.clone(), memory.clone(), ParallelismIR::default()), &ctx)
                .unwrap_or_else(|_| HardwareIR::default());
            let hardware_metrics = hardware_pass.compute_metrics(&mut hardware, &ctx)
                .unwrap_or_else(|_| HardwareMetrics::default());
            let _ = hardware_pass.validate(&hardware, &hardware_metrics);
            (hardware, hardware_metrics)
        },
    );

    // Emit progress for parallel phases
    emitter.emit(AnalysisEvent::PhaseStarted {
        job_id: job_id.to_string(),
        phase: "Parallelism+Hardware".to_string(),
        phase_index: 6,
        total_phases,
    });
    let t0 = Instant::now();

    // Re-run hardware with actual parallelism data
    let hardware_pass = HardwarePass;
    let (mut hardware, _) = hardware_pass.run(&(compute.clone(), memory.clone(), parallelism.clone()), &ctx)?;
    let _hardware_metrics = hardware_pass.compute_metrics(&mut hardware, &ctx)?;
    hardware_pass.validate(&hardware, &_hardware_metrics)?;

    let parallelism_duration = t0.elapsed().as_millis() as u64;
    phase_timeline.push(PhaseTimingEntry {
        name: "Parallelism+Hardware".to_string(),
        duration_ms: parallelism_duration,
        status: "completed".to_string(),
    });

    emitter.emit(AnalysisEvent::PhaseCompleted {
        job_id: job_id.to_string(),
        phase: "Parallelism+Hardware".to_string(),
        phase_index: 6,
        total_phases,
        duration_ms: parallelism_duration,
    });
    emitter.emit(AnalysisEvent::Progress {
        job_id: job_id.to_string(),
        phase: "Parallelism+Hardware".to_string(),
        phase_index: 6,
        total_phases,
        progress_pct: 7.0 / (total_phases as f64) * 100.0,
        elapsed_ms: start.elapsed().as_millis() as u64,
    });

    // Phase 9: Cost
    let cost_pass = CostPass;
    let (mut cost, _cost_metrics) = run_phase!("Cost", 7, {
        let (mut c, _) = cost_pass.run(&(hardware.clone(), parallelism.clone()), &ctx)?;
        let m = cost_pass.compute_metrics(&mut c, &ctx)?;
        cost_pass.validate(&c, &m)?;
        (c, m)
    });

    // Phase 10: Report
    let report_pass = ReportPass;
    let mut report = run_phase!("Report", 8, {
        report_pass.build_report(&ReportInput {
            arch: &arch,
            graph: &graph,
            tensor: &tensor,
            operator: &operator,
            compute: &compute,
            memory: &memory,
            parallelism: &parallelism,
            hardware: &hardware,
            cost: &cost,
        }, &ctx)?
    });

    // Phase 11: Dynamic Analysis
    let dynamic_config = DynamicConfig::default();

    emitter.emit(AnalysisEvent::PhaseStarted {
        job_id: job_id.to_string(),
        phase: "Dynamic".to_string(),
        phase_index: 9,
        total_phases,
    });
    let dyn_start = Instant::now();

    let (vm_metrics, (sta_metrics, bps_metrics)) = rayon::join(
        || {
            let vm_pass = VirtualMemoryPass::new();
            Some(vm_pass.run(&memory.metrics))
        },
        || rayon::join(
            || {
                let sta_pass = StabilityAnalysisPass::new();
                Some(sta_pass.run(&graph, &memory.metrics))
            },
            || {
                let bps_pass = BehavioralSynthesisPass::new();
                Some(bps_pass.run(&compute, &dynamic_config))
            }
        )
    );

    let dynamic = DynamicResults {
        virtual_memory: vm_metrics,
        stability: sta_metrics,
        behavioral: bps_metrics,
    };

    let dyn_duration = dyn_start.elapsed().as_millis() as u64;
    phase_timeline.push(PhaseTimingEntry {
        name: "Dynamic".to_string(),
        duration_ms: dyn_duration,
        status: "completed".to_string(),
    });

    emitter.emit(AnalysisEvent::PhaseCompleted {
        job_id: job_id.to_string(),
        phase: "Dynamic".to_string(),
        phase_index: 9,
        total_phases,
        duration_ms: dyn_duration,
    });

    // Assign phase timeline to report now that all phases are complete
    report.phase_timeline = phase_timeline;

    let analysis_time_ms = start.elapsed().as_millis() as u64;

    let result = AnalysisResult {
        arch,
        graph,
        tensor,
        operator,
        compute,
        memory,
        parallelism,
        hardware,
        cost,
        report,
        dynamic,
        analysis_time_ms,
    };

    // Emit completed event
    emitter.emit(AnalysisEvent::Completed {
        job_id: job_id.to_string(),
        total_ms: analysis_time_ms,
    });

    Ok(result)
}

/// Run analysis streaming, returning the result or emitting a failure event
pub fn run_analysis_streaming_fallible(
    config: ModelConfig,
    emitter: SharedEmitter,
    job_id: &str,
) -> Result<AnalysisResult, NeuraxError> {
    match run_analysis_streaming(config, &emitter, job_id) {
        Ok(result) => Ok(result),
        Err(e) => {
            emitter.emit(AnalysisEvent::Failed {
                job_id: job_id.to_string(),
                error: e.to_string(),
                phase: "unknown".to_string(),
            });
            Err(e)
        }
    }
}