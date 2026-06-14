//! NEURAX Core - Pipeline orchestration
//!
//! Industrial-grade compiler for neural network architectures.

pub mod ir;
mod engine;
mod runner;
pub mod units;

pub use ir::{IrBackend, MlirBackend, select_backend};
pub use engine::*;
pub use runner::*;
pub use units::{FLOPs, Bytes, LatencyMs, TokensPerSec, ParamCount};

use neurax_parser::ModelConfig;
use neurax_ir::{
    NeuraxContext, NeuraxError,
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
use neurax_ir::traits::{IrPass, ReportPass as ReportPassTrait};
use neurax_ir::report::ReportInput;
use std::time::Instant;

/// Analysis result containing all IR outputs
#[derive(Debug)]
pub struct AnalysisResult {
    pub arch: ArchitectureIR,
    pub graph: GraphIR,
    pub tensor: TensorIR,
    pub operator: OperatorIR,
    pub compute: ComputeIR,
    pub memory: MemoryIR,
    pub parallelism: ParallelismIR,
    pub hardware: HardwareIR,
    pub cost: CostIR,
    pub report: ReportIR,
    /// Dynamic analysis results (M36-M55)
    pub dynamic: DynamicResults,
    pub analysis_time_ms: u64,
}

impl AnalysisResult {
    /// Export all metrics to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        use neurax_ir::report::JsonOutput;
        
        let output = JsonOutput::from_report_with_dynamic(
            &self.report, 
            "model.json", 
            self.analysis_time_ms,
            &self.dynamic
        );
        output.to_json()
    }
    
    /// Export all metrics to JSON bytes
    pub fn to_json_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        use neurax_ir::report::JsonOutput;
        
        let output = JsonOutput::from_report_with_dynamic(
            &self.report, 
            "model.json", 
            self.analysis_time_ms,
            &self.dynamic
        );
        output.to_json_bytes()
    }
    
    /// Save metrics to a JSON file
    pub fn save_json(&self, path: &str) -> std::io::Result<()> {
        let json = self.to_json().map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }
}

/// Run full analysis pipeline
pub fn run_analysis(config: ModelConfig) -> Result<AnalysisResult, NeuraxError> {
    let start = Instant::now();
    let ctx = NeuraxContext::new(config.clone());
    
    // Phase 1: Architecture
    let arch_pass = ArchitecturePass;
    let (mut arch, _) = arch_pass.run(&config, &ctx)?;
    let arch_metrics = arch_pass.compute_metrics(&mut arch, &ctx)?;
    arch_pass.validate(&arch, &arch_metrics)?;
    
    // Phase 2: Graph
    let graph_pass = GraphPass;
    let (mut graph, _) = graph_pass.run(&arch, &ctx)?;
    let graph_metrics = graph_pass.compute_metrics(&mut graph, &ctx)?;
    graph_pass.validate(&graph, &graph_metrics)?;
    
    // Phase 3: Tensor
    let tensor_pass = TensorPass;
    let (mut tensor, _) = tensor_pass.run(&graph, &ctx)?;
    let tensor_metrics = tensor_pass.compute_metrics(&mut tensor, &ctx)?;
    tensor_pass.validate(&tensor, &tensor_metrics)?;
    
    // Phase 4: Operator
    let operator_pass = OperatorPass;
    let (mut operator, _) = operator_pass.run(&(tensor.clone(), arch.clone()), &ctx)?;
    let operator_metrics = operator_pass.compute_metrics(&mut operator, &ctx)?;
    operator_pass.validate(&operator, &operator_metrics)?;
    
    // Phase 5: Compute
    let compute_pass = ComputePass;
    let (mut compute, _) = compute_pass.run(&operator, &ctx)?;
    let compute_metrics = compute_pass.compute_metrics(&mut compute, &ctx)?;
    compute_pass.validate(&compute, &compute_metrics)?;
    
    // Phase 6: Memory
    let memory_pass = MemoryPass;
    let (mut memory, _) = memory_pass.run(&(compute.clone(), tensor.clone(), arch.clone()), &ctx)?;
    let memory_metrics = memory_pass.compute_metrics(&mut memory, &ctx)?;
    memory_pass.validate(&memory, &memory_metrics)?;
    
    // Phase 7 & 8: Parallelism and Hardware in parallel (rayon::join per impl_2.md)
    // These passes are independent and can run concurrently
    let ((parallelism, parallelism_metrics), (hardware, hardware_metrics)) = rayon::join(
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
    
    // Re-run hardware with actual parallelism data (quick update)
    let hardware_pass = HardwarePass;
    let (mut hardware, _) = hardware_pass.run(&(compute.clone(), memory.clone(), parallelism.clone()), &ctx)?;
    let hardware_metrics = hardware_pass.compute_metrics(&mut hardware, &ctx)?;
    hardware_pass.validate(&hardware, &hardware_metrics)?;
    
    // Phase 9: Cost
    let cost_pass = CostPass;
    let (mut cost, _) = cost_pass.run(&(hardware.clone(), parallelism.clone()), &ctx)?;
    let cost_metrics = cost_pass.compute_metrics(&mut cost, &ctx)?;
    cost_pass.validate(&cost, &cost_metrics)?;
    
    // Phase 10: Report
    let report_pass = ReportPass;
    let report = report_pass.build_report(&ReportInput {
        arch: &arch,
        graph: &graph,
        tensor: &tensor,
        operator: &operator,
        compute: &compute,
        memory: &memory,
        parallelism: &parallelism,
        hardware: &hardware,
        cost: &cost,
    }, &ctx)?;
    
    // Phase 11: Dynamic Analysis (M36-M55)
    let dynamic_config = DynamicConfig::default();
    
    // Run dynamic passes in parallel
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
    
    let analysis_time_ms = start.elapsed().as_millis() as u64;
    
    // Return result with owned values
    Ok(AnalysisResult {
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
    })
}

/// Trait extension for running passes
pub trait IrPassExt: IrPass {
    fn run(&self, input: &Self::Input, ctx: &NeuraxContext) -> Result<(Self::Output, Self::Metrics), NeuraxError> {
        let mut output = self.build(input, ctx).map_err(|e| e.into())?;
        let metrics = self.compute_metrics(&mut output, ctx).map_err(|e| e.into())?;
        self.validate(&output, &metrics).map_err(|e| e.into())?;
        Ok((output, metrics))
    }
}

impl<T: IrPass> IrPassExt for T {}
