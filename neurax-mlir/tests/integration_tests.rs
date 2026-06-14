//! Integration tests for MLIR dialects

use neurax_mlir::{
    NeuraxContext, NeuraxModule,
    dialects::{
        ArchitectureDialect, OperatorDialect, ComputeDialect, MemoryDialect,
        HardwareDialect, ParallelismDialect, CostDialect, ReportDialect,
    },
};
use melior::ir::Location;

/// Test complete MLIR pipeline for a transformer model
#[test]
fn test_transformer_model_mlir() {
    let ctx = NeuraxContext::new();
    let context = ctx.as_context();
    let loc = Location::unknown(context);
    
    // Create model operation
    let model = ArchitectureDialect::model(context, "Llama-3-8B", "transformer", loc);
    assert!(model.is_ok(), "Failed to create model operation");
    
    // Create global params
    let params = ArchitectureDialect::global_params_full(
        context,
        4096,   // hidden_size
        14336,  // intermediate_size
        32,     // num_attention_heads
        8,      // num_key_value_heads
        32,     // num_layers
        128256, // vocab_size
        8192,   // sequence_length
        4096,   // embedding_dim
        0.0,    // dropout_rate
        1e-5,   // layer_norm_eps
        loc,
    );
    assert!(params.is_ok(), "Failed to create global_params operation");
    
    // Create architecture metrics
    let metrics = ArchitectureDialect::metrics_full(
        context,
        8030000000,  // total_parameters
        32,          // num_layers
        32,          // num_attention_layers
        32,          // num_mlp_layers
        1,           // num_embedding_layers
        64,          // num_normalization_layers
        15200.0,     // model_size_mb
        loc,
    );
    assert!(metrics.is_ok(), "Failed to create arch metrics operation");
}

/// Test operator dialect operations
#[test]
fn test_operator_dialect() {
    let ctx = NeuraxContext::new();
    let context = ctx.as_context();
    let loc = Location::unknown(context);
    
    // Test MatMul
    let matmul = OperatorDialect::matmul(context, 4096 * 14336, 1.2e12, loc);
    assert!(matmul.is_ok(), "Failed to create matmul operation");
    
    // Test Attention
    let attn = OperatorDialect::attention(context, 4096, 32, 4194304, 6.29e11, loc);
    assert!(attn.is_ok(), "Failed to create attention operation");
    
    // Test Embedding
    let embed = OperatorDialect::embedding(context, 128256, 4096, 524289024, loc);
    assert!(embed.is_ok(), "Failed to create embedding operation");
    
    // Test RMSNorm
    let norm = OperatorDialect::rms_norm(context, 4096, 1e-5, 4096, loc);
    assert!(norm.is_ok(), "Failed to create rms_norm operation");
    
    // Test MoE
    let moe = OperatorDialect::moe(context, 4096, 8, 2, 1024, 1.5e12, loc);
    assert!(moe.is_ok(), "Failed to create moe operation");
    
    // Test SSM (Mamba)
    let ssm = OperatorDialect::ssm(context, 16, 4096, 4, 4096000, 1.0e12, loc);
    assert!(ssm.is_ok(), "Failed to create ssm operation");
}

/// Test compute dialect
#[test]
fn test_compute_dialect() {
    let ctx = NeuraxContext::new();
    let context = ctx.as_context();
    let loc = Location::unknown(context);
    
    // Test FLOPs
    let flops = ComputeDialect::flops(context, 1.05e23, 2.10e23, loc);
    assert!(flops.is_ok(), "Failed to create flops operation");
    
    // Test intensity
    let intensity = ComputeDialect::intensity(context, 1.05e23, 500000000000, 210.0, loc);
    assert!(intensity.is_ok(), "Failed to create intensity operation");
    
    // Test metrics
    let metrics = ComputeDialect::metrics(context, 3.14e23, 1.57e23, 7.5e6, 127.5, 3.14e23, 6.28e23, 0.0, 9.42e23, loc);
    assert!(metrics.is_ok(), "Failed to create compute metrics operation");
}

/// Test memory dialect
#[test]
fn test_memory_dialect() {
    let ctx = NeuraxContext::new();
    let context = ctx.as_context();
    let loc = Location::unknown(context);
    
    // Test liveness
    let liveness = MemoryDialect::liveness(context, "activation_0", 0, 100, 4194304, loc);
    assert!(liveness.is_ok(), "Failed to create liveness operation");
    
    // Test alloc
    let alloc = MemoryDialect::alloc(context, 4194304, "tensor_0", loc);
    assert!(alloc.is_ok(), "Failed to create alloc operation");
    
    // Test peak
    let peak = MemoryDialect::peak(context, 80000000000, Some(50), loc);
    assert!(peak.is_ok(), "Failed to create peak operation");
    
    // Test OOM risk
    let oom = MemoryDialect::oom_risk(context, "low", 0.75, loc);
    assert!(oom.is_ok(), "Failed to create oom_risk operation");
    
    // Test metrics
    let metrics = MemoryDialect::metrics(
        context,
        80000000000,  // parameter_memory
        12000000000,  // activation_memory
        80000000000,  // gradient_memory
        160000000000, // optimizer_state_memory
        92000000000,  // peak_vram
        4,            // max_batch_size_fit
        loc,
    );
    assert!(metrics.is_ok(), "Failed to create memory metrics operation");
}

/// Test hardware dialect
#[test]
fn test_hardware_dialect() {
    let ctx = NeuraxContext::new();
    let context = ctx.as_context();
    let loc = Location::unknown(context);
    
    // Test GPU profile
    let gpu = HardwareDialect::gpu_full(
        context,
        "H100-SXM",
        8,
        80,
        989.4,
        1979.0,
        3352.0,
        true,
        true,
        loc,
    );
    assert!(gpu.is_ok(), "Failed to create GPU profile operation");
    
    // Test roofline
    let roofline = HardwareDialect::roofline(context, 989.4, 3352.0, 0.295, loc);
    assert!(roofline.is_ok(), "Failed to create roofline operation");
    
    // Test timing
    let timing = HardwareDialect::timing(context, "layer_0", 25.0, 17.5, 42.5, loc);
    assert!(timing.is_ok(), "Failed to create timing operation");
}

/// Test parallelism dialect
#[test]
fn test_parallelism_dialect() {
    let ctx = NeuraxContext::new();
    let context = ctx.as_context();
    let loc = Location::unknown(context);
    
    // Test data parallel
    let dp = ParallelismDialect::data_parallel(context, 2, 0.95, loc);
    assert!(dp.is_ok(), "Failed to create data_parallel operation");
    
    // Test tensor parallel
    let tp = ParallelismDialect::tensor_parallel(context, 4, loc);
    assert!(tp.is_ok(), "Failed to create tensor_parallel operation");
    
    // Test pipeline parallel
    let pp = ParallelismDialect::pipeline_parallel(context, 1, 4, 1.0, loc);
    assert!(pp.is_ok(), "Failed to create pipeline_parallel operation");
    
    // Test hybrid
    let hybrid = ParallelismDialect::hybrid(context, 2, 4, 1, loc);
    assert!(hybrid.is_ok(), "Failed to create hybrid operation");
    
    // Test ZeRO
    let zero = ParallelismDialect::zero(context, 3, 35000000000, loc);
    assert!(zero.is_ok(), "Failed to create zero operation");
}

/// Test cost dialect
#[test]
fn test_cost_dialect() {
    let ctx = NeuraxContext::new();
    let context = ctx.as_context();
    let loc = Location::unknown(context);
    
    // Test pricing model
    let pricing = CostDialect::pricing_model(
        context,
        2.5,   // gpu_hour_usd
        0.10,  // energy_kwh_usd
        1.2,   // pue_factor
        700.0, // gpu_tdp_watts
        0.4,   // co2_per_kwh
        loc,
    );
    assert!(pricing.is_ok(), "Failed to create pricing_model operation");
    
    // Test training cost
    let training = CostDialect::training_cost(context, 720.0, 2500000.0, 5760.0, loc);
    assert!(training.is_ok(), "Failed to create training_cost operation");
    
    // Test energy
    let energy = CostDialect::energy(context, 125000.0, 62500.0, loc);
    assert!(energy.is_ok(), "Failed to create energy operation");
}

/// Test report dialect
#[test]
fn test_report_dialect() {
    let ctx = NeuraxContext::new();
    let context = ctx.as_context();
    let loc = Location::unknown(context);
    
    // Test report
    let report = ReportDialect::report(context, "Llama-3-8B", "transformer", "1.0", 150, loc);
    assert!(report.is_ok(), "Failed to create report operation");
    
    // Test all metrics
    let metrics = ReportDialect::all_metrics(
        context,
        8030000000,   // total_parameters
        3.14e23,      // total_flops
        80000000000,  // peak_vram
        42.5,         // latency_ms
        2500000.0,    // training_cost_usd
        loc,
    );
    assert!(metrics.is_ok(), "Failed to create all_metrics operation");
    
    // Test diagnostic
    let diag = ReportDialect::diagnostic(context, "memory_overflow", "warning", "Peak memory exceeds 80GB", loc);
    assert!(diag.is_ok(), "Failed to create diagnostic operation");
    
    // Test recommendation
    let rec = ReportDialect::recommendation(
        context,
        "memory_optimization",
        "Enable gradient checkpointing",
        "Reduces peak memory by 40%",
        "high",
        loc,
    );
    assert!(rec.is_ok(), "Failed to create recommendation operation");
}

/// Test full integration pipeline
#[test]
fn test_full_integration_pipeline() {
    let ctx = NeuraxContext::new();
    let _module = NeuraxModule::new(ctx.as_context());
    let context = ctx.as_context();
    let loc = Location::unknown(context);
    
    // Simulate full analysis pipeline
    let model = ArchitectureDialect::model(context, "TestModel", "transformer", loc);
    assert!(model.is_ok());
    
    let params = ArchitectureDialect::global_params(context, &[("hidden_size", 4096), ("num_layers", 32)], loc);
    assert!(params.is_ok());
    
    let arch_metrics = ArchitectureDialect::metrics(context, 7000000000, 32, loc);
    assert!(arch_metrics.is_ok());
    
    let compute_metrics = ComputeDialect::metrics(context, 1.0e23, 5.0e22, 5.0e6, 100.0, 1.0e23, 2.0e23, 0.0, 3.0e23, loc);
    assert!(compute_metrics.is_ok());
    
    let mem_metrics = MemoryDialect::metrics(context, 28000000000, 8000000000, 28000000000, 56000000000, 80000000000, 4, loc);
    assert!(mem_metrics.is_ok());
    
    let hw_metrics = HardwareDialect::metrics(context, 50.0, 20000.0, 0.85, 0.72, 450.0, 2800.0, loc);
    assert!(hw_metrics.is_ok());
    
    let cost_metrics = CostDialect::training_cost(context, 100.0, 50000.0, 800.0, loc);
    assert!(cost_metrics.is_ok());
    
    let report = ReportDialect::report(context, "TestModel", "transformer", "1.0", 100, loc);
    assert!(report.is_ok());
}
