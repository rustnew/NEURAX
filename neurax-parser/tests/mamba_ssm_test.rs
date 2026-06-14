//! Test compilation of a State Space Model (Mamba-2 style)
//! Validates complete absorption pipeline with SSM architecture

use neurax_parser::{parse_model_config, AbsorbedModel};
use neurax_ir::IrInjector;

/// Mamba-2 8B - State Space Model with selective scan
const MAMBA_8B_JSON: &str = r#"
{
    "schema_version": "1.0",
    "model": {
        "name": "Mamba-2-8B",
        "type": "ssm",
        "layers": [
            {"id": "embed_tokens", "layer_type": "embedding", "input_shape": [4096], "output_shape": [4096], "params": {"vocab_size": 128000, "embedding_dim": 4096}},
            
            {"id": "mamba_block_0", "layer_type": "mamba_block", "input_shape": [4096, 4096], "output_shape": [4096, 4096], "params": {"d_model": 4096, "d_state": 128, "d_conv": 4, "expand": 2}},
            {"id": "norm_0", "layer_type": "normalization", "input_shape": [4096, 4096], "output_shape": [4096, 4096], "params": {"hidden_size": 4096}},
            
            {"id": "mamba_block_1", "layer_type": "mamba_block", "input_shape": [4096, 4096], "output_shape": [4096, 4096], "params": {"d_model": 4096, "d_state": 128, "d_conv": 4, "expand": 2}},
            {"id": "norm_1", "layer_type": "normalization", "input_shape": [4096, 4096], "output_shape": [4096, 4096], "params": {"hidden_size": 4096}},
            
            {"id": "mamba_block_2", "layer_type": "mamba_block", "input_shape": [4096, 4096], "output_shape": [4096, 4096], "params": {"d_model": 4096, "d_state": 128, "d_conv": 4, "expand": 2}},
            {"id": "norm_2", "layer_type": "normalization", "input_shape": [4096, 4096], "output_shape": [4096, 4096], "params": {"hidden_size": 4096}},
            
            {"id": "mamba_block_3", "layer_type": "mamba_block", "input_shape": [4096, 4096], "output_shape": [4096, 4096], "params": {"d_model": 4096, "d_state": 128, "d_conv": 4, "expand": 2}},
            {"id": "norm_3", "layer_type": "normalization", "input_shape": [4096, 4096], "output_shape": [4096, 4096], "params": {"hidden_size": 4096}},
            
            {"id": "mamba_block_4", "layer_type": "mamba_block", "input_shape": [4096, 4096], "output_shape": [4096, 4096], "params": {"d_model": 4096, "d_state": 128, "d_conv": 4, "expand": 2}},
            {"id": "norm_4", "layer_type": "normalization", "input_shape": [4096, 4096], "output_shape": [4096, 4096], "params": {"hidden_size": 4096}},
            
            {"id": "mamba_block_5", "layer_type": "mamba_block", "input_shape": [4096, 4096], "output_shape": [4096, 4096], "params": {"d_model": 4096, "d_state": 128, "d_conv": 4, "expand": 2}},
            {"id": "norm_5", "layer_type": "normalization", "input_shape": [4096, 4096], "output_shape": [4096, 4096], "params": {"hidden_size": 4096}},
            
            {"id": "mamba_block_6", "layer_type": "mamba_block", "input_shape": [4096, 4096], "output_shape": [4096, 4096], "params": {"d_model": 4096, "d_state": 128, "d_conv": 4, "expand": 2}},
            {"id": "norm_6", "layer_type": "normalization", "input_shape": [4096, 4096], "output_shape": [4096, 4096], "params": {"hidden_size": 4096}},
            
            {"id": "mamba_block_7", "layer_type": "mamba_block", "input_shape": [4096, 4096], "output_shape": [4096, 4096], "params": {"d_model": 4096, "d_state": 128, "d_conv": 4, "expand": 2}},
            {"id": "norm_7", "layer_type": "normalization", "input_shape": [4096, 4096], "output_shape": [4096, 4096], "params": {"hidden_size": 4096}},
            
            {"id": "final_norm", "layer_type": "normalization", "input_shape": [4096, 4096], "output_shape": [4096, 4096], "params": {"hidden_size": 4096}},
            {"id": "lm_head", "layer_type": "dense", "input_shape": [4096, 4096], "output_shape": [4096, 128000], "params": {"in_features": 4096, "out_features": 128000}}
        ],
        "global_params": {
            "hidden_size": 4096,
            "num_layers": 32,
            "vocab_size": 128000,
            "ssm_state_size": 128,
            "ssm_expand": 2,
            "ssm_conv_kernel": 4,
            "intermediate_size": 8192
        }
    },
    "training": {
        "batch_size": 128,
        "optimizer": "adamw",
        "learning_rate": 0.0003,
        "precision": "bf16",
        "gradient_checkpointing": true,
        "zero_stage": 2,
        "max_steps": 100000,
        "warmup_steps": 2000,
        "parallelism": {
            "data_parallel": 4,
            "tensor_parallel": 1,
            "pipeline_parallel": 2
        }
    },
    "hardware": {
        "gpus": [
            {
                "name": "H100-80GB-HBM3",
                "count": 8,
                "memory_gb": 80,
                "tflops_fp16": 1979,
                "tflops_fp32": 67,
                "tflops_fp8": 3958,
                "memory_bandwidth_gb_s": 3352,
                "tensor_cores": true,
                "nvlink": true
            }
        ],
        "interconnect": "NVLink4",
        "interconnect_bandwidth_gb_s": 900
    },
    "data": {
        "input_shape": [4096],
        "dtype": "bf16",
        "vocab_size": 128000
    },
    "cost_config": {
        "provider": "lambda",
        "gpu_hour_usd": 2.50,
        "energy_kwh_usd": 0.08,
        "pue_factor": 1.1
    }
}
"#;

#[test]
fn test_mamba_ssm_compilation() {
    println!("=== Compiling State Space Model (Mamba-2-8B) ===");
    println!("Model: Mamba-2-8B");
    println!("Architecture: Selective State Space Model");
    println!("State Size: 128, Expand: 2, Conv Kernel: 4");
    println!("GPUs: 8× H100-80GB");
    
    // ── Step 1: Parse JSON ─────────────────────────────────────────────
    let start = std::time::Instant::now();
    let config = parse_model_config(MAMBA_8B_JSON)
        .expect("Failed to parse Mamba JSON");
    let parse_time = start.elapsed();
    println!("✓ Parsed in {:?}", parse_time);
    
    // ── Step 2: Absorb into AbsorbedModel ───────────────────────────────
    let start = std::time::Instant::now();
    let absorbed = AbsorbedModel::absorb(config);
    let absorb_time = start.elapsed();
    println!("✓ Absorbed in {:?}", absorb_time);
    
    // ── Step 3: Validate GlobalResolutionContext ────────────────────────
    let grc = &absorbed.resolution_context;
    
    // ── SSM-Specific Parameters ────────────────────────────────────────
    println!("\n=== SSM Parameters ===");
    
    // State size (N in Mamba paper)
    assert_eq!(grc.ssm_state_size, Some(128), "ssm_state_size should be 128");
    println!("  ssm_state_size (N): {}", grc.ssm_state_size.unwrap());
    
    // Expand factor
    assert_eq!(grc.ssm_expand, Some(2), "ssm_expand should be 2");
    println!("  ssm_expand: {}", grc.ssm_expand.unwrap());
    
    // Convolution kernel size
    assert_eq!(grc.ssm_conv_kernel, Some(4), "ssm_conv_kernel should be 4");
    println!("  ssm_conv_kernel: {}", grc.ssm_conv_kernel.unwrap());
    
    // ── Standard Transformer-like Parameters ───────────────────────────
    println!("\n=== Standard Parameters ===");
    
    assert_eq!(grc.hidden_size, Some(4096), "hidden_size should be 4096");
    println!("  hidden_size (d_model): {}", grc.hidden_size.unwrap());
    
    assert_eq!(grc.num_layers, Some(32), "num_layers should be 32");
    println!("  num_layers: {}", grc.num_layers.unwrap());
    
    assert_eq!(grc.vocab_size, Some(128000), "vocab_size should be 128000");
    println!("  vocab_size: {}", grc.vocab_size.unwrap());
    
    // Intermediate size (d_inner = d_model * expand)
    assert_eq!(grc.intermediate_size, Some(8192), "intermediate_size should be 8192");
    println!("  intermediate_size (d_inner): {}", grc.intermediate_size.unwrap());
    
    // ── Derived Values ─────────────────────────────────────────────────
    println!("\n=== Derived Values ===");
    
    assert_eq!(grc.dtype_bytes, 2, "bf16 = 2 bytes");
    println!("  dtype_bytes: {}", grc.dtype_bytes);
    
    assert_eq!(grc.optimizer_bytes_per_param, 8, "AdamW = 8 bytes");
    println!("  optimizer_bytes: {}", grc.optimizer_bytes_per_param);
    
    // d_inner calculation
    let d_inner = grc.d_inner();
    assert_eq!(d_inner, Some(8192), "d_inner should be hidden_size * expand");
    println!("  d_inner (derived): {:?}", d_inner);
    
    // ── Hardware & Parallelism ─────────────────────────────────────────
    println!("\n=== Hardware & Parallelism ===");
    
    assert_eq!(grc.num_gpus, 8, "8 GPUs");
    println!("  num_gpus: {}", grc.num_gpus);
    
    assert!((grc.primary_gpu_tflops - 1979.0).abs() < 1.0, "H100 TFLOPs");
    println!("  GPU TFLOPs: {}", grc.primary_gpu_tflops);
    
    assert_eq!(grc.dp, 4, "Data parallel = 4");
    assert_eq!(grc.tp, 1, "No tensor parallel for SSM");
    assert_eq!(grc.pp, 2, "Pipeline parallel = 2");
    println!("  Parallelism: DP={}, TP={}, PP={}", grc.dp, grc.tp, grc.pp);
    
    // ── Symbol Table ───────────────────────────────────────────────────
    println!("\n=== Symbol Table ===");
    
    assert!(grc.symbol_table.contains_key("B"), "B in symbol table");
    assert!(grc.symbol_table.contains_key("H"), "H in symbol table");
    assert!(grc.symbol_table.contains_key("V"), "V in symbol table");
    assert!(grc.symbol_table.contains_key("I"), "I (intermediate) in symbol table");
    
    println!("  B (batch): {:?}", grc.symbol_table.get("B"));
    println!("  H (hidden): {:?}", grc.symbol_table.get("H"));
    println!("  V (vocab): {:?}", grc.symbol_table.get("V"));
    println!("  I (intermediate): {:?}", grc.symbol_table.get("I"));
    
    // Confidence score
    println!("\n  Confidence score: {:.2}%", grc.confidence_score * 100.0);
    
    // ── Step 4: IR Injection ───────────────────────────────────────────
    let start = std::time::Instant::now();
    let arch_input = IrInjector::to_architecture_ir(&absorbed);
    let mem_config = IrInjector::configure_memory_pass(&absorbed);
    let hw_config = IrInjector::configure_hardware_pass(&absorbed);
    let cost_config = IrInjector::configure_cost_pass(&absorbed);
    let inject_time = start.elapsed();
    println!("\n✓ IRs injected in {:?}", inject_time);
    
    // Validate Architecture IR
    assert_eq!(arch_input.hidden_size, Some(4096));
    assert_eq!(arch_input.num_layers, Some(32));
    assert_eq!(arch_input.vocab_size, Some(128000));
    
    // Validate Memory config
    assert_eq!(mem_config.dtype_bytes, 2);
    assert_eq!(mem_config.optimizer_bytes, 8);
    assert!(mem_config.checkpointing_enabled);
    assert_eq!(mem_config.zero_stage, 2);
    
    // Validate Hardware config
    assert!((hw_config.gpu_tflops_fp16 - 1979.0).abs() < 1.0);
    assert_eq!(hw_config.dp, 4);
    assert_eq!(hw_config.pp, 2);
    
    // Validate Cost config
    assert!((cost_config.gpu_hour_usd - 2.50).abs() < 0.01);
    assert_eq!(cost_config.num_gpus, 8);
    
    // ── Step 5: Parameter Calculation ───────────────────────────────────
    let start = std::time::Instant::now();
    let total_params = IrInjector::calculate_total_params(&absorbed);
    let calc_time = start.elapsed();
    println!("✓ Parameters calculated in {:?}", calc_time);
    
    // Mamba-8B has ~8B parameters
    println!("  Total parameters: {:.2}B", total_params as f64 / 1e9);
    assert!(total_params > 5_000_000_000, "Expected > 5B params, got {}", total_params);
    assert!(total_params < 12_000_000_000, "Expected < 12B params, got {}", total_params);
    
    // ── Step 6: FLOPs Calculation ───────────────────────────────────────
    let start = std::time::Instant::now();
    let flops = IrInjector::calculate_flops_per_token(&absorbed);
    let flops_time = start.elapsed();
    println!("✓ FLOPs calculated in {:?}", flops_time);
    println!("  FLOPs per token: {:.2e}", flops as f64);
    assert!(flops > 0, "FLOPs should be > 0");
    
    // ── Summary ─────────────────────────────────────────────────────────
    println!("\n=== Mamba Compilation Summary ===");
    println!("Total time: {:?}", parse_time + absorb_time + inject_time + calc_time + flops_time);
    println!("✓ All SSM fields absorbed correctly");
    println!("✓ All IRs configured");
    println!("✓ Parameter count: {:.2}B", total_params as f64 / 1e9);
    println!("✓ FLOPs/token: {:.2e}", flops as f64);
}

#[test]
fn test_ssm_state_dimensions() {
    let config = parse_model_config(MAMBA_8B_JSON).unwrap();
    let absorbed = AbsorbedModel::absorb(config);
    let grc = &absorbed.resolution_context;
    
    // Verify SSM-specific dimensions
    assert_eq!(grc.ssm_state_size, Some(128), "State size (N)");
    assert_eq!(grc.ssm_expand, Some(2), "Expand factor");
    assert_eq!(grc.ssm_conv_kernel, Some(4), "Conv kernel");
    
    // Verify d_inner derivation
    // d_inner = d_model * expand = 4096 * 2 = 8192
    let d_inner = grc.d_inner();
    assert_eq!(d_inner, Some(8192), "d_inner should be derived from hidden_size * expand");
    
    // Verify it matches intermediate_size if provided
    assert_eq!(grc.intermediate_size, d_inner, "intermediate_size should match d_inner");
}

#[test]
fn test_ssm_symbol_table() {
    let config = parse_model_config(MAMBA_8B_JSON).unwrap();
    let absorbed = AbsorbedModel::absorb(config);
    let grc = &absorbed.resolution_context;
    
    // Verify symbol table has SSM-relevant symbols
    let sym = &grc.symbol_table;
    
    // Standard symbols
    assert!(sym.contains_key("B"), "B (batch)");
    assert!(sym.contains_key("H"), "H (hidden)");
    assert!(sym.contains_key("V"), "V (vocab)");
    assert!(sym.contains_key("I"), "I (intermediate)");
    
    // Verify values
    assert_eq!(sym.get("B"), Some(&128u64), "Batch size");
    assert_eq!(sym.get("H"), Some(&4096u64), "Hidden size");
    assert_eq!(sym.get("V"), Some(&128000u64), "Vocab size");
    assert_eq!(sym.get("I"), Some(&8192u64), "Intermediate size");
    
    // dtype_bytes
    assert_eq!(sym.get("dtype_bytes"), Some(&2u64), "bf16 = 2 bytes");
}

#[test]
fn test_ssm_vs_transformer_distinction() {
    let config = parse_model_config(MAMBA_8B_JSON).unwrap();
    let absorbed = AbsorbedModel::absorb(config);
    let grc = &absorbed.resolution_context;
    
    // SSM should NOT have attention-related parameters
    assert_eq!(grc.num_attention_heads, None, "SSM has no attention heads");
    assert_eq!(grc.num_key_value_heads, None, "SSM has no KV heads");
    
    // SSM SHOULD have state space parameters
    assert!(grc.ssm_state_size.is_some(), "SSM should have state size");
    assert!(grc.ssm_expand.is_some(), "SSM should have expand factor");
    assert!(grc.ssm_conv_kernel.is_some(), "SSM should have conv kernel");
    
    // tied_embeddings should be true for SSM language models
    assert!(grc.tied_embeddings, "SSM LM should have tied embeddings");
}
