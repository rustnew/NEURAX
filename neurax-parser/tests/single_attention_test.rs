//! Test compilation of a single attention block
//! Shows detailed absorption results for one attention layer

use neurax_parser::{parse_model_config, AbsorbedModel};
use neurax_ir::IrInjector;

/// Single attention block JSON
const SINGLE_ATTENTION_JSON: &str = r#"
{
    "schema_version": "1.0",
    "model": {
        "name": "SingleAttention-Block",
        "type": "transformer",
        "layers": [
            {
                "id": "layer_0_self_attn",
                "layer_type": "attention",
                "input_shape": [2048, 4096],
                "output_shape": [2048, 4096],
                "params": {
                    "hidden_size": 4096,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 8
                }
            }
        ],
        "global_params": {
            "hidden_size": 4096,
            "num_layers": 1,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 32000
        }
    },
    "training": {
        "batch_size": 32,
        "optimizer": "adamw",
        "learning_rate": 0.0001,
        "precision": "bf16",
        "gradient_checkpointing": false,
        "zero_stage": 0,
        "max_steps": 1000,
        "warmup_steps": 100,
        "parallelism": {
            "data_parallel": 1,
            "tensor_parallel": 1,
            "pipeline_parallel": 1
        }
    },
    "hardware": {
        "gpus": [
            {
                "name": "A100-80GB",
                "count": 1,
                "memory_gb": 80,
                "tflops_fp16": 312,
                "tflops_fp32": 19.5,
                "tflops_fp8": 624,
                "memory_bandwidth_gb_s": 2039,
                "tensor_cores": true,
                "nvlink": false
            }
        ],
        "interconnect": "None",
        "interconnect_bandwidth_gb_s": 0
    },
    "data": {
        "input_shape": [2048, 4096],
        "dtype": "bf16",
        "vocab_size": 32000
    },
    "cost_config": {
        "provider": "local",
        "gpu_hour_usd": 0.0,
        "energy_kwh_usd": 0.0,
        "pue_factor": 1.0
    }
}
"#;

#[test]
fn test_single_attention_block() {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║      SINGLE ATTENTION BLOCK COMPILATION                    ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");
    
    // ── Parse ──────────────────────────────────────────────────────────
    let config = parse_model_config(SINGLE_ATTENTION_JSON)
        .expect("Failed to parse attention JSON");
    println!("✓ JSON parsed successfully");
    
    // ── Absorb ─────────────────────────────────────────────────────────
    let absorbed = AbsorbedModel::absorb(config);
    println!("✓ Model absorbed\n");
    
    // ── Detailed Results ──────────────────────────────────────────────
    let grc = &absorbed.resolution_context;
    
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│                  ATTENTION PARAMETERS                       │");
    println!("├─────────────────────────────────────────────────────────────┤");
    
    // Hidden size
    println!("│ hidden_size (H):        {:>10}                     │", 
             grc.hidden_size.unwrap_or(0));
    
    // Number of attention heads (Q)
    println!("│ num_attention_heads:    {:>10}                     │", 
             grc.num_attention_heads.unwrap_or(0));
    
    // Number of KV heads (GQA)
    println!("│ num_key_value_heads:    {:>10} (GQA enabled)        │", 
             grc.num_key_value_heads.unwrap_or(0));
    
    // Head dimension
    println!("│ head_dim:               {:>10}                     │", 
             grc.head_dim);
    
    // H_kv (KV hidden size for GQA)
    if let Some(h_kv) = grc.h_kv {
        println!("│ h_kv (KV hidden):       {:>10}                     │", h_kv);
    }
    
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│                  DERIVED VALUES                             │");
    println!("├─────────────────────────────────────────────────────────────┤");
    
    // dtype
    println!("│ dtype_bytes:            {:>10} (bf16)              │", grc.dtype_bytes);
    
    // optimizer
    println!("│ optimizer_bytes:        {:>10} (AdamW)             │", grc.optimizer_bytes_per_param);
    
    // Tied embeddings
    println!("│ tied_embeddings:        {:>10}                     │", grc.tied_embeddings);
    
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│                  SYMBOL TABLE                               │");
    println!("├─────────────────────────────────────────────────────────────┤");
    
    for (key, value) in &grc.symbol_table {
        println!("│ {::<20} = {:>10}                     │", key, value);
    }
    
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│                  HARDWARE                                   │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ num_gpus:               {:>10}                     │", grc.num_gpus);
    println!("│ GPU TFLOPs:             {:>10.1}                     │", grc.primary_gpu_tflops);
    println!("│ GPU Memory (GB):        {:>10.1}                     │", grc.primary_gpu_memory_gb);
    
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│                  CONFIDENCE                                 │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ confidence_score:       {:>10.2}%                   │", grc.confidence_score * 100.0);
    println!("│ missing_fields:         {:>10?}              │", grc.missing_fields);
    
    println!("└─────────────────────────────────────────────────────────────┘\n");
    
    // ── Parameter Calculation ─────────────────────────────────────────
    let total_params = IrInjector::calculate_total_params(&absorbed);
    
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│                  PARAMETER COUNT                            │");
    println!("├─────────────────────────────────────────────────────────────┤");
    
    // Attention params breakdown
    let h = grc.hidden_size.unwrap() as f64;
    let heads = grc.num_attention_heads.unwrap() as f64;
    let kv_heads = grc.num_key_value_heads.unwrap_or(grc.num_attention_heads.unwrap()) as f64;
    let head_dim = grc.head_dim as f64;
    
    // Q projection: H x H
    let q_params = h * h;
    println!("│ Q projection:           {:>12.0} params           │", q_params);
    
    // K projection: H x (kv_heads * head_dim)
    let k_params = h * (kv_heads * head_dim);
    println!("│ K projection:           {:>12.0} params           │", k_params);
    
    // V projection: H x (kv_heads * head_dim)
    let v_params = h * (kv_heads * head_dim);
    println!("│ V projection:           {:>12.0} params           │", v_params);
    
    // O projection: H x H
    let o_params = h * h;
    println!("│ O projection:           {:>12.0} params           │", o_params);
    
    let attn_total = q_params + k_params + v_params + o_params;
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ ATTENTION TOTAL:        {:>12.0} params           │", attn_total);
    println!("│                       {:>10.2}M                    │", attn_total / 1e6);
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ Calculated total:       {:>12.0} params           │", total_params);
    println!("│                        {:>10.2}M                    │", total_params as f64 / 1e6);
    println!("└─────────────────────────────────────────────────────────────┘\n");
    
    // ── FLOPs Calculation ─────────────────────────────────────────────
    let flops = IrInjector::calculate_flops_per_token(&absorbed);
    
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│                  FLOPs PER TOKEN                            │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ FLOPs/token:            {:>12.2e}                │", flops as f64);
    println!("└─────────────────────────────────────────────────────────────┘\n");
    
    // ── IR Injection ──────────────────────────────────────────────────
    let arch_input = IrInjector::to_architecture_ir(&absorbed);
    let mem_config = IrInjector::configure_memory_pass(&absorbed);
    let hw_config = IrInjector::configure_hardware_pass(&absorbed);
    
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│                  IR INJECTION RESULTS                       │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ ArchitectureIR:                                         │");
    println!("│   hidden_size:          {:>10?}                     │", arch_input.hidden_size);
    println!("│   num_layers:           {:>10?}                     │", arch_input.num_layers);
    println!("│   head_dim:             {:>10}                     │", arch_input.head_dim);
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ MemoryPass:                                             │");
    println!("│   dtype_bytes:          {:>10}                     │", mem_config.dtype_bytes);
    println!("│   optimizer_bytes:      {:>10}                     │", mem_config.optimizer_bytes);
    println!("│   num_kv_heads:         {:>10}                     │", mem_config.num_kv_heads);
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ HardwarePass:                                           │");
    println!("│   gpu_tflops_fp16:      {:>10.1}                     │", hw_config.gpu_tflops_fp16);
    println!("│   has_tensor_cores:     {:>10}                     │", hw_config.has_tensor_cores);
    println!("└─────────────────────────────────────────────────────────────┘\n");
    
    // ── Assertions ────────────────────────────────────────────────────
    assert_eq!(grc.hidden_size, Some(4096));
    assert_eq!(grc.num_attention_heads, Some(32));
    assert_eq!(grc.num_key_value_heads, Some(8), "GQA with 8 KV heads");
    assert_eq!(grc.head_dim, 128);
    assert!(grc.h_kv.is_some(), "h_kv should be calculated for GQA");
    
    println!("✓ All assertions passed!");
    println!("✓ Single attention block compiled successfully!\n");
}
