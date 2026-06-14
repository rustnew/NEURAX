//! Integration test for JSON absorption with LLaMA-like model
//! Validates that every JSON field is correctly absorbed and injected into IRs

use neurax_parser::{parse_model_config, AbsorbedModel};
use neurax_ir::IrInjector;

/// LLaMA-7B style JSON for testing complete absorption
const LLAMA_7B_JSON: &str = r#"
{
    "schema_version": "1.0",
    "model": {
        "name": "LLaMA-7B",
        "type": "transformer",
        "layers": [
            {
                "id": "embed_tokens",
                "layer_type": "embedding",
                "input_shape": [2048],
                "output_shape": [4096],
                "params": {
                    "vocab_size": 32000,
                    "embedding_dim": 4096
                }
            },
            {
                "id": "layers_0_self_attn",
                "layer_type": "attention",
                "input_shape": [2048, 4096],
                "output_shape": [2048, 4096],
                "params": {
                    "hidden_size": 4096,
                    "num_heads": 32,
                    "num_kv_heads": 8
                }
            },
            {
                "id": "layers_0_mlp",
                "layer_type": "mlp",
                "input_shape": [2048, 4096],
                "output_shape": [2048, 11008],
                "params": {
                    "hidden_size": 4096,
                    "intermediate_size": 11008
                }
            }
        ],
        "global_params": {
            "hidden_size": 4096,
            "num_layers": 32,
            "vocab_size": 32000,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128
        }
    },
    "training": {
        "batch_size": 32,
        "optimizer": "adamw",
        "learning_rate": 0.0001,
        "precision": "bf16",
        "gradient_checkpointing": true,
        "zero_stage": 2,
        "max_steps": 10000,
        "warmup_steps": 1000,
        "parallelism": {
            "data_parallel": 2,
            "tensor_parallel": 1,
            "pipeline_parallel": 1
        }
    },
    "hardware": {
        "gpus": [
            {
                "name": "A100-80GB",
                "count": 8,
                "memory_gb": 80,
                "tflops_fp16": 312,
                "tflops_fp32": 19.5,
                "tflops_fp8": 624,
                "memory_bandwidth_gb_s": 2039,
                "tensor_cores": true,
                "nvlink": true
            }
        ],
        "interconnect": "NVLink",
        "interconnect_bandwidth_gb_s": 600
    },
    "data": {
        "input_shape": [2048],
        "dtype": "bf16",
        "vocab_size": 32000
    },
    "cost_config": {
        "provider": "aws",
        "gpu_hour_usd": 4.35,
        "energy_kwh_usd": 0.12,
        "pue_factor": 1.2
    }
}
"#;

#[test]
fn test_llama7b_absorption() {
    // Parse JSON
    let config = parse_model_config(LLAMA_7B_JSON)
        .expect("Failed to parse LLaMA-7B JSON");
    
    // Absorb into AbsorbedModel
    let absorbed = AbsorbedModel::absorb(config);
    
    // ── Validate GlobalResolutionContext ─────────────────────────────────
    let grc = &absorbed.resolution_context;
    
    // Core dimensions
    assert_eq!(grc.hidden_size, Some(4096), "hidden_size should be 4096");
    assert_eq!(grc.num_layers, Some(32), "num_layers should be 32");
    assert_eq!(grc.vocab_size, Some(32000), "vocab_size should be 32000");
    assert_eq!(grc.intermediate_size, Some(11008), "intermediate_size should be 11008");
    assert_eq!(grc.num_attention_heads, Some(32), "num_attention_heads should be 32");
    assert_eq!(grc.num_key_value_heads, Some(8), "num_key_value_heads should be 8 (GQA)");
    assert_eq!(grc.head_dim, 128, "head_dim should be 128");
    
    // Derived values
    assert_eq!(grc.dtype_bytes, 2, "bf16 should be 2 bytes");
    assert_eq!(grc.optimizer_bytes_per_param, 8, "AdamW should be 8 bytes");
    assert!(grc.h_kv.is_some(), "h_kv should be calculated for GQA");
    
    // Boolean flags
    assert!(grc.gradient_checkpointing, "gradient_checkpointing should be true");
    assert!(grc.tied_embeddings, "tied_embeddings should be true for transformer");
    
    // Hardware
    assert_eq!(grc.num_gpus, 8, "num_gpus should be 8");
    assert!((grc.primary_gpu_tflops - 312.0).abs() < 0.1, "GPU TFLOPs should be 312");
    assert!((grc.primary_gpu_memory_gb - 80.0).abs() < 0.1, "GPU memory should be 80GB");
    
    // Parallelism
    assert_eq!(grc.dp, 2, "Data parallel should be 2");
    assert_eq!(grc.tp, 1, "Tensor parallel should be 1");
    assert_eq!(grc.pp, 1, "Pipeline parallel should be 1");
    assert_eq!(grc.zero, 2, "ZeRO stage should be 2");
    
    // Symbol table
    assert!(grc.symbol_table.contains_key("B"), "B should be in symbol table");
    assert!(grc.symbol_table.contains_key("H"), "H should be in symbol table");
    assert!(grc.symbol_table.contains_key("V"), "V should be in symbol table");
    
    // Confidence score
    assert!(grc.confidence_score > 0.5, "Confidence score should be > 0.5");
    
    // ── Validate IR Injection ───────────────────────────────────────────
    let arch_input = IrInjector::to_architecture_ir(&absorbed);
    
    assert_eq!(arch_input.hidden_size, Some(4096));
    assert_eq!(arch_input.num_layers, Some(32));
    assert_eq!(arch_input.vocab_size, Some(32000));
    assert_eq!(arch_input.head_dim, 128);
    
    // Memory config
    let mem_config = IrInjector::configure_memory_pass(&absorbed);
    
    assert_eq!(mem_config.dtype_bytes, 2);
    assert_eq!(mem_config.optimizer_bytes, 8);
    assert!(mem_config.checkpointing_enabled);
    assert_eq!(mem_config.zero_stage, 2);
    assert_eq!(mem_config.num_gpus, 8);
    assert_eq!(mem_config.num_kv_heads, 8);
    
    // Hardware config
    let hw_config = IrInjector::configure_hardware_pass(&absorbed);
    
    assert!((hw_config.gpu_tflops_fp16 - 312.0).abs() < 0.1);
    assert!(hw_config.has_tensor_cores);
    assert_eq!(hw_config.dp, 2);
    
    // Cost config
    let cost_config = IrInjector::configure_cost_pass(&absorbed);
    
    assert!((cost_config.gpu_hour_usd - 4.35).abs() < 0.01);
    assert!((cost_config.energy_kwh_usd - 0.12).abs() < 0.01);
    assert_eq!(cost_config.num_gpus, 8);
    
    // ── Validate Parameter Calculation ───────────────────────────────────
    let total_params = IrInjector::calculate_total_params(&absorbed);
    
    // Our calculation: embed + 32*(attn + mlp) + norms
    // = 131M + 32 * 177M + 266K ≈ 5.8B
    // (LLaMA-7B has ~6.7B with additional components we don't model)
    assert!(total_params > 5_500_000_000, "Expected > 5.5B params, got {}", total_params);
    assert!(total_params < 6_500_000_000, "Expected < 6.5B params, got {}", total_params);
    
    // ── Validate FLOPs Calculation ───────────────────────────────────────
    let flops = IrInjector::calculate_flops_per_token(&absorbed);
    
    // Should be non-zero for a valid model
    assert!(flops > 0, "FLOPs per token should be > 0");
}

#[test]
fn test_symbol_resolution() {
    let config = parse_model_config(LLAMA_7B_JSON).unwrap();
    let absorbed = AbsorbedModel::absorb(config);
    
    // Test symbol table resolution
    let grc = &absorbed.resolution_context;
    
    // B should resolve to batch_size
    assert_eq!(grc.symbol_table.get("B"), Some(&32u64));
    
    // H should resolve to hidden_size
    assert_eq!(grc.symbol_table.get("H"), Some(&4096u64));
    
    // V should resolve to vocab_size
    assert_eq!(grc.symbol_table.get("V"), Some(&32000u64));
    
    // dtype_bytes should be 2 for bf16
    assert_eq!(grc.symbol_table.get("dtype_bytes"), Some(&2u64));
}
