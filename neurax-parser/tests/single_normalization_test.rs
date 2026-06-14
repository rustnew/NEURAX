//! Test compilation of a single normalization block
//! Shows detailed absorption results for normalization layers (LayerNorm, RMSNorm)

use neurax_parser::{parse_model_config, AbsorbedModel};
use neurax_ir::IrInjector;

/// Single normalization block JSON (RMSNorm style)
const SINGLE_NORM_JSON: &str = r#"
{
    "schema_version": "1.0",
    "model": {
        "name": "SingleNorm-Block",
        "type": "transformer",
        "layers": [
            {
                "id": "rms_norm_0",
                "layer_type": "normalization",
                "input_shape": [2048, 4096],
                "output_shape": [2048, 4096],
                "params": {
                    "hidden_size": 4096,
                    "norm_type": "rms",
                    "epsilon": 1e-6
                }
            }
        ],
        "global_params": {
            "hidden_size": 4096,
            "num_layers": 1,
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
fn test_single_normalization_block() {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║      SINGLE NORMALIZATION BLOCK COMPILATION                ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");
    
    // ── Parse ──────────────────────────────────────────────────────────
    let config = parse_model_config(SINGLE_NORM_JSON)
        .expect("Failed to parse normalization JSON");
    println!("✓ JSON parsed successfully");
    
    // ── Absorb ─────────────────────────────────────────────────────────
    let absorbed = AbsorbedModel::absorb(config);
    println!("✓ Model absorbed\n");
    
    // ── Detailed Results ──────────────────────────────────────────────
    let grc = &absorbed.resolution_context;
    
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│                NORMALIZATION PARAMETERS                     │");
    println!("├─────────────────────────────────────────────────────────────┤");
    
    // Hidden size
    println!("│ hidden_size (H):        {:>10}                     │", 
             grc.hidden_size.unwrap_or(0));
    
    // Number of layers
    println!("│ num_layers:             {:>10}                     │", 
             grc.num_layers.unwrap_or(0));
    
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│                  DERIVED VALUES                             │");
    println!("├─────────────────────────────────────────────────────────────┤");
    
    // dtype
    println!("│ dtype_bytes:            {:>10} (bf16)              │", grc.dtype_bytes);
    
    // optimizer
    println!("│ optimizer_bytes:        {:>10} (AdamW)             │", grc.optimizer_bytes_per_param);
    
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│                  SYMBOL TABLE                               │");
    println!("├─────────────────────────────────────────────────────────────┤");
    
    for (key, value) in &grc.symbol_table {
        println!("│ {::<20} = {:>10}                     │", key, value);
    }
    
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│                  CONFIDENCE                                 │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ confidence_score:       {:>10.2}%                   │", grc.confidence_score * 100.0);
    println!("│ missing_fields:         {:>10?}              │", grc.missing_fields);
    
    println!("└─────────────────────────────────────────────────────────────┘\n");
    
    // ── Layer Details ─────────────────────────────────────────────────
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│                  LAYER DETAILS                              │");
    println!("├─────────────────────────────────────────────────────────────┤");
    
    let layers = &absorbed.config.model.layers;
    for layer in layers {
        println!("│ Layer ID:       {:>42} │", layer.id);
        println!("│ Layer Type:     {:>42} │", format!("{:?}", layer.layer_type));
        println!("│ Input Shape:    {:>42?} │", layer.input_shape);
        println!("│ Output Shape:   {:>42?} │", layer.output_shape);
        
        // Show key params from LayerParams struct
        println!("│ Params:                                                 │");
        if let Some(h) = layer.params.hidden_size {
            println!("│   hidden_size:          {:>20}          │", h);
        }
        if let Some(activation) = &layer.params.activation {
            println!("│   activation:           {:>20}          │", activation);
        }
        println!("│   causal:               {:>20}          │", layer.params.causal);
        println!("│   gated:                {:>20}          │", layer.params.gated);
    }
    
    println!("└─────────────────────────────────────────────────────────────┘\n");
    
    // ── Parameter Calculation ─────────────────────────────────────────
    let total_params = IrInjector::calculate_total_params(&absorbed);
    
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│                  PARAMETER COUNT                            │");
    println!("├─────────────────────────────────────────────────────────────┤");
    
    // Normalization params: weight (gamma) + optional bias (beta)
    let h = grc.hidden_size.unwrap() as f64;
    
    // RMSNorm: only weight (gamma), no bias
    let rms_params = h;  // just gamma
    println!("│ RMSNorm weight (γ):     {:>12.0} params           │", rms_params);
    
    // LayerNorm: weight + bias
    let ln_params = h * 2.0;  // gamma + beta
    println!("│ LayerNorm (γ + β):      {:>12.0} params           │", ln_params);
    
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ Calculated total:       {:>12.0} params           │", total_params);
    println!("│                        {:>10.2}K                    │", total_params as f64 / 1e3);
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
    
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│                  IR INJECTION RESULTS                       │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ ArchitectureIR:                                         │");
    println!("│   hidden_size:          {:>10?}                     │", arch_input.hidden_size);
    println!("│   num_layers:           {:>10?}                     │", arch_input.num_layers);
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ MemoryPass:                                             │");
    println!("│   dtype_bytes:          {:>10}                     │", mem_config.dtype_bytes);
    println!("│   optimizer_bytes:      {:>10}                     │", mem_config.optimizer_bytes);
    println!("└─────────────────────────────────────────────────────────────┘\n");
    
    // ── Assertions ────────────────────────────────────────────────────
    assert_eq!(grc.hidden_size, Some(4096));
    assert_eq!(grc.num_layers, Some(1));
    
    println!("✓ All assertions passed!");
    println!("✓ Single normalization block compiled successfully!\n");
}

#[test]
fn test_norm_types_comparison() {
    println!("\n=== Normalization Types Comparison ===\n");
    
    let h = 4096f64;
    
    println!("┌───────────────────┬───────────────┬───────────────────────┐");
    println!("│ Normalization     │ Parameters    │ Formula               │");
    println!("├───────────────────┼───────────────┼───────────────────────┤");
    println!("│ RMSNorm           │ {:>10.0}    │ γ * x / √(mean(x²)+ε) │", h);
    println!("│ LayerNorm         │ {:>10.0}    │ γ * (x-μ)/σ + β       │", h * 2.0);
    println!("│ BatchNorm         │ {:>10.0}    │ γ * (x-μ)/σ + β       │", h * 2.0);
    println!("│ GroupNorm         │ {:>10.0}    │ γ * (x-μ)/σ + β       │", h * 2.0);
    println!("└───────────────────┴───────────────┴───────────────────────┘\n");
    
    println!("RMSNorm (used in LLaMA):");
    println!("  - No bias term (β)");
    println!("  - No mean subtraction");
    println!("  - Simpler computation: x * γ / √(mean(x²) + ε)");
    println!("  - Parameters: {} (weight only)\n", h as u64);
    
    println!("LayerNorm (used in BERT/GPT):");
    println!("  - Has both weight (γ) and bias (β)");
    println!("  - Full normalization: (x - μ) / σ");
    println!("  - Parameters: {} (weight + bias)\n", (h * 2.0) as u64);
}
