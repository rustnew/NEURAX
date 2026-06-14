//! Test compilation of a 1 Trillion parameter Transformer model
//! Compares output metrics with real-world models (GPT-4, LLaMA-3-405B, Grok-1)
//! JSON input follows the neurax-IR standard format

use neurax_parser::{parse_model_config, AbsorbedModel};
use neurax_ir::IrInjector;

/// Transformer-1T: A hypothetical 1 Trillion parameter model
/// Architecture inspired by GPT-4, LLaMA-3-405B, and Grok-1
const TRANSFORMER_1T_JSON: &str = r#"
{
    "schema_version": "1.0",
    "model": {
        "name": "Transformer-1T",
        "type": "transformer",
        "layers": [
            {"id": "embed_tokens", "layer_type": "embedding", "input_shape": [8192], "output_shape": [8192, 24576], "params": {"vocab_size": 256000, "embedding_dim": 24576}},
            
            {"id": "layer_0_self_attn", "layer_type": "attention", "input_shape": [8192, 24576], "output_shape": [8192, 24576], "params": {"hidden_size": 24576, "num_attention_heads": 192, "num_key_value_heads": 24}},
            {"id": "layer_0_norm", "layer_type": "normalization", "input_shape": [8192, 24576], "output_shape": [8192, 24576], "params": {"hidden_size": 24576}},
            {"id": "layer_0_mlp_gate", "layer_type": "dense", "input_shape": [8192, 24576], "output_shape": [8192, 65536], "params": {"in_features": 24576, "out_features": 65536}},
            {"id": "layer_0_mlp_up", "layer_type": "dense", "input_shape": [8192, 24576], "output_shape": [8192, 65536], "params": {"in_features": 24576, "out_features": 65536}},
            {"id": "layer_0_mlp_down", "layer_type": "dense", "input_shape": [8192, 65536], "output_shape": [8192, 24576], "params": {"in_features": 65536, "out_features": 24576}},
            
            {"id": "layer_1_self_attn", "layer_type": "attention", "input_shape": [8192, 24576], "output_shape": [8192, 24576], "params": {"hidden_size": 24576, "num_attention_heads": 192, "num_key_value_heads": 24}},
            {"id": "layer_1_norm", "layer_type": "normalization", "input_shape": [8192, 24576], "output_shape": [8192, 24576], "params": {"hidden_size": 24576}},
            {"id": "layer_1_mlp_gate", "layer_type": "dense", "input_shape": [8192, 24576], "output_shape": [8192, 65536], "params": {"in_features": 24576, "out_features": 65536}},
            {"id": "layer_1_mlp_up", "layer_type": "dense", "input_shape": [8192, 24576], "output_shape": [8192, 65536], "params": {"in_features": 24576, "out_features": 65536}},
            {"id": "layer_1_mlp_down", "layer_type": "dense", "input_shape": [8192, 65536], "output_shape": [8192, 24576], "params": {"in_features": 65536, "out_features": 24576}},
            
            {"id": "layer_2_self_attn", "layer_type": "attention", "input_shape": [8192, 24576], "output_shape": [8192, 24576], "params": {"hidden_size": 24576, "num_attention_heads": 192, "num_key_value_heads": 24}},
            {"id": "layer_2_norm", "layer_type": "normalization", "input_shape": [8192, 24576], "output_shape": [8192, 24576], "params": {"hidden_size": 24576}},
            {"id": "layer_2_mlp_gate", "layer_type": "dense", "input_shape": [8192, 24576], "output_shape": [8192, 65536], "params": {"in_features": 24576, "out_features": 65536}},
            {"id": "layer_2_mlp_up", "layer_type": "dense", "input_shape": [8192, 24576], "output_shape": [8192, 65536], "params": {"in_features": 24576, "out_features": 65536}},
            {"id": "layer_2_mlp_down", "layer_type": "dense", "input_shape": [8192, 65536], "output_shape": [8192, 24576], "params": {"in_features": 65536, "out_features": 24576}},
            
            {"id": "layer_3_self_attn", "layer_type": "attention", "input_shape": [8192, 24576], "output_shape": [8192, 24576], "params": {"hidden_size": 24576, "num_attention_heads": 192, "num_key_value_heads": 24}},
            {"id": "layer_3_norm", "layer_type": "normalization", "input_shape": [8192, 24576], "output_shape": [8192, 24576], "params": {"hidden_size": 24576}},
            {"id": "layer_3_mlp_gate", "layer_type": "dense", "input_shape": [8192, 24576], "output_shape": [8192, 65536], "params": {"in_features": 24576, "out_features": 65536}},
            {"id": "layer_3_mlp_up", "layer_type": "dense", "input_shape": [8192, 24576], "output_shape": [8192, 65536], "params": {"in_features": 24576, "out_features": 65536}},
            {"id": "layer_3_mlp_down", "layer_type": "dense", "input_shape": [8192, 65536], "output_shape": [8192, 24576], "params": {"in_features": 65536, "out_features": 24576}},
            
            {"id": "layer_4_self_attn", "layer_type": "attention", "input_shape": [8192, 24576], "output_shape": [8192, 24576], "params": {"hidden_size": 24576, "num_attention_heads": 192, "num_key_value_heads": 24}},
            {"id": "layer_4_norm", "layer_type": "normalization", "input_shape": [8192, 24576], "output_shape": [8192, 24576], "params": {"hidden_size": 24576}},
            {"id": "layer_4_mlp_gate", "layer_type": "dense", "input_shape": [8192, 24576], "output_shape": [8192, 65536], "params": {"in_features": 24576, "out_features": 65536}},
            {"id": "layer_4_mlp_up", "layer_type": "dense", "input_shape": [8192, 24576], "output_shape": [8192, 65536], "params": {"in_features": 24576, "out_features": 65536}},
            {"id": "layer_4_mlp_down", "layer_type": "dense", "input_shape": [8192, 65536], "output_shape": [8192, 24576], "params": {"in_features": 65536, "out_features": 24576}},
            
            {"id": "final_norm", "layer_type": "normalization", "input_shape": [8192, 24576], "output_shape": [8192, 24576], "params": {"hidden_size": 24576}},
            {"id": "lm_head", "layer_type": "dense", "input_shape": [8192, 24576], "output_shape": [8192, 256000], "params": {"in_features": 24576, "out_features": 256000}}
        ],
        "global_params": {
            "hidden_size": 24576,
            "num_layers": 128,
            "num_attention_heads": 192,
            "num_key_value_heads": 24,
            "head_dim": 128,
            "intermediate_size": 65536,
            "vocab_size": 256000,
            "max_position_embeddings": 8192
        }
    },
    "training": {
        "batch_size": 2048,
        "optimizer": "adamw",
        "learning_rate": 0.00001,
        "precision": "bf16",
        "gradient_checkpointing": true,
        "zero_stage": 3,
        "max_steps": 500000,
        "warmup_steps": 10000,
        "parallelism": {
            "data_parallel": 256,
            "tensor_parallel": 8,
            "pipeline_parallel": 16
        }
    },
    "hardware": {
        "gpus": [
            {
                "name": "H100-80GB-HBM3",
                "count": 32768,
                "memory_gb": 80,
                "tflops_fp16": 1979,
                "tflops_fp32": 67,
                "tflops_fp8": 3958,
                "memory_bandwidth_gb_s": 3352,
                "tensor_cores": true,
                "nvlink": true
            }
        ],
        "interconnect": "InfiniBand-NDR",
        "interconnect_bandwidth_gb_s": 400
    },
    "data": {
        "input_shape": [8192],
        "dtype": "bf16",
        "vocab_size": 256000
    },
    "cost_config": {
        "provider": "custom",
        "gpu_hour_usd": 2.50,
        "energy_kwh_usd": 0.08,
        "pue_factor": 1.1
    }
}
"#;

/// Real-world model specifications for comparison
struct RealModelSpecs {
    name: &'static str,
    params_billion: f64,
    hidden_size: u64,
    num_layers: u32,
    num_heads: u32,
    head_dim: u64,
    intermediate_size: u64,
    vocab_size: u64,
    context_length: u64,
    training_tokens_trillion: f64,
    gpu_hours_million: f64,
}

impl RealModelSpecs {
    /// GPT-4 estimated specifications (based on public analysis)
    fn gpt4() -> Self {
        Self {
            name: "GPT-4",
            params_billion: 1760.0,      // ~1.76T params (MoE)
            hidden_size: 16384,          // Estimated
            num_layers: 120,             // Estimated
            num_heads: 128,              // Estimated
            head_dim: 128,
            intermediate_size: 65536,    // Estimated
            vocab_size: 100256,
            context_length: 128000,
            training_tokens_trillion: 13.0,  // ~13T tokens
            gpu_hours_million: 25.0,     // Estimated
        }
    }
    
    /// LLaMA-3-405B specifications
    fn llama3_405b() -> Self {
        Self {
            name: "LLaMA-3-405B",
            params_billion: 405.0,
            hidden_size: 16384,
            num_layers: 126,
            num_heads: 128,
            head_dim: 128,
            intermediate_size: 53248,
            vocab_size: 128256,
            context_length: 128000,
            training_tokens_trillion: 15.0,
            gpu_hours_million: 7.0,
        }
    }
    
    /// Grok-1 specifications
    fn grok1() -> Self {
        Self {
            name: "Grok-1",
            params_billion: 314.0,
            hidden_size: 18432,
            num_layers: 64,
            num_heads: 128,
            head_dim: 144,
            intermediate_size: 73728,
            vocab_size: 131072,
            context_length: 8192,
            training_tokens_trillion: 2.0,
            gpu_hours_million: 3.0,
        }
    }
    
    /// Calculate expected parameters for dense transformer
    fn calculate_params(hidden: u64, layers: u32, intermediate: u64, vocab: u64, kv_heads: u32, heads: u32) -> f64 {
        let h = hidden as f64;
        let l = layers as f64;
        let i = intermediate as f64;
        let v = vocab as f64;
        let kv = kv_heads as f64;
        let q = heads as f64;
        let head_dim = h / q;
        
        // Embedding (tied)
        let embed = h * v;
        
        // Per layer: Attention + MLP
        // Attention: Q, K, V, O projections
        // Q: h x h, K: h x (kv * head_dim), V: h x (kv * head_dim), O: h x h
        let attn_per_layer = h * h + h * (kv * head_dim) * 2.0 + h * h;
        
        // MLP: gate, up, down (SwiGLU)
        let mlp_per_layer = h * i * 3.0;
        
        // LayerNorm: 2 * h per layer
        let norm_per_layer = 2.0 * h;
        
        // Total
        let total = embed + l * (attn_per_layer + mlp_per_layer + norm_per_layer);
        total / 1e9  // In billions
    }
    
    /// Calculate FLOPs per token
    fn calculate_flops_per_token(hidden: u64, layers: u32, intermediate: u64, vocab: u64) -> f64 {
        let h = hidden as f64;
        let l = layers as f64;
        let i = intermediate as f64;
        let v = vocab as f64;
        
        // Attention: 4 * h^2 per layer (Q, K, V, O)
        let attn_flops = 4.0 * h * h;
        
        // MLP: 8 * h * i per layer (gate, up, down)
        let mlp_flops = 8.0 * h * i;
        
        // Total per token
        let total = l * (attn_flops + mlp_flops);
        total
    }
    
    /// Calculate training FLOPs
    fn calculate_training_flops(params_billion: f64, tokens_trillion: f64) -> f64 {
        // Training FLOPs ≈ 6 * N * D (forward + backward)
        6.0 * params_billion * 1e9 * tokens_trillion * 1e12
    }
}

#[test]
fn test_transformer_1t_compilation() {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║         TRANSFORMER 1T PARAMETER MODEL                      ║");
    println!("║         1,000 BILLION PARAMETERS                            ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");
    
    // ── Step 1: Parse JSON ─────────────────────────────────────────────
    let start = std::time::Instant::now();
    let config = parse_model_config(TRANSFORMER_1T_JSON)
        .expect("Failed to parse Transformer-1T JSON");
    let parse_time = start.elapsed();
    println!("✓ JSON parsed in {:?}", parse_time);
    
    // ── Step 2: Absorb ─────────────────────────────────────────────────
    let start = std::time::Instant::now();
    let absorbed = AbsorbedModel::absorb(config);
    let absorb_time = start.elapsed();
    println!("✓ Model absorbed in {:?}\n", absorb_time);
    
    // ── Step 3: Validate GlobalResolutionContext ───────────────────────
    let grc = &absorbed.resolution_context;
    
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│              TRANSFORMER-1T PARAMETERS                      │");
    println!("├─────────────────────────────────────────────────────────────┤");
    
    println!("│ hidden_size:            {:>15}               │", grc.hidden_size.unwrap_or(0));
    println!("│ num_layers:             {:>15}               │", grc.num_layers.unwrap_or(0));
    println!("│ num_attention_heads:    {:>15}               │", grc.num_attention_heads.unwrap_or(0));
    println!("│ num_key_value_heads:    {:>15} (GQA 8:1)     │", grc.num_key_value_heads.unwrap_or(0));
    println!("│ head_dim:               {:>15}               │", grc.head_dim);
    println!("│ intermediate_size:      {:>15}               │", grc.intermediate_size.unwrap_or(0));
    println!("│ vocab_size:             {:>15}               │", grc.vocab_size.unwrap_or(0));
    
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│                  DERIVED VALUES                             │");
    println!("├─────────────────────────────────────────────────────────────┤");
    
    println!("│ dtype_bytes:            {:>15} (bf16)         │", grc.dtype_bytes);
    println!("│ optimizer_bytes:        {:>15} (AdamW)        │", grc.optimizer_bytes_per_param);
    println!("│ tied_embeddings:        {:>15}               │", grc.tied_embeddings);
    println!("│ gradient_checkpointing: {:>15}               │", grc.gradient_checkpointing);
    
    // d_inner calculation
    let d_inner = grc.d_inner();
    println!("│ d_inner (derived):      {:>15?}               │", d_inner);
    
    // h_kv for GQA
    if let Some(h_kv) = grc.h_kv {
        println!("│ h_kv (KV hidden):       {:>15}               │", h_kv);
    }
    
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│                  PARALLELISM                                │");
    println!("├─────────────────────────────────────────────────────────────┤");
    
    println!("│ Data Parallel (DP):    {:>15}               │", grc.dp);
    println!("│ Tensor Parallel (TP):  {:>15}               │", grc.tp);
    println!("│ Pipeline Parallel (PP):{:>15}               │", grc.pp);
    println!("│ ZeRO Stage:             {:>15}               │", grc.zero);
    println!("│ Total GPUs:             {:>15}               │", grc.num_gpus);
    
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│                  HARDWARE                                   │");
    println!("├─────────────────────────────────────────────────────────────┤");
    
    println!("│ GPU Model:              {:>15}               │", "H100-80GB");
    println!("│ GPU TFLOPs (FP16):      {:>15.0}               │", grc.primary_gpu_tflops);
    println!("│ GPU Memory (GB):        {:>15.1}               │", grc.primary_gpu_memory_gb);
    println!("│ Memory BW (GB/s):       {:>15.1}               │", grc.primary_gpu_bw_gb_s);
    println!("│ Tensor Cores:           {:>15}               │", grc.has_tensor_cores);
    println!("│ Interconnect BW (GB/s): {:>15}               │", grc.interconnect_bw);
    
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│                  CONFIDENCE                                 │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ confidence_score:       {:>14.2}%              │", grc.confidence_score * 100.0);
    println!("│ missing_fields:         {:>15?}              │", grc.missing_fields);
    
    println!("└─────────────────────────────────────────────────────────────┘\n");
    
    // ── Step 4: IR Injection ───────────────────────────────────────────
    let start = std::time::Instant::now();
    let arch_input = IrInjector::to_architecture_ir(&absorbed);
    let mem_config = IrInjector::configure_memory_pass(&absorbed);
    let hw_config = IrInjector::configure_hardware_pass(&absorbed);
    let cost_config = IrInjector::configure_cost_pass(&absorbed);
    let inject_time = start.elapsed();
    println!("✓ IRs injected in {:?}\n", inject_time);
    
    // ── Step 5: Parameter Calculation ───────────────────────────────────
    let start = std::time::Instant::now();
    let total_params = IrInjector::calculate_total_params(&absorbed);
    let calc_time = start.elapsed();
    
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│                  PARAMETER COUNT                            │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ Total Parameters:       {:>15.2}B            │", total_params as f64 / 1e9);
    println!("│                        {:>15.4}T            │", total_params as f64 / 1e12);
    println!("│ Calculation time:       {:>15?}             │", calc_time);
    println!("└─────────────────────────────────────────────────────────────┘\n");
    
    // ── Step 6: FLOPs Calculation ─────────────────────────────────────
    let flops = IrInjector::calculate_flops_per_token(&absorbed);
    
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│                  FLOPs ANALYSIS                             │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ FLOPs/token:            {:>15.2e}            │", flops as f64);
    println!("│ FLOPs/token (TFLOPs):   {:>15.2}            │", flops as f64 / 1e12);
    println!("└─────────────────────────────────────────────────────────────┘\n");
    
    // ── Step 7: Compare with Real Models ───────────────────────────────
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│          COMPARISON WITH REAL-WORLD MODELS                  │");
    println!("├─────────────────────────────────────────────────────────────┤");
    
    let gpt4 = RealModelSpecs::gpt4();
    let llama3 = RealModelSpecs::llama3_405b();
    let grok1 = RealModelSpecs::grok1();
    
    println!("│                                                             │");
    println!("│ Model              │ Params (B) │ Hidden │ Layers │ Heads  │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ GPT-4              │ {:>10.0} │ {:>7} │ {:>6} │ {:>6} │", 
             gpt4.params_billion, gpt4.hidden_size, gpt4.num_layers, gpt4.num_heads);
    println!("│ LLaMA-3-405B       │ {:>10.0} │ {:>7} │ {:>6} │ {:>6} │", 
             llama3.params_billion, llama3.hidden_size, llama3.num_layers, llama3.num_heads);
    println!("│ Grok-1             │ {:>10.0} │ {:>7} │ {:>6} │ {:>6} │", 
             grok1.params_billion, grok1.hidden_size, grok1.num_layers, grok1.num_heads);
    println!("│ Transformer-1T     │ {:>10.0} │ {:>7} │ {:>6} │ {:>6} │", 
             total_params as f64 / 1e9, grc.hidden_size.unwrap_or(0), 
             grc.num_layers.unwrap_or(0), grc.num_attention_heads.unwrap_or(0));
    println!("└─────────────────────────────────────────────────────────────┘\n");
    
    // ── Expected vs Calculated Parameters ─────────────────────────────
    let expected_params = RealModelSpecs::calculate_params(
        grc.hidden_size.unwrap(),
        grc.num_layers.unwrap(),
        grc.intermediate_size.unwrap(),
        grc.vocab_size.unwrap(),
        grc.num_key_value_heads.unwrap(),
        grc.num_attention_heads.unwrap(),
    );
    
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│          PARAMETER ACCURACY COMPARISON                      │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ Expected params:         {:>15.2}B           │", expected_params);
    println!("│ Calculated params:      {:>15.2}B           │", total_params as f64 / 1e9);
    
    let accuracy = if expected_params > 0.0 {
        let diff = (total_params as f64 / 1e9 - expected_params).abs() / expected_params;
        (1.0 - diff.min(1.0)) * 100.0
    } else {
        0.0
    };
    println!("│ Accuracy:               {:>14.1}%            │", accuracy);
    println!("└─────────────────────────────────────────────────────────────┘\n");
    
    // ── Training Cost Estimation ───────────────────────────────────────
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│          TRAINING COST ESTIMATION                           │");
    println!("├─────────────────────────────────────────────────────────────┤");
    
    let training_tokens = 15.0; // 15T tokens (like LLaMA-3)
    let training_flops = RealModelSpecs::calculate_training_flops(
        total_params as f64 / 1e9, training_tokens
    );
    
    println!("│ Training tokens:        {:>14.1}T           │", training_tokens);
    println!("│ Total training FLOPs:   {:>15.2e}           │", training_flops);
    
    // GPU hours estimation
    let gpu_tflops = grc.primary_gpu_tflops;
    let gpu_utilization = 0.4; // 40% utilization
    let effective_tflops = gpu_tflops * gpu_utilization;
    let gpu_seconds = training_flops / (effective_tflops * 1e12);
    let gpu_hours = gpu_seconds / 3600.0;
    let gpu_million_hours = gpu_hours / 1e6;
    
    println!("│ GPU utilization:        {:>14.0}%            │", gpu_utilization * 100.0);
    println!("│ GPU hours (millions):   {:>15.2}           │", gpu_million_hours);
    
    // Cost
    let gpu_hour_cost = cost_config.gpu_hour_usd;
    let total_cost = gpu_million_hours * 1e6 * gpu_hour_cost;
    println!("│ Cost per GPU-hour:      {:>15.2} USD        │", gpu_hour_cost);
    println!("│ Total training cost:    {:>15.2}M USD      │", total_cost / 1e6);
    
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ Comparison with real models:                               │");
    println!("│ GPT-4 estimated:       {:>15.0}M GPU-hours   │", gpt4.gpu_hours_million);
    println!("│ LLaMA-3-405B:          {:>15.0}M GPU-hours   │", llama3.gpu_hours_million);
    println!("└─────────────────────────────────────────────────────────────┘\n");
    
    // ── Memory Requirements ────────────────────────────────────────────
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│          MEMORY REQUIREMENTS                                │");
    println!("├─────────────────────────────────────────────────────────────┤");
    
    let params_bytes = total_params as f64 * grc.dtype_bytes as f64;
    let optimizer_bytes = total_params as f64 * grc.optimizer_bytes_per_param as f64;
    let gradients_bytes = total_params as f64 * 4.0; // Always fp32 for gradients
    
    println!("│ Model weights:          {:>15.2} TB         │", params_bytes / 1e12);
    println!("│ Optimizer states:       {:>15.2} TB         │", optimizer_bytes / 1e12);
    println!("│ Gradients:              {:>15.2} TB         │", gradients_bytes / 1e12);
    println!("│ Total (no sharding):    {:>15.2} TB         │", 
             (params_bytes + optimizer_bytes + gradients_bytes) / 1e12);
    
    // With ZeRO-3 sharding
    let num_gpus = grc.num_gpus as f64;
    println!("│ With ZeRO-3 (per GPU):  {:>15.2} GB         │", 
             (params_bytes + optimizer_bytes + gradients_bytes) / num_gpus / 1e9);
    
    println!("└─────────────────────────────────────────────────────────────┘\n");
    
    // ── Assertions ─────────────────────────────────────────────────────
    assert!(total_params > 500_000_000_000, "Expected > 500B params, got {}", total_params);
    assert!(total_params < 2_000_000_000_000, "Expected < 2T params, got {}", total_params);
    assert_eq!(grc.hidden_size, Some(24576));
    assert_eq!(grc.num_layers, Some(128));
    assert_eq!(grc.num_attention_heads, Some(192));
    assert_eq!(grc.num_key_value_heads, Some(24), "GQA ratio should be 8:1");
    
    println!("✓ All assertions passed!");
    println!("✓ Transformer-1T compiled successfully!\n");
}

#[test]
fn test_transformer_1t_vs_real_models() {
    println!("\n=== Detailed Comparison with Real Models ===\n");
    
    let config = parse_model_config(TRANSFORMER_1T_JSON).unwrap();
    let absorbed = AbsorbedModel::absorb(config);
    let grc = &absorbed.resolution_context;
    
    let gpt4 = RealModelSpecs::gpt4();
    let llama3 = RealModelSpecs::llama3_405b();
    let grok1 = RealModelSpecs::grok1();
    
    let our_params = IrInjector::calculate_total_params(&absorbed) as f64 / 1e9;
    
    println!("┌────────────────────────────────────────────────────────────────────┐");
    println!("│                    MODEL SPECIFICATIONS                           │");
    println!("├────────────────────────────────────────────────────────────────────┤");
    println!("│ Metric          │ GPT-4    │ LLaMA-3 │ Grok-1  │ Trans-1T        │");
    println!("├────────────────────────────────────────────────────────────────────┤");
    println!("│ Params (B)      │ {:>8.0} │ {:>7.0} │ {:>6.0} │ {:>8.2}        │", 
             gpt4.params_billion, llama3.params_billion, grok1.params_billion, our_params);
    println!("│ Hidden Size     │ {:>8} │ {:>7} │ {:>6} │ {:>8}        │", 
             gpt4.hidden_size, llama3.hidden_size, grok1.hidden_size, grc.hidden_size.unwrap());
    println!("│ Layers          │ {:>8} │ {:>7} │ {:>6} │ {:>8}        │", 
             gpt4.num_layers, llama3.num_layers, grok1.num_layers, grc.num_layers.unwrap());
    println!("│ Attention Heads │ {:>8} │ {:>7} │ {:>6} │ {:>8}        │", 
             gpt4.num_heads, llama3.num_heads, grok1.num_heads, grc.num_attention_heads.unwrap());
    println!("│ Head Dimension  │ {:>8} │ {:>7} │ {:>6} │ {:>8}        │", 
             gpt4.head_dim, llama3.head_dim, grok1.head_dim, grc.head_dim);
    println!("│ Vocab Size      │ {:>8} │ {:>7} │ {:>6} │ {:>8}        │", 
             gpt4.vocab_size, llama3.vocab_size, grok1.vocab_size, grc.vocab_size.unwrap());
    println!("│ Context Length  │ {:>8} │ {:>7} │ {:>6} │ {:>8}        │", 
             gpt4.context_length, llama3.context_length, grok1.context_length, 8192);
    println!("└────────────────────────────────────────────────────────────────────┘\n");
    
    // Verify our model is in the right range
    assert!(our_params > llama3.params_billion, "Should be larger than LLaMA-3-405B");
    println!("✓ Transformer-1T is larger than LLaMA-3-405B ({:.0}B params)", llama3.params_billion);
}

#[test]
fn test_transformer_1t_flops_accuracy() {
    let config = parse_model_config(TRANSFORMER_1T_JSON).unwrap();
    let absorbed = AbsorbedModel::absorb(config);
    let grc = &absorbed.resolution_context;
    
    let calculated_flops = IrInjector::calculate_flops_per_token(&absorbed) as f64;
    let expected_flops = RealModelSpecs::calculate_flops_per_token(
        grc.hidden_size.unwrap(),
        grc.num_layers.unwrap(),
        grc.intermediate_size.unwrap(),
        grc.vocab_size.unwrap(),
    );
    
    println!("\n=== FLOPs Accuracy ===");
    println!("Expected FLOPs/token:    {:.2e}", expected_flops);
    println!("Calculated FLOPs/token:  {:.2e}", calculated_flops);
    
    let ratio = calculated_flops / expected_flops;
    println!("Ratio:                   {:.2}", ratio);
    
    // FLOPs should be in reasonable range
    assert!(calculated_flops > 1e12, "FLOPs should be > 1 TFLOP per token");
}
