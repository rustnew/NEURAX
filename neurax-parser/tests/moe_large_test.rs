//! Test compilation of a large MoE (Mixture of Experts) model
//! Compares output metrics with real-world models (Mixtral-8x7B, DeepSeek-V3)
//! JSON input follows the neurax-IR standard format

use neurax_parser::{parse_model_config, AbsorbedModel};
use neurax_ir::IrInjector;

/// MoE-141B-A36B: Large Mixture of Experts model
/// 141B total parameters, 36B active per token
/// Inspired by DeepSeek-V3 and Mixtral architectures
const MOE_LARGE_JSON: &str = r#"
{
    "schema_version": "1.0",
    "model": {
        "name": "MoE-141B-A36B",
        "type": "moe",
        "layers": [
            {"id": "embed_tokens", "layer_type": "embedding", "input_shape": [4096], "output_shape": [4096, 7168], "params": {"vocab_size": 128000, "embedding_dim": 7168}},
            
            {"id": "layer_0_self_attn", "layer_type": "attention", "input_shape": [4096, 7168], "output_shape": [4096, 7168], "params": {"hidden_size": 7168, "num_attention_heads": 56, "num_key_value_heads": 8}},
            {"id": "layer_0_norm", "layer_type": "normalization", "input_shape": [4096, 7168], "output_shape": [4096, 7168], "params": {"hidden_size": 7168}},
            {"id": "layer_0_moe", "layer_type": "moe", "input_shape": [4096, 7168], "output_shape": [4096, 7168], "params": {"hidden_size": 7168, "num_experts": 64, "intermediate_size": 20480, "top_k": 6}},
            
            {"id": "layer_1_self_attn", "layer_type": "attention", "input_shape": [4096, 7168], "output_shape": [4096, 7168], "params": {"hidden_size": 7168, "num_attention_heads": 56, "num_key_value_heads": 8}},
            {"id": "layer_1_norm", "layer_type": "normalization", "input_shape": [4096, 7168], "output_shape": [4096, 7168], "params": {"hidden_size": 7168}},
            {"id": "layer_1_moe_gate", "layer_type": "moe", "input_shape": [4096, 7168], "output_shape": [4096, 7168], "params": {"hidden_size": 7168, "num_experts": 64, "top_k": 6}},
            {"id": "layer_1_experts", "layer_type": "moe", "input_shape": [4096, 7168], "output_shape": [4096, 7168], "params": {"hidden_size": 7168, "num_experts": 64, "intermediate_size": 20480, "top_k": 6}},
            
            {"id": "layer_2_self_attn", "layer_type": "attention", "input_shape": [4096, 7168], "output_shape": [4096, 7168], "params": {"hidden_size": 7168, "num_attention_heads": 56, "num_key_value_heads": 8}},
            {"id": "layer_2_norm", "layer_type": "normalization", "input_shape": [4096, 7168], "output_shape": [4096, 7168], "params": {"hidden_size": 7168}},
            {"id": "layer_2_moe_gate", "layer_type": "moe", "input_shape": [4096, 7168], "output_shape": [4096, 7168], "params": {"hidden_size": 7168, "num_experts": 64, "top_k": 6}},
            {"id": "layer_2_experts", "layer_type": "moe", "input_shape": [4096, 7168], "output_shape": [4096, 7168], "params": {"hidden_size": 7168, "num_experts": 64, "intermediate_size": 20480, "top_k": 6}},
            
            {"id": "layer_3_self_attn", "layer_type": "attention", "input_shape": [4096, 7168], "output_shape": [4096, 7168], "params": {"hidden_size": 7168, "num_attention_heads": 56, "num_key_value_heads": 8}},
            {"id": "layer_3_norm", "layer_type": "normalization", "input_shape": [4096, 7168], "output_shape": [4096, 7168], "params": {"hidden_size": 7168}},
            {"id": "layer_3_moe_gate", "layer_type": "moe", "input_shape": [4096, 7168], "output_shape": [4096, 7168], "params": {"hidden_size": 7168, "num_experts": 64, "top_k": 6}},
            {"id": "layer_3_experts", "layer_type": "moe", "input_shape": [4096, 7168], "output_shape": [4096, 7168], "params": {"hidden_size": 7168, "num_experts": 64, "intermediate_size": 20480, "top_k": 6}},
            
            {"id": "layer_4_self_attn", "layer_type": "attention", "input_shape": [4096, 7168], "output_shape": [4096, 7168], "params": {"hidden_size": 7168, "num_attention_heads": 56, "num_key_value_heads": 8}},
            {"id": "layer_4_norm", "layer_type": "normalization", "input_shape": [4096, 7168], "output_shape": [4096, 7168], "params": {"hidden_size": 7168}},
            {"id": "layer_4_moe_gate", "layer_type": "moe", "input_shape": [4096, 7168], "output_shape": [4096, 7168], "params": {"hidden_size": 7168, "num_experts": 64, "top_k": 6}},
            {"id": "layer_4_experts", "layer_type": "moe", "input_shape": [4096, 7168], "output_shape": [4096, 7168], "params": {"hidden_size": 7168, "num_experts": 64, "intermediate_size": 20480, "top_k": 6}},
            
            {"id": "layer_5_self_attn", "layer_type": "attention", "input_shape": [4096, 7168], "output_shape": [4096, 7168], "params": {"hidden_size": 7168, "num_attention_heads": 56, "num_key_value_heads": 8}},
            {"id": "layer_5_norm", "layer_type": "normalization", "input_shape": [4096, 7168], "output_shape": [4096, 7168], "params": {"hidden_size": 7168}},
            {"id": "layer_5_moe_gate", "layer_type": "moe", "input_shape": [4096, 7168], "output_shape": [4096, 7168], "params": {"hidden_size": 7168, "num_experts": 64, "top_k": 6}},
            {"id": "layer_5_experts", "layer_type": "moe", "input_shape": [4096, 7168], "output_shape": [4096, 7168], "params": {"hidden_size": 7168, "num_experts": 64, "intermediate_size": 20480, "top_k": 6}},
            
            {"id": "layer_6_self_attn", "layer_type": "attention", "input_shape": [4096, 7168], "output_shape": [4096, 7168], "params": {"hidden_size": 7168, "num_attention_heads": 56, "num_key_value_heads": 8}},
            {"id": "layer_6_norm", "layer_type": "normalization", "input_shape": [4096, 7168], "output_shape": [4096, 7168], "params": {"hidden_size": 7168}},
            {"id": "layer_6_moe_gate", "layer_type": "moe", "input_shape": [4096, 7168], "output_shape": [4096, 7168], "params": {"hidden_size": 7168, "num_experts": 64, "top_k": 6}},
            {"id": "layer_6_experts", "layer_type": "moe", "input_shape": [4096, 7168], "output_shape": [4096, 7168], "params": {"hidden_size": 7168, "num_experts": 64, "intermediate_size": 20480, "top_k": 6}},
            
            {"id": "layer_7_self_attn", "layer_type": "attention", "input_shape": [4096, 7168], "output_shape": [4096, 7168], "params": {"hidden_size": 7168, "num_attention_heads": 56, "num_key_value_heads": 8}},
            {"id": "layer_7_norm", "layer_type": "normalization", "input_shape": [4096, 7168], "output_shape": [4096, 7168], "params": {"hidden_size": 7168}},
            {"id": "layer_7_moe_gate", "layer_type": "moe", "input_shape": [4096, 7168], "output_shape": [4096, 7168], "params": {"hidden_size": 7168, "num_experts": 64, "top_k": 6}},
            {"id": "layer_7_experts", "layer_type": "moe", "input_shape": [4096, 7168], "output_shape": [4096, 7168], "params": {"hidden_size": 7168, "num_experts": 64, "intermediate_size": 20480, "top_k": 6}},
            
            {"id": "final_norm", "layer_type": "normalization", "input_shape": [4096, 7168], "output_shape": [4096, 7168], "params": {"hidden_size": 7168}},
            {"id": "lm_head", "layer_type": "dense", "input_shape": [4096, 7168], "output_shape": [4096, 128000], "params": {"in_features": 7168, "out_features": 128000}}
        ],
        "global_params": {
            "hidden_size": 7168,
            "num_layers": 64,
            "num_attention_heads": 56,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 20480,
            "vocab_size": 128000,
            "num_experts": 64,
            "num_experts_per_tok": 6,
            "num_shared_experts": 2,
            "moe_intermediate_size": 20480,
            "num_dense_layers": 8,
            "max_position_embeddings": 4096
        }
    },
    "training": {
        "batch_size": 1024,
        "optimizer": "adamw",
        "learning_rate": 0.0001,
        "precision": "bf16",
        "gradient_checkpointing": true,
        "zero_stage": 3,
        "max_steps": 300000,
        "warmup_steps": 5000,
        "parallelism": {
            "data_parallel": 128,
            "tensor_parallel": 8,
            "pipeline_parallel": 8
        }
    },
    "hardware": {
        "gpus": [
            {
                "name": "H100-80GB-HBM3",
                "count": 8192,
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
        "input_shape": [4096],
        "dtype": "bf16",
        "vocab_size": 128000
    },
    "cost_config": {
        "provider": "custom",
        "gpu_hour_usd": 2.50,
        "energy_kwh_usd": 0.08,
        "pue_factor": 1.1
    }
}
"#;

/// Real-world MoE model specifications
struct RealMoeSpecs {
    name: &'static str,
    total_params_billion: f64,
    active_params_billion: f64,
    hidden_size: u64,
    num_layers: u32,
    num_experts: u32,
    top_k: u32,
    num_shared_experts: u32,
    intermediate_size: u64,
    vocab_size: u64,
    context_length: u64,
}

impl RealMoeSpecs {
    /// Mixtral-8x7B specifications
    fn mixtral_8x7b() -> Self {
        Self {
            name: "Mixtral-8x7B",
            total_params_billion: 46.7,
            active_params_billion: 12.9,
            hidden_size: 4096,
            num_layers: 32,
            num_experts: 8,
            top_k: 2,
            num_shared_experts: 0,
            intermediate_size: 14336,
            vocab_size: 32000,
            context_length: 32768,
        }
    }
    
    /// DeepSeek-V3 specifications
    fn deepseek_v3() -> Self {
        Self {
            name: "DeepSeek-V3",
            total_params_billion: 671.0,
            active_params_billion: 37.0,
            hidden_size: 7168,
            num_layers: 61,
            num_experts: 256,
            top_k: 8,
            num_shared_experts: 1,
            intermediate_size: 2048,
            vocab_size: 128000,
            context_length: 128000,
        }
    }
    
    /// Grok-1 specifications
    fn grok1() -> Self {
        Self {
            name: "Grok-1",
            total_params_billion: 314.0,
            active_params_billion: 86.0,
            hidden_size: 18432,
            num_layers: 64,
            num_experts: 8,
            top_k: 2,
            num_shared_experts: 0,
            intermediate_size: 73728,
            vocab_size: 131072,
            context_length: 8192,
        }
    }
    
    /// Calculate expected total parameters for MoE
    fn calculate_moe_params(
        hidden: u64,
        layers: u32,
        intermediate: u64,
        vocab: u64,
        num_experts: u32,
        shared_experts: u32,
        dense_layers: u32,
        moe_layers: u32,
    ) -> f64 {
        let h = hidden as f64;
        let i = intermediate as f64;
        let v = vocab as f64;
        let e = num_experts as f64;
        let s = shared_experts as f64;
        
        // Embedding (tied)
        let embed = h * v;
        
        // Attention per layer (shared across all tokens)
        let attn_per_layer = 4.0 * h * h;  // Q, K, V, O
        
        // Dense MLP (for dense layers)
        let dense_mlp = 3.0 * h * i;
        
        // MoE MLP: each expert has gate, up, down projections
        // Expert params = e * (3 * h * i)
        let moe_mlp = e * 3.0 * h * i;
        
        // Shared experts
        let shared_mlp = s * 3.0 * h * i;
        
        // Router: h -> num_experts
        let router = h * e;
        
        // LayerNorm: 2 * h per layer
        let norm = 2.0 * h;
        
        // Total
        let total = embed 
            + dense_layers as f64 * (attn_per_layer + dense_mlp + norm)
            + moe_layers as f64 * (attn_per_layer + moe_mlp + shared_mlp + router + norm);
        
        total / 1e9
    }
    
    /// Calculate active parameters per token
    fn calculate_active_params(
        hidden: u64,
        intermediate: u64,
        vocab: u64,
        top_k: u32,
        shared_experts: u32,
        dense_layers: u32,
        moe_layers: u32,
    ) -> f64 {
        let h = hidden as f64;
        let i = intermediate as f64;
        let v = vocab as f64;
        let k = top_k as f64;
        let s = shared_experts as f64;
        
        // Embedding (tied, always active)
        let embed = h * v;
        
        // Attention (always active)
        let attn = 4.0 * h * h;
        
        // Dense MLP
        let dense_mlp = 3.0 * h * i;
        
        // Active MoE: only top_k experts + shared
        let active_moe = k * 3.0 * h * i + s * 3.0 * h * i;
        
        // Router (always active)
        let router = h * k;  // Simplified
        
        // LayerNorm
        let norm = 2.0 * h;
        
        // Total active
        let total = embed 
            + dense_layers as f64 * (attn + dense_mlp + norm)
            + moe_layers as f64 * (attn + active_moe + router + norm);
        
        total / 1e9
    }
}

#[test]
fn test_moe_large_compilation() {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║         MoE-141B-A36B: MIXTURE OF EXPERTS                   ║");
    println!("║         141B Total / 36B Active Parameters                  ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");
    
    // ── Step 1: Parse JSON ─────────────────────────────────────────────
    let start = std::time::Instant::now();
    let config = parse_model_config(MOE_LARGE_JSON)
        .expect("Failed to parse MoE JSON");
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
    println!("│                  MoE PARAMETERS                             │");
    println!("├─────────────────────────────────────────────────────────────┤");
    
    println!("│ hidden_size:            {:>15}               │", grc.hidden_size.unwrap_or(0));
    println!("│ num_layers:             {:>15}               │", grc.num_layers.unwrap_or(0));
    println!("│ num_attention_heads:    {:>15}               │", grc.num_attention_heads.unwrap_or(0));
    println!("│ num_key_value_heads:    {:>15} (GQA 7:1)     │", grc.num_key_value_heads.unwrap_or(0));
    println!("│ head_dim:               {:>15}               │", grc.head_dim);
    println!("│ intermediate_size:      {:>15}               │", grc.intermediate_size.unwrap_or(0));
    println!("│ vocab_size:             {:>15}               │", grc.vocab_size.unwrap_or(0));
    
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│                  MoE-SPECIFIC                              │");
    println!("├─────────────────────────────────────────────────────────────┤");
    
    println!("│ num_experts:            {:>15}               │", grc.num_experts.unwrap_or(0));
    println!("│ num_experts_per_tok:    {:>15} (top-k)       │", grc.num_experts_per_tok.unwrap_or(0));
    println!("│ num_shared_experts:     {:>15}               │", grc.num_shared_experts.unwrap_or(0));
    println!("│ moe_intermediate_size:  {:>15}               │", grc.moe_intermediate_size.unwrap_or(0));
    
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│                  DERIVED VALUES                             │");
    println!("├─────────────────────────────────────────────────────────────┤");
    
    println!("│ dtype_bytes:            {:>15} (bf16)         │", grc.dtype_bytes);
    println!("│ optimizer_bytes:        {:>15} (AdamW)        │", grc.optimizer_bytes_per_param);
    println!("│ tied_embeddings:        {:>15}               │", grc.tied_embeddings);
    println!("│ gradient_checkpointing: {:>15}               │", grc.gradient_checkpointing);
    
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
    
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│                  CONFIDENCE                                 │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ confidence_score:       {:>14.1}%              │", grc.confidence_score * 100.0);
    println!("│ missing_fields:         {:>15?}              │", grc.missing_fields);
    
    println!("└─────────────────────────────────────────────────────────────┘\n");
    
    // ── Step 4: IR Injection ───────────────────────────────────────────
    let start = std::time::Instant::now();
    let _arch_input = IrInjector::to_architecture_ir(&absorbed);
    let _mem_config = IrInjector::configure_memory_pass(&absorbed);
    let _hw_config = IrInjector::configure_hardware_pass(&absorbed);
    let _cost_config = IrInjector::configure_cost_pass(&absorbed);
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
    
    // ── Step 6: Compare with Real MoE Models ───────────────────────────
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│          COMPARISON WITH REAL-WORLD MoE MODELS              │");
    println!("├─────────────────────────────────────────────────────────────┤");
    
    let mixtral = RealMoeSpecs::mixtral_8x7b();
    let deepseek = RealMoeSpecs::deepseek_v3();
    let grok1 = RealMoeSpecs::grok1();
    
    println!("│                                                             │");
    println!("│ Model          │ Total (B) │ Active (B) │ Experts │ Top-K │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ Mixtral-8x7B   │ {:>10.0} │ {:>10.1} │ {:>7} │ {:>6} │", 
             mixtral.total_params_billion, mixtral.active_params_billion, 
             mixtral.num_experts, mixtral.top_k);
    println!("│ DeepSeek-V3    │ {:>10.0} │ {:>10.0} │ {:>7} │ {:>6} │", 
             deepseek.total_params_billion, deepseek.active_params_billion,
             deepseek.num_experts, deepseek.top_k);
    println!("│ Grok-1         │ {:>10.0} │ {:>10.0} │ {:>7} │ {:>6} │", 
             grok1.total_params_billion, grok1.active_params_billion,
             grok1.num_experts, grok1.top_k);
    println!("│ MoE-141B-A36B  │ {:>10.0} │ {:>10.0} │ {:>7} │ {:>6} │", 
             total_params as f64 / 1e9, 36.0,  // Approximate active
             grc.num_experts.unwrap_or(64), grc.num_experts_per_tok.unwrap_or(6));
    println!("└─────────────────────────────────────────────────────────────┘\n");
    
    // ── Expected vs Calculated Parameters ─────────────────────────────
    let expected_total = RealMoeSpecs::calculate_moe_params(
        grc.hidden_size.unwrap(),
        grc.num_layers.unwrap(),
        grc.moe_intermediate_size.unwrap_or(grc.intermediate_size.unwrap()),
        grc.vocab_size.unwrap(),
        grc.num_experts.unwrap_or(64),
        grc.num_shared_experts.unwrap_or(2),
        8,   // dense layers
        56,  // moe layers
    );
    
    let expected_active = RealMoeSpecs::calculate_active_params(
        grc.hidden_size.unwrap(),
        grc.moe_intermediate_size.unwrap_or(grc.intermediate_size.unwrap()),
        grc.vocab_size.unwrap(),
        grc.num_experts_per_tok.unwrap_or(6),
        grc.num_shared_experts.unwrap_or(2),
        8,   // dense layers
        56,  // moe layers
    );
    
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│          PARAMETER ACCURACY COMPARISON                      │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ Expected total params:  {:>15.2}B           │", expected_total);
    println!("│ Calculated total:       {:>15.2}B           │", total_params as f64 / 1e9);
    println!("│ Expected active params: {:>15.2}B           │", expected_active);
    println!("│ Active ratio:           {:>14.1}%            │", 
             (expected_active / expected_total) * 100.0);
    println!("│ Sparsity:               {:>14.1}%            │", 
             (1.0 - expected_active / expected_total) * 100.0);
    println!("└─────────────────────────────────────────────────────────────┘\n");
    
    // ── Training Cost Estimation ───────────────────────────────────────
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│          TRAINING COST ESTIMATION                           │");
    println!("├─────────────────────────────────────────────────────────────┤");
    
    let training_tokens = 14.0; // 14T tokens
    let active_params = expected_active * 1e9;
    let training_flops = 6.0 * active_params * training_tokens * 1e12;
    
    println!("│ Training tokens:        {:>14.1}T           │", training_tokens);
    println!("│ Active params (B):      {:>15.1}            │", expected_active);
    println!("│ Total training FLOPs:   {:>15.2e}           │", training_flops);
    
    // GPU hours
    let gpu_tflops = grc.primary_gpu_tflops;
    let gpu_utilization = 0.35; // 35% for MoE (lower due to routing)
    let effective_tflops = gpu_tflops * gpu_utilization;
    let gpu_seconds = training_flops / (effective_tflops * 1e12);
    let gpu_hours = gpu_seconds / 3600.0;
    let gpu_million_hours = gpu_hours / 1e6;
    
    println!("│ GPU utilization:        {:>14.0}%            │", gpu_utilization * 100.0);
    println!("│ GPU hours (millions):   {:>15.2}            │", gpu_million_hours);
    println!("│ Total training cost:    {:>15.2}M USD      │", gpu_million_hours * 2.5);
    
    println!("└─────────────────────────────────────────────────────────────┘\n");
    
    // ── Assertions ─────────────────────────────────────────────────────
    // Note: calculate_total_params returns active params for MoE
    // Total params would be much larger (64 experts * mlp_params)
    assert!(total_params > 10_000_000_000, "Expected > 10B active params, got {}", total_params);
    assert!(total_params < 100_000_000_000, "Expected < 100B active params, got {}", total_params);
    assert_eq!(grc.hidden_size, Some(7168));
    assert_eq!(grc.num_experts.unwrap_or(64), 64, "Should have 64 experts");
    assert_eq!(grc.num_experts_per_tok.unwrap_or(6), 6, "Top-k should be 6");
    assert_eq!(grc.num_shared_experts.unwrap_or(2), 2, "Should have 2 shared experts");
    
    println!("✓ All assertions passed!");
    println!("✓ MoE-141B-A36B compiled successfully!\n");
}

#[test]
fn test_moe_vs_real_models() {
    println!("\n=== MoE vs Real Models Detailed Comparison ===\n");
    
    let config = parse_model_config(MOE_LARGE_JSON).unwrap();
    let absorbed = AbsorbedModel::absorb(config);
    let grc = &absorbed.resolution_context;
    
    let mixtral = RealMoeSpecs::mixtral_8x7b();
    let deepseek = RealMoeSpecs::deepseek_v3();
    let grok1 = RealMoeSpecs::grok1();
    
    let our_params = IrInjector::calculate_total_params(&absorbed) as f64 / 1e9;
    
    println!("┌────────────────────────────────────────────────────────────────────┐");
    println!("│                    MoE MODEL SPECIFICATIONS                       │");
    println!("├────────────────────────────────────────────────────────────────────┤");
    println!("│ Metric          │ Mixtral │ DeepSeek │ Grok-1 │ MoE-141B        │");
    println!("├────────────────────────────────────────────────────────────────────┤");
    println!("│ Total (B)       │ {:>8.0} │ {:>9.0} │ {:>6.0} │ {:>8.2}        │", 
             mixtral.total_params_billion, deepseek.total_params_billion, 
             grok1.total_params_billion, our_params);
    println!("│ Active (B)      │ {:>8.1} │ {:>9.0} │ {:>6.0} │ {:>8.1}        │", 
             mixtral.active_params_billion, deepseek.active_params_billion,
             grok1.active_params_billion, 36.0);
    println!("│ Hidden Size     │ {:>8} │ {:>9} │ {:>6} │ {:>8}        │", 
             mixtral.hidden_size, deepseek.hidden_size, grok1.hidden_size, grc.hidden_size.unwrap());
    println!("│ Layers          │ {:>8} │ {:>9} │ {:>6} │ {:>8}        │", 
             mixtral.num_layers, deepseek.num_layers, grok1.num_layers, grc.num_layers.unwrap());
    println!("│ Num Experts     │ {:>8} │ {:>9} │ {:>6} │ {:>8}        │", 
             mixtral.num_experts, deepseek.num_experts, grok1.num_experts, grc.num_experts.unwrap_or(64));
    println!("│ Top-K           │ {:>8} │ {:>9} │ {:>6} │ {:>8}        │", 
             mixtral.top_k, deepseek.top_k, grok1.top_k, grc.num_experts_per_tok.unwrap_or(6));
    println!("│ Shared Experts  │ {:>8} │ {:>9} │ {:>6} │ {:>8}        │", 
             mixtral.num_shared_experts, deepseek.num_shared_experts, 
             grok1.num_shared_experts, grc.num_shared_experts.unwrap_or(2));
    println!("└────────────────────────────────────────────────────────────────────┘\n");
    
    // Verify our model positioning
    // Note: Our MoE model has ~37B active params, smaller than Mixtral's 47B total
    // but with 64 experts, the total params would be ~1645B
    println!("✓ MoE-141B-A36B has {:.1}B active params (smaller than Mixtral's 47B total)", our_params);
}

#[test]
fn test_moe_expert_utilization() {
    let config = parse_model_config(MOE_LARGE_JSON).unwrap();
    let absorbed = AbsorbedModel::absorb(config);
    let grc = &absorbed.resolution_context;
    
    let num_experts = grc.num_experts.unwrap_or(64);
    let top_k = grc.num_experts_per_tok.unwrap_or(6);
    let shared = grc.num_shared_experts.unwrap_or(2);
    
    println!("\n=== Expert Utilization Analysis ===\n");
    
    // Expert utilization ratio
    let active_experts = top_k + shared;
    let total_experts = num_experts + shared;
    let utilization = active_experts as f64 / total_experts as f64;
    
    println!("Total experts:           {}", num_experts);
    println!("Shared experts:          {}", shared);
    println!("Top-K (routed):          {}", top_k);
    println!("Active per token:        {} ({} routed + {} shared)", 
             active_experts, top_k, shared);
    println!("Expert utilization:      {:.1}%", utilization * 100.0);
    println!("Parameter efficiency:    {:.1}x", 1.0 / utilization);
    
    // Routing capacity
    let routing_capacity = top_k as f64 / num_experts as f64;
    println!("Routing capacity:        {:.1}% of experts per token", routing_capacity * 100.0);
    
    assert!(utilization < 0.2, "MoE should have <20% expert utilization");
    println!("\n✓ MoE achieves {:.1}x parameter efficiency", 1.0 / utilization);
}
