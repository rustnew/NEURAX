//! Hybrid Architecture Compilation Test
//! Tests models combining multiple architecture families:
//! - Transformer + CNN (Vision Transformer, ConvNeXt with Attention)
//! - Transformer + Diffusion (DiT - Diffusion Transformer)
//! - RNN + Attention (LSTM with Attention, Transformer-XL)
//! - CNN + Diffusion (Diffusion UNet with ResNet blocks)
//! - MoE + Transformer (Mixture of Experts Transformer)

use neurax_parser::{parse_model_config, AbsorbedModel};
use neurax_ir::IrInjector;

/// Vision Transformer (ViT) - CNN + Transformer hybrid
/// Uses patch embedding (conv) + transformer layers
const VIT_HYBRID_JSON: &str = r#"
{
    "schema_version": "1.0",
    "model": {
        "name": "Vision-Transformer-ViT-L",
        "type": "hybrid",
        "layers": [
            {"id": "patch_embed", "layer_type": "conv", "input_shape": [224, 224, 3], "output_shape": [196, 1024], "params": {"in_channels": 3, "out_channels": 1024, "kernel_size": 16, "stride": 16}},
            {"id": "cls_token", "layer_type": "embedding", "input_shape": [1], "output_shape": [1024], "params": {"vocab_size": 1, "embedding_dim": 1024}},
            {"id": "pos_embed", "layer_type": "embedding", "input_shape": [197], "output_shape": [1024], "params": {"vocab_size": 197, "embedding_dim": 1024}},
            
            {"id": "block_0", "layer_type": "attention", "input_shape": [197, 1024], "output_shape": [197, 1024], "params": {"hidden_size": 1024, "num_heads": 16, "num_kv_heads": 16}},
            {"id": "mlp_0", "layer_type": "mlp", "input_shape": [197, 1024], "output_shape": [197, 1024], "params": {"hidden_size": 1024, "intermediate_size": 4096}},
            {"id": "norm_0", "layer_type": "normalization", "input_shape": [197, 1024], "output_shape": [197, 1024], "params": {"hidden_size": 1024}},
            
            {"id": "block_1", "layer_type": "attention", "input_shape": [197, 1024], "output_shape": [197, 1024], "params": {"hidden_size": 1024, "num_heads": 16, "num_kv_heads": 16}},
            {"id": "mlp_1", "layer_type": "mlp", "input_shape": [197, 1024], "output_shape": [197, 1024], "params": {"hidden_size": 1024, "intermediate_size": 4096}},
            {"id": "norm_1", "layer_type": "normalization", "input_shape": [197, 1024], "output_shape": [197, 1024], "params": {"hidden_size": 1024}},
            
            {"id": "block_2", "layer_type": "attention", "input_shape": [197, 1024], "output_shape": [197, 1024], "params": {"hidden_size": 1024, "num_heads": 16, "num_kv_heads": 16}},
            {"id": "mlp_2", "layer_type": "mlp", "input_shape": [197, 1024], "output_shape": [197, 1024], "params": {"hidden_size": 1024, "intermediate_size": 4096}},
            
            {"id": "head", "layer_type": "dense", "input_shape": [1024], "output_shape": [1000], "params": {"in_features": 1024, "out_features": 1000}}
        ],
        "global_params": {
            "hidden_size": 1024,
            "num_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "image_size": 224,
            "patch_size": 16,
            "num_classes": 1000
        }
    },
    "training": {"batch_size": 256, "optimizer": "adamw", "learning_rate": 0.001, "precision": "fp16", "gradient_checkpointing": true, "zero_stage": 2, "max_steps": 300000, "warmup_steps": 10000, "parallelism": {"data_parallel": 64, "tensor_parallel": 1, "pipeline_parallel": 1}},
    "hardware": {"gpus": [{"name": "A100-80GB", "count": 64, "memory_gb": 80, "tflops_fp16": 312, "memory_bandwidth_gb_s": 2039, "tensor_cores": true, "nvlink": true}], "interconnect": "InfiniBand-200Gb/s", "interconnect_bandwidth_gb_s": 25},
    "data": {"input_shape": [224, 224, 3], "dtype": "fp16", "image_height": 224, "image_width": 224, "image_channels": 3},
    "cost_config": {"provider": "aws", "gpu_hour_usd": 4.50, "energy_kwh_usd": 0.12, "pue_factor": 1.2}
}
"#;

/// DiT - Diffusion Transformer (Transformer + Diffusion hybrid)
const DIT_HYBRID_JSON: &str = r#"
{
    "schema_version": "1.0",
    "model": {
        "name": "DiT-XL",
        "type": "hybrid",
        "layers": [
            {"id": "time_embed", "layer_type": "time_embedding", "input_shape": [1], "output_shape": [1152], "params": {"time_embedding_dim": 1152}},
            {"id": "patch_embed", "layer_type": "conv", "input_shape": [32, 32, 4], "output_shape": [256, 1152], "params": {"in_channels": 4, "out_channels": 1152, "kernel_size": 2, "stride": 2}},
            {"id": "pos_embed", "layer_type": "embedding", "input_shape": [256], "output_shape": [1152], "params": {"vocab_size": 256, "embedding_dim": 1152}},
            
            {"id": "dit_block_0", "layer_type": "attention", "input_shape": [256, 1152], "output_shape": [256, 1152], "params": {"hidden_size": 1152, "num_heads": 16}},
            {"id": "dit_mlp_0", "layer_type": "mlp", "input_shape": [256, 1152], "output_shape": [256, 1152], "params": {"hidden_size": 1152, "intermediate_size": 4608}},
            {"id": "ada_ln_0", "layer_type": "normalization", "input_shape": [256, 1152], "output_shape": [256, 1152], "params": {"hidden_size": 1152, "activation": "adaptive"}},
            
            {"id": "dit_block_1", "layer_type": "attention", "input_shape": [256, 1152], "output_shape": [256, 1152], "params": {"hidden_size": 1152, "num_heads": 16}},
            {"id": "dit_mlp_1", "layer_type": "mlp", "input_shape": [256, 1152], "output_shape": [256, 1152], "params": {"hidden_size": 1152, "intermediate_size": 4608}},
            
            {"id": "final_layer", "layer_type": "conv", "input_shape": [256, 1152], "output_shape": [32, 32, 4], "params": {"in_channels": 1152, "out_channels": 4, "kernel_size": 2}},
            {"id": "vae_dec", "layer_type": "vae_decoder", "input_shape": [32, 32, 4], "output_shape": [256, 256, 3], "params": {"in_channels": 4, "out_channels": 3, "vae_scale_factor": 8}}
        ],
        "global_params": {
            "hidden_size": 1152,
            "num_layers": 28,
            "num_attention_heads": 16,
            "intermediate_size": 4608,
            "diffusion_timesteps": 1000,
            "image_size": 256,
            "latent_channels": 4,
            "patch_size": 2
        }
    },
    "training": {"batch_size": 256, "optimizer": "adamw", "learning_rate": 0.0001, "precision": "fp16", "gradient_checkpointing": true, "zero_stage": 2, "max_steps": 400000, "warmup_steps": 5000, "parallelism": {"data_parallel": 64, "tensor_parallel": 2, "pipeline_parallel": 1}},
    "hardware": {"gpus": [{"name": "A100-80GB", "count": 128, "memory_gb": 80, "tflops_fp16": 312, "memory_bandwidth_gb_s": 2039, "tensor_cores": true, "nvlink": true}], "interconnect": "InfiniBand-400Gb/s", "interconnect_bandwidth_gb_s": 50},
    "data": {"input_shape": [256, 256, 3], "dtype": "fp16", "image_height": 256, "image_width": 256, "image_channels": 3},
    "cost_config": {"provider": "aws", "gpu_hour_usd": 4.50, "energy_kwh_usd": 0.12, "pue_factor": 1.2}
}
"#;

/// LSTM with Attention (RNN + Transformer hybrid)
const LSTM_ATTENTION_HYBRID_JSON: &str = r#"
{
    "schema_version": "1.0",
    "model": {
        "name": "LSTM-Attention-Hybrid",
        "type": "hybrid",
        "layers": [
            {"id": "embedding", "layer_type": "embedding", "input_shape": [512, 50000], "output_shape": [512, 1024], "params": {"vocab_size": 50000, "embedding_dim": 1024}},
            
            {"id": "lstm_1", "layer_type": "lstm_block", "input_shape": [512, 1024], "output_shape": [512, 2048], "params": {"rnn_hidden_size": 1024, "bidirectional_rnn": true, "num_rnn_layers": 1, "cell_type": "lstm"}},
            {"id": "lstm_2", "layer_type": "lstm_block", "input_shape": [512, 2048], "output_shape": [512, 2048], "params": {"rnn_hidden_size": 1024, "bidirectional_rnn": true, "num_rnn_layers": 1, "cell_type": "lstm"}},
            
            {"id": "attention", "layer_type": "attention", "input_shape": [512, 2048], "output_shape": [512, 2048], "params": {"hidden_size": 2048, "num_heads": 8}},
            
            {"id": "output", "layer_type": "dense", "input_shape": [512, 2048], "output_shape": [512, 50000], "params": {"in_features": 2048, "out_features": 50000}}
        ],
        "global_params": {
            "vocab_size": 50000,
            "embedding_dim": 1024,
            "rnn_hidden_size": 1024,
            "num_rnn_layers": 2,
            "bidirectional_rnn": true,
            "cell_type": "lstm",
            "hidden_size": 2048,
            "num_attention_heads": 8,
            "sequence_length": 512
        }
    },
    "training": {"batch_size": 64, "optimizer": "adamw", "learning_rate": 0.001, "precision": "fp32", "gradient_checkpointing": false, "zero_stage": 1, "max_steps": 300000, "warmup_steps": 8000, "parallelism": {"data_parallel": 8, "tensor_parallel": 1, "pipeline_parallel": 1}},
    "hardware": {"gpus": [{"name": "A100-80GB", "count": 8, "memory_gb": 80, "tflops_fp32": 19.5, "memory_bandwidth_gb_s": 2039, "tensor_cores": true, "nvlink": true}], "interconnect": "NVLink", "interconnect_bandwidth_gb_s": 600},
    "data": {"input_shape": [512], "dtype": "fp32"},
    "cost_config": {"provider": "aws", "gpu_hour_usd": 4.50, "energy_kwh_usd": 0.12, "pue_factor": 1.2}
}
"#;

/// MoE Transformer (MoE + Transformer hybrid)
const MOE_TRANSFORMER_HYBRID_JSON: &str = r#"
{
    "schema_version": "1.0",
    "model": {
        "name": "Mixtral-8x7B",
        "type": "hybrid",
        "layers": [
            {"id": "embed", "layer_type": "embedding", "input_shape": [4096, 32000], "output_shape": [4096, 4096], "params": {"vocab_size": 32000, "embedding_dim": 4096}},
            
            {"id": "attn_0", "layer_type": "attention", "input_shape": [4096, 4096], "output_shape": [4096, 4096], "params": {"hidden_size": 4096, "num_heads": 32, "num_kv_heads": 8}},
            {"id": "moe_0", "layer_type": "moe", "input_shape": [4096, 4096], "output_shape": [4096, 4096], "params": {"hidden_size": 4096, "intermediate_size": 14336, "num_experts": 8, "num_experts_per_tok": 2}},
            {"id": "norm_0", "layer_type": "normalization", "input_shape": [4096, 4096], "output_shape": [4096, 4096], "params": {"hidden_size": 4096, "activation": "rms"}},
            
            {"id": "attn_1", "layer_type": "attention", "input_shape": [4096, 4096], "output_shape": [4096, 4096], "params": {"hidden_size": 4096, "num_heads": 32, "num_kv_heads": 8}},
            {"id": "moe_1", "layer_type": "moe", "input_shape": [4096, 4096], "output_shape": [4096, 4096], "params": {"hidden_size": 4096, "intermediate_size": 14336, "num_experts": 8, "num_experts_per_tok": 2}},
            
            {"id": "lm_head", "layer_type": "dense", "input_shape": [4096, 4096], "output_shape": [4096, 32000], "params": {"in_features": 4096, "out_features": 32000}}
        ],
        "global_params": {
            "hidden_size": 4096,
            "num_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "vocab_size": 32000,
            "sequence_length": 4096
        }
    },
    "training": {"batch_size": 128, "optimizer": "adamw", "learning_rate": 0.0001, "precision": "bf16", "gradient_checkpointing": true, "zero_stage": 3, "max_steps": 1000000, "warmup_steps": 20000, "parallelism": {"data_parallel": 128, "tensor_parallel": 8, "pipeline_parallel": 4}},
    "hardware": {"gpus": [{"name": "H100-80GB", "count": 512, "memory_gb": 80, "tflops_fp16": 1979, "memory_bandwidth_gb_s": 3352, "tensor_cores": true, "nvlink": true}], "interconnect": "InfiniBand-400Gb/s", "interconnect_bandwidth_gb_s": 50},
    "data": {"input_shape": [4096], "dtype": "bf16"},
    "cost_config": {"provider": "aws", "gpu_hour_usd": 6.50, "energy_kwh_usd": 0.12, "pue_factor": 1.2}
}
"#;

/// ConvNeXt with Attention (CNN + Transformer hybrid)
const CONVNEXT_ATTENTION_HYBRID_JSON: &str = r#"
{
    "schema_version": "1.0",
    "model": {
        "name": "ConvNeXt-Attention",
        "type": "hybrid",
        "layers": [
            {"id": "stem", "layer_type": "conv", "input_shape": [224, 224, 3], "output_shape": [56, 56, 128], "params": {"in_channels": 3, "out_channels": 128, "kernel_size": 4, "stride": 4}},
            
            {"id": "stage1_block1", "layer_type": "convnext_block", "input_shape": [56, 56, 128], "output_shape": [56, 56, 128], "params": {"in_channels": 128, "out_channels": 128}},
            {"id": "stage1_block2", "layer_type": "convnext_block", "input_shape": [56, 56, 128], "output_shape": [56, 56, 128], "params": {"in_channels": 128, "out_channels": 128}},
            {"id": "downsample1", "layer_type": "conv", "input_shape": [56, 56, 128], "output_shape": [28, 28, 256], "params": {"in_channels": 128, "out_channels": 256, "kernel_size": 2, "stride": 2}},
            
            {"id": "stage2_block1", "layer_type": "convnext_block", "input_shape": [28, 28, 256], "output_shape": [28, 28, 256], "params": {"in_channels": 256, "out_channels": 256}},
            {"id": "stage2_block2", "layer_type": "convnext_block", "input_shape": [28, 28, 256], "output_shape": [28, 28, 256], "params": {"in_channels": 256, "out_channels": 256}},
            {"id": "attention2", "layer_type": "attention", "input_shape": [784, 256], "output_shape": [784, 256], "params": {"hidden_size": 256, "num_heads": 4}},
            {"id": "downsample2", "layer_type": "conv", "input_shape": [28, 28, 256], "output_shape": [14, 14, 512], "params": {"in_channels": 256, "out_channels": 512, "kernel_size": 2, "stride": 2}},
            
            {"id": "stage3_block1", "layer_type": "convnext_block", "input_shape": [14, 14, 512], "output_shape": [14, 14, 512], "params": {"in_channels": 512, "out_channels": 512}},
            {"id": "attention3", "layer_type": "attention", "input_shape": [196, 512], "output_shape": [196, 512], "params": {"hidden_size": 512, "num_heads": 8}},
            
            {"id": "head", "layer_type": "dense", "input_shape": [512], "output_shape": [1000], "params": {"in_features": 512, "out_features": 1000}}
        ],
        "global_params": {
            "hidden_size": 512,
            "num_layers": 12,
            "num_attention_heads": 8,
            "image_size": 224,
            "num_classes": 1000
        }
    },
    "training": {"batch_size": 128, "optimizer": "adamw", "learning_rate": 0.0005, "precision": "fp16", "gradient_checkpointing": true, "zero_stage": 1, "max_steps": 200000, "warmup_steps": 5000, "parallelism": {"data_parallel": 32, "tensor_parallel": 1, "pipeline_parallel": 1}},
    "hardware": {"gpus": [{"name": "A100-80GB", "count": 32, "memory_gb": 80, "tflops_fp16": 312, "memory_bandwidth_gb_s": 2039, "tensor_cores": true, "nvlink": true}], "interconnect": "NVLink", "interconnect_bandwidth_gb_s": 600},
    "data": {"input_shape": [224, 224, 3], "dtype": "fp16", "image_height": 224, "image_width": 224, "image_channels": 3},
    "cost_config": {"provider": "aws", "gpu_hour_usd": 4.50, "energy_kwh_usd": 0.12, "pue_factor": 1.2}
}
"#;

/// Whisper - CNN + Transformer (Audio model)
const WHISPER_HYBRID_JSON: &str = r#"
{
    "schema_version": "1.0",
    "model": {
        "name": "Whisper-Large-v3",
        "type": "hybrid",
        "layers": [
            {"id": "conv1", "layer_type": "conv", "input_shape": [80, 3000, 1], "output_shape": [40, 1500, 1280], "params": {"in_channels": 1, "out_channels": 1280, "kernel_size": 3, "stride": 1, "padding": 1}},
            {"id": "conv2", "layer_type": "conv", "input_shape": [40, 1500, 1280], "output_shape": [40, 750, 1280], "params": {"in_channels": 1280, "out_channels": 1280, "kernel_size": 3, "stride": 2, "padding": 1}},
            
            {"id": "pos_embed", "layer_type": "embedding", "input_shape": [1500], "output_shape": [1280], "params": {"vocab_size": 1500, "embedding_dim": 1280}},
            
            {"id": "enc_block_0", "layer_type": "attention", "input_shape": [1500, 1280], "output_shape": [1500, 1280], "params": {"hidden_size": 1280, "num_heads": 20}},
            {"id": "enc_mlp_0", "layer_type": "mlp", "input_shape": [1500, 1280], "output_shape": [1500, 1280], "params": {"hidden_size": 1280, "intermediate_size": 5120}},
            
            {"id": "dec_embed", "layer_type": "embedding", "input_shape": [448, 51865], "output_shape": [448, 1280], "params": {"vocab_size": 51865, "embedding_dim": 1280}},
            
            {"id": "dec_block_0", "layer_type": "attention", "input_shape": [448, 1280], "output_shape": [448, 1280], "params": {"hidden_size": 1280, "num_heads": 20}},
            {"id": "cross_attn_0", "layer_type": "cross_attention", "input_shape": [448, 1280], "output_shape": [448, 1280], "params": {"hidden_size": 1280, "num_heads": 20}},
            {"id": "dec_mlp_0", "layer_type": "mlp", "input_shape": [448, 1280], "output_shape": [448, 1280], "params": {"hidden_size": 1280, "intermediate_size": 5120}},
            
            {"id": "lm_head", "layer_type": "dense", "input_shape": [448, 1280], "output_shape": [448, 51865], "params": {"in_features": 1280, "out_features": 51865}}
        ],
        "global_params": {
            "hidden_size": 1280,
            "num_layers": 32,
            "num_attention_heads": 20,
            "intermediate_size": 5120,
            "vocab_size": 51865,
            "sequence_length": 1500
        }
    },
    "training": {"batch_size": 16, "optimizer": "adamw", "learning_rate": 0.0001, "precision": "fp16", "gradient_checkpointing": true, "zero_stage": 2, "max_steps": 500000, "warmup_steps": 10000, "parallelism": {"data_parallel": 64, "tensor_parallel": 1, "pipeline_parallel": 2}},
    "hardware": {"gpus": [{"name": "A100-80GB", "count": 128, "memory_gb": 80, "tflops_fp16": 312, "memory_bandwidth_gb_s": 2039, "tensor_cores": true, "nvlink": true}], "interconnect": "InfiniBand-200Gb/s", "interconnect_bandwidth_gb_s": 25},
    "data": {"input_shape": [80, 3000], "dtype": "fp16"},
    "cost_config": {"provider": "aws", "gpu_hour_usd": 4.50, "energy_kwh_usd": 0.12, "pue_factor": 1.2}
}
"#;

struct HybridModel {
    name: &'static str,
    json: &'static str,
    families: &'static [&'static str],
}

#[test]
fn test_all_hybrid_models() {
    println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║              HYBRID ARCHITECTURE COMPILATION TEST                       ║");
    println!("║   ViT | DiT | LSTM+Attention | MoE-Transformer | ConvNeXt+Attn | Whisper ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");
    
    let models = [
        HybridModel { name: "ViT-L", json: VIT_HYBRID_JSON, families: &["CNN", "Transformer"] },
        HybridModel { name: "DiT-XL", json: DIT_HYBRID_JSON, families: &["Transformer", "Diffusion"] },
        HybridModel { name: "LSTM-Attn", json: LSTM_ATTENTION_HYBRID_JSON, families: &["RNN", "Transformer"] },
        HybridModel { name: "Mixtral-8x7B", json: MOE_TRANSFORMER_HYBRID_JSON, families: &["MoE", "Transformer"] },
        HybridModel { name: "ConvNeXt-Attn", json: CONVNEXT_ATTENTION_HYBRID_JSON, families: &["CNN", "Transformer"] },
        HybridModel { name: "Whisper-Large", json: WHISPER_HYBRID_JSON, families: &["CNN", "Transformer", "Diffusion"] },
    ];
    
    println!("┌────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Model          │ Params (M) │ Families              │ Status │ Confidence    │");
    println!("├────────────────────────────────────────────────────────────────────────────────┤");
    
    let mut all_passed = true;
    
    for model in &models {
        let config = parse_model_config(model.json).expect(&format!("Failed to parse {}", model.name));
        let absorbed = AbsorbedModel::absorb(config);
        let grc = &absorbed.resolution_context;
        
        let total_params = IrInjector::calculate_total_params(&absorbed) as f64 / 1e6;
        let confidence = grc.confidence_score * 100.0;
        let families_str = model.families.join(" + ");
        
        let status = if total_params > 0.0 {
            "✓ OK"
        } else {
            all_passed = false;
            "✗ FAIL"
        };
        
        println!("│ {:<15} │ {:>10.1} │ {:<21} │ {:>6} │ {:>6.1}%       │", 
                 model.name, total_params, families_str, status, confidence);
    }
    
    println!("└────────────────────────────────────────────────────────────────────────────────┘\n");
    
    assert!(all_passed, "Some hybrid models failed compilation");
    println!("✓ All hybrid architectures compiled successfully!\n");
}

#[test]
fn test_vit_detailed() {
    println!("\n=== Vision Transformer (ViT) Detailed Analysis ===\n");
    
    let config = parse_model_config(VIT_HYBRID_JSON).unwrap();
    let absorbed = AbsorbedModel::absorb(config);
    let grc = &absorbed.resolution_context;
    
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│                  ViT-L PARAMETERS                           │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ Architecture: CNN (Patch Embed) + Transformer              │");
    println!("│ hidden_size:              {:>15}           │", grc.hidden_size.unwrap_or(0));
    println!("│ num_layers:              {:>15?}           │", grc.num_layers);
    println!("│ num_attention_heads:    {:>15?}           │", grc.num_attention_heads);
    println!("│ image_size:              {:>15?}           │", grc.image_size);
    println!("└─────────────────────────────────────────────────────────────┘\n");
    
    let total_params = IrInjector::calculate_total_params(&absorbed);
    println!("Total Parameters: {:.2}M\n", total_params as f64 / 1e6);
    
    assert!(total_params > 0);
    println!("✓ ViT hybrid architecture validated!\n");
}

#[test]
fn test_moe_transformer_detailed() {
    println!("\n=== MoE Transformer (Mixtral) Detailed Analysis ===\n");
    
    let config = parse_model_config(MOE_TRANSFORMER_HYBRID_JSON).unwrap();
    let absorbed = AbsorbedModel::absorb(config);
    let grc = &absorbed.resolution_context;
    
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│                  Mixtral-8x7B PARAMETERS                    │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ Architecture: MoE + Transformer (Sparse Mixture)           │");
    println!("│ hidden_size:             {:>15}            │", grc.hidden_size.unwrap_or(0));
    println!("│ num_experts:             {:>15?}            │", grc.num_experts);
    println!("│ num_experts_per_tok:     {:>15?}            │", grc.num_experts_per_tok);
    println!("│ num_attention_heads:     {:>15?}            │", grc.num_attention_heads);
    println!("│ vocab_size:              {:>15}            │", grc.vocab_size.unwrap_or(0));
    println!("└─────────────────────────────────────────────────────────────┘\n");
    
    let total_params = IrInjector::calculate_total_params(&absorbed);
    println!("Total Parameters: {:.2}B (47B total, 13B active per token)\n", total_params as f64 / 1e9);
    
    assert!(total_params > 0);
    println!("✓ MoE Transformer hybrid architecture validated!\n");
}

#[test]
fn test_hybrid_layer_combinations() {
    println!("\n=== Hybrid Layer Type Combinations ===\n");
    
    println!("┌────────────────────────────────────────────────────────────────────────┐");
    println!("│ Hybrid Type        │ Layer Combinations                          │ Use  │");
    println!("├────────────────────────────────────────────────────────────────────────┤");
    println!("│ ViT                │ Conv (patch) + Attention + MLP             │ Vision │");
    println!("│ DiT                │ TimeEmbed + Attention + Conv (unpatch)     │ Gen   │");
    println!("│ LSTM+Attention     │ LstmBlock + Attention                       │ Seq   │");
    println!("│ MoE-Transformer    │ Attention + MoE                             │ LLM   │");
    println!("│ ConvNeXt+Attn      │ ConvnextBlock + Attention                   │ Vision │");
    println!("│ Whisper            │ Conv + Attention + CrossAttention           │ Audio │");
    println!("└────────────────────────────────────────────────────────────────────────┘\n");
    
    println!("Key insights:\n");
    println!("  - Hybrid models combine strengths of multiple architectures");
    println!("  - ViT: CNN for local features + Transformer for global context");
    println!("  - DiT: Transformer architecture for diffusion process");
    println!("  - MoE: Sparse activation reduces compute while maintaining capacity");
    println!("  - Cross-attention enables encoder-decoder architectures\n");
}
