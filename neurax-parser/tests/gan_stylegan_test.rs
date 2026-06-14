//! Test compilation of a GAN (Generative Adversarial Network) model
//! Validates complete absorption pipeline with GAN architecture
//! Based on StyleGAN-3 and BigGAN architectures

use neurax_parser::{parse_model_config, AbsorbedModel};
use neurax_ir::IrInjector;

/// StyleGAN-3 style GAN for high-resolution image generation
const STYLEGAN_3_JSON: &str = r#"
{
    "schema_version": "1.0",
    "model": {
        "name": "StyleGAN3-1024",
        "type": "gan",
        "layers": [
            {
                "id": "generator_input",
                "layer_type": "custom",
                "input_shape": [512],
                "output_shape": [4, 4, 512],
                "params": {
                    "z_dim": 512,
                    "w_dim": 512,
                    "initial_resolution": 4,
                    "initial_channels": 512
                }
            },
            {
                "id": "gen_block_4x4",
                "layer_type": "generator_block",
                "input_shape": [4, 4, 512],
                "output_shape": [8, 8, 512],
                "params": {
                    "in_channels": 512,
                    "out_channels": 512,
                    "resolution": 4,
                    "style_dim": 512
                }
            },
            {
                "id": "gen_block_8x8",
                "layer_type": "generator_block",
                "input_shape": [8, 8, 512],
                "output_shape": [16, 16, 512],
                "params": {
                    "in_channels": 512,
                    "out_channels": 512,
                    "resolution": 8,
                    "style_dim": 512
                }
            },
            {
                "id": "gen_block_16x16",
                "layer_type": "generator_block",
                "input_shape": [16, 16, 512],
                "output_shape": [32, 32, 512],
                "params": {
                    "in_channels": 512,
                    "out_channels": 512,
                    "resolution": 16,
                    "style_dim": 512
                }
            },
            {
                "id": "gen_block_32x32",
                "layer_type": "generator_block",
                "input_shape": [32, 32, 512],
                "output_shape": [64, 64, 256],
                "params": {
                    "in_channels": 512,
                    "out_channels": 256,
                    "resolution": 32,
                    "style_dim": 512
                }
            },
            {
                "id": "gen_block_64x64",
                "layer_type": "generator_block",
                "input_shape": [64, 64, 256],
                "output_shape": [128, 128, 128],
                "params": {
                    "in_channels": 256,
                    "out_channels": 128,
                    "resolution": 64,
                    "style_dim": 512
                }
            },
            {
                "id": "gen_block_128x128",
                "layer_type": "generator_block",
                "input_shape": [128, 128, 128],
                "output_shape": [256, 256, 64],
                "params": {
                    "in_channels": 128,
                    "out_channels": 64,
                    "resolution": 128,
                    "style_dim": 512
                }
            },
            {
                "id": "gen_block_256x256",
                "layer_type": "generator_block",
                "input_shape": [256, 256, 64],
                "output_shape": [512, 512, 32],
                "params": {
                    "in_channels": 64,
                    "out_channels": 32,
                    "resolution": 256,
                    "style_dim": 512
                }
            },
            {
                "id": "gen_block_512x512",
                "layer_type": "generator_block",
                "input_shape": [512, 512, 32],
                "output_shape": [1024, 1024, 16],
                "params": {
                    "in_channels": 32,
                    "out_channels": 16,
                    "resolution": 512,
                    "style_dim": 512
                }
            },
            {
                "id": "gen_output",
                "layer_type": "conv",
                "input_shape": [1024, 1024, 16],
                "output_shape": [1024, 1024, 3],
                "params": {
                    "in_channels": 16,
                    "out_channels": 3,
                    "kernel_size": 1
                }
            },
            {
                "id": "disc_input",
                "layer_type": "conv",
                "input_shape": [1024, 1024, 3],
                "output_shape": [1024, 1024, 16],
                "params": {
                    "in_channels": 3,
                    "out_channels": 16,
                    "kernel_size": 1
                }
            },
            {
                "id": "disc_block_512x512",
                "layer_type": "discriminator_block",
                "input_shape": [1024, 1024, 16],
                "output_shape": [512, 512, 32],
                "params": {
                    "in_channels": 16,
                    "out_channels": 32,
                    "resolution": 1024
                }
            },
            {
                "id": "disc_block_256x256",
                "layer_type": "discriminator_block",
                "input_shape": [512, 512, 32],
                "output_shape": [256, 256, 64],
                "params": {
                    "in_channels": 32,
                    "out_channels": 64,
                    "resolution": 512
                }
            },
            {
                "id": "disc_block_128x128",
                "layer_type": "discriminator_block",
                "input_shape": [256, 256, 64],
                "output_shape": [128, 128, 128],
                "params": {
                    "in_channels": 64,
                    "out_channels": 128,
                    "resolution": 256
                }
            },
            {
                "id": "disc_block_64x64",
                "layer_type": "discriminator_block",
                "input_shape": [128, 128, 128],
                "output_shape": [64, 64, 256],
                "params": {
                    "in_channels": 128,
                    "out_channels": 256,
                    "resolution": 128
                }
            },
            {
                "id": "disc_block_32x32",
                "layer_type": "discriminator_block",
                "input_shape": [64, 64, 256],
                "output_shape": [32, 32, 512],
                "params": {
                    "in_channels": 256,
                    "out_channels": 512,
                    "resolution": 64
                }
            },
            {
                "id": "disc_block_16x16",
                "layer_type": "discriminator_block",
                "input_shape": [32, 32, 512],
                "output_shape": [16, 16, 512],
                "params": {
                    "in_channels": 512,
                    "out_channels": 512,
                    "resolution": 32
                }
            },
            {
                "id": "disc_block_8x8",
                "layer_type": "discriminator_block",
                "input_shape": [16, 16, 512],
                "output_shape": [8, 8, 512],
                "params": {
                    "in_channels": 512,
                    "out_channels": 512,
                    "resolution": 16
                }
            },
            {
                "id": "disc_block_4x4",
                "layer_type": "discriminator_block",
                "input_shape": [8, 8, 512],
                "output_shape": [4, 4, 512],
                "params": {
                    "in_channels": 512,
                    "out_channels": 512,
                    "resolution": 8
                }
            },
            {
                "id": "disc_output",
                "layer_type": "dense",
                "input_shape": [8192],
                "output_shape": [1],
                "params": {
                    "in_features": 8192,
                    "out_features": 1
                }
            }
        ],
        "global_params": {
            "num_layers": 10,
            "image_height": 1024,
            "image_width": 1024,
            "image_channels": 3,
            "hidden_size": 512,
            "initial_channels": 512,
            "base_channels": 64
        }
    },
    "training": {
        "batch_size": 32,
        "optimizer": "adam",
        "learning_rate": 0.002,
        "precision": "fp16",
        "gradient_checkpointing": false,
        "zero_stage": 0,
        "max_steps": 200000,
        "warmup_steps": 1000,
        "parallelism": {
            "data_parallel": 8,
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
        "input_shape": [1024, 1024, 3],
        "dtype": "fp16",
        "image_height": 1024,
        "image_width": 1024,
        "image_channels": 3
    },
    "cost_config": {
        "provider": "aws",
        "gpu_hour_usd": 4.35,
        "energy_kwh_usd": 0.12,
        "pue_factor": 1.2
    }
}
"#;

/// Real-world GAN specifications for comparison
struct RealGanSpecs {
    /// Generator parameters (millions)
    gen_params_million: f64,
    /// Discriminator parameters (millions)
    disc_params_million: f64,
    /// Total parameters
    total_params_million: f64,
    /// Output resolution
    resolution: u64,
    /// Latent dimension
    z_dim: u64,
    /// Style dimension
    w_dim: u64,
}

impl RealGanSpecs {
    /// StyleGAN-3-T (translation equivariant) specifications
    fn stylegan3_t_1024() -> Self {
        Self {
            gen_params_million: 24.0,      // ~24M for generator
            disc_params_million: 18.0,     // ~18M for discriminator
            total_params_million: 42.0,    // ~42M total
            resolution: 1024,
            z_dim: 512,
            w_dim: 512,
        }
    }
    
    /// StyleGAN-2 1024 specifications
    fn stylegan2_1024() -> Self {
        Self {
            gen_params_million: 30.0,
            disc_params_million: 20.0,
            total_params_million: 50.0,
            resolution: 1024,
            z_dim: 512,
            w_dim: 512,
        }
    }
    
    /// Calculate expected generator params
    fn calculate_gen_params(style_dim: u64, max_channels: u64, num_blocks: u32) -> f64 {
        // Each block: conv + style modulation
        // Rough estimate: 2 * (in_ch * out_ch * kernel^2) per block
        let base = style_dim * max_channels * 2;  // Mapping network
        let blocks = num_blocks as u64 * max_channels * max_channels * 9;  // 3x3 convs
        (base + blocks) as f64 / 1e6
    }
}

#[test]
fn test_gan_stylegan_compilation() {
    println!("=== Compiling GAN (StyleGAN-3-1024) ===");
    println!("Model: StyleGAN3-1024");
    println!("Architecture: Generative Adversarial Network");
    println!("Resolution: 1024×1024");
    println!("GPUs: 8× A100-80GB");
    
    // ── Step 1: Parse JSON ─────────────────────────────────────────────
    let start = std::time::Instant::now();
    let config = parse_model_config(STYLEGAN_3_JSON)
        .expect("Failed to parse GAN JSON");
    let parse_time = start.elapsed();
    println!("✓ Parsed in {:?}", parse_time);
    
    // ── Step 2: Absorb into AbsorbedModel ───────────────────────────────
    let start = std::time::Instant::now();
    let absorbed = AbsorbedModel::absorb(config);
    let absorb_time = start.elapsed();
    println!("✓ Absorbed in {:?}", absorb_time);
    
    // ── Step 3: Validate GlobalResolutionContext ────────────────────────
    let grc = &absorbed.resolution_context;
    
    // ── GAN-Specific Parameters ────────────────────────────────────────
    println!("\n=== GAN Parameters ===");
    
    // Image dimensions
    assert_eq!(grc.image_height, Some(1024), "image_height should be 1024");
    assert_eq!(grc.image_width, Some(1024), "image_width should be 1024");
    assert_eq!(grc.image_channels, Some(3), "image_channels should be 3");
    println!("  Output resolution: {}x{}x{}", 
             grc.image_width.unwrap(), 
             grc.image_height.unwrap(),
             grc.image_channels.unwrap());
    
    // Hidden size (style dimension)
    assert_eq!(grc.hidden_size, Some(512), "hidden_size should be 512");
    println!("  Style dimension (w_dim): {}", grc.hidden_size.unwrap());
    
    // Number of layers
    assert_eq!(grc.num_layers, Some(10), "num_layers should be 10");
    println!("  Generator blocks: {}", grc.num_layers.unwrap());
    
    // Initial channels
    assert_eq!(grc.initial_channels, Some(512), "initial_channels should be 512");
    println!("  Initial channels: {}", grc.initial_channels.unwrap());
    
    // Base channels
    assert_eq!(grc.base_channels, Some(64), "base_channels should be 64");
    println!("  Base channels: {}", grc.base_channels.unwrap());
    
    // ── Derived Values ─────────────────────────────────────────────────
    println!("\n=== Derived Values ===");
    
    assert_eq!(grc.dtype_bytes, 2, "fp16 = 2 bytes");
    println!("  dtype_bytes: {}", grc.dtype_bytes);
    
    assert_eq!(grc.optimizer_bytes_per_param, 8, "Adam = 8 bytes");
    println!("  optimizer_bytes: {}", grc.optimizer_bytes_per_param);
    
    // ── Hardware & Parallelism ─────────────────────────────────────────
    println!("\n=== Hardware & Parallelism ===");
    
    assert_eq!(grc.num_gpus, 8, "8 GPUs");
    println!("  num_gpus: {}", grc.num_gpus);
    
    assert!((grc.primary_gpu_tflops - 312.0).abs() < 1.0, "A100 TFLOPs");
    println!("  GPU TFLOPs: {}", grc.primary_gpu_tflops);
    
    assert_eq!(grc.dp, 8, "Data parallel = 8");
    println!("  Parallelism: DP={}", grc.dp);
    
    // ── Symbol Table ───────────────────────────────────────────────────
    println!("\n=== Symbol Table ===");
    
    assert!(grc.symbol_table.contains_key("B"), "B in symbol table");
    println!("  B (batch): {:?}", grc.symbol_table.get("B"));
    println!("  H_img: {:?}", grc.symbol_table.get("H_img"));
    println!("  W_img: {:?}", grc.symbol_table.get("W_img"));
    println!("  C_img: {:?}", grc.symbol_table.get("C_img"));
    
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
    assert_eq!(arch_input.hidden_size, Some(512));
    assert_eq!(arch_input.num_layers, Some(10));
    
    // Validate Memory config
    assert_eq!(mem_config.dtype_bytes, 2);
    assert_eq!(mem_config.num_gpus, 8);
    
    // ── Step 5: Parameter Calculation ───────────────────────────────────
    let start = std::time::Instant::now();
    let total_params = IrInjector::calculate_total_params(&absorbed);
    let calc_time = start.elapsed();
    println!("✓ Parameters calculated in {:?}", calc_time);
    println!("  Total parameters: {:.2}M", total_params as f64 / 1e6);
    
    // ── Step 6: Compare with Real Model Specs ───────────────────────────
    println!("\n=== Comparison with Real GAN ===");
    
    let real_stylegan3 = RealGanSpecs::stylegan3_t_1024();
    let real_stylegan2 = RealGanSpecs::stylegan2_1024();
    
    println!("  StyleGAN-3-T 1024 params: {:.2}M", real_stylegan3.total_params_million);
    println!("  StyleGAN-2 1024 params: {:.2}M", real_stylegan2.total_params_million);
    println!("  Calculated params: {:.2}M", total_params as f64 / 1e6);
    
    // Verify resolution matches
    assert_eq!(grc.image_width.unwrap(), real_stylegan3.resolution, 
               "Resolution should match real specs");
    
    // Verify params are in reasonable range for GAN
    assert!(total_params > 1_000_000, "Expected > 1M params, got {}", total_params);
    assert!(total_params < 500_000_000, "Expected < 500M params, got {}", total_params);
    
    // ── Summary ─────────────────────────────────────────────────────────
    println!("\n=== GAN Compilation Summary ===");
    println!("Total time: {:?}", parse_time + absorb_time + inject_time + calc_time);
    println!("✓ All GAN fields absorbed correctly");
    println!("✓ All IRs configured");
    println!("✓ Parameter count: {:.2}M", total_params as f64 / 1e6);
    println!("✓ Resolution matches StyleGAN-3 specs");
}

#[test]
fn test_gan_specific_fields() {
    let config = parse_model_config(STYLEGAN_3_JSON).unwrap();
    let absorbed = AbsorbedModel::absorb(config);
    let grc = &absorbed.resolution_context;
    
    // Verify GAN-specific fields are absorbed
    assert_eq!(grc.image_height, Some(1024), "Image height");
    assert_eq!(grc.image_width, Some(1024), "Image width");
    assert_eq!(grc.image_channels, Some(3), "Image channels");
    assert_eq!(grc.initial_channels, Some(512), "Initial channels");
    assert_eq!(grc.base_channels, Some(64), "Base channels");
    
    // GAN should NOT have language model parameters
    assert_eq!(grc.vocab_size, None, "GAN has no vocabulary");
    assert_eq!(grc.num_attention_heads, None, "GAN has no attention heads");
    assert_eq!(grc.seq_len, None, "GAN has no sequence length");
    
    // GAN should NOT have SSM parameters
    assert_eq!(grc.ssm_state_size, None, "GAN has no SSM state");
    
    // GAN should NOT have GNN parameters
    assert_eq!(grc.node_features, None, "GAN has no node features");
    assert_eq!(grc.edge_features, None, "GAN has no edge features");
}

#[test]
fn test_gan_vs_other_architectures() {
    let config = parse_model_config(STYLEGAN_3_JSON).unwrap();
    let absorbed = AbsorbedModel::absorb(config);
    let grc = &absorbed.resolution_context;
    
    // GAN should have image-related parameters
    assert!(grc.image_height.is_some(), "GAN should have image height");
    assert!(grc.image_width.is_some(), "GAN should have image width");
    assert!(grc.image_channels.is_some(), "GAN should have image channels");
    
    // GAN should NOT have transformer parameters
    assert_eq!(grc.num_attention_heads, None, "GAN has no attention heads");
    assert_eq!(grc.num_key_value_heads, None, "GAN has no KV heads");
    assert_eq!(grc.vocab_size, None, "GAN has no vocabulary");
    
    // GAN should NOT have SSM parameters
    assert_eq!(grc.ssm_state_size, None, "GAN has no SSM state");
    assert_eq!(grc.ssm_expand, None, "GAN has no SSM expand");
    
    // GAN should NOT have GNN parameters
    assert_eq!(grc.node_features, None, "GAN has no node features");
    assert_eq!(grc.num_message_passing, None, "GAN has no message passing");
    
    // tied_embeddings should be false for GAN
    assert!(!grc.tied_embeddings, "GAN should not have tied embeddings");
}

#[test]
fn test_gan_symbol_table() {
    let config = parse_model_config(STYLEGAN_3_JSON).unwrap();
    let absorbed = AbsorbedModel::absorb(config);
    let grc = &absorbed.resolution_context;
    
    let sym = &grc.symbol_table;
    
    // Standard symbols
    assert!(sym.contains_key("B"), "B (batch)");
    assert!(sym.contains_key("H_img"), "H_img (image height)");
    assert!(sym.contains_key("W_img"), "W_img (image width)");
    assert!(sym.contains_key("C_img"), "C_img (image channels)");
    
    // Verify values
    assert_eq!(sym.get("B"), Some(&32u64), "Batch size");
    assert_eq!(sym.get("H_img"), Some(&1024u64), "Image height");
    assert_eq!(sym.get("W_img"), Some(&1024u64), "Image width");
    assert_eq!(sym.get("C_img"), Some(&3u64), "Image channels");
    
    // dtype_bytes for fp16
    assert_eq!(sym.get("dtype_bytes"), Some(&2u64), "fp16 = 2 bytes");
}

#[test]
fn test_gan_metrics_accuracy() {
    let config = parse_model_config(STYLEGAN_3_JSON).unwrap();
    let absorbed = AbsorbedModel::absorb(config);
    let grc = &absorbed.resolution_context;
    
    // Compare with real-world specifications
    let real = RealGanSpecs::stylegan3_t_1024();
    
    // Verify resolution matches
    assert_eq!(grc.image_width.unwrap(), real.resolution, 
               "Resolution should match real specs");
    assert_eq!(grc.image_height.unwrap(), real.resolution,
               "Resolution should match real specs");
    
    // Verify style dimension matches
    assert_eq!(grc.hidden_size.unwrap(), real.w_dim,
               "Style dim should match real specs");
    
    // Calculate params
    let calculated = IrInjector::calculate_total_params(&absorbed) as f64 / 1e6;
    
    // Verify params are in reasonable range
    assert!(calculated > 1.0 && calculated < 500.0,
            "GAN params should be in reasonable range (1-500M), got {:.2}M", calculated);
    
    println!("✓ GAN Metrics accuracy verified:");
    println!("  Resolution: {}x{} (matches StyleGAN-3)", 
             grc.image_width.unwrap(), grc.image_height.unwrap());
    println!("  Style dim: {} (matches StyleGAN-3)", grc.hidden_size.unwrap());
    println!("  Calculated params: {:.2}M", calculated);
    println!("  Real StyleGAN-3 params: ~{:.2}M", real.total_params_million);
}
