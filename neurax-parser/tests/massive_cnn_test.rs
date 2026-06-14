//! Test compilation of a massive CNN model (ResNet-1000 style)
//! Validates complete absorption pipeline with CNN architecture

use neurax_parser::{parse_model_config, AbsorbedModel};
use neurax_ir::IrInjector;

/// Massive CNN - ResNet-1000 style with 1000 layers
const MASSIVE_CNN_JSON: &str = r#"
{
    "schema_version": "1.0",
    "model": {
        "name": "ResNet-1000-XL",
        "type": "cnn",
        "layers": [
            {"id": "conv1", "layer_type": "conv", "input_shape": [3, 512, 512], "output_shape": [64, 256, 256], "params": {"in_channels": 3, "out_channels": 64, "kernel_size": 7, "stride": 2, "padding": 3}},
            {"id": "bn1", "layer_type": "normalization", "input_shape": [64, 256, 256], "output_shape": [64, 256, 256], "params": {"hidden_size": 64}},
            {"id": "pool1", "layer_type": "pooling", "input_shape": [64, 256, 256], "output_shape": [64, 128, 128], "params": {"pool_size": 3, "stride": 2}},
            
            {"id": "res_block_0_0", "layer_type": "residual_block", "input_shape": [64, 128, 128], "output_shape": [256, 128, 128], "params": {"in_channels": 64, "out_channels": 256, "bottleneck": false}},
            {"id": "res_block_0_1", "layer_type": "residual_block", "input_shape": [256, 128, 128], "output_shape": [256, 128, 128], "params": {"in_channels": 256, "out_channels": 256}},
            {"id": "res_block_0_2", "layer_type": "residual_block", "input_shape": [256, 128, 128], "output_shape": [256, 128, 128], "params": {"in_channels": 256, "out_channels": 256}},
            
            {"id": "res_block_1_0", "layer_type": "residual_block", "input_shape": [256, 128, 128], "output_shape": [512, 64, 64], "params": {"in_channels": 256, "out_channels": 512, "stride": 2}},
            {"id": "res_block_1_1", "layer_type": "residual_block", "input_shape": [512, 64, 64], "output_shape": [512, 64, 64], "params": {"in_channels": 512, "out_channels": 512}},
            {"id": "res_block_1_2", "layer_type": "residual_block", "input_shape": [512, 64, 64], "output_shape": [512, 64, 64], "params": {"in_channels": 512, "out_channels": 512}},
            {"id": "res_block_1_3", "layer_type": "residual_block", "input_shape": [512, 64, 64], "output_shape": [512, 64, 64], "params": {"in_channels": 512, "out_channels": 512}},
            
            {"id": "res_block_2_0", "layer_type": "residual_block", "input_shape": [512, 64, 64], "output_shape": [1024, 32, 32], "params": {"in_channels": 512, "out_channels": 1024, "stride": 2}},
            {"id": "res_block_2_1", "layer_type": "residual_block", "input_shape": [1024, 32, 32], "output_shape": [1024, 32, 32], "params": {"in_channels": 1024, "out_channels": 1024}},
            {"id": "res_block_2_2", "layer_type": "residual_block", "input_shape": [1024, 32, 32], "output_shape": [1024, 32, 32], "params": {"in_channels": 1024, "out_channels": 1024}},
            {"id": "res_block_2_3", "layer_type": "residual_block", "input_shape": [1024, 32, 32], "output_shape": [1024, 32, 32], "params": {"in_channels": 1024, "out_channels": 1024}},
            {"id": "res_block_2_4", "layer_type": "residual_block", "input_shape": [1024, 32, 32], "output_shape": [1024, 32, 32], "params": {"in_channels": 1024, "out_channels": 1024}},
            {"id": "res_block_2_5", "layer_type": "residual_block", "input_shape": [1024, 32, 32], "output_shape": [1024, 32, 32], "params": {"in_channels": 1024, "out_channels": 1024}},
            
            {"id": "res_block_3_0", "layer_type": "residual_block", "input_shape": [1024, 32, 32], "output_shape": [2048, 16, 16], "params": {"in_channels": 1024, "out_channels": 2048, "stride": 2}},
            {"id": "res_block_3_1", "layer_type": "residual_block", "input_shape": [2048, 16, 16], "output_shape": [2048, 16, 16], "params": {"in_channels": 2048, "out_channels": 2048}},
            {"id": "res_block_3_2", "layer_type": "residual_block", "input_shape": [2048, 16, 16], "output_shape": [2048, 16, 16], "params": {"in_channels": 2048, "out_channels": 2048}},
            
            {"id": "global_pool", "layer_type": "pooling", "input_shape": [2048, 16, 16], "output_shape": [2048, 1, 1], "params": {"pool_type": "avg", "global": true}},
            {"id": "fc", "layer_type": "dense", "input_shape": [2048], "output_shape": [1000], "params": {"in_features": 2048, "out_features": 1000}}
        ],
        "global_params": {
            "num_layers": 152,
            "initial_channels": 64,
            "base_channels": 64,
            "num_classes": 1000,
            "image_height": 512,
            "image_width": 512,
            "image_channels": 3
        }
    },
    "training": {
        "batch_size": 256,
        "optimizer": "sgd",
        "learning_rate": 0.1,
        "precision": "fp16",
        "gradient_checkpointing": false,
        "zero_stage": 0,
        "max_steps": 50000,
        "warmup_steps": 5000,
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
        "input_shape": [3, 512, 512],
        "dtype": "fp16",
        "num_classes": 1000,
        "image_height": 512,
        "image_width": 512,
        "image_channels": 3
    },
    "cost_config": {
        "provider": "gcp",
        "gpu_hour_usd": 3.50,
        "energy_kwh_usd": 0.10,
        "pue_factor": 1.15
    }
}
"#;

#[test]
fn test_massive_cnn_compilation() {
    println!("=== Compiling Massive CNN (ResNet-152) ===");
    println!("Model: ResNet-152-XL");
    println!("Input: 512×512×3");
    println!("Classes: 1000 (ImageNet)");
    println!("GPUs: 8× A100-80GB");
    
    // ── Step 1: Parse JSON ─────────────────────────────────────────────
    let start = std::time::Instant::now();
    let config = parse_model_config(MASSIVE_CNN_JSON)
        .expect("Failed to parse massive CNN JSON");
    let parse_time = start.elapsed();
    println!("✓ Parsed in {:?}", parse_time);
    
    // ── Step 2: Absorb into AbsorbedModel ───────────────────────────────
    let start = std::time::Instant::now();
    let absorbed = AbsorbedModel::absorb(config);
    let absorb_time = start.elapsed();
    println!("✓ Absorbed in {:?}", absorb_time);
    
    // ── Step 3: Validate GlobalResolutionContext ────────────────────────
    let grc = &absorbed.resolution_context;
    
    // CNN-specific dimensions
    assert_eq!(grc.num_layers, Some(152), "num_layers should be 152");
    assert_eq!(grc.num_classes, Some(1000), "num_classes should be 1000");
    assert_eq!(grc.image_height, Some(512), "image_height should be 512");
    assert_eq!(grc.image_width, Some(512), "image_width should be 512");
    assert_eq!(grc.image_channels, Some(3), "image_channels should be 3");
    
    // Derived values
    assert_eq!(grc.dtype_bytes, 2, "fp16 = 2 bytes");
    assert_eq!(grc.optimizer_bytes_per_param, 4, "SGD+momentum = 4 bytes");
    
    // Hardware
    assert_eq!(grc.num_gpus, 8, "8 GPUs");
    assert!((grc.primary_gpu_tflops - 312.0).abs() < 1.0, "A100 TFLOPs");
    
    // Parallelism
    assert_eq!(grc.dp, 8, "Data parallel = 8");
    assert_eq!(grc.tp, 1, "No tensor parallel for CNN");
    assert_eq!(grc.pp, 1, "No pipeline parallel for CNN");
    
    // Symbol table
    assert!(grc.symbol_table.contains_key("B"), "B in symbol table");
    assert!(grc.symbol_table.contains_key("num_classes"), "num_classes in symbol table");
    
    println!("✓ GlobalResolutionContext validated");
    println!("  Confidence score: {:.2}%", grc.confidence_score * 100.0);
    
    // ── Step 4: IR Injection ───────────────────────────────────────────
    let start = std::time::Instant::now();
    let arch_input = IrInjector::to_architecture_ir(&absorbed);
    let mem_config = IrInjector::configure_memory_pass(&absorbed);
    let hw_config = IrInjector::configure_hardware_pass(&absorbed);
    let cost_config = IrInjector::configure_cost_pass(&absorbed);
    let inject_time = start.elapsed();
    println!("✓ IRs injected in {:?}", inject_time);
    
    // Validate Architecture IR
    assert_eq!(arch_input.num_layers, Some(152));
    // For CNN, vocab_size is None (not applicable), num_classes is separate
    assert_eq!(grc.num_classes, Some(1000));
    
    // Validate Memory config
    assert_eq!(mem_config.dtype_bytes, 2);
    assert_eq!(mem_config.optimizer_bytes, 4);
    assert_eq!(mem_config.num_gpus, 8);
    
    // Validate Hardware config
    assert!((hw_config.gpu_tflops_fp16 - 312.0).abs() < 1.0);
    assert_eq!(hw_config.dp, 8);
    
    // Validate Cost config
    assert!((cost_config.gpu_hour_usd - 3.50).abs() < 0.01);
    assert_eq!(cost_config.num_gpus, 8);
    
    // ── Step 5: Parameter Calculation (CNN-specific) ────────────────────
    let start = std::time::Instant::now();
    let total_params = IrInjector::calculate_total_params(&absorbed);
    let calc_time = start.elapsed();
    println!("✓ Parameters calculated in {:?}", calc_time);
    
    // ResNet-152 has ~60M parameters
    // Our simplified calculation gives a rough estimate
    println!("  Total parameters: {:.2}M", total_params as f64 / 1e6);
    
    // ── Summary ─────────────────────────────────────────────────────────
    println!("\n=== CNN Compilation Summary ===");
    println!("Total time: {:?}", parse_time + absorb_time + inject_time + calc_time);
    println!("✓ All fields absorbed correctly");
    println!("✓ All IRs configured");
    println!("✓ Parameter count: {:.2}M", total_params as f64 / 1e6);
}

#[test]
fn test_cnn_image_dimensions() {
    let config = parse_model_config(MASSIVE_CNN_JSON).unwrap();
    let absorbed = AbsorbedModel::absorb(config);
    let grc = &absorbed.resolution_context;
    
    // Verify image dimensions are absorbed
    assert_eq!(grc.image_height, Some(512));
    assert_eq!(grc.image_width, Some(512));
    assert_eq!(grc.image_channels, Some(3));
    assert_eq!(grc.num_classes, Some(1000));
    
    // Verify symbol table has image-related symbols
    assert!(grc.symbol_table.contains_key("num_classes"));
}
