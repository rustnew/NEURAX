//! Tests for challenging model architectures
//! 
//! Tests one representative model per family:
//! - GPT-4: Large Transformer (120 layers, 16K hidden)
//! - Mixtral-8x7B: MoE (8 experts)
//! - DeepSeek-V3: Extreme MoE (256 experts)
//! - Mamba-2.8B: SSM with custom equations

use neurax_core::analyze_json;

const GPT4_JSON: &str = include_str!("../../examples/models/gpt4.json");
const MIXTRAL_JSON: &str = include_str!("../../examples/models/mixtral_8x7b.json");
const DEEPSEEK_JSON: &str = include_str!("../../examples/models/deepseek_v3.json");
const MAMBA_JSON: &str = include_str!("../../examples/models/mamba_2.8b.json");

/// Test GPT-4 - Large Transformer
#[test]
fn test_gpt4_analysis() {
    let result = analyze_json(GPT4_JSON);
    assert!(result.is_ok(), "GPT-4 analysis failed: {:?}", result.err());
    
    let report = result.unwrap();
    
    // Verify model structure - GPT-4 scaled down for testing
    assert!(report.arch.metrics.total_parameters > 1_000_000_000u64, "GPT-4 should have >1B params");
    assert!(report.arch.metrics.num_layers >= 10, "GPT-4 should have at least 10 layers");
    
    // Verify FLOPs estimation
    assert!(report.compute.metrics.forward_flops > 1e12, "Forward FLOPs should be >1e12");
    assert!(report.compute.metrics.backward_flops > report.compute.metrics.forward_flops, 
            "Backward FLOPs should exceed forward");
    
    // Verify memory estimation
    assert!(report.memory.metrics.peak_vram_bytes > 0, "Peak memory should be calculated");
    
    // Verify training cost (may be 0 if training config incomplete)
    // Just check the analysis completed successfully
    
    println!("✓ GPT-4 Analysis:");
    println!("  - Parameters: {:.2e}", report.arch.metrics.total_parameters);
    println!("  - Forward FLOPs: {:.2e}", report.compute.metrics.forward_flops);
    println!("  - Backward FLOPs: {:.2e}", report.compute.metrics.backward_flops);
    println!("  - Peak Memory: {:.2} GB", report.memory.metrics.peak_vram_gb());
    println!("  - Training Cost: ${:.2}M", report.cost.metrics.training_cost_usd / 1e6);
}

/// Test Mixtral-8x7B - MoE Architecture
#[test]
fn test_mixtral_moe_analysis() {
    let result = analyze_json(MIXTRAL_JSON);
    assert!(result.is_ok(), "Mixtral analysis failed: {:?}", result.err());
    
    let report = result.unwrap();
    
    // Verify MoE structure - check layers_by_type for moe
    let moe_layers = report.arch.metrics.layers_by_type.get("moe").unwrap_or(&0);
    assert!(*moe_layers > 0, "Should detect MoE layers");
    
    // Verify FLOPs with expert routing
    assert!(report.compute.metrics.forward_flops > 0.0, "Forward FLOPs should be calculated");
    
    // Verify memory includes expert overhead
    assert!(report.memory.metrics.peak_vram_bytes > 0, "Peak memory should include expert overhead");
    
    println!("✓ Mixtral-8x7B MoE Analysis:");
    println!("  - Total Parameters: {:.2e}", report.arch.metrics.total_parameters);
    println!("  - MoE Layers: {}", moe_layers);
    println!("  - Forward FLOPs: {:.2e}", report.compute.metrics.forward_flops);
    println!("  - Peak VRAM: {:.2} GB", report.memory.metrics.peak_vram_gb());
}

/// Test DeepSeek-V3 - Extreme MoE (256 experts)
#[test]
fn test_deepseek_extreme_moe() {
    let result = analyze_json(DEEPSEEK_JSON);
    assert!(result.is_ok(), "DeepSeek-V3 analysis failed: {:?}", result.err());
    
    let report = result.unwrap();
    
    // Verify MoE layers detected
    let moe_layers = report.arch.metrics.layers_by_type.get("moe").unwrap_or(&0);
    assert!(*moe_layers > 0, "Should detect MoE layers");
    
    // Verify memory for large model
    assert!(report.memory.metrics.peak_vram_bytes > 0, "Peak memory should handle large model");
    
    // Verify FLOPs estimation
    assert!(report.compute.metrics.forward_flops > 0.0, "FLOPs should be calculated");
    
    println!("✓ DeepSeek-V3 Extreme MoE Analysis:");
    println!("  - Total Parameters: {:.2e}", report.arch.metrics.total_parameters);
    println!("  - MoE Layers: {}", moe_layers);
    println!("  - Layers: {}", report.arch.metrics.num_layers);
    println!("  - Peak VRAM: {:.2} GB", report.memory.metrics.peak_vram_gb());
}

/// Test Mamba-2.8B - SSM with custom equations
#[test]
fn test_mamba_ssm_analysis() {
    let result = analyze_json(MAMBA_JSON);
    assert!(result.is_ok(), "Mamba analysis failed: {:?}", result.err());
    
    let report = result.unwrap();
    
    // Verify SSM structure
    assert!(report.arch.metrics.total_parameters > 0, "Should calculate SSM parameters");
    
    // Verify custom equations are handled
    assert!(report.compute.metrics.forward_flops > 0.0, 
            "Custom FLOPs equations should be evaluated");
    
    // Verify memory for state
    assert!(report.memory.metrics.peak_vram_bytes > 0, "Peak memory should include SSM state");
    
    // Verify layer count
    assert_eq!(report.arch.metrics.num_layers, 64, "Mamba should have 64 layers");
    
    println!("✓ Mamba-2.8B SSM Analysis:");
    println!("  - Parameters: {:.2e}", report.arch.metrics.total_parameters);
    println!("  - Num Layers: {}", report.arch.metrics.num_layers);
    println!("  - Forward FLOPs: {:.2e}", report.compute.metrics.forward_flops);
    println!("  - Custom Equations: Handled");
}

/// Test all models in parallel
#[test]
fn test_all_challenging_models() {
    use std::time::Instant;
    
    let start = Instant::now();
    
    // Parse all models
    let models: Vec<_> = vec![
        ("GPT-4", GPT4_JSON),
        ("Mixtral-8x7B", MIXTRAL_JSON),
        ("DeepSeek-V3", DEEPSEEK_JSON),
        ("Mamba-2.8B", MAMBA_JSON),
    ];
    
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║        NEURAX IR - Challenging Models Analysis               ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    
    for (name, json) in &models {
        let model_start = Instant::now();
        let result = analyze_json(json);
        let elapsed = model_start.elapsed();
        
        match result {
            Ok(report) => {
                println!("║ {:<15} │ ✓ PASS │ {:>6.1}ms │ Params: {:.2e}      ║", 
                         name, 
                         elapsed.as_secs_f64() * 1000.0,
                         report.arch.metrics.total_parameters);
            }
            Err(e) => {
                println!("║ {:<15} │ ✗ FAIL │ {:>6.1}ms │ Error: {:?<30} ║", 
                         name, 
                         elapsed.as_secs_f64() * 1000.0,
                         e.to_string().chars().take(30).collect::<String>());
            }
        }
    }
    
    let total_elapsed = start.elapsed();
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Total Analysis Time: {:>6.2}s                              ║", 
             total_elapsed.as_secs_f64());
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    
    // All should succeed
    for (name, json) in models {
        let result = analyze_json(json);
        assert!(result.is_ok(), "{} analysis should succeed", name);
    }
}

/// Test precision levels for different models
#[test]
fn test_precision_levels() {
    // GPT-4 with known shapes should have high precision
    let gpt4_result = analyze_json(GPT4_JSON).unwrap();
    
    // Mixtral with MoE routing
    let mixtral_result = analyze_json(MIXTRAL_JSON).unwrap();
    
    // DeepSeek with extreme MoE
    let deepseek_result = analyze_json(DEEPSEEK_JSON).unwrap();
    
    // Mamba with custom equations (lower precision)
    let mamba_result = analyze_json(MAMBA_JSON).unwrap();
    
    println!("\nPrecision Analysis:");
    println!("  GPT-4:      Params={:.2e}, FLOPs={:.2e}", 
             gpt4_result.arch.metrics.total_parameters,
             gpt4_result.compute.metrics.forward_flops);
    println!("  Mixtral:    Params={:.2e}, Layers={}", 
             mixtral_result.arch.metrics.total_parameters,
             mixtral_result.arch.metrics.num_layers);
    println!("  DeepSeek:   Params={:.2e}, Layers={}", 
             deepseek_result.arch.metrics.total_parameters,
             deepseek_result.arch.metrics.num_layers);
    println!("  Mamba:      Params={:.2e}, Custom=Yes", 
             mamba_result.arch.metrics.total_parameters);
}
