//! Tests de validation VRAM selon profile.md Partie III
//!
//! Vérifie que les estimations VRAM NEURAX sont cohérentes

use neurax_core::analyze_json;

/// Test de cohérence VRAM: params + activations + gradients + optimizer <= peak_vram
#[test]
fn test_vram_components_sum() {
    let json = include_str!("../../../../Neurax-IR/models/gpt2_medium.json");
    let result = analyze_json(json).expect("Analysis should succeed");
    
    let peak_vram = result.memory.metrics.peak_vram_bytes;
    let params = result.memory.metrics.parameter_memory_bytes;
    let activations = result.memory.metrics.activation_memory_bytes;
    let gradients = result.memory.metrics.gradient_memory_bytes;
    let optimizer = result.memory.metrics.optimizer_state_bytes;
    
    // La somme des composants doit être <= peak_vram (avec marge pour fragmentation)
    let sum_components = params + activations + gradients + optimizer;
    let fragmentation = result.memory.metrics.fragmentation_estimate;
    
    // Peak VRAM = sum × (1 + fragmentation)
    let expected_peak = (sum_components as f64 * (1.0 + fragmentation)) as u64;
    
    // Tolérance ±20% (estimation fragmentation approximative)
    let error = ((peak_vram as i64 - expected_peak as i64).abs() as f64) / expected_peak as f64;
    
    assert!(
        error < 0.20,
        "VRAM components: peak={}, expected={}, error={:.1}%",
        peak_vram, expected_peak, error * 100.0
    );
    
    println!("✓ VRAM components sum: peak={} bytes, components={} bytes", peak_vram, sum_components);
}

/// Test de cohérence mémoire paramètres vs architecture
/// NOTE: La précision peut varier (fp32=4 bytes, fp16/bf16=2 bytes)
#[test]
fn test_param_memory_consistency() {
    let json = include_str!("../../../../Neurax-IR/models/gpt2_medium.json");
    let result = analyze_json(json).expect("Analysis should succeed");
    
    let param_memory = result.memory.metrics.parameter_memory_bytes;
    let total_params = result.arch.metrics.total_parameters;
    
    // La précision peut être fp32 (4 bytes) ou fp16/bf16 (2 bytes)
    // On vérifie que le ratio est cohérent (2-4 bytes par param)
    let bytes_per_param = param_memory as f64 / total_params as f64;
    
    assert!(
        bytes_per_param >= 2.0 && bytes_per_param <= 4.0,
        "Param memory: {} bytes for {} params = {:.1} bytes/param (expected 2-4)",
        param_memory, total_params, bytes_per_param
    );
    
    println!("✓ Param memory consistency: {} bytes for {} params ({:.1} bytes/param)", param_memory, total_params, bytes_per_param);
}

/// Test de cohérence gradients = params (pour training)
#[test]
fn test_gradient_memory_equals_params() {
    let json = include_str!("../../../../Neurax-IR/models/gpt2_medium.json");
    let result = analyze_json(json).expect("Analysis should succeed");
    
    let gradient_memory = result.memory.metrics.gradient_memory_bytes;
    let param_memory = result.memory.metrics.parameter_memory_bytes;
    
    // En training, gradients = params (1 gradient par param)
    // Tolérance ±5%
    let error = ((gradient_memory as f64 - param_memory as f64).abs()) / param_memory as f64;
    
    assert!(
        error < 0.05,
        "Gradient memory: actual={}, expected={} (should equal params), error={:.1}%",
        gradient_memory, param_memory, error * 100.0
    );
    
    println!("✓ Gradient memory = param memory: {} bytes", gradient_memory);
}

/// Test de cohérence optimizer state (AdamW = 2× params)
#[test]
fn test_optimizer_state_adamw() {
    let json = include_str!("../../../../Neurax-IR/models/gpt2_medium.json");
    let result = analyze_json(json).expect("Analysis should succeed");
    
    let optimizer_memory = result.memory.metrics.optimizer_state_bytes;
    let param_memory = result.memory.metrics.parameter_memory_bytes;
    
    // AdamW stocke momentum + variance = 2× params
    let expected_optimizer = param_memory * 2;
    
    // Tolérance ±10%
    let error = ((optimizer_memory as i64 - expected_optimizer as i64).abs() as f64) / expected_optimizer as f64;
    
    assert!(
        error < 0.10,
        "Optimizer state: actual={}, expected={} (2× params for AdamW), error={:.1}%",
        optimizer_memory, expected_optimizer, error * 100.0
    );
    
    println!("✓ Optimizer state (AdamW): {} bytes = 2× params", optimizer_memory);
}

/// Test que max_batch_size_fit est cohérent avec VRAM GPU
#[test]
fn test_max_batch_size_fits_vram() {
    let json = include_str!("../../../../Neurax-IR/models/gpt2_medium.json");
    let result = analyze_json(json).expect("Analysis should succeed");
    
    let max_batch = result.memory.metrics.max_batch_size_fit;
    let gpu_memory_gb = result.report.metrics.gpu_memory_gb;
    let peak_vram = result.memory.metrics.peak_vram_bytes;
    let gpu_memory_bytes = (gpu_memory_gb * 1e9) as u64;
    
    // max_batch_size_fit doit être > 0 si peak_vram < gpu_memory
    if peak_vram < gpu_memory_bytes {
        assert!(
            max_batch > 0,
            "max_batch_size_fit should be > 0 when peak_vram ({}) < gpu_memory ({})",
            peak_vram, gpu_memory_bytes
        );
    }
    
    println!("✓ Max batch size fit: {} (peak_vram={}, gpu_memory={:.1}GB)", max_batch, peak_vram, gpu_memory_gb);
}

/// Test de non-régression: VRAM ne doit pas être 0 pour un modèle valide
#[test]
fn test_vram_nonzero() {
    let json = include_str!("../../../../Neurax-IR/models/gpt2_small.json");
    let result = analyze_json(json).expect("Analysis should succeed");
    
    assert!(
        result.memory.metrics.peak_vram_bytes > 0,
        "Peak VRAM should be > 0 for valid model"
    );
    assert!(
        result.memory.metrics.parameter_memory_bytes > 0,
        "Parameter memory should be > 0 for valid model"
    );
    
    println!("✓ VRAM non-zero: peak={} bytes", result.memory.metrics.peak_vram_bytes);
}

/// Test de fragmentation raisonnable (5-20%)
#[test]
fn test_fragmentation_reasonable() {
    let json = include_str!("../../../../Neurax-IR/models/gpt2_medium.json");
    let result = analyze_json(json).expect("Analysis should succeed");
    
    let frag = result.memory.metrics.fragmentation_estimate;
    
    // Fragmentation typique: 5-20%
    assert!(
        frag >= 0.05 && frag <= 0.20,
        "Fragmentation should be 5-20%, got {:.1}%",
        frag * 100.0
    );
    
    println!("✓ Fragmentation reasonable: {:.1}%", frag * 100.0);
}

/// Test de cohérence activation memory vs FLOPs
/// NOTE: Le ratio peut varier considérablement selon l'architecture
#[test]
fn test_activation_memory_scale() {
    let json = include_str!("../../../../Neurax-IR/models/gpt2_medium.json");
    let result = analyze_json(json).expect("Analysis should succeed");
    
    let activation_memory = result.memory.metrics.activation_memory_bytes;
    let forward_flops = result.compute.metrics.forward_flops;
    
    // Activation memory devrait être proportionnelle aux FLOPs
    // Ratio typique: 1 byte pour ~1000-20000 FLOPs
    let ratio = forward_flops / activation_memory as f64;
    
    // Le ratio devrait être entre 100 et 20000
    assert!(
        ratio > 100.0 && ratio < 20000.0,
        "FLOPs/activation_memory ratio should be 100-20000, got {:.0}",
        ratio
    );
    
    println!("✓ Activation memory scale: {:.0} FLOPs/byte", ratio);
}
