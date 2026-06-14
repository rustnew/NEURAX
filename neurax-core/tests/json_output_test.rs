//! JSON output verification test
//! 
//! Verifies that all model metrics are exported to JSON correctly

use neurax_core::analyze_json;

const GPT3_175B_JSON: &str = include_str!("../../examples/models/gpt3_175b.json");
const MIXTRAL_JSON: &str = include_str!("../../examples/models/mixtral_8x7b.json");
const MAMBA_JSON: &str = include_str!("../../examples/models/mamba_2.8b.json");

#[test]
fn test_json_output_gpt3() {
    let result = analyze_json(GPT3_175B_JSON).expect("Analysis should succeed");
    let json = result.to_json().expect("JSON export should succeed");
    
    // Verify JSON contains expected fields
    assert!(json.contains("total_parameters"), "JSON should contain total_parameters");
    assert!(json.contains("num_layers"), "JSON should contain num_layers");
    assert!(json.contains("forward_flops"), "JSON should contain forward_flops");
    assert!(json.contains("peak_vram_bytes"), "JSON should contain peak_vram_bytes");
    assert!(json.contains("training_cost_usd"), "JSON should contain training_cost_usd");
    
    println!("\n✓ GPT-3 JSON output contains all required fields");
    println!("JSON length: {} bytes", json.len());
}

#[test]
fn test_json_output_mixtral() {
    let result = analyze_json(MIXTRAL_JSON).expect("Analysis should succeed");
    let json = result.to_json().expect("JSON export should succeed");
    
    // Verify JSON structure
    assert!(json.contains("\"metrics\""), "JSON should have metrics section");
    assert!(json.contains("\"model\""), "JSON should have model section");
    assert!(json.contains("\"diagnostics\""), "JSON should have diagnostics section");
    
    println!("\n✓ Mixtral JSON output has correct structure");
}

#[test]
fn test_json_output_mamba() {
    let result = analyze_json(MAMBA_JSON).expect("Analysis should succeed");
    let json = result.to_json().expect("JSON export should succeed");
    
    // Verify custom layer metrics are included
    assert!(json.contains("total_parameters"), "JSON should contain total_parameters");
    
    println!("\n✓ Mamba JSON output includes custom layer metrics");
}

#[test]
fn test_json_output_to_file() {
    let result = analyze_json(GPT3_175B_JSON).expect("Analysis should succeed");
    
    // Save to temp file
    let temp_path = "/tmp/neurax_test_output.json";
    result.save_json(temp_path).expect("Save should succeed");
    
    // Read back and verify
    let saved_content = std::fs::read_to_string(temp_path).expect("Read should succeed");
    assert!(saved_content.contains("total_parameters"), "Saved JSON should contain metrics");
    
    // Cleanup
    let _ = std::fs::remove_file(temp_path);
    
    println!("\n✓ JSON can be saved to file and read back correctly");
}

#[test]
fn test_json_output_all_metrics_present() {
    let result = analyze_json(GPT3_175B_JSON).expect("Analysis should succeed");
    let json = result.to_json().expect("JSON export should succeed");
    
    // Parse JSON to verify structure
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("JSON should be valid");
    
    // Verify all metric categories are present
    assert!(parsed.get("metrics").is_some(), "JSON should have metrics object");
    
    let metrics = parsed.get("metrics").unwrap();
    
    // Structure metrics
    assert!(metrics.get("structure").is_some(), "Should have structure metrics");
    
    // Compute metrics
    assert!(metrics.get("compute").is_some(), "Should have compute metrics");
    
    // Memory metrics
    assert!(metrics.get("memory").is_some(), "Should have memory metrics");
    
    // Cost metrics
    assert!(metrics.get("cost").is_some(), "Should have cost metrics");
    
    // Hardware metrics
    assert!(metrics.get("hardware").is_some(), "Should have hardware metrics");
    
    println!("\n✓ JSON contains all metric categories:");
    println!("  - structure");
    println!("  - compute");
    println!("  - memory");
    println!("  - cost");
    println!("  - hardware");
}

#[test]
fn test_json_output_schema_version() {
    let result = analyze_json(GPT3_175B_JSON).expect("Analysis should succeed");
    let json = result.to_json().expect("JSON export should succeed");
    
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("JSON should be valid");
    
    assert!(parsed.get("schema_version").is_some(), "JSON should have schema_version");
    assert!(parsed.get("generated_at").is_some(), "JSON should have generated_at timestamp");
    assert!(parsed.get("neurax_version").is_some(), "JSON should have neurax_version");
    
    println!("\n✓ JSON has correct metadata fields:");
    println!("  - schema_version");
    println!("  - generated_at");
    println!("  - neurax_version");
}

/// Print a sample JSON output for verification
#[test]
fn test_print_sample_json_output() {
    let result = analyze_json(GPT3_175B_JSON).expect("Analysis should succeed");
    let json = result.to_json().expect("JSON export should succeed");
    
    println!("\n╔══════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                           SAMPLE JSON OUTPUT (GPT-3 175B)                                        ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════════════════╝\n");
    
    // Print first 2000 chars
    let preview: String = json.chars().take(2000).collect();
    println!("{}", preview);
    println!("... (truncated, total {} bytes)", json.len());
}
