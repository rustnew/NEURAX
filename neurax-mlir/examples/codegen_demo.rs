//! Code Generation Demo
//! 
//! Demonstrates step-by-step MLIR generation for various hardware targets.
//!
//! Run with: cargo run --example codegen_demo

use neurax_mlir::{CpuBackend, CudaBackend, RocmBackend, MetalBackend, VulkanBackend, TargetLowering};
use std::fs;

fn main() {
    println!("=== NEURAX-MLIR Code Generation Demo ===");
    println!();
    
    // Step 1: Define model parameters
    println!("Step 1: Define Model Parameters");
    println!("--------------------------------");
    let batch = 1;
    let seq_len = 2048;
    let hidden_size = 4096;
    let num_heads = 32;
    let m = 1024;
    let k = 1024;
    let n = 1024;
    println!("  Batch size: {}", batch);
    println!("  Sequence length: {}", seq_len);
    println!("  Hidden size: {}", hidden_size);
    println!("  Num heads: {}", num_heads);
    println!("  Matmul dims: {}x{}x{}", m, k, n);
    println!();
    
    // Step 2: Generate MLIR for CPU
    println!("Step 2: Generate CPU MLIR");
    println!("-------------------------");
    let cpu_matmul = CpuBackend::lower_matmul(batch, m, k, n, "f32").unwrap();
    println!("CPU Matmul MLIR:");
    println!("{}", indent(&cpu_matmul, "  "));
    println!();
    
    // Step 3: Generate MLIR for CUDA
    println!("Step 3: Generate CUDA MLIR");
    println!("--------------------------");
    let cuda_matmul = CudaBackend::lower_matmul(batch, m, k, n, "f16").unwrap();
    println!("CUDA Matmul MLIR:");
    println!("{}", indent(&cuda_matmul, "  "));
    println!();
    
    // Step 4: Generate Attention for CUDA
    println!("Step 4: Generate Flash Attention (CUDA)");
    println!("--------------------------------------");
    let cuda_attn = CudaBackend::lower_attention(seq_len, hidden_size, num_heads, "f16").unwrap();
    println!("CUDA Flash Attention MLIR:");
    println!("{}", indent(&cuda_attn, "  "));
    println!();
    
    // Step 5: Generate Conv2D for CPU
    println!("Step 5: Generate Conv2D (CPU)");
    println!("-----------------------------");
    let cpu_conv = CpuBackend::lower_conv2d(1, 3, 64, 224, 224, 7, "f32").unwrap();
    println!("CPU Conv2D MLIR:");
    println!("{}", indent(&cpu_conv, "  "));
    println!();
    
    // Step 6: Compare backends
    println!("Step 6: Compare Backend Attributes");
    println!("----------------------------------");
    println!("CPU module attributes:   {}", CpuBackend::module_attributes());
    println!("CPU function attributes: {}", CpuBackend::function_attributes());
    println!("CUDA module attributes:  {}", CudaBackend::module_attributes());
    println!("CUDA function attrs:     {}", CudaBackend::function_attributes());
    println!("ROCm module attributes:  {}", RocmBackend::module_attributes());
    println!("Metal module attributes: {}", MetalBackend::module_attributes());
    println!("Vulkan module attrs:     {}", VulkanBackend::module_attributes());
    println!();
    
    // Step 7: Save to file
    println!("Step 7: Save MLIR to File");
    println!("------------------------");
    let output_dir = std::path::Path::new("output_mlir");
    fs::create_dir_all(output_dir).expect("Failed to create output directory");
    
    let cpu_file = output_dir.join("cpu_matmul.mlir");
    fs::write(&cpu_file, &cpu_matmul).expect("Failed to write CPU MLIR");
    println!("  Saved: {}", cpu_file.display());
    
    let cuda_file = output_dir.join("cuda_matmul.mlir");
    fs::write(&cuda_file, &cuda_matmul).expect("Failed to write CUDA MLIR");
    println!("  Saved: {}", cuda_file.display());
    
    let attn_file = output_dir.join("cuda_attention.mlir");
    fs::write(&attn_file, &cuda_attn).expect("Failed to write attention MLIR");
    println!("  Saved: {}", attn_file.display());
    println!();
    
    // Step 8: Show compilation commands
    println!("Step 8: Compilation Commands");
    println!("---------------------------");
    println!("To compile the generated MLIR, run:");
    println!();
    println!("  # CPU target:");
    println!("  iree-compile output_mlir/cpu_matmul.mlir --iree-hal-target-device=local-task -o output_mlir/cpu_matmul.vmfb");
    println!();
    println!("  # CUDA target:");
    println!("  iree-compile output_mlir/cuda_matmul.mlir --iree-hal-target-device=cuda -o output_mlir/cuda_matmul.vmfb");
    println!();
    
    // Step 9: Show inference command
    println!("Step 9: Run Inference");
    println!("--------------------");
    println!("After compilation, run inference with:");
    println!();
    println!("  iree-run-module --device=cuda --module=output_mlir/cuda_matmul.vmfb --function=matmul --input=\"1x1024x1024xf16\" --input=\"1x1024x1024xf16\"");
    println!();
    
    println!("=== Demo Complete ===");
    println!();
    println!("Generated files in: output_mlir/");
    println!("  - cpu_matmul.mlir");
    println!("  - cuda_matmul.mlir");
    println!("  - cuda_attention.mlir");
}

fn indent(text: &str, prefix: &str) -> String {
    text.lines().map(|line| format!("{}{}", prefix, line)).collect::<Vec<_>>().join("\n")
}
