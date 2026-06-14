//! Integration tests for MLIR lowering infrastructure

use neurax_mlir::*;

// ============================================================================
// Target Backend Tests
// ============================================================================

#[test]
fn test_all_backends_available() {
    let backends = TargetBackend::all();
    assert_eq!(backends.len(), 5);
    assert!(backends.contains(&TargetBackend::Cpu));
    assert!(backends.contains(&TargetBackend::Cuda));
    assert!(backends.contains(&TargetBackend::Vulkan));
    assert!(backends.contains(&TargetBackend::Metal));
    assert!(backends.contains(&TargetBackend::Rocm));
}

#[test]
fn test_backend_parsing() {
    assert_eq!(TargetBackend::from_str("cpu"), Some(TargetBackend::Cpu));
    assert_eq!(TargetBackend::from_str("cuda"), Some(TargetBackend::Cuda));
    assert_eq!(TargetBackend::from_str("vulkan"), Some(TargetBackend::Vulkan));
    assert_eq!(TargetBackend::from_str("metal"), Some(TargetBackend::Metal));
    assert_eq!(TargetBackend::from_str("rocm"), Some(TargetBackend::Rocm));
    assert_eq!(TargetBackend::from_str("invalid"), None);
}

#[test]
fn test_gpu_detection() {
    assert!(!TargetBackend::Cpu.is_gpu());
    assert!(TargetBackend::Cuda.is_gpu());
    assert!(TargetBackend::Vulkan.is_gpu());
    assert!(TargetBackend::Metal.is_gpu());
    assert!(TargetBackend::Rocm.is_gpu());
}

#[test]
fn test_iree_targets() {
    assert_eq!(TargetBackend::Cpu.iree_target(), "local-task");
    assert_eq!(TargetBackend::Cuda.iree_target(), "cuda");
    assert_eq!(TargetBackend::Vulkan.iree_target(), "vulkan");
    assert_eq!(TargetBackend::Metal.iree_target(), "metal");
    assert_eq!(TargetBackend::Rocm.iree_target(), "rocm");
}

// ============================================================================
// CPU Backend Tests
// ============================================================================

#[test]
fn test_cpu_matmul() {
    let code = CpuBackend::lower_matmul(1, 1024, 1024, 1024, "f32").unwrap();
    assert!(code.contains("linalg.matmul"));
    assert!(code.contains("tensor<1x1024x1024xf32>"));
    assert!(code.contains("llvm.readonly"));
}

#[test]
fn test_cpu_conv2d() {
    let code = CpuBackend::lower_conv2d(1, 3, 64, 224, 224, 3, "f32").unwrap();
    assert!(code.contains("linalg.conv_2d"));
    assert!(code.contains("tensor<1x224x224x3xf32>"));
}

#[test]
fn test_cpu_attention() {
    let code = CpuBackend::lower_attention(2048, 8192, 64, "f32").unwrap();
    assert!(code.contains("attention"));
    assert!(code.contains("tensor<2048x8192xf32>"));
}

#[test]
fn test_cpu_supported_dtypes() {
    let dtypes = CpuBackend::supported_dtypes();
    assert!(dtypes.contains(&"f32"));
    assert!(dtypes.contains(&"f64"));
    assert!(dtypes.contains(&"i32"));
}

#[test]
fn test_cpu_module_attributes() {
    let attrs = CpuBackend::module_attributes();
    assert!(attrs.contains("llvm"));
}

#[test]
fn test_cpu_function_attributes() {
    let attrs = CpuBackend::function_attributes();
    assert!(attrs.contains("readonly"));
}

// ============================================================================
// CUDA Backend Tests
// ============================================================================

#[test]
fn test_cuda_matmul() {
    let code = CudaBackend::lower_matmul(1, 1024, 1024, 1024, "f16").unwrap();
    assert!(code.contains("linalg.matmul"));
    assert!(code.contains("gpu.kernel"));
    assert!(code.contains("tensor<1x1024x1024xf16>"));
}

#[test]
fn test_cuda_conv2d() {
    let code = CudaBackend::lower_conv2d(1, 3, 64, 224, 224, 3, "f16").unwrap();
    assert!(code.contains("linalg.conv_2d"));
    assert!(code.contains("gpu.kernel"));
}

#[test]
fn test_cuda_flash_attention() {
    let code = CudaBackend::lower_attention(2048, 8192, 64, "f16").unwrap();
    assert!(code.contains("flash_attention"));
    assert!(code.contains("gpu.kernel"));
}

#[test]
fn test_cuda_supported_dtypes() {
    let dtypes = CudaBackend::supported_dtypes();
    assert!(dtypes.contains(&"f16"));
    assert!(dtypes.contains(&"bf16"));
    assert!(dtypes.contains(&"f8"));
}

#[test]
fn test_cuda_module_attributes() {
    let attrs = CudaBackend::module_attributes();
    assert!(attrs.contains("gpu.container_module"));
    assert!(attrs.contains("ptx"));
}

// ============================================================================
// Vulkan Backend Tests
// ============================================================================

#[test]
fn test_vulkan_matmul() {
    let code = VulkanBackend::lower_matmul(1, 1024, 1024, 1024, "f32").unwrap();
    assert!(code.contains("linalg.matmul"));
}

#[test]
fn test_vulkan_conv2d() {
    let code = VulkanBackend::lower_conv2d(1, 3, 64, 224, 224, 3, "f32").unwrap();
    assert!(code.contains("linalg.conv_2d"));
}

#[test]
fn test_vulkan_attention() {
    let code = VulkanBackend::lower_attention(2048, 8192, 64, "f32").unwrap();
    assert!(code.contains("attention"));
}

#[test]
fn test_vulkan_module_attributes() {
    let attrs = VulkanBackend::module_attributes();
    assert!(attrs.contains("gpu.container_module"));
}

// ============================================================================
// Metal Backend Tests
// ============================================================================

#[test]
fn test_metal_matmul() {
    let code = MetalBackend::lower_matmul(1, 1024, 1024, 1024, "f16").unwrap();
    assert!(code.contains("linalg.matmul"));
}

#[test]
fn test_metal_conv2d() {
    let code = MetalBackend::lower_conv2d(1, 3, 64, 224, 224, 3, "f16").unwrap();
    assert!(code.contains("linalg.conv_2d"));
}

#[test]
fn test_metal_attention() {
    let code = MetalBackend::lower_attention(2048, 8192, 64, "f16").unwrap();
    assert!(code.contains("attention"));
}

#[test]
fn test_metal_supported_dtypes() {
    let dtypes = MetalBackend::supported_dtypes();
    assert!(dtypes.contains(&"f16"));
    assert!(dtypes.contains(&"f32"));
}

// ============================================================================
// ROCm Backend Tests
// ============================================================================

#[test]
fn test_rocm_matmul() {
    let code = RocmBackend::lower_matmul(1, 1024, 1024, 1024, "bf16").unwrap();
    assert!(code.contains("linalg.matmul"));
    assert!(code.contains("gpu.kernel"));
}

#[test]
fn test_rocm_conv2d() {
    let code = RocmBackend::lower_conv2d(1, 3, 64, 224, 224, 3, "bf16").unwrap();
    assert!(code.contains("linalg.conv_2d"));
}

#[test]
fn test_rocm_attention() {
    let code = RocmBackend::lower_attention(2048, 8192, 64, "bf16").unwrap();
    assert!(code.contains("attention"));
}

#[test]
fn test_rocm_supported_dtypes() {
    let dtypes = RocmBackend::supported_dtypes();
    assert!(dtypes.contains(&"bf16"));
    assert!(dtypes.contains(&"f16"));
    assert!(dtypes.contains(&"f32"));
}

#[test]
fn test_rocm_module_attributes() {
    let attrs = RocmBackend::module_attributes();
    assert!(attrs.contains("gpu.container_module"));
    assert!(attrs.contains("hip"));
}

// ============================================================================
// IREE Integration Tests
// ============================================================================

#[test]
fn test_iree_device_flags() {
    // Verify device flags are generated (format may vary by IREE version)
    let cpu_flag = IreeDevice::Cpu.backend_flag();
    assert!(cpu_flag.contains("iree-hal") && cpu_flag.contains("llvm-cpu"));
    
    let cuda_flag = IreeDevice::Cuda.backend_flag();
    assert!(cuda_flag.contains("cuda"));
    
    let vulkan_flag = IreeDevice::Vulkan.backend_flag();
    assert!(vulkan_flag.contains("vulkan"));
}

#[test]
fn test_iree_compiler_creation() {
    let _compiler = IreeCompiler::new();
}

// ============================================================================
// MLIR Syntax Validation Tests
// ============================================================================

#[test]
fn test_mlir_tensor_syntax() {
    let code = CpuBackend::lower_matmul(2, 1024, 512, 256, "f32").unwrap();
    assert!(code.contains("tensor<2x1024x512xf32>"));
    assert!(code.contains("tensor<2x512x256xf32>"));
    assert!(code.contains("tensor<2x1024x256xf32>"));
}

#[test]
fn test_mlir_function_syntax() {
    let code = CpuBackend::lower_matmul(1, 128, 128, 128, "f32").unwrap();
    assert!(code.contains("func.func @matmul"));
    assert!(code.contains("return"));
    assert!(code.contains("attributes {"));
}

#[test]
fn test_mlir_linalg_syntax() {
    let code = CpuBackend::lower_matmul(1, 64, 64, 64, "f32").unwrap();
    assert!(code.contains("linalg.matmul ins("));
    assert!(code.contains("outs("));
}

#[test]
fn test_conv2d_output_shape() {
    // Test that conv2d output shape is calculated correctly
    let code = CpuBackend::lower_conv2d(1, 3, 64, 224, 224, 7, "f32").unwrap();
    // Output should be 218x218 (224 - 7 + 1)
    assert!(code.contains("tensor<1x218x218x64xf32>"));
}

#[test]
fn test_attention_head_dimension() {
    // Test attention with hidden_size=8192, num_heads=64 => head_dim=128
    let code = CudaBackend::lower_attention(2048, 8192, 64, "f16").unwrap();
    assert!(code.contains("tensor<2048x64x128xf16>"));
}
