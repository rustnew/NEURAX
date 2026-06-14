//! IREE Integration
//!
//! Provides integration with the IREE compiler for multi-backend deployment.
//! IREE can target: CPU, CUDA, Vulkan, Metal, ROCm from a single MLIR input.

mod compiler;

pub use compiler::{IreeCompiler, IreeTarget};

use crate::targets::TargetBackend;

/// IREE-supported target devices
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IreeDevice {
    /// CPU execution
    Cpu,
    /// NVIDIA CUDA
    Cuda,
    /// Vulkan (cross-platform GPU)
    Vulkan,
    /// Apple Metal
    Metal,
    /// AMD ROCm
    Rocm,
}

impl IreeDevice {
    /// Convert from TargetBackend
    pub fn from_backend(backend: TargetBackend) -> Self {
        match backend {
            TargetBackend::Cpu => Self::Cpu,
            TargetBackend::Cuda => Self::Cuda,
            TargetBackend::Vulkan => Self::Vulkan,
            TargetBackend::Metal => Self::Metal,
            TargetBackend::Rocm => Self::Rocm,
        }
    }
    
    /// Get IREE target device flag
    pub fn iree_flag(&self) -> &'static str {
        match self {
            Self::Cpu => "--iree-hal-target-device=local-task",
            Self::Cuda => "--iree-hal-target-device=cuda",
            Self::Vulkan => "--iree-hal-target-device=vulkan",
            Self::Metal => "--iree-hal-target-device=metal",
            Self::Rocm => "--iree-hal-target-device=rocm",
        }
    }
    
    /// Get IREE backend flag
    pub fn backend_flag(&self) -> &'static str {
        match self {
            Self::Cpu => "--iree-hal-local-target-device-backends=llvm-cpu",
            Self::Cuda => "--iree-hal-target-backends=cuda",
            Self::Vulkan => "--iree-hal-target-backends=vulkan-spirv",
            Self::Metal => "--iree-hal-target-backends=metal",
            Self::Rocm => "--iree-hal-target-backends=rocm",
        }
    }
    
    /// Get the output file extension
    pub fn output_extension(&self) -> &'static str {
        ".vmfb"
    }
}

impl std::fmt::Display for IreeDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Cuda => write!(f, "cuda"),
            Self::Vulkan => write!(f, "vulkan"),
            Self::Metal => write!(f, "metal"),
            Self::Rocm => write!(f, "rocm"),
        }
    }
}

/// Generate IREE-compatible MLIR (StableHLO or TOSA dialect)
pub fn generate_iree_compatible_mlir(
    model_name: &str,
    hidden_size: usize,
    num_heads: usize,
    seq_len: usize,
    dtype: &str,
) -> String {
    let head_dim = hidden_size / num_heads;
    
    // IREE prefers StableHLO or TOSA as input
    // For now, generate linalg which IREE can process
    format!(
r#"// IREE-compatible MLIR for {model_name}
// Input dialect: linalg (IREE will convert internally)
// Target: Multi-backend (CPU, CUDA, Vulkan, Metal, ROCm)

module @{model_name} attributes {{
  iree.module.export = true
}} {{
  // Model metadata
  func.func @model.info() -> (i64, i64, i64, i64) {{
    %hidden_size = arith.constant {hidden_size} : i64
    %num_heads = arith.constant {num_heads} : i64
    %head_dim = arith.constant {head_dim} : i64
    %seq_len = arith.constant {seq_len} : i64
    return %hidden_size, %num_heads, %head_dim, %seq_len : i64, i64, i64, i64
  }}

  // Entry point for inference
  func.func @forward(%input: tensor<{seq_len}x{hidden_size}x{dtype}>) -> tensor<{seq_len}x{hidden_size}x{dtype}> attributes {{
    iree.module.export,
    iree.abi.model = "raw"
  }} {{
    // Placeholder - actual implementation would have layer operations
    %output = tensor.empty() : tensor<{seq_len}x{hidden_size}x{dtype}>
    return %output : tensor<{seq_len}x{hidden_size}x{dtype}>
  }}

  // Attention layer
  func.func @attention(%input: tensor<{seq_len}x{hidden_size}x{dtype}>) -> tensor<{seq_len}x{hidden_size}x{dtype}> {{
    %output = tensor.empty() : tensor<{seq_len}x{hidden_size}x{dtype}>
    return %output : tensor<{seq_len}x{hidden_size}x{dtype}>
  }}

  // MLP layer
  func.func @mlp(%input: tensor<{seq_len}x{hidden_size}x{dtype}>) -> tensor<{seq_len}x{hidden_size}x{dtype}> {{
    %output = tensor.empty() : tensor<{seq_len}x{hidden_size}x{dtype}>
    return %output : tensor<{seq_len}x{hidden_size}x{dtype}>
  }}
}}
"#,
        model_name = model_name,
        hidden_size = hidden_size,
        num_heads = num_heads,
        head_dim = head_dim,
        seq_len = seq_len,
        dtype = dtype
    )
}

/// Generate IREE compile command
pub fn generate_iree_compile_command(
    input_mlir: &str,
    output_vmfb: &str,
    device: IreeDevice,
) -> String {
    format!(
        "iree-compile {} {} -o {}",
        input_mlir,
        device.backend_flag(),
        output_vmfb
    )
}

/// Generate IREE run command
pub fn generate_iree_run_command(
    vmfb_path: &str,
    function: &str,
    input_shape: &str,
) -> String {
    format!(
        "iree-run-module --module={} --function={} --input={}",
        vmfb_path, function, input_shape
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_iree_device_flags() {
        assert!(IreeDevice::Cuda.iree_flag().contains("cuda"));
        assert!(IreeDevice::Vulkan.backend_flag().contains("vulkan"));
    }
    
    #[test]
    fn test_iree_compatible_mlir() {
        let code = generate_iree_compatible_mlir("test", 768, 12, 512, "f32");
        assert!(code.contains("iree.module.export"));
    }
    
    #[test]
    fn test_iree_compile_command() {
        let cmd = generate_iree_compile_command("model.mlir", "model.vmfb", IreeDevice::Cuda);
        assert!(cmd.contains("iree-compile"));
        assert!(cmd.contains("cuda"));
    }
}
