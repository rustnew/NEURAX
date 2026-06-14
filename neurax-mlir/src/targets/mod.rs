//! Target Backend Trait - Interface for target-specific code generation

mod cpu;
mod cuda;
mod vulkan;
mod metal;
mod rocm;

pub use cpu::CpuBackend;
pub use cuda::CudaBackend;
pub use vulkan::VulkanBackend;
pub use metal::MetalBackend;
pub use rocm::RocmBackend;

/// Supported target backends
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetBackend {
    /// CPU via LLVM IR
    Cpu,
    /// NVIDIA GPU via CUDA
    Cuda,
    /// Vulkan via SPIR-V
    Vulkan,
    /// Apple Metal
    Metal,
    /// AMD ROCm
    Rocm,
}

impl TargetBackend {
    /// Get the string name of the backend
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Vulkan => "vulkan",
            Self::Metal => "metal",
            Self::Rocm => "rocm",
        }
    }
    
    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "cpu" | "llvm" => Some(Self::Cpu),
            "cuda" | "nvidia" | "gpu" => Some(Self::Cuda),
            "vulkan" | "spirv" => Some(Self::Vulkan),
            "metal" | "apple" => Some(Self::Metal),
            "rocm" | "amd" => Some(Self::Rocm),
            _ => None,
        }
    }
    
    /// Get all supported backends
    pub fn all() -> &'static [TargetBackend] {
        &[Self::Cpu, Self::Cuda, Self::Vulkan, Self::Metal, Self::Rocm]
    }
    
    /// Check if this is a GPU backend
    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::Cuda | Self::Vulkan | Self::Metal | Self::Rocm)
    }
    
    /// Get the IREE target device name
    pub fn iree_target(&self) -> &'static str {
        match self {
            Self::Cpu => "local-task",
            Self::Cuda => "cuda",
            Self::Vulkan => "vulkan",
            Self::Metal => "metal",
            Self::Rocm => "rocm",
        }
    }
    
    /// Get the IREE backend flag
    pub fn iree_backend_flag(&self) -> &'static str {
        match self {
            Self::Cpu => "llvm-cpu",
            Self::Cuda => "cuda",
            Self::Vulkan => "vulkan",
            Self::Metal => "metal",
            Self::Rocm => "rocm",
        }
    }
}

/// Trait for target-specific lowering strategies
pub trait TargetLowering {
    /// Get the target backend this lowering supports
    fn backend() -> TargetBackend;
    
    /// Get supported data types for this target
    fn supported_dtypes() -> &'static [&'static str];
    
    /// Lower a tensor operation to target-specific MLIR
    /// 
    /// Returns the lowered operation as MLIR text
    fn lower_matmul(
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        dtype: &str,
    ) -> Result<String, String>;
    
    /// Lower a convolution operation
    fn lower_conv2d(
        batch: usize,
        in_channels: usize,
        out_channels: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
        dtype: &str,
    ) -> Result<String, String>;
    
    /// Lower an attention operation
    fn lower_attention(
        seq_len: usize,
        hidden_size: usize,
        num_heads: usize,
        dtype: &str,
    ) -> Result<String, String>;
    
    /// Get target-specific module attributes
    fn module_attributes() -> String {
        String::new()
    }
    
    /// Get target-specific function attributes
    fn function_attributes() -> String {
        String::new()
    }
}

impl std::fmt::Display for TargetBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
