//! NEURAX MLIR Code Generation Library
//!
//! Provides MLIR dialects and lowering infrastructure for the NEURAX compiler:
//! - Custom dialects for architecture, operators, memory, hardware, parallelism
//! - Multi-target lowering to CPU, CUDA, Vulkan, Metal, ROCm
//! - IREE integration for cross-platform deployment

pub mod context;
pub mod module;
pub mod passes;
pub mod dialects;
pub mod integration;
pub mod compiler;
pub mod lowering;
pub mod targets;
pub mod iree;

// Re-export key types for convenience
pub use context::NeuraxContext;
pub use module::NeuraxModule;
pub use compiler::compile_model_to_mlir;

// Lowering infrastructure
pub use lowering::{LoweringContext, LoweringPass, ArchitectureLowering, OperatorLowering, MemoryLowering, HardwareLowering, ParallelismLowering};

// Target backends
pub use targets::{TargetBackend, TargetLowering, CpuBackend, CudaBackend, VulkanBackend, MetalBackend, RocmBackend};

// IREE integration
pub use iree::{IreeDevice, IreeCompiler, IreeTarget};

#[cfg(test)]
mod tests;
