//! NEURAX MLIR Passes
//!
//! This module provides MLIR passes for transforming and analyzing
//! NEURAX IR modules.

pub mod architecture;
pub mod compute;
pub mod memory;
pub mod hardware;
pub mod parallelism;
pub mod cost;

// Re-export pass traits
pub use architecture::ArchitecturePass;
pub use compute::ComputePass;
pub use memory::MemoryPass;
pub use hardware::HardwarePass;
pub use parallelism::ParallelismPass;
pub use cost::CostPass;
