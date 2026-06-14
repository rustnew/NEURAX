//! NEURAX MLIR Dialects
//!
//! This module defines all NEURAX dialects using melior's API.

pub mod utils;
pub mod architecture;
pub mod graph;
pub mod tensor;
pub mod operator;
pub mod compute;
pub mod memory;
pub mod hardware;
pub mod parallelism;
pub mod cost;
pub mod report;
pub mod training;
pub mod data;
pub mod optimization;

pub use architecture::ArchitectureDialect;
pub use graph::GraphDialect;
pub use tensor::TensorDialect;
pub use operator::OperatorDialect;
pub use compute::ComputeDialect;
pub use memory::MemoryDialect;
pub use hardware::HardwareDialect;
pub use parallelism::ParallelismDialect;
pub use cost::CostDialect;
pub use report::ReportDialect;
pub use training::TrainingDialect;
pub use data::DataDialect;
pub use optimization::OptimizationDialect;
