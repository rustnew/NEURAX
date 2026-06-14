//! MLIR Lowering Infrastructure
//!
//! This module provides lowering passes to convert NEURAX dialects
//! to standard MLIR operations (linalg, tensor, arith, scf).

mod context;
mod pass;
mod architecture;
mod operator;
mod memory;
mod hardware;
mod parallelism;

pub use context::LoweringContext;
pub use pass::LoweringPass;
pub use architecture::ArchitectureLowering;
pub use operator::OperatorLowering;
pub use memory::MemoryLowering;
pub use hardware::HardwareLowering;
pub use parallelism::ParallelismLowering;

use melior::ir::Module;

/// Run all lowering passes on a module
pub fn run_lowering_pipeline<'c>(
    module: &mut Module<'c>,
    context: &mut LoweringContext<'c>,
) -> Result<(), String> {
    // Order matters: architecture first, then operators, then memory/hardware
    ArchitectureLowering::run(module, context)?;
    OperatorLowering::run(module, context)?;
    MemoryLowering::run(module, context)?;
    HardwareLowering::run(module, context)?;
    ParallelismLowering::run(module, context)?;
    
    Ok(())
}
