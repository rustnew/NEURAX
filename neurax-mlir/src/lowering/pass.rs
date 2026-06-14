//! Lowering Pass Trait - Common interface for all lowering passes

use super::context::LoweringContext;
use melior::ir::Module;

/// Trait for all lowering passes
/// 
/// Each lowering pass converts NEURAX dialect operations to standard MLIR.
/// Passes are run in order: Architecture → Operator → Memory → Hardware → Parallelism
pub trait LoweringPass {
    /// Name of the lowering pass
    fn name() -> &'static str;
    
    /// Description of what this pass lowers
    fn description() -> &'static str;
    
    /// Run the lowering pass on the module
    /// 
    /// # Arguments
    /// * `module` - The MLIR module to transform
    /// * `context` - Lowering context tracking value mappings
    /// 
    /// # Returns
    /// * `Ok(())` if lowering succeeded
    /// * `Err(String)` with error message if lowering failed
    fn run<'c>(module: &mut Module<'c>, context: &mut LoweringContext<'c>) -> Result<(), String>;
}

/// Helper macro to implement LoweringPass for simple passes
#[macro_export]
macro_rules! impl_lowering_pass {
    ($name:ident, $desc:expr, $run_fn:expr) => {
        pub struct $name;
        
        impl $crate::lowering::pass::LoweringPass for $name {
            fn name() -> &'static str {
                stringify!($name)
            }
            
            fn description() -> &'static str {
                $desc
            }
            
            fn run<'c>(
                module: &mut melior::ir::Module<'c>,
                context: &mut $crate::lowering::context::LoweringContext<'c>,
            ) -> Result<(), String> {
                $run_fn(module, context)
            }
        }
    };
}
