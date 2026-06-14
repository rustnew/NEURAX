//! NEURAX MLIR Context management

use melior::Context;

/// NEURAX MLIR Context wrapper
pub struct NeuraxContext {
    pub context: Context,
}

impl NeuraxContext {
    /// Create a new NEURAX MLIR context
    pub fn new() -> Self {
        let context = Context::new();
        
        // Enable unregistered dialects for NEURAX custom dialects
        context.set_allow_unregistered_dialects(true);
        
        Self { context }
    }
    
    /// Get the underlying MLIR context
    pub fn as_context(&self) -> &Context {
        &self.context
    }
}

impl Default for NeuraxContext {
    fn default() -> Self {
        Self::new()
    }
}
