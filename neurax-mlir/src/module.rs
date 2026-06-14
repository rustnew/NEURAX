//! NEURAX MLIR Module management

use melior::ir::{Module, Location};
use melior::Context;

/// NEURAX MLIR Module wrapper
pub struct NeuraxModule<'c> {
    pub module: Module<'c>,
}

impl<'c> NeuraxModule<'c> {
    /// Create a new empty NEURAX module
    pub fn new(context: &'c Context) -> Self {
        let location = Location::unknown(context);
        let module = Module::new(location);
        
        Self { module }
    }
    
    /// Create a module from an existing MLIR module
    pub fn from_module(module: Module<'c>) -> Self {
        Self { module }
    }
    
    /// Get the underlying MLIR module
    pub fn as_module(&self) -> &Module<'c> {
        &self.module
    }
    
    /// Get a mutable reference to the underlying MLIR module
    pub fn as_module_mut(&mut self) -> &mut Module<'c> {
        &mut self.module
    }
}
