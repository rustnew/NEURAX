//! IR Backend Abstraction
//!
//! Provides a trait describing the IR/code-generation backend used by NEURAX.
//! NEURAX uses the MLIR backend (`neurax-mlir`) for all IR lowering and code
//! generation. The analytical pipeline is always available; the MLIR code
//! generation layer is enabled through the `mlir` Cargo feature.

use std::sync::Arc;

/// IR Backend trait - abstraction over the MLIR backend.
pub trait IrBackend: Send + Sync {
    /// Get backend name
    fn name(&self) -> &'static str;
    
    /// Check if backend is available
    fn is_available(&self) -> bool;
    
    /// Get backend version
    fn version(&self) -> &'static str;
}

/// MLIR backend implementation (powered by the `neurax-mlir` crate).
#[derive(Debug, Clone)]
pub struct MlirBackend {
    name: &'static str,
    version: &'static str,
}

impl MlirBackend {
    /// Create a new MLIR backend
    pub fn new() -> Self {
        MlirBackend {
            name: "neurax-mlir",
            version: "0.18.0",
        }
    }

    /// Whether the MLIR code-generation layer is compiled in (the `mlir` feature).
    pub fn codegen_enabled(&self) -> bool {
        cfg!(feature = "mlir")
    }
}

impl Default for MlirBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl IrBackend for MlirBackend {
    fn name(&self) -> &'static str {
        self.name
    }
    
    fn is_available(&self) -> bool {
        // The analytical pipeline is always available; MLIR codegen is feature-gated.
        true
    }
    
    fn version(&self) -> &'static str {
        self.version
    }
}

/// Backend selector - returns the NEURAX MLIR backend.
pub fn select_backend() -> Arc<dyn IrBackend> {
    Arc::new(MlirBackend::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlir_backend() {
        let backend = MlirBackend::new();
        assert_eq!(backend.name(), "neurax-mlir");
        assert!(backend.is_available());
    }

    #[test]
    fn test_select_backend() {
        let backend = select_backend();
        assert_eq!(backend.name(), "neurax-mlir");
        assert!(backend.is_available());
    }
}
