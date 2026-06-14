//! Lowering Context - Tracks SSA value mapping and symbols during lowering

use melior::ir::{Location, r#type::Type};
use melior::Context;
use std::collections::HashMap;

/// Lowering context that tracks mappings between NEURAX dialect ops and standard MLIR
pub struct LoweringContext<'c> {
    /// Reference to the MLIR context
    context: &'c Context,
    
    /// Map from NEURAX operation names to their lowered SSA value names
    value_map: HashMap<String, String>,
    
    /// Map from NEURAX attribute names to MLIR attribute names
    attribute_map: HashMap<String, String>,
    
    /// Symbol table for function names
    symbol_table: HashMap<String, String>,
    
    /// Current location for generated operations
    current_location: Option<Location<'c>>,
    
    /// Target backend name (cpu, cuda, vulkan, metal, rocm)
    target: String,
    
    /// Type cache for commonly used types
    type_cache: HashMap<String, Type<'c>>,
}

impl<'c> LoweringContext<'c> {
    /// Create a new lowering context
    pub fn new(context: &'c Context, target: &str) -> Self {
        Self {
            context,
            value_map: HashMap::new(),
            attribute_map: HashMap::new(),
            symbol_table: HashMap::new(),
            current_location: None,
            target: target.to_string(),
            type_cache: HashMap::new(),
        }
    }
    
    /// Get the MLIR context
    pub fn context(&self) -> &'c Context {
        self.context
    }
    
    /// Get the target backend name
    pub fn target(&self) -> &str {
        &self.target
    }
    
    /// Set the current location
    pub fn set_location(&mut self, location: Location<'c>) {
        self.current_location = Some(location);
    }
    
    /// Get the current location or create an unknown location
    pub fn location(&self) -> Location<'c> {
        self.current_location.unwrap_or_else(|| Location::unknown(self.context))
    }
    
    /// Map a NEURAX value name to its lowered MLIR value name
    pub fn map_value(&mut self, neurax_name: &str, mlir_value_name: &str) {
        self.value_map.insert(neurax_name.to_string(), mlir_value_name.to_string());
    }
    
    /// Look up a previously lowered value name
    pub fn lookup_value(&self, neurax_name: &str) -> Option<&String> {
        self.value_map.get(neurax_name)
    }
    
    /// Map a NEURAX attribute to its MLIR attribute name
    pub fn map_attribute(&mut self, neurax_name: &str, mlir_attr_name: &str) {
        self.attribute_map.insert(neurax_name.to_string(), mlir_attr_name.to_string());
    }
    
    /// Look up a previously mapped attribute name
    pub fn lookup_attribute(&self, neurax_name: &str) -> Option<&String> {
        self.attribute_map.get(neurax_name)
    }
    
    /// Register a symbol (function name, global, etc.)
    pub fn register_symbol(&mut self, neurax_name: &str, mlir_name: &str) {
        self.symbol_table.insert(neurax_name.to_string(), mlir_name.to_string());
    }
    
    /// Look up a symbol's MLIR name
    pub fn lookup_symbol(&self, neurax_name: &str) -> Option<&String> {
        self.symbol_table.get(neurax_name)
    }
    
    /// Cache a type for reuse
    pub fn cache_type(&mut self, name: &str, ty: Type<'c>) {
        self.type_cache.insert(name.to_string(), ty);
    }
    
    /// Look up a cached type
    pub fn lookup_type(&self, name: &str) -> Option<&Type<'c>> {
        self.type_cache.get(name)
    }
    
    /// Check if targeting CPU
    pub fn is_cpu(&self) -> bool {
        self.target == "cpu" || self.target == "llvm"
    }
    
    /// Check if targeting NVIDIA GPU
    pub fn is_cuda(&self) -> bool {
        self.target == "cuda" || self.target == "nvidia"
    }
    
    /// Check if targeting Vulkan
    pub fn is_vulkan(&self) -> bool {
        self.target == "vulkan" || self.target == "spirv"
    }
    
    /// Check if targeting Metal
    pub fn is_metal(&self) -> bool {
        self.target == "metal" || self.target == "apple"
    }
    
    /// Check if targeting ROCm
    pub fn is_rocm(&self) -> bool {
        self.target == "rocm" || self.target == "amd"
    }
    
    /// Clear all mappings (for starting a new module)
    pub fn clear(&mut self) {
        self.value_map.clear();
        self.attribute_map.clear();
        self.symbol_table.clear();
        self.type_cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use melior::Context;
    
    #[test]
    fn test_context_creation() {
        let ctx = Context::new();
        ctx.set_allow_unregistered_dialects(true);
        let lowering_ctx = LoweringContext::new(&ctx, "cpu");
        assert_eq!(lowering_ctx.target(), "cpu");
        assert!(lowering_ctx.is_cpu());
        assert!(!lowering_ctx.is_cuda());
    }
}
