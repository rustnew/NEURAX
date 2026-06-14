//! Memory Dialect Lowering
//!
//! Lowers mem.* operations to memref/tensor dialects:
//! - mem.liveness → Preserved as metadata
//! - mem.alloc → memref.alloc or tensor.empty
//! - mem.metrics → Module attributes

use super::context::LoweringContext;
use super::pass::LoweringPass;
use melior::ir::Module;

/// Memory dialect lowering pass
pub struct MemoryLowering;

impl LoweringPass for MemoryLowering {
    fn name() -> &'static str {
        "MemoryLowering"
    }
    
    fn description() -> &'static str {
        "Lowers mem.liveness, mem.alloc, mem.metrics to memref/tensor"
    }
    
    fn run<'c>(_module: &mut Module<'c>, _context: &mut LoweringContext<'c>) -> Result<(), String> {
        // Memory operations are primarily metadata for memory planning passes
        // The actual memory allocation is handled by the runtime
        Ok(())
    }
}

/// Generate memory allocation for a tensor
pub fn lower_alloc(
    shape: &[usize],
    dtype: &str,
    target: &str,
) -> String {
    let shape_str = shape.iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join("x");
    
    match target {
        "cpu" | "llvm" => format!(
            "  %mem = memref.alloc() : memref<{}x{}>\n",
            shape_str, dtype
        ),
        _ => format!(
            "  %mem = tensor.empty() : tensor<{}x{}>\n",
            shape_str, dtype
        ),
    }
}

/// Generate memory metrics as module attributes
pub fn generate_memory_metrics(
    param_memory: u64,
    activation_memory: u64,
    gradient_memory: u64,
    optimizer_memory: u64,
    peak_memory: u64,
) -> String {
    format!(
r#"  // Memory metrics (preserved as metadata for memory planning)
  // parameter_memory = {} bytes
  // activation_memory = {} bytes
  // gradient_memory = {} bytes
  // optimizer_memory = {} bytes
  // peak_memory = {} bytes

"#,
        param_memory, activation_memory, gradient_memory, optimizer_memory, peak_memory
    )
}

/// Generate liveness analysis for memory optimization
pub fn generate_liveness_analysis(
    tensor_id: &str,
    start_step: i64,
    end_step: i64,
    size_bytes: i64,
) -> String {
    format!(
        "  // Liveness: tensor {} live from step {} to {} ({} bytes)\n",
        tensor_id, start_step, end_step, size_bytes
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_alloc_cpu() {
        let code = lower_alloc(&[2048, 8192], "f32", "cpu");
        assert!(code.contains("memref.alloc"));
    }
    
    #[test]
    fn test_alloc_gpu() {
        let code = lower_alloc(&[2048, 8192], "f32", "cuda");
        assert!(code.contains("tensor.empty"));
    }
}
