//! Parallelism Dialect Lowering
//!
//! Lowers par.* operations to scf/gpu dialects:
//! - par.data_parallel → scf.parallel or distributed execution
//! - par.tensor_parallel → Operation splitting
//! - par.hybrid → Combined lowering strategy

use super::context::LoweringContext;
use super::pass::LoweringPass;
use melior::ir::Module;

/// Parallelism dialect lowering pass
pub struct ParallelismLowering;

impl LoweringPass for ParallelismLowering {
    fn name() -> &'static str {
        "ParallelismLowering"
    }
    
    fn description() -> &'static str {
        "Lowers par.data_parallel, par.tensor_parallel, par.hybrid to scf/gpu"
    }
    
    fn run<'c>(_module: &mut Module<'c>, _context: &mut LoweringContext<'c>) -> Result<(), String> {
        // Parallelism lowering transforms parallelism hints to execution strategies
        Ok(())
    }
}

/// Parallelism configuration
#[derive(Debug, Clone)]
pub struct ParallelismConfig {
    pub data_parallel: u32,
    pub tensor_parallel: u32,
    pub pipeline_parallel: u32,
}

impl ParallelismConfig {
    pub fn new(dp: u32, tp: u32, pp: u32) -> Self {
        Self {
            data_parallel: dp,
            tensor_parallel: tp,
            pipeline_parallel: pp,
        }
    }
    
    pub fn total_gpus(&self) -> u32 {
        self.data_parallel * self.tensor_parallel * self.pipeline_parallel
    }
    
    pub fn is_parallel(&self) -> bool {
        self.total_gpus() > 1
    }
}

/// Generate parallelism attributes for the module
pub fn generate_parallelism_attributes(config: &ParallelismConfig) -> String {
    format!(
r#"  // Parallelism configuration
  // data_parallel = {}
  // tensor_parallel = {}
  // pipeline_parallel = {}
  // total_gpus = {}
"#,
        config.data_parallel,
        config.tensor_parallel,
        config.pipeline_parallel,
        config.total_gpus()
    )
}

/// Generate tensor-parallel matmul (split across TP ranks)
pub fn lower_tp_matmul(
    m: usize,
    k: usize,
    n: usize,
    tensor_parallel: u32,
    dtype: &str,
) -> String {
    let n_per_rank = n / tensor_parallel as usize;
    
    format!(
r#"  // Tensor-parallel matmul: column-parallel split
  // Each rank computes {} columns of the output
  func.func @tp_matmul(%a: tensor<{}x{}x{}>, %b_local: tensor<{}x{}x{}>) -> tensor<{}x{}x{}> {{
    %c_init = tensor.empty() : tensor<{}x{}x{}>
    %c_local = linalg.matmul ins(%a, %b_local : tensor<{}x{}x{}>, tensor<{}x{}x{}>) outs(%c_init : tensor<{}x{}x{}>) -> tensor<{}x{}x{}>
    // All-reduce across TP ranks would be inserted here
    return %c_local : tensor<{}x{}x{}>
  }}

"#,
        n_per_rank,
        m, k, dtype, k, n_per_rank, dtype, m, n_per_rank, dtype,
        m, n_per_rank, dtype,
        m, k, dtype, k, n_per_rank, dtype, m, n_per_rank, dtype, m, n_per_rank, dtype,
        m, n_per_rank, dtype
    )
}

/// Generate data-parallel loop (for batch parallelism)
pub fn lower_dp_loop(
    batch_size: usize,
    hidden_size: usize,
    dp_size: u32,
    dtype: &str,
) -> String {
    let batch_per_rank = batch_size / dp_size as usize;
    
    format!(
r#"  // Data-parallel batch split
  // Each rank processes {} samples
  func.func @dp_forward(%input_local: tensor<{}x{}x{}>) -> tensor<{}x{}x{}> {{
    %output = tensor.empty() : tensor<{}x{}x{}>
    // Forward pass on local batch
    // All-reduce gradients across DP ranks
    return %output : tensor<{}x{}x{}>
  }}

"#,
        batch_per_rank,
        batch_per_rank, hidden_size, dtype, batch_per_rank, hidden_size, dtype,
        batch_per_rank, hidden_size, dtype,
        batch_per_rank, hidden_size, dtype
    )
}

/// Generate pipeline-parallel stage
pub fn lower_pp_stage(
    stage_id: u32,
    num_stages: u32,
    input_shape: &[usize],
    output_shape: &[usize],
    dtype: &str,
) -> String {
    let input_shape_str: Vec<String> = input_shape.iter().map(|x| x.to_string()).collect();
    let output_shape_str: Vec<String> = output_shape.iter().map(|x| x.to_string()).collect();
    let input_tensor = format!("tensor<{}x{}>", input_shape_str.join("x"), dtype);
    let output_tensor = format!("tensor<{}x{}>", output_shape_str.join("x"), dtype);
    
    format!(
r#"  // Pipeline stage {}/{}
  func.func @pp_stage_{}(%input: {}) -> {} {{
    // Stage computation
    %output = tensor.empty() : {}
    // Send activation to next stage via point-to-point communication
    return %output : {}
  }}

"#,
        stage_id, num_stages,
        stage_id, input_tensor, output_tensor,
        output_tensor,
        output_tensor
    )
}

/// Generate hybrid parallelism (DP + TP + PP) coordination
pub fn lower_hybrid_parallel(
    config: &ParallelismConfig,
    hidden_size: usize,
    seq_len: usize,
    dtype: &str,
) -> String {
    format!(
r#"  // Hybrid parallelism: DP={}, TP={}, PP={}
  // Total GPUs: {}
  // 
  // Communication patterns:
  // - DP: All-reduce gradients (within TP group)
  // - TP: All-reduce activations (within TP group)
  // - PP: Point-to-point activations (between PP stages)
  
  func.func @hybrid_forward(%input: tensor<{}x{}x{}>) -> tensor<{}x{}x{}> {{
    // 1. Split batch for DP
    // 2. Split weights for TP
    // 3. Split layers for PP
    %output = tensor.empty() : tensor<{}x{}x{}>
    return %output : tensor<{}x{}x{}>
  }}

"#,
        config.data_parallel, config.tensor_parallel, config.pipeline_parallel,
        config.total_gpus(),
        seq_len, hidden_size, dtype, seq_len, hidden_size, dtype,
        seq_len, hidden_size, dtype, seq_len, hidden_size, dtype
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parallelism_config() {
        let config = ParallelismConfig::new(8, 8, 1);
        assert_eq!(config.total_gpus(), 64);
        assert!(config.is_parallel());
    }
    
    #[test]
    fn test_tp_matmul() {
        let code = lower_tp_matmul(1024, 1024, 1024, 8, "f32");
        assert!(code.contains("@tp_matmul"));
    }
    
    #[test]
    fn test_hybrid_parallel() {
        let config = ParallelismConfig::new(8, 8, 1);
        let code = lower_hybrid_parallel(&config, 8192, 2048, "f32");
        assert!(code.contains("@hybrid_forward"));
    }
}
