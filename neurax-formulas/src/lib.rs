//! NEURAX Formulas - Analytical formulas for ML operations
//!
//! This crate provides pure analytical formulas for computing FLOPs,
//! memory requirements, and other metrics for various ML layer types.

pub mod attention;
pub mod conv;
pub mod mlp;
pub mod embedding;
pub mod normalization;
pub mod moe;
pub mod ssm;
pub mod rnn;
pub mod diffusion;
pub mod gnn;
pub mod custom;
pub mod cnn_blocks;

pub use attention::*;
pub use conv::*;
pub use mlp::*;
pub use embedding::*;
pub use normalization::*;
pub use ssm::*;
pub use rnn::*;

/// Returns the number of bytes per element for a given dtype
pub fn dtype_bytes(dtype: &str) -> usize {
    match dtype {
        "fp32" | "float32" => 4,
        "fp16" | "float16" => 2,
        "bf16" | "bfloat16" => 2,
        "fp8" | "float8" => 1,
        "int8" => 1,
        "int4" => 1, // packed
        _ => 4, // default to fp32
    }
}

/// Returns the multiplier for backward pass FLOPs
pub fn backward_flops_multiplier() -> f64 {
    2.0 // Standard approximation: backward ≈ 2× forward
}

/// Returns the optimizer FLOPs overhead (Adam)
pub fn optimizer_flops_multiplier() -> f64 {
    0.1 // ~10% overhead for Adam optimizer
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_dtype_bytes() {
        assert_eq!(dtype_bytes("fp32"), 4);
        assert_eq!(dtype_bytes("fp16"), 2);
        assert_eq!(dtype_bytes("bf16"), 2);
        assert_eq!(dtype_bytes("int8"), 1);
    }
}
