//! NEURAX Formulas - Analytical formulas for ML operations
//!
//! This crate provides pure analytical formulas for computing FLOPs,
//! memory requirements, and other metrics for various ML layer types.

pub mod activation;
pub mod attention;
pub mod cnn_blocks;
pub mod conv;
pub mod custom;
pub mod diffusion;
pub mod embedding;
pub mod gnn;
pub mod lora;
pub mod mlp;
pub mod moe;
pub mod normalization;
pub mod rnn;
pub mod ssm;

pub use attention::*;
pub use conv::*;
pub use embedding::*;
pub use mlp::*;
pub use normalization::*;
pub use rnn::*;
pub use ssm::*;

/// Returns the number of bytes per element for a given dtype.
///
/// `f64`, not an integer type: int4 packs two values per byte (GPTQ/AWQ/
/// QLoRA-NF4/GGUF Q4 — the most-cited LLM quantization scheme in practice),
/// so its real per-element cost is 0.5 bytes, not representable as `usize`.
/// Returning `1` for it reported the same memory footprint as int8 — twice
/// the real size for exactly the quantization users reach for specifically
/// to fit a model in less memory.
pub fn dtype_bytes(dtype: &str) -> f64 {
    match dtype {
        "fp64" | "float64" => 8.0,
        "fp32" | "float32" => 4.0,
        "fp16" | "float16" => 2.0,
        "bf16" | "bfloat16" => 2.0,
        "fp8" | "float8" => 1.0,
        "int8" => 1.0,
        "int4" => 0.5, // packed, two values per byte
        _ => 4.0,      // default to fp32
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

    #[test]
    fn test_dtype_bytes() {
        assert_eq!(dtype_bytes("fp32"), 4.0);
        assert_eq!(dtype_bytes("fp16"), 2.0);
        assert_eq!(dtype_bytes("bf16"), 2.0);
        assert_eq!(dtype_bytes("int8"), 1.0);
    }

    #[test]
    fn test_int4_is_half_a_byte_not_a_full_one() {
        // Packed two-per-byte storage (GPTQ/AWQ/QLoRA-NF4/GGUF Q4) — reporting
        // 1.0 here (int8's size) doubled every int4 model's memory footprint.
        assert_eq!(dtype_bytes("int4"), 0.5);
        assert!(dtype_bytes("int4") < dtype_bytes("int8"));
    }
}
