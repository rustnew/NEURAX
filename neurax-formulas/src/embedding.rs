//! Embedding layer formulas

/// Compute FLOPs for embedding lookup
///
/// Embedding is essentially a lookup operation, so FLOPs are minimal.
/// We count the memory reads as the cost.
pub fn embedding_flops(batch: usize, seq_len: usize, embedding_dim: usize) -> f64 {
    // Embedding lookup: just reading embedding_dim values per token
    // This is more of a memory operation than compute
    batch as f64 * seq_len as f64 * embedding_dim as f64
}

/// Compute parameters for embedding layer
pub fn embedding_params(vocab_size: usize, embedding_dim: usize) -> u64 {
    (vocab_size * embedding_dim) as u64
}

/// Compute FLOPs for positional encoding (learned)
pub fn learned_positional_encoding_flops(batch: usize, seq_len: usize, embedding_dim: usize) -> f64 {
    // Just addition with learned embeddings
    batch as f64 * seq_len as f64 * embedding_dim as f64
}

/// Compute FLOPs for sinusoidal positional encoding
pub fn sinusoidal_positional_encoding_flops(seq_len: usize, embedding_dim: usize) -> f64 {
    // Computed once and cached, so effectively 0 during forward pass
    // But for analysis: sin/cos for each position and dimension
    seq_len as f64 * embedding_dim as f64 * 10.0 // approximate sin/cos cost
}

/// Compute FLOPs for rotary position embedding (RoPE)
pub fn rope_flops(batch: usize, seq_len: usize, num_heads: usize, head_dim: usize) -> f64 {
    // RoPE applies rotation to each head
    // Complex multiplication for each position and head
    batch as f64 * num_heads as f64 * seq_len as f64 * head_dim as f64 * 4.0
}

/// Compute parameters for positional encoding
pub fn positional_encoding_params(max_seq_len: usize, embedding_dim: usize, learned: bool) -> u64 {
    if learned {
        (max_seq_len * embedding_dim) as u64
    } else {
        0 // Sinusoidal has no learned parameters
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_params() {
        // GPT-2 small vocab: 50257, embedding_dim: 768
        let params = embedding_params(50257, 768);
        assert_eq!(params, 50257 * 768);
    }

    #[test]
    fn test_embedding_flops() {
        let flops = embedding_flops(1, 1024, 768);
        assert_eq!(flops, 768.0 * 1024.0);
    }
}
