//! Attention layer formulas
//!
//! Hot path — all functions are #[inline(always)] for zero-cost abstraction.

/// Compute FLOPs for self-attention layer
///
/// # Arguments
/// * `batch` - Batch size
/// * `seq_len` - Sequence length
/// * `hidden_size` - Hidden dimension
/// * `num_heads` - Number of attention heads
/// * `causal` - Whether attention is causal (masked)
///
/// # Returns
/// Total FLOPs for the attention layer (forward pass)
#[inline(always)]
pub fn attention_flops(batch: usize, seq_len: usize, hidden_size: usize, num_heads: usize, causal: bool) -> f64 {
    let head_dim = hidden_size / num_heads;
    
    // Q, K, V projections: 3 × (B × S × H × H) matmuls
    // Each matmul: 2 × B × S × H × H
    let qkv_flops = 3.0 * (2.0 * batch as f64 * seq_len as f64 * hidden_size as f64 * hidden_size as f64);
    
    // QK^T attention scores: B × heads × S × S × head_dim
    // Matmul: [B, heads, S, head_dim] × [B, heads, head_dim, S] → [B, heads, S, S]
    let attn_scores_flops = 2.0 * batch as f64 * num_heads as f64 * seq_len as f64 * seq_len as f64 * head_dim as f64;
    
    // Attention × V: [B, heads, S, S] × [B, heads, S, head_dim] → [B, heads, S, head_dim]
    let attn_v_flops = 2.0 * batch as f64 * num_heads as f64 * seq_len as f64 * seq_len as f64 * head_dim as f64;
    
    // Output projection: [B, S, H] × [H, H] → [B, S, H]
    let out_proj_flops = 2.0 * batch as f64 * seq_len as f64 * hidden_size as f64 * hidden_size as f64;
    
    // Softmax: ~5 × B × heads × S × S (exp, sum, div per position)
    let softmax_flops = 5.0 * batch as f64 * num_heads as f64 * seq_len as f64 * seq_len as f64;
    
    // For causal attention, we only compute half the attention matrix
    let causal_factor = if causal { 0.5 } else { 1.0 };
    
    qkv_flops + (attn_scores_flops + attn_v_flops + softmax_flops) * causal_factor + out_proj_flops
}

/// Compute FLOPs for FlashAttention (memory-optimized, same FLOPs as standard)
#[inline(always)]
pub fn flash_attention_flops(
    batch: usize,
    seq_len: usize,
    hidden_size: usize,
    num_heads: usize,
    causal: bool,
) -> f64 {
    // FlashAttention has same FLOPs as standard attention but lower memory
    attention_flops(batch, seq_len, hidden_size, num_heads, causal)
}

/// Compute FLOPs for multi-query attention (MQA) or grouped-query attention (GQA)
#[inline(always)]
pub fn gqa_flops(
    batch: usize,
    seq_len: usize,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    causal: bool,
) -> f64 {
    let head_dim = hidden_size / num_heads;
    
    // Q projection (full heads)
    let q_flops = 2.0 * batch as f64 * seq_len as f64 * hidden_size as f64 * hidden_size as f64;
    
    // K, V projections (reduced heads)
    let kv_dim = num_kv_heads * head_dim;
    let kv_flops = 2.0 * 2.0 * batch as f64 * seq_len as f64 * hidden_size as f64 * kv_dim as f64;
    
    // Attention computation
    let attn_scores_flops = 2.0 * batch as f64 * num_heads as f64 * seq_len as f64 * seq_len as f64 * head_dim as f64;
    let attn_v_flops = 2.0 * batch as f64 * num_heads as f64 * seq_len as f64 * seq_len as f64 * head_dim as f64;
    
    // Output projection
    let out_proj_flops = 2.0 * batch as f64 * seq_len as f64 * hidden_size as f64 * hidden_size as f64;
    
    let causal_factor = if causal { 0.5 } else { 1.0 };
    
    q_flops + kv_flops + (attn_scores_flops + attn_v_flops) * causal_factor + out_proj_flops
}

/// Compute parameters for attention layer
#[inline(always)]
pub fn attention_params(hidden_size: usize, num_heads: usize, bias: bool) -> u64 {
    let head_dim = hidden_size / num_heads;
    
    // Q, K, V projections: 3 × (H × H) weights
    let qkv_params = 3 * hidden_size * hidden_size;
    
    // Output projection: H × H
    let out_params = hidden_size * hidden_size;
    
    // Biases (optional)
    let bias_params = if bias {
        4 * hidden_size // Q, K, V, Out biases
    } else {
        0
    };
    
    (qkv_params + out_params + bias_params) as u64
}

/// Compute parameters for GQA/MQA
#[inline(always)]
pub fn gqa_params(hidden_size: usize, num_heads: usize, num_kv_heads: usize, bias: bool) -> u64 {
    let head_dim = hidden_size / num_heads;
    
    // Q projection
    let q_params = hidden_size * hidden_size;
    
    // K, V projections (reduced)
    let kv_dim = num_kv_heads * head_dim;
    let kv_params = 2 * hidden_size * kv_dim;
    
    // Output projection
    let out_params = hidden_size * hidden_size;
    
    let bias_params = if bias {
        hidden_size + 2 * kv_dim + hidden_size
    } else {
        0
    };
    
    (q_params + kv_params + out_params + bias_params) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_attention_flops_gpt2_small() {
        // GPT-2 small: batch=1, seq=1024, hidden=768, heads=12
        let flops = attention_flops(1, 1024, 768, 12, true);
        // Should be around 3-4e9 FLOPs per layer
        assert!(flops > 1e9 && flops < 1e10);
    }

    #[test]
    fn test_attention_params() {
        // GPT-2 small attention: Q, K, V, O projections
        // Q: 768×768, K: 768×768, V: 768×768, O: 768×768
        let params = attention_params(768, 12, true);
        // 4 × 768² + 4 × 768 = 2,359,296 + 3,072 = 2,362,368
        assert_eq!(params, 2_362_368);
    }
}
