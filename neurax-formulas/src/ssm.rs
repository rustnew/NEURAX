//! State Space Model (SSM) formulas - Mamba, S4, H3
//!
//! Hot path — all functions are #[inline(always)] for zero-cost abstraction.

/// Compute FLOPs for Mamba SSM block
///
/// # Arguments
/// * `batch` - Batch size
/// * `seq_len` - Sequence length
/// * `hidden_size` - Hidden dimension (d_model)
/// * `state_dim` - SSM state dimension (d_state, typically 16)
/// * `expand_factor` - Expansion factor (typically 2)
#[inline(always)]
pub fn mamba_flops(
    batch: usize,
    seq_len: usize,
    hidden_size: usize,
    state_dim: usize,
    expand_factor: usize,
) -> f64 {
    let d_inner = hidden_size * expand_factor;
    
    // Input projection: [B, S, H] -> [B, S, d_inner * 2] (for x and z branches)
    let in_proj = 2.0 * batch as f64 * seq_len as f64 * hidden_size as f64 * (d_inner * 2) as f64;
    
    // Conv1d: [B, d_inner, S] with kernel size 4
    let conv1d = 2.0 * batch as f64 * d_inner as f64 * seq_len as f64 * 4.0;
    
    // SSM state update: selective scan
    // A, B, C, D parameters computed from x
    // State update: h[t] = A * h[t-1] + B * x[t]
    // Output: y[t] = C * h[t] + D * x[t]
    let ssm_params = 4.0 * d_inner as f64 * state_dim as f64; // A, B, C, D
    let ssm_state = 2.0 * batch as f64 * seq_len as f64 * d_inner as f64 * state_dim as f64;
    let ssm_output = 2.0 * batch as f64 * seq_len as f64 * d_inner as f64;
    
    // Output projection: [B, S, d_inner] -> [B, S, H]
    let out_proj = 2.0 * batch as f64 * seq_len as f64 * d_inner as f64 * hidden_size as f64;
    
    in_proj + conv1d + ssm_params + ssm_state + ssm_output + out_proj
}

/// Compute FLOPs for S4 (Structured State Space) block
#[inline(always)]
pub fn s4_flops(
    batch: usize,
    seq_len: usize,
    hidden_size: usize,
    state_dim: usize,
) -> f64 {
    // S4 uses a structured matrix for efficient computation
    // FFT-based convolution: O(N log N) instead of O(N^2)
    
    // Input projection
    let in_proj = 2.0 * batch as f64 * seq_len as f64 * hidden_size as f64 * hidden_size as f64;
    
    // FFT convolution (simplified)
    let fft_flops = batch as f64 * hidden_size as f64 * seq_len as f64 * (seq_len as f64).log2() * 2.0;
    
    // State update
    let state_flops = 2.0 * batch as f64 * seq_len as f64 * hidden_size as f64 * state_dim as f64;
    
    // Output projection
    let out_proj = 2.0 * batch as f64 * seq_len as f64 * hidden_size as f64 * hidden_size as f64;
    
    in_proj + fft_flops + state_flops + out_proj
}

/// Compute FLOPs for H3 (Hungry Hungry Hippos) block
#[inline(always)]
pub fn h3_flops(
    batch: usize,
    seq_len: usize,
    hidden_size: usize,
    state_dim: usize,
) -> f64 {
    // H3 combines SSM with attention-like computation
    
    // Input projection
    let in_proj = 2.0 * batch as f64 * seq_len as f64 * hidden_size as f64 * hidden_size as f64;
    
    // SSM layers (2 for H3)
    let ssm_flops = 2.0 * s4_flops(batch, seq_len, hidden_size, state_dim);
    
    // Output projection
    let out_proj = 2.0 * batch as f64 * seq_len as f64 * hidden_size as f64 * hidden_size as f64;
    
    in_proj + ssm_flops + out_proj
}

/// Compute parameters for Mamba block
#[inline(always)]
pub fn mamba_params(
    hidden_size: usize,
    state_dim: usize,
    expand_factor: usize,
) -> u64 {
    let d_inner = hidden_size * expand_factor;
    
    // Input projection: H -> d_inner * 2
    let in_proj = hidden_size * (d_inner * 2);
    
    // Conv1d: d_inner * kernel_size (typically 4)
    let conv1d = d_inner * 4;
    
    // SSM parameters: A, B, C, D for d_inner
    // A: d_inner * state_dim (log NDT)
    // B, C: d_inner * state_dim each
    // D: d_inner
    let ssm = d_inner * state_dim * 3 + d_inner;
    
    // Output projection: d_inner -> H
    let out_proj = d_inner * hidden_size;
    
    (in_proj + conv1d + ssm + out_proj) as u64
}

/// Compute FLOPs for Mamba Conv1d operation specifically
#[inline(always)]
pub fn mamba_conv1d_flops(
    batch: usize,
    seq_len: usize,
    d_inner: usize,
    kernel_size: usize,
) -> f64 {
    2.0 * batch as f64 * d_inner as f64 * seq_len as f64 * kernel_size as f64
}

/// Compute FLOPs for SSM state update specifically
#[inline(always)]
pub fn ssm_state_update_flops(
    batch: usize,
    seq_len: usize,
    d_inner: usize,
    state_dim: usize,
) -> f64 {
    // h[t] = A * h[t-1] + B * x[t]
    // y[t] = C * h[t] + D * x[t]
    4.0 * batch as f64 * seq_len as f64 * d_inner as f64 * state_dim as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mamba_flops() {
        // Mamba-1.4B: hidden=2048, state=16, expand=2
        let flops = mamba_flops(1, 2048, 2048, 16, 2);
        assert!(flops > 0.0);
    }

    #[test]
    fn test_mamba_params() {
        let params = mamba_params(2048, 16, 2);
        // Should be roughly 4 * hidden^2 for projections plus SSM params
        assert!(params > 0);
    }
}
