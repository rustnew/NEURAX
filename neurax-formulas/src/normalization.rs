//! Normalization layer formulas (LayerNorm, BatchNorm, RMSNorm)

/// Compute FLOPs for Layer Normalization
///
/// LayerNorm: mean, variance, normalize, scale, shift
/// ~5 operations per element
pub fn layer_norm_flops(batch: usize, seq_len: usize, hidden_size: usize) -> f64 {
    // Mean: H additions
    // Variance: H additions + H multiplications
    // Normalize: H divisions
    // Scale: H multiplications
    // Shift: H additions
    // Total: ~5 × H per position
    5.0 * batch as f64 * seq_len as f64 * hidden_size as f64
}

/// Compute FLOPs for RMS Normalization (used in LLaMA)
///
/// RMSNorm: compute RMS, normalize, scale
/// ~3 operations per element
pub fn rms_norm_flops(batch: usize, seq_len: usize, hidden_size: usize) -> f64 {
    // RMS: sum of squares
    // Normalize: divide by sqrt(RMS)
    // Scale: multiply by weight
    3.0 * batch as f64 * seq_len as f64 * hidden_size as f64
}

/// Compute FLOPs for Batch Normalization (training)
pub fn batch_norm_flops_training(batch: usize, channels: usize, height: usize, width: usize) -> f64 {
    let spatial = height * width;
    // Mean: B × spatial additions per channel
    // Variance: B × spatial operations per channel
    // Normalize, scale, shift: 4 × B × spatial × channels
    6.0 * batch as f64 * channels as f64 * spatial as f64
}

/// Compute FLOPs for Batch Normalization (inference)
pub fn batch_norm_flops_inference(batch: usize, channels: usize, height: usize, width: usize) -> f64 {
    // Uses running mean/var, just normalize, scale, shift
    let spatial = height * width;
    3.0 * batch as f64 * channels as f64 * spatial as f64
}

/// Compute parameters for Layer Normalization
pub fn layer_norm_params(hidden_size: usize, elementwise_affine: bool) -> u64 {
    if elementwise_affine {
        // weight + bias
        (2 * hidden_size) as u64
    } else {
        0
    }
}

/// Compute parameters for RMS Normalization
pub fn rms_norm_params(hidden_size: usize) -> u64 {
    // Only weight, no bias
    hidden_size as u64
}

/// Compute parameters for Batch Normalization
pub fn batch_norm_params(num_features: usize, affine: bool) -> u64 {
    if affine {
        // weight + bias + running_mean + running_var (not trainable but stored)
        (4 * num_features) as u64
    } else {
        (2 * num_features) as u64 // running stats only
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm_flops() {
        let flops = layer_norm_flops(1, 1024, 768);
        assert_eq!(flops, 5.0 * 1024.0 * 768.0);
    }

    #[test]
    fn test_layer_norm_params() {
        let params = layer_norm_params(768, true);
        assert_eq!(params, 2 * 768);
    }

    #[test]
    fn test_rms_norm_params() {
        let params = rms_norm_params(4096);
        assert_eq!(params, 4096);
    }
}
