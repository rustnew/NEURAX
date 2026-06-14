//! Convolution layer formulas
//!
//! Hot path — all functions are #[inline(always)] for zero-cost abstraction.

/// Compute FLOPs for Conv2D layer
///
/// # Arguments
/// * `batch` - Batch size
/// * `in_channels` - Number of input channels
/// * `out_channels` - Number of output channels
/// * `height` - Input height
/// * `width` - Input width
/// * `kernel_h` - Kernel height
/// * `kernel_w` - Kernel width
/// * `stride` - Stride
/// * `padding` - Padding
/// * `groups` - Groups (1 for standard conv, in_channels for depthwise)
#[inline(always)]
pub fn conv2d_flops(
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    height: usize,
    width: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride: usize,
    padding: usize,
    groups: usize,
) -> f64 {
    // Output dimensions
    let out_h = (height + 2 * padding - kernel_h) / stride + 1;
    let out_w = (width + 2 * padding - kernel_w) / stride + 1;
    
    // FLOPs per output position
    let flops_per_pos = 2.0 * (in_channels / groups) as f64 * kernel_h as f64 * kernel_w as f64;
    
    // Total FLOPs
    batch as f64 * out_channels as f64 * out_h as f64 * out_w as f64 * flops_per_pos
}

/// Compute FLOPs for depthwise separable convolution
#[inline(always)]
pub fn depthwise_separable_conv2d_flops(
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride: usize,
    padding: usize,
) -> f64 {
    // Depthwise conv
    let depthwise = conv2d_flops(batch, channels, channels, height, width, kernel_h, kernel_w, stride, padding, channels);
    
    // Pointwise conv (1x1)
    let pointwise = conv2d_flops(
        batch, channels, channels,
        (height + 2 * padding - kernel_h) / stride + 1,
        (width + 2 * padding - kernel_w) / stride + 1,
        1, 1, 1, 0, 1
    );
    
    depthwise + pointwise
}

/// Compute FLOPs for Conv3D layer
#[inline(always)]
pub fn conv3d_flops(
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    depth: usize,
    height: usize,
    width: usize,
    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride: usize,
    padding: usize,
) -> f64 {
    let out_d = (depth + 2 * padding - kernel_d) / stride + 1;
    let out_h = (height + 2 * padding - kernel_h) / stride + 1;
    let out_w = (width + 2 * padding - kernel_w) / stride + 1;
    
    let flops_per_pos = 2.0 * in_channels as f64 * kernel_d as f64 * kernel_h as f64 * kernel_w as f64;
    
    batch as f64 * out_channels as f64 * out_d as f64 * out_h as f64 * out_w as f64 * flops_per_pos
}

/// Compute parameters for Conv2D layer
#[inline(always)]
pub fn conv2d_params(
    in_channels: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    groups: usize,
    bias: bool,
) -> u64 {
    let weight_params = out_channels * (in_channels / groups) * kernel_h * kernel_w;
    let bias_params = if bias { out_channels } else { 0 };
    (weight_params + bias_params) as u64
}

/// Compute output shape for Conv2D
#[inline(always)]
pub fn conv2d_output_shape(
    height: usize,
    width: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride: usize,
    padding: usize,
) -> (usize, usize) {
    let out_h = (height + 2 * padding - kernel_h) / stride + 1;
    let out_w = (width + 2 * padding - kernel_w) / stride + 1;
    (out_h, out_w)
}

/// Compute FLOPs for pooling layer
#[inline(always)]
pub fn pooling_flops(
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    kernel_size: usize,
    stride: usize,
) -> f64 {
    let out_h = (height - kernel_size) / stride + 1;
    let out_w = (width - kernel_size) / stride + 1;
    // Each pool operation: kernel_size² comparisons or additions
    batch as f64 * channels as f64 * out_h as f64 * out_w as f64 * kernel_size.pow(2) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_conv2d_flops_resnet() {
        // ResNet first conv: 3×224×224 → 64 channels, 7×7 kernel, stride 2
        let flops = conv2d_flops(1, 3, 64, 224, 224, 7, 7, 2, 3, 1);
        // Output: 112×112, FLOPs = 2 × 3 × 7 × 7 × 64 × 112 × 112 ≈ 236M
        assert!(flops > 200_000_000.0 && flops < 300_000_000.0);
    }

    #[test]
    fn test_conv2d_params() {
        let params = conv2d_params(3, 64, 7, 7, 1, true);
        // 64 × 3 × 7 × 7 + 64 = 9,408 + 64 = 9,472
        assert_eq!(params, 9_472);
    }

    #[test]
    fn test_conv2d_output_shape() {
        let (h, w) = conv2d_output_shape(224, 224, 7, 7, 2, 3);
        assert_eq!(h, 112);
        assert_eq!(w, 112);
    }
}
