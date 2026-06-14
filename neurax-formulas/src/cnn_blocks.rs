//! CNN Block formulas - ResNet, Inception, MobileNet, DenseNet blocks

/// Compute parameters for ResNet Basic Block (2 convs + skip connection)
///
/// Structure: conv3x3 → BN → ReLU → conv3x3 → BN → add(skip) → ReLU
pub fn resnet_basic_block_params(
    in_channels: usize,
    out_channels: usize,
    stride: usize,
    bias: bool,
) -> u64 {
    // First conv: in_channels → out_channels, 3×3
    let conv1 = conv_params(in_channels, out_channels, 3, bias);
    
    // Second conv: out_channels → out_channels, 3×3
    let conv2 = conv_params(out_channels, out_channels, 3, bias);
    
    // Two BatchNorm layers (weight + bias + running_mean + running_var = 4 × channels)
    let bn_params = (2 * 4 * out_channels) as u64;
    
    // Downsample projection if dimensions don't match
    let downsample = if stride != 1 || in_channels != out_channels {
        // 1×1 conv for projection + BN
        conv_params(in_channels, out_channels, 1, bias) + (4 * out_channels) as u64
    } else {
        0
    };
    
    conv1 + conv2 + bn_params + downsample
}

/// Compute parameters for ResNet Bottleneck Block (3 convs)
///
/// Structure: 1×1 → 3×3 → 1×1 with expansion factor (typically 4)
pub fn resnet_bottleneck_block_params(
    in_channels: usize,
    mid_channels: usize,
    out_channels: usize,
    stride: usize,
    bias: bool,
) -> u64 {
    // 1×1 reduction
    let conv1 = conv_params(in_channels, mid_channels, 1, bias);
    
    // 3×3 conv
    let conv2 = conv_params(mid_channels, mid_channels, 3, bias);
    
    // 1×1 expansion
    let conv3 = conv_params(mid_channels, out_channels, 1, bias);
    
    // Three BatchNorm layers
    let bn_params = (3 * 4 * mid_channels + 4 * out_channels) as u64;
    
    // Downsample if needed
    let downsample = if stride != 1 || in_channels != out_channels {
        conv_params(in_channels, out_channels, 1, bias) + (4 * out_channels) as u64
    } else {
        0
    };
    
    conv1 + conv2 + conv3 + bn_params + downsample
}

/// Compute parameters for Inception Module (multiple parallel branches)
///
/// Structure: 1×1 + 3×3 + 5×5 + pool branches
pub fn inception_module_params(
    in_channels: usize,
    out_1x1: usize,
    out_3x3_reduce: usize,
    out_3x3: usize,
    out_5x5_reduce: usize,
    out_5x5: usize,
    pool_proj: usize,
    bias: bool,
) -> u64 {
    // 1×1 branch
    let branch1x1 = conv_params(in_channels, out_1x1, 1, bias);
    
    // 3×3 branch (with reduction)
    let branch3x3 = conv_params(in_channels, out_3x3_reduce, 1, bias)
                  + conv_params(out_3x3_reduce, out_3x3, 3, bias);
    
    // 5×5 branch (with reduction)
    let branch5x5 = conv_params(in_channels, out_5x5_reduce, 1, bias)
                  + conv_params(out_5x5_reduce, out_5x5, 5, bias);
    
    // Pool branch
    let branch_pool = conv_params(in_channels, pool_proj, 1, bias);
    
    // BatchNorm for each conv (simplified: 2 per branch)
    let bn_params = (2 * (out_1x1 + out_3x3_reduce + out_3x3 + out_5x5_reduce + out_5x5 + pool_proj)) as u64;
    
    branch1x1 + branch3x3 + branch5x5 + branch_pool + bn_params
}

/// Compute parameters for MBConv (MobileNet Inverted Residual Block)
///
/// Structure: 1×1 expand → 3×3/5×5 depthwise → 1×1 project
pub fn mbconv_params(
    in_channels: usize,
    out_channels: usize,
    expand_factor: usize,
    kernel_size: usize,
    stride: usize,
    bias: bool,
) -> u64 {
    let expanded = in_channels * expand_factor;
    
    // Expansion phase (skip if expand_factor=1)
    let expand_params = if expand_factor > 1 {
        conv_params(in_channels, expanded, 1, bias) + (4 * expanded) as u64 // conv + BN
    } else {
        0
    };
    
    // Depthwise conv
    let depthwise = (expanded * kernel_size * kernel_size + 4 * expanded) as u64; // depthwise + BN
    
    // Projection phase (always present, no bias in original MobileNet)
    let project = conv_params(expanded, out_channels, 1, false) + (4 * out_channels) as u64;
    
    expand_params + depthwise + project
}

/// Compute parameters for DenseNet Dense Block
///
/// Structure: BN → ReLU → 1×1 conv → BN → ReLU → 3×3 conv
/// Growth rate determines output channels per layer
pub fn dense_block_params(
    in_channels: usize,
    growth_rate: usize,
    num_layers: usize,
    bottleneck_factor: usize,
    bias: bool,
) -> u64 {
    let mut total_params: u64 = 0;
    let mut channels = in_channels;
    
    for _ in 0..num_layers {
        // Bottleneck (1×1 conv)
        let bn_channels = channels * bottleneck_factor;
        let bottleneck = conv_params(channels, bn_channels, 1, bias) + (4 * bn_channels) as u64;
        
        // Main conv (3×3)
        let main_conv = conv_params(bn_channels, growth_rate, 3, bias) + (4 * growth_rate) as u64;
        
        total_params += bottleneck + main_conv;
        channels += growth_rate; // Concatenation
    }
    
    total_params
}

/// Compute parameters for ConvNeXt Block
///
/// Structure: 7×7 depthwise → LayerNorm → 1×1 → GELU → 1×1
pub fn convnext_block_params(
    channels: usize,
    mlp_ratio: f64,
    bias: bool,
) -> u64 {
    // Depthwise 7×7 conv
    let depthwise = (channels * 7 * 7 + 4 * channels) as u64; // weights + LN
    
    // MLP: 1×1 expand → 1×1 project
    let mlp_hidden = (channels as f64 * mlp_ratio) as usize;
    let mlp = conv_params(channels, mlp_hidden, 1, bias) 
            + conv_params(mlp_hidden, channels, 1, bias);
    
    // Two LayerNorm layers
    let ln_params = (2 * 2 * channels) as u64;
    
    depthwise + mlp + ln_params
}

/// Compute parameters for ShuffleNet Unit
///
/// Structure: 1×1 → channel shuffle → 3×3 depthwise → 1×1
pub fn shuffle_unit_params(
    in_channels: usize,
    out_channels: usize,
    groups: usize,
    stride: usize,
    bias: bool,
) -> u64 {
    let mid_channels = out_channels / 2;
    
    // First 1×1 group conv
    let conv1 = ((in_channels / groups) * mid_channels * groups * 1 * 1 + 4 * mid_channels) as u64;
    
    // Depthwise 3×3
    let depthwise = (mid_channels * 3 * 3 + 4 * mid_channels) as u64;
    
    // Second 1×1 group conv
    let conv2 = ((mid_channels / groups) * out_channels * groups * 1 * 1 + 4 * out_channels) as u64;
    
    // Skip connection projection if needed
    let skip = if stride != 1 || in_channels != out_channels {
        conv_params(in_channels, out_channels, 1, bias) + (4 * out_channels) as u64
    } else {
        0
    };
    
    conv1 + depthwise + conv2 + skip
}

/// Compute parameters for C2f block (YOLOv8 style)
///
/// Structure: Split → multiple Bottlenecks → Concat
pub fn c2f_block_params(
    in_channels: usize,
    out_channels: usize,
    num_bottlenecks: usize,
    shortcut: bool,
    bias: bool,
) -> u64 {
    let hidden = out_channels / 2;
    
    // Initial conv
    let init_conv = conv_params(in_channels, out_channels, 1, bias);
    
    // Bottlenecks (simplified as 2 convs each)
    let mut bottleneck_params: u64 = 0;
    for _ in 0..num_bottlenecks {
        bottleneck_params += conv_params(hidden, hidden, 3, bias) * 2;
        if shortcut {
            bottleneck_params += (4 * hidden) as u64; // BN
        }
    }
    
    // Final conv
    let final_conv = conv_params(out_channels + hidden * num_bottlenecks, out_channels, 1, bias);
    
    init_conv + bottleneck_params + final_conv
}

/// Helper: Standard conv params
fn conv_params(in_ch: usize, out_ch: usize, kernel: usize, bias: bool) -> u64 {
    let weights = in_ch * out_ch * kernel * kernel;
    let bias_params = if bias { out_ch } else { 0 };
    (weights + bias_params) as u64
}

/// Compute FLOPs for ResNet Basic Block
pub fn resnet_basic_block_flops(
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    height: usize,
    width: usize,
    stride: usize,
) -> f64 {
    let out_h = height / stride;
    let out_w = width / stride;
    
    // First conv
    let conv1_flops = 2.0 * batch as f64 * out_channels as f64 * out_h as f64 * out_w as f64 
                    * in_channels as f64 * 3.0 * 3.0 / stride as f64;
    
    // Second conv
    let conv2_flops = 2.0 * batch as f64 * out_channels as f64 * out_h as f64 * out_w as f64 
                    * out_channels as f64 * 3.0 * 3.0;
    
    // Skip projection if needed
    let skip_flops = if stride != 1 || in_channels != out_channels {
        2.0 * batch as f64 * out_channels as f64 * out_h as f64 * out_w as f64 
        * in_channels as f64 * 1.0 * 1.0
    } else {
        0.0
    };
    
    conv1_flops + conv2_flops + skip_flops
}

/// Compute FLOPs for MBConv block
pub fn mbconv_flops(
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    height: usize,
    width: usize,
    expand_factor: usize,
    kernel_size: usize,
    stride: usize,
) -> f64 {
    let expanded = in_channels * expand_factor;
    let out_h = height / stride;
    let out_w = width / stride;
    
    // Expansion
    let expand_flops = if expand_factor > 1 {
        2.0 * batch as f64 * height as f64 * width as f64 * in_channels as f64 * expanded as f64
    } else {
        0.0
    };
    
    // Depthwise
    let depthwise_flops = 2.0 * batch as f64 * expanded as f64 * out_h as f64 * out_w as f64 
                        * kernel_size as f64 * kernel_size as f64;
    
    // Projection
    let project_flops = 2.0 * batch as f64 * out_channels as f64 * out_h as f64 * out_w as f64 
                      * expanded as f64;
    
    expand_flops + depthwise_flops + project_flops
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resnet_basic_block() {
        // ResNet-18 style: 64 → 64, stride=1
        let params = resnet_basic_block_params(64, 64, 1, false);
        // 2 × (64 × 64 × 3 × 3) = 73,728 params
        assert!(params > 70000 && params < 80000);
    }

    #[test]
    fn test_resnet_bottleneck() {
        // ResNet-50 style: 256 → 512 → 1024
        let params = resnet_bottleneck_block_params(256, 128, 512, 1, false);
        assert!(params > 0);
    }

    #[test]
    fn test_mbconv() {
        // MobileNetV2 style
        let params = mbconv_params(32, 16, 6, 3, 1, false);
        assert!(params > 0);
    }

    #[test]
    fn test_inception() {
        // InceptionV3 style
        let params = inception_module_params(288, 64, 48, 64, 8, 32, 64, false);
        assert!(params > 0);
    }

    #[test]
    fn test_dense_block() {
        // DenseNet-121 style: growth_rate=32, 6 layers
        let params = dense_block_params(64, 32, 6, 4, false);
        assert!(params > 0);
    }

    #[test]
    fn test_convnext_block() {
        // ConvNeXt-Tiny style
        let params = convnext_block_params(96, 4.0, false);
        assert!(params > 0);
    }
}
