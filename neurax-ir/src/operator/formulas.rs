//! Operator FLOPs formulas

/// MatMul FLOPs: [M, K] × [K, N] → [M, N]
pub fn matmul_flops(m: usize, k: usize, n: usize) -> f64 {
    2.0 * m as f64 * k as f64 * n as f64
}

/// Batched MatMul FLOPs
pub fn batched_matmul_flops(batch: usize, m: usize, k: usize, n: usize) -> f64 {
    batch as f64 * matmul_flops(m, k, n)
}

/// Conv2D FLOPs: batch * out_h * out_w * out_ch * kernel_h * kernel_w * in_ch
pub fn conv2d_flops(batch: usize, out_h: usize, out_w: usize, out_ch: usize, kernel_h: usize, kernel_w: usize, in_ch: usize, groups: usize) -> f64 {
    let g = groups as f64;
    2.0 * batch as f64 * out_h as f64 * out_w as f64 * out_ch as f64 * kernel_h as f64 * kernel_w as f64 * (in_ch as f64 / g)
}

/// Depthwise Conv2D FLOPs
pub fn depthwise_conv2d_flops(batch: usize, out_h: usize, out_w: usize, channels: usize, kernel_h: usize, kernel_w: usize) -> f64 {
    conv2d_flops(batch, out_h, out_w, channels, kernel_h, kernel_w, channels, channels)
}

/// Softmax FLOPs: ~5 ops per element (exp, sum, div)
pub fn softmax_flops(num_elements: usize) -> f64 {
    5.0 * num_elements as f64
}

/// ReLU FLOPs: 1 comparison per element
pub fn relu_flops(num_elements: usize) -> f64 {
    num_elements as f64
}

/// GELU FLOPs: ~10 ops per element (approximation)
pub fn gelu_flops(num_elements: usize) -> f64 {
    10.0 * num_elements as f64
}

/// SiLU/Swish FLOPs: ~4 ops per element
pub fn silu_flops(num_elements: usize) -> f64 {
    4.0 * num_elements as f64
}

/// Add/Mul FLOPs: 1 op per element
pub fn elementwise_flops(num_elements: usize) -> f64 {
    num_elements as f64
}

/// MoE FLOPs: router + top_k experts
/// Router: batch * seq * hidden * num_experts (softmax over experts)
/// Expert: top_k * batch * seq * expert_flops (typically 2x MLP)
pub fn moe_flops(batch: usize, seq: usize, hidden: usize, num_experts: usize, top_k: usize, expert_intermediate: usize) -> f64 {
    // Router FLOPs: linear projection + softmax
    let router_flops = batch as f64 * seq as f64 * hidden as f64 * num_experts as f64 * 2.0;
    
    // Expert FLOPs: top_k experts, each expert is an MLP
    // MLP = 2 * hidden * intermediate (up projection + down projection)
    let expert_flops = 2.0 * hidden as f64 * expert_intermediate as f64;
    let total_expert_flops = top_k as f64 * batch as f64 * seq as f64 * expert_flops;
    
    router_flops + total_expert_flops
}

/// Pooling FLOPs: minimal (just comparison or addition per element)
/// Max pooling: 1 comparison per element in kernel
/// Avg pooling: 1 addition per element in kernel
pub fn pooling_flops(batch: usize, channels: usize, out_h: usize, out_w: usize, kernel_size: usize) -> f64 {
    batch as f64 * channels as f64 * out_h as f64 * out_w as f64 * kernel_size as f64
}
