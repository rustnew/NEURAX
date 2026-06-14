//! Backward pass ratios per AtomOp per tuning.md §12
//! 
//! Exact backward/forward ratios based on mathematical analysis of backpropagation

use serde::Serialize;

/// Backward ratio with precision metadata
#[derive(Debug, Clone, Serialize)]
pub struct BackwardRatio {
    /// Ratio backward_flops / forward_flops
    pub ratio: f64,
    /// Mathematical formula explanation
    pub formula: &'static str,
    /// Confidence in this ratio (0.0-1.0)
    pub confidence: f64,
}

/// Atom operation kinds for backward ratio lookup
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AtomOpKind {
    // Linear operations
    MatMul,
    Linear,
    BatchedMatMul,
    
    // Attention operations
    MultiHeadAttention,
    GroupedQueryAttention,
    FlashAttention,
    AttentionScores,
    AttentionOutput,
    
    // Normalization
    LayerNorm,
    RmsNorm,
    BatchNorm,
    
    // Activations
    Softmax,
    Relu,
    Gelu,
    Silu,
    Swish,
    
    // Convolution
    Conv2d,
    DepthwiseConv2d,
    
    // Embedding
    TokenEmbedding,
    PositionalEmbedding,
    
    // MoE
    MoeExpertGroup,
    MoeRouter,
    
    // SSM (Mamba)
    SsmStateUpdate,
    
    // Elementwise
    ElementwiseAdd,
    ElementwiseMul,
    
    // Pooling
    MaxPool2d,
    AvgPool2d,
    GlobalAvgPool,
    
    // Other
    Dropout,
    Residual,
    Unknown,
}

/// Backward ratios lookup table
pub struct BackwardRatios;

impl BackwardRatios {
    /// Get exact backward ratio for an operation
    pub fn ratio_for_op(op: &AtomOpKind) -> BackwardRatio {
        match op {
            // ── Linear operations ─────────────────────────────────────────────
            // dL/dX = dL/dY × W^T : 1 matmul same size as forward
            // dL/dW = X^T × dL/dY : 1 matmul same size
            // Total backward = 2× forward
            AtomOpKind::MatMul | AtomOpKind::Linear | AtomOpKind::BatchedMatMul => BackwardRatio {
                ratio: 2.0,
                formula: "dX = dY×W^T, dW = X^T×dY → 2×forward",
                confidence: 1.0,
            },

            // ── Attention ───────────────────────────────────────────────────────
            // Backward attention ≈ 2.5× forward:
            // - dQ, dK, dV: each identical to forward (3×)
            // - dScore = dAttn × V^T: 1× identical
            // - softmax backward: gradient Jacobian dense (0.5× additional)
            AtomOpKind::MultiHeadAttention | AtomOpKind::GroupedQueryAttention => BackwardRatio {
                ratio: 2.5,
                formula: "dQ/dK/dV = 3× forward + softmax_bwd = 0.5× forward",
                confidence: 0.95,
            },

            AtomOpKind::FlashAttention => BackwardRatio {
                ratio: 2.5, // Same math operations, better IO
                formula: "Flash backward = Flash forward structure, ratio identical",
                confidence: 0.90,
            },

            AtomOpKind::AttentionScores => BackwardRatio {
                ratio: 1.5,
                formula: "dScore = dAttn × V^T + softmax_bwd",
                confidence: 0.90,
            },

            AtomOpKind::AttentionOutput => BackwardRatio {
                ratio: 2.0,
                formula: "dV = Attn^T × dOut, dAttn = dOut × V^T",
                confidence: 0.95,
            },

            // ── Normalization ───────────────────────────────────────────────────
            // LayerNorm backward: mean_grad + var_grad + x_hat_grad
            // = 4 passes over data
            AtomOpKind::LayerNorm => BackwardRatio {
                ratio: 4.0,
                formula: "dγ, dβ, dx_hat, d(mean), d(var) → 4×forward approx",
                confidence: 0.90,
            },

            // RMSNorm: simpler, no mean → 3× forward
            AtomOpKind::RmsNorm => BackwardRatio {
                ratio: 3.0,
                formula: "dγ, d(rms), dx → 3×forward approx",
                confidence: 0.90,
            },

            AtomOpKind::BatchNorm => BackwardRatio {
                ratio: 5.0, // running_mean/var gradient + batch sync
                formula: "dγ, dβ, dx with batch stats → 5×forward approx",
                confidence: 0.85,
            },

            // ── Softmax ─────────────────────────────────────────────────────────
            // dL/dx_i = p_i × (dL/dy_i - Σ_j(p_j × dL/dy_j))
            // = 3 passes over elements
            AtomOpKind::Softmax => BackwardRatio {
                ratio: 3.0,
                formula: "Jacobian diagonal + global correction → 3×forward",
                confidence: 0.95,
            },

            // ── Activations ─────────────────────────────────────────────────────
            // ReLU: binary mask already calculated, 1 multiply → 1×
            AtomOpKind::Relu => BackwardRatio {
                ratio: 1.0,
                formula: "dx = dL/dy × (x > 0) → 1×forward (mask precomputed)",
                confidence: 1.0,
            },

            // GELU: complex analytical gradient, ≈ 4× forward
            AtomOpKind::Gelu => BackwardRatio {
                ratio: 4.0,
                formula: "d/dx GELU(x) = 0.5*(1+tanh(...))+x*sech²(...) → 4×approx",
                confidence: 0.85,
            },

            // SiLU/Swish: sigmoid(x)*(1+x*(1-sigmoid(x))) → 3× forward
            AtomOpKind::Silu | AtomOpKind::Swish => BackwardRatio {
                ratio: 3.0,
                formula: "d/dx SiLU = sigmoid(x)*(1+x*(1-sig(x))) → 3×forward",
                confidence: 0.85,
            },

            // ── Convolution ──────────────────────────────────────────────────────
            // dL/dX: transposed convolution → same FLOPs
            // dL/dW: cross-correlation → same FLOPs
            // Total → 2× + 0.5× for strided dX
            AtomOpKind::Conv2d => BackwardRatio {
                ratio: 2.5,
                formula: "dX=deconv(dY,W), dW=conv(X,dY) → ~2.5× forward",
                confidence: 0.90,
            },

            AtomOpKind::DepthwiseConv2d => BackwardRatio {
                ratio: 2.0,
                formula: "Depthwise: dX and dW = 2× forward exact",
                confidence: 0.95,
            },

            // ── Embedding ────────────────────────────────────────────────────────
            // Embedding backward = scatter_add on indices
            // ≈ same FLOPs as embedding forward lookup (0 ops) but
            // gradient = accumulation, so 1× the gradient size
            AtomOpKind::TokenEmbedding | AtomOpKind::PositionalEmbedding => BackwardRatio {
                ratio: 1.0,
                formula: "scatter_add(grad_weight, indices, dL/dY) → 1×embed_size",
                confidence: 0.95,
            },

            // ── MoE ───────────────────────────────────────────────────────────────
            // Router backward + experts backward
            AtomOpKind::MoeExpertGroup => BackwardRatio {
                ratio: 2.5, // experts = 2× + router = 0.5×
                formula: "Expert FFN backward (2×) + router backward (0.5×)",
                confidence: 0.85,
            },

            AtomOpKind::MoeRouter => BackwardRatio {
                ratio: 2.0,
                formula: "Linear router backward → 2×forward",
                confidence: 0.90,
            },

            // ── SSM (Mamba) ──────────────────────────────────────────────────────
            // Backward SSM: reverse associative scan ≈ 1.5× forward
            AtomOpKind::SsmStateUpdate => BackwardRatio {
                ratio: 1.5,
                formula: "Reverse associative scan ≈ 1.5× forward",
                confidence: 0.80,
            },

            // ── Elementwise ──────────────────────────────────────────────────────
            AtomOpKind::ElementwiseAdd => BackwardRatio {
                ratio: 1.0, // Gradient distributed to both branches
                formula: "dL/dX = dL/dY, dL/dZ = dL/dY → 1×",
                confidence: 1.0,
            },

            AtomOpKind::ElementwiseMul => BackwardRatio {
                ratio: 2.0, // dX = Y×grad, dY = X×grad
                formula: "dX = dY×Y, dY = dY×X → 2×",
                confidence: 1.0,
            },

            // ── Pooling ───────────────────────────────────────────────────────────
            AtomOpKind::MaxPool2d => BackwardRatio {
                ratio: 1.5, // mask lookup + scatter
                formula: "max_mask backward: 1×forward approx (sparse)",
                confidence: 0.80,
            },

            AtomOpKind::AvgPool2d | AtomOpKind::GlobalAvgPool => BackwardRatio {
                ratio: 1.0, // broadcast gradient
                formula: "broadcast gradient / n_elements → 1×",
                confidence: 1.0,
            },

            // ── Dropout ───────────────────────────────────────────────────────────
            // Dropout: precomputed mask → 1×
            AtomOpKind::Dropout => BackwardRatio {
                ratio: 1.0,
                formula: "Apply stored mask → 1×",
                confidence: 1.0,
            },

            // ── Residual ───────────────────────────────────────────────────────────
            AtomOpKind::Residual => BackwardRatio {
                ratio: 1.0,
                formula: "Gradient copy to both branches → 1×",
                confidence: 1.0,
            },

            // ── Unknown ───────────────────────────────────────────────────────────
            AtomOpKind::Unknown => BackwardRatio {
                ratio: 2.0, // Conservative default
                formula: "Default conservative estimate",
                confidence: 0.60,
            },
        }
    }
    
    /// Calculate total backward FLOPs from forward FLOPs
    pub fn backward_flops(op: &AtomOpKind, forward_flops: u64) -> u64 {
        let ratio = Self::ratio_for_op(op);
        (forward_flops as f64 * ratio.ratio) as u64
    }
    
    /// Get all ratios as a map
    pub fn all_ratios() -> std::collections::HashMap<AtomOpKind, BackwardRatio> {
        use AtomOpKind::*;
        let ops = [
            MatMul, Linear, BatchedMatMul,
            MultiHeadAttention, GroupedQueryAttention, FlashAttention, AttentionScores, AttentionOutput,
            LayerNorm, RmsNorm, BatchNorm,
            Softmax, Relu, Gelu, Silu, Swish,
            Conv2d, DepthwiseConv2d,
            TokenEmbedding, PositionalEmbedding,
            MoeExpertGroup, MoeRouter,
            SsmStateUpdate,
            ElementwiseAdd, ElementwiseMul,
            MaxPool2d, AvgPool2d, GlobalAvgPool,
            Dropout, Residual, Unknown,
        ];
        
        ops.iter().map(|op| (*op, Self::ratio_for_op(op))).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_backward() {
        let ratio = BackwardRatios::ratio_for_op(&AtomOpKind::MatMul);
        assert_eq!(ratio.ratio, 2.0);
        assert_eq!(ratio.confidence, 1.0);
    }

    #[test]
    fn test_attention_backward() {
        let ratio = BackwardRatios::ratio_for_op(&AtomOpKind::MultiHeadAttention);
        assert_eq!(ratio.ratio, 2.5);
        assert!(ratio.confidence >= 0.90);
    }

    #[test]
    fn test_layernorm_backward() {
        let ratio = BackwardRatios::ratio_for_op(&AtomOpKind::LayerNorm);
        assert_eq!(ratio.ratio, 4.0);
    }

    #[test]
    fn test_backward_flops_calculation() {
        let forward = 1000u64;
        let backward = BackwardRatios::backward_flops(&AtomOpKind::MatMul, forward);
        assert_eq!(backward, 2000);
        
        let backward = BackwardRatios::backward_flops(&AtomOpKind::MultiHeadAttention, forward);
        assert_eq!(backward, 2500);
    }

    #[test]
    fn test_all_ratios() {
        let ratios = BackwardRatios::all_ratios();
        assert!(!ratios.is_empty());
        assert!(ratios.contains_key(&AtomOpKind::MatMul));
        assert!(ratios.contains_key(&AtomOpKind::FlashAttention));
    }
}
