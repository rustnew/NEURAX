//! Operator IR pass

use super::{AtomOp, OpType, OperatorIR, OperatorMetrics};
use crate::architecture::ArchitectureIR;
use crate::error::OperatorError;
use crate::tensor::Shape;
use crate::tensor::TensorIR;
use crate::traits::IrPass;
use crate::NeuraxContext;
use neurax_parser::LayerType;

/// Reads a `usize` out of a `global_params.extra` map, falling back when the
/// key is absent or not a plain non-negative integer — the same map
/// `GlobalResolutionContext` reads `num_nodes`/`num_edges` from, so a GNN
/// design's real graph size (if the client supplied one) reaches its FLOPs
/// the same way it reaches everything else this map backs. `pub(crate)`
/// so `tensor::shape_inference` reads the same `num_nodes` this pass does,
/// instead of a second, independently-drifting lookup.
pub(crate) fn extra_usize(
    extra: &std::collections::HashMap<String, serde_json::Value>,
    key: &str,
    default: usize,
) -> usize {
    extra
        .get(key)
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(default)
}

/// Operator pass implementation
pub struct OperatorPass;

impl IrPass for OperatorPass {
    type Input = (TensorIR, ArchitectureIR);
    type Output = OperatorIR;
    type Metrics = OperatorMetrics;
    type PassError = OperatorError;

    fn name(&self) -> &'static str {
        "OperatorIR"
    }

    fn build(
        &self,
        input: &Self::Input,
        ctx: &NeuraxContext,
    ) -> Result<Self::Output, Self::PassError> {
        let (_tensor_ir, arch_ir) = input;
        let mut op_ir = OperatorIR::default();
        let batch = ctx.config.training.batch_size;
        let seq = ctx
            .config
            .training
            .sequence_length
            .or(arch_ir.global_params.sequence_length)
            .unwrap_or(512);
        let dtype = &ctx.config.training.precision;

        for layer in &arch_ir.layers {
            let mut layer_ops = Vec::new();

            let mut ops = decompose_layer_to_ops(layer, batch, seq, dtype, ctx);

            // The real per-kind, per-role scale — the same one
            // `scaled_total_parameters` and the memory pass's liveness
            // intervals already use, looked up by matching `id` against the
            // parsed config's own layer list (`repeat_scale_for` reads
            // `encoder_decoder_role`/`repeat_fraction` off that layer, which
            // `ArchitectureIR`'s own `LayerDef` doesn't carry).
            //
            // This replaces a second, independent implementation that used
            // to live here: it counted only `Attention`/`Mlp` (ignoring
            // `CrossAttention`, `MambaBlock`, MoE and every other repeatable
            // kind — each of those got a silent, permanent 1.0), applied one
            // blended scale to both Attention and Mlp instead of each
            // kind's own count the way `repeat_scale_for` does, and had no
            // way to know about `encoder_decoder_role`/`repeat_fraction`.
            // FLOPs for a T5 import's decoder cross-attention, and for
            // every MoE model's routed-expert layers, were understated as a
            // result — parameter counts were already correct through
            // `scaled_total_parameters`, only FLOPs used this second path.
            let scale = ctx
                .config
                .model
                .layers
                .iter()
                .find(|l| l.id == layer.id)
                .map(|l| crate::architecture::repeat_scale_for(&ctx.config, l))
                .unwrap_or(1.0);
            if scale > 1.0 {
                for op in &mut ops {
                    op.flops *= scale;
                }
            }

            for op in ops {
                let op_id = op_ir.operations.len();
                layer_ops.push(op_id);
                op_ir.operations.push(op);
            }

            op_ir.layer_ops.insert(layer.id.clone(), layer_ops);
        }

        Ok(op_ir)
    }

    fn compute_metrics(
        &self,
        output: &mut Self::Output,
        _ctx: &NeuraxContext,
    ) -> Result<Self::Metrics, Self::PassError> {
        let mut metrics = OperatorMetrics {
            total_op_count: output.operations.len(),
            ..Default::default()
        };

        for op in &output.operations {
            // FLOPs per layer
            let entry = metrics
                .flops_per_layer
                .entry(op.layer_id.clone())
                .or_insert(0.0);
            *entry += op.flops;
            metrics.total_flops_approx += op.flops;

            // Op type distribution
            let type_str = op.op_type.as_str().to_string();
            *metrics.op_type_distribution.entry(type_str).or_insert(0) += 1;

            if op.is_custom {
                metrics.custom_op_count += 1;
            }
        }

        output.metrics = metrics.clone();
        output.metrics_done = true;
        Ok(metrics)
    }

    fn validate(
        &self,
        _output: &Self::Output,
        metrics: &Self::Metrics,
    ) -> Result<(), Self::PassError> {
        if metrics.total_op_count == 0 {
            return Err(OperatorError::UnknownOperator(
                "No operations generated".to_string(),
            ));
        }
        Ok(())
    }
}

/// Decompose a layer into atomic operations
fn decompose_layer_to_ops(
    layer: &crate::architecture::LayerDef,
    batch: usize,
    seq: usize,
    dtype: &str,
    ctx: &NeuraxContext,
) -> Vec<AtomOp> {
    let mut ops = Vec::new();

    // OpSpec-IR (`neurax-opspec`): a migrated layer type's FLOPs formula
    // lives in exactly one place, alongside its params formula. Only
    // `Custom` (permanently) falls through to the match below unchanged.
    // `OpType` and activation-memory bytes are decided here, not by
    // `neurax-opspec` — it owns no `neurax-ir` concept, see its own docs.
    if let Some(spec) = neurax_opspec::op_spec(layer.layer_type) {
        let parser_layer = to_parser_layer(layer);
        let flops_ctx = neurax_opspec::FlopsContext {
            global_params: &ctx.config.model.global_params,
            image_channels: ctx.config.data.image_channels,
            image_height: ctx.config.data.image_height,
            image_width: ctx.config.data.image_width,
        };
        let flops = (spec.flops_fn)(&parser_layer, batch, seq, &flops_ctx);
        let activation_memory = spec
            .activation_memory_fn
            .map(|f| f(&parser_layer, batch, seq, dtype))
            .unwrap_or(0);
        ops.push(AtomOp {
            id: ops.len(),
            op_type: op_type_for(&parser_layer),
            layer_id: layer.id.clone(),
            input_shapes: vec![],
            output_shape: crate::tensor::Shape::default(),
            flops,
            param_count: layer.param_count,
            activation_memory,
            is_custom: false,
        });
        // `Attention` is the one migrated type that still decomposes into
        // two ops: RoPE (applied to Q/K) is a second, real atomic operation
        // — `neurax_opspec::attention_flops_fn` reports the main op's FLOPs
        // only, precisely so this second op keeps existing. See
        // `attention_rope_flops`'s own doc for why (kernel_launch_count
        // counts real ops and feeds a real latency-overhead model).
        if layer.layer_type == LayerType::Attention {
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Mul, // RoPE is element-wise rotation
                layer_id: layer.id.clone(),
                input_shapes: vec![],
                output_shape: crate::tensor::Shape::default(),
                flops: neurax_opspec::attention_rope_flops(&parser_layer, batch, seq),
                param_count: 0,
                activation_memory: 0,
                is_custom: false,
            });
        }
        return ops;
    }

    match layer.layer_type {
        // Migrated to OpSpec-IR (`neurax-opspec`) — the early-return above
        // already returns for every one of these before the match runs, so
        // reaching this arm would mean `op_spec()` and this match disagree
        // about which types are migrated. Every real `LayerType` except
        // `Custom` is migrated.
        LayerType::Embedding
        | LayerType::Attention
        | LayerType::Mlp
        | LayerType::Conv
        | LayerType::Dense
        | LayerType::LoraLinear
        | LayerType::DoraLinear
        | LayerType::Normalization
        | LayerType::Pooling
        | LayerType::MoE
        | LayerType::MoeRouter
        | LayerType::MoeCombine
        | LayerType::MoeSharedExpert
        | LayerType::ResidualBlock
        | LayerType::ResnetBottleneck
        | LayerType::Mbconv
        | LayerType::Inception
        | LayerType::DenseBlock
        | LayerType::ConvnextBlock
        | LayerType::ShuffleUnit
        | LayerType::C2f
        | LayerType::Detection
        | LayerType::Transition
        | LayerType::MambaBlock
        | LayerType::S4Block
        | LayerType::StateSpace
        | LayerType::H3Block
        | LayerType::RwkvBlock
        | LayerType::RetentionBlock
        | LayerType::GeneratorBlock
        | LayerType::DiscriminatorBlock
        | LayerType::StyleMod
        | LayerType::AdaIN
        | LayerType::MinibatchStd
        | LayerType::PixelNorm
        | LayerType::SelfAttention
        | LayerType::SpectralNorm
        | LayerType::ProgressiveBlock
        | LayerType::LstmBlock
        | LayerType::GruBlock
        | LayerType::RnnCell
        | LayerType::Bidirectional
        | LayerType::EncoderBlock
        | LayerType::DecoderBlock
        | LayerType::UnetBlock
        | LayerType::TimeEmbedding
        | LayerType::CrossAttention
        | LayerType::DownBlock
        | LayerType::UpBlock
        | LayerType::MidBlock
        | LayerType::ResnetBlock
        | LayerType::TimestepBlock
        | LayerType::ConditionBlock
        | LayerType::NoisePredictor
        | LayerType::VaeEncoder
        | LayerType::VaeDecoder
        | LayerType::GraphConvNet
        | LayerType::MessagePassing
        | LayerType::GraphAttentionNet
        | LayerType::RgcnConv => unreachable!(
            "{:?} is registered in OpSpec-IR (neurax-opspec); the early return above must have handled it",
            layer.layer_type
        ),
        // Custom layer - use custom equations if provided
        LayerType::Custom => {
            let hidden = layer.params.hidden_size.unwrap_or(512);
            // Use custom equations or default FLOPs
            // Evaluate the author's FLOP equation.
            //
            // This used to `parse::<f64>()` the equation string, which succeeds
            // only for a bare literal: any real formula such as "2 * B * S * H"
            // failed to parse and the block was silently costed at zero FLOPs.
            let equation = layer
                .custom_equations
                .as_ref()
                .and_then(|eqs| eqs.flops_forward.as_ref());
            let flops = match equation {
                Some(equation) => {
                    let evaluated = crate::architecture::evaluate_custom_equation_with(
                        equation,
                        &to_parser_layer(layer),
                        batch,
                        seq,
                    );
                    if evaluated.is_none() {
                        // Report the broken formula. Otherwise the block costs
                        // nothing and the failure surfaces much later as an
                        // unexplained "zero FLOPs" error.
                        ctx.add_diagnostic(crate::Diagnostic {
                            severity: crate::Severity::Critical,
                            category: crate::DiagnosticCategory::CustomLayerFallback,
                            code: crate::DiagnosticCode::E003,
                            message: format!(
                                "Custom FLOP equation for layer '{}' could not be evaluated: `{}`",
                                layer.id, equation
                            ),
                            layer_id: Some(layer.id.clone()),
                            suggestion: Some(
                                "Use the documented variables (B, S, H, D, I, V, N) and operators \
                                 supported by the expression evaluator."
                                    .to_string(),
                            ),
                            precision_impact: 1.0,
                        });
                    }
                    evaluated.unwrap_or(0.0)
                }
                None => {
                    ctx.add_diagnostic(crate::Diagnostic {
                        severity: crate::Severity::Warning,
                        category: crate::DiagnosticCategory::CustomLayerFallback,
                        code: crate::DiagnosticCode::W001,
                        message: format!(
                            "Custom layer '{}' has no FLOP equation; it is costed as free.",
                            layer.id
                        ),
                        layer_id: Some(layer.id.clone()),
                        suggestion: Some(
                            "Add `custom_equations.flops_forward` so the block is accounted for."
                                .to_string(),
                        ),
                        precision_impact: 0.5,
                    });
                    0.0
                }
            };
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Custom,
                layer_id: layer.id.clone(),
                input_shapes: vec![Shape::known(vec![batch, seq, hidden])],
                output_shape: Shape::known(vec![batch, seq, hidden]),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: true,
            });
        }
    }

    ops
}

/// Total FLOPs for one layer — the sum of every atomic operation
/// `decompose_layer_to_ops` breaks it into.
///
/// Exposed publicly so other consumers of the real per-layer-type formulas
/// this pass already applies don't have to re-derive their own, separate
/// approximation. A since-deleted MLIR backend did exactly that: it counted
/// attention as `4 × hidden²` FLOPs and left every CNN, SSM, GAN, RNN and
/// diffusion layer type silently uncounted, simply because nothing connected
/// it to this pass.
pub fn layer_flops(
    layer: &crate::architecture::LayerDef,
    batch: usize,
    seq: usize,
    dtype: &str,
    ctx: &NeuraxContext,
) -> f64 {
    decompose_layer_to_ops(layer, batch, seq, dtype, ctx)
        .iter()
        .map(|op| op.flops)
        .sum()
}

/// Chooses the `OpType` a migrated layer's single `AtomOp` should carry — a
/// decision this pass makes, not `neurax-opspec` (which owns no `neurax-ir`
/// concept). Matches exactly what each type's own pre-migration arm set,
/// with one runtime branch (`Normalization`'s RMS vs LayerNorm) that can't
/// be a static per-`LayerType` table entry.
fn op_type_for(layer: &neurax_parser::Layer) -> OpType {
    use neurax_parser::LayerType::*;
    match layer.layer_type {
        Embedding => OpType::Embedding,
        Attention | SelfAttention | CrossAttention => OpType::Attention,
        Normalization => {
            if layer.params.activation.as_deref() == Some("rms") {
                OpType::RMSNorm
            } else {
                OpType::LayerNorm
            }
        }
        Pooling => OpType::Pooling(super::PoolingType::Max),
        MoE | MoeRouter | MoeCombine | MoeSharedExpert => OpType::MoE,
        Dense | LoraLinear | DoraLinear => OpType::MatMul,
        Mlp | MambaBlock | S4Block | StateSpace | H3Block | RwkvBlock | RetentionBlock
        | StyleMod | AdaIN | PixelNorm | MinibatchStd | SpectralNorm | LstmBlock | GruBlock
        | RnnCell | Bidirectional | EncoderBlock | DecoderBlock | TimeEmbedding | TimestepBlock
        | ConditionBlock | GraphConvNet | MessagePassing | GraphAttentionNet | RgcnConv => {
            OpType::Linear
        }
        // CNN blocks, GAN conv-like blocks, diffusion conv/U-Net blocks, and
        // `Conv` itself — every one of these emitted `Conv2D` before
        // migrating.
        _ => OpType::Conv2D,
    }
}

/// Adapt an IR layer back to the parser type the shared helpers expect.
fn to_parser_layer(layer: &crate::architecture::LayerDef) -> neurax_parser::Layer {
    neurax_parser::Layer {
        id: layer.id.clone(),
        layer_type: layer.layer_type,
        input_shape: layer.input_shape.clone(),
        output_shape: layer.output_shape.clone(),
        params: layer.params.clone(),
        custom_equations: layer.custom_equations.clone(),
    }
}
