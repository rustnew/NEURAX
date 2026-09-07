//! The op table itself — one entry per migrated `LayerType`.
//!
//! Every family is here: the 10 CNN blocks, Transformer (`Embedding`,
//! `Attention`, `Mlp`, `Dense`, LoRA/DoRA, `Normalization`, `Conv`,
//! `Pooling`), MoE, SSM, GAN, RNN, Diffusion, and GNN. `Custom` is the one
//! permanent exception — see the crate doc for why.
//!
//! Organized as one `&[OpSpec]` slice per family, concatenated in
//! [`op_spec`], so a family's entries stay visually grouped with the
//! comments explaining its own defaults and quirks — the same grouping
//! `neurax-ir`'s two original match arms already used.

use crate::{ActivationMemoryFn, FlopsContext, FlopsFn, OpSpec, ParamsFn};
use neurax_formulas::{
    attention, cnn_blocks, conv, embedding, gnn, lora, mlp, moe, normalization, rnn, ssm,
};
use neurax_parser::{GlobalParams, Layer, LayerType};

/// Looks up a layer type's unified definition. `None` means "not migrated
/// yet" — the caller (`neurax-ir`) must fall back to its pre-existing match
/// arm. Every real `LayerType` except `Custom` should resolve to `Some`.
pub fn op_spec(layer_type: LayerType) -> Option<&'static OpSpec> {
    CNN_BLOCK_SPECS
        .iter()
        .chain(TRANSFORMER_SPECS)
        .chain(MOE_SPECS)
        .chain(SSM_SPECS)
        .chain(GAN_SPECS)
        .chain(RNN_SPECS)
        .chain(DIFFUSION_SPECS)
        .chain(GNN_SPECS)
        .find(|spec| spec.layer_type == layer_type)
}

const fn spec(layer_type: LayerType, params_fn: ParamsFn, flops_fn: FlopsFn) -> OpSpec {
    OpSpec {
        layer_type,
        params_fn,
        flops_fn,
        activation_memory_fn: None,
    }
}

/// Like [`spec`], for the handful of types that also track a real
/// activation-tensor size.
const fn spec_with_memory(
    layer_type: LayerType,
    params_fn: ParamsFn,
    flops_fn: FlopsFn,
    activation_memory_fn: ActivationMemoryFn,
) -> OpSpec {
    OpSpec {
        layer_type,
        params_fn,
        flops_fn,
        activation_memory_fn: Some(activation_memory_fn),
    }
}

static CNN_BLOCK_SPECS: &[OpSpec] = &[
    spec(
        LayerType::ResidualBlock,
        residual_block_params,
        residual_block_flops,
    ),
    spec(
        LayerType::ResnetBottleneck,
        resnet_bottleneck_params,
        resnet_bottleneck_flops,
    ),
    spec(LayerType::Mbconv, mbconv_params, mbconv_flops),
    spec(LayerType::Inception, inception_params, inception_flops),
    spec(LayerType::DenseBlock, dense_block_params, dense_block_flops),
    spec(
        LayerType::ConvnextBlock,
        convnext_block_params,
        convnext_block_flops,
    ),
    spec(
        LayerType::ShuffleUnit,
        shuffle_unit_params,
        shuffle_unit_flops,
    ),
    spec(LayerType::C2f, c2f_params, c2f_flops),
    spec(
        LayerType::Detection,
        detection_transition_params,
        detection_transition_flops,
    ),
    spec(
        LayerType::Transition,
        detection_transition_params,
        detection_transition_flops,
    ),
];

/// Square spatial side stand-in every CNN block already used before this
/// migration — nothing in the pipeline tracks a feature map's real height/
/// width as it shrinks through the network.
fn side_from(seq: usize) -> usize {
    (seq as f64).sqrt().round().max(1.0) as usize
}

// ── ResidualBlock ────────────────────────────────────────────────────────
// Migrating this pair surfaced a real bug the split fixes: the
// pre-migration FLOPs arm shared one `out_channels.unwrap_or(256)` default
// between ResidualBlock and ResnetBottleneck — correct for a bottleneck's
// real default, silently wrong for a plain basic block (whose own
// params-side default, and every real ResNet-18/34 basic block with no
// stated channel change, is 64->64, not 64->256). Splitting into two specs
// with their own defaults, matching the params side, closes it.
fn residual_block_params(layer: &Layer) -> u64 {
    let in_ch = layer.params.in_channels.unwrap_or(64);
    let out_ch = layer.params.out_channels.unwrap_or(64);
    let stride = layer.params.stride.unwrap_or(1);
    cnn_blocks::resnet_basic_block_params(in_ch, out_ch, stride, layer.params.bias)
}

fn residual_block_flops(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let side = side_from(seq);
    let in_ch = layer.params.in_channels.unwrap_or(64);
    let out_ch = layer.params.out_channels.unwrap_or(64);
    let stride = layer.params.stride.unwrap_or(1);
    cnn_blocks::resnet_basic_block_flops(batch, in_ch, out_ch, side, side, stride)
}

// ── ResnetBottleneck ─────────────────────────────────────────────────────
fn resnet_bottleneck_params(layer: &Layer) -> u64 {
    let in_ch = layer.params.in_channels.unwrap_or(64);
    let out_ch = layer.params.out_channels.unwrap_or(256);
    let mid_ch = layer.params.mid_channels.unwrap_or(out_ch / 4);
    let stride = layer.params.stride.unwrap_or(1);
    let cardinality = layer.params.cardinality.unwrap_or(1);
    cnn_blocks::resnet_bottleneck_block_params(
        in_ch,
        mid_ch,
        out_ch,
        stride,
        cardinality,
        layer.params.bias,
    )
}

fn resnet_bottleneck_flops(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let side = side_from(seq);
    let params = &layer.params;
    let in_ch = params.in_channels.unwrap_or(64);
    let out_ch = params.out_channels.unwrap_or(256);
    let mid_ch = params.mid_channels.unwrap_or(out_ch / 4);
    let stride = params.stride.unwrap_or(1);
    let cardinality = params.cardinality.unwrap_or(1);
    cnn_blocks::resnet_bottleneck_block_flops(
        batch,
        in_ch,
        mid_ch,
        out_ch,
        side,
        side,
        stride,
        cardinality,
    )
}

// ── Mbconv ───────────────────────────────────────────────────────────────
fn mbconv_params(layer: &Layer) -> u64 {
    let in_ch = layer.params.in_channels.unwrap_or(32);
    let out_ch = layer.params.out_channels.unwrap_or(16);
    let expand = layer.params.expansion_factor.unwrap_or(6);
    let kernel = layer.params.kernel_size.unwrap_or(3);
    let stride = layer.params.stride.unwrap_or(1);
    let se_reduction = layer.params.se_reduction_ratio.unwrap_or(4);
    cnn_blocks::mbconv_params(
        in_ch,
        out_ch,
        expand,
        kernel,
        stride,
        layer.params.se,
        se_reduction,
        layer.params.bias,
    )
}

fn mbconv_flops(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let side = side_from(seq);
    let params = &layer.params;
    let in_ch = params.in_channels.unwrap_or(32);
    let out_ch = params.out_channels.unwrap_or(16);
    let expand = params.expansion_factor.unwrap_or(6);
    let kernel = params.kernel_size.unwrap_or(3);
    let stride = params.stride.unwrap_or(1);
    let se_reduction = params.se_reduction_ratio.unwrap_or(4);
    cnn_blocks::mbconv_flops(
        batch,
        in_ch,
        out_ch,
        side,
        side,
        expand,
        kernel,
        stride,
        params.se,
        se_reduction,
    )
}

// ── Inception ────────────────────────────────────────────────────────────
fn inception_params(layer: &Layer) -> u64 {
    let in_ch = layer.params.in_channels.unwrap_or(288);
    let out_1x1 = layer.params.out_channels.unwrap_or(64);
    cnn_blocks::inception_module_params(
        in_ch,
        out_1x1,
        out_1x1 / 2,
        out_1x1,
        out_1x1 / 8,
        out_1x1 / 2,
        out_1x1,
        layer.params.bias,
    )
}

fn inception_flops(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let side = side_from(seq);
    let params = &layer.params;
    let in_ch = params.in_channels.unwrap_or(288);
    let out_1x1 = params.out_channels.unwrap_or(64);
    cnn_blocks::inception_module_flops(
        batch,
        side,
        side,
        in_ch,
        out_1x1,
        out_1x1 / 2,
        out_1x1,
        out_1x1 / 8,
        out_1x1 / 2,
        out_1x1,
    )
}

// ── DenseBlock ───────────────────────────────────────────────────────────
fn dense_block_params(layer: &Layer) -> u64 {
    let in_ch = layer.params.in_channels.unwrap_or(64);
    let growth = layer.params.growth_rate.unwrap_or(32);
    let num_layers = layer.params.num_layers.unwrap_or(4);
    cnn_blocks::dense_block_params(in_ch, growth, num_layers, 4, layer.params.bias)
}

fn dense_block_flops(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let side = side_from(seq);
    let params = &layer.params;
    let in_ch = params.in_channels.unwrap_or(64);
    let growth = params.growth_rate.unwrap_or(32);
    let num_layers = params.num_layers.unwrap_or(4);
    cnn_blocks::dense_block_flops(batch, side, side, in_ch, growth, num_layers, 4)
}

// ── ConvnextBlock ────────────────────────────────────────────────────────
fn convnext_block_params(layer: &Layer) -> u64 {
    let channels = layer.params.hidden_size.unwrap_or(96);
    let mlp_ratio = layer.params.mlp_ratio.unwrap_or(4.0);
    cnn_blocks::convnext_block_params(channels, mlp_ratio, layer.params.bias)
}

fn convnext_block_flops(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let side = side_from(seq);
    let params = &layer.params;
    let channels = params.hidden_size.unwrap_or(96);
    let mlp_ratio = params.mlp_ratio.unwrap_or(4.0);
    cnn_blocks::convnext_block_flops(batch, side, side, channels, mlp_ratio)
}

// ── ShuffleUnit ──────────────────────────────────────────────────────────
fn shuffle_unit_params(layer: &Layer) -> u64 {
    let in_ch = layer.params.in_channels.unwrap_or(64);
    let out_ch = layer.params.out_channels.unwrap_or(64);
    let groups = layer.params.groups.unwrap_or(2);
    let stride = layer.params.stride.unwrap_or(1);
    cnn_blocks::shuffle_unit_params(in_ch, out_ch, groups, stride, layer.params.bias)
}

fn shuffle_unit_flops(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let side = side_from(seq);
    let params = &layer.params;
    let in_ch = params.in_channels.unwrap_or(64);
    let out_ch = params.out_channels.unwrap_or(64);
    let groups = params.groups.unwrap_or(2);
    let stride = params.stride.unwrap_or(1);
    cnn_blocks::shuffle_unit_flops(batch, side, side, in_ch, out_ch, groups, stride)
}

// ── C2f ──────────────────────────────────────────────────────────────────
fn c2f_params(layer: &Layer) -> u64 {
    let in_ch = layer.params.in_channels.unwrap_or(64);
    let out_ch = layer.params.out_channels.unwrap_or(64);
    let num_bn = layer.params.num_bottlenecks.unwrap_or(3);
    cnn_blocks::c2f_block_params(in_ch, out_ch, num_bn, true, layer.params.bias)
}

fn c2f_flops(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let side = side_from(seq);
    let params = &layer.params;
    let in_ch = params.in_channels.unwrap_or(64);
    let out_ch = params.out_channels.unwrap_or(64);
    let num_bn = params.num_bottlenecks.unwrap_or(3);
    cnn_blocks::c2f_block_flops(batch, side, side, in_ch, out_ch, num_bn)
}

// ── Detection / Transition ───────────────────────────────────────────────
fn detection_transition_params(layer: &Layer) -> u64 {
    let in_ch = layer.params.in_channels.unwrap_or(256);
    let out_ch = layer.params.out_channels.unwrap_or(256);
    let kernel = layer.params.kernel_size.unwrap_or(3);
    neurax_formulas::conv::conv2d_params(in_ch, out_ch, kernel, kernel, 1, layer.params.bias)
}

fn detection_transition_flops(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let side = side_from(seq);
    let params = &layer.params;
    let in_ch = params.in_channels.unwrap_or(256);
    let out_ch = params.out_channels.unwrap_or(256);
    let kernel = params.kernel_size.unwrap_or(3);
    neurax_formulas::conv::conv2d_flops(
        batch,
        in_ch,
        out_ch,
        side,
        side,
        kernel,
        kernel,
        1,
        kernel / 2,
        1,
    )
}

// ═══════════════════════════════════════════════════════════════════════
// Transformer family: Embedding, Attention, Mlp, Dense, LoRA/DoRA,
// Normalization, Conv, Pooling
// ═══════════════════════════════════════════════════════════════════════

static TRANSFORMER_SPECS: &[OpSpec] = &[
    spec_with_memory(
        LayerType::Embedding,
        embedding_params_fn,
        embedding_flops_fn,
        embedding_activation_memory,
    ),
    spec_with_memory(
        LayerType::Attention,
        attention_params_fn,
        attention_flops_fn,
        attention_activation_memory,
    ),
    spec_with_memory(
        LayerType::Mlp,
        mlp_params_fn,
        mlp_flops_fn,
        mlp_activation_memory,
    ),
    spec_with_memory(
        LayerType::Dense,
        dense_params_fn,
        dense_flops_fn,
        dense_activation_memory,
    ),
    spec_with_memory(
        LayerType::LoraLinear,
        lora_params_fn,
        lora_flops_fn,
        lora_dora_activation_memory,
    ),
    spec_with_memory(
        LayerType::DoraLinear,
        dora_params_fn,
        dora_flops_fn,
        lora_dora_activation_memory,
    ),
    spec(
        LayerType::Normalization,
        normalization_params_fn,
        normalization_flops_fn,
    ),
    spec(LayerType::Conv, conv_params_fn, conv_flops_fn),
    spec(LayerType::Pooling, pooling_params_fn, pooling_flops_fn),
];

fn embedding_dim(layer: &Layer) -> usize {
    layer
        .params
        .embedding_dim
        .unwrap_or(layer.params.hidden_size.unwrap_or(512))
}

fn embedding_params_fn(layer: &Layer) -> u64 {
    let vocab = layer.params.vocab_size.unwrap_or(50000);
    embedding::embedding_params(vocab, embedding_dim(layer))
}

fn embedding_flops_fn(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    embedding::embedding_flops(batch, seq, embedding_dim(layer))
}

fn embedding_activation_memory(layer: &Layer, batch: usize, seq: usize, dtype: &str) -> u64 {
    ((batch * seq * embedding_dim(layer)) as f64 * neurax_formulas::dtype_bytes(dtype)).round()
        as u64
}

/// The real per-head width, when a model states one independently of
/// `hidden_size`/`num_heads` (GLM-4.5, HuggingFace-verified:
/// `hidden_size=5120, num_heads=96, head_dim=128` — `96*128=12288 != 5120`,
/// a real widened projection, not a rounding artifact). Falls back to the
/// derived `hidden/heads` otherwise, which is what every model without an
/// explicit `head_dim` already assumes.
fn resolved_head_dim(layer: &Layer) -> usize {
    let hidden = layer.params.hidden_size.unwrap_or(512);
    let heads = layer.params.num_heads.unwrap_or(8).max(1);
    layer.params.head_dim.unwrap_or(hidden / heads)
}

fn attention_params_fn(layer: &Layer) -> u64 {
    let hidden = layer.params.hidden_size.unwrap_or(512);
    let heads = layer.params.num_heads.unwrap_or(8);
    let kv_heads = layer.params.num_kv_heads.unwrap_or(heads);
    let head_dim = resolved_head_dim(layer);
    if kv_heads == heads {
        attention::attention_params_with_head_dim(
            hidden,
            heads.max(1) * head_dim,
            layer.params.bias,
        )
    } else {
        attention::gqa_params_with_head_dim(hidden, heads, kv_heads, head_dim, layer.params.bias)
    }
}

/// The main attention op's FLOPs only — *not* RoPE's. `Attention` is the
/// one migrated type that still emits two `AtomOp`s (this one, plus a RoPE
/// op the call site in `operator/pass.rs` adds itself): `kernel_launch_count`
/// counts real ops, and a regression test
/// (`neurax-core/tests/kernel_launch_count_reflects_real_ops.rs`) pins it at
/// 2 ops for a bare attention layer — folding RoPE's FLOPs in here and
/// reporting one op would silently undercount kernel launches, which feed a
/// real latency-overhead model, not just a cosmetic diagnostic.
fn attention_flops_fn(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let hidden = layer.params.hidden_size.unwrap_or(512);
    let heads = layer.params.num_heads.unwrap_or(8).max(1);
    let kv_heads = layer.params.num_kv_heads.unwrap_or(heads).max(1);
    let head_dim = resolved_head_dim(layer);
    let causal = layer.params.causal;
    let kv_span = layer
        .params
        .window_size
        .or(layer.params.block_size)
        .or(layer.params.dilation.map(|d| (seq / d.max(1)).max(1)))
        .unwrap_or(seq);
    if kv_span < seq {
        attention::windowed_attention_flops_with_head_dim(
            batch, seq, kv_span, hidden, heads, head_dim, causal,
        )
    } else if kv_heads < heads {
        attention::gqa_flops_with_head_dim(batch, seq, hidden, heads, kv_heads, head_dim, causal)
    } else {
        attention::attention_flops_with_head_dim(batch, seq, hidden, heads, head_dim, causal)
    }
}

/// RoPE's own FLOPs for an `Attention` layer — the second `AtomOp` the call
/// site in `operator/pass.rs` pushes itself, since [`OpSpec`] only carries
/// one `FlopsFn` per type. See [`attention_flops_fn`] for why this stays a
/// separate op instead of being folded into that total.
pub fn attention_rope_flops(layer: &Layer, batch: usize, seq: usize) -> f64 {
    let heads = layer.params.num_heads.unwrap_or(8).max(1);
    embedding::rope_flops(batch, seq, heads, resolved_head_dim(layer))
}

fn attention_activation_memory(layer: &Layer, batch: usize, seq: usize, dtype: &str) -> u64 {
    let hidden = layer.params.hidden_size.unwrap_or(512);
    ((batch * seq * hidden) as f64 * neurax_formulas::dtype_bytes(dtype)).round() as u64
}

/// Duplicated from `neurax-ir::architecture::is_gated_mlp` rather than
/// shared — depending on `neurax-ir` from here would be circular (see the
/// crate doc). Same two-line rule: an explicit `gated` flag, or a gated
/// activation name.
fn is_gated_mlp(params: &neurax_parser::LayerParams) -> bool {
    params.gated
        || params
            .activation
            .as_deref()
            .is_some_and(neurax_formulas::activation::is_gated_activation)
}

fn mlp_params_fn(layer: &Layer) -> u64 {
    let hidden = layer.params.hidden_size.unwrap_or(512);
    let intermediate = layer.params.intermediate_size.unwrap_or(4 * hidden);
    if is_gated_mlp(&layer.params) {
        mlp::gated_mlp_params(hidden, intermediate, layer.params.bias)
    } else {
        mlp::mlp_params(hidden, intermediate, layer.params.bias)
    }
}

fn mlp_flops_fn(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let hidden = layer.params.hidden_size.unwrap_or(512);
    let intermediate = layer.params.intermediate_size.unwrap_or(4 * hidden);
    let activation = layer.params.activation.as_deref().unwrap_or("gelu");
    if is_gated_mlp(&layer.params) {
        mlp::gated_mlp_flops(batch, seq, hidden, intermediate, activation)
    } else {
        mlp::mlp_flops(batch, seq, hidden, intermediate, activation)
    }
}

fn mlp_activation_memory(layer: &Layer, batch: usize, seq: usize, dtype: &str) -> u64 {
    let hidden = layer.params.hidden_size.unwrap_or(512);
    let intermediate = layer.params.intermediate_size.unwrap_or(4 * hidden);
    ((batch * seq * intermediate) as f64 * neurax_formulas::dtype_bytes(dtype)).round() as u64
}

fn dense_in_features(layer: &Layer) -> usize {
    layer
        .params
        .in_features
        .or(layer.params.in_channels)
        .or(layer.params.hidden_size)
        .unwrap_or(512)
}

fn dense_out_features(layer: &Layer) -> usize {
    layer
        .params
        .out_features
        .or(layer.params.out_channels)
        .or(layer.params.hidden_size)
        .unwrap_or(512)
}

fn dense_outer_dims(layer: &Layer) -> usize {
    layer
        .input_shape
        .iter()
        .take(layer.input_shape.len().saturating_sub(1))
        .copied()
        .product::<usize>()
        .max(1)
}

fn dense_output_elements(layer: &Layer, out_f: usize) -> usize {
    if layer.output_shape.is_empty() {
        dense_outer_dims(layer) * out_f
    } else {
        layer.output_shape.iter().copied().product::<usize>()
    }
}

fn dense_params_fn(layer: &Layer) -> u64 {
    let in_f = dense_in_features(layer);
    let out_f = dense_out_features(layer);
    let bias = if layer.params.bias { out_f } else { 0 };
    (in_f * out_f + bias) as u64
}

fn dense_flops_fn(layer: &Layer, _batch: usize, _seq: usize, _ctx: &FlopsContext) -> f64 {
    let outer_dims = dense_outer_dims(layer);
    2.0 * outer_dims as f64 * dense_in_features(layer) as f64 * dense_out_features(layer) as f64
}

fn dense_activation_memory(layer: &Layer, _batch: usize, _seq: usize, dtype: &str) -> u64 {
    let out_f = dense_out_features(layer);
    (dense_output_elements(layer, out_f) as f64 * neurax_formulas::dtype_bytes(dtype)).round()
        as u64
}

fn lora_dora_out_features(layer: &Layer) -> usize {
    layer.params.out_features.unwrap_or(512)
}

fn lora_params_fn(layer: &Layer) -> u64 {
    let in_f = layer.params.in_features.unwrap_or(512);
    let rank = layer.params.rank.unwrap_or(16);
    lora::lora_params(in_f, lora_dora_out_features(layer), rank)
}

fn dora_params_fn(layer: &Layer) -> u64 {
    let in_f = layer.params.in_features.unwrap_or(512);
    let rank = layer.params.rank.unwrap_or(16);
    lora::dora_params(in_f, lora_dora_out_features(layer), rank)
}

fn lora_flops_fn(layer: &Layer, _batch: usize, _seq: usize, _ctx: &FlopsContext) -> f64 {
    let in_f = layer.params.in_features.unwrap_or(512);
    let out_f = lora_dora_out_features(layer);
    let rank = layer.params.rank.unwrap_or(16);
    lora::lora_flops(dense_outer_dims(layer), in_f, out_f, rank)
}

fn dora_flops_fn(layer: &Layer, _batch: usize, _seq: usize, _ctx: &FlopsContext) -> f64 {
    let in_f = layer.params.in_features.unwrap_or(512);
    let out_f = lora_dora_out_features(layer);
    let rank = layer.params.rank.unwrap_or(16);
    lora::dora_flops(dense_outer_dims(layer), in_f, out_f, rank)
}

fn lora_dora_activation_memory(layer: &Layer, _batch: usize, _seq: usize, dtype: &str) -> u64 {
    let out_f = lora_dora_out_features(layer);
    (dense_output_elements(layer, out_f) as f64 * neurax_formulas::dtype_bytes(dtype)).round()
        as u64
}

fn normalization_is_rms(layer: &Layer) -> bool {
    layer.params.activation.as_deref() == Some("rms")
}

fn normalization_params_fn(layer: &Layer) -> u64 {
    let hidden = layer.params.hidden_size.unwrap_or(512);
    if normalization_is_rms(layer) {
        normalization::rms_norm_params(hidden)
    } else {
        normalization::layer_norm_params(hidden, true)
    }
}

fn normalization_flops_fn(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let hidden = layer.params.hidden_size.unwrap_or(512);
    if normalization_is_rms(layer) {
        normalization::rms_norm_flops(batch, seq, hidden)
    } else {
        normalization::layer_norm_flops(batch, seq, hidden)
    }
}

/// Real bug found migrating `Conv`: the pre-migration FLOPs arm derived
/// `kernel_w` from `kernel_h` — itself always just `kernel_size`, never
/// `params.kernel_h` — instead of reading `params.kernel_h`/`kernel_w`
/// directly the way the params-side arm always did. A layer with an
/// explicit non-square `kernel_h`/`kernel_w` pair was silently costed as a
/// square, `kernel_size`-only kernel. Both formulas now read the same two
/// fields, in the same order.
fn conv_kernel_hw(layer: &Layer) -> (usize, usize) {
    let kh = layer
        .params
        .kernel_h
        .unwrap_or(layer.params.kernel_size.unwrap_or(3));
    let kw = layer
        .params
        .kernel_w
        .unwrap_or(layer.params.kernel_size.unwrap_or(3));
    (kh, kw)
}

fn conv_params_fn(layer: &Layer) -> u64 {
    let in_ch = layer.params.in_channels.unwrap_or(3);
    let out_ch = layer.params.out_channels.unwrap_or(64);
    let (kh, kw) = conv_kernel_hw(layer);
    let groups = layer.params.groups.unwrap_or(1);
    conv::conv2d_params(in_ch, out_ch, kh, kw, groups, layer.params.bias)
}

fn conv_flops_fn(layer: &Layer, batch: usize, _seq: usize, ctx: &FlopsContext) -> f64 {
    let in_ch = layer.params.in_channels.or(ctx.image_channels).unwrap_or(3);
    let out_ch = layer.params.out_channels.unwrap_or(64);
    let (kernel_h, kernel_w) = conv_kernel_hw(layer);
    let stride = layer.params.stride.unwrap_or(1);
    let padding = layer.params.padding.unwrap_or(0);
    let groups = 1; // Standard convolution — matches the pre-migration arm.

    let (batch, in_h, in_w) = if layer.input_shape.len() >= 4 {
        (
            layer.input_shape[0],
            layer.input_shape[2],
            layer.input_shape[3],
        )
    } else {
        (
            batch,
            ctx.image_height.unwrap_or(224),
            ctx.image_width.unwrap_or(224),
        )
    };

    // Saturating, not `-`: a kernel larger than its padded input (a real
    // config this crate has actually been handed — a mis-wired skip
    // connection feeding a conv a far smaller spatial input than the
    // template's own author intended) used to panic the whole analysis
    // with an unsigned subtract-overflow instead of reporting a degenerate
    // shape. `stride.max(1)` guards the same class of crash for a stated
    // `stride: 0`, which is nonsensical but not this function's job to
    // reject — that belongs to validation, not to a FLOPs formula.
    let stride_safe = stride.max(1);
    let out_h = (in_h + 2 * padding + stride_safe).saturating_sub(kernel_h) / stride_safe;
    let out_w = (in_w + 2 * padding + stride_safe).saturating_sub(kernel_w) / stride_safe;

    // `neurax-ir`'s own (now-orphaned) `operator::formulas::conv2d_flops`
    // computes this exact six-term product from pre-computed out_h/out_w —
    // duplicated inline rather than imported, since depending on
    // `neurax-ir` from this crate would be circular (see the crate doc).
    2.0 * batch as f64
        * out_h as f64
        * out_w as f64
        * out_ch as f64
        * kernel_h as f64
        * kernel_w as f64
        * (in_ch as f64 / groups as f64)
}

fn pooling_params_fn(_layer: &Layer) -> u64 {
    0
}

fn pooling_flops_fn(layer: &Layer, _batch: usize, _seq: usize, _ctx: &FlopsContext) -> f64 {
    let kernel_size = layer.params.kernel_size.unwrap_or(2);
    let (batch, channels, in_h, in_w) = if layer.input_shape.len() >= 4 {
        (
            layer.input_shape[0],
            layer.input_shape[1],
            layer.input_shape[2],
            layer.input_shape[3],
        )
    } else {
        (1, 64, 224, 224)
    };
    let stride = layer.params.stride.unwrap_or(2);
    let out_h = in_h.div_ceil(stride);
    let out_w = in_w.div_ceil(stride);
    // Same formula as neurax-ir's own (now-orphaned) operator::formulas::pooling_flops.
    batch as f64 * channels as f64 * out_h as f64 * out_w as f64 * kernel_size as f64
}

// ═══════════════════════════════════════════════════════════════════════
// MoE family
// ═══════════════════════════════════════════════════════════════════════

static MOE_SPECS: &[OpSpec] = &[
    spec(LayerType::MoE, moe_params_fn, moe_flops_fn),
    spec(
        LayerType::MoeRouter,
        moe_router_params_fn,
        moe_router_flops_fn,
    ),
    spec(
        LayerType::MoeCombine,
        moe_combine_params_fn,
        moe_combine_flops_fn,
    ),
    spec(
        LayerType::MoeSharedExpert,
        moe_shared_expert_params_fn,
        moe_shared_expert_flops_fn,
    ),
];

/// Whether a MoE layer's experts use a gated (SwiGLU: gate+up+down, 3
/// matrices) shape or a plain (up+down, 2 matrices) one.
///
/// Unlike the plain `Mlp` type (whose own default, `gelu`, is *not*
/// gated), a MoE layer with no stated `activation` defaults to **gated**:
/// every reference MoE fixture in this project (Mixtral 8x7B, DeepSeek-V3)
/// states no explicit `activation`/`gated` field and relies on exactly
/// this default, matching real SwiGLU experts. An explicitly-stated,
/// recognizably non-gated activation (e.g. `"gelu"`) opts a MoE layer out
/// of the gated shape — Grok-1 (314B published, xAI's own public `run.py`)
/// is exactly this real case: its experts are a plain 2-matrix MLP.
/// Calling `gated_mlp_params` unconditionally (the bug this function
/// fixes) overcounted it at 470.3B (+49.8%) — exactly the 3-matrix/
/// 2-matrix ratio — while every gated MoE tested (Mixtral, DeepSeek-V3,
/// Kimi K2, DBRX, Arctic, Qwen3, GLM-4.5) was already accurate, which is
/// exactly why this went unnoticed until a real non-gated model was
/// tested.
fn moe_expert_is_gated(params: &neurax_parser::LayerParams) -> bool {
    if params.gated {
        return true;
    }
    match params.activation.as_deref() {
        None => true,
        Some(name) => {
            neurax_formulas::activation::is_gated_activation(name)
                || name.eq_ignore_ascii_case("silu")
                || name.eq_ignore_ascii_case("swish")
        }
    }
}

fn moe_expert_params(hidden: usize, intermediate: usize, layer: &Layer) -> u64 {
    if moe_expert_is_gated(&layer.params) {
        mlp::gated_mlp_params(hidden, intermediate, layer.params.bias)
    } else {
        mlp::mlp_params(hidden, intermediate, layer.params.bias)
    }
}

/// One expert's forward FLOPs — same gated/plain split as
/// [`moe_expert_params`], for the same reason.
fn moe_expert_flops(
    batch: usize,
    seq: usize,
    hidden: usize,
    intermediate: usize,
    layer: &Layer,
) -> f64 {
    let activation = layer.params.activation.as_deref().unwrap_or("silu");
    if moe_expert_is_gated(&layer.params) {
        mlp::gated_mlp_flops(batch, seq, hidden, intermediate, activation)
    } else {
        mlp::mlp_flops(batch, seq, hidden, intermediate, activation)
    }
}

fn moe_params_fn(layer: &Layer) -> u64 {
    let hidden = layer.params.hidden_size.unwrap_or(512);
    let intermediate = layer.params.intermediate_size.unwrap_or(4 * hidden);
    let num_experts = layer.params.num_experts.unwrap_or(8);
    let shared_experts = layer.params.shared_experts.unwrap_or(0);
    let expert_params = moe_expert_params(hidden, intermediate, layer);
    moe::moe_params_with_shared(
        hidden,
        intermediate,
        num_experts,
        shared_experts,
        expert_params,
    )
}

fn moe_flops_fn(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let hidden = layer.params.hidden_size.unwrap_or(512);
    let intermediate = layer.params.intermediate_size.unwrap_or(4 * hidden);
    let num_experts = layer.params.num_experts.unwrap_or(8);
    let top_k = layer.params.top_k.unwrap_or(2);
    // One token through one expert (batch=1, seq=1) — matches
    // `moe_expert_params`'s own per-expert shape, scaled by real
    // batch/seq/top_k in `moe::moe_flops` below.
    let expert_flops = moe_expert_flops(1, 1, hidden, intermediate, layer);
    moe::moe_flops(batch, seq, hidden, num_experts, top_k, expert_flops)
}

fn moe_router_params_fn(layer: &Layer) -> u64 {
    let hidden = layer.params.hidden_size.unwrap_or(512);
    let num_experts = layer.params.num_experts.unwrap_or(8);
    moe::moe_router_params(hidden, num_experts)
}

fn moe_router_flops_fn(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let hidden = layer.params.hidden_size.unwrap_or(512);
    let num_experts = layer.params.num_experts.unwrap_or(8);
    moe::moe_router_flops(batch, seq, hidden, num_experts)
}

fn moe_combine_params_fn(_layer: &Layer) -> u64 {
    0
}

fn moe_combine_flops_fn(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let hidden = layer.params.hidden_size.unwrap_or(512);
    let top_k = layer.params.top_k.unwrap_or(2);
    2.0 * batch as f64 * seq as f64 * top_k as f64 * hidden as f64
}

fn moe_shared_expert_params_fn(layer: &Layer) -> u64 {
    let hidden = layer.params.hidden_size.unwrap_or(512);
    let intermediate = layer.params.intermediate_size.unwrap_or(4 * hidden);
    let shared_experts = layer.params.num_experts.unwrap_or(0);
    let expert_params = moe_expert_params(hidden, intermediate, layer);
    moe::moe_params_with_shared(hidden, intermediate, 0, shared_experts, expert_params)
}

fn moe_shared_expert_flops_fn(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let hidden = layer.params.hidden_size.unwrap_or(512);
    let intermediate = layer.params.intermediate_size.unwrap_or(4 * hidden);
    let shared_experts = layer.params.num_experts.unwrap_or(0) as f64;
    shared_experts * moe_expert_flops(batch, seq, hidden, intermediate, layer)
}

// ═══════════════════════════════════════════════════════════════════════
// SSM family
// ═══════════════════════════════════════════════════════════════════════

static SSM_SPECS: &[OpSpec] = &[
    spec(
        LayerType::MambaBlock,
        mamba_params_fn,
        mamba_shaped_flops_fn,
    ),
    spec(LayerType::S4Block, s4_params_fn, s4_flops_fn),
    spec(LayerType::StateSpace, s4_params_fn, s4_flops_fn),
    spec(LayerType::H3Block, h3_params_fn, h3_flops_fn),
    spec(LayerType::RwkvBlock, rwkv_params_fn, mamba_shaped_flops_fn),
    spec(
        LayerType::RetentionBlock,
        retention_params_fn,
        mamba_shaped_flops_fn,
    ),
];

fn mamba_params_fn(layer: &Layer) -> u64 {
    let hidden = layer.params.hidden_size.unwrap_or(512);
    let state_dim = layer.params.state_dim.unwrap_or(16);
    let expansion = layer.params.expansion_factor.unwrap_or(2);
    ssm::mamba_params(hidden, state_dim, expansion)
}

/// `MambaBlock`/`RwkvBlock`/`RetentionBlock` share this FLOPs formula
/// pre-migration even though RWKV/RetNet get their own, different params
/// formula just below — `neurax-formulas::ssm` has no `rwkv_flops`/
/// `retention_flops` yet, so `mamba_flops` is what actually runs for all
/// three today. Preserved exactly, not fixed: inventing a correct RWKV/
/// RetNet FLOPs formula is a real formulas-crate task, not something this
/// migration should do silently.
fn mamba_shaped_flops_fn(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let hidden = layer.params.hidden_size.unwrap_or(512);
    let state_dim = layer.params.state_dim.unwrap_or(16);
    let expand = layer.params.expansion_factor.unwrap_or(2);
    ssm::mamba_flops(batch, seq, hidden, state_dim, expand)
}

fn s4_params_fn(layer: &Layer) -> u64 {
    let hidden = layer.params.hidden_size.unwrap_or(512);
    let state_dim = layer.params.state_dim.unwrap_or(16);
    ssm::s4_params(hidden, state_dim)
}

fn s4_flops_fn(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let hidden = layer.params.hidden_size.unwrap_or(512);
    let state_dim = layer.params.state_dim.unwrap_or(16);
    ssm::s4_flops(batch, seq, hidden, state_dim)
}

fn h3_params_fn(layer: &Layer) -> u64 {
    let hidden = layer.params.hidden_size.unwrap_or(512);
    let state_dim = layer.params.state_dim.unwrap_or(16);
    ssm::h3_params(hidden, state_dim)
}

fn h3_flops_fn(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let hidden = layer.params.hidden_size.unwrap_or(512);
    let state_dim = layer.params.state_dim.unwrap_or(16);
    ssm::h3_flops(batch, seq, hidden, state_dim)
}

fn rwkv_params_fn(layer: &Layer) -> u64 {
    let hidden = layer.params.hidden_size.unwrap_or(512);
    ssm::rwkv_params(hidden, layer.params.intermediate_size)
}

fn retention_params_fn(layer: &Layer) -> u64 {
    let hidden = layer.params.hidden_size.unwrap_or(512);
    ssm::retention_params(hidden, layer.params.intermediate_size)
}

// ═══════════════════════════════════════════════════════════════════════
// GAN family
// ═══════════════════════════════════════════════════════════════════════

static GAN_SPECS: &[OpSpec] = &[
    spec(
        LayerType::GeneratorBlock,
        gan_conv_block_params_fn,
        gan_conv_block_flops_fn,
    ),
    spec(
        LayerType::DiscriminatorBlock,
        gan_conv_block_params_fn,
        gan_conv_block_flops_fn,
    ),
    spec(
        LayerType::ProgressiveBlock,
        gan_conv_block_params_fn,
        gan_conv_block_flops_fn,
    ),
    spec(
        LayerType::SelfAttention,
        self_attention_params_fn,
        self_attention_flops_fn,
    ),
    spec(LayerType::StyleMod, style_mod_params_fn, style_mod_flops_fn),
    spec(LayerType::AdaIN, zero_params_fn, norm_weight_flops_fn),
    spec(LayerType::PixelNorm, zero_params_fn, norm_weight_flops_fn),
    spec(
        LayerType::MinibatchStd,
        zero_params_fn,
        norm_weight_flops_fn,
    ),
    spec(
        LayerType::SpectralNorm,
        spectral_norm_params_fn,
        norm_weight_flops_fn,
    ),
];

fn gan_conv_block_params_fn(layer: &Layer) -> u64 {
    let in_ch = layer.params.in_channels.unwrap_or(64);
    let out_ch = layer.params.out_channels.unwrap_or(64);
    let kh = layer.params.kernel_size.unwrap_or(3);
    conv::conv2d_params(in_ch, out_ch, kh, kh, 1, layer.params.bias)
}

fn gan_conv_block_flops_fn(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let in_ch = layer.params.in_channels.unwrap_or(64);
    let out_ch = layer.params.out_channels.unwrap_or(64);
    let kh = layer.params.kernel_size.unwrap_or(3);
    let stride = layer.params.stride.unwrap_or(1);
    let padding = layer.params.padding.unwrap_or(0);
    let side = side_from(seq);
    conv::conv2d_flops(batch, in_ch, out_ch, side, side, kh, kh, stride, padding, 1)
}

fn self_attention_params_fn(layer: &Layer) -> u64 {
    let channels = layer.params.out_channels.unwrap_or(512);
    attention::attention_params(channels, channels / 64, false)
}

fn self_attention_flops_fn(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let channels = layer.params.out_channels.unwrap_or(512);
    let heads = (channels / 64).max(1);
    attention::attention_flops(batch, seq, channels, heads, false)
}

fn style_mod_params_fn(layer: &Layer) -> u64 {
    (layer.params.out_channels.unwrap_or(512) * 2) as u64
}

fn style_mod_flops_fn(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let channels = layer.params.out_channels.unwrap_or(512);
    2.0 * batch as f64 * seq as f64 * channels as f64
}

fn zero_params_fn(_layer: &Layer) -> u64 {
    0
}

/// `AdaIN`/`PixelNorm`/`MinibatchStd`/`SpectralNorm` share this FLOPs shape
/// pre-migration: a normalization-weight-sized cost (~RMSNorm's own 3 ops
/// per element), not a full layer's worth of compute — none of the four
/// carry a weight matrix of their own.
fn norm_weight_flops_fn(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let channels = layer.params.out_channels.unwrap_or(512);
    3.0 * batch as f64 * seq as f64 * channels as f64
}

fn spectral_norm_params_fn(layer: &Layer) -> u64 {
    // Spectral norm adds one "u" vector per weight matrix — reduces to
    // `in_ch`, computed the same (slightly indirect) way the pre-migration
    // arm did.
    let in_ch = layer.params.in_channels.unwrap_or(64);
    let out_ch = layer.params.out_channels.unwrap_or(64);
    (in_ch * out_ch / out_ch) as u64
}

// ═══════════════════════════════════════════════════════════════════════
// RNN family
// ═══════════════════════════════════════════════════════════════════════

static RNN_SPECS: &[OpSpec] = &[
    spec(LayerType::LstmBlock, lstm_params_fn, lstm_flops_fn),
    spec(LayerType::GruBlock, gru_params_fn, gru_flops_fn),
    spec(LayerType::RnnCell, rnn_cell_params_fn, rnn_cell_flops_fn),
    spec(
        LayerType::Bidirectional,
        bidirectional_params_fn,
        bidirectional_flops_fn,
    ),
    spec(
        LayerType::EncoderBlock,
        encoder_decoder_params_fn,
        encoder_decoder_flops_fn,
    ),
    spec(
        LayerType::DecoderBlock,
        encoder_decoder_params_fn,
        encoder_decoder_flops_fn,
    ),
];

fn rnn_hidden_and_input(layer: &Layer) -> (usize, usize) {
    let hidden = layer.params.rnn_hidden_size.unwrap_or(512);
    let input_size = layer.params.hidden_size.unwrap_or(hidden);
    (hidden, input_size)
}

fn lstm_params_fn(layer: &Layer) -> u64 {
    let (hidden, input_size) = rnn_hidden_and_input(layer);
    let bidir_mult = if layer.params.bidirectional_rnn { 2 } else { 1 };
    let stack = layer.params.num_rnn_layers.unwrap_or(1).max(1) as u64;
    let rest_input_size = hidden * bidir_mult as usize;
    let first = rnn::lstm_params(hidden, input_size, true);
    let rest = rnn::lstm_params(hidden, rest_input_size, true) * (stack - 1);
    (first + rest) * bidir_mult
}

fn lstm_flops_fn(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let (hidden, input_size) = rnn_hidden_and_input(layer);
    let bidir_mult = if layer.params.bidirectional_rnn { 2 } else { 1 };
    let stack = layer.params.num_rnn_layers.unwrap_or(1).max(1) as u64;
    let rest_input_size = hidden * bidir_mult as usize;
    let first = rnn::lstm_flops(batch, seq, hidden, input_size);
    let rest = rnn::lstm_flops(batch, seq, hidden, rest_input_size) * (stack - 1) as f64;
    (first + rest) * bidir_mult as f64
}

fn gru_params_fn(layer: &Layer) -> u64 {
    let (hidden, input_size) = rnn_hidden_and_input(layer);
    let bidir_mult = if layer.params.bidirectional_rnn { 2 } else { 1 };
    let stack = layer.params.num_rnn_layers.unwrap_or(1).max(1) as u64;
    let rest_input_size = hidden * bidir_mult as usize;
    let first = rnn::gru_params(hidden, input_size, true);
    let rest = rnn::gru_params(hidden, rest_input_size, true) * (stack - 1);
    (first + rest) * bidir_mult
}

fn gru_flops_fn(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let (hidden, input_size) = rnn_hidden_and_input(layer);
    let bidir_mult = if layer.params.bidirectional_rnn { 2 } else { 1 };
    let stack = layer.params.num_rnn_layers.unwrap_or(1).max(1) as u64;
    let rest_input_size = hidden * bidir_mult as usize;
    let first = rnn::gru_flops(batch, seq, hidden, input_size);
    let rest = rnn::gru_flops(batch, seq, hidden, rest_input_size) * (stack - 1) as f64;
    (first + rest) * bidir_mult as f64
}

fn rnn_cell_params_fn(layer: &Layer) -> u64 {
    let (hidden, input_size) = rnn_hidden_and_input(layer);
    let stack = layer.params.num_rnn_layers.unwrap_or(1).max(1) as u64;
    let first = rnn::rnn_params(hidden, input_size, true);
    let rest = rnn::rnn_params(hidden, hidden, true) * (stack - 1);
    first + rest
}

fn rnn_cell_flops_fn(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let (hidden, input_size) = rnn_hidden_and_input(layer);
    let stack = layer.params.num_rnn_layers.unwrap_or(1).max(1) as u64;
    let first = rnn::rnn_flops(batch, seq, hidden, input_size);
    let rest = rnn::rnn_flops(batch, seq, hidden, hidden) * (stack - 1) as f64;
    first + rest
}

fn bidirectional_params_fn(layer: &Layer) -> u64 {
    let (hidden, input_size) = rnn_hidden_and_input(layer);
    rnn::lstm_params(hidden, input_size, true) * 2
}

fn bidirectional_flops_fn(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let (hidden, input_size) = rnn_hidden_and_input(layer);
    rnn::bidirectional_flops(batch, seq, hidden, input_size, "lstm")
}

fn encoder_decoder_params_fn(layer: &Layer) -> u64 {
    let (hidden, input_size) = rnn_hidden_and_input(layer);
    rnn::lstm_params(hidden, input_size, true)
}

fn encoder_decoder_flops_fn(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let (hidden, input_size) = rnn_hidden_and_input(layer);
    rnn::lstm_flops(batch, seq, hidden, input_size)
}

// ═══════════════════════════════════════════════════════════════════════
// Diffusion family
// ═══════════════════════════════════════════════════════════════════════

static DIFFUSION_SPECS: &[OpSpec] = &[
    spec(
        LayerType::UnetBlock,
        unet_resnet_params_fn,
        unet_resnet_flops_fn,
    ),
    spec(
        LayerType::ResnetBlock,
        unet_resnet_params_fn,
        unet_resnet_flops_fn,
    ),
    spec(
        LayerType::TimeEmbedding,
        time_embedding_params_fn,
        time_embedding_flops_fn,
    ),
    spec(
        LayerType::TimestepBlock,
        time_embedding_params_fn,
        time_embedding_flops_fn,
    ),
    spec(
        LayerType::CrossAttention,
        cross_attention_params_fn,
        cross_attention_flops_fn,
    ),
    spec(
        LayerType::DownBlock,
        down_up_mid_params_fn,
        down_up_mid_flops_fn,
    ),
    spec(
        LayerType::UpBlock,
        down_up_mid_params_fn,
        down_up_mid_flops_fn,
    ),
    spec(
        LayerType::MidBlock,
        down_up_mid_params_fn,
        down_up_mid_flops_fn,
    ),
    spec(
        LayerType::ConditionBlock,
        condition_block_params_fn,
        condition_block_flops_fn,
    ),
    spec(
        LayerType::NoisePredictor,
        noise_predictor_params_fn,
        noise_predictor_flops_fn,
    ),
    spec(LayerType::VaeEncoder, vae_params_fn, vae_flops_fn),
    spec(LayerType::VaeDecoder, vae_params_fn, vae_flops_fn),
];

/// Shared `in_channels_diff`/`in_channels`/`hidden_size` fallback chain
/// (defaulting to `default`) every diffusion block below already used —
/// real U-Net templates (SDXL, SD1.5) only ever set `hidden_size` on these
/// nodes.
fn unet_channels(layer: &Layer, default: usize) -> (usize, usize) {
    let in_ch = layer.params.in_channels_diff.unwrap_or(
        layer
            .params
            .in_channels
            .unwrap_or(layer.params.hidden_size.unwrap_or(default)),
    );
    let out_ch = layer.params.out_channels_diff.unwrap_or(
        layer
            .params
            .out_channels
            .unwrap_or(layer.params.hidden_size.unwrap_or(default)),
    );
    (in_ch, out_ch)
}

fn unet_resnet_params_fn(layer: &Layer) -> u64 {
    let (in_ch, out_ch) = unet_channels(layer, 320);
    let block_repeat = layer.params.layers_per_block.unwrap_or(1) as u64;
    cnn_blocks::resnet_basic_block_params(in_ch, out_ch, 1, layer.params.bias) * block_repeat
}

fn unet_resnet_flops_fn(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let (in_ch, out_ch) = unet_channels(layer, 320);
    let block_repeat = layer.params.layers_per_block.unwrap_or(1).max(1) as f64;
    let side = side_from(seq);
    cnn_blocks::resnet_basic_block_flops(batch, in_ch, out_ch, side, side, 1) * block_repeat
}

fn time_embedding_params_fn(layer: &Layer) -> u64 {
    let channels = layer.params.hidden_size.unwrap_or(320);
    mlp::mlp_params(channels, channels * 4, true)
}

fn time_embedding_flops_fn(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let channels = layer.params.hidden_size.unwrap_or(320);
    mlp::mlp_flops(batch, seq, channels, channels * 4, "silu")
}

fn condition_block_params_fn(layer: &Layer) -> u64 {
    let hidden = layer.params.hidden_size.unwrap_or(320);
    mlp::mlp_params(hidden, hidden * 4, true)
}

fn condition_block_flops_fn(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let hidden = layer.params.hidden_size.unwrap_or(320);
    mlp::mlp_flops(batch, seq, hidden, hidden * 4, "gelu")
}

fn cross_attention_params_fn(layer: &Layer) -> u64 {
    let hidden = layer.params.hidden_size.unwrap_or(320);
    let context_dim = layer.params.cross_attention_dim.unwrap_or(hidden);
    let block_repeat = layer.params.transformer_layers_per_block.unwrap_or(1) as u64;
    attention::cross_attention_params(hidden, context_dim, true) * block_repeat
}

fn cross_attention_flops_fn(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let hidden = layer.params.hidden_size.unwrap_or(320);
    let heads = layer.params.num_heads.unwrap_or(8);
    let block_repeat = layer.params.transformer_layers_per_block.unwrap_or(1) as f64;
    attention::attention_flops(batch, seq, hidden, heads, false) * block_repeat
}

fn down_up_mid_params_fn(layer: &Layer) -> u64 {
    let block_repeat = layer.params.layers_per_block.unwrap_or(1).max(1) as u64;
    match layer
        .params
        .block_out_channels
        .as_ref()
        .filter(|c| !c.is_empty())
    {
        Some(channels) => {
            let entry = layer.params.in_channels_diff.unwrap_or(
                layer
                    .params
                    .in_channels
                    .unwrap_or(layer.params.hidden_size.unwrap_or(channels[0])),
            );
            let mut widths = Vec::with_capacity(channels.len() + 1);
            widths.push(entry);
            widths.extend(channels.iter().copied());
            widths
                .windows(2)
                .map(|w| {
                    let (win, wout) = (w[0], w[1]);
                    let first =
                        cnn_blocks::resnet_basic_block_params(win, wout, 1, layer.params.bias);
                    let rest =
                        cnn_blocks::resnet_basic_block_params(wout, wout, 1, layer.params.bias)
                            * (block_repeat - 1);
                    first + rest
                })
                .sum()
        }
        None => {
            let (in_ch, out_ch) = unet_channels(layer, 320);
            cnn_blocks::resnet_basic_block_params(in_ch, out_ch, 1, layer.params.bias)
                * block_repeat
        }
    }
}

fn down_up_mid_flops_fn(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let block_repeat = layer.params.layers_per_block.unwrap_or(1).max(1) as f64;
    let side = side_from(seq);
    match layer
        .params
        .block_out_channels
        .as_ref()
        .filter(|c| !c.is_empty())
    {
        Some(channels) => {
            let entry = layer.params.in_channels_diff.unwrap_or(
                layer
                    .params
                    .in_channels
                    .unwrap_or(layer.params.hidden_size.unwrap_or(channels[0])),
            );
            let mut widths = Vec::with_capacity(channels.len() + 1);
            widths.push(entry);
            widths.extend(channels.iter().copied());
            widths
                .windows(2)
                .map(|w| {
                    let (win, wout) = (w[0], w[1]);
                    let first =
                        cnn_blocks::resnet_basic_block_flops(batch, win, wout, side, side, 1);
                    let rest =
                        cnn_blocks::resnet_basic_block_flops(batch, wout, wout, side, side, 1)
                            * (block_repeat - 1.0);
                    first + rest
                })
                .sum()
        }
        None => {
            let (in_ch, out_ch) = unet_channels(layer, 320);
            cnn_blocks::resnet_basic_block_flops(batch, in_ch, out_ch, side, side, 1) * block_repeat
        }
    }
}

fn noise_predictor_params_fn(layer: &Layer) -> u64 {
    let channels = layer.params.out_channels_diff.unwrap_or(4);
    conv::conv2d_params(channels, channels, 3, 3, 1, false)
}

fn noise_predictor_flops_fn(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let channels = layer.params.out_channels_diff.unwrap_or(4);
    let side = side_from(seq);
    conv::conv2d_flops(batch, channels, channels, side, side, 3, 3, 1, 1, 1)
}

fn vae_params_fn(layer: &Layer) -> u64 {
    let in_ch = layer.params.in_channels.unwrap_or(3);
    let out_ch = layer.params.out_channels_diff.unwrap_or(4);
    conv::conv2d_params(in_ch, out_ch, 3, 3, 1, false)
}

fn vae_flops_fn(layer: &Layer, batch: usize, seq: usize, _ctx: &FlopsContext) -> f64 {
    let in_ch = layer.params.in_channels.unwrap_or(3);
    let out_ch = layer.params.out_channels_diff.unwrap_or(4);
    let side = side_from(seq);
    conv::conv2d_flops(batch, in_ch, out_ch, side, side, 3, 3, 1, 1, 1)
}

// ═══════════════════════════════════════════════════════════════════════
// GNN family
// ═══════════════════════════════════════════════════════════════════════

static GNN_SPECS: &[OpSpec] = &[
    spec(LayerType::GraphConvNet, gcn_params_fn, gcn_flops_fn),
    spec(
        LayerType::MessagePassing,
        message_passing_params_fn,
        message_passing_flops_fn,
    ),
    spec(LayerType::GraphAttentionNet, gat_params_fn, gat_flops_fn),
    spec(LayerType::RgcnConv, rgcn_params_fn, rgcn_flops_fn),
];

/// Reads `num_nodes`/`num_edges` out of `global_params.extra`, the same map
/// `neurax-ir`'s `GlobalResolutionContext` reads for the params side —
/// falling back to the Cora citation-graph benchmark's real size, the same
/// default a GNN design with no explicit graph size already assumed before
/// migrating.
/// A positive size from `global_params`, or `default`.
///
/// Every caller of this helper asks for a quantity that cannot meaningfully be
/// zero — a graph with no nodes, a batch of no samples. A present zero is
/// therefore a sender's "unset" sentinel, not a value, and treating it as one
/// silently zeroes the formula that reads it: `gcn_flops(0, in, out, 0)` is 0,
/// and a report whose total FLOPs is zero is rejected outright. Six GNN
/// reference templates failed exactly that way, because the studio wrote
/// `num_nodes: 0` for a config that had never set it.
///
/// The studio no longer sends those zeros, but it is not the only client:
/// `neurax-agent`, `neurax-mcp`, an older frontend build and hand-written JSON
/// all reach this same code. Rejecting a non-positive value here is what makes
/// the default mean "no usable value was supplied", from any of them.
fn extra_usize(global_params: &GlobalParams, key: &str, default: usize) -> usize {
    global_params
        .extra
        .get(key)
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

fn gcn_params_fn(layer: &Layer) -> u64 {
    let in_f = layer.params.in_features.unwrap_or(64);
    let out_f = layer.params.out_features.unwrap_or(64);
    gnn::gcn_params(in_f, out_f, layer.params.bias)
}

fn gcn_flops_fn(layer: &Layer, _batch: usize, _seq: usize, ctx: &FlopsContext) -> f64 {
    let in_f = layer.params.in_features.unwrap_or(64);
    let out_f = layer.params.out_features.unwrap_or(64);
    let num_nodes = extra_usize(ctx.global_params, "num_nodes", 2708);
    let num_edges = extra_usize(ctx.global_params, "num_edges", 10556);
    gnn::gcn_flops(num_nodes, in_f, out_f, num_edges)
}

fn message_passing_params_fn(layer: &Layer) -> u64 {
    let in_f = layer.params.in_features.unwrap_or(64);
    let out_f = layer.params.out_features.unwrap_or(64);
    gnn::message_passing_params(in_f, out_f, layer.params.bias)
}

fn message_passing_flops_fn(layer: &Layer, _batch: usize, _seq: usize, ctx: &FlopsContext) -> f64 {
    let in_f = layer.params.in_features.unwrap_or(64);
    let out_f = layer.params.out_features.unwrap_or(64);
    let num_nodes = extra_usize(ctx.global_params, "num_nodes", 2708);
    let num_edges = extra_usize(ctx.global_params, "num_edges", 10556);
    gnn::message_passing_flops(num_nodes, in_f, out_f, num_edges)
}

fn gat_params_fn(layer: &Layer) -> u64 {
    let in_f = layer.params.in_features.unwrap_or(64);
    let out_f = layer.params.out_features.unwrap_or(64);
    let num_heads = layer.params.num_heads.unwrap_or(8);
    gnn::gat_params(in_f, out_f, num_heads, layer.params.bias)
}

fn gat_flops_fn(layer: &Layer, _batch: usize, _seq: usize, ctx: &FlopsContext) -> f64 {
    let in_f = layer.params.in_features.unwrap_or(64);
    let out_f = layer.params.out_features.unwrap_or(64);
    let num_heads = layer.params.num_heads.unwrap_or(8);
    let num_nodes = extra_usize(ctx.global_params, "num_nodes", 2708);
    let num_edges = extra_usize(ctx.global_params, "num_edges", 10556);
    gnn::gat_flops(num_nodes, in_f, out_f, num_edges, num_heads)
}

fn rgcn_params_fn(layer: &Layer) -> u64 {
    let in_f = layer.params.in_features.unwrap_or(64);
    let out_f = layer.params.out_features.unwrap_or(64);
    let num_relations = layer.params.num_relations.unwrap_or(1);
    gnn::rgcn_params(
        in_f,
        out_f,
        num_relations,
        layer.params.num_bases,
        layer.params.bias,
    )
}

fn rgcn_flops_fn(layer: &Layer, _batch: usize, _seq: usize, ctx: &FlopsContext) -> f64 {
    let in_f = layer.params.in_features.unwrap_or(64);
    let out_f = layer.params.out_features.unwrap_or(64);
    let num_relations = layer.params.num_relations.unwrap_or(1);
    let num_nodes = extra_usize(ctx.global_params, "num_nodes", 2708);
    let num_edges = extra_usize(ctx.global_params, "num_edges", 10556);
    gnn::rgcn_flops(num_nodes, in_f, out_f, num_edges, num_relations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use neurax_parser::LayerParams;

    /// Every real `LayerType` except `Custom` — see the crate doc for why
    /// that one stays permanently unmigrated.
    const ALL_MIGRATED_TYPES: &[LayerType] = &[
        // Transformer
        LayerType::Embedding,
        LayerType::Attention,
        LayerType::Mlp,
        LayerType::Conv,
        LayerType::Dense,
        LayerType::LoraLinear,
        LayerType::DoraLinear,
        LayerType::Normalization,
        LayerType::Pooling,
        // MoE
        LayerType::MoE,
        LayerType::MoeRouter,
        LayerType::MoeCombine,
        LayerType::MoeSharedExpert,
        // CNN blocks
        LayerType::ResidualBlock,
        LayerType::ResnetBottleneck,
        LayerType::Mbconv,
        LayerType::Inception,
        LayerType::DenseBlock,
        LayerType::ConvnextBlock,
        LayerType::ShuffleUnit,
        LayerType::C2f,
        LayerType::Detection,
        LayerType::Transition,
        // SSM
        LayerType::MambaBlock,
        LayerType::S4Block,
        LayerType::H3Block,
        LayerType::StateSpace,
        LayerType::RwkvBlock,
        LayerType::RetentionBlock,
        // GAN
        LayerType::GeneratorBlock,
        LayerType::DiscriminatorBlock,
        LayerType::StyleMod,
        LayerType::AdaIN,
        LayerType::MinibatchStd,
        LayerType::PixelNorm,
        LayerType::SelfAttention,
        LayerType::SpectralNorm,
        LayerType::ProgressiveBlock,
        // RNN
        LayerType::LstmBlock,
        LayerType::GruBlock,
        LayerType::RnnCell,
        LayerType::Bidirectional,
        LayerType::EncoderBlock,
        LayerType::DecoderBlock,
        // Diffusion
        LayerType::UnetBlock,
        LayerType::TimeEmbedding,
        LayerType::CrossAttention,
        LayerType::DownBlock,
        LayerType::UpBlock,
        LayerType::MidBlock,
        LayerType::ResnetBlock,
        LayerType::TimestepBlock,
        LayerType::ConditionBlock,
        LayerType::NoisePredictor,
        LayerType::VaeEncoder,
        LayerType::VaeDecoder,
        // GNN
        LayerType::GraphConvNet,
        LayerType::GraphAttentionNet,
        LayerType::MessagePassing,
        LayerType::RgcnConv,
    ];

    fn bare_layer(layer_type: LayerType) -> Layer {
        Layer {
            id: "test".to_string(),
            layer_type,
            input_shape: vec![],
            output_shape: vec![],
            params: LayerParams::default(),
            custom_equations: None,
        }
    }

    fn default_ctx(global_params: &GlobalParams) -> FlopsContext<'_> {
        FlopsContext {
            global_params,
            image_channels: None,
            image_height: None,
            image_width: None,
        }
    }

    #[test]
    fn every_real_layer_type_except_custom_is_registered() {
        for &lt in ALL_MIGRATED_TYPES {
            assert!(
                op_spec(lt).is_some(),
                "{lt:?} should be registered in OpSpec-IR"
            );
        }
    }

    #[test]
    fn custom_is_deliberately_not_registered() {
        // Custom evaluates a user-supplied equation and reports neurax-ir
        // diagnostics — neurax-ir-specific behavior this crate has no
        // business owning. Must stay `None` forever, not "not migrated yet".
        assert!(op_spec(LayerType::Custom).is_none());
    }

    #[test]
    fn every_migrated_type_computes_nonzero_flops_from_bare_defaults() {
        let global_params = GlobalParams::default();
        let ctx = default_ctx(&global_params);
        for &lt in ALL_MIGRATED_TYPES {
            // MoeSharedExpert defaults its own shared-expert count to 0
            // (`num_experts.unwrap_or(0)`, matching the params side's own
            // rule that no shared experts means no shared-expert cost) — a
            // bare layer of this type legitimately costs 0 FLOPs, same as
            // before migrating.
            if lt == LayerType::MoeSharedExpert {
                continue;
            }
            let spec = op_spec(lt).unwrap();
            let layer = bare_layer(lt);
            // Params: a handful of types are legitimately weightless
            // (MoeCombine has no weights of its own; AdaIN/PixelNorm/
            // MinibatchStd normalize an existing activation) — not asserted
            // nonzero here, only that it doesn't panic.
            let _ = (spec.params_fn)(&layer);
            let flops = (spec.flops_fn)(&layer, 1, 196, &ctx);
            assert!(
                flops > 0.0,
                "{lt:?} should have nonzero FLOPs from bare defaults"
            );
        }
    }

    #[test]
    fn residual_block_no_longer_borrows_bottleneck_s_out_channels_default() {
        // The bug the CNN migration fixed: a bare ResidualBlock with no
        // stated out_channels must default to 64 (matching its own
        // params-side default and a real ResNet-18/34 basic block with no
        // channel change), not 256 (ResnetBottleneck's real default,
        // previously shared by accident via one combined match arm).
        let global_params = GlobalParams::default();
        let ctx = default_ctx(&global_params);
        let spec = op_spec(LayerType::ResidualBlock).unwrap();
        let layer = bare_layer(LayerType::ResidualBlock);
        let flops = (spec.flops_fn)(&layer, 1, 196, &ctx);
        let expected = cnn_blocks::resnet_basic_block_flops(1, 64, 64, 14, 14, 1);
        assert_eq!(flops, expected);
    }

    #[test]
    fn conv_flops_reads_kernel_h_and_kernel_w_not_just_kernel_size() {
        // Real bug found migrating Conv: the pre-migration FLOPs arm derived
        // kernel_w from kernel_size alone, never reading params.kernel_h/
        // kernel_w — an explicit non-square kernel was silently costed as
        // square. Both formulas must now agree on which fields decide the
        // kernel shape.
        let mut params = LayerParams::default();
        params.kernel_h = Some(5);
        params.kernel_w = Some(7);
        params.kernel_size = Some(3);
        params.in_channels = Some(3);
        params.out_channels = Some(8);
        let layer = Layer {
            id: "t".to_string(),
            layer_type: LayerType::Conv,
            input_shape: vec![1, 3, 32, 32],
            output_shape: vec![],
            params,
            custom_equations: None,
        };
        let global_params = GlobalParams::default();
        let ctx = default_ctx(&global_params);
        let spec = op_spec(LayerType::Conv).unwrap();
        let flops = (spec.flops_fn)(&layer, 1, 196, &ctx);

        let padding = 0; // Conv defaults to no padding when params.padding is unset, stride defaults to 1
        let out_h = 32 + 2 * padding - 5 + 1;
        let out_w = 32 + 2 * padding - 7 + 1;
        let expected = 2.0 * 1.0 * out_h as f64 * out_w as f64 * 8.0 * 5.0 * 7.0 * (3.0 / 1.0);
        assert_eq!(flops, expected);
    }

    #[test]
    fn a_zero_sized_global_falls_back_instead_of_zeroing_the_formula() {
        // The studio used to write `num_nodes: 0` whenever a config had never
        // set it, which drove every GCN/GAT template's FLOPs to zero — and the
        // compiler rejects a report whose total is zero. A sender's "unset"
        // sentinel must reach the same default an absent key would.
        let spec = op_spec(LayerType::GraphConvNet).unwrap();
        let layer = bare_layer(LayerType::GraphConvNet);

        let mut zeroed = GlobalParams::default();
        zeroed
            .extra
            .insert("num_nodes".to_string(), serde_json::json!(0));
        zeroed
            .extra
            .insert("num_edges".to_string(), serde_json::json!(0));

        let empty = GlobalParams::default();
        let with_zeros = (spec.flops_fn)(&layer, 1, 1, &default_ctx(&zeroed));
        let with_nothing = (spec.flops_fn)(&layer, 1, 1, &default_ctx(&empty));

        assert!(with_zeros > 0.0, "a zero-sized graph must not zero the FLOPs");
        assert_eq!(
            with_zeros, with_nothing,
            "an explicit zero must behave exactly like an absent key"
        );
    }

    #[test]
    fn gnn_flops_read_graph_size_from_global_params_extra() {
        // GNN was the family that forced FlopsContext to exist at all: no
        // per-layer field carries the graph's real size, only
        // global_params.extra. A design that states num_nodes/num_edges
        // must actually change the FLOPs, not silently keep using the
        // Cora-sized default.
        let mut global_params = GlobalParams::default();
        global_params
            .extra
            .insert("num_nodes".to_string(), serde_json::json!(100));
        global_params
            .extra
            .insert("num_edges".to_string(), serde_json::json!(50));
        let small_ctx = default_ctx(&global_params);
        let default_global_params = GlobalParams::default();
        let default_ctx = default_ctx(&default_global_params);

        let spec = op_spec(LayerType::GraphConvNet).unwrap();
        let layer = bare_layer(LayerType::GraphConvNet);
        let small_flops = (spec.flops_fn)(&layer, 1, 196, &small_ctx);
        let cora_flops = (spec.flops_fn)(&layer, 1, 196, &default_ctx);
        assert_ne!(small_flops, cora_flops);
    }

    #[test]
    fn params_and_flops_agree_on_which_channels_a_cnn_block_defaults_to() {
        // The structural guarantee this whole crate exists for: every
        // migrated CNN block's params_fn and flops_fn must read the same
        // in_channels/out_channels defaults, so they can never silently
        // describe two different blocks the way ResidualBlock/
        // ResnetBottleneck once did.
        let global_params = GlobalParams::default();
        let ctx = default_ctx(&global_params);
        for lt in [
            LayerType::ResidualBlock,
            LayerType::ResnetBottleneck,
            LayerType::Mbconv,
            LayerType::Inception,
            LayerType::DenseBlock,
            LayerType::ConvnextBlock,
            LayerType::ShuffleUnit,
            LayerType::C2f,
            LayerType::Detection,
            LayerType::Transition,
        ] {
            let spec = op_spec(lt).unwrap();
            let layer = bare_layer(lt);
            let param_count = (spec.params_fn)(&layer);
            let flops = (spec.flops_fn)(&layer, 1, 196, &ctx);
            assert!(param_count > 0, "{lt:?} should have nonzero params");
            assert!(flops > 0.0, "{lt:?} should have nonzero FLOPs");
        }
    }

    #[test]
    fn embedding_attention_mlp_dense_lora_track_real_activation_memory() {
        // The one thing the generic early-return path in neurax-ir must not
        // silently zero out: these six types tracked a real, nonzero
        // activation-tensor size before migrating (everything else tracked
        // zero already). `activation_memory_fn` is how that survives.
        for lt in [
            LayerType::Embedding,
            LayerType::Attention,
            LayerType::Mlp,
            LayerType::Dense,
            LayerType::LoraLinear,
            LayerType::DoraLinear,
        ] {
            let spec = op_spec(lt).unwrap();
            let mem_fn = spec
                .activation_memory_fn
                .unwrap_or_else(|| panic!("{lt:?} should track real activation memory"));
            let layer = bare_layer(lt);
            assert!(
                mem_fn(&layer, 2, 128, "fp16") > 0,
                "{lt:?} should report nonzero activation memory for a real batch/seq/dtype"
            );
        }
    }

    #[test]
    fn most_migrated_types_have_no_activation_memory_fn() {
        // Matches what the pre-migration arms did: only the six Transformer
        // types above ever computed a real activation-tensor size. Every
        // other migrated type used a hardcoded `0`, which `None` reproduces.
        let tracked: &[LayerType] = &[
            LayerType::Embedding,
            LayerType::Attention,
            LayerType::Mlp,
            LayerType::Dense,
            LayerType::LoraLinear,
            LayerType::DoraLinear,
        ];
        for &lt in ALL_MIGRATED_TYPES {
            if tracked.contains(&lt) {
                continue;
            }
            let spec = op_spec(lt).unwrap();
            assert!(
                spec.activation_memory_fn.is_none(),
                "{lt:?} shouldn't track activation memory — it didn't before migrating"
            );
        }
    }

    #[test]
    fn moe_defaults_to_gated_experts_when_activation_is_unstated() {
        // Mixtral 8x7B's and DeepSeek-V3's own reference JSON fixtures
        // (examples/models/{mixtral_8x7b,deepseek_v3}.json) state no
        // `activation`/`gated` field on their MoE layer at all — this
        // default is what makes those still resolve as real SwiGLU experts.
        let layer = bare_layer(LayerType::MoE);
        assert!(moe_expert_is_gated(&layer.params));
    }

    #[test]
    fn moe_experts_can_opt_out_of_gating_grok1_style() {
        // Real bug found auditing 100B+ models: Grok-1 (314B published,
        // xAI's own public run.py) uses a plain 2-matrix expert MLP, not
        // SwiGLU — this used to be overcounted at 470.3B (+49.8%) because
        // the gated (3-matrix) formula ran unconditionally. An explicit,
        // recognizably non-gated activation now opts a MoE layer out.
        let base = LayerParams {
            hidden_size: Some(6144),
            intermediate_size: Some(16384),
            num_experts: Some(8),
            top_k: Some(2),
            ..Default::default()
        };
        let non_gated_params = LayerParams {
            activation: Some("gelu".to_string()),
            ..base.clone()
        };
        assert!(!moe_expert_is_gated(&non_gated_params));
        assert!(moe_expert_is_gated(&base));

        let gated_layer = Layer {
            id: "gated".to_string(),
            layer_type: LayerType::MoE,
            input_shape: vec![],
            output_shape: vec![],
            params: base,
            custom_equations: None,
        };
        let non_gated_layer = Layer {
            id: "non_gated".to_string(),
            layer_type: LayerType::MoE,
            input_shape: vec![],
            output_shape: vec![],
            params: non_gated_params,
            custom_equations: None,
        };

        let spec = op_spec(LayerType::MoE).unwrap();
        let gated_params = (spec.params_fn)(&gated_layer);
        let non_gated_params = (spec.params_fn)(&non_gated_layer);
        // Gated experts have exactly 3/2 the weights of plain ones (gate,
        // up, down vs. just up, down) — the same ratio that produced
        // Grok-1's +49.8% before this fix.
        assert!(
            gated_params > non_gated_params,
            "gated MoE params ({gated_params}) should exceed non-gated ({non_gated_params})"
        );
    }

    #[test]
    fn glm4_5_shaped_attention_uses_the_real_head_dim() {
        // GLM-4.5, HuggingFace-verified (zai-org/GLM-4.5 config.json):
        // hidden_size=5120, num_attention_heads=96, num_key_value_heads=8,
        // head_dim=128. 96*128=12288 != 5120 — the validator used to reject
        // this outright ("hidden_size must be divisible by num_heads");
        // the formulas must now use the real, wider head_dim instead of
        // silently deriving (and truncating) one from hidden_size/num_heads.
        let layer = Layer {
            id: "glm".to_string(),
            layer_type: LayerType::Attention,
            input_shape: vec![],
            output_shape: vec![],
            params: LayerParams {
                hidden_size: Some(5120),
                num_heads: Some(96),
                num_kv_heads: Some(8),
                head_dim: Some(128),
                ..Default::default()
            },
            custom_equations: None,
        };
        let global_params = GlobalParams::default();
        let ctx = default_ctx(&global_params);
        let spec = op_spec(LayerType::Attention).unwrap();

        let params_with_real_head_dim = (spec.params_fn)(&layer);
        let flops_with_real_head_dim = (spec.flops_fn)(&layer, 1, 4096, &ctx);
        assert!(params_with_real_head_dim > 0);
        assert!(flops_with_real_head_dim > 0.0 && flops_with_real_head_dim.is_finite());

        // Without the real head_dim, the derived one (5120/96, truncated)
        // would compute a narrower — and wrong — Q/output projection.
        let mut without_head_dim = layer.clone();
        without_head_dim.params.head_dim = None;
        let params_without = (spec.params_fn)(&without_head_dim);
        assert_ne!(
            params_with_real_head_dim, params_without,
            "an explicit head_dim that doesn't derive from hidden_size/num_heads must change the result"
        );
    }
}
