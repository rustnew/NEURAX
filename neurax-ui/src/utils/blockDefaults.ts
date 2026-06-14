/**
 * blockDefaults.ts
 * Maps catalogue_v2 block type strings to sensible default params.
 * Used when loading macro "Quick Start" blocks onto the canvas so
 * that every node is pre-filled and ready for compilation.
 */

import { ParameterValue } from '@/types/architecture.ts';

type Params = Record<string, ParameterValue>;

const DEFAULTS: Record<string, Params> = {
    // ── I/O ──────────────────────────────────────────────────
    input: { shape: '[B, seq_len]' },
    output: { shape: '[B, output_dim]' },

    // ── Transformer / Attention ──────────────────────────────
    embedding: { vocab_size: 32000, d_model: 768, padding_idx: 0 },
    token_embedding: { vocab_size: 32000, d_model: 768, padding_idx: 0 },
    positional_encoding: { d_model: 768, max_len: 2048, encoding_type: 'sinusoidal' },
    pos_absolute: { d_model: 768, max_len: 2048, encoding_type: 'sinusoidal', dropout: 0.0 },
    rope: { d_model: 768, base: 10000, max_len: 2048 },
    pos_rope: { d_model: 768, max_len: 2048, base: 10000, scaling_type: 'none', partial_rotary_factor: 1.0 },
    alibi: { n_heads: 12, causal: true },
    pos_alibi: { n_heads: 12, causal: true },
    mha: { n_heads: 12, head_dim: 64, d_model: 768, dropout: 0.0, bias: false, causal: true },
    gqa: { n_heads: 32, n_kv_heads: 8, d_model: 4096, dropout: 0.0, causal: true },
    mha_attention: { n_heads: 12, head_dim: 64, d_model: 768, dropout: 0.0, bias: false, causal: true, output_projection: true },
    gqa_attention: { n_heads: 32, n_kv_heads: 8, d_model: 4096, dropout: 0.0, causal: true },
    cross_attention: { d_model: 512, n_heads: 8, head_dim: 64, dropout: 0.1, causal_decoder: false },
    flash_attention: { n_heads: 12, d_model: 768, version: 3, causal: true },
    ffn: { d_model: 768, d_ff: 3072, bias: true, dropout: 0.0 },
    ffn_standard: { d_model: 768, d_ff: 3072, bias: true, dropout: 0.0, activation: 'gelu' },
    swiglu: { d_model: 4096, d_ff: 11008, gate_type: 'swiglu', bias: false },
    ffn_gated: { d_model: 4096, d_ff: 11008, gate_type: 'swiglu', bias: false, activation: 'silu' },
    lm_head: { d_model: 768, vocab_size: 32000, bias: false, tie_weights: true },

    // ── Normalization ────────────────────────────────────────
    layernorm: { normalized_shape: 768, eps: 1e-6, elementwise_affine: true, position: 'pre' },
    rmsnorm: { normalized_shape: 768, eps: 1e-5, elementwise_affine: true, position: 'pre' },
    batchnorm: { num_features: 64, eps: 1e-5, momentum: 0.1, affine: true },
    groupnorm: { num_groups: 32, num_channels: 256, eps: 1e-5, affine: true },
    instancenorm: { num_features: 64, eps: 1e-5, affine: false },
    spectral_norm: { num_features: 256, power_iterations: 1 },

    // ── Structure ────────────────────────────────────────────
    add: { dropout: 0.0, pre_norm: true },
    residual: { dropout: 0.0, pre_norm: true },
    residual_add: { dropout: 0.0, pre_norm: true },
    concat: { dim: -1 },
    dropout: { rate: 0.1 },
    flatten: { start_dim: 1 },
    merge: { strategy: 'add' },
    dense: { in_features: 768, out_features: 768, bias: true },
    linear_projection: { in_features: 768, out_features: 768, bias: true, dtype: 'float32', activation: 'none' },
    lora_linear: { in_features: 768, out_features: 768, rank: 16, alpha: 32, bias: true, activation: 'none' },
    dora_linear: { in_features: 768, out_features: 768, rank: 16, alpha: 32, bias: true, activation: 'none' },
    layer_stack: { num_layers: 12, share_parameters: false },

    // ── CNN ──────────────────────────────────────────────────
    stem_block: { type: 'conv', in_channels: 3, out_channels: 64, stride: 2, activation: 'relu' },
    stage_block: { num_blocks: 3, block_type: 'bottleneck', in_channels: 64, out_channels: 256, stride: 1 },
    conv2d: { in_channels: 64, out_channels: 64, kernel_size: 3, stride: 1, padding: 1, bias: true },
    depthwise_conv2d: { in_channels: 64, kernel_size: 3, stride: 1, padding: 1, bias: false },
    depthwise_conv: { in_channels: 64, kernel_size: 3, stride: 1, padding: 1, bias: false },
    conv_transpose2d: { in_channels: 256, out_channels: 128, kernel_size: 4, stride: 2, padding: 1 },
    transposed_conv: { in_channels: 256, out_channels: 128, kernel_size: 4, stride: 2, padding: 1 },
    se_block: { in_channels: 256, reduction: 16 },
    max_pool: { kernel_size: 2, stride: 2, padding: 0 },
    avg_pool: { kernel_size: 2, stride: 2, padding: 0 },
    global_pool: { type: 'avg' },
    upsample: { scale_factor: 2, mode: 'bilinear' },
    downsample: { in_channels: 64, out_channels: 128, stride: 2, method: 'conv' },
    downsample_2d: { channels: 320, use_conv: true, stride: 2 },
    pixelnorm: { eps: 1e-8 },
    diff_resblock: { in_channels: 64, out_channels: 64, stride: 1 },
    basic_block: { in_channels: 64, out_channels: 64, stride: 1, downsample: false, activation: 'relu' },
    bottleneck_block: { in_channels: 256, out_channels: 256, stride: 1, expansion: 4, activation: 'relu' },
    mbconv_block: { in_channels: 32, out_channels: 64, expand_ratio: 4, stride: 1, se_ratio: 0.25, kernel_size: 3, activation: 'silu' },
    dense_block: { num_layers: 6, in_channels: 64, growth_rate: 32, bn_size: 4, drop_rate: 0.0 },

    // ── Activations ──────────────────────────────────────────
    relu: {},
    gelu: { approximate: 'tanh' },
    silu: {},
    tanh: {},
    sigmoid: {},
    leaky_relu: { negative_slope: 0.2 },

    // ── MoE ──────────────────────────────────────────────────
    gate: { num_experts: 8, d_model: 768, top_k: 2, router_bias: false },
    router_linear: { num_experts: 8, d_model: 768, top_k: 2, router_type: 'top_k', router_bias: false, capacity_factor: 1.25, dropout: 0.0 },
    router_softmax: { num_experts: 8, d_model: 768, router_bias: false, dropout: 0.0 },
    expert: { d_model: 768, d_ff: 3072, num_experts: 8, dropout: 0.0 },
    expert_linear: { d_model: 768, d_ff: 3072, num_experts: 8, bias: true, dropout: 0.0, activation: 'gelu' },
    expert_gated_ffn: { d_model: 768, d_ff: 14336, num_experts: 8, gate_type: 'swiglu', bias: false, dropout: 0.0, activation: 'silu' },
    expert_combine: { combination_type: 'weighted_sum' },
    moe_block: { num_experts: 8, top_k: 2, d_model: 768, expert_d_ff: 3072, expert_type: 'gated_ffn', router_type: 'top_k', router_bias: false, capacity_factor: 1.25, aux_loss_coef: 0.01, shared_experts: 0 },
    moe_layer: { num_experts: 64, top_k: 1, d_model: 768, expert_d_ff: 3072, expert_type: 'ffn', router_type: 'softmax', shared_experts: 0, aux_loss_coef: 0.01, z_loss_coef: 0.0001, capacity_factor: 1.25, dropout: 0.0 },

    // ── SSM ──────────────────────────────────────────────────
    mamba_block: { d_model: 768, d_state: 16, d_conv: 4, expand: 2, dt_rank: 'auto', bias: false },
    s4_block: { d_model: 512, d_state: 64, ssm_type: 's4d', init_method: 'hippo', normalization: 'layer_norm' },
    s6_block: { d_model: 768, d_state: 16, expand: 2, dt_rank: 48, conv_kernel: 4, bias: false },
    conv1d: { in_channels: 768, out_channels: 768, kernel_size: 4, groups: 768 },

    // ── RNN ──────────────────────────────────────────────────
    lstm: { input_size: 256, hidden_size: 512, num_layers: 1, batch_first: true, dropout: 0.0 },
    lstm_cell: { input_size: 256, hidden_size: 512, num_layers: 1, batch_first: true, dropout: 0.0 },
    gru: { input_size: 256, hidden_size: 512, num_layers: 1, batch_first: true, dropout: 0.0 },
    gru_cell: { input_size: 256, hidden_size: 512, num_layers: 1, batch_first: true, dropout: 0.0 },
    bilstm: { input_size: 256, hidden_size: 256, num_layers: 2, batch_first: true, dropout: 0.2, bidirectional: true },
    bigru: { input_size: 256, hidden_size: 256, num_layers: 2, batch_first: true, dropout: 0.2, bidirectional: true },
    bahdanau_attention: { encoder_size: 512, decoder_size: 512, attention_dim: 256 },
    seq2seq_attention: { attentionType: 'bahdanau' },
    rnn_cell: { hiddenSize: 256, numLayers: 1, activation: 'tanh' },

    // ── GNN ──────────────────────────────────────────────────
    gcn_conv: { in_channels: 16, out_channels: 64, improved: false, cached: false, add_self_loops: true, normalize: true, bias: true },
    gat_conv: { in_channels: 16, out_channels: 8, heads: 8, concat: true, negative_slope: 0.2, dropout: 0.6, add_self_loops: true, bias: true },
    sage_conv: { in_channels: 16, out_channels: 64, normalize: true, concat: false, root_weight: true, bias: true },
    edge_conv: { nn: 'mlp', aggr: 'max' },
    graphnorm: { in_channels: 64, eps: 1e-5 },
    graph_norm: { in_channels: 64, eps: 1e-5 },
    global_mean_pool: {},
    global_max_pool: {},
    global_add_pool: {},
    classification_head: { d_model: 768, num_classes: 10, pooling: 'mean' },

    // ── Diffusion ────────────────────────────────────────────
    timestep_embedding: { channels: 320, time_embed_dim: 1280, act_fn: 'silu' },
    noise_scheduler: { num_train_timesteps: 1000, beta_start: 0.0001, beta_end: 0.02, beta_schedule: 'scaled_linear' },
    vae_encoder: { in_channels: 3, latent_channels: 4, base_channels: 128, ch_mult: '1,2,4,4', num_res_blocks: 2, dropout: 0.0 },
    vae_decoder: { latent_channels: 4, out_channels: 3, base_channels: 128, ch_mult: '1,2,4,4', num_res_blocks: 2 },
    unet_block: { in_channels: 64, out_channels: 128, skip_channels: 64, activation: 'silu' },
    residual_block: { in_channels: 256, out_channels: 256, d_time_emb: 1280, dropout: 0.0, activation: 'silu' },
    patchify: { patch_size: 2, in_channels: 4, embed_dim: 1152 },
    depatchify: { patch_size: 2, out_channels: 8 },
    spatial_transformer: { in_channels: 320, n_heads: 8, d_head: 40, depth: 1, context_dim: 768 },
    dit_block: { input_size: 32, patch_size: 2, in_channels: 4, embed_dim: 1152, depth: 28, num_heads: 16, mlp_ratio: 4.0, learn_sigma: true },

    // ── SNN ──────────────────────────────────────────────────
    rate_encoder: { encoding: 'rate', timesteps: 100 },
    latency_encoder: { encoding: 'latency', timesteps: 50 },
    lif_neuron: { tau: 10.0, threshold: 1.0, reset: 0.0, decay: 0.9 },
    leaky_neuron: { tau: 5.0, threshold: 0.9, reset: 0.0 },
    synaptic_layer: { in_features: 256, out_features: 256, delay: 1 },
    stdp_synapse: { in_features: 256, out_features: 256, a_plus: 0.01, a_minus: 0.012, tau_plus: 20, tau_minus: 20 },
    gan_noise_z: { dim: 100, distribution: 'gaussian' },
    dcgan_generator_block: { out_channels: 256, kernel: 4, stride: 2, padding: 1, activation: 'relu' },
    dcgan_discriminator_block: { out_channels: 256, kernel: 4, stride: 2, padding: 1, activation: 'leaky_relu' },
    stylegan_synthesis_block: { resolution: 1024, channels: 512 },

    // ── RL ───────────────────────────────────────────────────
    policy_head: { d_model: 768, action_dim: 32000, distribution: 'categorical' },
    value_head: { d_model: 768, hidden_dim: 256 },
    advantage_stream: { d_model: 768, action_dim: 18, hidden_dim: 256 },

    // ── Custom / Experimental ────────────────────────────────
    custom: { flops_formula: 'B*N*D', memory_formula: 'B*N*D*4' },
};

/**
 * Returns default params for a given catalogue block type string.
 * Falls back to empty object if the type is not yet mapped.
 */
export function getBlockDefaults(blockType: string): Params {
    return DEFAULTS[blockType] ?? {};
}
