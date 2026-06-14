import { CanvasNode, Connection, LayerType, NodeGroup } from '@/types/architecture.ts';
import { ArchitectureFamily } from '@/types/plugins.ts';

/**
 * NEURAX IR v3 — Strict TopologyInput JSON
 * Must parse into Rust struct TopologyInput and pass
 * canonicalization + shape inference + cost modeling.
 */

export interface NeuraxLayer {
  id: string;
  layer_type: string;
  input_shape: number[];
  output_shape: number[];
  params: Record<string, any>;
  custom_equations?: Record<string, string>;
}

export interface NeuraxGlobalParams {
  hidden_size?: number;
  num_layers?: number;
  vocab_size?: number;
  sequence_length?: number;
  num_heads?: number;
  head_dim?: number;
  ffn_dim?: number;
  num_experts?: number;
  top_k?: number;
  [key: string]: any;
}

export interface NeuraxModel {
  name: string;
  type: string;
  global_params: NeuraxGlobalParams;
  layers: NeuraxLayer[];
}

export interface NeuraxTraining {
  batch_size: number;
  optimizer?: string;
  learning_rate?: number;
  num_epochs?: number;
  sequence_length?: number;
  precision: string;
  gradient_checkpointing?: boolean;
  zero_stage?: number;
  max_steps?: number;
  warmup_steps?: number;
  parallelism?: {
    data_parallel: number;
    tensor_parallel: number;
    pipeline_parallel: number;
  };
}

export interface NeuraxData {
  dataset_size?: number;
  vocab_size?: number;
  num_classes?: number;
}

export interface NeuraxGpu {
  name: string;
  count: number;
  memory_gb?: number;
  tflops_fp16?: number;
  tflops_fp32?: number;
}

export interface NeuraxHardware {
  gpus: NeuraxGpu[];
  interconnect?: string;
  interconnect_bandwidth_gbs?: number;
}

export interface NeuraxIR {
  schema_version: string;
  model: NeuraxModel;
  training: NeuraxTraining;
  hardware: NeuraxHardware;
  data?: NeuraxData;
}

// Legacy compat for blocks-based format (internal use only)
export interface NeuraxBlock {
  id: string;
  type: string;
  inputs: string[];
  outputs: string[];
  params: Record<string, any> | null;
  ui_node_type?: LayerType;
  repeat?: number;
  trainable?: boolean;
  sub_blocks?: NeuraxBlock[];
  comment?: string;
}

export interface NeuraxEnv {
  hw: string;
  prec: 'fp32' | 'fp16' | 'bf16' | 'int8' | 'int4';
  batch: number;
  seed?: number;
  device?: string;
  compile?: boolean;

  // Transformers / LLM common
  seq?: number;
  vocab?: number;
  d?: number;
  h?: number;
  hd?: number;
  ff?: number;
  L?: number;
  kv?: number;
  bias?: boolean;
  drop?: number;
  flash?: boolean;
  rope_theta?: number;
  max_seq_len?: number;
  alibi?: boolean;
  relative_bias?: boolean;
  use_cache?: boolean;
  activation?: string;

  // CNN / Spatial
  h_img?: number;
  w_img?: number;
  cin?: number;
  cout?: number;
  norm?: string;
  act?: string;
  pool?: string;

  // ViT / DiT
  patch?: number;
  patches?: number;
  steps?: number;
  cfg?: number;
  mlp_ratio?: number;
  qkv_bias?: boolean;
  proj_drop?: number;
  attn_drop?: number;
  pos_embed?: string;
  use_flash?: boolean;

  // GNN
  num_nodes?: number;
  num_edges?: number;
  feat_dim?: number;
  out_dim?: number;
  edge_dim?: number;
  aggr?: string;
  normalize?: boolean;
  add_self_loops?: boolean;

  // RNN / SSM / SNN
  hid?: number;
  bidir?: boolean;
  state?: number;
  dt?: number;
  conv_kernel?: number;
  expand?: number;
  use_fast_path?: boolean;
  proj_size?: number;
  time?: number;
  spike?: number;

  // MoE
  exp?: number;
  topk?: number;
  expert_capacity?: number;
  shared_expert?: boolean;

  // Diffusion Base
  model_channels?: number;
  num_res_blocks?: number;
  channel_mult?: string;
  attention_resolutions?: string;
  dropout?: number;
  use_checkpoint?: boolean;
  out_channels?: number;

  // RL (legacy/custom)
  act_dim?: number;
  st_dim?: number;
  max_new_tokens?: number;
}

// Legacy compat
export type NeuraxGraph = NeuraxBlock;

/** Canonical op type mapping */
const BLOCK_TYPE_MAP: Partial<Record<LayerType, string>> = {
  input: 'Input',
  output: 'DenseProjection',
  dense: 'DenseProjection',
  conv2d: 'Conv2D',
  relu: 'ReLU',
  gelu: 'GELU',
  attention: 'Attention',
  residual: 'ResidualAdd',
  transformer: 'Group',
  linear_projection: 'DenseProjection',
  lora_linear: 'DenseProjection',
  dora_linear: 'DenseProjection',
  q_projection: 'DenseProjection',
  k_projection: 'DenseProjection',
  v_projection: 'DenseProjection',
  qkv_combined: 'DenseProjection',
  mqa_projection: 'DenseProjection',
  gqa_projection: 'DenseProjection',
  embedding: 'Embedding',
  token_embedding: 'Embedding',
  conv1d: 'Conv1D',
  conv3d: 'Conv3D',
  depthwise_conv: 'DepthwiseSep',
  transposed_conv: 'TransposeConv',
  attention_score: 'ScaledDotProductAttn',
  attention_aggregation: 'ScaledDotProductAttn',
  cross_attention: 'CrossAttention',
  mha_attention: 'Attention',
  mqa_attention: 'ScaledDotProductAttn',
  gqa_attention: 'ScaledDotProductAttn',
  mla_attention: 'ScaledDotProductAttn',
  sliding_window_attention: 'ScaledDotProductAttn',
  dilated_attention: 'sdpa',
  sparse_attention: 'sdpa',
  linear_attention: 'sdpa',
  flash_attention: 'sdpa',
  flex_attention: 'sdpa',
  max_pool: 'MaxPool',
  avg_pool: 'AvgPool',
  adaptive_pool: 'AdaptivePool',
  global_pool: 'GlobalPool',
  layernorm: 'LayerNorm',
  rmsnorm: 'RmsNorm',
  batchnorm: 'BatchNorm',
  groupnorm: 'GroupNorm',
  instancenorm: 'InstanceNorm',
  ffn_standard: 'Mlp',
  ffn_gated: 'Mlp',
  ffn_parallel: 'Mlp',
  moe_block: 'TopKRouter',
  pos_absolute: 'Opaque',
  pos_rope: 'Opaque',
  pos_alibi: 'Opaque',
  pos_relative_bias: 'Opaque',
  pos_xpos: 'Opaque',
  pos_fire: 'Opaque',
  residual_add: 'ResidualAdd',
  concat: 'Concat',
  flatten: 'Flatten',
  reshape: 'Reshape',
  dropout: 'Dropout',
  layer_stack: 'Opaque',
  gradient_checkpoint: 'Opaque',
  lm_head: 'LMHead',
  classification_head: 'DenseProjection',
  router_linear: 'TopKRouter',
  router_softmax: 'SoftmaxRouter',
  noisy_topk_router: 'TopKRouter',
  expert_choice_router: 'ExpertChoiceRouter',
  non_trainable_router: 'FixedRouter',
  product_key_router: 'ProductKeyRouter',
  hierarchical_router: 'HierarchicalRouter',
  sinkhorn_router: 'SinkhornRouter',
  expert_linear: 'ExpertFFN',
  expert_gated_ffn: 'ExpertFFN',
  expert_multihead: 'ExpertMultiHead',
  expert_scalar: 'ExpertScalar',
  expert_memory: 'ExpertMemory',
  expert_dispatch: 'ExpertDispatch',
  expert_combine: 'ExpertCombine',
  output_combination: 'OutputCombination',
  concat_projection: 'ConcatProjection',
  attention_pooling: 'AttentionPooling',
  shared_expert: 'SharedExpert',
  load_balancing_loss: 'Opaque',
  expert_capacity_limit: 'Opaque',
  z_loss: 'Opaque',
  router_regularization: 'Opaque',
  jitter_noise: 'Opaque',
  capacity_computation: 'Opaque',
  mask_generation: 'Opaque',
  moe_layer: 'MoELayer',
  moa_block: 'MoABlock',
  fine_grained_moe: 'FineGrainedMoE',
  soft_moe: 'SoftMoE',
  peer_layer: 'PEERLayer',
  hierarchical_moe: 'HierarchicalMoE',
  ssm_discretize: 'SSMDiscretize',
  delta_computation: 'DeltaComputation',
  state_matrix_a: 'S4Layer',
  state_matrix_b: 'S4Layer',
  state_matrix_c: 'S4Layer',
  s4_block: 'S4Layer',
  s5_block: 'S5Layer',
  s6_block: 'S6Layer',
  lru_block: 'LRULayer',
  linoss_block: 'LinOSSLayer',
  selective_scan: 'SelectiveScan',
  input_dependent_timescale: 'Opaque',
  s7_selection: 'Opaque',
  serpent_selection: 'Opaque',
  scan_1d: 'Opaque',
  scan_2d: 'Opaque',
  scan_multidirectional: 'Opaque',
  scan_spiral: 'Opaque',
  scan_diagonal: 'Opaque',
  causal_conv1d: 'CausalConv1D',
  ssm_in_proj: 'DenseProjection',
  ssm_out_proj: 'DenseProjection',
  delta_proj: 'DenseProjection',
  bc_proj: 'DenseProjection',
  glu_block: 'Opaque',
  hadamard_product: 'Opaque',
  vss_block: 'VSSBlock',
  rssg_block: 'RSSGBlock',
  basic_layer_ssm: 'Opaque',
  stg_mamba_block: 'Opaque',
  dual_path_mamba: 'Opaque',
  mamba_mixer: 'MambaMixer',
  multiscale_ssm: 'Opaque',
  parallel_differencing: 'Opaque',
  spiking_ssm: 'Opaque',
  neuromorphic_activation: 'Opaque',
  h3_block: 'H3Block',
  hyena_conv: 'HyenaConv',
  gated_ssm: 'GatedSSM',
  neural_cde: 'NeuralCDE',
  a_parameterization: 'Opaque',
  hybrid_ssm_attn: 'HybridLayer',
  multimodal_mamba: 'MultiModalMamba',
  scan_operator: 'Opaque',
  scan_block: 'Opaque',
  state_reset: 'Opaque',
  ssm_layernorm: 'LayerNorm',
  ssm_output_head: 'DenseProjection',
  forecasting_head: 'DenseProjection',
  message_aggregate: 'MessageAggregate',
  graph_conv: 'GraphConv',
  graph_attention: 'GraphAttention',
  graph_readout: 'GraphReadout',
  // GNN — Message Passing
  gcn_conv: 'GCNConv',
  sage_conv: 'SAGEConv',
  gat_conv: 'GATConv',
  gat_v2_conv: 'GATv2Conv',
  gin_conv: 'GINConv',
  cheb_conv: 'ChebConv',
  mf_conv: 'MFConv',
  rgcn_conv: 'RGCNConv',
  tag_conv: 'TAGConv',
  arma_conv: 'ARMAConv',
  sg_conv: 'SGConv',
  appnp: 'APPNP',
  dna_conv: 'DNAConv',
  cluster_gcn_conv: 'ClusterGCNConv',
  // GNN — Edge Features
  edge_conv: 'EdgeConv',
  nn_conv: 'NNConv',
  // GNN — Pooling
  topk_pooling: 'TopKPooling',
  sag_pool: 'SAGPool',
  edge_pooling: 'EdgePooling',
  asa_pooling: 'ASAPooling',
  global_max_pool: 'GlobalMaxPool',
  global_mean_pool: 'GlobalMeanPool',
  global_add_pool: 'GlobalAddPool',
  // GNN — Normalization & Regularization
  graph_norm: 'GraphNorm',
  pair_norm: 'PairNorm',
  mean_subtraction_norm: 'MeanSubtractionNorm',
  edge_dropout: 'EdgeDropout',
  // GNN — Heterogeneous
  hetero_conv: 'HeteroConv',
  hgt_conv: 'HGTConv',
  han_conv: 'HANConv',
  // GNN — Temporal
  tgcn: 'TGCN',
  stgcn: 'STGCN',
  dy_gr_encoder: 'DyGrEncoder',
  // GNN — Readout
  global_attention_readout: 'GlobalAttentionReadout',
  set2set_readout: 'Set2SetReadout',
  graph_readout_general: 'GraphReadout',
  // GNN — Positioning & Structural
  random_walk_pe: 'RandomWalkPE',
  laplacian_pe: 'LaplacianPE',
  distance_encoding: 'DistanceEncoding',
  // GNN — Transformation
  add_self_loops: 'AddSelfLoops',
  t_normalize_features: 'NormalizeFeatures',
  t_to_undirected: 'ToUndirected',
  // GNN — Advanced
  mpnn_structure: 'MPNN',
  deeper_gcn: 'DeeperGCN',
  rnn_cell: 'RNNCell',
  lstm_cell: 'LSTMCell',
  gru_cell: 'GRUCell',
  // GAN — Styles & Modulation
  noise_injection: 'Opaque',
  style_modulation: 'Opaque',
  modulated_conv2d: 'Opaque',
  adain_style: 'InstanceNorm',
  weight_demodulation: 'Opaque',
  style_demodulation: 'Opaque',
  style_projection: 'DenseProjection',
  // Diffusion — Encoders
  vae_encoder: 'VAEEncoder',
  vae_decoder: 'VAEDecoder',
  image_encoder: 'ImageEncoder',
  text_encoder: 'TextEncoder',
  class_encoder: 'ClassEncoder',
  // Diffusion — Denoisers
  unet_2d_cond: 'UNet2DConditionModel',
  unet_model: 'UNetModel',
  dit_block: 'DiTBlock',
  mmdit_block: 'MMDiTBlock',
  flag_dit: 'FlagDiT',
  next_dit: 'NextDiT',
  // Diffusion — U-Net Base
  diff_resblock: 'DiffResBlock',
  spatial_transformer: 'SpatialTransformer',
  basic_transformer_block: 'BasicTransformerBlock',
  downsample_2d: 'Downsample2D',
  upsample_2d: 'Upsample2D',
  timestep_embed_seq: 'Opaque',
  cross_attn_down_block: 'CrossAttnDownBlock2D',
  cross_attn_up_block: 'CrossAttnUpBlock2D',
  down_block_2d: 'DownBlock2D',
  up_block_2d: 'UpBlock2D',
  unet_mid_block: 'UNetMidBlock2DCrossAttn',
  // Diffusion — Conditioning
  timestep_embedding: 'TimestepEmbedding',
  timestep_projection: 'TimestepProjection',
  sinusoidal_timestep_embed: 'SinusoidalTimestepEmbedding',
  text_projection: 'DenseProjection',
  clip_embedding: 'CLIPEmbedding',
  class_embedding: 'ClassEmbedding',
  classifier_free_guidance: 'Opaque',
  image_projection: 'DenseProjection',
  controlnet_block: 'ControlNetBlock',
  ip_adapter: 'Opaque',
  // Diffusion — Noise & Scheduling
  noise_schedule: 'Opaque',
  ddpm_scheduler: 'Opaque',
  ddim_scheduler: 'Opaque',
  euler_scheduler: 'Opaque',
  dpm_solver: 'Opaque',
  flow_match_scheduler: 'Opaque',
  gaussian_noise: 'Opaque',
  forward_diffusion: 'Opaque',
  reverse_diffusion: 'Opaque',
  latent_diffusion_step: 'Opaque',
  // Diffusion — Normalization
  ada_group_norm: 'AdaGroupNorm',
  sandwich_norm: 'Opaque',
  // Diffusion — SD/LDM
  autoencoder_kl: 'AutoencoderKL',
  // Diffusion — PixArt/DiT
  adaln_single: 'AdaLNSingle',
  patchify: 'Patchify',
  depatchify: 'Depatchify',
  // Diffusion — SD3/Flux
  double_stream_block: 'DoubleStreamBlock',
  single_stream_block: 'SingleStreamBlock',
  rectified_flow: 'Opaque',
  // Diffusion — Lumina
  rope_3d: 'Opaque',
  freq_aware_rope: 'Opaque',
  time_aware_rope: 'Opaque',
  context_drop: 'Opaque',
  // Diffusion — Output
  diff_output_layer: 'DenseProjection',
  final_conv: 'Conv2D',
  image_decoder: 'VAEDecoder',
  // Diffusion — Action-Conditioned
  action_encoder: 'DenseProjection',
  action_cond_unet: 'UNet2DConditionModel',
  dynamics_predictor: 'DenseProjection',
  world_model: 'Opaque',
  // Diffusion — Emerging
  reg_injector: 'Opaque',
  self_attention_guidance: 'Opaque',
  cascade_multiscale: 'Opaque',
  patch_ddm: 'Opaque',
  // Diffusion — Legacy
  time_embedding: 'Opaque',
  unet_downsample: 'Opaque',
  unet_upsample: 'Opaque',
  policy_head: 'DenseProjection',
  value_head: 'DenseProjection',
  advantage_stream: 'DenseProjection',
  lif_neuron: 'Opaque',
  spike_encoder: 'Opaque',
  stdp_synapse: 'Opaque',
  // CNN — Convolutions
  pointwise_conv: 'Conv2D',
  separable_conv: 'SeparableConv',
  deformable_conv: 'DeformableConv',
  large_kernel_conv: 'Conv2D',
  mixed_kernel_conv: 'Conv2D',
  dynamic_conv: 'DynamicConv',
  sparse_conv: 'SparseConv',
  involution: 'Involution',
  gated_conv: 'Conv2D',
  // CNN — Pooling
  roi_pool: 'ROIPool',
  roi_align: 'ROIAlign',
  // CNN — Normalization
  sync_batchnorm: 'BatchNorm',
  switchable_norm: 'BatchNorm',
  filter_response_norm: 'BatchNorm',
  adaptive_instance_norm: 'InstanceNorm',
  conditional_batchnorm: 'BatchNorm',
  spectral_norm: 'Opaque',
  // CNN — Structure
  permute: 'Opaque',
  stochastic_depth: 'Opaque',
  skip_connection: 'ResidualAdd',
  upsample: 'Upsample',
  // CNN — Output heads
  detection_head: 'DenseProjection',
  segmentation_head: 'DenseProjection',
  mask_head: 'DenseProjection',
  roi_head: 'DenseProjection',
  rpn_head: 'DenseProjection',
  // CNN — Residual & Dense
  basic_block: 'ResBlock',
  bottleneck_block: 'BottleneckBlock',
  preact_block: 'ResBlock',
  dense_layer: 'DenseLayer',
  dense_block: 'DenseBlock',
  transition_layer: 'TransitionLayer',
  inverted_bottleneck: 'InvertedBottleneck',
  convnext_block: 'ConvNeXtBlock',
  mbconv_block: 'MBConvBlock',
  fused_mbconv: 'MBConvBlock',
  // CNN — Attention
  se_layer: 'SELayer',
  cbam_layer: 'CBAMLayer',
  eca_layer: 'ECALayer',
  non_local_block: 'NonLocalBlock',
  gc_net: 'GCNet',
  coord_conv: 'CoordConv',
  self_attention_2d: 'sdpa',
  // CNN — ViT
  patch_embed: 'PatchEmbed',
  class_token: 'Opaque',
  transformer_encoder_block: 'TransformerBlock',
  cvt_block: 'CvTBlock',
  levit_block: 'LeViTBlock',
  // CNN — Skeleton
  stem_block: 'Conv2D',
  stage_block: 'Opaque',
  downsample_block: 'Conv2D',
  upsample_block: 'Upsample',
  // CNN — Segmentation & Detection
  unet_block: 'Opaque',
  fpn_block: 'FPNBlock',
  anchor_generator: 'Opaque',
  // CNN — Mixer & Future
  mlp_mixer: 'MLPMixer',
  res_mlp: 'DenseProjection',
  conv_mixer: 'ConvMixer',
  adaptive_inference_block: 'Opaque',
  nas_cell: 'Opaque',
  // GAN — Primary Components
  gan_noise_z: 'Input',
  gan_noise_w: 'Input',
  gan_label_embedding: 'Embedding',
  dcgan_generator_block: 'TransposeConv',
  dcgan_discriminator_block: 'Conv2D',
  torgb_layer: 'Conv2D',
  fromrgb_layer: 'Conv2D',
  // StyleGAN
  mapping_network: 'Group',
  stylegan_synthesis_block: 'Group',
  stylegan_res_block: 'ResBlock',
  progressive_growing_step: 'Opaque',
  fading_layer: 'Opaque',
  // Translation
  cyclegan_block: 'Group',
  pix2pix_generator: 'Group',
  patch_gan_discriminator: 'Conv2D',
  pix2pix_discriminator: 'Conv2D',
  multiscale_discriminator: 'Group',
  // Video & Temporal
  // Video & Temporal
  video_generator_3d: 'Group',
  temporal_discriminator: 'Group',
  optic_flow_loss: 'Opaque',
  frame_interpolation_block: 'Group',
  recurrent_gan_cell: 'RNNCell',
  // BigGAN & High-Res
  biggan_res_block: 'ResBlock',
  self_attention_gan: 'sdpa',
  // Special Apps
  super_res_block: 'Group',
  esrgan_dense_block: 'DenseBlock',
  // Losses & Reg
  gan_adversarial_loss: 'Opaque',
  gan_minimax_loss: 'Opaque',
  gan_non_saturating_loss: 'Opaque',
  perceptual_loss: 'Opaque',
  pixel_wise_loss: 'Opaque',
  total_variation_loss: 'Opaque',
  id_loss: 'Opaque',
  r1_regularization: 'Opaque',
  r2_regularization: 'Opaque',
  lazy_regularization: 'Opaque',
  gradient_penalty: 'Opaque',
  // Metrics
  fid_metric_node: 'Opaque',
  is_metric_node: 'Opaque',
  // --- Remaining GAN Mappings ---
  // --- Remaining GAN Mappings ---
  gan_discriminator: 'Group',
  upsample_conv_block: 'Group',
  pixel_shuffle: 'Opaque',
  pixel_unshuffle: 'Opaque',
  checkerboard_removal: 'Opaque',
  conditional_embedding: 'Embedding',
  label_conditioning: 'Opaque',
  infogan_latent_code: 'Input',
  q_network_head: 'Group',
  mutual_info_loss: 'Opaque',
  auxiliary_classifier: 'Group',
  wgan_critic: 'Group',
  lipschitz_constraint: 'Opaque',
  weight_clipping: 'Opaque',
  lsgan_loss: 'Opaque',
  hinge_loss: 'Opaque',
  wasserstein_distance: 'Opaque',
  mapping_linear: 'DenseProjection',
  non_local_gan: 'Group',
  spatiotemporal_conv: 'Conv3D',
  flow_warping_layer: 'Opaque',
  equalized_lr_linear: 'DenseProjection',
  equalized_lr_conv: 'Conv2D',
  blur_filter: 'Opaque',
  antialias_upsample: 'Opaque',
  noise_broadcast: 'Opaque',
  vgg_feature_extractor: 'Group',
  orthogonal_reg: 'Opaque',
  spectral_norm_wrapper: 'Opaque',
  truncation_trick: 'Opaque',
  shared_residual_block: 'ResBlock',
  text_to_image_fusion: 'Opaque',
  mask_conditioned_gan: 'Group',
  medical_gan_block: 'Group',
  domain_adaptation_layer: 'Opaque',
  latent_interpolation: 'Opaque',
  latent_traversal: 'Opaque',
  eigen_discovery: 'Opaque',
  gan_inversion: 'Opaque',
  pivotal_tuning_step: 'Opaque',
  r3gan_block: 'Group',
  diffusion_gan_hybrid: 'Group',
  any_res_gan_block: 'Group',
  lightweight_gan_block: 'Group',
  filtered_lrelu: 'Opaque',
  fourier_features: 'Opaque',
  transformed_conv2d: 'Conv2D',
  config_independent_conv: 'Conv2D',
  sinc_filter: 'Opaque',
};

// ─── camelCase → snake_case canonicalization ───────────────────────

const PARAM_RENAME_MAP: Record<string, string> = {
  outFeatures: 'out_features',
  inFeatures: 'in_features',
  vocabSize: 'vocab_size',
  dModel: 'd_model',
  hiddenSize: 'hidden_size',
  inputSize: 'input_size',
  numLayers: 'num_layers',
  numHeads: 'num_heads',
  headDim: 'head_dim',
  topK: 'top_k',
  useBias: 'use_bias',
  inChannels: 'in_channels',
  outChannels: 'out_channels',
  kernelSize: 'kernel_size',
  stateSize: 'state_size',
  ffnDim: 'ffn_dim',
  normalizedShape: 'normalized_shape',
  numFeatures: 'num_features',
  numGroups: 'num_groups',
  numChannels: 'num_channels',
  batchFirst: 'batch_first',
  projSize: 'proj_size',
  startDim: 'start_dim',
  attentionType: 'attention_type',
  aPlus: 'a_plus',
  aMinus: 'a_minus',
  tauPlus: 'tau_plus',
  tauMinus: 'tau_minus',
  numActions: 'action_dim',
};

/** Converts camelCase params to NEURAX snake_case */
function canonicalizeParams(params: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(params)) {
    const canonical = PARAM_RENAME_MAP[key] ?? key;
    out[canonical] = value;
  }
  return out;
}

/** For sdpa blocks, ensure `num_heads` not `heads`, and add in_features */
function fixSdpaParams(params: Record<string, unknown>, inputDim: number | null): Record<string, unknown> {
  const out = { ...params };
  // Rename heads → num_heads
  if ('heads' in out && !('num_heads' in out)) {
    out.num_heads = out.heads;
    delete out.heads;
  }
  // Infer in_features from dim or d_model or upstream
  if (!('in_features' in out)) {
    if ('dim' in out) {
      out.in_features = out.dim;
    } else if ('d_model' in out) {
      out.in_features = out.d_model;
    } else if (inputDim !== null) {
      out.in_features = inputDim;
    }
  }
  return out;
}

/** Embedding: must use vocab_size + d_model, never dim/embedding_dim */
function fixEmbeddingParams(params: Record<string, unknown>): Record<string, unknown> {
  const out = { ...params };
  // Rewrite forbidden aliases → d_model
  for (const alias of ['dim', 'embedding_dim', 'model_dim']) {
    if (alias in out && !('d_model' in out)) {
      out.d_model = out[alias];
      delete out[alias];
    }
  }
  return out;
}

function deleteParamKeys(params: Record<string, unknown>, keys: string[]): void {
  for (const key of keys) {
    delete params[key];
  }
}

/** DenseProjection / LMHead: must have in_features + out_features, never units/dim/d_model */
function fixDenseParams(
  nodeType: LayerType,
  params: Record<string, unknown>,
  inputDim: number | null,
): Record<string, unknown> {
  const out = { ...params };

  // Map forbidden aliases → correct keys
  if ('units' in out && !('out_features' in out)) {
    out.out_features = out.units;
    delete out.units;
  }
  if (
    'dim' in out &&
    !('out_features' in out) &&
    !['classification_head', 'policy_head', 'value_head', 'advantage_stream'].includes(nodeType)
  ) {
    out.out_features = out.dim;
    delete out.dim;
  }

  switch (nodeType) {
    case 'classification_head':
      if ('d_model' in out && !('in_features' in out)) {
        out.in_features = out.d_model;
      }
      if ('num_classes' in out && !('out_features' in out)) {
        out.out_features = out.num_classes;
      }
      deleteParamKeys(out, ['d_model', 'num_classes', 'pooling']);
      break;
    case 'policy_head':
      if ('d_model' in out && !('in_features' in out)) {
        out.in_features = out.d_model;
      }
      if ('action_dim' in out && !('out_features' in out)) {
        out.out_features = out.action_dim;
      }
      deleteParamKeys(out, ['d_model', 'action_dim', 'distribution']);
      break;
    case 'value_head':
      if ('d_model' in out && !('in_features' in out)) {
        out.in_features = out.d_model;
      }
      if (!('out_features' in out)) {
        out.out_features = 1;
      }
      deleteParamKeys(out, ['d_model', 'hidden_dim']);
      break;
    case 'advantage_stream':
      if ('d_model' in out && !('in_features' in out)) {
        out.in_features = out.d_model;
      }
      if ('action_dim' in out && !('out_features' in out)) {
        out.out_features = out.action_dim;
      }
      deleteParamKeys(out, ['d_model', 'action_dim', 'hidden_dim']);
      break;
    default:
      if ('d_model' in out && !('out_features' in out)) {
        out.out_features = out.d_model;
        delete out.d_model;
      }
      break;
  }

  // Ensure in_features exists
  if (!('in_features' in out)) {
    if ('d_model' in out) {
      out.in_features = out.d_model;
      delete out.d_model;
    } else if (inputDim !== null) {
      out.in_features = inputDim;
    }
  }

  // Fallback: if still missing out_features, try in_features
  if (!('out_features' in out) && 'in_features' in out) {
    out.out_features = out.in_features;
  }

  return out;
}

function getNumericParam(
  params: Record<string, unknown>,
  keys: string[],
): number | undefined {
  for (const key of keys) {
    const value = params[key];
    if (typeof value === 'number' && Number.isFinite(value)) {
      return value;
    }
  }
  return undefined;
}

function inferNodeOutputDim(node: CanvasNode): number | undefined {
  const params = (node.params ?? {}) as Record<string, unknown>;

  switch (node.type) {
    case 'embedding':
    case 'token_embedding':
      return getNumericParam(params, ['d_model', 'dim', 'dModel']);
    case 'layernorm':
    case 'rmsnorm':
      return getNumericParam(params, ['normalized_shape', 'd_model']);
    case 'batchnorm':
    case 'instancenorm':
      return getNumericParam(params, ['num_features']);
    case 'groupnorm':
      return getNumericParam(params, ['num_channels']);
    case 'graph_norm':
      return getNumericParam(params, ['in_channels', 'in_features']);
    case 'classification_head':
      return getNumericParam(params, ['num_classes', 'out_features', 'outFeatures']);
    case 'policy_head':
    case 'advantage_stream':
      return getNumericParam(params, ['action_dim', 'out_features', 'outFeatures']);
    case 'value_head':
      return 1;
    case 'lm_head':
      return getNumericParam(params, ['vocab_size', 'vocabSize', 'out_features', 'outFeatures']);
    case 'gat_conv': {
      const outChannels = getNumericParam(params, ['out_channels', 'outChannels']);
      const heads = getNumericParam(params, ['heads', 'num_heads']);
      if (outChannels == null) return undefined;
      return params.concat === false || heads == null ? outChannels : outChannels * heads;
    }
    default:
      return getNumericParam(params, [
        'out_features',
        'outFeatures',
        'dim',
        'd_model',
        'dModel',
        'hidden_size',
        'hiddenSize',
        'out_channels',
        'outChannels',
        'vocab_size',
      ]);
  }
}

// ─── Topological sort (Kahn's algorithm) ──────────────────────────

function topologicalSort(nodes: CanvasNode[], connections: Connection[]): string[] {
  const inDegree: Record<string, number> = {};
  const adjacency: Record<string, string[]> = {};
  const nodeIds = new Set(nodes.map(n => n.id));

  for (const id of nodeIds) {
    inDegree[id] = 0;
    adjacency[id] = [];
  }

  for (const conn of connections) {
    if (nodeIds.has(conn.from) && nodeIds.has(conn.to)) {
      adjacency[conn.from].push(conn.to);
      inDegree[conn.to]++;
    }
  }

  const queue: string[] = [];
  for (const id of nodeIds) {
    if (inDegree[id] === 0) queue.push(id);
  }

  const sorted: string[] = [];
  while (queue.length > 0) {
    const current = queue.shift()!;
    sorted.push(current);
    for (const neighbor of adjacency[current]) {
      inDegree[neighbor]--;
      if (inDegree[neighbor] === 0) queue.push(neighbor);
    }
  }

  // Append orphans
  for (const id of nodeIds) {
    if (!sorted.includes(id)) sorted.push(id);
  }

  return sorted;
}

// ─── Tensor naming ────────────────────────────────────────────────

function makeTensorName(node: CanvasNode): string {
  const base = node.name
    .toLowerCase()
    .replace(/[^a-z0-9_]/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_|_$/g, '');
  return `${base}_out`;
}

// ─── Param extraction ─────────────────────────────────────────────

const SKIP_KEYS = new Set(['shape']);

function extractParams(node: CanvasNode, blockType: string): Record<string, unknown> {
  const raw: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(node.params)) {
    if (SKIP_KEYS.has(key)) continue;
    if (key === 'activation' && value === 'none') continue;
    raw[key] = value;
  }

  // Canonicalize camelCase → snake_case
  const params = canonicalizeParams(raw);

  return params;
}

// ─── Hardware validation ──────────────────────────────────────────

const VALID_HARDWARE = new Set(['H100', 'A100', 'L40', 'V100', 'T4', 'RTX4090', 'RTX4080', 'RTX3090']);
const DEFAULT_HARDWARE = 'RTX4090';

function normalizeHardware(hw: string | null | undefined): { hardware: string; wasAutoFixed: boolean } {
  if (!hw) return { hardware: DEFAULT_HARDWARE, wasAutoFixed: true };

  // Fast path: already canonical and supported as-is
  if (VALID_HARDWARE.has(hw)) return { hardware: hw, wasAutoFixed: false };

  const s = hw.trim();
  const l = s.toLowerCase();

  // Accept canonical names with different casing
  if (l === 'rtx4090') return { hardware: 'RTX4090', wasAutoFixed: l !== 'rtx4090' };
  if (l === 'rtx4080') return { hardware: 'RTX4080', wasAutoFixed: true };
  if (l === 'rtx3090') return { hardware: 'RTX3090', wasAutoFixed: true };
  if (l === 'a100') return { hardware: 'A100', wasAutoFixed: true };
  if (l === 'h100') return { hardware: 'H100', wasAutoFixed: true };

  // Map current UI hardware ids to canonical labels expected by backend
  // Consumer GPUs
  if (l === 'rtx3090') return { hardware: 'RTX3090', wasAutoFixed: true };
  if (l === 'rtx4090') return { hardware: 'RTX4090', wasAutoFixed: true };

  // Cloud GPUs (A100 variants and H100)
  if (l === 'a100-40') return { hardware: 'A100', wasAutoFixed: true };
  if (l === 'a100-80') return { hardware: 'A100', wasAutoFixed: true };
  if (l === 'h100' || l.startsWith('h100')) return { hardware: 'H100', wasAutoFixed: true };

  // TPU
  if (l === 'tpu-v5p' || l === 'tpu v5p' || l === 'tpuv5p') return { hardware: 'TPU v5p', wasAutoFixed: true };

  // Cluster presets — encode GPU type + count in the label so backend can branch if needed
  if (l === 'cluster-64') return { hardware: 'H100-Cluster-64', wasAutoFixed: true };
  if (l === 'cluster-256') return { hardware: 'H100-Cluster-256', wasAutoFixed: true };
  if (l === 'cluster-512') return { hardware: 'H100-Cluster-512', wasAutoFixed: true };

  // Generic fallbacks for strings like "a100-xyz" etc.
  if (l.startsWith('a100')) return { hardware: 'A100', wasAutoFixed: true };
  if (l.startsWith('h100')) return { hardware: 'H100', wasAutoFixed: true };

  // Unknown — keep a safe default
  return { hardware: DEFAULT_HARDWARE, wasAutoFixed: true };
}

// ─── Env inference helpers ────────────────────────────────────────

const SEQ_REQUIRING_OPS = new Set([
  'TokenInput', 'Embedding', 'DenseProjection', 'LMHead',
  'sdpa', 'ScaledDotProductAttn', 'Group',
  'TopKRouter', 'ExpertFFN', 'layernorm', 'rmsnorm',
]);

const IMAGE_OPS = new Set(['Conv2D', 'Conv3D', 'DepthwiseSep', 'TransposeConv', 'MaxPool', 'AvgPool']);

function inferEnvRequirements(blocks: NeuraxBlock[]): { needsSeqLen: boolean; needsImage: boolean } {
  let needsSeqLen = false;
  let needsImage = false;
  for (const b of blocks) {
    if (SEQ_REQUIRING_OPS.has(b.type)) needsSeqLen = true;
    // Input block: check mode from params
    if (b.type === 'Input') {
      const p = b.params ?? {};
      if ('dim' in p || 'd_model' in p) needsSeqLen = true;
      if ('channels' in p || 'in_channels' in p) needsImage = true;
    }
    if (IMAGE_OPS.has(b.type)) needsImage = true;
  }
  return { needsSeqLen, needsImage };
}

// ─── Precision validation ─────────────────────────────────────────

const VALID_PRECISIONS = new Set(['fp32', 'fp16', 'bf16', 'int8', 'int4']);

function normalizePrecision(p: string | null): NeuraxEnv['prec'] {
  if (p && VALID_PRECISIONS.has(p)) return p as NeuraxEnv['prec'];
  return 'fp16';
}

function normalizeLayerTypeKey(value: string): string {
  return value
    .trim()
    .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
    .replace(/[\s-]+/g, '_')
    .replace(/_+/g, '_')
    .toLowerCase();
}

function toParserLayerType(blockType: string): string {
  const normalized = normalizeLayerTypeKey(blockType);

  const directTypes = new Set([
    'embedding',
    'attention',
    'mlp',
    'conv',
    'dense',
    'normalization',
    'pooling',
    'moe',
    'residual_block',
    'mbconv',
    'inception',
    'dense_block',
    'convnext_block',
    'shuffle_unit',
    'c2f',
    'detection',
    'transition',
    'mamba_block',
    's4_block',
    'h3_block',
    'state_space',
    'rwkv_block',
    'retention_block',
    'generator_block',
    'discriminator_block',
    'style_mod',
    'adain',
    'minibatch_std',
    'pixel_norm',
    'self_attention',
    'spectral_norm',
    'progressive_block',
    'lstm_block',
    'gru_block',
    'rnn_cell',
    'bidirectional',
    'encoder_block',
    'decoder_block',
    'unet_block',
    'time_embedding',
    'cross_attention',
    'down_block',
    'up_block',
    'mid_block',
    'resnet_block',
    'timestep_block',
    'condition_block',
    'noise_predictor',
    'vae_encoder',
    'vae_decoder',
    'custom',
  ]);

  if (directTypes.has(normalized)) {
    return normalized;
  }

  if (normalized === 'input' || normalized === 'output' || normalized === 'group' || normalized === 'opaque' || normalized === 'transformer') {
    return 'custom';
  }

  if (
    normalized === 'embedding' ||
    normalized === 'token_embedding' ||
    normalized === 'patch_embed' ||
    normalized === 'clip_embedding' ||
    normalized === 'class_embedding'
  ) {
    return 'embedding';
  }

  if (
    normalized === 'cross_attention' ||
    normalized === 'attention_score' ||
    normalized === 'attention_aggregation' ||
    normalized === 'mha_attention' ||
    normalized === 'mqa_attention' ||
    normalized === 'gqa_attention' ||
    normalized === 'mla_attention' ||
    normalized === 'flash_attention' ||
    normalized === 'flex_attention' ||
    normalized === 'sparse_attention' ||
    normalized === 'linear_attention' ||
    normalized === 'sliding_window_attention' ||
    normalized === 'dilated_attention' ||
    normalized === 'scaled_dot_product_attn' ||
    normalized === 'sdpa' ||
    normalized === 'graph_attention' ||
    normalized === 'global_attention_readout'
  ) {
    return normalized === 'cross_attention' ? 'cross_attention' : 'attention';
  }

  if (
    normalized === 'layernorm' ||
    normalized === 'rmsnorm' ||
    normalized === 'batchnorm' ||
    normalized === 'groupnorm' ||
    normalized === 'instancenorm' ||
    normalized === 'sync_batchnorm' ||
    normalized === 'switchable_norm' ||
    normalized === 'filter_response_norm' ||
    normalized === 'conditional_batchnorm' ||
    normalized === 'graph_norm' ||
    normalized === 'pair_norm' ||
    normalized === 'mean_subtraction_norm' ||
    normalized === 'ssm_layernorm'
  ) {
    return 'normalization';
  }

  if (normalized === 'adaptive_instance_norm' || normalized === 'adain_style') {
    return 'adain';
  }
  if (normalized === 'pixel_norm') {
    return 'pixel_norm';
  }
  if (normalized === 'spectral_norm') {
    return 'spectral_norm';
  }

  if (
    normalized === 'ffn_standard' ||
    normalized === 'ffn_gated' ||
    normalized === 'ffn_parallel' ||
    normalized === 'mlp_mixer' ||
    normalized === 'res_mlp'
  ) {
    return 'mlp';
  }

  if (
    normalized === 'max_pool' ||
    normalized === 'avg_pool' ||
    normalized === 'adaptive_pool' ||
    normalized === 'global_pool' ||
    normalized === 'roi_pool' ||
    normalized === 'roi_align' ||
    normalized === 'topk_pooling' ||
    normalized === 'sag_pool' ||
    normalized === 'edge_pooling' ||
    normalized === 'asa_pooling' ||
    normalized === 'global_max_pool' ||
    normalized === 'global_mean_pool' ||
    normalized === 'global_add_pool' ||
    normalized === 'graph_readout' ||
    normalized === 'graph_readout_general' ||
    normalized === 'set2set_readout'
  ) {
    return 'pooling';
  }

  if (
    normalized === 'moe_block' ||
    normalized === 'router_linear' ||
    normalized === 'router_softmax' ||
    normalized === 'noisy_topk_router' ||
    normalized === 'expert_choice_router' ||
    normalized === 'non_trainable_router' ||
    normalized === 'product_key_router' ||
    normalized === 'hierarchical_router' ||
    normalized === 'sinkhorn_router' ||
    normalized === 'expert_linear' ||
    normalized === 'expert_gated_ffn' ||
    normalized === 'expert_multihead' ||
    normalized === 'expert_scalar' ||
    normalized === 'expert_memory' ||
    normalized === 'expert_dispatch' ||
    normalized === 'expert_combine' ||
    normalized === 'output_combination' ||
    normalized === 'concat_projection' ||
    normalized === 'attention_pooling' ||
    normalized === 'shared_expert' ||
    normalized === 'moe_layer' ||
    normalized === 'moa_block' ||
    normalized === 'fine_grained_moe' ||
    normalized === 'soft_moe' ||
    normalized === 'peer_layer' ||
    normalized === 'hierarchical_moe'
  ) {
    return 'moe';
  }

  if (
    normalized === 'conv1d' ||
    normalized === 'conv2d' ||
    normalized === 'conv3d' ||
    normalized === 'depthwise_conv' ||
    normalized === 'transposed_conv' ||
    normalized === 'pointwise_conv' ||
    normalized === 'separable_conv' ||
    normalized === 'deformable_conv' ||
    normalized === 'large_kernel_conv' ||
    normalized === 'mixed_kernel_conv' ||
    normalized === 'dynamic_conv' ||
    normalized === 'sparse_conv' ||
    normalized === 'involution' ||
    normalized === 'gated_conv' ||
    normalized === 'causal_conv1d' ||
    normalized === 'hyena_conv'
  ) {
    return 'conv';
  }

  if (
    normalized === 'linear_projection' ||
    normalized === 'lora_linear' ||
    normalized === 'dora_linear' ||
    normalized === 'q_projection' ||
    normalized === 'k_projection' ||
    normalized === 'v_projection' ||
    normalized === 'qkv_combined' ||
    normalized === 'mqa_projection' ||
    normalized === 'gqa_projection' ||
    normalized === 'lm_head' ||
    normalized === 'classification_head' ||
    normalized === 'text_projection' ||
    normalized === 'image_projection' ||
    normalized === 'diff_output_layer' ||
    normalized === 'action_encoder' ||
    normalized === 'dynamics_predictor' ||
    normalized === 'policy_head' ||
    normalized === 'value_head' ||
    normalized === 'advantage_stream' ||
    normalized === 'forecasting_head' ||
    normalized === 'mapping_linear' ||
    normalized === 'equalized_lr_linear'
  ) {
    return 'dense';
  }

  if (
    normalized === 'basic_block' ||
    normalized === 'bottleneck_block' ||
    normalized === 'preact_block' ||
    normalized === 'stem_block' ||
    normalized === 'stage_block' ||
    normalized === 'skip_connection'
  ) {
    return 'residual_block';
  }
  if (normalized === 'mbconv_block' || normalized === 'fused_mbconv' || normalized === 'inverted_bottleneck') {
    return 'mbconv';
  }
  if (normalized === 'transition_layer') {
    return 'transition';
  }
  if (normalized === 'dense_layer') {
    return 'dense_block';
  }
  if (normalized === 'self_attention_2d') {
    return 'self_attention';
  }

  if (normalized === 'vae_encoder') {
    return 'vae_encoder';
  }
  if (normalized === 'vae_decoder' || normalized === 'image_decoder') {
    return 'vae_decoder';
  }
  if (
    normalized === 'unet_2d_cond' ||
    normalized === 'unet_model' ||
    normalized === 'autoencoder_kl'
  ) {
    return 'unet_block';
  }
  if (normalized.includes('down_block') || normalized.includes('downsample')) {
    return 'down_block';
  }
  if (normalized.includes('up_block') || normalized.includes('upsample')) {
    return 'up_block';
  }
  if (normalized.includes('mid_block')) {
    return 'mid_block';
  }
  if (normalized.includes('timestep') || normalized === 'time_embedding') {
    return normalized.includes('projection') ? 'timestep_block' : 'time_embedding';
  }
  if (
    normalized.includes('condition') ||
    normalized.includes('controlnet') ||
    normalized.includes('adapter') ||
    normalized.includes('guidance')
  ) {
    return 'condition_block';
  }
  if (normalized.includes('noise') || normalized.includes('scheduler')) {
    return 'noise_predictor';
  }
  if (normalized.includes('resnet')) {
    return 'resnet_block';
  }

  if (normalized.includes('generator')) {
    return 'generator_block';
  }
  if (normalized.includes('discriminator') || normalized.includes('critic')) {
    return 'discriminator_block';
  }
  if (normalized.includes('style') && normalized.includes('mod')) {
    return 'style_mod';
  }
  if (normalized.includes('progressive')) {
    return 'progressive_block';
  }

  if (normalized.includes('mamba')) {
    return 'mamba_block';
  }
  if (normalized.startsWith('s4')) {
    return 's4_block';
  }
  if (normalized.startsWith('h3')) {
    return 'h3_block';
  }
  if (normalized.includes('rwkv')) {
    return 'rwkv_block';
  }
  if (normalized.includes('retention')) {
    return 'retention_block';
  }
  if (normalized.includes('ssm') || normalized.includes('state_space') || normalized.includes('scan')) {
    return 'state_space';
  }

  if (normalized.includes('lstm')) {
    return 'lstm_block';
  }
  if (normalized.includes('gru')) {
    return 'gru_block';
  }
  if (normalized === 'rnn' || normalized === 'rnn_cell') {
    return 'rnn_cell';
  }
  if (normalized.includes('bidirectional') || normalized === 'bilstm' || normalized === 'bigru') {
    return 'bidirectional';
  }
  if (normalized.endsWith('encoder') || normalized.includes('_encoder_')) {
    return 'encoder_block';
  }
  if (normalized.endsWith('decoder') || normalized.includes('_decoder_')) {
    return 'decoder_block';
  }

  if (normalized.includes('router') || normalized.includes('expert') || normalized.includes('moe')) {
    return 'moe';
  }
  if (normalized.includes('attention') || normalized.endsWith('_attn')) {
    return 'attention';
  }
  if (normalized.includes('norm')) {
    return 'normalization';
  }
  if (normalized.includes('pool') || normalized.includes('readout')) {
    return 'pooling';
  }
  if (normalized.includes('conv')) {
    return 'conv';
  }
  if (normalized.includes('projection') || normalized.includes('linear') || normalized.includes('dense') || normalized.endsWith('_head')) {
    return 'dense';
  }

  return 'custom';
}

function flattenBlocks(blocks: NeuraxBlock[]): NeuraxBlock[] {
  const flat: NeuraxBlock[] = [];

  for (const block of blocks) {
    if (Array.isArray(block.sub_blocks) && block.sub_blocks.length > 0) {
      flat.push(...flattenBlocks(block.sub_blocks));
      continue;
    }
    flat.push(block);
  }

  return flat;
}

function toParserModelType(family: ArchitectureFamily): string {
  switch (family) {
    case 'rl':
    case 'snn':
      return 'rnn';
    case 'experimental':
      return 'transformer';
    default:
      return family;
  }
}

// ─── Main compiler ────────────────────────────────────────────────

export function compileToNeuraxIR(
  nodes: CanvasNode[],
  connections: Connection[],
  options: {
    modelName?: string;
    family?: ArchitectureFamily;
    hardware?: string | null;
    precision?: string | null;
    batchSize?: number | null;
    seed?: number | null;
    device?: string | null;
    useCompile?: boolean | null;

    // Transformers
    seqLen?: number | null;
    vocabSize?: number | null;
    hiddenDim?: number | null;
    numHeads?: number | null;
    headDim?: number | null;
    ffnDim?: number | null;
    numLayers?: number | null;
    kvHeads?: number | null;
    useBias?: boolean | null;
    dropout?: number | null;
    useFlash?: boolean | null;
    ropeTheta?: number | null;
    maxSeqLen?: number | null;
    useAlibi?: boolean | null;
    useRelativeBias?: boolean | null;
    useCache?: boolean | null;
    activation?: string | null;

    // CNN
    imgHeight?: number | null;
    imgWidth?: number | null;
    inChannels?: number | null;
    numClasses?: number | null;
    normType?: string | null;
    convActivation?: string | null;
    poolType?: string | null;

    // ViT / DiT
    patchSize?: number | null;
    numPatches?: number | null;
    numDenoisingSteps?: number | null;
    guidanceScale?: number | null;
    mlpRatio?: number | null;
    qkvBias?: boolean | null;
    projDrop?: number | null;
    attnDrop?: number | null;
    posEmbedType?: string | null;
    useFlashVit?: boolean | null;

    // GNN
    numNodes?: number | null;
    numEdges?: number | null;
    nodeFeatDim?: number | null;
    outDim?: number | null;
    edgeFeatDim?: number | null;
    aggrType?: string | null;
    useNormalize?: boolean | null;
    addSelfLoops?: boolean | null;

    // RNN / SSM
    hiddenSize?: number | null;
    isBidirectional?: boolean | null;
    dState?: number | null;
    dtRank?: number | null;
    convKernel?: number | null;
    expandFactor?: number | null;
    useFastPath?: boolean | null;
    projSize?: number | null;
    timesteps?: number | null;
    spikeRate?: number | null;

    // MoE
    numExperts?: number | null;
    topK?: number | null;
    expertCapacity?: number | null;
    useSharedExpert?: boolean | null;

    // RL
    actionDim?: number | null;
    stateDim?: number | null;

    // Diffusion specific
    modelChannels?: number | null;
    numResBlocks?: number | null;
    channelMult?: string | null;
    attnResolutions?: string | null;
    useCheckpoint?: boolean | null;
    outChannels?: number | null;

    maxNewTokens?: number | null;
    groups?: NodeGroup[];

    // Training config
    learningRate?: number | null;
    numEpochs?: number | null;

    // Hardware GPU detail
    gpuCount?: number | null;
    gpuMemoryGb?: number | null;

    // Data config
    datasetSize?: number | null;
  } = {}
) {
  const {
    modelName = 'NeuraxModel',
    family = 'transformer',
    hardware = 'CPU',
    precision = 'fp16',
    batchSize = 1,
    seed = null,
    device = null,
    useCompile = null,
    learningRate = null,
    numEpochs = null,
    gpuCount = null,
    gpuMemoryGb = null,
    datasetSize = null,

    seqLen = null,
    vocabSize = null,
    hiddenDim = null,
    numHeads = null,
    headDim = null,
    ffnDim = null,
    numLayers = null,
    kvHeads = null,
    useBias = null,
    dropout = null,
    useFlash = null,
    ropeTheta = null,
    maxSeqLen = null,
    useAlibi = null,
    useRelativeBias = null,
    useCache = null,
    activation = null,

    imgHeight = null,
    imgWidth = null,
    inChannels = null,
    numClasses = null,
    normType = null,
    convActivation = null,
    poolType = null,

    patchSize = null,
    numPatches = null,
    numDenoisingSteps = null,
    guidanceScale = null,
    mlpRatio = null,
    qkvBias = null,
    projDrop = null,
    attnDrop = null,
    posEmbedType = null,
    useFlashVit = null,

    numNodes = null,
    numEdges = null,
    nodeFeatDim = null,
    outDim = null,
    edgeFeatDim = null,
    aggrType = null,
    useNormalize = null,
    addSelfLoops = null,

    hiddenSize = null,
    isBidirectional = null,
    dState = null,
    dtRank = null,
    convKernel = null,
    expandFactor = null,
    useFastPath = null,
    projSize = null,
    timesteps = null,
    spikeRate = null,

    numExperts = null,
    topK = null,
    expertCapacity = null,
    useSharedExpert = null,

    actionDim = null,
    stateDim = null,

    modelChannels = null,
    numResBlocks = null,
    channelMult = null,
    attnResolutions = null,
    useCheckpoint = null,
    outChannels = null,

    maxNewTokens = null,
    groups = [],
  } = options;

  // ─── Group serialization (Topology v3 native): emit a single Group block with repeat+sub_blocks ────
  // Backend expands groups itself via {group_input}/{group_output} placeholders.
  // We must *not* pre-expand groups here.
  const nodeById = new Map(nodes.map(n => [n.id, n]));
  const groupByNodeId = new Map<string, NodeGroup>();
  for (const g of groups) {
    for (const nodeId of g.nodeIds) {
      groupByNodeId.set(nodeId, g);
    }
  }

  // All groups are emitted as Group IR blocks — the `collapsed` flag is a UI-only
  // visual hint and must NOT gate IR serialisation.
  const collapsedGroups = groups; // treat all groups as collapsed for IR purposes
  const groupNodeIds = new Set(collapsedGroups.flatMap(g => g.nodeIds));
  const groupIdSet = new Set(collapsedGroups.map(g => g.id));

  const outerItemId = (nodeId: string): string => {
    // Already a group ID (new-style connection from group port)
    if (groupIdSet.has(nodeId)) return nodeId;
    // Child node of any group
    const g = groupByNodeId.get(nodeId);
    if (g) return g.id;
    return nodeId;
  };

  const outerConnections: Connection[] = [];
  const seenOuter = new Set<string>();
  for (const c of connections) {
    const fromOuter = outerItemId(c.from);
    const toOuter = outerItemId(c.to);
    if (fromOuter === toOuter) continue;
    const key = `${fromOuter}::${toOuter}`;
    if (seenOuter.has(key)) continue;
    seenOuter.add(key);
    outerConnections.push({ id: c.id, from: fromOuter, to: toOuter });
  }

  const outerNodes: CanvasNode[] = nodes.filter(n => !groupNodeIds.has(n.id));
  const sortedOuterIds = topologicalSort(
    // topologicalSort expects nodes; represent groups as virtual nodes
    [
      ...outerNodes,
      ...collapsedGroups.map(g => ({
        id: g.id,
        type: 'transformer' as LayerType,
        name: g.name,
        x: g.x,
        y: g.y,
        params: {},
      } satisfies CanvasNode)),
    ],
    outerConnections
  );

  const outerNodeMap = new Map<string, CanvasNode>();
  for (const n of outerNodes) outerNodeMap.set(n.id, n);
  for (const g of collapsedGroups) {
    outerNodeMap.set(g.id, {
      id: g.id,
      type: 'transformer' as LayerType,
      name: g.name,
      x: g.x,
      y: g.y,
      params: {},
    });
  }

  // Pre-compute tensor names
  const tensorNames = new Map<string, string>();
  for (const id of sortedOuterIds) {
    if (!groupIdSet.has(id)) {
      const node = outerNodeMap.get(id);
      if (node) tensorNames.set(node.id, makeTensorName(node));
    } else {
      tensorNames.set(id, `${id}_out`);
    }
  }

  // Build incoming edges per node
  const incomingMap = new Map<string, string[]>();
  for (const conn of outerConnections) {
    if (!incomingMap.has(conn.to)) incomingMap.set(conn.to, []);
    incomingMap.get(conn.to)!.push(conn.from);
  }

  // Pre-compute output dimensions per node for auto-fill
  const nodeDimMap = new Map<string, number>();
  for (const node of nodes) {
    const dim = inferNodeOutputDim(node);
    if (dim != null) {
      nodeDimMap.set(node.id, dim);
    }
  }

  // Second pass: propagate dims through nodes that don't define their own
  for (const id of sortedOuterIds) {
    if (nodeDimMap.has(id)) continue;
    const inc = incomingMap.get(id);
    if (inc && inc.length > 0) {
      for (const srcId of inc) {
        if (nodeDimMap.has(srcId)) {
          nodeDimMap.set(id, nodeDimMap.get(srcId)!);
          break;
        }
      }
    }
  }

  const blocks: NeuraxBlock[] = [];

  const buildTransformerGroupBlock = (node: CanvasNode, incoming: string[], outgoingTensor: string): NeuraxBlock => {
    const p: any = node.params ?? {};
    const dim = (p.dim ?? p.d_model ?? p.dModel ?? nodeDimMap.get(node.id) ?? null) as number | null;
    const heads = (p.num_heads ?? p.numHeads ?? p.heads ?? numHeads ?? null) as number | null;
    const ffn = (p.ffn ?? p.ffn_dim ?? p.ffnDim ?? ffnDim ?? null) as number | null;
    const repeat = Math.max(1, Number(p.repeat ?? p.repeatCount ?? p.layers ?? 1) || 1);

    const attnId = `${node.id}_attn`;
    const ffnId = `${node.id}_ffn`;

    const sub_blocks: NeuraxBlock[] = [
      {
        id: attnId,
        type: 'ScaledDotProductAttn',
        ui_node_type: 'attention',
        inputs: ['{group_input}'],
        outputs: [`${attnId}_out`],
        params: {
          ...(heads != null ? { num_heads: heads } : {}),
          ...(dim != null ? { in_features: dim } : {}),
        },
        trainable: true,
      },
      {
        id: ffnId,
        type: 'MLP',
        ui_node_type: 'ffn_standard',
        inputs: [`${attnId}_out`],
        outputs: ['{group_output}'],
        params: {
          ...(dim != null ? { in_features: dim, out_features: dim } : {}),
          ...(ffn != null ? { intermediate_size: ffn } : {}),
          activation: activation ?? 'gelu',
        },
        trainable: true,
      },
    ];

    return {
      id: node.id,
      type: 'Group',
      inputs: incoming,
      outputs: [outgoingTensor],
      params: {},
      ui_node_type: node.type,
      repeat,
      trainable: true,
      sub_blocks,
    };
  };

  const buildAtomicBlock = (
    node: CanvasNode,
    incomingIds: string[],
    incomingTensors: string[],
    outgoingTensor: string,
  ): NeuraxBlock => {
    const blockType = BLOCK_TYPE_MAP[node.type] ?? 'Opaque';

    if (blockType === 'Group' && node.type === 'transformer') {
      return buildTransformerGroupBlock(node, incomingTensors, outgoingTensor);
    }

    let params = extractParams(node, blockType);

    if (blockType === 'DenseProjection') {
      const inputDim = incomingIds.length > 0 ? (nodeDimMap.get(incomingIds[0]) ?? null) : null;
      params = fixDenseParams(node.type, params, inputDim);
    }
    if (blockType === 'Embedding') {
      params = fixEmbeddingParams(params);
    }
    if (blockType === 'ScaledDotProductAttn' || blockType === 'CrossAttention' || blockType === 'FlashAttention') {
      const inputDim = incomingIds.length > 0 ? (nodeDimMap.get(incomingIds[0]) ?? null) : null;
      params = fixSdpaParams(params, inputDim);
    }

    // @ts-ignore
    const NORM_TYPES = new Set(['layernorm', 'rmsnorm', 'LayerNorm', 'RMSNorm', 'BatchNorm', 'GroupNorm', 'InstanceNorm']);
    if (NORM_TYPES.has(blockType) && !('dim' in params)) {
      const inputDim = incomingIds.length > 0 ? (nodeDimMap.get(incomingIds[0]) ?? null) : null;
      if (inputDim !== null) {
        params.dim = inputDim;
      }
    }

    return {
      id: node.id,
      type: blockType,
      ui_node_type: node.type,
      inputs: incomingTensors,
      outputs: [outgoingTensor],
      params,
    };
  };

  for (const id of sortedOuterIds) {
    if (!groupIdSet.has(id)) {
      const node = outerNodeMap.get(id);
      if (!node) continue;
      const incomingOuter = incomingMap.get(id) || [];
      const inputTensors = incomingOuter.map(srcId => tensorNames.get(srcId)!).filter(Boolean);
      const outTensor = tensorNames.get(id)!;
      blocks.push(buildAtomicBlock(node, incomingOuter, inputTensors, outTensor));
      continue;
    }

    const group = collapsedGroups.find(g => g.id === id);
    if (!group) continue;

    const groupNodeSet = new Set(group.nodeIds);
    const groupInternalConnections = connections.filter(
      c => groupNodeSet.has(c.from) && groupNodeSet.has(c.to)
    );

    // Determine group entry/exit nodes based on boundary edges
    const externalIncoming = connections.filter(
      c => !groupNodeSet.has(c.from) && groupNodeSet.has(c.to)
    );
    const externalOutgoing = connections.filter(
      c => groupNodeSet.has(c.from) && !groupNodeSet.has(c.to)
    );

    const groupInputsOuter = (incomingMap.get(group.id) || []).map(srcId => tensorNames.get(srcId)!).filter(Boolean);
    const groupOutputsOuter = (() => {
      const outTensor = tensorNames.get(group.id);
      return outTensor ? [outTensor] : [];
    })();

    // Internal tensor names must be relative so backend Group expansion can prefix them.
    const innerTensorName = (nodeId: string): string => `${nodeId}_out`;

    // Build per-node incoming inside group (both internal and external)
    const innerIncomingMap = new Map<string, string[]>();
    for (const c of groupInternalConnections) {
      if (!innerIncomingMap.has(c.to)) innerIncomingMap.set(c.to, []);
      innerIncomingMap.get(c.to)!.push(c.from);
    }
    for (const c of externalIncoming) {
      if (!innerIncomingMap.has(c.to)) innerIncomingMap.set(c.to, []);
      innerIncomingMap.get(c.to)!.push(c.from);
    }

    // Sort nodes within group using internal edges only; append any disconnected nodes.
    const groupNodes: CanvasNode[] = group.nodeIds
      .map(nid => nodeById.get(nid))
      .filter((n): n is CanvasNode => Boolean(n));
    const sortedInnerIds = topologicalSort(groupNodes, groupInternalConnections);
    for (const n of groupNodes) {
      if (!sortedInnerIds.includes(n.id)) sortedInnerIds.push(n.id);
    }

    // Choose a single external input/output tensor for placeholder substitution.
    const groupInputTensor = groupInputsOuter[0] ?? '';
    const groupOutputTensor = groupOutputsOuter[0] ?? `${group.id}_out`;

    const subBlocks: NeuraxBlock[] = sortedInnerIds.map(innerId => {
      const node = nodeById.get(innerId)!;
      const incIds = innerIncomingMap.get(innerId) || [];

      const inputs = incIds.map(srcId => {
        if (groupNodeSet.has(srcId)) {
          return innerTensorName(srcId);
        }
        // external source
        const outerSrc = outerItemId(srcId);
        const extTensor = tensorNames.get(outerSrc) ?? '';
        return extTensor === groupInputTensor ? '{group_input}' : extTensor;
      }).filter(Boolean);

      const out = innerTensorName(innerId);
      const outputs = [
        externalOutgoing.some(c => c.from === innerId)
          ? '{group_output}'
          : out,
      ];

      const blockType = BLOCK_TYPE_MAP[node.type] ?? 'Opaque';
      let params = extractParams(node, blockType);

      if (blockType === 'DenseProjection') {
        const inputDim = incIds.length > 0 ? (nodeDimMap.get(incIds[0]) ?? null) : null;
        params = fixDenseParams(node.type, params, inputDim);
      }
      if (blockType === 'Embedding') {
        params = fixEmbeddingParams(params);
      }
      if (blockType === 'ScaledDotProductAttn' || blockType === 'CrossAttention' || blockType === 'FlashAttention') {
        const inputDim = incIds.length > 0 ? (nodeDimMap.get(incIds[0]) ?? null) : null;
        params = fixSdpaParams(params, inputDim);
      }

      // @ts-ignore
      const NORM_TYPES = new Set(['layernorm', 'rmsnorm', 'LayerNorm', 'RMSNorm', 'BatchNorm', 'GroupNorm', 'InstanceNorm']);
      if (NORM_TYPES.has(blockType) && !('dim' in params)) {
        const inputDim = incIds.length > 0 ? (nodeDimMap.get(incIds[0]) ?? null) : null;
        if (inputDim !== null) {
          params.dim = inputDim;
        }
      }

      return {
        id: innerId,
        type: blockType,
        ui_node_type: node.type,
        inputs,
        outputs,
        trainable: true,
        params,
      };
    });

    blocks.push({
      id: group.id,
      type: 'Group',
      ui_node_type: 'transformer',
      inputs: [groupInputTensor].filter(Boolean),
      outputs: [groupOutputTensor].filter(Boolean),
      repeat: Math.max(1, group.repeatCount ?? 1),
      trainable: true,
      params: {},
      sub_blocks: subBlocks,
    });
  }

  // ─── Auto-fix pass ────────────────────────────────────────────
  const autoFixNotes: string[] = [];

  // 1. Hardware validation
  const { hardware: resolvedHw, wasAutoFixed: hwFixed } = normalizeHardware(hardware);
  if (hwFixed) {
    autoFixNotes.push(`Hardware auto-fixed to "${resolvedHw}" (was "${hardware ?? 'null'}").`);
  }

  // 2. Input block validation: must have dim (seq) or channels (img), never empty params
  for (const block of blocks) {
    if (block.type === 'Input') {
      const params = block.params || {};
      const hasDim = 'dim' in params || 'd_model' in params;
      const hasChannels = 'channels' in params || 'in_channels' in params;
      if (!hasDim && !hasChannels) {
        // Auto-fix: default to sequence mode with dim from downstream
        if (!block.params) block.params = {};
        block.params.dim = 2048;
        autoFixNotes.push(`Input block "${block.id}" had empty params; auto-fixed to seq mode dim=2048.`);
      }
    }
  }

  // 2b. DenseProjection/LMHead validation: must have in_features + out_features
  for (const block of blocks) {
    if (block.type === 'DenseProjection' || block.type === 'LMHead') {
      const params = block.params || {};
      if (!('in_features' in params)) {
        autoFixNotes.push(`Block "${block.id}" (${block.type}) missing params.in_features.`);
      }
      if (!('out_features' in params)) {
        autoFixNotes.push(`Block "${block.id}" (${block.type}) missing params.out_features.`);
      }
    }
  }

  // 3. MoE shape rule: TopKRouter/ExpertFFN need [B,S,D], never directly after TokenInput
  const tokenOutputTensors = new Set(
    blocks.filter(b => b.type === 'TokenInput').flatMap(b => b.outputs)
  );
  for (const block of blocks) {
    if (block.type === 'TopKRouter' || block.type === 'ExpertFFN' || block.type === 'DenseProjection') {
      for (const inp of block.inputs) {
        if (tokenOutputTensors.has(inp)) {
          autoFixNotes.push(
            `Block "${block.id}" (${block.type}) receives raw token ids from TokenInput — requires Embedding in between.`
          );
        }
      }
    }
  }

  // 4. Validate block id uniqueness
  const seenIds = new Set<string>();
  for (const block of blocks) {
    if (seenIds.has(block.id)) {
      autoFixNotes.push(`Duplicate block id "${block.id}" detected.`);
    }
    seenIds.add(block.id);
  }

  // 5. Validate tensor chain integrity
  const producedTensors = new Set<string>();
  for (const block of blocks) {
    for (const inp of block.inputs) {
      if (inp && !producedTensors.has(inp)) {
        // Allow if it's the first block (external input)
        if (block.inputs.length > 0 && blocks.indexOf(block) > 0) {
          // Tensor not yet produced — might be an issue but don't break, just note
        }
      }
    }
    for (const out of block.outputs) {
      producedTensors.add(out);
    }
  }

  // Infer env requirements from block types
  const { needsSeqLen, needsImage } = inferEnvRequirements(blocks);

  const resolvedSeqLen = needsSeqLen ? (seqLen ?? 128) : seqLen;
  const resolvedImgH = needsImage ? (imgHeight ?? 224) : imgHeight;
  const resolvedImgW = needsImage ? (imgWidth ?? 224) : imgWidth;

  // Build env, only including defined values
  // Build env, only including defined values
  const env: NeuraxEnv = {
    hw: resolvedHw,
    prec: normalizePrecision(precision),
    batch: batchSize ?? 1,
  };

  if (seed != null) env.seed = seed;
  if (device != null) env.device = device;
  if (useCompile != null) env.compile = useCompile;

  // Transformers
  if (resolvedSeqLen != null) env.seq = resolvedSeqLen;
  if (vocabSize != null) env.vocab = vocabSize;
  if (hiddenDim != null) env.d = hiddenDim;
  if (numHeads != null) env.h = numHeads;
  if (headDim != null) env.hd = headDim;
  if (ffnDim != null) env.ff = ffnDim;
  if (numLayers != null) env.L = numLayers;
  if (kvHeads != null) env.kv = kvHeads;
  if (useBias != null) env.bias = useBias;
  if (dropout != null) env.drop = dropout;
  if (useFlash != null) env.flash = useFlash;
  if (ropeTheta != null) env.rope_theta = ropeTheta;
  if (maxSeqLen != null) env.max_seq_len = maxSeqLen;
  if (useAlibi != null) env.alibi = useAlibi;
  if (useRelativeBias != null) env.relative_bias = useRelativeBias;
  if (useCache != null) env.use_cache = useCache;
  if (activation != null) env.activation = activation;

  // CNN / Spatial
  if (resolvedImgH != null) env.h_img = resolvedImgH;
  if (resolvedImgW != null) env.w_img = resolvedImgW;
  if (inChannels != null) env.cin = inChannels;
  if (numClasses != null) env.cout = numClasses;
  if (normType != null) env.norm = normType;
  if (convActivation != null) env.act = convActivation;
  if (poolType != null) env.pool = poolType;

  // ViT / DiT
  if (patchSize != null) env.patch = patchSize;
  if (numPatches != null) env.patches = numPatches;
  if (numDenoisingSteps != null) env.steps = numDenoisingSteps;
  if (guidanceScale != null) env.cfg = guidanceScale;
  if (mlpRatio != null) env.mlp_ratio = mlpRatio;
  if (qkvBias != null) env.qkv_bias = qkvBias;
  if (projDrop != null) env.proj_drop = projDrop;
  if (attnDrop != null) env.attn_drop = attnDrop;
  if (posEmbedType != null) env.pos_embed = posEmbedType;
  if (useFlashVit != null) env.use_flash = useFlashVit;

  // GNN
  if (numNodes != null) env.num_nodes = numNodes;
  if (numEdges != null) env.num_edges = numEdges;
  if (nodeFeatDim != null) env.feat_dim = nodeFeatDim;
  if (outDim != null) env.out_dim = outDim;
  if (edgeFeatDim != null) env.edge_dim = edgeFeatDim;
  if (aggrType != null) env.aggr = aggrType;
  if (useNormalize != null) env.normalize = useNormalize;
  if (addSelfLoops != null) env.add_self_loops = addSelfLoops;

  // RNN / SSM / SNN
  if (hiddenSize != null) env.hid = hiddenSize;
  if (isBidirectional != null) env.bidir = isBidirectional;
  if (dState != null) env.state = dState;
  if (dtRank != null) env.dt = dtRank;
  if (convKernel != null) env.conv_kernel = convKernel;
  if (expandFactor != null) env.expand = expandFactor;
  if (useFastPath != null) env.use_fast_path = useFastPath;
  if (projSize != null) env.proj_size = projSize;
  if (timesteps != null) env.time = timesteps;
  if (spikeRate != null) env.spike = spikeRate;

  // MoE
  if (numExperts != null) env.exp = numExperts;
  if (topK != null) env.topk = topK;
  if (expertCapacity != null) env.expert_capacity = expertCapacity;
  if (useSharedExpert != null) env.shared_expert = useSharedExpert;

  // Diffusion Base (UNet)
  if (modelChannels != null) env.model_channels = modelChannels;
  if (numResBlocks != null) env.num_res_blocks = numResBlocks;
  if (channelMult != null) env.channel_mult = channelMult;
  if (attnResolutions != null) env.attention_resolutions = attnResolutions;
  if (useCheckpoint != null) env.use_checkpoint = useCheckpoint;
  if (outChannels != null) env.out_channels = outChannels;

  // RL / Legend
  if (actionDim != null) env.act_dim = actionDim;
  if (stateDim != null) env.st_dim = stateDim;
  if (maxNewTokens != null) env.max_new_tokens = maxNewTokens;

  // Convert blocks to layers format expected by backend
  // Use numeric shapes: [batch, seq_len, hidden_dim] or derived from params
  const batch = batchSize ?? 1;
  const seq = resolvedSeqLen ?? 128;
  const hidden = hiddenDim ?? 768;

  const flatBlocks = flattenBlocks(blocks).filter((block) => block.ui_node_type !== 'input' && block.ui_node_type !== 'output');

  const layers: NeuraxLayer[] = flatBlocks.map(block => {
    // Derive shapes from block type and params
    let inputShape: number[] = [];
    let outputShape: number[] = [];

    const p = block.params ?? {};

    switch (block.type) {
      case 'Input':
        inputShape = [];
        outputShape = [batch, seq, p.dim ?? hidden];
        break;
      case 'Embedding':
        inputShape = [batch, seq];
        outputShape = [batch, seq, p.d_model ?? hidden];
        break;
      case 'DenseProjection':
      case 'LMHead':
        inputShape = [batch, seq, p.in_features ?? hidden];
        outputShape = [batch, seq, p.out_features ?? p.vocab_size ?? hidden];
        break;
      default:
        // For attention, norms, etc. - shapes pass through
        inputShape = [batch, seq, hidden];
        outputShape = [batch, seq, hidden];
    }

    return {
      id: block.id,
      layer_type: toParserLayerType(block.ui_node_type ?? block.type),
      input_shape: inputShape,
      output_shape: outputShape,
      params: p,
      ...(block.comment && { comment: block.comment }),
    };
  });

  // Build global_params from env
  const global_params: NeuraxGlobalParams = {};
  if (hiddenDim != null) global_params.hidden_size = hiddenDim;
  if (numLayers != null) global_params.num_layers = numLayers;
  if (vocabSize != null) global_params.vocab_size = vocabSize;
  if (resolvedSeqLen != null) global_params.sequence_length = resolvedSeqLen;
  if (numHeads != null) global_params.num_heads = numHeads;
  if (headDim != null) global_params.head_dim = headDim;
  if (ffnDim != null) global_params.ffn_dim = ffnDim;
  if (numExperts != null) global_params.num_experts = numExperts;
  if (topK != null) global_params.top_k = topK;

  // Build training config
  const trainingConfig: NeuraxTraining = {
    batch_size: batchSize ?? 1,
    precision: normalizePrecision(precision),
    ...(learningRate != null && { learning_rate: learningRate }),
    ...(numEpochs != null && { num_epochs: numEpochs }),
    ...(resolvedSeqLen != null && resolvedSeqLen > 0 && { sequence_length: resolvedSeqLen }),
  };

  // Build hardware config
  const gpuName = resolvedHw === 'CPU' ? 'CPU' : resolvedHw;
  const hardwareConfig: NeuraxHardware = {
    gpus: [{
      name: gpuName,
      count: gpuCount ?? 1,
      ...(gpuMemoryGb != null && { memory_gb: gpuMemoryGb }),
    }],
  };

  // Build data config
  const dataConfig: NeuraxData | undefined =
    (datasetSize != null || vocabSize != null || numClasses != null)
      ? {
        ...(datasetSize != null && { dataset_size: datasetSize }),
        ...(vocabSize != null && vocabSize > 0 && { vocab_size: vocabSize }),
        ...(numClasses != null && numClasses > 0 && { num_classes: numClasses }),
      }
      : undefined;

  return {
    schema_version: '1.0',
    model: {
      name: modelName,
      type: toParserModelType(family),
      global_params,
      layers,
    },
    training: trainingConfig,
    hardware: hardwareConfig,
    ...(dataConfig && { data: dataConfig }),
  };
}
