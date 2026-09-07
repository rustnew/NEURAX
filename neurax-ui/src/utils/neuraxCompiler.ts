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
  /** How many of `num_layers` are dense (no MoE routing) — see `numDenseLayers`. */
  num_dense_layers?: number;
  /** An encoder-decoder model's decoder depth, when it differs from `num_layers` (the encoder's). */
  num_decoder_layers?: number;
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

/** For sdpa blocks, ensure `num_heads` not `heads`, and add hidden_size */
function fixSdpaParams(params: Record<string, unknown>, inputDim: number | null): Record<string, unknown> {
  const out = { ...params };
  // Rename heads → num_heads
  if ('heads' in out && !('num_heads' in out)) {
    out.num_heads = out.heads;
    delete out.heads;
  }
  // Infer hidden_size from dim or d_model or upstream
  // Backend calculate_layer_params reads hidden_size for Attention
  if (!('hidden_size' in out)) {
    if ('dim' in out) {
      out.hidden_size = out.dim;
    } else if ('d_model' in out) {
      out.hidden_size = out.d_model;
    } else if (inputDim !== null) {
      out.hidden_size = inputDim;
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

/** LSTM/GRU/RNN: the backend reads the recurrent state size from
 * `rnn_hidden_size` and the layer's input dimension from bare `hidden_size`
 * — but a template author's `hidden_size` on an RNN node means the
 * recurrent state size (see every real template: "BiGRU Layer 2 (256)"
 * sets `hidden_size: 256` meaning *its own* width, not the previous
 * layer's). Nothing here ever renamed it, so `rnn_hidden_size` was always
 * absent and the backend fell back to a hardcoded 512 regardless of what
 * the canvas said — silently discarding the user's configured size for
 * every LSTM/GRU/RNN block in the app. */
function fixRnnParams(params: Record<string, unknown>, inputDim: number | null): Record<string, unknown> {
  const out = { ...params };
  if (!('rnn_hidden_size' in out) && 'hidden_size' in out) {
    out.rnn_hidden_size = out.hidden_size;
    out.hidden_size = inputDim ?? out.hidden_size;
  }
  if (!('bidirectional_rnn' in out) && out.bidirectional === true) {
    out.bidirectional_rnn = true;
  }
  if (!('num_rnn_layers' in out) && 'num_layers' in out) {
    out.num_rnn_layers = out.num_layers;
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
    !['classification_head', 'policy_head', 'value_head'].includes(nodeType)
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

function extractParams(node: CanvasNode, _blockType: string): Record<string, unknown> {
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

/**
 * FLOPs-per-element and gated-ness for one activation, ported verbatim from
 * `neurax-formulas::activation::activation_spec` (the Rust source of truth
 * every backend formula reads from) so this one code path can't drift from
 * it the way it already had: this table used to hardcode `gated ? 4 : 8`
 * and treat plain `silu`/`swish` as gated (they aren't — only `swiglu`/
 * `geglu`/`glu`/`reglu` are), which double-counted a plain SiLU MLP as
 * SwiGLU and used the wrong per-element cost for every other activation
 * (a plain `gelu` MLP got charged 8, not the real, calibrated 10).
 * Unrecognised names fall back to `{ flops: 5, gated: false }`, matching
 * `activation_flops_per_element`'s own unknown-name fallback.
 */
const ACTIVATION_SPECS: Record<string, { flops: number; gated: boolean }> = {
  none: { flops: 0, gated: false },
  linear: { flops: 0, gated: false },
  identity: { flops: 0, gated: false },
  relu: { flops: 1, gated: false },
  leaky_relu: { flops: 2, gated: false },
  leakyrelu: { flops: 2, gated: false },
  relu6: { flops: 2, gated: false },
  sigmoid: { flops: 4, gated: false },
  logistic: { flops: 4, gated: false },
  tanh: { flops: 6, gated: false },
  silu: { flops: 4, gated: false },
  swish: { flops: 4, gated: false },
  hard_sigmoid: { flops: 2, gated: false },
  hardsigmoid: { flops: 2, gated: false },
  hard_swish: { flops: 3, gated: false },
  hardswish: { flops: 3, gated: false },
  h_swish: { flops: 3, gated: false },
  gelu: { flops: 10, gated: false },
  gelu_erf: { flops: 10, gated: false },
  gelu_tanh: { flops: 8, gated: false },
  gelu_new: { flops: 8, gated: false },
  gelu_approx: { flops: 8, gated: false },
  softplus: { flops: 7, gated: false },
  mish: { flops: 12, gated: false },
  elu: { flops: 4, gated: false },
  selu: { flops: 5, gated: false },
  glu: { flops: 4, gated: true },
  swiglu: { flops: 4, gated: true },
  silu_glu: { flops: 4, gated: true },
  swish_glu: { flops: 4, gated: true },
  geglu: { flops: 10, gated: true },
  gelu_glu: { flops: 10, gated: true },
  reglu: { flops: 1, gated: true },
  relu_glu: { flops: 1, gated: true },
};

function activationSpec(name: string | null | undefined): { flops: number; gated: boolean } {
  const normalized = String(name ?? '').trim().toLowerCase().replace(/-/g, '_');
  return ACTIVATION_SPECS[normalized] ?? { flops: 5, gated: false };
}

/** Block types that take their normalized dimension from `dim` rather than
 * `hidden_size`/`d_model` — hoisted from two identical inline copies. */
const NORM_TYPES = new Set<string>([
  'layernorm',
  'rmsnorm',
  'LayerNorm',
  'RMSNorm',
  'BatchNorm',
  'GroupNorm',
  'InstanceNorm',
]);

/**
 * Params and forward-FLOPs for a `layer_stack` block — the compact "N×
 * Decoder Blocks" node every preset in the Templates catalogue compiles to,
 * instead of N individual Attention/Mlp/Normalization nodes.
 *
 * The backend's `LayerType::Custom` arm has no formula keyed to "a block
 * that secretly repeats N times" — without this, the stack silently priced
 * out at 0 params and 0 FLOPs. For GPT-2 Small that undercounted total
 * parameters by roughly 3x (39M shown vs. the model's real ~124M), because
 * the stack holds nearly all of a transformer's weight; the params/tab and
 * per-layer FLOPs breakdown for "12x Decoder Blocks" both read as literal
 * zero, and the GPU-utilization figure computed from those same per-layer
 * numbers came out NaN (serialized as a bare JSON `null`) for every model
 * that uses this representation — which is every preset in the catalogue.
 *
 * Mirrors neurax-formulas' attention/mlp/normalization formulas (standard
 * MHA or GQA, standard or SwiGLU-gated MLP, causal attention, LayerNorm ×2
 * per block — pre-attention and pre-FFN) so the number this emits agrees
 * with what the backend would compute for an equivalent architecture built
 * from individually typed blocks. No bias terms, matching how this
 * catalogue's other blocks (e.g. the LM head) are already compiled.
 */
function decoderStackParamsAndFlops(cfg: {
  numLayers: number;
  hidden: number;
  heads: number;
  kvHeads: number | null;
  headDim: number | null;
  ffnDim: number;
  activation: string | null;
  gated: boolean | null;
  batch: number;
  seq: number;
}): { paramCount: number; flopsForward: number } {
  const { numLayers, hidden, heads, ffnDim, batch, seq } = cfg;
  const kvHeads = cfg.kvHeads ?? heads;
  // A model that states its own head_dim (GLM-4.5: hidden_size=5120,
  // num_heads=96, head_dim=128 — 96*128=12288 != 5120, a real widened
  // Q/output projection, not a rounding artifact) must use it instead of
  // silently deriving — and truncating — one from hidden/heads. Mirrors
  // neurax-formulas::attention's `*_with_head_dim` variants.
  const headDim = cfg.headDim ?? hidden / Math.max(heads, 1);
  const qkWidth = heads * headDim;
  const activation = activationSpec(cfg.activation);
  const gated = cfg.gated === true || activation.gated;

  let attnParams: number;
  let attnFlops: number;
  if (kvHeads === heads) {
    // Standard MHA: Q, K, V, out — hidden -> qkWidth each (== hidden*hidden
    // when head_dim derives evenly, wider when it doesn't).
    attnParams = 3 * hidden * qkWidth + qkWidth * hidden;
    const qkv = 3 * (2 * batch * seq * hidden * qkWidth);
    const scores = 2 * batch * heads * seq * seq * headDim;
    const av = 2 * batch * heads * seq * seq * headDim;
    const softmax = 5 * batch * heads * seq * seq;
    const outProj = 2 * batch * seq * qkWidth * hidden;
    // Causal: on average only half the attention matrix is real work.
    attnFlops = qkv + (scores + av + softmax) * 0.5 + outProj;
  } else {
    // GQA/MQA: K, V projections shrink to the KV head count.
    const kvDim = kvHeads * headDim;
    attnParams = hidden * qkWidth + 2 * hidden * kvDim + qkWidth * hidden;
    const q = 2 * batch * seq * hidden * qkWidth;
    const kv = 2 * (2 * batch * seq * hidden * kvDim);
    const scores = 2 * batch * heads * seq * seq * headDim;
    const av = 2 * batch * heads * seq * seq * headDim;
    const outProj = 2 * batch * seq * qkWidth * hidden;
    attnFlops = q + kv + (scores + av) * 0.5 + outProj;
  }

  // MLP: standard (up + down) or SwiGLU-gated (gate + up + down).
  const mlpParams = gated ? 3 * hidden * ffnDim : 2 * hidden * ffnDim;
  const linear1 = 2 * batch * seq * hidden * ffnDim;
  const linear2 = 2 * batch * seq * ffnDim * hidden;
  const gateExtra = gated ? 2 * batch * seq * hidden * ffnDim + batch * seq * ffnDim : 0;
  const actFlops = activation.flops * batch * seq * ffnDim;
  const mlpFlops = linear1 + linear2 + gateExtra + actFlops;

  // Two LayerNorms per block (pre-attention, pre-FFN): weight + bias each.
  const normParams = 2 * (2 * hidden);
  const normFlops = 2 * (5 * batch * seq * hidden);

  const perLayerParams = attnParams + mlpParams + normParams;
  const perLayerFlops = attnFlops + mlpFlops + normFlops;

  return {
    paramCount: Math.round(perLayerParams * numLayers),
    flopsForward: perLayerFlops * numLayers,
  };
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

  // Graph Neural Networks — explicit, not left to the generic `attention`/
  // `custom` fuzzy fallbacks below. `gat_attention` used to fall into
  // `.includes('attention')` and cost a plain transformer attention block;
  // `message_passing` matched nothing at all and fell all the way through
  // to `custom` (0 parameters, 0 FLOPs) regardless of the design. Both now
  // reach the real GCN/GAT/message-passing formulas in neurax-formulas.
  if (normalized === 'graph_conv' || normalized === 'gcn_conv' || normalized === 'gcn') {
    return 'graph_conv';
  }
  if (
    normalized === 'graph_attention' ||
    normalized === 'gat_attention' ||
    normalized === 'gat_conv'
  ) {
    return 'graph_attention';
  }
  if (
    normalized === 'message_passing' ||
    normalized === 'mpnn' ||
    normalized === 'graph_sage' ||
    normalized === 'graphsage'
  ) {
    return 'message_passing';
  }
  // rgcn_conv used to fall through to the generic `.includes('conv')` catch-all
  // below and get costed as a plain Conv2D (in_channels/out_channels/kernel) —
  // a weight matrix per relation type (num_relations) is what actually makes
  // an RGCN layer expensive, and it has none of a Conv2D's params at all.
  if (normalized === 'rgcn_conv' || normalized === 'rgcn' || normalized === 'relational_gcn') {
    return 'rgcn_conv';
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

  // A block diagram's MoE layer is one router node, one experts node, one
  // combine node, and (for a DeepSeek-style model) one shared-expert node —
  // four separate nodes standing in for one conceptual layer, so the diagram
  // reads left to right instead of as a single opaque box. All four used to
  // collapse to the same generic `moe` parser type, which the Rust side reads
  // as "this node alone is a complete MoE layer": the router's `hidden ×
  // num_experts` gating matrix got costed as `num_experts` full experts, and
  // a real model's params/FLOPs came out wrong by 20-180%, confirmed against
  // Mixtral-8x7B and DeepSeek-MoE-16B's published sizes. Splitting the three
  // roles that have a concrete, tested real-world shape (the router, the
  // combine step, and a DeepSeek-style shared expert) onto their own parser
  // types fixes both the misattributed cost and — because
  // `repeat_scale_for` counts occurrences per type — the depth multiplier
  // that was being diluted across however many same-typed nodes one logical
  // layer happened to use.
  //
  // The remaining, more exotic MoE block variants below (expert-choice
  // routing, product-key routers, soft-MoE, ...) have no reference template
  // or import path exercising them yet, so there is no real config to
  // measure them against — left mapped to the generic `moe` type rather than
  // guessed into a bucket that can't be verified.
  if (normalized === 'noisy_topk_router') {
    return 'moe_router';
  }
  if (normalized === 'expert_combine') {
    return 'moe_combine';
  }
  if (normalized === 'shared_expert') {
    return 'moe_shared_expert';
  }

  if (
    normalized === 'moe_block' ||
    normalized === 'router_linear' ||
    normalized === 'router_softmax' ||
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
    normalized === 'output_combination' ||
    normalized === 'concat_projection' ||
    normalized === 'attention_pooling' ||
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

  // lora_linear/dora_linear used to collapse into the generic 'dense' bucket
  // below, costing a fine-tuning adapter as if it were the full dense layer
  // it exists specifically to avoid materializing — rank * (in + out)
  // instead of in * out, routinely a 100x+ difference.
  if (normalized === 'lora_linear' || normalized === 'lora') {
    return 'lora_linear';
  }
  if (normalized === 'dora_linear' || normalized === 'dora') {
    return 'dora_linear';
  }

  if (
    normalized === 'linear_projection' ||
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
    normalized === 'forecasting_head' ||
    normalized === 'mapping_linear' ||
    normalized === 'equalized_lr_linear'
  ) {
    return 'dense';
  }

  // `bottleneck_block` is deliberately its own branch, not folded in with
  // `basic_block` below: the parser accepts "bottleneck_block" itself as a
  // direct alias for `LayerType::ResnetBottleneck` (the real 3-conv
  // 1x1/3x3/1x1 block ResNet-50+ uses, with its own `mid_channels`/
  // `cardinality`-aware formula) — routing it to `residual_block`
  // (`LayerType::ResidualBlock`, ResNet-18/34's plain 2-conv block) instead
  // silently cost every bottleneck-family template the wrong architecture,
  // not just the wrong numbers. `basic_block` genuinely is `ResidualBlock`.
  if (normalized === 'bottleneck_block') {
    return 'bottleneck_block';
  }
  if (
    normalized === 'basic_block' ||
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
  if (
    normalized.includes('down_block') ||
    normalized.includes('downsample') ||
    normalized === 'unet_encoder'
  ) {
    return 'down_block';
  }
  if (
    normalized.includes('up_block') ||
    normalized.includes('upsample') ||
    normalized === 'unet_decoder'
  ) {
    return 'up_block';
  }
  if (normalized.includes('mid_block') || normalized === 'unet_mid') {
    return 'mid_block';
  }
  // MM-DiT (Stable Diffusion 3 / FLUX's joint image+text attention block)
  // and the UNet variants below used to match nothing here and fall to
  // `custom` — 0 parameters regardless of the design. Routed to the generic
  // diffusion block formula (real, if not block-specific) rather than left
  // at zero.
  if (
    normalized === 'mmdit_block' ||
    normalized === 'unet_latent' ||
    normalized === 'unet_eff' ||
    normalized === 'refiner' ||
    normalized === 'caption_refiner'
  ) {
    return 'unet_block';
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

  // StyleGAN's synthesis network and its `to_rgb` projection — both real
  // convolutions in the actual architecture, both used to fall to `custom`
  // (0 parameters) because neither name contains "generator" or "conv".
  if (normalized === 'synthesis_block') {
    return 'generator_block';
  }
  if (normalized === 'to_rgb' || normalized === 'from_rgb') {
    return 'conv';
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

/**
 * CNN shape propagation.
 *
 * Every non-Input/Embedding/DenseProjection/LMHead block falls through to a
 * `default` case elsewhere in this file that hardcodes `[batch, seq,
 * hidden]` — values that describe a sequence model, not an image one, and
 * are 0 whenever nothing set them. A CNN's actual shape changes at every
 * conv/pool layer (channels grow, spatial size shrinks) and nothing here
 * ever tracked that: every block computed its shape in isolation, with no
 * awareness of what the block before it actually produced. This computes
 * the real thing — standard conv/pool output-size arithmetic, propagated
 * along the graph's real edges from the entry blocks (whatever is
 * connected from the input) through to the head.
 */
type ShapeVec = number[];

function convOutSpatial(inSize: number, kernel: number, stride: number, padding: number, dilation: number): number {
  const size = Math.floor((inSize + 2 * padding - dilation * (kernel - 1) - 1) / Math.max(stride, 1) + 1);
  return Math.max(1, size);
}

function transposeConvOutSpatial(inSize: number, kernel: number, stride: number, padding: number, outputPadding: number, dilation: number): number {
  const size = (inSize - 1) * stride - 2 * padding + dilation * (kernel - 1) + outputPadding + 1;
  return Math.max(1, size);
}

/** Output shape for one CNN block, given the shape(s) of whatever feeds it. */
function cnnBlockOutputShape(blockType: string, params: Record<string, any> | undefined | null, inputShapes: ShapeVec[]): ShapeVec {
  const primary = inputShapes[0] ?? [1, 1, 1, 1];
  const [B, C, H, W] = [
    primary[0] ?? 1,
    primary[1] ?? 1,
    primary[2] ?? 1,
    primary[3] ?? 1,
  ];
  const p = params ?? {};
  const kernel = Number(p.kernel_size ?? 3);
  const stride = Number(p.stride ?? 1);
  const padding = Number(p.padding ?? 0);
  const dilation = Number(p.dilation ?? 1);

  switch (blockType) {
    case 'Conv2D':
    case 'DepthwiseSep': {
      const outC = Number(p.out_channels ?? C);
      return [B, outC, convOutSpatial(H, kernel, stride, padding, dilation), convOutSpatial(W, kernel, stride, padding, dilation)];
    }
    case 'TransposeConv': {
      const outC = Number(p.out_channels ?? C);
      const outputPadding = Number(p.output_padding ?? 0);
      return [
        B, outC,
        transposeConvOutSpatial(H, kernel, stride, padding, outputPadding, dilation),
        transposeConvOutSpatial(W, kernel, stride, padding, outputPadding, dilation),
      ];
    }
    case 'MaxPool':
    case 'AvgPool': {
      const poolSize = Number(p.pool_size ?? p.kernel_size ?? 2);
      const poolStride = Number(p.stride ?? poolSize);
      const poolPadding = Number(p.padding ?? 0);
      return [B, C, convOutSpatial(H, poolSize, poolStride, poolPadding, 1), convOutSpatial(W, poolSize, poolStride, poolPadding, 1)];
    }
    case 'AdaptivePool': {
      const outSize = Math.max(1, Number(p.output_size ?? 1));
      return [B, C, outSize, outSize];
    }
    case 'GlobalPool':
      return [B, C, 1, 1];
    case 'Upsample': {
      const scale = Number(p.scale_factor ?? 2);
      const outC = Number(p.out_channels ?? C);
      return [B, outC, Math.max(1, Math.round(H * scale)), Math.max(1, Math.round(W * scale))];
    }
    case 'Flatten':
      return [B, C * H * W];
    case 'DenseProjection': {
      const inFeatures = primary.slice(1).reduce((a, b) => a * (b || 1), 1);
      const outFeatures = Number(p.out_features ?? p.num_classes ?? inFeatures);
      return [B, outFeatures];
    }
    case 'Concat': {
      // Branches merging back together (e.g. Inception-style): same B/H/W,
      // channels add up. Falls back to the primary branch's shape if the
      // branches don't actually line up (mismatched resolution is a real
      // architecture error the validator should catch, not something to
      // paper over with a wrong number here).
      const fourD = inputShapes.filter((s) => s.length === 4);
      if (fourD.length === inputShapes.length && fourD.length > 1) {
        const [b0, , h0, w0] = fourD[0];
        const sameSpatial = fourD.every((s) => s[0] === b0 && s[2] === h0 && s[3] === w0);
        if (sameSpatial) {
          const totalC = fourD.reduce((sum, s) => sum + (s[1] ?? 0), 0);
          return [b0, totalC, h0, w0];
        }
      }
      return primary;
    }
    case 'BatchNorm':
    case 'GroupNorm':
    case 'InstanceNorm':
    case 'LayerNorm':
    case 'RmsNorm':
    case 'Dropout':
    case 'ResidualAdd':
    case 'Add':
    default:
      return primary;
  }
}

/**
 * Propagates shapes along a graph's real edges, starting from whatever the
 * entry blocks (directly connected from the canvas's `input` node, which
 * never appears in `flatBlocks` itself) actually see — a real per-family
 * tensor shape, not a sequence model's `[batch, seq, hidden]` placeholder.
 * `shapeFn` supplies the actual per-block-type arithmetic; this function
 * only owns getting blocks visited in dependency order.
 */
function computeShapesViaGraph(
  flatBlocks: NeuraxBlock[],
  connections: Connection[],
  entryShape: ShapeVec,
  shapeFn: (blockType: string, params: Record<string, any> | undefined | null, inputShapes: ShapeVec[]) => ShapeVec,
): Map<string, { input: ShapeVec; output: ShapeVec }> {
  const shapes = new Map<string, { input: ShapeVec; output: ShapeVec }>();
  const byId = new Map(flatBlocks.map((b) => [b.id, b]));
  const predsOf = new Map<string, string[]>();
  for (const b of flatBlocks) predsOf.set(b.id, []);
  for (const c of connections) {
    if (byId.has(c.to) && byId.has(c.from)) predsOf.get(c.to)!.push(c.from);
  }

  const pending = new Set(flatBlocks.map((b) => b.id));
  const queue: string[] = flatBlocks
    .filter((b) => (predsOf.get(b.id) ?? []).length === 0)
    .map((b) => b.id);

  // Kahn-style traversal with requeueing: bounded so a cycle (which
  // shouldn't exist — the validator rejects one — but this must never hang
  // the compiler if one somehow does) degrades to "unresolved nodes fall
  // back to the entry shape" instead of an infinite loop.
  let guard = flatBlocks.length * flatBlocks.length + flatBlocks.length + 8;
  while (queue.length > 0 && guard-- > 0) {
    const id = queue.shift()!;
    if (!pending.has(id)) continue;
    const preds = predsOf.get(id) ?? [];
    if (preds.some((p) => pending.has(p))) {
      queue.push(id);
      continue;
    }
    const block = byId.get(id)!;
    const inputShapes = preds.length > 0
      ? preds.map((p) => shapes.get(p)?.output ?? entryShape)
      : [entryShape];
    const outputShape = shapeFn(block.type, block.params, inputShapes);
    shapes.set(id, { input: inputShapes[0], output: outputShape });
    pending.delete(id);
    for (const c of connections) {
      if (c.from === id && byId.has(c.to) && pending.has(c.to)) queue.push(c.to);
    }
  }
  for (const id of pending) {
    shapes.set(id, { input: entryShape, output: entryShape });
  }
  return shapes;
}

function computeCnnShapes(
  flatBlocks: NeuraxBlock[],
  connections: Connection[],
  batch: number,
  inChannels: number,
  imgHeight: number,
  imgWidth: number,
): Map<string, { input: ShapeVec; output: ShapeVec }> {
  return computeShapesViaGraph(flatBlocks, connections, [batch, inChannels, imgHeight, imgWidth], cnnBlockOutputShape);
}

/**
 * GNN's tensor convention is `[batch, num_nodes, node_features]`, not an
 * image's `[B, C, H, W]` — a graph conv layer changes the feature
 * dimension (like a per-node dense layer), while node count only changes
 * at an explicit pooling/coarsening block. Reuses the same graph-traversal
 * engine as CNN with GNN-appropriate arithmetic instead.
 */
function gnnBlockOutputShape(blockType: string, params: Record<string, any> | undefined | null, inputShapes: ShapeVec[]): ShapeVec {
  const primary = inputShapes[0] ?? [1, 1, 1];
  const [B, N, F] = [primary[0] ?? 1, primary[1] ?? 1, primary[2] ?? 1];
  const p = params ?? {};

  switch (blockType) {
    case 'GCNConv':
    case 'SAGEConv':
    case 'GINConv':
    case 'ChebConv':
    case 'RGCNConv':
    case 'TAGConv':
    case 'ClusterGCNConv':
    case 'EdgeConv': {
      const outF = Number(p.out_channels ?? p.out_features ?? F);
      return [B, N, outF];
    }
    case 'GATConv':
    case 'GATv2Conv': {
      // Multi-head attention concatenates head outputs unless told not to.
      const perHead = Number(p.out_channels ?? p.out_features ?? F);
      const heads = Number(p.num_heads ?? p.heads ?? 1);
      const concatHeads = p.concat !== false;
      return [B, N, concatHeads ? perHead * heads : perHead];
    }
    case 'TopKPooling':
    case 'SAGPool': {
      const ratio = Number(p.ratio ?? 0.5);
      return [B, Math.max(1, Math.round(N * ratio)), F];
    }
    case 'GlobalMeanPool':
    case 'GlobalMaxPool':
    case 'GlobalAddPool':
      // Graph-level readout: pools across all nodes into one vector.
      return [B, F];
    case 'DenseProjection': {
      const inFeatures = primary.length === 2 ? primary[1] : F;
      const outFeatures = Number(p.out_features ?? p.num_classes ?? inFeatures);
      return [B, outFeatures];
    }
    default:
      return primary;
  }
}

function computeGnnShapes(
  flatBlocks: NeuraxBlock[],
  connections: Connection[],
  batch: number,
  numNodes: number,
  nodeFeatDim: number,
): Map<string, { input: ShapeVec; output: ShapeVec }> {
  return computeShapesViaGraph(flatBlocks, connections, [batch, numNodes, nodeFeatDim], gnnBlockOutputShape);
}

function toParserModelType(family: ArchitectureFamily): string {
  return family;
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
    /**
     * DeepSeek-style MoE models keep their first few layers dense (plain
     * feed-forward, no routing) before switching to routed experts —
     * `first_k_dense_replace` in the config. `repeat_scale_for` on the Rust
     * side reads this to scale the model's `mlp` and `moe` blocks against
     * their own real share of the depth instead of the full depth each.
     */
    numDenseLayers?: number | null;
    /**
     * An encoder-decoder model's (T5, BART, Pegasus...) decoder depth, when
     * it differs from `numLayers` (the encoder's). Read by
     * `repeat_scale_for` on the Rust side alongside a decoder-role layer's
     * own `encoder_decoder_role` tag — see `neurax-ir/src/architecture/mod.rs`.
     */
    numDecoderLayers?: number | null;
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

    // Training / optimisation
    optimizer?: string | null;
    weightDecay?: number | null;
    warmupSteps?: number | null;
    lrScheduler?: string | null;
    maxSteps?: number | null;
    gradientCheckpointing?: boolean | null;
    zeroStage?: number | null;
    earlyStoppingPatience?: number | null;

    // Parallelism
    tensorParallel?: number | null;
    pipelineParallel?: number | null;
    expertParallel?: number | null;
    microBatchSize?: number | null;
    gradAccumSteps?: number | null;

    /** User-defined hyperparameters, forwarded into `global_params`. */
    customParams?: Record<string, string | number | boolean> | null;

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
    numDenseLayers = null,
    numDecoderLayers = null,
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

    optimizer = null,
    weightDecay = null,
    warmupSteps = null,
    lrScheduler = null,
    maxSteps = null,
    gradientCheckpointing = null,
    zeroStage = null,
    earlyStoppingPatience = null,
    tensorParallel = null,
    pipelineParallel = null,
    expertParallel = null,
    microBatchSize = null,
    gradAccumSteps = null,
    customParams = null,

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
          ...(dim != null ? { hidden_size: dim } : {}),
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
          ...(dim != null ? { hidden_size: dim } : {}),
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
    // node.type, not blockType: BLOCK_TYPE_MAP only covers 'lstm_cell'/
    // 'gru_cell' explicitly, and everything else in this family (bilstm,
    // bigru, rnn_cell, lstm_peephole_cell, indylstm_cell, ...) falls
    // through to 'Opaque' there — but toParserLayerType() below resolves
    // any of them to a real backend RNN layer_type via the exact same
    // substring rule used here, so this must match just as broadly.
    const rnnNormalized = normalizeLayerTypeKey(node.type);
    if (
      rnnNormalized.includes('lstm') ||
      rnnNormalized.includes('gru') ||
      rnnNormalized === 'rnn' ||
      rnnNormalized === 'rnn_cell'
    ) {
      const inputDim = incomingIds.length > 0 ? (nodeDimMap.get(incomingIds[0]) ?? null) : null;
      params = fixRnnParams(params, inputDim);
    }

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
      const inputTensors = incomingOuter.map(srcId => tensorNames.get(srcId)).filter(Boolean) as string[];
      const outTensor = tensorNames.get(id) ?? `${id}_out`;
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

    const groupInputsOuter = (incomingMap.get(group.id) || []).map(srcId => tensorNames.get(srcId)).filter(Boolean) as string[];
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
        if (block.inputs.length > 0 && blocks.indexOf(block) > 0) {
          autoFixNotes.push(`Block "${block.id}" expects input tensor "${inp}" which is not produced by any prior block.`);
        }
      }
    }
    for (const out of block.outputs) {
      producedTensors.add(out);
    }
  }

  // 6. Client-side shape validation (pass through topological order)
  const blockOutputDims = new Map<string, number>();
  for (const block of blocks) {
    const params = block.params || {};
    let outputDim: number | undefined;
    switch (block.type) {
      case 'Embedding':
        outputDim = getNumericParam(params, ['d_model', 'dim', 'dModel']);
        break;
      case 'DenseProjection':
      case 'LMHead':
        outputDim = getNumericParam(params, ['out_features', 'outFeatures']);
        break;
      case 'RMSNorm':
      case 'LayerNorm':
        outputDim = getNumericParam(params, ['normalized_shape', 'dim', 'd_model']);
        break;
    }
    if (outputDim != null) {
      blockOutputDims.set(block.id, outputDim);
    }
    if (block.inputs.length > 0) {
      const srcId = block.inputs[0];
      const prevDim = blockOutputDims.get(srcId);
      if (prevDim != null) {
        const expectedIn = getNumericParam(params, ['in_features', 'inFeatures', 'd_model', 'hidden_size']);
        if (expectedIn != null && expectedIn !== prevDim) {
          autoFixNotes.push(
            `Shape mismatch: "${block.id}" expects in_features=${expectedIn} but previous block outputs dim=${prevDim}.`
          );
        }
      }
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
  // RNN's own hwConfig preset sets `hiddenSize` (its own field, distinct
  // from the transformer-oriented `hiddenDim`) — falls back to it before
  // the hardcoded default so an RNN build's actual configured hidden size
  // reaches shape inference instead of silently landing on 0. `||`, not
  // `??`: `hiddenDim` defaults to a real `0` (not null/undefined) whenever
  // a family's own preset doesn't set it, and 0 is never a valid size.
  const hidden = hiddenDim || hiddenSize || 768;

  const flatBlocksRaw = flattenBlocks(blocks).filter((block) => block.ui_node_type !== 'input' && block.ui_node_type !== 'output');

  // A "Hybrid SSM+Attention" canvas block (Jamba-style) represents two real
  // sublayers sharing one node. Left as-is, `toParserLayerType` resolves
  // "hybrid_ssm_attn" via its generic `includes('ssm')` rule to a plain
  // `state_space` block — the attention half never reached the compiled
  // model at all, silently. Split it here into its own `attention` and
  // `mamba_block` representatives, each carrying `repeat_fraction` (from
  // `mix_ratio`) so `repeat_scale_for` on the Rust side scales them against
  // their own share of the model's depth (`ai21labs/Jamba-v0.1`: attention
  // on 4 of 32 layers) instead of the full depth each.
  // A CNN "stage" node (the Templates catalogue's `bottleneck_block`/
  // `basic_block`/... convention: `{ planes, blocks, stride, expansion }`
  // standing for `blocks` repeated real blocks) used to compile in a way
  // that silently dropped every one of those fields. Two bugs stacked:
  // `toParserLayerType` folded `bottleneck_block` into the same bucket as
  // `basic_block` (`residual_block`/`LayerType::ResidualBlock`, ResNet's
  // 2-conv block) even though the parser already accepts `bottleneck_block`
  // itself as a direct alias for `LayerType::ResnetBottleneck` (the real
  // 3-conv 1x1/3x3/1x1 block ResNet-50+ actually uses); and neither
  // `LayerType` reads `planes`/`blocks`/`expansion` at all, only
  // `in_channels`/`out_channels`/`mid_channels`/`stride`/`cardinality` — so
  // every stage silently fell back to the same default 64->64 block
  // regardless of what stage it was. Verified empirically against the
  // existing ResNet-50 template before this fix: 384,808 measured against
  // a published 25,600,000 (98.5% off).
  //
  // `cnnStageChannels` tracks the running output-channel count across
  // stages in encounter order (seeded at 64, the real stem output for every
  // template that uses this convention) so each stage's *first* block gets
  // the real previous stage's channel count as its `in_channels` — exactly
  // how a real ResNet-family stage transition works (only the first block
  // changes channels/stride; the rest are in==out, stride 1).
  let cnnStageChannels = 64;
  const flatBlocks: NeuraxBlock[] = [];
  for (const block of flatBlocksRaw) {
    const sourceType = String(block.ui_node_type ?? block.type ?? '').toLowerCase();

    if (sourceType === 'hybrid_ssm_attn') {
      const p = block.params ?? {};
      const width = Number(p.d_model ?? hidden);
      const mixRatio = Math.min(1, Math.max(0, Number(p.mix_ratio ?? 0.5)));
      const numHeadsHybrid = Number(p.n_heads ?? 8);
      flatBlocks.push({
        ...block,
        id: `${block.id}-attn`,
        ui_node_type: 'attention' as LayerType,
        params: {
          hidden_size: width,
          num_heads: numHeadsHybrid,
          causal: true,
          repeat_fraction: mixRatio,
        },
      });
      flatBlocks.push({
        ...block,
        id: `${block.id}-mamba`,
        ui_node_type: 'mamba_block' as LayerType,
        params: {
          hidden_size: width,
          state_dim: Number(p.d_state ?? 16),
          expansion_factor: Number(p.expand ?? 2),
          repeat_fraction: Number((1 - mixRatio).toFixed(6)),
        },
      });
      continue;
    }

    const p = block.params ?? {};
    const isCnnStage =
      (sourceType === 'bottleneck_block' || sourceType === 'basic_block' || sourceType === 'stage_block') &&
      typeof p.planes === 'number' &&
      typeof p.blocks === 'number';
    if (isCnnStage) {
      const outCh = Number(p.planes);
      const numBlocks = Math.max(1, Math.round(Number(p.blocks)));
      const stageStride = Number(p.stride ?? 1);
      const expansion = Number(p.expansion ?? 4);
      const cardinality = typeof p.cardinality === 'number' ? Number(p.cardinality) : undefined;
      const baseWidth = typeof p.base_width === 'number' ? Number(p.base_width) : undefined;
      // ResNeXt/RegNet-style grouped bottlenecks state their real per-group
      // inner width directly (cardinality x base_width) rather than the
      // plain out/expansion a standard bottleneck uses — reading it when
      // present, instead of always deriving it, is what makes those two
      // families' grouped convolutions cost what they really cost instead
      // of a plain bottleneck's.
      const midCh =
        cardinality !== undefined && baseWidth !== undefined
          ? cardinality * baseWidth
          : Math.max(1, Math.round(outCh / Math.max(expansion, 1)));
      const isBottleneck = expansion > 1;
      const startInCh = cnnStageChannels;

      for (let i = 0; i < numBlocks; i++) {
        const blockInCh = i === 0 ? startInCh : outCh;
        const blockStride = i === 0 ? stageStride : 1;
        flatBlocks.push({
          ...block,
          id: `${block.id}-b${i}`,
          ui_node_type: (isBottleneck ? 'bottleneck_block' : 'residual_block') as LayerType,
          params: {
            in_channels: blockInCh,
            out_channels: outCh,
            ...(isBottleneck ? { mid_channels: midCh } : {}),
            stride: blockStride,
            ...(cardinality !== undefined ? { cardinality } : {}),
          },
        });
      }
      cnnStageChannels = outCh;
      continue;
    }

    // A plain VGG-style `conv2d` node whose *name* states a repeat
    // ("Conv Block 3: 4× [256, 3x3]") but whose params never did — the
    // label was cosmetic, the compiled graph always held exactly one conv,
    // regardless of what the name claimed. `blocks` makes the repeat real:
    // only the first of the `blocks` convs changes channels (VGG-16/19's
    // stem->256->512 transitions happen once per stage, at its first conv),
    // the rest are same-channel 3x3 convs, mirroring the CNN stage-block
    // handling above for the same reason.
    if (sourceType === 'conv2d' && typeof p.blocks === 'number' && Number(p.blocks) > 1) {
      const outCh = Number(p.out_channels ?? p.in_channels ?? 64);
      const inCh = Number(p.in_channels ?? outCh);
      const numBlocks = Math.max(1, Math.round(Number(p.blocks)));
      for (let i = 0; i < numBlocks; i++) {
        flatBlocks.push({
          ...block,
          id: `${block.id}-b${i}`,
          params: {
            ...p,
            in_channels: i === 0 ? inCh : outCh,
            out_channels: outCh,
          },
        });
      }
      continue;
    }

    flatBlocks.push(block);
  }

  // A `layer_stack` node means one of two different things depending on who
  // built the design. The Templates catalogue compiles a repeated block down
  // to the stack node alone — nothing else in `flatBlocks` describes what's
  // inside it, which is exactly the shape that priced at 0 params/FLOPs (see
  // `decoderStackParamsAndFlops`). The HuggingFace importer instead lays out
  // real Attention/Mlp/Normalization nodes as the stack's "body" and connects
  // them back to it (see `buildGraph` in huggingfaceImporter.ts) — the
  // backend's existing `repeat_scale_for` already scales *those* nodes by
  // `num_layers`, so also pricing the stack node itself would double-count
  // every repeated layer. Only synthesize params/FLOPs for the former case.
  const hasIndividuallyTypedDecoderBody = flatBlocks.some((b) => {
    const t = toParserLayerType(b.ui_node_type ?? b.type);
    return t === 'attention' || t === 'mlp';
  });

  // gan and diffusion are the same [B, C, H, W] image convention cnn is —
  // same conv/pool/upsample arithmetic, same graph-propagation engine.
  // Defaults mirror each family's own hwConfig preset (diffusion starts
  // smaller: 64px, 4 latent channels, vs. cnn/gan's 224px/64px RGB).
  const isImageFamily = family === 'cnn' || family === 'gan' || family === 'diffusion';
  const imageDefaultRes = family === 'diffusion' ? 64 : family === 'gan' ? 64 : 224;
  const imageDefaultChannels = family === 'diffusion' ? 4 : 3;
  const cnnShapes = isImageFamily
    ? computeCnnShapes(
        flatBlocks, connections, batch,
        inChannels ?? imageDefaultChannels,
        imgHeight ?? imageDefaultRes,
        imgWidth ?? imageDefaultRes,
      )
    : null;
  const gnnShapes = family === 'gnn'
    ? computeGnnShapes(flatBlocks, connections, batch, numNodes ?? 2708, nodeFeatDim ?? 16)
    : null;

  const layers: NeuraxLayer[] = flatBlocks.map(block => {
    // Derive shapes from block type and params
    let inputShape: number[] = [];
    let outputShape: number[] = [];

    const p = block.params ?? {};

    const graphShape = cnnShapes?.get(block.id) ?? gnnShapes?.get(block.id);
    if (graphShape) {
      inputShape = graphShape.input;
      outputShape = graphShape.output;
    } else switch (block.type) {
      case 'Input': {
        inputShape = [];
        // Image-shaped families describe their input as [B, C, H, W], not
        // the [B, S, D] a sequence model uses — using the latter regardless
        // of family (as this unconditionally did) fed every CNN build a
        // shape built from `seq`/`hidden`, values that mean nothing for
        // images, and 0 whenever nothing had set them. That 0 propagated
        // through every downstream layer's shape inference, so the whole
        // network came out empty and the compiler rejected it.
        const isImageShaped = family === 'cnn' || family === 'gan' || family === 'diffusion';
        outputShape = isImageShaped
          ? [batch, inChannels ?? 3, imgHeight ?? 224, imgWidth ?? 224]
          : [batch, seq, p.dim ?? hidden];
        break;
      }
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

    // A gated feed-forward block collapses to `mlp` for the compiler, which
    // then has no way to know it holds three weight matrices rather than two.
    // Without this the SwiGLU stacks in LLaMA, Mistral and friends were counted
    // as plain MLPs — about 1.5B parameters short on a 7B model.
    const sourceType = String(block.ui_node_type ?? block.type ?? '').toLowerCase();
    const isGatedFeedForward =
      sourceType === 'ffn_gated' || sourceType === 'swiglu' || sourceType === 'geglu';

    // The repeated decoder-block placeholder ("12x Decoder Blocks" on the
    // canvas) compiles to a single opaque `custom` layer with nothing but its
    // own repeat count — the backend has no formula for "a block that
    // secretly repeats N times" and silently prices it at 0 params/FLOPs.
    // See `decoderStackParamsAndFlops` for the full story.
    let stackOverrides: { params?: Record<string, any>; custom_equations?: Record<string, string> } = {};
    if (sourceType === 'layer_stack' && !hasIndividuallyTypedDecoderBody) {
      const stackLayers = Number(p.num_layers ?? numLayers ?? 1);
      const { paramCount, flopsForward } = decoderStackParamsAndFlops({
        numLayers: stackLayers,
        hidden,
        heads: numHeads ?? 8,
        kvHeads: kvHeads ?? null,
        headDim: headDim ?? p.head_dim ?? null,
        ffnDim: ffnDim ?? 4 * hidden,
        activation: activation ?? null,
        gated: typeof p.gated === 'boolean' ? p.gated : null,
        batch,
        seq,
      });
      stackOverrides = {
        params: { ...p, param_count: paramCount },
        custom_equations: { flops_forward: String(flopsForward) },
      };
    }

    return {
      id: block.id,
      layer_type: toParserLayerType(block.ui_node_type ?? block.type),
      input_shape: inputShape,
      output_shape: outputShape,
      params: stackOverrides.params ?? (isGatedFeedForward ? { ...p, gated: true } : p),
      ...(stackOverrides.custom_equations && { custom_equations: stackOverrides.custom_equations }),
      ...(block.comment && { comment: block.comment }),
    };
  });

  // Build global_params from env
  const global_params: NeuraxGlobalParams = {};
  if (hiddenDim != null) global_params.hidden_size = hiddenDim;
  if (numLayers != null) global_params.num_layers = numLayers;
  if (numDenseLayers != null) global_params.num_dense_layers = numDenseLayers;
  if (numDecoderLayers != null) global_params.num_decoder_layers = numDecoderLayers;
  if (vocabSize != null) global_params.vocab_size = vocabSize;
  if (resolvedSeqLen != null) global_params.sequence_length = resolvedSeqLen;
  if (numHeads != null) global_params.num_heads = numHeads;
  if (headDim != null) global_params.head_dim = headDim;
  if (ffnDim != null) global_params.ffn_dim = ffnDim;
  if (numExperts != null) global_params.num_experts = numExperts;
  if (topK != null) global_params.top_k = topK;
  // GNN graph size — read by the backend's message-passing/GCN/GAT FLOPs
  // formulas (edge count sets how many messages actually get passed). These
  // used to only ever be written onto a local `env` object that nothing
  // downstream of this function ever read; `global_params` is the field the
  // backend's `GlobalResolutionContext` actually parses (`get_u64("num_nodes")`).
  if (numNodes != null) global_params.num_nodes = numNodes;
  if (numEdges != null) global_params.num_edges = numEdges;
  if (nodeFeatDim != null) global_params.node_features = nodeFeatDim;
  if (edgeFeatDim != null) global_params.edge_features = edgeFeatDim;

  // Settings that describe the training regime rather than the tensor shapes.
  // They ride in global_params, which the backend keeps as a flattened
  // catch-all, so they reach the analysis without a schema change on either side.
  if (weightDecay != null) global_params.weight_decay = weightDecay;
  if (lrScheduler != null) global_params.lr_scheduler = lrScheduler;
  if (earlyStoppingPatience != null) global_params.early_stopping_patience = earlyStoppingPatience;
  if (expertParallel != null) global_params.expert_parallel = expertParallel;
  if (microBatchSize != null) global_params.micro_batch_size = microBatchSize;
  if (gradAccumSteps != null) global_params.gradient_accumulation_steps = gradAccumSteps;

  // User-defined hyperparameters last, so an explicit entry wins over a
  // built-in of the same name rather than being silently discarded.
  if (customParams) {
    for (const [key, value] of Object.entries(customParams)) {
      if (key.trim() !== '' && value !== undefined && value !== null) {
        global_params[key] = value;
      }
    }
  }

  // Build training config
  const trainingConfig: NeuraxTraining = {
    batch_size: batchSize ?? 1,
    precision: normalizePrecision(precision),
    ...(learningRate != null && { learning_rate: learningRate }),
    ...(numEpochs != null && { num_epochs: numEpochs }),
    ...(resolvedSeqLen != null && resolvedSeqLen > 0 && { sequence_length: resolvedSeqLen }),
    // Optimisation and parallelism settings. These were declared on
    // NeuraxTraining but never emitted, so choosing an optimizer, a warmup or a
    // parallelism degree in the UI had no effect on the analysis.
    ...(optimizer != null && { optimizer }),
    ...(warmupSteps != null && { warmup_steps: warmupSteps }),
    ...(maxSteps != null && maxSteps > 0 && { max_steps: maxSteps }),
    ...(gradientCheckpointing != null && { gradient_checkpointing: gradientCheckpointing }),
    ...(zeroStage != null && { zero_stage: zeroStage }),
    ...((tensorParallel != null || pipelineParallel != null) && {
      parallelism: {
        // Data parallelism takes whatever the other two degrees leave.
        data_parallel: Math.max(
          1,
          Math.floor((gpuCount ?? 1) / Math.max(1, (tensorParallel ?? 1) * (pipelineParallel ?? 1))),
        ),
        tensor_parallel: tensorParallel ?? 1,
        pipeline_parallel: pipelineParallel ?? 1,
      },
    }),
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
    ...(autoFixNotes.length > 0 && { _warnings: autoFixNotes }),
  };
}
