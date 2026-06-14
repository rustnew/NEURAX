export type LayerType =
  | 'input'
  | 'output'
  // Legacy compat
  | 'dense'
  | 'conv2d'
  | 'relu'
  | 'gelu'
  | 'attention'
  | 'residual'
  | 'transformer'
  // Projections
  | 'linear_projection'
  | 'lora_linear'
  | 'dora_linear'
  | 'q_projection'
  | 'k_projection'
  | 'v_projection'
  | 'qkv_combined'
  | 'mqa_projection'
  | 'gqa_projection'
  | 'embedding'
  | 'token_embedding'
  // Convolutions
  | 'conv1d'
  | 'conv3d'
  | 'depthwise_conv'
  | 'transposed_conv'
  | 'pointwise_conv'
  | 'separable_conv'
  | 'deformable_conv'
  | 'large_kernel_conv'
  | 'mixed_kernel_conv'
  | 'dynamic_conv'
  | 'sparse_conv'
  | 'involution'
  | 'gated_conv'
  // Attention
  | 'attention_score'
  | 'attention_aggregation'
  | 'cross_attention'
  | 'mha_attention'
  | 'mqa_attention'
  | 'gqa_attention'
  | 'mla_attention'
  | 'sliding_window_attention'
  | 'dilated_attention'
  | 'sparse_attention'
  | 'linear_attention'
  | 'flash_attention'
  | 'flex_attention'
  // Pooling
  | 'max_pool'
  | 'avg_pool'
  | 'adaptive_pool'
  | 'global_pool'
  | 'roi_pool'
  | 'roi_align'
  // Normalization
  | 'layernorm'
  | 'rmsnorm'
  | 'batchnorm'
  | 'groupnorm'
  | 'instancenorm'
  | 'sync_batchnorm'
  | 'switchable_norm'
  | 'filter_response_norm'
  | 'adaptive_instance_norm'
  | 'conditional_batchnorm'
  | 'spectral_norm'
  // Feed-Forward
  | 'ffn_standard'
  | 'ffn_gated'
  | 'ffn_parallel'
  | 'moe_block'
  // Positional Encoding
  | 'pos_absolute'
  | 'pos_rope'
  | 'pos_alibi'
  | 'pos_relative_bias'
  | 'pos_xpos'
  | 'pos_fire'
  // Structure
  | 'residual_add'
  | 'concat'
  | 'flatten'
  | 'reshape'
  | 'dropout'
  | 'layer_stack'
  | 'gradient_checkpoint'
  | 'permute'
  | 'stochastic_depth'
  | 'skip_connection'
  | 'upsample'
  | 'global_pool'
  // CNN — Architecture Skeleton (extra)
  | 'stem_block'
  | 'stage_block'
  | 'residual_block'
  // Output heads
  | 'lm_head'
  | 'classification_head'
  | 'detection_head'
  | 'segmentation_head'
  | 'mask_head'
  | 'roi_head'
  | 'rpn_head'
  // CNN — Residual & Dense Blocks
  | 'basic_block'
  | 'bottleneck_block'
  | 'preact_block'
  | 'dense_layer'
  | 'dense_block'
  | 'transition_layer'
  | 'inverted_bottleneck'
  | 'convnext_block'
  | 'mbconv_block'
  | 'fused_mbconv'
  // CNN — Attention Mechanisms
  | 'se_layer'
  | 'cbam_layer'
  | 'eca_layer'
  | 'non_local_block'
  | 'gc_net'
  | 'coord_conv'
  | 'self_attention_2d'
  // CNN — Vision Transformer Blocks
  | 'patch_embed'
  | 'class_token'
  | 'transformer_encoder_block'
  | 'cvt_block'
  | 'levit_block'
  // CNN — Architecture Skeleton
  | 'stem_block'
  | 'stage_block'
  | 'downsample_block'
  | 'upsample_block'
  // CNN — Segmentation & Detection
  | 'unet_block'
  | 'fpn_block'
  | 'anchor_generator'
  // CNN — Mixer & Future
  | 'mlp_mixer'
  | 'res_mlp'
  | 'conv_mixer'
  | 'adaptive_inference_block'
  | 'nas_cell'
  // MoE — Routing
  | 'router_linear'
  | 'router_softmax'
  | 'noisy_topk_router'
  | 'expert_choice_router'
  | 'non_trainable_router'
  | 'product_key_router'
  | 'hierarchical_router'
  | 'sinkhorn_router'
  // MoE — Experts
  | 'expert_linear'
  | 'expert_gated_ffn'
  | 'expert_multihead'
  | 'expert_scalar'
  | 'expert_memory'
  // MoE — Aggregation
  | 'expert_dispatch'
  | 'expert_combine'
  | 'output_combination'
  | 'concat_projection'
  | 'attention_pooling'
  | 'shared_expert'
  // MoE — Regularization
  | 'load_balancing_loss'
  | 'expert_capacity_limit'
  | 'z_loss'
  | 'router_regularization'
  | 'jitter_noise'
  | 'capacity_computation'
  | 'mask_generation'
  // MoE — Composite
  | 'moe_layer'
  | 'moa_block'
  | 'fine_grained_moe'
  | 'soft_moe'
  | 'peer_layer'
  | 'hierarchical_moe'
  // SSM — Discretization
  | 'ssm_discretize'
  | 'delta_computation'
  // SSM — Fundamental
  | 'state_matrix_a'
  | 'state_matrix_b'
  | 'state_matrix_c'
  | 's4_block'
  | 's5_block'
  | 's6_block'
  | 'lru_block'
  | 'linoss_block'
  // SSM — Selectivity
  | 'selective_scan'
  | 'input_dependent_timescale'
  | 's7_selection'
  | 'serpent_selection'
  // SSM — Scan Strategies
  | 'scan_1d'
  | 'scan_2d'
  | 'scan_multidirectional'
  | 'scan_spiral'
  | 'scan_diagonal'
  // SSM — Convolution / Projection
  | 'causal_conv1d'
  | 'ssm_in_proj'
  | 'ssm_out_proj'
  | 'delta_proj'
  | 'bc_proj'
  // SSM — Gating
  | 'glu_block'
  | 'hadamard_product'
  // SSM — Composite Architectures
  | 'vss_block'
  | 'rssg_block'
  | 'basic_layer_ssm'
  | 'stg_mamba_block'
  | 'dual_path_mamba'
  | 'mamba_mixer'
  // SSM — Multi-Scale
  | 'multiscale_ssm'
  | 'parallel_differencing'
  // SSM — Neuromorphic
  | 'spiking_ssm'
  | 'neuromorphic_activation'
  // SSM — Extended Fundamental
  | 'h3_block'
  | 'hyena_conv'
  | 'gated_ssm'
  | 'neural_cde'
  | 'a_parameterization'
  // SSM — Hybrid & Multimodal
  | 'hybrid_ssm_attn'
  | 'multimodal_mamba'
  // SSM — Scan Extended
  | 'scan_operator'
  | 'scan_block'
  // SSM — Utilities
  | 'state_reset'
  | 'ssm_layernorm'
  // SSM — Output
  | 'ssm_output_head'
  | 'forecasting_head'
  // GNN — Message Passing
  | 'gcn_conv'
  | 'sage_conv'
  | 'gat_conv'
  | 'gat_v2_conv'
  | 'gin_conv'
  | 'cheb_conv'
  | 'mf_conv'
  | 'rgcn_conv'
  | 'tag_conv'
  | 'arma_conv'
  | 'sg_conv'
  | 'appnp'
  | 'dna_conv'
  | 'cluster_gcn_conv'
  // GNN — Edge Features
  | 'edge_conv'
  | 'nn_conv'
  // GNN — Pooling
  | 'topk_pooling'
  | 'sag_pool'
  | 'edge_pooling'
  | 'asa_pooling'
  | 'global_max_pool'
  | 'global_mean_pool'
  | 'global_add_pool'
  // GNN — Normalization & Regularization
  | 'graph_norm'
  | 'pair_norm'
  | 'mean_subtraction_norm'
  | 'edge_dropout'
  // GNN — Heterogeneous
  | 'hetero_conv'
  | 'hgt_conv'
  | 'han_conv'
  // GNN — Temporal
  | 'tgcn'
  | 'stgcn'
  | 'dy_gr_encoder'
  // GNN — Readout
  | 'global_attention_readout'
  | 'set2set_readout'
  | 'graph_readout_general'
  // GNN — Positioning & Structural
  | 'random_walk_pe'
  | 'laplacian_pe'
  | 'distance_encoding'
  // GNN — Transformation
  | 'add_self_loops'
  | 't_normalize_features'
  | 't_to_undirected'
  // GNN — Advanced
  | 'mpnn_structure'
  | 'deeper_gcn'
  // GNN — Legacy / Core
  | 'message_aggregate'
  | 'graph_conv'
  | 'graph_attention'
  | 'graph_readout'
  // RNN
  | 'rnn_cell'
  | 'lstm_cell'
  | 'gru_cell'
  // RNN/LSTM/GRU — Fundamental Cells
  | 'elman_rnn_cell'
  | 'jordan_rnn_cell'
  | 'lstm_peephole_cell'
  | 'lstm_coupled_cell'
  | 'lstm_projection_cell'
  | 'mgu_cell'
  | 'ugrnn_cell'
  | 'sru_cell'
  | 'srupp_cell'
  | 'indrnn_cell'
  | 'indylstm_cell'
  | 'phased_lstm_cell'
  | 'qrnn_cell'
  | 'lmu_cell'
  | 'fast_weights_rnn_cell'
  | 'hebbian_rnn_cell'
  // RNN/LSTM/GRU — Wrappers
  | 'bidirectional_wrapper'
  | 'stacked_rnn'
  | 'residual_rnn_layer'
  | 'dropout_rnn_layer'
  | 'zoneout_wrapper'
  // RNN/LSTM/GRU — Specialized Architectures
  | 'conv_lstm_cell'
  | 'conv_gru_cell'
  | 'stlstm_cell'
  | 'grid_lstm'
  | 'tree_lstm'
  | 'graph_lstm'
  | 'depth_gated_lstm'
  | 'gated_feedback_lstm'
  // RNN/LSTM/GRU — Seq2Seq
  | 'many_to_one_encoder'
  | 'one_to_many_decoder'
  | 'many_to_many_encoder_decoder'
  | 'seq2seq_attention'
  | 'siamese_rnn'
  // RNN/LSTM/GRU — Internal Gates
  | 'lstm_input_gate'
  | 'lstm_forget_gate'
  | 'lstm_output_gate'
  | 'lstm_cell_modulation'
  | 'gru_reset_gate'
  | 'gru_update_gate'
  | 'gru_candidate_activation'
  // RNN/LSTM/GRU — Normalization & Regularization
  | 'layernorm_rnn'
  | 'weightnorm_rnn'
  | 'recurrent_batchnorm'
  | 'variational_dropout_rnn'
  | 'zoneout'
  // RNN/LSTM/GRU — Future Concepts
  | 'memory_augmented_rnn'
  | 'act_rnn'
  | 'rim_block'
  | 'neural_turing_machine'
  // GAN — Fundamental
  | 'gan_noise_z'
  | 'gan_noise_w'
  | 'gan_label_embedding'
  | 'gan_generator'
  | 'gan_discriminator'
  | 'gan_adversarial_loss'
  | 'gan_minimax_loss'
  | 'gan_non_saturating_loss'
  // GAN — DCGAN & Basic Conv
  | 'dcgan_generator_block'
  | 'dcgan_discriminator_block'
  | 'upsample_conv_block'
  | 'pixel_shuffle'
  | 'pixel_unshuffle'
  | 'checkerboard_removal'
  // GAN — Conditional & InfoGAN
  | 'conditional_embedding'
  | 'label_conditioning'
  | 'infogan_latent_code'
  | 'q_network_head'
  | 'mutual_info_loss'
  | 'auxiliary_classifier'
  // GAN — WGAN & LSGAN
  | 'wgan_critic'
  | 'gradient_penalty'
  | 'lipschitz_constraint'
  | 'weight_clipping'
  | 'lsgan_loss'
  | 'hinge_loss'
  | 'wasserstein_distance'
  // GAN — Projection & Mapping
  | 'mapping_network'
  | 'mapping_linear'
  | 'style_projection'
  | 'noise_injection'
  | 'style_modulation'
  | 'style_demodulation'
  | 'weight_demodulation'
  | 'modulated_conv2d'
  | 'adain_style'
  // GAN — StyleGAN Specific
  | 'stylegan_synthesis_block'
  | 'stylegan_res_block'
  | 'progressive_growing_step'
  | 'fading_layer'
  | 'torgb_layer'
  | 'fromrgb_layer'
  | 'equalized_lr_linear'
  | 'equalized_lr_conv'
  | 'blur_filter'
  | 'antialias_upsample'
  | 'noise_broadcast'
  // GAN — StyleGAN3 Specific
  | 'filtered_lrelu'
  | 'fourier_features'
  | 'transformed_conv2d'
  | 'config_independent_conv'
  | 'sinc_filter'
  // GAN — Multi-Scale & Image Translation
  | 'cyclegan_block'
  | 'pix2pix_generator'
  | 'pix2pix_discriminator'
  | 'patch_gan_discriminator'
  | 'multiscale_discriminator'
  | 'perceptual_loss'
  | 'vgg_feature_extractor'
  | 'pixel_wise_loss'
  | 'total_variation_loss'
  | 'id_loss'
  // GAN — BigGAN & High-Res
  | 'biggan_res_block'
  | 'non_local_gan'
  | 'self_attention_gan'
  | 'orthogonal_reg'
  | 'spectral_norm_wrapper'
  | 'truncation_trick'
  | 'shared_residual_block'
  // GAN — Video & Temporal
  | 'video_generator_3d'
  | 'temporal_discriminator'
  | 'spatiotemporal_conv'
  | 'flow_warping_layer'
  | 'optic_flow_loss'
  | 'frame_interpolation_block'
  | 'recurrent_gan_cell'
  // GAN — Special Applications
  | 'super_res_block'
  | 'esrgan_dense_block'
  | 'text_to_image_fusion'
  | 'mask_conditioned_gan'
  | 'medical_gan_block'
  | 'domain_adaptation_layer'
  // GAN — Latent Space & Sampling
  | 'latent_interpolation'
  | 'latent_traversal'
  | 'eigen_discovery'
  | 'gan_inversion'
  | 'pivotal_tuning_step'
  // GAN — Regularization & Metrics
  | 'r1_regularization'
  | 'r2_regularization'
  | 'lazy_regularization'
  | 'fid_metric_node'
  | 'is_metric_node'
  // GAN — Future Concepts
  | 'r3gan_block'
  | 'diffusion_gan_hybrid'
  | 'any_res_gan_block'
  | 'lightweight_gan_block'
  // Diffusion — Encoders
  | 'vae_encoder'
  | 'vae_decoder'
  | 'image_encoder'
  | 'text_encoder'
  | 'class_encoder'
  // Diffusion — Denoisers
  | 'unet_2d_cond'
  | 'unet_model'
  | 'dit_block'
  | 'mmdit_block'
  | 'flag_dit'
  | 'next_dit'
  // Diffusion — U-Net Base
  | 'diff_resblock'
  | 'spatial_transformer'
  | 'basic_transformer_block'
  | 'downsample_2d'
  | 'upsample_2d'
  | 'timestep_embed_seq'
  | 'cross_attn_down_block'
  | 'cross_attn_up_block'
  | 'down_block_2d'
  | 'up_block_2d'
  | 'unet_mid_block'
  // Diffusion — Conditioning
  | 'timestep_embedding'
  | 'timestep_projection'
  | 'sinusoidal_timestep_embed'
  | 'text_projection'
  | 'clip_embedding'
  | 'class_embedding'
  | 'classifier_free_guidance'
  | 'image_projection'
  | 'controlnet_block'
  | 'ip_adapter'
  // Diffusion — Noise & Scheduling
  | 'noise_schedule'
  | 'ddpm_scheduler'
  | 'ddim_scheduler'
  | 'euler_scheduler'
  | 'dpm_solver'
  | 'flow_match_scheduler'
  | 'gaussian_noise'
  | 'forward_diffusion'
  | 'reverse_diffusion'
  | 'latent_diffusion_step'
  // Diffusion — Normalization
  | 'ada_group_norm'
  | 'sandwich_norm'
  // Diffusion — SD/LDM Specific
  | 'autoencoder_kl'
  // Diffusion — PixArt / DiT Specific
  | 'adaln_single'
  | 'patchify'
  | 'depatchify'
  // Diffusion — SD3/Flux Specific
  | 'double_stream_block'
  | 'single_stream_block'
  | 'rectified_flow'
  // Diffusion — Lumina Specific
  | 'rope_3d'
  | 'freq_aware_rope'
  | 'time_aware_rope'
  | 'context_drop'
  // Diffusion — Output
  | 'diff_output_layer'
  | 'final_conv'
  | 'image_decoder'
  // Diffusion — Action-Conditioned
  | 'action_encoder'
  | 'action_cond_unet'
  | 'dynamics_predictor'
  | 'world_model'
  // Diffusion — Emerging
  | 'reg_injector'
  | 'self_attention_guidance'
  | 'cascade_multiscale'
  | 'patch_ddm'
  // Diffusion — Legacy compat
  | 'time_embedding'
  | 'unet_downsample'
  | 'unet_upsample'
  // RL
  | 'policy_head'
  | 'value_head'
  | 'advantage_stream'
  // SNN
  | 'lif_neuron'
  | 'spike_encoder'
  | 'stdp_synapse'
  | 'rate_encoder'
  | 'synaptic_layer'
  | 'leaky_neuron'
  // Diffusion — extra composite
  | 'unet_block'
  | 'dit_block'
  | 'noise_scheduler'
  // MoE — composite
  | 'moe_layer';

export type ActivationFunction =
  | 'none'
  | 'relu'
  | 'gelu'
  | 'gelu_tanh'
  | 'gelu_fast'
  | 'gelu_new'
  | 'silu'
  | 'swish'
  | 'tanh'
  | 'sigmoid'
  | 'leaky_relu'
  | 'elu'
  | 'mish'
  | 'softmax'
  | 'swiglu'
  | 'geglu'
  | 'reglu'
  | 'bilinear'
  | 'solu'
  | 'solu_ln';

export type ParameterValue = number | string | boolean | object | any[];

export interface LayerConfig {
  id: string;
  type: LayerType;
  name: string;
  icon: string;
  description: string;
  defaultParams: Record<string, ParameterValue>;
  /** Which params support activation dropdowns */
  hasActivation?: boolean;
  /** Tooltip for the block */
  tooltip?: string;
  /** Category for grouping in palette */
  category?: string;
  /** Custom color for the block (e.g. hsl, hex) */
  color?: string;
}

export interface CanvasNode {
  id: string;
  type: LayerType;
  name: string;
  x: number;
  y: number;
  params: Record<string, ParameterValue>;
  inputShape?: string;
  outputShape?: string;
}

export interface Connection {
  id: string;
  from: string;
  to: string;
}

export interface NodeGroup {
  id: string;
  name: string;
  nodeIds: string[];
  connectionIds: string[];
  repeatCount: number;
  x: number;
  y: number;
  /** Collapsed = shown as single block; expanded = shown as individual nodes */
  collapsed: boolean;
}

export interface OpsDistribution {
  [opType: string]: number;
}

export interface AnalysisResult {
  // Model stats
  totalParams: number;
  numLayers: number;
  modelType: string;
  graphDepth: number;
  totalOperations: number;
  criticalPathLength: number;
  tensorResolutionRatio: number;
  unresolvedDimCount: number;
  totalTensorCount: number;
  largestTensorBytes: number;
  opsDistribution: OpsDistribution;

  // Compute
  totalFlops: number;
  forwardFlops: number;
  backwardFlops: number;
  flopsPerToken: number;
  arithmeticIntensity: number;
  bottleneck: string;
  rooflinePosition: number;

  // Formatted strings for display
  estimatedFlops: string;
  forwardFlopsHuman: string;
  backwardFlopsHuman: string;

  // Memory (bytes)
  peakVramBytes: number;
  parameterMemoryBytes: number;
  activationMemoryBytes: number;
  gradientMemoryBytes: number;
  optimizerStateBytes: number;
  maxBatchSizeFit: number;
  memoryFragmentation: number;
  memoryUsage: string;

  // Hardware
  gpuName: string;
  gpuCount: number;
  gpuMemoryGb: number;
  gpuTflops: number;
  gpuBandwidthGbs: number;
  interconnect?: string;
  interconnectBandwidthGbs?: number;

  // Parallelism
  dataParallelEfficiency: number;
  communicationOverhead: number;
  optimalGpuCount: number;
  pipelineStages: number;
  tensorParallelDegree: number;
  dataParallel?: number;
  tensorParallel?: number;
  pipelineParallel?: number;

  // Performance
  latencyMs: number | null;
  throughputTokensPerS: number;
  throughputGraphsPerS?: number | null;
  gpuUtilization: number | null;

  // Training cost
  trainingCostUsd: number;
  trainingTimeHours: number;
  energyKwh: number;
  co2Kg: number;
  costPerMillionTokensUsd: number;
  provider?: string;

  // Runtime context
  selectedPrecision?: string;
  selectedBatchSize?: number;

  // Meta
  confidenceScore: number;
  depth: number;
  isSequenceModel?: boolean;
  customLayerCount?: number;
  diagnosticCount?: number;
  reportWarnings?: string[];
  recommendations?: AnalysisRecommendation[];

  // New fields for real-time simulation
  compilation?: {
    current_phase: string;
    total_progress: number;
    phase_timeline: Array<{
      name: string;
      duration_ms: number;
      status: string;
    }>;
  };
  live_trace?: {
    partial_metrics: [number, number][];
    throughput_trace: [number, number][];
    memory_liveness?: Array<{
      step: number;
      value: number;
    }>;
    memory_heatmap?: Array<{
      layer: string;
      timeline: number[];
    }>;
    gradient_memory_breakdown?: Array<{
      name: string;
      forward: number;
      backward: number;
    }>;
    kv_cache_scaling?: Array<{
      seq: number;
      value: number;
    }>;
  };
  memory_liveness?: Array<{
    step: number;
    value: number;
  }>;
  memory_heatmap?: Array<{
    layer: string;
    timeline: number[]; // 0 or 1 for active status at each normalized step
  }>;
  gradient_memory_breakdown?: Array<{
    name: string;
    forward: number;
    backward: number;
  }>;
  kv_cache_scaling?: Array<{
    seq: number;
    value: number;
  }>;
  // ─── Per Layer optional maps (keyed by layer name/id) ──────────────────────
  perLayerLatency?: Record<string, number>; // ms per layer from compiler
  perLayerVram?: Record<string, number>;    // bytes per layer from compiler

  diagnostics?: Array<{
    category: string;
    severity: string;
    code?: string;
    message: string;
    layer_id?: string;
    suggestion?: string;
    precision_impact?: number;
  }>;

  // ─── Dynamic Metrics (M36-M55) ──────────────────────────────────
  dynamic?: {
    virtual_memory?: {
      fragmentation_overhead_gb: number;
      fragmentation_pct: number;
      defrag_savings_gb: number;
      virtual_savings_gb: number;
      virtual_savings_pct: number;
      peak_vram_with_defrag_gb: number;
      peak_vram_with_virtual_gb: number;
      recommended_strategy: string;
      confidence: number;
    };
    stability?: {
      lyapunov_exponent_mean: number;
      chaos_index: number;
      high_risk_layers_count: number;
      fp32_required_pct: number;
      global_robustness_score: number;
      fp32_fallback_memory_overhead_gb: number;
      confidence: number;
    };
    behavioral?: {
      expert_load_imbalance: number;
      memory_contention_score: number;
      cache_locality_score: number;
      numerical_sensitivity: number;
      load_balance_efficiency: number;
      memory_bank_conflict_rate: number;
      prediction_confidence: number;
    };
  };
}

export interface AnalysisRecommendation {
  category: string;
  priority: string;
  title: string;
  description: string;
  impact: string;
}

export interface PerLayerBreakdownRow {
  id: string;
  name: string;
  params?: number;
  flops?: string;
  memory?: string;
  latency?: string;
}

export interface Warning {
  id: string;
  type: 'error' | 'warning' | 'info';
  message: string;
  hint?: string;
  code?: string;
  nodeId?: string;
}

export const ACTIVATION_OPTIONS: { value: ActivationFunction; label: string }[] = [
  { value: 'none', label: 'None' },
  { value: 'relu', label: 'ReLU' },
  { value: 'gelu', label: 'GELU' },
  { value: 'gelu_tanh', label: 'GELU (tanh approx)' },
  { value: 'gelu_fast', label: 'GELU Fast' },
  { value: 'gelu_new', label: 'GELU New' },
  { value: 'silu', label: 'SiLU / Swish β=1' },
  { value: 'swish', label: 'Swish (β param)' },
  { value: 'tanh', label: 'Tanh' },
  { value: 'sigmoid', label: 'Sigmoid' },
  { value: 'leaky_relu', label: 'Leaky ReLU' },
  { value: 'elu', label: 'ELU' },
  { value: 'mish', label: 'Mish' },
  { value: 'softmax', label: 'Softmax' },
  { value: 'swiglu', label: 'SwiGLU' },
  { value: 'geglu', label: 'GeGLU' },
  { value: 'reglu', label: 'ReGLU' },
  { value: 'bilinear', label: 'Bilinear' },
  { value: 'solu', label: 'SoLU' },
  { value: 'solu_ln', label: 'SoLU + LayerNorm' },
];
