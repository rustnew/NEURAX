import { MacroBlock, ArchitectureConstraint } from '@/types/catalog.ts';
import { ArchitectureFamily } from '@/types/plugins.ts';

// ─── MACRO BLOCKS ──────────────────────────────────────────────────────────
//
// All internalBlocks strings should match a valid block type from the UI registry.

export const MACRO_BLOCKS: MacroBlock[] = [

  // ── Transformer ────────────────────────────────────────────────────────
  {
    id: 'macro-transformer-block',
    name: 'Standard Transformer Block',
    family: 'transformer',
    description: 'LayerNorm → MHA → Residual → LayerNorm → FFN → Residual',
    icon: 'Box',
    tags: ['attention', 'norm', 'ffn'],
    internalBlocks: ['layernorm', 'mha_attention', 'residual_add', 'layernorm', 'ffn_standard', 'residual_add'],
    color: 'hsl(199, 89%, 48%)',
  },
  {
    id: 'macro-decoder-block',
    name: 'Decoder-Only Block',
    family: 'transformer',
    description: 'RMSNorm → Causal MHA → Residual → RMSNorm → SwiGLU → Residual',
    icon: 'Box',
    tags: ['decoder', 'causal', 'swiglu'],
    internalBlocks: ['rmsnorm', 'mha_attention', 'residual_add', 'rmsnorm', 'ffn_gated', 'residual_add'],
    color: 'hsl(199, 89%, 48%)',
  },
  {
    id: 'macro-cross-attn',
    name: 'Cross-Attention Block',
    family: 'transformer',
    description: 'LayerNorm → Cross-MHA → Residual (encoder-decoder)',
    icon: 'GitBranch',
    tags: ['cross-attention', 'encoder-decoder'],
    internalBlocks: ['layernorm', 'cross_attention', 'residual_add'],
    color: 'hsl(199, 89%, 48%)',
  },
  {
    id: 'macro-rope-block',
    name: 'RoPE Attention Block',
    family: 'transformer',
    description: 'RoPE → GQA (causal) → Residual → RMSNorm → SwiGLU → Residual',
    icon: 'Rotate3D',
    tags: ['rope', 'gqa', 'llama-style'],
    internalBlocks: ['pos_rope', 'gqa_attention', 'residual_add', 'rmsnorm', 'ffn_gated', 'residual_add'],
    color: 'hsl(199, 89%, 48%)',
  },

  // ── MoE ───────────────────────────────────────────────────────────────
  {
    id: 'macro-moe-block',
    name: 'MoE Transformer Block',
    family: 'moe',
    description: 'LayerNorm → MHA → Residual → LayerNorm → Gate → Expert FFN → Combine',
    icon: 'Network',
    tags: ['routing', 'sparse', 'attention'],
    internalBlocks: ['layernorm', 'mha_attention', 'residual_add', 'moe_block', 'residual_add'],
    color: 'hsl(280, 70%, 55%)',
  },
  {
    id: 'macro-moe-ffn',
    name: 'MoE FFN Layer',
    family: 'moe',
    description: 'Gate (Router) → Top-K Dispatch → Expert FFN → Combine',
    icon: 'Route',
    tags: ['routing', 'top-k'],
    internalBlocks: ['moe_block'],
    color: 'hsl(280, 70%, 55%)',
  },
  {
    id: 'macro-moe-full-layer',
    name: 'Full MoE Layer',
    family: 'moe',
    description: 'RMSNorm → GQA → Add → RMSNorm → Gate → Experts → Combine → Add',
    icon: 'Layers',
    tags: ['full-layer', 'mixtral-style'],
    internalBlocks: ['rmsnorm', 'gqa_attention', 'residual_add', 'moe_block', 'residual_add'],
    color: 'hsl(280, 70%, 55%)',
  },

  // ── CNN ───────────────────────────────────────────────────────────────
  {
    id: 'macro-resnet-block',
    name: 'ResNet Block',
    family: 'cnn',
    description: 'Conv2D → BN → ReLU → Conv2D → BN → (+Skip) → ReLU',
    icon: 'GitBranch',
    tags: ['residual', 'skip-connection'],
    internalBlocks: ['basic_block'],
    color: 'hsl(142, 71%, 45%)',
  },
  {
    id: 'macro-bottleneck',
    name: 'Bottleneck Block',
    family: 'cnn',
    description: '1×1 Conv → 3×3 Conv → 1×1 Conv with residual',
    icon: 'Layers',
    tags: ['bottleneck', 'resnet50'],
    internalBlocks: ['bottleneck_block'],
    color: 'hsl(142, 71%, 45%)',
  },
  {
    id: 'macro-mbconv',
    name: 'MBConv Block',
    family: 'cnn',
    description: 'Expand Conv → Depthwise Conv2D → SE Block → Project Conv',
    icon: 'Zap',
    tags: ['mobile', 'efficient', 'depthwise'],
    internalBlocks: ['mbconv_block'],
    color: 'hsl(142, 71%, 45%)',
  },
  {
    id: 'macro-dense-block',
    name: 'DenseNet Block',
    family: 'cnn',
    description: 'Concat(all prev) → BN → ReLU → Conv2D (growth rate)',
    icon: 'Share2',
    tags: ['densenet', 'feature-reuse', 'concat'],
    internalBlocks: ['dense_block'],
    color: 'hsl(142, 71%, 45%)',
  },

  // ── GAN ───────────────────────────────────────────────────────────────
  {
    id: 'macro-gen-block',
    name: 'Generator Block',
    family: 'gan',
    description: 'ConvTranspose2D → BatchNorm → ReLU (upsample block)',
    icon: 'Wand2',
    tags: ['generator', 'upsample'],
    internalBlocks: ['dcgan_generator_block'],
    color: 'hsl(38, 92%, 50%)',
  },
  {
    id: 'macro-disc-block',
    name: 'Discriminator Block',
    family: 'gan',
    description: 'Conv2D → SpectralNorm → InstanceNorm → LeakyReLU (downsample)',
    icon: 'Shield',
    tags: ['discriminator', 'downsample'],
    internalBlocks: ['dcgan_discriminator_block'],
    color: 'hsl(38, 92%, 50%)',
  },
  {
    id: 'macro-style-gan-block',
    name: 'StyleGAN Synthesis Stage',
    family: 'gan',
    description: 'ModulatedConv2D → Noise Injection → Bias → Activation',
    icon: 'Sparkles',
    tags: ['stylegan', 'modulated', 'style-mixing'],
    internalBlocks: ['stylegan_synthesis_block'],
    color: 'hsl(45, 100%, 50%)',
  },
  {
    id: 'macro-patchgan-disc',
    name: 'PatchGAN Discriminator',
    family: 'gan',
    description: 'Conv2D ×N → LeakyReLU → InstanceNorm → Conv2D (patch score)',
    icon: 'Grid',
    tags: ['patchgan', 'conditional', 'pix2pix'],
    internalBlocks: ['conv2d', 'spectral_norm', 'instancenorm'],
    color: 'hsl(38, 92%, 50%)',
  },

  // ── SSM ───────────────────────────────────────────────────────────────
  {
    id: 'macro-mamba-block',
    name: 'Mamba Block',
    family: 'ssm',
    description: 'Linear Proj → Conv1D → SSM Core (selective scan) → SiLU Gate → Output Proj',
    icon: 'Workflow',
    tags: ['selective-scan', 'mamba', 'ssm'],
    internalBlocks: ['linear_projection', 'conv1d', 's6_block'],
    color: 'hsl(160, 70%, 45%)',
  },
  {
    id: 'macro-s4-block',
    name: 'S4 Block',
    family: 'ssm',
    description: 'SSM Core (HiPPO) → GLU Gate → Residual',
    icon: 'Workflow',
    tags: ['s4', 'hippo', 'linear'],
    internalBlocks: ['s4_block', 'residual_add'],
    color: 'hsl(160, 70%, 45%)',
  },
  {
    id: 'macro-mamba-layer',
    name: 'Full Mamba Layer',
    family: 'ssm',
    description: 'RMSNorm → Mamba Block → Residual Add',
    icon: 'Layers',
    tags: ['mamba', 'full-layer', 'residual'],
    internalBlocks: ['rmsnorm', 's6_block', 'residual_add'],
    color: 'hsl(160, 70%, 45%)',
  },

  // ── GNN ───────────────────────────────────────────────────────────────
  {
    id: 'macro-gcn-res',
    name: 'ResNet GCN Block',
    family: 'gnn',
    description: 'GCN Conv → GraphNorm → ReLU → Dropout → Residual',
    icon: 'GitMerge',
    tags: ['residual', 'gcn', 'graphnorm'],
    internalBlocks: ['gcn_conv', 'graph_norm', 'dropout', 'residual_add'],
    color: 'hsl(340, 75%, 55%)',
  },
  {
    id: 'macro-mpnn-block',
    name: 'MPNN Block',
    family: 'gnn',
    description: 'Edge Conv (Message) → Global Aggregation → Dense (Update)',
    icon: 'MessageSquare',
    tags: ['message-passing', 'mpnn'],
    internalBlocks: ['edge_conv', 'global_add_pool', 'linear_projection'],
    color: 'hsl(340, 75%, 55%)',
  },
  {
    id: 'macro-gat-block',
    name: 'GAT Block',
    family: 'gnn',
    description: 'GAT Conv (multi-head, concat) → BatchNorm → ReLU → Dropout',
    icon: 'Eye',
    tags: ['attention', 'gat', 'multi-head'],
    internalBlocks: ['gat_conv', 'batchnorm', 'dropout'],
    color: 'hsl(340, 75%, 55%)',
  },
  {
    id: 'macro-graph-readout',
    name: 'Graph Readout',
    family: 'gnn',
    description: 'Global Mean Pool → Dense → ReLU → Classification Head',
    icon: 'Crosshair',
    tags: ['readout', 'graph-classification'],
    internalBlocks: ['global_mean_pool', 'linear_projection', 'classification_head'],
    color: 'hsl(340, 75%, 55%)',
  },

  // ── RNN ───────────────────────────────────────────────────────────────
  {
    id: 'macro-lstm-cell',
    name: 'LSTM Cell',
    family: 'rnn',
    description: 'Forget → Input → Cell → Output gates',
    icon: 'Repeat',
    tags: ['lstm', 'gated', 'memory'],
    internalBlocks: ['lstm'],
    color: 'hsl(25, 80%, 52%)',
  },
  {
    id: 'macro-gru-cell',
    name: 'GRU Cell',
    family: 'rnn',
    description: 'Reset → Update → Candidate gates',
    icon: 'Repeat',
    tags: ['gru', 'gated', 'efficient'],
    internalBlocks: ['gru'],
    color: 'hsl(25, 80%, 52%)',
  },
  {
    id: 'macro-bidirectional',
    name: 'Bidirectional Layer',
    family: 'rnn',
    description: 'Forward + backward LSTM/GRU pass with merge',
    icon: 'ArrowLeftRight',
    tags: ['bidirectional', 'bilstm'],
    internalBlocks: ['bilstm'],
    color: 'hsl(25, 80%, 52%)',
  },
  {
    id: 'macro-seq2seq-attn',
    name: 'Seq2Seq + Attention',
    family: 'rnn',
    description: 'LSTM Encoder → Bahdanau Attention → GRU Decoder',
    icon: 'Route',
    tags: ['seq2seq', 'bahdanau', 'encoder-decoder'],
    internalBlocks: ['lstm', 'seq2seq_attention', 'gru'],
    color: 'hsl(25, 80%, 52%)',
  },
  {
    id: 'macro-lstmp',
    name: 'LSTM Projection (LSTMP)',
    family: 'rnn',
    description: 'LSTM with linear output projection (reduces hidden dim)',
    icon: 'Minimize2',
    tags: ['lstmp', 'projection'],
    internalBlocks: ['lstm', 'linear_projection'],
    color: 'hsl(25, 80%, 52%)',
  },
  {
    id: 'macro-stacked-rnn',
    name: 'Stacked RNN',
    family: 'rnn',
    description: 'Multi-layer recurrent stack with dropout between layers',
    icon: 'Layers',
    tags: ['stacked', 'deep-rnn'],
    internalBlocks: ['lstm', 'dropout', 'gru'],
    color: 'hsl(25, 80%, 52%)',
  },
  {
    id: 'macro-srupp',
    name: 'SRU++',
    family: 'rnn',
    description: 'SRU with internal self-attention for long-range context',
    icon: 'Zap',
    tags: ['sru', 'parallel', 'attention'],
    internalBlocks: ['mha_attention', 'linear_projection'],
    color: 'hsl(25, 80%, 52%)',
  },
  {
    id: 'macro-qrnn',
    name: 'QRNN',
    family: 'rnn',
    description: 'Temporal Conv1D followed by minimal recurrence gate',
    icon: 'Activity',
    tags: ['qrnn', 'temporal-conv', 'parallel'],
    internalBlocks: ['conv1d', 'linear_projection'],
    color: 'hsl(25, 80%, 52%)',
  },
  {
    id: 'macro-elman-rnn',
    name: 'Elman RNN Cell',
    family: 'rnn',
    description: 'h_t = tanh(Wx·x_t + Wh·h_{t-1} + b)',
    icon: 'Repeat',
    tags: ['vanilla-rnn', 'elman'],
    internalBlocks: ['rnn_cell'],
    color: 'hsl(25, 80%, 52%)',
  },
  {
    id: 'macro-zoneout',
    name: 'Zoneout Wrapper',
    family: 'rnn',
    description: 'Stochastic identity regularisation on hidden state',
    icon: 'Shield',
    tags: ['regularisation', 'zoneout'],
    internalBlocks: ['rnn_cell'],
    color: 'hsl(25, 80%, 52%)',
  },
  {
    id: 'macro-lmu',
    name: 'Legendre Memory Unit',
    family: 'rnn',
    description: 'Neurally plausible memory via Legendre polynomials',
    icon: 'Database',
    tags: ['lmu', 'memory', 'neuromorphic'],
    internalBlocks: ['rnn_cell'],
    color: 'hsl(25, 80%, 52%)',
  },
  {
    id: 'macro-mgu',
    name: 'Minimal Gated Unit',
    family: 'rnn',
    description: 'GRU-like cell with single combined forget/update gate',
    icon: 'Minimize',
    tags: ['mgu', 'minimal', 'gated'],
    internalBlocks: ['rnn_cell'],
    color: 'hsl(25, 80%, 52%)',
  },
  {
    id: 'macro-siamese-rnn',
    name: 'Siamese RNN',
    family: 'rnn',
    description: 'Twin LSTM towers with shared weights for similarity',
    icon: 'Users',
    tags: ['siamese', 'similarity', 'shared-weights'],
    internalBlocks: ['lstm'],
    color: 'hsl(25, 80%, 52%)',
  },

  // ── Diffusion ──────────────────────────────────────────────────────────
  {
    id: 'macro-unet-down',
    name: 'U-Net Downsample Block',
    family: 'diffusion',
    description: 'ResidualBlock → GroupNorm → SiLU → Downsample',
    icon: 'ArrowDown',
    tags: ['unet', 'encoder', 'downsample'],
    internalBlocks: ['residual_block', 'groupnorm', 'downsample_2d'],
    color: 'hsl(262, 60%, 55%)',
  },
  {
    id: 'macro-unet-up',
    name: 'U-Net Upsample Block',
    family: 'diffusion',
    description: 'Upsample → Concat (skip) → ResidualBlock → GroupNorm → SiLU',
    icon: 'ArrowUp',
    tags: ['unet', 'decoder', 'upsample', 'skip'],
    internalBlocks: ['upsample', 'concat', 'residual_block', 'groupnorm'],
    color: 'hsl(262, 60%, 55%)',
  },
  {
    id: 'macro-unet-mid',
    name: 'U-Net Bottleneck Block',
    family: 'diffusion',
    description: 'ResidualBlock → MHA (self-attn) → ResidualBlock',
    icon: 'Minimize2',
    tags: ['unet', 'bottleneck', 'attention'],
    internalBlocks: ['residual_block', 'spatial_transformer', 'residual_block'],
    color: 'hsl(262, 60%, 55%)',
  },
  {
    id: 'macro-timestep-cond',
    name: 'Timestep Conditioning',
    family: 'diffusion',
    description: 'Timestep Embedding → Dense → SiLU → Scale+Shift injection',
    icon: 'Clock',
    tags: ['timestep', 'conditioning', 'adagn'],
    internalBlocks: ['timestep_embedding', 'linear_projection'],
    color: 'hsl(262, 60%, 55%)',
  },

  // ── SNN ────────────────────────────────────────────────────────────────
  {
    id: 'macro-lif-layer',
    name: 'LIF Layer',
    family: 'snn',
    description: 'Dense → BatchNorm → LIF Neuron (with leak and reset)',
    icon: 'Zap',
    tags: ['lif', 'spiking', 'neuromorphic'],
    internalBlocks: ['linear_projection', 'batchnorm', 'lif_neuron'],
    color: 'hsl(54, 80%, 45%)',
  },
  {
    id: 'macro-snn-encoder',
    name: 'SNN Encoder',
    family: 'snn',
    description: 'Rate Encoder → Dense → LIF Neuron → Dropout',
    icon: 'Radio',
    tags: ['encoder', 'rate-coding', 'spike'],
    internalBlocks: ['rate_encoder', 'linear_projection', 'lif_neuron', 'dropout'],
    color: 'hsl(54, 80%, 45%)',
  },
  {
    id: 'macro-synaptic-layer',
    name: 'Synaptic Layer',
    family: 'snn',
    description: 'Synaptic Layer → LIF Neuron → Leaky Neuron (membrane)',
    icon: 'GitBranch',
    tags: ['synaptic', 'membrane', 'bio-inspired'],
    internalBlocks: ['synaptic_layer', 'lif_neuron', 'leaky_neuron'],
    color: 'hsl(54, 80%, 45%)',
  },

  // ── Experimental ──────────────────────────────────────────────────────
  {
    id: 'macro-custom-block',
    name: 'Custom Block',
    family: 'experimental',
    description: 'User-defined operation with custom FLOPs and memory equations',
    icon: 'Code2',
    tags: ['custom', 'research', 'experimental'],
    internalBlocks: ['linear_projection'],
    color: 'hsl(0, 0%, 55%)',
  },
  {
    id: 'macro-hybrid-attn-ssm',
    name: 'Hybrid Attention + SSM',
    family: 'experimental',
    description: 'MHA branch + Mamba branch → Add → RMSNorm (Jamba-style)',
    icon: 'Shuffle',
    tags: ['hybrid', 'jamba', 'research'],
    internalBlocks: ['mha_attention', 's6_block', 'residual_add', 'rmsnorm'],
    color: 'hsl(0, 0%, 55%)',
  },

  // ── RL (forward-compatible — no catalogue blocks yet) ──────────────────
  {
    id: 'macro-actor-critic',
    name: 'Actor-Critic Head',
    family: 'rl',
    description: 'Shared backbone → Policy Head + Value Head (parallel)',
    icon: 'GitBranch',
    tags: ['a2c', 'actor-critic', 'policy'],
    internalBlocks: ['linear_projection', 'policy_head', 'value_head'],
    color: 'hsl(15, 80%, 52%)',
  },
  {
    id: 'macro-dueling-dqn',
    name: 'Dueling DQN Streams',
    family: 'rl',
    description: 'Value Stream + Advantage Stream → Q-Values (combined)',
    icon: 'Columns',
    tags: ['dqn', 'dueling', 'value-advantage'],
    internalBlocks: ['linear_projection', 'value_head', 'policy_head'],
    color: 'hsl(15, 80%, 52%)',
  },
];

// ─── ARCHITECTURE CONSTRAINTS ──────────────────────────────────────────────

export const ARCHITECTURE_CONSTRAINTS: ArchitectureConstraint[] = [
  {
    family: 'transformer',
    requiredBlocks: ['input', 'output', 'mha_attention'],
    incompatibleBlocks: [
      'gcn_conv', 'gat_conv', 'sage_conv',
      's6_block', 's4_block',
      'lstm', 'gru', 'bilstm',
    ],
    warningRules: [
      {
        id: 'no-norm',
        condition: 'missing_block',
        blockType: 'layernorm',
        message: 'Transformer blocks should include LayerNorm or RMSNorm for training stability',
        severity: 'warning',
      },
      {
        id: 'no-attn',
        condition: 'missing_block',
        blockType: 'mha_attention',
        message: 'No attention block found — required for Transformer architectures',
        severity: 'error',
      },
    ],
  },

  {
    family: 'cnn',
    requiredBlocks: ['input', 'output', 'conv2d'],
    incompatibleBlocks: [
      'mha_attention', 'gqa_attention',
      'moe_block', 'router_linear',
      's6_block', 's4_block',
      'gcn_conv', 'gat_conv', 'sage_conv',
      'lstm', 'gru', 'bilstm',
    ],
    warningRules: [
      {
        id: 'no-conv',
        condition: 'missing_block',
        blockType: 'conv2d',
        message: 'No convolutional layer found — required for CNN architectures',
        severity: 'error',
      },
      {
        id: 'no-bn',
        condition: 'missing_block',
        blockType: 'batchnorm',
        message: 'Consider adding BatchNorm for faster convergence',
        severity: 'info',
      },
    ],
  },

  {
    family: 'moe',
    requiredBlocks: ['input', 'output', 'moe_block'],
    incompatibleBlocks: [
      'gcn_conv', 'gat_conv', 'sage_conv',
      's6_block', 's4_block',
      'lstm', 'gru', 'bilstm',
    ],
    warningRules: [
      {
        id: 'no-moe',
        condition: 'missing_block',
        blockType: 'moe_block',
        message: 'MoE requires a moe_block or moe_layer for expert routing',
        severity: 'error',
      },
      {
        id: 'no-combine',
        condition: 'missing_block',
        blockType: 'expert_combine',
        message: 'MoE should include an expert_combine block to merge expert outputs',
        severity: 'warning',
      },
    ],
  },

  {
    family: 'ssm',
    requiredBlocks: ['input', 'output'],
    incompatibleBlocks: [
      'gcn_conv', 'gat_conv', 'sage_conv',
      'conv2d',
      'moe_block', 'router_linear',
      'lstm', 'gru', 'bilstm',
    ],
    warningRules: [
      {
        id: 'no-ssm',
        condition: 'missing_block',
        blockType: 's6_block',
        message: 'No SSM block found — add s6_block (Mamba) or s4_block',
        severity: 'warning',
      },
    ],
  },

  {
    family: 'gan',
    requiredBlocks: ['input', 'output'],
    incompatibleBlocks: [
      'gcn_conv', 'gat_conv', 'sage_conv',
      's6_block', 's4_block',
      'lstm', 'gru', 'bilstm',
    ],
    warningRules: [
      {
        id: 'no-gen',
        condition: 'missing_block',
        blockType: 'dcgan_generator_block',
        message: 'GAN generator typically requires upsampling (e.g., dcgan_generator_block)',
        severity: 'info',
      },
    ],
  },

  {
    family: 'diffusion',
    requiredBlocks: ['input', 'output', 'noise_scheduler'],
    incompatibleBlocks: [
      'gcn_conv', 'gat_conv', 'sage_conv',
      'lstm', 'gru', 'bilstm',
      's6_block', 's4_block',
    ],
    warningRules: [
      {
        id: 'no-noise',
        condition: 'missing_block',
        blockType: 'noise_scheduler',
        message: 'Diffusion models require a noise_scheduler block (beta schedule config)',
        severity: 'error',
      },
      {
        id: 'no-timestep',
        condition: 'missing_block',
        blockType: 'timestep_embedding',
        message: 'Diffusion U-Net should include timestep conditioning',
        severity: 'warning',
      },
    ],
  },

  {
    family: 'gnn',
    requiredBlocks: ['input', 'output'],
    incompatibleBlocks: [
      'conv2d',
      'moe_block', 'router_linear',
      's6_block', 's4_block',
      'lstm', 'gru', 'bilstm',
    ],
    warningRules: [
      {
        id: 'no-graph-conv',
        condition: 'missing_block',
        blockType: 'gcn_conv',
        message: 'GNN requires at least one graph convolution layer (gcn_conv, gat_conv, or sage_conv)',
        severity: 'warning',
      },
    ],
  },

  {
    family: 'rnn',
    requiredBlocks: ['input', 'output'],
    incompatibleBlocks: [
      'conv2d',
      'mha_attention', 'gqa_attention',
      'moe_block', 'router_linear',
      's6_block', 's4_block',
      'gcn_conv', 'gat_conv', 'sage_conv',
    ],
    warningRules: [
      {
        id: 'vanishing',
        condition: 'order_violation',
        message: 'Deep vanilla RNNs may suffer from vanishing gradients — consider lstm or gru',
        severity: 'warning',
      },
    ],
  },

  {
    family: 'snn',
    requiredBlocks: ['input', 'output'],
    incompatibleBlocks: [
      'moe_block', 'router_linear',
      'gcn_conv', 'gat_conv', 'sage_conv',
      's6_block', 's4_block',
      'conv2d',
    ],
    warningRules: [
      {
        id: 'no-lif',
        condition: 'missing_block',
        blockType: 'lif_neuron',
        message: 'SNN requires at least one spiking neuron layer (lif_neuron)',
        severity: 'error',
      },
      {
        id: 'no-encoder',
        condition: 'missing_block',
        blockType: 'rate_encoder',
        message: 'SNN should include a spike encoder to convert real-valued input to spikes',
        severity: 'warning',
      },
    ],
  },

  {
    // RL family uses policy/value heads for now
    family: 'rl',
    requiredBlocks: ['input', 'output'],
    incompatibleBlocks: [
      'gcn_conv', 'gat_conv', 'sage_conv',
      's6_block', 's4_block',
      'noise_scheduler',
    ],
    warningRules: [
      {
        id: 'no-policy',
        condition: 'missing_block',
        blockType: 'policy_head',
        message: 'RL models need a policy_head or value_head for action/value outputs',
        severity: 'warning',
      },
    ],
  },

  {
    family: 'experimental',
    requiredBlocks: ['input', 'output'],
    incompatibleBlocks: [],
    warningRules: [],
  },
];

// ─── UTILITY FUNCTIONS ──────────────────────────────────────────────────────

export function getMacroBlocksForFamily(family: ArchitectureFamily): MacroBlock[] {
  return MACRO_BLOCKS.filter(m => m.family === family);
}

export function getConstraintsForFamily(family: ArchitectureFamily): ArchitectureConstraint | undefined {
  return ARCHITECTURE_CONSTRAINTS.find(c => c.family === family);
}
