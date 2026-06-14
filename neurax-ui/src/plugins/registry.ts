import { ArchitectureFamily, ArchitecturePlugin, PluginTool, PluginMetric } from '@/types/plugins.ts';
import { LayerConfig } from '@/types/architecture.ts';

// ─── SHARED BLOCKS ────────────────────────────────────────

const IO_BLOCKS: LayerConfig[] = [
  { id: 'input', type: 'input', name: 'Input', icon: 'ArrowRightToLine', description: 'Data entry point', defaultParams: {}, category: 'I/O' },
  { id: 'output', type: 'output', name: 'Output', icon: 'ArrowRightFromLine', description: 'Model output', defaultParams: {}, category: 'I/O' },
];

const NORM_BLOCKS: Record<string, LayerConfig> = {
  layernorm: { id: 'layernorm', type: 'layernorm', name: 'LayerNorm', icon: 'AlignCenter', description: 'Layer normalization', defaultParams: { normalized_shape: 768, eps: 1e-6, elementwise_affine: true }, category: 'Normalization', tooltip: 'Normalizes across feature dimension' },
  rmsnorm: { id: 'rmsnorm', type: 'rmsnorm', name: 'RMSNorm', icon: 'AlignCenter', description: 'Root mean square normalization', defaultParams: { normalized_shape: 768, eps: 1e-5, elementwise_affine: true }, category: 'Normalization', tooltip: 'Faster than LayerNorm, used in LLaMA' },
  batchnorm: { id: 'batchnorm', type: 'batchnorm', name: 'BatchNorm', icon: 'AlignJustify', description: 'Batch normalization', defaultParams: { num_features: 64, eps: 1e-5, momentum: 0.1, affine: true }, category: 'Normalization', tooltip: 'Normalizes across batch dimension' },
  groupnorm: { id: 'groupnorm', type: 'groupnorm', name: 'GroupNorm', icon: 'AlignJustify', description: 'Group normalization', defaultParams: { num_groups: 32, num_channels: 256, eps: 1e-5, affine: true }, category: 'Normalization', tooltip: 'Normalizes within channel groups' },
  instancenorm: { id: 'instancenorm', type: 'instancenorm', name: 'InstanceNorm', icon: 'AlignJustify', description: 'Instance normalization', defaultParams: { num_features: 64, eps: 1e-5, affine: false }, category: 'Normalization', tooltip: 'Per-instance normalization, used in style transfer' },
};

const STRUCT_BLOCKS: Record<string, LayerConfig> = {
  residual_add: { id: 'residual_add', type: 'residual_add', name: 'Residual Add', icon: 'GitBranch', description: 'Skip connection addition', defaultParams: {}, category: 'Structure', tooltip: 'Adds skip connection input to current tensor' },
  concat: { id: 'concat', type: 'concat', name: 'Concat', icon: 'GitBranch', description: 'Tensor concatenation', defaultParams: { dim: -1 }, category: 'Structure' },
  dropout: { id: 'dropout', type: 'dropout', name: 'Dropout', icon: 'Zap', description: 'Regularization dropout', defaultParams: { rate: 0.1 }, category: 'Structure' },
  flatten: { id: 'flatten', type: 'flatten', name: 'Flatten', icon: 'Layers', description: 'Flatten spatial dims', defaultParams: { start_dim: 1 }, category: 'Structure' },
  reshape: { id: 'reshape', type: 'reshape', name: 'Reshape', icon: 'Layers', description: 'Reshape tensor dimensions', defaultParams: { shape: 'auto' }, category: 'Structure' },
};

// ─── TRANSFORMER BLOCKS ───────────────────────────────────

const transformerBlocks: LayerConfig[] = [
  ...IO_BLOCKS,

  // ── Input / Embedding ──
  { id: 'token_embedding', type: 'token_embedding', name: 'Token Embedding', icon: 'Layers', description: 'Token → dense vector projection', defaultParams: { vocab_size: 32000, d_model: 768, padding_idx: 0, dtype: 'float32' }, category: 'Input', tooltip: 'Maps discrete token IDs to continuous vectors' },
  { id: 'embedding', type: 'embedding', name: 'Embedding', icon: 'Layers', description: 'Token + positional embedding', defaultParams: { vocab_size: 32000, d_model: 768, padding_idx: 0, dtype: 'float32' }, category: 'Input', hasActivation: false, tooltip: 'Combined token and positional embedding' },

  // ── Projection ──
  { id: 'linear_projection', type: 'linear_projection', name: 'Linear Projection', icon: 'Layers', description: 'Dense linear Wx + b', defaultParams: { in_features: 768, out_features: 768, bias: true, dtype: 'float32', activation: 'none' }, category: 'Projection', hasActivation: true, tooltip: 'Standard dense linear transform' },
  { id: 'lora_linear', type: 'lora_linear', name: 'LoRA Linear', icon: 'Layers', description: 'Low-rank adapted linear', defaultParams: { in_features: 768, out_features: 768, rank: 16, alpha: 32, bias: true, activation: 'none' }, category: 'Projection', hasActivation: true, tooltip: 'Low-Rank Adaptation for efficient fine-tuning' },
  { id: 'dora_linear', type: 'dora_linear', name: 'DoRA Linear', icon: 'Layers', description: 'Weight-decomposed low-rank', defaultParams: { in_features: 768, out_features: 768, rank: 16, alpha: 32, bias: true, activation: 'none' }, category: 'Projection', hasActivation: true, tooltip: 'Weight-Decomposed Low-Rank Adaptation' },
  { id: 'q_projection', type: 'q_projection', name: 'Q Projection', icon: 'Focus', description: 'Query projection', defaultParams: { d_model: 768, n_heads: 12, head_dim: 64, bias: false }, category: 'Projection', tooltip: 'Projects input to query space' },
  { id: 'k_projection', type: 'k_projection', name: 'K Projection', icon: 'Focus', description: 'Key projection', defaultParams: { d_model: 768, n_heads: 12, head_dim: 64, bias: false }, category: 'Projection', tooltip: 'Projects input to key space' },
  { id: 'v_projection', type: 'v_projection', name: 'V Projection', icon: 'Focus', description: 'Value projection', defaultParams: { d_model: 768, n_heads: 12, head_dim: 64, bias: false }, category: 'Projection', tooltip: 'Projects input to value space' },
  { id: 'qkv_combined', type: 'qkv_combined', name: 'QKV Combined', icon: 'Focus', description: 'Single fused Q/K/V projection', defaultParams: { d_model: 768, n_heads: 12, head_dim: 64, qkv_bias: false }, category: 'Projection', tooltip: 'One matrix for all three projections (faster)' },
  { id: 'mqa_projection', type: 'mqa_projection', name: 'MQA Projection', icon: 'Focus', description: 'Multi-Query (shared K/V)', defaultParams: { d_model: 768, n_heads: 12, n_kv_heads: 1, head_dim: 64, bias: false }, category: 'Projection', tooltip: 'K and V shared across all heads' },
  { id: 'gqa_projection', type: 'gqa_projection', name: 'GQA Projection', icon: 'Focus', description: 'Grouped-Query (grouped K/V)', defaultParams: { d_model: 768, n_heads: 12, n_kv_heads: 4, head_dim: 64, bias: false }, category: 'Projection', tooltip: 'K and V shared within groups of heads' },

  // ── Attention ──
  { id: 'attention_score', type: 'attention_score', name: 'Scaled Dot-Product', icon: 'Focus', description: 'softmax(QK^T/√d)V core', defaultParams: { scale: 'auto', dropout: 0.0, causal: true, mask_type: 'causal' }, category: 'Attention', tooltip: 'Fundamental attention scoring kernel' },
  { id: 'mha_attention', type: 'mha_attention', name: 'Multi-Head Attention', icon: 'Focus', description: 'Standard MHA with projections', defaultParams: { n_heads: 12, head_dim: 64, d_model: 768, dropout: 0.0, bias: false, output_projection: true }, category: 'Attention', tooltip: 'Full multi-head attention with output projection' },
  { id: 'mqa_attention', type: 'mqa_attention', name: 'Multi-Query Attention', icon: 'Focus', description: 'Shared K/V heads (MQA)', defaultParams: { n_heads: 12, n_kv_heads: 1, d_model: 768, dropout: 0.0 }, category: 'Attention', tooltip: 'All heads share one K/V — reduces KV cache' },
  { id: 'gqa_attention', type: 'gqa_attention', name: 'Grouped-Query Attention', icon: 'Focus', description: 'Grouped K/V heads (GQA)', defaultParams: { n_heads: 12, n_kv_heads: 4, d_model: 768, dropout: 0.0 }, category: 'Attention', tooltip: 'K/V shared within head groups — used in LLaMA 2/3' },
  { id: 'mla_attention', type: 'mla_attention', name: 'Multi-head Latent Attn', icon: 'Focus', description: 'Latent compression (DeepSeek)', defaultParams: { n_heads: 12, d_model: 768, latent_dim: 128, compression_ratio: 4 }, category: 'Attention', tooltip: 'Attention with latent compression — DeepSeek V2/V3' },
  { id: 'sliding_window_attention', type: 'sliding_window_attention', name: 'Sliding Window Attn', icon: 'Focus', description: 'Local windowed attention', defaultParams: { n_heads: 12, d_model: 768, window_size: 4096, causal: true }, category: 'Attention', tooltip: 'Each token attends to a fixed-size local window' },
  { id: 'dilated_attention', type: 'dilated_attention', name: 'Dilated Attention', icon: 'Focus', description: 'Attention with dilation gaps', defaultParams: { n_heads: 12, d_model: 768, dilation: 2, window_size: 512 }, category: 'Attention', tooltip: 'Expands receptive field via dilated patterns' },
  { id: 'sparse_attention', type: 'sparse_attention', name: 'Sparse Attention', icon: 'Focus', description: 'Block-sparse attention pattern', defaultParams: { n_heads: 12, d_model: 768, sparsity_pattern: 'fixed', block_size: 64 }, category: 'Attention', tooltip: 'Reduces O(N²) via sparse patterns' },
  { id: 'linear_attention', type: 'linear_attention', name: 'Linear Attention', icon: 'Focus', description: 'O(N) approximation', defaultParams: { n_heads: 12, d_model: 768, kernel_type: 'elu' }, category: 'Attention', tooltip: 'Linear-time via kernel approximation' },
  { id: 'flash_attention', type: 'flash_attention', name: 'Flash Attention', icon: 'Zap', description: 'IO-aware fused attention', defaultParams: { n_heads: 12, d_model: 768, version: 3, causal: true }, category: 'Attention', tooltip: 'GPU memory-efficient tiled attention' },
  { id: 'flex_attention', type: 'flex_attention', name: 'Flex Attention', icon: 'Zap', description: 'Dynamic pattern attention', defaultParams: { n_heads: 12, d_model: 768, score_mod: 'none', mask_mod: 'causal' }, category: 'Attention', tooltip: 'Runtime-defined attention patterns' },
  { id: 'attention_aggregation', type: 'attention_aggregation', name: 'Attention Aggregation', icon: 'Focus', description: 'Weighted value aggregation', defaultParams: { n_heads: 12 }, category: 'Attention', tooltip: 'Multiplies attention weights by values' },
  { id: 'cross_attention', type: 'cross_attention', name: 'Cross Attention', icon: 'Focus', description: 'Encoder-decoder cross attention', defaultParams: { d_model: 768, n_heads: 12, head_dim: 64, dropout: 0.0, causal_decoder: false }, category: 'Attention', tooltip: 'Attends to encoder output from decoder' },

  // ── Normalization ──
  { ...NORM_BLOCKS.layernorm, defaultParams: { normalized_shape: 768, eps: 1e-6, elementwise_affine: true, position: 'pre' } },
  { ...NORM_BLOCKS.rmsnorm, defaultParams: { normalized_shape: 768, eps: 1e-5, elementwise_affine: true, position: 'pre' } },

  // ── Feed-Forward Networks ──
  { id: 'ffn_standard', type: 'ffn_standard', name: 'FFN Standard', icon: 'Layers', description: '2-layer MLP (d→4d→d)', defaultParams: { d_model: 768, d_ff: 3072, bias: true, dropout: 0.0, activation: 'gelu' }, category: 'Feed-Forward', hasActivation: true, tooltip: 'Standard narrow-wide-narrow FFN' },
  { id: 'ffn_gated', type: 'ffn_gated', name: 'FFN Gated (SwiGLU)', icon: 'Zap', description: 'Gated FFN with SwiGLU/GeGLU', defaultParams: { d_model: 768, d_ff: 2048, gate_type: 'swiglu', bias: false, activation: 'silu' }, category: 'Feed-Forward', hasActivation: true, tooltip: 'gate_proj * act(up_proj) → down_proj — LLaMA/Mistral' },
  { id: 'ffn_parallel', type: 'ffn_parallel', name: 'FFN Parallel', icon: 'Workflow', description: 'Parallel Attention + MLP', defaultParams: { d_model: 768, d_ff: 3072, parallel_attn_mlp: true }, category: 'Feed-Forward', hasActivation: true, tooltip: 'Attention and MLP run in parallel — GPT-J/NeoX' },
  { id: 'moe_block', type: 'moe_block', name: 'MoE Block', icon: 'Network', description: 'Mixture of experts FFN', defaultParams: { num_experts: 8, top_k: 2, d_model: 768, expert_d_ff: 2048, expert_type: 'gated_ffn', router_type: 'top_k', router_bias: false, capacity_factor: 1.25, aux_loss_coef: 0.01, shared_experts: 0 }, category: 'Feed-Forward', tooltip: 'Sparse expert routing — Mixtral/DeepSeek' },

  // ── Positional Encoding ──
  { id: 'pos_absolute', type: 'pos_absolute', name: 'Absolute Position', icon: 'Activity', description: 'Sinusoidal or learned', defaultParams: { d_model: 768, max_len: 2048, encoding_type: 'sinusoidal', dropout: 0.0 }, category: 'Position', tooltip: 'Classic fixed sinusoidal or learned positional encoding' },
  { id: 'pos_rope', type: 'pos_rope', name: 'RoPE', icon: 'Activity', description: 'Rotary position embedding', defaultParams: { d_model: 768, max_len: 131072, base: 10000, scaling_type: 'none', partial_rotary_factor: 1.0 }, category: 'Position', tooltip: 'Rotation-based encoding — LLaMA, GPT-NeoX, PaLM' },
  { id: 'pos_alibi', type: 'pos_alibi', name: 'ALiBi', icon: 'Activity', description: 'Attention linear biases', defaultParams: { n_heads: 12, causal: true }, category: 'Position', tooltip: 'Linear bias on attention scores — no learned params' },
  { id: 'pos_relative_bias', type: 'pos_relative_bias', name: 'Relative Position Bias', icon: 'Activity', description: 'T5/Swin relative bias', defaultParams: { n_heads: 12, max_relative_position: 128, buckets: 32 }, category: 'Position', tooltip: 'Learned bucketed relative bias — T5' },
  { id: 'pos_xpos', type: 'pos_xpos', name: 'xPos', icon: 'Activity', description: 'Extended position for long ctx', defaultParams: { d_model: 768, max_len: 131072 }, category: 'Position', tooltip: 'Position encoding optimized for long context' },
  { id: 'pos_fire', type: 'pos_fire', name: 'FIRE', icon: 'Activity', description: 'Functional relative PE', defaultParams: { d_model: 768, max_len: 131072 }, category: 'Position', tooltip: 'Functional Interpolatable Relative Encoding' },

  // ── Structure ──
  { ...STRUCT_BLOCKS.residual_add, defaultParams: { dropout: 0.0, pre_norm: true } },
  { ...STRUCT_BLOCKS.dropout },
  { id: 'layer_stack', type: 'layer_stack', name: 'Layer Stack', icon: 'Layers', description: 'Repeat N identical blocks', defaultParams: { num_layers: 12, share_parameters: false }, category: 'Structure', tooltip: 'Stack N copies of a block (e.g. 32× decoder layers)' },
  { id: 'gradient_checkpoint', type: 'gradient_checkpoint', name: 'Gradient Checkpoint', icon: 'Box', description: 'Memory-compute tradeoff', defaultParams: { checkpoint_ratio: 1.0, offload_to_cpu: false }, category: 'Structure', tooltip: 'Re-computes activations during backward pass to save memory' },

  // ── Output Heads ──
  { id: 'lm_head', type: 'lm_head', name: 'LM Head', icon: 'ArrowRightFromLine', description: 'Next-token prediction head', defaultParams: { d_model: 768, vocab_size: 32000, bias: false, tie_weights: true }, category: 'Output', tooltip: 'Projects hidden states to vocabulary logits' },
  { id: 'classification_head', type: 'classification_head', name: 'Classification Head', icon: 'ArrowRightFromLine', description: 'Downstream classification', defaultParams: { d_model: 768, num_classes: 10, pooling: 'cls' }, category: 'Output', tooltip: 'CLS pooling → linear → class logits' },
  { id: 'value_head', type: 'value_head', name: 'Value Head (RL)', icon: 'Activity', description: 'RLHF value head', defaultParams: { d_model: 768, hidden_dim: 256 }, category: 'Output', tooltip: 'Scalar value prediction for PPO/RLHF' },
  { id: 'policy_head', type: 'policy_head', name: 'Policy Head (RL)', icon: 'Route', description: 'RLHF policy head', defaultParams: { d_model: 768, action_dim: 32000, distribution: 'categorical' }, category: 'Output', tooltip: 'Action distribution for reinforcement learning' },
];

// ─── MOE BLOCKS ───────────────────────────────────────────

const moeBlocks: LayerConfig[] = [
  ...IO_BLOCKS,

  // ── Input / Embedding ──
  { id: 'token_embedding', type: 'token_embedding', name: 'Token Embedding', icon: 'Layers', description: 'Token → dense vector projection', defaultParams: { vocab_size: 32000, d_model: 768, padding_idx: 0, dtype: 'float32' }, category: 'Input', tooltip: 'Maps discrete token IDs to continuous vectors' },
  { id: 'embedding', type: 'embedding', name: 'Embedding', icon: 'Layers', description: 'Token + positional embedding', defaultParams: { vocab_size: 32000, d_model: 768, padding_idx: 0, dtype: 'float32' }, category: 'Input' },

  // ── Projection ──
  { id: 'linear_projection', type: 'linear_projection', name: 'Linear Projection', icon: 'Layers', description: 'Dense linear Wx + b', defaultParams: { in_features: 768, out_features: 768, bias: true, dtype: 'float32', activation: 'none' }, category: 'Projection', hasActivation: true },
  { id: 'q_projection', type: 'q_projection', name: 'Q Projection', icon: 'Focus', description: 'Query projection', defaultParams: { d_model: 768, n_heads: 12, head_dim: 64, bias: false }, category: 'Projection' },
  { id: 'k_projection', type: 'k_projection', name: 'K Projection', icon: 'Focus', description: 'Key projection', defaultParams: { d_model: 768, n_heads: 12, head_dim: 64, bias: false }, category: 'Projection' },
  { id: 'v_projection', type: 'v_projection', name: 'V Projection', icon: 'Focus', description: 'Value projection', defaultParams: { d_model: 768, n_heads: 12, head_dim: 64, bias: false }, category: 'Projection' },
  { id: 'gqa_projection', type: 'gqa_projection', name: 'GQA Projection', icon: 'Focus', description: 'Grouped-Query K/V', defaultParams: { d_model: 768, n_heads: 12, n_kv_heads: 4, head_dim: 64, bias: false }, category: 'Projection', tooltip: 'K and V shared within groups — used in Qwen MoE' },

  // ── Routing ──
  { id: 'router_softmax', type: 'router_softmax', name: 'Router Softmax', icon: 'Route', description: 'Standard softmax gating over all experts', defaultParams: { num_experts: 8, d_model: 768, router_bias: false, dropout: 0.0 }, category: 'Routing', tooltip: 'Full softmax distribution over experts — no top-K selection' },
  { id: 'router_linear', type: 'router_linear', name: 'Top-K Router', icon: 'Route', description: 'Softmax gating with top-K selection', defaultParams: { num_experts: 8, d_model: 768, top_k: 2, router_type: 'top_k', router_bias: false, capacity_factor: 1.25, dropout: 0.0, jitter_noise: 0.01 }, category: 'Routing', tooltip: 'Selects top-K experts per token via learned scores' },
  { id: 'noisy_topk_router', type: 'noisy_topk_router', name: 'Noisy Top-K Router', icon: 'Route', description: 'Top-K with Gaussian noise for exploration', defaultParams: { num_experts: 8, d_model: 768, top_k: 2, noise_std: 0.1, noise_regularizer: 0.01, noise_type: 'gaussian', router_bias: false }, category: 'Routing', tooltip: 'Adds Gaussian noise to logits during training' },
  { id: 'expert_choice_router', type: 'expert_choice_router', name: 'Expert-Choice Router', icon: 'Route', description: 'Experts select tokens (inverse routing)', defaultParams: { num_experts: 8, d_model: 768, capacity_per_expert: 64, routing_type: 'expert_choice', balancing_regularization: 0.01 }, category: 'Routing', tooltip: 'Each expert picks which tokens to process' },
  { id: 'non_trainable_router', type: 'non_trainable_router', name: 'Non-Trainable Router', icon: 'Route', description: 'Fixed routing via hash/random/round-robin', defaultParams: { num_experts: 8, routing_strategy: 'hash', num_buckets: 8, seed: 42 }, category: 'Routing', tooltip: 'No learned parameters — deterministic routing' },
  { id: 'product_key_router', type: 'product_key_router', name: 'Product-Key Router', icon: 'Route', description: 'Decomposed keys for millions of experts', defaultParams: { num_experts: 1000000, d_model: 768, key_dim: 256, num_subkeys: 2, num_heads: 4, top_k: 32 }, category: 'Routing', tooltip: 'Scales to 1M+ experts via product-key decomposition (PEER)' },
  { id: 'hierarchical_router', type: 'hierarchical_router', name: 'Hierarchical Router', icon: 'Route', description: 'Two-level group → expert routing', defaultParams: { num_groups: 4, top_k_groups: 2, experts_per_group: 4, d_model: 768, router_per_group: false }, category: 'Routing', tooltip: 'First selects expert groups, then experts within groups' },
  { id: 'sinkhorn_router', type: 'sinkhorn_router', name: 'Sinkhorn Router', icon: 'Route', description: 'Balanced soft routing via Sinkhorn iterations', defaultParams: { num_experts: 8, d_model: 768, num_iters: 10, epsilon: 0.01, top_k: 2 }, category: 'Routing', tooltip: 'Iterative normalization for balanced expert assignment' },

  // ── Experts ──
  { id: 'expert_linear', type: 'expert_linear', name: 'Expert FFN Standard', icon: 'Users', description: '2-layer expert feed-forward network', defaultParams: { d_model: 768, d_ff: 3072, num_experts: 8, bias: true, dropout: 0.0, use_norm: false, activation: 'gelu' }, category: 'Experts', hasActivation: true, tooltip: 'Standard narrow-wide-narrow expert FFN' },
  { id: 'expert_gated_ffn', type: 'expert_gated_ffn', name: 'Expert Gated FFN', icon: 'Zap', description: 'Gated expert with SwiGLU/GeGLU (Mixtral)', defaultParams: { d_model: 768, d_ff: 14336, num_experts: 8, gate_type: 'swiglu', bias: false, dropout: 0.0, activation: 'silu' }, category: 'Experts', hasActivation: true, tooltip: 'gate_proj * act(up_proj) → down_proj per expert — Mixtral/DeepSeek' },
  { id: 'expert_multihead', type: 'expert_multihead', name: 'Expert Multi-Head (PEER)', icon: 'Users', description: 'Multi-head expert with independent sub-experts', defaultParams: { num_heads: 4, head_dim: 192, num_experts_per_head: 256, d_model: 768, key_dim: 128 }, category: 'Experts', tooltip: 'Each head selects independently — PEER architecture' },
  { id: 'expert_scalar', type: 'expert_scalar', name: 'Expert 1D (Scalar)', icon: 'Users', description: 'Scalar output expert for massive expert counts', defaultParams: { d_model: 768, output_dim: 1, num_experts: 1000000 }, category: 'Experts', tooltip: 'Tiny single-output expert for 1M+ expert architectures' },
  { id: 'expert_memory', type: 'expert_memory', name: 'Expert with Memory', icon: 'Database', description: 'Expert with external memory read/write', defaultParams: { d_model: 768, d_ff: 3072, memory_size: 1024, memory_dim: 768 }, category: 'Experts', tooltip: 'Memory-augmented expert — reads/writes external memory bank' },

  // ── Aggregation / Combination ──
  { id: 'expert_dispatch', type: 'expert_dispatch', name: 'Expert Dispatch', icon: 'Network', description: 'Dispatch tokens to selected experts', defaultParams: { capacity_factor: 1.25, drop_tokens: false }, category: 'Aggregation', tooltip: 'Permutes tokens into [E, capacity, D] for parallel expert execution' },
  { id: 'expert_combine', type: 'expert_combine', name: 'Expert Combine', icon: 'GitMerge', description: 'Reassemble expert outputs into batch', defaultParams: {}, category: 'Aggregation', tooltip: 'Inverse of dispatch — gathers expert outputs back to [B,S,D]' },
  { id: 'output_combination', type: 'output_combination', name: 'Weighted Sum', icon: 'GitMerge', description: 'Weighted sum of expert outputs by router scores', defaultParams: { combination_type: 'weighted_sum' }, category: 'Aggregation', tooltip: 'Standard combination: Σ weight_i × expert_i(x)' },
  { id: 'concat_projection', type: 'concat_projection', name: 'Concat + Projection', icon: 'GitMerge', description: 'Concatenate expert outputs then project', defaultParams: { projection_bias: true, d_model: 768 }, category: 'Aggregation', tooltip: 'Concatenates top-K outputs then projects back to D' },
  { id: 'attention_pooling', type: 'attention_pooling', name: 'Attention Pooling', icon: 'Focus', description: 'Attention-based expert output pooling', defaultParams: { num_heads: 4, dropout: 0.0, d_model: 768 }, category: 'Aggregation', tooltip: 'Learns to attend over expert outputs for combination' },
  { id: 'shared_expert', type: 'shared_expert', name: 'Shared Expert Block', icon: 'Share2', description: 'Always-active experts + routed experts (DeepSeek)', defaultParams: { num_shared_experts: 2, num_routed_experts: 160, shared_expert_ffn_dim: 2048, d_model: 768, top_k: 6 }, category: 'Aggregation', tooltip: 'shared_out + routed_out — DeepSeek-V2/V3' },

  // ── Regularization / Aux Losses ──
  { id: 'load_balancing_loss', type: 'load_balancing_loss', name: 'Load Balancing Loss', icon: 'Scale', description: 'Aux loss for equal token distribution', defaultParams: { aux_loss_coef: 0.01, loss_type: 'both' }, category: 'Regularization', tooltip: 'loss = E/B² × Σ(f_i × P_i)' },
  { id: 'expert_capacity_limit', type: 'expert_capacity_limit', name: 'Expert Capacity Limit', icon: 'Shield', description: 'Cap tokens per expert', defaultParams: { capacity_factor: 1.25, drop_tokens: true }, category: 'Regularization', tooltip: 'Excess tokens dropped or overflow-routed' },
  { id: 'z_loss', type: 'z_loss', name: 'Z-Loss', icon: 'Activity', description: 'Penalizes large router logits', defaultParams: { z_loss_coef: 0.0001 }, category: 'Regularization', tooltip: 'loss = coef × Σ logits²' },
  { id: 'router_regularization', type: 'router_regularization', name: 'Router Entropy Reg', icon: 'Activity', description: 'Entropy + confidence penalties', defaultParams: { router_entropy_coef: 0.01, router_confidence_penalty: 0.001 }, category: 'Regularization', tooltip: 'Maximizes entropy to prevent expert collapse' },
  { id: 'jitter_noise', type: 'jitter_noise', name: 'Jitter Noise', icon: 'Zap', description: 'Multiplicative noise on router input', defaultParams: { jitter_noise: 0.01 }, category: 'Regularization', tooltip: 'Regularizes routing by adding multiplicative noise' },
  { id: 'capacity_computation', type: 'capacity_computation', name: 'Capacity Computation', icon: 'Calculator', description: 'Compute per-expert token capacity', defaultParams: { capacity_factor: 1.25 }, category: 'Utilities', tooltip: 'capacity = ceil(tokens × top_k / E × factor)' },
  { id: 'mask_generation', type: 'mask_generation', name: 'Mask Generation', icon: 'Grid3x3', description: 'Generate binary dispatch masks from indices', defaultParams: {}, category: 'Utilities', tooltip: 'Converts top-K indices to binary [B,S,E] dispatch mask' },

  // ── Composite MoE Blocks ──
  { id: 'moe_block', type: 'moe_block', name: 'MoE Layer (Standard)', icon: 'Network', description: 'Full MoE replacing FFN with router + experts', defaultParams: { num_experts: 8, top_k: 2, d_model: 768, expert_d_ff: 14336, expert_type: 'gated_ffn', router_type: 'top_k', router_bias: false, capacity_factor: 1.25, aux_loss_coef: 0.01, shared_experts: 0 }, category: 'MoE Composite', tooltip: 'Complete MoE layer — Mixtral/DeepSeek style' },
  { id: 'moe_layer', type: 'moe_layer', name: 'MoE Layer (Configurable)', icon: 'Network', description: 'Fully configurable MoE with all variants', defaultParams: { num_experts: 8, top_k: 2, d_model: 768, expert_d_ff: 14336, expert_type: 'gated_ffn', router_type: 'top_k', shared_experts: 0, aux_loss_coef: 0.01, z_loss_coef: 0.0001, capacity_factor: 1.25, dropout: 0.0 }, category: 'MoE Composite', tooltip: 'Advanced MoE with fine-grained control' },
  { id: 'moa_block', type: 'moa_block', name: 'Mixture of Attention (MoA)', icon: 'Focus', description: 'Expert routing for attention heads (JetMoE)', defaultParams: { num_attention_experts: 8, d_model: 768, n_kv_heads: 4, kv_channels: 64, top_k: 2 }, category: 'MoE Composite', tooltip: 'Attention heads as routed experts — JetMoE' },
  { id: 'fine_grained_moe', type: 'fine_grained_moe', name: 'Fine-Grained MoE', icon: 'Network', description: 'Many small experts from split originals', defaultParams: { num_experts: 8, granularity: 4, d_model: 768, d_ff: 3072, top_k: 2 }, category: 'MoE Composite', tooltip: 'Each expert split into G sub-experts — DeepSeek-V2' },
  { id: 'soft_moe', type: 'soft_moe', name: 'Soft MoE', icon: 'Blend', description: 'Soft routing via token/expert merging', defaultParams: { num_experts: 8, d_model: 768, merging_strategy: 'token_merging', temperature: 1.0 }, category: 'MoE Composite', tooltip: 'Non-discrete expert selection via soft merging' },
  { id: 'peer_layer', type: 'peer_layer', name: 'PEER Layer', icon: 'Network', description: 'Parameter Efficient Expert Retrieval', defaultParams: { num_experts: 1000000, d_model: 768, num_heads: 4, head_dim: 192, key_dim: 128, top_k: 2 }, category: 'MoE Composite', tooltip: '1M+ tiny experts with product-key retrieval' },
  { id: 'hierarchical_moe', type: 'hierarchical_moe', name: 'Hierarchical MoE', icon: 'Network', description: 'Two-level group→expert MoE', defaultParams: { num_groups: 4, experts_per_group: 4, top_k_groups: 2, top_k_experts: 1, d_model: 768, d_ff: 3072 }, category: 'MoE Composite', tooltip: 'Two-stage routing: select groups then experts within' },

  // ── Attention (shared with Transformer) ──
  { id: 'attention_score', type: 'attention_score', name: 'Scaled Dot-Product', icon: 'Focus', description: 'softmax(QK^T/√d)V core', defaultParams: { scale: 'auto', dropout: 0.0, causal: true, mask_type: 'causal' }, category: 'Attention' },
  { id: 'mha_attention', type: 'mha_attention', name: 'Multi-Head Attention', icon: 'Focus', description: 'Standard MHA', defaultParams: { n_heads: 12, head_dim: 64, d_model: 768, dropout: 0.0, bias: false, output_projection: true }, category: 'Attention' },
  { id: 'gqa_attention', type: 'gqa_attention', name: 'Grouped-Query Attention', icon: 'Focus', description: 'GQA for MoE models', defaultParams: { n_heads: 12, n_kv_heads: 4, d_model: 768, dropout: 0.0 }, category: 'Attention' },
  { id: 'mla_attention', type: 'mla_attention', name: 'Multi-head Latent Attn', icon: 'Focus', description: 'Latent compression (DeepSeek)', defaultParams: { n_heads: 12, d_model: 768, latent_dim: 128, compression_ratio: 4 }, category: 'Attention' },
  { id: 'cross_attention', type: 'cross_attention', name: 'Cross Attention', icon: 'Focus', description: 'Encoder-decoder cross attention', defaultParams: { d_model: 768, n_heads: 12, head_dim: 64, dropout: 0.0, causal_decoder: false }, category: 'Attention' },
  { id: 'flash_attention', type: 'flash_attention', name: 'Flash Attention', icon: 'Zap', description: 'IO-aware fused attention', defaultParams: { n_heads: 12, d_model: 768, version: 3, causal: true }, category: 'Attention' },

  // ── Normalization ──
  { ...NORM_BLOCKS.layernorm, defaultParams: { normalized_shape: 768, eps: 1e-6, elementwise_affine: true, position: 'pre' } },
  { ...NORM_BLOCKS.rmsnorm, defaultParams: { normalized_shape: 768, eps: 1e-5, elementwise_affine: true, position: 'pre' } },

  // ── Positional Encoding ──
  { id: 'pos_rope', type: 'pos_rope', name: 'RoPE', icon: 'Activity', description: 'Rotary position embedding', defaultParams: { d_model: 768, max_len: 131072, base: 10000, scaling_type: 'none' }, category: 'Position' },

  // ── Structure ──
  { ...STRUCT_BLOCKS.residual_add, defaultParams: { dropout: 0.0, pre_norm: true } },
  { ...STRUCT_BLOCKS.dropout },
  { id: 'layer_stack', type: 'layer_stack', name: 'Layer Stack', icon: 'Layers', description: 'Repeat N identical blocks', defaultParams: { num_layers: 32, share_parameters: false }, category: 'Structure' },
  { id: 'gradient_checkpoint', type: 'gradient_checkpoint', name: 'Gradient Checkpoint', icon: 'Box', description: 'Memory-compute tradeoff', defaultParams: { checkpoint_ratio: 1.0, offload_to_cpu: false }, category: 'Structure' },

  // ── Output Heads ──
  { id: 'lm_head', type: 'lm_head', name: 'LM Head', icon: 'ArrowRightFromLine', description: 'Next-token prediction head', defaultParams: { d_model: 768, vocab_size: 32000, bias: false, tie_weights: true }, category: 'Output' },
  { id: 'classification_head', type: 'classification_head', name: 'Classification Head', icon: 'ArrowRightFromLine', description: 'Downstream classification', defaultParams: { d_model: 768, num_classes: 10, pooling: 'cls' }, category: 'Output' },
];

// ─── SSM BLOCKS ───────────────────────────────────────────

const ssmBlocks: LayerConfig[] = [
  ...IO_BLOCKS,

  // ── Input / Embedding ──
  { id: 'token_embedding', type: 'token_embedding', name: 'Token Embedding', icon: 'Layers', description: 'Token → dense vector', defaultParams: { vocab_size: 32000, d_model: 512, padding_idx: 0 }, category: 'Input' },
  { id: 'embedding', type: 'embedding', name: 'Embedding', icon: 'Layers', description: 'Token + positional embedding', defaultParams: { vocab_size: 32000, d_model: 512 }, category: 'Input' },

  // ── Projection ──
  { id: 'linear_projection', type: 'linear_projection', name: 'Linear Projection', icon: 'Layers', description: 'Dense linear Wx + b', defaultParams: { in_features: 512, out_features: 512, bias: true, activation: 'none' }, category: 'Projection', hasActivation: true },
  { id: 'mha_attention', type: 'mha_attention', name: 'Multi-Head Attention', icon: 'Focus', description: 'Standard MHA for hybrid SSM models', defaultParams: { n_heads: 12, head_dim: 64, d_model: 512, dropout: 0.0, bias: false, output_projection: true }, category: 'Attention', tooltip: 'Attention branch used in Jamba-style hybrid stacks' },

  // ── Discretization ──
  { id: 'ssm_discretize', type: 'ssm_discretize', name: 'Discretization', icon: 'Workflow', description: 'Continuous → discrete (A,B) via ZOH/bilinear', defaultParams: { method: 'zoh', discretization_rule: 'exp', precision: 'float32' }, category: 'Discretization', tooltip: 'Transforms continuous A,B,Δ into discrete Ā,B̄' },
  { id: 'delta_computation', type: 'delta_computation', name: 'Delta Computation', icon: 'Workflow', description: 'Compute timestep Δ from input', defaultParams: { delta_type: 'input_dependent', d_model: 512, d_state: 16, bias: true }, category: 'Discretization', hasActivation: true, tooltip: 'Projects input to Δ with softplus activation' },

  // ── SSM Fundamental Blocks ──
  { id: 'state_matrix_a', type: 'state_matrix_a', name: 'State Matrix A', icon: 'Workflow', description: 'State transition matrix h\'=Ah+Bx', defaultParams: { d_state: 64, init_method: 'hippo', parameterization: 'diagonal' }, category: 'SSM Core', tooltip: 'Continuous-time state transition dynamics' },
  { id: 'state_matrix_b', type: 'state_matrix_b', name: 'State Matrix B', icon: 'Workflow', description: 'Input-to-state projection', defaultParams: { d_state: 64, d_model: 512 }, category: 'SSM Core', tooltip: 'Maps input x(t) into state space' },
  { id: 'state_matrix_c', type: 'state_matrix_c', name: 'State Matrix C', icon: 'Workflow', description: 'State-to-output projection y=Ch+Dx', defaultParams: { d_state: 64, d_model: 512 }, category: 'SSM Core', tooltip: 'Maps state h(t) to output y(t)' },
  { id: 's4_block', type: 's4_block', name: 'S4 Block', icon: 'Workflow', description: 'Structured State Space with HiPPO init', defaultParams: { d_model: 512, d_state: 64, ssm_type: 's4d', init_method: 'hippo', normalization: 'layer_norm' }, category: 'SSM Core', tooltip: 'S4/S4D — long-range memory via HiPPO matrices' },
  { id: 's5_block', type: 's5_block', name: 'S5 Block', icon: 'Workflow', description: 'Structured SSM with block parameterization', defaultParams: { d_model: 512, d_state: 64, blocks: 4, ssm_parameterization: 'diagonal_plus_low_rank' }, category: 'SSM Core', tooltip: 'Improved S4 with structured parameterization' },
  { id: 's6_block', type: 's6_block', name: 'S6 Block (Mamba)', icon: 'Zap', description: 'Selective SSM — input-dependent A,B,C,Δ', defaultParams: { d_model: 768, d_state: 16, expand: 2, dt_rank: 48, conv_kernel: 4, bias: false }, category: 'SSM Core', tooltip: 'Core of Mamba — selective scan with O(n) complexity' },
  { id: 'lru_block', type: 'lru_block', name: 'LRU Block', icon: 'Workflow', description: 'Linear Recurrent Unit with gating', defaultParams: { d_model: 512, d_state: 64, rnn_type: 'glru', gate_activation: 'sigmoid' }, category: 'SSM Core', tooltip: 'Gated linear recurrence — simpler than S4' },
  { id: 'linoss_block', type: 'linoss_block', name: 'LinOSS Block', icon: 'Workflow', description: 'Linear Oscillatory State-Space', defaultParams: { d_model: 512, d_state: 64, oscillation_type: 'damped', discretization: 'IMEX', A_diag: true, steps: 1 }, category: 'SSM Core', tooltip: 'Complex-valued oscillatory dynamics for cortical modeling' },

  // ── Selectivity Mechanisms ──
  { id: 'selective_scan', type: 'selective_scan', name: 'Selective Scan', icon: 'Zap', description: 'Input-dependent parallel SSM scan', defaultParams: { scan_type: 'selective_scan', d_state: 16, use_fast_scan: true }, category: 'Selectivity', tooltip: 'Hardware-aware associative parallel scan — O(n)' },
  { id: 'input_dependent_timescale', type: 'input_dependent_timescale', name: 'Input-Dependent Timescales', icon: 'Clock', description: 'Learned timescales from input', defaultParams: { timescale_type: 'input_dependent' }, category: 'Selectivity', hasActivation: true, tooltip: 'Softplus-activated Δ for adaptive time steps' },
  { id: 's7_selection', type: 's7_selection', name: 'S7 Selection', icon: 'Zap', description: 'Enhanced selectivity with gating', defaultParams: { selection_type: 's7', gating_mechanism: 'multiplicative' }, category: 'Selectivity', tooltip: 'Improved selection mechanism over S6' },
  { id: 'serpent_selection', type: 'serpent_selection', name: 'SeRpEnt Selection', icon: 'Zap', description: 'Selective resampling for compression', defaultParams: { resampling_rate: 0.5, time_interval_learner: true }, category: 'Selectivity', tooltip: 'Learns to compress sequence via resampling' },

  // ── Scan Strategies ──
  { id: 'scan_1d', type: 'scan_1d', name: '1D Scan (Causal)', icon: 'ArrowRight', description: 'Sequential scan for text/audio', defaultParams: { direction: 'forward', causal: true }, category: 'Scan', tooltip: 'Standard forward causal scan or bidirectional' },
  { id: 'scan_2d', type: 'scan_2d', name: '2D Scan (SS2D)', icon: 'Maximize', description: '4-direction scan for images', defaultParams: { scan_directions: 'h,v,h_flip,v_flip', alternating: false, merge_strategy: 'sum' }, category: 'Scan', tooltip: 'Horizontal + vertical + flipped scans for vision' },
  { id: 'scan_multidirectional', type: 'scan_multidirectional', name: 'Multi-D Scan', icon: 'Move3d', description: 'N-dimensional scan (Mamba-ND)', defaultParams: { dimensions: 2, scan_orders: 'auto', block_scan: false }, category: 'Scan', tooltip: 'Generalized scan for images, video, 3D data' },
  { id: 'scan_spiral', type: 'scan_spiral', name: 'Spiral Scan', icon: 'RotateCw', description: 'Spiral traversal for 2D data', defaultParams: { spiral_type: 'square', resolution: 'auto' }, category: 'Scan', tooltip: 'Archimedean or square spiral — I2I-Mamba' },
  { id: 'scan_diagonal', type: 'scan_diagonal', name: 'Diagonal Scan', icon: 'TrendingUp', description: 'Diagonal traversal for oblique correlations', defaultParams: { direction: 'top_left' }, category: 'Scan', tooltip: 'Captures diagonal spatial dependencies' },

  // ── Convolution & Projection ──
  { id: 'causal_conv1d', type: 'causal_conv1d', name: 'Causal Conv1D', icon: 'Workflow', description: 'Causal 1D convolution before SSM', defaultParams: { in_channels: 768, out_channels: 768, kernel_size: 4, groups: 1 }, category: 'Conv/Proj', tooltip: 'Local causal convolution — padding = kernel_size-1' },
  { id: 'conv1d', type: 'conv1d', name: 'Conv1D', icon: 'Workflow', description: 'Generic 1D convolution for SSM macros', defaultParams: { in_channels: 768, out_channels: 768, kernel_size: 4, groups: 768 }, category: 'Conv/Proj', tooltip: 'Non-causal 1D convolution used by macro templates' },
  { id: 'ssm_in_proj', type: 'ssm_in_proj', name: 'SSM InProj', icon: 'Layers', description: 'Input expansion projection', defaultParams: { d_model: 768, expand: 2, bias: false }, category: 'Conv/Proj', tooltip: '[B,S,D] → [B,S,E×D] expansion before SSM' },
  { id: 'ssm_out_proj', type: 'ssm_out_proj', name: 'SSM OutProj', icon: 'Layers', description: 'Output compression projection', defaultParams: { d_model: 768, expand: 2, bias: false }, category: 'Conv/Proj', tooltip: '[B,S,E×D] → [B,S,D] compression after SSM' },
  { id: 'delta_proj', type: 'delta_proj', name: 'Delta Projection', icon: 'Layers', description: 'Δ parameter projection with softplus', defaultParams: { d_in: 1536, dt_rank: 48, d_state: 16 }, category: 'Conv/Proj', hasActivation: true, tooltip: 'Projects expanded input to Δ timestep' },
  { id: 'bc_proj', type: 'bc_proj', name: 'B/C Projection', icon: 'Layers', description: 'Selective B and C parameter projection', defaultParams: { d_in: 1536, d_state: 16 }, category: 'Conv/Proj', tooltip: 'Input-dependent B,C for selectivity' },

  // ── Gating & Activation ──
  { id: 'glu_block', type: 'glu_block', name: 'Gated Linear Unit', icon: 'Zap', description: 'GLU gate for SSM architectures', defaultParams: { d_model: 768 }, category: 'Gating', hasActivation: true, tooltip: 'σ(xW₁) ⊙ xW₂ gating mechanism' },
  { id: 'hadamard_product', type: 'hadamard_product', name: 'Hadamard Product', icon: 'X', description: 'Element-wise multiplication', defaultParams: {}, category: 'Gating', tooltip: 'Combines gate and up projections element-wise' },

  // ── Normalization ──
  { ...NORM_BLOCKS.layernorm, defaultParams: { normalized_shape: 512, eps: 1e-6, elementwise_affine: true } },
  { ...NORM_BLOCKS.rmsnorm, defaultParams: { normalized_shape: 512, eps: 1e-5, elementwise_affine: true } },
  { ...NORM_BLOCKS.batchnorm },

  // ── Extended SSM Fundamentals ──
  { id: 'h3_block', type: 'h3_block', name: 'H3 Block', icon: 'Workflow', description: 'SSM + convolution hybrid (H3)', defaultParams: { d_model: 512, d_state: 64, kernel_size: 4, ssm_type: 's4d' }, category: 'SSM Core', tooltip: 'Hungry Hungry Hippos — combines SSM with learned convolution' },
  { id: 'hyena_conv', type: 'hyena_conv', name: 'Hyena Operator', icon: 'Workflow', description: 'Long convolution with learned filters', defaultParams: { d_model: 512, order: 2, filter_dim: 64, max_len: 8192 }, category: 'SSM Core', tooltip: 'Sub-quadratic operator via implicit long convolutions' },
  { id: 'gated_ssm', type: 'gated_ssm', name: 'Gated SSM', icon: 'Workflow', description: 'SSM with learned gate activation', defaultParams: { d_model: 512, d_state: 64, gate_activation: 'sigmoid' }, category: 'SSM Core', hasActivation: true, tooltip: 'Gated State Space — gate controls information flow' },
  { id: 'neural_cde', type: 'neural_cde', name: 'Neural CDE', icon: 'Workflow', description: 'Neural Controlled Differential Equation', defaultParams: { d_model: 256, d_state: 64, solver: 'euler', step_size: 0.01, adjoint: false }, category: 'SSM Core', tooltip: 'Continuous-time SSM variant for irregular time series' },
  { id: 'a_parameterization', type: 'a_parameterization', name: 'A Parameterization', icon: 'Workflow', description: 'Structure of state matrix A', defaultParams: { type: 'diagonal', rank: 1, init_type: 'hippo', d_state: 64 }, category: 'Discretization', tooltip: 'Diagonal, DPLR, companion, or dense A matrix structure' },

  // ── Hybrid & Multimodal ──
  { id: 'hybrid_ssm_attn', type: 'hybrid_ssm_attn', name: 'Hybrid SSM+Attention', icon: 'GitBranch', description: 'Combined attention + SSM layer (Jamba)', defaultParams: { d_model: 768, attn_type: 'gqa', ssm_type: 's6', mix_ratio: 0.5, n_heads: 12 }, category: 'SSM Composite', tooltip: 'Interleaves attention and SSM blocks — Jamba architecture' },
  { id: 'multimodal_mamba', type: 'multimodal_mamba', name: 'MultiModal Mamba', icon: 'Layers', description: 'Multimodal fusion SSM block', defaultParams: { d_model: 768, modalities: 'text,image', fusion_type: 'concat', d_state: 16 }, category: 'SSM Composite', tooltip: 'Fuses multiple modalities through selective SSM' },

  // ── Scan Extended ──
  { id: 'scan_operator', type: 'scan_operator', name: 'Scan Operator', icon: 'Workflow', description: 'Generic associative scan primitive', defaultParams: { associative_op: 'plus', reverse: false, bidirectional: false }, category: 'Scan', tooltip: 'Configurable parallel scan with custom associative ops' },
  { id: 'scan_block', type: 'scan_block', name: 'Block Scan', icon: 'Grid3X3', description: 'Scan by spatial blocks', defaultParams: { block_size: 8 }, category: 'Scan', tooltip: 'Processes data in spatial blocks before scanning' },

  // ── Utilities ──
  { id: 'state_reset', type: 'state_reset', name: 'State Reset', icon: 'RotateCcw', description: 'Reset latent state between sequences', defaultParams: { reset_type: 'zero' }, category: 'Structure', tooltip: 'Reinitializes hidden state h between independent batches' },
  { id: 'ssm_layernorm', type: 'ssm_layernorm', name: 'SSM LayerNorm', icon: 'AlignCenter', description: 'LayerNorm adapted for post-scan', defaultParams: { normalized_shape: 512, eps: 1e-6, elementwise_affine: true }, category: 'Normalization', tooltip: 'Applied after SSM scan for training stability' },

  // ── Composite SSM Architectures ──
  { id: 'vss_block', type: 'vss_block', name: 'VSS Block (Vision)', icon: 'Eye', description: 'Visual State Space with SS2D + channel attention', defaultParams: { d_model: 96, d_state: 16, expand: 2, dropout: 0.0, channel_attention: true }, category: 'SSM Composite', tooltip: 'Vision Mamba block — SS2D + normalization + channel attn' },
  { id: 'rssg_block', type: 'rssg_block', name: 'RSSG Block', icon: 'Layers', description: 'Residual State-Space Group (MambaIR)', defaultParams: { num_blocks: 6, input_resolution: 64, dropout: 0.0 }, category: 'SSM Composite', tooltip: 'Hierarchical group of VSS blocks with residual connections' },
  { id: 'basic_layer_ssm', type: 'basic_layer_ssm', name: 'Basic SSM Layer', icon: 'Layers', description: 'Hierarchical SSM layer at one scale', defaultParams: { depth: 4, resolution: 64, d_model: 96 }, category: 'SSM Composite', tooltip: 'Stack of SSM blocks at a single resolution level' },
  { id: 'stg_mamba_block', type: 'stg_mamba_block', name: 'STG-Mamba Block', icon: 'GitBranch', description: 'Spatiotemporal graph SSM with Kalman fusion', defaultParams: { node_dim: 64, time_steps: 12, kalman_fusion: true, d_state: 16 }, category: 'SSM Composite', tooltip: 'Spatiotemporal graph modeling with Kalman filter fusion' },
  { id: 'dual_path_mamba', type: 'dual_path_mamba', name: 'Dual-Path Mamba', icon: 'GitBranch', description: 'Local + global dual-path SSM', defaultParams: { path1_type: 'local', path2_type: 'global', interleaving: 'alternate', d_model: 256 }, category: 'SSM Composite', tooltip: 'Two parallel SSM paths for speech separation' },
  { id: 'mamba_mixer', type: 'mamba_mixer', name: 'MambaMixer', icon: 'Shuffle', description: 'Token + channel selective mixing', defaultParams: { d_model: 768, token_mixer: 'selective', channel_mixer: 'selective', skip_connection: 'residual' }, category: 'SSM Composite', tooltip: 'Dual selective mixing on both token and channel dimensions' },

  // ── Multi-Scale ──
  { id: 'multiscale_ssm', type: 'multiscale_ssm', name: 'Multiscale SSM Backbone', icon: 'BarChart3', description: 'Multi-resolution SSM for fast/slow dynamics', defaultParams: { num_scales: 3, scale_factors: '1,2,4', interpolation: 'linear', d_model: 256 }, category: 'Multi-Scale', tooltip: 'NeuroSSM — captures dynamics at multiple timescales' },
  { id: 'parallel_differencing', type: 'parallel_differencing', name: 'Parallel Differencing', icon: 'GitBranch', description: 'Detect transient state changes', defaultParams: { diff_order: 1, merge_method: 'concat' }, category: 'Multi-Scale', tooltip: 'Parallel branch computing temporal differences' },

  // ── Neuromorphic ──
  { id: 'spiking_ssm', type: 'spiking_ssm', name: 'Spiking SSM Layer', icon: 'Zap', description: 'SSM with spiking neurons for energy efficiency', defaultParams: { d_model: 256, d_state: 16, threshold: 1.0, reset_mechanism: 'subtract', spike_rate: 0.1 }, category: 'Neuromorphic', tooltip: 'SpikySpace — spike-driven SSM with sparse activations' },
  { id: 'neuromorphic_activation', type: 'neuromorphic_activation', name: 'Neuromorphic Activation', icon: 'Zap', description: 'Hardware-friendly activation approximations', defaultParams: { approx_type: 'piecewise_linear', precision: 'ternary' }, category: 'Neuromorphic', tooltip: 'Binary/ternary activations for neuromorphic hardware' },

  // ── Positional Encoding ──
  { id: 'pos_absolute', type: 'pos_absolute', name: 'Absolute Position', icon: 'Activity', description: 'Sinusoidal or learned positional encoding', defaultParams: { d_model: 512, max_len: 8192, encoding_type: 'sinusoidal' }, category: 'Position' },
  { id: 'pos_rope', type: 'pos_rope', name: 'RoPE', icon: 'Activity', description: 'Rotary position embedding', defaultParams: { d_model: 512, max_len: 131072, base: 10000 }, category: 'Position' },

  // ── Structure ──
  { ...STRUCT_BLOCKS.residual_add, defaultParams: { dropout: 0.0, pre_norm: true } },
  { ...STRUCT_BLOCKS.dropout },
  { id: 'layer_stack', type: 'layer_stack', name: 'Layer Stack', icon: 'Layers', description: 'Repeat N identical blocks', defaultParams: { num_layers: 24, share_parameters: false }, category: 'Structure' },
  { id: 'gradient_checkpoint', type: 'gradient_checkpoint', name: 'Gradient Checkpoint', icon: 'Box', description: 'Memory-compute tradeoff', defaultParams: { checkpoint_ratio: 1.0, offload_to_cpu: false }, category: 'Structure' },

  // ── Output Heads ──
  { id: 'lm_head', type: 'lm_head', name: 'LM Head', icon: 'ArrowRightFromLine', description: 'Next-token prediction head', defaultParams: { d_model: 768, vocab_size: 32000, bias: false, tie_weights: true }, category: 'Output' },
  { id: 'classification_head', type: 'classification_head', name: 'Classification Head', icon: 'ArrowRightFromLine', description: 'Downstream classification', defaultParams: { d_model: 512, num_classes: 10, pooling: 'mean' }, category: 'Output' },
  { id: 'ssm_output_head', type: 'ssm_output_head', name: 'SSM Output Head', icon: 'ArrowRightFromLine', description: 'Task-adaptive SSM output', defaultParams: { d_model: 512, task: 'classification', pooling: 'mean' }, category: 'Output', tooltip: 'Flexible output with mean/last/max/attention pooling' },
  { id: 'forecasting_head', type: 'forecasting_head', name: 'Forecasting Head', icon: 'TrendingUp', description: 'Time-series prediction head', defaultParams: { d_model: 512, horizon: 24, strategy: 'direct' }, category: 'Output', tooltip: 'Direct/recursive/seq2seq forecasting strategies' },
];

// ─── MAMBA BLOCKS ─────────────────────────────────────────

const mambaBlocks: LayerConfig[] = [
  ...IO_BLOCKS,

  // ── Input ──
  { id: 'token_embedding', type: 'token_embedding', name: 'Token Embedding', icon: 'Layers', description: 'Token → dense vector', defaultParams: { vocab_size: 32000, d_model: 768, padding_idx: 0 }, category: 'Input' },
  { id: 'embedding', type: 'embedding', name: 'Embedding', icon: 'Layers', description: 'Token + positional embedding', defaultParams: { vocab_size: 32000, d_model: 768 }, category: 'Input' },

  // ── Projection ──
  { id: 'ssm_in_proj', type: 'ssm_in_proj', name: 'InProj (Expand)', icon: 'Layers', description: 'D → 2×E×D expansion', defaultParams: { d_model: 768, expand: 2, bias: false }, category: 'Projection', tooltip: 'Splits into gate and SSM branches' },
  { id: 'ssm_out_proj', type: 'ssm_out_proj', name: 'OutProj (Compress)', icon: 'Layers', description: 'E×D → D compression', defaultParams: { d_model: 768, expand: 2, bias: false }, category: 'Projection' },
  { id: 'linear_projection', type: 'linear_projection', name: 'Linear Projection', icon: 'Layers', description: 'Dense linear transform', defaultParams: { in_features: 768, out_features: 768, bias: true }, category: 'Projection', hasActivation: true },

  // ── Mamba Core ──
  { id: 'causal_conv1d', type: 'causal_conv1d', name: 'Causal Conv1D', icon: 'Workflow', description: 'Causal 1D convolution before SSM', defaultParams: { in_channels: 1536, out_channels: 1536, kernel_size: 4, groups: 1536 }, category: 'Mamba Core', tooltip: 'Depthwise causal conv — local context before scan' },
  { id: 's6_block', type: 's6_block', name: 'S6 Block (Mamba Core)', icon: 'Zap', description: 'Selective SSM with input-dependent params', defaultParams: { d_model: 768, d_state: 16, expand: 2, dt_rank: 48, conv_kernel: 4, bias: false }, category: 'Mamba Core', tooltip: 'Full Mamba S6 block — selective scan O(n)' },
  { id: 'selective_scan', type: 'selective_scan', name: 'Selective Scan', icon: 'Zap', description: 'Input-dependent parallel SSM scan', defaultParams: { scan_type: 'selective_scan', d_state: 16, use_fast_scan: true }, category: 'Mamba Core', tooltip: 'Hardware-aware associative parallel scan' },
  { id: 'delta_proj', type: 'delta_proj', name: 'Delta Projection', icon: 'Layers', description: 'Δ timestep projection + softplus', defaultParams: { d_in: 1536, dt_rank: 48, d_state: 16 }, category: 'Mamba Core', hasActivation: true },
  { id: 'bc_proj', type: 'bc_proj', name: 'B/C Projection', icon: 'Layers', description: 'Input-dependent B,C projection', defaultParams: { d_in: 1536, d_state: 16 }, category: 'Mamba Core' },
  { id: 'state_matrix_a', type: 'state_matrix_a', name: 'State Matrix A', icon: 'Workflow', description: 'State transition (input-dependent)', defaultParams: { d_state: 16, init_method: 'hippo', parameterization: 'diagonal' }, category: 'Mamba Core' },
  { id: 'state_matrix_b', type: 'state_matrix_b', name: 'State Matrix B', icon: 'Workflow', description: 'Input projection (selective)', defaultParams: { d_state: 16, d_model: 768 }, category: 'Mamba Core' },
  { id: 'state_matrix_c', type: 'state_matrix_c', name: 'State Matrix C', icon: 'Workflow', description: 'Output projection (selective)', defaultParams: { d_state: 16, d_model: 768 }, category: 'Mamba Core' },

  // ── Gating ──
  { id: 'glu_block', type: 'glu_block', name: 'Gated Linear Unit', icon: 'Zap', description: 'SiLU gate for Mamba', defaultParams: { d_model: 768 }, category: 'Gating', hasActivation: true },
  { id: 'hadamard_product', type: 'hadamard_product', name: 'Hadamard Product', icon: 'X', description: 'Element-wise gate × value', defaultParams: {}, category: 'Gating' },

  // ── Scan Strategies ──
  { id: 'scan_1d', type: 'scan_1d', name: '1D Scan', icon: 'ArrowRight', description: 'Forward/backward causal scan', defaultParams: { direction: 'forward', causal: true }, category: 'Scan' },
  { id: 'scan_2d', type: 'scan_2d', name: '2D Scan (SS2D)', icon: 'Maximize', description: '4-direction scan for vision', defaultParams: { scan_directions: 'h,v,h_flip,v_flip', merge_strategy: 'sum' }, category: 'Scan' },
  { id: 'scan_multidirectional', type: 'scan_multidirectional', name: 'Multi-D Scan', icon: 'Move3d', description: 'N-dimensional Mamba-ND scan', defaultParams: { dimensions: 2, block_scan: false }, category: 'Scan' },
  { id: 'scan_operator', type: 'scan_operator', name: 'Scan Operator', icon: 'Workflow', description: 'Generic associative scan', defaultParams: { associative_op: 'plus', reverse: false, bidirectional: false }, category: 'Scan' },
  { id: 'scan_block', type: 'scan_block', name: 'Block Scan', icon: 'Grid3X3', description: 'Scan by spatial blocks', defaultParams: { block_size: 8 }, category: 'Scan' },

  // ── Composite ──
  { id: 'vss_block', type: 'vss_block', name: 'VSS Block', icon: 'Eye', description: 'Vision Mamba block', defaultParams: { d_model: 96, d_state: 16, expand: 2, channel_attention: true }, category: 'Mamba Composite' },
  { id: 'mamba_mixer', type: 'mamba_mixer', name: 'MambaMixer', icon: 'Shuffle', description: 'Token + channel selective mixing', defaultParams: { d_model: 768, token_mixer: 'selective', channel_mixer: 'selective' }, category: 'Mamba Composite' },
  { id: 'dual_path_mamba', type: 'dual_path_mamba', name: 'Dual-Path Mamba', icon: 'GitBranch', description: 'Local + global dual path', defaultParams: { path1_type: 'local', path2_type: 'global', d_model: 256 }, category: 'Mamba Composite' },
  { id: 'hybrid_ssm_attn', type: 'hybrid_ssm_attn', name: 'Hybrid SSM+Attention', icon: 'GitBranch', description: 'Attention + SSM (Jamba)', defaultParams: { d_model: 768, attn_type: 'gqa', ssm_type: 's6', mix_ratio: 0.5, n_heads: 12 }, category: 'Mamba Composite', tooltip: 'Interleaved attention and Mamba blocks' },
  { id: 'multimodal_mamba', type: 'multimodal_mamba', name: 'MultiModal Mamba', icon: 'Layers', description: 'Multimodal fusion via Mamba', defaultParams: { d_model: 768, modalities: 'text,image', fusion_type: 'concat', d_state: 16 }, category: 'Mamba Composite' },

  // ── Normalization ──
  { ...NORM_BLOCKS.rmsnorm, defaultParams: { normalized_shape: 768, eps: 1e-5, elementwise_affine: true } },
  { ...NORM_BLOCKS.layernorm, defaultParams: { normalized_shape: 768, eps: 1e-6, elementwise_affine: true } },
  { id: 'ssm_layernorm', type: 'ssm_layernorm', name: 'SSM LayerNorm', icon: 'AlignCenter', description: 'Post-scan LayerNorm', defaultParams: { normalized_shape: 768, eps: 1e-6, elementwise_affine: true }, category: 'Normalization' },

  // ── Structure ──
  { ...STRUCT_BLOCKS.residual_add, defaultParams: { dropout: 0.0, pre_norm: true } },
  { ...STRUCT_BLOCKS.dropout },
  { id: 'layer_stack', type: 'layer_stack', name: 'Layer Stack', icon: 'Layers', description: 'Repeat N Mamba blocks', defaultParams: { num_layers: 48, share_parameters: false }, category: 'Structure' },
  { id: 'state_reset', type: 'state_reset', name: 'State Reset', icon: 'RotateCcw', description: 'Reset hidden state between sequences', defaultParams: { reset_type: 'zero' }, category: 'Structure' },

  // ── Output ──
  { id: 'lm_head', type: 'lm_head', name: 'LM Head', icon: 'ArrowRightFromLine', description: 'Next-token prediction head', defaultParams: { d_model: 768, vocab_size: 32000, bias: false, tie_weights: true }, category: 'Output' },
  { id: 'classification_head', type: 'classification_head', name: 'Classification Head', icon: 'ArrowRightFromLine', description: 'Classification output', defaultParams: { d_model: 768, num_classes: 10, pooling: 'last' }, category: 'Output' },
  { id: 'ssm_output_head', type: 'ssm_output_head', name: 'SSM Output Head', icon: 'ArrowRightFromLine', description: 'Task-adaptive output', defaultParams: { d_model: 768, task: 'lm', pooling: 'last' }, category: 'Output' },
  { id: 'forecasting_head', type: 'forecasting_head', name: 'Forecasting Head', icon: 'TrendingUp', description: 'Time-series prediction', defaultParams: { d_model: 768, horizon: 24, strategy: 'direct' }, category: 'Output' },
];

// ─── CNN BLOCKS ───────────────────────────────────────────

const cnnBlocks: LayerConfig[] = [
  { ...IO_BLOCKS[0], defaultParams: { shape: '[B, C, H, W]' } },
  { ...IO_BLOCKS[1], defaultParams: { shape: '[B, classes]' } },

  // ── Convolutions ──
  { id: 'conv2d', type: 'conv2d', name: 'Conv2D', icon: 'Grid3X3', description: 'Standard 2D convolution', defaultParams: { in_channels: 3, out_channels: 64, kernel_size: 3, stride: 1, padding: 1, dilation: 1, groups: 1, bias: true }, category: 'Convolution', hasActivation: true, tooltip: 'Standard spatial convolution' },
  { id: 'conv1d', type: 'conv1d', name: 'Conv1D', icon: 'Grid3X3', description: '1D convolution', defaultParams: { in_channels: 64, out_channels: 64, kernel_size: 3, stride: 1, padding: 1, bias: true }, category: 'Convolution', hasActivation: true },
  { id: 'conv3d', type: 'conv3d', name: 'Conv3D', icon: 'Grid3X3', description: '3D convolution (video/volumetric)', defaultParams: { in_channels: 3, out_channels: 64, kernel_size: 3, stride: 1, padding: 1, bias: true }, category: 'Convolution', hasActivation: true },
  { id: 'transposed_conv', type: 'transposed_conv', name: 'ConvTranspose2D', icon: 'Grid3X3', description: 'Transposed (deconv) upsampling', defaultParams: { in_channels: 256, out_channels: 128, kernel_size: 4, stride: 2, padding: 1, output_padding: 0, bias: true }, category: 'Convolution', hasActivation: true },
  { id: 'depthwise_conv', type: 'depthwise_conv', name: 'Depthwise Conv2D', icon: 'Grid3X3', description: 'Per-channel convolution', defaultParams: { in_channels: 64, kernel_size: 3, stride: 1, padding: 1, bias: false }, category: 'Convolution', hasActivation: true, tooltip: 'Each channel convolved independently' },
  { id: 'pointwise_conv', type: 'pointwise_conv', name: 'Pointwise Conv2D', icon: 'Grid3X3', description: '1×1 channel mixing', defaultParams: { in_channels: 64, out_channels: 128, bias: false }, category: 'Convolution', hasActivation: true, tooltip: '1×1 convolution for channel projection' },
  { id: 'separable_conv', type: 'separable_conv', name: 'Separable Conv2D', icon: 'Grid3X3', description: 'Depthwise + pointwise', defaultParams: { in_channels: 64, out_channels: 128, kernel_size: 3, stride: 1, padding: 1, bias: false }, category: 'Convolution', hasActivation: true, tooltip: 'Factored conv: depthwise then pointwise' },
  { id: 'deformable_conv', type: 'deformable_conv', name: 'Deformable Conv2D', icon: 'Grid3X3', description: 'Learnable sampling offsets', defaultParams: { in_channels: 64, out_channels: 64, kernel_size: 3, stride: 1, padding: 1, deformable_groups: 1 }, category: 'Convolution', hasActivation: true, tooltip: 'Offsets learned for adaptive receptive field' },
  { id: 'large_kernel_conv', type: 'large_kernel_conv', name: 'Large Kernel Conv', icon: 'Grid3X3', description: 'Large kernel via depthwise decomposition', defaultParams: { in_channels: 64, out_channels: 64, kernel_size: 31, decomposition: 'depthwise' }, category: 'Convolution', hasActivation: true, tooltip: 'ConvNeXt/RepLKNet-style large kernels' },
  { id: 'mixed_kernel_conv', type: 'mixed_kernel_conv', name: 'Mixed Kernel Conv', icon: 'Grid3X3', description: 'Multiple kernel sizes in parallel', defaultParams: { in_channels: 64, out_channels: 64, kernel_sizes: '3,5,7' }, category: 'Convolution', hasActivation: true, tooltip: 'Inception-style multi-scale convolutions' },
  { id: 'dynamic_conv', type: 'dynamic_conv', name: 'Dynamic Conv', icon: 'Grid3X3', description: 'Input-dependent weights', defaultParams: { in_channels: 64, out_channels: 64, num_experts: 4, k: 2 }, category: 'Convolution', hasActivation: true, tooltip: 'Dynamically generated convolutional weights' },
  { id: 'sparse_conv', type: 'sparse_conv', name: 'Sparse Conv', icon: 'Grid3X3', description: 'Sparse convolution for efficiency', defaultParams: { in_channels: 64, out_channels: 64, kernel_size: 3, sparsity: 0.5 }, category: 'Convolution', hasActivation: true },
  { id: 'involution', type: 'involution', name: 'Involution', icon: 'Grid3X3', description: 'Spatial-specific, channel-agnostic op', defaultParams: { in_channels: 64, out_channels: 64, kernel_size: 7, stride: 1 }, category: 'Convolution', hasActivation: true, tooltip: 'Inverse of convolution — location-specific kernels' },
  { id: 'gated_conv', type: 'gated_conv', name: 'Gated Conv', icon: 'Grid3X3', description: 'Convolution with gating mechanism', defaultParams: { in_channels: 64, out_channels: 64, kernel_size: 3 }, category: 'Convolution', hasActivation: true },
  { id: 'coord_conv', type: 'coord_conv', name: 'CoordConv', icon: 'Grid3X3', description: 'Conv with coordinate channels', defaultParams: { in_channels: 64, out_channels: 64, kernel_size: 3, with_r: false }, category: 'Convolution', hasActivation: true, tooltip: 'Appends (x,y) coordinate channels to input' },

  // ── Pooling ──
  { id: 'max_pool', type: 'max_pool', name: 'MaxPool2D', icon: 'Grid3X3', description: 'Max pooling', defaultParams: { kernel_size: 2, stride: 2, padding: 0 }, category: 'Pooling' },
  { id: 'avg_pool', type: 'avg_pool', name: 'AvgPool2D', icon: 'Grid3X3', description: 'Average pooling', defaultParams: { kernel_size: 2, stride: 2, padding: 0 }, category: 'Pooling' },
  { id: 'adaptive_pool', type: 'adaptive_pool', name: 'Adaptive AvgPool', icon: 'Grid3X3', description: 'Adaptive output-size pooling', defaultParams: { output_size: 1 }, category: 'Pooling' },
  { id: 'global_pool', type: 'global_pool', name: 'Global AvgPool', icon: 'Grid3X3', description: 'Global average pooling → [B,C]', defaultParams: { type: 'avg' }, category: 'Pooling', tooltip: 'Collapses spatial dims to single vector' },
  { id: 'roi_pool', type: 'roi_pool', name: 'ROI Pool', icon: 'Grid3X3', description: 'Region-of-interest pooling', defaultParams: { output_size: 7, spatial_scale: 0.0625 }, category: 'Pooling', tooltip: 'Pools features for each detection ROI' },
  { id: 'roi_align', type: 'roi_align', name: 'ROI Align', icon: 'Grid3X3', description: 'Bilinear ROI alignment', defaultParams: { output_size: 7, spatial_scale: 0.0625, sampling_ratio: 2 }, category: 'Pooling', tooltip: 'Sub-pixel accurate ROI feature extraction' },

  // ── Normalization ──
  { ...NORM_BLOCKS.batchnorm, defaultParams: { num_features: 64, eps: 1e-5, momentum: 0.1, affine: true } },
  { ...NORM_BLOCKS.groupnorm, defaultParams: { num_groups: 32, num_channels: 64, eps: 1e-5, affine: true } },
  { ...NORM_BLOCKS.instancenorm, defaultParams: { num_features: 64, eps: 1e-5, affine: false } },
  { ...NORM_BLOCKS.layernorm, defaultParams: { normalized_shape: 768, eps: 1e-6, elementwise_affine: true } },
  { id: 'sync_batchnorm', type: 'sync_batchnorm', name: 'SyncBatchNorm', icon: 'AlignJustify', description: 'Cross-GPU synchronized BatchNorm', defaultParams: { num_features: 64, eps: 1e-5, momentum: 0.1, affine: true }, category: 'Normalization', tooltip: 'Syncs statistics across distributed GPUs' },
  { id: 'switchable_norm', type: 'switchable_norm', name: 'SwitchableNorm', icon: 'AlignJustify', description: 'Learnable mix of BN/IN/LN', defaultParams: { num_features: 64, eps: 1e-5, momentum: 0.1 }, category: 'Normalization', tooltip: 'Weighted combination of normalization types' },
  { id: 'filter_response_norm', type: 'filter_response_norm', name: 'Filter Response Norm', icon: 'AlignJustify', description: 'FRN — batch-independent norm', defaultParams: { num_features: 64, eps: 1e-6, learnable_eps: true }, category: 'Normalization', tooltip: 'No batch dependency — stable for small batch sizes' },
  { id: 'adaptive_instance_norm', type: 'adaptive_instance_norm', name: 'AdaIN', icon: 'AlignJustify', description: 'Adaptive Instance Normalization', defaultParams: { in_channels: 256 }, category: 'Normalization', tooltip: 'Style transfer via external style statistics' },
  { id: 'conditional_batchnorm', type: 'conditional_batchnorm', name: 'Conditional BN', icon: 'AlignJustify', description: 'Class-conditional BatchNorm', defaultParams: { in_channels: 256, num_classes: 1000 }, category: 'Normalization', tooltip: 'Per-class affine parameters for class-conditional generation' },
  { id: 'spectral_norm', type: 'spectral_norm', name: 'Spectral Norm', icon: 'AlignJustify', description: 'Spectral normalization wrapper', defaultParams: { num_features: 256, power_iterations: 1 }, category: 'Normalization', tooltip: 'Constrains Lipschitz constant for training stability' },

  // ── Residual & Dense Blocks ──
  { id: 'basic_block', type: 'basic_block', name: 'BasicBlock (ResNet)', icon: 'Box', description: '2×Conv3×3 + skip (ResNet-18/34)', defaultParams: { in_channels: 64, out_channels: 64, stride: 1, downsample: false, activation: 'relu' }, category: 'Residual Blocks', hasActivation: true, tooltip: 'Standard ResNet basic residual block' },
  { id: 'bottleneck_block', type: 'bottleneck_block', name: 'Bottleneck (ResNet)', icon: 'Box', description: '1×1 → 3×3 → 1×1 bottleneck', defaultParams: { in_channels: 256, out_channels: 256, stride: 1, expansion: 4, activation: 'relu' }, category: 'Residual Blocks', hasActivation: true, tooltip: 'ResNet-50/101/152 bottleneck with expansion=4' },
  { id: 'preact_block', type: 'preact_block', name: 'Pre-Act Block', icon: 'Box', description: 'BN → ReLU → Conv (pre-activation)', defaultParams: { in_channels: 64, out_channels: 64, stride: 1 }, category: 'Residual Blocks', hasActivation: true, tooltip: 'Pre-activation ordering — improved gradient flow' },
  { id: 'dense_layer', type: 'dense_layer', name: 'Dense Layer', icon: 'Layers', description: 'BN → ReLU → Conv (DenseNet)', defaultParams: { in_channels: 64, growth_rate: 32, bn_size: 4, drop_rate: 0.0 }, category: 'Residual Blocks', hasActivation: true, tooltip: 'Single DenseNet layer producing growth_rate channels' },
  { id: 'dense_block', type: 'dense_block', name: 'Dense Block', icon: 'Layers', description: 'Stacked dense layers with concat', defaultParams: { num_layers: 6, in_channels: 64, growth_rate: 32, bn_size: 4, drop_rate: 0.0 }, category: 'Residual Blocks', tooltip: 'DenseNet block — all layers connected to all subsequent' },
  { id: 'transition_layer', type: 'transition_layer', name: 'Transition Layer', icon: 'Layers', description: 'BN → 1×1 Conv → AvgPool (DenseNet)', defaultParams: { in_channels: 256, out_channels: 128, stride: 2 }, category: 'Residual Blocks', tooltip: 'Reduces spatial dims and channels between dense blocks' },
  { id: 'inverted_bottleneck', type: 'inverted_bottleneck', name: 'Inverted Bottleneck', icon: 'Box', description: 'Expand → DWConv → Compress (MobileNet)', defaultParams: { in_channels: 32, out_channels: 64, expand_ratio: 6, stride: 1 }, category: 'Residual Blocks', hasActivation: true, tooltip: 'MobileNetV2 inverted residual block' },
  { id: 'convnext_block', type: 'convnext_block', name: 'ConvNeXt Block', icon: 'Box', description: 'DWConv7×7 → LN → FC → GELU → FC', defaultParams: { in_channels: 96, drop_path: 0.0, layer_scale_init: 1e-6 }, category: 'Residual Blocks', tooltip: 'Modernized ConvNet block with large kernel and GELU' },
  { id: 'mbconv_block', type: 'mbconv_block', name: 'MBConv Block', icon: 'Box', description: 'Mobile inverted bottleneck + SE', defaultParams: { in_channels: 32, out_channels: 64, expand_ratio: 4, stride: 1, se_ratio: 0.25, kernel_size: 3, activation: 'silu' }, category: 'Residual Blocks', hasActivation: true, tooltip: 'EfficientNet MBConv with squeeze-excitation' },
  { id: 'fused_mbconv', type: 'fused_mbconv', name: 'Fused-MBConv', icon: 'Box', description: 'Fused expand conv + SE', defaultParams: { in_channels: 32, out_channels: 64, expand_ratio: 4, stride: 1, se_ratio: 0.25 }, category: 'Residual Blocks', hasActivation: true, tooltip: 'EfficientNetV2 fused convolution variant' },

  // ── Attention Mechanisms ──
  { id: 'se_layer', type: 'se_layer', name: 'SE Layer', icon: 'Focus', description: 'Squeeze-and-Excitation channel attention', defaultParams: { in_channels: 256, reduction: 16 }, category: 'Channel Attention', tooltip: 'Global pool → FC → ReLU → FC → Sigmoid channel gate' },
  { id: 'cbam_layer', type: 'cbam_layer', name: 'CBAM Layer', icon: 'Focus', description: 'Channel + spatial attention', defaultParams: { in_channels: 256, reduction: 16, kernel_size: 7 }, category: 'Channel Attention', tooltip: 'Sequential channel and spatial attention modules' },
  { id: 'eca_layer', type: 'eca_layer', name: 'ECA Layer', icon: 'Focus', description: 'Efficient channel attention', defaultParams: { in_channels: 256, gamma: 2, b: 1 }, category: 'Channel Attention', tooltip: 'Adaptive 1D conv for channel attention — no FC' },
  { id: 'non_local_block', type: 'non_local_block', name: 'Non-Local Block', icon: 'Focus', description: 'Self-attention for spatial features', defaultParams: { in_channels: 256, inter_channels: 128, mode: 'embedded' }, category: 'Channel Attention', tooltip: 'Captures long-range spatial dependencies' },
  { id: 'gc_net', type: 'gc_net', name: 'GC-Net (Global Context)', icon: 'Focus', description: 'Simplified non-local + SE', defaultParams: { in_channels: 256, reduction: 16 }, category: 'Channel Attention', tooltip: 'Global context block combining non-local and SE ideas' },
  { id: 'self_attention_2d', type: 'self_attention_2d', name: 'Self-Attention 2D', icon: 'Focus', description: 'Pure spatial self-attention', defaultParams: { embed_dim: 256, num_heads: 8 }, category: 'Channel Attention', tooltip: 'Stand-alone self-attention replacing convolution' },

  // ── Vision Transformer Blocks ──
  { id: 'patch_embed', type: 'patch_embed', name: 'Patch Embed', icon: 'Grid3X3', description: 'Image → patch sequence', defaultParams: { img_size: 224, patch_size: 16, in_channels: 3, embed_dim: 768 }, category: 'ViT Blocks', tooltip: 'Splits image into non-overlapping patches and projects' },
  { id: 'class_token', type: 'class_token', name: 'Class Token', icon: 'Layers', description: 'Prepends learnable [CLS] token', defaultParams: { embed_dim: 768 }, category: 'ViT Blocks', tooltip: 'Learnable class token prepended to patch sequence' },
  { id: 'pos_absolute', type: 'pos_absolute', name: 'Positional Encoding', icon: 'Activity', description: 'Learnable or sinusoidal position', defaultParams: { num_patches: 196, embed_dim: 768, type: 'learned' }, category: 'ViT Blocks', tooltip: 'Adds positional information to patch embeddings' },
  { id: 'transformer_encoder_block', type: 'transformer_encoder_block', name: 'ViT Encoder Block', icon: 'Focus', description: 'MHA + FFN with LayerNorm', defaultParams: { embed_dim: 768, num_heads: 12, ffn_dim: 3072, dropout: 0.0, attention_dropout: 0.0 }, category: 'ViT Blocks', tooltip: 'Standard Vision Transformer encoder layer' },
  { id: 'cross_attention', type: 'cross_attention', name: 'Cross Attention', icon: 'Focus', description: 'Query-key-value cross attention', defaultParams: { embed_dim: 768, num_heads: 12, ffn_dim: 3072 }, category: 'ViT Blocks' },
  { id: 'cvt_block', type: 'cvt_block', name: 'CvT Block', icon: 'Focus', description: 'Convolutional vision transformer', defaultParams: { in_channels: 64, embed_dim: 256, num_heads: 4, stride: 2 }, category: 'ViT Blocks', tooltip: 'Convolutional token embedding + transformer' },
  { id: 'levit_block', type: 'levit_block', name: 'LeViT Block', icon: 'Focus', description: 'Efficient vision transformer', defaultParams: { in_channels: 128, out_channels: 256, num_heads: 4, stride: 2 }, category: 'ViT Blocks', tooltip: 'Lightweight efficient vision transformer block' },

  // ── Architecture Skeleton ──
  { id: 'stem_block', type: 'stem_block', name: 'Stem', icon: 'Box', description: 'Initial image processing', defaultParams: { type: 'conv', in_channels: 3, out_channels: 64, stride: 2, activation: 'relu' }, category: 'Skeleton', hasActivation: true, tooltip: 'First layers processing raw image input' },
  { id: 'stage_block', type: 'stage_block', name: 'Stage', icon: 'Layers', description: 'Repeating block stage', defaultParams: { num_blocks: 3, block_type: 'bottleneck', in_channels: 256, out_channels: 512, stride: 2 }, category: 'Skeleton', tooltip: 'One resolution stage with N repeated blocks' },
  { id: 'downsample_block', type: 'downsample_block', name: 'Downsample', icon: 'Grid3X3', description: 'Spatial downsampling', defaultParams: { in_channels: 64, out_channels: 128, stride: 2, method: 'conv' }, category: 'Skeleton', tooltip: 'Strided conv or pool for resolution reduction' },
  { id: 'upsample_block', type: 'upsample_block', name: 'Upsample', icon: 'Grid3X3', description: 'Spatial upsampling', defaultParams: { in_channels: 256, out_channels: 128, scale_factor: 2, method: 'interpolate' }, category: 'Skeleton', tooltip: 'ConvTranspose or interpolation for resolution increase' },
  { id: 'upsample', type: 'upsample', name: 'Interpolation', icon: 'Grid3X3', description: 'Bilinear/nearest upsample', defaultParams: { scale_factor: 2, mode: 'bilinear' }, category: 'Skeleton' },

  // ── Segmentation & Detection ──
  { id: 'unet_block', type: 'unet_block', name: 'U-Net Block', icon: 'Layers', description: 'U-Net encoder-decoder block', defaultParams: { in_channels: 64, out_channels: 128, skip_channels: 64, activation: 'relu' }, category: 'Seg/Det', hasActivation: true, tooltip: 'Decode block with skip connection from encoder' },
  { id: 'fpn_block', type: 'fpn_block', name: 'FPN Block', icon: 'Layers', description: 'Feature Pyramid Network', defaultParams: { in_channels_list: '256,512,1024,2048', out_channels: 256 }, category: 'Seg/Det', tooltip: 'Multi-scale feature pyramid for detection' },
  { id: 'anchor_generator', type: 'anchor_generator', name: 'Anchor Generator', icon: 'Box', description: 'Generate detection anchors', defaultParams: { sizes: '32,64,128,256,512', ratios: '0.5,1.0,2.0' }, category: 'Seg/Det', tooltip: 'Generates bounding box anchors at multiple scales' },
  { id: 'rpn_head', type: 'rpn_head', name: 'RPN Head', icon: 'ArrowRightFromLine', description: 'Region Proposal Network head', defaultParams: { in_channels: 256, num_anchors: 3 }, category: 'Seg/Det', tooltip: 'Proposes regions of interest for object detection' },
  { id: 'detection_head', type: 'detection_head', name: 'Detection Head', icon: 'ArrowRightFromLine', description: 'Classification + box regression', defaultParams: { in_channels: 256, num_classes: 80, num_anchors: 9 }, category: 'Seg/Det', tooltip: 'Final detection output: class + bbox' },
  { id: 'segmentation_head', type: 'segmentation_head', name: 'Segmentation Head', icon: 'ArrowRightFromLine', description: 'Per-pixel classification', defaultParams: { in_channels: 256, num_classes: 21, upsample_factor: 4 }, category: 'Seg/Det', tooltip: 'Dense pixel-wise segmentation output' },
  { id: 'mask_head', type: 'mask_head', name: 'Mask Head', icon: 'ArrowRightFromLine', description: 'Instance mask prediction', defaultParams: { in_channels: 256, num_classes: 80 }, category: 'Seg/Det', tooltip: 'Predicts binary masks per instance (Mask R-CNN)' },
  { id: 'roi_head', type: 'roi_head', name: 'ROI Head', icon: 'ArrowRightFromLine', description: 'ROI classification head', defaultParams: { in_channels: 256, num_classes: 80 }, category: 'Seg/Det', tooltip: 'Classifies each ROI proposal' },

  // ── Projection ──
  { id: 'linear_projection', type: 'linear_projection', name: 'Linear / FC', icon: 'Layers', description: 'Fully connected layer', defaultParams: { in_features: 2048, out_features: 1000, bias: true, activation: 'none' }, category: 'Projection', hasActivation: true },
  { id: 'embedding', type: 'embedding', name: 'Embedding', icon: 'Layers', description: 'Embedding lookup', defaultParams: { vocab_size: 50000, embed_dim: 768 }, category: 'Projection' },

  // ── Mixer & Future Blocks ──
  { id: 'mlp_mixer', type: 'mlp_mixer', name: 'MLP-Mixer', icon: 'Workflow', description: 'Token-mixing + channel-mixing MLP', defaultParams: { num_patches: 196, embed_dim: 512, token_mix_dim: 256, channel_mix_dim: 2048 }, category: 'Mixer', tooltip: 'Pure MLP architecture for vision — no convolution or attention' },
  { id: 'res_mlp', type: 'res_mlp', name: 'ResMLP', icon: 'Workflow', description: 'Residual MLP mixer', defaultParams: { embed_dim: 384, expansion_factor: 4 }, category: 'Mixer', tooltip: 'Residual connections in MLP mixing' },
  { id: 'conv_mixer', type: 'conv_mixer', name: 'ConvMixer', icon: 'Workflow', description: 'Depthwise conv + pointwise mixing', defaultParams: { in_channels: 768, kernel_size: 7, depth: 20 }, category: 'Mixer', tooltip: 'Simple architecture: patch embed + DWConv mixing' },
  { id: 'adaptive_inference_block', type: 'adaptive_inference_block', name: 'Adaptive Inference', icon: 'Zap', description: 'Early exit block', defaultParams: { input_size: 256, threshold: 0.9 }, category: 'Mixer', tooltip: 'Conditional computation — exits early when confident' },
  { id: 'nas_cell', type: 'nas_cell', name: 'NAS Cell', icon: 'Workflow', description: 'Neural architecture search cell', defaultParams: { operations: 'conv3x3,conv5x5,maxpool,skip' }, category: 'Mixer', tooltip: 'Searchable cell for architecture optimization' },

  // ── Structure ──
  { ...STRUCT_BLOCKS.residual_add, defaultParams: { mode: 'add' } },
  { ...STRUCT_BLOCKS.concat },
  { ...STRUCT_BLOCKS.flatten },
  { ...STRUCT_BLOCKS.reshape },
  { ...STRUCT_BLOCKS.dropout, defaultParams: { rate: 0.5 } },
  { id: 'skip_connection', type: 'skip_connection', name: 'Skip Connection', icon: 'GitBranch', description: 'Add or concat skip', defaultParams: { mode: 'add' }, category: 'Structure', tooltip: 'Add or concat bypass connection' },
  { id: 'permute', type: 'permute', name: 'Permute', icon: 'Layers', description: 'Reorder tensor dimensions', defaultParams: { dims: '0,2,1' }, category: 'Structure' },
  { id: 'stochastic_depth', type: 'stochastic_depth', name: 'Stochastic Depth', icon: 'Zap', description: 'Random layer dropping', defaultParams: { p: 0.1 }, category: 'Structure', tooltip: 'Randomly skips entire residual block during training' },
  { id: 'layer_stack', type: 'layer_stack', name: 'Layer Stack', icon: 'Layers', description: 'Repeat N identical blocks', defaultParams: { num_layers: 12, share_parameters: false }, category: 'Structure' },

  // ── Output Heads ──
  { id: 'classification_head', type: 'classification_head', name: 'Classification Head', icon: 'ArrowRightFromLine', description: 'Image classification output', defaultParams: { d_model: 2048, num_classes: 1000, pooling: 'avg' }, category: 'Output', tooltip: 'GlobalPool → FC → num_classes' },
];

// ─── DIFFUSION BLOCKS ─────────────────────────────────────

const diffusionBlocks: LayerConfig[] = [
  ...IO_BLOCKS,

  // ── 1. Encoders ──
  { id: 'vae_encoder', type: 'vae_encoder', name: 'VAE Encoder', icon: 'Layers', description: 'Variational autoencoder encoder (pixel → latent)', defaultParams: { in_channels: 3, latent_channels: 4, base_channels: 128, ch_mult: '1,2,4,4', num_res_blocks: 2, dropout: 0.0 }, category: 'Encoders', tooltip: 'Compresses image to latent space [B,4,H/8,W/8]' },
  { id: 'vae_decoder', type: 'vae_decoder', name: 'VAE Decoder', icon: 'Layers', description: 'Variational autoencoder decoder (latent → pixel)', defaultParams: { latent_channels: 4, out_channels: 3, base_channels: 128, ch_mult: '1,2,4,4', num_res_blocks: 2 }, category: 'Encoders', tooltip: 'Reconstructs image from latent space' },
  { id: 'image_encoder', type: 'image_encoder', name: 'Image Encoder', icon: 'Layers', description: 'Pixel-space image encoder with downsampling', defaultParams: { in_channels: 3, channels: 64, num_res_blocks: 2, downsample_steps: 3 }, category: 'Encoders', tooltip: 'Progressive downsampling for pixel-space diffusion' },
  { id: 'text_encoder', type: 'text_encoder', name: 'Text Encoder', icon: 'Layers', description: 'CLIP / T5 text encoder for conditioning', defaultParams: { model_type: 'clip', max_length: 77, output_dim: 768, freeze: true, layer: 'penultimate' }, category: 'Encoders', tooltip: 'Encodes text prompts to condition the denoiser' },
  { id: 'class_encoder', type: 'class_encoder', name: 'Class Encoder', icon: 'Layers', description: 'Class label → embedding', defaultParams: { num_classes: 1000, embed_dim: 768 }, category: 'Encoders', tooltip: 'Embeds discrete class labels for class-conditioned generation' },

  // ── 2. Denoisers ──
  { id: 'unet_2d_cond', type: 'unet_2d_cond', name: 'UNet2D Conditional', icon: 'Network', description: 'Full conditional U-Net denoiser (Stable Diffusion)', defaultParams: { sample_size: 64, in_channels: 4, out_channels: 4, layers_per_block: 2, down_block_types: 'CrossAttnDownBlock2D,CrossAttnDownBlock2D,CrossAttnDownBlock2D,DownBlock2D', up_block_types: 'UpBlock2D,CrossAttnUpBlock2D,CrossAttnUpBlock2D,CrossAttnUpBlock2D' }, category: 'Denoisers', tooltip: 'Primary denoiser for latent diffusion models' },
  { id: 'unet_model', type: 'unet_model', name: 'UNet Model', icon: 'Network', description: 'Configurable U-Net denoiser', defaultParams: { in_channels: 4, out_channels: 4, channels: 320, n_res_blocks: 2, attention_levels: '1,2,4', channel_multipliers: '1,2,4,4', n_heads: 8, tf_layers: 1, d_cond: 768 }, category: 'Denoisers', tooltip: 'Flexible U-Net with configurable attention levels' },
  { id: 'unet_block', type: 'unet_block', name: 'U-Net Block', icon: 'Layers', description: 'Preset-compatible U-Net block', defaultParams: { in_channels: 64, out_channels: 128, skip_channels: 64, activation: 'silu' }, category: 'Denoisers', hasActivation: true, tooltip: 'Shared U-Net block alias used by reference presets' },
  { id: 'dit_block', type: 'dit_block', name: 'DiT Block', icon: 'Sparkles', description: 'Diffusion Transformer block', defaultParams: { input_size: 32, patch_size: 2, in_channels: 4, embed_dim: 1152, depth: 28, num_heads: 16, mlp_ratio: 4.0, learn_sigma: true }, category: 'Denoisers', tooltip: 'Transformer-based denoiser replacing U-Net — PixArt, DiT' },
  { id: 'mmdit_block', type: 'mmdit_block', name: 'MMDiT Block', icon: 'Sparkles', description: 'Multimodal Diffusion Transformer (SD3/Flux)', defaultParams: { embed_dim: 3072, depth: 38, num_heads: 24, double_stream: true, modality_embed_dim: 768 }, category: 'Denoisers', tooltip: 'Joint image+text transformer for SD3 and Flux.1' },
  { id: 'flag_dit', type: 'flag_dit', name: 'Flag-DiT', icon: 'Sparkles', description: 'Flag-DiT with zero-init attention + RoPE', defaultParams: { embed_dim: 4096, depth: 32, num_heads: 32, zero_init_attention: true, rope: true }, category: 'Denoisers', tooltip: 'Lumina-T2I architecture with 3D RoPE' },
  { id: 'next_dit', type: 'next_dit', name: 'Next-DiT', icon: 'Sparkles', description: 'Next-generation DiT with sandwich norm', defaultParams: { embed_dim: 4096, depth: 32, num_heads: 32, rope_3d: true, sandwich_norm: true }, category: 'Denoisers', tooltip: 'Lumina-Next improved DiT with sandwich normalization' },

  // ── 3. U-Net Base Blocks ──
  { id: 'diff_resblock', type: 'diff_resblock', name: 'ResBlock (Diffusion)', icon: 'Box', description: 'Residual block with time conditioning', defaultParams: { in_channels: 256, out_channels: 256, d_time_emb: 1280, dropout: 0.0, activation: 'silu' }, category: 'U-Net Base', hasActivation: true, tooltip: 'Core U-Net residual block with timestep modulation' },
  { id: 'residual_block', type: 'residual_block', name: 'Residual Block', icon: 'Box', description: 'Preset-compatible diffusion resblock', defaultParams: { in_channels: 256, out_channels: 256, d_time_emb: 1280, dropout: 0.0, activation: 'silu' }, category: 'U-Net Base', hasActivation: true, tooltip: 'Alias used by reference diffusion presets' },
  { id: 'spatial_transformer', type: 'spatial_transformer', name: 'Spatial Transformer', icon: 'Focus', description: 'Attention over spatial features with optional cross-attn', defaultParams: { in_channels: 320, n_heads: 8, d_head: 40, depth: 1, context_dim: 768 }, category: 'U-Net Base', tooltip: 'Spatial self-attention + cross-attention for conditioning' },
  { id: 'basic_transformer_block', type: 'basic_transformer_block', name: 'Basic Transformer Block', icon: 'Focus', description: 'Self-attn → Cross-attn → FFN within SpatialTransformer', defaultParams: { dim: 320, n_heads: 8, d_head: 40, context_dim: 768, dropout: 0.0 }, category: 'U-Net Base', tooltip: 'Standard transformer block inside spatial transformer' },
  { id: 'downsample_2d', type: 'downsample_2d', name: 'Downsample2D', icon: 'Layers', description: 'Strided conv or pool downsample', defaultParams: { channels: 320, use_conv: true, stride: 2 }, category: 'U-Net Base', tooltip: 'Reduces spatial resolution by 2×' },
  { id: 'upsample_2d', type: 'upsample_2d', name: 'Upsample2D', icon: 'Layers', description: 'Interpolation + conv upsample', defaultParams: { channels: 320, use_conv: true, scale_factor: 2 }, category: 'U-Net Base', tooltip: 'Doubles spatial resolution' },
  { id: 'timestep_embed_seq', type: 'timestep_embed_seq', name: 'TimestepEmbedSequential', icon: 'Workflow', description: 'Wrapper for heterogeneous block sequences', defaultParams: {}, category: 'U-Net Base', tooltip: 'Passes timestep embedding through a sequence of blocks' },

  // ── 4. Specialized U-Net Blocks ──
  { id: 'cross_attn_down_block', type: 'cross_attn_down_block', name: 'CrossAttn DownBlock', icon: 'Layers', description: 'Down block with cross-attention layers', defaultParams: { in_channels: 320, out_channels: 640, n_res_blocks: 2, attn_num_heads: 8, dropout: 0.0 }, category: 'U-Net Blocks', tooltip: 'Encoder block with ResBlocks + SpatialTransformer + Downsample' },
  { id: 'cross_attn_up_block', type: 'cross_attn_up_block', name: 'CrossAttn UpBlock', icon: 'Layers', description: 'Up block with cross-attention layers', defaultParams: { in_channels: 640, out_channels: 320, prev_output_channels: 640, n_res_blocks: 2, attn_num_heads: 8 }, category: 'U-Net Blocks', tooltip: 'Decoder block with ResBlocks + SpatialTransformer + Upsample' },
  { id: 'down_block_2d', type: 'down_block_2d', name: 'DownBlock2D', icon: 'Layers', description: 'Simple down block (no attention)', defaultParams: { in_channels: 640, out_channels: 1280, n_res_blocks: 2, dropout: 0.0 }, category: 'U-Net Blocks', tooltip: 'Pure convolutional encoder block' },
  { id: 'up_block_2d', type: 'up_block_2d', name: 'UpBlock2D', icon: 'Layers', description: 'Simple up block (no attention)', defaultParams: { in_channels: 1280, out_channels: 640, prev_output_channels: 1280, n_res_blocks: 2 }, category: 'U-Net Blocks', tooltip: 'Pure convolutional decoder block' },
  { id: 'unet_mid_block', type: 'unet_mid_block', name: 'U-Net Mid Block', icon: 'Layers', description: 'Middle bottleneck block with cross-attention', defaultParams: { in_channels: 1280, n_res_blocks: 1, attn_num_heads: 8 }, category: 'U-Net Blocks', tooltip: 'Bottleneck between encoder and decoder' },

  // ── 5. Timestep Embedding ──
  { id: 'timestep_embedding', type: 'timestep_embedding', name: 'Timestep Embedding', icon: 'Clock', description: 'Maps timestep → dense embedding via SiLU MLP', defaultParams: { channels: 320, time_embed_dim: 1280, act_fn: 'silu' }, category: 'Conditioning', tooltip: 'Timestep → sinusoidal → MLP → time_embed_dim' },
  { id: 'timestep_projection', type: 'timestep_projection', name: 'Timestep Projection', icon: 'Clock', description: 'Projects time embedding to target dim', defaultParams: { time_embed_dim: 1280, out_dim: 512 }, category: 'Conditioning', tooltip: 'Linear projection of timestep embedding' },
  { id: 'sinusoidal_timestep_embed', type: 'sinusoidal_timestep_embed', name: 'Sinusoidal Timestep', icon: 'Clock', description: 'Raw sinusoidal timestep encoding', defaultParams: { channels: 320, max_period: 10000 }, category: 'Conditioning', tooltip: 'sin/cos encoding of scalar timestep' },

  // ── 6. Text Conditioning ──
  { id: 'cross_attention', type: 'cross_attention', name: 'Cross Attention', icon: 'Focus', description: 'Text-conditioned cross attention', defaultParams: { query_dim: 320, context_dim: 768, n_heads: 8, d_head: 40, dropout: 0.0 }, category: 'Conditioning', tooltip: 'Injects text conditioning into spatial features' },
  { id: 'text_projection', type: 'text_projection', name: 'Text Projection', icon: 'Layers', description: 'Projects text embeddings to model dim', defaultParams: { text_dim: 768, proj_dim: 320 }, category: 'Conditioning', tooltip: 'Aligns text encoder output to denoiser dimension' },
  { id: 'clip_embedding', type: 'clip_embedding', name: 'CLIP Embedding', icon: 'Layers', description: 'CLIP vocabulary + position embedding', defaultParams: { vocab_size: 49408, max_position_embeddings: 77, hidden_size: 768 }, category: 'Conditioning', tooltip: 'Token + position embeddings for CLIP text encoder' },

  // ── 7. Class Conditioning ──
  { id: 'class_embedding', type: 'class_embedding', name: 'Class Embedding', icon: 'Layers', description: 'Embeds class labels for conditional generation', defaultParams: { num_classes: 1000, embed_dim: 768 }, category: 'Conditioning', tooltip: 'Learned class embeddings added to timestep embedding' },
  { id: 'classifier_free_guidance', type: 'classifier_free_guidance', name: 'Classifier-Free Guidance', icon: 'Workflow', description: 'Balances conditioned/unconditioned outputs', defaultParams: { guidance_scale: 7.5, uncond_prob: 0.1 }, category: 'Conditioning', tooltip: 'out = uncond + scale × (cond - uncond)' },

  // ── 8. Image Conditioning ──
  { id: 'image_projection', type: 'image_projection', name: 'Image Projection', icon: 'Layers', description: 'Projects image features for conditioning', defaultParams: { in_channels: 3, out_channels: 320 }, category: 'Conditioning', tooltip: 'Image prompt projection for img2img / inpainting' },
  { id: 'controlnet_block', type: 'controlnet_block', name: 'ControlNet Block', icon: 'Network', description: 'Spatial conditioning via ControlNet', defaultParams: { in_channels: 320, conditioning_channels: 3, block_out_channels: '320,640,1280' }, category: 'Conditioning', tooltip: 'Adds structural control (edges, depth, pose) to generation' },
  { id: 'ip_adapter', type: 'ip_adapter', name: 'IP-Adapter', icon: 'Layers', description: 'Image prompt adapter via decoupled cross-attention', defaultParams: { image_embed_dim: 1024, cross_attention_dim: 768 }, category: 'Conditioning', tooltip: 'Decoupled cross-attention for image-prompted generation' },

  // ── 9. Schedulers ──
  { id: 'ddpm_scheduler', type: 'ddpm_scheduler', name: 'DDPM Scheduler', icon: 'Waves', description: 'Classic DDPM noise schedule', defaultParams: { num_train_timesteps: 1000, beta_start: 0.0001, beta_end: 0.02, beta_schedule: 'scaled_linear' }, category: 'Schedulers', tooltip: 'Denoising Diffusion Probabilistic Models scheduler' },
  { id: 'ddim_scheduler', type: 'ddim_scheduler', name: 'DDIM Scheduler', icon: 'Waves', description: 'Deterministic sampling in fewer steps', defaultParams: { num_train_timesteps: 1000, beta_schedule: 'scaled_linear', set_alpha_to_one: false }, category: 'Schedulers', tooltip: 'Non-Markovian sampling — 50 steps instead of 1000' },
  { id: 'euler_scheduler', type: 'euler_scheduler', name: 'Euler Scheduler', icon: 'Waves', description: 'Euler method ODE solver', defaultParams: { num_train_timesteps: 1000, beta_schedule: 'scaled_linear' }, category: 'Schedulers', tooltip: 'Simple and effective ODE-based sampler' },
  { id: 'dpm_solver', type: 'dpm_solver', name: 'DPM-Solver++', icon: 'Waves', description: 'Fast high-order ODE solver', defaultParams: { num_train_timesteps: 1000, beta_schedule: 'scaled_linear', algorithm_type: 'dpmsolver++' }, category: 'Schedulers', tooltip: 'Fast convergence in 20-25 steps — DPM-Solver++' },
  { id: 'flow_match_scheduler', type: 'flow_match_scheduler', name: 'Flow Match Scheduler', icon: 'Waves', description: 'Rectified flow / flow matching', defaultParams: { num_steps: 50, solver: 'euler' }, category: 'Schedulers', tooltip: 'For Flux.1 and rectified flow models' },
  { id: 'noise_schedule', type: 'noise_schedule', name: 'Noise Schedule', icon: 'Waves', description: 'Configurable noise schedule', defaultParams: { steps: 1000, schedule: 'cosine' }, category: 'Schedulers', tooltip: 'Controls noise intensity per timestep' },
  { id: 'noise_scheduler', type: 'noise_scheduler', name: 'Noise Scheduler', icon: 'Waves', description: 'Preset-compatible DDPM scheduler alias', defaultParams: { num_train_timesteps: 1000, beta_start: 0.0001, beta_end: 0.02, beta_schedule: 'scaled_linear' }, category: 'Schedulers', tooltip: 'Alias used by reference diffusion presets' },

  // ── 10. Noise Operations ──
  { id: 'gaussian_noise', type: 'gaussian_noise', name: 'Gaussian Noise', icon: 'Zap', description: 'Samples Gaussian noise ε ~ N(0,I)', defaultParams: { mean: 0, std: 1 }, category: 'Noise', tooltip: 'Pure noise source for forward diffusion' },
  { id: 'forward_diffusion', type: 'forward_diffusion', name: 'Forward Diffusion', icon: 'Workflow', description: 'Add noise: x₀ → xₜ', defaultParams: { scheduler: 'ddpm', timesteps: 1000 }, category: 'Noise', tooltip: 'q(xₜ|x₀) = √(ᾱₜ)x₀ + √(1-ᾱₜ)ε' },
  { id: 'reverse_diffusion', type: 'reverse_diffusion', name: 'Reverse Diffusion', icon: 'Workflow', description: 'Iterative denoising: xₜ → x₀', defaultParams: { model: 'unet', scheduler: 'ddpm', guidance_scale: 7.5 }, category: 'Noise', tooltip: 'Full reverse process from noise to image' },
  { id: 'latent_diffusion_step', type: 'latent_diffusion_step', name: 'Latent Diffusion Step', icon: 'Workflow', description: 'Single denoising step in latent space', defaultParams: { model: 'unet', scheduler: 'ddpm' }, category: 'Noise', tooltip: 'zₜ → z_{t-1} single step' },

  // ── 11. Diffusion Normalization ──
  { ...NORM_BLOCKS.groupnorm, defaultParams: { num_groups: 32, num_channels: 320, eps: 1e-5 } },
  { ...NORM_BLOCKS.layernorm, defaultParams: { normalized_shape: 320, eps: 1e-6, elementwise_affine: true } },
  { id: 'ada_group_norm', type: 'ada_group_norm', name: 'AdaGroupNorm', icon: 'AlignJustify', description: 'Adaptive GroupNorm with conditioning', defaultParams: { num_groups: 32, num_channels: 320, cond_dim: 1280 }, category: 'Normalization', tooltip: 'GroupNorm with scale/shift from conditioning signal' },
  { id: 'sandwich_norm', type: 'sandwich_norm', name: 'Sandwich Norm', icon: 'AlignJustify', description: 'Pre + post normalization (Lumina)', defaultParams: { dim: 4096, type: 'layernorm' }, category: 'Normalization', tooltip: 'Normalization before AND after attention/FFN' },

  // ── 12. SD / LDM Specific ──
  { id: 'autoencoder_kl', type: 'autoencoder_kl', name: 'AutoencoderKL', icon: 'Network', description: 'Full VAE (encoder + decoder) for latent diffusion', defaultParams: { embed_dim: 4, ch_mult: '1,2,4,4', num_res_blocks: 2 }, category: 'LDM', tooltip: 'Stable Diffusion VAE — encodes/decodes to/from 4-channel latent' },

  // ── 13. PixArt / DiT Specific ──
  { id: 'adaln_single', type: 'adaln_single', name: 'AdaLN-Single', icon: 'Layers', description: 'Single adaptive LayerNorm (PixArt-α)', defaultParams: { embed_dim: 1152, hidden_dim: 4608 }, category: 'DiT Blocks', tooltip: 'Efficient conditioning via single AdaLN shared across layers' },
  { id: 'patchify', type: 'patchify', name: 'Patchify', icon: 'Grid3X3', description: 'Latent → patch tokens for DiT', defaultParams: { patch_size: 2, in_channels: 4, embed_dim: 1152 }, category: 'DiT Blocks', tooltip: 'Splits latent into N = H×W/p² patches and projects to D' },
  { id: 'depatchify', type: 'depatchify', name: 'Depatchify', icon: 'Grid3X3', description: 'Patch tokens → spatial output', defaultParams: { patch_size: 2, out_channels: 8 }, category: 'DiT Blocks', tooltip: 'Reverses patchification to reconstruct spatial output' },

  // ── 14. SD3 / Flux Specific ──
  { id: 'double_stream_block', type: 'double_stream_block', name: 'Double Stream Block', icon: 'GitBranch', description: 'Parallel image + text streams (SD3/Flux)', defaultParams: { embed_dim: 3072, num_heads: 24, mlp_ratio: 4.0 }, category: 'SD3/Flux', tooltip: 'Independent attention per modality + cross-modality mixing' },
  { id: 'single_stream_block', type: 'single_stream_block', name: 'Single Stream Block', icon: 'Layers', description: 'Merged image+text stream (Flux late layers)', defaultParams: { embed_dim: 3072, num_heads: 24, mlp_ratio: 4.0 }, category: 'SD3/Flux', tooltip: 'Concatenated tokens processed jointly' },
  { id: 'rectified_flow', type: 'rectified_flow', name: 'Rectified Flow', icon: 'Workflow', description: 'Straight-line flow matching training', defaultParams: { num_steps: 50, solver: 'euler' }, category: 'SD3/Flux', tooltip: 'Linear interpolation training for fast inference' },

  // ── 15. Lumina Specific ──
  { id: 'rope_3d', type: 'rope_3d', name: '3D-RoPE', icon: 'Activity', description: '3D rotary positional encoding', defaultParams: { embed_dim: 4096, max_len: 16384, base: 10000 }, category: 'Lumina', tooltip: 'Positional encoding for variable-resolution images' },
  { id: 'freq_aware_rope', type: 'freq_aware_rope', name: 'Freq-Aware RoPE', icon: 'Activity', description: 'Frequency-scaled RoPE', defaultParams: { embed_dim: 4096, freq_scaling: 'linear' }, category: 'Lumina', tooltip: 'RoPE with per-frequency scaling for resolution generalization' },
  { id: 'time_aware_rope', type: 'time_aware_rope', name: 'Time-Aware RoPE', icon: 'Activity', description: 'Time-sensitive positional encoding', defaultParams: { embed_dim: 4096, time_scale: 1.0 }, category: 'Lumina', tooltip: 'Incorporates diffusion timestep into positional encoding' },
  { id: 'context_drop', type: 'context_drop', name: 'Context Drop', icon: 'Zap', description: 'Redundant token fusion/dropping', defaultParams: { drop_ratio: 0.3 }, category: 'Lumina', tooltip: 'Reduces sequence length by merging similar tokens' },

  // ── 16. Output / Post-Processing ──
  { id: 'diff_output_layer', type: 'diff_output_layer', name: 'Output Layer', icon: 'ArrowRightFromLine', description: 'Final projection to noise prediction', defaultParams: { in_channels: 320, out_channels: 4, use_norm: true }, category: 'Output', tooltip: 'GroupNorm → SiLU → Conv to predict ε or v' },
  { id: 'final_conv', type: 'final_conv', name: 'Final Conv', icon: 'ArrowRightFromLine', description: 'Final convolution output', defaultParams: { in_channels: 320, out_channels: 4, kernel_size: 3, padding: 1 }, category: 'Output', tooltip: 'Last convolution before output' },
  { id: 'image_decoder', type: 'image_decoder', name: 'Image Decoder (VAE)', icon: 'ArrowRightFromLine', description: 'Decodes latent to full-resolution image', defaultParams: { z_channels: 4, out_channels: 3 }, category: 'Output', tooltip: 'Latent [B,4,H/8,W/8] → image [B,3,H,W]' },

  // ── 17. Action-Conditioned ──
  { id: 'action_encoder', type: 'action_encoder', name: 'Action Encoder', icon: 'Layers', description: 'Encodes action vectors for world models', defaultParams: { action_dim: 7, embed_dim: 256 }, category: 'Action', tooltip: 'Projects robot/game actions to embedding space' },
  { id: 'action_cond_unet', type: 'action_cond_unet', name: 'Action-Conditioned UNet', icon: 'Network', description: 'U-Net conditioned on actions + time', defaultParams: { in_channels: 4, action_embed_dim: 256, time_embed_dim: 1280 }, category: 'Action', tooltip: 'Video prediction U-Net with action conditioning' },
  { id: 'dynamics_predictor', type: 'dynamics_predictor', name: 'Dynamics Predictor', icon: 'Workflow', description: 'Predicts next state from state + action', defaultParams: { state_dim: 512, action_dim: 7, hidden_dim: 256 }, category: 'Action', tooltip: 'Forward dynamics model for world simulation' },
  { id: 'world_model', type: 'world_model', name: 'World Model', icon: 'Network', description: 'Full world model (observation + action → next obs)', defaultParams: { observation_dim: 512, action_dim: 7, latent_dim: 256 }, category: 'Action', tooltip: 'Complete world model for planning and simulation' },

  // ── 18. Emerging / Future ──
  { id: 'reg_injector', type: 'reg_injector', name: 'REG Injector', icon: 'Zap', description: 'Representation entanglement injection', defaultParams: { semantic_dim: 768, inject_layer: 'mid' }, category: 'Emerging', tooltip: 'Injects semantic embeddings for improved generation' },
  { id: 'self_attention_guidance', type: 'self_attention_guidance', name: 'Self-Attn Guidance', icon: 'Focus', description: 'Guidance via self-attention maps', defaultParams: { guidance_scale: 3.0 }, category: 'Emerging', tooltip: 'Uses internal attention maps to guide generation quality' },
  { id: 'cascade_multiscale', type: 'cascade_multiscale', name: 'Cascade Diffusion', icon: 'Layers', description: 'Multi-scale cascaded diffusion', defaultParams: { num_scales: 3, scale_factors: '64,256,1024' }, category: 'Emerging', tooltip: 'Low-res → mid-res → high-res cascaded generation' },
  { id: 'patch_ddm', type: 'patch_ddm', name: 'Patch DDM', icon: 'Grid3X3', description: 'Patch-based diffusion for memory efficiency', defaultParams: { patch_size: 64, stride: 32 }, category: 'Emerging', tooltip: 'Processes image patches independently for large images' },

  // ── 19. Shared Blocks ──
  { id: 'conv2d', type: 'conv2d', name: 'Conv2D', icon: 'Grid3X3', description: '2D convolution', defaultParams: { in_channels: 320, out_channels: 320, kernel_size: 3, stride: 1, padding: 1 }, category: 'Convolution', hasActivation: true },
  { id: 'transposed_conv', type: 'transposed_conv', name: 'ConvTranspose2D', icon: 'Grid3X3', description: 'Transposed convolution for upsampling', defaultParams: { in_channels: 256, out_channels: 128, kernel_size: 4, stride: 2, padding: 1 }, category: 'Convolution', hasActivation: true },
  { id: 'attention_score', type: 'attention_score', name: 'Self Attention', icon: 'Focus', description: 'Self attention in denoiser', defaultParams: { n_heads: 8, d_head: 40 }, category: 'Attention' },
  { id: 'linear_projection', type: 'linear_projection', name: 'Linear Projection', icon: 'Layers', description: 'Dense linear transform', defaultParams: { in_features: 320, out_features: 320, bias: true, activation: 'none' }, category: 'Projection', hasActivation: true },

  // ── 20. Structure ──
  STRUCT_BLOCKS.residual_add,
  STRUCT_BLOCKS.concat,
  STRUCT_BLOCKS.dropout,
  { id: 'upsample', type: 'upsample', name: 'Interpolation', icon: 'Grid3X3', description: 'Preset-compatible upsample block', defaultParams: { scale_factor: 2, mode: 'bilinear' }, category: 'Structure' },
  { id: 'skip_connection', type: 'skip_connection', name: 'Skip Connection', icon: 'GitBranch', description: 'U-Net skip connection', defaultParams: { mode: 'concat' }, category: 'Structure', tooltip: 'Encoder-to-decoder bypass connection' },
  { id: 'layer_stack', type: 'layer_stack', name: 'Layer Stack', icon: 'Layers', description: 'Repeat N blocks', defaultParams: { num_layers: 28, share_parameters: false }, category: 'Structure' },
  { id: 'gradient_checkpoint', type: 'gradient_checkpoint', name: 'Gradient Checkpoint', icon: 'Box', description: 'Memory-compute tradeoff', defaultParams: { checkpoint_ratio: 1.0 }, category: 'Structure' },

  // ── 21. Legacy Compat ──
  { id: 'time_embedding', type: 'time_embedding', name: 'Time Embedding (Legacy)', icon: 'Clock', description: 'Sinusoidal timestep embedding', defaultParams: { dim: 256 }, category: 'Conditioning' },
  { id: 'unet_downsample', type: 'unet_downsample', name: 'U-Net Downsample (Legacy)', icon: 'Layers', description: 'Encoder downsample', defaultParams: { channels: 256 }, category: 'U-Net Base', hasActivation: true },
  { id: 'unet_upsample', type: 'unet_upsample', name: 'U-Net Upsample (Legacy)', icon: 'Layers', description: 'Decoder upsample', defaultParams: { channels: 256 }, category: 'U-Net Base', hasActivation: true },
];

// ─── GNN BLOCKS ───────────────────────────────────────────

const gnnBlocks: LayerConfig[] = [
  ...IO_BLOCKS,
  // 1. Message Passing (Convolution)
  { id: 'gcn_conv', type: 'gcn_conv', name: 'GCNConv', icon: 'Share2', description: 'Graph Convolutional Network layer', defaultParams: { in_channels: 16, out_channels: 32, improved: false, cached: false, add_self_loops: true, normalize: true, bias: true }, category: 'Message Passing', tooltip: 'Standard GCN layer with spectral normalization' },
  { id: 'sage_conv', type: 'sage_conv', name: 'SAGEConv', icon: 'Share2', description: 'GraphSAGE layer (Sample and AggreGatE)', defaultParams: { in_channels: 16, out_channels: 32, normalize: false, concat: false, root_weight: true, bias: true }, category: 'Message Passing', tooltip: 'Inductive aggregation from neighbors' },
  { id: 'gat_conv', type: 'gat_conv', name: 'GATConv', icon: 'Focus', description: 'Graph Attention Network layer', defaultParams: { in_channels: 16, out_channels: 8, heads: 8, concat: true, negative_slope: 0.2, dropout: 0.0, add_self_loops: true, bias: true }, category: 'Message Passing', tooltip: 'Multi-head attention over neighborhoods' },
  { id: 'gat_v2_conv', type: 'gat_v2_conv', name: 'GATv2Conv', icon: 'Focus', description: 'Improved Graph Attention Network layer', defaultParams: { in_channels: 16, out_channels: 8, heads: 8, concat: true, negative_slope: 0.2, dropout: 0.0, add_self_loops: true, bias: true }, category: 'Message Passing', tooltip: 'GAT variant with static attention fix' },
  { id: 'gin_conv', type: 'gin_conv', name: 'GINConv', icon: 'Share2', description: 'Graph Isomorphism Network layer', defaultParams: { eps: 0.0, train_eps: false }, category: 'Message Passing', tooltip: 'Maximally powerful GNN for graph isomorphism' },
  { id: 'cheb_conv', type: 'cheb_conv', name: 'ChebConv', icon: 'Share2', description: 'Chebyshev Spectral Graph Convolution', defaultParams: { in_channels: 16, out_channels: 32, K: 3, normalization: 'sym', bias: true }, category: 'Message Passing' },
  { id: 'mf_conv', type: 'mf_conv', name: 'MFConv', icon: 'Share2', description: 'Convolutional layer for Molecular Fingerprints', defaultParams: { in_channels: 16, out_channels: 32, max_degree: 10, bias: true }, category: 'Message Passing' },
  { id: 'rgcn_conv', type: 'rgcn_conv', name: 'RGCNConv', icon: 'Share2', description: 'Relational Graph Convolutional Network', defaultParams: { in_channels: 16, out_channels: 32, num_relations: 5, num_bases: 2, bias: true }, category: 'Message Passing' },
  { id: 'tag_conv', type: 'tag_conv', name: 'TAGConv', icon: 'Share2', description: 'Topology Adaptive Graph Convolutional Network', defaultParams: { in_channels: 16, out_channels: 32, K: 3, bias: true }, category: 'Message Passing' },
  { id: 'arma_conv', type: 'arma_conv', name: 'ARMAConv', icon: 'Share2', description: 'Auto-Regressive Moving Average Graph Conv', defaultParams: { in_channels: 16, out_channels: 32, num_stacks: 1, num_layers: 1, shared_weights: false, dropout: 0.0, bias: true }, category: 'Message Passing' },
  { id: 'sg_conv', type: 'sg_conv', name: 'SGConv', icon: 'Share2', description: 'Simplifying Graph Convolutional Network', defaultParams: { in_channels: 16, out_channels: 32, K: 2, cached: false, add_self_loops: true, bias: true }, category: 'Message Passing' },
  { id: 'appnp', type: 'appnp', name: 'Appnp', icon: 'Share2', description: 'Approximate Personalized PageRank of GCNs', defaultParams: { K: 10, alpha: 0.1, dropout: 0.0, cached: false, add_self_loops: true }, category: 'Message Passing' },
  { id: 'dna_conv', type: 'dna_conv', name: 'DNAConv', icon: 'Share2', description: 'Dynamic Neighborhood Aggregation', defaultParams: { in_channels: 16, heads: 8, groups: 1, dropout: 0.0, bias: true }, category: 'Message Passing' },
  { id: 'cluster_gcn_conv', type: 'cluster_gcn_conv', name: 'ClusterGCNConv', icon: 'Share2', description: 'Cluster-GCN efficient training layer', defaultParams: { in_channels: 16, out_channels: 32, diag_lambda: 0.0, bias: true }, category: 'Message Passing' },

  // 2. Edge Features
  { id: 'edge_conv', type: 'edge_conv', name: 'EdgeConv', icon: 'GitBranch', description: 'Dynamic Graph CNN for edge feature extraction', defaultParams: { nn: 'mlp', aggr: 'max' }, category: 'Edge Features' },
  { id: 'nn_conv', type: 'nn_conv', name: 'NNConv', icon: 'GitBranch', description: 'Graph convolution with edge feature transform', defaultParams: { in_channels: 16, out_channels: 32, nn: 'mlp', aggr: 'add', root_weight: true, bias: true }, category: 'Edge Features' },

  // 3. Pooling
  { id: 'topk_pooling', type: 'topk_pooling', name: 'TopKPooling', icon: 'AlignJustify', description: 'Pooling based on Top-K node selection', defaultParams: { in_channels: 16, ratio: 0.5, min_score: null, multiplier: 1.0, nonlinearity: 'tanh' }, category: 'Pooling' },
  { id: 'sag_pool', type: 'sag_pool', name: 'SAGPool', icon: 'Focus', description: 'Self-Attention Graph Pooling', defaultParams: { in_channels: 16, ratio: 0.5, GNN: 'GCN', min_score: null, multiplier: 1.0, nonlinearity: 'tanh' }, category: 'Pooling' },
  { id: 'edge_pooling', type: 'edge_pooling', name: 'EdgePooling', icon: 'GitBranch', description: 'Edge-based hierarchical pooling', defaultParams: { in_channels: 16, edge_score_method: 'mlp', dropout: 0.0 }, category: 'Pooling' },
  { id: 'asa_pooling', type: 'asa_pooling', name: 'ASAPooling', icon: 'AlignJustify', description: 'Adaptive Structure-Aware Pooling', defaultParams: { in_channels: 16, ratio: 0.5, dropout: 0.0 }, category: 'Pooling' },
  { id: 'global_max_pool', type: 'global_max_pool', name: 'GlobalMaxPool', icon: 'AlignJustify', description: 'Global maximum pooling', defaultParams: {}, category: 'Pooling' },
  { id: 'global_mean_pool', type: 'global_mean_pool', name: 'GlobalMeanPool', icon: 'AlignJustify', description: 'Global mean pooling', defaultParams: {}, category: 'Pooling' },
  { id: 'global_add_pool', type: 'global_add_pool', name: 'GlobalAddPool', icon: 'AlignJustify', description: 'Global sum pooling', defaultParams: {}, category: 'Pooling' },

  // 4. Normalization and Regularization
  { id: 'graph_norm', type: 'graph_norm', name: 'GraphNorm', icon: 'AlignCenter', description: 'Graph-specific normalization layer', defaultParams: { in_channels: 16, eps: 1e-5 }, category: 'Normalization' },
  { id: 'pair_norm', type: 'pair_norm', name: 'PairNorm', icon: 'AlignCenter', description: 'Normalization for mitigating over-smoothing', defaultParams: { scale: 1.0, scale_individually: false, eps: 1e-5 }, category: 'Normalization' },
  { id: 'mean_subtraction_norm', type: 'mean_subtraction_norm', name: 'MeanSubtractionNorm', icon: 'AlignCenter', description: 'Mean subtraction normalization', defaultParams: {}, category: 'Normalization' },
  { id: 'edge_dropout', type: 'edge_dropout', name: 'EdgeDropout', icon: 'Zap', description: 'Randomly dropping edges during training', defaultParams: { p: 0.5, force_undirected: false }, category: 'Regularization' },

  // 5. Heterogeneous Graphs
  { id: 'hetero_conv', type: 'hetero_conv', name: 'HeteroConv', icon: 'Share2', description: 'Convolutional layer for heterogeneous graphs', defaultParams: { convs: {}, aggr: 'sum' }, category: 'Heterogeneous' },
  { id: 'hgt_conv', type: 'hgt_conv', name: 'HGTConv', icon: 'Focus', description: 'Heterogeneous Graph Transformer', defaultParams: { in_channels: 16, out_channels: 32, heads: 8, num_types: 3, num_relations: 5 }, category: 'Heterogeneous' },
  { id: 'han_conv', type: 'han_conv', name: 'HANConv', icon: 'Focus', description: 'Heterogeneous Attention Network', defaultParams: { in_channels: 16, out_channels: 32, heads: 8, semantic_attention_heads: 4 }, category: 'Heterogeneous' },

  // 6. Temporal and Dynamic Graphs
  { id: 'tgcn', type: 'tgcn', name: 'TGCN', icon: 'Clock', description: 'Temporal Graph Convolutional Network', defaultParams: { in_channels: 16, out_channels: 32, improved: false, cached: false, add_self_loops: true }, category: 'Temporal' },
  { id: 'stgcn', type: 'stgcn', name: 'STGCN', icon: 'Clock', description: 'Spatio-Temporal Graph Convolutional Network', defaultParams: { in_channels: 16, out_channels: 32, kernel_size: 3, K: 3 }, category: 'Temporal' },
  { id: 'dy_gr_encoder', type: 'dy_gr_encoder', name: 'DyGrEncoder', icon: 'Clock', description: 'Encoding layer for dynamic graphs', defaultParams: { in_channels: 16, out_channels: 32, conv_type: 'GCN' }, category: 'Temporal' },

  // 7. Readout and Prediction Heads
  { id: 'global_attention_readout', type: 'global_attention_readout', name: 'GlobalAttentionReadout', icon: 'Focus', description: 'Attention-based global pooling', defaultParams: { gate_nn: 'mlp', nn: 'mlp' }, category: 'Readout' },
  { id: 'set2set_readout', type: 'set2set_readout', name: 'Set2SetReadout', icon: 'Layers', description: 'Set2Set aggregation mechanism', defaultParams: { in_channels: 16, processing_steps: 3, num_layers: 1 }, category: 'Readout' },
  { id: 'graph_readout_general', type: 'graph_readout_general', name: 'GraphReadout', icon: 'Layers', description: 'General graph-level output layer', defaultParams: { type: 'mean' }, category: 'Readout' },
  { id: 'linear_projection', type: 'linear_projection', name: 'Linear Projection', icon: 'Layers', description: 'Graph-level dense projection', defaultParams: { in_features: 64, out_features: 64, bias: true, activation: 'none' }, category: 'Readout', hasActivation: true },
  { id: 'classification_head', type: 'classification_head', name: 'Classification Head', icon: 'ArrowRightFromLine', description: 'Graph classification output', defaultParams: { d_model: 64, num_classes: 10, pooling: 'mean' }, category: 'Readout', tooltip: 'Graph-level classification head after readout pooling' },

  // 8. Positioning and Structural Encoding
  { id: 'random_walk_pe', type: 'random_walk_pe', name: 'RandomWalkPositionalEncoding', icon: 'Activity', description: 'Positional encoding from random walks', defaultParams: { walk_length: 20 }, category: 'Positioning' },
  { id: 'laplacian_pe', type: 'laplacian_pe', name: 'LaplacianPositionalEncoding', icon: 'Activity', description: 'Eigenvector-based structural encoding', defaultParams: { k: 10 }, category: 'Positioning' },
  { id: 'distance_encoding', type: 'distance_encoding', name: 'DistanceEncoding', icon: 'Activity', description: 'Encoding based on graph distances', defaultParams: { max_distance: 5 }, category: 'Positioning' },

  // 9. Graph Transformation
  { id: 'add_self_loops', type: 'add_self_loops', name: 'AddSelfLoops', icon: 'RotateCw', description: 'Adding self-loops to the graph', defaultParams: { fill_value: 1.0 }, category: 'Transformation' },
  { id: 't_normalize_features', type: 't_normalize_features', name: 'TNormalizeFeatures', icon: 'AlignCenter', description: 'Feature normalization', defaultParams: {}, category: 'Transformation' },
  { id: 't_to_undirected', type: 't_to_undirected', name: 'TToUndirected', icon: 'ArrowLeftRight', description: 'Converting directed graphs to undirected', defaultParams: {}, category: 'Transformation' },

  // 10. Advanced Architectures
  { id: 'mpnn_structure', type: 'mpnn_structure', name: 'MPNN', icon: 'MessageSquare', description: 'Message Passing Neural Network structure', defaultParams: { message_nn: 'mlp', update_nn: 'mlp', aggr: 'add' }, category: 'Advanced' },
  { id: 'deeper_gcn', type: 'deeper_gcn', name: 'DeeperGCN', icon: 'Layers', description: 'Architecture optimized for deep GNNs', defaultParams: { in_channels: 16, out_channels: 32, num_layers: 28, block: 'res+', norm: 'norm+', msg_norm: true }, category: 'Advanced' },

  // Shared blocks
  NORM_BLOCKS.layernorm,
  NORM_BLOCKS.batchnorm,
  STRUCT_BLOCKS.residual_add,
  STRUCT_BLOCKS.dropout,
  { ...IO_BLOCKS[0], defaultParams: { shape: '[N, F]' } },
  { ...IO_BLOCKS[1], defaultParams: { shape: '[N, C]' } },
];

// ─── GAN BLOCKS ───────────────────────────────────────────

const ganBlocks: LayerConfig[] = [
  ...IO_BLOCKS,

  // ── 1. Fundamental GAN ──
  { id: 'gan_noise_z', type: 'gan_noise_z', name: 'Latent Noise (z)', icon: 'Waves', description: 'Input latent vector z ~ N(0,I)', defaultParams: { dim: 512, distribution: 'gaussian' }, category: 'Fundamental GAN', color: 'hsl(210, 100%, 50%)', tooltip: 'The source of randomness for the Generator' },
  { id: 'gan_generator', type: 'gan_generator', name: 'Generator Base', icon: 'Wand2', description: 'Generic generator container', defaultParams: { architecture: 'dcgan', latent_dim: 100 }, category: 'Fundamental GAN', color: 'hsl(210, 100%, 50%)' },
  { id: 'gan_discriminator', type: 'gan_discriminator', name: 'Discriminator Base', icon: 'Shield', description: 'Generic discriminator container', defaultParams: { type: 'standard' }, category: 'Fundamental GAN', color: 'hsl(210, 100%, 50%)' },
  { id: 'gan_adversarial_loss', type: 'gan_adversarial_loss', name: 'Adversarial Loss', icon: 'Activity', description: 'Standard GAN binary cross-entropy loss', defaultParams: { label_smoothing: 0.0 }, category: 'Fundamental GAN', color: 'hsl(210, 100%, 50%)' },
  { id: 'gan_minimax_loss', type: 'gan_minimax_loss', name: 'Minimax Loss', icon: 'Activity', description: 'Original GAN minimax objective', defaultParams: {}, category: 'Fundamental GAN', color: 'hsl(210, 100%, 50%)' },
  { id: 'gan_non_saturating_loss', type: 'gan_non_saturating_loss', name: 'Non-Saturating Loss', icon: 'Activity', description: 'Standard loss with log(D(G(z))) trick', defaultParams: {}, category: 'Fundamental GAN', color: 'hsl(210, 100%, 50%)' },

  // ── 2. DCGAN & Conv-Base ──
  { id: 'dcgan_generator_block', type: 'dcgan_generator_block', name: 'DCGAN Gen Block', icon: 'Grid3X3', description: 'Transposed Conv + BN + ReLU', defaultParams: { out_channels: 256, kernel: 4, stride: 2, padding: 1, activation: 'relu' }, category: 'DCGAN & Conv-Base', color: 'hsl(280, 80%, 60%)', hasActivation: true },
  { id: 'dcgan_discriminator_block', type: 'dcgan_discriminator_block', name: 'DCGAN Disc Block', icon: 'Grid3X3', description: 'Conv + LeakyReLU + BN', defaultParams: { out_channels: 256, kernel: 4, stride: 2, padding: 1, activation: 'leaky_relu' }, category: 'DCGAN & Conv-Base', color: 'hsl(280, 80%, 60%)', hasActivation: true },
  { id: 'upsample_conv_block', type: 'upsample_conv_block', name: 'Upsample-Conv Block', icon: 'Maximize', description: 'Nearest/Bilinear Upsample then Conv', defaultParams: { scale: 2, mode: 'nearest', out_channels: 128 }, category: 'DCGAN & Conv-Base', color: 'hsl(280, 80%, 60%)', tooltip: 'Reduces checkerboard artifacts compared to transposed conv' },
  { id: 'pixel_shuffle', type: 'pixel_shuffle', name: 'Pixel Shuffle', icon: 'Shuffle', description: 'Sub-pixel convolution upsampling', defaultParams: { upscale_factor: 2 }, category: 'DCGAN & Conv-Base', color: 'hsl(280, 80%, 60%)', tooltip: 'Rearranges [B, C*r^2, H, W] to [B, C, H*r, W*r]' },
  { id: 'pixel_unshuffle', type: 'pixel_unshuffle', name: 'Pixel Unshuffle', icon: 'Shuffle', description: 'Inverse pixel shuffle downsampling', defaultParams: { downscale_factor: 2 }, category: 'DCGAN & Conv-Base', color: 'hsl(280, 80%, 60%)' },
  { id: 'checkerboard_removal', type: 'checkerboard_removal', name: 'Checkerboard Filter', icon: 'Filter', description: 'Post-upsampling artifact removal', defaultParams: { kernel_size: 3 }, category: 'DCGAN & Conv-Base', color: 'hsl(280, 80%, 60%)' },

  // ── 3. Conditional & InfoGAN ──
  { id: 'conditional_embedding', type: 'conditional_embedding', name: 'Class Conditioning', icon: 'Tag', description: 'Project class label to GAN latent', defaultParams: { num_classes: 10, embed_dim: 128 }, category: 'Conditional & InfoGAN', color: 'hsl(140, 80%, 45%)' },
  { id: 'label_conditioning', type: 'label_conditioning', name: 'Label Conditioning', icon: 'Fingerprint', description: 'Concatenate labels to input/features', defaultParams: { method: 'concat' }, category: 'Conditional & InfoGAN', color: 'hsl(140, 80%, 45%)' },
  { id: 'infogan_latent_code', type: 'infogan_latent_code', name: 'InfoGAN Code', icon: 'Code', description: 'Structured latent codes (c)', defaultParams: { continuous_dim: 2, discrete_dim: 10 }, category: 'Conditional & InfoGAN', color: 'hsl(140, 80%, 45%)', tooltip: 'Interpretable latent codes for disentangled reps' },
  { id: 'q_network_head', type: 'q_network_head', name: 'Q-Network Head', icon: 'Focus', description: 'Predicts latent codes from D features', defaultParams: { out_dim: 12 }, category: 'Conditional & InfoGAN', color: 'hsl(140, 80%, 45%)' },
  { id: 'mutual_info_loss', type: 'mutual_info_loss', name: 'Mutual Info Loss', icon: 'Activity', description: 'InfoGAN regularization loss', defaultParams: { lambda: 1.0 }, category: 'Conditional & InfoGAN', color: 'hsl(140, 80%, 45%)' },
  { id: 'auxiliary_classifier', type: 'auxiliary_classifier', name: 'AC-GAN Classifier', icon: 'ShieldCheck', description: 'Secondary classification head in D', defaultParams: { num_classes: 10 }, category: 'Conditional & InfoGAN', color: 'hsl(140, 80%, 45%)' },

  // ── 4. WGAN & LSGAN ──
  { id: 'wgan_critic', type: 'wgan_critic', name: 'WGAN Critic', icon: 'ShieldAlert', description: 'Discriminator without final sigmoid', defaultParams: {}, category: 'WGAN & LSGAN', color: 'hsl(10, 90%, 55%)', tooltip: 'Estimates Wasserstein distance' },
  { id: 'gradient_penalty', type: 'gradient_penalty', name: 'Gradient Penalty (GP)', icon: 'Zap', description: 'Gradient norm constraint (WGAN-GP)', defaultParams: { lambda: 10.0 }, category: 'WGAN & LSGAN', color: 'hsl(10, 90%, 55%)', tooltip: 'Enforces 1-Lipschitz continuity on Critic' },
  { id: 'lipschitz_constraint', type: 'lipschitz_constraint', name: 'Lipschitz Constraint', icon: 'Lock', description: 'General Lipschitz enforcement', defaultParams: { k: 1.0 }, category: 'WGAN & LSGAN', color: 'hsl(10, 90%, 55%)' },
  { id: 'weight_clipping', type: 'weight_clipping', name: 'Weight Clipping', icon: 'Scissors', description: 'Hard weight constraints for WGAN', defaultParams: { limit: 0.01 }, category: 'WGAN & LSGAN', color: 'hsl(10, 90%, 55%)' },
  { id: 'lsgan_loss', type: 'lsgan_loss', name: 'Least Squares Loss', icon: 'Activity', description: 'Mean Squared Error (LSGAN)', defaultParams: {}, category: 'WGAN & LSGAN', color: 'hsl(10, 90%, 55%)' },
  { id: 'hinge_loss', type: 'hinge_loss', name: 'Hinge Loss', icon: 'Activity', description: 'Hinge-based adversarial loss', defaultParams: {}, category: 'WGAN & LSGAN', color: 'hsl(10, 90%, 55%)', tooltip: 'Common in BigGAN and SAGAN' },
  { id: 'wasserstein_distance', type: 'wasserstein_distance', name: 'Wasserstein Dist', icon: 'Activity', description: 'EMD distance metric', defaultParams: {}, category: 'WGAN & LSGAN', color: 'hsl(10, 90%, 55%)' },

  // ── 5. Style & Modulation ──
  { id: 'mapping_network', type: 'mapping_network', name: 'Mapping Network', icon: 'Workflow', description: '8-layer MLP for style space (W)', defaultParams: { layers: 8, dim: 512 }, category: 'Style & Modulation', color: 'hsl(320, 90%, 60%)', tooltip: 'Transforms z to intermediate style space W' },
  { id: 'mapping_linear', type: 'mapping_linear', name: 'Style Linear', icon: 'Layers', description: 'Style-space projection layer', defaultParams: { in: 512, out: 512 }, category: 'Style & Modulation', color: 'hsl(320, 90%, 60%)' },
  { id: 'style_projection', type: 'style_projection', name: 'Style Projection (A)', icon: 'ArrowRight', description: 'Affine transform mapping W to s', defaultParams: { out_channels: 512 }, category: 'Style & Modulation', color: 'hsl(320, 90%, 60%)' },
  { id: 'noise_injection', type: 'noise_injection', name: 'Noise Injection (B)', icon: 'Zap', description: 'Per-pixel noise addition (StyleGAN)', defaultParams: { gain: 1.0 }, category: 'Style & Modulation', color: 'hsl(320, 90%, 60%)', tooltip: 'Adds stochastic detail to features' },
  { id: 'style_modulation', type: 'style_modulation', name: 'Style Modulation', icon: 'Wand2', description: 'Scales weights by style code', defaultParams: {}, category: 'Style & Modulation', color: 'hsl(320, 90%, 60%)' },
  { id: 'style_demodulation', type: 'style_demodulation', name: 'Style Demodulation', icon: 'Wand2', description: 'Normalizes weights after modulation', defaultParams: {}, category: 'Style & Modulation', color: 'hsl(320, 90%, 60%)' },
  { id: 'modulated_conv2d', type: 'modulated_conv2d', name: 'Modulated Conv2D', icon: 'Grid3X3', description: 'Full StyleGAN2 conv block', defaultParams: { out_channels: 512, kernel: 3, demodulate: true }, category: 'Style & Modulation', color: 'hsl(320, 90%, 60%)', tooltip: 'Modulation + Conv + Demodulation' },
  { id: 'adain_style', type: 'adain_style', name: 'AdaIN Style', icon: 'AlignJustify', description: 'Adaptive Instance Norm (StyleGAN1)', defaultParams: { epsilon: 1e-5 }, category: 'Style & Modulation', color: 'hsl(320, 90%, 60%)', tooltip: 'Applies scale/shift from style vector to normalized features' },

  // ── 6. StyleGAN / Synthesis ──
  { id: 'stylegan_synthesis_block', type: 'stylegan_synthesis_block', name: 'Synthesis Block', icon: 'Sparkles', description: 'StyleGAN2 resolution stage', defaultParams: { resolution: 1024, channels: 512 }, category: 'StyleGAN & Synthesis', color: 'hsl(45, 100%, 50%)' },
  { id: 'stylegan_res_block', type: 'stylegan_res_block', name: 'StyleGAN ResBlock', icon: 'Box', description: 'Residual block for StyleGAN Discriminator', defaultParams: { in_channels: 512, out_channels: 512 }, category: 'StyleGAN & Synthesis', color: 'hsl(45, 100%, 50%)' },
  { id: 'progressive_growing_step', type: 'progressive_growing_step', name: 'Progressive Growth', icon: 'TrendingUp', description: 'Resolution growth step (ProGAN)', defaultParams: { from_res: 4, to_res: 8 }, category: 'StyleGAN & Synthesis', color: 'hsl(45, 100%, 50%)' },
  { id: 'fading_layer', type: 'fading_layer', name: 'Fading / Alpha Blend', icon: 'Blend', description: 'Smooth resolution transition', defaultParams: { alpha: 0.5 }, category: 'StyleGAN & Synthesis', color: 'hsl(45, 100%, 50%)' },
  { id: 'torgb_layer', type: 'torgb_layer', name: 'ToRGB', icon: 'Image', description: 'Project features to RGB', defaultParams: { in_channels: 512 }, category: 'StyleGAN & Synthesis', color: 'hsl(45, 100%, 50%)' },
  { id: 'fromrgb_layer', type: 'fromrgb_layer', name: 'FromRGB', icon: 'Image', description: 'RGB project to features', defaultParams: { out_channels: 512 }, category: 'StyleGAN & Synthesis', color: 'hsl(45, 100%, 50%)' },
  { id: 'equalized_lr_linear', type: 'equalized_lr_linear', name: 'Equalized Linear', icon: 'Calculator', description: 'Linear with runtime scaling', defaultParams: {}, category: 'StyleGAN & Synthesis', color: 'hsl(45, 100%, 50%)' },
  { id: 'equalized_lr_conv', type: 'equalized_lr_conv', name: 'Equalized Conv', icon: 'Calculator', description: 'Conv with runtime scaling', defaultParams: {}, category: 'StyleGAN & Synthesis', color: 'hsl(45, 100%, 50%)' },
  { id: 'blur_filter', type: 'blur_filter', name: 'Blur (Low-Pass)', icon: 'Filter', description: 'Anti-aliasing blur kernel', defaultParams: { kernel: '[1, 2, 1]' }, category: 'StyleGAN & Synthesis', color: 'hsl(45, 100%, 50%)' },
  { id: 'antialias_upsample', type: 'antialias_upsample', name: 'Anti-alias Upsample', icon: 'Maximize', description: 'Upsample + Blur', defaultParams: { scale: 2 }, category: 'StyleGAN & Synthesis', color: 'hsl(45, 100%, 50%)' },
  { id: 'noise_broadcast', type: 'noise_broadcast', name: 'Noise Broadcast', icon: 'Zap', description: 'Broadcast noise to feature shape', defaultParams: {}, category: 'StyleGAN & Synthesis', color: 'hsl(45, 100%, 50%)' },

  // ── 7. High-Res & Attention ──
  { id: 'biggan_res_block', type: 'biggan_res_block', name: 'BigGAN ResBlock', icon: 'Box', description: 'Class-conditioned high-res resblock', defaultParams: { in_channels: 256, out_channels: 256, upsample: true }, category: 'Attention & BigGAN', color: 'hsl(190, 100%, 40%)' },
  { id: 'non_local_gan', type: 'non_local_gan', name: 'Non-Local Block', icon: 'Focus', description: 'Global context for GANs', defaultParams: { channels: 128 }, category: 'Attention & BigGAN', color: 'hsl(190, 100%, 40%)' },
  { id: 'self_attention_gan', type: 'self_attention_gan', name: 'Self-Attention GAN (SAGAN)', icon: 'Focus', description: 'Attention-based spatial dependencies', defaultParams: { channels: 128 }, category: 'Attention & BigGAN', color: 'hsl(190, 100%, 40%)' },
  { id: 'orthogonal_reg', type: 'orthogonal_reg', name: 'Orthogonal Regularization', icon: 'Lock', description: 'Constraint on weights orthogonality', defaultParams: { beta: 0.0001 }, category: 'Attention & BigGAN', color: 'hsl(190, 100%, 40%)' },
  { id: 'spectral_norm_wrapper', type: 'spectral_norm_wrapper', name: 'Spectral Norm', icon: 'Activity', description: 'Wrapper for spectral normalization', defaultParams: {}, category: 'Attention & BigGAN', color: 'hsl(190, 100%, 40%)' },
  { id: 'truncation_trick', type: 'truncation_trick', name: 'Truncation Trick', icon: 'Scissors', description: 'Clamping latent space sampling', defaultParams: { threshold: 0.7 }, category: 'Attention & BigGAN', color: 'hsl(190, 100%, 40%)', tooltip: 'Trades diversity for higher quality' },
  { id: 'shared_residual_block', type: 'shared_residual_block', name: 'Shared ResBlock', icon: 'Share2', description: 'Shared weights across resblocks', defaultParams: {}, category: 'Attention & BigGAN', color: 'hsl(190, 100%, 40%)' },

  // ── 8. Image Translation ──
  { id: 'cyclegan_block', type: 'cyclegan_block', name: 'CycleGAN ResBlock', icon: 'Box', description: 'ResNet block for domain transfer', defaultParams: { channels: 256 }, category: 'Cycle & Pix2Pix', color: 'hsl(260, 100%, 65%)' },
  { id: 'pix2pix_generator', type: 'pix2pix_generator', name: 'Pix2Pix U-Net', icon: 'Network', description: 'Encoder-decoder for translation', defaultParams: { skip: true }, category: 'Cycle & Pix2Pix', color: 'hsl(260, 100%, 65%)' },
  { id: 'patch_gan_discriminator', type: 'patch_gan_discriminator', name: 'PatchGAN', icon: 'Grid3X3', description: 'Markovian patches discriminator', defaultParams: { patch_size: 70 }, category: 'Cycle & Pix2Pix', color: 'hsl(260, 100%, 65%)', tooltip: 'Classifies NxN patches as real/fake' },
  { id: 'multiscale_discriminator', type: 'multiscale_discriminator', name: 'Multi-scale Disc', icon: 'Layers', description: 'Discriminator acting at multiple scales', defaultParams: { scales: 3 }, category: 'Cycle & Pix2Pix', color: 'hsl(260, 100%, 65%)' },
  { id: 'perceptual_loss', type: 'perceptual_loss', name: 'Perceptual Loss', icon: 'Activity', description: 'VGG-based feature matching loss', defaultParams: { lambda: 10.0 }, category: 'Cycle & Pix2Pix', color: 'hsl(260, 100%, 65%)' },
  { id: 'vgg_feature_extractor', type: 'vgg_feature_extractor', name: 'VGG Extractor', icon: 'Layers', description: 'Pre-trained VGG for losses', defaultParams: { layers: 'relu3_3,relu4_3' }, category: 'Cycle & Pix2Pix', color: 'hsl(260, 100%, 65%)' },
  { id: 'pixel_wise_loss', type: 'pixel_wise_loss', name: 'L1/L2 Reconstruction', icon: 'Activity', description: 'Pixel-level reconstruction loss', defaultParams: { type: 'l1' }, category: 'Cycle & Pix2Pix', color: 'hsl(260, 100%, 65%)' },
  { id: 'total_variation_loss', type: 'total_variation_loss', name: 'TV Loss', icon: 'Activity', description: 'Regularization for spatial smoothness', defaultParams: { weight: 1e-5 }, category: 'Cycle & Pix2Pix', color: 'hsl(260, 100%, 65%)' },
  { id: 'id_loss', type: 'id_loss', name: 'Identity Loss', icon: 'Activity', description: 'CycleGAN identity preservation', defaultParams: { lambda: 0.5 }, category: 'Cycle & Pix2Pix', color: 'hsl(260, 100%, 65%)' },

  // ── 9. Video & Special ──
  { id: 'video_generator_3d', type: 'video_generator_3d', name: '3D Conv Generator', icon: 'Video', description: 'Spatio-temporal generation', defaultParams: { channels: 64, frames: 16 }, category: 'Video & Temporal GAN', color: 'hsl(15, 100%, 50%)' },
  { id: 'temporal_discriminator', type: 'temporal_discriminator', name: 'Temporal Disc', icon: 'Shield', description: 'Discriminator for video dynamics', defaultParams: { kernel_size: 3 }, category: 'Video & Temporal GAN', color: 'hsl(15, 100%, 50%)' },
  { id: 'esrgan_dense_block', type: 'esrgan_dense_block', name: 'ESRGAN RRDB', icon: 'Box', description: 'Residual-in-Residual Dense Block', defaultParams: { channels: 64, growth_rate: 32 }, category: 'Special & Regularization', color: 'hsl(0, 0%, 50%)', tooltip: 'Core block for Super-Resolution GAN' },
  { id: 'super_res_block', type: 'super_res_block', name: 'Super-Res Block', icon: 'Maximize', description: 'Upsampling for ESRGAN/SRGAN', defaultParams: { scale: 4 }, category: 'Special & Regularization', color: 'hsl(0, 0%, 50%)' },
  { id: 'medical_gan_block', type: 'medical_gan_block', name: 'Medical GAN Block', icon: 'Stethoscope', description: 'Custom block for volumetric medical imaging', defaultParams: { dims: 3 }, category: 'Special & Regularization', color: 'hsl(0, 0%, 50%)' },
  { id: 'domain_adaptation_layer', type: 'domain_adaptation_layer', name: 'Domain Adaptation', icon: 'Globe', description: 'Alignment for domain-shift', defaultParams: {}, category: 'Special & Regularization', color: 'hsl(0, 0%, 50%)' },
  { id: 'r1_regularization', type: 'r1_regularization', name: 'R1 Gradient Penalty', icon: 'Zap', description: 'Regularization on discriminator gradients', defaultParams: { gamma: 10.0 }, category: 'Special & Regularization', color: 'hsl(0, 0%, 50%)' },
  { id: 'lazy_regularization', type: 'lazy_regularization', name: 'Lazy Regularization', icon: 'Clock', description: 'Regularization applied every K steps', defaultParams: { interval: 16 }, category: 'Special & Regularization', color: 'hsl(0, 0%, 50%)' },
  { id: 'r3gan_block', type: 'r3gan_block', name: 'R3GAN Block', icon: 'Zap', description: 'Recent high-efficiency GAN block', defaultParams: {}, category: 'Special & Regularization', color: 'hsl(0, 0%, 50%)' },
  { id: 'diffusion_gan_hybrid', type: 'diffusion_gan_hybrid', name: 'Diffusion-GAN Hybrid', icon: 'Sparkles', description: 'Combined sampling strategy', defaultParams: {}, category: 'Special & Regularization', color: 'hsl(280, 70%, 55%)' },

  ...IO_BLOCKS,
  { id: 'layernorm', type: 'layernorm', name: 'LayerNorm', icon: 'AlignCenter', description: 'Layer normalization', defaultParams: { eps: 1e-6 }, category: 'Normalization' },
  { id: 'batchnorm', type: 'batchnorm', name: 'BatchNorm', icon: 'AlignJustify', description: 'Batch normalization', defaultParams: { eps: 1e-5 }, category: 'Normalization' },
  { id: 'instancenorm', type: 'instancenorm', name: 'InstanceNorm', icon: 'AlignJustify', description: 'Instance normalization', defaultParams: { eps: 1e-5 }, category: 'Normalization' },
  { id: 'sync_batchnorm', type: 'sync_batchnorm', name: 'SyncBN', icon: 'AlignJustify', description: 'Synchronized BN', defaultParams: {}, category: 'Normalization' },
  { id: 'spectral_norm', type: 'spectral_norm', name: 'SpectralNorm', icon: 'Shield', description: 'Weight normalization', defaultParams: {}, category: 'Normalization' },
  { id: 'conv2d', type: 'conv2d', name: 'Conv2D', icon: 'Grid3X3', description: 'Generic convolution block', defaultParams: { in_channels: 64, out_channels: 64, kernel_size: 3, stride: 1, padding: 1 }, category: 'DCGAN & Conv-Base' },
  { id: 'linear_projection', type: 'linear_projection', name: 'Linear Projection', icon: 'Layers', description: 'Dense discriminator head', defaultParams: { in_features: 4096, out_features: 1, bias: true, activation: 'none' }, category: 'Projection', hasActivation: true },
  STRUCT_BLOCKS.residual_add,
  STRUCT_BLOCKS.concat,
  STRUCT_BLOCKS.dropout,
  STRUCT_BLOCKS.flatten,
  STRUCT_BLOCKS.reshape,
];

// ─── RL BLOCKS ────────────────────────────────────────────

const rlBlocks: LayerConfig[] = [
  ...IO_BLOCKS,
  { id: 'linear_projection', type: 'linear_projection', name: 'Linear / FC', icon: 'Layers', description: 'Dense feature layer', defaultParams: { in_features: 256, out_features: 256, bias: true, activation: 'none' }, category: 'Projection', hasActivation: true },
  { id: 'conv2d', type: 'conv2d', name: 'Conv2D', icon: 'Grid3X3', description: 'Visual feature extraction', defaultParams: { filters: 32, kernel: 8, stride: 4 }, category: 'Convolution', hasActivation: true },
  { id: 'policy_head', type: 'policy_head', name: 'Policy Head', icon: 'Route', description: 'Action probability distribution', defaultParams: { d_model: 256, action_dim: 18, distribution: 'categorical' }, category: 'RL Core', hasActivation: false, tooltip: 'Outputs π(a|s) action probabilities' },
  { id: 'value_head', type: 'value_head', name: 'Value Head', icon: 'Activity', description: 'State value estimation V(s)', defaultParams: { d_model: 256, hidden_dim: 256 }, category: 'RL Core', tooltip: 'Estimates expected return' },
  { id: 'advantage_stream', type: 'advantage_stream', name: 'Advantage Stream', icon: 'GitBranch', description: 'Dueling DQN advantage branch', defaultParams: { d_model: 256, action_dim: 18, hidden_dim: 256 }, category: 'RL Core', tooltip: 'A(s,a) = Q(s,a) - V(s)' },
  NORM_BLOCKS.layernorm,
  STRUCT_BLOCKS.residual_add,
  STRUCT_BLOCKS.flatten,
  STRUCT_BLOCKS.dropout,
];

// ─── SNN BLOCKS ───────────────────────────────────────────

const snnBlocks: LayerConfig[] = [
  ...IO_BLOCKS,
  { id: 'spike_encoder', type: 'spike_encoder', name: 'Spike Encoder', icon: 'Zap', description: 'Rate/temporal spike encoding', defaultParams: { encoding: 'rate', timesteps: 100 }, category: 'SNN Core', tooltip: 'Converts analog input to spike trains' },
  { id: 'rate_encoder', type: 'rate_encoder', name: 'Rate Encoder', icon: 'Zap', description: 'Preset-compatible rate encoder alias', defaultParams: { encoding: 'rate', timesteps: 100 }, category: 'SNN Core', tooltip: 'Alias used by reference SNN presets' },
  { id: 'lif_neuron', type: 'lif_neuron', name: 'LIF Neuron', icon: 'Brain', description: 'Leaky integrate-and-fire neuron', defaultParams: { tau: 10, threshold: 1.0, reset: 0.0 }, category: 'SNN Core', hasActivation: false, tooltip: 'Bio-inspired spiking neuron model' },
  { id: 'leaky_neuron', type: 'leaky_neuron', name: 'Leaky Neuron', icon: 'Brain', description: 'Leaky spiking neuron', defaultParams: { tau: 5.0, threshold: 0.9, reset: 0.0 }, category: 'SNN Core', tooltip: 'Preset-compatible leaky neuron alias' },
  { id: 'synaptic_layer', type: 'synaptic_layer', name: 'Synaptic Layer', icon: 'Layers', description: 'Delayed synaptic projection', defaultParams: { in_features: 256, out_features: 256, delay: 1 }, category: 'SNN Core', tooltip: 'Preset-compatible synaptic layer alias' },
  { id: 'stdp_synapse', type: 'stdp_synapse', name: 'STDP Synapse', icon: 'Network', description: 'Spike-timing dependent plasticity', defaultParams: { in_features: 256, out_features: 256, a_plus: 0.01, a_minus: 0.012, tau_plus: 20, tau_minus: 20 }, category: 'SNN Core', tooltip: 'Hebbian-like learning rule' },
  { id: 'linear_projection', type: 'linear_projection', name: 'Linear Projection', icon: 'Layers', description: 'Synaptic weight layer', defaultParams: { in_features: 128, out_features: 128, bias: true }, category: 'Projection' },
  { id: 'conv2d', type: 'conv2d', name: 'Conv2D', icon: 'Grid3X3', description: 'Convolutional spiking layer', defaultParams: { filters: 32, kernel: 3 }, category: 'Convolution' },
  NORM_BLOCKS.batchnorm,
  STRUCT_BLOCKS.residual_add,
  STRUCT_BLOCKS.dropout,
];

// ─── RNN BLOCKS ───────────────────────────────────────────

const rnnBlocks: LayerConfig[] = [
  ...IO_BLOCKS,
  { id: 'embedding', type: 'embedding', name: 'Embedding', icon: 'Layers', description: 'Token embedding', defaultParams: { vocab_size: 50000, d_model: 256, padding_idx: 0 }, category: 'Projection' },
  { id: 'linear_projection', type: 'linear_projection', name: 'Linear / FC', icon: 'Layers', description: 'Dense transform', defaultParams: { in_features: 256, out_features: 256, bias: true, activation: 'none' }, category: 'Projection', hasActivation: true },
  { id: 'conv1d', type: 'conv1d', name: 'Conv1D', icon: 'Grid3X3', description: 'Temporal 1D convolution for QRNN-style blocks', defaultParams: { in_channels: 768, out_channels: 768, kernel_size: 4, groups: 768 }, category: 'Projection', tooltip: 'Shared 1D convolution block used by QRNN-style macros' },
  { id: 'mha_attention', type: 'mha_attention', name: 'Multi-Head Attention', icon: 'Focus', description: 'Internal attention for SRU++-style recurrent hybrids', defaultParams: { n_heads: 12, head_dim: 64, d_model: 768, dropout: 0.0, bias: false, output_projection: true }, category: 'Projection', tooltip: 'Attention side-branch used by SRU++-style recurrent blocks' },
  { id: 'lstm', type: 'lstm', name: 'LSTM', icon: 'Repeat', description: 'Preset-compatible stacked LSTM block', defaultParams: { input_size: 256, hidden_size: 512, num_layers: 1, batch_first: true, dropout: 0.0 }, category: 'Recurrent Cells', tooltip: 'Alias used by reference RNN presets' },
  { id: 'gru', type: 'gru', name: 'GRU', icon: 'Repeat', description: 'Preset-compatible stacked GRU block', defaultParams: { input_size: 256, hidden_size: 512, num_layers: 1, batch_first: true, dropout: 0.0 }, category: 'Recurrent Cells', tooltip: 'Alias used by reference RNN presets' },
  { id: 'bilstm', type: 'bilstm', name: 'BiLSTM', icon: 'Repeat', description: 'Preset-compatible bidirectional LSTM block', defaultParams: { input_size: 256, hidden_size: 256, num_layers: 2, batch_first: true, dropout: 0.2, bidirectional: true }, category: 'Recurrent Cells', tooltip: 'Alias used by reference RNN presets' },
  { id: 'bigru', type: 'bigru', name: 'BiGRU', icon: 'Repeat', description: 'Preset-compatible bidirectional GRU block', defaultParams: { input_size: 256, hidden_size: 256, num_layers: 2, batch_first: true, dropout: 0.2, bidirectional: true }, category: 'Recurrent Cells', tooltip: 'Alias used by reference RNN presets' },
  // Fundamental cells
  { id: 'rnn_cell', type: 'rnn_cell', name: 'RNN Cell', icon: 'Repeat', description: 'Vanilla recurrent cell', defaultParams: { hiddenSize: 256, numLayers: 1, activation: 'tanh' }, category: 'Recurrent Cells', tooltip: 'Simple Elman RNN' },
  { id: 'elman_rnn_cell', type: 'elman_rnn_cell', name: 'Elman RNN Cell', icon: 'Repeat', description: 'Classic Elman recurrent cell', defaultParams: { inputSize: 256, hiddenSize: 256, nonlinearity: 'tanh', bias: true }, category: 'Recurrent Cells' },
  { id: 'jordan_rnn_cell', type: 'jordan_rnn_cell', name: 'Jordan RNN Cell', icon: 'Repeat', description: 'Jordan variant (output feedback)', defaultParams: { inputSize: 256, hiddenSize: 256, nonlinearity: 'tanh', bias: true }, category: 'Recurrent Cells' },
  { id: 'lstm_cell', type: 'lstm_cell', name: 'LSTM Cell', icon: 'Repeat', description: 'Long Short-Term Memory', defaultParams: { hiddenSize: 512, numLayers: 2, bidirectional: false }, category: 'Recurrent Cells', tooltip: 'Gated memory with forget/input/output gates' },
  { id: 'lstm_peephole_cell', type: 'lstm_peephole_cell', name: 'LSTM Peephole', icon: 'Repeat', description: 'LSTM with peephole connections', defaultParams: { inputSize: 256, hiddenSize: 512, bias: true, peephole: true }, category: 'Recurrent Cells' },
  { id: 'lstm_coupled_cell', type: 'lstm_coupled_cell', name: 'LSTM Coupled Gates (CIFG)', icon: 'Repeat', description: 'Coupled input/forget gates', defaultParams: { inputSize: 256, hiddenSize: 512, bias: true }, category: 'Recurrent Cells' },
  { id: 'lstm_projection_cell', type: 'lstm_projection_cell', name: 'LSTM with Projection (LSTMP)', icon: 'Repeat', description: 'LSTM + linear projection', defaultParams: { inputSize: 256, hiddenSize: 512, projSize: 256, bias: true }, category: 'Recurrent Cells' },
  { id: 'gru_cell', type: 'gru_cell', name: 'GRU Cell', icon: 'Repeat', description: 'Gated Recurrent Unit', defaultParams: { hiddenSize: 512, numLayers: 2 }, category: 'Recurrent Cells', tooltip: 'Simplified LSTM with update/reset gates' },
  { id: 'mgu_cell', type: 'mgu_cell', name: 'MGU Cell', icon: 'Repeat', description: 'Minimal Gated Unit', defaultParams: { inputSize: 256, hiddenSize: 512, bias: true }, category: 'Recurrent Cells' },
  { id: 'ugrnn_cell', type: 'ugrnn_cell', name: 'UGRNN Cell', icon: 'Repeat', description: 'Update Gate RNN', defaultParams: { inputSize: 256, hiddenSize: 512, bias: true }, category: 'Recurrent Cells' },
  { id: 'sru_cell', type: 'sru_cell', name: 'SRU Cell', icon: 'Repeat', description: 'Simple Recurrent Unit (parallelizable)', defaultParams: { inputSize: 256, hiddenSize: 512, bias: true, highway: false }, category: 'Recurrent Cells' },
  { id: 'srupp_cell', type: 'srupp_cell', name: 'SRU++ Cell', icon: 'Repeat', description: 'SRU with internal attention (SRU++)', defaultParams: { inputSize: 256, hiddenSize: 512, bias: true, numHeads: 4 }, category: 'Recurrent Cells' },
  { id: 'indrnn_cell', type: 'indrnn_cell', name: 'IndRNN Cell', icon: 'Repeat', description: 'Independently recurrent neurons', defaultParams: { inputSize: 256, hiddenSize: 512, nonlinearity: 'relu', bias: true }, category: 'Recurrent Cells' },
  { id: 'indylstm_cell', type: 'indylstm_cell', name: 'IndyLSTM Cell', icon: 'Repeat', description: 'Independently parameterized LSTM', defaultParams: { inputSize: 256, hiddenSize: 512, bias: true }, category: 'Recurrent Cells' },
  { id: 'phased_lstm_cell', type: 'phased_lstm_cell', name: 'Phased LSTM Cell', icon: 'Repeat', description: 'Time-gated LSTM for irregular sampling', defaultParams: { inputSize: 256, hiddenSize: 512, period: 1.0, tau: 0.1, rOn: 0.05 }, category: 'Recurrent Cells' },
  { id: 'qrnn_cell', type: 'qrnn_cell', name: 'QRNN Cell', icon: 'Repeat', description: 'Quasi-RNN (conv + minimal recurrence)', defaultParams: { inputSize: 256, hiddenSize: 512, kernelSize: 2, convLayers: 1 }, category: 'Recurrent Cells' },
  { id: 'lmu_cell', type: 'lmu_cell', name: 'Legendre Memory Unit (LMU)', icon: 'Repeat', description: 'Orthogonal memory using Legendre polynomials', defaultParams: { inputSize: 256, hiddenSize: 512, order: 10, theta: 100.0 }, category: 'Recurrent Cells' },
  { id: 'fast_weights_rnn_cell', type: 'fast_weights_rnn_cell', name: 'Fast Weights RNN', icon: 'Repeat', description: 'RNN with fast associative weights', defaultParams: { inputSize: 256, hiddenSize: 512, slowUpdate: 0.1, fastDecay: 0.95 }, category: 'Recurrent Cells' },
  { id: 'hebbian_rnn_cell', type: 'hebbian_rnn_cell', name: 'Hebbian RNN', icon: 'Repeat', description: 'Local Hebbian plasticity in recurrence', defaultParams: { inputSize: 256, hiddenSize: 512, learningRateHebb: 0.01 }, category: 'Recurrent Cells' },

  // Wrappers
  { id: 'bidirectional_wrapper', type: 'bidirectional_wrapper', name: 'Bidirectional Wrapper', icon: 'Repeat', description: 'Make a cell bidirectional', defaultParams: { mergeMode: 'concat' }, category: 'Wrappers' },
  { id: 'stacked_rnn', type: 'stacked_rnn', name: 'Stacked RNN', icon: 'Repeat', description: 'Stack multiple recurrent layers', defaultParams: { numLayers: 2, cellType: 'lstm_cell', dropout: 0.0 }, category: 'Wrappers' },
  { id: 'residual_rnn_layer', type: 'residual_rnn_layer', name: 'Residual RNN Layer', icon: 'GitBranch', description: 'Residual connection around a recurrent layer', defaultParams: { projection: false }, category: 'Wrappers' },
  { id: 'dropout_rnn_layer', type: 'dropout_rnn_layer', name: 'Dropout RNN Layer', icon: 'Box', description: 'Dropout on inputs/hidden states', defaultParams: { dropout: 0.1, recurrentDropout: 0.0 }, category: 'Wrappers' },
  { id: 'zoneout_wrapper', type: 'zoneout_wrapper', name: 'Zoneout Wrapper', icon: 'Box', description: 'Zoneout regularization for RNN states', defaultParams: { zoneoutRate: 0.1 }, category: 'Wrappers' },

  // Specialized architectures
  { id: 'conv_lstm_cell', type: 'conv_lstm_cell', name: 'ConvLSTM Cell', icon: 'Grid3X3', description: 'Convolutional LSTM (spatio-temporal)', defaultParams: { inChannels: 3, hiddenChannels: 64, kernelSize: 3, padding: 1 }, category: 'Specialized RNN' },
  { id: 'conv_gru_cell', type: 'conv_gru_cell', name: 'ConvGRU Cell', icon: 'Grid3X3', description: 'Convolutional GRU (spatio-temporal)', defaultParams: { inChannels: 3, hiddenChannels: 64, kernelSize: 3, padding: 1 }, category: 'Specialized RNN' },
  { id: 'stlstm_cell', type: 'stlstm_cell', name: 'ST-LSTM Cell', icon: 'Grid3X3', description: 'Spatio-Temporal LSTM (dual state)', defaultParams: { inChannels: 3, hiddenChannels: 64, kernelSize: 3 }, category: 'Specialized RNN' },
  { id: 'grid_lstm', type: 'grid_lstm', name: 'GridLSTM', icon: 'Grid3X3', description: 'Multi-dimensional LSTM', defaultParams: { dims: 2, hiddenSizes: [128, 128] }, category: 'Specialized RNN' },
  { id: 'tree_lstm', type: 'tree_lstm', name: 'TreeLSTM', icon: 'Share2', description: 'LSTM over tree structures', defaultParams: { hiddenSize: 256, numChildren: 2, mode: 'child_sum' }, category: 'Specialized RNN' },
  { id: 'graph_lstm', type: 'graph_lstm', name: 'GraphLSTM', icon: 'Share2', description: 'LSTM over graphs', defaultParams: { hiddenSize: 256, edgeTypes: 1 }, category: 'Specialized RNN' },
  { id: 'depth_gated_lstm', type: 'depth_gated_lstm', name: 'Depth-Gated LSTM', icon: 'Repeat', description: 'LSTM with depth gating', defaultParams: { inputSize: 256, hiddenSize: 512, numLayers: 2 }, category: 'Specialized RNN' },
  { id: 'gated_feedback_lstm', type: 'gated_feedback_lstm', name: 'Gated Feedback LSTM', icon: 'Repeat', description: 'LSTM with inter-layer feedback', defaultParams: { inputSize: 256, hiddenSize: 512, numLayers: 2 }, category: 'Specialized RNN' },

  // Seq2Seq
  { id: 'many_to_one_encoder', type: 'many_to_one_encoder', name: 'Many-to-One Encoder', icon: 'ArrowRightToLine', description: 'Sequence → single vector', defaultParams: { pooling: 'last' }, category: 'Seq2Seq' },
  { id: 'one_to_many_decoder', type: 'one_to_many_decoder', name: 'One-to-Many Decoder', icon: 'ArrowRightFromLine', description: 'Vector → sequence', defaultParams: { outputLength: 32, teacherForcing: 0.0 }, category: 'Seq2Seq' },
  { id: 'many_to_many_encoder_decoder', type: 'many_to_many_encoder_decoder', name: 'Encoder–Decoder (No Attention)', icon: 'Route', description: 'Seq2Seq without attention', defaultParams: { context: 'final_state' }, category: 'Seq2Seq' },
  { id: 'seq2seq_attention', type: 'seq2seq_attention', name: 'Seq2Seq Attention', icon: 'Focus', description: 'Encoder–decoder with attention', defaultParams: { attentionType: 'bahdanau' }, category: 'Seq2Seq' },
  { id: 'siamese_rnn', type: 'siamese_rnn', name: 'Siamese RNN', icon: 'Users', description: 'Two RNN towers with shared weights', defaultParams: { distanceMetric: 'cosine' }, category: 'Seq2Seq' },

  // Internal gates (visual building blocks)
  { id: 'lstm_input_gate', type: 'lstm_input_gate', name: 'LSTM Input Gate', icon: 'GitBranch', description: 'i_t gate', defaultParams: { bias: true }, category: 'Gates' },
  { id: 'lstm_forget_gate', type: 'lstm_forget_gate', name: 'LSTM Forget Gate', icon: 'GitBranch', description: 'f_t gate', defaultParams: { bias: true }, category: 'Gates' },
  { id: 'lstm_output_gate', type: 'lstm_output_gate', name: 'LSTM Output Gate', icon: 'GitBranch', description: 'o_t gate', defaultParams: { bias: true }, category: 'Gates' },
  { id: 'lstm_cell_modulation', type: 'lstm_cell_modulation', name: 'LSTM Cell Modulation', icon: 'GitBranch', description: 'g_t candidate modulation', defaultParams: { bias: true }, category: 'Gates' },
  { id: 'gru_reset_gate', type: 'gru_reset_gate', name: 'GRU Reset Gate', icon: 'GitBranch', description: 'r_t gate', defaultParams: { bias: true }, category: 'Gates' },
  { id: 'gru_update_gate', type: 'gru_update_gate', name: 'GRU Update Gate', icon: 'GitBranch', description: 'z_t gate', defaultParams: { bias: true }, category: 'Gates' },
  { id: 'gru_candidate_activation', type: 'gru_candidate_activation', name: 'GRU Candidate', icon: 'GitBranch', description: 'h~_t candidate activation', defaultParams: { bias: true }, category: 'Gates' },

  // Normalization & regularization
  { id: 'layernorm_rnn', type: 'layernorm_rnn', name: 'LayerNorm (RNN)', icon: 'AlignCenter', description: 'LayerNorm for recurrent hidden states', defaultParams: { normalizedShape: 512, eps: 1e-6, affine: true }, category: 'Normalization' },
  { id: 'weightnorm_rnn', type: 'weightnorm_rnn', name: 'WeightNorm (RNN)', icon: 'Shield', description: 'Weight normalization for recurrent weights', defaultParams: { gain: 1.0 }, category: 'Normalization' },
  { id: 'recurrent_batchnorm', type: 'recurrent_batchnorm', name: 'Recurrent BatchNorm', icon: 'AlignJustify', description: 'BatchNorm adapted for recurrent networks', defaultParams: { numFeatures: 512, momentum: 0.1, position: 'pre' }, category: 'Normalization' },
  { id: 'variational_dropout_rnn', type: 'variational_dropout_rnn', name: 'Variational Dropout (RNN)', icon: 'Box', description: 'Dropout mask constant across time', defaultParams: { dropout: 0.1, recurrentDropout: 0.0 }, category: 'Regularization' },
  { id: 'zoneout', type: 'zoneout', name: 'Zoneout', icon: 'Box', description: 'Zoneout state regularization', defaultParams: { zoneoutRate: 0.1 }, category: 'Regularization' },

  // Future concepts
  { id: 'memory_augmented_rnn', type: 'memory_augmented_rnn', name: 'Memory Augmented RNN', icon: 'MessageSquare', description: 'External memory (NTM/DNC style)', defaultParams: { memorySize: 128, memoryDim: 64, numReadHeads: 1, numWriteHeads: 1 }, category: 'Future Concepts' },
  { id: 'act_rnn', type: 'act_rnn', name: 'ACT RNN', icon: 'Activity', description: 'Adaptive Computation Time', defaultParams: { ponderCost: 0.01, threshold: 0.99 }, category: 'Future Concepts' },
  { id: 'rim_block', type: 'rim_block', name: 'RIM Block', icon: 'Network', description: 'Recurrent Independent Mechanisms', defaultParams: { numMechanisms: 4, topK: 2, hiddenSize: 256 }, category: 'Future Concepts' },
  { id: 'neural_turing_machine', type: 'neural_turing_machine', name: 'Neural Turing Machine', icon: 'Network', description: 'NTM core controller + memory', defaultParams: { memorySize: 128, memoryDim: 64 }, category: 'Future Concepts' },

  NORM_BLOCKS.layernorm,
  STRUCT_BLOCKS.residual_add,
  STRUCT_BLOCKS.dropout,
];

// ─── EXPERIMENTAL BLOCKS ──────────────────────────────────

const experimentalBlocks: LayerConfig[] = [
  ...transformerBlocks,
  ...mambaBlocks.filter(l => !IO_BLOCKS.some(io => io.id === l.id) && !transformerBlocks.some(t => t.id === l.id)),
];

// ─── PLUGIN REGISTRY ─────────────────────────────────────

export const PLUGINS: Record<ArchitectureFamily, ArchitecturePlugin> = {
  transformer: {
    id: 'transformer',
    config: { id: 'transformer', name: 'Transformer / LLM', description: 'Attention-based sequence models', icon: 'Sparkles', color: 'hsl(199, 89%, 48%)' },
    layers: transformerBlocks,
    tools: [
      { id: 'attention-viz', name: 'Attention Heads', icon: 'Focus', description: 'Visualize attention patterns', component: 'AttentionHeadsPanel' },
      { id: 'token-flow', name: 'Token Flow', icon: 'ArrowRight', description: 'Token → Embedding → Attention', component: 'TokenFlowPanel' },
      { id: 'param-estimation', name: 'Parameters', icon: 'Calculator', description: 'Parameter & FLOPs estimation', component: 'ParameterPanel' },
    ],
    metrics: [
      { id: 'attention-heads', name: 'Attention Heads', value: 12, status: 'normal' },
      { id: 'hidden-dim', name: 'Hidden Dimension', value: 768, status: 'normal' },
      { id: 'context-length', name: 'Context Length', value: '2048', status: 'normal' },
    ],
    visualizations: [
      { id: 'attention-map', name: 'Attention Map', type: 'heatmap' },
      { id: 'token-embedding', name: 'Token Embeddings', type: 'chart' },
    ],
  },
  moe: {
    id: 'moe',
    config: { id: 'moe', name: 'Mixture of Experts', description: 'Sparse expert routing models', icon: 'Network', color: 'hsl(280, 70%, 55%)' },
    layers: moeBlocks,
    tools: [
      { id: 'routing-viz', name: 'Expert Routing', icon: 'Route', description: 'Token-to-expert routing', component: 'ExpertRoutingPanel' },
      { id: 'load-balance', name: 'Load Balancing', icon: 'Scale', description: 'Expert utilization balance', component: 'LoadBalancePanel' },
    ],
    metrics: [
      { id: 'num-experts', name: 'Total Experts', value: 8, status: 'normal' },
      { id: 'top-k', name: 'Top-K Routing', value: 2, status: 'normal' },
    ],
    visualizations: [
      { id: 'routing-heatmap', name: 'Routing Heatmap', type: 'heatmap' },
      { id: 'expert-load', name: 'Expert Load', type: 'chart' },
    ],
  },
  ssm: {
    id: 'ssm',
    config: { id: 'ssm', name: 'State Space Models', description: 'S4, Mamba, H3, and linear recurrence', icon: 'Workflow', color: 'hsl(160, 70%, 45%)' },
    layers: [...ssmBlocks, ...mambaBlocks.filter(l => !ssmBlocks.some(s => s.id === l.id))],
    tools: [
      { id: 'state-transition', name: 'State Transitions', icon: 'Workflow', description: 'State transition diagrams', component: 'StateTransitionPanel' },
      { id: 'stability', name: 'Stability', icon: 'Shield', description: 'Numerical stability', component: 'StabilityPanel' },
    ],
    metrics: [
      { id: 'state-size', name: 'State Size', value: 64, status: 'normal' },
      { id: 'seq-length', name: 'Sequence Length', value: '128K', status: 'normal' },
    ],
    visualizations: [
      { id: 'state-diagram', name: 'State Diagram', type: 'diagram' },
    ],
  },
  cnn: {
    id: 'cnn',
    config: { id: 'cnn', name: 'CNN / Vision', description: 'Convolutional neural networks', icon: 'Grid3X3', color: 'hsl(210, 70%, 50%)' },
    layers: cnnBlocks,
    tools: [
      { id: 'feature-maps', name: 'Feature Maps', icon: 'Grid3X3', description: 'Visualize conv features', component: 'FeatureMapsPanel' },
      { id: 'receptive-field', name: 'Receptive Field', icon: 'Target', description: 'Receptive field size', component: 'ReceptiveFieldPanel' },
    ],
    metrics: [
      { id: 'conv-layers', name: 'Conv Layers', value: 5, status: 'normal' },
      { id: 'receptive-field', name: 'Receptive Field', value: '32×32', status: 'normal' },
    ],
    visualizations: [
      { id: 'feature-viz', name: 'Feature Viz', type: 'heatmap' },
    ],
  },
  diffusion: {
    id: 'diffusion',
    config: { id: 'diffusion', name: 'Diffusion Models', description: 'Denoising and score-based generative', icon: 'Waves', color: 'hsl(38, 92%, 50%)' },
    layers: diffusionBlocks,
    tools: [
      { id: 'noise-schedule', name: 'Noise Schedule', icon: 'Waves', description: 'Noise schedule visualization', component: 'NoiseSchedulePanel' },
      { id: 'sampling', name: 'Sampling Pipeline', icon: 'PlayCircle', description: 'Generation pipeline view', component: 'SamplingPanel' },
    ],
    metrics: [
      { id: 'diffusion-steps', name: 'Diffusion Steps', value: 1000, status: 'normal' },
      { id: 'channels', name: 'Base Channels', value: 256, status: 'normal' },
    ],
    visualizations: [
      { id: 'noise-viz', name: 'Noise Schedule', type: 'chart' },
    ],
  },
  gnn: {
    id: 'gnn',
    config: { id: 'gnn', name: 'Graph Neural Networks', description: 'Node, edge, and graph learning', icon: 'Share2', color: 'hsl(340, 75%, 55%)' },
    layers: gnnBlocks,
    tools: [
      { id: 'graph-input', name: 'Graph Structure', icon: 'Share2', description: 'Node & edge config', component: 'GraphInputPanel' },
      { id: 'message-pass', name: 'Message Passing', icon: 'MessageSquare', description: 'Message flow', component: 'MessagePassPanel' },
    ],
    metrics: [
      { id: 'num-nodes', name: 'Nodes', value: '10K', status: 'normal' },
      { id: 'avg-degree', name: 'Avg Degree', value: '5.0', status: 'normal' },
    ],
    visualizations: [
      { id: 'graph-viz', name: 'Graph Viz', type: 'graph' },
    ],
  },
  gan: {
    id: 'gan',
    config: { id: 'gan', name: 'GANs', description: 'Generative adversarial networks', icon: 'Wand2', color: 'hsl(45, 90%, 50%)' },
    layers: ganBlocks,
    tools: [
      { id: 'gen-disc', name: 'Gen ↔ Disc', icon: 'ArrowLeftRight', description: 'Generator-Discriminator flow', component: 'GenDiscPanel' },
      { id: 'mode-collapse', name: 'Mode Collapse', icon: 'AlertTriangle', description: 'Mode collapse indicators', component: 'ModeCollapsePanel' },
    ],
    metrics: [
      { id: 'gen-loss', name: 'Generator Loss', value: '0.82', status: 'normal' },
      { id: 'disc-loss', name: 'Discriminator Loss', value: '0.45', status: 'warning' },
    ],
    visualizations: [
      { id: 'loss-chart', name: 'Loss Curves', type: 'chart' },
    ],
  },
  rl: {
    id: 'rl',
    config: { id: 'rl', name: 'Reinforcement Learning', description: 'Actor–Critic and policy gradient', icon: 'Gamepad2', color: 'hsl(15, 80%, 52%)' },
    layers: rlBlocks,
    tools: [
      { id: 'policy-viz', name: 'Policy Network', icon: 'Route', description: 'Action distribution', component: 'PolicyVizPanel' },
      { id: 'value-viz', name: 'Value Network', icon: 'Activity', description: 'Value estimation', component: 'ValueVizPanel' },
    ],
    metrics: [
      { id: 'num-actions', name: 'Action Space', value: 18, status: 'normal' },
      { id: 'architecture', name: 'Type', value: 'Actor-Critic', status: 'normal' },
    ],
    visualizations: [
      { id: 'reward-chart', name: 'Reward Curve', type: 'chart' },
    ],
  },
  snn: {
    id: 'snn',
    config: { id: 'snn', name: 'Spiking Neural Networks', description: 'Bio-inspired spike-based models', icon: 'Brain', color: 'hsl(300, 60%, 50%)' },
    layers: snnBlocks,
    tools: [
      { id: 'spike-viz', name: 'Spike Raster', icon: 'Zap', description: 'Spike timing visualization', component: 'SpikeRasterPanel' },
      { id: 'membrane-viz', name: 'Membrane Potential', icon: 'Activity', description: 'Neuron membrane dynamics', component: 'MembranePotentialPanel' },
    ],
    metrics: [
      { id: 'timesteps', name: 'Timesteps', value: 100, status: 'normal' },
      { id: 'threshold', name: 'Threshold', value: '1.0', status: 'normal' },
    ],
    visualizations: [
      { id: 'raster-plot', name: 'Raster Plot', type: 'chart' },
    ],
  },
  rnn: {
    id: 'rnn',
    config: { id: 'rnn', name: 'RNN / LSTM / GRU', description: 'Recurrent sequence models', icon: 'Repeat', color: 'hsl(25, 80%, 52%)' },
    layers: rnnBlocks,
    tools: [
      { id: 'gate-analysis', name: 'Gate Analysis', icon: 'Repeat', description: 'LSTM/GRU gate visualization', component: 'GateAnalysisPanel' },
      { id: 'gradient-flow', name: 'Gradient Flow', icon: 'TrendingUp', description: 'Vanishing gradient analysis', component: 'GradientFlowPanel' },
    ],
    metrics: [
      { id: 'hidden-size', name: 'Hidden Size', value: 512, status: 'normal' },
      { id: 'num-layers', name: 'Layers', value: 3, status: 'normal' },
    ],
    visualizations: [
      { id: 'gate-viz', name: 'Gate Visualization', type: 'heatmap' },
    ],
  },
  experimental: {
    id: 'experimental',
    config: { id: 'experimental', name: 'Experimental', description: 'Early access R&D models', icon: 'FlaskConical', color: 'hsl(280, 70%, 55%)' },
    layers: experimentalBlocks,
    tools: [
      { id: 'neural-ode', name: 'Neural ODE', icon: 'Sigma', description: 'Continuous-depth networks', component: 'NeuralODEPanel' },
    ],
    metrics: [
      { id: 'research-status', name: 'Research Status', value: 'Active', status: 'warning' },
    ],
    visualizations: [
      { id: 'research-flow', name: 'Research Flow', type: 'flow' },
    ],
  },
};

export function getPlugin(family: ArchitectureFamily): ArchitecturePlugin | undefined {
  return PLUGINS[family];
}

export function getPluginLayers(family: ArchitectureFamily): LayerConfig[] {
  return PLUGINS[family]?.layers ?? [];
}

export function getPluginTools(family: ArchitectureFamily): PluginTool[] {
  return PLUGINS[family]?.tools ?? [];
}

export function getPluginMetrics(family: ArchitectureFamily): PluginMetric[] {
  return PLUGINS[family]?.metrics ?? [];
}
