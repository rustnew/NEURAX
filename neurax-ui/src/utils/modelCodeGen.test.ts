/**
 * The code this module generates must be a real reflection of the design —
 * not merely code that parses. Two things are checked for every supported
 * layer type: the parameter count matches a hand-derived expected value
 * (the same formula `neurax-formulas` uses), and the generated `forward()`
 * actually calls the layer rather than passing the tensor through untouched
 * — the exact defect (`x2 = x1`) that got the previous code-generation
 * attempt removed from `GitHubExportPanel`.
 */
import { describe, it, expect } from 'vitest';
import { CanvasNode, Connection, ParameterValue } from '@/types/architecture.ts';
import { HardwareConfig } from '@/contexts/HardwareContext.tsx';
import {
  generateModelCode,
  generateModelCodeVerified,
  verifyCodegenAgainstAnalysis,
  compileUntilVerified,
  isCodegenSupported,
} from './modelCodeGen.ts';

const BASE_HW: HardwareConfig = {
  hardware: 'a100', precision: 'fp16', batchSize: 1,
  learningRate: 1e-4, numEpochs: 1, gpuCount: 1, gpuMemoryGb: 80,
  datasetSize: 1000, seqLen: 128, vocabSize: 32000, hiddenDim: 512,
  numHeads: 8, ffnDim: 2048, numLayers: 4, useBias: false, dropout: 0.0,
  useFlash: true, imgHeight: 224, imgWidth: 224, inChannels: 3,
  numDenoisingSteps: 0, guidanceScale: 0, numNodes: 0, numEdges: 0,
  nodeFeatDim: 0, dState: 16, dtRank: 32, timesteps: 0, spikeRate: 0,
  numExperts: 8, topK: 2, actionDim: 0, stateDim: 0,
};

function chain(nodes: CanvasNode[]): Connection[] {
  const conns: Connection[] = [];
  for (let i = 0; i < nodes.length - 1; i++) {
    conns.push({ id: `c${i}`, from: nodes[i].id, to: nodes[i + 1].id });
  }
  return conns;
}

function node(id: string, type: CanvasNode['type'], params: Record<string, ParameterValue> = {}): CanvasNode {
  return { id, type, name: id, x: 0, y: 0, params };
}

describe('generateModelCode — no silent identity pass-through', () => {
  it('never emits a forward body that just reassigns the tensor for a supported layer', () => {
    const nodes = [
      node('n1', 'token_embedding', { vocab_size: 1000, hidden_size: 64 }),
      node('n2', 'mha_attention', { hidden_size: 64, num_heads: 4 }),
      node('n3', 'ffn_gated', { hidden_size: 64, intermediate_size: 256 }),
      node('n4', 'layernorm', { hidden_size: 64 }),
    ];
    const result = generateModelCode(nodes, chain(nodes), BASE_HW, 'TinyModel');
    expect(result.fullySupported).toBe(true);
    for (const layer of result.layers) {
      const forward = layer.forwardLines.join('\n');
      // The historical bug: `x2 = x1` — a bare reassignment with no call.
      expect(forward).not.toMatch(/^x\s*=\s*x\s*$/);
      expect(forward).toMatch(new RegExp(`self\\.${layer.varName}\\(`));
    }
  });

  it('makes an unsupported layer fail loudly instead of passing through', () => {
    // A type the palette does not offer at all — which is the whole point:
    // `quantum_circuit` used to sit in `LayerType` as a declared-but-
    // unreachable name, so this test asserted the unsupported path using a
    // type the union claimed was valid. `as any` now says plainly that this
    // is not a block NEURAX has.
    const nodes = [node('n1', 'quantum_circuit' as any, {})];
    const result = generateModelCode(nodes, [], BASE_HW, 'Weird');
    expect(result.fullySupported).toBe(false);
    expect(result.unsupportedTypes).toContain('quantum_circuit');
    const layer = result.layers[0];
    expect(layer.supported).toBe(false);
    expect(layer.paramCount).toBe(0);
    expect(layer.forwardLines.join('\n')).toMatch(/raise NotImplementedError/);
    expect(isCodegenSupported('quantum_circuit' as any)).toBe(false);
  });
});

describe('generateModelCode — parameter counts match neurax-formulas', () => {
  it('embedding: vocab_size * hidden_size', () => {
    const n = [node('n1', 'token_embedding', { vocab_size: 30522, hidden_size: 768 })];
    const result = generateModelCode(n, [], BASE_HW, 'Embed');
    expect(result.totalParams).toBe(30522 * 768);
  });

  it('MHA attention: 4*H*H (no bias) — qkv (3H^2) + out (H^2)', () => {
    const n = [node('n1', 'mha_attention', { hidden_size: 768, num_heads: 12 })];
    const result = generateModelCode(n, [], { ...BASE_HW, useBias: false }, 'Attn');
    expect(result.totalParams).toBe(4 * 768 * 768);
  });

  it('MHA attention with bias: adds 4*H for Q,K,V,Out biases', () => {
    const n = [node('n1', 'mha_attention', { hidden_size: 768, num_heads: 12, bias: true })];
    const result = generateModelCode(n, [], BASE_HW, 'Attn');
    expect(result.totalParams).toBe(4 * 768 * 768 + 4 * 768);
  });

  it('GQA attention: matches gqa_params from neurax-formulas', () => {
    // hidden=4096, heads=32, kv_heads=8 (LLaMA-2-7B shape), no bias
    const hidden = 4096, heads = 32, kvHeads = 8;
    const headDim = hidden / heads;
    const q = hidden * hidden;
    const kvDim = kvHeads * headDim;
    const kv = hidden * kvDim * 2;
    const out = hidden * hidden;
    const expected = q + kv + out;

    const n = [node('n1', 'gqa_attention', { hidden_size: hidden, num_heads: heads, num_kv_heads: kvHeads })];
    const result = generateModelCode(n, [], { ...BASE_HW, useBias: false }, 'GQA');
    expect(result.totalParams).toBe(expected);
  });

  it('ffn_gated (SwiGLU): 3*H*I, no bias', () => {
    const n = [node('n1', 'ffn_gated', { hidden_size: 4096, intermediate_size: 11008 })];
    const result = generateModelCode(n, [], { ...BASE_HW, useBias: false }, 'FFN');
    expect(result.totalParams).toBe(3 * 4096 * 11008);
  });

  it('ffn_standard: 2*H*I, no bias', () => {
    const n = [node('n1', 'ffn_standard', { hidden_size: 768, intermediate_size: 3072 })];
    const result = generateModelCode(n, [], { ...BASE_HW, useBias: false }, 'FFN');
    expect(result.totalParams).toBe(2 * 768 * 3072);
  });

  it('layernorm: 2*H (weight + bias)', () => {
    const n = [node('n1', 'layernorm', { hidden_size: 768 })];
    const result = generateModelCode(n, [], BASE_HW, 'LN');
    expect(result.totalParams).toBe(2 * 768);
  });

  it('rmsnorm: H (weight only)', () => {
    const n = [node('n1', 'rmsnorm', { hidden_size: 768 })];
    const result = generateModelCode(n, [], BASE_HW, 'RMS');
    expect(result.totalParams).toBe(768);
  });

  it('mamba_block: matches mamba_params from neurax-formulas (2.8B shape)', () => {
    const hidden = 2560, stateDim = 16, expand = 2;
    const dInner = hidden * expand;
    const inProj = hidden * (dInner * 2);
    const conv1d = dInner * 4;
    const ssm = dInner * stateDim * 3 + dInner;
    const outProj = dInner * hidden;
    const expected = inProj + conv1d + ssm + outProj;

    const n = [node('n1', 'mamba_block', { hidden_size: hidden, d_state: stateDim, expand_factor: expand })];
    const result = generateModelCode(n, [], BASE_HW, 'Mamba');
    expect(result.totalParams).toBe(expected);
  });

  it('bottleneck_block stage: matches resnet_bottleneck_block_params summed over repeats', () => {
    // ResNet-50 stage 1: in=64 -> mid=64 -> out=256, 3 blocks, stride 1
    const n = [node('n1', 'bottleneck_block', { planes: 256, blocks: 3, stride: 1, expansion: 4 })];
    const hw = { ...BASE_HW, inChannels: 64 };
    const result = generateModelCode(n, [], hw, 'Stage1');

    // Hand-computed via the same conv/BN formula as neurax-formulas:
    const convP = (i: number, o: number, k: number) => i * o * k * k;
    let expected = 0;
    let curIn = 64;
    for (let i = 0; i < 3; i++) {
      const mid = 64, out = 256;
      const c1 = convP(curIn, mid, 1);
      const c2 = convP(mid, mid, 3);
      const c3 = convP(mid, out, 1);
      const bn = 3 * 2 * mid + 2 * out; // 3 BN at mid + 1 BN at out, trainable-only (2/channel)
      const downsample = (curIn !== out) ? convP(curIn, out, 1) + 2 * out : 0;
      expected += c1 + c2 + c3 + bn + downsample;
      curIn = out;
    }
    expect(result.totalParams).toBe(expected);
  });

  it('moe_block: router (H*E) + E * gated-expert params', () => {
    const hidden = 512, numExperts = 8, intermediate = 1024;
    const n = [node('n1', 'moe_block', { hidden_size: hidden, num_experts: numExperts, intermediate_size: intermediate, top_k: 2 })];
    const result = generateModelCode(n, [], { ...BASE_HW, useBias: false }, 'MoE');
    const expertParams = 3 * hidden * intermediate; // gated MLP, no bias
    const expected = hidden * numExperts + numExperts * expertParams;
    expect(result.totalParams).toBe(expected);
  });

  it('conv2d: in*out*k*k, no bias', () => {
    const n = [node('n1', 'conv2d', { in_channels: 3, out_channels: 64, kernel_size: 7 })];
    const result = generateModelCode(n, [], { ...BASE_HW, useBias: false }, 'Conv');
    expect(result.totalParams).toBe(3 * 64 * 7 * 7);
  });

  it('activation and pooling layers contribute zero parameters but still call a real op', () => {
    const nodes = [
      node('n1', 'conv2d', { in_channels: 3, out_channels: 8, kernel_size: 3 }),
      node('n2', 'relu', {}),
      node('n3', 'avg_pool', {}),
    ];
    const result = generateModelCode(nodes, chain(nodes), { ...BASE_HW, useBias: false }, 'Pool');
    const relu = result.layers.find((l) => l.layerType === 'relu')!;
    const pool = result.layers.find((l) => l.layerType === 'avg_pool')!;
    expect(relu.paramCount).toBe(0);
    expect(relu.initCode).toContain('nn.ReLU');
    expect(pool.initCode).toContain('AdaptiveAvgPool2d');
  });

  it('dcgan_generator_block: in*out*k*k, defaults to bias=False (matches the real DCGAN convention)', () => {
    const n = [node('n1', 'dcgan_generator_block', { in_channels: 100, out_channels: 512, kernel_size: 4 })];
    const result = generateModelCode(n, [], BASE_HW, 'Gen');
    expect(result.totalParams).toBe(100 * 512 * 4 * 4);
    expect(result.layers[0].initCode).toContain('nn.ConvTranspose2d');
    expect(result.layers[0].initCode).toContain('bias=False');
  });

  it('dcgan_discriminator_block: in*out*k*k, defaults to bias=False', () => {
    const n = [node('n1', 'dcgan_discriminator_block', { in_channels: 3, out_channels: 64, kernel_size: 4 })];
    const result = generateModelCode(n, [], BASE_HW, 'Disc');
    expect(result.totalParams).toBe(3 * 64 * 4 * 4);
    expect(result.layers[0].initCode).toContain('nn.Conv2d');
    expect(result.layers[0].initCode).toContain('bias=False');
  });

  it('a full DCGAN generator+discriminator (the official PyTorch tutorial shape) matches its independently re-derived conv-weight total', () => {
    // Same channel progression as examples/models/dcgan.json after its fix:
    // nz=100, ngf=ndf=64, nc=3 — the canonical PyTorch DCGAN tutorial
    // config. This checks only conv weights (bias=False throughout);
    // NEURAX's own analysis doesn't model BatchNorm's affine parameters
    // for these two layer types either, so the two stay comparable.
    const nodes = [
      node('g1', 'dcgan_generator_block', { in_channels: 100, out_channels: 512, kernel_size: 4 }),
      node('g2', 'dcgan_generator_block', { in_channels: 512, out_channels: 256, kernel_size: 4 }),
      node('g3', 'dcgan_generator_block', { in_channels: 256, out_channels: 128, kernel_size: 4 }),
      node('g4', 'dcgan_generator_block', { in_channels: 128, out_channels: 64, kernel_size: 4 }),
      node('g5', 'dcgan_generator_block', { in_channels: 64, out_channels: 3, kernel_size: 4 }),
      node('d1', 'dcgan_discriminator_block', { in_channels: 3, out_channels: 64, kernel_size: 4 }),
      node('d2', 'dcgan_discriminator_block', { in_channels: 64, out_channels: 128, kernel_size: 4 }),
      node('d3', 'dcgan_discriminator_block', { in_channels: 128, out_channels: 256, kernel_size: 4 }),
      node('d4', 'dcgan_discriminator_block', { in_channels: 256, out_channels: 512, kernel_size: 4 }),
      node('d5', 'dcgan_discriminator_block', { in_channels: 512, out_channels: 1, kernel_size: 4 }),
    ];
    const result = generateModelCode(nodes, chain(nodes), BASE_HW, 'DCGAN');

    const genConvWeights = 100 * 512 * 16 + 512 * 256 * 16 + 256 * 128 * 16 + 128 * 64 * 16 + 64 * 3 * 16;
    const discConvWeights = 3 * 64 * 16 + 64 * 128 * 16 + 128 * 256 * 16 + 256 * 512 * 16 + 512 * 1 * 16;
    expect(genConvWeights).toBe(3_574_784);
    expect(discConvWeights).toBe(2_763_776);
    expect(result.totalParams).toBe(genConvWeights + discConvWeights);
    expect(result.fullySupported).toBe(true);
  });

  it('gcn_conv: in*out + bias, matches gcn_params (bias defaults to False like the GAN blocks)', () => {
    const n = [node('n1', 'gcn_conv', { in_features: 128, out_features: 256, bias: true })];
    const result = generateModelCode(n, [], BASE_HW, 'GCN');
    expect(result.totalParams).toBe(128 * 256 + 256);
    expect(result.layers[0].initCode).toContain('GCNConv(128, 256, bias=True)');
    expect(result.layers[0].forwardLines[0]).toContain('edge_index');
  });

  it('a 3-layer GCN (the ogbn-arxiv shape verified against the OGB leaderboard in published_model_accuracy.rs) matches its conv-only total', () => {
    // Same layer widths and bias=True as examples/models/gcn_ogbn_arxiv.json
    // (128->256->256->256->40); that file's full total (110,120, including
    // two BatchNorm1d layers) is checked against the real OGB leaderboard
    // figure on the Rust side. This test isolates just the three GCNConv
    // layers' contribution: 33,024 + 65,792 + 10,280 = 109,096 — the
    // remaining 1,024 comes from the two 256-wide normalization layers.
    const nodes = [
      node('conv1', 'gcn_conv', { in_features: 128, out_features: 256, bias: true }),
      node('conv2', 'gcn_conv', { in_features: 256, out_features: 256, bias: true }),
      node('conv3', 'gcn_conv', { in_features: 256, out_features: 40, bias: true }),
    ];
    const result = generateModelCode(nodes, chain(nodes), BASE_HW, 'GCN');
    expect(result.totalParams).toBe(33_024 + 65_792 + 10_280);
    expect(result.totalParams).toBe(109_096);
    expect(result.fullySupported).toBe(true);
    expect(result.code).toContain('from torch_geometric.nn import GCNConv, GATConv, RGCNConv');
    expect(result.code).toContain('def forward(self, x, edge_index):');
  });

  it('gat_conv: per-head weight + attention vectors, matches gat_params', () => {
    const inFeatures = 4096, outFeatures = 4096, heads = 8;
    const headDim = outFeatures / heads;
    const expected = inFeatures * heads * headDim + heads * headDim * 2;
    const n = [node('n1', 'gat_conv', { in_features: inFeatures, out_features: outFeatures, num_heads: heads })];
    const result = generateModelCode(n, [], BASE_HW, 'GAT');
    expect(result.totalParams).toBe(expected);
    expect(result.layers[0].initCode).toContain(`GATConv(${inFeatures}, ${headDim}, heads=${heads}`);
  });

  it('rgcn_conv: per-relation weights + self-loop, matches rgcn_params, forward takes edge_type too', () => {
    const inFeatures = 64, outFeatures = 64, numRelations = 5;
    const expected = numRelations * inFeatures * outFeatures + inFeatures * outFeatures;
    const n = [node('n1', 'rgcn_conv', { in_features: inFeatures, out_features: outFeatures, num_relations: numRelations })];
    const result = generateModelCode(n, [], BASE_HW, 'RGCN');
    expect(result.totalParams).toBe(expected);
    expect(result.code).toContain('def forward(self, x, edge_index, edge_type):');
    expect(result.layers[0].forwardLines[0]).toContain('edge_type');
  });

  it('lstm_cell: 4 gates x (hidden+input) x hidden + 4*hidden bias, matches lstm_params', () => {
    // Input size, absent an upstream conv/attention node to derive it from,
    // falls back to ctx.channels' own initial value (hw.inChannels, 3 on
    // BASE_HW) — the same fallback conv2d's first layer in a design uses.
    const hidden = 512, input = BASE_HW.inChannels ?? 3;
    const expected = 4 * (hidden + input) * hidden + 4 * hidden;
    const n = [node('n1', 'lstm_cell', { hidden_size: hidden, num_layers: 1, bidirectional: false })];
    const result = generateModelCode(n, [], BASE_HW, 'LSTM');
    expect(result.layers[0].paramCount).toBe(expected);
    expect(result.layers[0].initCode).toContain(`nn.LSTM(${input}, ${hidden}, num_layers=1, batch_first=True, bidirectional=False)`);
    expect(result.layers[0].forwardLines[0]).toBe('x, _ = self.l1_n1(x)');
  });

  it('bigru: doubles the single-direction gru_params total (matches architecture/mod.rs\'s bidir_mult)', () => {
    const hidden = 256, input = BASE_HW.inChannels ?? 3;
    const singleDirection = 2 * (hidden + input) * hidden + (hidden + input) * hidden + 3 * hidden;
    const n = [node('n1', 'bigru', { hidden_size: hidden, num_layers: 1, bidirectional: true })];
    const result = generateModelCode(n, [], BASE_HW, 'BiGRU');
    expect(result.layers[0].paramCount).toBe(singleDirection * 2);
    expect(result.layers[0].initCode).toContain('nn.GRU');
    expect(result.layers[0].initCode).toContain('bidirectional=True');
  });

  it('a 4-layer stacked LSTM costs more than a 1-layer one at the same hidden size (matches the num_rnn_layers fix)', () => {
    const single = [
      node('emb', 'token_embedding', { vocab_size: 1000, hidden_size: 256 }),
      node('n1', 'lstm_cell', { hidden_size: 512, num_layers: 1 }),
    ];
    const stacked = [
      node('emb', 'token_embedding', { vocab_size: 1000, hidden_size: 256 }),
      node('n1', 'lstm_cell', { hidden_size: 512, num_layers: 4 }),
    ];
    const singleResult = generateModelCode(single, chain(single), BASE_HW, 'LSTM1');
    const stackedResult = generateModelCode(stacked, chain(stacked), BASE_HW, 'LSTM4');
    expect(stackedResult.layers[1].paramCount).toBeGreaterThan(singleResult.layers[1].paramCount * 1.5);
    expect(stackedResult.layers[1].initCode).toContain('num_layers=4');
  });

  it('unet_block: layers_per_block repeats of a ResNet block, matches the real SD1.5/SDXL down-stage shape', () => {
    // Same conv_in -> down_0 shape as examples/models/stable_diffusion_1.5.json:
    // conv_in produces 320 channels, down_0 is hidden_size=320,
    // layers_per_block=2 — in==out==320, so no downsample branch.
    const hidden = 320, layersPerBlock = 2;
    const oneBlock = 320 * 320 * 9 + 320 * 320 * 9 + 2 * 2 * 320; // conv1 + conv2 + BN, no downsample (in==out, stride=1)
    const n = [
      node('conv_in', 'conv2d', { in_channels: 4, out_channels: hidden, kernel_size: 3 }),
      node('n1', 'unet_block', { hidden_size: hidden, layers_per_block: layersPerBlock }),
    ];
    const result = generateModelCode(n, chain(n), BASE_HW, 'UNet');
    expect(result.layers[1].paramCount).toBe(oneBlock * layersPerBlock);
    expect(result.layers[1].initCode).toContain(`NeuraxBasicBlockStage(${hidden}, planes=${hidden}, blocks=${layersPerBlock}, stride=1)`);
    expect(result.fullySupported).toBe(true);
  });
});

describe('a full small transformer — end-to-end total', () => {
  it('sums every layer correctly and matches when compared to itself as "the analysis"', () => {
    const nodes = [
      node('n1', 'token_embedding', { vocab_size: 1000, hidden_size: 128 }),
      node('n2', 'mha_attention', { hidden_size: 128, num_heads: 4 }),
      node('n3', 'layernorm', { hidden_size: 128 }),
      node('n4', 'ffn_gated', { hidden_size: 128, intermediate_size: 512 }),
      node('n5', 'layernorm', { hidden_size: 128 }),
      node('n6', 'lm_head', { hidden_size: 128, vocab_size: 1000 }),
    ];
    const result = generateModelCode(nodes, chain(nodes), { ...BASE_HW, useBias: false }, 'Tiny');
    expect(result.fullySupported).toBe(true);

    const expected =
      1000 * 128 // embedding
      + 4 * 128 * 128 // attention
      + 2 * 128 // layernorm
      + 3 * 128 * 512 // gated ffn
      + 2 * 128 // layernorm
      + 128 * 1000; // lm_head (dense, no bias)
    expect(result.totalParams).toBe(expected);

    const verification = verifyCodegenAgainstAnalysis(result, expected);
    expect(verification.matches).toBe(true);
    expect(verification.deltaPct).toBe(0);
  });

  it('reports a mismatch when the "analysis" total disagrees, rather than hiding it', () => {
    const nodes = [node('n1', 'token_embedding', { vocab_size: 1000, hidden_size: 128 })];
    const result = generateModelCode(nodes, [], BASE_HW, 'Tiny');
    const verification = verifyCodegenAgainstAnalysis(result, result.totalParams * 2);
    expect(verification.matches).toBe(false);
    // analyzed = 2x generated, so |generated - analyzed| / analyzed = 0.5
    expect(verification.deltaPct).toBeCloseTo(0.5, 5);
  });

  it('the generated file is importable-looking Python: balanced class/def structure', () => {
    const nodes = [node('n1', 'token_embedding', { vocab_size: 100, hidden_size: 32 })];
    const result = generateModelCode(nodes, [], BASE_HW, 'My Cool Model');
    expect(result.code).toContain('class MyCoolModel(nn.Module):');
    expect(result.code).toContain('def __init__(self):');
    expect(result.code).toContain('def forward(self, x):');
    expect(result.code).toContain('return x');
    // Every opened class has a matching, non-empty body — a very cheap
    // sanity check that this isn't the empty-shell failure mode again.
    expect(result.code).toMatch(/self\.l1_n1 = nn\.Embedding\(100, 32\)/);
  });
});

describe('compileUntilVerified — the generate/check/retry loop', () => {
  it('accepts a correct first attempt without retrying', () => {
    let calls = 0;
    const loop = compileUntilVerified(
      () => { calls += 1; return 'ok'; },
      (r) => r === 'ok',
      5,
    );
    expect(loop.converged).toBe(true);
    expect(loop.attempts).toBe(1);
    expect(calls).toBe(1);
  });

  it('retries a flaky generator until it produces a verifiable result', () => {
    let calls = 0;
    // Simulates an LLM-assisted generator that gets it wrong twice before
    // matching — the scenario this loop exists for, once a non-deterministic
    // generation strategy is plugged in.
    const loop = compileUntilVerified(
      () => { calls += 1; return calls; },
      (r) => r >= 3,
      5,
    );
    expect(loop.converged).toBe(true);
    expect(loop.attempts).toBe(3);
    expect(calls).toBe(3);
  });

  it('gives up honestly after maxAttempts rather than pretending success', () => {
    let calls = 0;
    const loop = compileUntilVerified(
      () => { calls += 1; return 'always wrong'; },
      (r) => r === 'never matches',
      3,
    );
    expect(loop.converged).toBe(false);
    expect(loop.attempts).toBe(3);
    expect(calls).toBe(3);
    // The failing result is still returned — callers can inspect it — but
    // `converged: false` means it must not be trusted or shipped as-is.
    expect(loop.result).toBe('always wrong');
  });
});

describe('generateModelCodeVerified — the entry point export should use', () => {
  it('converges in exactly one attempt for a fully-supported, correctly-analysed design', () => {
    const nodes = [node('n1', 'token_embedding', { vocab_size: 1000, hidden_size: 128 })];
    const expected = 1000 * 128;
    const loop = generateModelCodeVerified(nodes, [], BASE_HW, 'Tiny', expected);
    expect(loop.converged).toBe(true);
    expect(loop.attempts).toBe(1);
    expect(loop.verification.matches).toBe(true);
  });

  it('does not demand a match for a design with honestly-unsupported layers', () => {
    const nodes = [
      node('n1', 'token_embedding', { vocab_size: 1000, hidden_size: 128 }),
      node('n2', 'quantum_circuit' as any, {}),
    ];
    // The real analysis total includes the unsupported layer's contribution;
    // the generated code cannot possibly match it, and the loop must not
    // spin `maxAttempts` times pretending a retry could fix that.
    const loop = generateModelCodeVerified(nodes, [], BASE_HW, 'Tiny', 1000 * 128 + 999, 3);
    expect(loop.attempts).toBe(1);
    expect(loop.result.fullySupported).toBe(false);
  });
});
