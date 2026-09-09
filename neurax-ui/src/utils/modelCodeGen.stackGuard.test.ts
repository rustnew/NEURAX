/**
 * A design that would generate in the wrong order must be refused, not run.
 *
 * `layer_stack` stands for N repeated blocks. Some designs also carry the
 * block's body — an attention and a feed-forward node — as separate blocks.
 * The compiler copes, because it sums per layer and never walks the graph in
 * order. A generated `forward()` cannot: it is a sequence, so it applies the
 * body once outside the stack, and the topological sort puts those body nodes
 * first because nothing feeds them.
 *
 * That produced BERT-base as attention → feed-forward → token embedding. It
 * trains, and it is not the model that was analysed — so every figure the
 * Accuracy view compares would be a comparison between two different models.
 */
import { describe, it, expect } from 'vitest';
import { generateModelCode } from './modelCodeGen';
import type { CanvasNode, Connection } from '@/types/architecture';
import { DEFAULT_HARDWARE_CONFIG } from '@/contexts/HardwareContext';

const node = (id: string, type: string, params: Record<string, unknown> = {}): CanvasNode =>
  ({ id, type, name: id, x: 0, y: 0, params } as unknown as CanvasNode);

const hw = { ...DEFAULT_HARDWARE_CONFIG, hiddenDim: 768, numLayers: 12, numHeads: 12, ffnDim: 3072 };

describe('code generation refuses what it cannot order', () => {
  it('refuses a stack that also has its body on the canvas', () => {
    const nodes = [
      node('n1', 'input'),
      node('n2', 'token_embedding', { vocab_size: 30522 }),
      node('n3', 'layer_stack', { num_layers: 12 }),
      node('n4', 'mha_attention'),
      node('n5', 'ffn_standard'),
      node('n6', 'output'),
    ];
    const connections: Connection[] = [
      { id: 'c1', from: 'n1', to: 'n2' },
      { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n6' },
    ] as unknown as Connection[];

    const result = generateModelCode(nodes, connections, hw, 'Ambiguous');
    expect(result.fullySupported).toBe(false);
    expect(result.unsupportedTypes.join(' ')).toContain('layer_stack');
  });

  it('generates a stack on its own, because the order is then unambiguous', () => {
    const nodes = [
      node('n1', 'input'),
      node('n2', 'token_embedding', { vocab_size: 1000 }),
      node('n3', 'layer_stack', { num_layers: 4 }),
      node('n4', 'output'),
    ];
    const connections: Connection[] = [
      { id: 'c1', from: 'n1', to: 'n2' },
      { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' },
    ] as unknown as Connection[];

    const result = generateModelCode(nodes, connections, hw, 'Clean');
    expect(result.fullySupported).toBe(true);
    expect(result.code).toContain('NeuraxStack(4');
    // And it declares what it eats, rather than leaving the harness to guess.
    expect(result.inputKind).toBe('tokens');
    expect(result.vocabSize).toBe(1000);
  });

  it('reads the input shape off the model, never off the hardware config', () => {
    // The bug: an image size set anywhere in the config meant the input was
    // assumed to be an image, so BERT was fed `[3, 224, 224]` floats.
    const imageConfig = { ...hw, imgHeight: 224, imgWidth: 224, inChannels: 3, seqLen: 512 };
    const nodes = [node('n1', 'input'), node('n2', 'token_embedding', { vocab_size: 500 }), node('n3', 'output')];
    const connections: Connection[] = [
      { id: 'c1', from: 'n1', to: 'n2' },
      { id: 'c2', from: 'n2', to: 'n3' },
    ] as unknown as Connection[];

    const result = generateModelCode(nodes, connections, imageConfig, 'Tokens');
    expect(result.inputKind).toBe('tokens');
    expect(result.inputShape).toEqual([512]);
  });

  it('marks a convolutional design as taking an image', () => {
    const imageConfig = { ...hw, imgHeight: 64, imgWidth: 64, inChannels: 3 };
    const nodes = [node('n1', 'input'), node('n2', 'conv2d', { out_channels: 16 }), node('n3', 'output')];
    const connections: Connection[] = [
      { id: 'c1', from: 'n1', to: 'n2' },
      { id: 'c2', from: 'n2', to: 'n3' },
    ] as unknown as Connection[];

    const result = generateModelCode(nodes, connections, imageConfig, 'Vision');
    expect(result.inputKind).toBe('image');
    expect(result.inputShape).toEqual([3, 64, 64]);
  });
});
