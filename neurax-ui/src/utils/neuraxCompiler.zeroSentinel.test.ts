/**
 * An unset size must not reach the backend as a zero.
 *
 * `HardwareConfig` uses 0 as its "not set for this family" sentinel — a
 * transformer config ships `numNodes: 0`, a GNN config ships `vocabSize: 0`.
 * Those zeros used to travel all the way into `global_params`, where a
 * *present* key beats the backend formula's own fallback. Six GNN reference
 * templates and pix2pix answered `400 Compute IR error: Total FLOPs is zero`
 * instead of a number, because `gcn_flops(0, in, out, 0)` is 0 and the
 * compiler rejects a report whose total is zero.
 */
import { describe, expect, it } from 'vitest';
import { compileToNeuraxIR } from './neuraxCompiler.ts';

const gnn = () => ({
  nodes: [
    { id: 'a', type: 'input', name: 'In', x: 0, y: 0, params: {} },
    { id: 'b', type: 'gcn_conv', name: 'GCN', x: 1, y: 0, params: { in_features: 1433, out_features: 16 } },
    { id: 'c', type: 'output', name: 'Out', x: 2, y: 0, params: {} },
  ] as never[],
  conns: [
    { id: 'e1', from: 'a', to: 'b' },
    { id: 'e2', from: 'b', to: 'c' },
  ] as never[],
});

describe('unset sizes never travel as zeros', () => {
  it('omits a graph size of 0 so the backend default can apply', () => {
    const { nodes, conns } = gnn();
    const ir: any = compileToNeuraxIR(nodes, conns, {
      modelName: 'M', family: 'gnn', numNodes: 0, numEdges: 0, nodeFeatDim: 0,
    } as never);

    const g = ir.model.global_params;
    expect(g).not.toHaveProperty('num_nodes');
    expect(g).not.toHaveProperty('num_edges');
    expect(g).not.toHaveProperty('node_features');
  });

  it('still carries a graph size the user really set', () => {
    const { nodes, conns } = gnn();
    const ir: any = compileToNeuraxIR(nodes, conns, {
      modelName: 'M', family: 'gnn', numNodes: 2708, numEdges: 10556,
    } as never);

    expect(ir.model.global_params.num_nodes).toBe(2708);
    expect(ir.model.global_params.num_edges).toBe(10556);
  });

  it('drops every other zero-sentinel size, not just the graph ones', () => {
    const { nodes, conns } = gnn();
    const ir: any = compileToNeuraxIR(nodes, conns, {
      modelName: 'M', family: 'transformer',
      vocabSize: 0, hiddenDim: 0, numHeads: 0, ffnDim: 0, numExperts: 0, topK: 0,
    } as never);

    for (const key of ['vocab_size', 'hidden_size', 'num_heads', 'ffn_dim', 'num_experts', 'top_k']) {
      expect(ir.model.global_params, key).not.toHaveProperty(key);
    }
  });

  it('keeps a setting whose zero is a real value', () => {
    // 0 weight decay means no regularization — a choice, not an unset field.
    const { nodes, conns } = gnn();
    const ir: any = compileToNeuraxIR(nodes, conns, {
      modelName: 'M', family: 'transformer', weightDecay: 0, numDenseLayers: 0,
    } as never);

    expect(ir.model.global_params.weight_decay).toBe(0);
    expect(ir.model.global_params.num_dense_layers).toBe(0);
  });

  it('starts an image model at a real resolution when none was given', () => {
    // pix2pix began at a 0x0 image, its convolutions collapsed to 1x1, and the
    // whole design priced at zero FLOPs.
    const ir: any = compileToNeuraxIR(
      [
        { id: 'a', type: 'input', name: 'In', x: 0, y: 0, params: {} },
        { id: 'b', type: 'conv2d', name: 'Conv', x: 1, y: 0, params: { in_channels: 3, out_channels: 64, kernel_size: 4, stride: 2 } },
      ] as never[],
      [{ id: 'e1', from: 'a', to: 'b' }] as never[],
      { modelName: 'M', family: 'gan', imgHeight: 0, imgWidth: 0, inChannels: 0 } as never,
    );

    const conv = ir.model.layers.find((l: any) => l.layer_type === 'conv');
    expect(conv, 'the conv layer should be in the IR').toBeTruthy();
    const [, , h, w] = conv.input_shape;
    expect(h, 'height must not start at 0').toBeGreaterThan(1);
    expect(w, 'width must not start at 0').toBeGreaterThan(1);
  });
});
