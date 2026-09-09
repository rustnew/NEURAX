/**
 * A design saved to a file must come back as the same design.
 *
 * That sounds obvious enough not to test, which is exactly why file formats
 * lose data: the loss is always in the field nobody thought about — the groups,
 * the KV head count, the one hyperparameter that was set through a dialog
 * rather than on a block. So the round-trip test below compares the *whole*
 * snapshot rather than spot-checking, and will fail the moment a new field is
 * added to the design without being added to the file.
 *
 * The other half of these tests is about the file being usable by things that
 * are not this application: git, a diff viewer, a script. That is what makes it
 * a document instead of a save-state.
 */
import { describe, it, expect } from 'vitest';
import {
  serializeDesign,
  parseNeuraxFile,
  suggestedFileName,
  DesignSnapshot,
  NEURAX_FORMAT,
  NEURAX_FORMAT_VERSION,
} from './neuraxFile';
import { CanvasNode, Connection, NodeGroup } from '@/types/architecture.ts';
import { INITIALIZATION_METHODS } from './weightInitialization.ts';

const nodes: CanvasNode[] = [
  { id: 'n1', type: 'input', name: 'Input', x: 50, y: 140, params: { sequence_length: 4096 } },
  {
    id: 'n2',
    type: 'token_embedding',
    name: 'Token Embedding',
    x: 250,
    y: 140,
    params: { vocab_size: 32000, hidden_size: 4096 },
  },
  {
    id: 'n3',
    type: 'gqa_attention',
    name: 'GQA',
    x: 450,
    y: 140,
    params: { hidden_size: 4096, num_heads: 32, num_kv_heads: 8, causal: true },
    inputShape: '[B, 4096]',
    outputShape: '[B, 4096]',
  },
];

const connections: Connection[] = [
  { id: 'c1', from: 'n1', to: 'n2' },
  { id: 'c2', from: 'n2', to: 'n3' },
];

const groups: NodeGroup[] = [
  {
    id: 'g1',
    name: 'Decoder Block',
    nodeIds: ['n2', 'n3'],
    connectionIds: ['c2'],
    repeatCount: 32,
    x: 200,
    y: 100,
    collapsed: false,
  },
];

const snapshot: DesignSnapshot = {
  name: 'My LLaMA',
  architecture: 'transformer',
  nodes,
  connections,
  groups,
  hardware: {
    hardware: 'H100',
    precision: 'bf16',
    batchSize: 8,
    numLayers: 32,
    hiddenDim: 4096,
    numHeads: 32,
    kvHeads: 8,
    vocabSize: 32000,
    seqLen: 4096,
    gpuCount: 8,
  },
  analysis: null,
};

const FIXED_TIME = '2026-08-14T10:00:00.000Z';

describe('the .neurax document', () => {
  describe('round-trips a design', () => {
    it('returns every block, connection and group unchanged', () => {
      const parsed = parseNeuraxFile(serializeDesign(snapshot));
      expect(parsed.ok).toBe(true);
      if (!parsed.ok) return;

      expect(parsed.document.design.nodes).toEqual(nodes);
      expect(parsed.document.design.connections).toEqual(connections);
      expect(parsed.document.design.groups).toEqual(groups);
    });

    it('returns the name, family and full hardware configuration', () => {
      const parsed = parseNeuraxFile(serializeDesign(snapshot));
      if (!parsed.ok) throw new Error(parsed.error);

      expect(parsed.document.name).toBe('My LLaMA');
      expect(parsed.document.architecture).toBe('transformer');
      // Compared whole: a hardware field dropped by the format fails here.
      expect(parsed.document.hardware).toEqual(snapshot.hardware);
    });

    it('survives a second trip through the format unchanged', () => {
      // A format that loses something loses it on the first save; a format that
      // *adds* something drifts on every save. Both show up here.
      const once = serializeDesign(snapshot, { savedAt: FIXED_TIME });
      const parsed = parseNeuraxFile(once);
      if (!parsed.ok) throw new Error(parsed.error);

      const twice = serializeDesign(
        {
          name: parsed.document.name,
          architecture: parsed.document.architecture,
          nodes: parsed.document.design.nodes,
          connections: parsed.document.design.connections,
          groups: parsed.document.design.groups,
          hardware: parsed.document.hardware,
          analysis: parsed.document.analysis,
        },
        { savedAt: FIXED_TIME },
      );

      expect(twice).toBe(once);
    });

    it('keeps an analysis when one was recorded', () => {
      const withAnalysis = {
        ...snapshot,
        analysis: { totalParams: 6.74e9, numLayers: 32, modelType: 'transformer' } as never,
      };
      const parsed = parseNeuraxFile(serializeDesign(withAnalysis));
      if (!parsed.ok) throw new Error(parsed.error);
      expect((parsed.document.analysis as unknown as Record<string, unknown>)?.totalParams).toBe(
        6.74e9,
      );
    });
  });

  describe('is a file a person and a repository can work with', () => {
    it('declares what it is in the first two keys', () => {
      const text = serializeDesign(snapshot);
      const firstKeys = Object.keys(JSON.parse(text)).slice(0, 2);
      expect(firstKeys).toEqual(['format', 'version']);
      expect(JSON.parse(text).format).toBe(NEURAX_FORMAT);
      expect(JSON.parse(text).version).toBe(NEURAX_FORMAT_VERSION);
    });

    it('is pretty-printed, so a diff is line by line', () => {
      const text = serializeDesign(snapshot);
      expect(text.split('\n').length).toBeGreaterThan(50);
      expect(text.endsWith('\n'), 'a text file ends with a newline').toBe(true);
    });

    it('produces identical bytes for a design whose keys were written in another order', () => {
      // This is the property that keeps `git diff` honest. Editing a parameter
      // and editing it back reorders the object's keys; without sorting, the
      // file changes when the design did not.
      const reordered: DesignSnapshot = {
        ...snapshot,
        nodes: snapshot.nodes.map((n) => ({
          ...n,
          params: Object.fromEntries(Object.entries(n.params).reverse()),
        })),
      };

      expect(serializeDesign(reordered, { savedAt: FIXED_TIME })).toBe(
        serializeDesign(snapshot, { savedAt: FIXED_TIME }),
      );
    });

    it('changes exactly the lines that changed', () => {
      const before = serializeDesign(snapshot, { savedAt: FIXED_TIME }).split('\n');
      const deeper: DesignSnapshot = {
        ...snapshot,
        hardware: { ...snapshot.hardware, numLayers: 48 },
      };
      const after = serializeDesign(deeper, { savedAt: FIXED_TIME }).split('\n');

      expect(after.length).toBe(before.length);
      const changed = before.filter((line, i) => line !== after[i]);
      expect(changed, `expected one changed line, got ${changed.length}`).toHaveLength(1);
      expect(changed[0]).toContain('numLayers');
    });
  });

  describe('refuses files it cannot honestly open', () => {
    it('rejects invalid JSON with the parser message', () => {
      const result = parseNeuraxFile('{ nope');
      expect(result.ok).toBe(false);
      if (result.ok) return;
      expect(result.error).toMatch(/not valid json/i);
    });

    it('points at Import when given a HuggingFace config', () => {
      const result = parseNeuraxFile(
        JSON.stringify({ model_type: 'llama', hidden_size: 4096, num_hidden_layers: 32 }),
      );
      expect(result.ok).toBe(false);
      if (result.ok) return;
      expect(result.error).toMatch(/import/i);
    });

    it('refuses a file from a newer NEURAX rather than dropping what it cannot read', () => {
      const future = JSON.stringify({
        format: NEURAX_FORMAT,
        version: NEURAX_FORMAT_VERSION + 1,
        design: { nodes: [], connections: [], groups: [] },
      });
      const result = parseNeuraxFile(future);
      expect(result.ok).toBe(false);
      if (result.ok) return;
      expect(result.error).toMatch(/newer version/i);
    });

    it('reports a truncated file rather than opening an empty canvas', () => {
      const result = parseNeuraxFile(
        JSON.stringify({ format: NEURAX_FORMAT, version: 1, name: 'x' }),
      );
      expect(result.ok).toBe(false);
      if (result.ok) return;
      expect(result.error).toMatch(/no blocks/i);
    });
  });

  describe('repairs what it safely can', () => {
    it('drops connections whose endpoints are missing, and says so', () => {
      const handEdited = JSON.stringify({
        format: NEURAX_FORMAT,
        version: 1,
        name: 'edited',
        architecture: 'transformer',
        design: {
          nodes: [nodes[0], nodes[1]],
          connections: [...connections, { id: 'c9', from: 'n2', to: 'deleted-node' }],
          groups: [],
        },
        hardware: {},
      });

      const result = parseNeuraxFile(handEdited);
      expect(result.ok).toBe(true);
      if (!result.ok) return;

      // `c2` reaches n3 and `c9` reaches a deleted node; neither endpoint is in
      // this file, so only the edge between the two surviving blocks remains.
      expect(result.document.design.connections.map((c) => c.id)).toEqual(['c1']);
      expect(result.warnings.join(' ')).toMatch(/2 connections refer/i);
    });

    it('opens a file with no groups key', () => {
      const minimal = JSON.stringify({
        format: NEURAX_FORMAT,
        version: 1,
        design: { nodes, connections },
      });
      const result = parseNeuraxFile(minimal);
      expect(result.ok).toBe(true);
      if (!result.ok) return;
      expect(result.document.design.groups).toEqual([]);
      expect(result.document.name).toBe('Untitled');
    });
  });

  describe('suggests a filename that works', () => {
    it('uses the design name', () => {
      expect(suggestedFileName('My LLaMA')).toBe('My-LLaMA.neurax');
    });

    it('strips characters a filesystem would reject', () => {
      expect(suggestedFileName('a/b:c*d?e')).toBe('abcde.neurax');
    });

    it('falls back rather than producing a nameless file', () => {
      expect(suggestedFileName('   ')).toBe('untitled-design.neurax');
      expect(suggestedFileName('///')).toBe('untitled-design.neurax');
    });

    it('keeps the name short enough for every filesystem', () => {
      expect(suggestedFileName('x'.repeat(500)).length).toBeLessThanOrEqual(88);
    });
  });

  describe('carries Production\'s initialisation recipe', () => {
    const withInit: DesignSnapshot = {
      ...snapshot,
      initialization: {
        method: 'xavier_normal',
        gain: 1.0,
        hyperparameters: {
          learningRate: 0.0003,
          dropout: 0.1,
          weightDecay: 0.01,
          warmupSteps: 500,
          optimizer: 'AdamW',
          gradientClipping: 1.0,
        },
        layers: [
          {
            layerId: 'n3',
            layerName: 'GQA',
            layerType: 'gqa_attention',
            shape: [4096, 12288],
            fanIn: 4096,
            fanOut: 12288,
            variance: 0.00032552083333333335,
          },
        ],
      },
    };

    it('round-trips the recipe, not just the architecture', () => {
      const parsed = parseNeuraxFile(serializeDesign(withInit));
      expect(parsed.ok).toBe(true);
      if (!parsed.ok) return;
      expect(parsed.document.initialization).toEqual(withInit.initialization);
    });

    it('is additive: an older file with no initialization section still opens', () => {
      const parsed = parseNeuraxFile(serializeDesign(snapshot));
      expect(parsed.ok).toBe(true);
      if (!parsed.ok) return;
      expect(parsed.document.initialization).toBeNull();
    });

    it('never stores the generated weight values, only the recipe that reproduces them', () => {
      // The point of storing method + config instead of the weight arrays: a
      // git diff of a few million random floats is noise. Only the recipe and
      // the real per-layer facts derived from it belong in the file.
      const text = serializeDesign(withInit);
      const parsed = JSON.parse(text);
      expect(Object.keys(parsed.initialization).sort()).toEqual(
        ['gain', 'hyperparameters', 'layers', 'method', 'sparsity'].filter(
          (k) => k in parsed.initialization,
        ),
      );
      expect(Object.keys(parsed.initialization.layers[0]).sort()).toEqual(
        ['fanIn', 'fanOut', 'layerId', 'layerName', 'layerType', 'shape', 'variance'].sort(),
      );
    });

    it('rejects a malformed initialization section rather than crashing the canvas', () => {
      const handEdited = JSON.stringify({
        format: NEURAX_FORMAT,
        version: 1,
        design: { nodes, connections: [], groups: [] },
        initialization: { method: 'xavier_normal' }, // no `layers` — incomplete
      });
      const parsed = parseNeuraxFile(handEdited);
      expect(parsed.ok).toBe(true);
      if (!parsed.ok) return;
      expect(parsed.document.initialization).toBeNull();
    });

    it('drops one malformed layer entry rather than trusting the whole array', () => {
      // A record only had its top-level shape checked (`method` is a string,
      // `layers` is an array) before being cast straight through — a single
      // entry missing `shape` reached Production's own rendering as `undefined`
      // instead of being caught here, next to the data that was actually bad.
      const handEdited = JSON.stringify({
        format: NEURAX_FORMAT,
        version: 1,
        design: { nodes, connections: [], groups: [] },
        initialization: {
          method: 'xavier_normal',
          hyperparameters: withInit.initialization!.hyperparameters,
          layers: [
            withInit.initialization!.layers[0], // valid
            { layerId: 'n4', layerName: 'broken' }, // missing shape/fanIn/fanOut/variance
          ],
        },
      });
      const parsed = parseNeuraxFile(handEdited);
      expect(parsed.ok).toBe(true);
      if (!parsed.ok) return;
      expect(parsed.document.initialization?.layers).toEqual([withInit.initialization!.layers[0]]);
      expect(parsed.warnings.some((w) => w.includes('malformed'))).toBe(true);
    });

    it('drops the whole record when every layer is malformed', () => {
      const handEdited = JSON.stringify({
        format: NEURAX_FORMAT,
        version: 1,
        design: { nodes, connections: [], groups: [] },
        initialization: {
          method: 'xavier_normal',
          hyperparameters: withInit.initialization!.hyperparameters,
          layers: [{ layerId: 'n4' }],
        },
      });
      const parsed = parseNeuraxFile(handEdited);
      expect(parsed.ok).toBe(true);
      if (!parsed.ok) return;
      expect(parsed.document.initialization).toBeNull();
    });

    it('accepts every real InitializationMethod and rejects an unrecognised one', () => {
      // The accepted set here is duplicated from `InitializationMethod`
      // rather than imported (importing it would make this module and
      // weightInitialization.ts import each other). Driven off
      // `INITIALIZATION_METHODS` — the same list Production's method picker
      // renders from — so an eighth or ninth method added there is exercised
      // here automatically instead of silently exempt from this check.
      for (const { id } of INITIALIZATION_METHODS) {
        const handEdited = JSON.stringify({
          format: NEURAX_FORMAT,
          version: 1,
          design: { nodes, connections: [], groups: [] },
          initialization: {
            method: id,
            hyperparameters: withInit.initialization!.hyperparameters,
            layers: [withInit.initialization!.layers[0]],
          },
        });
        const parsed = parseNeuraxFile(handEdited);
        expect(parsed.ok, id).toBe(true);
        if (!parsed.ok) continue;
        expect(parsed.document.initialization?.method, id).toBe(id);
      }

      const bogus = JSON.stringify({
        format: NEURAX_FORMAT,
        version: 1,
        design: { nodes, connections: [], groups: [] },
        initialization: {
          method: 'gradient_boosted_vibes',
          hyperparameters: withInit.initialization!.hyperparameters,
          layers: [withInit.initialization!.layers[0]],
        },
      });
      const parsed = parseNeuraxFile(bogus);
      expect(parsed.ok).toBe(true);
      if (!parsed.ok) return;
      expect(parsed.document.initialization).toBeNull();
      expect(parsed.warnings.some((w) => w.includes('unrecognised method'))).toBe(true);
    });
  });

  describe('carries Time Machine configuration', () => {
    // The panel stayed mounted for the whole session and simply forgot its
    // sliders the moment the design was reopened, because nothing about it
    // was in this file at all.
    const timeMachine: DesignSnapshot['timeMachine'] = {
      growthRate: 150,
      budgetMax: 750000,
      horizon: 3,
      hardware: 'h200',
    };
    const withPanels: DesignSnapshot = { ...snapshot, timeMachine };

    it('round-trips the panel’s configuration, not just the architecture', () => {
      const parsed = parseNeuraxFile(serializeDesign(withPanels));
      expect(parsed.ok).toBe(true);
      if (!parsed.ok) return;
      expect(parsed.document.timeMachine).toEqual(timeMachine);
    });

    it('is additive: an older file with neither section still opens', () => {
      const parsed = parseNeuraxFile(serializeDesign(snapshot));
      expect(parsed.ok).toBe(true);
      if (!parsed.ok) return;
      expect(parsed.document.timeMachine).toBeNull();
    });

    it('still opens a file written before the Inference panel was removed', () => {
      // Files in the wild carry an `inference` block. The panel that wrote it
      // is gone, but the design around it is not, and refusing the file — or
      // warning about a section the user can no longer see — would punish
      // them for a change they did not make.
      const olderFile = JSON.stringify({
        format: NEURAX_FORMAT,
        version: 1,
        name: 'Written by an older NEURAX',
        design: { nodes, connections: [], groups: [] },
        inference: { temperature: 0.7, top_k: 50, prompt_length: 2048 },
      });
      const parsed = parseNeuraxFile(olderFile);
      expect(parsed.ok).toBe(true);
      if (!parsed.ok) return;
      expect(parsed.document.name).toBe('Written by an older NEURAX');
      expect(parsed.document.design.nodes.length).toBe(nodes.length);
      expect(parsed.warnings.some((w) => w.includes('inference'))).toBe(false);
    });

    it('rejects a Time Machine section with an unknown hardware track', () => {
      const handEdited = JSON.stringify({
        format: NEURAX_FORMAT,
        version: 1,
        design: { nodes, connections: [], groups: [] },
        timeMachine: { ...timeMachine, hardware: 'tpu-v9-imaginary' },
      });
      const parsed = parseNeuraxFile(handEdited);
      expect(parsed.ok).toBe(true);
      if (!parsed.ok) return;
      expect(parsed.document.timeMachine).toBeNull();
      expect(parsed.warnings.some((w) => w.includes('Time Machine'))).toBe(true);
    });
  });
});
