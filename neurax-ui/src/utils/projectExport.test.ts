import { describe, it, expect } from 'vitest';
import { CanvasNode, Connection, AnalysisResult, ParameterValue } from '@/types/architecture.ts';
import { HardwareConfig } from '@/contexts/HardwareContext.tsx';
import { buildProjectFiles } from './projectExport.ts';

const BASE_HW: HardwareConfig = {
  hardware: 'a100', precision: 'fp16', batchSize: 4,
  learningRate: 3e-4, numEpochs: 3, gpuCount: 1, gpuMemoryGb: 80,
  datasetSize: 1000, seqLen: 128, vocabSize: 1000, hiddenDim: 128,
  numHeads: 4, ffnDim: 512, numLayers: 2, useBias: false, dropout: 0.0,
  useFlash: true, imgHeight: 224, imgWidth: 224, inChannels: 3,
  numDenoisingSteps: 0, guidanceScale: 0, numNodes: 0, numEdges: 0,
  nodeFeatDim: 0, dState: 16, dtRank: 32, timesteps: 0, spikeRate: 0,
  numExperts: 1, topK: 1, actionDim: 0, stateDim: 0,
};

function node(id: string, type: CanvasNode['type'], params: Record<string, ParameterValue> = {}): CanvasNode {
  return { id, type, name: id, x: 0, y: 0, params };
}

const NODES: CanvasNode[] = [
  node('n1', 'token_embedding', { vocab_size: 1000, hidden_size: 128 }),
  node('n2', 'mha_attention', { hidden_size: 128, num_heads: 4 }),
];
const CONNS: Connection[] = [{ id: 'c1', from: 'n1', to: 'n2' }];

const ANALYSIS: AnalysisResult = {
  totalParams: 1000 * 128 + 4 * 128 * 128,
  activeParams: 1000 * 128 + 4 * 128 * 128,
  numLayers: 2,
  modelType: 'transformer',
  graphDepth: 2,
  totalOperations: 2,
  criticalPathLength: 2,
  tensorResolutionRatio: 1,
  unresolvedDimCount: 0,
  totalTensorCount: 2,
  largestTensorBytes: 0,
  opsDistribution: {},
  totalFlops: 0, forwardFlops: 0, backwardFlops: 0, flopsPerToken: 0,
  flopsIncrementalDecode: 0, arithmeticIntensity: 0, bottleneck: 'compute',
  rooflinePosition: 0, estimatedFlops: '1.2 GFLOPs', forwardFlopsHuman: '', backwardFlopsHuman: '',
  peakVramBytes: 0, parameterMemoryBytes: 0, activationMemoryBytes: 0, gradientMemoryBytes: 0,
  optimizerStateBytes: 0, maxBatchSizeFit: 4, memoryFragmentation: 0, memoryUsage: '1.1 GB',
  gpuName: 'A100', gpuCount: 1, gpuMemoryGb: 80, gpuTflops: 300, gpuBandwidthGbs: 2000,
  dataParallelEfficiency: 1, communicationOverhead: 0, optimalGpuCount: 1, pipelineStages: 1,
  tensorParallelDegree: 1, latencyMs: 1, throughputTokensPerS: 100, gpuUtilization: 0.9,
  trainingCostUsd: 0.5, trainingTimeHours: 0.1, energyKwh: 0.1, co2Kg: 0.05,
  costPerMillionTokensUsd: 0.01, confidenceScore: 0.9, depth: 2,
};

describe('buildProjectFiles', () => {
  it('produces a modular package layout, not a pile of flat scripts', () => {
    const result = buildProjectFiles(NODES, CONNS, BASE_HW, 'transformer', 'My Model', ANALYSIS);
    const paths = result.files.map((f) => f.path).sort();
    expect(paths).toEqual([
      'README.md', 'data/.gitkeep', 'hyperparameters.json',
      'requirements.txt', 'src/__init__.py', 'src/model.py', 'train.py',
    ].sort());
  });

  it('src/ is a real Python package', () => {
    const result = buildProjectFiles(NODES, CONNS, BASE_HW, 'transformer', 'My Model', ANALYSIS);
    expect(result.files.find((f) => f.path === 'src/__init__.py')).toBeTruthy();
    const train = result.files.find((f) => f.path === 'train.py')!.content;
    expect(train).toContain('from src.model import');
  });

  it('hyperparameters.json round-trips the exact hardware config', () => {
    const result = buildProjectFiles(NODES, CONNS, BASE_HW, 'transformer', 'My Model', ANALYSIS);
    const hpFile = result.files.find((f) => f.path === 'hyperparameters.json')!;
    const parsed = JSON.parse(hpFile.content);
    expect(parsed.learningRate).toBe(BASE_HW.learningRate);
    expect(parsed.batchSize).toBe(BASE_HW.batchSize);
    expect(parsed.hiddenDim).toBe(BASE_HW.hiddenDim);
  });

  it('verification matches when the design is fully within the supported set', () => {
    const result = buildProjectFiles(NODES, CONNS, BASE_HW, 'transformer', 'My Model', ANALYSIS);
    expect(result.codegen.fullySupported).toBe(true);
    expect(result.verification.matches).toBe(true);
  });

  it('README honestly reports a mismatch when unsupported layers are present', () => {
    const withUnsupported = [...NODES, node('n3', 'quantum_circuit' as any, {})];
    const result = buildProjectFiles(withUnsupported, CONNS, BASE_HW, 'transformer', 'My Model', ANALYSIS);
    const readme = result.files.find((f) => f.path === 'README.md')!.content;
    expect(result.codegen.fullySupported).toBe(false);
    expect(readme).toMatch(/not yet/i);
    expect(readme).toContain('quantum_circuit');
  });

  it('requirements.txt is family-aware', () => {
    const cnn = buildProjectFiles(NODES, CONNS, BASE_HW, 'cnn', 'CNN Model', ANALYSIS);
    const transformer = buildProjectFiles(NODES, CONNS, BASE_HW, 'transformer', 'T Model', ANALYSIS);
    expect(cnn.files.find((f) => f.path === 'requirements.txt')!.content).toContain('torchvision');
    expect(transformer.files.find((f) => f.path === 'requirements.txt')!.content).toContain('tokenizers');
  });

  it('train.py wires the real hyperparameter values, not placeholders', () => {
    const result = buildProjectFiles(NODES, CONNS, BASE_HW, 'transformer', 'My Model', ANALYSIS);
    const train = result.files.find((f) => f.path === 'train.py')!.content;
    expect(train).toContain(`hp.get("batchSize", ${BASE_HW.batchSize})`);
    expect(train).toContain(`hp.get("numEpochs", ${BASE_HW.numEpochs})`);
  });

  it('respects a target directory prefix for every file', () => {
    const result = buildProjectFiles(NODES, CONNS, BASE_HW, 'transformer', 'My Model', ANALYSIS, 'models/my-model');
    for (const f of result.files) {
      expect(f.path.startsWith('models/my-model/')).toBe(true);
    }
  });

  it('slugifies the model name for the zip filename base', () => {
    const result = buildProjectFiles(NODES, CONNS, BASE_HW, 'transformer', 'My Cool Model! 2.0', ANALYSIS);
    expect(result.slug).toBe('my_cool_model_2_0');
  });
});

describe('NEURAX signature across the generated project', () => {
  it('appears, consistently, in every file — not just the README', () => {
    const result = buildProjectFiles(NODES, CONNS, BASE_HW, 'transformer', 'My Model', ANALYSIS);
    for (const f of result.files) {
      expect(f.content.toLowerCase()).toContain('neurax');
    }
  });

  it('the README badge and file headers say "verified" only when the code actually matches the analysis', () => {
    const verifiedResult = buildProjectFiles(NODES, CONNS, BASE_HW, 'transformer', 'My Model', ANALYSIS);
    expect(verifiedResult.verification.matches).toBe(true);
    const readme = verifiedResult.files.find((f) => f.path === 'README.md')!.content;
    const modelPy = verifiedResult.files.find((f) => f.path === 'src/model.py')!.content;
    expect(readme).toMatch(/parameters-verified/);
    expect(modelPy).toMatch(/cross-checked against its own analysis/);

    const withUnsupported = [...NODES, node('n3', 'quantum_circuit' as any, {})];
    const unverifiedResult = buildProjectFiles(withUnsupported, CONNS, BASE_HW, 'transformer', 'My Model', ANALYSIS);
    const readme2 = unverifiedResult.files.find((f) => f.path === 'README.md')!.content;
    const modelPy2 = unverifiedResult.files.find((f) => f.path === 'src/model.py')!.content;
    expect(readme2).toMatch(/needs%20review/);
    expect(modelPy2).toMatch(/outside NEURAX's verified set/);
  });

  it('links back to the real NEURAX repository, not a placeholder', () => {
    const result = buildProjectFiles(NODES, CONNS, BASE_HW, 'transformer', 'My Model', ANALYSIS);
    for (const f of result.files) {
      if (f.content.includes('github.com')) {
        expect(f.content).toContain('github.com/rustnew/NEURAX');
      }
    }
  });
});

/**
 * The assistant can write architectures the translator refuses to express.
 * What must never happen is a folder that says NEURAX and contains a
 * different model from the one the studio is running.
 */
describe('buildProjectFiles with verified assistant code', () => {
  const OVERRIDE = {
    code: 'import torch.nn as nn\n\n\nclass HandWritten(nn.Module):\n    pass\n',
    modelClassName: 'HandWritten',
    builtParams: ANALYSIS.totalParams,
  };

  it('ships the assistant\'s file as src/model.py, not the translator\'s', () => {
    const result = buildProjectFiles(NODES, CONNS, BASE_HW, 'transformer', 'My Model', ANALYSIS, '', OVERRIDE);
    const model = result.files.find((f) => f.path === 'src/model.py');
    expect(model?.content).toContain('class HandWritten');
    expect(result.codegen.modelClassName).toBe('HandWritten');
  });

  it('the training script imports the class that is actually in the file', () => {
    // The failure this catches: model.py defines HandWritten, train.py
    // imports NeuraxModel, and the project does not run at all.
    const result = buildProjectFiles(NODES, CONNS, BASE_HW, 'transformer', 'My Model', ANALYSIS, '', OVERRIDE);
    const train = result.files.find((f) => f.path === 'train.py');
    expect(train?.content).toContain('HandWritten');
  });

  it('says in the README who wrote the code and what it survived', () => {
    const result = buildProjectFiles(NODES, CONNS, BASE_HW, 'transformer', 'My Model', ANALYSIS, '', OVERRIDE);
    const readme = result.files.find((f) => f.path === 'README.md')?.content ?? '';
    expect(readme).toContain("written by NEURAX's assistant");
    expect(readme).toContain('PyTorch actually built');
  });

  it('reports the count PyTorch built, not the one the translator predicted', () => {
    const result = buildProjectFiles(
      NODES, CONNS, BASE_HW, 'transformer', 'My Model', ANALYSIS, '',
      { ...OVERRIDE, builtParams: ANALYSIS.totalParams },
    );
    expect(result.codegen.totalParams).toBe(ANALYSIS.totalParams);
    expect(result.verification.matches).toBe(true);
  });

  it('without an override, nothing about the normal export changes', () => {
    const plain = buildProjectFiles(NODES, CONNS, BASE_HW, 'transformer', 'My Model', ANALYSIS);
    const explicitNull = buildProjectFiles(NODES, CONNS, BASE_HW, 'transformer', 'My Model', ANALYSIS, '', null);
    expect(explicitNull.files).toEqual(plain.files);
  });
});
