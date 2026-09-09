import { useEffect, useState, useCallback, useMemo, useRef, lazy, Suspense } from 'react';
import { useNavigate } from 'react-router-dom';
import { TopNav } from '@/components/layout/TopNav.tsx';
import { LayerPalette } from '@/components/layout/LayerPalette.tsx';
import { WorkspaceTabs, WorkspaceTab } from '@/components/layout/WorkspaceTabs.tsx';
import { ArchitectureCanvas } from '@/components/canvas/ArchitectureCanvas.tsx';
import { RightPanelTabs } from '@/components/panels/RightPanelTabs.tsx';
import type { RightPanelTabId } from '@/components/panels/RightPanelTabs.tsx';
import { InspectorPanel } from '@/components/panels/InspectorPanel.tsx';
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '@/components/ui/resizable.tsx';
import { Sheet, SheetContent } from '@/components/ui/sheet.tsx';
import { Button } from '@/components/ui/button.tsx';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog.tsx';
import { SimulationTargetPanel } from '@/components/panels/SimulationTargetPanel.tsx';
import type { TimeMachineConfig } from '@/components/timemachine/TimeMachineWorkspace.tsx';

// Code-split: none of these render anything at first paint (dialogs/panels
// gated on `isOpen`, or workspace tabs other than "architecture" hidden via
// CSS — see WorkspaceTabs' "stay mounted" comment). Loading them lazily keeps
// their ~4,700 combined lines out of the main chunk without changing when
// each actually mounts.
const AIChatDrawer = lazy(() => import('@/components/panels/AIChatDrawer.tsx'));
const ExportPanel = lazy(() =>
  import('@/components/panels/ExportPanel.tsx').then((m) => ({ default: m.ExportPanel }))
);
const ImportPanel = lazy(() =>
  import('@/components/panels/ImportPanel.tsx').then((m) => ({ default: m.ImportPanel }))
);
const SharePanel = lazy(() =>
  import('@/components/panels/SharePanel.tsx').then((m) => ({ default: m.SharePanel }))
);
const ComparePanel = lazy(() =>
  import('@/components/panels/ComparePanel.tsx').then((m) => ({ default: m.ComparePanel }))
);
const DocumentationPanel = lazy(() =>
  import('@/components/panels/DocumentationPanel.tsx').then((m) => ({ default: m.DocumentationPanel }))
);
const ModelHyperparametersDialog = lazy(() =>
  import('@/components/panels/ModelHyperparametersPanel.tsx').then((m) => ({ default: m.ModelHyperparametersDialog }))
);
const ProductionWorkspace = lazy(() =>
  import('@/components/production/ProductionWorkspace.tsx').then((m) => ({ default: m.ProductionWorkspace }))
);
const SimulationWorkspace = lazy(() =>
  import('@/components/simulation/SimulationWorkspace.tsx').then((m) => ({ default: m.SimulationWorkspace }))
);
const TrainingWorkspace = lazy(() =>
  import('@/components/training/TrainingWorkspace.tsx').then((m) => ({ default: m.TrainingWorkspace }))
);
const TimeMachineWorkspace = lazy(() =>
  import('@/components/timemachine/TimeMachineWorkspace.tsx').then((m) => ({ default: m.TimeMachineWorkspace }))
);

import { IS_MOCK, mockHardware, mockImageDataset, profileForSelection } from '@/services/mockRuntime.ts';
import { detectHardware, measureHardware } from '@/services/neuraxApi.ts';
import { isCpuOnly, primaryGpu } from '@/types/runtime.ts';
import type { DatasetProfile, HardwareProfile } from '@/types/runtime.ts';

import { ArchitectureFamily } from '@/types/plugins.ts';
import { VariantPreset } from '@/types/catalog.ts';
import { AnalysisResult, CanvasNode, Connection, LayerConfig, NodeGroup, PerLayerBreakdownRow, Warning, ParameterValue } from '@/types/architecture.ts';
import { ImportResult } from '@/utils/architectureImporter.ts';
import { compileToNeuraxIR } from '@/utils/neuraxCompiler.ts';
import { generateModelCode } from '@/utils/modelCodeGen.ts';
import {
  serializeDesign,
  parseNeuraxFile,
  suggestedFileName,
  NEURAX_EXTENSION,
  InitializationRecord,
} from '@/utils/neuraxFile.ts';
import { useDesignHistory } from '@/hooks/useDesignHistory.ts';
import { DesignVariant } from '@/utils/designComparison.ts';
import {
  saveTextFile,
  openTextFile,
  writeTextFile,
  canWriteInPlace,
} from '@/services/desktopRuntime.ts';
import { getBlockDefaults, normalizeBlockParams } from '@/utils/blockDefaults.ts';
import { DEFAULT_HARDWARE_CONFIG, HardwareConfig, useHardware, validateHardwareConfig, ArchitectureFamily as HwFamily } from '@/contexts/HardwareContext.tsx';
import { useAuth } from '@/contexts/AuthContext.tsx';
import { explainAnalysisFailure, failureAsWarnings } from '@/services/compilerErrors.ts';
import { analyze, analyzeStream, listProjects, createProject, updateProject, deleteProject, getCredits, type Project, type CreditInfo } from '@/services/neuraxApi.ts';
import { useToast } from '@/hooks/use-toast.ts';
import { getPluginLayers } from '@/plugins/registry.ts';
import { hasAnalysisReportData } from '@/components/simulation/simulationData.ts';

/**
 * neurax-agent's Python catalogue (catalogue.json) describes each family's
 * blocks with its own short type names (`mha`, `ffn`, `residual`, ...) —
 * authored independently of this canvas's own plugin registry, which uses
 * longer, more specific ones (`mha_attention`, `ffn_standard`,
 * `residual_add`, ...). `add_node` below looks a tool call's `layer_type`
 * straight up in `layerConfigByType`; when the agent's name isn't a literal
 * key there, the node was silently dropped — no error reached the agent or
 * the user, just a toast easy to miss mid-stream. A live build asking the
 * agent to grow an existing GPT-2 came back as a stack of bare LayerNorms:
 * every `mha`/`ffn`/`residual`/`swiglu` node it added was silently no-op'd,
 * and the few types that happened to match by coincidence (`layernorm`,
 * `embedding`) were all that survived.
 *
 * This is the direct fix for that: on a miss, retry the lookup through this
 * alias before giving up. Covers every 1:1 rename confirmed across the
 * agent's catalogue (catalogue.json) against this canvas's registry, family
 * by family. A few agent type names have no real counterpart at all rather
 * than just a different spelling:
 *   - `relu`/`gelu`/`silu`/`tanh`/`sigmoid`/`leaky_relu` aren't separate
 *     blocks here — activation is a *parameter* on the block before them
 *     (`hasActivation: true` + an `activation` field), never its own node.
 *   - SNN neuron/encoder types (`rate_encoder`, `lif_neuron`, ...) have no
 *     equivalent on this canvas at all — the SNN family isn't buildable by
 *     the agent yet.
 *   - A handful of rare ones (`pixelnorm`, `bahdanau_attention`) don't have
 *     a close-enough match to alias safely.
 * Every one of those still add_node-fails today — but `handleAgentToolEvent`
 * bridges connections across whatever gap that leaves (see
 * `droppedNodeIdsRef` below), so the design stays fully wired even where an
 * alias genuinely can't help.
 */
const AGENT_TYPE_ALIASES: Record<string, string> = {
  // transformer / moe
  mha: 'mha_attention',
  gqa: 'gqa_attention',
  mqa: 'mqa_attention',
  mla: 'mla_attention',
  ffn: 'ffn_standard',
  swiglu: 'ffn_gated',
  residual: 'residual_add',
  positional_encoding: 'pos_absolute',
  rope: 'pos_rope',
  alibi: 'pos_alibi',
  gate: 'router_linear',
  expert: 'expert_gated_ffn',
  add: 'residual_add',
  merge: 'concat',
  // cnn / gan / diffusion / gnn
  dense: 'linear_projection',
  depthwise_conv2d: 'depthwise_conv',
  conv_transpose2d: 'transposed_conv',
  se_block: 'se_layer',
  graphnorm: 'graph_norm',
  downsample: 'downsample_block',
  // ssm
  mamba_block: 'mamba_mixer',
};

const _hashString = (input: string): string => {
  let h = 5381;
  for (let i = 0; i < input.length; i++) {
    h = ((h << 5) + h) ^ input.charCodeAt(i);
  }
  return `cat-${(h >>> 0).toString(16)}`;
};

const _uniqueId = (prefix: string): string => {
  const p = prefix && prefix.trim().length > 0 ? prefix.trim() : 'id';
  const ru = (globalThis as any)?.crypto?.randomUUID?.bind((globalThis as any)?.crypto);
  if (typeof ru === 'function') {
    return `${p}-${ru()}`;
  }
  return `${p}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
};

const initialNodes: CanvasNode[] = [];

const initialConnections: Connection[] = [];

const initialAnalysis: AnalysisResult = {
  modelName: undefined,
  totalParams: 0, activeParams: 0, numLayers: 0, modelType: '', graphDepth: 0,
  totalOperations: 0, criticalPathLength: 0, tensorResolutionRatio: 1,
  unresolvedDimCount: 0, totalTensorCount: 0, largestTensorBytes: 0,
  opsDistribution: {},
  totalFlops: 0, forwardFlops: 0, backwardFlops: 0,
  flopsPerToken: 0, flopsIncrementalDecode: 0, arithmeticIntensity: 0, bottleneck: '', rooflinePosition: 0,
  estimatedFlops: '0 FLOPs', forwardFlopsHuman: '0 FLOPs', backwardFlopsHuman: '0 FLOPs',
  peakVramBytes: 0, parameterMemoryBytes: 0, activationMemoryBytes: 0,
  gradientMemoryBytes: 0, optimizerStateBytes: 0, maxBatchSizeFit: 0,
  memoryFragmentation: 0, memoryUsage: '0 GB',
  gpuName: '', gpuCount: 1, gpuMemoryGb: 0, gpuTflops: 0, gpuBandwidthGbs: 0,
  interconnect: '', interconnectBandwidthGbs: 0,
  dataParallelEfficiency: 1, communicationOverhead: 0, optimalGpuCount: 1,
  pipelineStages: 0, tensorParallelDegree: 0,
  dataParallel: 1, tensorParallel: 1, pipelineParallel: 1,
  latencyMs: null, throughputTokensPerS: 0, throughputGraphsPerS: null, gpuUtilization: null,
  trainingCostUsd: 0, trainingTimeHours: 0, energyKwh: 0, co2Kg: 0,
  costPerMillionTokensUsd: 0, gpuHours: 0, provider: '',
  selectedPrecision: 'fp16', selectedBatchSize: 1,
  confidenceScore: 1, depth: 0,
  isSequenceModel: true, customLayerCount: 0, diagnosticCount: 0,
  sequenceLength: undefined, numAttentionHeads: undefined, numKeyValueHeads: undefined,
  intermediateSize: undefined, layersByType: undefined,
  analysisTimeMs: undefined, generatedAt: undefined,
  reportWarnings: [], recommendations: [],
  live_trace: {
    partial_metrics: [],
    throughput_trace: [],
    memory_liveness: [],
    memory_heatmap: [],
    gradient_memory_breakdown: [],
    kv_cache_scaling: [],
  },
  memory_liveness: [],
  memory_heatmap: [],
  gradient_memory_breakdown: [],
  kv_cache_scaling: [],
  diagnostics: [],
};


const initialWarnings: Warning[] = [];

const ALL_ARCHITECTURE_FAMILIES: ArchitectureFamily[] = [
  'transformer',
  'moe',
  'ssm',
  'cnn',
  'diffusion',
  'gnn',
  'gan',
  'rnn',
];

const isPositiveNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value > 0;

export const withPositiveFallback = (value: number | undefined, fallback: number): number =>
  isPositiveNumber(value) ? value : fallback;

export const findNumericParam = (
  nodes: CanvasNode[],
  nodeTypes: string[],
  paramKeys: string[],
): number | undefined => {
  for (const node of nodes) {
    if (!nodeTypes.includes(node.type)) continue;
    const params = (node.params ?? {}) as Record<string, unknown>;
    for (const key of paramKeys) {
      const value = params[key];
      if (isPositiveNumber(value)) return value;
    }
  }
  return undefined;
};

export function hydrateNodesForFamily(
  family: ArchitectureFamily,
  nodes: CanvasNode[],
): CanvasNode[] {
  const layerMap = new Map(getPluginLayers(family).map((layer) => [layer.type, layer]));

  // The model's width, taken from whichever blocks do state it.
  //
  // Templates do not repeat the width on every block — GPT-2 XL gives its FFN
  // an intermediate size but no `hidden_size` — and a block that falls back to
  // the schema's 768 default instead silently shrinks that layer. Inheriting
  // keeps a design consistent with itself.
  const statedWidths = nodes
    .map((node) => {
      const p = (node.params ?? {}) as Record<string, unknown>;
      const value = p.d_model ?? p.hidden_size ?? p.embedding_dim;
      return typeof value === 'number' && value > 0 ? value : null;
    })
    .filter((v): v is number => v !== null);
  const modelWidth = statedWidths.length
    ? statedWidths.sort((a, b) => statedWidths.filter((v) => v === a).length - statedWidths.filter((v) => v === b).length).pop()
    : undefined;

  return nodes.map((node) => {
    const config = layerMap.get(node.type);
    if (!config) return node;

    // Normalise before merging: a template states `hidden_size` where the block
    // schema says `d_model`, and without this the template's value sits beside
    // the default instead of replacing it.
    const stated = normalizeBlockParams(
      node.type,
      (node.params ?? {}) as Record<string, unknown>,
    );
    const schema = getBlockDefaults(node.type);
    // Inherit the model's width when this block does not state one of its own.
    if (modelWidth && 'd_model' in schema && stated.d_model === undefined) {
      stated.d_model = modelWidth;
    }
    if (modelWidth && 'normalized_shape' in schema && stated.normalized_shape === undefined) {
      stated.normalized_shape = modelWidth;
    }

    const params: Record<string, unknown> = {
      ...schema,
      ...(config.defaultParams ?? {}),
      ...stated,
    };

    if (config.hasActivation && !('activation' in params)) {
      params.activation = 'none';
    }

    return {
      ...node,
      params: params as any,
    };
  });
}

/**
 * The agent's hyperparameter names, translated into this codebase's.
 *
 * The agent names them the way papers and training scripts do
 * (`learning_rate`, `warmup_steps`); `HardwareConfig` names them the way the
 * rest of this app does. Anything without a field of its own goes to
 * `customParams`, which exists for exactly that and is forwarded verbatim into
 * the model's `global_params` — dropping it would lose a value the agent is
 * actively reasoning about (betas, dropout, gradient clipping).
 */
const AGENT_HYPERPARAM_FIELDS: Record<string, keyof HardwareConfig> = {
  learning_rate: 'learningRate',
  weight_decay: 'weightDecay',
  warmup_steps: 'warmupSteps',
  max_steps: 'maxSteps',
  batch_size: 'batchSize',
  gradient_accumulation_steps: 'gradAccumSteps',
  lr_schedule: 'lrScheduler',
  optimizer: 'optimizer',
  num_epochs: 'numEpochs',
  zero_stage: 'zeroStage',
  gradient_checkpointing: 'gradientCheckpointing',
  precision: 'precision',
};

export function mapAgentHyperparams(
  updates: Record<string, unknown>,
  currentCustomParams?: Record<string, string | number | boolean>,
): Record<string, unknown> {
  const mapped: Record<string, unknown> = {};
  const custom: Record<string, string | number | boolean> = {};

  for (const [key, value] of Object.entries(updates)) {
    const field = AGENT_HYPERPARAM_FIELDS[key];
    if (field) {
      mapped[field] = value;
    } else if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
      custom[key] = value;
    }
  }

  if (Object.keys(custom).length > 0) {
    // Merged, not replaced: a later `set_hyperparams` must not wipe what an
    // earlier one put there.
    mapped.customParams = { ...(currentCustomParams ?? {}), ...custom };
  }
  return mapped;
}

export function buildHardwareConfigFromPreset(
  preset: VariantPreset,
  current: HardwareConfig,
): HardwareConfig {
  const nodes = preset.nodes;
  const embeddingDim = findNumericParam(
    nodes,
    ['token_embedding', 'embedding', 'lm_head', 'layernorm', 'rmsnorm', 'pos_absolute', 'pos_rope'],
    ['d_model', 'normalized_shape'],
  );
  const layerCount = findNumericParam(nodes, ['layer_stack', 'dit_block'], ['num_layers', 'depth']);
  const vocabSize = findNumericParam(
    nodes,
    ['token_embedding', 'embedding', 'lm_head'],
    ['vocab_size', 'vocabSize'],
  );
  const maxSeqLen = findNumericParam(
    nodes,
    ['pos_absolute', 'pos_rope', 'noise_scheduler'],
    ['max_len', 'num_train_timesteps'],
  );
  const attentionHeads = findNumericParam(
    nodes,
    ['mha_attention', 'mqa_attention', 'gqa_attention', 'cross_attention', 'flash_attention', 'dit_block'],
    ['n_heads', 'num_heads', 'heads'],
  );
  const feedForwardDim = findNumericParam(
    nodes,
    ['ffn_standard', 'ffn_gated', 'moe_block', 'moe_layer', 'expert_linear', 'expert_gated_ffn'],
    ['d_ff', 'expert_d_ff'],
  );
  const inChannels = findNumericParam(
    nodes,
    ['stem_block', 'conv2d', 'conv1d', 'conv3d', 'patchify', 'dit_block', 'patch_embed', 'unet_block', 'dcgan_generator_block', 'dcgan_discriminator_block'],
    ['in_channels', 'channels'],
  );
  const numClasses = findNumericParam(nodes, ['classification_head'], ['num_classes']);
  const nodeFeatDim = findNumericParam(
    nodes,
    ['gcn_conv', 'gat_conv', 'sage_conv', 'gin_conv'],
    ['in_channels', 'in_features'],
  );
  const dState = findNumericParam(
    nodes,
    ['s6_block', 's4_block', 's5_block', 'h3_block', 'gated_ssm', 'mamba_block'],
    ['d_state'],
  );
  const dtRank = findNumericParam(nodes, ['s6_block', 'mamba_block'], ['dt_rank']);
  const convKernel = findNumericParam(nodes, ['s6_block', 'mamba_block'], ['conv_kernel', 'd_conv']);
  const hiddenSize = findNumericParam(
    nodes,
    ['lstm', 'gru', 'lstm_cell', 'gru_cell', 'bilstm', 'bigru'],
    ['hidden_size', 'hiddenSize'],
  );
  const numExperts = findNumericParam(
    nodes,
    ['moe_block', 'moe_layer', 'router_linear', 'router_softmax'],
    ['num_experts'],
  );
  const topK = findNumericParam(nodes, ['moe_block', 'moe_layer', 'router_linear'], ['top_k']);
  const diffusionInputSize = findNumericParam(nodes, ['dit_block'], ['input_size']);
  const modelChannels = findNumericParam(
    nodes,
    ['timestep_embedding', 'spatial_transformer'],
    ['channels', 'in_channels'],
  );

  const base: HardwareConfig = {
    ...DEFAULT_HARDWARE_CONFIG,
    hardware: current.hardware,
    precision: current.precision,
    batchSize: current.batchSize > 0 ? current.batchSize : DEFAULT_HARDWARE_CONFIG.batchSize,
    seed: current.seed,
    device: current.device,
    useCompile: current.useCompile,
  };

  switch (preset.family) {
    case 'transformer':
      return {
        ...base,
        seqLen: withPositiveFallback(maxSeqLen, 1024),
        vocabSize: withPositiveFallback(vocabSize, 32000),
        hiddenDim: withPositiveFallback(embeddingDim, 768),
        numHeads: withPositiveFallback(attentionHeads, 12),
        ffnDim: withPositiveFallback(feedForwardDim, 3072),
        numLayers: withPositiveFallback(layerCount, 12),
      };
    case 'moe':
      return {
        ...base,
        seqLen: withPositiveFallback(maxSeqLen, 4096),
        vocabSize: withPositiveFallback(vocabSize, 32000),
        hiddenDim: withPositiveFallback(embeddingDim, 2048),
        numHeads: withPositiveFallback(attentionHeads, 16),
        ffnDim: withPositiveFallback(feedForwardDim, 8192),
        numLayers: withPositiveFallback(layerCount, 32),
        numExperts: withPositiveFallback(numExperts, 8),
        topK: withPositiveFallback(topK, 2),
      };
    case 'ssm':
      return {
        ...base,
        seqLen: withPositiveFallback(maxSeqLen, 2048),
        vocabSize: withPositiveFallback(vocabSize, 32000),
        hiddenDim: withPositiveFallback(embeddingDim, 768),
        numLayers: withPositiveFallback(layerCount, 24),
        dState: withPositiveFallback(dState, 16),
        dtRank: withPositiveFallback(dtRank, 48),
        convKernel: withPositiveFallback(convKernel, 4),
      };
    case 'cnn':
      return {
        ...base,
        imgHeight: withPositiveFallback(findNumericParam(nodes, ['patch_embed'], ['img_size']), 224),
        imgWidth: withPositiveFallback(findNumericParam(nodes, ['patch_embed'], ['img_size']), 224),
        inChannels: withPositiveFallback(inChannels, 3),
        numClasses: withPositiveFallback(numClasses, 1000),
        numLayers: withPositiveFallback(layerCount, 50),
      };
    case 'diffusion':
      return {
        ...base,
        imgHeight: withPositiveFallback(diffusionInputSize, 64),
        imgWidth: withPositiveFallback(diffusionInputSize, 64),
        inChannels: withPositiveFallback(inChannels, 4),
        numDenoisingSteps: withPositiveFallback(findNumericParam(nodes, ['noise_scheduler'], ['num_train_timesteps']), 50),
        guidanceScale: withPositiveFallback(current.guidanceScale, 7.5),
        modelChannels: withPositiveFallback(modelChannels, 320),
      };
    case 'gnn':
      return {
        ...base,
        numNodes: 2708,
        numEdges: 10556,
        nodeFeatDim: withPositiveFallback(nodeFeatDim, 16),
        outDim: withPositiveFallback(numClasses, 64),
        numLayers: withPositiveFallback(layerCount, 2),
      };
    case 'gan':
      return {
        ...base,
        imgHeight: 64,
        imgWidth: 64,
        inChannels: withPositiveFallback(inChannels, 3),
      };
    case 'rnn':
      return {
        ...base,
        seqLen: withPositiveFallback(maxSeqLen, 128),
        vocabSize: withPositiveFallback(vocabSize, 32000),
        hiddenSize: withPositiveFallback(hiddenSize, 512),
        numLayers: withPositiveFallback(layerCount, 2),
      };
    default:
      return base;
  }
}

// ─── Report Parsing Helper ───────────────────────────────────────────
// Extracted so both synchronous and streaming analysis handlers can share it.

interface ParsedReportState {
  analysis: AnalysisResult;
  perLayer: PerLayerBreakdownRow[];
  warnings: Warning[];
  perLayerLatency: Record<string, number>;
  perLayerVram: Record<string, number>;
}

function parseAnalysisReport(
  rawReport: unknown,
  precision: string,
  batchSize: number,
  nodes: CanvasNode[] = [],
): ParsedReportState {
  const r = rawReport as Record<string, unknown>;
  const rpt = ((r as any)?.report ?? r) as Record<string, unknown>;
  const metricsRoot = ((rpt as any)?.metrics ?? rpt) as Record<string, unknown>;

  const sub = (key: string) => {
    const nested = (metricsRoot as any)[key];
    if (nested && typeof nested === 'object' && !Array.isArray(nested) && Object.keys(nested).length > 0)
      return nested as any;
    return metricsRoot as any;
  };

  const struct = sub('structure');
  const compute = sub('compute');
  const memory = sub('memory');
  const hardware = sub('hardware');
  const parallelism = sub('parallelism');
  const performance = sub('performance');
  const cost = sub('cost');
  const graph = sub('graph');
  const dynamic = (metricsRoot.dynamic ?? {}) as any;

  // Extract top-level model config & report metadata
  const modelSection = (rpt.model ?? {}) as Record<string, unknown>;
  const reportMeta = {
    analysisTimeMs: typeof (rpt as any)?.analysis_time_ms === 'number' ? (rpt as any).analysis_time_ms as number : undefined,
    generatedAt: typeof (rpt as any)?.generated_at === 'string' ? (rpt as any).generated_at as string : undefined,
  };

  const formatFlopsHuman = (flops: number): string => {
    if (!Number.isFinite(flops) || flops <= 0) return '0 FLOPs';
    if (flops >= 1e12) return `${(flops / 1e12).toFixed(2)} TFLOPs`;
    if (flops >= 1e9) return `${(flops / 1e9).toFixed(2)} GFLOPs`;
    if (flops >= 1e6) return `${(flops / 1e6).toFixed(2)} MFLOPs`;
    if (flops >= 1e3) return `${(flops / 1e3).toFixed(2)} KFLOPs`;
    return `${flops.toFixed(0)} FLOPs`;
  };

  const formatBytesGb = (bytes: number): string => {
    if (!Number.isFinite(bytes) || bytes <= 0) return '0 GB';
    const gb = bytes / 1e9;
    return `${gb.toFixed(gb >= 10 ? 1 : 2)} GB`;
  };

  const forwardFlops = compute.forward_flops ?? 0;
  const backwardFlops = compute.backward_flops ?? 0;
  const totalFlops = compute.total_flops ?? forwardFlops;
  const peakVramBytes = memory.peak_vram_bytes ?? 0;
  const throughputGraphsPerS = typeof performance.throughput_graphs_per_s === 'number'
    ? performance.throughput_graphs_per_s
    : null;

  const compilationRaw = (((rpt as any)?.compilation ?? {}) as Record<string, unknown>);
  const phaseTimelineRaw = Array.isArray((rpt as any)?.phase_timeline)
    ? (rpt as any).phase_timeline
    : Array.isArray((compilationRaw as any)?.phase_timeline)
      ? (compilationRaw as any).phase_timeline
      : [];
  const normalizedPhaseTimeline = phaseTimelineRaw.map((phase: any) => ({
    name: typeof phase?.name === 'string' ? phase.name : 'Unknown',
    duration_ms: typeof phase?.duration_ms === 'number' ? phase.duration_ms : 0,
    status: typeof phase?.status === 'string' ? phase.status.toLowerCase() : 'completed',
  }));
  const hasCompilationPayload =
    typeof compilationRaw.current_phase === 'string'
    || typeof compilationRaw.total_progress === 'number'
    || normalizedPhaseTimeline.length > 0;
  const compilation = hasCompilationPayload
    ? {
      current_phase: typeof compilationRaw.current_phase === 'string'
        ? compilationRaw.current_phase
        : 'Completed',
      total_progress: typeof compilationRaw.total_progress === 'number'
        ? compilationRaw.total_progress
        : 1,
      phase_timeline: normalizedPhaseTimeline,
    }
    : undefined;

  const liveTraceRaw = (((rpt as any)?.live_trace ?? {}) as Record<string, unknown>);
  const normalizeTupleSeries = (value: unknown): [number, number][] =>
    Array.isArray(value)
      ? value
        .map((entry) => {
          if (!Array.isArray(entry) || entry.length < 2) return null;
          const x = entry[0];
          const y = entry[1];
          if (typeof x !== 'number' || typeof y !== 'number') return null;
          return [x, y] as [number, number];
        })
        .filter((entry): entry is [number, number] => entry !== null)
      : [];

  const memoryLiveness = Array.isArray(liveTraceRaw.memory_liveness)
    ? liveTraceRaw.memory_liveness
      .map((entry: any) => (
        typeof entry?.[0] === 'number' && typeof entry?.[1] === 'number'
          ? { step: entry[0], value: entry[1] }
          : (typeof entry?.step === 'number' && typeof entry?.value === 'number'
            ? { step: entry.step, value: entry.value }
            : null)
      ))
      .filter((entry): entry is { step: number; value: number } => entry !== null)
    : [];
  const memoryHeatmap = Array.isArray(liveTraceRaw.memory_heatmap)
    ? liveTraceRaw.memory_heatmap
      .map((entry: any) => (
        typeof entry?.layer === 'string' && Array.isArray(entry?.timeline)
          ? {
            layer: entry.layer,
            timeline: entry.timeline.filter((step: unknown): step is number => typeof step === 'number'),
          }
          : null
      ))
      .filter((entry): entry is { layer: string; timeline: number[] } => entry !== null)
    : [];
  const gradientRaw = Array.isArray((metricsRoot as any)?.gradient_memory_per_layer)
    ? (metricsRoot as any).gradient_memory_per_layer
    : Array.isArray(liveTraceRaw.gradient_memory_breakdown)
      ? liveTraceRaw.gradient_memory_breakdown
      : [];
  const gradientMemoryBreakdown = gradientRaw
    .map((entry: any) => (
      typeof entry?.name === 'string'
        ? {
          name: entry.name,
          forward: typeof entry?.forward === 'number' ? entry.forward : 0,
          backward: typeof entry?.backward === 'number' ? entry.backward : 0,
        }
        : null
    ))
    .filter((entry: any): entry is { name: string; forward: number; backward: number } => entry !== null);
  const kvRaw = Array.isArray((metricsRoot as any)?.kv_cache_scaling)
    ? (metricsRoot as any).kv_cache_scaling
    : Array.isArray(liveTraceRaw.kv_cache_scaling)
      ? liveTraceRaw.kv_cache_scaling
      : [];
  const kvCacheScaling = kvRaw
    .map((entry: any) => (
      typeof entry?.seq === 'number' && typeof entry?.value === 'number'
        ? { seq: entry.seq, value: entry.value }
        : (typeof entry?.[0] === 'number' && typeof entry?.[1] === 'number'
          ? { seq: entry[0], value: entry[1] }
          : null)
    ))
    .filter((entry: any): entry is { seq: number; value: number } => entry !== null);
  const liveTrace = {
    partial_metrics: normalizeTupleSeries(liveTraceRaw.partial_metrics),
    throughput_trace: normalizeTupleSeries(liveTraceRaw.throughput_trace),
    memory_liveness: memoryLiveness,
    memory_heatmap: memoryHeatmap,
    gradient_memory_breakdown: gradientMemoryBreakdown,
    kv_cache_scaling: kvCacheScaling,
  };

  const normalizedDiagnostics = (Array.isArray((rpt as any)?.diagnostics) ? (rpt as any).diagnostics : [])
    .map((diag: any) => ({
      category: typeof diag?.category === 'string' ? diag.category.toLowerCase() : 'general',
      severity: typeof diag?.severity === 'string' ? diag.severity.toLowerCase() : 'info',
      code: typeof diag?.code === 'string' ? diag.code : undefined,
      message: typeof diag?.message === 'string' ? diag.message : 'Unknown diagnostic',
      layer_id: typeof diag?.layer_id === 'string' ? diag.layer_id : undefined,
      suggestion: typeof diag?.suggestion === 'string' ? diag.suggestion : undefined,
      precision_impact: typeof diag?.precision_impact === 'number' ? diag.precision_impact : undefined,
    }));

  const recommendations = (Array.isArray((rpt as any)?.recommendations) ? (rpt as any).recommendations : [])
    .map((rec: any) => ({
      category: typeof rec?.category === 'string' ? rec.category.toLowerCase() : 'general',
      priority: typeof rec?.priority === 'string' ? rec.priority.toLowerCase() : 'medium',
      title: typeof rec?.title === 'string' ? rec.title : 'Recommendation',
      description: typeof rec?.description === 'string' ? rec.description : '',
      impact: typeof rec?.impact === 'string' ? rec.impact : '',
    }));
  const reportWarnings = (Array.isArray((rpt as any)?.warnings) ? (rpt as any).warnings : [])
    .filter((warning: unknown): warning is string => typeof warning === 'string' && warning.trim().length > 0);

  const analysis: AnalysisResult = {
    modelName: typeof modelSection.name === 'string' ? modelSection.name : undefined,
    totalParams: struct.total_parameters ?? 0,
    // Parameters actually touched per token — equal to totalParams for a
    // dense model, smaller for a mixture of experts. Falls back to
    // totalParams if the backend didn't send one, which is the correct
    // value for a dense model anyway.
    activeParams: struct.active_parameters ?? struct.total_parameters ?? 0,
    numLayers: struct.num_layers ?? 0,
    modelType: struct.model_type ?? '',
    hiddenSize: struct.hidden_size ?? 0,
    vocabSize: struct.vocab_size ?? 0,
    sequenceLength: typeof modelSection.sequence_length === 'number' ? modelSection.sequence_length as number : undefined,
    numAttentionHeads: typeof modelSection.num_attention_heads === 'number' ? modelSection.num_attention_heads as number : undefined,
    numKeyValueHeads: typeof modelSection.num_key_value_heads === 'number' ? modelSection.num_key_value_heads as number : undefined,
    intermediateSize: typeof modelSection.intermediate_size === 'number' ? modelSection.intermediate_size as number : undefined,
    layersByType: (struct.layers_by_type && typeof struct.layers_by_type === 'object' && !Array.isArray(struct.layers_by_type))
      ? struct.layers_by_type as Record<string, number>
      : undefined,
    analysisTimeMs: reportMeta.analysisTimeMs,
    generatedAt: reportMeta.generatedAt,
    graphDepth: graph.graph_depth ?? 0,
    totalOperations: graph.total_operations ?? 0,
    criticalPathLength: graph.critical_path_length ?? 0,
    tensorResolutionRatio: graph.tensor_resolution_ratio ?? 1,
    unresolvedDimCount: graph.unresolved_dim_count ?? 0,
    totalTensorCount: graph.total_tensor_count ?? 0,
    largestTensorBytes: graph.largest_tensor_bytes ?? 0,
    opsDistribution: compute.ops_distribution ?? compute.op_type_distribution ?? {},

    totalFlops,
    forwardFlops,
    backwardFlops,
    flopsPerToken: compute.flops_per_token ?? 0,
    flopsIncrementalDecode: compute.flops_incremental_decode ?? 0,
    arithmeticIntensity: compute.arithmetic_intensity ?? 0,
    macs: compute.macs ?? 0,
    totalStepFlops: compute.total_step_flops ?? 0,
    flopsPerBatch: compute.flops_per_batch ?? 0,
    bytesAccessed: compute.bytes_accessed ?? 0,
    bottleneck: performance.bottleneck ?? '',
    rooflinePosition: compute.roofline_position ?? performance.roofline_position ?? 0,

    estimatedFlops: formatFlopsHuman(totalFlops),
    forwardFlopsHuman: formatFlopsHuman(forwardFlops),
    backwardFlopsHuman: formatFlopsHuman(backwardFlops),

    peakVramBytes,
    parameterMemoryBytes: memory.parameter_memory_bytes ?? 0,
    activationMemoryBytes: memory.activation_memory_bytes ?? 0,
    gradientMemoryBytes: memory.gradient_memory_bytes ?? 0,
    optimizerStateBytes: memory.optimizer_state_bytes ?? 0,
    maxBatchSizeFit: memory.max_batch_size_fit ?? 0,
    memoryFragmentation: (dynamic.virtual_memory?.fragmentation_pct ?? memory.memory_fragmentation_pct ?? 0) / 100,
    memoryFragmentationPct: memory.memory_fragmentation_pct ?? dynamic.virtual_memory?.fragmentation_pct ?? 0,
    // `?? 'low'` here used to silently assert the comfortable answer
    // whenever the backend sent no oom_risk at all — the Memory tab then
    // showed e.g. "OOM Risk: Low" next to a live "Utilization: 257%"
    // computed from the same peakVramBytes/gpu_memory_gb two lines below.
    // Derive the same tier from that number instead of guessing low.
    oomRisk: memory.oom_risk ?? (
      hardware.gpu_memory_gb > 0
        ? (() => {
          const utilization = peakVramBytes / 1e9 / hardware.gpu_memory_gb;
          return utilization >= 1 ? 'high' : utilization >= 0.85 ? 'medium' : 'low';
        })()
        : 'low'
    ),
    memoryUsage: formatBytesGb(peakVramBytes),

    gpuName: hardware.gpu_name ?? '',
    gpuCount: hardware.gpu_count ?? 1,
    gpuMemoryGb: hardware.gpu_memory_gb ?? 0,
    gpuTflops: hardware.gpu_tflops_fp16 ?? 0,
    gpuBandwidthGbs: hardware.gpu_memory_bandwidth_gbs ?? 0,
    interconnect: hardware.interconnect ?? '',
    interconnectBandwidthGbs: hardware.interconnect_bandwidth_gbs ?? 0,

    dataParallelEfficiency: parallelism.data_parallel_efficiency ?? 1,
    communicationOverhead: parallelism.communication_overhead ?? 0,
    optimalGpuCount: parallelism.optimal_gpu_count ?? 1,
    pipelineStages: parallelism.pipeline_stages ?? parallelism.pipeline_parallel ?? 1,
    tensorParallelDegree: parallelism.tensor_parallel_degree ?? parallelism.tensor_parallel ?? 1,
    dataParallel: parallelism.data_parallel ?? 1,
    tensorParallel: parallelism.tensor_parallel ?? 1,
    pipelineParallel: parallelism.pipeline_parallel ?? 1,

    latencyMs: performance.latency_ms ?? null,
    throughputTokensPerS: performance.throughput_tokens_per_s ?? 0,
    throughputGraphsPerS,
    gpuUtilization: performance.gpu_utilization ?? null,
    tensorCoreUtilization: performance.tensor_core_utilization ?? 0,
    effectiveTflops: performance.effective_tflops ?? 0,
    samplesPerS: performance.samples_per_s ?? 0,

    trainingCostUsd: cost.training_cost_usd ?? 0,
    trainingTimeHours: cost.training_time_hours ?? 0,
    energyKwh: cost.energy_kwh ?? 0,
    co2Kg: cost.co2_kg ?? 0,
    costPerMillionTokensUsd: cost.cost_per_million_tokens_usd ?? 0,
    gpuHours: cost.gpu_hours ?? 0,
    provider: cost.provider ?? '',

    selectedPrecision: precision,
    selectedBatchSize: batchSize,

    confidenceScore: (rpt as any)?.confidence_score ?? (metricsRoot as any)?.confidence_score ?? 1.0,
    depth: (rpt as any)?.depth ?? 1,
    isSequenceModel: ((metricsRoot as any)?.is_sequence_model ?? true),
    customLayerCount: (metricsRoot as any)?.custom_layer_count ?? 0,
    diagnosticCount: normalizedDiagnostics.length,
    reportWarnings,
    recommendations,

    compilation,
    live_trace: liveTrace,
    memory_liveness: memoryLiveness,
    memory_heatmap: memoryHeatmap,
    gradient_memory_breakdown: gradientMemoryBreakdown,
    kv_cache_scaling: kvCacheScaling,
    diagnostics: normalizedDiagnostics,

    dynamic: {
      virtual_memory: dynamic.virtual_memory ? {
        fragmentation_overhead_gb: dynamic.virtual_memory.fragmentation_overhead_gb ?? 0,
        fragmentation_pct: dynamic.virtual_memory.fragmentation_pct ?? 0,
        defrag_savings_gb: dynamic.virtual_memory.defrag_savings_gb ?? 0,
        virtual_savings_gb: dynamic.virtual_memory.virtual_savings_gb ?? 0,
        virtual_savings_pct: dynamic.virtual_memory.virtual_savings_pct ?? 0,
        peak_vram_with_defrag_gb: dynamic.virtual_memory.peak_vram_with_defrag_gb ?? 0,
        peak_vram_with_virtual_gb: dynamic.virtual_memory.peak_vram_with_virtual_gb ?? 0,
        recommended_strategy: dynamic.virtual_memory.recommended_strategy ?? 'NoAction',
        confidence: dynamic.virtual_memory.confidence ?? 0,
      } : undefined,
      stability: dynamic.stability ? {
        lyapunov_exponent_mean: dynamic.stability.lyapunov_exponent_mean ?? 0,
        chaos_index: dynamic.stability.chaos_index ?? 0,
        high_risk_layers_count: dynamic.stability.high_risk_layers_count ?? 0,
        fp32_required_pct: dynamic.stability.fp32_required_pct ?? 0,
        global_robustness_score: dynamic.stability.global_robustness_score ?? 1.0,
        fp32_fallback_memory_overhead_gb: dynamic.stability.fp32_fallback_memory_overhead_gb ?? 0,
        confidence: dynamic.stability.confidence ?? 0,
      } : undefined,
      behavioral: dynamic.behavioral ? {
        expert_load_imbalance: dynamic.behavioral.expert_load_imbalance ?? 0,
        has_moe: dynamic.behavioral.has_moe ?? false,
        load_balance_efficiency: dynamic.behavioral.load_balance_efficiency ?? 100,
      } : undefined,
    },
  };

  // Per-layer breakdown
  const perLayerMetrics: Record<string, number> = compute.flops_per_layer ?? {};
  const perLayerParams: Record<string, number> = struct.params_per_layer ?? {};
  const perLayerLatency: Record<string, number> = performance.latency_per_layer ?? {};
  const perLayerVram: Record<string, number> = memory.vram_per_layer ?? {};

  const allLayerIds = Array.from(new Set([
    ...Object.keys(perLayerMetrics),
    ...Object.keys(perLayerParams),
    ...Object.keys(perLayerLatency),
    ...Object.keys(perLayerVram),
  ]));

  // The backend keys every per-layer map by the compiler's internal node id
  // ("n8") — the canvas node carries the real, human name ("Residual Add")
  // the user actually gave the block. Falling back to the id itself only
  // when a node can't be found keeps this honest rather than inventing a
  // name for something that isn't on the canvas at all.
  const nameById = new Map(nodes.map((n) => [n.id, n.name]));

  const perLayer: PerLayerBreakdownRow[] = allLayerIds.map(id => {
    const vramBytes = perLayerVram[id] ?? 0;
    const memStr = vramBytes >= 1e9
      ? `${(vramBytes / 1e9).toFixed(2)} GB`
      : vramBytes >= 1e6
        ? `${(vramBytes / 1e6).toFixed(1)} MB`
        : vramBytes > 0 ? `${(vramBytes / 1e3).toFixed(1)} KB` : '—';

    const latencyVal = perLayerLatency[id];
    const latencyStr = latencyVal !== undefined
      ? latencyVal < 1 ? `${(latencyVal * 1000).toFixed(1)}µs` : `${latencyVal.toFixed(2)}ms`
      : '—';

    return {
      id,
      name: nameById.get(id) ?? id,
      params: perLayerParams[id] ?? 0,
      flops: formatFlopsHuman(perLayerMetrics[id] ?? 0),
      memory: memStr,
      latency: latencyStr,
    };
  });

  // Warnings
  const confidence = (r as any)?.confidence as
    | { verdict?: string; blocked_reason?: string; }
    | undefined;

  const newWarnings: Warning[] = [];

  if (confidence?.verdict === 'blocked') {
    const reason = confidence.blocked_reason?.trim();
    newWarnings.push({
      id: `blocked:${reason ?? 'unknown'}`,
      type: 'error',
      message: reason ? `Blocked: ${reason}` : 'Blocked: the backend could not analyze this model.',
      hint: 'Provide more complete shapes/params or simplify unknown dimensions, then retry.',
      code: 'BLOCKED',
    });
  }

  for (let i = 0; i < normalizedDiagnostics.length; i++) {
    const d = normalizedDiagnostics[i];
    const sev = d.severity.toLowerCase();
    const type: Warning['type'] =
      sev === 'critical' || sev === 'error'
        ? 'error'
        : sev === 'warning'
          ? 'warning'
          : 'info';
    const code = d.code;
    const message = d.message;
    const id = code ? `${code}:${message}` : `diag-${i}:${message}`;
    newWarnings.push({
      id,
      type,
      message,
      hint: d.suggestion ?? undefined,
      code: code ?? undefined,
      nodeId: d.layer_id,
    });
  }

  for (const warningText of reportWarnings) {
    newWarnings.push({
      id: `warn:${warningText}`,
      type: 'warning',
      message: warningText,
    });
  }

  if (newWarnings.length === 0) {
    newWarnings.push({
      id: 'ok',
      type: 'info',
      message: 'Architecture validated successfully by backend.',
    });
  }

  return { analysis, perLayer, warnings: newWarnings, perLayerLatency, perLayerVram };
}

const Index = () => {
  const { isAuthenticated } = useAuth();
  const navigate = useNavigate();

  // ── Auth guard : seul un utilisateur connecté peut accéder au studio ──
  useEffect(() => {
    if (!isAuthenticated) {
      navigate('/', { replace: true });
    }
  }, [isAuthenticated, navigate]);

  const [nodes, setNodes] = useState<CanvasNode[]>(initialNodes);
  const [connections, setConnections] = useState<Connection[]>(initialConnections);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectionRevision, setSelectionRevision] = useState(0);
  const [analysis, setAnalysis] = useState<AnalysisResult>(initialAnalysis);
  const [warnings, setWarnings] = useState<Warning[]>(initialWarnings);
  const [perLayer, setPerLayer] = useState<PerLayerBreakdownRow[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showNewCanvasDialog, setShowNewCanvasDialog] = useState(false);
  const [showExportPanel, setShowExportPanel] = useState(false);
  const [showImportPanel, setShowImportPanel] = useState(false);
  const [showSharePanel, setShowSharePanel] = useState(false);
  const [showTargetPanel, setShowTargetPanel] = useState(false);
  const [showHyperparametersPanel, setShowHyperparametersPanel] = useState(false);

  const [selectedArchitecture, setSelectedArchitecture] = useState<ArchitectureFamily>('transformer');
  const [activeWorkspaceTab, setActiveWorkspaceTab] = useState<WorkspaceTab>('architecture');
  /**
   * The machine, and the data the model is being built for.
   *
   * Both are seeded from `mockRuntime` while the detection and profiling
   * endpoints do not exist. They are held here rather than inside the Training
   * workspace because neither belongs to it: the hardware strip is on screen in
   * every workspace, and the dataset is what fills in the shape fields the
   * Architecture tab would otherwise ask the user to type.
   */
  /**
   * The machine, detected rather than assumed.
   *
   * This used to be `mockHardware` unconditionally — an RTX 4090 with 22.6 GB
   * free, shown on every machine including ones with no discrete GPU at all.
   * Every memory verdict, every "this fits", was computed against a card the
   * user may not own.
   *
   * Now the service is asked, and the example profile is a labelled fallback
   * for when it is not running: the studio's design surface works with no
   * backend, so a missing service must not blank the toolbar — but it must
   * not be mistaken for a real reading either, which is what
   * `isExampleHardware` is for.
   */
  const [hardwareProfile, setHardwareProfile] = useState<HardwareProfile | null>(null);
  const [isExampleHardware, setIsExampleHardware] = useState(false);
  const [isMeasuring, setIsMeasuring] = useState(false);


  const [datasetProfile, setDatasetProfile] = useState<DatasetProfile | null>(IS_MOCK ? mockImageDataset : null);
  const [activeRightPanelTab, setActiveRightPanelTab] = useState<RightPanelTabId>('architecture');
  const [jumpToIssuesSignal, setJumpToIssuesSignal] = useState(0);
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [isDesktopLayout, setIsDesktopLayout] = useState(true);
  const [currentPresetId, setCurrentPresetId] = useState<string | null>(null);
  const [presetAutoAnalysisTick, setPresetAutoAnalysisTick] = useState(0);
  const [autoAnalysisTick, setAutoAnalysisTick] = useState(0);
  const [groups, setGroups] = useState<NodeGroup[]>([]);
  // Carried through open → save round trips only — Production's own Save
  // button always writes the initialisation it just computed regardless of
  // this. Without this, opening a file that had one and then saving from
  // the Architecture tab (Ctrl+S, not Production's Save) silently dropped
  // it: `currentDesignSnapshot` never mentioned it, so the file it wrote
  // carried no `initialization` section at all.
  const [openedInitialization, setOpenedInitialization] = useState<InitializationRecord | null>(null);
  // The exact `nodes`/`connections` arrays `openedInitialization` was set
  // alongside — not their content, their reference. A later edit produces a
  // new array either way, so comparing by reference is enough to notice the
  // design has moved on since the recipe was captured, without a deep
  // comparison on every keystroke. See the invalidation effect below.
  const initializationSourceRef = useRef<{ nodes: CanvasNode[]; connections: Connection[] } | null>(null);
  // Inference Intelligence and Time Machine keep their own sliders as local
  // state, so this is the copy that actually survives a save: each panel
  // reports its current values here through onChange, and is handed them
  // back as `initialParams`/`initialConfig`. Both panels stay permanently
  // mounted (see WorkspaceTabs), so their `useState` initialiser only runs
  // once — `panelLoadGeneration` below forces a remount when a file is
  // opened, so a saved value actually reaches the sliders rather than only
  // updating state nothing rereads.
  const [timeMachineConfig, setTimeMachineConfig] = useState<TimeMachineConfig | null>(null);
  const [panelLoadGeneration, setPanelLoadGeneration] = useState(0);
  const [savedProjects, setSavedProjects] = useState<Project[]>([]);
  const [currentProjectId, setCurrentProjectId] = useState<string | null>(null);
  const [isProjectsLoading, setIsProjectsLoading] = useState(false);
  const [creditInfo, setCreditInfo] = useState<CreditInfo | null>(null);

  // ── The open document ──────────────────────────────────────────────
  // A design is a file, not just application state. `documentPath` is only ever
  // known on the desktop — a browser hands over contents without revealing a
  // path — which is why saving falls back to Save As on the web.
  const [documentName, setDocumentName] = useState<string | undefined>(undefined);
  const [documentPath, setDocumentPath] = useState<string | undefined>(undefined);
  const [isDirty, setIsDirty] = useState(false);

  // ── A/B comparison ─────────────────────────────────────────────────
  const [comparisonBaseline, setComparisonBaseline] = useState<DesignVariant | null>(null);
  const [showComparePanel, setShowComparePanel] = useState(false);

  // ── Documentation ──────────────────────────────────────────────────
  const [showDocumentation, setShowDocumentation] = useState(false);
  const [docSectionId, setDocSectionId] = useState<string | undefined>(undefined);
  const { toast } = useToast();

  /**
   * Measure this machine's sustained throughput.
   *
   * Explicit rather than automatic: it costs about half a second of real
   * compute, and a studio that quietly ran a benchmark on every page load
   * would be spending the user's machine without being asked. The result is
   * cached by the service, so a second call is free.
   */
  const measureThisMachine = useCallback(async () => {
    if (isExampleHardware || isMeasuring) return;
    setIsMeasuring(true);
    const measured = await measureHardware();
    if (measured) {
      setHardwareProfile(measured);
    } else {
      toast({
        title: 'Could not measure this machine',
        description: 'The local NEURAX service did not answer.',
        variant: 'destructive',
      });
    }
    setIsMeasuring(false);
  }, [isExampleHardware, isMeasuring, toast]);
  const { config: hwConfig, setConfig: setHwConfig, updateConfig: updateHwConfig, triggerAttempt } = useHardware();

  /**
   * Make the detected machine the machine the analysis computes for.
   *
   * Without this the studio detected a laptop and went on computing against
   * `DEFAULT_HARDWARE_CONFIG` — an RTX 4090 with 80 GB, a card that does not
   * exist (a 4090 has 24) on a machine that has no discrete GPU at all. The
   * target panel said "Intel UHD 620, detected" and the footer said
   * "Analysing for RTX4090". Detection that does not reach the analysis is
   * decoration.
   *
   * The memory mapping is the part worth stating. `gpuMemoryGb` is the budget
   * every fit check divides by, so on a machine with a discrete card it is
   * that card's VRAM — and on a CPU-only machine it is *system RAM*, because
   * that is genuinely the memory the work would run in. Leaving the default
   * 80 GB there would tell a 24 GB laptop that anything fits.
   */
  const applyDetectedHardware = useCallback(
    (profile: HardwareProfile) => {
      const gpu = primaryGpu(profile);
      const GB = 1024 ** 3;

      if (gpu && !gpu.integrated && gpu.vramTotalBytes != null) {
        updateHwConfig({
          hardware: gpu.name,
          gpuMemoryGb: Math.round(gpu.vramTotalBytes / GB),
          gpuCount: profile.gpus.filter((g) => !g.integrated).length || 1,
          device: gpu.backend === 'cuda' ? 'cuda' : gpu.backend === 'rocm' ? 'rocm' : 'cpu',
        });
        return;
      }

      // No usable accelerator: the CPU is the target, and system RAM is the
      // budget. Rounded down, because a design that exactly fills available
      // memory does not run.
      updateHwConfig({
        hardware: profile.cpu.model,
        gpuMemoryGb: Math.max(1, Math.floor(profile.ramAvailableBytes / GB)),
        gpuCount: 1,
        device: 'cpu',
      });
    },
    [updateHwConfig],
  );

  useEffect(() => {
    let cancelled = false;
    void detectHardware().then((profile) => {
      if (cancelled) return;
      if (profile) {
        setHardwareProfile(profile);
        setIsExampleHardware(false);
        applyDetectedHardware(profile);
      } else {
        setHardwareProfile(mockHardware);
        setIsExampleHardware(true);
      }
    });
    return () => {
      cancelled = true;
    };
    // `applyDetectedHardware` is stable for the life of the page and this
    // must run exactly once, on load.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const toHwFamily = useCallback((fam: ArchitectureFamily): HwFamily => {
    switch (fam) {
      case 'transformer':
        return 'transformer';
      case 'cnn':
        return 'cnn';
      case 'gnn':
        return 'gnn';
      case 'rnn':
        return 'rnn';
      case 'ssm':
        return 'ssm';
      case 'moe':
        return 'moe';
      case 'diffusion':
        return 'diffusion';
      case 'gan':
        return 'gan';
      default:
        return fam;
    }
  }, []);

  useEffect(() => {
    const mql = window.matchMedia('(min-width: 768px)');
    const update = () => setIsDesktopLayout(mql.matches);
    update();
    mql.addEventListener('change', update);
    return () => mql.removeEventListener('change', update);
  }, []);

  const hasCanvasBlocks = nodes.length > 0 || groups.length > 0;
  const hasCanvasContent = hasCanvasBlocks || connections.length > 0;

  // ─── Undo / redo ─────────────────────────────────────────────
  //
  // The design is three pieces of state edited from a couple of dozen call
  // sites. Rather than route every one through a reducer, the history observes
  // them; see `useDesignHistory`. Memoised because the hook's effect keys off
  // this object's identity — rebuilt every render, the coalescing window would
  // restart on every render too.
  const designSnapshot = useMemo(
    () => ({ nodes, connections, groups }),
    [nodes, connections, groups],
  );

  const applyDesign = useCallback(
    (snapshot: { nodes: CanvasNode[]; connections: Connection[]; groups: NodeGroup[] }) => {
      setNodes(snapshot.nodes);
      setConnections(snapshot.connections);
      setGroups(snapshot.groups);
      // The selected block may not exist in the restored design.
      setSelectedNodeId((current) =>
        current && snapshot.nodes.some((n) => n.id === current) ? current : null,
      );
    },
    [],
  );

  // Destructured because the hook returns a fresh object each render, while the
  // callbacks inside it are stable. Depending on the object would invalidate
  // every handler below on every render.
  const {
    canUndo,
    canRedo,
    undo: undoDesign,
    redo: redoDesign,
    reset: resetHistory,
  } = useDesignHistory(designSnapshot, applyDesign);

  // Any change to the design is a change not yet in the file.
  //
  // Opening a file, loading a template or clearing the canvas also changes the
  // design, and this effect runs *after* those handlers have already set the
  // flag — so without a way to exempt them, a freshly opened document would
  // show as unsaved the instant it appeared. `cleanDesign` records the design
  // that is currently on disk; a change away from it is what makes the document
  // dirty, and a change back to it does not.
  //
  // Compared by the three inner references rather than by the snapshot object,
  // which `useMemo` rebuilds whenever any of them changes: every handler here
  // replaces an array instead of mutating it, so an unchanged reference means
  // genuinely no change.
  const cleanDesign = useRef(designSnapshot);
  useEffect(() => {
    const clean = cleanDesign.current;
    if (
      designSnapshot.nodes === clean.nodes &&
      designSnapshot.connections === clean.connections &&
      designSnapshot.groups === clean.groups
    ) {
      return;
    }
    setIsDirty(true);
  }, [designSnapshot]);

  /** Mark the design now on screen as the one the file holds. */
  const markSaved = useCallback(
    (snapshot: { nodes: CanvasNode[]; connections: Connection[]; groups: NodeGroup[] }) => {
      cleanDesign.current = snapshot;
      setIsDirty(false);
    },
    [],
  );

  const resetWorkspace = useCallback(() => {
    const empty = { nodes: [], connections: [], groups: [] };
    setNodes(empty.nodes);
    setConnections(empty.connections);
    setGroups(empty.groups);
    setSelectedNodeId(null);
    setCurrentPresetId(null);
    setWarnings([]);
    setPerLayer([]);
    setAnalysis(initialAnalysis);
    setActiveWorkspaceTab('architecture');
    initializationSourceRef.current = null;
    setOpenedInitialization(null);
    setTimeMachineConfig(null);
    setPanelLoadGeneration((g) => g + 1);
    pendingConnectionsRef.current.clear();

    // A blank page is a new document: undoing back into the previous one would
    // resurrect work the user just chose to leave.
    resetHistory(empty);
    setDocumentName(undefined);
    setDocumentPath(undefined);
    markSaved(empty);
  }, [resetHistory, markSaved]);

  const downloadCanvasSnapshot = useCallback(() => {
    if (!hasCanvasContent) return null;

    const neuraxIR = compileToNeuraxIR(nodes, connections, {
      modelName: 'NeuraxModel',
      family: selectedArchitecture,
      groups,
      ...hwConfig,
      learningRate: hwConfig.learningRate,
      numEpochs: hwConfig.numEpochs,
      gpuCount: hwConfig.gpuCount,
      gpuMemoryGb: hwConfig.gpuMemoryGb,
      datasetSize: hwConfig.datasetSize,
      numClasses: hwConfig.numClasses,
    });
    const content = JSON.stringify(neuraxIR, null, 2);
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `neurax-${selectedArchitecture}-${timestamp}.neurax.json`;
    const blob = new Blob([content], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);

    return filename;
  }, [hasCanvasContent, nodes, connections, selectedArchitecture, groups, hwConfig]);

  const handleArchitectureChange = useCallback((family: ArchitectureFamily) => {
    setSelectedArchitecture(family);

    toast({
      title: "Architecture Changed",
      description: `Switched to ${family.charAt(0).toUpperCase() + family.slice(1)} analysis mode`,
    });
  }, [toast]);

  const handleLoadPreset = useCallback((preset: VariantPreset) => {
    const hydratedNodes = hydrateNodesForFamily(preset.family, preset.nodes);
    const hydratedPreset: VariantPreset = {
      ...preset,
      nodes: hydratedNodes,
    };

    setSelectedArchitecture(preset.family);
    setNodes(hydratedNodes);
    setConnections(preset.connections);
    setGroups([]);
    setHwConfig(buildHardwareConfigFromPreset(hydratedPreset, hwConfig));
    setCurrentPresetId(preset.id);
    setSelectedNodeId(null);
    setWarnings([]);
    setPerLayer([]);
    setPresetAutoAnalysisTick(tick => tick + 1);

    // Loading a template starts a new document: it has a name to suggest but no
    // file behind it, and undo must not reach back into whatever was open.
    resetHistory({ nodes: hydratedNodes, connections: preset.connections, groups: [] });
    setDocumentName(preset.name);
    setDocumentPath(undefined);
    setIsDirty(true);
    toast({
      title: "Template Loaded",
      description: `Loaded "${preset.name}" — all blocks are editable`,
    });
  }, [hwConfig, setHwConfig, toast, resetHistory]);

  const handleSelectNode = useCallback((id: string | null) => {
    setSelectedNodeId(id);
    if (id) setSelectionRevision(r => r + 1);
  }, []);

  const selectedNode = nodes.find(n => n.id === selectedNodeId) || null;
  const selectedGroup = groups.find(g => g.id === selectedNodeId) || null;

  // Normalize existing nodes whenever architecture family changes to CNN
  useEffect(() => {
    if (selectedArchitecture !== 'cnn') return;

    setNodes(prev => prev.map(n => {
      // Fix input shape for CNN
      if (n.type === 'input') {
        const shape = (n.params as any)?.shape;
        if (shape === '[B, seq_len]' || !shape) {
          return { ...n, params: { ...(n.params ?? {}), shape: '[B, C, H, W]' } };
        }
      }
      // Fix classification_head pooling for CNN
      if (n.type === 'classification_head') {
        const p = (n.params ?? {}) as any;
        if (p?.pooling === 'cls') {
          const next: Record<string, any> = { ...p, pooling: 'avg' };
          if (next.in_features == null && typeof p.d_model === 'number') {
            next.in_features = p.d_model;
          }
          return { ...n, params: next };
        }
      }
      return n;
    }));
  }, [selectedArchitecture]);

  const addNodeFromConfig = useCallback((config: LayerConfig, x: number, y: number, forcedId?: string) => {
    const requestedId = forcedId && forcedId.trim().length > 0 ? forcedId.trim() : '';
    let created: CanvasNode | null = null;

    setNodes(prev => {
      if (requestedId) {
        const existing = prev.find(n => n.id === requestedId);
        if (existing) {
          created = existing;
          return prev;
        }
      }

      const id = requestedId || _uniqueId(config.type);

      // Normalize params based on current architecture family
      let params = {
        ...getBlockDefaults(config.type),
        ...config.defaultParams,
      };
      if (config.hasActivation && !('activation' in params)) {
        params = { ...params, activation: 'none' };
      }
      if (selectedArchitecture === 'cnn') {
        if (config.type === 'input') {
          const shape = (params as any)?.shape;
          if (shape === '[B, seq_len]' || !shape) {
            params = { ...params, shape: '[B, C, H, W]' };
          }
        }
        if (config.type === 'classification_head') {
          const p = params as any;
          if (p?.pooling === 'cls') {
            params = { ...p, pooling: 'avg' };
            if (params.in_features == null && typeof p.d_model === 'number') {
              params.in_features = p.d_model;
            }
          }
        }
      }

      const newNode: CanvasNode = {
        id,
        type: config.type,
        name: config.name,
        x,
        y,
params: params as Record<string, ParameterValue>,
        inputShape: 'auto',
        outputShape: 'auto',
      };
      created = newNode;
      return [...prev, newNode];
    });

    return created;
  }, [selectedArchitecture]);

  const handleAddNode = useCallback((config: LayerConfig, x: number, y: number) => {
    addNodeFromConfig(config, x, y);
    toast({
      title: "Layer added",
      description: `${config.name} layer added to canvas`,
    });
  }, [addNodeFromConfig, toast]);

  const handleUpdateNode = useCallback((id: string, updates: Partial<CanvasNode>) => {
    setNodes(prev =>
      prev.map(node =>
        node.id === id ? { ...node, ...updates } : node
      )
    );
  }, []);

  const handleDeleteNode = useCallback((id: string) => {
    const node = nodes.find(n => n.id === id);
    setNodes(prev => prev.filter(n => n.id !== id));
    setConnections(prev => prev.filter(c => c.from !== id && c.to !== id));
    // Clean up any groups that referenced this node
    setGroups(prev => prev
      .map(g => ({
        ...g,
        nodeIds: g.nodeIds.filter(nid => nid !== id),
        connectionIds: g.connectionIds.filter(() => {
          // We don't have connection objects here easily, so keep all connection IDs
          return true;
        }),
      }))
      // Dissolve groups that have fewer than 2 nodes left
      .filter(g => g.nodeIds.length >= 2)
    );
    setSelectedNodeId(null);
    toast({
      title: "Layer deleted",
      description: `${node?.name || 'Layer'} removed from architecture`,
    });
  }, [nodes, toast]);

  const handleDuplicateNode = useCallback((id: string) => {
    const node = nodes.find(n => n.id === id);
    if (!node) return;

    const hydratedNode = hydrateNodesForFamily(selectedArchitecture, [node])[0] ?? node;
    const newNode: CanvasNode = {
      ...hydratedNode,
      id: _uniqueId(node.type),
      name: `${node.name}_copy`,
      x: node.x + 30,
      y: node.y + 30,
    };
    setNodes(prev => [...prev, newNode]);
    setSelectedNodeId(newNode.id);
    toast({
      title: "Layer duplicated",
      description: `Created copy of ${node.name}`,
    });
  }, [nodes, selectedArchitecture, toast]);

  const pendingConnectionsRef = useRef<Set<string>>(new Set());

  const handleAddConnection = useCallback((from: string, to: string, force = false) => {
    const toNode = nodes.find(n => n.id === to) || (groups.find(g => g.id === to) as any);
    const toNodeType = toNode?.type;

    // Synchronized with backend MERGE_BLOCK_TYPES (neurax-agent/block_constraints.json's
    // merge_capable_types — checked by Index.fanin.test.ts against that file directly)
    const isFanInCapable = [
      'concat', 'merge', 'add', 'residual', 'residual_add', 'skip_connection',
      'expert_combine', 'gate', 'lm_head', 'moe_block', 'unet_block', 'router_softmax',
      'output_combination'
    ].includes(toNodeType || '');

    // Allow connection if:
    // 1. Target node is not yet known (defensive against state lag)
    // 2. OR Target is fan-in capable
    // 3. OR it currently has 0 connections
    const allowsFanin = !toNodeType || isFanInCapable;

    // Check both committed connections and pending connections (synchronous guard)
    const existingIncoming = connections.filter(c => c.to === to).length;
    const pendingIncoming = pendingConnectionsRef.current.has(to) ? 1 : 0;
    const totalIncoming = existingIncoming + pendingIncoming;

    const exists = connections.some(c => c.from === from && c.to === to);
    if (exists && !force) {
      toast({
        title: "Connection exists",
        description: "These layers are already connected",
        variant: "destructive",
      });
      return;
    }

    if (totalIncoming > 0 && !allowsFanin && !force) {
      console.warn(`[neurax-ui] Connection to ${to} rejected: fan-in not allowed for type ${toNodeType}`);
      toast({
        title: "Invalid connection",
        description: `${toNode?.name || 'Layer'} can only have one incoming edge. Use a merge block to combine paths.`,
        variant: "destructive",
      });
      return;
    }

    if (from === to) {
      toast({
        title: "Invalid connection",
        description: "Cannot connect a layer to itself",
        variant: "destructive",
      });
      return;
    }

    // Mark this target as having a pending connection (synchronous)
    pendingConnectionsRef.current.add(to);

    const newConnection: Connection = {
      id: _uniqueId('conn'),
      from,
      to,
    };

    setConnections(prev => {
      // Double-check inside the functional update (latest state)
      // If 'force' is true, we trust the caller (agent) and skip the manual fan-in guard.
      const latestIncoming = prev.filter(c => c.to === to).length;
      if (!force && latestIncoming > 0 && !allowsFanin) {
        pendingConnectionsRef.current.delete(to);
        return prev;
      }
      return [...prev, newConnection];
    });

    // Clear pending after a tick
    setTimeout(() => pendingConnectionsRef.current.delete(to), 0);

    // Look up name from nodes OR groups
    const fromName = nodes.find(n => n.id === from)?.name ?? groups.find(g => g.id === from)?.name ?? from;
    const toName = nodes.find(n => n.id === to)?.name ?? groups.find(g => g.id === to)?.name ?? to;
    toast({
      title: "Connection created",
      description: `${fromName} → ${toName}`,
    });
  }, [connections, nodes, groups, toast]);

  const handleDeleteConnection = useCallback((id: string) => {
    const connection = connections.find(c => c.id === id);
    if (!connection) return;

    const fromNode = nodes.find(n => n.id === connection.from);
    const toNode = nodes.find(n => n.id === connection.to);

    setConnections(prev => prev.filter(c => c.id !== id));
    toast({
      title: "Connection deleted",
      description: `Removed ${fromNode?.name} → ${toNode?.name}`,
    });
  }, [connections, nodes, toast]);

  // ─── Group handlers ──────────────────────────────────────────
  const handleGroupSelected = useCallback((nodeIds: string[]) => {
    if (nodeIds.length < 2) return;

    const selectedNodes = nodes.filter(n => nodeIds.includes(n.id));
    const avgX = selectedNodes.reduce((s, n) => s + n.x, 0) / selectedNodes.length;
    const avgY = selectedNodes.reduce((s, n) => s + n.y, 0) / selectedNodes.length;

    // Find internal connections (both endpoints in the group)
    const nodeIdSet = new Set(nodeIds);
    const internalConns = connections.filter(c => nodeIdSet.has(c.from) && nodeIdSet.has(c.to));

    const group: NodeGroup = {
      id: `group-${Date.now()}`,
      name: `Group (${selectedNodes.length} blocks)`,
      nodeIds: [...nodeIds],
      connectionIds: internalConns.map(c => c.id),
      repeatCount: 1,
      x: avgX,
      y: avgY,
      collapsed: true,
    };

    setGroups(prev => [...prev, group]);
    toast({
      title: "Group created",
      description: `Grouped ${nodeIds.length} blocks — adjust ×N to repeat`,
    });
  }, [nodes, connections, toast]);

  const handleUngroupGroup = useCallback((groupId: string) => {
    setGroups(prev => prev.filter(g => g.id !== groupId));
    toast({ title: "Group dissolved", description: "Blocks are now individual again" });
  }, [toast]);

  const handleDeleteGroup = useCallback((groupId: string) => {
    const group = groups.find(g => g.id === groupId);
    if (!group) return;
    // Delete all nodes in the group
    setNodes(prev => prev.filter(n => !group.nodeIds.includes(n.id)));
    setConnections(prev => prev.filter(c => !group.nodeIds.includes(c.from) && !group.nodeIds.includes(c.to)));
    setGroups(prev => prev.filter(g => g.id !== groupId));
    setSelectedNodeId(null);
    toast({ title: "Group deleted", description: `Removed ${group.nodeIds.length} blocks` });
  }, [groups, toast]);

  const handleUpdateGroup = useCallback((groupId: string, updates: Partial<NodeGroup>) => {
    setGroups(prev => prev.map(g => g.id === groupId ? { ...g, ...updates } : g));
  }, []);

  const handleRunAnalysis = useCallback(async () => {
    // 1. Mandatory Hyperparameter Validation
    const validation = validateHardwareConfig(hwConfig, toHwFamily(selectedArchitecture));
    if (!validation.isValid) {
      // Trigger visual feedback in the panel
      triggerAttempt();

      // Update warnings to show errors
      setWarnings(validation.missingFields.map(field => ({
        id: `missing-${field}`,
        type: 'error',
        message: `Mandatory Hyperparameter: ${field} is unset or zero.`,
        code: 'E_MISSING_HYPERPARAMETER',
      })));

      return;
    }

    setIsAnalyzing(true);
    // Every message below is appended onto whatever this run finds, not
    // replaced — without clearing first, a warning from a run three clicks
    // ago is still on screen, and gets a duplicate of itself added next to
    // it on every subsequent run that reproduces the same issue.
    setWarnings([]);

    try {
      // Compile canvas to NEURAX IR topology
      const ir = compileToNeuraxIR(nodes, connections, {
        modelName: 'NeuraxModel',
        family: selectedArchitecture,
        groups,
        // Spread the whole hyperparameter set: listing fields by hand silently
        // dropped every family-specific one (dropout, kvHeads, ropeTheta,
        // numExperts, dState, ...), so the compiler analysed a model the user
        // had not configured.
        ...hwConfig,
      });

      const compilerWarnings: string[] = (ir as any)?._warnings ?? [];
      if (compilerWarnings.length > 0) {
        setWarnings(prev => [
          ...prev,
          ...compilerWarnings.map((w, i) => ({
            id: `compiler-${i}`,
            type: 'warning' as const,
            code: 'COMPILER_AUTO_FIX' as const,
            message: w,
          })),
        ]);
      }

      // Send to backend — topology IS the full IR (env already embedded)
      const { report } = await analyze({
        topology: ir as unknown as Record<string, unknown>,
      });

      // Parse report using shared helper
      const parsed = parseAnalysisReport(report, hwConfig.precision, hwConfig.batchSize, nodes);
      setAnalysis(parsed.analysis);
      setPerLayer(parsed.perLayer);
      setWarnings(prev => [...prev, ...parsed.warnings]);

      // Backfill perLayerLatency and perLayerVram into the analysis state
      if (Object.keys(parsed.perLayerLatency).length > 0 || Object.keys(parsed.perLayerVram).length > 0) {
        setAnalysis(prev => prev ? {
          ...prev,
          perLayerLatency: parsed.perLayerLatency,
          perLayerVram: parsed.perLayerVram,
        } : prev);
      }

      toast({
        title: "Analysis complete",
        description: `Found ${parsed.warnings.filter(w => w.type === 'error').length} errors, ${parsed.warnings.filter(w => w.type === 'warning').length} warnings`,
      });
      // Trigger success toast only if there are no errors
      if (parsed.warnings.some(w => w.type === 'error')) {
        toast({
          title: "Compilation warnings",
          description: `Architecture has ${parsed.warnings.filter(w => w.type === 'error').length} design issues. See issues tab for details.`,
        });
      } else {
        toast({
          title: "Analysis complete",
          description: `Found ${parsed.warnings.length} architectural warnings. Performance metrics are now live.`,
        });
      }
    } catch (err) {
      console.error('[neurax] Analysis failed:', err);

      // Everything the compiler said reaches the user, translated into
      // something actionable where it is recognised and shown verbatim where it
      // is not. See `explainAnalysisFailure`.
      const failure = explainAnalysisFailure(err);

      // A failed analysis has no numbers. Leaving the previous run's figures on
      // screen beside an error is how someone ends up quoting a stale cost.
      setAnalysis(initialAnalysis);
      setPerLayer([]);

      // Local checks add the causes the compiler never gets to see, because a
      // design missing an input never reaches it.
      const localWarnings: Warning[] = [];
      if (!nodes.some((n) => n.type === 'input')) {
        localWarnings.push({
          id: 'no-input',
          type: 'error',
          message: 'Missing Input block — nothing marks where data enters the model.',
        });
      }
      if (!nodes.some((n) => n.type === 'output')) {
        localWarnings.push({
          id: 'no-output',
          type: 'error',
          message: 'Missing Output block — nothing marks where the model produces its result.',
        });
      }

      setWarnings([...failureAsWarnings(failure), ...localWarnings]);

      if (failure.diagnostics.length > 0) {
        setAnalysis((prev) => ({
          ...prev,
          diagnostics: failure.diagnostics,
          diagnosticCount: failure.diagnostics.length,
        }));
      }

      toast({
        title: failure.title,
        description: failure.hint ? `${failure.detail} ${failure.hint}` : failure.detail,
        variant: 'destructive',
      });
    } finally {
      setIsAnalyzing(false);
    }
  }, [nodes, connections, groups, selectedArchitecture, hwConfig, toast, triggerAttempt, toHwFamily]);

  // ─── Streaming Analysis Handler ────────────────────────────────────────
  // Uses SSE to show real-time compilation progress, then falls back to
  // synchronous analysis if streaming is unavailable.
  const handleRunAnalysisStream = useCallback(async () => {
    const validation = validateHardwareConfig(hwConfig, toHwFamily(selectedArchitecture));
    if (!validation.isValid) {
      triggerAttempt();
      setWarnings(validation.missingFields.map(field => ({
        id: `missing-${field}`,
        type: 'error',
        message: `Mandatory Hyperparameter: ${field} is unset or zero.`,
        code: 'E_MISSING_HYPERPARAMETER',
      })));
      return;
    }

    setIsAnalyzing(true);
    // See handleRunAnalysis: without this, a warning from an earlier run
    // stays on screen and accumulates a duplicate of itself on every
    // subsequent run that still reproduces the same issue.
    setWarnings([]);

    try {
      const ir = compileToNeuraxIR(nodes, connections, {
        modelName: 'NeuraxModel',
        family: selectedArchitecture,
        groups,
        // See handleRunAnalysis: spread so no hyperparameter is dropped.
        ...hwConfig,
      });

      const cw: string[] = (ir as any)?._warnings ?? [];
      if (cw.length > 0) {
        setWarnings(prev => [
          ...prev,
          ...cw.map((w, i) => ({
            id: `compiler-${i}`,
            type: 'warning' as const,
            code: 'COMPILER_AUTO_FIX' as const,
            message: w,
          })),
        ]);
      }

      // Set initial compilation state
      setAnalysis(prev => prev ? {
        ...prev,
        compilation: {
          current_phase: 'Initializing',
          total_progress: 0,
          phase_timeline: [],
        },
      } : prev);

      await new Promise<void>((resolve, reject) => {
        analyzeStream(
          { topology: ir as unknown as Record<string, unknown> },
          {
            onStarted: () => {
              setAnalysis(prev => prev ? {
                ...prev,
                compilation: { current_phase: 'Started', total_progress: 0.05, phase_timeline: [] },
              } : prev);
            },
            onPhaseStarted: (phase) => {
              setAnalysis(prev => {
                const existing = prev?.compilation?.phase_timeline ?? [];
                const progress = phase.total_phases > 0 ? phase.phase_index / phase.total_phases : 0;
                return prev ? {
                  ...prev,
                  compilation: {
                    current_phase: phase.phase,
                    total_progress: progress,
                    phase_timeline: [
                      ...existing,
                      { name: phase.phase, duration_ms: 0, status: 'running' },
                    ],
                  },
                } : prev;
              });
            },
            onPhaseCompleted: (phase) => {
              setAnalysis(prev => {
                const existing = prev?.compilation?.phase_timeline ?? [];
                const updated = existing.map(p =>
                  p.name === phase.phase && p.status === 'running'
                    ? { ...p, status: 'completed', duration_ms: phase.duration_ms }
                    : p
                );
                // If no running entry was found, add it as completed
                if (!existing.some(p => p.name === phase.phase)) {
                  updated.push({ name: phase.phase, duration_ms: phase.duration_ms, status: 'completed' });
                }
                const progress = phase.total_phases > 0 ? phase.phase_index / phase.total_phases : 0;
                return prev ? {
                  ...prev,
                  compilation: {
                    current_phase: phase.phase,
                    total_progress: progress,
                    phase_timeline: updated,
                  },
                } : prev;
              });
            },
            onProgress: (progress) => {
              setAnalysis(prev => prev ? {
                ...prev,
                compilation: {
                  ...prev.compilation!,
                  current_phase: prev.compilation?.current_phase ?? 'Processing',
                  total_progress: progress.progress_pct / 100,
                  phase_timeline: prev.compilation?.phase_timeline ?? [],
                },
              } : prev);
            },
            onDiagnostic: (_diag) => {
              // Diagnostics processed silently
            },
            onCompleted: () => {
              setAnalysis(prev => prev ? {
                ...prev,
                compilation: {
                  current_phase: 'Completed',
                  total_progress: 1,
                  phase_timeline: prev.compilation?.phase_timeline ?? [],
                },
              } : prev);
            },
            onResult: (result) => {
              const parsed = parseAnalysisReport(result, hwConfig.precision, hwConfig.batchSize, nodes);
              setAnalysis(parsed.analysis);
              setPerLayer(parsed.perLayer);
              setWarnings(parsed.warnings);

              if (Object.keys(parsed.perLayerLatency).length > 0 || Object.keys(parsed.perLayerVram).length > 0) {
                setAnalysis(prev => prev ? {
                  ...prev,
                  perLayerLatency: parsed.perLayerLatency,
                  perLayerVram: parsed.perLayerVram,
                } : prev);
              }

              toast({
                title: "Analysis complete",
                description: `Found ${parsed.warnings.filter(w => w.type === 'error').length} errors, ${parsed.warnings.filter(w => w.type === 'warning').length} warnings`,
              });
              if (parsed.warnings.some(w => w.type === 'error')) {
                toast({
                  title: "Compilation warnings",
                  description: `Architecture has ${parsed.warnings.filter(w => w.type === 'error').length} design issues. See issues tab for details.`,
                });
              } else {
                toast({
                  title: "Analysis complete",
                  description: `Found ${parsed.warnings.length} architectural warnings. Performance metrics are now live.`,
                });
              }
              resolve();
            },
            onFailed: (error) => {
              console.error('[neurax] Streaming analysis failed:', error);
              const errorMsg = error.error || 'Streaming analysis failed.';
              setWarnings([{
                id: 'stream-failed',
                type: 'error',
                message: errorMsg,
              }]);
              toast({
                title: "Analysis failed",
                description: errorMsg,
                variant: "destructive",
              });
              reject(new Error(errorMsg));
            },
            onError: (error) => {
              console.error('[neurax] SSE connection error:', error);
              // Fall back to synchronous analysis
              reject(new Error('SSE connection error'));
            },
          },
        );
      });
    } catch (err) {
      console.warn('[neurax] Streaming analysis failed, falling back to synchronous:', err);
      // Fall back to synchronous analysis
      await handleRunAnalysis();
    } finally {
      setIsAnalyzing(false);
    }
  }, [nodes, connections, groups, selectedArchitecture, hwConfig, toast, triggerAttempt, toHwFamily, handleRunAnalysis]);

  useEffect(() => {
    if (presetAutoAnalysisTick <= 0) return;
    const timeout = window.setTimeout(() => {
      void handleRunAnalysis();
    }, 500);
    return () => window.clearTimeout(timeout);
  }, [presetAutoAnalysisTick, handleRunAnalysis]);

  // Auto-analysis when nodes or connections change (debounced)
  const isFirstRender = useRef(true);
  useEffect(() => {
    if (isFirstRender.current) {
      isFirstRender.current = false;
      return;
    }
    if (nodes.length === 0) return;
    const timeout = window.setTimeout(() => {
      setAutoAnalysisTick(t => t + 1);
    }, 800);
    return () => window.clearTimeout(timeout);
  }, [nodes, connections]);

  useEffect(() => {
    if (autoAnalysisTick <= 0) return;
    const timeout = window.setTimeout(() => {
      void handleRunAnalysis();
    }, 200);
    return () => window.clearTimeout(timeout);
  }, [autoAnalysisTick, handleRunAnalysis]);

  const [agentAnalysisTick, setAgentAnalysisTick] = useState(0);

  const triggerAgentAutoAnalysis = useCallback(() => {
    setAgentAnalysisTick((v) => v + 1);
  }, []);

  useEffect(() => {
    if (agentAnalysisTick <= 0) return;
    const t = window.setTimeout(() => {
      void handleRunAnalysis();
    }, 650);
    return () => window.clearTimeout(t);
  }, [agentAnalysisTick, handleRunAnalysis]);

  const agentGetSnapshot = useCallback(() => {
    const preferredOrder: ArchitectureFamily[] = [
      selectedArchitecture,
      ...ALL_ARCHITECTURE_FAMILIES.filter((f) => f !== selectedArchitecture),
    ];

    const allLayers = preferredOrder.flatMap((f) => getPluginLayers(f));
    const allowedLayerTypes = Array.from(new Set(allLayers.map((l) => l.type)));

    // De-duplicate by type, preferring the currently selected architecture family's definition.
    //
    // `description` rides along here rather than as a second, separate
    // payload: the agent's explanation mode (agent_graph.py's
    // `explain_layer_type` tool) reads it straight off each catalogue
    // entry, so there is exactly one place — this one — that ever holds
    // NEURAX's per-block descriptions, instead of a second backend copy
    // that would drift from this file the same way LAYER_TYPE_MAP once did
    // between neurax-mcp and neurax-agent.
    const catalogueByType = new Map<string, { type: string; name: string; category: any; defaultParams: any; description?: string }>();
    for (const l of allLayers) {
      if (!l?.type) continue;
      if (!catalogueByType.has(l.type)) {
        catalogueByType.set(l.type, {
          type: l.type,
          name: l.name,
          category: l.category,
          defaultParams: l.defaultParams,
          description: l.description,
        });
      }
    }
    const catalogue = Array.from(catalogueByType.values());
    const catalogueId = _hashString(`${JSON.stringify(ALL_ARCHITECTURE_FAMILIES)}:${JSON.stringify(catalogue)}`);
    const hwFamily = toHwFamily(selectedArchitecture);
    const hwValidation = validateHardwareConfig(hwConfig, hwFamily);
    return {
      family: selectedArchitecture,
      nodes,
      connections,
      groups,
      allowed_layer_types: allowedLayerTypes,
      allowed_families: ALL_ARCHITECTURE_FAMILIES,
      catalogue_id: catalogueId,
      catalogue,
      missing_mandatory_fields: hwValidation.missingFields,
      hw_config: hwConfig as any,
      analysis_warnings: warnings,
      active_tab: activeWorkspaceTab,
      /**
       * The machine and the data, so the agent designs for what exists.
       *
       * It received neither, and could not: `hw_config` carries the target's
       * name and memory budget, which is what the *analysis* needs, but says
       * nothing about whether a run could start here — no runtime, no usable
       * precisions, no measured throughput, and no idea whether the machine
       * has an accelerator at all. An assistant asked to size a model for
       * "your hardware" was working from a name.
       *
       * Trimmed rather than sent whole: the agent needs what constrains a
       * design, not telemetry that changes between two of its own steps.
       */
      machine: hardwareProfile
        ? {
            cpu: `${hardwareProfile.cpu.model} (${hardwareProfile.cpu.cores}c/${hardwareProfile.cpu.threads}t)`,
            cpu_features: hardwareProfile.cpu.features ?? [],
            ram_available_gb: Math.round(hardwareProfile.ramAvailableBytes / 1024 ** 3),
            disk_available_gb: Math.round(hardwareProfile.diskAvailableBytes / 1024 ** 3),
            accelerators: hardwareProfile.gpus.map((g) => ({
              name: g.name,
              integrated: g.integrated,
              recognised: g.recognised,
              vram_free_gb: g.vramFreeBytes != null ? Math.round(g.vramFreeBytes / 1024 ** 3) : null,
            })),
            cpu_only: isCpuOnly(hardwareProfile),
            measured_gflops: hardwareProfile.compute?.gflopsF32 ?? null,
            measured_bandwidth_gbs: hardwareProfile.compute?.memoryBandwidthGbs ?? null,
            is_example: isExampleHardware,
          }
        : null,
      dataset: datasetProfile
        ? {
            kind: datasetProfile.kind,
            samples: datasetProfile.samples,
            sample_shape: datasetProfile.sampleShape ?? null,
            num_classes: datasetProfile.suggested.numClasses ?? null,
            family_hint: datasetProfile.suggested.familyHint ?? null,
          }
        : null,
    };
  }, [selectedArchitecture, nodes, connections, groups, hwConfig, warnings, toHwFamily]);

  const layerConfigByType = useMemo(() => {
    const preferredOrder: ArchitectureFamily[] = [
      selectedArchitecture,
      ...ALL_ARCHITECTURE_FAMILIES.filter((f) => f !== selectedArchitecture),
    ];

    const allLayers = preferredOrder.flatMap((f) => getPluginLayers(f));
    const map = new Map<string, LayerConfig>();
    for (const l of allLayers) {
      if (!l?.type) continue;
      if (!map.has(l.type)) map.set(l.type, l as LayerConfig);
    }
    return map;
  }, [selectedArchitecture]);

  const handleClearCanvas = useCallback(() => {
    if (!hasCanvasContent) {
      return;
    }

    resetWorkspace();

    toast({
      title: "Canvas cleared",
      description: "Removed all blocks and connections from the current workspace.",
    });
  }, [hasCanvasContent, resetWorkspace, toast]);

  const handleCreateNewCanvas = useCallback(() => {
    if (hasCanvasBlocks) {
      setShowNewCanvasDialog(true);
      return;
    }

    resetWorkspace();
    toast({
      title: "Blank page ready",
      description: "Started a fresh workspace.",
    });
  }, [hasCanvasBlocks, resetWorkspace, toast]);

  const handleSaveCanvas = useCallback(() => {
    const filename = downloadCanvasSnapshot();
    if (!filename) {
      toast({
        title: "Nothing to save",
        description: "Add blocks to the canvas before saving a snapshot.",
      });
      return;
    }

    toast({
      title: "Canvas saved",
      description: `Downloaded ${filename}.`,
    });
  }, [downloadCanvasSnapshot, toast]);

  const handleDiscardCanvasAndStartNew = useCallback(() => {
    setShowNewCanvasDialog(false);
    resetWorkspace();
    toast({
      title: "Blank page ready",
      description: "Started a fresh workspace without saving the previous canvas.",
    });
  }, [resetWorkspace, toast]);

  const handleSaveCanvasAndStartNew = useCallback(() => {
    const filename = downloadCanvasSnapshot();
    setShowNewCanvasDialog(false);
    resetWorkspace();
    toast({
      title: "Canvas saved",
      description: filename
        ? `Downloaded ${filename} and started a fresh workspace.`
        : "Started a fresh workspace.",
    });
  }, [downloadCanvasSnapshot, resetWorkspace, toast]);

  // ─── The design as a file ───────────────────────────────────────────
  //
  // Separate from the project store below, and deliberately so. A project is
  // application state, kept for you; a `.neurax` file is a document you own,
  // that lives beside your training code, goes into a repository, and can be
  // handed to someone else. Both are useful; only one of them can be reviewed
  // in a pull request.

  /**
   * The open document's name, stripped of its file extension — what a saved
   * design is called everywhere in the UI that isn't the raw filename.
   */
  const documentBaseName = useMemo(
    () => documentName?.replace(/\.neurax(\.json)?$/i, '') ?? 'Untitled design',
    [documentName],
  );

  /**
   * Sets `openedInitialization` together with the design it was computed
   * against, so the invalidation effect below can tell "just set alongside
   * this exact design" apart from "the design has since moved on."
   */
  const setOpenedInitializationFor = useCallback(
    (record: InitializationRecord | null, forNodes: CanvasNode[], forConnections: Connection[]) => {
      initializationSourceRef.current = record ? { nodes: forNodes, connections: forConnections } : null;
      setOpenedInitialization(record);
    },
    [],
  );

  /**
   * A saved initialisation recipe describes one specific design's layer
   * shapes and fan-in/fan-out — it stops being true the moment a node is
   * added, removed, or reshaped. Rather than track every place nodes or
   * connections can change (drag, delete, undo, import, preset load, agent
   * edit...), this notices generically: whenever either array is no longer
   * the exact reference `openedInitialization` was captured against, the
   * recipe no longer describes what's on screen and is dropped. A stale
   * recipe reaching the next save would be worse than none — it would claim
   * to describe a design it doesn't, the same kind of drift this session's
   * `.neurax` format was built to refuse elsewhere.
   */
  useEffect(() => {
    const source = initializationSourceRef.current;
    if (!source) return;
    if (source.nodes !== nodes || source.connections !== connections) {
      initializationSourceRef.current = null;
      setOpenedInitialization(null);
    }
  }, [nodes, connections]);

  /** Everything that has to survive a round trip through a file. */
  const currentDesignSnapshot = useCallback(
    () => ({
      name: documentBaseName,
      architecture: selectedArchitecture,
      nodes,
      connections,
      groups,
      hardware: hwConfig,
      analysis,
      initialization: openedInitialization,
      timeMachine: timeMachineConfig,
    }),
    [
      documentBaseName, selectedArchitecture, nodes, connections, groups, hwConfig, analysis,
      openedInitialization, timeMachineConfig,
    ],
  );

  /** Write the design somewhere new, asking the user where. */
  const handleSaveDesignAs = useCallback(async () => {
    const snapshot = currentDesignSnapshot();
    const contents = serializeDesign(snapshot, { generator: 'NEURAX Studio' });

    try {
      const result = await saveTextFile(
        contents,
        suggestedFileName(snapshot.name),
        'application/json',
      );
      if (!result.saved) return; // The user dismissed the dialog.

      if (result.path) {
        setDocumentPath(result.path);
        setDocumentName(result.path.split(/[/\\]/).pop());
      } else {
        // A browser never reveals where the download went, so the document
        // stays pathless and the next save asks again.
        setDocumentName(suggestedFileName(snapshot.name));
      }
      markSaved({ nodes: snapshot.nodes, connections: snapshot.connections, groups: snapshot.groups });

      toast({
        title: 'Design saved',
        description: result.path ?? suggestedFileName(snapshot.name),
      });
    } catch (err) {
      toast({ title: 'Save failed', description: String(err), variant: 'destructive' });
    }
  }, [currentDesignSnapshot, markSaved, toast]);

  /**
   * Save to the file this design came from.
   *
   * Falls back to Save As when there is no such file — a design that has never
   * been saved, or a browser, where a page cannot write to a path at all.
   */
  const handleSaveDesign = useCallback(async () => {
    if (!documentPath || !canWriteInPlace()) {
      await handleSaveDesignAs();
      return;
    }

    const contents = serializeDesign(currentDesignSnapshot(), { generator: 'NEURAX Studio' });
    try {
      await writeTextFile(documentPath, contents);
      markSaved({ nodes, connections, groups });
      toast({ title: 'Saved', description: documentPath });
    } catch (err) {
      // The path may have been removed or made read-only since it was chosen;
      // offering the dialog is more useful than reporting a failure.
      toast({
        title: 'Could not save in place',
        description: `${String(err)} — choose a new location.`,
        variant: 'destructive',
      });
      await handleSaveDesignAs();
    }
  }, [documentPath, currentDesignSnapshot, handleSaveDesignAs, nodes, connections, groups, markSaved, toast]);

  /** Open a `.neurax` design from disk, replacing what is on the canvas. */
  const handleOpenDesign = useCallback(async () => {
    // `writable`: this open starts an editing session on the document, so a
    // later Ctrl+S may write back to it without asking for the name again.
    let picked;
    try {
      picked = await openTextFile([NEURAX_EXTENSION, 'json'], { writable: true });
    } catch (err) {
      toast({ title: 'Could not read that file', description: String(err), variant: 'destructive' });
      return;
    }
    if (!picked) return; // Dismissed.

    const parsed = parseNeuraxFile(picked.contents);
    if (!parsed.ok) {
      toast({ title: 'Cannot open this file', description: parsed.error, variant: 'destructive' });
      return;
    }

    const { document: doc, warnings: fileWarnings } = parsed;
    const design = doc.design;

    setNodes(design.nodes);
    setConnections(design.connections);
    setGroups(design.groups);
    setSelectedArchitecture(doc.architecture);
    if (doc.hardware && Object.keys(doc.hardware).length > 0) {
      // Merged over the defaults rather than assigned. A file written by an
      // older build, or hand-edited down to the fields someone cared about,
      // carries only part of the configuration — assigning it would leave the
      // rest `undefined`, and the analysis would run against a config with
      // holes in it rather than against the defaults.
      setHwConfig({ ...DEFAULT_HARDWARE_CONFIG, ...doc.hardware } as HardwareConfig);
    }
    setSelectedNodeId(null);
    setCurrentPresetId(null);
    setWarnings([]);
    setPerLayer([]);
    setAnalysis(initialAnalysis);
    setOpenedInitializationFor(doc.initialization ?? null, design.nodes, design.connections);
    setTimeMachineConfig(doc.timeMachine ?? null);
    // Forces Time Machine to remount: the panel stays mounted across the whole
    // session, so without this its sliders
    // would keep showing whatever was there before this file was opened —
    // the state above would be correct and the screen would still be wrong.
    setPanelLoadGeneration((g) => g + 1);
    pendingConnectionsRef.current.clear();

    // A freshly opened file is the start of a history, not a step in the
    // previous document's.
    resetHistory(design);
    setDocumentName(picked.name);
    setDocumentPath(picked.path);
    markSaved(design);

    toast({
      title: `Opened ${doc.name}`,
      description: fileWarnings.length
        ? fileWarnings.join(' ')
        : `${design.nodes.length} blocks. Run an analysis to compute its metrics.`,
      variant: fileWarnings.length ? 'destructive' : undefined,
    });
  }, [resetHistory, setHwConfig, markSaved, toast, setOpenedInitializationFor]);

  // ─── Project Save/Load ──────────────────────────────────────────────

  const handleLoadProjects = useCallback(async () => {
    setIsProjectsLoading(true);
    try {
      const resp = await listProjects();
      setSavedProjects(resp.projects);
    } catch (err) {
      console.error('[neurax] Failed to load projects:', err);
    } finally {
      setIsProjectsLoading(false);
    }
  }, []);

  const handleSaveProject = useCallback(async () => {
    const canvasData = { nodes, connections, groups };
    const projectBody = {
      name: `Project ${new Date().toLocaleDateString()}`,
      architecture: selectedArchitecture,
      canvas: canvasData,
      hardware_config: hwConfig as unknown as Record<string, unknown>,
      last_analysis: analysis as unknown as Record<string, unknown> | undefined,
    };

    try {
      if (currentProjectId) {
        // Update existing project
        await updateProject(currentProjectId, {
          canvas: canvasData,
          architecture: selectedArchitecture,
          hardware_config: hwConfig as unknown as Record<string, unknown>,
          last_analysis: analysis as unknown as Record<string, unknown>,
        });
        toast({ title: 'Project saved', description: 'Changes saved successfully.' });
      } else {
        // Create new project
        const resp = await createProject(projectBody);
        setCurrentProjectId(resp.project.id);
        toast({ title: 'Project saved', description: `Created "${resp.project.name}".` });
      }
      // Refresh project list
      await handleLoadProjects();
    } catch (err) {
      toast({ title: 'Save failed', description: String(err), variant: 'destructive' });
    }
  }, [nodes, connections, groups, selectedArchitecture, hwConfig, analysis, currentProjectId, toast, handleLoadProjects]);

  const handleLoadProject = useCallback(async (project: Project) => {
    const canvas = project.canvas as any;
    if (canvas?.nodes) setNodes(canvas.nodes);
    if (canvas?.connections) setConnections(canvas.connections);
    if (canvas?.groups) setGroups(canvas.groups);
    if (project.architecture) setSelectedArchitecture(project.architecture as ArchitectureFamily);
    if (project.hardware_config) setHwConfig(project.hardware_config as any);
    setCurrentProjectId(project.id);
    // The project store doesn't carry an initialisation recipe at all — clear
    // whatever a previously opened `.neurax` file left behind, or a save
    // right after loading this project would attach someone else's recipe
    // to it.
    initializationSourceRef.current = null;
    setOpenedInitialization(null);
    // Same reasoning: the project store doesn't carry these either.
    setTimeMachineConfig(null);
    setPanelLoadGeneration((g) => g + 1);
    toast({ title: 'Project loaded', description: `Loaded "${project.name}".` });
  }, [setNodes, setConnections, setGroups, setSelectedArchitecture, setHwConfig, toast]);

  const handleDeleteProject = useCallback(async (projectId: string) => {
    try {
      await deleteProject(projectId);
      if (currentProjectId === projectId) {
        setCurrentProjectId(null);
      }
      await handleLoadProjects();
      toast({ title: 'Project deleted', description: 'The project has been removed.' });
    } catch (err) {
      toast({ title: 'Delete failed', description: String(err), variant: 'destructive' });
    }
  }, [currentProjectId, toast, handleLoadProjects]);

  // Load projects on mount
  useEffect(() => {
    void handleLoadProjects();
  }, [handleLoadProjects]);

  // Load credits on mount
  useEffect(() => {
    getCredits()
      .then((res) => setCreditInfo(res.credits))
      .catch(() => { /* credits unavailable in dev mode */ });
  }, []);

  // Bridges connections across a node the agent tried to add but that
  // failed to land on the canvas (unknown/unaliased type, or any other
  // add_node rejection) — instead of two dangling edges into and out of a
  // node that was never actually there, wire its would-be neighbors
  // straight to each other. Without this, every silently-dropped node left
  // the design with a hole: real blocks on either side of it, no path
  // between them, which is exactly what disconnected, "same block
  // everywhere" architectures came from — the agent kept adding whatever
  // few types happened to survive, oblivious that everything around them
  // had stopped connecting.
  const droppedNodeIdsRef = useRef<Set<string>>(new Set());
  const bridgeSourcesRef = useRef<Map<string, Set<string>>>(new Map());

  const handleAgentToolEvent = useCallback((tool: { name: string; args?: Record<string, unknown> }) => {
    const name = tool?.name;
    const args = tool?.args ?? {};

    if (name === 'set_hw_config') {
      const updates = (args as any)?.updates;
      if (!updates || typeof updates !== 'object') return;
      updateHwConfig(updates as any);
      triggerAgentAutoAnalysis();
      return;
    }

    // `initialize_hyperparams` and `set_hyperparams` were granted to three of
    // the agent's four modes and applied to its own snapshot, but nothing here
    // handled them — so a training configuration the agent believed it had set
    // never reached the canvas, silently, in both directions. Both arrive as
    // `{updates: {...}}` (agent_graph.py fills in the values that
    // `initialize_hyperparams` derives, since the call itself carries none).
    if (name === 'initialize_hyperparams' || name === 'set_hyperparams') {
      const updates = (args as any)?.updates;
      if (!updates || typeof updates !== 'object') return;

      const mapped = mapAgentHyperparams(
        updates as Record<string, unknown>,
        hwConfig.customParams,
      );
      if (Object.keys(mapped).length === 0) return;

      updateHwConfig(mapped as any);
      triggerAgentAutoAnalysis();
      return;
    }

    if (name === 'set_family') {
      const family = String(args.family ?? '');
      if (!family) return;
      // Validate family is in the allowed list
      const validFamilies = ALL_ARCHITECTURE_FAMILIES;
      if (!validFamilies.includes(family as ArchitectureFamily)) {
        console.warn(`Invalid family from agent: ${family}, ignoring`);
        return;
      }
      handleArchitectureChange(family as ArchitectureFamily);
      triggerAgentAutoAnalysis();
      return;
    }

    if (name === 'add_node') {
      const layerType = String(args.layer_type ?? '');
      const nodeId = typeof args.node_id === 'string' ? args.node_id : undefined;
      const x = Number(args.x ?? 100);
      const y = Number(args.y ?? 100);

      const cfg = layerConfigByType.get(layerType)
        ?? layerConfigByType.get(AGENT_TYPE_ALIASES[layerType] ?? '');
      if (!cfg) {
        if (nodeId) droppedNodeIdsRef.current.add(nodeId);
        toast({
          title: 'Agent tool rejected',
          description: `Unknown layer type: ${layerType}`,
          variant: 'destructive',
        });
        return;
      }

      addNodeFromConfig(cfg as LayerConfig, x, y, nodeId);
      triggerAgentAutoAnalysis();
      return;
    }

    if (name === 'set_node_params') {
      const nodeId = String((args as any)?.node_id ?? '');
      const updates = (args as any)?.updates;
      if (!nodeId || !updates || typeof updates !== 'object') return;
      const patch = updates as Record<string, any>;
      handleUpdateNode(nodeId, {
        params: {
          ...(nodes.find((n) => n.id === nodeId)?.params ?? {}),
          ...patch,
        },
      });
      triggerAgentAutoAnalysis();
      return;
    }

    if (name === 'connect') {
      const fromId = String(args.from_id ?? '');
      const toId = String(args.to_id ?? '');
      if (!fromId || !toId) return;

      // Resolve a dropped id to the real (or already-bridged) node(s) that
      // actually feed it, recursing through any run of consecutive dropped
      // nodes. `seen` guards against a cycle turning this into a loop.
      const resolveSources = (id: string, seen: Set<string> = new Set()): string[] => {
        if (!droppedNodeIdsRef.current.has(id)) return [id];
        if (seen.has(id)) return [];
        seen.add(id);
        const upstream = bridgeSourcesRef.current.get(id);
        if (!upstream || upstream.size === 0) return [];
        return [...upstream].flatMap((u) => resolveSources(u, seen));
      };

      if (droppedNodeIdsRef.current.has(toId)) {
        // The target never landed — hold onto its real source(s) so that
        // whatever the agent next connects FROM `toId` gets wired to them
        // directly instead.
        const sources = resolveSources(fromId);
        if (sources.length > 0) {
          const set = bridgeSourcesRef.current.get(toId) ?? new Set<string>();
          sources.forEach((s) => set.add(s));
          bridgeSourcesRef.current.set(toId, set);
        }
        return;
      }

      const sources = resolveSources(fromId);
      sources.forEach((src) => handleAddConnection(src, toId, true));
      if (sources.length > 0) triggerAgentAutoAnalysis();
      return;
    }

    if (name === 'disconnect') {
      const fromId = String(args.from_id ?? '');
      const toId = String(args.to_id ?? '');
      if (!fromId || !toId) return;
      const conn = connections.find((c) => c.from === fromId && c.to === toId);
      if (conn) {
        handleDeleteConnection(conn.id);
        triggerAgentAutoAnalysis();
      }
      return;
    }

    if (name === 'delete_node') {
      const nodeId = String(args.node_id ?? '');
      if (!nodeId) return;
      handleDeleteNode(nodeId);
      triggerAgentAutoAnalysis();
      return;
    }

    if (name === 'navigate_to') {
      const validTabs: WorkspaceTab[] = [
        'architecture',
        'simulation',
        'production',
        'training',
        'timemachine',
      ];
      const tab = String(args.tab ?? '') as WorkspaceTab;
      if (!validTabs.includes(tab)) {
        console.warn(`Agent navigate_to: unknown tab "${tab}", ignoring`);
        return;
      }
      setActiveWorkspaceTab(tab);
      return;
    }

    if (name === 'run_analysis') {
      void handleRunAnalysis();
      return;
    }

    if (name === 'select_node') {
      const nodeId = String(args.node_id ?? '');
      handleSelectNode(nodeId || null);
      return;
    }
  }, [layerConfigByType, toast, handleAddNode, handleUpdateNode, handleAddConnection, handleDeleteConnection, handleDeleteNode, handleSelectNode, triggerAgentAutoAnalysis, handleArchitectureChange, updateHwConfig, nodes, connections, setActiveWorkspaceTab, handleRunAnalysis]);

  const handleImportArchitecture = useCallback((result: ImportResult) => {
    // 1. Update family if present
    if (result.family) {
      handleArchitectureChange(result.family);
    }

    // 2. Update hardware config if present
    if (result.hardwareConfig) {
      updateHwConfig(result.hardwareConfig);
    }

    // 3. Replace current architecture with imported one
    const targetFamily = result.family || selectedArchitecture;
    const importedNodes = hydrateNodesForFamily(targetFamily, result.nodes);
    setNodes(importedNodes);
    setConnections(result.connections);
    // An import replaces the whole design, so any groups belonged to the design
    // that was just discarded and would now point at blocks that do not exist.
    setGroups([]);
    setSelectedNodeId(null);
    setCurrentPresetId(null);

    // 4. An import is a new document. It has no file yet — saving it should ask
    // where to put it rather than overwrite whatever was open before — but it
    // does have a name, which becomes the suggested filename.
    resetHistory({ nodes: importedNodes, connections: result.connections, groups: [] });
    setDocumentName(result.modelName);
    setDocumentPath(undefined);
    setIsDirty(true);

    // 5. Re-run analysis after import
    setTimeout(() => {
      handleRunAnalysis();
    }, 500);
  }, [handleArchitectureChange, updateHwConfig, selectedArchitecture, hydrateNodesForFamily, handleRunAnalysis, resetHistory]);

  // ─── A/B comparison ─────────────────────────────────────────────────

  /** The design on the canvas now, as something a comparison can hold. */
  const currentVariant = useMemo<DesignVariant | null>(() => {
    // Only an analysed design has numbers to compare. `totalParams` at zero
    // means the analysis has not run, and comparing against it would show a
    // page of meaningless deltas.
    if (!nodes.length || !analysis || !analysis.totalParams) return null;

    return {
      id: 'current',
      name: documentName?.replace(/\.neurax(\.json)?$/i, '') ?? 'Current design',
      capturedAt: new Date().toISOString(),
      architecture: selectedArchitecture,
      blockCount: nodes.length,
      connectionCount: connections.length,
      analysis,
      hardware: hwConfig,
    };
  }, [nodes, connections, analysis, documentName, selectedArchitecture, hwConfig]);

  const handleCaptureBaseline = useCallback(() => {
    if (!currentVariant) return;
    setComparisonBaseline({
      ...currentVariant,
      id: `baseline-${Date.now()}`,
      name: `${currentVariant.name} (baseline)`,
      capturedAt: new Date().toISOString(),
    });
    toast({
      title: 'Baseline captured',
      description: 'Change the design and reopen Compare to see what moved.',
    });
  }, [currentVariant, toast]);

  // ─── Keyboard shortcuts ─────────────────────────────────────────────
  //
  // Registered here rather than on the canvas because these act on the
  // document, not on the selection, and must work wherever focus happens to be
  // — including the analysis panel and the inspector.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      // Help, before anything else: F1 works even while typing, because the
      // moment you need the guide is often the moment you are staring at a
      // field you do not understand.
      if (event.key === 'F1') {
        event.preventDefault();
        setDocSectionId(undefined);
        setShowDocumentation(true);
        return;
      }

      const accel = event.ctrlKey || event.metaKey;
      if (!accel) return;

      // Never steal a keystroke from a field the user is typing in: Ctrl+Z in a
      // text input means undo the typing, and the browser already does that.
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (target.isContentEditable ||
          ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName))
      ) {
        return;
      }

      const key = event.key.toLowerCase();

      if (key === 'z' && !event.shiftKey) {
        event.preventDefault();
        undoDesign();
        return;
      }

      // Both conventions for redo: Ctrl+Shift+Z everywhere, Ctrl+Y on Windows.
      if ((key === 'z' && event.shiftKey) || key === 'y') {
        event.preventDefault();
        redoDesign();
        return;
      }

      if (key === 's') {
        event.preventDefault();
        void (event.shiftKey ? handleSaveDesignAs() : handleSaveDesign());
        return;
      }

      if (key === 'o') {
        event.preventDefault();
        void handleOpenDesign();
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [undoDesign, redoDesign, handleSaveDesign, handleSaveDesignAs, handleOpenDesign]);

  // Warn before losing unsaved work. The browser shows its own wording; on the
  // desktop this is what stops a quit from discarding an hour of design.
  useEffect(() => {
    if (!isDirty) return;
    const onBeforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = '';
    };
    window.addEventListener('beforeunload', onBeforeUnload);
    return () => window.removeEventListener('beforeunload', onBeforeUnload);
  }, [isDirty]);

  // Architecture workspace content
  const architectureContent = (
    <div className="flex-1 flex overflow-hidden min-w-0">
      {isChatOpen && !isDesktopLayout ? (
        <Sheet open={isChatOpen} onOpenChange={setIsChatOpen}>
          <SheetContent side="left" className="p-0 w-[360px] sm:w-[420px]">
            <Suspense fallback={null}>
              <AIChatDrawer
                open={isChatOpen}
                onOpenChange={setIsChatOpen}
                onAddCredits={undefined}
                creditsLeft={creditInfo ? creditInfo.limit - creditInfo.used : undefined}
                creditsLimit={creditInfo?.limit}
                getSnapshot={agentGetSnapshot}
                onToolEvent={handleAgentToolEvent}
                projectId={currentProjectId}
                hardware={hardwareProfile}
                dataset={datasetProfile}
                className="h-full"
              />
            </Suspense>
          </SheetContent>
        </Sheet>
      ) : null}

      {isDesktopLayout ? (
        <ResizablePanelGroup direction="horizontal" className="flex-1 min-w-0">
          {isChatOpen ? (
            <>
              <ResizablePanel defaultSize={28} minSize={18} maxSize={42} className="border-r border-border/70 min-w-0">
                <Suspense fallback={null}>
                  <AIChatDrawer
                    open={isChatOpen}
                    onOpenChange={setIsChatOpen}
                    onAddCredits={undefined}
                    creditsLeft={creditInfo ? creditInfo.limit - creditInfo.used : undefined}
                    creditsLimit={creditInfo?.limit}
                    getSnapshot={agentGetSnapshot}
                    onToolEvent={handleAgentToolEvent}
                    projectId={currentProjectId}
                    hardware={hardwareProfile}
                    dataset={datasetProfile}
                    className="h-full"
                  />
                </Suspense>
              </ResizablePanel>
              <ResizableHandle withHandle />
            </>
          ) : null}

          <ResizablePanel defaultSize={isChatOpen ? 72 : 100} minSize={40} className="min-w-0">
            <div className="h-full flex overflow-hidden min-w-0">
              <div className="hidden sm:block flex-shrink-0">
                <LayerPalette onDragStart={() => { }} selectedArchitecture={selectedArchitecture} />
              </div>

              <div className="flex-1 flex flex-col min-w-0">
                {/* Main Canvas */}
                <div className="flex-1 relative overflow-hidden min-w-0">
                  <ArchitectureCanvas
                    nodes={nodes}
                    connections={connections}
                    groups={groups}
                    selectedNodeId={selectedNodeId}
                    onSelectNode={handleSelectNode}
                    onUpdateNode={handleUpdateNode}
                    onAddNode={handleAddNode}
                    onDeleteNode={handleDeleteNode}
                    onDuplicateNode={handleDuplicateNode}
                    onAddConnection={handleAddConnection}
                    onDeleteConnection={handleDeleteConnection}
                    onGroupSelected={handleGroupSelected}
                    onUngroupGroup={handleUngroupGroup}
                    onDeleteGroup={handleDeleteGroup}
                    onUpdateGroup={handleUpdateGroup}
                  />

                </div>

                <InspectorPanel
                  node={selectedNode}
                  group={selectedGroup}
                  nodes={nodes}
                  selectionRevision={selectionRevision}
                  analysis={analysis}
                  perLayer={perLayer}
                  warnings={warnings}
                  onJumpToWarnings={() => {
                    setActiveRightPanelTab('architecture');
                    setJumpToIssuesSignal((v) => v + 1);
                  }}
                  onUpdateNode={handleUpdateNode}
                  onUpdateGroup={handleUpdateGroup}
                  onClose={() => handleSelectNode(null)}
                  onDelete={handleDeleteNode}
                  onDeleteGroup={handleDeleteGroup}
                  selectedArchitecture={selectedArchitecture}
                />
              </div>

              <div className="hidden md:block flex-shrink-0">
                <RightPanelTabs
                  nodes={nodes}
                  selectedNodeId={selectedNodeId}
                  onSelectNode={handleSelectNode}
                  analysis={analysis}
                  warnings={warnings}
                  perLayer={perLayer}
                  selectedArchitecture={selectedArchitecture}
                  activeTab={activeRightPanelTab}
                  onActiveTabChange={setActiveRightPanelTab}
                  jumpToIssuesSignal={jumpToIssuesSignal}
                />
              </div>
            </div>
          </ResizablePanel>
        </ResizablePanelGroup>
      ) : (
        <div className="flex-1 flex overflow-hidden min-w-0">
          <div className="flex-1 flex flex-col min-w-0">
            <div className="flex-1 relative overflow-hidden min-w-0">
              <ArchitectureCanvas
                nodes={nodes}
                connections={connections}
                groups={groups}
                selectedNodeId={selectedNodeId}
                onSelectNode={handleSelectNode}
                onUpdateNode={handleUpdateNode}
                onAddNode={handleAddNode}
                onDeleteNode={handleDeleteNode}
                onDuplicateNode={handleDuplicateNode}
                onAddConnection={handleAddConnection}
                onDeleteConnection={handleDeleteConnection}
                onGroupSelected={handleGroupSelected}
                onUngroupGroup={handleUngroupGroup}
                onDeleteGroup={handleDeleteGroup}
                onUpdateGroup={handleUpdateGroup}
              />

            </div>
            <InspectorPanel
              node={selectedNode}
              group={selectedGroup}
              nodes={nodes}
              selectionRevision={selectionRevision}
              analysis={analysis}
              perLayer={perLayer}
              warnings={warnings}
              onJumpToWarnings={() => {
                setActiveRightPanelTab('architecture');
                setJumpToIssuesSignal((v) => v + 1);
              }}
              onUpdateNode={handleUpdateNode}
              onUpdateGroup={handleUpdateGroup}
              onClose={() => handleSelectNode(null)}
              onDelete={handleDeleteNode}
              onDeleteGroup={handleDeleteGroup}
              selectedArchitecture={selectedArchitecture}
            />
          </div>
        </div>
      )}
    </div>
  );

  return (
    <div className="h-screen flex flex-col bg-background">
      <TopNav
        datasetProfile={datasetProfile}
        detectedHardware={hardwareProfile}
        onPickDataset={(selection) => setDatasetProfile(profileForSelection(selection))}
        onRunAnalysis={handleRunAnalysisStream}
        isAnalyzing={isAnalyzing}
        onNewCanvas={handleCreateNewCanvas}
        onSaveCanvas={handleSaveCanvas}
        onExport={() => setShowExportPanel(true)}
        onImport={() => setShowImportPanel(true)}
        onShare={() => setShowSharePanel(true)}
        onOpenDesign={() => void handleOpenDesign()}
        onSaveDesign={() => void handleSaveDesign()}
        documentName={documentName}
        documentBaseName={documentBaseName}
        onRenameDocument={(name) => setDocumentName(name)}
        isDirty={isDirty}
        onUndo={undoDesign}
        onRedo={redoDesign}
        canUndo={canUndo}
        canRedo={canRedo}
        onCompare={() => setShowComparePanel(true)}
        hasBaseline={comparisonBaseline !== null}
        onOpenDocumentation={() => {
          setDocSectionId(undefined);
          setShowDocumentation(true);
        }}
        onSelectTarget={() => setShowTargetPanel(true)}
        onHyperparameters={() => setShowHyperparametersPanel(true)}
        isChatOpen={isChatOpen}
        onToggleChat={() => setIsChatOpen((v) => !v)}
        selectedArchitecture={selectedArchitecture}
        onArchitectureChange={handleArchitectureChange}

        onLoadPreset={handleLoadPreset}
        onClearCanvas={handleClearCanvas}
        currentPresetId={currentPresetId}
        nodes={nodes}
        connections={connections}
        projects={savedProjects}
        currentProjectId={currentProjectId}
        onSaveProject={handleSaveProject}
        onLoadProject={handleLoadProject}
        onDeleteProject={handleDeleteProject}
        isProjectsLoading={isProjectsLoading}
      />

      <div className="flex-1 flex flex-col overflow-hidden">
        <WorkspaceTabs
          activeTab={activeWorkspaceTab}
          onTabChange={setActiveWorkspaceTab}
          hasDesign={hasCanvasBlocks}
          hasAnalysis={hasAnalysisReportData(analysis)}
          onExport={() => setShowExportPanel(true)}
          architectureContent={architectureContent}
          simulationContent={
            <Suspense fallback={null}>
              <SimulationWorkspace nodes={nodes} connections={connections} analysis={analysis} perLayer={perLayer} warnings={warnings} />
            </Suspense>
          }
          productionContent={
            <Suspense fallback={null}>
              <ProductionWorkspace
                nodes={nodes}
                connections={connections}
                modelName={documentBaseName}
                architectureFamily={selectedArchitecture}
                groups={groups}
                hardware={hwConfig}
                analysis={analysis}
                onSaved={(initialization) => {
                  setOpenedInitializationFor(initialization, nodes, connections);
                  markSaved({ nodes, connections, groups });
                }}
              />
            </Suspense>
          }
          trainingContent={
            <Suspense fallback={null}>
              <TrainingWorkspace
                modelName={documentBaseName}
                /**
                 * The design, as PyTorch, generated on demand.
                 *
                 * Passed as a function rather than a value so the canvas is
                 * read at the moment a run starts, not on every render of a
                 * workspace nobody is looking at — generating a large model's
                 * source is not free.
                 */
                onGenerateModel={() => {
                  if (nodes.length === 0) return null;
                  const generated = generateModelCode(nodes, connections, hwConfig, documentBaseName);
                  return {
                    code: generated.code,
                    modelClassName: generated.modelClassName,
                    totalParams: generated.totalParams,
                    fullySupported: generated.fullySupported,
                    unsupportedTypes: generated.unsupportedTypes,
                    // The shape and the dtype come from the model, not from
                    // the config. Guessing them here — image size set
                    // anywhere means images — fed `[3, 224, 224]` floats to
                    // BERT's token embedding and killed the run in its first
                    // attention block.
                    inputShape: generated.inputShape,
                    inputKind: generated.inputKind,
                    vocabSize: generated.vocabSize,
                    numClasses: hwConfig.numClasses ?? 10,
                  };
                }}
                analysis={analysis}
                hardware={hardwareProfile}
                dataset={datasetProfile}
                onChooseDataset={() => setDatasetProfile(mockImageDataset)}
              />
            </Suspense>
          }
          timeMachineContent={
            <Suspense fallback={null}>
              <TimeMachineWorkspace
                key={`timemachine-${panelLoadGeneration}`}
                nodes={nodes}
                connections={connections}
                initialConfig={timeMachineConfig ?? undefined}
                onConfigChange={setTimeMachineConfig}
              />
            </Suspense>
          }
        >
          {null}
        </WorkspaceTabs>
      </div>

      <Dialog open={showNewCanvasDialog} onOpenChange={setShowNewCanvasDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Save current canvas?</DialogTitle>
            <DialogDescription>
              This canvas has content. Save the current graph before starting a blank page?
            </DialogDescription>
          </DialogHeader>

          <div className="rounded-md border border-border bg-secondary/30 px-3 py-2 text-sm text-muted-foreground">
            {nodes.length} {nodes.length === 1 ? 'block' : 'blocks'} and {connections.length} {connections.length === 1 ? 'connection' : 'connections'}
            <div className="mt-1 text-xs">
              Saving downloads a `.neurax.json` snapshot that you can import later.
            </div>
          </div>

          <DialogFooter className="gap-2 sm:gap-0">
            <Button variant="outline" onClick={() => setShowNewCanvasDialog(false)}>
              Cancel
            </Button>
            <Button variant="secondary" onClick={handleDiscardCanvasAndStartNew}>
              Don&apos;t Save
            </Button>
            <Button onClick={handleSaveCanvasAndStartNew}>
              Save &amp; New Page
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Suspense fallback={null}>
        <ExportPanel
          isOpen={showExportPanel}
          onClose={() => setShowExportPanel(false)}
          architectureName="NeuraxModel"
          nodes={nodes}
          connections={connections}
          groups={groups}
          selectedArchitecture={selectedArchitecture}
          // Only a real, completed analysis — not the zeroed placeholder shown
          // before the first run — is worth cross-checking generated code
          // against. `generatedAt` is only ever set by a real analysis result.
          analysisResult={analysis.generatedAt ? analysis : null}
        />

        <ImportPanel
          isOpen={showImportPanel}
          onClose={() => setShowImportPanel(false)}
          onImport={handleImportArchitecture}
        />

        <SharePanel
          isOpen={showSharePanel}
          onClose={() => setShowSharePanel(false)}
          nodes={nodes}
          connections={connections}
          groups={groups}
          selectedArchitecture={selectedArchitecture}
          analysisResult={analysis.generatedAt ? analysis : null}
        />

        <ComparePanel
          isOpen={showComparePanel}
          onClose={() => setShowComparePanel(false)}
          baseline={comparisonBaseline}
          candidate={currentVariant}
          onCapture={handleCaptureBaseline}
          onClear={() => setComparisonBaseline(null)}
        />

        <DocumentationPanel
          isOpen={showDocumentation}
          onClose={() => setShowDocumentation(false)}
          initialSectionId={docSectionId}
        />

        <ModelHyperparametersDialog
          isOpen={showHyperparametersPanel}
          onClose={() => setShowHyperparametersPanel(false)}
          family={selectedArchitecture}
        />
      </Suspense>

      <SimulationTargetPanel
        detected={hardwareProfile}
        isExample={isExampleHardware}
        isMeasuring={isMeasuring}
        onMeasure={() => void measureThisMachine()}
        isOpen={showTargetPanel}
        onClose={() => setShowTargetPanel(false)}
      />

    </div>
  );
};

export default Index;
