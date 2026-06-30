import { useEffect, useState, useCallback, useMemo, useRef } from 'react';
import { TopNav } from '@/components/layout/TopNav.tsx';
import { LayerPalette } from '@/components/layout/LayerPalette.tsx';
import { WorkspaceTabs, WorkspaceTab } from '@/components/layout/WorkspaceTabs.tsx';
import { ArchitectureCanvas } from '@/components/canvas/ArchitectureCanvas.tsx';
import { RightPanelTabs } from '@/components/panels/RightPanelTabs.tsx';
import type { RightPanelTabId } from '@/components/panels/RightPanelTabs.tsx';
import { InspectorPanel } from '@/components/panels/InspectorPanel.tsx';
import AIChatDrawer from '@/components/panels/AIChatDrawer.tsx';
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
import { ExportPanel } from '@/components/panels/ExportPanel.tsx';
import { ImportPanel } from '@/components/panels/ImportPanel.tsx';
import { PricingPage } from '@/pages/Pricing.tsx';
import { InferenceIntelligence } from '@/components/inference';
import { ProductionWorkspace } from '@/components/production/ProductionWorkspace.tsx';
import { SimulationWorkspace } from '@/components/simulation/SimulationWorkspace.tsx';
import { TimeMachineWorkspace } from '@/components/timemachine/TimeMachineWorkspace.tsx';

import { ArchitectureFamily } from '@/types/plugins.ts';
import { VariantPreset } from '@/types/catalog.ts';
import { AnalysisResult, CanvasNode, Connection, LayerConfig, NodeGroup, PerLayerBreakdownRow, Warning, ParameterValue } from '@/types/architecture.ts';
import { ImportResult } from '@/utils/architectureImporter.ts';
import { compileToNeuraxIR } from '@/utils/neuraxCompiler.ts';
import { getBlockDefaults } from '@/utils/blockDefaults.ts';
import { DEFAULT_HARDWARE_CONFIG, HardwareConfig, useHardware, validateHardwareConfig, ArchitectureFamily as HwFamily } from '@/contexts/HardwareContext.tsx';
import { analyze, analyzeStream, NeuraxApiError, listProjects, createProject, updateProject, deleteProject, getCredits, type Project, type CreditInfo } from '@/services/neuraxApi.ts';
import { useToast } from '@/hooks/use-toast.ts';
import { getPluginLayers } from '@/plugins/registry.ts';

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
  totalParams: 0, numLayers: 0, modelType: '', graphDepth: 0,
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
  costPerMillionTokensUsd: 0, provider: '',
  selectedPrecision: 'fp16', selectedBatchSize: 1,
  confidenceScore: 1, depth: 0,
  isSequenceModel: true, customLayerCount: 0, diagnosticCount: 0,
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
  'rl',
  'snn',
  'rnn',
  'experimental',
];

const isPositiveNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value > 0;

const withPositiveFallback = (value: number | undefined, fallback: number): number =>
  isPositiveNumber(value) ? value : fallback;

const findNumericParam = (
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

function hydrateNodesForFamily(
  family: ArchitectureFamily,
  nodes: CanvasNode[],
): CanvasNode[] {
  const layerMap = new Map(getPluginLayers(family).map((layer) => [layer.type, layer]));

  return nodes.map((node) => {
    const config = layerMap.get(node.type);
    if (!config) return node;

    const params: Record<string, unknown> = {
      ...getBlockDefaults(node.type),
      ...(config.defaultParams ?? {}),
      ...(node.params ?? {}),
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

function buildHardwareConfigFromPreset(
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
  const timesteps = findNumericParam(nodes, ['rate_encoder', 'latency_encoder'], ['timesteps']);
  const numExperts = findNumericParam(
    nodes,
    ['moe_block', 'moe_layer', 'router_linear', 'router_softmax'],
    ['num_experts'],
  );
  const topK = findNumericParam(nodes, ['moe_block', 'moe_layer', 'router_linear'], ['top_k']);
  const actionDim = findNumericParam(nodes, ['policy_head'], ['action_dim']);
  const stateDim = findNumericParam(nodes, ['linear_projection', 'dense'], ['in_features']);
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
    case 'rl':
      return {
        ...base,
        hiddenDim: withPositiveFallback(embeddingDim, 256),
        numLayers: withPositiveFallback(layerCount, 2),
        actionDim: withPositiveFallback(actionDim, 4),
        stateDim: withPositiveFallback(stateDim, 256),
      };
    case 'snn':
      return {
        ...base,
        timesteps: withPositiveFallback(timesteps, 100),
        spikeRate: withPositiveFallback(current.spikeRate, 0.1),
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
    totalParams: struct.total_parameters ?? 0,
    numLayers: struct.num_layers ?? 0,
    modelType: struct.model_type ?? '',
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
    bottleneck: performance.bottleneck ?? '',
    rooflinePosition: compute.roofline_position ?? 0,

    estimatedFlops: formatFlopsHuman(totalFlops),
    forwardFlopsHuman: formatFlopsHuman(forwardFlops),
    backwardFlopsHuman: formatFlopsHuman(backwardFlops),

    peakVramBytes,
    parameterMemoryBytes: memory.parameter_memory_bytes ?? 0,
    activationMemoryBytes: memory.activation_memory_bytes ?? 0,
    gradientMemoryBytes: memory.gradient_memory_bytes ?? 0,
    optimizerStateBytes: memory.optimizer_state_bytes ?? 0,
    maxBatchSizeFit: memory.max_batch_size_fit ?? 0,
    memoryFragmentation: (dynamic.virtual_memory?.fragmentation_pct ?? 0) / 100,
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

    trainingCostUsd: cost.training_cost_usd ?? 0,
    trainingTimeHours: cost.training_time_hours ?? 0,
    energyKwh: cost.energy_kwh ?? 0,
    co2Kg: cost.co2_kg ?? 0,
    costPerMillionTokensUsd: cost.cost_per_million_tokens_usd ?? 0,
    provider: cost.provider ?? '',

    selectedPrecision: precision,
    selectedBatchSize: batchSize,

    confidenceScore: (rpt as any)?.confidence_score ?? 1.0,
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
        memory_contention_score: dynamic.behavioral.memory_contention_score ?? 0,
        cache_locality_score: dynamic.behavioral.cache_locality_score ?? 0,
        numerical_sensitivity: dynamic.behavioral.numerical_sensitivity ?? 0,
        load_balance_efficiency: dynamic.behavioral.load_balance_efficiency ?? 100,
        memory_bank_conflict_rate: dynamic.behavioral.memory_bank_conflict_rate ?? 0,
        prediction_confidence: dynamic.behavioral.prediction_confidence ?? 0,
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
      name: id,
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
  const [nodes, setNodes] = useState<CanvasNode[]>(initialNodes);
  const [connections, setConnections] = useState<Connection[]>(initialConnections);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectionRevision, setSelectionRevision] = useState(0);
  const [analysis, setAnalysis] = useState<AnalysisResult>(initialAnalysis);
  const [compiledTopology, setCompiledTopology] = useState<Record<string, unknown> | undefined>(undefined);
  const [warnings, setWarnings] = useState<Warning[]>(initialWarnings);
  const [perLayer, setPerLayer] = useState<PerLayerBreakdownRow[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showNewCanvasDialog, setShowNewCanvasDialog] = useState(false);
  const [showExportPanel, setShowExportPanel] = useState(false);
  const [showImportPanel, setShowImportPanel] = useState(false);
  const [showPricingPage, setShowPricingPage] = useState(false);
  const [selectedArchitecture, setSelectedArchitecture] = useState<ArchitectureFamily>('transformer');
  const [activeWorkspaceTab, setActiveWorkspaceTab] = useState<WorkspaceTab>('architecture');
  const [activeRightPanelTab, setActiveRightPanelTab] = useState<RightPanelTabId>('issues');
  const [jumpToIssuesSignal, setJumpToIssuesSignal] = useState(0);
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [isDesktopLayout, setIsDesktopLayout] = useState(true);
  const [currentPresetId, setCurrentPresetId] = useState<string | null>(null);
  const [presetAutoAnalysisTick, setPresetAutoAnalysisTick] = useState(0);
  const [groups, setGroups] = useState<NodeGroup[]>([]);
  const [savedProjects, setSavedProjects] = useState<Project[]>([]);
  const [currentProjectId, setCurrentProjectId] = useState<string | null>(null);
  const [isProjectsLoading, setIsProjectsLoading] = useState(false);
  const [creditInfo, setCreditInfo] = useState<CreditInfo | null>(null);
  const { toast } = useToast();
  const { config: hwConfig, setConfig: setHwConfig, updateConfig: updateHwConfig, triggerAttempt } = useHardware();

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
      case 'rl':
        return 'rl';
      case 'snn':
        return 'snn';
      case 'experimental':
        return 'experimental';
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

  const resetWorkspace = useCallback(() => {
    setNodes([]);
    setConnections([]);
    setGroups([]);
    setSelectedNodeId(null);
    setCurrentPresetId(null);
    setWarnings([]);
    setPerLayer([]);
    setAnalysis(initialAnalysis);
    setActiveWorkspaceTab('architecture');
    pendingConnectionsRef.current.clear();
  }, []);

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
    toast({
      title: "Template Loaded",
      description: `Loaded "${preset.name}" — all blocks are editable`,
    });
  }, [hwConfig, setHwConfig, toast]);

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

    // Synchronized with backend MERGE_BLOCK_TYPES
    const isFanInCapable = [
      'concat', 'merge', 'add', 'residual', 'residual_add', 'skip_connection',
      'expert_combine', 'gate', 'lm_head', 'moe_block', 'unet_block', 'router_softmax'
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

    try {
      // Compile canvas to NEURAX IR topology
      const ir = compileToNeuraxIR(nodes, connections, {
        modelName: 'NeuraxModel',
        family: selectedArchitecture,
        hardware: hwConfig.hardware,
        precision: hwConfig.precision,
        batchSize: hwConfig.batchSize,
        groups,
        // Training
        learningRate: hwConfig.learningRate,
        numEpochs: hwConfig.numEpochs,
        seqLen: hwConfig.seqLen,
        // Hardware
        gpuCount: hwConfig.gpuCount,
        gpuMemoryGb: hwConfig.gpuMemoryGb,
        // Data
        datasetSize: hwConfig.datasetSize,
        vocabSize: hwConfig.vocabSize,
        numClasses: hwConfig.numClasses,
      });

      // Send to backend — topology IS the full IR (env already embedded)
      setCompiledTopology(ir as unknown as Record<string, unknown>);
      const { report } = await analyze({
        topology: ir as unknown as Record<string, unknown>,
      });

      // Parse report using shared helper
      const parsed = parseAnalysisReport(report, hwConfig.precision, hwConfig.batchSize);
      setAnalysis(parsed.analysis);
      setPerLayer(parsed.perLayer);
      setWarnings(parsed.warnings);

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

      let skipToast = false;

      // Handle Authentication / Authorization
      if (err instanceof NeuraxApiError && (err.status === 401 || err.status === 403)) {
        const bodyStr = typeof err.body === 'string' ? err.body : '';
        const bodyLower = bodyStr.toLowerCase();
        const isEmailNotVerified = err.status === 403
          && bodyLower.includes('verify')
          && bodyLower.includes('email');

        const msg = err.status === 401
          ? 'Please sign in to run analysis.'
          : isEmailNotVerified
            ? 'Please verify your email (check your inbox) to run analysis.'
            : 'Your plan does not allow this analysis.';

        setWarnings([
          { id: 'auth', type: 'error', message: msg },
        ]);

        toast({
          title: err.status === 401
            ? 'Authentication required'
            : isEmailNotVerified
              ? 'Email verification required'
              : 'Upgrade required',
          description: msg,
          variant: 'destructive',
        });

        return;
      }

      // Handle 400 Bad Request (Compilation Errors/Warnings)
      if (err instanceof NeuraxApiError && err.status === 400) {
        skipToast = true; // DO NOT show a intrusive toast for 400s

        // Attempt to parse diagnostics from the error body
        if (err.body && typeof err.body === 'string') {
          try {
            const bodyJson = JSON.parse(err.body);
            const rpt = (bodyJson?.report ?? bodyJson) as Record<string, unknown>;

            if (Array.isArray(rpt?.diagnostics)) {
              const parsedDiagnostics = rpt.diagnostics.map((diag: any) => ({
                category: typeof diag?.category === 'string' ? diag.category.toLowerCase() : 'compiler',
                severity: typeof diag?.severity === 'string' ? diag.severity.toLowerCase() : 'error',
                code: typeof diag?.code === 'string' ? diag.code : 'E_COMPILE',
                message: typeof diag?.message === 'string' ? diag.message : 'Unknown compilation error',
                layer_id: typeof diag?.layer_id === 'string' ? diag.layer_id : undefined,
                suggestion: typeof diag?.suggestion === 'string' ? diag.suggestion : undefined,
                precision_impact: typeof diag?.precision_impact === 'number' ? diag.precision_impact : undefined,
              }));

              // Use these diagnostics as warnings so they show up in the issues panel
              setWarnings(parsedDiagnostics.map((d: any, idx: number) => ({
                id: `diag-${idx}`,
                type: d.severity === 'info' ? 'warning' : 'error',
                message: d.message,
                code: d.code,
              })));

              // Also update analysis to show these diagnostics in the dedicated panel
              setAnalysis(prev => ({
                ...prev,
                diagnostics: parsedDiagnostics,
                diagnosticCount: parsedDiagnostics.length,
              }));
            } else {
              setWarnings([{ id: 'compile-fail', type: 'error', message: 'Architecture compilation failed. Please check the block connections.' }]);
            }
          } catch (e) {
            setWarnings([{ id: 'bad-request', type: 'error', message: 'The compiler rejected the current topology (400).' }]);
          }
        }
      }

      if (!skipToast) {
        setAnalysis(initialAnalysis);
        setPerLayer([]);

        // Local validation warnings
        const localWarnings: Warning[] = [];
        if (!nodes.some(n => n.type === 'input')) {
          localWarnings.push({ id: 'no-input', type: 'error', message: 'Missing Input layer.' });
        }
        if (!nodes.some(n => n.type === 'output')) {
          localWarnings.push({ id: 'no-output', type: 'error', message: 'Missing Output layer.' });
        }
        setWarnings(localWarnings.length > 0 ? localWarnings : [
          { id: 'offline', type: 'warning', message: 'Backend unreachable — metrics unavailable.' },
        ]);

        const errorMsg = err instanceof NeuraxApiError
          ? `Server returned ${err.status}`
          : 'Backend unreachable';

        toast({
          title: "Analysis (offline mode)",
          description: `${errorMsg} — metrics unavailable`,
          variant: "destructive",
        });
      }
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

    try {
      const ir = compileToNeuraxIR(nodes, connections, {
        modelName: 'NeuraxModel',
        family: selectedArchitecture,
        hardware: hwConfig.hardware,
        precision: hwConfig.precision,
        batchSize: hwConfig.batchSize,
        groups,
        learningRate: hwConfig.learningRate,
        numEpochs: hwConfig.numEpochs,
        seqLen: hwConfig.seqLen,
        gpuCount: hwConfig.gpuCount,
        gpuMemoryGb: hwConfig.gpuMemoryGb,
        datasetSize: hwConfig.datasetSize,
        vocabSize: hwConfig.vocabSize,
        numClasses: hwConfig.numClasses,
      });

      setCompiledTopology(ir as unknown as Record<string, unknown>);

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
            onDiagnostic: (diag) => {
              // Optionally show diagnostics as they arrive
              console.log('[neurax] Streaming diagnostic:', diag);
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
              const parsed = parseAnalysisReport(result, hwConfig.precision, hwConfig.batchSize);
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
    const catalogueByType = new Map<string, { type: string; name: string; category: any; defaultParams: any }>();
    for (const l of allLayers) {
      if (!l?.type) continue;
      if (!catalogueByType.has(l.type)) {
        catalogueByType.set(l.type, {
          type: l.type,
          name: l.name,
          category: l.category,
          defaultParams: l.defaultParams,
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

  const handleAgentToolEvent = useCallback((tool: { name: string; args?: Record<string, unknown> }) => {
    const name = tool?.name;
    const args = tool?.args ?? {};

    if (name === 'clear_canvas') {
      handleClearCanvas();
      return;
    }

    if (name === 'set_hw_config') {
      const updates = (args as any)?.updates;
      if (!updates || typeof updates !== 'object') return;
      updateHwConfig(updates as any);
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

      const cfg = layerConfigByType.get(layerType);
      if (!cfg) {
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

    if (name === 'move_node') {
      const nodeId = String(args.node_id ?? '');
      const x = Number(args.x ?? 0);
      const y = Number(args.y ?? 0);
      if (!nodeId) return;
      handleUpdateNode(nodeId, { x, y });
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
      handleAddConnection(fromId, toId, true);
      triggerAgentAutoAnalysis();
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
      const validTabs: WorkspaceTab[] = ['architecture', 'simulation', 'production', 'inference', 'timemachine'];
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
  }, [handleClearCanvas, layerConfigByType, toast, handleAddNode, handleUpdateNode, handleAddConnection, handleDeleteConnection, handleDeleteNode, handleSelectNode, triggerAgentAutoAnalysis, handleArchitectureChange, updateHwConfig, nodes, connections, setActiveWorkspaceTab, handleRunAnalysis]);

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
    setNodes(hydrateNodesForFamily(targetFamily, result.nodes));
    setConnections(result.connections);
    setSelectedNodeId(null);

    // 4. Re-run analysis after import
    setTimeout(() => {
      handleRunAnalysis();
    }, 500);
  }, [handleArchitectureChange, updateHwConfig, selectedArchitecture, hydrateNodesForFamily, handleRunAnalysis]);

  // Architecture workspace content
  const architectureContent = (
    <div className="flex-1 flex overflow-hidden min-w-0">
      {isChatOpen && !isDesktopLayout ? (
        <Sheet open={isChatOpen} onOpenChange={setIsChatOpen}>
          <SheetContent side="left" className="p-0 w-[360px] sm:w-[420px]">
            <AIChatDrawer
              open={isChatOpen}
              onOpenChange={setIsChatOpen}
              onAddCredits={() => setShowPricingPage(true)}
              creditsLeft={creditInfo ? creditInfo.limit - creditInfo.used : undefined}
              creditsLimit={creditInfo?.limit}
              getSnapshot={agentGetSnapshot}
              onToolEvent={handleAgentToolEvent}
              className="h-full"
            />
          </SheetContent>
        </Sheet>
      ) : null}

      {isDesktopLayout ? (
        <ResizablePanelGroup direction="horizontal" className="flex-1 min-w-0">
          {isChatOpen ? (
            <>
              <ResizablePanel defaultSize={28} minSize={18} maxSize={42} className="border-r border-border/70 min-w-0">
                <AIChatDrawer
                  open={isChatOpen}
                  onOpenChange={setIsChatOpen}
                  onAddCredits={() => setShowPricingPage(true)}
                  creditsLeft={creditInfo ? creditInfo.limit - creditInfo.used : undefined}
                  creditsLimit={creditInfo?.limit}
                  getSnapshot={agentGetSnapshot}
                  onToolEvent={handleAgentToolEvent}
                  className="h-full"
                />
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
                    setActiveRightPanelTab('issues');
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
                setActiveRightPanelTab('issues');
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
        onRunAnalysis={handleRunAnalysisStream}
        isAnalyzing={isAnalyzing}
        onNewCanvas={handleCreateNewCanvas}
        onSaveCanvas={handleSaveCanvas}
        onExport={() => setShowExportPanel(true)}
        onImport={() => setShowImportPanel(true)}
        isChatOpen={isChatOpen}
        onToggleChat={() => setIsChatOpen((v) => !v)}
        selectedArchitecture={selectedArchitecture}
        onArchitectureChange={handleArchitectureChange}
        onOpenPricing={() => setShowPricingPage(true)}
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
          architectureContent={architectureContent}
          simulationContent={<SimulationWorkspace nodes={nodes} connections={connections} analysis={analysis} perLayer={perLayer} warnings={warnings} topology={compiledTopology} />}
          productionContent={<ProductionWorkspace nodes={nodes} connections={connections} modelName="NeuraxModel" />}
          inferenceContent={<InferenceIntelligence architectureType={selectedArchitecture} />}
          timeMachineContent={<TimeMachineWorkspace nodes={nodes} connections={connections} analysis={analysis} />}
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

      <ExportPanel
        isOpen={showExportPanel}
        onClose={() => setShowExportPanel(false)}
        architectureName="NeuraxModel"
        nodes={nodes}
        connections={connections}
        groups={groups}
        selectedArchitecture={selectedArchitecture}
      />

      <ImportPanel
        isOpen={showImportPanel}
        onClose={() => setShowImportPanel(false)}
        onImport={handleImportArchitecture}
      />

      <PricingPage
        isOpen={showPricingPage}
        onClose={() => setShowPricingPage(false)}
      />
    </div>
  );
};

export default Index;
