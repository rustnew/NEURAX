import { HardwareDetail } from '@/services/neuraxApi.ts';
import { AnalysisResult, CanvasNode, Connection, PerLayerBreakdownRow, Warning } from '@/types/architecture.ts';

const BYTES_IN_MB = 1024 ** 2;
const BYTES_IN_GB = 1024 ** 3;

export const SIMULATION_COLORS = {
  blue: '#0ea5e9',
  green: '#10b981',
  amber: '#f59e0b',
  red: '#ef4444',
  violet: '#8b5cf6',
  cyan: '#22d3ee',
  slate: '#64748b',
  indigo: '#6366f1',
  teal: '#14b8a6',
};

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function formatBytes(bytes: number, digits = 1): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 10 ? 0 : digits)} ${units[unit]}`;
}

export function formatCompactNumber(value: number, digits = 1): string {
  if (!Number.isFinite(value)) return '0';
  const abs = Math.abs(value);
  if (abs >= 1e12) return `${(value / 1e12).toFixed(digits)}T`;
  if (abs >= 1e9) return `${(value / 1e9).toFixed(digits)}B`;
  if (abs >= 1e6) return `${(value / 1e6).toFixed(digits)}M`;
  if (abs >= 1e3) return `${(value / 1e3).toFixed(digits)}K`;
  return `${value.toFixed(abs >= 10 ? 0 : digits)}`;
}

export function formatPercent(ratio: number, digits = 0): string {
  return `${(ratio * 100).toFixed(digits)}%`;
}

export function hasAnalysisReportData(analysis?: AnalysisResult | null): analysis is AnalysisResult {
  if (!analysis) return false;

  return (
    analysis.totalParams > 0
    || analysis.numLayers > 0
    || analysis.totalOperations > 0
    || analysis.totalTensorCount > 0
    || analysis.peakVramBytes > 0
    || analysis.latencyMs !== null
    || analysis.diagnosticCount !== undefined && analysis.diagnosticCount > 0
    || (analysis.diagnostics?.length ?? 0) > 0
    || (analysis.reportWarnings?.length ?? 0) > 0
    || Object.keys(analysis.opsDistribution ?? {}).length > 0
  );
}

export function hasCompilationProgress(analysis?: AnalysisResult | null): boolean {
  if (!analysis?.compilation) return false;
  const { current_phase, total_progress, phase_timeline } = analysis.compilation;
  return Boolean(
    (typeof current_phase === 'string' && current_phase.trim().length > 0)
    || typeof total_progress === 'number'
    || (phase_timeline?.length ?? 0) > 0,
  );
}

export function hasPhaseTimeline(analysis?: AnalysisResult | null): boolean {
  return (analysis?.compilation?.phase_timeline?.length ?? 0) > 0;
}

export function hasLivePartialMetrics(analysis?: AnalysisResult | null): boolean {
  return (analysis?.live_trace?.partial_metrics?.length ?? 0) > 0;
}

export function hasLiveThroughputTrace(analysis?: AnalysisResult | null): boolean {
  return (analysis?.live_trace?.throughput_trace?.length ?? 0) > 0;
}

export function hasMemoryHeatmap(analysis?: AnalysisResult | null): boolean {
  return (analysis?.memory_heatmap?.length ?? 0) > 0 || (analysis?.live_trace?.memory_heatmap?.length ?? 0) > 0;
}

export function hasMemoryLiveness(analysis?: AnalysisResult | null): boolean {
  return (analysis?.memory_liveness?.length ?? 0) > 0 || (analysis?.live_trace?.memory_liveness?.length ?? 0) > 0;
}

export function hasGradientMemoryBreakdown(analysis?: AnalysisResult | null): boolean {
  return (analysis?.gradient_memory_breakdown?.length ?? 0) > 0 || (analysis?.live_trace?.gradient_memory_breakdown?.length ?? 0) > 0;
}

export function hasKvCacheScaling(analysis?: AnalysisResult | null): boolean {
  return (analysis?.kv_cache_scaling?.length ?? 0) > 0 || (analysis?.live_trace?.kv_cache_scaling?.length ?? 0) > 0;
}

export function hasPerLayerRows(perLayer?: ArrayLike<unknown> | null): boolean {
  return (perLayer?.length ?? 0) > 0;
}

export function hasPerLayerLatencyMap(analysis?: AnalysisResult | null): boolean {
  return Object.keys(analysis?.perLayerLatency ?? {}).length > 0;
}

export function hasPerLayerVramMap(analysis?: AnalysisResult | null): boolean {
  return Object.keys(analysis?.perLayerVram ?? {}).length > 0;
}

export function normalizeSeverity(value?: string): 'critical' | 'warning' | 'info' | 'hint' {
  const normalized = (value ?? '').toLowerCase();
  if (normalized === 'critical' || normalized === 'error') return 'critical';
  if (normalized === 'warning') return 'warning';
  if (normalized === 'hint') return 'hint';
  return 'info';
}

export function severityColor(severity: string): string {
  switch (normalizeSeverity(severity)) {
    case 'critical':
      return SIMULATION_COLORS.red;
    case 'warning':
      return SIMULATION_COLORS.amber;
    case 'hint':
      return SIMULATION_COLORS.teal;
    default:
      return SIMULATION_COLORS.blue;
  }
}

export function severityWeight(severity: string): number {
  switch (normalizeSeverity(severity)) {
    case 'critical':
      return 3;
    case 'warning':
      return 2;
    case 'hint':
      return 1;
    default:
      return 1;
  }
}

export function normalizePhaseStatus(status?: string): 'completed' | 'inprogress' | 'pending' | 'failed' {
  const normalized = (status ?? '').toLowerCase();
  if (normalized.includes('complete')) return 'completed';
  if (normalized.includes('progress') || normalized.includes('running')) return 'inprogress';
  if (normalized.includes('fail') || normalized.includes('error')) return 'failed';
  return 'pending';
}

export function parseFlopsValue(value: string | number | undefined): number {
  if (value === undefined || value === null) return 0;
  if (typeof value === 'number') return value;
  const match = value.trim().match(/^(-?\d+(\.\d+)?)\s*([A-Za-z]+)?$/);
  if (!match) return 0;
  let numeric = parseFloat(match[1]);
  const unit = match[3]?.toUpperCase() ?? '';
  if (unit.startsWith('TFLOP') || unit === 'T') numeric *= 1e12;
  else if (unit.startsWith('GFLOP') || unit === 'G') numeric *= 1e9;
  else if (unit.startsWith('MFLOP') || unit === 'M') numeric *= 1e6;
  else if (unit === 'K') numeric *= 1e3;
  return numeric;
}

export interface DerivedLayerMetric {
  key: string;
  name: string;
  kind: string;
  flops: number;
  params: number;
  share: number;
  latencyMs: number;
  weightsMb: number;
  activationsMb: number;
  gradientsMb: number;
  optimizerMb: number;
  forwardMb: number;
  backwardMb: number;
  memoryMb: number;
}

function classifyLayerKind(name: string): string {
  const value = name.toLowerCase();
  if (value.includes('attn') || value.includes('attention')) return 'attention';
  if (value.includes('ffn') || value.includes('mlp') || value.includes('expert')) return 'ffn';
  if (value.includes('embed')) return 'embedding';
  if (value.includes('norm')) return 'norm';
  if (value.includes('conv') || value.includes('unet')) return 'conv';
  if (value.includes('head') || value.includes('classifier')) return 'head';
  return 'other';
}

export function buildDerivedLayerMetrics(
  analysis: AnalysisResult,
  perLayer: PerLayerBreakdownRow[] = [],
): DerivedLayerMetric[] {
  const rawRows = perLayer.length > 0
    ? perLayer
    : Object.entries(analysis.opsDistribution ?? {})
      .map(([name, count]) => ({
        id: name,
        name,
        params: Math.round((analysis.totalParams || 0) * (count / Math.max(1, Object.values(analysis.opsDistribution).reduce((sum, value) => sum + value, 0)))),
        flops: ((analysis.totalFlops || 0) * (count / Math.max(1, Object.values(analysis.opsDistribution).reduce((sum, value) => sum + value, 0)))).toString(),
      }));

  const totalFlops = rawRows.reduce((sum, row) => sum + parseFlopsValue(row.flops), 0) || 1;
  const totalParams = rawRows.reduce((sum, row) => sum + (row.params ?? 0), 0) || 1;
  const parameterMb = analysis.parameterMemoryBytes / BYTES_IN_MB;
  const activationMb = analysis.activationMemoryBytes / BYTES_IN_MB;
  const gradientMb = analysis.gradientMemoryBytes / BYTES_IN_MB;
  const optimizerMb = analysis.optimizerStateBytes / BYTES_IN_MB;
  const gradientBreakdown = new Map(
    (analysis.gradient_memory_breakdown ?? analysis.live_trace?.gradient_memory_breakdown ?? []).map((entry) => [
      entry.name.toLowerCase(),
      {
        forwardMb: entry.forward / BYTES_IN_MB,
        backwardMb: entry.backward / BYTES_IN_MB,
      },
    ]),
  );

  return rawRows
    .map((row, index) => {
      const name = row.name ?? row.id ?? `Layer ${index + 1}`;
      const key = row.id ?? name;
      const flops = parseFlopsValue(row.flops);
      const params = row.params ?? 0;
      const flopsShare = flops / totalFlops;
      const paramShare = params / totalParams;
      const grad = gradientBreakdown.get(name.toLowerCase()) ?? gradientBreakdown.get(key.toLowerCase());
      const weights = Math.max(parameterMb * paramShare, (params * 2) / BYTES_IN_MB);
      const activations = Math.max(activationMb * flopsShare, 0);
      const gradients = Math.max(gradientMb * paramShare, 0);
      const optimizer = Math.max(optimizerMb * paramShare, 0);
      const forwardMb = grad?.forwardMb ?? (weights * 0.35 + activations * 0.65);
      const backwardMb = grad?.backwardMb ?? (gradients * 0.7 + optimizer * 0.6);
      const memoryMb = weights + activations + gradients + optimizer;

      return {
        key,
        name,
        kind: classifyLayerKind(name),
        flops: flops / 1e9,
        params,
        share: flopsShare,
        latencyMs: (analysis.latencyMs ?? 0) * flopsShare,
        weightsMb: weights,
        activationsMb: activations,
        gradientsMb: gradients,
        optimizerMb: optimizer,
        forwardMb,
        backwardMb,
        memoryMb,
      };
    })
    .sort((a, b) => b.flops - a.flops)
    .slice(0, 10);
}

export interface HardwareProjection {
  name: string;
  latencyMs: number;
  throughput: number;
  fit: boolean;
  costIndex: number;
  peakTflops: number;
}

export function deriveHardwareProjections(
  analysis: AnalysisResult,
  hardwareList: HardwareDetail[],
): HardwareProjection[] {
  if (hardwareList.length === 0) return [];
  const achievedTflops = analysis.latencyMs && analysis.latencyMs > 0
    ? (analysis.totalFlops / 1e12) / (analysis.latencyMs / 1000)
    : Math.max(0.1, (analysis.gpuTflops || 1) * clamp(analysis.gpuUtilization ?? 0.45, 0.15, 1));
  const currentRoof = Math.max(
    0.1,
    Math.min(
      analysis.gpuTflops || 1,
      analysis.arithmeticIntensity > 0
        ? (analysis.arithmeticIntensity * (analysis.gpuBandwidthGbs || 1)) / 1000
        : analysis.gpuTflops || 1,
    ),
  );
  const efficiency = clamp(achievedTflops / currentRoof, 0.15, 1);
  const baselineThroughput = analysis.isSequenceModel
    ? analysis.throughputTokensPerS
    : (analysis.throughputGraphsPerS ?? 0);
  const baselineCost = Math.max(analysis.costPerMillionTokensUsd || 0, 1);

  return hardwareList
    .map((hardware) => {
      const peakTflops = hardware.tflops_fp16;
      const memoryRoof = analysis.arithmeticIntensity > 0
        ? (analysis.arithmeticIntensity * hardware.memory_bandwidth_gbs) / 1000
        : peakTflops;
      const attainable = Math.max(0.05, Math.min(peakTflops, memoryRoof) * efficiency);
      const latencyMs = analysis.totalFlops > 0
        ? ((analysis.totalFlops / 1e12) / attainable) * 1000
        : 0;
      const throughput = baselineThroughput > 0 && (analysis.latencyMs ?? 0) > 0
        ? baselineThroughput * ((analysis.latencyMs ?? latencyMs) / Math.max(latencyMs, 0.001))
        : attainable * 100;
      const vramBytes = hardware.memory_gb * 1024 * 1024 * 1024;
      const fit = vramBytes >= analysis.peakVramBytes;
      const costIndex = baselineCost * (latencyMs > 0 && (analysis.latencyMs ?? 0) > 0
        ? latencyMs / Math.max(analysis.latencyMs ?? latencyMs, 0.001)
        : 1);
      return {
        name: hardware.name,
        latencyMs,
        throughput,
        fit,
        costIndex,
        peakTflops,
      };
    })
    .sort((a, b) => b.throughput - a.throughput);
}

export function deriveBatchScaling(analysis: AnalysisResult): Array<{ batch: number; throughput: number; latencyMs: number }> {
  const currentBatch = Math.max(1, analysis.selectedBatchSize ?? 1);
  const limit = Math.max(currentBatch, Math.min(analysis.maxBatchSizeFit || 128, 128));
  const points = new Set<number>([1, 2, 4, 8, 16, 32, 64, 128, currentBatch, limit]);
  const ordered = Array.from(points).filter((value) => value <= Math.max(128, limit)).sort((a, b) => a - b);
  const throughputBase = Math.max(analysis.isSequenceModel ? analysis.throughputTokensPerS : (analysis.throughputGraphsPerS ?? 0), 1);
  const perSample = throughputBase / currentBatch;
  const latencyBase = Math.max(analysis.latencyMs ?? 0, 1);

  return ordered.map((batch) => {
    const utilizationGain = 0.72 + 0.28 * (1 - Math.exp(-batch / Math.max(currentBatch, 2)));
    const saturationPenalty = batch > limit ? clamp(1 - ((batch - limit) / batch), 0.45, 1) : 1;
    const throughput = perSample * batch * utilizationGain * saturationPenalty;
    const latencyMs = latencyBase * (0.65 + 0.35 * (batch / currentBatch));
    return { batch, throughput, latencyMs };
  });
}

export function deriveMultiGpuScaling(analysis: AnalysisResult): Array<{ gpus: number; actual: number; ideal: number }> {
  const target = Math.max(8, analysis.optimalGpuCount || analysis.gpuCount || 1);
  const points = [1, 2, 4, 8, 16].filter((value) => value <= target);
  if (!points.includes(target)) points.push(target);
  const ordered = Array.from(new Set(points)).sort((a, b) => a - b);
  const efficiency = clamp(
    (analysis.dataParallelEfficiency || 0.6) * (1 - (analysis.communicationOverhead || 0) * 0.25),
    0.25,
    0.98,
  );

  return ordered.map((gpus) => {
    const actual = gpus === 1
      ? 1
      : clamp((gpus * efficiency) / (1 + (analysis.communicationOverhead || 0) * Math.log2(gpus)), 1, gpus);
    return { gpus, actual, ideal: gpus };
  });
}

function precisionBytes(precision?: string): number {
  const normalized = (precision ?? 'fp16').toLowerCase();
  if (normalized === 'fp32') return 4;
  if (normalized === 'int8') return 1;
  if (normalized === 'int4') return 0.5;
  return 2;
}

export function derivePrecisionComparison(analysis: AnalysisResult): Array<{
  precision: string;
  latencyPct: number;
  vramPct: number;
  confidencePct: number;
  speedup: number;
  memorySavingsPct: number;
}> {
  const precisions = [
    { precision: 'FP32', bytes: 4, penalty: 0 },
    { precision: 'BF16', bytes: 2, penalty: 0.01 },
    { precision: 'INT8', bytes: 1, penalty: 0.03 },
    { precision: 'INT4', bytes: 0.5, penalty: 0.1 },
  ];
  const currentBytes = precisionBytes(analysis.selectedPrecision);
  const currentLatency = Math.max(analysis.latencyMs ?? 0, 1);
  const currentVram = Math.max(analysis.peakVramBytes, BYTES_IN_GB);
  const fp32Latency = currentLatency * Math.pow(4 / currentBytes, 0.65);
  const fp32Vram = currentVram * (4 / currentBytes);

  return precisions.map(({ precision, bytes, penalty }) => {
    const latency = currentLatency * Math.pow(bytes / currentBytes, 0.65);
    const vram = currentVram * (bytes / currentBytes);
    const confidencePct = clamp((analysis.confidenceScore - penalty) * 100, 45, 100);
    return {
      precision,
      latencyPct: clamp((latency / fp32Latency) * 100, 5, 160),
      vramPct: clamp((vram / fp32Vram) * 100, 5, 160),
      confidencePct,
      speedup: clamp(fp32Latency / latency, 0.5, 8),
      memorySavingsPct: clamp((1 - (vram / fp32Vram)) * 100, 0, 90),
    };
  });
}

export function deriveVariantRows(analysis: AnalysisResult): Array<{
  label: string;
  paramsLabel: string;
  memoryLabel: string;
  speedLabel: string;
  confidenceLabel: string;
  tone: string;
}> {
  const baselineThroughput = analysis.isSequenceModel
    ? analysis.throughputTokensPerS
    : (analysis.throughputGraphsPerS ?? 0);
  const scaling = deriveMultiGpuScaling(analysis);
  const optimalScale = scaling.find((item) => item.gpus === (analysis.optimalGpuCount || 1))?.actual ?? 1;

  return [
    {
      label: 'Baseline',
      paramsLabel: formatCompactNumber(analysis.totalParams, 1),
      memoryLabel: formatBytes(analysis.peakVramBytes),
      speedLabel: '1.0x',
      confidenceLabel: `${Math.round(analysis.confidenceScore * 100)}%`,
      tone: SIMULATION_COLORS.blue,
    },
    {
      label: 'Checkpointed',
      paramsLabel: formatCompactNumber(analysis.totalParams, 1),
      memoryLabel: formatBytes(Math.max(0, analysis.peakVramBytes - analysis.activationMemoryBytes * 0.45)),
      speedLabel: '0.88x',
      confidenceLabel: `${Math.round(analysis.confidenceScore * 100)}%`,
      tone: SIMULATION_COLORS.teal,
    },
    {
      label: 'INT8 Quant',
      paramsLabel: formatCompactNumber(analysis.totalParams, 1),
      memoryLabel: formatBytes(analysis.peakVramBytes * 0.55),
      speedLabel: '1.35x',
      confidenceLabel: `${Math.max(0, Math.round(analysis.confidenceScore * 100) - 3)}%`,
      tone: SIMULATION_COLORS.green,
    },
    {
      label: 'Scaled Cluster',
      paramsLabel: formatCompactNumber(analysis.totalParams, 1),
      memoryLabel: formatBytes(Math.max(analysis.peakVramBytes / Math.max(analysis.optimalGpuCount || 1, 1), 1)),
      speedLabel: `${optimalScale.toFixed(1)}x`,
      confidenceLabel: `${Math.round(analysis.confidenceScore * 100)}%`,
      tone: SIMULATION_COLORS.amber,
    },
  ].filter((row) => baselineThroughput > 0 || row.label !== 'Scaled Cluster');
}

function scoreFromPriority(priority: string): number {
  const normalized = priority.toLowerCase();
  if (normalized === 'high') return 80;
  if (normalized === 'low') return 35;
  return 55;
}

function scoreFromImpact(impact: string, priority: string): number {
  const numeric = impact.match(/(\d+(\.\d+)?)/);
  if (!numeric) return scoreFromPriority(priority);
  const value = parseFloat(numeric[1]);
  if (/x/i.test(impact)) return clamp(value * 15, 20, 95);
  if (/%/.test(impact)) return clamp(value, 15, 95);
  if (/gb/i.test(impact)) return clamp(value * 12, 20, 95);
  return clamp(scoreFromPriority(priority) + value, 20, 95);
}

export function deriveOptimizationOpportunities(analysis: AnalysisResult): Array<{
  title: string;
  description: string;
  score: number;
  priority: string;
}> {
  const recommendations = analysis.recommendations ?? [];
  if (recommendations.length > 0) {
    return recommendations.map((recommendation) => ({
      title: recommendation.title,
      description: recommendation.impact || recommendation.description,
      score: scoreFromImpact(recommendation.impact, recommendation.priority),
      priority: recommendation.priority,
    }));
  }

  const fallback = [];
  if (analysis.bottleneck === 'memory-bound') {
    fallback.push({
      title: 'Higher Bandwidth GPU',
      description: 'Reduce the memory roofline limit.',
      score: 72,
      priority: 'medium',
    });
  }
  if (analysis.activationMemoryBytes > 0) {
    fallback.push({
      title: 'Gradient Checkpointing',
      description: `Recover ~${formatBytes(analysis.activationMemoryBytes * 0.45)} of activation VRAM.`,
      score: 78,
      priority: 'high',
    });
  }
  fallback.push({
    title: 'Batch Tuning',
    description: `Scale up toward batch ${analysis.maxBatchSizeFit || 1} for better device occupancy.`,
    score: 48,
    priority: 'medium',
  });
  return fallback;
}

export function deriveZeroStageComparison(analysis: AnalysisResult): Array<{
  stage: string;
  weightsMb: number;
  gradientsMb: number;
  optimizerMb: number;
}> {
  const gpus = Math.max(analysis.optimalGpuCount || analysis.gpuCount || 1, 1);
  const weights = analysis.parameterMemoryBytes / BYTES_IN_MB;
  const gradients = analysis.gradientMemoryBytes / BYTES_IN_MB;
  const optimizer = analysis.optimizerStateBytes / BYTES_IN_MB;
  return [
    {
      stage: 'ZeRO-1',
      weightsMb: weights,
      gradientsMb: gradients,
      optimizerMb: optimizer / gpus,
    },
    {
      stage: 'ZeRO-2',
      weightsMb: weights,
      gradientsMb: gradients / gpus,
      optimizerMb: optimizer / gpus,
    },
    {
      stage: 'ZeRO-3',
      weightsMb: weights / gpus,
      gradientsMb: gradients / gpus,
      optimizerMb: optimizer / gpus,
    },
  ];
}

export function deriveLoraSavings(analysis: AnalysisResult): Array<{ name: string; value: number }> {
  const baseline = (analysis.parameterMemoryBytes + analysis.gradientMemoryBytes + analysis.optimizerStateBytes) / BYTES_IN_GB;
  return [
    { name: 'Full FT', value: baseline },
    { name: 'LoRA r=16', value: baseline * 0.16 },
    { name: 'LoRA r=8', value: baseline * 0.11 },
    { name: 'QLoRA', value: baseline * 0.07 },
  ];
}

export function deriveTrainingScaleCurve(analysis: AnalysisResult): Array<{ gpus: number; hours: number; idealHours: number }> {
  const baseHours = Math.max(analysis.trainingTimeHours || 0, 0.25);
  return deriveMultiGpuScaling(analysis).map((entry) => ({
    gpus: entry.gpus,
    hours: baseHours / entry.actual,
    idealHours: baseHours / entry.ideal,
  }));
}

export function deriveCommunicationCurve(analysis: AnalysisResult): Array<{ gpus: number; communicationPct: number; computePct: number }> {
  const scaling = deriveMultiGpuScaling(analysis);
  return scaling.map((entry) => {
    const communicationPct = clamp(
      (analysis.communicationOverhead || 0.05) * 100 * Math.max(1, Math.log2(entry.gpus || 1) + 1),
      4,
      85,
    );
    return {
      gpus: entry.gpus,
      communicationPct,
      computePct: clamp(100 - communicationPct, 15, 96),
    };
  });
}

function bucketCategory(category?: string, code?: string): 'shape' | 'memory' | 'parallel' | 'op' | 'config' | 'general' {
  const value = `${category ?? ''} ${code ?? ''}`.toLowerCase();
  if (value.includes('shape') || value.includes('e002') || value.includes('w002')) return 'shape';
  if (value.includes('memory') || value.includes('e001') || value.includes('w005')) return 'memory';
  if (value.includes('parallel') || value.includes('w006')) return 'parallel';
  if (value.includes('unsupported') || value.includes('custom') || value.includes('op')) return 'op';
  if (value.includes('config')) return 'config';
  return 'general';
}

export function deriveIssueSummary(
  analysis: AnalysisResult,
  warnings: Warning[] = [],
): Array<{ severity: string; count: number; fill: string }> {
  if (!hasAnalysisReportData(analysis)) return [];

  const diagnostics = analysis.diagnostics ?? [];
  const counts = {
    critical: 0,
    warning: 0,
    info: 0,
    hint: 0,
  };

  diagnostics.forEach((diagnostic) => {
    counts[normalizeSeverity(diagnostic.severity)] += 1;
  });
  warnings.forEach((warning) => {
    if (warning.type === 'error') counts.critical += 1;
    else if (warning.type === 'warning') counts.warning += 1;
    else counts.info += 1;
  });

  return [
    { severity: 'Critical', count: counts.critical, fill: SIMULATION_COLORS.red },
    { severity: 'Warning', count: counts.warning, fill: SIMULATION_COLORS.amber },
    { severity: 'Info', count: counts.info, fill: SIMULATION_COLORS.blue },
    { severity: 'Hint', count: counts.hint, fill: SIMULATION_COLORS.green },
  ].filter((entry) => entry.count > 0);
}

export function deriveDiagnosticsByLayer(
  analysis: AnalysisResult,
  warnings: Warning[] = [],
  perLayer: PerLayerBreakdownRow[] = [],
  nodes: CanvasNode[] = [],
): Array<Record<string, string | number>> {
  if (!hasAnalysisReportData(analysis)) return [];

  const labelById = new Map(nodes.map((node) => [node.id, node.name || node.type]));
  const rows = new Map<string, Record<string, string | number>>();

  const ensureRow = (key: string) => {
    if (!rows.has(key)) {
      rows.set(key, {
        layer: key,
        shape: 0,
        memory: 0,
        parallel: 0,
        op: 0,
        config: 0,
        general: 0,
      });
    }
    return rows.get(key)!;
  };

  (analysis.diagnostics ?? []).forEach((diagnostic) => {
    const rowKey = diagnostic.layer_id ? (labelById.get(diagnostic.layer_id) ?? diagnostic.layer_id) : 'Global';
    const row = ensureRow(rowKey);
    const bucket = bucketCategory(diagnostic.category, diagnostic.code);
    row[bucket] = (row[bucket] as number) + severityWeight(diagnostic.severity);
  });

  warnings.forEach((warning) => {
    const rowKey = warning.nodeId ? (labelById.get(warning.nodeId) ?? warning.nodeId) : 'Global';
    const row = ensureRow(rowKey);
    row.general = (row.general as number) + severityWeight(warning.type);
  });

  if (rows.size === 0) {
    perLayer.slice(0, 5).forEach((row) => ensureRow(row.name));
  }

  return Array.from(rows.values()).slice(0, 8);
}

export function deriveConfidenceBars(
  analysis: AnalysisResult,
  warnings: Warning[] = [],
): Array<{ label: string; value: number; fill: string }> {
  if (!hasAnalysisReportData(analysis)) return [];

  const diagnostics = analysis.diagnostics ?? [];
  const critical = diagnostics.filter((diagnostic) => normalizeSeverity(diagnostic.severity) === 'critical').length
    + warnings.filter((warning) => warning.type === 'error').length;
  const warningCount = diagnostics.filter((diagnostic) => normalizeSeverity(diagnostic.severity) === 'warning').length
    + warnings.filter((warning) => warning.type === 'warning').length;
  const tensorConfidence = analysis.totalTensorCount > 0
    ? (1 - (analysis.unresolvedDimCount / analysis.totalTensorCount))
    : analysis.confidenceScore;
  const diagnosticCleanliness = clamp(1 - (critical * 0.2 + warningCount * 0.08), 0, 1);
  const memoryFit = analysis.gpuMemoryGb > 0
    ? clamp(1 - (analysis.peakVramBytes / (analysis.gpuMemoryGb * BYTES_IN_GB)) + 0.35, 0, 1)
    : 0.6;

  return [
    { label: 'Shape', value: clamp(analysis.tensorResolutionRatio * 100, 0, 100), fill: SIMULATION_COLORS.green },
    { label: 'Concrete Dims', value: clamp(tensorConfidence * 100, 0, 100), fill: SIMULATION_COLORS.teal },
    { label: 'Diagnostics', value: clamp(diagnosticCleanliness * 100, 0, 100), fill: SIMULATION_COLORS.blue },
    { label: 'Memory Fit', value: clamp(memoryFit * 100, 0, 100), fill: SIMULATION_COLORS.amber },
    { label: 'Overall', value: clamp(analysis.confidenceScore * 100, 0, 100), fill: SIMULATION_COLORS.indigo },
  ];
}

export function deriveUnsupportedOps(analysis: AnalysisResult): Array<{
  name: string;
  detail: string;
  count: number;
  severity: string;
}> {
  if (!hasAnalysisReportData(analysis)) return [];

  const items: Array<{ name: string; detail: string; count: number; severity: string }> = [];
  const customWarnings = (analysis.reportWarnings ?? []).filter((warning) => /custom|unsupported|estimated flops/i.test(warning));
  customWarnings.forEach((warning) => {
    items.push({
      name: 'custom_ops',
      detail: warning,
      count: analysis.customLayerCount ?? 1,
      severity: 'warning',
    });
  });

  (analysis.diagnostics ?? [])
    .filter((diagnostic) => {
      const code = diagnostic.code?.toLowerCase() ?? '';
      return code === 'e003' || code === 'e004' || code === 'w001';
    })
    .forEach((diagnostic) => {
      items.push({
        name: diagnostic.code?.toLowerCase() ?? diagnostic.category,
        detail: diagnostic.message,
        count: 1,
        severity: diagnostic.severity,
      });
    });

  if (items.length === 0) {
    items.push({
      name: 'supported_ops',
      detail: 'No unsupported or fallback operations were reported by the compiler.',
      count: 0,
      severity: 'info',
    });
  }

  return items;
}

export function deriveResolutionDistribution(
  analysis: AnalysisResult,
  _warnings: Warning[] = [],
): Array<{ name: string; value: number; fill: string }> {
  if (!hasAnalysisReportData(analysis) || analysis.totalTensorCount <= 0) return [];

  const total = Math.max(Math.round(analysis.totalTensorCount || 0), 0);
  const certain = Math.min(total, Math.max(0, Math.round(total * clamp(analysis.tensorResolutionRatio, 0, 1))));

  let remaining = Math.max(0, total - certain);
  const unknown = Math.min(remaining, Math.max(analysis.customLayerCount ?? 0, 0));
  remaining -= unknown;

  const ambiguous = analysis.unresolvedDimCount > 0 ? remaining : 0;
  const inferred = Math.max(0, remaining - ambiguous);

  return [
    { name: 'Certain', value: certain, fill: SIMULATION_COLORS.green },
    { name: 'Inferred', value: inferred, fill: SIMULATION_COLORS.blue },
    { name: 'Ambiguous', value: ambiguous, fill: SIMULATION_COLORS.amber },
    { name: 'Unknown', value: unknown, fill: SIMULATION_COLORS.red },
  ].filter((entry) => entry.value > 0);
}

export function derivePenaltyWaterfall(analysis: AnalysisResult): Array<{ label: string; value: number; delta: number }> {
  if (!hasAnalysisReportData(analysis)) return [];

  const initial = 100;
  const afterShape = initial * clamp(analysis.tensorResolutionRatio, 0, 1);
  const hasCustomPenalty = (analysis.reportWarnings ?? []).some((warning) => /custom operations/i.test(warning));
  const afterCustom = hasCustomPenalty ? afterShape * 0.6 : afterShape;
  const afterDimensions = analysis.unresolvedDimCount > 0 ? afterCustom * 0.8 : afterCustom;
  const final = clamp(analysis.confidenceScore * 100, 0, 100);

  return [
    { label: 'Initial', value: initial, delta: 0 },
    { label: 'Shape', value: afterShape, delta: afterShape - initial },
    { label: 'Custom Ops', value: afterCustom, delta: afterCustom - afterShape },
    { label: 'Concrete Dims', value: afterDimensions, delta: afterDimensions - afterCustom },
    { label: 'Final', value: final, delta: final - afterDimensions },
  ];
}

export function deriveFusionCandidates(
  nodes: CanvasNode[] = [],
  connections: Connection[] = [],
  perLayer: PerLayerBreakdownRow[] = [],
): Array<{ label: string; gainPct: number; difficulty: string }> {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const layerWeights = new Map(
    perLayer.map((row) => [row.id, parseFlopsValue(row.flops)]),
  );
  const candidates = connections
    .map((connection) => {
      const source = byId.get(connection.from);
      const target = byId.get(connection.to);
      if (!source || !target) return null;
      const sourceType = source.type.toLowerCase();
      const targetType = target.type.toLowerCase();
      let gainPct = 0;
      let difficulty = 'Medium';

      if ((sourceType.includes('norm') && (targetType.includes('attention') || targetType.includes('ffn')))
        || (targetType.includes('norm') && (sourceType.includes('attention') || sourceType.includes('ffn')))) {
        gainPct = 8;
        difficulty = 'Easy';
      } else if ((sourceType.includes('linear') || sourceType.includes('projection'))
        && (targetType.includes('linear') || targetType.includes('projection') || targetType.includes('head'))) {
        gainPct = 10;
        difficulty = 'Easy';
      } else if ((sourceType.includes('attention') && targetType.includes('ffn'))
        || (sourceType.includes('conv') && targetType.includes('block'))) {
        gainPct = 14;
        difficulty = 'Medium';
      } else if (sourceType.includes('conv') || targetType.includes('conv')) {
        gainPct = 6;
        difficulty = 'Medium';
      } else {
        gainPct = 4;
      }

      const weightBoost = clamp(
        ((layerWeights.get(connection.from) ?? 0) + (layerWeights.get(connection.to) ?? 0)) / 5e9,
        0,
        10,
      );
      return {
        label: `${source.name || source.type} + ${target.name || target.type}`,
        gainPct: clamp(gainPct + weightBoost, 3, 24),
        difficulty,
      };
    })
    .filter((entry): entry is { label: string; gainPct: number; difficulty: string } => entry !== null)
    .sort((a, b) => b.gainPct - a.gainPct)
    .slice(0, 4);

  return candidates;
}
