import React, { useMemo } from 'react';
import {
  BarChart3,
  Zap,
  Cpu,
  HardDrive,
  Activity,
  Layers,
  DollarSign,
  Server,
  Gauge,
  FlaskConical,
  ShieldCheck,
  Binary,
  BrainCircuit,
} from 'lucide-react';
import { AnalysisResult, CanvasNode, PerLayerBreakdownRow } from '@/types/architecture.ts';
import { cn } from '@/lib/utils.ts';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.tsx';

interface MetricsDashboardProps {
  nodes: CanvasNode[];
  selectedNodeId: string | null;
  onSelectNode: (id: string) => void;
  analysis?: AnalysisResult;
  perLayer?: PerLayerBreakdownRow[];
  view?: 'overview' | 'deep';
}

// ─── Helpers ──────────────────────────────────────────────────────

function fmt(n: number | null | undefined, decimals = 2, showZero = false): string {
  if (n == null || !Number.isFinite(n)) return '—';
  if (n === 0 && !showZero) return '—';
  return n.toFixed(decimals);
}

function fmtBytes(bytes: number): string {
  if (!bytes || bytes === 0) return '—';
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(1)} KB`;
  return `${bytes} B`;
}

function fmtNum(n: number): string {
  if (!Number.isFinite(n) || n === 0) return '—';
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

function fmtPercent(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return '—';
  return `${(n * 100).toFixed(1)}%`;
}

// ─── Sub-components ───────────────────────────────────────────────

function SectionHeader({ icon: Icon, label }: { icon: React.ElementType; label: string }) {
  return (
    <div className="flex items-center gap-2 mb-2">
      <Icon className="w-3.5 h-3.5 text-primary" />
      <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        {label}
      </span>
    </div>
  );
}

function StatRow({ label, value, mono = true }: { label: string; value: string | number; mono?: boolean }) {
  return (
    <div className="flex items-center justify-between py-0.5">
      <span className="text-[10px] text-muted-foreground">{label}</span>
      <span className={cn('text-[10px] font-medium', mono && 'font-mono')}>{value || '—'}</span>
    </div>
  );
}

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="metric-card">
      <div className="text-[9px] text-muted-foreground uppercase tracking-wide mb-1">{label}</div>
      <div className="text-base font-bold font-mono text-foreground leading-tight">{value || '—'}</div>
      {sub && <div className="text-[9px] text-muted-foreground/60 mt-0.5">{sub}</div>}
    </div>
  );
}

function MemoryBar({ label, bytes, totalBytes, colorClass }: { label: string; bytes: number; totalBytes: number; colorClass: string }) {
  const pct = totalBytes > 0 ? Math.min((bytes / totalBytes) * 100, 100) : 0;
  return (
    <div className="space-y-0.5">
      <div className="flex justify-between text-[10px]">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono">{fmtBytes(bytes)}</span>
      </div>
      <div className="relative h-1.5 bg-secondary rounded-full overflow-hidden">
        <div className={cn('absolute inset-y-0 left-0 rounded-full transition-all', colorClass)} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────

export function MetricsDashboard({ nodes: _nodes, selectedNodeId, onSelectNode, analysis, perLayer, view = 'overview' }: MetricsDashboardProps) {
  const hasData = !!analysis && (analysis.totalParams > 0 || analysis.totalFlops > 0 || analysis.peakVramBytes > 0);

  const perLayerFlopsRows = useMemo(() => {
    if (!perLayer || perLayer.length === 0) return undefined;
    const parseFlopsHuman = (s: string | undefined): number => {
      if (!s) return 0;
      const m = s.trim().match(/^(-?\d+(?:\.\d+)?)\s*(TFLOPs|GFLOPs|MFLOPs|KFLOPs|FLOPs)$/i);
      if (!m) return 0;
      const v = Number(m[1]);
      switch (m[2].toUpperCase()) {
        case 'TFLOPS': return v * 1e12;
        case 'GFLOPS': return v * 1e9;
        case 'MFLOPS': return v * 1e6;
        case 'KFLOPS': return v * 1e3;
        default: return v;
      }
    };
    const rows = perLayer.map(r => ({ id: r.id, name: r.name, flopsHuman: r.flops ?? '—', flops: parseFlopsHuman(r.flops) })).filter(r => r.id && r.name);
    return rows.length > 0 ? rows : undefined;
  }, [perLayer]);

  const maxFlops = useMemo(() =>
    perLayerFlopsRows ? Math.max(...perLayerFlopsRows.map(r => r.flops), 1) : 1,
    [perLayerFlopsRows]
  );

  const totalVram = analysis?.peakVramBytes ?? 0;

  const bottleneckColor = analysis?.bottleneck === 'compute-bound'
    ? 'text-orange-400'
    : analysis?.bottleneck === 'memory-bound'
      ? 'text-blue-400'
      : 'text-muted-foreground';

  return (
    <div className="h-full flex flex-col bg-card overflow-hidden">
      {/* Header */}
      <div className="h-10 px-4 flex items-center border-b border-border bg-panel-header shrink-0">
        <BarChart3 className="w-4 h-4 text-primary mr-2" />
        <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Metrics Dashboard
        </span>
        {analysis && analysis.confidenceScore > 0 && (
          <span className={cn('ml-auto text-[10px] font-mono px-1.5 py-0.5 rounded',
            analysis.confidenceScore >= 0.9 ? 'bg-success/10 text-success' : 'bg-warning/10 text-warning'
          )}>
            {fmtPercent(analysis.confidenceScore)} conf
          </span>
        )}
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-thin p-3 space-y-4">
        {!hasData && (
          <div className="p-4 rounded-lg border border-border bg-secondary/20">
            <div className="text-xs text-muted-foreground">No metrics report available yet.</div>
            <div className="text-[10px] text-muted-foreground/80 mt-1">
              Run analysis to populate FLOPs, memory, and per-layer metrics.
            </div>
          </div>
        )}

        {hasData && (
          <Tabs defaultValue={view === 'overview' ? 'summary' : 'compute'} className="flex flex-col">
            <div className="mb-1">
              <TabsList className={cn(
                "grid h-8 w-full bg-secondary/40",
                view === 'overview' ? 'grid-cols-2' : 'grid-cols-3'
              )}>
                {view === 'overview' ? (
                  <>
                    <TabsTrigger value="summary" className="text-[10px] uppercase tracking-wide">Summary</TabsTrigger>
                    <TabsTrigger value="structure" className="text-[10px] uppercase tracking-wide">Structure</TabsTrigger>
                  </>
                ) : (
                  <>
                    <TabsTrigger value="compute" className="text-[10px] uppercase tracking-wide">Compute</TabsTrigger>
                    <TabsTrigger value="memory" className="text-[10px] uppercase tracking-wide">Memory</TabsTrigger>
                    <TabsTrigger value="system" className="text-[10px] uppercase tracking-wide">System</TabsTrigger>
                  </>
                )}
              </TabsList>
            </div>

            {view === 'overview' && (
              <TabsContent value="summary" className="mt-0 space-y-4">
              <div className="grid grid-cols-2 gap-2">
                <StatCard label="Total Params" value={fmtNum(analysis!.totalParams)} sub="trainable parameters" />
                <StatCard label="Peak VRAM" value={analysis!.memoryUsage} sub={analysis!.gpuName || 'GPU estimate'} />
                <StatCard label="Forward FLOPs" value={analysis!.forwardFlopsHuman} sub="per pass" />
                <StatCard label="Backward FLOPs" value={analysis!.backwardFlopsHuman} sub="per pass" />
              </div>

              <div>
                <SectionHeader icon={Gauge} label="Performance" />
                <div className="rounded-lg border border-border/50 bg-secondary/20 px-3 py-2 space-y-0.5">
                  <StatRow label="Latency" value={analysis!.latencyMs != null ? `${fmt(analysis!.latencyMs)} ms` : '—'} />
                  <StatRow label="Throughput" value={analysis!.throughputTokensPerS > 0 ? `${fmtNum(analysis!.throughputTokensPerS)} tok/s` : '—'} />
                  <StatRow label="GPU Utilization" value={analysis!.gpuUtilization != null ? fmtPercent(analysis!.gpuUtilization) : '—'} />
                  <StatRow label="Confidence" value={fmtPercent(analysis!.confidenceScore)} />
                </div>
              </div>

              <div>
                <SectionHeader icon={DollarSign} label="Training Cost" />
                <div className="rounded-lg border border-border/50 bg-secondary/20 px-3 py-2 space-y-0.5">
                  <StatRow label="Estimated Cost" value={analysis!.trainingCostUsd > 0 ? `$${fmt(analysis!.trainingCostUsd, 2)}` : '—'} />
                  <StatRow label="Training Time" value={analysis!.trainingTimeHours > 0 ? `${fmt(analysis!.trainingTimeHours, 2)} hrs` : '—'} />
                  <StatRow label="Energy" value={analysis!.energyKwh > 0 ? `${fmt(analysis!.energyKwh, 2)} kWh` : '—'} />
                  <StatRow label="CO₂" value={analysis!.co2Kg > 0 ? `${fmt(analysis!.co2Kg, 3)} kg` : '—'} />
                </div>
              </div>
              </TabsContent>
            )}

            {view === 'overview' && (
              <TabsContent value="structure" className="mt-0 space-y-4">
              <div>
                <SectionHeader icon={Layers} label="Model Structure" />
                <div className="rounded-lg border border-border/50 bg-secondary/20 px-3 py-2 space-y-0.5">
                  <StatRow label="Architecture" value={analysis!.modelType || '—'} mono={false} />
                  <StatRow label="Layers" value={analysis!.numLayers} />
                  <StatRow label="Graph Depth" value={analysis!.graphDepth} />
                  <StatRow label="Total Operations" value={fmtNum(analysis!.totalOperations)} />
                  <StatRow label="Critical Path" value={analysis!.criticalPathLength} />
                  <StatRow label="Total Tensors" value={analysis!.totalTensorCount} />
                  <StatRow label="Largest Tensor" value={fmtBytes(analysis!.largestTensorBytes)} />
                  <StatRow label="Tensor Resolution" value={fmtPercent(analysis!.tensorResolutionRatio)} />
                  {analysis!.unresolvedDimCount > 0 && <StatRow label="Unresolved Dims" value={analysis!.unresolvedDimCount} />}
                </div>
              </div>

              {perLayerFlopsRows && (
                <div>
                  <SectionHeader icon={Activity} label="FLOPs per Layer" />
                  <div className="space-y-1.5">
                    {perLayerFlopsRows.map((row) => {
                      const percentage = (row.flops / maxFlops) * 100;
                      const isSelected = selectedNodeId === row.id;
                      return (
                        <button
                          key={row.id}
                          className={cn(
                            'w-full text-left p-2 rounded transition-colors',
                            isSelected ? 'bg-primary/20' : 'hover:bg-secondary/50'
                          )}
                          onClick={() => onSelectNode(row.id)}
                        >
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-[10px] font-medium truncate max-w-[60%]">{row.name}</span>
                            <span className="text-[10px] font-mono text-muted-foreground">{row.flopsHuman}</span>
                          </div>
                          <div className="relative h-1.5 bg-secondary rounded-full overflow-hidden">
                            <div className="absolute inset-y-0 left-0 rounded-full transition-all bg-primary" style={{ width: `${percentage}%` }} />
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}
              </TabsContent>
            )}

            {view === 'deep' && (
              <TabsContent value="compute" className="mt-0 space-y-4">
              <div>
                <SectionHeader icon={Zap} label="Compute" />
                <div className="rounded-lg border border-border/50 bg-secondary/20 px-3 py-2 space-y-0.5">
                  <StatRow label="Total FLOPs" value={analysis!.estimatedFlops} />
                  <StatRow label="FLOPs/Token" value={fmtNum(analysis!.flopsPerToken)} />
                  <StatRow label="Arithmetic Intensity" value={`${fmt(analysis!.arithmeticIntensity)} FLOP/byte`} />
                  <div className="flex items-center justify-between py-0.5">
                    <span className="text-[10px] text-muted-foreground">Bottleneck</span>
                    <span className={cn('text-[10px] font-medium font-mono', bottleneckColor)}>
                      {analysis!.bottleneck || '—'}
                    </span>
                  </div>
                  <StatRow label="Roofline Position" value={fmt(analysis!.rooflinePosition, 3)} />
                </div>
              </div>

              {Object.keys(analysis!.opsDistribution).length > 0 && (
                <div>
                  <SectionHeader icon={BarChart3} label="Ops Distribution" />
                  <div className="rounded-lg border border-border/50 bg-secondary/20 px-3 py-2 space-y-0.5">
                    {Object.entries(analysis!.opsDistribution).map(([op, count]) => (
                      <StatRow key={op} label={op} value={count} />
                    ))}
                  </div>
                </div>
              )}
              </TabsContent>
            )}

            {view === 'deep' && (
              <TabsContent value="memory" className="mt-0 space-y-4">
              <div>
                <SectionHeader icon={HardDrive} label="Memory Breakdown" />
                <div className="rounded-lg border border-border/50 bg-secondary/20 px-3 py-2 space-y-2">
                  <MemoryBar label="Parameters" bytes={analysis!.parameterMemoryBytes} totalBytes={totalVram} colorClass="bg-primary" />
                  <MemoryBar label="Activations" bytes={analysis!.activationMemoryBytes} totalBytes={totalVram} colorClass="bg-blue-500" />
                  <MemoryBar label="Gradients" bytes={analysis!.gradientMemoryBytes} totalBytes={totalVram} colorClass="bg-orange-500" />
                  <MemoryBar label="Optimizer States" bytes={analysis!.optimizerStateBytes} totalBytes={totalVram} colorClass="bg-purple-500" />
                  <div className="border-t border-border/40 pt-1 space-y-0.5">
                    <StatRow label="Peak VRAM" value={fmtBytes(analysis!.peakVramBytes)} />
                    <StatRow label="Max Batch Size" value={fmtNum(analysis!.maxBatchSizeFit)} />
                    <StatRow label="Fragmentation" value={fmtPercent(analysis!.memoryFragmentation)} />
                  </div>
                </div>
              </div>
              </TabsContent>
            )}

            {view === 'deep' && (
              <TabsContent value="system" className="mt-0 space-y-4">
              {analysis!.gpuName && (
                <div>
                  <SectionHeader icon={Server} label="Hardware" />
                  <div className="rounded-lg border border-border/50 bg-secondary/20 px-3 py-2 space-y-0.5">
                    <StatRow label="GPU" value={analysis!.gpuName} mono={false} />
                    <StatRow label="GPU Count" value={analysis!.gpuCount} />
                    <StatRow label="VRAM" value={`${fmt(analysis!.gpuMemoryGb, 1)} GB`} />
                    <StatRow label="Peak TFLOPs (FP16)" value={`${fmt(analysis!.gpuTflops, 1)} TFLOPS`} />
                    <StatRow label="HBM Bandwidth" value={`${fmt(analysis!.gpuBandwidthGbs, 0)} GB/s`} />
                  </div>
                </div>
              )}

              <div>
                <SectionHeader icon={Cpu} label="Parallelism" />
                <div className="rounded-lg border border-border/50 bg-secondary/20 px-3 py-2 space-y-0.5">
                  <StatRow label="Optimal GPU Count" value={analysis!.optimalGpuCount} />
                  <StatRow label="Data Parallel Eff." value={fmtPercent(analysis!.dataParallelEfficiency)} />
                  <StatRow label="Communication OH" value={fmtPercent(analysis!.communicationOverhead)} />
                  <StatRow label="Pipeline Stages" value={analysis!.pipelineStages || '—'} />
                  <StatRow label="Tensor Parallel Deg." value={analysis!.tensorParallelDegree || '—'} />
                </div>
              </div>

              {analysis!.dynamic && (
                <div className="space-y-4">
                  {analysis!.dynamic.virtual_memory && (
                    <div>
                      <SectionHeader icon={Binary} label="Virtual Memory Opt." />
                      <div className="rounded-lg border border-border/50 bg-secondary/20 px-3 py-2 space-y-0.5">
                        <StatRow label="Fragmentation" value={fmtPercent(analysis!.dynamic.virtual_memory.fragmentation_pct / 100)} />
                        <StatRow label="Virtual Savings" value={`${fmt(analysis!.dynamic.virtual_memory.virtual_savings_gb, 2, true)} GB`} />
                        <StatRow label="Defrag Savings" value={`${fmt(analysis!.dynamic.virtual_memory.defrag_savings_gb, 2, true)} GB`} />
                        <div className="flex items-center justify-between py-0.5 border-t border-border/30 mt-1 pt-1">
                          <span className="text-[10px] text-muted-foreground uppercase text-[8px] font-bold">Strategy</span>
                          <span className="text-[10px] font-bold text-primary">{analysis!.dynamic.virtual_memory.recommended_strategy}</span>
                        </div>
                      </div>
                    </div>
                  )}

                  {analysis!.dynamic.stability && (
                    <div>
                      <SectionHeader icon={ShieldCheck} label="Numerical Stability" />
                      <div className="rounded-lg border border-border/50 bg-secondary/20 px-3 py-2 space-y-0.5">
                        <StatRow label="Robustness Score" value={fmt(analysis!.dynamic.stability.global_robustness_score, 2)} />
                        <StatRow label="Chaos Index" value={fmt(analysis!.dynamic.stability.chaos_index, 3, true)} />
                        <StatRow label="FP32 Required" value={fmtPercent(analysis!.dynamic.stability.fp32_required_pct / 100)} />
                        <StatRow label="High Risk Layers" value={analysis!.dynamic.stability.high_risk_layers_count} />
                      </div>
                    </div>
                  )}

                  {analysis!.dynamic.behavioral && (
                    <div>
                      <SectionHeader icon={BrainCircuit} label="Behavioral Metrics" />
                      <div className="rounded-lg border border-border/50 bg-secondary/20 px-3 py-2 space-y-0.5">
                        <StatRow label="Expert Balance" value={fmtPercent(analysis!.dynamic.behavioral.load_balance_efficiency / 100)} />
                        <StatRow label="Cache Locality" value={fmt(analysis!.dynamic.behavioral.cache_locality_score, 2)} />
                        <StatRow label="Mem Contention" value={fmt(analysis!.dynamic.behavioral.memory_contention_score, 2, true)} />
                      </div>
                    </div>
                  )}
                </div>
              )}
              </TabsContent>
            )}
          </Tabs>
        )}

        {/* ── Experiment Note ── */}
        {hasData && (
          <div className="flex items-center gap-1.5 text-[9px] text-muted-foreground/50 pb-2">
            <FlaskConical className="w-3 h-3" />
            <span>All values are backend estimates — run analysis to refresh.</span>
          </div>
        )}
      </div>
    </div>
  );
}
