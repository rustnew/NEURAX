import {
  Cpu,
  Database,
  HardDrive,
  Layers,
  Zap,
  Weight,
  CircleDot
} from 'lucide-react';
import { AnalysisResult, CanvasNode, PerLayerBreakdownRow } from '@/types/architecture.ts';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.tsx';

interface ParameterPanelProps {
  nodes: CanvasNode[];
  selectedNodeId: string | null;
  onSelectNode: (id: string) => void;
  analysis: AnalysisResult;
  perLayer?: PerLayerBreakdownRow[];
}

export function ParameterPanel({ analysis, perLayer = [] }: ParameterPanelProps) {
  const fmt = (num: number): string => {
    if (num >= 1e9) return `${(num / 1e9).toFixed(1)}B`;
    if (num >= 1e6) return `${(num / 1e6).toFixed(1)}M`;
    if (num >= 1e3) return `${(num / 1e3).toFixed(1)}K`;
    return num.toString();
  };

  const fmtBytes = (b: number): string => {
    if (b >= 1e9) return `${(b / 1e9).toFixed(1)} GB`;
    if (b >= 1e6) return `${(b / 1e6).toFixed(0)} MB`;
    if (b >= 1e3) return `${(b / 1e3).toFixed(0)} KB`;
    return `${b} B`;
  };

  return (
    <div className="h-full flex flex-col bg-card border-l border-border overflow-hidden">
      {/* Header */}
      <div className="h-10 px-4 flex items-center border-b border-border bg-panel-header shrink-0">
        <Database className="w-4 h-4 text-primary mr-2" />
        <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Parameters
        </span>
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-thin">
        <Tabs defaultValue="overview" className="h-full flex flex-col">
          <div className="px-3 py-2 border-b border-border bg-card/50">
            <TabsList className="grid h-8 w-full grid-cols-3 bg-secondary/40">
              <TabsTrigger value="overview" className="text-[11px] uppercase tracking-wide">
                Overview
              </TabsTrigger>
              <TabsTrigger value="memory" className="text-[11px] uppercase tracking-wide">
                Memory
              </TabsTrigger>
              <TabsTrigger value="layers" className="text-[11px] uppercase tracking-wide">
                Layers
              </TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="overview" className="mt-0 flex-1 overflow-y-auto scrollbar-thin">
            <div className="p-3 grid grid-cols-2 gap-2 border-b border-border bg-card/50">
              <div className="metric-card">
                <div className="flex items-center gap-1.5 mb-1">
                  <Layers className="w-3.5 h-3.5 text-primary" />
                  <span className="text-[10px] text-muted-foreground uppercase tracking-wide">Parameters</span>
                </div>
                <div className="text-lg font-semibold font-mono text-foreground">
                  {analysis.totalParams > 0 ? fmt(analysis.totalParams) : '0'}
                </div>
                <div className="text-[10px] text-muted-foreground/70">Total trainable</div>
              </div>

              <div className="metric-card">
                <div className="flex items-center gap-1.5 mb-1">
                  <HardDrive className="w-3.5 h-3.5 text-primary" />
                  <span className="text-[10px] text-muted-foreground uppercase tracking-wide">Memory</span>
                </div>
                <div className="text-lg font-semibold font-mono text-foreground">
                  {analysis.peakVramBytes > 0 ? fmtBytes(analysis.peakVramBytes) : '0 B'}
                </div>
                <div className="text-[10px] text-muted-foreground/70">Peak VRAM</div>
              </div>

              <div className="metric-card">
                <div className="flex items-center gap-1.5 mb-1">
                  <Zap className="w-3.5 h-3.5 text-primary" />
                  <span className="text-[10px] text-muted-foreground uppercase tracking-wide">FLOPs</span>
                </div>
                <div className="text-lg font-semibold font-mono text-foreground">
                  {analysis.estimatedFlops || '0 FLOPs'}
                </div>
                <div className="text-[10px] text-muted-foreground/70">Per forward pass</div>
              </div>

              <div className="metric-card">
                <div className="flex items-center gap-1.5 mb-1">
                  <Cpu className="w-3.5 h-3.5 text-primary" />
                  <span className="text-[10px] text-muted-foreground uppercase tracking-wide">Depth</span>
                </div>
                <div className="text-lg font-semibold font-mono text-foreground">
                  {analysis.graphDepth || analysis.numLayers || 1}
                </div>
                <div className="text-[10px] text-muted-foreground/70">Layer depth</div>
              </div>
            </div>

            <div className="px-4 py-4 space-y-3">
              <div className="text-xs font-semibold text-foreground">Global Statistics</div>
              <div className="grid grid-cols-2 gap-2">
                <div className="metric-card">
                  <div className="flex items-center gap-1.5 mb-1">
                    <Layers className="w-3 h-3 text-primary" />
                    <span className="text-[10px] text-muted-foreground uppercase">Total Params</span>
                  </div>
                  <div className="text-sm font-semibold font-mono">{fmt(analysis.totalParams)}</div>
                </div>
                <div className="metric-card">
                  <div className="flex items-center gap-1.5 mb-1">
                    <Weight className="w-3 h-3 text-primary" />
                    <span className="text-[10px] text-muted-foreground uppercase">Param Mem</span>
                  </div>
                  <div className="text-sm font-semibold font-mono">
                    {analysis.parameterMemoryBytes > 0 ? fmtBytes(analysis.parameterMemoryBytes) : '—'}
                  </div>
                </div>
                <div className="metric-card">
                  <div className="flex items-center gap-1.5 mb-1">
                    <CircleDot className="w-3 h-3 text-primary" />
                    <span className="text-[10px] text-muted-foreground uppercase">Layers</span>
                  </div>
                  <div className="text-sm font-semibold font-mono">
                    {analysis.numLayers > 0 ? analysis.numLayers : '—'}
                  </div>
                </div>
                <div className="metric-card">
                  <div className="flex items-center gap-1.5 mb-1">
                    <HardDrive className="w-3 h-3 text-primary" />
                    <span className="text-[10px] text-muted-foreground uppercase">Model Size</span>
                  </div>
                  <div className="text-sm font-semibold font-mono">
                    {analysis.parameterMemoryBytes > 0 ? fmtBytes(analysis.parameterMemoryBytes) : '—'}
                  </div>
                </div>
              </div>

              {analysis.bottleneck && (
                <div className="flex items-center gap-2 text-xs">
                  <span className="text-muted-foreground">Bottleneck:</span>
                  <span className={`px-2 py-0.5 rounded font-medium ${analysis.bottleneck === 'memory-bound' ? 'bg-amber-500/20 text-amber-400' :
                    analysis.bottleneck === 'compute-bound' ? 'bg-blue-500/20 text-blue-400' :
                      'bg-green-500/20 text-green-400'
                    }`}>{analysis.bottleneck}</span>
                </div>
              )}
            </div>
          </TabsContent>

          <TabsContent value="memory" className="mt-0 flex-1 overflow-y-auto scrollbar-thin">
            <div className="p-4 space-y-3">
              <div className="grid grid-cols-2 gap-2">
                <div className="metric-card">
                  <div className="flex items-center gap-1.5 mb-1">
                    <HardDrive className="w-3 h-3 text-primary" />
                    <span className="text-[10px] text-muted-foreground uppercase">Peak VRAM</span>
                  </div>
                  <div className="text-sm font-semibold font-mono">
                    {analysis.peakVramBytes > 0 ? fmtBytes(analysis.peakVramBytes) : '—'}
                  </div>
                </div>
                <div className="metric-card">
                  <div className="flex items-center gap-1.5 mb-1">
                    <Weight className="w-3 h-3 text-primary" />
                    <span className="text-[10px] text-muted-foreground uppercase">Param Mem</span>
                  </div>
                  <div className="text-sm font-semibold font-mono">
                    {analysis.parameterMemoryBytes > 0 ? fmtBytes(analysis.parameterMemoryBytes) : '—'}
                  </div>
                </div>
              </div>

              <div className="p-3 bg-secondary/30 rounded-lg space-y-2">
                <div className="text-[10px] font-semibold uppercase text-muted-foreground mb-2">
                  Memory Breakdown
                </div>
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">Activations</span>
                    <span className="font-mono">{analysis.activationMemoryBytes > 0 ? fmtBytes(analysis.activationMemoryBytes) : '—'}</span>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">Gradients</span>
                    <span className="font-mono">{analysis.gradientMemoryBytes > 0 ? fmtBytes(analysis.gradientMemoryBytes) : '—'}</span>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">Optimizer</span>
                    <span className="font-mono">{analysis.optimizerStateBytes > 0 ? fmtBytes(analysis.optimizerStateBytes) : '—'}</span>
                  </div>
                  <div className="pt-1 border-t border-border/30 flex items-center justify-between text-xs font-semibold">
                    <span className="text-foreground">Peak VRAM</span>
                    <span className="font-mono text-primary">{analysis.peakVramBytes > 0 ? fmtBytes(analysis.peakVramBytes) : '—'}</span>
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="layers" className="mt-0 flex-1 overflow-y-auto scrollbar-thin">
            <div className="px-2 py-3 space-y-0.5">
              {perLayer.length > 0 ? (
                <div className="flex flex-col">
                  {/* Header */}
                  <div className="flex items-center justify-between text-[10px] uppercase tracking-wider text-muted-foreground/60 px-2 py-1 mb-1 border-b border-border/10">
                    <span className="w-24">Layer ID</span>
                    <div className="flex gap-4 items-center">
                      <span className="w-12 text-right">Params</span>
                      <span className="w-12 text-right">FLOPS</span>
                      <span className="w-14 text-right">Latency</span>
                      <span className="w-14 text-right">VRAM</span>
                    </div>
                  </div>
                  {perLayer.map(row => (
                    <div
                      key={row.id}
                      className="flex items-center justify-between text-[11px] py-1.5 px-2 hover:bg-secondary/30 rounded transition-colors group"
                    >
                      <span className="text-foreground/80 truncate w-24 font-medium" title={row.name}>
                        {row.name}
                      </span>
                      <div className="flex gap-4 shrink-0 text-right items-center">
                        <span className="w-12 font-mono text-muted-foreground/90 tabular-nums">
                          {row.params && row.params > 0 ? fmt(row.params) : '—'}
                        </span>
                        <span className="w-12 font-mono text-muted-foreground/90 tabular-nums">{row.flops ?? '—'}</span>
                        <span className="w-14 font-mono text-accent/80 tabular-nums">{row.latency ?? '—'}</span>
                        <span className="w-14 font-mono text-primary/70 tabular-nums">{row.memory && row.memory !== '—' ? row.memory : '—'}</span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-xs text-muted-foreground/70 text-center py-4">
                  Run analysis to see per-layer breakdown.
                </div>
              )}
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
