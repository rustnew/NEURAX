import React, { useMemo, useRef, useEffect } from 'react';
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
  AlertTriangle,
  Lightbulb,
  Info,
  CheckCircle2,
  XCircle,
  Database,
  AlertCircle,
  BookOpen,
  ArrowRight,
} from 'lucide-react';
import { AnalysisResult, CanvasNode, PerLayerBreakdownRow, Warning } from '@/types/architecture.ts';
import { cn } from '@/lib/utils.ts';
import {
  normalizeDiagnostics,
  normalizeRecommendations,
  countDiagnostics,
  summariseDiagnostics,
  severityPresentation,
} from '@/utils/diagnostics.ts';

/**
 * How many diagnostics the panel lists before deferring to the report.
 *
 * They are sorted with the blocking ones first, so a truncated list never hides
 * something that stops the model running — and the count of what was left out
 * is always shown.
 */
const DIAGNOSTIC_LIMIT = 15;


interface MetricsDashboardProps {
  nodes: CanvasNode[];
  selectedNodeId: string | null;
  onSelectNode: (id: string) => void;
  analysis?: AnalysisResult;
  perLayer?: PerLayerBreakdownRow[];
  warnings?: Warning[];
  category: 'architecture' | 'performance' | 'hardware';
  jumpToIssuesSignal?: number;
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

function SectionIcon({ icon: Icon, label, count }: { icon: React.ElementType; label: string; count?: number }) {
  return (
    <div className="flex items-center gap-2 mb-2">
      <Icon className="w-3.5 h-3.5 text-primary" />
      <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        {label}{count !== undefined ? ` (${count})` : ''}
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

function StatCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="metric-card">
      <div className="text-[9px] text-muted-foreground uppercase tracking-wide mb-1">{label}</div>
      <div className="text-base font-bold font-mono text-foreground leading-tight">{value ?? '—'}</div>
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

function IssuesBlock({ warnings, jumpToIssuesSignal }: { warnings: Warning[]; jumpToIssuesSignal?: number }) {
  const issuesRef = useRef<HTMLDivElement | null>(null);
  const firstCompilerWarningRef = useRef<HTMLDivElement | null>(null);
  const compilerWarningPattern = /custom operations|estimated flops|unsupported/i;

  const errorCount = warnings.filter(w => w.type === 'error').length;
  const warningCount = warnings.filter(w => w.type === 'warning').length;
  const infoCount = warnings.filter(w => w.type === 'info').length;

  useEffect(() => {
    if (jumpToIssuesSignal && jumpToIssuesSignal > 0) {
      const target = firstCompilerWarningRef.current ?? issuesRef.current;
      target?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [jumpToIssuesSignal]);

  useEffect(() => {
    firstCompilerWarningRef.current = null;
  }, [warnings]);

  return (
    <div ref={issuesRef}>
      <SectionIcon icon={AlertCircle} label="Issues & Diagnostics" />
      {errorCount === 0 && warningCount === 0 ? (
        <div className="rounded-lg border border-border/50 bg-secondary/20 px-3 py-4 mb-4">
          <div className="flex items-center gap-2 text-success">
            <CheckCircle2 className="w-4 h-4" />
            <span className="text-xs font-medium">No issues detected</span>
          </div>
        </div>
      ) : (
        <div className="space-y-1.5 mb-4">
          <div className="flex items-center gap-2 pb-1">
            <div className={cn(
              "flex items-center gap-1 px-2 py-0.5 rounded text-[10px]",
              errorCount > 0 ? "bg-destructive/10 text-destructive" : "bg-success/10 text-success"
            )}>
              {errorCount > 0 ? <AlertCircle className="w-3 h-3" /> : <CheckCircle2 className="w-3 h-3" />}
              <span>{errorCount} errors</span>
            </div>
            <div className={cn(
              "flex items-center gap-1 px-2 py-0.5 rounded text-[10px]",
              warningCount > 0 ? "bg-warning/10 text-warning" : "bg-success/10 text-success"
            )}>
              <AlertTriangle className="w-3 h-3" />
              <span>{warningCount} warnings</span>
            </div>
            {infoCount > 0 && (
              <span className="text-[10px] text-muted-foreground">{infoCount} info</span>
            )}
          </div>
          {warnings.map((warning) => (
            <div
              key={warning.id}
              ref={(el) => {
                if (!el) return;
                if (!compilerWarningPattern.test(warning.message)) return;
                if (!firstCompilerWarningRef.current) {
                  firstCompilerWarningRef.current = el;
                }
              }}
              /**
               * The body text is `text-foreground`, not the severity's
               * `-foreground` token.
               *
               * `--destructive-foreground` and `--warning-foreground` are the
               * colours meant to sit *on top of* a solid red or amber — white
               * on a filled button. The background here is `bg-destructive/5`,
               * a 5 % tint that is, for contrast purposes, the panel itself.
               * On the dark theme those tokens happen to be near-white and the
               * text read anyway; on the light theme they are pure white
               * (`--destructive-foreground: 0 0% 100%`), so every error and
               * warning rendered white-on-white and the panel looked empty.
               *
               * Severity is carried by the icon and the border, which are the
               * right places for it — the message is body text and takes the
               * body colour, in every theme.
               */
              className={cn(
                "p-2 rounded-md border text-[10px] text-foreground",
                warning.type === 'error' && "bg-destructive/5 border-destructive/30",
                warning.type === 'warning' && "bg-warning/5 border-warning/30",
                warning.type === 'info' && "bg-primary/5 border-primary/20"
              )}
            >
              <div className="flex items-start gap-1.5">
                {warning.type === 'error' ? (
                  <XCircle className="w-3 h-3 mt-0.5 shrink-0 text-destructive" />
                ) : warning.type === 'warning' ? (
                  <AlertTriangle className="w-3 h-3 mt-0.5 shrink-0 text-warning" />
                ) : (
                  <CheckCircle2 className="w-3 h-3 mt-0.5 shrink-0 text-primary" />
                )}
                <div className="min-w-0">
                  {warning.code ? (
                    <span className="font-mono text-[9px] opacity-70 mr-1">[{warning.code}]</span>
                  ) : null}
                  <span>{warning.message}</span>
                  {warning.hint ? (
                    <div className="mt-0.5 text-[9px] opacity-75">{warning.hint}</div>
                  ) : null}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────

export function MetricsDashboard({
  nodes: _nodes,
  selectedNodeId,
  onSelectNode,
  analysis,
  perLayer,
  warnings = [],
  category,
  jumpToIssuesSignal = 0,
}: MetricsDashboardProps) {
  const hasData = !!analysis && (analysis.totalParams > 0 || analysis.totalFlops > 0 || analysis.peakVramBytes > 0);

  const totalVram = analysis?.peakVramBytes ?? 0;

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

  const bottleneckColor = analysis?.bottleneck === 'compute-bound'
    ? 'text-orange-400'
    : analysis?.bottleneck === 'memory-bound'
      ? 'text-blue-400'
      : 'text-muted-foreground';

  const categoryTitle = category === 'architecture' ? 'Architecture & Diagnostics'
    : category === 'performance' ? 'Performance & Memory'
    : 'Hardware & Cost';

  const categoryIcon = category === 'architecture' ? Layers
    : category === 'performance' ? Zap
    : Server;

  const CategoryIcon = categoryIcon;

  // ─── Render ────────────────────────────────────────────────────

  return (
    <div className="h-full flex flex-col bg-card overflow-hidden">
      {/* Header */}
      <div className="h-10 px-4 flex items-center border-b border-border bg-panel-header shrink-0">
        <CategoryIcon className="w-4 h-4 text-primary mr-2" />
        <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          {categoryTitle}
        </span>
        {hasData && analysis && analysis.confidenceScore > 0 && (
          <span className={cn('ml-auto text-[10px] font-mono px-1.5 py-0.5 rounded',
            analysis.confidenceScore >= 0.9 ? 'bg-success/10 text-success' : 'bg-warning/10 text-warning'
          )}>
            {fmtPercent(analysis.confidenceScore)} conf
          </span>
        )}
      </div>

      {/* Scrollable Content */}
      <div className="flex-1 overflow-y-auto scrollbar-thin p-3 space-y-4">
        {!hasData && (
          <div className="p-4 rounded-lg border border-border bg-secondary/20">
            <div className="text-xs text-muted-foreground">No metrics report available yet.</div>
            <div className="text-[10px] text-muted-foreground/80 mt-1">
              Run analysis to populate all architecture, performance, and cost metrics.
            </div>
          </div>
        )}

        {hasData && category === 'architecture' && (
          <ArchitectureCategory
            analysis={analysis!}
            perLayerFlopsRows={perLayerFlopsRows}
            maxFlops={maxFlops}
            selectedNodeId={selectedNodeId}
            onSelectNode={onSelectNode}
            warnings={warnings}
            jumpToIssuesSignal={jumpToIssuesSignal}
          />
        )}

        {hasData && category === 'performance' && (
          <PerformanceCategory
            analysis={analysis!}
            totalVram={totalVram}
            perLayerFlopsRows={perLayerFlopsRows}
            maxFlops={maxFlops}
            selectedNodeId={selectedNodeId}
            onSelectNode={onSelectNode}
            bottleneckColor={bottleneckColor}
          />
        )}

        {hasData && category === 'hardware' && (
          <HardwareCategory analysis={analysis!} />
        )}

        {/* Footer note */}
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

// ════════════════════════════════════════════════════════════════════
//  CATEGORY 1: Architecture & Diagnostics
// ════════════════════════════════════════════════════════════════════

function ArchitectureCategory({
  analysis,
  perLayerFlopsRows,
  maxFlops,
  selectedNodeId,
  onSelectNode,
  warnings,
  jumpToIssuesSignal,
}: {
  analysis: AnalysisResult;
  perLayerFlopsRows?: { id: string; name: string; flopsHuman: string; flops: number }[];
  maxFlops: number;
  selectedNodeId: string | null;
  onSelectNode: (id: string) => void;
  warnings: Warning[];
  jumpToIssuesSignal: number;
}) {
  // Normalised once here rather than at each use: the compiler's severities are
  // PascalCase (`Critical`, `Hint`) and its categories are too, and reading
  // them raw is what once turned every optimisation hint into a red error row.
  const normalizedDiagnostics = useMemo(
    () => normalizeDiagnostics(analysis?.diagnostics),
    [analysis?.diagnostics],
  );
  const diagnosticCounts = useMemo(
    () => countDiagnostics(normalizedDiagnostics),
    [normalizedDiagnostics],
  );
  const normalizedRecommendations = useMemo(
    () => normalizeRecommendations(analysis?.recommendations),
    [analysis?.recommendations],
  );

  return (
    <div className="space-y-4">
      {/* ════ Model Structure ════ */}
      <div>
        <SectionIcon icon={BookOpen} label="Model Structure" />
        <div className="grid grid-cols-2 gap-2 mb-3">
          {(analysis.modelName && analysis.modelName !== analysis.modelType) && (
            <StatCard label="Model Name" value={analysis.modelName} sub={analysis.modelType || '—'} />
          )}
          <StatCard label="Total Params" value={fmtNum(analysis.totalParams)} sub="trainable parameters" />
          {analysis.activeParams > 0 && analysis.activeParams < analysis.totalParams && (
            <StatCard
              label="Active Params"
              value={fmtNum(analysis.activeParams)}
              sub={`per token · ${Math.round((analysis.activeParams / analysis.totalParams) * 100)}% of total`}
            />
          )}
          <StatCard label="Layers" value={analysis.numLayers} sub={`depth ${analysis.graphDepth}`} />
        </div>
        <div className="rounded-lg border border-border/50 bg-secondary/20 px-3 py-2 space-y-0.5">
          <StatRow label="Architecture" value={analysis.modelType || '—'} mono={false} />
          {analysis.modelName && analysis.modelName !== analysis.modelType && (
            <StatRow label="Model Name" value={analysis.modelName} mono={false} />
          )}
          <StatRow label="Graph Depth" value={analysis.graphDepth} />
          <StatRow label="Total Operations" value={fmtNum(analysis.totalOperations)} />
          <StatRow label="Critical Path Length" value={analysis.criticalPathLength} />
          {(analysis.sequenceLength ?? 0) > 0 && <StatRow label="Sequence Length" value={analysis.sequenceLength!} />}
          {(analysis.hiddenSize ?? 0) > 0 && <StatRow label="Hidden Size" value={analysis.hiddenSize!} />}
          {(analysis.vocabSize ?? 0) > 0 && <StatRow label="Vocab Size" value={analysis.vocabSize!} />}
          {(analysis.numAttentionHeads ?? 0) > 0 && <StatRow label="Attention Heads" value={analysis.numAttentionHeads!} />}
          {(analysis.numKeyValueHeads ?? 0) > 0 && <StatRow label="KV Heads" value={analysis.numKeyValueHeads!} />}
          {(analysis.intermediateSize ?? 0) > 0 && <StatRow label="Intermediate Size" value={analysis.intermediateSize!} />}
        </div>
      </div>

      {/* ════ Layers by Type ════ */}
      {analysis.layersByType && Object.keys(analysis.layersByType).length > 0 && (
        <div>
          <SectionIcon icon={Layers} label="Layers by Type" />
          <div className="rounded-lg border border-border/50 bg-secondary/20 px-3 py-2 space-y-0.5">
            {Object.entries(analysis.layersByType).map(([t, cnt]) => (
              <StatRow key={t} label={t} value={cnt} mono={false} />
            ))}
          </div>
        </div>
      )}

      {/* ════ Tensors ════ */}
      <div>
        <SectionIcon icon={Database} label="Tensors" />
        <div className="rounded-lg border border-border/50 bg-secondary/20 px-3 py-2 space-y-0.5">
          <StatRow label="Total Tensors" value={analysis.totalTensorCount} />
          <StatRow label="Largest Tensor" value={fmtBytes(analysis.largestTensorBytes)} />
          <StatRow label="Tensor Resolution" value={fmtPercent(analysis.tensorResolutionRatio)} />
          {analysis.unresolvedDimCount > 0 && (
            <StatRow label="Unresolved Dims" value={analysis.unresolvedDimCount} />
          )}
        </div>
      </div>

      {/* ════ FLOPs per Layer ════ */}
      {perLayerFlopsRows && (
        <div>
          <SectionIcon icon={Activity} label="FLOPs per Layer" />
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

      {/* ════ Per-Layer Details (params, memory, latency) ════ */}
      {perLayerFlopsRows && (
        <div>
          <SectionIcon icon={Info} label="Per-Layer Metrics" />
          <div className="rounded-lg border border-border/50 bg-secondary/20 px-3 py-2 space-y-0.5">
            {perLayerFlopsRows.map((row) => {
              const isSelected = selectedNodeId === row.id;
              return (
                <button
                  key={row.id}
                  onClick={() => onSelectNode(row.id)}
                  className={cn(
                    'w-full text-left px-2 py-1 rounded text-[10px] transition-colors',
                    isSelected ? 'bg-primary/20' : 'hover:bg-secondary/50'
                  )}
                >
                  <span className="font-medium">{row.name}</span>
                  <span className="text-muted-foreground ml-2">{row.flopsHuman}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* ════ Issues, Warnings & Diagnostics (merged from old Issues tab) ════ */}
      <IssuesBlock warnings={warnings} jumpToIssuesSignal={jumpToIssuesSignal} />

      {/* ════ Diagnostics from the compiler ════ */}
      {normalizedDiagnostics.length > 0 && (
        <div>
          <SectionIcon
            icon={AlertTriangle}
            label={`Compiler Diagnostics (${normalizedDiagnostics.length})`}
          />
          {/* What was found, most consequential first. */}
          <p className="text-[10px] text-muted-foreground mb-1.5 px-0.5">
            {summariseDiagnostics(diagnosticCounts)}
          </p>
          <div className="space-y-1.5">
            {normalizedDiagnostics.slice(0, DIAGNOSTIC_LIMIT).map((d, i) => {
              const look = severityPresentation(d.severity);
              // Red stops you, amber warns you, green is a way to improve the
              // model. The icon carries the same distinction as the colour, so
              // the row still reads without it.
              const Icon =
                d.severity === 'critical'
                  ? XCircle
                  : d.severity === 'warning'
                    ? AlertTriangle
                    : d.severity === 'hint'
                      ? Lightbulb
                      : Info;

              return (
                <div key={`${d.code ?? 'diag'}-${i}`} className={`rounded-lg border px-2.5 py-2 ${look.colors.container}`}>
                  <div className="flex items-start gap-1.5">
                    <Icon className={`w-3 h-3 mt-0.5 shrink-0 ${look.colors.accent}`} />
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-1.5 flex-wrap mb-0.5">
                        <span className={`text-[8px] uppercase font-bold tracking-wide ${look.colors.accent}`}>
                          {look.label}
                        </span>
                        <span className="text-[8px] uppercase tracking-wide text-muted-foreground/70">
                          {d.categoryLabel}
                        </span>
                        {d.code && (
                          <span className="text-[8px] font-mono text-muted-foreground/50">{d.code}</span>
                        )}
                      </div>
                      <div className="text-[10px] font-medium leading-snug">{d.message}</div>
                      {/* The suggestion is the reason this row exists: it is
                          what the reader can actually do. */}
                      {d.suggestion && (
                        <div className="text-[9px] text-muted-foreground mt-1 flex items-start gap-1">
                          <ArrowRight className="w-2.5 h-2.5 mt-[0.15rem] shrink-0" />
                          <span>{d.suggestion}</span>
                        </div>
                      )}
                      {d.layerId && (
                        <div className="text-[9px] text-muted-foreground/60 mt-0.5">Block: {d.layerId}</div>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
          {/* Never truncate silently — a hidden blocking issue is the one that
              costs a training run. */}
          {normalizedDiagnostics.length > DIAGNOSTIC_LIMIT && (
            <p className="text-[9px] text-muted-foreground/70 mt-1.5 px-0.5">
              {normalizedDiagnostics.length - DIAGNOSTIC_LIMIT} more, in the exported report.
            </p>
          )}
        </div>
      )}

      {/* ════ Recommendations ════ */}
      {normalizedRecommendations.length > 0 && (
        <div>
          <SectionIcon icon={Lightbulb} label={`Recommendations (${normalizedRecommendations.length})`} />
          <div className="space-y-1.5">
            {normalizedRecommendations.slice(0, 8).map((rec, i) => {
              // A recommendation is always an improvement, so it is always
              // green; priority varies its weight, not its meaning.
              const prioCls =
                rec.priority === 'high'
                  ? 'text-emerald-600 dark:text-emerald-400'
                  : rec.priority === 'medium'
                    ? 'text-emerald-600/70 dark:text-emerald-400/70'
                    : 'text-muted-foreground';
              return (
                <div
                  key={`${rec.title}-${i}`}
                  className="rounded-lg border border-emerald-500/25 bg-emerald-500/[0.07] px-2.5 py-2"
                >
                  <div className="flex items-center justify-between gap-2 mb-1">
                    <span className="text-[10px] font-semibold">{rec.title}</span>
                    <span className={`text-[8px] uppercase font-bold shrink-0 ${prioCls}`}>
                      {rec.priority}
                    </span>
                  </div>
                  <div className="text-[9px] text-muted-foreground leading-snug">{rec.description}</div>
                  {/* The quantified gain — the reason to act on this at all. */}
                  {rec.impact && (
                    <div className="text-[9px] font-medium text-emerald-600 dark:text-emerald-400 mt-1">
                      {rec.impact}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ════ Compiler Warnings ════ */}
      {(analysis.reportWarnings?.length ?? 0) > 0 && (
        <div>
          <SectionIcon icon={Info} label="Compiler Warnings" />
          <div className="rounded-lg border border-border/50 bg-secondary/20 px-3 py-2 space-y-1">
            {analysis.reportWarnings!.map((w, i) => (
              <div key={i} className="text-[10px] text-warning flex items-start gap-1.5">
                <AlertTriangle className="w-3 h-3 shrink-0 mt-0.5" />
                <span>{w}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ════ Analysis metadata ════ */}
      {(analysis.analysisTimeMs ?? 0) > 0 && (
        <div className="flex items-center gap-2 text-[9px] text-muted-foreground/70 px-1 pb-2">
          <Info className="w-2.5 h-2.5" />
          <span>Analysis completed in <span className="font-mono font-medium text-muted-foreground/90">{(analysis.analysisTimeMs! / 1000).toFixed(2)}s</span>
          {analysis.generatedAt && <> &middot; {new Date(analysis.generatedAt!).toLocaleString()}</>}
          </span>
        </div>
      )}
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════
//  CATEGORY 2: Performance & Memory
// ════════════════════════════════════════════════════════════════════

function PerformanceCategory({
  analysis,
  totalVram,
  perLayerFlopsRows,
  maxFlops,
  selectedNodeId,
  onSelectNode,
  bottleneckColor,
}: {
  analysis: AnalysisResult;
  totalVram: number;
  perLayerFlopsRows?: { id: string; name: string; flopsHuman: string; flops: number }[];
  maxFlops: number;
  selectedNodeId: string | null;
  onSelectNode: (id: string) => void;
  bottleneckColor: string;
}) {
  return (
    <div className="space-y-4">
      {/* ════ Summary Cards ════ */}
      <div className="grid grid-cols-2 gap-2">
        <StatCard label="Forward FLOPs" value={analysis.forwardFlopsHuman} sub="per pass" />
        <StatCard label="Backward FLOPs" value={analysis.backwardFlopsHuman} sub="per pass" />
        <StatCard label="Peak VRAM" value={analysis.memoryUsage} sub={analysis.gpuName || 'GPU estimate'} />
        <StatCard label="Total Params" value={fmtNum(analysis.totalParams)} sub="trainable parameters" />
        {analysis.activeParams > 0 && analysis.activeParams < analysis.totalParams && (
          <StatCard
            label="Active Params"
            value={fmtNum(analysis.activeParams)}
            sub={`per token · ${Math.round((analysis.activeParams / analysis.totalParams) * 100)}% of total`}
          />
        )}
      </div>

      {/* ════ FLOPs ════ */}
      <div>
        <SectionIcon icon={Zap} label="FLOPs" />
        <div className="rounded-lg border border-border/50 bg-secondary/20 px-3 py-2 space-y-0.5">
          <StatRow label="Total FLOPs" value={analysis.estimatedFlops} />
          <StatRow label="Forward FLOPs" value={analysis.forwardFlopsHuman} />
          <StatRow label="Backward FLOPs" value={analysis.backwardFlopsHuman} />
          {(analysis.totalStepFlops ?? 0) > 0 && <StatRow label="Step FLOPs" value={fmtNum(analysis.totalStepFlops!)} />}
          {(analysis.flopsPerBatch ?? 0) > 0 && <StatRow label="FLOPs/Batch" value={fmtNum(analysis.flopsPerBatch!)} />}
          <StatRow label="FLOPs/Token" value={fmtNum(analysis.flopsPerToken)} />
          <StatRow label="FLOPs/Token (decode)" value={fmtNum(analysis.flopsIncrementalDecode)} />
          {(analysis.macs ?? 0) > 0 && <StatRow label="MACs" value={fmtNum(analysis.macs!)} />}
        </div>
      </div>

      {/* ════ Compute Efficiency ════ */}
      <div>
        <SectionIcon icon={Gauge} label="Compute Efficiency" />
        <div className="rounded-lg border border-border/50 bg-secondary/20 px-3 py-2 space-y-0.5">
          <StatRow label="Arithmetic Intensity" value={`${fmt(analysis.arithmeticIntensity)} FLOP/B`} />
          {(analysis.bytesAccessed ?? 0) > 0 && <StatRow label="Bytes Accessed" value={fmtBytes(analysis.bytesAccessed!)} />}
          <div className="flex items-center justify-between py-0.5">
            <span className="text-[10px] text-muted-foreground">Bottleneck</span>
            <span className={cn('text-[10px] font-medium font-mono', bottleneckColor)}>
              {analysis.bottleneck || '—'}
            </span>
          </div>
          <StatRow label="Roofline Position" value={`${fmt(analysis.rooflinePosition * 100, 1)}%`} />
          {(analysis.effectiveTflops ?? 0) > 0 && <StatRow label="Effective TFLOPs" value={`${fmt(analysis.effectiveTflops, 2)} TFLOPs`} />}
          {(analysis.samplesPerS ?? 0) > 0 && <StatRow label="Samples/s" value={fmtNum(analysis.samplesPerS!)} />}
        </div>
      </div>

      {/* ════ Ops Distribution ════ */}
      {Object.keys(analysis.opsDistribution).length > 0 && (
        <div>
          <SectionIcon icon={BarChart3} label="Ops Distribution" />
          <div className="rounded-lg border border-border/50 bg-secondary/20 px-3 py-2 space-y-0.5">
            {Object.entries(analysis.opsDistribution).map(([op, count]) => (
              <StatRow key={op} label={op} value={count} />
            ))}
          </div>
        </div>
      )}

      {/* ════ Memory Breakdown (Bar Chart) ════ */}
      <div>
        <SectionIcon icon={HardDrive} label="Memory Breakdown" />
        <div className="rounded-lg border border-border/50 bg-secondary/20 px-3 py-2 space-y-2">
          <MemoryBar label="Parameters" bytes={analysis.parameterMemoryBytes} totalBytes={totalVram} colorClass="bg-primary" />
          <MemoryBar label="Activations" bytes={analysis.activationMemoryBytes} totalBytes={totalVram} colorClass="bg-blue-500" />
          <MemoryBar label="Gradients" bytes={analysis.gradientMemoryBytes} totalBytes={totalVram} colorClass="bg-orange-500" />
          <MemoryBar label="Optimizer States" bytes={analysis.optimizerStateBytes} totalBytes={totalVram} colorClass="bg-purple-500" />
          <div className="border-t border-border/40 pt-1 space-y-0.5">
            <StatRow label="Peak VRAM" value={fmtBytes(analysis.peakVramBytes)} />
            <StatRow label="Max Batch Size Fit" value={fmtNum(analysis.maxBatchSizeFit)} />
            <StatRow label="Fragmentation" value={fmtPercent(analysis.memoryFragmentation)} />
            {(analysis.memoryFragmentationPct ?? 0) > 0 && <StatRow label="Frag. PCT" value={`${fmt(analysis.memoryFragmentationPct, 1)}%`} />}
            {analysis.oomRisk && <StatRow label="OOM Risk" value={analysis.oomRisk.toUpperCase()} />}
          </div>
        </div>
      </div>

      {/* ════ Virtual Memory Optimization ════ */}
      {analysis.dynamic?.virtual_memory && (
        <div>
          <SectionIcon icon={Binary} label="Virtual Memory Optimization" />
          <div className="rounded-lg border border-border/50 bg-secondary/20 px-3 py-2 space-y-0.5">
            <StatRow label="Frag. Overhead" value={`${fmt(analysis.dynamic.virtual_memory.fragmentation_overhead_gb, 2, true)} GB`} />
            <StatRow label="Frag. PCT" value={`${fmt(analysis.dynamic.virtual_memory.fragmentation_pct, 1)}%`} />
            <StatRow label="Defrag Savings" value={`${fmt(analysis.dynamic.virtual_memory.defrag_savings_gb, 2, true)} GB`} />
            <StatRow label="Virtual Savings" value={`${fmt(analysis.dynamic.virtual_memory.virtual_savings_gb, 2, true)} GB (${fmt(analysis.dynamic.virtual_memory.virtual_savings_pct, 1)}%)`} />
            <StatRow label="VRAM w/ Defrag" value={`${fmt(analysis.dynamic.virtual_memory.peak_vram_with_defrag_gb, 2)} GB`} />
            <StatRow label="VRAM w/ Virtual" value={`${fmt(analysis.dynamic.virtual_memory.peak_vram_with_virtual_gb, 2)} GB`} />
            <div className="flex items-center justify-between py-0.5 border-t border-border/30 mt-1 pt-1">
              <span className="text-[10px] text-muted-foreground uppercase text-[8px] font-bold">Strategy</span>
              <span className="text-[10px] font-bold text-primary">{analysis.dynamic.virtual_memory.recommended_strategy}</span>
            </div>
          </div>
        </div>
      )}

      {/* ════ Gradient Memory per Layer ════ */}
      {analysis.gradient_memory_breakdown && analysis.gradient_memory_breakdown.length > 0 && (
        <div>
          <SectionIcon icon={Layers} label="Gradient Memory per Layer" />
          <div className="rounded-lg border border-border/50 bg-secondary/20 px-3 py-2 space-y-1.5">
            {analysis.gradient_memory_breakdown.slice(0, 8).map(e => (
              <div key={e.name}>
                <div className="flex justify-between text-[9px] mb-0.5">
                  <span className="text-muted-foreground truncate max-w-[50%]">{e.name}</span>
                  <span className="font-mono">fwd {fmtBytes(e.forward)} / bwd {fmtBytes(e.backward)}</span>
                </div>
                <div className="relative h-1 bg-secondary rounded-full overflow-hidden">
                  <div className="absolute inset-y-0 left-0 rounded-full bg-blue-500" style={{ width: `${Math.min((e.forward / (e.forward + e.backward + 1)) * 100, 100)}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ════ KV Cache Scaling ════ */}
      {analysis.kv_cache_scaling && analysis.kv_cache_scaling.length > 0 && (
        <div>
          <SectionIcon icon={Database} label="KV Cache Scaling" />
          <div className="rounded-lg border border-border/50 bg-secondary/20 px-3 py-2 space-y-0.5">
            {analysis.kv_cache_scaling.slice(0, 6).map(e => (
              <StatRow key={e.seq} label={`seq=${e.seq}`} value={fmtBytes(e.value)} />
            ))}
          </div>
        </div>
      )}

      {/* ════ Throughput & Latency ════ */}
      <div>
        <SectionIcon icon={Activity} label="Throughput & Latency" />
        <div className="rounded-lg border border-border/50 bg-secondary/20 px-3 py-2 space-y-0.5">
          <StatRow label="Latency" value={analysis.latencyMs != null ? `${fmt(analysis.latencyMs)} ms` : '—'} />
          <StatRow label="Throughput" value={analysis.throughputTokensPerS > 0 ? `${fmtNum(analysis.throughputTokensPerS)} tok/s` : '—'} />
          {(analysis.throughputGraphsPerS ?? 0) > 0 && <StatRow label="Throughput (graphs)" value={`${fmt(analysis.throughputGraphsPerS, 2)} graphs/s`} />}
          <StatRow label="GPU Utilization" value={analysis.gpuUtilization != null ? fmtPercent(analysis.gpuUtilization) : '—'} />
          {(analysis.tensorCoreUtilization ?? 0) > 0 && <StatRow label="Tensor Core Util." value={fmtPercent(analysis.tensorCoreUtilization)} />}
        </div>
      </div>

      {/* ════ FLOPs per Layer (clickable) ════ */}
      {perLayerFlopsRows && (
        <div>
          <SectionIcon icon={Activity} label="FLOPs per Layer" />
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

      {/* ════ Compilation Timeline ════ */}
      {analysis.compilation?.phase_timeline && analysis.compilation.phase_timeline.length > 0 && (
        <div>
          <SectionIcon icon={Activity} label="Compilation Phases" />
          <div className="rounded-lg border border-border/50 bg-secondary/20 px-3 py-2 space-y-0.5">
            <StatRow label="Current Phase" value={analysis.compilation.current_phase} mono={false} />
            {analysis.compilation.phase_timeline.map((phase, i) => {
              const statusColor = phase.status === 'completed' ? 'text-success'
                : phase.status === 'running' ? 'text-blue-400'
                : phase.status === 'error' ? 'text-destructive' : 'text-muted-foreground';
              return (
                <div key={i} className="flex items-center justify-between py-0.5">
                  <span className={cn('text-[10px]', statusColor)}>{phase.name}</span>
                  <span className="text-[9px] font-mono text-muted-foreground">
                    {phase.duration_ms >= 1000 ? `${(phase.duration_ms / 1000).toFixed(1)}s` : `${phase.duration_ms.toFixed(0)}ms`}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════
//  CATEGORY 3: Hardware & Cost
// ════════════════════════════════════════════════════════════════════

function HardwareCategory({ analysis }: { analysis: AnalysisResult }) {
  return (
    <div className="space-y-4">
      {/* ════ Summary Cards ════ */}
      <div className="grid grid-cols-2 gap-2">
        <StatCard label="Training Cost" value={analysis.trainingCostUsd > 0 ? `$${fmt(analysis.trainingCostUsd, 2)}` : '—'} sub={analysis.provider || ''} />
        <StatCard label="Training Time" value={analysis.trainingTimeHours > 0 ? `${fmt(analysis.trainingTimeHours, 2)} hrs` : '—'} sub={`${fmt(analysis.energyKwh, 2)} kWh`} />
        <StatCard label="CO₂ Emissions" value={analysis.co2Kg > 0 ? `${fmt(analysis.co2Kg, 3)} kg` : '—'} sub={analysis.gpuHours ? `${fmt(analysis.gpuHours, 2)} GPU hrs` : ''} />
        <StatCard label="GPU" value={analysis.gpuName || '—'} sub={analysis.gpuCount > 1 ? `${analysis.gpuCount} × ${fmt(analysis.gpuMemoryGb, 1)} GB` : `${fmt(analysis.gpuMemoryGb, 1)} GB`} />
      </div>

      {/* ════ Hardware Specs ════ */}
      {analysis.gpuName && (
        <div>
          <SectionIcon icon={Server} label="Hardware Specs" />
          <div className="rounded-lg border border-border/50 bg-secondary/20 px-3 py-2 space-y-0.5">
            <StatRow label="GPU" value={analysis.gpuName} mono={false} />
            <StatRow label="GPU Count" value={analysis.gpuCount} />
            <StatRow label="VRAM" value={`${fmt(analysis.gpuMemoryGb, 1)} GB`} />
            <StatRow label="Peak TFLOPs FP16" value={`${fmt(analysis.gpuTflops, 1)} TFLOPs`} />
            <StatRow label="HBM Bandwidth" value={`${fmt(analysis.gpuBandwidthGbs, 0)} GB/s`} />
            {analysis.interconnect && <StatRow label="Interconnect" value={analysis.interconnect} mono={false} />}
            {(analysis.interconnectBandwidthGbs ?? 0) > 0 && <StatRow label="IC Bandwidth" value={`${fmt(analysis.interconnectBandwidthGbs, 0)} GB/s`} />}
            {(analysis.tensorCoreUtilization ?? 0) > 0 && <StatRow label="Tensor Core Util." value={fmtPercent(analysis.tensorCoreUtilization)} />}
          </div>
        </div>
      )}

      {/* ════ Parallelism ════ */}
      <div>
        <SectionIcon icon={Cpu} label="Parallelism" />
        <div className="rounded-lg border border-border/50 bg-secondary/20 px-3 py-2 space-y-0.5">
          <StatRow label="Optimal GPUs" value={analysis.optimalGpuCount} />
          <StatRow label="Data Parallel Eff." value={fmtPercent(analysis.dataParallelEfficiency)} />
          <StatRow label="Comm. Overhead" value={fmtPercent(analysis.communicationOverhead)} />
          <StatRow label="Pipeline Stages" value={analysis.pipelineStages || '—'} />
          <StatRow label="Tensor Parallel" value={analysis.tensorParallelDegree || '—'} />
          {(analysis.dataParallel ?? 0) > 1 && <StatRow label="Data Parallel DP" value={analysis.dataParallel!} />}
          {(analysis.tensorParallel ?? 0) > 1 && <StatRow label="Tensor Parallel TP" value={analysis.tensorParallel!} />}
          {(analysis.pipelineParallel ?? 0) > 1 && <StatRow label="Pipeline Parallel PP" value={analysis.pipelineParallel!} />}
        </div>
      </div>

      {/* ════ Training Cost Details ════ */}
      <div>
        <SectionIcon icon={DollarSign} label="Training Cost Details" />
        <div className="rounded-lg border border-border/50 bg-secondary/20 px-3 py-2 space-y-0.5">
          <StatRow label="Estimated Cost" value={analysis.trainingCostUsd > 0 ? `$${fmt(analysis.trainingCostUsd, 2)}` : '—'} />
          <StatRow label="Training Time" value={analysis.trainingTimeHours > 0 ? `${fmt(analysis.trainingTimeHours, 2)} hrs` : '—'} />
          <StatRow label="Energy" value={analysis.energyKwh > 0 ? `${fmt(analysis.energyKwh, 2)} kWh` : '—'} />
          <StatRow label="CO₂" value={analysis.co2Kg > 0 ? `${fmt(analysis.co2Kg, 3)} kg` : '—'} />
          {(analysis.gpuHours ?? 0) > 0 && <StatRow label="GPU Hours" value={`${fmt(analysis.gpuHours, 2)} hrs`} />}
          {(analysis.costPerMillionTokensUsd ?? 0) > 0 && <StatRow label="Cost/M Tokens" value={`$${fmt(analysis.costPerMillionTokensUsd, 4)}`} />}
          {analysis.provider && <StatRow label="Provider" value={analysis.provider} mono={false} />}
        </div>
      </div>

      {/* ════ Numerical Stability ════ */}
      {analysis.dynamic?.stability && (
        <div>
          <SectionIcon icon={ShieldCheck} label="Numerical Stability" />
          <div className="rounded-lg border border-border/50 bg-secondary/20 px-3 py-2 space-y-0.5">
            <StatRow label="Robustness Score" value={fmt(analysis.dynamic.stability.global_robustness_score, 3)} />
            <StatRow label="Lyapunov Exp." value={fmt(analysis.dynamic.stability.lyapunov_exponent_mean, 4, true)} />
            <StatRow label="Chaos Index" value={fmt(analysis.dynamic.stability.chaos_index, 3, true)} />
            <StatRow label="FP32 Required" value={`${fmt(analysis.dynamic.stability.fp32_required_pct, 1)}%`} />
            <StatRow label="FP32 Mem OH" value={`${fmt(analysis.dynamic.stability.fp32_fallback_memory_overhead_gb, 2, true)} GB`} />
            <StatRow label="High Risk Layers" value={analysis.dynamic.stability.high_risk_layers_count} />
          </div>
        </div>
      )}

      {/* ════ MoE Routing ════ */}
      {/* Only the fields NEURAX can actually derive statically: whether the
          model routes through MoE at all, and an assumed-balanced baseline
          (no runtime trace exists to measure real imbalance from). The rest
          of this panel used to show cache locality, memory contention,
          numerical sensitivity and a "prediction confidence" — all hardcoded
          constants identical for every model, removed rather than left as
          fake precision. */}
      {analysis.dynamic?.behavioral?.has_moe && (
        <div>
          <SectionIcon icon={BrainCircuit} label="MoE Routing" />
          <div className="rounded-lg border border-border/50 bg-secondary/20 px-3 py-2 space-y-0.5">
            <StatRow label="Load Balance (assumed)" value={`${fmt(analysis.dynamic.behavioral.load_balance_efficiency, 1)}%`} />
            <StatRow label="Expert Imbalance (assumed)" value={fmt(analysis.dynamic.behavioral.expert_load_imbalance, 3, true)} />
          </div>
        </div>
      )}

      {/* ════ Runtime Config ════ */}
      <div>
        <SectionIcon icon={Info} label="Runtime Config" />
        <div className="rounded-lg border border-border/50 bg-secondary/20 px-3 py-2 space-y-0.5">
          {analysis.selectedPrecision && <StatRow label="Precision" value={analysis.selectedPrecision} mono={false} />}
          {(analysis.selectedBatchSize ?? 0) > 0 && <StatRow label="Batch Size" value={analysis.selectedBatchSize!} />}
          {analysis.isSequenceModel !== undefined && <StatRow label="Sequence Model" value={analysis.isSequenceModel ? 'Yes' : 'No'} mono={false} />}
          {(analysis.customLayerCount ?? 0) > 0 && <StatRow label="Custom Layers" value={analysis.customLayerCount!} />}
          <StatRow label="Confidence" value={fmtPercent(analysis.confidenceScore)} />
          {(analysis.analysisTimeMs ?? 0) > 0 && (
            <StatRow label="Analysis Time" value={`${(analysis.analysisTimeMs! / 1000).toFixed(2)}s`} />
          )}
        </div>
      </div>
    </div>
  );
}
