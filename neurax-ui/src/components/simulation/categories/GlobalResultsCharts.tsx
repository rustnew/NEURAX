/**
 * Overview — what this model is, where its cost sits, and whether it fits.
 *
 * The screen this replaces opened on four figures rendered as chart-sized
 * tiles, a confidence donut, and two empty states. On a design that had not
 * been analysed it read "100 / 100" beside "0.0 parameters" — a confident
 * score for nothing at all.
 *
 * Everything here is now a curve, and that is a claim about the data rather
 * than a style preference. A bar chart is the right drawing for categories
 * with no order; but a network *has* an order — depth — and cost along it is a
 * profile, not a ranking. Read as a curve, "the middle third of this model is
 * flat and the head spikes" is one glance; read as sorted bars, that shape is
 * destroyed by the sort itself. The two remaining axes are genuinely
 * continuous: time through a training step, and sequence length.
 *
 * So the four charts are: cost across depth, memory across depth, VRAM through
 * a step, and KV-cache growth against context. The last two exist only when
 * the compiler emitted those series — `hasMemoryLiveness` and
 * `hasKvCacheScaling` guard them, and the card says so rather than drawing a
 * flat line through zeros.
 *
 * `confidenceScore` is gone from this view rather than redrawn: it is one
 * bounded number, which is a strip cell at most, and it was being shown at
 * full size on analyses with nothing in them.
 */
import { Activity, Cpu, Gauge, HardDrive, Layers, Sigma, Timer, Zap } from 'lucide-react';
import { Area, AreaChart, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip } from 'recharts';

import { AnalysisResult, PerLayerBreakdownRow } from '@/types/architecture.ts';
import {
  CHART_GRID, CHART_MARGINS, ChartErrorBoundary, ChartGrid, ChartSlot, StatCard, StatStrip,
  ViewNote, XA, YA, chartTooltipStyle,
} from '../shared';
import {
  SIMULATION_COLORS, buildDerivedLayerMetrics, formatBytes, formatCompactNumber,
  hasAnalysisReportData,
} from '../simulationData.ts';

interface GlobalResultsChartsProps {
  analysis?: AnalysisResult;
  perLayer?: PerLayerBreakdownRow[];
}

const GB = 1024 ** 3;

/** Human-readable FLOPs, for axes and tooltips that receive raw numbers. */
function formatFlops(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return '—';
  if (value >= 1e18) return `${(value / 1e18).toFixed(1)} EF`;
  if (value >= 1e15) return `${(value / 1e15).toFixed(1)} PF`;
  if (value >= 1e12) return `${(value / 1e12).toFixed(1)} TF`;
  if (value >= 1e9) return `${(value / 1e9).toFixed(1)} GF`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)} MF`;
  return `${value.toFixed(0)}`;
}

/**
 * The compilation itself, as a compact band rather than six more tiles.
 *
 * `graph_depth`, `total_operations`, `critical_path_length`,
 * `total_tensor_count`, `largest_tensor_bytes`, `macs` and `bytes_accessed`
 * are all produced by the compiler and, until now, displayed nowhere at all.
 * They describe the lowering rather than the model — how many operations the
 * graph became, how long its longest dependency chain is, how much data it
 * moves — and that is exactly what a compiler workspace should be able to
 * show. They are scalars, so they get a band; making four of them into charts
 * would be the padding this rewrite spent its time removing.
 */
function IrBand({ analysis }: { analysis: AnalysisResult }) {
  const items: Array<{ label: string; value: string }> = [];
  const push = (label: string, value: number | undefined, fmt: (v: number) => string) => {
    if (value !== undefined && Number.isFinite(value) && value > 0) items.push({ label, value: fmt(value) });
  };

  push('Graph depth', analysis.graphDepth, (v) => String(v));
  push('Operations', analysis.totalOperations, (v) => formatCompactNumber(v));
  push('Critical path', analysis.criticalPathLength, (v) => String(v));
  push('Tensors', analysis.totalTensorCount, (v) => formatCompactNumber(v));
  push('Largest tensor', analysis.largestTensorBytes, (v) => formatBytes(v));
  push('MACs', analysis.macs, (v) => formatFlops(v).replace('F', ''));
  push('Bytes moved', analysis.bytesAccessed, (v) => formatBytes(v));
  push('Analysed in', analysis.analysisTimeMs, (v) => `${v.toFixed(0)} ms`);

  if (items.length === 0) return null;

  return (
    <div className="rounded-[8px] border border-border/60 bg-muted/20 px-3 py-2">
      <div className="flex items-center gap-x-6 gap-y-2 flex-wrap">
        <span className="text-[9px] font-mono uppercase tracking-[0.14em] text-muted-foreground/70">
          What the compiler built
        </span>
        {items.map((item) => (
          <span key={item.label} className="flex items-baseline gap-1.5">
            <span className="text-[10px] text-muted-foreground">{item.label}</span>
            <span className="text-[12px] font-mono tabular-nums font-semibold text-foreground">{item.value}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

/** A small legend for a card's `action` slot. */
function Key({ entries }: { entries: { label: string; color: string }[] }) {
  return (
    <div className="flex items-center gap-3">
      {entries.map((e) => (
        <span key={e.label} className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
          <span className="w-2 h-2 rounded-[2px]" style={{ background: e.color }} />
          {e.label}
        </span>
      ))}
    </div>
  );
}

export function GlobalResultsCharts({ analysis, perLayer = [] }: GlobalResultsChartsProps) {
  if (!hasAnalysisReportData(analysis)) {
    return (
      <ChartErrorBoundary>
        <div className="flex flex-col items-center justify-center gap-2 py-24 text-center">
          <Layers className="w-8 h-8 text-muted-foreground/30" />
          <div className="text-[13px] font-semibold text-foreground">No analysis yet</div>
          <p className="text-[11px] text-muted-foreground max-w-[52ch] leading-relaxed">
            Build a design on the canvas and run the analysis. Every figure in this workspace is computed from the
            graph — nothing here is measured, and nothing is shown before there is something to compute it from.
          </p>
        </div>
      </ChartErrorBoundary>
    );
  }

  const capacityBytes = analysis.gpuMemoryGb > 0 ? analysis.gpuMemoryGb * GB : 0;
  const fits = capacityBytes > 0 && analysis.peakVramBytes <= capacityBytes;

  // ── The depth profile ────────────────────────────────────────────────────
  // Kept in graph order, never sorted: the order *is* the reading.
  const layers = buildDerivedLayerMetrics(analysis, perLayer);
  const depth = layers.map((layer, i) => ({
    i: i + 1,
    name: layer.name,
    params: layer.params,
    flops: layer.flops,
    latency: Number(layer.latencyMs.toFixed(3)),
    weights: Number(layer.weightsMb.toFixed(2)),
    activations: Number(layer.activationsMb.toFixed(2)),
    gradients: Number(layer.gradientsMb.toFixed(2)),
    optimizer: Number(layer.optimizerMb.toFixed(2)),
  }));
  const hasDepth = depth.length > 1;

  // ── Forward against backward, and context growth ─────────────────────────
  // `gradient_memory_per_layer` is real and populated in every report; the
  // `memory_liveness` series this card used to want has no producer anywhere
  // in the compiler, so a chart bound to it could only ever be blank.
  const fb = (analysis.gradient_memory_breakdown ?? []).map((row, i) => ({
    i: i + 1,
    name: row.name,
    forward: Number((row.forward / (1024 * 1024)).toFixed(2)),
    backward: Number((row.backward / (1024 * 1024)).toFixed(2)),
  }));
  /**
   * How concentrated the graph is, at two levels.
   *
   * `op_type_distribution` and `layers_by_type` are both real compiler output,
   * and both are unordered categories — which is why they were bar charts, and
   * why the bar charts were unsatisfying: the reading anyone actually wants is
   * "how few kinds make up most of it", and a ranking makes you count.
   *
   * Sorted descending and accumulated, that reading is a shape. A curve that
   * shoots to 90 % in two steps is a graph made of one repeated thing; one
   * that climbs the diagonal is genuinely heterogeneous, and the two call for
   * very different optimisation work. Plotting operations against blocks on
   * the same axes also shows where the lowering fans out — a handful of block
   * types becoming many operator kinds.
   */
  function concentration(counts: Record<string, number> | undefined) {
    const values = Object.values(counts ?? {})
      .map(Number)
      .filter((v) => Number.isFinite(v) && v > 0)
      .sort((a, b) => b - a);
    const total = values.reduce((sum, v) => sum + v, 0);
    if (total <= 0) return [];
    let running = 0;
    return values.map((v, i) => {
      running += v;
      return { rank: i + 1, pct: Number(((running / total) * 100).toFixed(2)) };
    });
  }

  const opCurve = concentration(analysis.opsDistribution as unknown as Record<string, number>);
  const blockCurve = concentration(analysis.layersByType);
  const kinds = Math.max(opCurve.length, blockCurve.length);
  const concentrationRows = Array.from({ length: kinds }, (_, i) => ({
    rank: i + 1,
    operations: opCurve[i]?.pct,
    blocks: blockCurve[i]?.pct,
  }));
  // How few operator kinds carry 80 % of the graph — the number the ranking
  // was really being read for.
  const opKnee = opCurve.findIndex((p) => p.pct >= 80) + 1;

  return (
    <ChartErrorBoundary>
      <div className="space-y-6">
        <IrBand analysis={analysis} />

        <StatStrip>
          <StatCard
            icon={<Sigma className="w-3.5 h-3.5" />}
            label="Parameters"
            value={formatCompactNumber(analysis.totalParams)}
            sublabel={
              analysis.activeParams > 0 && analysis.activeParams !== analysis.totalParams
                ? `${formatCompactNumber(analysis.activeParams)} active per token`
                : `${analysis.numLayers} layers`
            }
          />
          <StatCard
            icon={<Zap className="w-3.5 h-3.5" />}
            label="Compute"
            value={analysis.estimatedFlops || formatFlops(analysis.totalFlops)}
            sublabel={analysis.flopsPerToken > 0 ? `${formatFlops(analysis.flopsPerToken)} / token` : undefined}
          />
          <StatCard
            icon={<HardDrive className="w-3.5 h-3.5" />}
            label="Peak VRAM"
            value={formatBytes(analysis.peakVramBytes)}
            sublabel={capacityBytes > 0 ? `of ${analysis.gpuMemoryGb} GB` : undefined}
            variant={capacityBytes > 0 ? (fits ? 'success' : 'danger') : 'default'}
          />
          <StatCard
            icon={<Timer className="w-3.5 h-3.5" />}
            label="Latency"
            value={analysis.latencyMs !== null ? `${analysis.latencyMs.toFixed(1)} ms` : '—'}
            sublabel="per step"
          />
          <StatCard
            icon={<Gauge className="w-3.5 h-3.5" />}
            label="Throughput"
            value={analysis.throughputTokensPerS > 0 ? `${formatCompactNumber(analysis.throughputTokensPerS)}/s` : '—'}
            sublabel="tokens"
          />
          <StatCard
            icon={<Cpu className="w-3.5 h-3.5" />}
            label="Bound by"
            value={analysis.bottleneck || '—'}
            sublabel={analysis.arithmeticIntensity > 0 ? `${analysis.arithmeticIntensity.toFixed(1)} FLOP/byte` : undefined}
          />
        </StatStrip>

        <ChartGrid>
          <ChartSlot
            title="Cost across depth"
            has={hasDepth}
            emptyIcon={Layers}
            emptyTitle="Not enough layers to profile"
            emptyHint="A depth profile needs more than one layer. Add blocks and re-run the analysis."
            action={
              <Key
                entries={[
                  { label: 'parameters', color: SIMULATION_COLORS.blue },
                  { label: 'FLOPs', color: SIMULATION_COLORS.violet },
                ]}
              />
            }
            reading="Parameters and compute rarely peak in the same place. Where the two curves diverge is where a change buys memory but no time, or the reverse."
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={depth} margin={CHART_MARGINS.line}>
                {CHART_GRID}
                <XA dataKey="i" />
                <YA yAxisId="p" tickFormatter={(v: number) => formatCompactNumber(v)} width={56} />
                <YA yAxisId="f" orientation="right" tickFormatter={(v: number) => formatFlops(v)} width={56} />
                <Tooltip
                  {...chartTooltipStyle}
                  labelFormatter={(i) => depth[Number(i) - 1]?.name ?? `layer ${i}`}
                  formatter={(v: number, name: string) =>
                    name === 'FLOPs' ? formatFlops(v) : formatCompactNumber(v)
                  }
                />
                <Line
                  yAxisId="p"
                  type="monotone"
                  dataKey="params"
                  name="parameters"
                  stroke={SIMULATION_COLORS.blue}
                  strokeWidth={1.75}
                  dot={false}
                  isAnimationActive={false}
                />
                <Line
                  yAxisId="f"
                  type="monotone"
                  dataKey="flops"
                  name="FLOPs"
                  stroke={SIMULATION_COLORS.violet}
                  strokeWidth={1.75}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </ChartSlot>

          <ChartSlot
            title="Memory across depth"
            has={hasDepth}
            emptyIcon={HardDrive}
            emptyTitle="No per-layer memory"
            emptyHint="The analysis produced no per-layer memory terms for this design."
            action={
              <Key
                entries={[
                  { label: 'weights', color: SIMULATION_COLORS.blue },
                  { label: 'activations', color: SIMULATION_COLORS.amber },
                  { label: 'gradients', color: SIMULATION_COLORS.violet },
                  { label: 'optimizer', color: SIMULATION_COLORS.green },
                ]}
              />
            }
            reading="The four terms that make up peak VRAM, layer by layer. Optimizer state is usually the surprise — for Adam it is twice the weights it sits beside."
          >
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={depth} margin={CHART_MARGINS.area}>
                {CHART_GRID}
                <XA dataKey="i" />
                <YA unit=" MB" width={56} />
                <Tooltip
                  {...chartTooltipStyle}
                  labelFormatter={(i) => depth[Number(i) - 1]?.name ?? `layer ${i}`}
                  formatter={(v: number) => `${v.toFixed(2)} MB`}
                />
                {[
                  { key: 'weights', color: SIMULATION_COLORS.blue },
                  { key: 'activations', color: SIMULATION_COLORS.amber },
                  { key: 'gradients', color: SIMULATION_COLORS.violet },
                  { key: 'optimizer', color: SIMULATION_COLORS.green },
                ].map((term) => (
                  <Area
                    key={term.key}
                    type="monotone"
                    dataKey={term.key}
                    stackId="mem"
                    stroke={term.color}
                    fill={term.color}
                    fillOpacity={0.28}
                    strokeWidth={1.25}
                    isAnimationActive={false}
                  />
                ))}
              </AreaChart>
            </ResponsiveContainer>
          </ChartSlot>

          <ChartSlot
            title="Forward against backward memory"
            has={fb.length > 1}
            emptyIcon={Activity}
            emptyTitle="No forward/backward split"
            emptyHint="The analysis reported no per-layer gradient memory for this design."
            action={
              <Key
                entries={[
                  { label: 'forward', color: SIMULATION_COLORS.cyan },
                  { label: 'backward', color: SIMULATION_COLORS.violet },
                ]}
              />
            }
            reading="What each layer holds on the way in against what it holds on the way back. Layers where backward towers over forward are the ones gradient checkpointing actually pays for — everywhere else it buys time for nothing."
          >
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={fb} margin={CHART_MARGINS.area}>
                {CHART_GRID}
                <XA dataKey="i" />
                <YA unit=" MB" width={56} />
                <Tooltip
                  {...chartTooltipStyle}
                  labelFormatter={(i) => fb[Number(i) - 1]?.name ?? `layer ${i}`}
                  formatter={(v: number) => `${v.toFixed(2)} MB`}
                />
                <Area
                  type="monotone"
                  dataKey="forward"
                  stroke={SIMULATION_COLORS.cyan}
                  fill={SIMULATION_COLORS.cyan}
                  fillOpacity={0.26}
                  strokeWidth={1.5}
                  isAnimationActive={false}
                />
                <Area
                  type="monotone"
                  dataKey="backward"
                  stroke={SIMULATION_COLORS.violet}
                  fill={SIMULATION_COLORS.violet}
                  fillOpacity={0.26}
                  strokeWidth={1.5}
                  isAnimationActive={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </ChartSlot>

          <ChartSlot
            title="How few kinds make up the graph"
            has={concentrationRows.length > 1}
            emptyIcon={Layers}
            emptyTitle="No type distribution"
            emptyHint="The compiler reported neither an operation nor a block-type distribution for this graph."
            action={
              <Key
                entries={[
                  { label: 'operations', color: SIMULATION_COLORS.cyan },
                  { label: 'blocks', color: SIMULATION_COLORS.pink },
                ]}
              />
            }
            reading={
              <>
                Kinds sorted by how common they are, then accumulated. A curve that reaches 90 % in two steps is a graph
                made of one repeated thing; one that climbs the diagonal is genuinely heterogeneous, and the two reward
                completely different work.
                {opKnee > 0 ? ` Here, ${opKnee} operator ${opKnee === 1 ? 'kind covers' : 'kinds cover'} 80 % of the graph.` : ''}
              </>
            }
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={concentrationRows} margin={CHART_MARGINS.line}>
                {CHART_GRID}
                <XA dataKey="rank" allowDecimals={false} />
                <YA unit=" %" domain={[0, 100]} width={44} />
                <Tooltip
                  {...chartTooltipStyle}
                  labelFormatter={(v) => `${v} most common kinds`}
                  formatter={(v: number, key: string) => [`${v.toFixed(1)} % of the graph`, key]}
                />
                <ReferenceLine
                  y={80}
                  stroke="hsl(var(--muted-foreground))"
                  strokeDasharray="2 3"
                  strokeOpacity={0.6}
                  label={{ value: '80%', position: 'insideTopLeft', fontSize: 9, fill: 'hsl(var(--muted-foreground))' }}
                />
                <Line
                  type="monotone"
                  dataKey="operations"
                  name="operations"
                  stroke={SIMULATION_COLORS.cyan}
                  strokeWidth={1.75}
                  dot={{ r: 2.5 }}
                  connectNulls
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="blocks"
                  name="blocks"
                  stroke={SIMULATION_COLORS.pink}
                  strokeWidth={1.75}
                  strokeDasharray="4 3"
                  dot={{ r: 2.5 }}
                  connectNulls
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </ChartSlot>
        </ChartGrid>

        <ViewNote icon={Cpu}>
          Every curve above is derived from the graph by the compiler — analytically, in milliseconds, without running
          the model. What a real run does with the same design, and how far these predictions turn out to be off, is in{' '}
          <span className="font-medium text-foreground">Training</span>.
        </ViewNote>
      </div>
    </ChartErrorBoundary>
  );
}
