/**
 * Per Layer — the profile of the model along its own depth.
 *
 * The two charts this replaces were "Layers by FLOPs" and "Layers by VRAM":
 * horizontal bars, sorted descending, truncated to the top handful. Sorting is
 * what made them wrong. A network has an order, and the ranking destroys it —
 * "the first block is enormous and everything after it is flat" and "the cost
 * climbs steadily with depth" produce the same bar chart, and they call for
 * opposite changes.
 *
 * So every chart here keeps graph order on the x-axis and draws a curve. What
 * the ranking was genuinely good at — *which* layers dominate — is answered
 * better by the cumulative-share curve: it says "four layers carry 80 % of the
 * compute" as a shape, and reading a knee off a curve is easier than counting
 * bars.
 *
 * Arithmetic intensity per layer is the one figure here with no aggregate
 * equivalent anywhere else in the studio. FLOPs divided by bytes moved decides
 * whether a layer is limited by the card's compute or by its memory bus, and
 * it is per-layer by nature: a model is almost never uniformly one or the
 * other, and the mixed case is exactly the one a single headline number
 * cannot describe.
 */
import { Activity, Gauge, HardDrive, Layers, Timer } from 'lucide-react';
import { Area, AreaChart, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip } from 'recharts';

import { AnalysisResult, PerLayerBreakdownRow } from '@/types/architecture.ts';
import {
  CHART_GRID, CHART_MARGINS, ChartErrorBoundary, ChartGrid, ChartSlot, StatCard, StatStrip,
  ViewNote, XA, YA, chartTooltipStyle,
} from '../shared';
import {
  SIMULATION_COLORS, buildDerivedLayerMetrics, formatCompactNumber, hasAnalysisReportData,
} from '../simulationData.ts';

interface PerLayerChartsProps {
  analysis?: AnalysisResult;
  perLayer?: PerLayerBreakdownRow[];
}

function formatFlops(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return '—';
  if (value >= 1e15) return `${(value / 1e15).toFixed(1)} PF`;
  if (value >= 1e12) return `${(value / 1e12).toFixed(1)} TF`;
  if (value >= 1e9) return `${(value / 1e9).toFixed(1)} GF`;
  return `${(value / 1e6).toFixed(1)} MF`;
}

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

export function PerLayerCharts({ analysis, perLayer = [] }: PerLayerChartsProps) {
  if (!hasAnalysisReportData(analysis)) {
    return (
      <ChartErrorBoundary>
        <div className="flex flex-col items-center justify-center gap-2 py-24 text-center">
          <Layers className="w-8 h-8 text-muted-foreground/30" />
          <div className="text-[13px] font-semibold text-foreground">No per-layer data yet</div>
          <p className="text-[11px] text-muted-foreground max-w-[52ch] leading-relaxed">
            Run the analysis. The compiler produces parameters, FLOPs, latency and memory for every layer it lowers —
            these curves are that output, in graph order.
          </p>
        </div>
      </ChartErrorBoundary>
    );
  }

  const layers = buildDerivedLayerMetrics(analysis, perLayer);
  const hasDepth = layers.length > 1;

  const totalFlops = layers.reduce((sum, l) => sum + l.flops, 0);
  const totalLatency = layers.reduce((sum, l) => sum + l.latencyMs, 0);

  // Cumulative share, in graph order — the Pareto reading without the sort.
  let runningFlops = 0;
  let runningLatency = 0;
  const rows = layers.map((layer, i) => {
    runningFlops += layer.flops;
    runningLatency += layer.latencyMs;
    // Bytes moved: what the layer has to read and write to do its work. The
    // ratio against FLOPs is what decides compute- versus memory-bound.
    const bytes = layer.memoryMb * 1024 * 1024;
    return {
      i: i + 1,
      name: layer.name,
      kind: layer.kind,
      latency: Number(layer.latencyMs.toFixed(3)),
      vram: Number(layer.memoryMb.toFixed(2)),
      flops: layer.flops,
      params: layer.params,
      cumFlops: totalFlops > 0 ? Number(((runningFlops / totalFlops) * 100).toFixed(2)) : 0,
      cumLatency: totalLatency > 0 ? Number(((runningLatency / totalLatency) * 100).toFixed(2)) : 0,
      intensity: bytes > 0 ? Number((layer.flops / bytes).toFixed(2)) : 0,
    };
  });

  const heaviest = layers.reduce((a, b) => (b.flops > a.flops ? b : a), layers[0]);
  const slowest = layers.reduce((a, b) => (b.latencyMs > a.latencyMs ? b : a), layers[0]);
  // How many layers it takes to reach 80 % of the compute — the number the
  // sorted bar chart was really being read for.
  const knee = [...layers].sort((a, b) => b.flops - a.flops).reduce<{ n: number; acc: number }>(
    (state, l) => (state.acc >= totalFlops * 0.8 ? state : { n: state.n + 1, acc: state.acc + l.flops }),
    { n: 0, acc: 0 },
  ).n;
  const hasIntensity = rows.some((r) => r.intensity > 0);

  return (
    <ChartErrorBoundary>
      <div className="space-y-6">
        <StatStrip>
          <StatCard icon={<Layers className="w-3.5 h-3.5" />} label="Layers profiled" value={String(layers.length)} />
          <StatCard
            icon={<Activity className="w-3.5 h-3.5" />}
            label="Heaviest"
            value={heaviest?.name ?? '—'}
            sublabel={heaviest ? formatFlops(heaviest.flops) : undefined}
          />
          <StatCard
            icon={<Timer className="w-3.5 h-3.5" />}
            label="Slowest"
            value={slowest?.name ?? '—'}
            sublabel={slowest ? `${slowest.latencyMs.toFixed(2)} ms` : undefined}
          />
          <StatCard
            icon={<Gauge className="w-3.5 h-3.5" />}
            label="80% of compute"
            value={knee > 0 ? `${knee} layers` : '—'}
            sublabel={layers.length > 0 ? `of ${layers.length}` : undefined}
          />
          <StatCard
            icon={<Timer className="w-3.5 h-3.5" />}
            label="Summed latency"
            value={totalLatency > 0 ? `${totalLatency.toFixed(2)} ms` : '—'}
            sublabel="profiled layers only"
          />
          <StatCard
            icon={<HardDrive className="w-3.5 h-3.5" />}
            label="Params profiled"
            value={formatCompactNumber(layers.reduce((s, l) => s + l.params, 0))}
          />
        </StatStrip>

        <ChartGrid>
          <ChartSlot
            title="Latency across depth"
            has={hasDepth && totalLatency > 0}
            emptyIcon={Timer}
            emptyTitle="No per-layer latency"
            emptyHint="The compiler reported no latency per layer. Select a target card and re-run the analysis."
            reading="Time spent in each layer, in graph order. A spike is one operator; a plateau is the repeating block, and there the saving is per-layer times depth."
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={rows} margin={CHART_MARGINS.line}>
                {CHART_GRID}
                <XA dataKey="i" />
                <YA unit=" ms" width={56} />
                <Tooltip
                  {...chartTooltipStyle}
                  labelFormatter={(i) => rows[Number(i) - 1]?.name ?? `layer ${i}`}
                  formatter={(v: number) => `${v.toFixed(3)} ms`}
                />
                <Line
                  type="monotone"
                  dataKey="latency"
                  name="latency"
                  stroke={SIMULATION_COLORS.amber}
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
            emptyHint="The analysis produced no memory figures per layer for this design."
            reading="What each layer holds while it runs. The area under this curve is the activation term of peak VRAM — the one gradient checkpointing trades for time."
          >
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={rows} margin={CHART_MARGINS.area}>
                {CHART_GRID}
                <XA dataKey="i" />
                <YA unit=" MB" width={56} />
                <Tooltip
                  {...chartTooltipStyle}
                  labelFormatter={(i) => rows[Number(i) - 1]?.name ?? `layer ${i}`}
                  formatter={(v: number) => `${v.toFixed(2)} MB`}
                />
                <Area
                  type="monotone"
                  dataKey="vram"
                  name="memory"
                  stroke={SIMULATION_COLORS.blue}
                  fill={SIMULATION_COLORS.blue}
                  fillOpacity={0.28}
                  strokeWidth={1.5}
                  isAnimationActive={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </ChartSlot>

          <ChartSlot
            title="Cumulative share of cost"
            has={hasDepth && totalFlops > 0}
            emptyIcon={Gauge}
            emptyTitle="No cost distribution"
            emptyHint="The analysis reported no FLOPs per layer for this design."
            action={
              <Key
                entries={[
                  { label: 'compute', color: SIMULATION_COLORS.violet },
                  { label: 'time', color: SIMULATION_COLORS.amber },
                ]}
              />
            }
            reading="How the total accumulates as you walk the graph. A curve that leaves the diagonal early means the cost is concentrated near the input; one that hugs it means every layer costs the same, and there is no single thing to optimise."
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={rows} margin={CHART_MARGINS.line}>
                {CHART_GRID}
                <XA dataKey="i" />
                <YA unit=" %" domain={[0, 100]} width={44} />
                <Tooltip
                  {...chartTooltipStyle}
                  labelFormatter={(i) => rows[Number(i) - 1]?.name ?? `layer ${i}`}
                  formatter={(v: number) => `${v.toFixed(1)} %`}
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
                  dataKey="cumFlops"
                  name="compute"
                  stroke={SIMULATION_COLORS.violet}
                  strokeWidth={1.75}
                  dot={false}
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="cumLatency"
                  name="time"
                  stroke={SIMULATION_COLORS.amber}
                  strokeWidth={1.75}
                  strokeDasharray="4 3"
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </ChartSlot>

          <ChartSlot
            title="Arithmetic intensity across depth"
            has={hasDepth && hasIntensity}
            emptyIcon={Activity}
            emptyTitle="No intensity to compute"
            emptyHint="Arithmetic intensity needs both FLOPs and bytes moved per layer; this analysis reported only one of them."
            reading={
              analysis.gpuTflops > 0 && analysis.gpuBandwidthGbs > 0 ? (
                <>
                  FLOPs per byte moved. The dashed line is this card's ridge point (
                  {((analysis.gpuTflops * 1e12) / (analysis.gpuBandwidthGbs * 1e9)).toFixed(0)} FLOP/byte): layers below
                  it are waiting on memory, layers above it are doing arithmetic.
                </>
              ) : (
                'FLOPs per byte moved. Layers low on this curve are limited by the memory bus rather than by compute.'
              )
            }
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={rows} margin={CHART_MARGINS.line}>
                {CHART_GRID}
                <XA dataKey="i" />
                <YA width={56} />
                <Tooltip
                  {...chartTooltipStyle}
                  labelFormatter={(i) => rows[Number(i) - 1]?.name ?? `layer ${i}`}
                  formatter={(v: number) => `${v.toFixed(1)} FLOP/byte`}
                />
                {analysis.gpuTflops > 0 && analysis.gpuBandwidthGbs > 0 ? (
                  <ReferenceLine
                    y={(analysis.gpuTflops * 1e12) / (analysis.gpuBandwidthGbs * 1e9)}
                    stroke={SIMULATION_COLORS.red}
                    strokeDasharray="4 3"
                    label={{
                      value: 'ridge point',
                      position: 'insideTopRight',
                      fontSize: 9,
                      fill: 'hsl(var(--muted-foreground))',
                    }}
                  />
                ) : null}
                <Line
                  type="monotone"
                  dataKey="intensity"
                  name="intensity"
                  stroke={SIMULATION_COLORS.teal}
                  strokeWidth={1.75}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </ChartSlot>
        </ChartGrid>

        <ViewNote icon={Layers}>
          Layers are kept in graph order, never sorted by cost — the order is the model, and a ranking hides whether the
          expense is one block or the whole stack. Which layers dominate is the cumulative curve's job.
        </ViewNote>
      </div>
    </ChartErrorBoundary>
  );
}
