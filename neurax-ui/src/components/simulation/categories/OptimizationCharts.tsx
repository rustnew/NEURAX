/**
 * Optimization — what is actually limiting this design, and by how much.
 *
 * Three charts stood here. The Bottleneck Pareto was a sorted bar chart of the
 * same per-layer costs the Per Layer view now draws as a cumulative curve, so
 * it is gone rather than duplicated. "Compute vs Memory Bound" was a two-slice
 * pie of one boolean, which is a sentence, not a chart — it is a strip cell
 * now.
 *
 * The Roofline stays, because it is the one drawing in the studio that cannot
 * be replaced by a number. A model is not simply "compute bound" or "memory
 * bound": every layer sits somewhere on that spectrum, and the ceiling it runs
 * into depends on where. Plotting the layers against the card's own ceiling
 * shows which of them are nowhere near it — and it is the honest shape,
 * because the ceiling really is two straight lines meeting at the ridge point.
 *
 * It is drawn from **one** dataset holding both series, rather than a chart
 * with no data and a `data` prop on each child. The latter is what this file
 * did before, and it is the same shape that silently rendered a zero-height
 * chart elsewhere in this rewrite: the axes have nothing to derive a domain
 * from, so the drawing collapses with no error to notice.
 *
 * Everything plotted comes from the compiler's per-layer FLOPs, bytes and
 * latency, and from the target card's published throughput and bandwidth. No
 * figure here is measured, and none is extrapolated past what was reported.
 */
import { Activity, Cpu, Gauge, Target, Timer, Zap } from 'lucide-react';
import { Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip } from 'recharts';

import { AnalysisResult, PerLayerBreakdownRow } from '@/types/architecture.ts';
import {
  CHART_GRID, CHART_MARGINS, ChartErrorBoundary, ChartGrid, ChartSlot, StatCard, StatStrip,
  ViewNote, XA, YA, chartTooltipStyle,
} from '../shared';
import { SIMULATION_COLORS, buildDerivedLayerMetrics, hasAnalysisReportData } from '../simulationData.ts';

interface OptimizationChartsProps {
  analysis?: AnalysisResult;
  perLayer?: PerLayerBreakdownRow[];
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

export function OptimizationCharts({ analysis, perLayer = [] }: OptimizationChartsProps) {
  if (!hasAnalysisReportData(analysis)) {
    return (
      <ChartErrorBoundary>
        <div className="flex flex-col items-center justify-center gap-2 py-24 text-center">
          <Target className="w-8 h-8 text-muted-foreground/30" />
          <div className="text-[13px] font-semibold text-foreground">No optimization analysis yet</div>
          <p className="text-[11px] text-muted-foreground max-w-[52ch] leading-relaxed">
            Run the analysis with a target card selected. What limits a design depends on the card it runs on, so there
            is nothing to say until both exist.
          </p>
        </div>
      </ChartErrorBoundary>
    );
  }

  const peakFlops = analysis.gpuTflops * 1e12;
  const bandwidth = analysis.gpuBandwidthGbs * 1e9;
  const ridge = bandwidth > 0 ? peakFlops / bandwidth : 0;
  const hasCard = peakFlops > 0 && bandwidth > 0;

  const layers = buildDerivedLayerMetrics(analysis, perLayer);

  // Each layer's own intensity and the throughput its predicted latency
  // implies — both from the compiler, no measurement involved.
  const points = layers
    .map((layer, i) => {
      const bytes = layer.memoryMb * 1024 * 1024;
      const intensity = bytes > 0 ? layer.flops / bytes : 0;
      const achievedFlops = layer.latencyMs > 0 ? layer.flops / (layer.latencyMs / 1000) : 0;
      const ceilingFlops = hasCard ? Math.min(peakFlops, bandwidth * intensity) : 0;
      return {
        i: i + 1,
        name: layer.name,
        latencyMs: layer.latencyMs,
        intensity,
        achieved: achievedFlops / 1e12,
        ceiling: ceilingFlops / 1e12,
        // Time this layer would save if it ran at its own ceiling. The
        // honest form of "what would optimising this buy": a layer at 20 %
        // of a ceiling it barely uses saves less than one at 60 % of a high
        // one, and only the millisecond figure says which.
        recoverableMs:
          achievedFlops > 0 && ceilingFlops > achievedFlops
            ? layer.latencyMs - layer.latencyMs * (achievedFlops / ceilingFlops)
            : 0,
      };
    })
    .filter((p) => p.intensity > 0 && p.achieved > 0);

  const hasPoints = points.length > 1;

  // ── The roofline, as a single ordered dataset ────────────────────────────
  const minI = points.length ? Math.min(...points.map((p) => p.intensity)) : 1;
  const maxI = points.length ? Math.max(...points.map((p) => p.intensity)) : 1;
  const lo = Math.max(0.05, Math.min(minI, ridge > 0 ? ridge : minI) / 4);
  const hi = Math.max(maxI, ridge) * 4;
  const rooflineRows =
    hasCard && hasPoints
      ? [
          ...Array.from({ length: 48 }, (_, k) => {
            const intensity = lo * Math.pow(hi / lo, k / 47);
            return {
              intensity: Number(intensity.toFixed(4)),
              ceiling: Number((Math.min(peakFlops, bandwidth * intensity) / 1e12).toFixed(4)),
              layer: undefined as number | undefined,
              name: '',
            };
          }),
          ...points.map((p) => ({
            intensity: Number(p.intensity.toFixed(4)),
            ceiling: Number(p.ceiling.toFixed(4)),
            layer: Number(p.achieved.toFixed(4)),
            name: p.name,
          })),
        ].sort((a, b) => a.intensity - b.intensity)
      : [];

  const efficiency = points.map((p) => ({
    i: p.i,
    name: p.name,
    pct: p.ceiling > 0 ? Number(((p.achieved / p.ceiling) * 100).toFixed(2)) : 0,
  }));

  const recoverable = points.map((p) => ({
    i: p.i,
    name: p.name,
    ms: Number(p.recoverableMs.toFixed(3)),
  }));
  const totalRecoverable = recoverable.reduce((sum, r) => sum + r.ms, 0);

  const memoryBound = analysis.arithmeticIntensity > 0 && ridge > 0 && analysis.arithmeticIntensity < ridge;

  return (
    <ChartErrorBoundary>
      <div className="space-y-6">
        <StatStrip>
          <StatCard
            icon={<Cpu className="w-3.5 h-3.5" />}
            label="Limited by"
            value={analysis.bottleneck || (hasCard ? (memoryBound ? 'memory' : 'compute') : '—')}
            sublabel={hasCard ? `ridge ${ridge.toFixed(0)} FLOP/byte` : undefined}
          />
          <StatCard
            icon={<Activity className="w-3.5 h-3.5" />}
            label="Intensity"
            value={analysis.arithmeticIntensity > 0 ? analysis.arithmeticIntensity.toFixed(1) : '—'}
            sublabel="FLOP per byte"
          />
          <StatCard
            icon={<Zap className="w-3.5 h-3.5" />}
            label="Effective"
            value={analysis.effectiveTflops ? `${analysis.effectiveTflops.toFixed(1)} TF` : '—'}
            sublabel={analysis.gpuTflops > 0 ? `of ${analysis.gpuTflops.toFixed(0)} TF peak` : undefined}
          />
          <StatCard
            icon={<Gauge className="w-3.5 h-3.5" />}
            label="GPU used"
            value={analysis.gpuUtilization !== null ? `${analysis.gpuUtilization.toFixed(0)} %` : '—'}
            variant={(analysis.gpuUtilization ?? 100) < 50 ? 'warning' : 'default'}
          />
          <StatCard
            icon={<Timer className="w-3.5 h-3.5" />}
            label="Recoverable"
            value={totalRecoverable > 0 ? `${totalRecoverable.toFixed(2)} ms` : '—'}
            sublabel="if every layer hit its ceiling"
          />
          <StatCard
            label="Optimal devices"
            value={analysis.optimalGpuCount > 0 ? String(analysis.optimalGpuCount) : '—'}
            sublabel={
              analysis.dataParallelEfficiency > 0
                ? `${(analysis.dataParallelEfficiency * 100).toFixed(0)}% scaling`
                : undefined
            }
          />
        </StatStrip>

        <ChartGrid>
          <ChartSlot
            title="Roofline — every layer against the card's ceiling"
            has={rooflineRows.length > 0}
            emptyIcon={Target}
            emptyTitle="No roofline to draw"
            emptyHint="A roofline needs the target card's peak throughput and bandwidth, and per-layer FLOPs and bytes. Select a card and re-run the analysis."
            action={
              <Key
                entries={[
                  { label: 'ceiling', color: SIMULATION_COLORS.red },
                  { label: 'layers', color: SIMULATION_COLORS.blue },
                ]}
              />
            }
            reading={
              <>
                Both axes are logarithmic. The ceiling is two straight lines: below the ridge point (
                {ridge.toFixed(0)} FLOP/byte) a layer can only go as fast as the memory bus feeds it; above it the
                arithmetic units are the limit. A dot far under the line is leaving performance unused.
              </>
            }
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={rooflineRows} margin={CHART_MARGINS.line}>
                {CHART_GRID}
                <XA
                  type="number"
                  dataKey="intensity"
                  scale="log"
                  domain={[lo, hi]}
                  allowDataOverflow
                  tickFormatter={(v: number) => (v >= 1 ? v.toFixed(0) : v.toFixed(2))}
                />
                <YA type="number" scale="log" domain={['auto', 'auto']} allowDataOverflow unit=" TF" width={62} />
                <Tooltip
                  {...chartTooltipStyle}
                  labelFormatter={(v) => `${Number(v).toFixed(1)} FLOP/byte`}
                  formatter={(v: number, key: string) => [`${v.toFixed(2)} TFLOP/s`, key]}
                />
                {ridge > 0 ? (
                  <ReferenceLine
                    x={ridge}
                    stroke="hsl(var(--muted-foreground))"
                    strokeDasharray="2 3"
                    label={{
                      value: 'ridge',
                      position: 'insideTopLeft',
                      fontSize: 9,
                      fill: 'hsl(var(--muted-foreground))',
                    }}
                  />
                ) : null}
                <Line
                  type="monotone"
                  dataKey="ceiling"
                  name="ceiling"
                  stroke={SIMULATION_COLORS.red}
                  strokeWidth={1.75}
                  dot={false}
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="layer"
                  name="layer"
                  stroke={SIMULATION_COLORS.blue}
                  strokeWidth={0}
                  dot={{ r: 3.5, fill: SIMULATION_COLORS.blue }}
                  connectNulls={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </ChartSlot>

          <ChartSlot
            title="Throughput across depth"
            has={hasPoints}
            emptyIcon={Zap}
            emptyTitle="No per-layer throughput"
            emptyHint="This needs FLOPs and latency for each layer; the analysis reported only one of them."
            action={
              <Key
                entries={[
                  { label: 'achieved', color: SIMULATION_COLORS.violet },
                  { label: 'ceiling', color: SIMULATION_COLORS.red },
                ]}
              />
            }
            reading="What each layer achieves, from the compiler's own FLOPs and predicted latency, against the best its intensity allows. The gap between the two curves is the whole opportunity."
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={points} margin={CHART_MARGINS.line}>
                {CHART_GRID}
                <XA dataKey="i" />
                <YA unit=" TF" width={56} />
                <Tooltip
                  {...chartTooltipStyle}
                  labelFormatter={(i) => points[Number(i) - 1]?.name ?? `layer ${i}`}
                  formatter={(v: number, key: string) => [`${v.toFixed(2)} TFLOP/s`, key]}
                />
                <Line
                  type="monotone"
                  dataKey="ceiling"
                  name="ceiling"
                  stroke={SIMULATION_COLORS.red}
                  strokeWidth={1.5}
                  strokeDasharray="4 3"
                  dot={false}
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="achieved"
                  name="achieved"
                  stroke={SIMULATION_COLORS.violet}
                  strokeWidth={1.75}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </ChartSlot>

          <ChartSlot
            title="How much of the ceiling each layer reaches"
            has={hasCard && efficiency.length > 1}
            emptyIcon={Gauge}
            emptyTitle="Nothing to compare"
            emptyHint="Reaching the ceiling needs both a target card and per-layer timings."
            reading="Each layer against the best its own intensity allows on this card. Unlike raw latency, this says whether a slow layer is slow because it is big or because it is inefficient."
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={efficiency} margin={CHART_MARGINS.line}>
                {CHART_GRID}
                <XA dataKey="i" />
                <YA unit=" %" domain={[0, 100]} width={44} />
                <Tooltip
                  {...chartTooltipStyle}
                  labelFormatter={(i) => efficiency[Number(i) - 1]?.name ?? `layer ${i}`}
                  formatter={(v: number) => `${v.toFixed(1)} % of ceiling`}
                />
                <Line
                  type="monotone"
                  dataKey="pct"
                  name="of ceiling"
                  stroke={SIMULATION_COLORS.teal}
                  strokeWidth={1.75}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </ChartSlot>

          <ChartSlot
            title="Time recoverable per layer"
            has={hasPoints && totalRecoverable > 0}
            emptyIcon={Timer}
            emptyTitle="Nothing recoverable"
            emptyHint="Every profiled layer is already at the ceiling its intensity allows, or there are no timings to compare."
            reading={
              <>
                Milliseconds each layer would give back if it ran at its own ceiling — the previous curve turned into
                the currency that matters. A layer at 20 % of a low ceiling is worth less than one at 60 % of a high
                one, and only this chart distinguishes them.
                {totalRecoverable > 0 ? ` Summed, ${totalRecoverable.toFixed(2)} ms per step.` : ''}
              </>
            }
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={recoverable} margin={CHART_MARGINS.line}>
                {CHART_GRID}
                <XA dataKey="i" />
                <YA unit=" ms" width={56} />
                <Tooltip
                  {...chartTooltipStyle}
                  labelFormatter={(i) => recoverable[Number(i) - 1]?.name ?? `layer ${i}`}
                  formatter={(v: number) => `${v.toFixed(3)} ms recoverable`}
                />
                <Line
                  type="monotone"
                  dataKey="ms"
                  name="recoverable"
                  stroke={SIMULATION_COLORS.amber}
                  strokeWidth={1.75}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </ChartSlot>
        </ChartGrid>

        {/*
          Parallelism and the remaining compute figures.

          `pipeline_stages`, `tensor_parallel_degree`, `communication_overhead`
          and the interconnect are all real report fields that no view carried,
          and they are scalars — a band, not four more charts. They belong here
          because they answer the same question the roofline does from the
          other side: what the design is limited by once it stops fitting on
          one device.
        */}
        <div className="rounded-[8px] border border-border/60 bg-muted/20 px-3 py-2">
          <div className="flex items-center gap-x-6 gap-y-2 flex-wrap">
            <span className="text-[9px] font-mono uppercase tracking-[0.14em] text-muted-foreground/70">
              Across devices
            </span>
            {[
              { label: 'Devices', value: analysis.gpuCount > 0 ? String(analysis.gpuCount) : null },
              { label: 'Data parallel', value: analysis.dataParallel ? String(analysis.dataParallel) : null },
              {
                label: 'Tensor parallel',
                value: analysis.tensorParallelDegree > 0 ? String(analysis.tensorParallelDegree) : null,
              },
              { label: 'Pipeline stages', value: analysis.pipelineStages > 0 ? String(analysis.pipelineStages) : null },
              {
                label: 'Comms overhead',
                value:
                  analysis.communicationOverhead > 0
                    ? `${(analysis.communicationOverhead * 100).toFixed(1)} %`
                    : null,
              },
              { label: 'Interconnect', value: analysis.interconnect || null },
              {
                label: 'Tensor cores',
                value:
                  analysis.tensorCoreUtilization !== undefined && analysis.tensorCoreUtilization > 0
                    ? `${analysis.tensorCoreUtilization.toFixed(0)} %`
                    : null,
              },
              {
                label: 'Samples/s',
                value: analysis.samplesPerS && analysis.samplesPerS > 0 ? analysis.samplesPerS.toFixed(1) : null,
              },
            ]
              .filter((item): item is { label: string; value: string } => item.value !== null)
              .map((item) => (
                <span key={item.label} className="flex items-baseline gap-1.5">
                  <span className="text-[10px] text-muted-foreground">{item.label}</span>
                  <span className="text-[12px] font-mono tabular-nums font-semibold text-foreground">{item.value}</span>
                </span>
              ))}
          </div>
        </div>

        <ViewNote icon={Target}>
          The ceiling is the card's published peak, which no real kernel reaches — a layer at 60 % of the roofline is
          doing well, not badly. What matters here is the gap between layers, not the absolute number.
        </ViewNote>
      </div>
    </ChartErrorBoundary>
  );
}
