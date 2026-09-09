/**
 * Memory — whether this design fits, and what to change if it does not.
 *
 * Two of the eight charts here had no data behind them and never could: the
 * Memory Heatmap and VRAM Liveness read `memory_heatmap` and
 * `memory_liveness`, and no producer for either exists anywhere in the
 * compiler — not in Rust, not in the Python service. The frontend defaulted
 * both to `[]`, so the cards rendered their empty state on every analysis that
 * has ever run. They are removed rather than fixed, because there is nothing
 * to fix: the series was never emitted.
 *
 * What remains is built only on figures the report actually carries, and each
 * one answers a question that changes a decision:
 *
 *  - **Where the budget runs out** — the four memory terms accumulated across
 *    depth against the card's capacity. A total that overflows tells you it
 *    does not fit; this tells you at which layer, which is what says whether
 *    to shorten the stack or narrow it.
 *  - **What precision buys** — the weight budget across dtype widths. This is
 *    the one lever that moves memory by a factor rather than a percentage, and
 *    the curve is ordered by width, so it reads as the trade it is.
 *  - **What context costs** — the KV cache against sequence length. This chart
 *    was blank until this session: the compiler computes it from
 *    `embedding_dim`, and the client only ever sent the same number under the
 *    name `hidden_size`, so the width arrived as zero and the series came back
 *    empty on every model.
 *  - **What headroom is left** — the peak against capacity, with fragmentation
 *    charged as the real overhead it is rather than a footnote.
 */
import { AlertTriangle, HardDrive, Layers, Scaling, Shrink } from 'lucide-react';
import { Area, AreaChart, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip } from 'recharts';

import { AnalysisResult, PerLayerBreakdownRow } from '@/types/architecture.ts';
import {
  CHART_GRID, CHART_MARGINS, ChartErrorBoundary, ChartGrid, ChartSlot, StatCard, StatStrip,
  ViewNote, XA, YA, chartTooltipStyle,
} from '../shared';
import {
  SIMULATION_COLORS, buildDerivedLayerMetrics, formatBytes, hasAnalysisReportData,
} from '../simulationData.ts';

interface MemoryChartsProps {
  analysis?: AnalysisResult;
  perLayer?: PerLayerBreakdownRow[];
}

const GB = 1024 ** 3;

/**
 * Bytes per parameter at each precision NEURAX can analyse — the exact values
 * `neurax-formulas::dtype_bytes` uses, so this matches what a full analysis at
 * that precision would report rather than a separate estimate beside it.
 *
 * INT4 is listed at 1 byte/parameter, the same as INT8, and deliberately: the
 * compiler does not model sub-byte packing for weights (two 4-bit values
 * sharing one byte), so this shows the same conservative number the rest of
 * the app would if INT4 were selected — not the ~0.5 bytes/parameter true
 * 4-bit packing achieves. Real quantized runtimes (llama.cpp, bitsandbytes) do
 * pack INT4, so this row is an upper bound, not GGUF-file-size parity.
 */
const PRECISION_BYTES: Array<{ id: string; label: string; bytes: number }> = [
  { id: 'fp32', label: 'FP32', bytes: 4 },
  { id: 'fp16', label: 'FP16', bytes: 2 },
  { id: 'bf16', label: 'BF16', bytes: 2 },
  { id: 'int8', label: 'INT8', bytes: 1 },
  { id: 'int4', label: 'INT4', bytes: 1 },
];

/** Weight memory at every precision NEURAX supports, for a real parameter
 *  count — kept out of the component so it stays testable without rendering a
 *  chart, and so the byte table cannot drift from the compiler's unnoticed. */
export function computePrecisionMemory(
  totalParams: number,
): Array<{ id: string; label: string; bytes: number }> {
  if (totalParams <= 0) return [];
  return PRECISION_BYTES.map(({ id, label, bytes }) => ({ id, label, bytes: totalParams * bytes }));
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

export function MemoryCharts({ analysis, perLayer = [] }: MemoryChartsProps) {
  if (!hasAnalysisReportData(analysis)) {
    return (
      <ChartErrorBoundary>
        <div className="flex flex-col items-center justify-center gap-2 py-24 text-center">
          <HardDrive className="w-8 h-8 text-muted-foreground/30" />
          <div className="text-[13px] font-semibold text-foreground">No memory analysis yet</div>
          <p className="text-[11px] text-muted-foreground max-w-[52ch] leading-relaxed">
            Run the analysis. Peak VRAM, its four terms and the headroom against your card are all computed from the
            graph — no run required.
          </p>
        </div>
      </ChartErrorBoundary>
    );
  }

  const capacity = analysis.gpuMemoryGb > 0 ? analysis.gpuMemoryGb * GB : 0;
  const fits = capacity > 0 && analysis.peakVramBytes <= capacity;
  const headroom = capacity - analysis.peakVramBytes;

  // ── The budget, accumulated in graph order ───────────────────────────────
  const layers = buildDerivedLayerMetrics(analysis, perLayer);
  let acc = { weights: 0, activations: 0, gradients: 0, optimizer: 0 };
  const cumulative = layers.map((layer, i) => {
    acc = {
      weights: acc.weights + layer.weightsMb,
      activations: acc.activations + layer.activationsMb,
      gradients: acc.gradients + layer.gradientsMb,
      optimizer: acc.optimizer + layer.optimizerMb,
    };
    return {
      i: i + 1,
      name: layer.name,
      weights: Number((acc.weights / 1024).toFixed(3)),
      activations: Number((acc.activations / 1024).toFixed(3)),
      gradients: Number((acc.gradients / 1024).toFixed(3)),
      optimizer: Number((acc.optimizer / 1024).toFixed(3)),
    };
  });
  const hasDepth = cumulative.length > 1;

  // ── What each precision costs in weights ─────────────────────────────────
  const precisionCurve = computePrecisionMemory(analysis.totalParams).map((row) => ({
    label: row.label,
    gb: Number((row.bytes / GB).toFixed(3)),
  }));
  const currentPrecision = analysis.selectedPrecision;

  // ── Context growth ───────────────────────────────────────────────────────
  const kv = (analysis.kv_cache_scaling ?? analysis.live_trace?.kv_cache_scaling ?? []).map((p) => ({
    seq: p.seq,
    gb: Number((p.value / GB).toFixed(3)),
  }));

  // ── Headroom as batch size grows ─────────────────────────────────────────
  // Only the activation term scales with the batch; weights, gradients and
  // optimizer state do not. That is the compiler's own model, applied to the
  // batch actually analysed — an extrapolation of one term, not an invented
  // series, and it stops at the batch the compiler says is the last that fits.
  const batchNow = analysis.selectedBatchSize ?? 0;
  const fixedBytes =
    analysis.parameterMemoryBytes + analysis.gradientMemoryBytes + analysis.optimizerStateBytes;
  const perSampleBytes = batchNow > 0 ? analysis.activationMemoryBytes / batchNow : 0;
  const maxBatch = analysis.maxBatchSizeFit > 0 ? analysis.maxBatchSizeFit : 0;
  const batchCurve =
    batchNow > 0 && perSampleBytes > 0 && capacity > 0
      ? Array.from({ length: 12 }, (_, i) => {
          const batch = Math.max(1, Math.round(((i + 1) / 12) * Math.max(maxBatch, batchNow * 2)));
          return { batch, gb: Number(((fixedBytes + perSampleBytes * batch) / GB).toFixed(3)) };
        })
      : [];

  return (
    <ChartErrorBoundary>
      <div className="space-y-6">
        <StatStrip>
          <StatCard
            icon={<HardDrive className="w-3.5 h-3.5" />}
            label="Peak VRAM"
            value={formatBytes(analysis.peakVramBytes)}
            sublabel={capacity > 0 ? `of ${analysis.gpuMemoryGb} GB` : undefined}
            variant={capacity > 0 ? (fits ? 'success' : 'danger') : 'default'}
          />
          <StatCard
            icon={<Shrink className="w-3.5 h-3.5" />}
            label="Headroom"
            value={capacity > 0 ? formatBytes(Math.abs(headroom)) : '—'}
            sublabel={capacity > 0 ? (fits ? 'spare' : 'over budget') : undefined}
            variant={capacity > 0 ? (fits ? 'success' : 'danger') : 'default'}
          />
          <StatCard label="Weights" value={formatBytes(analysis.parameterMemoryBytes)} />
          <StatCard label="Activations" value={formatBytes(analysis.activationMemoryBytes)} />
          <StatCard label="Optimizer" value={formatBytes(analysis.optimizerStateBytes)} />
          <StatCard
            icon={<AlertTriangle className="w-3.5 h-3.5" />}
            label="Fragmentation"
            value={
              analysis.memoryFragmentationPct !== undefined
                ? `${analysis.memoryFragmentationPct.toFixed(0)} %`
                : '—'
            }
            sublabel={analysis.oomRisk ? `OOM risk ${analysis.oomRisk}` : undefined}
            variant={(analysis.memoryFragmentationPct ?? 0) > 15 ? 'warning' : 'default'}
          />
        </StatStrip>

        <ChartGrid>
          <ChartSlot
            title="Where the budget runs out"
            has={hasDepth}
            emptyIcon={Layers}
            emptyTitle="No per-layer memory"
            emptyHint="The analysis produced no memory terms per layer, so the budget cannot be accumulated across depth."
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
            reading={
              capacity > 0
                ? `The four terms accumulated layer by layer. Where the curve crosses the ${analysis.gpuMemoryGb} GB line is the depth this card stops supporting.`
                : 'The four terms accumulated layer by layer. Select a target card to see where its capacity is crossed.'
            }
          >
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={cumulative} margin={CHART_MARGINS.area}>
                {CHART_GRID}
                <XA dataKey="i" />
                <YA unit=" GB" width={56} />
                <Tooltip
                  {...chartTooltipStyle}
                  labelFormatter={(i) => cumulative[Number(i) - 1]?.name ?? `layer ${i}`}
                  formatter={(v: number) => `${v.toFixed(2)} GB`}
                />
                {[
                  { key: 'weights', color: SIMULATION_COLORS.blue },
                  { key: 'activations', color: SIMULATION_COLORS.amber },
                  { key: 'gradients', color: SIMULATION_COLORS.violet },
                  { key: 'optimizer', color: SIMULATION_COLORS.green },
                ].map((t) => (
                  <Area
                    key={t.key}
                    type="monotone"
                    dataKey={t.key}
                    stackId="budget"
                    stroke={t.color}
                    fill={t.color}
                    fillOpacity={0.26}
                    strokeWidth={1.25}
                    isAnimationActive={false}
                  />
                ))}
                {capacity > 0 ? (
                  <ReferenceLine
                    y={analysis.gpuMemoryGb}
                    stroke={SIMULATION_COLORS.red}
                    strokeDasharray="4 3"
                    label={{
                      value: `${analysis.gpuName || 'card'} · ${analysis.gpuMemoryGb} GB`,
                      position: 'insideTopRight',
                      fontSize: 9,
                      fill: 'hsl(var(--muted-foreground))',
                    }}
                  />
                ) : null}
              </AreaChart>
            </ResponsiveContainer>
          </ChartSlot>

          <ChartSlot
            title="What precision buys"
            has={precisionCurve.length > 0}
            emptyIcon={Scaling}
            emptyTitle="No parameter count"
            emptyHint="The weight budget needs a parameter count the analysis did not produce."
            reading={
              <>
                Weight memory at each precision the compiler supports, widest first — the one lever that moves memory
                by a factor rather than a percentage. INT4 is charged at one byte per parameter like INT8, because the
                compiler does not model sub-byte packing: treat it as an upper bound, not as a GGUF file size.
                {currentPrecision ? ` This design is analysed at ${currentPrecision}.` : ''}
              </>
            }
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={precisionCurve} margin={CHART_MARGINS.line}>
                {CHART_GRID}
                <XA dataKey="label" />
                <YA unit=" GB" width={56} />
                <Tooltip {...chartTooltipStyle} formatter={(v: number) => `${v.toFixed(2)} GB`} />
                {capacity > 0 ? (
                  <ReferenceLine
                    y={analysis.gpuMemoryGb}
                    stroke={SIMULATION_COLORS.red}
                    strokeDasharray="4 3"
                    label={{
                      value: 'card capacity',
                      position: 'insideTopRight',
                      fontSize: 9,
                      fill: 'hsl(var(--muted-foreground))',
                    }}
                  />
                ) : null}
                <Line
                  type="monotone"
                  dataKey="gb"
                  name="weights"
                  stroke={SIMULATION_COLORS.blue}
                  strokeWidth={1.75}
                  dot={{ r: 3 }}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </ChartSlot>

          <ChartSlot
            title="KV cache against context length"
            has={kv.length > 1}
            emptyIcon={Layers}
            emptyTitle="No KV cache to grow"
            emptyHint="This design keeps no attention cache — the series exists only for models that do."
            reading="The cache grows linearly with context and is charged per sequence in the batch. It is the term that turns a model that fits at 2k tokens into one that does not at 32k."
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={kv} margin={CHART_MARGINS.line}>
                {CHART_GRID}
                <XA dataKey="seq" />
                <YA unit=" GB" width={56} />
                <Tooltip
                  {...chartTooltipStyle}
                  labelFormatter={(v) => `${Number(v).toLocaleString('en-US')} tokens`}
                  formatter={(v: number) => `${v.toFixed(2)} GB`}
                />
                <Line
                  type="monotone"
                  dataKey="gb"
                  name="KV cache"
                  stroke={SIMULATION_COLORS.pink}
                  strokeWidth={1.75}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </ChartSlot>

          <ChartSlot
            title="Peak VRAM as the batch grows"
            has={batchCurve.length > 1}
            emptyIcon={Scaling}
            emptyTitle="No batch to scale from"
            emptyHint="This needs an analysed batch size and an activation figure to scale; one of the two is missing."
            reading={
              <>
                Only activations scale with the batch — weights, gradients and optimizer state do not — so the curve is
                a line with a floor, and the floor is what a smaller batch can never recover.
                {maxBatch > 0 ? ` The compiler puts the largest batch that fits at ${maxBatch}.` : ''}
              </>
            }
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={batchCurve} margin={CHART_MARGINS.line}>
                {CHART_GRID}
                <XA dataKey="batch" />
                <YA unit=" GB" width={56} />
                <Tooltip
                  {...chartTooltipStyle}
                  labelFormatter={(v) => `batch ${v}`}
                  formatter={(v: number) => `${v.toFixed(2)} GB`}
                />
                {capacity > 0 ? (
                  <ReferenceLine
                    y={analysis.gpuMemoryGb}
                    stroke={SIMULATION_COLORS.red}
                    strokeDasharray="4 3"
                    label={{
                      value: 'capacity',
                      position: 'insideTopRight',
                      fontSize: 9,
                      fill: 'hsl(var(--muted-foreground))',
                    }}
                  />
                ) : null}
                {batchNow > 0 ? (
                  <ReferenceLine
                    x={batchNow}
                    stroke="hsl(var(--muted-foreground))"
                    strokeDasharray="2 3"
                    label={{
                      value: 'analysed',
                      position: 'insideTopLeft',
                      fontSize: 9,
                      fill: 'hsl(var(--muted-foreground))',
                    }}
                  />
                ) : null}
                <Line
                  type="monotone"
                  dataKey="gb"
                  name="peak VRAM"
                  stroke={SIMULATION_COLORS.teal}
                  strokeWidth={1.75}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </ChartSlot>
        </ChartGrid>

        <ViewNote icon={HardDrive}>
          Peak VRAM is one instant, not an average, and it is the instant that decides whether a run starts at all.
          Every curve here is computed from the graph — what a real run allocates, beside these figures, is in{' '}
          <span className="font-medium text-foreground">Training</span>.
        </ViewNote>
      </div>
    </ChartErrorBoundary>
  );
}
