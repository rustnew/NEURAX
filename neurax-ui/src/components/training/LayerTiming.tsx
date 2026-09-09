/**
 * Per-layer predicted time against per-layer observed time.
 *
 * The compiler already produces `latency_per_layer`, `params_per_layer` and
 * `flops_per_layer` — they are what the Simulation workspace's Per Layer view
 * draws. Until a run exists those are three columns with nothing to check them
 * against. This view is the fourth column.
 *
 * It matters more than the aggregate on the Accuracy screen. A total step time
 * that lands within 5 % can hide two layers wrong in opposite directions, and
 * the layer that is wrong is the one to fix. An aggregate says the model of the
 * machine is roughly right; this says which operator it is wrong about.
 */
import { Layers } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

import {
  CHART_MARGINS, ChartCard, ChartContainer, ChartErrorBoundary, EmptyChartState,
  chartTooltipStyle,
} from '@/components/simulation/shared';
import { formatCount } from '@/services/format.ts';
import type { LayerComparison } from '@/types/runtime.ts';
import { cn } from '@/lib/utils.ts';

interface LayerTimingProps {
  layers: LayerComparison[];
}

export function LayerTiming({ layers }: LayerTimingProps) {
  const measured = layers.filter((l) => l.observedTimeMs !== undefined && l.predictedTimeMs !== undefined);

  if (measured.length === 0) {
    return (
      <EmptyChartState
        icon={Layers}
        title="No per-layer timings yet"
        description="Per-layer timings arrive once the run has profiled a step. Until then only the compiler's predictions exist."
      />
    );
  }

  const rows = measured.map((l) => ({
    name: l.type,
    predicted: Number((l.predictedTimeMs ?? 0).toFixed(2)),
    observed: Number((l.observedTimeMs ?? 0).toFixed(2)),
  }));

  const axis = { stroke: 'hsl(var(--muted-foreground))', fontSize: 10 };
  const totalPredicted = measured.reduce((s, l) => s + (l.predictedTimeMs ?? 0), 0);
  const totalObserved = measured.reduce((s, l) => s + (l.observedTimeMs ?? 0), 0);

  return (
    <ChartErrorBoundary>
      <div className="space-y-6">
        <ChartCard
          title="Time per layer — predicted vs observed"
          badge={{ text: 'compared', variant: 'live' }}
          size="wide"
          action={
            <div className="flex items-center gap-3">
              {[
                { label: 'predicted', color: 'hsl(var(--muted-foreground))' },
                { label: 'observed', color: 'hsl(var(--primary))' },
              ].map((entry) => (
                <span key={entry.label} className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
                  <span className="w-2 h-2 rounded-[2px]" style={{ background: entry.color }} />
                  {entry.label}
                </span>
              ))}
            </div>
          }
        >
          <ChartContainer>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={rows} margin={CHART_MARGINS.bar}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.4} />
                <XAxis dataKey="name" {...axis} interval={0} angle={-12} textAnchor="end" height={54} />
                <YAxis {...axis} unit=" ms" />
                <Tooltip {...chartTooltipStyle} />
                <Bar dataKey="predicted" fill="hsl(var(--muted-foreground))" fillOpacity={0.45} name="predicted" isAnimationActive={false} />
                <Bar dataKey="observed" fill="hsl(var(--primary))" name="observed" isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </ChartContainer>
        </ChartCard>

        <div className="overflow-x-auto border border-border rounded-lg">
          <table className="w-full text-[12px]">
            <thead>
              <tr className="bg-muted/40 text-muted-foreground text-[10px] uppercase tracking-wider">
                <th className="text-left px-3 py-2 font-medium">Layer</th>
                <th className="text-right px-3 py-2 font-medium">Parameters</th>
                <th className="text-right px-3 py-2 font-medium">FLOPs</th>
                <th className="text-right px-3 py-2 font-medium">Predicted</th>
                <th className="text-right px-3 py-2 font-medium">Observed</th>
                <th className="text-right px-3 py-2 font-medium">Error</th>
              </tr>
            </thead>
            <tbody>
              {measured.map((l) => {
                const errorPct = (((l.observedTimeMs ?? 0) - (l.predictedTimeMs ?? 0)) / (l.predictedTimeMs || 1)) * 100;
                return (
                  <tr key={l.id} className="border-t border-border/60">
                    <td className="px-3 py-2 font-mono text-foreground">{l.type}</td>
                    <td className="px-3 py-2 text-right font-mono tabular-nums text-muted-foreground">
                      {formatCount(l.predictedParams)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono tabular-nums text-muted-foreground">
                      {l.predictedFlops > 0 ? l.predictedFlops.toExponential(2) : '—'}
                    </td>
                    <td className="px-3 py-2 text-right font-mono tabular-nums text-muted-foreground">
                      {(l.predictedTimeMs ?? 0).toFixed(2)} ms
                    </td>
                    <td className="px-3 py-2 text-right font-mono tabular-nums font-semibold text-foreground">
                      {(l.observedTimeMs ?? 0).toFixed(2)} ms
                    </td>
                    <td
                      className={cn(
                        'px-3 py-2 text-right font-mono tabular-nums font-semibold',
                        Math.abs(errorPct) < 5
                          ? 'text-emerald-600 dark:text-emerald-400'
                          : Math.abs(errorPct) < 20
                            ? 'text-amber-600 dark:text-amber-400'
                            : 'text-red-600 dark:text-red-400',
                      )}
                    >
                      {errorPct >= 0 ? '+' : ''}
                      {errorPct.toFixed(1)} %
                    </td>
                  </tr>
                );
              })}
            </tbody>
            <tfoot>
              <tr className="border-t-2 border-border bg-muted/20 font-semibold">
                <td className="px-3 py-2 text-foreground">Sum of profiled layers</td>
                <td className="px-3 py-2" />
                <td className="px-3 py-2" />
                <td className="px-3 py-2 text-right font-mono tabular-nums text-muted-foreground">
                  {totalPredicted.toFixed(2)} ms
                </td>
                <td className="px-3 py-2 text-right font-mono tabular-nums text-foreground">
                  {totalObserved.toFixed(2)} ms
                </td>
                <td className="px-3 py-2" />
              </tr>
            </tfoot>
          </table>
        </div>

        <p className="text-[11px] text-muted-foreground leading-relaxed max-w-[90ch]">
          These rows cover the layers the profiler timed individually. Their sum is below the full step time — the
          optimiser, the loss and the transfers between layers are real cost that belongs to no single layer, and
          attributing it to one would make this table tidier and wrong.
        </p>
      </div>
    </ChartErrorBoundary>
  );
}
