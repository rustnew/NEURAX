/**
 * NEURAX's predictions, beside what actually happened.
 *
 * This is the screen that only NEURAX can draw. Every other trainer shows you
 * a loss curve; none of them can show you what the model *was supposed to*
 * cost, because none of them computed it. Eleven IR phases exist to produce
 * the left-hand column here, and until now nothing has ever checked them
 * against a machine.
 *
 * The parameter count is separated from the rest and shown first, because it
 * is categorically different. Memory, timing and energy are estimates with
 * legitimate error bands — a few percent off is a good prediction. The
 * parameter count is arithmetic: it is either right or a formula is wrong.
 * It also arrives about a second into a run, before a single step, which
 * makes it the cheapest possible moment to discover a bad design. A mismatch
 * is a red banner, not a percentage.
 *
 * Error is signed and directional. `higherIsBetter` exists because
 * overshooting on throughput is good news and overshooting on VRAM is how a
 * run dies at step 4 000; colouring both the same way would be a lie of
 * presentation.
 */
import { CircleCheck, CircleAlert, Scale, TriangleAlert } from 'lucide-react';

import { EmptyChartState } from '@/components/simulation/shared';
import { formatBytes, formatCount } from '@/services/format.ts';
import type { ModelBuiltEvent, PredictionPair } from '@/types/runtime.ts';
import { cn } from '@/lib/utils.ts';

interface PredictionAccuracyProps {
  predictions: PredictionPair[];
  model: ModelBuiltEvent | null;
}

/** Percent bands. Below 5 % is a good analytical prediction; past 20 % the
 *  model of the machine is wrong in a way worth investigating, not a
 *  rounding difference. */
function band(errorPct: number, higherIsBetter: boolean | undefined): 'good' | 'fair' | 'poor' {
  const magnitude = Math.abs(errorPct);
  // A prediction that undershoots a "higher is better" metric — the run is
  // faster than promised — is not a problem, only a conservative estimate.
  const favourable = higherIsBetter ? errorPct > 0 : errorPct < 0;
  if (magnitude < 5) return 'good';
  if (magnitude < 20 || favourable) return 'fair';
  return 'poor';
}

const BAND_STYLE = {
  good: 'text-emerald-600 dark:text-emerald-400',
  fair: 'text-amber-600 dark:text-amber-400',
  poor: 'text-red-600 dark:text-red-400',
} as const;

function formatValue(value: number, unit: string): string {
  if (unit === 'B') return formatBytes(value);
  if (unit === '') return value >= 10_000 ? formatCount(Math.round(value)) : value.toFixed(1);
  if (unit === '%') return `${value.toFixed(0)} %`;
  return `${value.toFixed(2)} ${unit}`;
}

export function PredictionAccuracy({ predictions, model }: PredictionAccuracyProps) {
  const observedAny = predictions.some((p) => p.observed !== undefined);
  if (!observedAny) {
    return (
      <EmptyChartState
        icon={Scale}
        title="Nothing to compare yet"
        description="NEURAX's predictions are checked against the run as soon as it reports its first measurements."
      />
    );
  }

  const paramPair = predictions.find((p) => p.metric === 'parameters');
  const rest = predictions.filter((p) => p.metric !== 'parameters');
  const paramsMatch =
    paramPair?.observed !== undefined && Math.round(paramPair.observed) === Math.round(paramPair.predicted);

  return (
    <div className="space-y-6">
      {paramPair && paramPair.observed !== undefined ? (
        <div
          className={cn(
            'rounded-lg border p-4',
            paramsMatch
              ? 'border-emerald-500/30 bg-emerald-500/5'
              : 'border-red-500/40 bg-red-500/5',
          )}
        >
          <div className="flex items-center gap-2">
            {paramsMatch ? (
              <CircleCheck className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
            ) : (
              <TriangleAlert className="w-4 h-4 text-red-600 dark:text-red-400" />
            )}
            <span className="text-sm font-semibold text-foreground">
              {paramsMatch ? 'Parameter count confirmed' : 'Parameter count does not match'}
            </span>
          </div>
          <div className="mt-3 flex flex-wrap items-baseline gap-x-8 gap-y-2">
            <div>
              <div className="text-[9px] font-mono uppercase tracking-[0.12em] text-muted-foreground/70">
                NEURAX predicted
              </div>
              <div className="text-lg font-semibold font-mono tabular-nums text-foreground">
                {formatCount(paramPair.predicted)}
              </div>
            </div>
            <div>
              <div className="text-[9px] font-mono uppercase tracking-[0.12em] text-muted-foreground/70">
                PyTorch built
              </div>
              <div className="text-lg font-semibold font-mono tabular-nums text-foreground">
                {formatCount(paramPair.observed)}
              </div>
            </div>
            {model ? (
              <div>
                <div className="text-[9px] font-mono uppercase tracking-[0.12em] text-muted-foreground/70">On</div>
                <div className="text-[13px] font-mono text-muted-foreground">
                  {model.device} · torch {model.torchVersion} · {model.dtype}
                </div>
              </div>
            ) : null}
          </div>
          <p className="mt-2.5 text-[11px] text-muted-foreground leading-relaxed max-w-[80ch]">
            {paramsMatch
              ? 'Reported before the first step, at no GPU cost. The compiler counted this model exactly as PyTorch built it.'
              : 'A parameter count is arithmetic, not an estimate — this is a formula disagreeing with reality, and the run should be stopped and the design re-checked rather than trained.'}
          </p>
        </div>
      ) : null}

      <div>
        <div className="flex items-baseline justify-between mb-2">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Predicted against observed
          </h3>
          <span className="text-[10px] font-mono text-muted-foreground/60">
            error = (observed − predicted) / predicted
          </span>
        </div>
        <div className="overflow-x-auto border border-border rounded-lg">
          <table className="w-full text-[12px]">
            <thead>
              <tr className="bg-muted/40 text-muted-foreground text-[10px] uppercase tracking-wider">
                <th className="text-left px-3 py-2 font-medium">Metric</th>
                <th className="text-right px-3 py-2 font-medium">NEURAX predicted</th>
                <th className="text-right px-3 py-2 font-medium">Observed</th>
                <th className="text-right px-3 py-2 font-medium">Error</th>
                <th className="text-left px-3 py-2 font-medium w-[34%]">Reading</th>
              </tr>
            </thead>
            <tbody>
              {rest.map((p) => {
                if (p.observed === undefined) {
                  return (
                    <tr key={p.metric} className="border-t border-border/60">
                      <td className="px-3 py-2 text-foreground">{p.label}</td>
                      <td className="px-3 py-2 text-right font-mono tabular-nums text-foreground">
                        {formatValue(p.predicted, p.unit)}
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-muted-foreground/50">not yet</td>
                      <td className="px-3 py-2 text-right text-muted-foreground/50">—</td>
                      <td className="px-3 py-2 text-muted-foreground/50">waiting for the run</td>
                    </tr>
                  );
                }
                const errorPct = ((p.observed - p.predicted) / p.predicted) * 100;
                const tone = band(errorPct, p.higherIsBetter);
                const favourable = p.higherIsBetter ? errorPct > 0 : errorPct < 0;
                return (
                  <tr key={p.metric} className="border-t border-border/60">
                    <td className="px-3 py-2 text-foreground">{p.label}</td>
                    <td className="px-3 py-2 text-right font-mono tabular-nums text-muted-foreground">
                      {formatValue(p.predicted, p.unit)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono tabular-nums font-semibold text-foreground">
                      {formatValue(p.observed, p.unit)}
                    </td>
                    <td className={cn('px-3 py-2 text-right font-mono tabular-nums font-semibold', BAND_STYLE[tone])}>
                      {errorPct >= 0 ? '+' : ''}
                      {errorPct.toFixed(1)} %
                    </td>
                    <td className="px-3 py-2 text-[11px] text-muted-foreground leading-snug">
                      {tone === 'good'
                        ? 'Within the analytical band.'
                        : favourable
                          ? 'Better than predicted — the estimate was conservative.'
                          : Math.abs(errorPct) < 10
                            ? 'Slightly over, inside normal machine-to-machine variance.'
                            : 'Worse than predicted; worth looking at before scaling this design up.'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <p className="flex items-start gap-1.5 text-[11px] text-muted-foreground leading-relaxed max-w-[90ch]">
        <CircleAlert className="w-3.5 h-3.5 mt-px shrink-0 text-muted-foreground/60" />
        <span>
          Timing and energy figures are observed on this machine, under whatever else it was doing. They confirm or
          refute the model NEURAX has of this GPU; they are not a benchmark of the GPU itself.
        </span>
      </p>
    </div>
  );
}
