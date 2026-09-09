/**
 * What the run is doing, right now.
 *
 * Every number on this screen comes from the training process itself — no
 * formula, no estimate. That is the whole point of the workspace, and it is
 * why the charts here are deliberately plain: there is nothing to model, only
 * something to watch.
 *
 * The one editorial choice is `Step time`, which is drawn as two stacked
 * bands rather than one line. Total step time alone cannot tell you whether
 * the GPU is working or waiting, and waiting on the dataloader is the single
 * most common reason a run is slower than NEURAX predicted. Splitting the
 * band makes a data-bound run obvious at a glance instead of hiding it inside
 * an average — the same reason `TrainingStep` carries `dataTimeMs` at all.
 *
 * The series are downsampled before they are drawn. At 12 000 steps every
 * chart would otherwise hand recharts more points than the card has pixels,
 * which costs frames during a live run and shows nothing extra.
 */
import { Activity, Cpu, Gauge, HardDrive, Thermometer, Timer, TrendingDown } from 'lucide-react';
import {
  Area, AreaChart, CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip,
  XAxis, YAxis,
} from 'recharts';

import {
  CHART_MARGINS, ChartCard, ChartContainer, ChartErrorBoundary, EmptyChartState, StatCard,
  chartTooltipStyle,
} from '@/components/simulation/shared';
import { formatBytes } from '@/services/format.ts';
import type { TrainingStep } from '@/types/runtime.ts';

interface LiveMetricsProps {
  steps: TrainingStep[];
  /** Optimizer steps in one pass over the training split. Drives the epoch
   *  markers; absent for a stream, where there are no epochs to mark. */
  stepsPerEpoch?: number;
}

const GB = 1024 ** 3;
/** Roughly one point per horizontal pixel of a card. More is wasted work. */
const MAX_POINTS = 400;

/** Keep every nth step so the curve keeps its shape — averaging windows would
 *  smooth away the noise and loss spikes that are the reason to look. */
function downsample<T>(rows: T[], max = MAX_POINTS): T[] {
  if (rows.length <= max) return rows;
  const stride = Math.ceil(rows.length / max);
  const out = rows.filter((_, i) => i % stride === 0);
  const last = rows[rows.length - 1];
  if (out[out.length - 1] !== last) out.push(last);
  return out;
}

function mean(values: number[]): number {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
}

export function LiveMetrics({ steps, stepsPerEpoch }: LiveMetricsProps) {
  if (steps.length === 0) {
    return (
      <EmptyChartState
        icon={Activity}
        title="No steps yet"
        description="Metrics appear as soon as the training process reports its first step."
      />
    );
  }

  const latest = steps[steps.length - 1];
  // The last 100 steps, not the whole run: an average over a run that started
  // slow keeps reporting a throughput the run no longer has.
  const recent = steps.slice(-100);
  const avgStepTime = mean(recent.map((s) => s.stepTimeMs));
  const avgDataTime = mean(recent.map((s) => s.dataTimeMs));
  const dataShare = avgStepTime > 0 ? (avgDataTime / avgStepTime) * 100 : 0;
  const bestLoss = Math.min(...steps.map((s) => s.loss));
  const latestEpoch = latest.epoch ?? (stepsPerEpoch ? Math.floor((latest.step - 1) / stepsPerEpoch) : undefined);
  // Boundaries already crossed, not the whole planned schedule: a marker for a
  // pass that has not happened yet would be a prediction on a chart whose
  // entire premise is that it only shows what was observed.
  const epochBoundaries =
    stepsPerEpoch && stepsPerEpoch > 0
      ? Array.from({ length: Math.floor(latest.step / stepsPerEpoch) }, (_, i) => (i + 1) * stepsPerEpoch)
      : [];

  const series = downsample(steps).map((s) => ({
    step: s.step,
    loss: Number(s.loss.toFixed(4)),
    valLoss: s.valLoss !== undefined ? Number(s.valLoss.toFixed(4)) : undefined,
    compute: Number((s.stepTimeMs - s.dataTimeMs).toFixed(2)),
    data: Number(s.dataTimeMs.toFixed(2)),
    samplesPerSec: Math.round(s.samplesPerSec),
    allocated: Number((s.vramAllocatedBytes / GB).toFixed(3)),
    reserved: Number((s.vramReservedBytes / GB).toFixed(3)),
    lr: s.learningRate,
    gradNorm: s.gradNorm !== undefined ? Number(s.gradNorm.toFixed(3)) : undefined,
    util: s.gpuUtilisationPct !== undefined ? Math.round(s.gpuUtilisationPct) : undefined,
    temp: s.temperatureC !== undefined ? Math.round(s.temperatureC) : undefined,
    power: s.powerWatts !== undefined ? Math.round(s.powerWatts) : undefined,
  }));

  const axis = { stroke: 'hsl(var(--muted-foreground))', fontSize: 10 };
  const grid = <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.4} />;

  return (
    <ChartErrorBoundary>
      <div className="space-y-6">
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          <StatCard
            icon={<TrendingDown className="w-3.5 h-3.5" />}
            label="Loss"
            value={latest.loss.toFixed(4)}
            sublabel={
              latestEpoch !== undefined ? `epoch ${latestEpoch + 1} · best ${bestLoss.toFixed(4)}` : `best ${bestLoss.toFixed(4)}`
            }
          />
          <StatCard
            icon={<Timer className="w-3.5 h-3.5" />}
            label="Step time"
            value={`${avgStepTime.toFixed(1)} ms`}
            sublabel="mean of last 100"
          />
          <StatCard
            icon={<Activity className="w-3.5 h-3.5" />}
            label="Throughput"
            value={`${Math.round(latest.samplesPerSec)}/s`}
            sublabel="samples"
          />
          <StatCard
            icon={<HardDrive className="w-3.5 h-3.5" />}
            label="VRAM"
            value={formatBytes(latest.vramAllocatedBytes)}
            sublabel={`${formatBytes(latest.vramReservedBytes)} reserved`}
          />
          <StatCard
            icon={<Gauge className="w-3.5 h-3.5" />}
            label="Waiting on data"
            value={`${dataShare.toFixed(0)}%`}
            sublabel="of each step"
            variant={dataShare > 25 ? 'warning' : 'default'}
          />
          <StatCard
            icon={<Thermometer className="w-3.5 h-3.5" />}
            label="GPU"
            value={latest.temperatureC !== undefined ? `${Math.round(latest.temperatureC)} °C` : '—'}
            sublabel={
              latest.gpuUtilisationPct !== undefined ? `${Math.round(latest.gpuUtilisationPct)}% · ${Math.round(latest.powerWatts ?? 0)} W` : undefined
            }
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <ChartCard title="Loss" badge={{ text: 'observed', variant: 'live' }} size="wide">
            <ChartContainer>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={series} margin={CHART_MARGINS.line}>
                  {grid}
                  <XAxis dataKey="step" {...axis} />
                  <YAxis {...axis} domain={['auto', 'auto']} />
                  <Tooltip {...chartTooltipStyle} />
                  {epochBoundaries.map((at, i) => (
                    <ReferenceLine
                      key={at}
                      x={at}
                      stroke="hsl(var(--muted-foreground))"
                      strokeDasharray="2 3"
                      strokeOpacity={0.5}
                      label={{
                        value: `epoch ${i + 2}`,
                        position: 'insideTopRight',
                        fontSize: 9,
                        fill: 'hsl(var(--muted-foreground))',
                      }}
                    />
                  ))}
                  <Line type="monotone" dataKey="loss" stroke="hsl(var(--primary))" strokeWidth={1.5} dot={false} isAnimationActive={false} name="train" />
                  <Line
                    type="monotone"
                    dataKey="valLoss"
                    stroke="hsl(var(--warning, 38 92% 50%))"
                    strokeWidth={0}
                    dot={{ r: 3 }}
                    connectNulls
                    isAnimationActive={false}
                    name="validation"
                  />
                </LineChart>
              </ResponsiveContainer>
            </ChartContainer>
          </ChartCard>

          <ChartCard title="Step time — compute vs waiting on data" badge={{ text: 'observed', variant: 'live' }} size="wide">
            <ChartContainer>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={series} margin={CHART_MARGINS.area}>
                  {grid}
                  <XAxis dataKey="step" {...axis} />
                  <YAxis {...axis} unit=" ms" />
                  <Tooltip {...chartTooltipStyle} />
                  <Area type="monotone" dataKey="compute" stackId="t" stroke="hsl(var(--primary))" fill="hsl(var(--primary))" fillOpacity={0.35} isAnimationActive={false} name="compute" />
                  <Area type="monotone" dataKey="data" stackId="t" stroke="hsl(var(--destructive))" fill="hsl(var(--destructive))" fillOpacity={0.35} isAnimationActive={false} name="waiting on data" />
                </AreaChart>
              </ResponsiveContainer>
            </ChartContainer>
          </ChartCard>

          <ChartCard title="VRAM — allocated vs reserved" badge={{ text: 'observed', variant: 'live' }} size="wide">
            <ChartContainer>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={series} margin={CHART_MARGINS.area}>
                  {grid}
                  <XAxis dataKey="step" {...axis} />
                  <YAxis {...axis} unit=" GB" domain={['auto', 'auto']} />
                  <Tooltip {...chartTooltipStyle} />
                  <Area type="monotone" dataKey="reserved" stroke="hsl(var(--muted-foreground))" fill="hsl(var(--muted-foreground))" fillOpacity={0.15} isAnimationActive={false} name="reserved" />
                  <Area type="monotone" dataKey="allocated" stroke="hsl(var(--primary))" fill="hsl(var(--primary))" fillOpacity={0.3} isAnimationActive={false} name="allocated" />
                </AreaChart>
              </ResponsiveContainer>
            </ChartContainer>
          </ChartCard>

          <ChartCard title="Throughput" badge={{ text: 'observed', variant: 'live' }} size="wide">
            <ChartContainer>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={series} margin={CHART_MARGINS.line}>
                  {grid}
                  <XAxis dataKey="step" {...axis} />
                  <YAxis {...axis} unit="/s" />
                  <Tooltip {...chartTooltipStyle} />
                  <Line type="monotone" dataKey="samplesPerSec" stroke="hsl(var(--primary))" strokeWidth={1.5} dot={false} isAnimationActive={false} name="samples/s" />
                </LineChart>
              </ResponsiveContainer>
            </ChartContainer>
          </ChartCard>

          <ChartCard title="Learning rate" badge={{ text: 'observed', variant: 'live' }} size="wide">
            <ChartContainer>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={series} margin={CHART_MARGINS.line}>
                  {grid}
                  <XAxis dataKey="step" {...axis} />
                  <YAxis {...axis} tickFormatter={(v: number) => v.toExponential(0)} width={52} />
                  <Tooltip {...chartTooltipStyle} formatter={(v: number) => v.toExponential(2)} />
                  <Line type="monotone" dataKey="lr" stroke="hsl(var(--primary))" strokeWidth={1.5} dot={false} isAnimationActive={false} name="lr" />
                </LineChart>
              </ResponsiveContainer>
            </ChartContainer>
          </ChartCard>

          <ChartCard title="Gradient norm" badge={{ text: 'observed', variant: 'live' }} size="wide">
            <ChartContainer>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={series} margin={CHART_MARGINS.line}>
                  {grid}
                  <XAxis dataKey="step" {...axis} />
                  <YAxis {...axis} domain={['auto', 'auto']} />
                  <Tooltip {...chartTooltipStyle} />
                  <Line type="monotone" dataKey="gradNorm" stroke="hsl(var(--primary))" strokeWidth={1.5} dot={false} connectNulls isAnimationActive={false} name="‖g‖" />
                </LineChart>
              </ResponsiveContainer>
            </ChartContainer>
          </ChartCard>

          <ChartCard title="GPU utilisation, temperature and power" badge={{ text: 'observed', variant: 'live' }} size="wide" className="lg:col-span-2">
            <ChartContainer>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={series} margin={CHART_MARGINS.line}>
                  {grid}
                  <XAxis dataKey="step" {...axis} />
                  <YAxis yAxisId="pct" {...axis} domain={[0, 100]} width={40} />
                  <YAxis yAxisId="w" orientation="right" {...axis} unit=" W" width={52} />
                  <Tooltip {...chartTooltipStyle} />
                  <Line yAxisId="pct" type="monotone" dataKey="util" stroke="hsl(var(--primary))" strokeWidth={1.5} dot={false} isAnimationActive={false} name="utilisation %" />
                  <Line yAxisId="pct" type="monotone" dataKey="temp" stroke="hsl(var(--destructive))" strokeWidth={1.5} dot={false} isAnimationActive={false} name="temperature °C" />
                  <Line yAxisId="w" type="monotone" dataKey="power" stroke="hsl(var(--muted-foreground))" strokeWidth={1.5} dot={false} isAnimationActive={false} name="power W" />
                </LineChart>
              </ResponsiveContainer>
            </ChartContainer>
          </ChartCard>
        </div>

        <p className="flex items-start gap-1.5 text-[11px] text-muted-foreground leading-relaxed">
          <Cpu className="w-3.5 h-3.5 mt-px shrink-0 text-muted-foreground/60" />
          <span>
            Everything above is reported by the training process. Nothing on this screen is a NEURAX estimate — the
            estimates, and how far off they turned out to be, are in <span className="font-medium text-foreground">Accuracy</span>.
            {stepsPerEpoch ? (
              <> Dashed lines mark each completed pass over the training data, which is where validation runs.</>
            ) : null}
          </span>
        </p>
      </div>
    </ChartErrorBoundary>
  );
}
