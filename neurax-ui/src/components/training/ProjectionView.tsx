import { TrendingUp, DollarSign, Clock, Zap, Leaf } from 'lucide-react';
import { AnalysisResult } from '@/types/architecture.ts';
import {
  ChartCard,
  StatCard,
  ChartErrorBoundary,
  EmptyChartState,
} from '@/components/simulation/shared';

interface ProjectionViewProps {
  analysis?: AnalysisResult;
}

/**
 * Three charts used to live here — Cost Breakdown, Runtime Projection, and
 * a bar-chart version of Efficiency Metrics — and all three are gone now:
 *
 * - Cost Breakdown applied a fixed 70/15/10/5 compute/storage/network/other
 *   split to any total cost, regardless of the actual model or hardware.
 *   Nothing in the analysis says what the real split is; the split was
 *   invented and presented as if it were.
 * - Runtime Projection plotted cost ($), time (h) and energy (kWh) as three
 *   lines against 10 fake "epochs" — each line was `total * (i+1)/10` by
 *   construction, a perfectly straight ramp with no real per-epoch signal,
 *   duplicating the four stat cards above it with three extra incompatible
 *   units squeezed onto one shared axis.
 * - Efficiency Metrics plotted cost/token, energy/token, co2/token (small
 *   fractions) against GPU-hours (tens of thousands) on one bar chart —
 *   an 8-order-of-magnitude spread that made three of the four bars read
 *   as zero-height. Confirmed live: only the GPU Hours bar was visible.
 *
 * What's left is every real number those three charts touched, in the two
 * places that were already honest about them: the stat cards below, and
 * the value list this file now builds from the same fields Training
 * Details already used.
 */
export function ProjectionView({ analysis }: ProjectionViewProps) {
  if (!analysis || analysis.trainingTimeHours === 0) {
    return (
      <EmptyChartState
        icon={TrendingUp}
        title="No training analysis available"
        description="Run analysis to see training projections."
      />
    );
  }

  const efficiencyRows = [
    {
      label: 'Cost per token',
      value: analysis.costPerMillionTokensUsd > 0 ? `$${(analysis.costPerMillionTokensUsd / 1_000_000).toFixed(8)}` : null,
    },
    {
      label: 'Energy per token',
      value: analysis.energyKwh > 0 && analysis.totalParams > 0
        ? `${(analysis.energyKwh / (analysis.totalParams * 1000)).toExponential(2)} kWh`
        : null,
    },
    {
      label: 'CO2 per token',
      value: analysis.co2Kg > 0 && analysis.totalParams > 0
        ? `${(analysis.co2Kg / (analysis.totalParams * 1000)).toExponential(2)} kg`
        : null,
    },
    {
      label: 'GPU hours',
      value: (analysis.gpuHours ?? analysis.trainingTimeHours) > 0
        ? `${(analysis.gpuHours ?? analysis.trainingTimeHours).toFixed(1)} h`
        : null,
    },
  ].filter((row) => row.value !== null) as Array<{ label: string; value: string }>;

  return (
    <ChartErrorBoundary>
      <div className="space-y-6">
        {/* Summary Cards */}
        <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
          <StatCard
            icon={<DollarSign className="w-3 h-3" />}
            label="Train Cost"
            value={`$${analysis.trainingCostUsd.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
          />
          <StatCard
            icon={<Clock className="w-3 h-3" />}
            label="Duration"
            value={`${analysis.trainingTimeHours.toFixed(1)} h`}
          />
          <StatCard
            icon={<Zap className="w-3 h-3" />}
            label="Energy"
            value={`${analysis.energyKwh.toFixed(1)} kWh`}
          />
          <StatCard
            icon={<Leaf className="w-3 h-3" />}
            label="CO2"
            value={`${analysis.co2Kg.toFixed(1)} kg`}
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Efficiency — a list, not a chart. These four numbers span 8
              orders of magnitude (a fraction of a cent per token vs. tens
              of thousands of GPU-hours); no shared axis can show all four
              without three of them reading as zero. */}
          <ChartCard title="Efficiency" size="wide">
            {efficiencyRows.length > 0 ? (
              <div className="space-y-3">
                {efficiencyRows.map((row) => (
                  <div key={row.label} className="flex justify-between items-center text-xs">
                    <span className="text-muted-foreground">{row.label}</span>
                    <span className="font-mono font-semibold">{row.value}</span>
                  </div>
                ))}
              </div>
            ) : (
              <EmptyChartState icon={Zap} title="No efficiency data" description="Run analysis to see per-token cost and energy." />
            )}
          </ChartCard>

          {/* Training Details */}
          <ChartCard title="Training Details" size="wide">
            <div className="space-y-3">
              <div className="flex justify-between items-center text-xs">
                <span className="text-muted-foreground">Cost per 1M tokens</span>
                <span className="font-mono font-semibold">
                  ${analysis.costPerMillionTokensUsd.toFixed(4)}
                </span>
              </div>
              {(analysis.gpuHours ?? 0) > 0 && (
                <div className="flex justify-between items-center text-xs">
                  <span className="text-muted-foreground">GPU Hours</span>
                  <span className="font-mono font-semibold">
                    {analysis.gpuHours!.toFixed(1)} h
                  </span>
                </div>
              )}
              {analysis.provider && (
                <div className="flex justify-between items-center text-xs">
                  <span className="text-muted-foreground">Provider</span>
                  <span className="font-mono font-semibold">
                    {analysis.provider}
                  </span>
                </div>
              )}
              <div className="flex justify-between items-center text-xs">
                <span className="text-muted-foreground">Model Size</span>
                <span className="font-mono font-semibold">
                  {(analysis.totalParams / 1e9).toFixed(2)}B params
                </span>
              </div>
            </div>
          </ChartCard>
        </div>
      </div>
    </ChartErrorBoundary>
  );
}
