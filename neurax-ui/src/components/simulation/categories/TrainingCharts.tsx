import { TrendingUp, Info } from 'lucide-react';
import { AnalysisResult, PerLayerBreakdownRow } from '@/types/architecture.ts';


interface TrainingChartsProps {
  analysis?: AnalysisResult;
  perLayer?: PerLayerBreakdownRow[];
}

const CARD_CLS = 'panel-section p-4 bg-card/30 border-primary/5 rounded-xl';

export function TrainingCharts({ analysis }: TrainingChartsProps) {
  if (!analysis || analysis.trainingTimeHours === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-muted-foreground border-2 border-dashed border-border rounded-lg">
        <TrendingUp className="w-8 h-8 mb-2 opacity-20" />
        <p className="text-sm">No training analysis available</p>
        <p className="text-xs">Run analysis to see training projections.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-primary" />
          Training — Cost & Runtime Summary
        </h2>
        <div className="flex items-center gap-2 text-[10px] text-muted-foreground bg-secondary/50 px-2 py-1 rounded-md">
          <Info className="w-3 h-3" />
          Only compiler-emitted training summary fields are shown live
        </div>
      </div>

      <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
        {[
          { label: 'Train Cost', value: `$${analysis.trainingCostUsd.toLocaleString(undefined, { maximumFractionDigits: 0 })}` },
          { label: 'Duration', value: `${analysis.trainingTimeHours.toFixed(1)} h` },
          { label: 'Energy', value: `${analysis.energyKwh.toFixed(1)} kWh` },
          { label: 'Footprint', value: `${analysis.co2Kg.toFixed(1)} kg CO2` },
        ].map((item) => (
          <div key={item.label} className={CARD_CLS}>
            <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{item.label}</div>
            <div className="mt-2 text-lg font-semibold font-mono">{item.value}</div>
          </div>
        ))}
      </div>


    </div>
  );
}
