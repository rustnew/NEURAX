import { GitCompare, Info } from 'lucide-react';
import { AnalysisResult } from '@/types/architecture.ts';


interface ComparisonChartsProps {
  analysis?: AnalysisResult;
}

const CARD_CLS = 'panel-section p-4 bg-card/30 border-primary/5 rounded-xl';

export function ComparisonCharts({ analysis }: ComparisonChartsProps) {
  if (!analysis || analysis.totalParams === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-muted-foreground border-2 border-dashed border-border rounded-lg">
        <GitCompare className="w-8 h-8 mb-2 opacity-20" />
        <p className="text-sm">No comparison data available</p>
        <p className="text-xs">Run analysis to compare deployment trade-offs.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <GitCompare className="w-5 h-5 text-primary" />
          Comparison — Benchmarks & Variants
        </h2>
        <div className="flex items-center gap-2 text-[10px] text-muted-foreground bg-secondary/50 px-2 py-1 rounded-md">
          <Info className="w-3 h-3" />
          Reserved for true cross-hardware, cross-precision, and variant report outputs
        </div>
      </div>

      <div className={`${CARD_CLS} grid grid-cols-1 gap-4 md:grid-cols-3`}>
        <div className="rounded-lg border border-border/60 bg-secondary/20 px-4 py-3">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Current device</div>
          <div className="mt-1 text-sm font-semibold">{analysis.gpuName || 'Unknown GPU'}</div>
        </div>
        <div className="rounded-lg border border-border/60 bg-secondary/20 px-4 py-3">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Current precision</div>
          <div className="mt-1 text-sm font-semibold uppercase">{analysis.selectedPrecision || 'N/A'}</div>
        </div>
        <div className="rounded-lg border border-border/60 bg-secondary/20 px-4 py-3">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Batch size</div>
          <div className="mt-1 text-sm font-semibold">{analysis.selectedBatchSize ?? 'N/A'}</div>
        </div>
      </div>


    </div>
  );
}
