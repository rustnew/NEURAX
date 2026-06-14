import { ArchitectureFamily } from '@/types/plugins.ts';
import { EnvironmentSettings } from '../layout/EnvironmentSettings.tsx';

interface AnalysisPanelProps {
  selectedArchitecture: ArchitectureFamily;
}

export function AnalysisPanel({ selectedArchitecture }: AnalysisPanelProps) {
  return (
    <div className="h-full bg-sidebar flex flex-col overflow-hidden">
      <div className="h-10 px-4 flex items-center border-b border-sidebar-border">
        <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Hyperparameters
        </span>
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-thin">
        <EnvironmentSettings family={selectedArchitecture} />
      </div>
    </div>
  );
}
