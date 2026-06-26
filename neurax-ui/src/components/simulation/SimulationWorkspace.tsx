import { useState } from 'react';
import { RealTimeCharts } from './categories/RealTimeCharts.tsx';
import { GlobalResultsCharts } from './categories/GlobalResultsCharts.tsx';
import { PerLayerCharts } from './categories/PerLayerCharts.tsx';
import { MemoryCharts } from './categories/MemoryCharts.tsx';
import { ComparisonCharts } from './categories/ComparisonCharts.tsx';
import { OptimizationCharts } from './categories/OptimizationCharts.tsx';
import { TrainingCharts } from './categories/TrainingCharts.tsx';
import { DebuggingCharts } from './categories/DebuggingCharts.tsx';
import {
  Activity, BarChart3, Layers, HardDrive,
  GitCompare, Target, TrendingUp, Bug
} from 'lucide-react';
import { CanvasNode, Connection, AnalysisResult, PerLayerBreakdownRow, Warning } from '@/types/architecture.ts';

const CATEGORIES = [
  { id: 'realtime', label: 'Real-Time', icon: Activity, count: 6 },
  { id: 'global', label: 'Results', icon: BarChart3, count: 8 },
  { id: 'perlayer', label: 'Per Layer', icon: Layers, count: 7 },
  { id: 'memory', label: 'Memory', icon: HardDrive, count: 6 },
  { id: 'comparison', label: 'Comparison', icon: GitCompare, count: 9 },
  { id: 'optimization', label: 'Optimization', icon: Target, count: 5 },
  { id: 'training', label: 'Training', icon: TrendingUp, count: 6 },
  { id: 'debugging', label: 'Debugging', icon: Bug, count: 8 },
] as const;

interface SimulationWorkspaceProps {
  nodes: CanvasNode[];
  connections: Connection[];
  analysis?: AnalysisResult;
  perLayer?: PerLayerBreakdownRow[];
  warnings?: Warning[];
  topology?: Record<string, unknown>;
}

export function SimulationWorkspace({ nodes, connections, analysis, perLayer, warnings, topology }: SimulationWorkspaceProps) {
  const [activeCategory, setActiveCategory] = useState('realtime');

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-background">
      {/* Category Tabs */}
      <div className="border-b border-border bg-card px-4 py-2">
        <div className="flex items-center gap-1 overflow-x-auto scrollbar-thin">
          {CATEGORIES.map((cat) => {
            const Icon = cat.icon;
            const isActive = activeCategory === cat.id;
            return (
              <button
                key={cat.id}
                onClick={() => setActiveCategory(cat.id)}
                className={`flex items-center gap-1 sm:gap-2 px-2 sm:px-3 py-1.5 text-xs font-medium rounded-md transition-all whitespace-nowrap ${isActive
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:text-foreground hover:bg-secondary'
                  }`}
              >
                <Icon className="w-3.5 h-3.5 shrink-0" />
                <span className="hidden sm:inline">{cat.label}</span>
                <span className={`text-[10px] px-1 sm:px-1.5 py-0.5 rounded-full hidden sm:inline ${isActive ? 'bg-primary-foreground/20' : 'bg-muted'
                  }`}>
                  {cat.count}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Chart Content */}
      <div className="flex-1 overflow-auto p-4 scrollbar-thin">
        {activeCategory === 'realtime' && <RealTimeCharts analysis={analysis} />}
        {activeCategory === 'global' && <GlobalResultsCharts analysis={analysis} />}
        {activeCategory === 'perlayer' && <PerLayerCharts analysis={analysis} perLayer={perLayer} />}
        {activeCategory === 'memory' && <MemoryCharts analysis={analysis} />}
        {activeCategory === 'comparison' && <ComparisonCharts analysis={analysis} topology={topology} />}
        {activeCategory === 'optimization' && (
          <OptimizationCharts
            analysis={analysis}
            perLayer={perLayer}
            nodes={nodes}
            connections={connections}
          />
        )}
        {activeCategory === 'training' && (
          <TrainingCharts
            analysis={analysis}
            perLayer={perLayer}
          />
        )}
        {activeCategory === 'debugging' && (
          <DebuggingCharts
            analysis={analysis}
            perLayer={perLayer}
            warnings={warnings}
            nodes={nodes}
          />
        )}
      </div>
    </div>
  );
}
