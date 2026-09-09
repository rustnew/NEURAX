import { useState, lazy, Suspense } from 'react';
import {
  BarChart3, Layers, HardDrive,
  Target, Bug,
} from 'lucide-react';
import { CanvasNode, Connection, AnalysisResult, PerLayerBreakdownRow, Warning } from '@/types/architecture.ts';
import { ChartSkeleton } from './shared';

const GlobalResultsCharts = lazy(() => import('./categories/GlobalResultsCharts.tsx').then(m => ({ default: m.GlobalResultsCharts })));
const PerLayerCharts = lazy(() => import('./categories/PerLayerCharts.tsx').then(m => ({ default: m.PerLayerCharts })));
const MemoryCharts = lazy(() => import('./categories/MemoryCharts.tsx').then(m => ({ default: m.MemoryCharts })));
const OptimizationCharts = lazy(() => import('./categories/OptimizationCharts.tsx').then(m => ({ default: m.OptimizationCharts })));
const DebuggingCharts = lazy(() => import('./categories/DebuggingCharts.tsx').then(m => ({ default: m.DebuggingCharts })));

export type SimulationCategoryId =
  | 'overview' | 'perlayer' | 'memory'
  | 'optimization' | 'diagnostics';

/**
 * Analysis views, ordered as they are read rather than as they were written.
 *
 * The sequence follows the question a designer actually works through: what is
 * this model, where does its cost sit, will it fit, how could it be faster, how
 * does it compare, and what is wrong with it. What a training run will cost, and
 * what it actually costs once running, both live in the Training workspace —
 * this one never leaves the realm of prediction.
 *
 * `chartCount` is asserted against the number of cards each module renders, so
 * the badges cannot drift out of date the way they had.
 */
export const SIMULATION_CATEGORIES = [
  {
    id: 'overview',
    label: 'Overview',
    hint: 'Headline size, cost and hardware fit',
    icon: BarChart3,
    chartCount: 4,
  },
  {
    id: 'perlayer',
    label: 'Per Layer',
    hint: 'The profile of the model along its own depth',
    icon: Layers,
    chartCount: 4,
  },
  {
    id: 'memory',
    label: 'Memory',
    hint: 'Whether it fits, and what to change if it does not',
    icon: HardDrive,
    chartCount: 4,
  },
  {
    id: 'optimization',
    label: 'Optimization',
    hint: 'What limits this design, and by how much',
    icon: Target,
    chartCount: 4,
  },
  {
    id: 'diagnostics',
    label: 'Diagnostics',
    hint: 'What the compiler could not resolve, and where',
    icon: Bug,
    chartCount: 2,
  },
] as const satisfies ReadonlyArray<{
  id: SimulationCategoryId;
  label: string;
  hint: string;
  icon: typeof Bug;
  chartCount: number;
}>;

interface SimulationWorkspaceProps {
  nodes: CanvasNode[];
  connections: Connection[];
  analysis?: AnalysisResult;
  perLayer?: PerLayerBreakdownRow[];
  warnings?: Warning[];
}

export function SimulationWorkspace({
  nodes,
  analysis,
  perLayer,
  warnings,
}: SimulationWorkspaceProps) {
  const [activeCategory, setActiveCategory] = useState<SimulationCategoryId>('overview');

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-background">
      <div className="border-b border-border bg-card px-4 py-2">
        <div
          className="flex items-center gap-1 overflow-x-auto scrollbar-thin"
          role="tablist"
          aria-label="Analysis views"
        >
          {SIMULATION_CATEGORIES.map((category) => {
            const Icon = category.icon;
            const isActive = activeCategory === category.id;
            return (
              <button
                key={category.id}
                type="button"
                role="tab"
                aria-selected={isActive}
                title={category.hint}
                onClick={() => setActiveCategory(category.id)}
                className={`flex items-center gap-1 sm:gap-2 px-2 sm:px-3 py-1.5 text-xs font-medium rounded-md transition-all whitespace-nowrap ${
                  isActive
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:text-foreground hover:bg-secondary'
                }`}
              >
                <Icon className="w-3.5 h-3.5 shrink-0" />
                <span className="hidden sm:inline">{category.label}</span>
                <span
                  className={`text-[10px] px-1 sm:px-1.5 py-0.5 rounded-full hidden sm:inline ${
                    isActive ? 'bg-primary-foreground/20' : 'bg-muted'
                  }`}
                >
                  {category.chartCount}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Only the active category is mounted, so its charts load on demand. */}
      <div className="flex-1 overflow-auto p-4 scrollbar-thin" role="tabpanel">
        <Suspense fallback={<ChartSkeleton variant="stats-grid" />}>
          {activeCategory === 'overview' && <GlobalResultsCharts analysis={analysis} perLayer={perLayer} />}
          {activeCategory === 'perlayer' && <PerLayerCharts analysis={analysis} perLayer={perLayer} />}
          {activeCategory === 'memory' && <MemoryCharts analysis={analysis} perLayer={perLayer} />}
          {activeCategory === 'optimization' && (
            <OptimizationCharts analysis={analysis} perLayer={perLayer} />
          )}
          {activeCategory === 'diagnostics' && (
            <DebuggingCharts
              analysis={analysis}
              perLayer={perLayer}
              warnings={warnings}
              nodes={nodes}
            />
          )}
        </Suspense>
      </div>
    </div>
  );
}
