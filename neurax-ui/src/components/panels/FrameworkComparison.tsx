import { useState } from 'react';
import { 
  GitBranch
} from 'lucide-react';
import { cn } from '@/lib/utils.ts';

interface FrameworkComparisonProps {
  selectedFramework: string | null;
  onSelectFramework: (framework: string) => void;
}

export function FrameworkComparison(_props: FrameworkComparisonProps) {
  const [viewMode, setViewMode] = useState<'cards' | 'table'>('cards');

  return (
    <div className="h-full flex flex-col bg-card overflow-hidden">
      {/* Header */}
      <div className="h-10 px-4 flex items-center justify-between border-b border-border bg-panel-header shrink-0">
        <div className="flex items-center gap-2">
          <GitBranch className="w-4 h-4 text-primary" />
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Framework Comparison
          </span>
        </div>
        <div className="flex items-center gap-1">
          <button
            className={cn(
              "px-2 py-1 text-[10px] rounded transition-colors",
              viewMode === 'cards' ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-secondary"
            )}
            onClick={() => setViewMode('cards')}
          >
            Cards
          </button>
          <button
            className={cn(
              "px-2 py-1 text-[10px] rounded transition-colors",
              viewMode === 'table' ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-secondary"
            )}
            onClick={() => setViewMode('table')}
          >
            Table
          </button>
        </div>
      </div>
      
      <div className="flex-1 overflow-y-auto scrollbar-thin p-4">
        <div className="p-4 rounded-lg border border-border bg-secondary/20">
          <div className="text-xs text-muted-foreground">
            Framework comparison is not available yet (mock data removed).
          </div>
          <div className="text-[10px] text-muted-foreground/80 mt-1">
            Once the backend exposes framework/hardware comparison metrics, this panel will render them.
          </div>
        </div>
      </div>
    </div>
  );
}
