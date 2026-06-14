import { useCallback, useRef, useState } from 'react';
import {
  BarChart3,
  Database,
  SlidersHorizontal,
  AlertTriangle,
  Cpu,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { CanvasNode, AnalysisResult, Warning } from '@/types/architecture.ts';
import { ArchitectureFamily } from '@/types/plugins.ts';
import { ParameterPanel } from './ParameterPanel.tsx';
import { MetricsDashboard } from './MetricsDashboard.tsx';
import { AnalysisPanel } from './AnalysisPanel.tsx';
import { IssuesPanel } from './IssuesPanel.tsx';
import { Button } from '@/components/ui/button.tsx';
import { cn } from '@/lib/utils.ts';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip.tsx';
import { PerLayerBreakdownRow } from '@/types/architecture.ts';

export type RightPanelTabId = 'analysis' | 'issues' | 'parameters' | 'metrics' | 'deepMetrics';

interface RightPanelTabsProps {
  nodes: CanvasNode[];
  selectedNodeId: string | null;
  onSelectNode: (id: string) => void;
  analysis: AnalysisResult;
  warnings: Warning[];
  perLayer: PerLayerBreakdownRow[];
  selectedArchitecture: ArchitectureFamily;
  activeTab?: RightPanelTabId;
  onActiveTabChange?: (tab: RightPanelTabId) => void;
  jumpToIssuesSignal?: number;
}

const TABS: { id: RightPanelTabId; label: string; icon: React.ElementType; tooltip: string }[] = [
  { id: 'analysis', label: 'Tune', icon: SlidersHorizontal, tooltip: 'Model Hyperparameters' },
  { id: 'parameters', label: 'Params', icon: Database, tooltip: 'Parameter Generation' },
  { id: 'metrics', label: 'Metrics', icon: BarChart3, tooltip: 'FLOPs & Memory' },
  { id: 'deepMetrics', label: 'Deep', icon: Cpu, tooltip: 'Advanced Compute, Memory & System Metrics' },
  { id: 'issues', label: 'Issues', icon: AlertTriangle, tooltip: 'Warnings & Diagnostics' },
];

export function RightPanelTabs({
  nodes,
  selectedNodeId,
  onSelectNode,
  analysis,
  warnings,
  perLayer,
  selectedArchitecture,
  activeTab: controlledActiveTab,
  onActiveTabChange,
  jumpToIssuesSignal,
}: RightPanelTabsProps) {
  const [internalActiveTab, setInternalActiveTab] = useState<RightPanelTabId>('analysis');
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [panelWidth, setPanelWidth] = useState<number>(320);
  const isResizingWidth = useRef(false);
  const activeTab = controlledActiveTab ?? internalActiveTab;

  const setActiveTab = useCallback((tab: RightPanelTabId) => {
    if (controlledActiveTab === undefined) {
      setInternalActiveTab(tab);
    }
    onActiveTabChange?.(tab);
  }, [controlledActiveTab, onActiveTabChange]);

  const errorCount = warnings.filter(w => w.type === 'error').length;
  const warningCount = warnings.filter(w => w.type === 'warning').length;

  const clampPanelWidth = useCallback((w: number) => {
    return Math.max(256, Math.min(560, w));
  }, []);

  const handleResizeWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();

    const primaryDelta = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
    setPanelWidth(prev => clampPanelWidth(prev - primaryDelta));
  }, [clampPanelWidth]);

  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isResizingWidth.current = true;
    const startX = e.clientX;
    const startWidth = panelWidth;

    const onMouseMove = (ev: MouseEvent) => {
      if (!isResizingWidth.current) return;
      const delta = startX - ev.clientX;
      setPanelWidth(clampPanelWidth(startWidth + delta));
    };

    const onMouseUp = () => {
      isResizingWidth.current = false;
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    document.body.style.cursor = 'ew-resize';
    document.body.style.userSelect = 'none';
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }, [clampPanelWidth, panelWidth]);

  if (isCollapsed) {
    return (
      <aside className="w-12 h-full bg-sidebar border-l border-sidebar-border flex flex-col transition-all duration-300">
        {/* Expand Button */}
        <div className="h-10 flex items-center justify-center border-b border-sidebar-border bg-panel-header">
          <Button
            variant="ghost"
            size="icon"
            className="w-6 h-6 text-muted-foreground hover:text-foreground"
            onClick={() => setIsCollapsed(false)}
          >
            <ChevronLeft className="w-4 h-4" />
          </Button>
        </div>

        {/* Collapsed Tab Icons */}
        <div className="flex-1 flex flex-col items-center gap-2 py-2">
          {TABS.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            const showBadge = tab.id === 'issues' && (errorCount > 0 || warningCount > 0);

            return (
              <Tooltip key={tab.id}>
                <TooltipTrigger asChild>
                  <button
                    className={cn(
                      "relative w-8 h-8 rounded-md flex items-center justify-center transition-colors",
                      isActive
                        ? "bg-primary/20 text-primary"
                        : "text-muted-foreground hover:text-foreground hover:bg-secondary/50"
                    )}
                    onClick={() => {
                      setActiveTab(tab.id);
                      setIsCollapsed(false);
                    }}
                  >
                    <Icon className="w-4 h-4" />
                    {showBadge && (
                      <span className={cn(
                        "absolute -top-0.5 -right-0.5 w-4 h-4 rounded-full text-[9px] flex items-center justify-center",
                        errorCount > 0
                          ? "bg-destructive text-destructive-foreground"
                          : "bg-warning text-warning-foreground"
                      )}>
                        {errorCount > 0 ? errorCount : warningCount}
                      </span>
                    )}
                  </button>
                </TooltipTrigger>
                <TooltipContent side="left" className="text-xs">
                  {tab.tooltip}
                </TooltipContent>
              </Tooltip>
            );
          })}
        </div>
      </aside>
    );
  }

  return (
    <aside
      className="h-full bg-sidebar border-l border-sidebar-border flex flex-col transition-all duration-300 relative"
      style={{ width: panelWidth }}
    >
      <div
        className="absolute top-0 left-0 w-2 h-full cursor-ew-resize hover:bg-primary/10 transition-colors z-20 group"
        onMouseDown={handleResizeStart}
        onWheel={handleResizeWheel}
      >
        <div className="absolute top-1/2 -translate-y-1/2 left-0.5 w-0.5 h-10 rounded-full bg-muted-foreground/30 group-hover:bg-primary/50 transition-colors" />
      </div>
      {/* Tab Bar */}
      <div className="h-10 px-1 flex items-center gap-0.5 border-b border-sidebar-border bg-panel-header">
        <Button
          variant="ghost"
          size="icon"
          className="w-6 h-6 text-muted-foreground hover:text-foreground flex-shrink-0"
          onClick={() => setIsCollapsed(true)}
        >
          <ChevronRight className="w-4 h-4" />
        </Button>

        {TABS.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          const showBadge = tab.id === 'issues' && (errorCount > 0 || warningCount > 0);

          return (
            <Tooltip key={tab.id}>
              <TooltipTrigger asChild>
                <button
                  className={cn(
                    "relative w-8 h-8 rounded-md flex items-center justify-center transition-colors flex-shrink-0",
                    isActive
                      ? "bg-primary/20 text-primary"
                      : "text-muted-foreground hover:text-foreground hover:bg-secondary/50"
                  )}
                  onClick={() => setActiveTab(tab.id)}
                >
                  <Icon className="w-4 h-4" />
                  {showBadge && (
                    <span className="absolute -top-0.5 -right-0.5 w-4 h-4 rounded-full bg-destructive text-destructive-foreground text-[9px] flex items-center justify-center">
                      {errorCount}
                    </span>
                  )}
                </button>
              </TooltipTrigger>
              <TooltipContent side="bottom" className="text-xs">
                {tab.tooltip}
              </TooltipContent>
            </Tooltip>
          );
        })}
      </div>

      {/* Panel Content */}
      <div className="flex-1 overflow-hidden">
        {activeTab === 'analysis' && (
          <AnalysisPanel
            selectedArchitecture={selectedArchitecture}
          />
        )}
        {activeTab === 'issues' && (
          <IssuesPanel
            warnings={warnings}
            jumpToIssuesSignal={jumpToIssuesSignal}
          />
        )}
        {activeTab === 'parameters' && (
          <ParameterPanel
            nodes={nodes}
            selectedNodeId={selectedNodeId}
            onSelectNode={onSelectNode}
            analysis={analysis}
            perLayer={perLayer}
          />
        )}
        {activeTab === 'metrics' && (
          <MetricsDashboard
            nodes={nodes}
            selectedNodeId={selectedNodeId}
            onSelectNode={onSelectNode}
            analysis={analysis}
            perLayer={perLayer}
            view="overview"
          />
        )}
        {activeTab === 'deepMetrics' && (
          <MetricsDashboard
            nodes={nodes}
            selectedNodeId={selectedNodeId}
            onSelectNode={onSelectNode}
            analysis={analysis}
            perLayer={perLayer}
            view="deep"
          />
        )}
      </div>
    </aside>
  );
}
