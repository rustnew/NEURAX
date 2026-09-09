import { Layers, Play, Rocket, Activity, Clock, Check, Download } from 'lucide-react';
import { cn } from '@/lib/utils.ts';

export type WorkspaceTab =
  | 'architecture'
  | 'simulation'
  | 'production'
  | 'training'
  | 'timemachine';

interface WorkspaceTabsProps {
  activeTab: WorkspaceTab;
  onTabChange: (tab: WorkspaceTab) => void;
  children: React.ReactNode;
  architectureContent: React.ReactNode;
  simulationContent?: React.ReactNode;
  productionContent?: React.ReactNode;
  trainingContent?: React.ReactNode;
  timeMachineContent?: React.ReactNode;
  /** Real signal: the canvas has at least one block on it. Undefined hides
   *  the step-progress treatment entirely (falls back to the plain tab
   *  bar) rather than showing a step as "not done" for a caller that never
   *  wired the signal up. */
  hasDesign?: boolean;
  /** Real signal: an analysis has actually produced a report — the same
   *  check the Simulation tab itself uses to decide it has something to
   *  show, not a separate guess. */
  hasAnalysis?: boolean;
  /** Opens the export panel. Export isn't a workspace tab — there's no
   *  content view to switch to — so it renders as a distinct trailing
   *  action on the rail, not a numbered step. */
  onExport?: () => void;
}

const WORKSPACE_TABS = [
  { id: 'architecture' as const, label: 'Architecture', icon: Layers },
  { id: 'simulation' as const, label: 'Simulation', icon: Play },
  { id: 'production' as const, label: 'Production', icon: Rocket },
  { id: 'training' as const, label: 'Training', icon: Activity },
  { id: 'timemachine' as const, label: 'Time Machine', icon: Clock },
];

/**
 * The order below is the sequence a design actually goes through — build
 * it, analyze it, initialize it, train it, then project it forward — not the
 * order the tabs happen to be written in.
 * Training sits after Production because that is where it falls in time:
 * a run needs initialized weights, and it is the first step that leaves
 * the analytical world and spends real hours on a real machine. Only Architecture and
 * Simulation get a real "done" state: whether a design exists and whether
 * it's been analyzed are both things this component can check for itself.
 * Production, Training and Time Machine have no equivalent completion signal — they're lenses to revisit, not tasks with an end —
 * so they only ever show as current or upcoming, never done. Inventing a
 * checkmark for them would be exactly the kind of confident-looking number
 * nothing backs that this app spent tonight removing everywhere else.
 */
function stepState(
  tabId: WorkspaceTab,
  activeTab: WorkspaceTab,
  hasDesign: boolean,
  hasAnalysis: boolean,
): 'done' | 'current' | 'upcoming' {
  if (tabId === activeTab) return 'current';
  if (tabId === 'architecture') return hasDesign ? 'done' : 'upcoming';
  if (tabId === 'simulation') return hasAnalysis ? 'done' : 'upcoming';
  return 'upcoming';
}

export function WorkspaceTabs({
  activeTab,
  onTabChange,
  architectureContent,
  simulationContent,
  productionContent,
  trainingContent,
  timeMachineContent,
  hasDesign,
  hasAnalysis,
  onExport,
}: WorkspaceTabsProps) {
  const showProgress = hasDesign !== undefined && hasAnalysis !== undefined;

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Tab Bar — a chronology rail when the caller wires up real
          completion signals, the plain tab row otherwise. Every step stays
          freely clickable either way; nothing here blocks navigation. */}
      <div className="h-10 bg-card border-b border-border flex items-center px-2 sm:px-4 gap-0.5 sm:gap-1 overflow-x-auto scrollbar-thin">
        {WORKSPACE_TABS.map((tab, index) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          const state = showProgress ? stepState(tab.id, activeTab, hasDesign!, hasAnalysis!) : 'current';

          return (
            <div key={tab.id} className="flex items-center">
              {index > 0 && showProgress && (
                <div
                  className={cn(
                    'hidden sm:block w-4 h-px mx-0.5 shrink-0',
                    state !== 'upcoming' || isActive ? 'bg-primary/40' : 'bg-border',
                  )}
                />
              )}
              <button
                onClick={() => onTabChange(tab.id)}
                title={showProgress ? `Step ${index + 1} of ${WORKSPACE_TABS.length} — ${tab.label}` : tab.label}
                className={cn(
                  'flex items-center gap-1 sm:gap-2 px-2 sm:px-4 py-2 text-xs sm:text-sm font-medium rounded-t-md transition-all whitespace-nowrap',
                  'border-b-2 -mb-[1px]',
                  isActive
                    ? 'bg-background border-primary text-foreground'
                    : 'border-transparent text-muted-foreground hover:text-foreground hover:bg-secondary/50',
                )}
              >
                {showProgress ? (
                  <span
                    className={cn(
                      'hidden sm:flex items-center justify-center w-4 h-4 rounded-full text-[9px] font-semibold shrink-0',
                      state === 'done'
                        ? 'bg-primary/15 text-primary'
                        : isActive
                          ? 'bg-primary text-primary-foreground'
                          : 'bg-secondary text-muted-foreground',
                    )}
                  >
                    {state === 'done' ? <Check className="w-2.5 h-2.5" /> : index + 1}
                  </span>
                ) : (
                  <Icon className="w-4 h-4 shrink-0" />
                )}
                <span className="hidden sm:inline">{tab.label}</span>
              </button>
            </div>
          );
        })}

        {onExport && (
          <>
            <div className="flex-1 min-w-2" />
            <button
              onClick={onExport}
              title="Export the current design"
              className="flex items-center gap-1.5 px-2 sm:px-3 py-1.5 text-xs font-medium rounded-md border border-dashed border-border text-muted-foreground hover:text-foreground hover:border-primary/40 hover:bg-secondary/50 transition-all whitespace-nowrap shrink-0"
            >
              <Download className="w-3.5 h-3.5 shrink-0" />
              <span className="hidden sm:inline">Export</span>
            </button>
          </>
        )}
      </div>

      {/* Tab Content — all tabs stay mounted to avoid Radix portal
          removeChild errors during unmount. Inactive tabs are hidden
          via CSS instead of being conditionally unmounted. */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className={cn("flex-1 flex flex-col overflow-hidden", activeTab === 'architecture' ? '' : 'hidden')}>
          {architectureContent}
        </div>
        {simulationContent && (
          <div className={cn("flex-1 flex flex-col overflow-hidden", activeTab === 'simulation' ? '' : 'hidden')}>
            {simulationContent}
          </div>
        )}
        {productionContent && (
          <div className={cn("flex-1 flex flex-col overflow-hidden", activeTab === 'production' ? '' : 'hidden')}>
            {productionContent}
          </div>
        )}
        {trainingContent && (
          <div className={cn("flex-1 flex flex-col overflow-hidden", activeTab === 'training' ? '' : 'hidden')}>
            {trainingContent}
          </div>
        )}
        {timeMachineContent && (
          <div className={cn("flex-1 flex flex-col overflow-hidden", activeTab === 'timemachine' ? '' : 'hidden')}>
            {timeMachineContent}
          </div>
        )}
      </div>
    </div>
  );
}
