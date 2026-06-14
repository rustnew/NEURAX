import { Layers, Play, Rocket, Brain, Clock } from 'lucide-react';
import { cn } from '@/lib/utils.ts';

export type WorkspaceTab = 'architecture' | 'simulation' | 'production' | 'inference' | 'timemachine';

interface WorkspaceTabsProps {
  activeTab: WorkspaceTab;
  onTabChange: (tab: WorkspaceTab) => void;
  children: React.ReactNode;
  architectureContent: React.ReactNode;
  simulationContent?: React.ReactNode;
  productionContent?: React.ReactNode;
  inferenceContent?: React.ReactNode;
  timeMachineContent?: React.ReactNode;
}

const WORKSPACE_TABS = [
  { id: 'architecture' as const, label: 'Architecture', icon: Layers },
  { id: 'simulation' as const, label: 'Simulation', icon: Play },
  { id: 'production' as const, label: 'Production', icon: Rocket },
  { id: 'inference' as const, label: 'Inference Intelligence', icon: Brain },
  { id: 'timemachine' as const, label: 'Time Machine', icon: Clock },
];

export function WorkspaceTabs({
  activeTab,
  onTabChange,
  architectureContent,
  simulationContent,
  productionContent,
  inferenceContent,
  timeMachineContent,
}: WorkspaceTabsProps) {
  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Tab Bar */}
      <div className="h-10 bg-card border-b border-border flex items-center px-2 sm:px-4 gap-0.5 sm:gap-1 overflow-x-auto scrollbar-thin">
        {WORKSPACE_TABS.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;

          return (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={cn(
                "flex items-center gap-1 sm:gap-2 px-2 sm:px-4 py-2 text-xs sm:text-sm font-medium rounded-t-md transition-all whitespace-nowrap",
                "border-b-2 -mb-[1px]",
                isActive
                  ? "bg-background border-primary text-foreground"
                  : "border-transparent text-muted-foreground hover:text-foreground hover:bg-secondary/50"
              )}
            >
              <Icon className="w-4 h-4 shrink-0" />
              <span className="hidden sm:inline">{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* Tab Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {activeTab === 'architecture' && architectureContent}
        {activeTab === 'simulation' && simulationContent}
        {activeTab === 'production' && productionContent}
        {activeTab === 'inference' && inferenceContent}
        {activeTab === 'timemachine' && timeMachineContent}
      </div>
    </div>
  );
}
