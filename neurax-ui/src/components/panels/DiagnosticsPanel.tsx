import {
  Activity,
  Cpu,
  HardDrive,
  Layers,
  Lock,
  Server,
  TrendingUp,
  Zap,
  CheckCircle,
} from 'lucide-react';
import { usePlan } from '@/contexts/PlanContext.tsx';
import { DIAGNOSTIC_FEATURES, canAccessDiagnostic } from '@/types/plans.ts';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip.tsx';

const diagnosticIcons: Record<string, React.ElementType> = {
  shape: Layers,
  vram: HardDrive,
  flops: Zap,
  gradient: TrendingUp,
  router: Activity,
  stability: Activity,
  mup: Cpu,
  bottleneck: Server,
  cluster: Server,
};

export function DiagnosticsPanel() {
  const { currentPlan } = usePlan();

  const accessibleDiagnostics = DIAGNOSTIC_FEATURES.filter((d) =>
    canAccessDiagnostic(currentPlan, d),
  );
  const lockedDiagnostics = DIAGNOSTIC_FEATURES.filter((d) =>
    !canAccessDiagnostic(currentPlan, d),
  );

  return (
    <div className="h-full bg-sidebar flex flex-col overflow-hidden">
      <div className="h-10 px-4 flex items-center border-b border-sidebar-border">
        <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Diagnostics
        </span>
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-thin">
        <div className="p-2 space-y-1">
          {accessibleDiagnostics.map((diagnostic) => {
            const Icon = diagnosticIcons[diagnostic.id] || Cpu;
            return (
              <div
                key={diagnostic.id}
                className="flex items-center gap-2 p-2 rounded-md bg-secondary/30 hover:bg-secondary/50 transition-colors cursor-pointer"
              >
                <Icon className="w-3.5 h-3.5 text-primary" />
                <div className="flex-1">
                  <div className="text-xs font-medium">{diagnostic.name}</div>
                  <div className="text-[10px] text-muted-foreground">{diagnostic.description}</div>
                </div>
                <CheckCircle className="w-3.5 h-3.5 text-success" />
              </div>
            );
          })}
        </div>

        {lockedDiagnostics.length > 0 && (
          <div className="p-2 pt-0 space-y-1">
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider px-2 py-1">
              Upgrade to unlock
            </div>
            {lockedDiagnostics.map((diagnostic) => {
              const Icon = diagnosticIcons[diagnostic.id] || Cpu;
              return (
                <Tooltip key={diagnostic.id}>
                  <TooltipTrigger asChild>
                    <div className="flex items-center gap-2 p-2 rounded-md bg-secondary/10 opacity-50 cursor-not-allowed">
                      <Icon className="w-3.5 h-3.5 text-muted-foreground" />
                      <div className="flex-1">
                        <div className="text-xs font-medium">{diagnostic.name}</div>
                        <div className="text-[10px] text-muted-foreground">{diagnostic.description}</div>
                      </div>
                      <Lock className="w-3.5 h-3.5 text-muted-foreground" />
                    </div>
                  </TooltipTrigger>
                  <TooltipContent side="left">
                    <p className="text-xs">
                      Available on{' '}
                      <span className="font-semibold text-primary">
                        {diagnostic.minPlan.toUpperCase()}
                      </span>{' '}
                      plan
                    </p>
                  </TooltipContent>
                </Tooltip>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
