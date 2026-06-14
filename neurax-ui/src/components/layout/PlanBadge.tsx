import { Crown, ChevronDown } from 'lucide-react';
import { usePlan } from '@/contexts/PlanContext.tsx';
import { PLAN_CONFIGS } from '@/types/plans.ts';
import { cn } from '@/lib/utils.ts';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu.tsx';

export function PlanBadge() {
  const { currentPlan, planConfig } = usePlan();

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          className={cn(
            "flex items-center gap-1.5 px-2.5 py-1 rounded-md border text-xs font-medium transition-colors",
            "hover:bg-secondary/50 focus:outline-none focus:ring-1 focus:ring-primary/50",
            planConfig.badge
          )}
        >
          {currentPlan === 'elite' ? (
            <Crown className="w-3.5 h-3.5" />
          ) : null}
          <span className="tracking-wider">{planConfig.displayName}</span>
          <ChevronDown className="w-3 h-3 opacity-60" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-48">
        <div className="px-2 py-1.5 text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
          Current plan
        </div>
        <DropdownMenuItem className="cursor-default" onSelect={(e) => e.preventDefault()}>
          <div className="flex items-center gap-2 w-full">
            <div
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: PLAN_CONFIGS[currentPlan].color }}
            />
            <span className="flex-1">{PLAN_CONFIGS[currentPlan].name}</span>
            <span className="text-[10px] text-primary">Active</span>
          </div>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
