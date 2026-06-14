import { Lock } from 'lucide-react';
import { cn } from '@/lib/utils.ts';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip.tsx';
import { PlanTier, getRequiredPlanName } from '@/types/plans.ts';

interface LockedFeatureProps {
  minPlan: PlanTier;
  children: React.ReactNode;
  className?: string;
  showLockIcon?: boolean;
  tooltipSide?: 'top' | 'right' | 'bottom' | 'left';
}

export function LockedFeature({ 
  minPlan, 
  children, 
  className,
  showLockIcon = true,
  tooltipSide = 'top'
}: LockedFeatureProps) {
  const planName = getRequiredPlanName(minPlan);
  
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div 
          className={cn(
            "relative opacity-50 cursor-not-allowed select-none",
            className
          )}
        >
          {children}
          {showLockIcon && (
            <div className="absolute top-1/2 right-2 -translate-y-1/2 flex items-center gap-1.5 px-1.5 py-0.5 rounded bg-background/80 backdrop-blur-sm border border-border/50">
              <Lock className="w-3 h-3 text-muted-foreground" />
              <span className="text-[9px] font-medium text-muted-foreground uppercase tracking-wide">
                {planName}
              </span>
            </div>
          )}
        </div>
      </TooltipTrigger>
      <TooltipContent side={tooltipSide} className="max-w-[200px]">
        <p className="text-xs">
          This feature is available on the <span className="font-semibold text-primary">{planName}</span> plan.
        </p>
      </TooltipContent>
    </Tooltip>
  );
}

interface LockedOverlayProps {
  minPlan: PlanTier;
  children: React.ReactNode;
  isLocked: boolean;
  className?: string;
}

export function LockedOverlay({ 
  minPlan, 
  children, 
  isLocked,
  className 
}: LockedOverlayProps) {
  if (!isLocked) return <>{children}</>;
  
  return (
    <LockedFeature minPlan={minPlan} className={className}>
      {children}
    </LockedFeature>
  );
}
