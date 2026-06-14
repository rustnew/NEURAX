import { Users, MousePointer2, Eye, Lock } from 'lucide-react';
import { usePlan } from '@/contexts/PlanContext.tsx';
import { cn } from '@/lib/utils.ts';
import { Avatar, AvatarFallback } from '@/components/ui/avatar.tsx';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip.tsx';

interface CollaborationIndicatorProps {
  className?: string;
  variant?: 'compact' | 'full';
}

export function CollaborationIndicator({ 
  className,
  variant = 'compact' 
}: CollaborationIndicatorProps) {
  const { canAccess } = usePlan();
  const hasAccess = canAccess('elite');

  if (!hasAccess) {
    if (variant === 'compact') return null;
    
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <div className={cn(
            "flex items-center gap-2 px-3 py-2 rounded-lg bg-secondary/30 border border-border/50 cursor-not-allowed opacity-60",
            className
          )}>
            <Users className="w-4 h-4 text-muted-foreground" />
            <span className="text-xs text-muted-foreground">Collaboration</span>
            <Lock className="w-3 h-3 text-muted-foreground ml-auto" />
          </div>
        </TooltipTrigger>
        <TooltipContent side="bottom" className="max-w-[220px]">
          <p className="text-xs">
            Real-time collaboration is available on the <span className="font-semibold text-primary">ELITE/LAB</span> plan.
          </p>
        </TooltipContent>
      </Tooltip>
    );
  }

  const collaborators: Array<{ id: string; name: string; initials: string; color: string; isActive: boolean }> = [];
  const activeCount = collaborators.filter(c => c.isActive).length;

  return (
    <div className={cn("flex items-center gap-2", className)}>
      {/* Live indicator */}
      <div className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-emerald-500/10 border border-emerald-500/20">
        <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
        <span className="text-[10px] font-medium text-emerald-400 uppercase tracking-wide">
          Live
        </span>
      </div>

      {/* Collaborator avatars */}
      <div className="flex -space-x-2">
        {collaborators.slice(0, 3).map((collaborator) => (
          <Tooltip key={collaborator.id}>
            <TooltipTrigger asChild>
              <div className="relative">
                <Avatar className={cn(
                  "w-7 h-7 border-2 border-background cursor-pointer",
                  collaborator.color
                )}>
                  <AvatarFallback className={cn(
                    "text-[10px] font-medium text-white",
                    collaborator.color
                  )}>
                    {collaborator.initials}
                  </AvatarFallback>
                </Avatar>
                {collaborator.isActive && (
                  <div className="absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 rounded-full bg-emerald-500 border-2 border-background" />
                )}
              </div>
            </TooltipTrigger>
            <TooltipContent side="bottom">
              <div className="flex items-center gap-2">
                <span className="text-xs">{collaborator.name}</span>
                {collaborator.isActive ? (
                  <span className="text-[10px] text-emerald-400">Editing</span>
                ) : (
                  <span className="text-[10px] text-muted-foreground">Viewing</span>
                )}
              </div>
            </TooltipContent>
          </Tooltip>
        ))}
      </div>

      {/* Viewer count */}
      <Tooltip>
        <TooltipTrigger asChild>
          <div className="flex items-center gap-1 px-2 py-1 rounded-md bg-secondary/50 text-xs text-muted-foreground">
            <Eye className="w-3 h-3" />
            <span>{activeCount}</span>
          </div>
        </TooltipTrigger>
        <TooltipContent side="bottom">
          <p className="text-xs">{activeCount} collaborators online</p>
        </TooltipContent>
      </Tooltip>
    </div>
  );
}

// Canvas cursor component for showing live cursors
export function LiveCursors() {
  const { canAccess } = usePlan();
  
  if (!canAccess('elite')) return null;

  return null;
}
