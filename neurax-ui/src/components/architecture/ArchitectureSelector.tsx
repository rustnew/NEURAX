import { 
  Sparkles, 
  Network, 
  Workflow, 
  Wand2, 
  Share2,
  Lock,
  FlaskConical,
  Grid3X3,
  Gamepad2,
  Repeat,
  Zap,
  Waves,
  Brain,
  LucideIcon
} from 'lucide-react';
import { ArchitectureFamily, ARCHITECTURE_FAMILIES } from '@/types/plugins.ts';
import { usePlan } from '@/contexts/PlanContext.tsx';
import { canAccessArchitecture } from '@/types/plans.ts';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select.tsx';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip.tsx';
import { Badge } from '@/components/ui/badge.tsx';
import { cn } from '@/lib/utils.ts';

const iconMap: Record<string, LucideIcon> = {
  Sparkles,
  Network,
  Workflow,
  Wand2,
  Share2,
  FlaskConical,
  Grid3X3,
  Gamepad2,
  Repeat,
  Zap,
  Waves,
  Brain,
};

// Extended families with experimental
const EXTENDED_FAMILIES = [
  ...ARCHITECTURE_FAMILIES,
  {
    id: 'experimental' as ArchitectureFamily,
    name: 'Experimental',
    description: 'Early access R&D models',
    icon: 'FlaskConical',
    color: 'hsl(280, 70%, 55%)',
  },
];

interface ArchitectureSelectorProps {
  value: ArchitectureFamily;
  onChange: (value: ArchitectureFamily) => void;
  className?: string;
}

export function ArchitectureSelector({ value, onChange, className }: ArchitectureSelectorProps) {
  const { currentPlan } = usePlan();
  
  const currentFamily = EXTENDED_FAMILIES.find(f => f.id === value);
  const CurrentIcon = currentFamily ? iconMap[currentFamily.icon] : Sparkles;

  const accessibleFamilies = EXTENDED_FAMILIES.filter(f => 
    canAccessArchitecture(currentPlan, f.id)
  );
  
  const lockedFamilies = EXTENDED_FAMILIES.filter(f => 
    !canAccessArchitecture(currentPlan, f.id)
  );

  return (
    <Select value={value} onValueChange={(v) => onChange(v as ArchitectureFamily)}>
      <SelectTrigger 
        className={cn(
          "w-[280px] h-9 bg-secondary/50 border-border/50 hover:bg-secondary transition-colors",
          "focus:ring-1 focus:ring-primary/50",
          className
        )}
      >
        <div className="flex items-center gap-2">
          <div 
            className="w-5 h-5 rounded flex items-center justify-center"
            style={{ backgroundColor: `${currentFamily?.color}20` }}
          >
            <CurrentIcon 
              className="w-3.5 h-3.5" 
              style={{ color: currentFamily?.color }}
            />
          </div>
          <SelectValue placeholder="Select Architecture" />
        </div>
      </SelectTrigger>
      <SelectContent className="bg-popover border-border z-50">
        {accessibleFamilies.map((family) => {
          const Icon = iconMap[family.icon] || Sparkles;
          const isExperimental = family.id === 'experimental';
          
          return (
            <SelectItem 
              key={family.id} 
              value={family.id}
              className="cursor-pointer focus:bg-secondary"
            >
              <div className="flex items-center gap-3 py-0.5">
                <div 
                  className="w-6 h-6 rounded flex items-center justify-center"
                  style={{ backgroundColor: `${family.color}20` }}
                >
                  <Icon 
                    className="w-4 h-4" 
                    style={{ color: family.color }}
                  />
                </div>
                <div className="flex flex-col">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium">{family.name}</span>
                    {isExperimental && (
                      <Badge 
                        variant="outline" 
                        className="text-[8px] px-1 py-0 h-4 bg-amber-500/10 text-amber-400 border-amber-500/30"
                      >
                        EARLY ACCESS
                      </Badge>
                    )}
                  </div>
                  <span className="text-[10px] text-muted-foreground">{family.description}</span>
                </div>
              </div>
            </SelectItem>
          );
        })}

        {lockedFamilies.length > 0 && (
          <>
            <div className="h-px bg-border my-1" />
            <div className="px-2 py-1 text-[10px] text-muted-foreground uppercase tracking-wider">
              Upgrade to unlock
            </div>
            {lockedFamilies.map((family) => {
              const Icon = iconMap[family.icon] || Sparkles;
              const requiredPlan = ['snn', 'experimental'].includes(family.id) ? 'ELITE' : 'ARCHITECT';
              
              return (
                <Tooltip key={family.id}>
                  <TooltipTrigger asChild>
                    <div className="flex items-center gap-3 py-1.5 px-2 opacity-50 cursor-not-allowed">
                      <div 
                        className="w-6 h-6 rounded flex items-center justify-center"
                        style={{ backgroundColor: `${family.color}10` }}
                      >
                        <Icon 
                          className="w-4 h-4" 
                          style={{ color: family.color }}
                        />
                      </div>
                      <div className="flex flex-col flex-1">
                        <span className="text-sm font-medium">{family.name}</span>
                        <span className="text-[10px] text-muted-foreground">{family.description}</span>
                      </div>
                      <Lock className="w-3.5 h-3.5 text-muted-foreground" />
                    </div>
                  </TooltipTrigger>
                  <TooltipContent side="right">
                    <p className="text-xs">
                      Available on <span className="font-semibold text-primary">{requiredPlan}</span> plan
                    </p>
                  </TooltipContent>
                </Tooltip>
              );
            })}
          </>
        )}
      </SelectContent>
    </Select>
  );
}
