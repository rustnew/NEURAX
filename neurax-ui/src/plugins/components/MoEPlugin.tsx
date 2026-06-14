import { 
  Route
} from 'lucide-react';
import { cn } from '@/lib/utils.ts';

interface MoEPluginProps {
  className?: string;
}

export function MoEPlugin({ className }: MoEPluginProps) {
  return (
    <div className={cn("space-y-4 animate-fade-in", className)}>
      <div className="p-4 rounded-lg border border-border bg-secondary/20">
        <div className="flex items-center gap-2">
          <Route className="w-4 h-4 text-[hsl(280,70%,55%)]" />
          <div className="text-xs text-muted-foreground">
            MoE plugin metrics are not available yet (mock data removed).
          </div>
        </div>
        <div className="text-[10px] text-muted-foreground/80 mt-1">
          Once the backend exposes MoE routing/utilization/balance metrics, this panel will render them.
        </div>
      </div>
    </div>
  );
}
