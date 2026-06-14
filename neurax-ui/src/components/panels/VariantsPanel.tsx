import { 
  GitCompare
} from 'lucide-react';
import { cn } from '@/lib/utils.ts';
import { Button } from '@/components/ui/button.tsx';

interface VariantsPanelProps {
  onPromoteVariant?: (variantId: string) => void;
}

export function VariantsPanel(_props: VariantsPanelProps) {
  return (
    <div className="h-full flex flex-col bg-card overflow-hidden">
      {/* Header */}
      <div className="h-10 px-4 flex items-center justify-between border-b border-border bg-panel-header shrink-0">
        <div className="flex items-center gap-2">
          <GitCompare className="w-4 h-4 text-primary" />
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Variants
          </span>
        </div>
        <Button
          variant="ghost"
          size="sm"
          className={cn(
            "h-6 text-[10px]"
          )}
          disabled
        >
          Compare
        </Button>
      </div>
      
      <div className="flex-1 overflow-y-auto scrollbar-thin p-4 space-y-3">
        <div className="p-4 rounded-lg border border-border bg-secondary/20">
          <div className="text-xs text-muted-foreground">
            Variants are not available yet (mock data removed).
          </div>
          <div className="text-[10px] text-muted-foreground/80 mt-1">
            Once the backend exposes variant generation/optimization outputs, this panel will render them.
          </div>
        </div>
      </div>
    </div>
  );
}
