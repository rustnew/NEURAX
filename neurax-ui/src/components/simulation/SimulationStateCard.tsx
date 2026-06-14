import { Clock3 } from 'lucide-react';
import { ReactNode } from 'react';

interface SimulationStateCardProps {
  title: string;
  indexLabel?: string;
  message: string;
  hint?: string;
  className?: string;
  children?: ReactNode;
}

export function SimulationStateCard({
  title,
  indexLabel,
  message,
  hint = 'Waiting for compiler-side report fields.',
  className = '',
  children,
}: SimulationStateCardProps) {
  return (
    <div className={`panel-section rounded-xl border border-border/60 bg-card/30 p-4 ${className}`.trim()}>
      <div className="mb-4 flex items-center justify-between gap-3">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          {indexLabel ? `${indexLabel} — ${title}` : title}
        </h3>
      </div>

      <div className="flex min-h-[192px] flex-col items-center justify-center rounded-lg border border-dashed border-border/60 bg-secondary/10 px-6 py-8 text-center">
        <Clock3 className="mb-3 h-7 w-7 text-muted-foreground/55" />
        <div className="text-sm font-semibold text-foreground">{message}</div>
        <div className="mt-2 max-w-sm text-xs leading-relaxed text-muted-foreground">{hint}</div>
        {children ? <div className="mt-4 w-full">{children}</div> : null}
      </div>
    </div>
  );
}
