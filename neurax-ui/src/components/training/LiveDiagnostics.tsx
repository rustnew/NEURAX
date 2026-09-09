/**
 * The compiler's diagnostics, continued into the run.
 *
 * NEURAX already emits static diagnostics — E001, H001, H008 — with a rule
 * this file keeps: name the cause and what to change, never just the symptom.
 * "Loss is NaN" is a symptom; "the learning rate is 3e-4 with no warmup and
 * the gradient norm crossed 1e4 at step 812" is a cause.
 *
 * These are the codes only a running process can raise. `L002` (waiting on
 * data) cannot be predicted from a graph at all: it depends on the disk, the
 * loader workers and what else the machine is doing. That is exactly the class
 * of finding the analytical compiler cannot reach, and the reason this
 * workspace earns its place.
 */
import { CircleCheck, Info, OctagonAlert, TriangleAlert } from 'lucide-react';

import type { DiagnosticSeverity, LiveDiagnostic } from '@/types/runtime.ts';
import { cn } from '@/lib/utils.ts';

interface LiveDiagnosticsProps {
  diagnostics: LiveDiagnostic[];
}

const SEVERITY: Record<DiagnosticSeverity, { icon: typeof Info; label: string; box: string; text: string }> = {
  critical: {
    icon: OctagonAlert,
    label: 'Critical',
    box: 'border-red-500/40 bg-red-500/5',
    text: 'text-red-600 dark:text-red-400',
  },
  warning: {
    icon: TriangleAlert,
    label: 'Warning',
    box: 'border-amber-500/40 bg-amber-500/5',
    text: 'text-amber-600 dark:text-amber-400',
  },
  hint: {
    icon: Info,
    label: 'Hint',
    box: 'border-border bg-muted/20',
    text: 'text-muted-foreground',
  },
};

const ORDER: DiagnosticSeverity[] = ['critical', 'warning', 'hint'];

export function LiveDiagnostics({ diagnostics }: LiveDiagnosticsProps) {
  if (diagnostics.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center gap-2 py-16 text-center">
        <CircleCheck className="w-7 h-7 text-emerald-500/60" />
        <div className="text-[13px] font-semibold text-foreground">Nothing to report</div>
        <p className="text-[11px] text-muted-foreground max-w-[46ch] leading-relaxed">
          The run is behaving the way the analysis said it would. Findings appear here the moment it stops doing so.
        </p>
      </div>
    );
  }

  const sorted = [...diagnostics].sort(
    (a, b) => ORDER.indexOf(a.severity) - ORDER.indexOf(b.severity) || (b.atStep ?? 0) - (a.atStep ?? 0),
  );

  return (
    <div className="space-y-3">
      {sorted.map((d, i) => {
        const style = SEVERITY[d.severity];
        const Icon = style.icon;
        return (
          <div key={`${d.code}-${i}`} className={cn('rounded-lg border p-3', style.box)}>
            <div className="flex items-center gap-2">
              <Icon className={cn('w-3.5 h-3.5 shrink-0', style.text)} />
              <span className={cn('text-[11px] font-semibold uppercase tracking-wider', style.text)}>
                {style.label}
              </span>
              <span className="font-mono text-[11px] text-muted-foreground">{d.code}</span>
              {d.atStep !== undefined ? (
                <span className="ml-auto font-mono text-[10px] text-muted-foreground/70 tabular-nums">
                  step {d.atStep.toLocaleString('en-US')}
                </span>
              ) : null}
            </div>
            <p className="mt-1.5 text-[12px] text-foreground/90 leading-relaxed max-w-[92ch]">{d.message}</p>
          </div>
        );
      })}
    </div>
  );
}
