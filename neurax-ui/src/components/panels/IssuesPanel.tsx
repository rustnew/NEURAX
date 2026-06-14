import { useEffect, useRef } from 'react';
import {
  AlertCircle,
  AlertTriangle,
  CheckCircle,
} from 'lucide-react';
import { Warning } from '@/types/architecture.ts';
import { cn } from '@/lib/utils.ts';

interface IssuesPanelProps {
  warnings: Warning[];
  jumpToIssuesSignal?: number;
}

export function IssuesPanel({ warnings, jumpToIssuesSignal = 0 }: IssuesPanelProps) {
  const issuesRef = useRef<HTMLDivElement>(null);
  const firstCompilerWarningRef = useRef<HTMLDivElement>(null);
  const compilerWarningPattern = /custom operations|estimated flops|unsupported/i;

  const errorCount = warnings.filter(w => w.type === 'error').length;
  const warningCount = warnings.filter(w => w.type === 'warning').length;

  useEffect(() => {
    if (jumpToIssuesSignal <= 0) return;
    const target = firstCompilerWarningRef.current ?? issuesRef.current;
    target?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, [jumpToIssuesSignal]);

  useEffect(() => {
    firstCompilerWarningRef.current = null;
  }, [warnings]);

  return (
    <div className="h-full bg-sidebar flex flex-col overflow-hidden">
      <div className="h-10 px-4 flex items-center border-b border-sidebar-border">
        <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Issues
        </span>
      </div>

      <div className="px-3 py-2 border-t border-sidebar-border">
        <div className="flex items-center gap-3 text-xs">
          <div className={cn(
            "flex items-center gap-1 px-2 py-1 rounded",
            errorCount > 0 ? "bg-destructive/10 text-destructive" : "bg-success/10 text-success"
          )}>
            {errorCount > 0 ? (
              <AlertCircle className="w-3.5 h-3.5" />
            ) : (
              <CheckCircle className="w-3.5 h-3.5" />
            )}
            <span>{errorCount} errors</span>
          </div>
          <div className={cn(
            "flex items-center gap-1 px-2 py-1 rounded",
            warningCount > 0 ? "bg-warning/10 text-warning" : "bg-success/10 text-success"
          )}>
            <AlertTriangle className="w-3.5 h-3.5" />
            <span>{warningCount} warnings</span>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-thin">
        <div ref={issuesRef} className="panel-header">Compiler & Validation Issues</div>
        <div className="p-2 space-y-2">
          {warnings.length === 0 ? (
            <div className="text-xs text-muted-foreground/50 text-center py-4">
              No issues detected
            </div>
          ) : (
            warnings.map((warning) => (
              <div
                key={warning.id}
                ref={(el) => {
                  if (!el) return;
                  if (!compilerWarningPattern.test(warning.message)) return;
                  if (!firstCompilerWarningRef.current) {
                    firstCompilerWarningRef.current = el;
                  }
                }}
                className={cn(
                  "p-2.5 rounded-md border text-xs",
                  warning.type === 'error' && "status-error",
                  warning.type === 'warning' && "status-warning",
                  warning.type === 'info' && "bg-primary/10 text-primary border-primary/30"
                )}
              >
                <div className="flex items-start gap-2">
                  {warning.type === 'error' ? (
                    <AlertCircle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
                  ) : warning.type === 'warning' ? (
                    <AlertTriangle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
                  ) : (
                    <CheckCircle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
                  )}
                  <div className="min-w-0">
                    <div className="leading-snug break-words">
                      {warning.code ? (
                        <span className="font-mono text-[10px] opacity-70 mr-1.5">[{warning.code}]</span>
                      ) : null}
                      <span>{warning.message}</span>
                    </div>
                    {warning.hint ? (
                      <div className="mt-1 text-[10px] opacity-75 leading-snug break-words">
                        {warning.hint}
                      </div>
                    ) : null}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
