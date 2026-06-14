import { useEffect, useMemo, useRef } from 'react';
import { CheckCircle2, Circle, Loader2, X } from 'lucide-react';
import { cn } from '@/lib/utils.ts';
import { Button } from '@/components/ui/button.tsx';
import { Dialog, DialogContent } from '@/components/ui/dialog.tsx';

export type AgentRunPlanItem = {
  id: string;
  text: string;
  status: 'pending' | 'in_progress' | 'done';
};

export type AgentRunBlock =
  | { kind: 'request'; content: string; ts: number }
  | { kind: 'assistant'; content: string; ts: number }
  | { kind: 'tool'; content: string; ts: number }
  | { kind: 'error'; content: string; ts: number }
  | { kind: 'done'; content: string; ts: number };

interface AgentRunModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title?: string;
  status?: 'idle' | 'planning' | 'running' | 'done' | 'error';
  request?: string;
  plan?: AgentRunPlanItem[];
  blocks?: AgentRunBlock[];
}

export function AgentRunModal({
  open,
  onOpenChange,
  title,
  status,
  request,
  plan,
  blocks,
}: AgentRunModalProps) {
  const listRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const el = listRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [open, blocks?.length]);

  const statusLabel = useMemo(() => {
    if (status === 'planning') return 'Planning';
    if (status === 'running') return 'Running';
    if (status === 'done') return 'Done';
    if (status === 'error') return 'Error';
    return 'Idle';
  }, [status]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className={cn(
          'max-w-none w-[calc(100vw-24px)] h-[calc(100vh-24px)] sm:w-[calc(100vw-48px)] sm:h-[calc(100vh-48px)]',
          'p-0 overflow-hidden',
        )}
      >
        <div className="h-full w-full flex flex-col bg-gradient-to-b from-background via-background to-secondary/10">
          <div className="px-6 py-4 border-b border-border/60 flex items-start justify-between gap-4">
            <div className="min-w-0">
              <div className="text-sm font-semibold tracking-tight truncate">{title ?? 'Neurax AI Run'}</div>
              <div className="mt-1 text-[11px] text-muted-foreground flex items-center gap-2">
                <span className="inline-flex items-center gap-1 rounded-full border border-border/60 bg-secondary/20 px-2 py-0.5">
                  {status === 'running' || status === 'planning' ? (
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  ) : status === 'done' ? (
                    <CheckCircle2 className="w-3.5 h-3.5" />
                  ) : (
                    <Circle className="w-3.5 h-3.5" />
                  )}
                  <span>{statusLabel}</span>
                </span>
                <span className="truncate">Agent execution view</span>
              </div>
            </div>

            <Button type="button" variant="ghost" size="icon" className="h-9 w-9" onClick={() => onOpenChange(false)}>
              <X className="w-4 h-4" />
            </Button>
          </div>

          <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-[420px_1fr]">
            <div className="border-b lg:border-b-0 lg:border-r border-border/60 p-5 space-y-4 overflow-y-auto">
              <div className="rounded-2xl border border-primary/30 bg-primary/10 p-4 shadow-sm">
                <div className="text-xs font-medium text-primary flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-primary" />
                  Request
                </div>
                <div className="mt-2 text-[12px] leading-relaxed whitespace-pre-wrap break-words text-foreground/90">{request || '—'}</div>
              </div>

              <div className="rounded-2xl border border-border/60 bg-secondary/10 p-4">
                <div className="text-xs font-medium text-muted-foreground flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground/60" />
                  Plan
                </div>
                <div className="mt-2 space-y-2">
                  {(plan && plan.length > 0) ? (
                    plan.map((p) => (
                      <div key={p.id} className="flex items-start gap-2">
                        {p.status === 'done' ? (
                          <CheckCircle2 className="w-4 h-4 mt-0.5 text-primary" />
                        ) : p.status === 'in_progress' ? (
                          <Loader2 className="w-4 h-4 mt-0.5 animate-spin text-muted-foreground" />
                        ) : (
                          <Circle className="w-4 h-4 mt-0.5 text-muted-foreground/60" />
                        )}
                        <div className="text-[12px] leading-relaxed whitespace-pre-wrap break-words">{p.text}</div>
                      </div>
                    ))
                  ) : (
                    <div className="text-[11px] text-muted-foreground leading-relaxed">Plan pending…</div>
                  )}
                </div>
              </div>
            </div>

            <div ref={listRef} className="min-h-0 overflow-y-auto p-5">
              {(blocks && blocks.length > 0) ? (
                <div className="rounded-2xl border border-border/60 bg-secondary/10 p-4">
                  <ul className="space-y-2">
                    {blocks.map((b, idx) => (
                      <li key={`${b.kind}-${b.ts}-${idx}`} className="flex items-start gap-2">
                        <div
                          className={cn(
                            'mt-1.5 h-2 w-2 rounded-full',
                            b.kind === 'error' && 'bg-destructive/70',
                            b.kind === 'done' && 'bg-primary/70',
                            b.kind === 'request' && 'bg-primary/80',
                            b.kind === 'assistant' && 'bg-muted-foreground/50',
                            b.kind === 'tool' && 'bg-muted-foreground/40',
                          )}
                        />
                        <div className="min-w-0">
                          <div className="text-[11px] font-medium text-muted-foreground">
                            {b.kind === 'assistant' ? 'Thinking' : b.kind === 'request' ? 'Request' : b.kind}
                          </div>
                          <div className="mt-1 text-[13px] leading-relaxed whitespace-pre-wrap break-words">{b.content}</div>
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : (
                <div className="text-[12px] text-muted-foreground">No run yet. Send a request to start.</div>
              )}
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
