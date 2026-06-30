import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ArrowUp, ChevronRight, Mic, MessageSquareText, Plus, Sparkles, Wallet, X } from 'lucide-react';
import { Button } from '@/components/ui/button.tsx';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover.tsx';
import { Progress } from '@/components/ui/progress.tsx';
import { Textarea } from '@/components/ui/textarea.tsx';
import { cn } from '@/lib/utils.ts';
import type { AgentRunPlanItem } from '@/components/panels/AgentRunModal.tsx';

type ChatRole = 'user' | 'assistant';

type ChatMessage = {
  id: string;
  role: ChatRole;
  content: string;
  createdAt: number;
};

type AgentSnapshot = {
  family: string;
  nodes: unknown[];
  connections: unknown[];
  groups: unknown[];
  allowed_layer_types: string[];
  allowed_families?: string[];
  catalogue_id?: string;
  catalogue?: unknown[];
  missing_mandatory_fields?: unknown[];
  hw_config?: Record<string, unknown>;
  analysis_warnings?: unknown[];
  active_tab?: string;
};

type AgentToolEvent = {
  name: 'add_node' | 'move_node' | 'connect' | 'done' | string;
  args?: Record<string, unknown>;
};

interface AIChatDrawerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  creditsLeft?: number;
  creditsLimit?: number;
  onAddCredits?: () => void;
  agentBaseUrl?: string;
  getSnapshot?: () => AgentSnapshot;
  onToolEvent?: (tool: AgentToolEvent) => void;
  className?: string;
}

type RunBlock =
  | { kind: 'request'; content: string; ts: number }
  | { kind: 'assistant'; content: string; ts: number }
  | { kind: 'result'; content: string; ts: number }
  | { kind: 'tool'; content: string; ts: number }
  | { kind: 'error'; content: string; ts: number }
  | { kind: 'done'; content: string; ts: number };

export default function AIChatDrawer({
  open,
  onOpenChange,
  creditsLeft,
  creditsLimit,
  onAddCredits,
  agentBaseUrl,
  getSnapshot,
  onToolEvent,
  className,
}: AIChatDrawerProps) {
  const [draft, setDraft] = useState('');
  const [creativity, setCreativity] = useState(0.0);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const listRef = useRef<HTMLDivElement | null>(null);
  const [isSending, setIsSending] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);

  const [runStatus, setRunStatus] = useState<'idle' | 'running' | 'done' | 'error'>('idle');
  const [runRequest, setRunRequest] = useState<string>('');
  const [runPlan, setRunPlan] = useState<AgentRunPlanItem[]>([]);
  const [runBlocks, setRunBlocks] = useState<RunBlock[]>([]);
  const [toolCount, setToolCount] = useState<number>(0);
  const [toolSummaries, setToolSummaries] = useState<string[]>([]);
  const [toolsExpanded, setToolsExpanded] = useState<boolean>(false);

  const [autoRound, setAutoRound] = useState<number>(0);
  const autoRoundRef = useRef<number>(0);
  useEffect(() => {
    autoRoundRef.current = autoRound;
  }, [autoRound]);

  // Pending auto-fix consent — holds detected issues, cleared when user makes a choice
  const [pendingAutoFix, setPendingAutoFix] = useState<{ round: number; issues: string[] } | null>(null);

  // ── Stable ref so auto-round always reads the LATEST snapshot (not a stale closure) ──
  const getSnapshotRef = useRef<(() => AgentSnapshot) | undefined>(getSnapshot);
  useEffect(() => {
    getSnapshotRef.current = getSnapshot;
  }, [getSnapshot]);

  const lastRunRequestRef = useRef<string>('');

  const describeBlock = (b: RunBlock): { title: string; detail: string; tone: 'neutral' | 'error' | 'success' } => {
    if (b.kind === 'request') return { title: 'Request', detail: b.content, tone: 'neutral' };
    if (b.kind === 'assistant') return { title: 'Thinking', detail: b.content, tone: 'neutral' };
    if (b.kind === 'result') return { title: 'Result', detail: b.content, tone: 'success' };
    if (b.kind === 'error') return { title: 'Error', detail: b.content, tone: 'error' };
    if (b.kind === 'done') return { title: 'Done', detail: b.content, tone: 'success' };
    // tool
    const raw = String(b.content ?? '');
    const [name, rest] = raw.split(' ', 2);
    let args: any = null;
    if (rest) {
      try {
        args = JSON.parse(rest);
      } catch {
        args = null;
      }
    }

    const toolName = String(name ?? '').trim();
    if (toolName === 'add_node') {
      const layer = typeof args?.layer_type === 'string' ? args.layer_type : 'layer';
      return { title: 'Add layer', detail: `Added ${layer}.`, tone: 'neutral' };
    }
    if (toolName === 'move_node') {
      return { title: 'Adjust layout', detail: 'Moved a layer on the canvas.', tone: 'neutral' };
    }
    if (toolName === 'connect') {
      return { title: 'Connect', detail: 'Connected two layers.', tone: 'neutral' };
    }
    if (toolName === 'set_family') {
      const fam = typeof args?.family === 'string' ? args.family : '';
      return { title: 'Architecture', detail: fam ? `Switched to ${fam}.` : 'Switched architecture family.', tone: 'neutral' };
    }
    if (toolName === 'set_hw_config') {
      return { title: 'Hyperparameters', detail: 'Updated required hyperparameters.', tone: 'neutral' };
    }
    if (toolName === 'clear_canvas') {
      return { title: 'Reset canvas', detail: 'Cleared the canvas for a fresh build.', tone: 'neutral' };
    }
    if (toolName === 'disconnect') {
      return { title: 'Disconnect', detail: 'Removed a connection between blocks.', tone: 'neutral' };
    }
    if (toolName === 'delete_node') {
      const id = typeof args?.node_id === 'string' ? args.node_id : '';
      return { title: 'Remove block', detail: id ? `Removed block ${id}.` : 'Removed a block from the canvas.', tone: 'neutral' };
    }
    if (toolName === 'navigate_to') {
      const tab = typeof args?.tab === 'string' ? args.tab : '';
      return { title: 'Navigate', detail: tab ? `Switched to ${tab} workspace.` : 'Navigated to another tab.', tone: 'neutral' };
    }
    if (toolName === 'run_analysis') {
      return { title: 'Compile', detail: 'Triggered compiler analysis on the current canvas.', tone: 'neutral' };
    }
    if (toolName === 'select_node') {
      const id = typeof args?.node_id === 'string' ? args.node_id : '';
      return { title: 'Select', detail: id ? `Focused block ${id}.` : 'Selected a block.', tone: 'neutral' };
    }
    if (toolName === 'done') {
      return { title: 'Finalize', detail: 'Finalized the run.', tone: 'success' };
    }
    return { title: 'Action', detail: 'Applied an action.', tone: 'neutral' };
  };

  const onToolEventRef = useRef(onToolEvent);
  useEffect(() => {
    onToolEventRef.current = onToolEvent;
  }, [onToolEvent]);

  const canSend = draft.trim().length > 0 && !isSending;
  const resolvedCreditsLeft = typeof creditsLeft === 'number' ? creditsLeft : 0;
  const resolvedCreditsLimit = typeof creditsLimit === 'number' ? creditsLimit : 0;
  const creditPct = resolvedCreditsLimit > 0 ? Math.max(0, Math.min(100, (resolvedCreditsLeft / resolvedCreditsLimit) * 100)) : 0;

  const subtitle = useMemo(() => {
    if (messages.length === 0) return 'Reason about implementation and changes.';
    const last = messages[messages.length - 1];
    return last.role === 'user' ? 'Thinking…' : 'Ask for the next step.';
  }, [messages]);

  useEffect(() => {
    const el = listRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages.length, open]);

  const summarizeTool = useCallback((evt: AgentToolEvent): string | null => {
    const name = String(evt?.name ?? '');
    const args = (evt as any)?.args ?? {};

    if (!name) return null;
    if (name === 'clear_canvas') return 'Cleared the canvas for a fresh build.';
    if (name === 'set_family') {
      const fam = String(args.family ?? '').trim();
      return fam ? `Switched model family to ${fam.toUpperCase()}.` : 'Switched model family.';
    }
    if (name === 'set_hw_config') return 'Filled minimal hardware defaults.';
    if (name === 'add_node') return 'Added a required building block.';
    if (name === 'connect') return 'Wired blocks to ensure data flows end-to-end.';
    if (name === 'move_node') return 'Adjusted layout for readability.';
    if (name === 'disconnect') return 'Removed a connection between blocks.';
    if (name === 'delete_node') return 'Removed a block from the canvas.';
    if (name === 'navigate_to') {
      const tab = String(args.tab ?? '').trim();
      return tab ? `Navigated to the ${tab} workspace.` : 'Navigated to another tab.';
    }
    if (name === 'run_analysis') return 'Triggered compiler analysis.';
    if (name === 'select_node') return 'Focused a block on the canvas.';
    if (name === 'done') return null;
    return 'Applied a tool action.';
  }, []);

  const uniqueToolSummaries = useMemo(() => {
    const seen = new Set<string>();
    const out: string[] = [];
    for (const s of toolSummaries) {
      if (!s) continue;
      if (seen.has(s)) continue;
      seen.add(s);
      out.push(s);
    }
    return out;
  }, [toolSummaries]);

  const startRun = useCallback(async (overrideContent?: unknown, opts?: { isAuto?: boolean }) => {
    const base = typeof overrideContent === 'string' ? overrideContent : draft;
    const content = String(base ?? '').trim();
    if (!content || isSending) return;

    setRunRequest(content);
    lastRunRequestRef.current = content;
    if (!opts?.isAuto) setAutoRound(0);

    const baseUrl = (agentBaseUrl ?? (import.meta as any).env?.VITE_AGENT_BASE_URL ?? 'http://127.0.0.1:8099') as string;
    // Use ref so the snapshot is always fresh even when called from auto-round closure
    const snapshot = getSnapshotRef.current?.();
    if (!snapshot) {
      const now = Date.now();
      setMessages((prev) => [
        ...prev,
        { id: `u-${now}`, role: 'user', content, createdAt: now },
        {
          id: `a-${now}`,
          role: 'assistant',
          content: 'Agent snapshot is unavailable. Refresh the page and try again.',
          createdAt: now + 1,
        },
      ]);
      setDraft('');
      setRunRequest(content);
      return;
    }

    const { snapshotForSend, catalogueKeyToMarkSent } = (() => {
      const catId = typeof snapshot.catalogue_id === 'string' ? snapshot.catalogue_id : '';
      if (!catId) return { snapshotForSend: snapshot, catalogueKeyToMarkSent: null as string | null };

      const key = `neurax.catalogue.sent.${catId}`;
      const alreadySent = (() => {
        try {
          return window.sessionStorage.getItem(key) === '1';
        } catch {
          return false;
        }
      })();

      if (!alreadySent) {
        return { snapshotForSend: snapshot, catalogueKeyToMarkSent: key };
      }

      const { catalogue, ...rest } = snapshot as any;
      return { snapshotForSend: rest as AgentSnapshot, catalogueKeyToMarkSent: null as string | null };
    })();

    eventSourceRef.current?.close();
    eventSourceRef.current = null;

    const now = Date.now();
    setMessages((prev) => [...prev, { id: `u-${now}`, role: 'user', content, createdAt: now }]);
    setDraft('');
    setIsSending(true);

    setRunStatus('running');
    setRunPlan([]);
    setRunBlocks([]);
    setToolCount(0);
    setToolSummaries([]);
    setToolsExpanded(false);
    setPendingAutoFix(null);  // dismiss any pending consent banner when a new run starts

    setRunBlocks((prev) => [...prev, { kind: 'request', content, ts: Date.now() }]);

    let runId: string | null = null;
    try {
      const resp = await fetch(`${baseUrl.replace(/\/$/, '')}/runs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_message: content, snapshot: snapshotForSend, creativity }),
      });
      if (!resp.ok) throw new Error(await resp.text());
      const data = (await resp.json()) as { run_id?: string };
      runId = data.run_id ?? null;
      if (!runId) throw new Error('Missing run_id');

      if (catalogueKeyToMarkSent) {
        try {
          window.sessionStorage.setItem(catalogueKeyToMarkSent, '1');
        } catch {
          // ignore
        }
      }

      const es = new EventSource(`${baseUrl.replace(/\/$/, '')}/runs/${runId}/events`);
      eventSourceRef.current = es;

      es.addEventListener('assistant', (evt) => {
        try {
          const payload = JSON.parse((evt as MessageEvent).data);
          const content = String(payload?.content ?? '');
          if (content) {
            setRunBlocks((prev) => [...prev, { kind: 'assistant', content, ts: Date.now() }]);
          }
          setRunStatus('running');
        } catch {
          // ignore
        }
      });

      es.addEventListener('plan', (evt) => {
        try {
          const payload = JSON.parse((evt as MessageEvent).data) as { items?: unknown[] };
          const items = Array.isArray(payload.items) ? payload.items : [];
          const normalized: AgentRunPlanItem[] = items
            .map((it: any, i: number) => ({
              id: typeof it?.id === 'string' ? it.id : String(i + 1),
              text: String(it?.text ?? ''),
              status: (it?.status === 'done' || it?.status === 'in_progress' || it?.status === 'pending') ? it.status : 'pending',
            }))
            .filter((x) => x.text.trim().length > 0);
          setRunPlan(normalized);
          setRunStatus('running');
        } catch {
          // ignore
        }
      });

      es.addEventListener('tool', (evt) => {
        try {
          const payload = JSON.parse((evt as MessageEvent).data) as AgentToolEvent;
          onToolEventRef.current?.(payload);
          setToolCount((c) => c + 1);
          const s = summarizeTool(payload);
          if (s) setToolSummaries((prev) => [...prev, s]);
        } catch {
          // ignore
        }
      });

      es.addEventListener('done', () => {
        es.close();
        if (eventSourceRef.current === es) eventSourceRef.current = null;
        setIsSending(false);

        setRunBlocks((prev) => {
          const assistants = prev.filter((b) => b.kind === 'assistant');
          if (assistants.length === 0) {
            return [...prev, { kind: 'done', content: 'Done', ts: Date.now() }];
          }

          const last = assistants[assistants.length - 1] as Extract<RunBlock, { kind: 'assistant' }>;
          const kept = prev.filter((b) => b !== last);
          return [
            ...kept,
            { kind: 'result', content: last.content, ts: Date.now() },
            { kind: 'done', content: 'Done', ts: Date.now() },
          ];
        });
        setRunStatus('done');

        const maxAutoRounds = 2;
        const nextRound = autoRoundRef.current + 1;
        if (nextRound > maxAutoRounds) return;

        window.setTimeout(() => {
          // Use the ref so we get the post-analysis snapshot (stale if we used the closure directly)
          const snap = getSnapshotRef.current?.();
          if (!snap) return;

          const missing = Array.isArray((snap as any)?.missing_mandatory_fields)
            ? ((snap as any).missing_mandatory_fields as unknown[])
            : [];
          const warnings = Array.isArray((snap as any)?.analysis_warnings)
            ? ((snap as any).analysis_warnings as unknown[])
            : [];
          const hasError = warnings.some((w: any) => String(w?.type ?? '').toLowerCase() === 'error');

          if (missing.length > 0 || hasError) {
            // Present consent banner instead of silently starting another pass
            const issueList: string[] = [
              ...missing.map((f: any) => `Missing required field: ${String(f)}`),
              ...(hasError ? ['Analysis errors detected — model may not compile'] : []),
            ];
            setAutoRound(nextRound);
            setPendingAutoFix({ round: nextRound, issues: issueList });
          }
        }, 900);
      });

      es.addEventListener('error', (evt) => {
        try {
          const payload = JSON.parse((evt as MessageEvent).data) as { message?: string };
          const msg = String(payload?.message ?? 'Agent error');
          setRunStatus('error');
          setRunBlocks((prev) => [...prev, { kind: 'error', content: msg, ts: Date.now() }]);
        } catch {
          const msg = String((evt as MessageEvent).data ?? 'Agent error');
          setRunStatus('error');
          setRunBlocks((prev) => [...prev, { kind: 'error', content: msg, ts: Date.now() }]);
        }
      });

      es.onerror = () => {
        es.close();
        if (eventSourceRef.current === es) eventSourceRef.current = null;
        setMessages((prev) => [
          ...prev,
          { id: `a-${Date.now()}`, role: 'assistant', content: 'Agent connection error. Is neurax-agent running?', createdAt: Date.now() },
        ]);
        setIsSending(false);

        setRunStatus('error');
        setRunBlocks((prev) => [...prev, { kind: 'error', content: 'Agent connection error. Is neurax-agent running?', ts: Date.now() }]);
        if (runId && eventSourceRef.current) {
          eventSourceRef.current.close();
          eventSourceRef.current = null;
        }
      };
    } catch (e) {
      const err = e instanceof Error ? e.message : String(e);
      const ts = Date.now();
      setMessages((prev) => [
        ...prev,
        { id: `a-${ts}`, role: 'assistant', content: `Agent error: ${err}`, createdAt: ts },
      ]);
      setIsSending(false);

      setRunStatus('error');
      setRunBlocks((prev) => [...prev, { kind: 'error', content: `Agent error: ${err}`, ts: Date.now() }]);
      if (runId && eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    }
  }, [
    agentBaseUrl,
    draft,
    isSending,
    onToolEvent,
    summarizeTool,
    // getSnapshot intentionally excluded — accessed via getSnapshotRef to avoid stale closures
  ]);

  if (!open) return null;

  return (
    <div
      className={cn(
        'h-full w-full flex flex-col',
        'bg-gradient-to-br from-background via-background/95 to-primary/[0.03]',
        'backdrop-blur-xl',
        className,
      )}
    >
      <div className="px-5 pt-5 pb-4 border-b border-border/40 bg-background/40 backdrop-blur-md sticky top-0 z-10">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <div className="w-9 h-9 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center">
                <MessageSquareText className="w-4.5 h-4.5 text-primary" />
              </div>
              <div className="min-w-0">
                <div className="text-sm font-semibold tracking-tight truncate">Neurax AI</div>
                <div className="text-[11px] text-muted-foreground truncate">{subtitle}</div>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-1">
            <Popover>
              <PopoverTrigger asChild>
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  className="h-8 px-2.5 text-[11px] font-medium"
                  aria-label="Open credits"
                >
                  <Wallet className="w-3.5 h-3.5 mr-1.5" />
                  Credits
                  <span className="ml-1.5 text-muted-foreground">{resolvedCreditsLeft.toFixed(1)} left</span>
                  <ChevronRight className="w-3.5 h-3.5 ml-1" />
                </Button>
              </PopoverTrigger>
              <PopoverContent align="end" sideOffset={8} className="w-72 p-3">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="text-xs font-semibold">Credits</div>
                    <div className="text-xs text-muted-foreground">{resolvedCreditsLeft.toFixed(1)} left</div>
                  </div>
                  <Progress value={creditPct} className="h-2" />
                  <div className="text-[11px] text-muted-foreground">Daily credits used first</div>
                  <div className="flex items-center justify-between gap-2 pt-1">
                    <div className="text-[11px] text-muted-foreground">Low on credits?</div>
                    <Button type="button" size="sm" onClick={onAddCredits} disabled={!onAddCredits}>
                      Add credits
                    </Button>
                  </div>
                </div>
              </PopoverContent>
            </Popover>

            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={() => onOpenChange(false)}
              aria-label="Close chat"
            >
              <X className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </div>

      <div ref={listRef} className="flex-1 overflow-y-auto scrollbar-thin px-4 py-6 space-y-6">
        {runStatus !== 'idle' ? (
          <div className="space-y-4 animate-in fade-in slide-in-from-bottom-2 duration-500">
            <div className="rounded-3xl border border-border/40 bg-card/40 backdrop-blur-sm shadow-xl shadow-primary/5 overflow-hidden">
              <div className="px-5 py-3 border-b border-border/40 bg-muted/20 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {runStatus === 'running' ? (
                    <div className="h-2 w-2 rounded-full bg-primary animate-pulse" />
                  ) : runStatus === 'done' ? (
                    <div className="h-2 w-2 rounded-full bg-success" />
                  ) : (
                    <div className="h-2 w-2 rounded-full bg-destructive" />
                  )}
                  <span className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground/80">Active Session</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <div className="h-1 w-8 rounded-full bg-border/40 overflow-hidden">
                    <div className={cn(
                      "h-full bg-primary transition-all duration-700",
                      runStatus === 'running' ? "w-2/3 animate-pulse" : "w-full"
                    )} />
                  </div>
                </div>
              </div>

              <div className="p-5 space-y-5">
                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center">
                    <MessageSquareText className="w-4 h-4 text-primary" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="text-[10px] font-bold text-muted-foreground/60 uppercase tracking-wider mb-1">Objective</div>
                    <div className="text-[13px] leading-relaxed text-foreground/90 font-medium">
                      {runRequest || '—'}
                    </div>
                  </div>
                </div>

                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-muted/20 border border-border/40 flex items-center justify-center">
                    <ChevronRight className="w-4 h-4 text-muted-foreground" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="text-[10px] font-bold text-muted-foreground/60 uppercase tracking-wider mb-2">Strategy & Plan</div>
                    <div className="space-y-2.5">
                      {runPlan.length > 0 ? (
                        runPlan.map((p) => (
                          <div key={p.id} className="flex items-center gap-3 group">
                            <div
                              className={cn(
                                'h-1.5 w-1.5 rounded-full transition-all duration-300',
                                p.status === 'done' && 'bg-success shadow-[0_0_8px_hsl(var(--success))] scale-125',
                                p.status === 'in_progress' && 'bg-primary animate-pulse shadow-[0_0_8px_hsl(var(--primary))]',
                                p.status === 'pending' && 'bg-muted-foreground/20',
                              )}
                            />
                            <div className={cn(
                              "text-[12px] transition-colors",
                              p.status === 'done' ? "text-muted-foreground/70" : "text-foreground/80"
                            )}>{p.text}</div>
                          </div>
                        ))
                      ) : (
                        <div className="flex items-center gap-2 py-1">
                          <div className="h-1.5 w-1.5 rounded-full bg-primary animate-ping" />
                          <div className="text-[11px] text-muted-foreground italic font-light italic">Orchestrating approach…</div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {runBlocks
                  .filter((b) => b.kind !== 'request' && b.kind !== 'tool')
                  .map((b, idx) => {
                    const d = describeBlock(b);
                    return (
                      <div key={`${b.kind}-${b.ts}-${idx}`} className="flex items-start gap-4 animate-in fade-in slide-in-from-left-1 duration-300">
                        <div
                          className={cn(
                            'flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center border',
                            d.tone === 'error' && 'bg-destructive/10 border-destructive/20 text-destructive',
                            d.tone === 'success' && 'bg-success/10 border-success/20 text-success',
                            d.tone === 'neutral' && 'bg-primary/5 border-primary/10 text-primary',
                          )}
                        >
                          {d.tone === 'error' ? <X className="w-3.5 h-3.5" /> : d.tone === 'success' ? <Sparkles className="w-3.5 h-3.5" /> : <Sparkles className="w-3.5 h-3.5 animate-pulse" />}
                        </div>
                        <div className="min-w-0 flex-1">
                          <div className="text-[10px] font-bold text-muted-foreground/60 uppercase tracking-wider mb-1">{d.title}</div>
                          <div className="text-[12px] leading-relaxed text-foreground/85">{d.detail}</div>
                        </div>
                      </div>
                    );
                  })}
              </div>
            </div>

            {toolCount > 0 && (
              <div className="rounded-2xl border border-border/40 bg-primary/[0.02] backdrop-blur-sm px-5 py-3 transition-all hover:bg-primary/[0.04]">
                <button
                  type="button"
                  className="w-full flex items-center justify-between text-left group"
                  onClick={() => setToolsExpanded((v) => !v)}
                >
                  <div className="flex items-center gap-2 text-[12px] font-semibold text-primary/80">
                    <div className="w-5 h-5 rounded-md bg-primary/10 flex items-center justify-center">
                      <Plus className="w-3 h-3" />
                    </div>
                    {toolCount} Engine {toolCount === 1 ? 'Action' : 'Actions'}
                  </div>
                  <div className="text-muted-foreground/60 group-hover:text-foreground transition-colors text-[10px] font-bold uppercase tracking-widest">{toolsExpanded ? 'Hide' : 'Review'}</div>
                </button>

                {toolsExpanded && uniqueToolSummaries.length > 0 && (
                  <ul className="mt-4 space-y-2 border-t border-border/20 pt-3 animate-in fade-in zoom-in-95 duration-200">
                    {uniqueToolSummaries.map((s, i) => (
                      <li key={`${i}-${s}`} className="text-[12px] text-muted-foreground/80 flex items-start gap-3">
                        <div className="mt-1.5 h-1 w-1 rounded-full bg-primary/40 flex-shrink-0" />
                        <span className="min-w-0 leading-relaxed font-light">{s}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </div>
        ) : messages.length === 0 ? (
          <div className="rounded-xl border border-border/60 bg-secondary/10 p-4">
            <div className="text-xs font-medium">Start a conversation</div>
            <div className="mt-1 text-[11px] text-muted-foreground leading-relaxed">
              Describe what you want to build, then I can propose changes and help you implement them.
            </div>
          </div>
        ) : (
          messages.map((m) => (
            <div
              key={m.id}
              className={cn(
                'max-w-[88%] rounded-2xl px-4 py-3 text-[13px] leading-relaxed shadow-sm transition-all animate-in fade-in slide-in-from-bottom-2 duration-300',
                m.role === 'user'
                  ? 'ml-auto bg-primary text-primary-foreground border-transparent shadow-primary/20 hover:shadow-primary/30'
                  : 'mr-auto bg-card/80 backdrop-blur-sm border border-border/40 text-foreground shadow-sm hover:border-border/80',
              )}
            >
              <div className="whitespace-pre-wrap break-words font-medium">{m.content}</div>
              <div className={cn(
                "mt-1.5 text-[9px] font-bold uppercase tracking-wider",
                m.role === 'user' ? 'text-primary-foreground/60' : 'text-muted-foreground/50'
              )}>
                {new Date(m.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </div>
            </div>
          ))
        )}
      </div>

      {/* ── Auto-fix consent banner ──────────────────────────────────────── */}
      {pendingAutoFix && (
        <div className="mx-4 mb-4 rounded-2xl border border-warning/30 bg-warning/[0.03] backdrop-blur-md p-4 animate-in slide-in-from-bottom-4 duration-500 shadow-lg shadow-warning/5">
          <div className="flex items-start justify-between gap-2 mb-3">
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded-full bg-warning/20 flex items-center justify-center">
                <Sparkles className="w-3.5 h-3.5 text-warning" />
              </div>
              <div className="text-[12px] font-bold text-warning/90 tracking-tight">
                Architectural refinements ready
              </div>
            </div>
            <button
              type="button"
              aria-label="Dismiss"
              className="text-muted-foreground/40 hover:text-foreground transition-colors p-1 rounded-full hover:bg-muted/30"
              onClick={() => setPendingAutoFix(null)}
            >
              <X className="w-4 h-4" />
            </button>
          </div>
          <div className="space-y-1.5 mb-5 px-1">
            {pendingAutoFix.issues.map((issue, idx) => (
              <div key={idx} className="text-[11px] text-muted-foreground/80 flex items-center gap-2 font-medium">
                <div className="h-1 w-1 rounded-full bg-warning/50" />
                {issue}
              </div>
            ))}
          </div>
          <div className="flex gap-2">
            <Button
              type="button"
              size="sm"
              className="h-9 px-4 text-[11px] rounded-xl font-bold bg-warning text-warning-foreground hover:bg-warning/90 shadow-sm transition-all active:scale-95 flex-1"
              onClick={() => {
                setPendingAutoFix(null);
                void startRun(lastRunRequestRef.current, { isAuto: true });
              }}
            >
              Optimize Building
            </Button>
            <Button
              type="button"
              size="sm"
              variant="outline"
              className="h-9 px-4 text-[11px] rounded-xl font-bold border-border/60 hover:bg-background/80 transition-all active:scale-95"
              onClick={() => setPendingAutoFix(null)}
            >
              Handle Manually
            </Button>
          </div>
        </div>
      )}

      <div className="p-5 border-t border-border/40 bg-background/40 backdrop-blur-xl">
        <div
          className={cn(
            'rounded-[24px] border border-border/40 bg-secondary/20 shadow-lg transition-all focus-within:border-primary/40 focus-within:bg-secondary/30 focus-within:shadow-primary/5',
            'px-4 py-3',
          )}
        >
          <div className="min-h-[44px]">
            <Textarea
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="How can Neurax help with your architecture?"
              className={cn(
                'min-h-[44px] max-h-[160px] resize-none',
                'border-0 bg-transparent shadow-none focus-visible:ring-0',
                'px-0 py-1 text-[13px] leading-relaxed placeholder:text-muted-foreground/40 font-medium',
              )}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  startRun();
                }
              }}
            />
          </div>

          <div className="mt-3 flex items-center justify-between gap-2">
            <div className="flex items-center gap-1.5">
              <Button type="button" size="icon" variant="ghost" className="h-8 w-8 rounded-xl hover:bg-primary/10 hover:text-primary transition-colors" aria-label="Add context">
                <Plus className="w-4 h-4" />
              </Button>

              <div className="h-4 w-px bg-border/40 mx-1" />

              <Popover>
                <PopoverTrigger asChild>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="h-8 rounded-xl px-3 text-[11px] font-bold text-primary hover:bg-primary/5 transition-all border border-primary/20 bg-primary/5 backdrop-blur-md"
                  >
                    <Sparkles className="w-3.5 h-3.5 mr-1.5 animate-pulse" />
                    {creativity === 0.0 ? 'Canon' : creativity === 0.4 ? 'Balanced' : creativity === 0.7 ? 'Creative' : 'Research'}
                  </Button>
                </PopoverTrigger>
                <PopoverContent align="start" className="w-[240px] p-2 bg-background/80 backdrop-blur-2xl border-border/40 shadow-2xl rounded-2xl animate-in fade-in zoom-in duration-200">
                  <div className="px-2 py-1.5">
                    <h4 className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground/60 mb-2">Architectural Mode</h4>
                    <div className="flex flex-col gap-1">
                      {[
                        { label: 'Canon', value: 0.0, hint: 'Deterministic & reproducible', desc: 'Strictly follows established architectural patterns.' },
                        { label: 'Balanced', value: 0.4, hint: 'Standard architectural patterns', desc: 'Adapts standard patterns for specific constraints.' },
                        { label: 'Creative', value: 0.7, hint: 'Novel & hybrid arrangements', desc: 'Proposes novel combinations and hybrid layer counts.' },
                        { label: 'Research', value: 1.0, hint: 'Experimental research grade', desc: 'Generates experimental, state-of-the-art architectures.' },
                      ].map(({ label, value, desc }) => (
                        <button
                          key={label}
                          onClick={() => setCreativity(value)}
                          className={cn(
                            'group flex flex-col items-start px-3 py-2.5 rounded-xl text-left transition-all duration-200',
                            creativity === value
                              ? 'bg-primary/10 text-primary border border-primary/20'
                              : 'hover:bg-muted/40 text-muted-foreground'
                          )}
                        >
                          <span className="text-[11px] font-bold">{label}</span>
                          <span className="text-[9px] font-medium opacity-60 leading-tight mt-0.5 group-hover:opacity-100 transition-opacity">
                            {desc}
                          </span>
                        </button>
                      ))}
                    </div>
                  </div>
                </PopoverContent>
              </Popover>

              <div className="h-4 w-px bg-border/40 mx-1" />

              <Button type="button" variant="ghost" size="sm" className="h-8 rounded-xl px-3 text-[11px] font-bold text-muted-foreground/70 hover:text-primary hover:bg-primary/5 transition-all">
                Plan
              </Button>
            </div>

            <div className="flex items-center gap-1.5">
              <Button type="button" size="icon" variant="ghost" className="h-8 w-8 rounded-xl hover:bg-primary/10 hover:text-primary transition-colors" aria-label="Voice input">
                <Mic className="w-4 h-4" />
              </Button>

              <Button
                type="button"
                size="icon"
                disabled={!canSend}
                onClick={startRun}
                className={cn(
                  'h-9 w-9 rounded-2xl transition-all duration-300 active:scale-90',
                  canSend
                    ? 'bg-primary text-primary-foreground shadow-lg shadow-primary/20 hover:shadow-primary/30 hover:-translate-y-0.5'
                    : 'bg-muted/40 text-muted-foreground/40'
                )}
                aria-label="Send message"
              >
                <ArrowUp className="w-4.5 h-4.5" />
              </Button>
            </div>
          </div>
        </div>

        <div className="mt-2.5 px-1 flex items-center justify-between opacity-40">
          <div className="text-[9px] font-bold uppercase tracking-widest text-muted-foreground">Neurax Intelligence Engine v2</div>
          <div className="text-[9px] font-bold uppercase tracking-widest text-muted-foreground">⌘ + ↵ to send</div>
        </div>
      </div>
    </div>
  );
}
