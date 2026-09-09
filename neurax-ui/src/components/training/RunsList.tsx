/**
 * Every run this machine has, and the checkpoints that make them resumable.
 *
 * The product claim this view has to make good on is that closing the studio
 * does not end a run, and reopening it does not lose one. That only holds
 * because a run is a directory, not a process: `state.json` says where it got
 * to, and the checkpoints beside it say what it knew. Both survive the studio,
 * the terminal, and the power.
 *
 * `interrupted` is the row that carries the claim. Nothing reports it — the
 * studio infers it when a state file still says `running` but the process that
 * claimed it is gone. It is shown as a resumable session with a checkpoint
 * behind it, not as a failure, because that is what it is.
 *
 * `exactResume` is drawn on every checkpoint rather than assumed. A checkpoint
 * with weights and optimiser state resumes the *training*; only one that also
 * saved the RNG state and the dataloader position resumes the *same run*. A
 * tool promising a reproducible record has to say which of the two it is
 * holding, not quietly offer the weaker one.
 */
import { CircleCheck, CirclePlay, FolderOpen, Save, TriangleAlert, X } from 'lucide-react';

import { Button } from '@/components/ui/button.tsx';
import { formatBytes } from '@/services/format.ts';
import type { Checkpoint, RunState, RunStatus } from '@/types/runtime.ts';
import { cn } from '@/lib/utils.ts';

interface RunsListProps {
  current: RunState | null;
  previous: RunState[];
  checkpoints: Checkpoint[];
  onResume: (run: RunState) => void;
  onOpenDirectory: (run: RunState) => void;
}

const STATUS: Record<RunStatus, { label: string; dot: string; text: string }> = {
  running: { label: 'Running', dot: 'bg-emerald-500', text: 'text-emerald-600 dark:text-emerald-400' },
  paused: { label: 'Paused', dot: 'bg-blue-500', text: 'text-blue-600 dark:text-blue-400' },
  interrupted: { label: 'Interrupted', dot: 'bg-amber-500', text: 'text-amber-600 dark:text-amber-400' },
  finished: { label: 'Finished', dot: 'bg-emerald-600', text: 'text-emerald-600 dark:text-emerald-400' },
  failed: { label: 'Failed', dot: 'bg-red-500', text: 'text-red-600 dark:text-red-400' },
};

function formatWhen(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString('en-GB', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' });
}

function RunRow({
  run,
  isCurrent,
  onResume,
  onOpenDirectory,
}: {
  run: RunState;
  isCurrent: boolean;
  onResume: (run: RunState) => void;
  onOpenDirectory: (run: RunState) => void;
}) {
  const style = STATUS[run.status];
  const pct = run.totalSteps > 0 ? (run.step / run.totalSteps) * 100 : 0;
  const resumable = run.status === 'interrupted' || run.status === 'paused';

  return (
    <div
      className={cn(
        'rounded-lg border p-3',
        isCurrent ? 'border-primary/40 bg-primary/5' : 'border-border bg-card/40',
      )}
    >
      <div className="flex items-center gap-2 flex-wrap">
        <span className={cn('w-1.5 h-1.5 rounded-full shrink-0', style.dot)} />
        <span className="text-[13px] font-semibold text-foreground truncate">{run.name}</span>
        <span className={cn('text-[10px] font-medium uppercase tracking-wider shrink-0', style.text)}>
          {style.label}
        </span>
        {isCurrent ? (
          <span className="rounded-[4px] border border-primary/40 bg-primary/10 px-1.5 py-0.5 text-[9px] font-mono uppercase tracking-[0.1em] text-primary shrink-0">
            open
          </span>
        ) : null}
        <span className="ml-auto text-[10px] font-mono text-muted-foreground/70 shrink-0">
          {formatWhen(run.startedAt)}
        </span>
      </div>

      <div className="mt-2 flex items-center gap-3">
        <div className="flex-1 h-1 rounded-full bg-muted overflow-hidden">
          <div className={cn('h-full rounded-full', style.dot)} style={{ width: `${pct}%` }} />
        </div>
        <span className="text-[10px] font-mono tabular-nums text-muted-foreground shrink-0">
          {run.step.toLocaleString('en-US')} / {run.totalSteps.toLocaleString('en-US')}
        </span>
      </div>

      <div className="mt-2 flex items-center gap-2 flex-wrap">
        <span className="font-mono text-[10px] text-muted-foreground/60 truncate">{run.directory}</span>
        <div className="ml-auto flex items-center gap-1.5 shrink-0">
          <Button size="sm" variant="ghost" className="h-6 px-2 text-[11px]" onClick={() => onOpenDirectory(run)}>
            <FolderOpen className="w-3 h-3 mr-1" />
            Open folder
          </Button>
          {resumable && !isCurrent ? (
            <Button size="sm" variant="outline" className="h-6 px-2 text-[11px]" onClick={() => onResume(run)}>
              <CirclePlay className="w-3 h-3 mr-1" />
              Resume
            </Button>
          ) : null}
        </div>
      </div>

      {run.status === 'interrupted' ? (
        <p className="mt-2 flex items-start gap-1.5 text-[11px] text-amber-600 dark:text-amber-400 leading-relaxed">
          <TriangleAlert className="w-3.5 h-3.5 mt-px shrink-0" />
          <span>Stopped without closing its state file. The directory is intact and the run can be picked up.</span>
        </p>
      ) : null}
      {run.status === 'failed' ? (
        <p className="mt-2 flex items-start gap-1.5 text-[11px] text-red-600 dark:text-red-400 leading-relaxed">
          <X className="w-3.5 h-3.5 mt-px shrink-0" />
          <span>The process ended with an error. Its log is in the run directory.</span>
        </p>
      ) : null}
    </div>
  );
}

export function RunsList({ current, previous, checkpoints, onResume, onOpenDirectory }: RunsListProps) {
  return (
    <div className="grid grid-cols-1 xl:grid-cols-[1fr_minmax(320px,420px)] gap-6">
      <div className="space-y-3">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Sessions on this machine</h3>
        {current ? <RunRow run={current} isCurrent onResume={onResume} onOpenDirectory={onOpenDirectory} /> : null}
        {previous.map((run) => (
          <RunRow key={run.id} run={run} isCurrent={false} onResume={onResume} onOpenDirectory={onOpenDirectory} />
        ))}
        {!current && previous.length === 0 ? (
          <p className="text-[12px] text-muted-foreground py-8 text-center">
            No runs yet. Starting one creates a directory that outlives this window.
          </p>
        ) : null}
      </div>

      <div className="space-y-3">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Checkpoints in the open run
        </h3>
        {checkpoints.length === 0 ? (
          <p className="text-[12px] text-muted-foreground py-8 text-center border border-dashed border-border rounded-lg">
            None written yet.
          </p>
        ) : (
          <div className="border border-border rounded-lg divide-y divide-border/60">
            {checkpoints
              .slice()
              .sort((a, b) => b.step - a.step)
              .map((c) => (
                <div key={c.path} className="p-3">
                  <div className="flex items-center gap-2">
                    <Save className="w-3.5 h-3.5 text-muted-foreground/60 shrink-0" />
                    <span className="text-[12px] font-semibold font-mono tabular-nums text-foreground">
                      step {c.step.toLocaleString('en-US')}
                    </span>
                    <span className="ml-auto text-[10px] font-mono text-muted-foreground">
                      {formatBytes(c.sizeBytes)}
                    </span>
                  </div>
                  <div className="mt-1 font-mono text-[10px] text-muted-foreground/60 truncate">{c.path}</div>
                  <div
                    className={cn(
                      'mt-1.5 flex items-center gap-1.5 text-[10px]',
                      c.exactResume ? 'text-emerald-600 dark:text-emerald-400' : 'text-amber-600 dark:text-amber-400',
                    )}
                  >
                    {c.exactResume ? <CircleCheck className="w-3 h-3" /> : <TriangleAlert className="w-3 h-3" />}
                    {c.exactResume
                      ? 'Exact resume — RNG state and data position saved'
                      : 'Weights and optimiser only — resuming continues training, but not the same run'}
                  </div>
                </div>
              ))}
          </div>
        )}
        <p className="text-[11px] text-muted-foreground leading-relaxed">
          A run is a directory, not a process. Closing NEURAX — or losing power — leaves both the state file and these
          checkpoints on disk, which is what makes the session reopenable rather than merely repeatable.
        </p>
      </div>
    </div>
  );
}
