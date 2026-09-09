/**
 * The run, and the three things a human can do to it.
 *
 * This bar is the reason the Training workspace is a workspace and not a
 * report: everywhere else in NEURAX the user is designing, and nothing is
 * happening on the machine. Here something is, and it has to be visible and
 * stoppable from the same window that started it.
 *
 * Two decisions are load-bearing:
 *
 *  - **`interrupted` is a first-class status, not an error.** It is what the
 *    studio infers when a run's state file still says `running` but the
 *    process that wrote it is gone — a closed laptop, a killed terminal, a
 *    power cut. The run is not lost; the directory holds every step and every
 *    checkpoint. So the button offered is Resume, not Retry.
 *  - **Stop is destructive and asks; Pause is not and does not.** Pause writes
 *    a checkpoint and stops stepping, which loses nothing. Stop ends the run.
 *    They are not the same gesture and are not styled as if they were.
 */
import { CirclePause, CirclePlay, Loader2, Square, TriangleAlert, Check, X } from 'lucide-react';

import { Button } from '@/components/ui/button.tsx';
import { formatDuration } from '@/services/format.ts';
import type { RunState } from '@/types/runtime.ts';
import { cn } from '@/lib/utils.ts';

interface RunControlsProps {
  run: RunState;
  /** Seconds per step observed so far, for the estimate. Absent before the
   *  run has produced enough steps to estimate anything honestly. */
  secondsPerStep?: number;
  /** True when the figures are generated rather than reported by a real
   *  process. Shown next to the status, not as a banner over the page: this
   *  is a property of the run, and the status line is where a run says what
   *  it is. */
  isExample?: boolean;
  onPause: () => void;
  onResume: () => void;
  onStop: () => void;
}

const STATUS_STYLE: Record<RunState['status'], { label: string; dot: string; text: string; icon: typeof Check }> = {
  running: { label: 'Running', dot: 'bg-emerald-500', text: 'text-emerald-600 dark:text-emerald-400', icon: Loader2 },
  paused: { label: 'Paused', dot: 'bg-blue-500', text: 'text-blue-600 dark:text-blue-400', icon: CirclePause },
  interrupted: { label: 'Interrupted', dot: 'bg-amber-500', text: 'text-amber-600 dark:text-amber-400', icon: TriangleAlert },
  finished: { label: 'Finished', dot: 'bg-emerald-600', text: 'text-emerald-600 dark:text-emerald-400', icon: Check },
  failed: { label: 'Failed', dot: 'bg-red-500', text: 'text-red-600 dark:text-red-400', icon: X },
};

export function RunControls({ run, secondsPerStep, isExample, onPause, onResume, onStop }: RunControlsProps) {
  const style = STATUS_STYLE[run.status];
  const Icon = style.icon;
  const pct = run.totalSteps > 0 ? (run.step / run.totalSteps) * 100 : 0;
  const stepsLeft = Math.max(0, run.totalSteps - run.step);
  // The step is what the process counts; the epoch is what the user planned
  // in. Both are shown, because neither answers the other's question: the
  // step says how far through the budget, the epoch how many times the model
  // has now seen the data.
  const epoch = run.stepsPerEpoch ? Math.floor(run.step / run.stepsPerEpoch) : null;
  const totalEpochs = run.stepsPerEpoch ? Math.round(run.totalSteps / run.stepsPerEpoch) : null;
  const isLive = run.status === 'running';
  const canResume = run.status === 'paused' || run.status === 'interrupted';

  return (
    <div className="border-b border-border bg-card px-4 py-2.5">
      <div className="flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-2 min-w-0">
          <span className={cn('w-2 h-2 rounded-full shrink-0', style.dot, isLive && 'animate-pulse')} />
          <span className="text-sm font-semibold text-foreground truncate">{run.name}</span>
          <span className={cn('flex items-center gap-1 text-[11px] font-medium shrink-0', style.text)}>
            <Icon className={cn('w-3 h-3', isLive && 'animate-spin')} />
            {style.label}
          </span>
          {isExample ? (
            <span
              className="rounded-[4px] border border-amber-500/40 bg-amber-500/10 px-1.5 py-0.5 text-[9px] font-mono uppercase tracking-[0.1em] text-amber-600 dark:text-amber-400 shrink-0"
              title="The training runtime is not built yet. The controls, the clock and every view are real; the numbers behind them are generated."
            >
              example
            </span>
          ) : null}
        </div>

        <div className="flex items-baseline gap-2 font-mono tabular-nums shrink-0">
          {epoch !== null && totalEpochs !== null ? (
            <>
              <span className="text-sm font-semibold text-foreground">
                epoch {Math.min(epoch + 1, totalEpochs)}
              </span>
              <span className="text-[11px] text-muted-foreground">of {totalEpochs}</span>
              <span className="text-muted-foreground/40">·</span>
            </>
          ) : null}
          <span className="text-sm font-semibold text-foreground">{run.step.toLocaleString('en-US')}</span>
          <span className="text-[11px] text-muted-foreground">/ {run.totalSteps.toLocaleString('en-US')} steps</span>
          <span className="text-[11px] text-muted-foreground">· {pct.toFixed(1)}%</span>
        </div>

        {secondsPerStep !== undefined && stepsLeft > 0 && isLive ? (
          <span className="text-[11px] font-mono text-muted-foreground shrink-0">
            ~{formatDuration((stepsLeft * secondsPerStep) / 3600)} left
          </span>
        ) : null}

        <div className="flex items-center gap-1.5 ml-auto shrink-0">
          {isLive ? (
            <Button size="sm" variant="outline" onClick={onPause} className="h-7">
              <CirclePause className="w-3.5 h-3.5 mr-1.5" />
              Pause
            </Button>
          ) : null}
          {canResume ? (
            <Button size="sm" onClick={onResume} className="h-7">
              <CirclePlay className="w-3.5 h-3.5 mr-1.5" />
              {run.status === 'interrupted' ? 'Resume from last checkpoint' : 'Resume'}
            </Button>
          ) : null}
          {isLive || run.status === 'paused' ? (
            <Button
              size="sm"
              variant="outline"
              onClick={onStop}
              className="h-7 border-destructive/40 text-destructive hover:bg-destructive/10 hover:text-destructive"
            >
              <Square className="w-3 h-3 mr-1.5 fill-current" />
              Stop
            </Button>
          ) : null}
        </div>
      </div>

      <div className="mt-2 flex items-center gap-3">
        <div className="relative flex-1 h-1 rounded-full bg-muted overflow-hidden">
          <div
            className={cn('h-full rounded-full transition-[width] duration-500', isLive ? 'bg-primary' : style.dot)}
            style={{ width: `${pct}%` }}
          />
          {/* Epoch boundaries. Drawn on the bar rather than listed, because
              "which pass am I on" is a spatial question. */}
          {totalEpochs && totalEpochs > 1
            ? Array.from({ length: totalEpochs - 1 }, (_, i) => (
                <span
                  key={i}
                  className="absolute top-0 h-full w-px bg-background/70"
                  style={{ left: `${((i + 1) / totalEpochs) * 100}%` }}
                />
              ))
            : null}
        </div>
        <span className="text-[10px] font-mono text-muted-foreground/70 shrink-0" title="The run is this directory. Closing the studio does not end it.">
          {run.directory}
        </span>
      </div>

      {run.status === 'interrupted' ? (
        <p className="mt-2 text-[11px] text-amber-600 dark:text-amber-400 leading-relaxed">
          This run's state file still said <span className="font-mono">running</span>, but the process that wrote it is
          gone. Nothing was lost — every step and checkpoint is on disk. Resuming continues from the last checkpoint.
        </p>
      ) : null}
    </div>
  );
}
