/**
 * The last screen before the machine starts working.
 *
 * Everything NEURAX has done until this point is free and reversible: analysis
 * costs nothing, a bad design is deleted. Training is the first action that
 * spends real hours, real electricity and real disk, so it is the first that
 * asks. This dialog exists to be read, not clicked through — its job is to put
 * the whole cost on one screen while it is still cheap to change your mind.
 *
 * The VRAM line is the one check that can block. It compares the predicted
 * peak against what is *free*, not what the card has: the compositor, the
 * browser and this very webview hold VRAM, and a plan sized against the 24 GB
 * printed on the box dies at the first step. Where NEURAX can tell in advance
 * that a run will not fit, letting it start anyway would waste minutes to
 * report what was knowable in milliseconds — the whole argument for an
 * analytical compiler, applied to its own launch button.
 */
import { CircleAlert, CircleCheck, Clock, FileCode, FolderOpen, HardDrive, Play, Repeat, Save, Zap } from 'lucide-react';

import { Button } from '@/components/ui/button.tsx';
import {
  Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle,
} from '@/components/ui/dialog.tsx';
import { formatBytes, formatDuration } from '@/services/format.ts';
import type { TrainingPlan } from '@/types/runtime.ts';
import { cn } from '@/lib/utils.ts';

interface TrainingLaunchDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  plan: TrainingPlan;
  /** True when the machine has no dedicated video memory, so the budget being
   *  checked is system RAM. Every memory label changes: telling a laptop user
   *  their design "fits in VRAM" names a thing their machine does not have. */
  budgetIsSystemRam?: boolean;
  /** Why this design cannot be trained, if it cannot. Stated here rather than
   *  discovered on click: a confirmation screen whose one button silently
   *  does nothing is worse than one that says why up front. */
  blocker?: string | null;
  /**
   * Who wrote the `model.py` about to be trained.
   *
   * Shown because "the assistant wrote this" is a different sentence from
   * "NEURAX translated your canvas", even though both reach this screen by
   * passing the same check. Someone committing hours of their machine to a
   * run is entitled to know which one they are looking at.
   */
  modelSource?: 'generator' | 'assistant' | 'you' | null;
  onConfirm: () => void;
}

function Row({
  icon: Icon,
  label,
  value,
  detail,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  detail?: string;
}) {
  return (
    <div className="flex items-baseline gap-3 py-1.5">
      <Icon className="w-3.5 h-3.5 shrink-0 text-muted-foreground/60 translate-y-0.5" />
      <span className="text-[11px] text-muted-foreground w-[128px] shrink-0">{label}</span>
      <span className="text-[13px] font-semibold font-mono tabular-nums text-foreground">{value}</span>
      {detail ? <span className="text-[11px] text-muted-foreground/70">{detail}</span> : null}
    </div>
  );
}

export function TrainingLaunchDialog({
  open,
  onOpenChange,
  plan,
  budgetIsSystemRam = false,
  blocker = null,
  modelSource = null,
  onConfirm,
}: TrainingLaunchDialogProps) {
  const memoryLabel = budgetIsSystemRam ? 'Peak memory' : 'Peak VRAM';
  const memoryKind = budgetIsSystemRam ? 'system memory' : 'VRAM';
  const headroom = plan.availableVramBytes - plan.predictedPeakVramBytes;
  const fits = headroom > 0;
  // Under 10 % headroom a run can still start and die later on fragmentation
  // or a longer batch. Said plainly rather than shown as a green tick.
  const tight = fits && headroom / plan.availableVramBytes < 0.1;
  const checkpointCount = Math.floor(plan.totalSteps / plan.checkpointEverySteps);
  const totalCheckpointBytes = checkpointCount * plan.checkpointSizeBytes;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[540px]">
        <DialogHeader>
          <DialogTitle>Start training {plan.modelName}?</DialogTitle>
          <DialogDescription>
            This is the first thing NEURAX does that costs something. Everything below is predicted from the design —
            the run will report what it actually was.
          </DialogDescription>
        </DialogHeader>

        <div className="border-y border-border py-2 my-1">
          <Row icon={Zap} label="On" value={plan.gpu} detail={`${plan.precision} · batch ${plan.batchSize}`} />
          <Row
            icon={Repeat}
            label="Passes over data"
            value={`${plan.epochs} epochs`}
            detail={`${plan.stepsPerEpoch.toLocaleString('en-US')} steps each — ${plan.trainSamples.toLocaleString('en-US')} samples ÷ batch ${plan.batchSize}`}
          />
          <Row
            icon={Clock}
            label="Duration"
            value={formatDuration(plan.estimatedHours)}
            detail={`${plan.totalSteps.toLocaleString('en-US')} steps in total`}
          />
          <Row
            icon={HardDrive}
            label={memoryLabel}
            value={formatBytes(plan.predictedPeakVramBytes)}
            detail={`of ${formatBytes(plan.availableVramBytes)} free`}
          />
          <Row
            icon={Zap}
            label="Energy"
            value={`${plan.estimatedKwh.toFixed(2)} kWh`}
            detail={`about $${plan.estimatedCostUsd.toFixed(2)} of electricity`}
          />
          <Row
            icon={Save}
            label="Checkpoints"
            value={`every ${plan.checkpointEverySteps.toLocaleString('en-US')}`}
            detail={`${checkpointCount} × ${formatBytes(plan.checkpointSizeBytes)} = ${formatBytes(totalCheckpointBytes)}`}
          />
          {modelSource ? (
            <Row
              icon={FileCode}
              label="Model code"
              value={
                modelSource === 'assistant'
                  ? 'Written by the assistant'
                  : modelSource === 'you'
                    ? 'Edited by you, in Code'
                    : 'Translated from your canvas'
              }
              detail={
                modelSource === 'generator'
                  ? 'parameter count cross-checked against the analysis'
                  : 'built by PyTorch and matched against the analysis before this screen'
              }
            />
          ) : null}
          {plan.outputDirectory ? (
            <Row icon={FolderOpen} label="Writes to" value={plan.outputDirectory} />
          ) : (
            <Row
              icon={FolderOpen}
              label="Writes to"
              value="~/neurax/runs"
              detail="a new directory, named when the run starts"
            />
          )}
        </div>

        {blocker ? (
          <div className="rounded-lg border border-red-500/40 bg-red-500/5 p-3 text-[12px] leading-relaxed text-red-700 dark:text-red-400">
            <div className="flex items-center gap-1.5 font-semibold">
              <CircleAlert className="w-3.5 h-3.5" />
              This design cannot be trained yet
            </div>
            <p className="mt-1 text-foreground/80">{blocker}</p>
          </div>
        ) : null}

        <div
          className={cn(
            'rounded-lg border p-3 text-[12px] leading-relaxed',
            !fits
              ? 'border-red-500/40 bg-red-500/5 text-red-700 dark:text-red-400'
              : tight
                ? 'border-amber-500/40 bg-amber-500/5 text-amber-700 dark:text-amber-400'
                : 'border-emerald-500/30 bg-emerald-500/5 text-emerald-700 dark:text-emerald-400',
          )}
        >
          <div className="flex items-center gap-1.5 font-semibold">
            {fits ? <CircleCheck className="w-3.5 h-3.5" /> : <CircleAlert className="w-3.5 h-3.5" />}
            {!fits
              ? 'This will not fit'
              : tight
                ? `Fits, with ${formatBytes(headroom)} to spare`
                : `Fits — ${formatBytes(headroom)} of headroom`}
          </div>
          <p className="mt-1 text-foreground/80">
            {!fits
              ? `The predicted peak is ${formatBytes(-headroom)} more than this machine has free in ${memoryKind}. Reduce the batch size, lower the precision, or enable gradient checkpointing — all of them are one change away in Architecture, and none of them costs a run to find out.`
              : tight
                ? 'Enough to start, but not enough to be sure it finishes: fragmentation and a longer batch can both close a gap this size mid-run.'
                : budgetIsSystemRam
                ? 'Measured against free system memory: this machine has no dedicated video memory, so the model would live in RAM alongside everything else running.'
                : 'Measured against free VRAM, not the card total — the desktop and this window are already holding some.'}
          </p>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={onConfirm} disabled={!fits || blocker !== null}>
            <Play className="w-3.5 h-3.5 mr-1.5" />
            Start training
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
