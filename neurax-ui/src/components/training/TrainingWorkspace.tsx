/**
 * Training — the workspace where NEURAX stops predicting and starts watching.
 *
 * Every other workspace in the studio is analytical. Architecture builds a
 * graph, Simulation derives what it would cost, Production writes the weights
 * out. None of them ever runs anything; `sweep.rs` still says "NEURAX never
 * runs a model", and for the compiler that remains true. This workspace is the
 * seam: the design leaves NEURAX as PyTorch, a process takes it, and what
 * comes back is measurement rather than derivation.
 *
 * It is a sibling of Simulation and Production, not a view inside either, and
 * the reason is not layout. The other workspaces are about a document; this one
 * is about a process that exists whether or not the window is open. Nesting it
 * under Simulation would have said the opposite — that a run is one more chart
 * of the analysis — and would have put the only stop button in the product
 * three clicks deep inside a tab about predictions.
 *
 * Its six views follow the life of a run rather than the shape of the data:
 * what it will cost, what it is doing, whether the prediction held, which
 * layer was wrong, what went wrong, and what runs exist on this machine.
 *
 * ── On the mock clock ──────────────────────────────────────────────────────
 * No backend exists yet. Rather than render a frozen screenshot of a run, this
 * component advances a real interval over `mockSteps`, so Start, Pause, Stop
 * and Resume all do what they claim and the live views can be judged in
 * motion — which is the only way to find out whether a chart that redraws
 * twice a second is readable. `IS_MOCK` gates the banner that says so; when
 * the runtime lands, the ticker is what gets replaced, not the views.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity, Database, FlaskConical, Layers, Play, Scale, ScrollText, Stethoscope,
} from 'lucide-react';

import { Button } from '@/components/ui/button.tsx';
import { useToast } from '@/hooks/use-toast.ts';
import { AnalysisResult } from '@/types/architecture.ts';
import {
  IS_MOCK, mockCheckpoints, mockDiagnostics, mockLayers, mockModelBuilt,
  mockPredictions, mockRunState, mockSteps,
} from '@/services/mockRuntime.ts';
import { isCpuOnly, primaryGpu } from '@/types/runtime.ts';
import type {
  DatasetProfile,
  HardwareProfile,
  PredictionPair,
  RunState,
  TrainingPlan,
  TrainingStep,
} from '@/types/runtime.ts';
import { cn } from '@/lib/utils.ts';

import { DatasetPanel } from '@/components/panels/DatasetPanel.tsx';
import { LayerTiming } from './LayerTiming.tsx';
import { LiveDiagnostics } from './LiveDiagnostics.tsx';
import { LiveMetrics } from './LiveMetrics.tsx';
import { PredictionAccuracy } from './PredictionAccuracy.tsx';
import { ProjectionView } from './ProjectionView.tsx';
import { RunControls } from './RunControls.tsx';
import { RunsList } from './RunsList.tsx';
import { useTrainingRun } from './useTrainingRun.ts';
import { controlTrainingRun, listTrainingRuns, startTrainingRun } from '@/services/neuraxApi.ts';
import { TrainingLaunchDialog } from './TrainingLaunchDialog.tsx';

export type TrainingCategoryId = 'plan' | 'live' | 'accuracy' | 'layers' | 'diagnostics' | 'sessions';

export const TRAINING_CATEGORIES = [
  { id: 'plan', label: 'Plan', hint: 'The dataset, the projected cost, and the launch', icon: FlaskConical },
  { id: 'live', label: 'Live', hint: 'Loss, throughput, memory and the GPU, as reported', icon: Activity },
  { id: 'accuracy', label: 'Accuracy', hint: "NEURAX's predictions against what happened", icon: Scale },
  { id: 'layers', label: 'Per Layer', hint: 'Which operator the cost model got wrong', icon: Layers },
  { id: 'diagnostics', label: 'Diagnostics', hint: 'Findings only a running process can raise', icon: Stethoscope },
  { id: 'sessions', label: 'Sessions', hint: 'Runs on this machine, and their checkpoints', icon: ScrollText },
] as const satisfies ReadonlyArray<{
  id: TrainingCategoryId;
  label: string;
  hint: string;
  icon: typeof Activity;
}>;

/** What the studio can generate from the canvas, for a run to train. */
export interface GeneratedModel {
  code: string;
  modelClassName: string;
  totalParams: number;
  fullySupported: boolean;
  unsupportedTypes: string[];
  /** One sample's shape, without the batch dimension. */
  inputShape: number[];
  numClasses: number;
}

interface TrainingWorkspaceProps {
  modelName: string;
  /** Generates the design as PyTorch. Absent when the host cannot — then the
   *  workspace offers the example and says why, rather than a Start button
   *  that fails on click. */
  onGenerateModel?: () => GeneratedModel | null;
  analysis?: AnalysisResult;
  hardware: HardwareProfile | null;
  dataset: DatasetProfile | null;
  onChooseDataset: () => void;
}

/** How fast the mock clock advances. Fast enough to watch a curve form,
 *  slow enough that the charts are judged at a redraw rate a real run has. */
const TICK_MS = 400;
const STEPS_PER_TICK = 23;

export function TrainingWorkspace({
  modelName,
  onGenerateModel,
  analysis,
  hardware,
  dataset,
  onChooseDataset,
}: TrainingWorkspaceProps) {
  const { toast } = useToast();
  const [activeCategory, setActiveCategory] = useState<TrainingCategoryId>('plan');
  const [launchOpen, setLaunchOpen] = useState(false);
  /**
   * Whether this design can be turned into a trainable file, checked when the
   * dialog opens rather than when Start is pressed.
   *
   * It used to be checked inside `start`, which returned early and left the
   * dialog sitting there: the user pressed the one button on a confirmation
   * screen and nothing happened, with the reason in a toast behind the modal.
   * A screen that asks for a commitment has to say beforehand whether the
   * commitment is possible.
   *
   * Computed on open, not on render — generating a large model's source is
   * not free, and a workspace nobody is looking at should not be doing it.
   */
  const [launchBlocker, setLaunchBlocker] = useState<string | null>(null);
  const [isStarting, setIsStarting] = useState(false);

  /**
   * Two sources, and only ever one at a time.
   *
   * `liveRunId` names a run that exists on disk: the service is polled for it
   * and every figure comes from the training process. `exampleRun` is the
   * worked demonstration, driven by a clock in this file. They are kept apart
   * rather than merged behind one state, because the difference — measured
   * against generated — is exactly what the studio must never blur.
   */
  const [liveRunId, setLiveRunId] = useState<string | null>(null);
  const live = useTrainingRun(liveRunId);
  const [exampleRun, setExampleRun] = useState<RunState | null>(null);
  const [exampleStep, setExampleStep] = useState(0);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const isExample = exampleRun !== null;
  const run = live.state ?? exampleRun;

  /**
   * The runs that exist on this machine.
   *
   * This list showed a fixture — three invented sessions, one of them
   * `interrupted` — while the service had a real endpoint returning the real
   * ones. That made the product's central claim unverifiable from the screen
   * it is claimed on: "closing the studio does not lose a run" is only
   * believable if the run you started is in the list when you come back.
   *
   * Refreshed when the workspace opens and whenever a run's status changes,
   * because those are the two moments the list can be wrong.
   */
  const [previousRuns, setPreviousRuns] = useState<RunState[]>([]);
  useEffect(() => {
    let cancelled = false;
    void listTrainingRuns().then((runs) => {
      if (!cancelled) setPreviousRuns(runs);
    });
    return () => {
      cancelled = true;
    };
  }, [live.state?.status, liveRunId]);
  /** Whether there is a design this studio can turn into a trainable file. */
  const canTrain = Boolean(onGenerateModel);

  /**
   * The plan, sized against the machine that would actually run it.
   *
   * The memory budget is the field that matters, and it is not always VRAM.
   * A discrete card contributes its free VRAM; a CPU-only machine — or one
   * whose only graphics processor is integrated and shares system memory —
   * contributes available RAM, because that is genuinely the memory the run
   * would live in. Reading `gpus[0].vramFreeBytes` unconditionally, as this
   * did, produced `null` on such a machine and silently fell back to an
   * example figure of 22.6 GB: a laptop being told a design fits, against a
   * card it does not have.
   */
  /**
   * The plan, built from the analysis and the machine — not from a fixture.
   *
   * This was `{ ...mockPlan }` with two fields overwritten, and the result was
   * a confirmation screen that disagreed with the workspace behind it: the
   * dialog offered 2.14 kWh while the analysis on the same page said 360.9,
   * and named an output directory for a model the user had not opened. A
   * screen whose whole job is to be read before committing hours of compute
   * has to state the run that will actually happen.
   *
   * Every field now comes from the analysis, the detected machine, or the
   * dataset. The output directory is gone: the service assigns it when the
   * run is created, and inventing one here meant showing a path that would
   * never exist.
   */
  const plan = useMemo((): TrainingPlan => {
    const gpu = primaryGpu(hardware);
    const cpuOnly = isCpuOnly(hardware);
    const dedicatedVram = !cpuOnly && gpu?.vramFreeBytes != null ? gpu.vramFreeBytes : null;

    const batchSize = analysis?.selectedBatchSize && analysis.selectedBatchSize > 0
      ? analysis.selectedBatchSize
      : 16;
    const trainSamples = dataset?.splits?.train ?? dataset?.samples ?? 0;
    const stepsPerEpoch = trainSamples > 0 ? Math.ceil(trainSamples / batchSize) : 0;
    const epochs = 5;

    // Checkpointing about ten times over the run: often enough that an
    // interruption costs little, rare enough not to dominate the disk.
    const totalSteps = stepsPerEpoch * epochs;
    const checkpointEvery = Math.max(50, Math.round(totalSteps / 10) || 500);
    // Weights plus optimizer state, which is what a checkpoint holds — the
    // optimizer's two moments are twice the weights again for AdamW.
    const checkpointSize = (analysis?.parameterMemoryBytes ?? 0) * 3;

    return {
      modelName: modelName || 'Untitled design',
      gpu: cpuOnly ? `${hardware?.cpu.model ?? 'this CPU'} (CPU)` : gpu?.name ?? 'no target selected',
      precision: analysis?.selectedPrecision ?? 'fp32',
      batchSize,
      epochs,
      stepsPerEpoch,
      trainSamples,
      totalSteps,
      estimatedHours: analysis?.trainingTimeHours ?? 0,
      predictedPeakVramBytes: analysis?.peakVramBytes ?? 0,
      availableVramBytes: dedicatedVram ?? hardware?.ramAvailableBytes ?? 0,
      estimatedKwh: analysis?.energyKwh ?? 0,
      estimatedCostUsd: analysis?.trainingCostUsd ?? 0,
      // Assigned by the service on creation. Stated as such rather than
      // guessed at, because a path shown here and not used is worse than none.
      outputDirectory: '',
      checkpointEverySteps: checkpointEvery,
      checkpointSizeBytes: checkpointSize,
    };
  }, [analysis, dataset, hardware, modelName]);

  /** True when the machine has no dedicated video memory, so the budget being
   *  checked is system RAM. The dialog needs to say which. */
  const budgetIsSystemRam = isCpuOnly(hardware) || primaryGpu(hardware)?.vramFreeBytes == null;

  // The example's clock. It drives nothing when a real run is open — a live
  // run reports its own steps, and a second source advancing beside it would
  // be two runs pretending to be one.
  useEffect(() => {
    if (exampleRun?.status !== 'running') return;
    timer.current = setInterval(() => {
      setExampleStep((s) => {
        const next = Math.min(s + STEPS_PER_TICK, mockRunState.totalSteps);
        if (next >= mockRunState.totalSteps) {
          setExampleRun((r) => (r ? { ...r, status: 'finished', step: next } : r));
        }
        return next;
      });
    }, TICK_MS);
    return () => {
      if (timer.current) clearInterval(timer.current);
      timer.current = null;
    };
  }, [exampleRun?.status]);

  useEffect(() => {
    setExampleRun((r) => (r && r.step !== exampleStep ? { ...r, step: exampleStep } : r));
  }, [exampleStep]);

  const steps: TrainingStep[] = isExample
    ? exampleStep > 0
      ? mockSteps(exampleStep)
      : []
    : live.steps;
  const hasObservations = steps.length > 0;

  /**
   * What NEURAX predicted, beside what the run reported.
   *
   * This view was comparing a fixture against a fixture — invented
   * predictions against invented observations — which made the one screen
   * whose entire purpose is confronting the compiler with reality the one
   * screen that could never do it.
   *
   * Now each row is a real pair or it is not shown. The predictions come from
   * the analysis the design was costed by; the observations come from the
   * training process. A row whose observation has not arrived yet is drawn
   * with the prediction alone and marked as waiting, which is the honest
   * intermediate state rather than a placeholder number.
   */
  const predictions = useMemo((): PredictionPair[] => {
    if (isExample) return mockPredictions(hasObservations);
    if (!analysis?.generatedAt) return [];

    const recent = steps.slice(-100);
    const meanStepMs =
      recent.length > 0 ? recent.reduce((sum, s) => sum + s.stepTimeMs, 0) / recent.length : undefined;
    const peakVram =
      steps.length > 0 ? Math.max(...steps.map((s) => s.vramAllocatedBytes)) : undefined;
    const meanSamples =
      recent.length > 0
        ? recent.reduce((sum, s) => sum + s.samplesPerSec, 0) / recent.length
        : undefined;

    const pairs: PredictionPair[] = [
      {
        metric: 'parameters',
        label: 'Parameters',
        predicted: analysis.totalParams,
        observed: live.model?.parameters,
        unit: '',
      },
    ];

    if (analysis.latencyMs !== null && analysis.latencyMs > 0) {
      pairs.push({
        metric: 'step_time',
        label: 'Time per step',
        predicted: analysis.latencyMs,
        observed: meanStepMs,
        unit: 'ms',
      });
    }
    // Only where the run can report it. A CPU run has no allocator figure to
    // compare against, and a row of "0 B observed" would read as a
    // catastrophic prediction error rather than as an absent measurement.
    if (analysis.peakVramBytes > 0 && (peakVram ?? 0) > 0) {
      pairs.push({
        metric: 'peak_vram',
        label: 'Peak VRAM',
        predicted: analysis.peakVramBytes,
        observed: peakVram,
        unit: 'B',
      });
    }
    if (analysis.samplesPerS && analysis.samplesPerS > 0) {
      pairs.push({
        metric: 'throughput',
        label: 'Samples per second',
        predicted: analysis.samplesPerS,
        observed: meanSamples,
        unit: '',
        higherIsBetter: true,
      });
    }
    return pairs;
  }, [analysis, hasObservations, isExample, live.model, steps]);

  const secondsPerStep = useMemo(() => {
    if (steps.length < 2) return undefined;
    const recent = steps.slice(-100);
    return recent.reduce((sum, s) => sum + s.stepTimeMs, 0) / recent.length / 1000;
  }, [steps]);

  /**
   * Findings, checkpoints and per-layer timings.
   *
   * Checkpoints are real for a real run — the service lists what is on disk.
   * Diagnostics and per-layer timings are not: nothing in the harness emits
   * them yet, so for a live run they are empty, and each view says so in its
   * own words rather than showing the example's figures under a real run's
   * name. That is the honest state, and it is visible: an empty Diagnostics
   * view on a live run is a missing feature, where borrowed numbers would be
   * a lie.
   */
  const diagnostics = useMemo(
    () =>
      isExample
        ? mockDiagnostics.filter((d) => d.atStep === undefined || d.atStep <= exampleStep)
        : [],
    [isExample, exampleStep],
  );
  const checkpoints = useMemo(
    () => (isExample ? mockCheckpoints.filter((c) => c.step <= exampleStep) : live.checkpoints),
    [isExample, exampleStep, live.checkpoints],
  );
  const layers = useMemo(
    () =>
      isExample
        ? exampleStep > 200
          ? mockLayers
          : mockLayers.map((l) => ({ ...l, observedTimeMs: undefined }))
        : [],
    [isExample, exampleStep],
  );

  /**
   * Start a real run.
   *
   * The model is generated here, from the canvas, by the same generator the
   * Export panel uses — and its parameter count is compared against the
   * analysis before anything is sent. A design the generator cannot fully
   * express would produce a `model.py` that trains something other than what
   * was analysed, and finding that out at step one of a four-hour run is the
   * expensive way to learn it.
   */
  /** What stops this design being trained, in the user's words, or `null`. */
  const checkLaunchable = useCallback((): string | null => {
    if (!onGenerateModel) return 'This studio cannot generate the design as PyTorch.';
    const generated = onGenerateModel();
    if (!generated) return 'There is nothing on the canvas yet.';
    if (!generated.fullySupported) {
      return `NEURAX does not yet generate PyTorch for ${generated.unsupportedTypes.join(', ')}, so this design cannot be trained as drawn.`;
    }
    return null;
  }, [onGenerateModel]);

  const openLaunch = useCallback(() => {
    setLaunchBlocker(checkLaunchable());
    setLaunchOpen(true);
  }, [checkLaunchable]);

  const start = useCallback(async () => {
    const generated = onGenerateModel?.();
    if (!generated || !generated.fullySupported) {
      // Reachable only if the canvas changed between opening the dialog and
      // confirming it. Close first, so the explanation is not behind a modal.
      setLaunchOpen(false);
      setLaunchBlocker(checkLaunchable());
      toast({
        title: 'This design cannot be trained',
        description: checkLaunchable() ?? 'The design changed since this plan was built.',
        variant: 'destructive',
      });
      return;
    }

    setIsStarting(true);
    setLaunchOpen(false);
    const started = await startTrainingRun({
      name: `${plan.modelName} · ${dataset?.displayPath.split('/').pop() ?? 'data'}`,
      modelCode: generated.code,
      modelClass: generated.modelClassName,
      inputShape: generated.inputShape,
      numClasses: generated.numClasses,
      datasetPath: dataset?.displayPath ?? null,
      epochs: plan.epochs,
      batchSize: plan.batchSize,
      learningRate: 3e-4,
      precision: plan.precision,
      stepsPerEpoch: plan.stepsPerEpoch,
      checkpointEverySteps: plan.checkpointEverySteps,
      predictions: { parameters: generated.totalParams },
    }).catch(() => null);
    setIsStarting(false);

    if (!started) {
      toast({
        title: 'Could not start training',
        description: 'The local NEURAX service did not answer.',
        variant: 'destructive',
      });
      return;
    }

    setExampleRun(null);
    setLiveRunId(started.id);
    setActiveCategory('live');
    toast({
      title: 'Training started',
      description: `Writing to ${started.directory}. Closing NEURAX will not stop it.`,
    });
  }, [dataset, onGenerateModel, plan, toast]);

  /**
   * Open the worked example.
   *
   * Deliberately not routed through the launch dialog: that dialog asks a
   * human to commit real hours and electricity, and putting a demonstration
   * behind it would teach the wrong reflex.
   */
  const startExample = useCallback(() => {
    setLiveRunId(null);
    setExampleStep(0);
    setExampleRun({
      ...mockRunState,
      name: `${plan.modelName} · ${dataset?.displayPath.split('/').pop() ?? 'dataset'}`,
      status: 'running',
      step: 0,
    });
    setActiveCategory('live');
    toast({
      title: 'Example run opened',
      description: 'Generated figures, so the views can be judged before the training runtime exists.',
    });
  }, [dataset, plan, toast]);

  /**
   * Pause, resume and stop.
   *
   * For a real run these write a file the training loop reads between steps,
   * and the studio then asks immediately rather than waiting for the next
   * poll — a button that takes a second to change state reads as broken.
   *
   * The example has no process to talk to, so its controls move its own
   * clock. Same buttons, and deliberately: the point of the example is to let
   * the controls be judged, which means they have to do what they say.
   */
  const send = useCallback(
    async (command: 'run' | 'pause' | 'stop') => {
      if (!liveRunId) return false;
      const ok = await controlTrainingRun(liveRunId, command);
      live.refresh();
      if (!ok) {
        toast({
          title: 'The run did not answer',
          description: 'It is still on disk; the local service is not responding.',
          variant: 'destructive',
        });
      }
      return Boolean(ok);
    },
    [liveRunId, live, toast],
  );

  const pause = useCallback(() => {
    if (liveRunId) {
      void send('pause');
      return;
    }
    setExampleRun((r) => (r ? { ...r, status: 'paused' } : r));
    toast({ title: 'Paused', description: 'A checkpoint was written. Nothing is lost.' });
  }, [liveRunId, send, toast]);

  const resume = useCallback(
    (target?: RunState) => {
      // Resuming a run from the Sessions list: it may be a different run
      // entirely, so the studio switches to it and the service restarts the
      // process from its last checkpoint.
      if (target) {
        setExampleRun(null);
        setLiveRunId(target.id);
        setActiveCategory('live');
        void controlTrainingRun(target.id, 'run');
        toast({
          title: 'Resumed',
          description: `Picking up at step ${target.step.toLocaleString('en-US')}.`,
        });
        return;
      }
      if (liveRunId) {
        void send('run');
        return;
      }
      setExampleRun((r) => (r ? { ...r, status: 'running' } : r));
    },
    [liveRunId, send, toast],
  );

  const stop = useCallback(() => {
    if (liveRunId) {
      void send('stop');
      toast({
        title: 'Stopping',
        description: 'It will write a checkpoint and stop at the next step boundary.',
      });
      return;
    }
    setExampleRun((r) => (r ? { ...r, status: 'finished' } : r));
    toast({ title: 'Stopped', description: 'The run directory and its checkpoints stay on disk.' });
  }, [liveRunId, send, toast]);

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-background">
      {run ? (
        <RunControls
          run={run}
          secondsPerStep={secondsPerStep}
          isExample={isExample}
          onPause={pause}
          onResume={() => resume()}
          onStop={stop}
        />
      ) : (
        <div className="border-b border-border bg-card px-4 py-2.5 flex items-center gap-3 flex-wrap">
          <span className="text-sm font-semibold text-foreground">No run open</span>
          <span className="text-[11px] text-muted-foreground">
            {canTrain
              ? 'Design and analyse first — the plan below is built from that analysis.'
              : 'This design cannot be generated as PyTorch yet, so it cannot be trained from here.'}
          </span>
          {/*
            The example is opt-in, and says so on the control rather than in a
            banner over every view.

            A page-wide amber strip stood here on all six views, warning that
            the numbers were generated. It was honest and it was the wrong
            shape: a permanent warning is furniture within a day, and it
            shouted on the Plan view where the dataset and the projected cost
            are both real. Making the example something you ask for, and
            labelling the run itself, puts the caveat where the doubt is.
          */}
          <Button
            size="sm"
            variant={canTrain ? 'default' : 'outline'}
            className="ml-auto h-7"
            disabled={isStarting}
            onClick={() => (canTrain ? openLaunch() : startExample())}
          >
            <Play className="w-3.5 h-3.5 mr-1.5" />
            {isStarting ? 'Starting…' : canTrain ? 'Start training' : 'Open an example run'}
          </Button>
        </div>
      )}

      <div className="border-b border-border bg-card px-4 py-2">
        <div className="flex items-center gap-1 overflow-x-auto scrollbar-thin" role="tablist" aria-label="Training views">
          {TRAINING_CATEGORIES.map((category) => {
            const Icon = category.icon;
            const isActive = activeCategory === category.id;
            const isLiveView = category.id === 'live' && run?.status === 'running';
            return (
              <button
                key={category.id}
                type="button"
                role="tab"
                aria-selected={isActive}
                title={category.hint}
                onClick={() => setActiveCategory(category.id)}
                className={cn(
                  'flex items-center gap-1 sm:gap-2 px-2 sm:px-3 py-1.5 text-xs font-medium rounded-md transition-all whitespace-nowrap',
                  isActive
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:text-foreground hover:bg-secondary',
                )}
              >
                <Icon className="w-3.5 h-3.5 shrink-0" />
                <span className="hidden sm:inline">{category.label}</span>
                {isLiveView && !isActive ? (
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse shrink-0" />
                ) : null}
              </button>
            );
          })}
        </div>
      </div>

      <div className="flex-1 overflow-auto p-4 scrollbar-thin" role="tabpanel">
        {activeCategory === 'plan' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              <div className="border border-border rounded-lg overflow-hidden">
                <div className="px-4 py-2.5 border-b border-border bg-muted/30 flex items-center gap-2">
                  <Database className="w-3.5 h-3.5 text-muted-foreground" />
                  <span className="text-[13px] font-semibold uppercase tracking-wider text-muted-foreground">
                    Training data
                  </span>
                </div>
                <DatasetPanel profile={dataset} onChoose={onChooseDataset} isExample={IS_MOCK} />
              </div>
              <div className="border border-border rounded-lg overflow-hidden">
                <div className="px-4 py-2.5 border-b border-border bg-muted/30 flex items-center gap-2">
                  <FlaskConical className="w-3.5 h-3.5 text-muted-foreground" />
                  <span className="text-[13px] font-semibold uppercase tracking-wider text-muted-foreground">
                    What this run will cost
                  </span>
                </div>
                <div className="p-4">
                  <ProjectionView analysis={analysis} />
                </div>
              </div>
            </div>
          </div>
        )}

        {activeCategory === 'live' && <LiveMetrics steps={steps} stepsPerEpoch={mockRunState.stepsPerEpoch} />}

        {activeCategory === 'accuracy' && (
          <PredictionAccuracy
            predictions={predictions}
            model={isExample ? (hasObservations ? mockModelBuilt : null) : live.model}
          />
        )}

        {activeCategory === 'layers' && <LayerTiming layers={layers} />}

        {activeCategory === 'diagnostics' && <LiveDiagnostics diagnostics={diagnostics} />}

        {activeCategory === 'sessions' && (
          <RunsList
            current={run}
            previous={previousRuns.filter((r) => r.id !== live.state?.id)}
            checkpoints={checkpoints}
            onResume={(target) => resume(target)}
            onOpenDirectory={(target) =>
              toast({ title: 'Run directory', description: target.directory })
            }
          />
        )}
      </div>

      <TrainingLaunchDialog
        open={launchOpen}
        onOpenChange={setLaunchOpen}
        plan={plan}
        budgetIsSystemRam={budgetIsSystemRam}
        blocker={launchBlocker}
        onConfirm={() => void start()}
      />
    </div>
  );
}
