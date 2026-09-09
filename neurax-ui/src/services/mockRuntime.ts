/**
 * Stand-in data for the machine, the dataset and a training run.
 *
 * **Nothing here talks to anything.** It exists so the whole surface can be
 * built and judged before the Rust side has a single endpoint — and, more
 * usefully, so the shapes in `types/runtime.ts` get exercised by real
 * components rather than agreed on paper.
 *
 * Every figure below is plausible rather than invented at random: the memory
 * split, the step times and the throughput are the orders of magnitude a
 * GPT-2-class model on a 24 GB card actually produces, so a reader can tell
 * whether the layout survives real numbers. Where a value would be a lie if
 * mistaken for a measurement — a prediction's error, a class imbalance — it is
 * marked as example data in the UI, never presented as the user's own.
 *
 * Replacing this module is the whole of the wiring work: same names, same
 * shapes, real sources.
 */
import type {
  Checkpoint,
  DatasetKind,
  DatasetProfile,
  HardwareProfile,
  LayerComparison,
  LiveDiagnostic,
  ModelBuiltEvent,
  PredictionPair,
  RunState,
  TrainingPlan,
  TrainingRun,
  TrainingStep,
} from '@/types/runtime.ts';

/** Marks every value in this module for the UI, so example data can never be
 *  mistaken for a reading taken from the user's own machine. */
export const IS_MOCK = true;

const GB = 1024 ** 3;

// ─── Hardware ──────────────────────────────────────────────────────────────

export const mockHardware: HardwareProfile = {
  detectedAt: '2026-09-08T08:45:00Z',
  os: 'Pop!_OS 22.04 LTS',
  hostname: 'workstation',
  cpu: {
    model: 'AMD Ryzen 9 7950X',
    vendor: 'AuthenticAMD',
    cores: 16,
    threads: 32,
    architecture: 'x86_64',
    baseMhz: 4500,
  },
  ramTotalBytes: 64 * GB,
  ramAvailableBytes: 41 * GB,
  diskAvailableBytes: 892 * GB,
  gpus: [
    {
      name: 'NVIDIA GeForce RTX 4090',
      vendor: 'NVIDIA',
      recognised: true,
      integrated: false,
      backend: 'cuda',
      vramTotalBytes: 24 * GB,
      vramFreeBytes: 22.6 * GB,
      driverVersion: '560.35.03',
      temperatureC: 41,
      utilisationPct: 3,
      powerWatts: 28,
    },
  ],
  compute: null,
  notes: [],
};

/** A card the database holds no specification for. */
export const mockHardwareUnknownGpu: HardwareProfile = {
  ...mockHardware,
  gpus: [
    {
      ...mockHardware.gpus[0],
      name: 'Some Unreleased Card',
      recognised: false,
      driverVersion: undefined,
    },
  ],
  notes: ['NEURAX holds no published specification for at least one detected accelerator.'],
};

/**
 * A machine with no accelerator at all.
 *
 * This is not an edge case to be tolerated — it is the machine NEURAX was
 * first tested on, and it is what a great many users have. The integrated
 * graphics processor reports no VRAM because it has none of its own: it
 * borrows system memory, which is already counted as RAM.
 */
export const mockHardwareCpuOnly: HardwareProfile = {
  detectedAt: '2026-09-08T08:45:00Z',
  os: 'Pop!_OS 22.04 LTS',
  hostname: 'pop-os',
  cpu: {
    model: 'Intel(R) Core(TM) i5-8365U CPU @ 1.60GHz',
    vendor: 'GenuineIntel',
    cores: 4,
    threads: 8,
    architecture: 'x86_64',
    baseMhz: 1600,
  },
  ramTotalBytes: 25000636416,
  ramAvailableBytes: 4 * GB,
  diskAvailableBytes: 34 * GB,
  gpus: [
    {
      name: 'Intel WhiskeyLake-U GT2 [UHD Graphics 620]',
      vendor: 'Intel',
      recognised: false,
      integrated: true,
      backend: 'vulkan',
      vramTotalBytes: null,
      vramFreeBytes: null,
    },
  ],
  compute: {
    device: 'Intel(R) Core(TM) i5-8365U CPU @ 1.60GHz',
    gflopsF32: 29.57,
    memoryBandwidthGbs: 11.69,
    threads: 8,
    measuredAt: '2026-09-08T22:37:22Z',
    durationMs: 512,
    method: 'register-blocked f32 GEMM 1024x1024x1024 × 3 (avx2+fma), STREAM triad 96 MB × 5, best of each, 8 threads',
  },
  notes: [
    'The graphics processor found is integrated: it shares system memory with the CPU, so it has no VRAM budget of its own.',
  ],
};

export const mockImageDataset: DatasetProfile = {
  kind: 'image-folder',
  displayPath: '~/datasets/leaf-disease',
  profiledAt: new Date().toISOString(),
  samples: 47_312,
  storageBytes: 18.4 * GB,
  sampleShape: [3, 224, 224],
  classes: [
    { label: 'healthy', count: 14_820 },
    { label: 'rust', count: 9_104 },
    { label: 'scab', count: 8_733 },
    { label: 'blight', count: 6_201 },
    { label: 'mildew', count: 4_918 },
    { label: 'mosaic', count: 3_536 },
  ],
  splits: { train: 37_849, validation: 4_731, test: 4_732 },
  integrity: {
    unreadableFiles: 3,
    corruptHeaders: 1,
    inconsistentShapes: 214,
    missingValues: 0,
    notes: [
      '214 images are not 224×224 and will be resized.',
      '4 files could not be read and will be skipped.',
    ],
  },
  suggested: {
    numClasses: 6,
    inChannels: 3,
    imgHeight: 224,
    imgWidth: 224,
    datasetSize: 47_312,
    familyHint: 'cnn',
  },
  fingerprint: 'sha256:3f7a…c91d',
};

export const mockCsvDataset: DatasetProfile = {
  kind: 'csv',
  displayPath: '~/datasets/churn.csv',
  profiledAt: new Date().toISOString(),
  samples: 128_400,
  storageBytes: 41 * 1024 * 1024,
  sampleShape: [23],
  classes: [
    { label: 'retained', count: 104_912 },
    { label: 'churned', count: 23_488 },
  ],
  columns: [
    { name: 'tenure_months', type: 'numeric', missingCount: 0 },
    { name: 'monthly_charges', type: 'numeric', missingCount: 41 },
    { name: 'contract_type', type: 'categorical', missingCount: 0, distinctValues: 3 },
    { name: 'churned', type: 'categorical', missingCount: 0, distinctValues: 2 },
  ],
  integrity: {
    unreadableFiles: 0,
    corruptHeaders: 0,
    inconsistentShapes: 0,
    missingValues: 41,
    notes: ['`monthly_charges` has 41 missing values (0.03 %).'],
  },
  suggested: {
    numClasses: 2,
    datasetSize: 128_400,
    familyHint: 'transformer',
  },
  fingerprint: 'sha256:8b2e…41af',
};

/**
 * A profile for what the user actually picked.
 *
 * The selection is real — the path is the file or folder chosen from the OS
 * dialog, and the kind comes from its extension — while the statistics are
 * generated, because nothing reads the file yet. Keeping the real half real
 * matters: a picker that discarded the choice and always showed the same
 * example would teach the wrong thing about what this control does.
 *
 * An unrecognised extension returns a profile that says so and suggests
 * nothing, rather than quietly defaulting to a table.
 */
export function profileForSelection(selection: {
  displayPath: string;
  kind: DatasetKind;
  fileCount: number;
}): DatasetProfile {
  const base =
    selection.kind === 'image-folder'
      ? mockImageDataset
      : selection.kind === 'csv' || selection.kind === 'jsonl'
        ? mockCsvDataset
        : null;

  if (!base) {
    return {
      kind: 'unknown',
      displayPath: selection.displayPath,
      profiledAt: new Date().toISOString(),
      samples: 0,
      storageBytes: 0,
      integrity: {
        unreadableFiles: 0,
        corruptHeaders: 0,
        inconsistentShapes: 0,
        missingValues: 0,
        notes: ['NEURAX does not recognise this extension, so nothing was read from it.'],
      },
      suggested: {},
      fingerprint: 'not profiled',
    };
  }

  // A folder selection knows its real file count; a single file does not
  // reveal its row count without being read, so the example's stands.
  const samples = selection.kind === 'image-folder' ? selection.fileCount : base.samples;
  const scale = base.samples > 0 ? samples / base.samples : 1;

  return {
    ...base,
    kind: selection.kind,
    displayPath: selection.displayPath,
    profiledAt: new Date().toISOString(),
    samples,
    storageBytes: Math.round(base.storageBytes * scale),
    classes: base.classes?.map((c) => ({ ...c, count: Math.max(1, Math.round(c.count * scale)) })),
    splits: base.splits
      ? {
          train: Math.round(base.splits.train * scale),
          validation: Math.round(base.splits.validation * scale),
          test: Math.round(base.splits.test * scale),
        }
      : undefined,
    suggested: { ...base.suggested, datasetSize: samples },
  };
}

// ─── Training ──────────────────────────────────────────────────────────────

/**
 * The run is described in epochs, and the step count falls out of it.
 *
 * That is the direction the arithmetic actually runs: a user decides to make
 * five passes over their data, and how many optimizer steps that costs depends
 * on how big the dataset is and how many samples fit in a batch. Stating a
 * round step count first — 12 000, say — and leaving the user to work out what
 * fraction of an epoch it lands on is the version that reads as a machine
 * talking to itself.
 */
export const TRAIN_BATCH_SIZE = 16;
export const TRAIN_EPOCHS = 5;
export const STEPS_PER_EPOCH = Math.ceil(
  (mockImageDataset.splits?.train ?? mockImageDataset.samples) / TRAIN_BATCH_SIZE,
);
export const TOTAL_STEPS = STEPS_PER_EPOCH * TRAIN_EPOCHS;

/** Zero-based epoch a step falls in. */
export function epochOfStep(step: number): number {
  return Math.floor((step - 1) / STEPS_PER_EPOCH);
}

export const mockModelBuilt: ModelBuiltEvent = {
  parameters: 123_633_408,
  device: 'cuda:0',
  torchVersion: '2.4.1',
  dtype: 'bfloat16',
};

/** What NEURAX predicted, beside what the run is observing.
 *  The observed values are absent until a run has actually produced them. */
export function mockPredictions(observed: boolean): PredictionPair[] {
  const pairs: PredictionPair[] = [
    { metric: 'parameters', label: 'Parameters', predicted: 123_633_408, unit: '', observed: observed ? 123_633_408 : undefined },
    { metric: 'peak_vram', label: 'Peak VRAM', predicted: 1.45 * GB, unit: 'B', observed: observed ? 1.49 * GB : undefined },
    { metric: 'step_time', label: 'Time per step', predicted: 13.13, unit: 'ms', observed: observed ? 13.9 : undefined },
    { metric: 'tflops', label: 'Effective TFLOPS', predicted: 130.3, unit: '', observed: observed ? 124.1 : undefined, higherIsBetter: true },
    { metric: 'duration', label: 'Total duration', predicted: 4.45, unit: 'h', observed: observed ? 4.71 : undefined },
    { metric: 'energy', label: 'Energy', predicted: 2.14, unit: 'kWh', observed: observed ? 2.21 : undefined },
    { metric: 'fragmentation', label: 'Fragmentation', predicted: 10, unit: '%', observed: observed ? 13 : undefined },
  ];
  return pairs;
}

export const mockLayers: LayerComparison[] = [
  { id: 'emb', type: 'token_embedding', predictedParams: 38_597_376, predictedFlops: 0, predictedTimeMs: 0.4, observedTimeMs: 0.4 },
  { id: 'attn', type: 'mha_attention', predictedParams: 2_362_368, predictedFlops: 1.58e10, predictedTimeMs: 5.0, observedTimeMs: 5.1 },
  { id: 'mlp', type: 'ffn_standard', predictedParams: 4_722_432, predictedFlops: 3.17e10, predictedTimeMs: 7.0, observedTimeMs: 7.8 },
  { id: 'ln', type: 'layernorm', predictedParams: 1_536, predictedFlops: 0, predictedTimeMs: 0.2, observedTimeMs: 0.2 },
];

export const mockDiagnostics: LiveDiagnostic[] = [
  {
    severity: 'warning',
    code: 'L002',
    message:
      'The GPU is waiting on data 38 % of each step. The dataset is 18.4 GB and reads are the limit, not compute — more loader workers or a faster disk would raise throughput.',
    atStep: 1_240,
  },
  {
    severity: 'hint',
    code: 'L004',
    message: 'Observed VRAM is 2.8 % above the prediction. Within the usual band for activation estimates.',
    atStep: 200,
  },
];

export const mockCheckpoints: Checkpoint[] = [
  { step: 2_000, path: 'checkpoints/step_02000.pt', sizeBytes: 742 * 1024 * 1024, savedAt: '2026-09-08T09:14:22Z', exactResume: true },
  { step: 4_000, path: 'checkpoints/step_04000.pt', sizeBytes: 742 * 1024 * 1024, savedAt: '2026-09-08T09:41:05Z', exactResume: true },
];

/**
 * A synthetic training curve.
 *
 * Loss falls the way a real one does — fast at first, then slowly, with noise
 * that never fully settles — because a curve that descends smoothly makes a
 * chart look right for the wrong reasons and hides whether the rendering
 * survives a jagged series.
 *
 * The noise is hashed from the step index rather than summed from sines. Sine
 * noise is periodic, and at the density these charts draw the period becomes a
 * visible sawtooth — a curve that reads as generated at a glance, which
 * defeats the point of judging the surface against something plausible. The
 * hash is still deterministic, so the same step always gives the same value
 * and a re-render never redraws a different history.
 */
/** Deterministic [0,1) from an integer — a cheap integer hash, not a PRNG with
 *  state, so any step can be evaluated on its own. */
function jitter(seed: number): number {
  let h = Math.imul(seed ^ 0x9e3779b9, 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}
export function mockSteps(upTo: number): TrainingStep[] {
  const out: TrainingStep[] = [];
  for (let s = 1; s <= upTo; s += 1) {
    const progress = s / TOTAL_STEPS;
    const base = 0.35 + 4.1 * Math.exp(-progress * 6);
    // Two scales of noise: step-to-step jitter, and a slower drift that makes
    // the curve wander rather than sit on the trend line.
    const noise = (jitter(s) - 0.5) * 0.09 + (jitter(Math.floor(s / 37)) - 0.5) * 0.05;
    const stepTime = 13.9 + Math.sin(s * 0.005) * 0.5 + (jitter(s + 7919) - 0.5) * 0.7;
    out.push({
      step: s,
      epoch: epochOfStep(s),
      loss: Math.max(0.2, base + noise),
      // Validation is a pass over the held-out split, so it happens at an
      // epoch boundary — not every round number of steps.
      valLoss: s % STEPS_PER_EPOCH === 0 ? Math.max(0.25, base + 0.08) : undefined,
      learningRate: 3e-4 * Math.min(1, s / 500) * (1 - progress * 0.8),
      gradNorm: 0.9 + Math.sin(s * 0.3) * 0.25,
      stepTimeMs: stepTime,
      dataTimeMs: stepTime * (0.3 + Math.sin(s * 0.002) * 0.07 + (jitter(s + 104729) - 0.5) * 0.05),
      samplesPerSec: (16 * 1000) / stepTime,
      vramAllocatedBytes: 1.49 * GB + (jitter(s + 1299709) - 0.5) * 0.03 * GB,
      vramReservedBytes: 1.71 * GB,
      gpuUtilisationPct: 88 + Math.sin(s * 0.004) * 5 + (jitter(s + 15485863) - 0.5) * 6,
      // Temperature ramps and then plateaus; it is the one series with real
      // thermal inertia, so it stays smooth where the others are jagged.
      temperatureC: 64 + 8 * (1 - Math.exp(-s / 900)) + (jitter(s + 32452843) - 0.5) * 1.2,
      powerWatts: 310 + Math.sin(s * 0.003) * 18 + (jitter(s + 49979687) - 0.5) * 22,
    });
  }
  return out;
}

export const mockRunState: RunState = {
  id: '2026-09-08-gpt2-small',
  name: 'GPT-2 Small · leaf-disease',
  status: 'running',
  startedAt: '2026-09-08T08:52:11Z',
  lastHeartbeat: new Date().toISOString(),
  pid: 48_211,
  step: 4_217,
  totalSteps: TOTAL_STEPS,
  stepsPerEpoch: STEPS_PER_EPOCH,
  directory: '~/neurax/runs/2026-09-08-gpt2-small',
};

export function mockRun(step = 4_217): TrainingRun {
  return {
    state: { ...mockRunState, step },
    model: mockModelBuilt,
    steps: mockSteps(step),
    predictions: mockPredictions(true),
    diagnostics: mockDiagnostics,
    layers: mockLayers,
    checkpoints: mockCheckpoints,
  };
}

/** Earlier runs, for the sessions list. `interrupted` is the one that matters:
 *  a run whose state file still says `running` while its process is gone. */
export const mockPreviousRuns: RunState[] = [
  {
    id: '2026-09-07-cnn-baseline',
    name: 'CNN baseline · leaf-disease',
    status: 'finished',
    startedAt: '2026-09-07T14:02:00Z',
    step: 8_000,
    totalSteps: 8_000,
    directory: '~/neurax/runs/2026-09-07-cnn-baseline',
  },
  {
    id: '2026-09-06-wide-cnn',
    name: 'Wide CNN · leaf-disease',
    status: 'interrupted',
    startedAt: '2026-09-06T19:30:00Z',
    step: 3_140,
    totalSteps: 8_000,
    directory: '~/neurax/runs/2026-09-06-wide-cnn',
  },
  {
    id: '2026-09-05-oom-attempt',
    name: 'Deep transformer · leaf-disease',
    status: 'failed',
    startedAt: '2026-09-05T11:10:00Z',
    step: 340,
    totalSteps: 12_000,
    directory: '~/neurax/runs/2026-09-05-oom-attempt',
  },
];

export const mockPlan: TrainingPlan = {
  modelName: 'GPT-2 Small',
  gpu: 'RTX 4090',
  precision: 'bf16',
  batchSize: TRAIN_BATCH_SIZE,
  epochs: TRAIN_EPOCHS,
  stepsPerEpoch: STEPS_PER_EPOCH,
  trainSamples: mockImageDataset.splits?.train ?? mockImageDataset.samples,
  totalSteps: TOTAL_STEPS,
  estimatedHours: 4.45,
  predictedPeakVramBytes: 1.45 * GB,
  availableVramBytes: 22.6 * GB,
  estimatedKwh: 2.14,
  estimatedCostUsd: 0.42,
  outputDirectory: '~/neurax/runs/2026-09-08-gpt2-small',
  checkpointEverySteps: 500,
  checkpointSizeBytes: 742 * 1024 * 1024,
};

// Formatting moved to `format.ts`: it is shared with surfaces that hold
// nothing simulated, and this module has to stay deletable.
export { formatBytes, formatCount, formatDuration } from './format.ts';
