/**
 * The contracts between the studio and the machine it runs on.
 *
 * Nothing here is implemented yet on the backend. These types are written
 * first on purpose: they are the API the Rust side will have to satisfy, and
 * fixing their shape now is what lets the whole surface be built and judged
 * before a single endpoint exists.
 *
 * Three rules are encoded in the shapes themselves rather than left to
 * documentation:
 *
 *  - **A hardware profile can be partial.** A machine with no discrete GPU, an
 *    unrecognised card, a driver that will not answer — all are normal states,
 *    not errors. Every field that can be absent is optional, and
 *    `recognised` says plainly whether NEURAX matched the card to a spec it
 *    knows, because an unrecognised GPU silently falling back to a generic
 *    profile is exactly the kind of confident wrong number this project keeps
 *    removing.
 *
 *  - **A dataset profile carries structure and statistics, never content.**
 *    There is no field anywhere below that can hold a sample, a row, or a
 *    pixel. `fingerprint` is a digest.
 *
 *  - **A run is a directory, not a process.** `RunState` is what a `state.json`
 *    on disk would say, which is what makes a run survive the studio being
 *    closed, crashing, or the machine losing power.
 */

// ─── Hardware ──────────────────────────────────────────────────────────────

export type ComputeBackend = 'cuda' | 'rocm' | 'metal' | 'vulkan' | 'cpu';

/**
 * These mirror `neurax-probe`'s `MachineProfile` field for field.
 *
 * They were written before the probe existed, as a guess at what detection
 * would return, and the guess was wrong in one way that matters: every VRAM
 * figure was typed as a required `number`. The first machine the probe ran on
 * has an integrated Intel graphics processor, which has no VRAM of its own —
 * it borrows system memory — so the probe correctly reports `null`, and the
 * UI would have divided a null by 2^30 and displayed a confident "0.0 GB".
 *
 * So the optionality here is not defensive typing. It is the difference
 * between "this card has no dedicated memory" and "this card has none left",
 * and those two must not render the same way.
 */
export interface GpuInfo {
  /** Name as the driver reports it, verbatim — not necessarily one NEURAX
   *  holds a specification for. */
  name: string;
  vendor: string;
  /** True when the name matched a spec in `neurax-hardware-db`. When false,
   *  every derived throughput figure is an approximation and the UI says so. */
  recognised: boolean;
  /** Soldered beside the CPU and sharing its memory. Changes the reading of
   *  every memory figure: an integrated GPU's "VRAM" is system RAM. */
  integrated: boolean;
  backend: ComputeBackend;
  /** Absent for an integrated part, which has no dedicated memory to report,
   *  and for any driver that will not answer. Never treat absent as zero. */
  vramTotalBytes: number | null;
  /** What is free right now — the number that governs whether a design
   *  starts. Always lower than the total: the compositor, the browser and
   *  NEURAX's own window all hold some. */
  vramFreeBytes: number | null;
  driverVersion?: string | null;
  temperatureC?: number | null;
  utilisationPct?: number | null;
  powerWatts?: number | null;
}

export interface CpuInfo {
  model: string;
  vendor: string;
  cores: number;
  threads: number;
  architecture: string;
  baseMhz?: number | null;
  /** Instruction sets the chip advertises — `avx2`, `fma`, `avx512f`. Most of
   *  the gap between a fast kernel and a slow one, and the reason a measured
   *  throughput is worth reading beside them. */
  features?: string[];
}

/**
 * What this machine sustains, measured rather than looked up.
 *
 * The hardware database holds published peaks for the accelerators it knows.
 * For everything else — an integrated GPU, a laptop CPU, a part released
 * after the database was written — there is no published figure at all, and
 * latency would simply be unavailable. So it is measured: a blocked matrix
 * multiply for arithmetic, a STREAM triad for memory.
 *
 * `method` is carried so a figure can be judged rather than trusted blindly.
 * It names the kernel, the instruction set it ran under and the thread count,
 * which is what separates "this machine is slow" from "this measurement used
 * a baseline kernel".
 */
export interface ComputeMeasurement {
  device: string;
  gflopsF32: number;
  memoryBandwidthGbs: number;
  threads: number;
  measuredAt: string;
  durationMs: number;
  method: string;
}

export interface HardwareProfile {
  detectedAt: string;
  os: string;
  hostname?: string | null;
  cpu: CpuInfo;
  ramTotalBytes: number;
  ramAvailableBytes: number;
  diskAvailableBytes: number;
  /** Empty on a machine with no accelerator — a normal state, and the one
   *  this was first tested on. */
  gpus: GpuInfo[];
  /** Absent until a measurement has been asked for. Detection is free and
   *  runs on load; measuring costs half a second and is explicit. */
  compute?: ComputeMeasurement | null;
  /** What the probe could not read, named. A partial profile is never an
   *  error, but its gaps are stated rather than filled with a plausible
   *  default. */
  notes: string[];
}

/**
 * The accelerator the analysis should target, or `null` on a CPU-only
 * machine.
 *
 * Prefers a discrete part over an integrated one: a laptop with both is
 * asking about the discrete card, and an integrated GPU sharing system RAM is
 * almost never the thing anyone trains on. Mirrors `MachineProfile::
 * primary_gpu` so the two sides cannot disagree about which card is "the" one.
 */
export function primaryGpu(profile: HardwareProfile | null): GpuInfo | null {
  if (!profile) return null;
  return profile.gpus.find((g) => !g.integrated) ?? profile.gpus[0] ?? null;
}

/** True when nothing but the CPU is available to compute on. */
export function isCpuOnly(profile: HardwareProfile | null): boolean {
  if (!profile) return false;
  return profile.gpus.length === 0 || profile.gpus.every((g) => g.integrated);
}

// ─── Dataset ───────────────────────────────────────────────────────────────

export type DatasetKind = 'csv' | 'image-folder' | 'jsonl' | 'unknown';

export type ColumnType = 'numeric' | 'categorical' | 'text' | 'unknown';

export interface CsvColumn {
  name: string;
  type: ColumnType;
  missingCount: number;
  /** Populated for categorical columns only. */
  distinctValues?: number;
}

export interface ClassCount {
  label: string;
  count: number;
}

/** What NEURAX could not read, or read inconsistently. Never a reason to fail
 *  outright — a dataset with a few unreadable files is still profilable, and
 *  the user needs to know rather than be blocked. */
export interface IntegrityReport {
  unreadableFiles: number;
  corruptHeaders: number;
  inconsistentShapes: number;
  missingValues: number;
  notes: string[];
}

/** The half of the profile that matters most: what it lets NEURAX fill in for
 *  the user. These map onto `HardwareConfig`'s mandatory fields, which is why
 *  a real profile empties `missing_mandatory_fields` on its own. */
export interface SuggestedConfig {
  numClasses?: number;
  inChannels?: number;
  imgHeight?: number;
  imgWidth?: number;
  seqLen?: number;
  datasetSize?: number;
  /** A hint, not a decision — the agent still chooses. */
  familyHint?: string;
}

export interface DatasetProfile {
  kind: DatasetKind;
  /** Shown to the user so they can confirm they picked the right thing. Never
   *  sent anywhere. */
  displayPath: string;
  profiledAt: string;
  samples: number;
  storageBytes: number;
  /** `[channels, height, width]` for images, `[features]` for tabular. */
  sampleShape?: number[];
  classes?: ClassCount[];
  columns?: CsvColumn[];
  splits?: { train: number; validation: number; test: number };
  integrity: IntegrityReport;
  suggested: SuggestedConfig;
  /** A digest, never a sample. Identifies the dataset in a run record. */
  fingerprint: string;
}

// ─── Training ──────────────────────────────────────────────────────────────

/**
 * What a run's `state.json` says.
 *
 * `interrupted` is the state nothing reports — it is inferred when the file
 * says `running` but the process that claimed it is gone. Detecting it is why
 * the state file records a pid *and* a start time: pids are reused, and
 * attaching to whatever now holds 12345 would be worse than noticing nothing.
 */
export type RunStatus = 'running' | 'paused' | 'interrupted' | 'finished' | 'failed';

export interface RunState {
  id: string;
  name: string;
  status: RunStatus;
  startedAt: string;
  lastHeartbeat?: string;
  pid?: number;
  step: number;
  totalSteps: number;
  /**
   * Optimizer steps in one pass over the training split — `ceil(train
   * samples / batch size)`.
   *
   * The step is what the process counts, but the epoch is what the user
   * thinks in, and the two are only relatable through this number. It is
   * absent for a run over a stream with no defined end, where there is no
   * such thing as an epoch and inventing one would be a lie.
   */
  stepsPerEpoch?: number;
  /** Directory that *is* the session. Reopening the studio reads this. */
  directory: string;
}

/** One line of the training process's own output, one per step. */
export interface TrainingStep {
  step: number;
  /** Zero-based pass over the training split this step belongs to. Carried on
   *  the step rather than recomputed in the UI, because only the process knows
   *  whether the loader actually restarted — a resumed run does not begin its
   *  epoch where the arithmetic would put it. */
  epoch?: number;
  loss: number;
  valLoss?: number;
  learningRate: number;
  gradNorm?: number;
  stepTimeMs: number;
  /** Time the GPU spent waiting for data. The most common bottleneck, and
   *  the one other tools bury inside total step time. */
  dataTimeMs: number;
  samplesPerSec: number;
  vramAllocatedBytes: number;
  vramReservedBytes: number;
  gpuUtilisationPct?: number;
  temperatureC?: number;
  powerWatts?: number;
}

/**
 * The first thing a training process reports, before a single step.
 *
 * It carries the real parameter count, which confronts NEURAX's prediction in
 * about a second — no GPU time spent, no epoch run. If the two disagree, the
 * run should stop here: a formula is wrong, and finding that out now costs
 * nothing.
 */
export interface ModelBuiltEvent {
  parameters: number;
  device: string;
  torchVersion: string;
  dtype: string;
}

/** One predicted figure beside what actually happened. */
export interface PredictionPair {
  metric: string;
  label: string;
  predicted: number;
  observed?: number;
  unit: string;
  /** Lower is better for memory and time; for throughput it is the reverse.
   *  The UI needs to know which direction is good before colouring an error. */
  higherIsBetter?: boolean;
}

export type DiagnosticSeverity = 'critical' | 'warning' | 'hint';

/** Live equivalents of the compiler's static diagnostics (E001, H001, H008).
 *  Same discipline: name the cause, not just the symptom. */
export interface LiveDiagnostic {
  severity: DiagnosticSeverity;
  code: string;
  message: string;
  atStep?: number;
}

export interface TrainingRun {
  state: RunState;
  model: ModelBuiltEvent | null;
  steps: TrainingStep[];
  predictions: PredictionPair[];
  diagnostics: LiveDiagnostic[];
  /** Per-layer, from the compiler's own `params_per_layer` / `flops_per_layer`
   *  / `latency_per_layer` — the column nobody else can show. */
  layers: LayerComparison[];
  checkpoints: Checkpoint[];
}

export interface LayerComparison {
  id: string;
  type: string;
  predictedParams: number;
  predictedFlops: number;
  observedTimeMs?: number;
  predictedTimeMs?: number;
}

export interface Checkpoint {
  step: number;
  path: string;
  sizeBytes: number;
  savedAt: string;
  /** Whether RNG state and dataloader position were saved too. Without them a
   *  resume continues the training but is not the same run, and a product that
   *  promises a reproducible record has to say which it is. */
  exactResume: boolean;
}

/** What the confirmation dialog shows before a human starts anything. */
export interface TrainingPlan {
  modelName: string;
  gpu: string;
  precision: string;
  batchSize: number;
  /** Passes over the training split. This is the number a user sets; the step
   *  count below is derived from it and the dataset, not the other way round. */
  epochs: number;
  stepsPerEpoch: number;
  /** Size of the training split the two figures above are derived from, so the
   *  arithmetic can be shown rather than asserted. */
  trainSamples: number;
  totalSteps: number;
  estimatedHours: number;
  predictedPeakVramBytes: number;
  availableVramBytes: number;
  estimatedKwh: number;
  estimatedCostUsd: number;
  outputDirectory: string;
  checkpointEverySteps: number;
  checkpointSizeBytes: number;
}
