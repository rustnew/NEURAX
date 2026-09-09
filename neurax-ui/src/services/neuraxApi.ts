/**
 * Neurax Service API Client
 * Auto-generated from OpenAPI 3.0.3 spec — neurax-service v0.1.0
 *
 * Change NEURAX_API_BASE to point at your backend.
 */

// ─── Configuration ────────────────────────────────────────────────

import { resolveApiBase } from '@/services/desktopRuntime.ts';

function normalizeLocalApiBase(rawBase: string): string {
  try {
    const parsed = new URL(rawBase);
    if (parsed.hostname === 'localhost') {
      parsed.hostname = '127.0.0.1';
    }
    return parsed.toString().replace(/\/$/, '');
  } catch {
    return rawBase.replace(/\/$/, '');
  }
}

/**
 * Base URL of the NEURAX service.
 *
 * In the desktop application the service runs in-process on a port the OS
 * assigns at launch, so its address cannot be a build-time constant;
 * `resolveApiBase` prefers the value the host injected and otherwise keeps the
 * browser's behaviour exactly as it was.
 */
const NEURAX_API_BASE = resolveApiBase(
  normalizeLocalApiBase(import.meta.env.VITE_NEURAX_API_URL ?? 'http://127.0.0.1:9098'),
);

/**
 * Base URL of NEURAX's public hosted API — used only for endpoints that must
 * be reachable by someone other than whoever is running this code, like a
 * public share link.
 *
 * Deliberately does NOT go through `resolveApiBase`. On desktop that
 * resolves to the embedded, loopback-only service (see
 * `desktopRuntime.ts`'s `isDesktop`/`resolveApiBase` — the desktop's own
 * server only ever binds `127.0.0.1`), which cannot serve a link to anyone
 * but the machine it runs on. A share created from the desktop app has to
 * reach the same public API a browser session would, not the local instance
 * that dies the moment the app closes.
 */
const NEURAX_HOSTED_API_BASE = normalizeLocalApiBase(
  import.meta.env.VITE_NEURAX_API_URL ?? 'http://127.0.0.1:9098',
);

let accessToken: string | null = null;

export function setNeuraxAccessToken(token: string | null) {
  accessToken = token;
}

// NEURAX's account is a local profile (see AuthContext), not a session
// issued by an identity server — there's nothing real to forward here.
// `setNeuraxAccessToken` exists for callers (tests, or a deployment with its
// own auth in front of neurax-service) that want to override this; nothing
// in the app itself calls it, so this fixed placeholder is what every
// request actually sends today.
async function getAccessToken(): Promise<string | null> {
  return accessToken ?? 'dev-token';
}

import type {
  Checkpoint,
  HardwareProfile,
  ModelBuiltEvent,
  RunState,
  TrainingStep,
} from '@/types/runtime.ts';

// ─── Types (from OpenAPI schemas) ─────────────────────────────────

export interface AnalyzeEnvOverrides {
  hardware?: string;
  precision?: string;
  batch_size?: number;
  seq_len?: number;
  confidence_min?: number;
  no_variants?: boolean;
  parallel_scan?: boolean;
}

export interface AnalyzeRequest {
  topology: Record<string, unknown>;
  env?: AnalyzeEnvOverrides;
}

export interface AnalyzeResponse {
  report: Record<string, unknown>;
}

export interface HardwareDetail {
  name: string;
  manufacturer: string;
  memory_gb: number;
  memory_bandwidth_gbs: number;
  tflops_fp64: number;
  tflops_fp32: number;
  tflops_fp16: number;
  tflops_bf16: number;
  tflops_int8: number;
  tflops_fp8: number;
  tensor_cores: boolean;
  nvlink: boolean;
  nvlink_bandwidth_gbs: number;
  tdp_watts: number;
  launch_year: number;
}

// ─── Time Machine (compiler-backed multi-year projection) ─────────

/** What-if scenario params (snake_case → matches Rust TimeMachineParams) */
export interface TimeMachineParams {
  growth_rate_pct: number;
  horizon_years: number;
  annual_budget_usd: number;
  hardware_track: 'a100' | 'h200' | 'b100';
  start_year?: number;
}

export interface TmScenarioPoint {
  year: number;
  nominal: number;
  optimistic: number;
  pessimistic: number;
  breakingPoint: boolean;
  migration?: string;
  hardwareEvent?: string;
}

export interface TmCostBreakdownPoint {
  year: number;
  compute: number;
  storage: number;
  network: number;
  egress: number;
}

export interface TmCarbonPoint {
  year: number;
  baseline: number;
  optimized: number;
  withGreenRegions: number;
}

export interface TmRecommendation {
  title: string;
  description: string;
  savings: string;
  timing: string;
  priority: string;
}

export interface TmSummary {
  totalCostNominalUsd: number;
  firstBreakYear?: number;
  baseMonthlyUsd: number;
  costGrowthRatio: number;
  hardwareTrack: string;
}

export interface TimeMachineProjection {
  timeline: TmScenarioPoint[];
  costBreakdown: TmCostBreakdownPoint[];
  carbon: TmCarbonPoint[];
  recommendations: TmRecommendation[];
  summary: TmSummary;
}

export interface TimeMachineRequest {
  topology: Record<string, unknown>;
  params?: TimeMachineParams;
}

export interface TimeMachineResponse {
  projection: TimeMachineProjection;
}

export interface HealthResponse {
  status: string;
}

export interface MeResponse {
  user_id: string;
  plan: 'free' | 'essential' | 'architect' | 'elite';
}

export interface PluginValidateRequest {
  plugin: Record<string, unknown>;
}

export interface PluginValidateResponse {
  ok: boolean;
}

export interface BillingCheckoutRequest {
  plan: 'essential' | 'architect' | 'elite';
  interval: 'month' | 'year';
  success_url: string;
  cancel_url: string;
}

export interface BillingUrlResponse {
  url: string;
}

export interface StartAsyncResponse {
  session_id: string;
}

export interface SessionStatusResponse {
  status: string;
  report: Record<string, unknown> | null;
}

export interface PresetMetadata {
  id: string;
  name: string;
  family: string;
  description: string;
  tags: string[];
  node_count: number;
  connection_count: number;
}

export interface PresetNode {
  id: string;
  type: string;
  name: string;
  x: number;
  y: number;
  params: Record<string, unknown>;
}

export interface PresetConnection {
  id: string;
  from: string;
  to: string;
}

export interface PresetFull extends PresetMetadata {
  nodes: any[];
  connections: any[];
}

// ─── Inference Intelligence ───────────────────────────────────────────────────

export interface InferenceParams {
  temperature: number;
  top_k: number;
  top_p: number;
  beam_width: number;
  repetition_penalty: number;
  presence_penalty: number;
  frequency_penalty: number;
  prompt_length: number;
  max_output_tokens: number;
  sliding_window: boolean;
  kv_cache_reuse: boolean;
  architecture_family: string;
  attention_type: string;
  moe_router_mode?: string;
  quantization_level: string;
  long_context_simulation: boolean;
  adversarial_prompt: boolean;
  high_temperature_mode: boolean;
  low_temperature_mode: boolean;
}

export type StabilityLevel = 'stable' | 'drift' | 'unstable' | 'chaotic';
export type InferenceRiskLevel = 'low' | 'medium' | 'high';

/**
 * What the compiler knew about the model a report was computed for.
 *
 * Every field is optional: a report simulating sampling behaviour alone (no
 * design connected) carries no profile at all. When it does, this is the
 * evidence that `hallucination_risk`, `context_degradation`, `router_stability`
 * and `kv_cache` describe *this* design rather than an assumed default.
 */
export interface ModelProfile {
  total_parameters?: number;
  num_layers?: number;
  hidden_size?: number;
  num_heads?: number;
  num_kv_heads?: number;
  /** Context length the model was built for, in tokens. */
  trained_context?: number;
  num_experts?: number;
  top_k?: number;
  state_dim?: number;
  dtype_bytes?: number;
}

/**
 * Cost of the key/value cache for the simulated request.
 *
 * The one widget in this report that is pure arithmetic rather than a
 * heuristic: bytes per token and bytes total follow directly from the model's
 * real layer count, KV head count and head dimension, with no coefficient
 * anywhere in the formula.
 */
export interface KvCacheCost {
  bytes_per_token: number;
  bytes_total: number;
  /** How much smaller grouped-query attention makes it than full multi-head. */
  gqa_savings_factor: number;
}

export interface InferenceReport {
  stability_index: { score: number; level: StabilityLevel };
  entropy_evolution: number[];
  noise_schedule?: number[];
  hallucination_risk: {
    risk: InferenceRiskLevel;
    confidence: number;
    /** From the connected design's real parameter count — a smaller model
     * confabulates more at identical sampling settings. Absent when no
     * design is connected, in which case it had no effect on `risk`. */
    capacity_component: number | null;
    /** From sampling settings alone (temperature, beam width, top-p,
     * repetition penalty, the adversarial-prompt stress test) — the only
     * thing this score is built from when no design is connected. */
    sampling_component: number;
  };
  attention_focus: number[];
  /** SSM/Mamba only — a Transformer/MoE/diffusion model has no sequential
   * hidden state for "coherence across sequence length" to describe. */
  state_stability?: number;
  context_degradation: number;
  sampling_volatility: { diversity: number; determinism: number };
  router_stability?: { stability: number; distribution: number[] };
  risk_overview: {
    coherence: InferenceRiskLevel;
    overconfidence: InferenceRiskLevel;
    /** MoE only — a dense model has no router to collapse. */
    collapse?: InferenceRiskLevel;
    degeneration: InferenceRiskLevel;
  };
  /** `undefined` when no design was supplied — sampling behaviour alone. */
  kv_cache?: KvCacheCost;
  /** Echo of the model this report was computed for; see {@link ModelProfile}. */
  model_profile?: ModelProfile;
}

export interface InferenceSimulateRequest {
  params: InferenceParams;
  /**
   * The model being simulated, in the shape `/analyze` accepts.
   *
   * Optional. When supplied, context degradation, hallucination risk, router
   * load and KV cache are computed for this model instead of for assumed
   * defaults — the endpoint previously assumed a 32k window and an eight-expert
   * router for every design.
   */
  topology?: Record<string, unknown>;
}

export interface InferenceSimulateResponse {
  report: InferenceReport;
}

// ─── Error class ──────────────────────────────────────────────────

// Defined in its own module so that recognising a failed request does not
// mean importing this whole client. Imported for use below and re-exported
// so existing callers need not change.
import { NeuraxApiError } from '@/services/apiError.ts';

export { NeuraxApiError };

// ─── HTTP helpers ─────────────────────────────────────────────────

async function request<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const url = `${NEURAX_API_BASE}${path}`;

  const token = await getAccessToken();
  const res = await fetch(url, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(init?.headers ?? {}),
    },
  });

  if (!res.ok) {
    const body = await res.text().catch(() => null);
    throw new NeuraxApiError(res.status, res.statusText, body);
  }

  // 200/202 with JSON body
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

// ─── API methods ──────────────────────────────────────────────────

/** GET /health — Health check */
export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health');
}

/** GET /me — Current user + plan */
export async function getMe(): Promise<MeResponse> {
  return {
    user_id: 'dev-user',
    plan: 'elite',
  };
}

/** GET /hardware — List all supported hardware with full specs */
export async function listHardware(): Promise<HardwareDetail[]> {
  return request<HardwareDetail[]>('/hardware');
}

/**
 * GET /hardware/detect — what machine is this?
 *
 * Free enough to call on load: the service reads `/proc`, asks the graphics
 * driver, and answers in microseconds. Nothing is measured and nothing leaves
 * the machine.
 *
 * Returns `null` rather than throwing when the service is not running. A
 * studio with no backend is a normal state — the whole design surface works
 * without one — and refusing to render because detection failed would make a
 * missing service look like a broken application.
 */
export async function detectHardware(): Promise<HardwareProfile | null> {
  try {
    return await request<HardwareProfile>('/hardware/detect');
  } catch {
    return null;
  }
}

/**
 * POST /hardware/measure — what can this machine sustain?
 *
 * Runs a matrix multiply and a memory sweep, about half a second, and caches
 * the result on the machine. Separate from detection precisely because of
 * that cost: it is asked for, never implied.
 *
 * This is the only figure available for hardware the specification database
 * does not cover — an integrated GPU, a laptop CPU, a part newer than the
 * database — which is most machines that are not datacenter cards.
 */
export async function measureHardware(): Promise<HardwareProfile | null> {
  try {
    return await request<HardwareProfile>('/hardware/measure', { method: 'POST' });
  } catch {
    return null;
  }
}

// ─── Training runs ──────────────────────────────────────────────────────────

/**
 * What the studio sends to start a run.
 *
 * The model travels as generated PyTorch. The studio is the only side that
 * can produce it — `modelCodeGen` reads the canvas and is checked against the
 * analysis by `verifyCodegenAgainstAnalysis` — and regenerating it in Rust
 * would put a second source of truth behind the very number the Accuracy view
 * exists to confront.
 */
export interface StartRunRequest {
  name: string;
  modelCode: string;
  modelClass: string;
  /** One sample's shape, without the batch dimension. Required: a wrong
   *  input shape does not degrade a run, it kills it in the first layer. */
  inputShape: number[];
  /** What the first layer consumes. An embedding takes integer indices; a
   *  float tensor is the wrong dtype for it, not just the wrong shape. */
  inputKind: 'tokens' | 'image' | 'features';
  vocabSize?: number;
  numClasses: number;
  datasetPath?: string | null;
  epochs: number;
  batchSize: number;
  learningRate: number;
  precision: string;
  stepsPerEpoch: number;
  checkpointEverySteps: number;
  keepCheckpoints?: number;
  predictions?: Record<string, unknown>;
}

/** What `GET /training/runs/{id}` returns. */
export interface RunSnapshot {
  state: RunState;
  model: ModelBuiltEvent | null;
  steps: TrainingStep[];
  /** Where to resume reading from. Sent straight back on the next poll — the
   *  studio never interprets it, the same contract as tailing a log. */
  nextOffset: number;
  checkpoints: Checkpoint[];
}

/** POST /training/runs — create a run directory and start training in it. */
export async function startTrainingRun(body: StartRunRequest): Promise<RunState> {
  return request<RunState>('/training/runs', { method: 'POST', body: JSON.stringify(body) });
}

/** GET /training/runs — every run on this machine, newest first.
 *
 *  Returns an empty list rather than throwing when no service is running: a
 *  studio with no backend has no runs, which is a true answer, not an error. */
export async function listTrainingRuns(): Promise<RunState[]> {
  try {
    return await request<RunState[]>('/training/runs');
  } catch {
    return [];
  }
}

/**
 * GET /training/runs/{id} — one run, from a byte offset onward.
 *
 * `offset` is what makes live views cheap: the service seeks past what this
 * studio already holds instead of re-sending a hundred thousand steps every
 * second.
 */
export async function getTrainingRun(id: string, offset = 0): Promise<RunSnapshot | null> {
  try {
    return await request<RunSnapshot>(`/training/runs/${encodeURIComponent(id)}?offset=${offset}`);
  } catch {
    return null;
  }
}

/**
 * POST /training/runs/{id}/control — pause, resume or stop.
 *
 * `run` on an interrupted run does more than lift a pause: nothing is
 * listening, so the service starts the process again, and it picks up from
 * its last checkpoint.
 */
export async function controlTrainingRun(
  id: string,
  command: 'run' | 'pause' | 'stop',
): Promise<RunState | null> {
  try {
    return await request<RunState>(`/training/runs/${encodeURIComponent(id)}/control`, {
      method: 'POST',
      body: JSON.stringify({ command }),
    });
  } catch {
    return null;
  }
}

/** POST /analyze — Run analysis synchronously */
export async function analyze(
  body: AnalyzeRequest,
): Promise<AnalyzeResponse> {
  return request<AnalyzeResponse>('/analyze', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

/** POST /plugin/validate — Validate a plugin JSON */
export async function validatePlugin(
  body: PluginValidateRequest,
): Promise<PluginValidateResponse> {
  return request<PluginValidateResponse>('/plugin/validate', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

/** POST /billing/checkout — Create Stripe Checkout session */
export async function createCheckoutSession(
  body: BillingCheckoutRequest,
): Promise<BillingUrlResponse> {
  return request<BillingUrlResponse>('/billing/checkout', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

/** POST /billing/portal — Create Stripe Billing Portal session */
export async function createBillingPortalSession(): Promise<BillingUrlResponse> {
  return request<BillingUrlResponse>('/billing/portal', {
    method: 'POST',
  });
}

/** GET /presets — List all architecture presets (metadata only) */
export async function getPresets(): Promise<PresetMetadata[]> {
  return request<PresetMetadata[]>('/presets');
}

/** GET /presets/{id} — Get full preset details including topology */
export async function getPreset(id: string): Promise<PresetFull> {
  return request<PresetFull>(`/presets/${id}`);
}

/** POST /timemachine — Compiler-backed multi-year cost/carbon/scaling projection */
export async function runTimeMachine(
  body: TimeMachineRequest,
): Promise<TimeMachineResponse> {
  return request<TimeMachineResponse>('/timemachine', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

/** POST /inference/simulate — Analytical inference behavior prediction */
export async function simulateInference(
  body: InferenceSimulateRequest,
): Promise<InferenceSimulateResponse> {
  return request<InferenceSimulateResponse>('/inference/simulate', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

// ─── Streaming Analysis (SSE) ──────────────────────────────────────

export interface AnalysisStreamEvent {
  type: 'Started' | 'PhaseStarted' | 'PhaseCompleted' | 'Progress' | 'Diagnostic' | 'Completed' | 'Failed' | 'Result' | 'Lagged';
  data: Record<string, unknown>;
}

export interface AnalysisStreamCallbacks {
  onStarted?: (data: { job_id: string; model_name: string; model_type: string; num_layers: number }) => void;
  onPhaseStarted?: (data: { job_id: string; phase: string; phase_index: number; total_phases: number }) => void;
  onPhaseCompleted?: (data: { job_id: string; phase: string; phase_index: number; total_phases: number; duration_ms: number }) => void;
  onProgress?: (data: { job_id: string; phase: string; phase_index: number; total_phases: number; progress_pct: number; elapsed_ms: number }) => void;
  onDiagnostic?: (data: { job_id: string; phase: string; category: string; severity: string; code?: string; message: string; suggestion?: string }) => void;
  onCompleted?: (data: { job_id: string; total_ms: number }) => void;
  onFailed?: (data: { job_id: string; error: string; phase: string }) => void;
  onResult?: (report: Record<string, unknown>) => void;
  onError?: (error: Error) => void;
}

/**
 * Start a streaming analysis and listen for SSE events.
 * Returns a function to abort the stream.
 */
export function analyzeStream(
  body: AnalyzeRequest,
  callbacks: AnalysisStreamCallbacks,
): () => void {
  const controller = new AbortController();

  // Step 1: Start the job
  request<{ job_id: string; view_token: string }>('/analyze/stream', {
    method: 'POST',
    body: JSON.stringify(body),
  })
    .then(async (response) => {
      const jobId = response.job_id;
      const viewToken = response.view_token;

      // Step 2: Connect to SSE stream with view token auth
      const baseUrl = NEURAX_API_BASE;
      const eventSource = new EventSource(`${baseUrl}/analyze/stream/${jobId}?token=${encodeURIComponent(viewToken)}`);

      const cleanup = () => {
        eventSource.close();
      };

      controller.signal.addEventListener('abort', cleanup);

      eventSource.onmessage = (event) => {
        try {
          const parsed: AnalysisStreamEvent = JSON.parse(event.data);
          const type = parsed.type;
          const data = parsed.data || {};

          switch (type) {
            case 'Started':
              callbacks.onStarted?.(data as AnalysisStreamCallbacks['onStarted'] extends undefined ? never : Parameters<NonNullable<AnalysisStreamCallbacks['onStarted']>>[0]);
              break;
            case 'PhaseStarted':
              callbacks.onPhaseStarted?.(data as any);
              break;
            case 'PhaseCompleted':
              callbacks.onPhaseCompleted?.(data as any);
              break;
            case 'Progress':
              callbacks.onProgress?.(data as any);
              break;
            case 'Diagnostic':
              callbacks.onDiagnostic?.(data as any);
              break;
            case 'Completed':
              callbacks.onCompleted?.(data as any);
              break;
            case 'Failed':
              callbacks.onFailed?.(data as any);
              cleanup();
              break;
            case 'Result':
              callbacks.onResult?.(data as Record<string, unknown>);
              cleanup();
              break;
            case 'Lagged':
              // Client is behind, continue
              break;
          }
        } catch {
          // Ignore parse errors
        }
      };

      eventSource.onerror = () => {
        // On error, try to get the result via polling
        request<{ status: string; job_id: string; report?: Record<string, unknown>; error?: string }>(`/analyze/result/${jobId}`)
          .then((result) => {
            if (result.status === 'completed' && result.report) {
              callbacks.onResult?.(result.report);
            } else if (result.status === 'failed') {
              callbacks.onFailed?.({ job_id: jobId, error: result.error || 'Analysis failed', phase: 'unknown' });
            }
            cleanup();
          })
          .catch((err) => {
            callbacks.onError?.(err);
            cleanup();
          });
      };
    })
    .catch((err) => {
      callbacks.onError?.(err);
    });

  return () => controller.abort();
}

/** GET /analyze/status/{job_id} — Poll job status */
export async function getAnalysisStatus(jobId: string): Promise<{
  job_id: string;
  status: string;
  created_at_ms: number;
  completed_at_ms: number | null;
  error: string | null;
}> {
  return request(`/analyze/status/${jobId}`);
}

/** GET /analyze/result/{job_id} — Get final result */
export async function getAnalysisResult(jobId: string): Promise<{
  status: string;
  job_id: string;
  report?: Record<string, unknown>;
  error?: string;
}> {
  return request(`/analyze/result/${jobId}`);
}

// ─── Comparison ─────────────────────────────────────────────────────

export interface CompareHardwareConfig {
  hardware: string;
  precision?: string;
  batch_size?: number;
  gpu_count?: number;
  gpu_memory_gb?: number;
}

export interface CompareRequest {
  topology: Record<string, unknown>;
  configs: CompareHardwareConfig[];
}

export interface CompareResultItem {
  label: string;
  hardware: string;
  precision: string;
  batch_size: number;
  gpu_count: number;
  report?: Record<string, unknown>;
  error?: string;
}

export interface CompareResponse {
  results: CompareResultItem[];
}

/** POST /analyze/compare — Compare model across multiple hardware configs */
export async function compareAnalyses(
  body: CompareRequest,
): Promise<CompareResponse> {
  return request<CompareResponse>('/analyze/compare', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

// ─── Projects ────────────────────────────────────────────────────────

export interface Project {
  id: string;
  user_id: string;
  name: string;
  description: string | null;
  architecture: string | null;
  canvas: Record<string, unknown>;
  hardware_config: Record<string, unknown> | null;
  last_analysis: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
}

export interface CreateProjectRequest {
  name: string;
  description?: string;
  architecture?: string;
  canvas: Record<string, unknown>;
  hardware_config?: Record<string, unknown>;
  last_analysis?: Record<string, unknown>;
}

export interface UpdateProjectRequest {
  name?: string;
  description?: string;
  architecture?: string;
  canvas?: Record<string, unknown>;
  hardware_config?: Record<string, unknown>;
  last_analysis?: Record<string, unknown>;
}

export interface ProjectListResponse {
  projects: Project[];
}

export interface ProjectResponse {
  project: Project;
}

/** GET /projects — List all projects for the current user */
export async function listProjects(): Promise<ProjectListResponse> {
  return request<ProjectListResponse>('/projects');
}

/** POST /projects — Create a new project */
export async function createProject(
  body: CreateProjectRequest,
): Promise<ProjectResponse> {
  return request<ProjectResponse>('/projects', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

/** GET /projects/{id} — Get a specific project */
export async function getProject(id: string): Promise<ProjectResponse> {
  return request<ProjectResponse>(`/projects/${id}`);
}

/** PUT /projects/{id} — Update a project */
export async function updateProject(
  id: string,
  body: UpdateProjectRequest,
): Promise<ProjectResponse> {
  return request<ProjectResponse>(`/projects/${id}`, {
    method: 'PUT',
    body: JSON.stringify(body),
  });
}

/** DELETE /projects/{id} — Delete a project */
export async function deleteProject(
  id: string,
): Promise<{ deleted: boolean }> {
  return request<{ deleted: boolean }>(`/projects/${id}`, {
    method: 'DELETE',
  });
}

// ─── ONNX Export ──────────────────────────────────────────────────────

export interface ExportOnnxRequest {
  topology: Record<string, unknown>;
  model_name?: string;
}

export interface ExportOnnxResponse {
  data: string; // base64-encoded ONNX protobuf binary
  model_name: string;
  node_count: number;
  initializer_count: number;
  size_bytes: number;
}

export async function exportOnnx(
  body: ExportOnnxRequest,
): Promise<ExportOnnxResponse> {
  return request<ExportOnnxResponse>('/export/onnx', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

// ─── Credits ──────────────────────────────────────────────────────────

export interface CreditInfo {
  user_id: string;
  used: number;
  limit: number;
  plan: string;
  period_start: string;
  period_end: string;
}

export interface CreditsResponse {
  credits: CreditInfo;
}

/** GET /credits — Get current user's credit balance and usage */
export async function getCredits(): Promise<CreditsResponse> {
  return request<CreditsResponse>('/credits');
}

// ─── Compliance Config ────────────────────────────────────────────────

/**
 * `active` | `upcoming` (agreed and dated, not yet in force) | `uncertain`
 * (proposed, not yet formally adopted) | `repealed` (used to apply, does
 * not any more — kept rather than removed, so a reader who expects it finds
 * out it lapsed instead of finding nothing).
 *
 * The service only ever emits one of these four (`ComplianceStatus` is a
 * Rust enum, not a free string), but the union is written out here too
 * rather than trusted implicitly: a value this type doesn't recognise is a
 * real signal — an older/newer service build out of sync with this
 * client — and should read as "unrecognised", not silently render as
 * whichever branch a naive fallback happens to pick.
 */
export type ComplianceStatus = 'active' | 'upcoming' | 'uncertain' | 'repealed';

export interface ComplianceRegulation {
  name: string;
  year: number;
  limit: number | null;
  unit: string | null;
  status: ComplianceStatus;
  description: string;
  region: string;
}

export interface ComplianceThresholds {
  /** Cumulative training FLOPs — EU AI Act Article 51 systemic-risk threshold. */
  systemic_risk_training_flops: number;
  carbon_report_tonnes: number;
  cost_review_usd: number;
}

export interface ComplianceConfig {
  regulations: ComplianceRegulation[];
  thresholds: ComplianceThresholds;
  recommendations: string[];
  /** Date this dataset was last checked against primary sources (ISO). */
  verified_as_of: string;
}

/** GET /compliance/config — Get compliance configuration and regulations */
export async function getComplianceConfig(): Promise<ComplianceConfig> {
  return request<ComplianceConfig>('/compliance/config');
}

// ─── GitHub Export ─────────────────────────────────────────────────────────

export interface ExportGitHubFile {
  path: string;
  content: string;
}

export interface ExportGitHubRequest {
  files: ExportGitHubFile[];
  github_token: string;
  repo: string;
  branch?: string;
  commit_message?: string;
  create_pr?: boolean;
  pr_branch?: string;
  /** Visibility to create the repository with, if it doesn't already exist.
   * Ignored when it does. Defaults to private server-side if omitted. */
  private?: boolean;
}

export interface ExportGitHubResponse {
  success: boolean;
  file_urls: string[];
  pr_url: string | null;
  error: string | null;
}

export async function exportToGitHub(
  body: ExportGitHubRequest,
): Promise<ExportGitHubResponse> {
  return request<ExportGitHubResponse>('/export/github', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

// ─── Public Shares ──────────────────────────────────────────────────────
//
// Anonymous, read-with-no-account links — see `neurax-service`'s
// `Share`/`ShareMode` for the data model these mirror. Uses
// `NEURAX_HOSTED_API_BASE`, not `request()`, for the reason documented on
// that constant above.

export type ShareMode = 'card' | 'full';

/** The report is an opaque snapshot of `AnalysisResult` — its shape is owned
 * by the frontend, the backend never inspects it, so there is nothing to
 * gain from re-typing every field here. */
export type ShareReport = Record<string, unknown>;

export interface ShareDesign {
  nodes: unknown[];
  connections: unknown[];
  groups: unknown[];
}

export interface Share {
  id: string;
  mode: ShareMode;
  display_name: string;
  family: string | null;
  report: ShareReport;
  design: ShareDesign | null;
  created_at: string;
  view_count: number;
}

async function requestHosted<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${NEURAX_HOSTED_API_BASE}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    const body = await res.text().catch(() => null);
    throw new NeuraxApiError(res.status, res.statusText, body);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

/** The page a viewer actually opens — this app's own origin, not the API's. */
export function shareViewUrl(id: string): string {
  return `${window.location.origin}/s/${id}`;
}

/** A direct link to `GET /shares/{id}/download` — usable as a plain `href`,
 * not just from a fetch call, since it is meant to trigger a real browser
 * download. */
export function shareDownloadUrl(id: string): string {
  return `${NEURAX_HOSTED_API_BASE}/shares/${id}/download`;
}

export interface CreateShareParams {
  mode: ShareMode;
  /** Chosen by the sharer at publish time — never pass the document's own
   * internal name here without letting the user review/edit it first; it
   * may be an internal project codename. */
  displayName: string;
  family?: string | null;
  report: ShareReport;
  /** Required for `mode: 'full'`, ignored for `mode: 'card'` — the backend
   * enforces this server-side regardless of what is sent. */
  design?: ShareDesign | null;
}

export interface CreateShareResult {
  id: string;
  /** Bearer credential for `deleteShare` — shown to the user once, at
   * creation time, exactly like an API key. There is no way to recover it
   * later; losing it means the share can never be taken down. */
  editToken: string;
  url: string;
}

export async function createShare(params: CreateShareParams): Promise<CreateShareResult> {
  const { id, edit_token } = await requestHosted<{ id: string; edit_token: string }>(
    '/shares',
    {
      method: 'POST',
      body: JSON.stringify({
        mode: params.mode,
        display_name: params.displayName,
        family: params.family ?? null,
        report: params.report,
        design: params.mode === 'full' ? (params.design ?? null) : null,
      }),
    },
  );
  return { id, editToken: edit_token, url: shareViewUrl(id) };
}

export async function getShare(id: string): Promise<Share> {
  const { share } = await requestHosted<{ share: Share }>(`/shares/${id}`);
  return share;
}

export async function deleteShare(id: string, editToken: string): Promise<void> {
  await requestHosted<void>(`/shares/${id}`, {
    method: 'DELETE',
    headers: { 'X-Edit-Token': editToken },
  });
}
