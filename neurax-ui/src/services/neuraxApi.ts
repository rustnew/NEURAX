/**
 * Neurax Service API Client
 * Auto-generated from OpenAPI 3.0.3 spec — neurax-service v0.1.0
 *
 * Change NEURAX_API_BASE to point at your backend.
 */

// ─── Configuration ────────────────────────────────────────────────

import { supabase } from '@/lib/supabaseClient.ts';

const SUPABASE_DISABLED = import.meta.env.VITE_SUPABASE_DISABLED === 'true';

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

const NEURAX_API_BASE =
  normalizeLocalApiBase(import.meta.env.VITE_NEURAX_API_URL ?? 'http://127.0.0.1:9098');

let accessToken: string | null = null;

export function setNeuraxAccessToken(token: string | null) {
  accessToken = token;
}

async function getAccessToken(): Promise<string | null> {
  if (accessToken) return accessToken;

  if (SUPABASE_DISABLED) {
    return 'dev-token';
  }

  const { data, error } = await supabase.auth.getSession();
  if (error) return null;

  const token = data.session?.access_token ?? null;
  accessToken = token;
  return token;
}

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

export interface InferenceReport {
  stability_index: { score: number; level: StabilityLevel };
  entropy_evolution: number[];
  noise_schedule?: number[];
  hallucination_risk: { risk: InferenceRiskLevel; confidence: number };
  attention_focus: number[];
  state_stability: number;
  context_degradation: number;
  sampling_volatility: { diversity: number; determinism: number };
  router_stability?: { stability: number; distribution: number[] };
  risk_overview: {
    coherence: InferenceRiskLevel;
    overconfidence: InferenceRiskLevel;
    collapse: InferenceRiskLevel;
    degeneration: InferenceRiskLevel;
  };
}

export interface InferenceSimulateRequest {
  params: InferenceParams;
}

export interface InferenceSimulateResponse {
  report: InferenceReport;
}

// ─── Error class ──────────────────────────────────────────────────

export class NeuraxApiError extends Error {
  constructor(
    public status: number,
    public statusText: string,
    public body?: unknown,
  ) {
    super(`Neurax API ${status}: ${statusText}`);
    this.name = 'NeuraxApiError';
  }
}

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
  if (SUPABASE_DISABLED) {
    return {
      user_id: 'dev-user',
      plan: 'elite',
    };
  }
  return request<MeResponse>('/me');
}

/** GET /hardware — List all supported hardware with full specs */
export async function listHardware(): Promise<HardwareDetail[]> {
  return request<HardwareDetail[]>('/hardware');
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
  request<{ job_id: string }>('/analyze/stream', {
    method: 'POST',
    body: JSON.stringify(body),
  })
    .then(async (response) => {
      const jobId = response.job_id;

      // Step 2: Connect to SSE stream
      const baseUrl = NEURAX_API_BASE;
      const eventSource = new EventSource(`${baseUrl}/analyze/stream/${jobId}`);

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

export interface ComplianceRegulation {
  name: string;
  year: number;
  limit: number | null;
  unit: string | null;
  status: string;
  description: string;
  region: string;
}

export interface ComplianceThresholds {
  high_risk_gflops: number;
  carbon_report_tonnes: number;
  dsa_disclosure_flops: number;
  cost_review_usd: number;
}

export interface ComplianceConfig {
  regulations: ComplianceRegulation[];
  thresholds: ComplianceThresholds;
  recommendations: string[];
}

/** GET /compliance/config — Get compliance configuration and regulations */
export async function getComplianceConfig(): Promise<ComplianceConfig> {
  return request<ComplianceConfig>('/compliance/config');
}
