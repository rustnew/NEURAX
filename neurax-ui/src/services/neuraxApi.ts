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
  peak_ops_per_s_fp16: number;
  mem_bw_gbps: number;
  vram_bytes: number;
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

/** GET /hardware — List supported hardware details */
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
