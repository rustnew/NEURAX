use actix_cors::Cors;
use actix_web::http::header::HeaderName;
use actix_web::{
    http::{header, StatusCode},
    middleware, web, App, HttpRequest, HttpResponse, HttpServer, Responder,
};
use base64::Engine;
use chrono::Datelike;
use dashmap::DashMap;
use hmac::{Hmac, Mac};
use serde::Deserialize;
use sha2::Sha256;
use std::env;
use std::sync::Arc;
use std::time::Duration;
use subtle::ConstantTimeEq;
use tokio::sync::broadcast;
use tracing_subscriber::EnvFilter;

pub mod agent_memory;
pub mod persistence;
mod presets;

// ─── API Key Authentication ─────────────────────────────────────────

/// An API key for programmatic access (used by the agent system)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Hash, Eq, PartialEq)]
pub struct ApiKeyInfo {
    /// The API key itself (prefixed with "nrx_")
    pub key: String,
    /// User who owns this key
    pub user_id: String,
    /// Human-readable name/label
    pub name: String,
    /// When the key was created
    pub created_at: String,
    /// Last time the key was used
    pub last_used_at: Option<String>,
    /// Whether the key is active
    pub active: bool,
    /// Scopes: "analyze", "inference", "compare", "export", "projects", "agent"
    pub scopes: Vec<String>,
}

fn generate_api_key() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let random_part: String = (0..40)
        .map(|_| format!("{:02x}", rng.gen::<u8>()))
        .collect();
    format!("nrx_{}", random_part)
}

fn api_key_from_req(req: &HttpRequest) -> Option<String> {
    // Check X-API-Key header first
    if let Some(key) = req.headers().get("X-API-Key").and_then(|v| v.to_str().ok()) {
        let key = key.trim();
        if key.starts_with("nrx_") && !key.is_empty() {
            return Some(key.to_string());
        }
    }
    // Check Authorization: Bearer nrx_...
    if let Some(auth) = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
    {
        if let Some(token) = auth.strip_prefix("Bearer ") {
            let token = token.trim();
            if token.starts_with("nrx_") {
                return Some(token.to_string());
            }
        }
    }
    None
}

/// Authenticate via API key. Returns the user_id if valid.
async fn auth_api_key(req: &HttpRequest, state: &AppState) -> Result<String, HttpResponse> {
    let key = api_key_from_req(req).ok_or_else(|| {
        HttpResponse::build(StatusCode::UNAUTHORIZED)
            .body("Missing API key. Use X-API-Key header or Authorization: Bearer nrx_...")
    })?;

    let api_key_info = state
        .api_keys
        .get(&key)
        .ok_or_else(|| HttpResponse::build(StatusCode::UNAUTHORIZED).body("Invalid API key"))?;

    if !api_key_info.value().active {
        return Err(HttpResponse::build(StatusCode::FORBIDDEN).body("API key has been revoked"));
    }

    let user_id = api_key_info.value().user_id.clone();
    drop(api_key_info);

    // Update last_used_at
    if let Some(mut entry) = state.api_keys.get_mut(&key) {
        entry.value_mut().last_used_at = Some(chrono::Utc::now().to_rfc3339());
    }

    Ok(user_id)
}

/// Authenticate via either API key or Supabase JWT. Returns user_id.
async fn auth_any(req: &HttpRequest, state: &AppState) -> Result<String, HttpResponse> {
    // Try API key first
    if api_key_from_req(req).is_some() {
        return auth_api_key(req, state).await;
    }
    // Fall back to Supabase JWT
    let user = get_supabase_user(req).await?;
    Ok(user.id)
}

/// Check if an API key has the required scope
fn check_api_key_scope(
    req: &HttpRequest,
    state: &AppState,
    required_scope: &str,
) -> Result<(), HttpResponse> {
    let key = api_key_from_req(req)
        .ok_or_else(|| HttpResponse::build(StatusCode::UNAUTHORIZED).body("Missing API key"))?;

    let api_key_info = state
        .api_keys
        .get(&key)
        .ok_or_else(|| HttpResponse::build(StatusCode::UNAUTHORIZED).body("Invalid API key"))?;

    if !api_key_info.value().active {
        return Err(HttpResponse::build(StatusCode::FORBIDDEN).body("API key has been revoked"));
    }

    let scopes = &api_key_info.value().scopes;
    // "agent" scope grants access to all agent endpoints
    if !scopes.contains(&required_scope.to_string())
        && !scopes.contains(&"agent".to_string())
        && !scopes.contains(&"all".to_string())
    {
        return Err(HttpResponse::build(StatusCode::FORBIDDEN)
            .body(format!("API key lacks '{}' scope", required_scope)));
    }

    Ok(())
}

// ─── Project Storage ─────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Hash, Eq, PartialEq)]
pub struct ProjectKey {
    pub(crate) user_id: String,
    pub(crate) id: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Project {
    // Public because persistence and the integration tests read them; the
    // struct is already serialised out over the API, so nothing is hidden by
    // keeping them private that is not already on the wire.
    pub id: String,
    pub user_id: String,
    pub name: String,
    description: Option<String>,
    /// Architecture family (e.g. "transformer", "moe")
    architecture: Option<String>,
    /// Canvas state as JSON (nodes, connections, groups)
    canvas: serde_json::Value,
    /// Hardware config as JSON
    hardware_config: Option<serde_json::Value>,
    /// Last analysis result (optional, stored as JSON)
    last_analysis: Option<serde_json::Value>,
    created_at: String,
    updated_at: String,
}

#[derive(Debug, serde::Deserialize)]
struct CreateProjectRequest {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    architecture: Option<String>,
    canvas: serde_json::Value,
    #[serde(default)]
    hardware_config: Option<serde_json::Value>,
    #[serde(default)]
    last_analysis: Option<serde_json::Value>,
}

#[derive(Debug, serde::Deserialize)]
struct UpdateProjectRequest {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    architecture: Option<String>,
    #[serde(default)]
    canvas: Option<serde_json::Value>,
    #[serde(default)]
    hardware_config: Option<serde_json::Value>,
    #[serde(default)]
    last_analysis: Option<serde_json::Value>,
}

#[derive(Debug, serde::Serialize)]
struct ProjectListResponse {
    projects: Vec<Project>,
}

#[derive(Debug, serde::Serialize)]
struct ProjectResponse {
    project: Project,
}

#[derive(Debug, serde::Deserialize)]
struct AnalyzeRequest {
    topology: serde_json::Value,
}

#[derive(Debug, serde::Serialize)]
struct AnalyzeResponse {
    report: neurax_ir::report::ReportIR,
}

#[derive(Debug, serde::Deserialize)]
struct SweepCandidatesRequest {
    batch_sizes: Option<Vec<usize>>,
    zero_stages: Option<Vec<u8>>,
    gpu_counts: Option<Vec<u32>>,
    precisions: Option<Vec<String>>,
}

#[derive(Debug, serde::Deserialize)]
struct SweepRequest {
    topology: serde_json::Value,
    #[serde(default)]
    candidates: Option<SweepCandidatesRequest>,
    #[serde(default = "default_sweep_objective")]
    objective: neurax_core::sweep::SweepObjective,
}

fn default_sweep_objective() -> neurax_core::sweep::SweepObjective {
    neurax_core::sweep::SweepObjective::MaxThroughput
}

#[derive(Debug, serde::Serialize)]
struct SweepResponse {
    result: neurax_core::sweep::SweepResult,
}

#[derive(Debug, serde::Deserialize)]
struct TimeMachineRequest {
    topology: serde_json::Value,
    #[serde(default)]
    params: neurax_ir::report::TimeMachineParams,
}

#[derive(Debug, serde::Serialize)]
struct TimeMachineResponse {
    projection: neurax_ir::report::TimeMachineProjection,
}

#[derive(Debug, serde::Serialize)]
struct HealthResponse {
    status: &'static str,
}

#[derive(Debug, serde::Serialize)]
struct HardwareDetailEntry {
    name: String,
    manufacturer: String,
    memory_gb: u64,
    memory_bandwidth_gbs: f64,
    tflops_fp64: f64,
    tflops_fp32: f64,
    tflops_fp16: f64,
    tflops_bf16: f64,
    tflops_int8: f64,
    tflops_fp8: f64,
    tensor_cores: bool,
    nvlink: bool,
    nvlink_bandwidth_gbs: f64,
    tdp_watts: u64,
    launch_year: u32,
}

/// What machine is this, and — once measured — what can it do?
///
/// Two routes rather than one, because the two halves cost very differently.
/// Detection reads `/proc` and asks the graphics driver: microseconds, safe to
/// call on every page load. Measurement runs two compute kernels for about
/// half a second, so it is a separate, explicit call — and it caches, so the
/// second caller pays nothing.
///
/// The spec lookup happens here rather than in the probe. `neurax-probe`
/// reads hardware and knows nothing about NEURAX's database; this handler owns
/// the database, so it is the right place to decide whether a detected part is
/// one we hold published figures for. That is what `recognised` means, and
/// setting it is the whole reason the two are separate crates.
async fn hardware_detect() -> impl Responder {
    let mut profile = neurax_probe::detect();
    resolve_against_database(&mut profile);
    HttpResponse::Ok().json(profile)
}

/// Measure this machine's sustained throughput, and remember it.
///
/// Answers with the full profile rather than the measurement alone, so the
/// caller ends up holding one object that describes the machine completely
/// instead of having to merge two.
async fn hardware_measure() -> impl Responder {
    let mut profile = neurax_probe::detect();
    resolve_against_database(&mut profile);
    profile.compute = Some(neurax_probe::measure_cached(&profile.cpu.model));
    HttpResponse::Ok().json(profile)
}

/// Mark the detected accelerators the database has published figures for.
///
/// Drivers report "NVIDIA GeForce RTX 4090" where the database says
/// "RTX4090", so the comparison ignores spaces, punctuation and the vendor
/// prefix — an exact match would declare every real card unknown and throw
/// away the whole specification sheet.
fn resolve_against_database(profile: &mut neurax_probe::MachineProfile) {
    let db = neurax_hardware_db::HardwareDatabase::new();
    let normalise = |name: &str| -> String {
        name.to_lowercase()
            .replace("nvidia", "")
            .replace("geforce", "")
            .replace("amd", "")
            .replace("radeon", "")
            .replace("intel", "")
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .collect()
    };
    let known: Vec<String> = db.list_gpus().iter().map(|g| normalise(&g.name)).collect();
    for gpu in &mut profile.gpus {
        let wanted = normalise(&gpu.name);
        gpu.recognised = known.iter().any(|k| *k == wanted);
    }
    if profile.gpus.iter().any(|g| !g.recognised) {
        profile.notes.push(
            "NEURAX holds no published specification for at least one detected accelerator, so its \
             throughput has to be measured rather than looked up."
                .into(),
        );
    }
}

use serde_json::json;

// ─── Training runs ──────────────────────────────────────────────────────────
//
// The studio is a viewer onto a directory. Every handler below reads or
// writes files under the runs root and holds no state of its own — which is
// what lets a run survive the service being restarted, and lets two studios
// look at the same run without either one owning it.

/// POST /training/runs — create a run directory and start training in it.
///
/// The model arrives as generated PyTorch rather than being generated here.
/// The studio's `modelCodeGen` is verified against the analysis
/// (`verifyCodegenAgainstAnalysis`), and reimplementing it in Rust would put
/// a second source of truth behind exactly the number the Accuracy view
/// exists to check.
async fn training_start(body: web::Json<neurax_runtime::StartRequest>) -> impl Responder {
    let created = match neurax_runtime::create_run(&body) {
        Ok(state) => state,
        Err(e) => return HttpResponse::BadRequest().json(json!({ "error": e.to_string() })),
    };
    match neurax_runtime::start_run(&created) {
        Ok(state) => HttpResponse::Ok().json(state),
        // The directory survives a failed spawn on purpose: what was about to
        // run is still on disk to look at, rather than vanishing with the
        // error message.
        Err(e) => HttpResponse::InternalServerError()
            .json(json!({ "error": e.to_string(), "run": created })),
    }
}

/// POST /model/verify — does this code build the model it claims to?
///
/// The gate every candidate passes before it is trained, and the reason the
/// assistant is allowed to write model code at all. The deterministic
/// generator's worth was never that a machine wrote it — it was that its
/// parameter count is confronted with the analysis and refuses on
/// disagreement. Code from anywhere else earns the same trust the same way or
/// it does not earn it.
///
/// Runs off the async threads: it starts a Python process and waits for it.
async fn model_verify(body: web::Json<neurax_runtime::VerifyRequest>) -> impl Responder {
    let request = body.into_inner();
    match web::block(move || neurax_runtime::verify_model(&request)).await {
        Ok(Ok(verdict)) => HttpResponse::Ok().json(verdict),
        // The checker could not be run at all — no Python, no temp directory.
        // Distinct from a verdict of "this model is wrong", and the studio
        // must not read it as one.
        Ok(Err(e)) => HttpResponse::InternalServerError()
            .json(json!({ "error": e.to_string(), "stage": "checker" })),
        Err(e) => HttpResponse::InternalServerError()
            .json(json!({ "error": e.to_string(), "stage": "checker" })),
    }
}

/// GET /training/runs — every run on this machine, newest first.
async fn training_list() -> impl Responder {
    HttpResponse::Ok().json(neurax_runtime::list_runs())
}

/// GET /training/runs/{id} — one run, with everything the studio draws.
///
/// `after` lets a reattaching studio ask only for steps it has not seen. A
/// run of 100 000 steps would otherwise re-send its whole history on every
/// poll, which is both slow and the reason live views stutter.
/// Where the caller got to last time, in bytes.
///
/// Byte offset rather than step number: the service seeks past what the
/// studio already has instead of re-reading and re-parsing the whole history
/// on every poll. The studio does not interpret the number, it just sends
/// back the `nextOffset` it was given — the same contract as tailing a log.
#[derive(serde::Deserialize)]
struct StepQuery {
    #[serde(default)]
    offset: u64,
}

async fn training_get(path: web::Path<String>, query: web::Query<StepQuery>) -> impl Responder {
    let id = path.into_inner();
    let dir = match neurax_runtime::run_dir(&id) {
        Ok(dir) => dir,
        Err(e) => return HttpResponse::NotFound().json(json!({ "error": e.to_string() })),
    };
    let state = match neurax_runtime::read_state_reconciled(&dir) {
        Ok(state) => state,
        Err(e) => return HttpResponse::InternalServerError().json(json!({ "error": e.to_string() })),
    };
    let (steps, next_offset) =
        neurax_runtime::read_steps_from(&dir, query.offset).unwrap_or_default();
    HttpResponse::Ok().json(json!({
        "state": state,
        "model": neurax_runtime::read_model_built(&dir),
        "steps": steps,
        "nextOffset": next_offset,
        "checkpoints": neurax_runtime::list_checkpoints(&dir),
    }))
}

#[derive(serde::Deserialize)]
struct ControlBody {
    command: neurax_runtime::Control,
}

/// POST /training/runs/{id}/control — pause, resume or stop.
///
/// Writes a file the training loop reads between steps, rather than signalling
/// the process. A signal would freeze it mid-kernel with memory held and no
/// checkpoint written; a file lets it stop at a step boundary, which is the
/// only place stopping is clean.
///
/// Resuming an *interrupted* run is different from resuming a paused one:
/// nothing is listening, so the process has to be started again. It picks up
/// from its last checkpoint.
async fn training_control(path: web::Path<String>, body: web::Json<ControlBody>) -> impl Responder {
    let id = path.into_inner();
    let dir = match neurax_runtime::run_dir(&id) {
        Ok(dir) => dir,
        Err(e) => return HttpResponse::NotFound().json(json!({ "error": e.to_string() })),
    };
    if let Err(e) = neurax_runtime::write_control(&dir, body.command) {
        return HttpResponse::InternalServerError().json(json!({ "error": e.to_string() }));
    }

    let state = match neurax_runtime::read_state_reconciled(&dir) {
        Ok(state) => state,
        Err(e) => return HttpResponse::InternalServerError().json(json!({ "error": e.to_string() })),
    };

    if matches!(body.command, neurax_runtime::Control::Run)
        && state.status == neurax_runtime::RunStatus::Interrupted
    {
        return match neurax_runtime::start_run(&state) {
            Ok(restarted) => HttpResponse::Ok().json(restarted),
            Err(e) => HttpResponse::InternalServerError().json(json!({ "error": e.to_string() })),
        };
    }
    HttpResponse::Ok().json(state)
}

async fn hardware_list() -> impl Responder {
    let db = neurax_hardware_db::HardwareDatabase::new();
    let gpus = db.list_gpus();
    let out: Vec<HardwareDetailEntry> = gpus
        .iter()
        .map(|g| HardwareDetailEntry {
            name: g.name.clone(),
            manufacturer: g.manufacturer.clone(),
            memory_gb: g.memory_gb,
            memory_bandwidth_gbs: g.memory_bandwidth_gbs,
            tflops_fp64: g.tflops_fp64,
            tflops_fp32: g.tflops_fp32,
            tflops_fp16: g.tflops_fp16,
            tflops_bf16: g.tflops_bf16,
            tflops_int8: g.tflops_int8,
            tflops_fp8: g.tflops_fp8,
            tensor_cores: g.tensor_cores,
            nvlink: g.nvlink,
            nvlink_bandwidth_gbs: g.nvlink_bandwidth_gbs,
            tdp_watts: g.tdp_watts,
            launch_year: g.launch_year,
        })
        .collect();
    HttpResponse::Ok().json(out)
}

#[derive(Debug, serde::Deserialize)]
struct PluginValidateRequest {
    plugin: serde_json::Value,
}

#[derive(Debug, serde::Serialize)]
struct PluginValidateResponse {
    ok: bool,
}

#[derive(Debug, serde::Deserialize)]
struct InferenceRequest {
    #[serde(default)]
    params: neurax_ir::inference::InferenceParams,
    /// The model being simulated, in the same shape `/analyze` accepts.
    ///
    /// Optional, so callers that only want sampling behaviour keep working. When
    /// present, context degradation, hallucination risk, router load and KV
    /// cache are computed for this model instead of for assumed defaults.
    #[serde(default)]
    topology: Option<serde_json::Value>,
}

/// Derive the inference-relevant dimensions from a model config.
///
/// Reuses the analysis parser rather than reading the JSON directly, so the
/// simulation and the compiler always agree on what a topology means.
fn model_profile_from_topology(
    topology: &serde_json::Value,
) -> Option<neurax_ir::inference::ModelProfile> {
    let json = serde_json::to_string(topology).ok()?;
    let config = neurax_parser::parse_model_config(&json).ok()?;

    let attention = config
        .model
        .layers
        .iter()
        .find(|l| l.layer_type == neurax_parser::LayerType::Attention);
    let moe = config
        .model
        .layers
        .iter()
        .find(|l| l.layer_type == neurax_parser::LayerType::MoE);
    let ssm = config.model.layers.iter().find(|l| {
        matches!(
            l.layer_type,
            neurax_parser::LayerType::MambaBlock | neurax_parser::LayerType::S4Block
        )
    });

    let hidden_size = attention
        .and_then(|l| l.params.hidden_size)
        .or(config.model.global_params.embedding_dim);
    let num_heads = attention.and_then(|l| l.params.num_heads);

    Some(neurax_ir::inference::ModelProfile {
        total_parameters: Some(neurax_ir::architecture::scaled_total_parameters(&config)),
        num_layers: config
            .model
            .global_params
            .num_layers
            .or(Some(config.model.layers.len() as u64)),
        hidden_size: hidden_size.map(|v| v as u64),
        num_heads: num_heads.map(|v| v as u64),
        num_kv_heads: attention
            .and_then(|l| l.params.num_kv_heads)
            .or(num_heads)
            .map(|v| v as u64),
        trained_context: config
            .model
            .global_params
            .sequence_length
            .or(config.training.sequence_length)
            .map(|v| v as u64),
        num_experts: moe.and_then(|l| l.params.num_experts).map(|v| v as u64),
        top_k: moe.and_then(|l| l.params.top_k).map(|v| v as u64),
        state_dim: ssm.and_then(|l| l.params.state_dim).map(|v| v as u64),
        dtype_bytes: Some(match config.training.precision.as_str() {
            "fp32" | "float32" => 4,
            "fp16" | "float16" | "bf16" | "bfloat16" => 2,
            "fp8" | "float8" | "int8" => 1,
            _ => 4,
        }),
    })
}

#[derive(Debug, serde::Serialize)]
struct InferenceResponse {
    report: neurax_ir::inference::InferenceReport,
}

#[derive(Debug, Deserialize)]
struct SupabaseUser {
    id: String,
    email_confirmed_at: Option<String>,
    confirmed_at: Option<String>,
}

/// Whether to skip Supabase authentication and treat every caller as `dev-user`.
///
/// Off unless explicitly switched on. This used to default to *on*, which meant
/// a deployment that simply never set the variable served every endpoint —
/// projects, credits, API keys — to anonymous callers. Two legitimate users of
/// the bypass remain, and both set it deliberately: local development against a
/// frontend with no Supabase project, and `neurax-desktop`, which is
/// single-user, loopback-only, and has no account to authenticate against.
fn noauth_enabled() -> bool {
    env::var("NEURAX_DEBUG_NOAUTH")
        .map(|v| v == "true" || v == "1")
        .unwrap_or(false)
}

#[derive(Debug, serde::Serialize)]
struct MeResponse {
    user_id: String,
    plan: String,
}

#[derive(Debug, serde::Deserialize)]
struct BillingCheckoutRequest {
    plan: String,
    interval: String,
    success_url: String,
    cancel_url: String,
}

#[derive(Debug, serde::Serialize)]
struct BillingUrlResponse {
    url: String,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct UserProfileRow {
    id: String,
    stripe_customer_id: Option<String>,
    plan_override: Option<String>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct StripeSubscriptionRow {
    user_id: String,
    status: String,
    plan_tier: String,
}

type HmacSha256 = Hmac<Sha256>;

// ─── Streaming Analysis Job Store ──────────────────────────────────

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct JobInfo {
    pub job_id: String,
    pub user_id: String,
    pub view_token: String,
    pub status: String,
    pub created_at_ms: u64,
    pub completed_at_ms: Option<u64>,
    pub error: Option<String>,
}

/// Shared state for streaming analysis jobs
#[derive(Clone)]
pub struct AppState {
    /// Job metadata store
    pub jobs: Arc<DashMap<String, JobInfo>>,
    /// Broadcast channels for each job (for SSE streaming)
    pub channels: Arc<DashMap<String, broadcast::Sender<String>>>,
    /// Completed analysis results stored as JSON
    pub results: Arc<DashMap<String, serde_json::Value>>,
    /// Projects store (keyed by user_id + project_id)
    pub projects: Arc<DashMap<ProjectKey, Project>>,
    /// Credit tracking per user
    pub credits: Arc<DashMap<String, CreditInfo>>,
    /// API keys for programmatic access (keyed by the API key string)
    pub api_keys: Arc<DashMap<String, ApiKeyInfo>>,
    /// Analysis results cache keyed by user_id (for agent to read back)
    pub user_analyses: Arc<DashMap<String, serde_json::Value>>,
    /// Inference results cache keyed by user_id (for agent to read back)
    pub user_inferences: Arc<DashMap<String, serde_json::Value>>,
    /// Public, anonymous share links (keyed by the share's short id)
    pub shares: Arc<DashMap<String, Share>>,
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

impl AppState {
    pub fn new() -> Self {
        Self {
            jobs: Arc::new(DashMap::new()),
            channels: Arc::new(DashMap::new()),
            results: Arc::new(DashMap::new()),
            projects: Arc::new(DashMap::new()),
            credits: Arc::new(DashMap::new()),
            api_keys: Arc::new(DashMap::new()),
            user_analyses: Arc::new(DashMap::new()),
            user_inferences: Arc::new(DashMap::new()),
            shares: Arc::new(DashMap::new()),
        }
    }
}

/// Reclaim streaming-analysis job/result entries older than the retention
/// window.
///
/// `state.channels` is cleaned up 30s after each job finishes (see
/// `analyze_stream_start`), but `state.jobs` and `state.results` were never
/// removed anywhere — every `/analyze/stream` call inserts a job_id-keyed
/// entry, including the job's full JSON report in `results`, that then lived
/// for the rest of the process's uptime. A long-running service (or a
/// desktop app left open for days) accumulates one of these per streaming
/// analysis ever run, unbounded.
///
/// This can't be as aggressive as the 30s `channels` cleanup: `/analyze/
/// result/{job_id}` and `/analyze/status/{job_id}` are meant to be pollable
/// well after the SSE stream itself has closed, so removing entries too
/// early would break that legitimate "come back later for the result" use.
/// 24h is ample time for that, while still bounding memory to roughly a
/// day's worth of streaming jobs instead of the process's entire lifetime.
const JOB_RETENTION_MS: u64 = 24 * 60 * 60 * 1000;
const JOB_SWEEP_INTERVAL: Duration = Duration::from_secs(60 * 60);

/// Remove `jobs`/`results` entries whose `created_at_ms` is older than
/// `retention_ms` relative to `now_ms`. Returns how many were reclaimed.
/// Pulled out of `spawn_job_retention_sweeper`'s loop so the sweep logic
/// itself — not just "does a background task exist" — has a direct test.
fn sweep_expired_jobs(
    jobs: &DashMap<String, JobInfo>,
    results: &DashMap<String, serde_json::Value>,
    now_ms: u64,
    retention_ms: u64,
) -> usize {
    let expired: Vec<String> = jobs
        .iter()
        .filter(|entry| now_ms.saturating_sub(entry.created_at_ms) > retention_ms)
        .map(|entry| entry.key().clone())
        .collect();
    for job_id in &expired {
        jobs.remove(job_id);
        results.remove(job_id);
    }
    expired.len()
}

fn spawn_job_retention_sweeper(state: &AppState) {
    let jobs = state.jobs.clone();
    let results = state.results.clone();
    actix_web::rt::spawn(async move {
        loop {
            actix_web::rt::time::sleep(JOB_SWEEP_INTERVAL).await;
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            let reclaimed = sweep_expired_jobs(&jobs, &results, now_ms, JOB_RETENTION_MS);
            if reclaimed > 0 {
                tracing::info!("[JOB_SWEEP] Reclaimed {} job(s) older than 24h", reclaimed);
            }
        }
    });
}

#[cfg(test)]
mod job_sweep_tests {
    use super::*;

    fn job(created_at_ms: u64) -> JobInfo {
        JobInfo {
            job_id: "irrelevant".to_string(),
            user_id: "u1".to_string(),
            view_token: "t".to_string(),
            status: "completed".to_string(),
            created_at_ms,
            completed_at_ms: Some(created_at_ms),
            error: None,
        }
    }

    /// The bug this sweeper fixes: before it existed, nothing ever called
    /// `jobs.remove`/`results.remove` for these two maps (unlike `channels`,
    /// which already had a 30s cleanup) — so a job from a week ago and a job
    /// from a second ago were indistinguishable, both kept forever. This
    /// pins the actual boundary: strictly-older-than-retention is reclaimed,
    /// within-retention is not.
    #[test]
    fn only_entries_past_the_retention_window_are_reclaimed() {
        let jobs: DashMap<String, JobInfo> = DashMap::new();
        let results: DashMap<String, serde_json::Value> = DashMap::new();
        let retention_ms = 24 * 60 * 60 * 1000;
        let now_ms = 10 * retention_ms; // arbitrary "current time" far from zero

        jobs.insert("old".to_string(), job(now_ms - retention_ms - 1));
        jobs.insert("boundary".to_string(), job(now_ms - retention_ms));
        jobs.insert("recent".to_string(), job(now_ms - 1_000));
        results.insert("old".to_string(), serde_json::json!({"report": "old"}));
        results.insert(
            "boundary".to_string(),
            serde_json::json!({"report": "boundary"}),
        );
        results.insert(
            "recent".to_string(),
            serde_json::json!({"report": "recent"}),
        );

        let reclaimed = sweep_expired_jobs(&jobs, &results, now_ms, retention_ms);

        assert_eq!(
            reclaimed, 1,
            "only the entry strictly past retention should go"
        );
        assert!(!jobs.contains_key("old"), "old job should be removed");
        assert!(
            !results.contains_key("old"),
            "old job's result should be removed too"
        );
        assert!(
            jobs.contains_key("boundary"),
            "exactly-at-retention should survive"
        );
        assert!(jobs.contains_key("recent"), "recent job should survive");
        assert!(
            results.contains_key("recent"),
            "recent job's result should survive"
        );
    }

    #[test]
    fn an_empty_store_sweeps_cleanly() {
        let jobs: DashMap<String, JobInfo> = DashMap::new();
        let results: DashMap<String, serde_json::Value> = DashMap::new();
        assert_eq!(sweep_expired_jobs(&jobs, &results, 1_000_000, 1000), 0);
    }
}

#[derive(Debug, serde::Deserialize)]
struct AnalyzeStreamRequest {
    topology: serde_json::Value,
}

/// A single hardware configuration override for comparison
#[derive(Debug, serde::Deserialize, Clone)]
struct CompareHardwareConfig {
    /// GPU name (e.g. "H100-SXM", "A100-PCIe", "RTX4090")
    hardware: String,
    /// Precision (e.g. "fp16", "fp32", "bf16", "int8", "fp8")
    #[serde(default)]
    precision: Option<String>,
    /// Batch size
    #[serde(default)]
    batch_size: Option<u32>,
    /// Number of GPUs
    #[serde(default)]
    gpu_count: Option<u32>,
    /// GPU memory in GB (overrides spec default)
    #[serde(default)]
    gpu_memory_gb: Option<u64>,
}

#[derive(Debug, serde::Deserialize)]
struct CompareRequest {
    topology: serde_json::Value,
    configs: Vec<CompareHardwareConfig>,
}

#[derive(Debug, serde::Serialize)]
struct CompareResultItem {
    label: String,
    hardware: String,
    precision: String,
    batch_size: u32,
    gpu_count: u32,
    report: Option<neurax_ir::report::ReportIR>,
    error: Option<String>,
}

#[derive(Debug, serde::Serialize)]
struct CompareResponse {
    results: Vec<CompareResultItem>,
}

#[derive(Debug, serde::Serialize)]
struct AnalyzeStreamResponse {
    job_id: String,
    view_token: String,
}

fn bearer_token_from_req(req: &HttpRequest) -> Result<String, HttpResponse> {
    let auth = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    let token = auth
        .strip_prefix("Bearer ")
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            HttpResponse::build(StatusCode::UNAUTHORIZED).body("Missing Authorization bearer token")
        })?;
    Ok(token.to_string())
}

async fn get_supabase_user(req: &HttpRequest) -> Result<SupabaseUser, HttpResponse> {
    if noauth_enabled() {
        return Ok(SupabaseUser {
            id: "dev-user".to_string(),
            email_confirmed_at: Some("dev".to_string()),
            confirmed_at: Some("dev".to_string()),
        });
    }

    let supabase_url = env::var("SUPABASE_URL").map_err(|_| {
        HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR).body("SUPABASE_URL not set")
    })?;
    let service_role_key = env::var("SUPABASE_SERVICE_ROLE_KEY").map_err(|_| {
        HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR)
            .body("SUPABASE_SERVICE_ROLE_KEY not set")
    })?;

    let token = bearer_token_from_req(req)?;

    let url = format!("{}/auth/v1/user", supabase_url.trim_end_matches('/'));
    let client = reqwest::Client::new();
    let res = client
        .get(url)
        .header("apikey", &service_role_key)
        .header(reqwest::header::AUTHORIZATION, format!("Bearer {token}"))
        .send()
        .await
        .map_err(|_| {
            HttpResponse::build(StatusCode::BAD_GATEWAY).body("Failed to reach Supabase auth")
        })?;

    if res.status() == reqwest::StatusCode::UNAUTHORIZED
        || res.status() == reqwest::StatusCode::FORBIDDEN
    {
        return Err(
            HttpResponse::build(StatusCode::UNAUTHORIZED).body("Invalid or expired session")
        );
    }

    let status = res.status();
    let body_text = res
        .text()
        .await
        .unwrap_or_else(|_| "<failed to read body>".to_string());

    if !status.is_success() {
        return Err(HttpResponse::build(StatusCode::BAD_GATEWAY).body(format!(
            "Supabase auth returned an error (status={status}): {body_text}"
        )));
    }

    let user = serde_json::from_str::<SupabaseUser>(&body_text).map_err(|e| {
        HttpResponse::build(StatusCode::BAD_GATEWAY).body(format!(
            "Supabase auth returned non-JSON or unexpected JSON (status={status}, err={e}): {body_text}"
        ))
    })?;

    Ok(user)
}

async fn require_verified_email(req: &HttpRequest) -> Result<SupabaseUser, HttpResponse> {
    let user = get_supabase_user(req).await?;

    // Debug bypass: skip email verification check
    if noauth_enabled()
        || env::var("NEURAX_DEBUG_BYPASS")
            .map(|v| v == "true")
            .unwrap_or(false)
    {
        return Ok(user);
    }

    if user.email_confirmed_at.is_none() && user.confirmed_at.is_none() {
        return Err(HttpResponse::build(StatusCode::FORBIDDEN)
            .body("Please verify your email to run analysis."));
    }
    Ok(user)
}

fn normalize_plan_tier(s: &str) -> Option<String> {
    match s.trim().to_lowercase().as_str() {
        "free" => Some("free".to_string()),
        "essential" => Some("essential".to_string()),
        "architect" => Some("architect".to_string()),
        "elite" => Some("elite".to_string()),
        _ => None,
    }
}

fn active_subscription_status(status: &str) -> bool {
    matches!(status, "active" | "trialing")
}

fn stripe_price_env_key(plan: &str, interval: &str) -> Option<&'static str> {
    match (plan, interval) {
        ("essential", "month") => Some("STRIPE_PRICE_ESSENTIAL_MONTHLY"),
        ("essential", "year") => Some("STRIPE_PRICE_ESSENTIAL_ANNUAL"),
        ("architect", "month") => Some("STRIPE_PRICE_ARCHITECT_MONTHLY"),
        ("architect", "year") => Some("STRIPE_PRICE_ARCHITECT_ANNUAL"),
        ("elite", "month") => Some("STRIPE_PRICE_ELITE_MONTHLY"),
        ("elite", "year") => Some("STRIPE_PRICE_ELITE_ANNUAL"),
        _ => None,
    }
}

pub(crate) async fn supabase_rest_client() -> Result<(String, String, reqwest::Client), HttpResponse> {
    let supabase_url = env::var("SUPABASE_URL").map_err(|_| {
        HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR).body("SUPABASE_URL not set")
    })?;
    let service_role_key = env::var("SUPABASE_SERVICE_ROLE_KEY").map_err(|_| {
        HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR)
            .body("SUPABASE_SERVICE_ROLE_KEY not set")
    })?;
    let client = reqwest::Client::new();
    Ok((supabase_url, service_role_key, client))
}

async fn fetch_user_profile(user_id: &str) -> Result<UserProfileRow, HttpResponse> {
    let (supabase_url, service_role_key, client) = supabase_rest_client().await?;
    let url = format!(
        "{}/rest/v1/user_profiles?id=eq.{}&select=id,stripe_customer_id,plan_override",
        supabase_url.trim_end_matches('/'),
        user_id
    );

    let res = client
        .get(url)
        .header("apikey", &service_role_key)
        .header(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {service_role_key}"),
        )
        .header(reqwest::header::ACCEPT, "application/json")
        .send()
        .await
        .map_err(|_| {
            HttpResponse::build(StatusCode::BAD_GATEWAY).body("Failed to reach Supabase REST")
        })?;

    if !res.status().is_success() {
        return Err(
            HttpResponse::build(StatusCode::BAD_GATEWAY).body("Supabase REST returned an error")
        );
    }

    let mut rows = res.json::<Vec<UserProfileRow>>().await.map_err(|_| {
        HttpResponse::build(StatusCode::BAD_GATEWAY).body("Supabase REST returned invalid JSON")
    })?;
    rows.pop()
        .ok_or_else(|| HttpResponse::build(StatusCode::NOT_FOUND).body("Profile not found"))
}

async fn update_user_profile_stripe_customer(
    user_id: &str,
    stripe_customer_id: &str,
) -> Result<(), HttpResponse> {
    let (supabase_url, service_role_key, client) = supabase_rest_client().await?;
    let url = format!(
        "{}/rest/v1/user_profiles?id=eq.{}",
        supabase_url.trim_end_matches('/'),
        user_id
    );

    let body = serde_json::json!({ "stripe_customer_id": stripe_customer_id });
    let res = client
        .patch(url)
        .header("apikey", &service_role_key)
        .header(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {service_role_key}"),
        )
        .header("Prefer", "return=minimal")
        .json(&body)
        .send()
        .await
        .map_err(|_| {
            HttpResponse::build(StatusCode::BAD_GATEWAY).body("Failed to update Supabase profile")
        })?;

    if !res.status().is_success() {
        return Err(
            HttpResponse::build(StatusCode::BAD_GATEWAY).body("Supabase profile update failed")
        );
    }
    Ok(())
}

async fn fetch_active_subscription_plan(user_id: &str) -> Result<Option<String>, HttpResponse> {
    let (supabase_url, service_role_key, client) = supabase_rest_client().await?;
    let url = format!(
        "{}/rest/v1/stripe_subscriptions?user_id=eq.{}&select=user_id,status,plan_tier",
        supabase_url.trim_end_matches('/'),
        user_id
    );

    let res = client
        .get(url)
        .header("apikey", &service_role_key)
        .header(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {service_role_key}"),
        )
        .header(reqwest::header::ACCEPT, "application/json")
        .send()
        .await
        .map_err(|_| {
            HttpResponse::build(StatusCode::BAD_GATEWAY).body("Failed to reach Supabase REST")
        })?;

    if !res.status().is_success() {
        return Err(
            HttpResponse::build(StatusCode::BAD_GATEWAY).body("Supabase REST returned an error")
        );
    }

    let rows = res
        .json::<Vec<StripeSubscriptionRow>>()
        .await
        .map_err(|_| {
            HttpResponse::build(StatusCode::BAD_GATEWAY).body("Supabase REST returned invalid JSON")
        })?;

    for r in rows {
        if active_subscription_status(r.status.as_str()) {
            if let Some(p) = normalize_plan_tier(&r.plan_tier) {
                return Ok(Some(p));
            }
        }
    }

    Ok(None)
}

async fn me(http_req: HttpRequest) -> impl Responder {
    let user = match get_supabase_user(&http_req).await {
        Ok(u) => u,
        Err(resp) => return resp,
    };

    // Debug bypass: Mock subscription tier
    let mock_plan = env::var("NEURAX_MOCK_PLAN").ok();
    if let Some(plan_str) = mock_plan.as_deref().or(Some("elite")) {
        if let Some(normalized) = normalize_plan_tier(plan_str) {
            return HttpResponse::Ok().json(MeResponse {
                user_id: user.id,
                plan: normalized,
            });
        }
    }

    let profile = match fetch_user_profile(&user.id).await {
        Ok(p) => p,
        Err(resp) => return resp,
    };

    if let Some(override_plan) = profile
        .plan_override
        .as_deref()
        .and_then(normalize_plan_tier)
    {
        return HttpResponse::Ok().json(MeResponse {
            user_id: user.id,
            plan: override_plan,
        });
    }

    let paid_plan = match fetch_active_subscription_plan(&user.id).await {
        Ok(p) => p,
        Err(resp) => return resp,
    };

    HttpResponse::Ok().json(MeResponse {
        user_id: user.id,
        plan: paid_plan.unwrap_or_else(|| "free".to_string()),
    })
}

async fn stripe_create_customer(
    user_id: &str,
    email: Option<&str>,
) -> Result<String, HttpResponse> {
    let stripe_secret_key = env::var("STRIPE_SECRET_KEY").map_err(|_| {
        HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR).body("STRIPE_SECRET_KEY not set")
    })?;

    let client = reqwest::Client::new();
    let mut form = vec![("metadata[supabase_user_id]", user_id.to_string())];
    if let Some(e) = email {
        form.push(("email", e.to_string()));
    }

    let res = client
        .post("https://api.stripe.com/v1/customers")
        .bearer_auth(stripe_secret_key)
        .form(&form)
        .send()
        .await
        .map_err(|_| HttpResponse::build(StatusCode::BAD_GATEWAY).body("Failed to reach Stripe"))?;

    if !res.status().is_success() {
        return Err(HttpResponse::build(StatusCode::BAD_GATEWAY).body("Stripe returned an error"));
    }

    let v = res.json::<serde_json::Value>().await.map_err(|_| {
        HttpResponse::build(StatusCode::BAD_GATEWAY).body("Stripe returned invalid JSON")
    })?;
    let id = v.get("id").and_then(|x| x.as_str()).ok_or_else(|| {
        HttpResponse::build(StatusCode::BAD_GATEWAY).body("Stripe response missing id")
    })?;
    Ok(id.to_string())
}

async fn billing_checkout(
    http_req: HttpRequest,
    req: web::Json<BillingCheckoutRequest>,
) -> impl Responder {
    let user = match get_supabase_user(&http_req).await {
        Ok(u) => u,
        Err(resp) => return resp,
    };

    let plan = match normalize_plan_tier(&req.plan) {
        Some(p) if p != "free" => p,
        _ => return HttpResponse::build(StatusCode::BAD_REQUEST).body("Invalid plan"),
    };

    let interval = match req.interval.trim().to_lowercase().as_str() {
        "month" | "monthly" => "month",
        "year" | "annual" | "annually" => "year",
        _ => return HttpResponse::build(StatusCode::BAD_REQUEST).body("Invalid interval"),
    };

    let price_env = stripe_price_env_key(plan.as_str(), interval).ok_or_else(|| {
        HttpResponse::build(StatusCode::BAD_REQUEST).body("Unsupported plan/interval")
    });
    let price_env = match price_env {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let price_id = match env::var(price_env) {
        Ok(v) if !v.trim().is_empty() => v,
        _ => {
            return HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR)
                .body("Stripe price env var not set")
        }
    };

    let mut profile = match fetch_user_profile(&user.id).await {
        Ok(p) => p,
        Err(resp) => return resp,
    };

    let stripe_customer_id = match profile.stripe_customer_id.as_deref() {
        Some(id) if !id.trim().is_empty() => id.to_string(),
        _ => {
            let created = match stripe_create_customer(&user.id, None).await {
                Ok(v) => v,
                Err(resp) => return resp,
            };
            if let Err(resp) = update_user_profile_stripe_customer(&user.id, &created).await {
                return resp;
            }
            profile.stripe_customer_id = Some(created.clone());
            created
        }
    };

    let stripe_secret_key = match env::var("STRIPE_SECRET_KEY") {
        Ok(v) => v,
        Err(_) => {
            return HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR)
                .body("STRIPE_SECRET_KEY not set")
        }
    };

    let client = reqwest::Client::new();
    let form = vec![
        ("mode", "subscription".to_string()),
        ("customer", stripe_customer_id),
        ("line_items[0][price]", price_id),
        ("line_items[0][quantity]", "1".to_string()),
        ("success_url", req.success_url.clone()),
        ("cancel_url", req.cancel_url.clone()),
    ];

    let res = client
        .post("https://api.stripe.com/v1/checkout/sessions")
        .bearer_auth(stripe_secret_key)
        .form(&form)
        .send()
        .await
        .map_err(|_| HttpResponse::build(StatusCode::BAD_GATEWAY).body("Failed to reach Stripe"));
    let res = match res {
        Ok(r) => r,
        Err(resp) => return resp,
    };

    if !res.status().is_success() {
        return HttpResponse::build(StatusCode::BAD_GATEWAY).body("Stripe returned an error");
    }

    let v = match res.json::<serde_json::Value>().await {
        Ok(v) => v,
        Err(_) => {
            return HttpResponse::build(StatusCode::BAD_GATEWAY)
                .body("Stripe returned invalid JSON")
        }
    };
    let url = v.get("url").and_then(|x| x.as_str()).ok_or_else(|| {
        HttpResponse::build(StatusCode::BAD_GATEWAY).body("Stripe response missing url")
    });
    let url = match url {
        Ok(u) => u,
        Err(resp) => return resp,
    };

    HttpResponse::Ok().json(BillingUrlResponse {
        url: url.to_string(),
    })
}

async fn billing_portal(http_req: HttpRequest) -> impl Responder {
    let user = match get_supabase_user(&http_req).await {
        Ok(u) => u,
        Err(resp) => return resp,
    };

    let profile = match fetch_user_profile(&user.id).await {
        Ok(p) => p,
        Err(resp) => return resp,
    };

    let stripe_customer_id = match profile.stripe_customer_id.as_deref() {
        Some(id) if !id.trim().is_empty() => id.to_string(),
        _ => {
            return HttpResponse::build(StatusCode::BAD_REQUEST).body("No Stripe customer for user")
        }
    };

    let stripe_secret_key = match env::var("STRIPE_SECRET_KEY") {
        Ok(v) => v,
        Err(_) => {
            return HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR)
                .body("STRIPE_SECRET_KEY not set")
        }
    };
    let return_url = match env::var("STRIPE_PORTAL_RETURN_URL") {
        Ok(v) => v,
        Err(_) => {
            return HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR)
                .body("STRIPE_PORTAL_RETURN_URL not set")
        }
    };

    let client = reqwest::Client::new();
    let form = vec![("customer", stripe_customer_id), ("return_url", return_url)];
    let res = client
        .post("https://api.stripe.com/v1/billing_portal/sessions")
        .bearer_auth(stripe_secret_key)
        .form(&form)
        .send()
        .await
        .map_err(|_| HttpResponse::build(StatusCode::BAD_GATEWAY).body("Failed to reach Stripe"));
    let res = match res {
        Ok(r) => r,
        Err(resp) => return resp,
    };

    if !res.status().is_success() {
        return HttpResponse::build(StatusCode::BAD_GATEWAY).body("Stripe returned an error");
    }

    let v = match res.json::<serde_json::Value>().await {
        Ok(v) => v,
        Err(_) => {
            return HttpResponse::build(StatusCode::BAD_GATEWAY)
                .body("Stripe returned invalid JSON")
        }
    };
    let url = v.get("url").and_then(|x| x.as_str()).ok_or_else(|| {
        HttpResponse::build(StatusCode::BAD_GATEWAY).body("Stripe response missing url")
    });
    let url = match url {
        Ok(u) => u,
        Err(resp) => return resp,
    };

    HttpResponse::Ok().json(BillingUrlResponse {
        url: url.to_string(),
    })
}

fn stripe_verify_signature(payload: &[u8], sig_header: &str, secret: &str) -> bool {
    let mut timestamp: Option<&str> = None;
    let mut signature: Option<&str> = None;
    for part in sig_header.split(',') {
        let part = part.trim();
        if let Some(v) = part.strip_prefix("t=") {
            timestamp = Some(v);
        } else if let Some(v) = part.strip_prefix("v1=") {
            signature = Some(v);
        }
    }
    let (t, v1) = match (timestamp, signature) {
        (Some(t), Some(v1)) => (t, v1),
        _ => return false,
    };

    let signed_payload = [t.as_bytes(), b".", payload].concat();
    let mut mac = match HmacSha256::new_from_slice(secret.as_bytes()) {
        Ok(m) => m,
        Err(_) => return false,
    };
    mac.update(&signed_payload);
    let expected = mac.finalize().into_bytes();
    let expected_hex = hex::encode(expected);
    expected_hex.as_bytes().ct_eq(v1.as_bytes()).into()
}

async fn stripe_webhook(http_req: HttpRequest, body: web::Bytes) -> impl Responder {
    let sig = match http_req
        .headers()
        .get("Stripe-Signature")
        .and_then(|v| v.to_str().ok())
    {
        Some(v) => v,
        None => {
            return HttpResponse::build(StatusCode::BAD_REQUEST).body("Missing Stripe-Signature")
        }
    };
    let secret = match env::var("STRIPE_WEBHOOK_SECRET") {
        Ok(v) => v,
        Err(_) => {
            return HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR)
                .body("STRIPE_WEBHOOK_SECRET not set")
        }
    };
    if !stripe_verify_signature(&body, sig, &secret) {
        return HttpResponse::build(StatusCode::BAD_REQUEST).body("Invalid webhook signature");
    }

    let event: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return HttpResponse::build(StatusCode::BAD_REQUEST).body("Invalid JSON"),
    };
    let event_id = match event.get("id").and_then(|v| v.as_str()) {
        Some(v) => v,
        None => return HttpResponse::build(StatusCode::BAD_REQUEST).body("Missing event id"),
    };
    let event_type = event
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    let (supabase_url, service_role_key, client) = match supabase_rest_client().await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    // Idempotency: insert event row; if already exists, treat as ok.
    let insert_url = format!(
        "{}/rest/v1/stripe_webhook_events",
        supabase_url.trim_end_matches('/')
    );
    let insert_body = serde_json::json!({
        "id": event_id,
        "type": event_type,
        "payload": event.clone(),
    });
    let insert_res = client
        .post(insert_url)
        .header("apikey", &service_role_key)
        .header(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {service_role_key}"),
        )
        .header("Prefer", "return=minimal")
        .json(&insert_body)
        .send()
        .await;

    if let Ok(r) = &insert_res {
        // PostgREST returns 409 conflict on duplicate primary key.
        if !(r.status().is_success() || r.status().as_u16() == 409) {
            return HttpResponse::build(StatusCode::BAD_GATEWAY)
                .body("Failed to persist webhook event");
        }
    } else {
        return HttpResponse::build(StatusCode::BAD_GATEWAY).body("Failed to reach Supabase REST");
    }

    // Handle subscription updates
    if matches!(
        event_type,
        "customer.subscription.created"
            | "customer.subscription.updated"
            | "customer.subscription.deleted"
    ) {
        // For subscription.* events, object is the subscription.
        let obj = event
            .pointer("/data/object")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));

        let subscription_id = obj.get("id").and_then(|v| v.as_str());
        let customer_id = obj.get("customer").and_then(|v| v.as_str());
        let status = obj
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        // price id: items.data[0].price.id
        let price_id = obj
            .pointer("/items/data/0/price/id")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // map price id to plan tier via env vars
        let plan_tier = if price_id
            == env::var("STRIPE_PRICE_ESSENTIAL_MONTHLY").unwrap_or_default()
            || price_id == env::var("STRIPE_PRICE_ESSENTIAL_ANNUAL").unwrap_or_default()
        {
            "essential"
        } else if price_id == env::var("STRIPE_PRICE_ARCHITECT_MONTHLY").unwrap_or_default()
            || price_id == env::var("STRIPE_PRICE_ARCHITECT_ANNUAL").unwrap_or_default()
        {
            "architect"
        } else if price_id == env::var("STRIPE_PRICE_ELITE_MONTHLY").unwrap_or_default()
            || price_id == env::var("STRIPE_PRICE_ELITE_ANNUAL").unwrap_or_default()
        {
            "elite"
        } else {
            "free"
        };

        if let (Some(sub_id), Some(cust_id)) = (subscription_id, customer_id) {
            // find user_profiles by stripe_customer_id
            let find_url = format!(
                "{}/rest/v1/user_profiles?stripe_customer_id=eq.{}&select=id",
                supabase_url.trim_end_matches('/'),
                cust_id
            );
            let find_res = client
                .get(find_url)
                .header("apikey", &service_role_key)
                .header(
                    reqwest::header::AUTHORIZATION,
                    format!("Bearer {service_role_key}"),
                )
                .send()
                .await;

            if let Ok(fr) = find_res {
                if fr.status().is_success() {
                    let rows = fr
                        .json::<Vec<serde_json::Value>>()
                        .await
                        .unwrap_or_default();
                    if let Some(row) = rows.first() {
                        if let Some(user_id) = row.get("id").and_then(|v| v.as_str()) {
                            let upsert_url = format!(
                                "{}/rest/v1/stripe_subscriptions?on_conflict=stripe_subscription_id",
                                supabase_url.trim_end_matches('/')
                            );
                            let upsert_body = serde_json::json!({
                                "user_id": user_id,
                                "stripe_customer_id": cust_id,
                                "stripe_subscription_id": sub_id,
                                "stripe_price_id": price_id,
                                "plan_tier": plan_tier,
                                "status": status,
                            });
                            // Upsert by unique stripe_subscription_id
                            let upsert_res = client
                                .post(upsert_url)
                                .header("apikey", &service_role_key)
                                .header(
                                    reqwest::header::AUTHORIZATION,
                                    format!("Bearer {service_role_key}"),
                                )
                                .header("Prefer", "resolution=merge-duplicates,return=minimal")
                                .json(&upsert_body)
                                .send()
                                .await;

                            if let Ok(ur) = upsert_res {
                                if !ur.status().is_success() {
                                    return HttpResponse::build(StatusCode::BAD_GATEWAY)
                                        .body("Failed to upsert subscription");
                                }
                            } else {
                                return HttpResponse::build(StatusCode::BAD_GATEWAY)
                                    .body("Failed to reach Supabase REST");
                            }
                        }
                    }
                }
            }
        }
    }

    HttpResponse::Ok().body("ok")
}

async fn inference_simulate(
    http_req: HttpRequest,
    req: web::Json<InferenceRequest>,
) -> impl Responder {
    let start = std::time::Instant::now();
    tracing::info!("[INFERENCE] Request received");

    if let Err(resp) = require_verified_email(&http_req).await {
        tracing::warn!(
            "[INFERENCE] Auth failed after {}ms",
            start.elapsed().as_millis()
        );
        return resp;
    }

    let params = req.params.clone();
    // A topology that cannot be parsed is reported as such rather than silently
    // simulated as some other model.
    let profile = match req.topology.as_ref() {
        Some(topology) => match model_profile_from_topology(topology) {
            Some(profile) => Some(profile),
            None => {
                tracing::warn!("[INFERENCE] Topology could not be parsed; rejecting");
                return HttpResponse::build(StatusCode::BAD_REQUEST).body(
                    "topology could not be parsed; omit it to simulate sampling behaviour alone",
                );
            }
        },
        None => None,
    };

    let result = actix_web::rt::task::spawn_blocking(move || {
        neurax_ir::inference::InferencePass::run_with_model(&params, profile.as_ref())
    });

    let timeout_result = actix_web::rt::time::timeout(Duration::from_secs(30), result).await;

    let elapsed = start.elapsed();
    match timeout_result {
        Ok(Ok(report)) => {
            tracing::info!("[INFERENCE] Success in {}ms", elapsed.as_millis());
            HttpResponse::Ok().json(InferenceResponse { report })
        }
        Ok(Err(_join_err)) => {
            tracing::error!(
                "[INFERENCE] Task join error after {}ms",
                elapsed.as_millis()
            );
            HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR)
                .body("Inference task failed unexpectedly")
        }
        Err(_timeout) => {
            tracing::error!("[INFERENCE] Timeout after {}ms", elapsed.as_millis());
            HttpResponse::build(StatusCode::GATEWAY_TIMEOUT)
                .body("Inference timed out after 30 seconds")
        }
    }
}

// ─── ONNX Export ────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ExportOnnxRequest {
    topology: serde_json::Value,
    /// Optional model name override
    model_name: Option<String>,
}

#[derive(Debug, serde::Serialize)]
struct ExportOnnxResponse {
    /// Base64-encoded ONNX protobuf binary
    data: String,
    /// Model name used
    model_name: String,
    /// Number of nodes in the ONNX graph
    node_count: usize,
    /// Number of initializers (weight tensors)
    initializer_count: usize,
    /// Size in bytes
    size_bytes: usize,
}

async fn export_onnx(http_req: HttpRequest, req: web::Json<ExportOnnxRequest>) -> impl Responder {
    let start = std::time::Instant::now();
    tracing::info!("[EXPORT_ONNX] Request received");

    if let Err(resp) = require_verified_email(&http_req).await {
        tracing::warn!(
            "[EXPORT_ONNX] Auth failed after {}ms",
            start.elapsed().as_millis()
        );
        return resp;
    }

    let input = match serde_json::to_string(&req.topology) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("[EXPORT_ONNX] Failed to serialize topology: {}", e);
            return HttpResponse::build(StatusCode::BAD_REQUEST).body(e.to_string());
        }
    };

    // Parse the topology JSON into ModelConfig
    tracing::info!("[EXPORT_ONNX] Parsing model config...");
    let config = match neurax_parser::parse_model_config(&input) {
        Ok(c) => {
            tracing::info!(
                "[EXPORT_ONNX] Parse OK: model_type={:?}, layers={}",
                c.model.model_type,
                c.model.layers.len()
            );
            c
        }
        Err(e) => {
            tracing::error!("[EXPORT_ONNX] Parse failed: {}", e);
            return HttpResponse::build(StatusCode::BAD_REQUEST).body(e.to_string());
        }
    };

    let model_name = req.model_name.clone();
    let result = actix_web::rt::task::spawn_blocking(move || {
        // Run the analysis pipeline to get the ArchitectureIR
        let analysis = neurax_core::run_analysis(config.clone()).map_err(|e| e.to_string())?;

        // Export to ONNX
        neurax_core::export::export_onnx(
            &analysis.arch,
            &config.training,
            &config.data,
            model_name.as_deref(),
        )
    });

    let timeout_result = actix_web::rt::time::timeout(Duration::from_secs(60), result).await;

    let elapsed = start.elapsed();
    match timeout_result {
        Ok(Ok(Ok(onnx_result))) => {
            tracing::info!(
                "[EXPORT_ONNX] Success in {}ms - {} nodes, {} initializers, {} bytes",
                elapsed.as_millis(),
                onnx_result.node_count,
                onnx_result.initializer_count,
                onnx_result.bytes.len()
            );
            let size_bytes = onnx_result.bytes.len();
            let model_name = onnx_result.model_name.clone();
            let node_count = onnx_result.node_count;
            let initializer_count = onnx_result.initializer_count;
            HttpResponse::Ok().json(ExportOnnxResponse {
                data: base64::engine::general_purpose::STANDARD.encode(&onnx_result.bytes),
                model_name,
                node_count,
                initializer_count,
                size_bytes,
            })
        }
        Ok(Ok(Err(e))) => {
            tracing::error!(
                "[EXPORT_ONNX] Export error after {}ms: {}",
                elapsed.as_millis(),
                e
            );
            HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR).body(e.to_string())
        }
        Ok(Err(_join_err)) => {
            tracing::error!(
                "[EXPORT_ONNX] Task join error after {}ms",
                elapsed.as_millis()
            );
            HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR)
                .body("Export task failed unexpectedly")
        }
        Err(_timeout) => {
            tracing::error!("[EXPORT_ONNX] Timeout after {}ms", elapsed.as_millis());
            HttpResponse::build(StatusCode::GATEWAY_TIMEOUT)
                .body("Export timed out after 60 seconds")
        }
    }
}

// ─── GitHub Export ─────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ExportGitHubFile {
    path: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ExportGitHubRequest {
    /// Files to push to the repository
    files: Vec<ExportGitHubFile>,
    /// GitHub Personal Access Token
    github_token: String,
    /// Repository in "owner/repo" format
    repo: String,
    /// Branch to push to (default "main")
    branch: Option<String>,
    /// Commit message
    commit_message: Option<String>,
    /// Whether to create a PR instead of pushing directly
    create_pr: Option<bool>,
    /// Branch name for the PR (ignored unless create_pr is true)
    pr_branch: Option<String>,
    /// Visibility for a repository NEURAX creates because it didn't already
    /// exist. Ignored when the repository is already there. Defaults to
    /// private — an architecture is the client's own, and NEURAX has no
    /// business making it public on their behalf without being told to.
    private: Option<bool>,
}

#[derive(Debug, serde::Serialize)]
struct ExportGitHubResponse {
    success: bool,
    /// URLs of the created/updated files on GitHub
    file_urls: Vec<String>,
    /// PR URL if create_pr was true
    pr_url: Option<String>,
    /// Error message if success is false
    error: Option<String>,
}

/// Makes sure `owner/repo_name` exists, creating it if it doesn't, and
/// returns its actual default branch — never assumed to be "main", since
/// plenty of real repositories still default to "master".
///
/// Creating it is what makes the client-supplied name enough on its own: no
/// repository has to be created by hand on GitHub first. `auto_init: true`
/// is the important part of that — it gives a brand-new repository a real
/// first commit, so it has a default branch at all. Without it, an empty
/// repository has zero branches, and every call later in the export that
/// reads or branches off of one fails with an opaque 404.
async fn ensure_github_repo(
    client: &reqwest::Client,
    token: &str,
    owner: &str,
    repo_name: &str,
    private: bool,
) -> Result<String, (StatusCode, String)> {
    let get_url = format!("https://api.github.com/repos/{}/{}", owner, repo_name);
    let get_resp = client
        .get(&get_url)
        .header("Authorization", format!("Bearer {}", token))
        .header("User-Agent", "NEURAX-Export")
        .send()
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_GATEWAY,
                format!("GitHub API request failed: {}", e),
            )
        })?;

    if get_resp.status().is_success() {
        let data: serde_json::Value = get_resp.json().await.map_err(|e| {
            (
                StatusCode::BAD_GATEWAY,
                format!("Failed to read repository info: {}", e),
            )
        })?;
        return Ok(data["default_branch"]
            .as_str()
            .unwrap_or("main")
            .to_string());
    }

    if get_resp.status().as_u16() != 404 {
        let status = get_resp.status();
        let body = get_resp.text().await.unwrap_or_default();
        return Err((
            StatusCode::BAD_GATEWAY,
            format!(
                "GitHub API error checking repository ({}): {}",
                status, body
            ),
        ));
    }

    tracing::info!(
        "[EXPORT_GITHUB] Repository {}/{} not found, creating it",
        owner,
        repo_name
    );

    // GitHub uses a different endpoint depending on whether `owner` is the
    // token's own account or an organization — ask which, rather than guess
    // and get a second, unrelated 404.
    let who_resp = client
        .get("https://api.github.com/user")
        .header("Authorization", format!("Bearer {}", token))
        .header("User-Agent", "NEURAX-Export")
        .send()
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_GATEWAY,
                format!("GitHub API request failed: {}", e),
            )
        })?;

    if !who_resp.status().is_success() {
        let status = who_resp.status();
        let body = who_resp.text().await.unwrap_or_default();
        return Err((
            StatusCode::BAD_GATEWAY,
            format!(
                "{}/{} doesn't exist, and the token could not be verified to create it ({}): {}",
                owner, repo_name, status, body
            ),
        ));
    }

    let who: serde_json::Value = who_resp.json().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!("Failed to read token identity: {}", e),
        )
    })?;
    let login = who["login"].as_str().unwrap_or_default();

    let create_url = if login.eq_ignore_ascii_case(owner) {
        "https://api.github.com/user/repos".to_string()
    } else {
        format!("https://api.github.com/orgs/{}/repos", owner)
    };

    let create_body = serde_json::json!({
        "name": repo_name,
        "private": private,
        "auto_init": true,
        "description": "Created by NEURAX — https://github.com/rustnew/NEURAX",
    });

    let create_resp = client
        .post(&create_url)
        .header("Authorization", format!("Bearer {}", token))
        .header("User-Agent", "NEURAX-Export")
        .json(&create_body)
        .send()
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_GATEWAY,
                format!("GitHub API request failed: {}", e),
            )
        })?;

    if !create_resp.status().is_success() {
        let status = create_resp.status();
        let body = create_resp.text().await.unwrap_or_default();
        return Err((
            StatusCode::BAD_GATEWAY,
            format!(
                "{}/{} doesn't exist, and NEURAX could not create it ({}): {}. \
                 If {} is an organization, the token needs permission to create \
                 repositories in it.",
                owner, repo_name, status, body, owner
            ),
        ));
    }

    let created: serde_json::Value = create_resp.json().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!("Failed to read created-repository response: {}", e),
        )
    })?;
    let default_branch = created["default_branch"]
        .as_str()
        .unwrap_or("main")
        .to_string();
    tracing::info!(
        "[EXPORT_GITHUB] Created {}/{} (default branch: {})",
        owner,
        repo_name,
        default_branch
    );
    Ok(default_branch)
}

/// Makes sure `branch` actually has a commit on it — not just that the
/// repository containing it exists.
///
/// This is the gap `ensure_github_repo` alone doesn't close: a repository
/// created by hand on github.com without the "initialize with a README"
/// checkbox exists, and reports a `default_branch` name, but has zero
/// commits and zero real branches — `default_branch` on an empty repository
/// is GitHub's answer to "what the branch will be called", not proof it
/// exists yet. Pushing to it through the normal Contents API then fails
/// with a 404 that looks identical to "the repository doesn't exist" —
/// exactly the symptom this closes.
///
/// A brand-new repository this same request just created via `auto_init`
/// never reaches the "create it" branch below — its first commit already
/// exists, so the ref lookup here succeeds and this is a no-op.
async fn ensure_branch_exists(
    client: &reqwest::Client,
    token: &str,
    owner: &str,
    repo_name: &str,
    branch: &str,
    seed_file: &ExportGitHubFile,
) -> Result<(), (StatusCode, String)> {
    let ref_url = format!(
        "https://api.github.com/repos/{}/{}/git/ref/heads/{}",
        owner, repo_name, branch
    );
    let check = client
        .get(&ref_url)
        .header("Authorization", format!("Bearer {}", token))
        .header("User-Agent", "NEURAX-Export")
        .send()
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_GATEWAY,
                format!("GitHub API request failed: {}", e),
            )
        })?;

    if check.status().is_success() {
        return Ok(()); // Branch already has a commit — nothing to do.
    }
    if check.status().as_u16() != 404 {
        let status = check.status();
        let body = check.text().await.unwrap_or_default();
        return Err((
            StatusCode::BAD_GATEWAY,
            format!("GitHub API error checking branch ({}): {}", status, body),
        ));
    }

    tracing::info!(
        "[EXPORT_GITHUB] {}/{}@{} has no commits yet, creating the first one",
        owner,
        repo_name,
        branch
    );

    let api_base = format!("https://api.github.com/repos/{}/{}", owner, repo_name);
    let auth = format!("Bearer {}", token);

    // An empty repository's first commit cannot be made through the Contents
    // API (the endpoint the rest of this export uses) — it has no parent
    // commit to attach to. The Git Data API builds one by hand instead: a
    // blob (the file's bytes), a tree (the blob at its path), a commit (the
    // tree, no parent), then a ref pointing `refs/heads/{branch}` at it.
    let blob_resp = client
        .post(format!("{}/git/blobs", api_base))
        .header("Authorization", &auth)
        .header("User-Agent", "NEURAX-Export")
        .json(&serde_json::json!({
            "content": base64::engine::general_purpose::STANDARD.encode(seed_file.content.as_bytes()),
            "encoding": "base64",
        }))
        .send()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("GitHub API request failed: {}", e)))?;
    if !blob_resp.status().is_success() {
        let status = blob_resp.status();
        let body = blob_resp.text().await.unwrap_or_default();
        return Err((
            StatusCode::BAD_GATEWAY,
            format!("Failed to seed the first commit ({}): {}", status, body),
        ));
    }
    let blob: serde_json::Value = blob_resp.json().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!("Failed to read blob response: {}", e),
        )
    })?;
    let blob_sha = blob["sha"].as_str().unwrap_or_default().to_string();

    let tree_resp = client
        .post(format!("{}/git/trees", api_base))
        .header("Authorization", &auth)
        .header("User-Agent", "NEURAX-Export")
        .json(&serde_json::json!({
            "tree": [{
                "path": seed_file.path,
                "mode": "100644",
                "type": "blob",
                "sha": blob_sha,
            }],
        }))
        .send()
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_GATEWAY,
                format!("GitHub API request failed: {}", e),
            )
        })?;
    if !tree_resp.status().is_success() {
        let status = tree_resp.status();
        let body = tree_resp.text().await.unwrap_or_default();
        return Err((
            StatusCode::BAD_GATEWAY,
            format!("Failed to seed the first commit ({}): {}", status, body),
        ));
    }
    let tree: serde_json::Value = tree_resp.json().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!("Failed to read tree response: {}", e),
        )
    })?;
    let tree_sha = tree["sha"].as_str().unwrap_or_default().to_string();

    let commit_resp = client
        .post(format!("{}/git/commits", api_base))
        .header("Authorization", &auth)
        .header("User-Agent", "NEURAX-Export")
        .json(&serde_json::json!({
            "message": "Initial commit — NEURAX",
            "tree": tree_sha,
            "parents": [],
        }))
        .send()
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_GATEWAY,
                format!("GitHub API request failed: {}", e),
            )
        })?;
    if !commit_resp.status().is_success() {
        let status = commit_resp.status();
        let body = commit_resp.text().await.unwrap_or_default();
        return Err((
            StatusCode::BAD_GATEWAY,
            format!("Failed to seed the first commit ({}): {}", status, body),
        ));
    }
    let commit: serde_json::Value = commit_resp.json().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!("Failed to read commit response: {}", e),
        )
    })?;
    let commit_sha = commit["sha"].as_str().unwrap_or_default().to_string();

    let ref_resp = client
        .post(format!("{}/git/refs", api_base))
        .header("Authorization", &auth)
        .header("User-Agent", "NEURAX-Export")
        .json(&serde_json::json!({
            "ref": format!("refs/heads/{}", branch),
            "sha": commit_sha,
        }))
        .send()
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_GATEWAY,
                format!("GitHub API request failed: {}", e),
            )
        })?;
    if !ref_resp.status().is_success() {
        let status = ref_resp.status();
        let body = ref_resp.text().await.unwrap_or_default();
        return Err((
            StatusCode::BAD_GATEWAY,
            format!(
                "Failed to create branch \"{}\" ({}): {}",
                branch, status, body
            ),
        ));
    }

    tracing::info!(
        "[EXPORT_GITHUB] Seeded {}/{}@{} with an initial commit",
        owner,
        repo_name,
        branch
    );
    Ok(())
}

async fn export_github(
    http_req: HttpRequest,
    req: web::Json<ExportGitHubRequest>,
) -> impl Responder {
    let start = std::time::Instant::now();
    tracing::info!("[EXPORT_GITHUB] Request received");

    if let Err(resp) = require_verified_email(&http_req).await {
        tracing::warn!(
            "[EXPORT_GITHUB] Auth failed after {}ms",
            start.elapsed().as_millis()
        );
        return resp;
    }

    let commit_message = req
        .commit_message
        .clone()
        .unwrap_or_else(|| "Add model architecture from NEURAX".to_string());
    let create_pr = req.create_pr.unwrap_or(false);
    let pr_branch = req.pr_branch.clone().unwrap_or_else(|| {
        format!(
            "neurax/update-{}",
            chrono::Utc::now().format("%Y%m%d%H%M%S")
        )
    });

    let github_token = req.github_token.trim().to_string();
    let repo = req.repo.trim().to_string();

    if github_token.is_empty() || repo.is_empty() {
        return HttpResponse::build(StatusCode::BAD_REQUEST).json(ExportGitHubResponse {
            success: false,
            file_urls: vec![],
            pr_url: None,
            error: Some("GitHub token and repository are required".to_string()),
        });
    }

    let Some((owner, repo_name)) = repo.split_once('/') else {
        return HttpResponse::build(StatusCode::BAD_REQUEST).json(ExportGitHubResponse {
            success: false,
            file_urls: vec![],
            pr_url: None,
            error: Some(format!(
                "\"{}\" is not a valid repository — expected \"owner/repo\"",
                repo
            )),
        });
    };

    if req.files.is_empty() {
        return HttpResponse::build(StatusCode::BAD_REQUEST).json(ExportGitHubResponse {
            success: false,
            file_urls: vec![],
            pr_url: None,
            error: Some("No files to export".to_string()),
        });
    }

    let client = reqwest::Client::new();
    let api_base = format!("https://api.github.com/repos/{}", repo);

    // The bug this closes: a client exporting to a repository that either
    // doesn't exist yet, or exists but has zero commits (so no branch has
    // ever been created), got an opaque "404 Not Found" from a much later
    // GitHub API call with no indication *why*. Resolving (and, if needed,
    // creating) the repository up front turns that into either a working
    // export or one clear error message, and — since a freshly created repo
    // is `auto_init`ed — guarantees a real default branch exists before
    // anything below tries to read or branch off of one.
    let repo_default_branch = match ensure_github_repo(
        &client,
        &github_token,
        owner,
        repo_name,
        req.private.unwrap_or(true),
    )
    .await
    {
        Ok(default_branch) => default_branch,
        Err((status, message)) => {
            tracing::error!("[EXPORT_GITHUB] Could not resolve repository: {}", message);
            return HttpResponse::build(status).json(ExportGitHubResponse {
                success: false,
                file_urls: vec![],
                pr_url: None,
                error: Some(message),
            });
        }
    };

    // Kept around (not just consumed into `branch`) so a later 404 can say
    // exactly how a client-supplied branch differs from the real default,
    // instead of just repeating GitHub's generic "Not Found".
    let branch = req
        .branch
        .clone()
        .unwrap_or_else(|| repo_default_branch.clone());

    // Covers the case `ensure_github_repo` alone can't: a repository that
    // already existed (created by hand, without "initialize with a README")
    // and has zero commits, so `branch` names a real repository but not yet
    // a real ref. `req.files[0]` exists — `req.files.is_empty()` was already
    // rejected above.
    if let Err((status, message)) = ensure_branch_exists(
        &client,
        &github_token,
        owner,
        repo_name,
        &branch,
        &req.files[0],
    )
    .await
    {
        tracing::error!("[EXPORT_GITHUB] Could not prepare branch: {}", message);
        return HttpResponse::build(status).json(ExportGitHubResponse {
            success: false,
            file_urls: vec![],
            pr_url: None,
            error: Some(message),
        });
    }

    // Determine the target branch (use pr_branch if creating a PR)
    let target_branch = if create_pr { &pr_branch } else { &branch };

    // If creating a PR, first get the SHA of the base branch to create a new branch
    let base_sha = if create_pr {
        let branch_url = format!("{}/git/ref/heads/{}", api_base, branch);
        match client
            .get(&branch_url)
            .header("Authorization", format!("Bearer {}", github_token))
            .header("User-Agent", "NEURAX-Export")
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                let data: serde_json::Value = match resp.json().await {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::error!("[EXPORT_GITHUB] Failed to parse branch response: {}", e);
                        return HttpResponse::build(StatusCode::BAD_GATEWAY).json(
                            ExportGitHubResponse {
                                success: false,
                                file_urls: vec![],
                                pr_url: None,
                                error: Some(format!("Failed to read base branch: {}", e)),
                            },
                        );
                    }
                };
                let sha = data["object"]["sha"].as_str().map(|s| s.to_string());
                match sha {
                    Some(s) => s,
                    None => {
                        return HttpResponse::build(StatusCode::BAD_GATEWAY).json(
                            ExportGitHubResponse {
                                success: false,
                                file_urls: vec![],
                                pr_url: None,
                                error: Some("Could not resolve base branch SHA".to_string()),
                            },
                        );
                    }
                }
            }
            Ok(resp) => {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                tracing::error!(
                    "[EXPORT_GITHUB] Failed to get branch: {} - {}",
                    status,
                    body
                );
                return HttpResponse::build(StatusCode::BAD_GATEWAY).json(ExportGitHubResponse {
                    success: false,
                    file_urls: vec![],
                    pr_url: None,
                    error: Some(format!("GitHub API error ({}): {}", status, body)),
                });
            }
            Err(e) => {
                tracing::error!("[EXPORT_GITHUB] Request failed: {}", e);
                return HttpResponse::build(StatusCode::BAD_GATEWAY).json(ExportGitHubResponse {
                    success: false,
                    file_urls: vec![],
                    pr_url: None,
                    error: Some(format!("GitHub API request failed: {}", e)),
                });
            }
        }
    } else {
        String::new()
    };

    // If creating a PR, create the new branch
    if create_pr {
        let ref_url = format!("{}/git/refs", api_base);
        let create_ref_body = serde_json::json!({
            "ref": format!("refs/heads/{}", pr_branch),
            "sha": base_sha,
        });
        match client
            .post(&ref_url)
            .header("Authorization", format!("Bearer {}", github_token))
            .header("User-Agent", "NEURAX-Export")
            .json(&create_ref_body)
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() || resp.status().as_u16() == 422 => {
                // 422 = already exists, which is fine
                tracing::info!("[EXPORT_GITHUB] Branch {} ready", pr_branch);
            }
            Ok(resp) => {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                tracing::error!(
                    "[EXPORT_GITHUB] Failed to create branch: {} - {}",
                    status,
                    body
                );
                return HttpResponse::build(StatusCode::BAD_GATEWAY).json(ExportGitHubResponse {
                    success: false,
                    file_urls: vec![],
                    pr_url: None,
                    error: Some(format!("Failed to create branch ({}): {}", status, body)),
                });
            }
            Err(e) => {
                tracing::error!("[EXPORT_GITHUB] Request failed: {}", e);
                return HttpResponse::build(StatusCode::BAD_GATEWAY).json(ExportGitHubResponse {
                    success: false,
                    file_urls: vec![],
                    pr_url: None,
                    error: Some(format!("GitHub API request failed: {}", e)),
                });
            }
        }
    }

    // Push each file to GitHub
    let mut file_urls: Vec<String> = vec![];

    for file in &req.files {
        let content_base64 =
            base64::engine::general_purpose::STANDARD.encode(file.content.as_bytes());
        let put_url = format!("{}/contents/{}", api_base, file.path);

        // Check if file already exists to get the SHA
        let existing_sha: Option<String> = {
            let check_resp = client
                .get(&put_url)
                .query(&[("ref", target_branch)])
                .header("Authorization", format!("Bearer {}", github_token))
                .header("User-Agent", "NEURAX-Export")
                .send()
                .await;
            match check_resp {
                Ok(resp) if resp.status().is_success() => resp
                    .json::<serde_json::Value>()
                    .await
                    .ok()
                    .and_then(|data| data["sha"].as_str().map(|s| s.to_string())),
                _ => None,
            }
        };

        let mut body = serde_json::json!({
            "message": commit_message,
            "content": content_base64,
            "branch": target_branch,
        });
        if let Some(sha) = existing_sha {
            body["sha"] = serde_json::Value::String(sha);
        }

        match client
            .put(&put_url)
            .header("Authorization", format!("Bearer {}", github_token))
            .header("User-Agent", "NEURAX-Export")
            .json(&body)
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() || resp.status().as_u16() == 201 => {
                let data: serde_json::Value = resp.json().await.unwrap_or_default();
                if let Some(html_url) = data["content"]["html_url"].as_str() {
                    file_urls.push(html_url.to_string());
                }
                tracing::info!("[EXPORT_GITHUB] Pushed {}", file.path);
            }
            Ok(resp) => {
                let status = resp.status();
                let response_body = resp.text().await.unwrap_or_default();
                tracing::error!(
                    "[EXPORT_GITHUB] Failed to push {}: {} - {}",
                    file.path,
                    status,
                    response_body
                );
                // A 404 here almost always means the *branch* in the request
                // body doesn't exist — not the file or the repository, both
                // already confirmed above. Spell that out explicitly instead
                // of surfacing GitHub's generic "Not Found", especially when
                // the branch actually used differs from the repository's own
                // default: that mismatch is the single most common cause,
                // e.g. a client that sent an explicit "main" for a repository
                // whose real default branch is "master".
                let error = if status.as_u16() == 404 {
                    if target_branch != &repo_default_branch {
                        format!(
                            "Branch \"{}\" doesn't exist in {} — its default branch is \"{}\". \
                             Leave the Branch field blank to use it automatically, or push to a \
                             branch that actually exists.",
                            target_branch, repo, repo_default_branch
                        )
                    } else {
                        format!(
                            "GitHub returned 404 pushing {} to {}@{} even though that branch was \
                             just resolved as the repository's default — the repository may still \
                             be finishing initialization; wait a few seconds and retry. ({})",
                            file.path, repo, target_branch, response_body
                        )
                    }
                } else {
                    format!(
                        "Failed to push {} ({}): {}",
                        file.path, status, response_body
                    )
                };
                return HttpResponse::build(StatusCode::BAD_GATEWAY).json(ExportGitHubResponse {
                    success: false,
                    file_urls,
                    pr_url: None,
                    error: Some(error),
                });
            }
            Err(e) => {
                tracing::error!("[EXPORT_GITHUB] Request failed for {}: {}", file.path, e);
                return HttpResponse::build(StatusCode::BAD_GATEWAY).json(ExportGitHubResponse {
                    success: false,
                    file_urls,
                    pr_url: None,
                    error: Some(format!(
                        "GitHub API request failed for {}: {}",
                        file.path, e
                    )),
                });
            }
        }
    }

    // If creating a PR, open it
    let pr_url: Option<String> = if create_pr {
        let pr_create_url = format!("{}/pulls", api_base);
        let pr_body = serde_json::json!({
            "title": commit_message,
            "head": pr_branch,
            "base": branch,
            "body": "This PR was automatically generated by NEURAX — the AI architecture design platform.\n\nChanges include the model architecture files as designed in the NEURAX canvas.",
        });
        match client
            .post(&pr_create_url)
            .header("Authorization", format!("Bearer {}", github_token))
            .header("User-Agent", "NEURAX-Export")
            .json(&pr_body)
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() || resp.status().as_u16() == 201 => {
                let data: serde_json::Value = resp.json().await.unwrap_or_default();
                let url = data["html_url"].as_str().map(|s| s.to_string());
                tracing::info!("[EXPORT_GITHUB] PR created: {:?}", url);
                url
            }
            Ok(resp) => {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                tracing::warn!("[EXPORT_GITHUB] Failed to create PR: {} - {}", status, body);
                None
            }
            Err(e) => {
                tracing::warn!("[EXPORT_GITHUB] Failed to create PR: {}", e);
                None
            }
        }
    } else {
        None
    };

    let elapsed = start.elapsed();
    tracing::info!(
        "[EXPORT_GITHUB] Success in {}ms - {} files pushed to {}/{}",
        elapsed.as_millis(),
        req.files.len(),
        repo,
        target_branch,
    );

    HttpResponse::Ok().json(ExportGitHubResponse {
        success: true,
        file_urls,
        pr_url,
        error: None,
    })
}

async fn analyze(http_req: HttpRequest, req: web::Json<AnalyzeRequest>) -> impl Responder {
    let start = std::time::Instant::now();
    tracing::info!("[ANALYZE] Request received");

    // Serialized once and reused below for parsing — this used to run a
    // second, identical serialization after the auth check purely to get
    // the same string again. Kept at debug, not info: this is the busiest
    // endpoint in the app (fired on every canvas auto-analysis), and a raw
    // preview of the user's model design is exactly what the desktop build
    // promises never leaves the machine — echoing it into the log level
    // users see by default undercuts that promise for no operational gain.
    let payload_str = serde_json::to_string(&req.topology);
    if let Ok(ref s) = payload_str {
        tracing::debug!("[ANALYZE] Payload preview: {}...", &s[..s.len().min(500)]);
        tracing::debug!("[ANALYZE] Payload size: {} bytes", s.len());
    }

    if let Err(resp) = require_verified_email(&http_req).await {
        tracing::warn!(
            "[ANALYZE] Auth failed after {}ms",
            start.elapsed().as_millis()
        );
        return resp;
    }
    tracing::debug!(
        "[ANALYZE] Auth passed after {}ms",
        start.elapsed().as_millis()
    );

    let input = match payload_str {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("[ANALYZE] Failed to serialize topology: {}", e);
            return HttpResponse::build(StatusCode::BAD_REQUEST).body(e.to_string());
        }
    };

    // Parse the topology JSON into ModelConfig
    tracing::info!("[ANALYZE] Parsing model config...");
    let config = match neurax_parser::parse_model_config(&input) {
        Ok(c) => {
            tracing::info!(
                "[ANALYZE] Parse OK: model_type={:?}, layers={}",
                c.model.model_type,
                c.model.layers.len()
            );
            c
        }
        Err(e) => {
            tracing::error!(
                "[ANALYZE] Parse failed after {}ms: {}",
                start.elapsed().as_millis(),
                e
            );
            return HttpResponse::build(StatusCode::BAD_REQUEST).body(e.to_string());
        }
    };

    tracing::info!("[ANALYZE] Starting core analysis...");
    let result = actix_web::rt::task::spawn_blocking(move || neurax_core::run_analysis(config));

    let timeout_result = actix_web::rt::time::timeout(Duration::from_secs(60), result).await;

    let elapsed = start.elapsed();
    match timeout_result {
        Ok(Ok(Ok(analysis_result))) => {
            tracing::info!(
                "[ANALYZE] Success in {}ms - report generated",
                elapsed.as_millis()
            );
            HttpResponse::Ok().json(AnalyzeResponse {
                report: analysis_result.report,
            })
        }
        Ok(Ok(Err(e))) => {
            tracing::error!(
                "[ANALYZE] Analysis error after {}ms: {}",
                elapsed.as_millis(),
                e
            );
            HttpResponse::build(StatusCode::BAD_REQUEST).body(e.to_string())
        }
        Ok(Err(_join_err)) => {
            tracing::error!("[ANALYZE] Task join error after {}ms", elapsed.as_millis());
            HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR)
                .body("Analysis task failed unexpectedly")
        }
        Err(_timeout) => {
            tracing::error!("[ANALYZE] Timeout after {}ms", elapsed.as_millis());
            HttpResponse::build(StatusCode::GATEWAY_TIMEOUT)
                .body("Analysis timed out after 60 seconds")
        }
    }
}

/// Sweep batch_size × zero_stage × gpu_count × precision for the fastest/
/// cheapest/largest-batch feasible configuration. Each point is a full
/// `run_analysis` call — cheap (no execution, ~0-1ms each) but a large
/// candidate grid still adds up, so the grid size is capped the same way
/// `analyze_compare` caps its config count, and the whole sweep shares
/// `analyze`'s 60s timeout.
async fn sweep(http_req: HttpRequest, req: web::Json<SweepRequest>) -> impl Responder {
    let start = std::time::Instant::now();
    tracing::info!("[SWEEP] Request received");

    if let Err(resp) = require_verified_email(&http_req).await {
        return resp;
    }

    let input = match serde_json::to_string(&req.topology) {
        Ok(v) => v,
        Err(e) => return HttpResponse::build(StatusCode::BAD_REQUEST).body(e.to_string()),
    };
    let config = match neurax_parser::parse_model_config(&input) {
        Ok(c) => c,
        Err(e) => return HttpResponse::build(StatusCode::BAD_REQUEST).body(e.to_string()),
    };

    let defaults = neurax_core::sweep::SweepCandidates::defaults_for(&config);
    let requested = req.candidates.as_ref();
    let candidates = neurax_core::sweep::SweepCandidates {
        batch_sizes: requested
            .and_then(|c| c.batch_sizes.clone())
            .unwrap_or(defaults.batch_sizes),
        zero_stages: requested
            .and_then(|c| c.zero_stages.clone())
            .unwrap_or(defaults.zero_stages),
        gpu_counts: requested
            .and_then(|c| c.gpu_counts.clone())
            .unwrap_or(defaults.gpu_counts),
        precisions: requested
            .and_then(|c| c.precisions.clone())
            .unwrap_or(defaults.precisions),
    };

    let grid_size = candidates.batch_sizes.len()
        * candidates.zero_stages.len()
        * candidates.gpu_counts.len()
        * candidates.precisions.len();
    if grid_size > 512 {
        return HttpResponse::build(StatusCode::BAD_REQUEST).body(format!(
            "Sweep grid too large ({grid_size} combinations, max 512) — narrow the candidate lists"
        ));
    }

    let objective = req.objective;
    let task = actix_web::rt::task::spawn_blocking(move || {
        neurax_core::sweep::sweep_hyperparameters(&config, &candidates, objective)
    });
    let timeout_result = actix_web::rt::time::timeout(Duration::from_secs(60), task).await;

    let elapsed = start.elapsed();
    match timeout_result {
        Ok(Ok(Ok(result))) => {
            tracing::info!(
                "[SWEEP] Success in {}ms - {} points evaluated",
                elapsed.as_millis(),
                result.points.len()
            );
            HttpResponse::Ok().json(SweepResponse { result })
        }
        Ok(Ok(Err(e))) => HttpResponse::build(StatusCode::BAD_REQUEST).body(e.to_string()),
        Ok(Err(_join_err)) => HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR)
            .body("Sweep task failed unexpectedly"),
        Err(_timeout) => HttpResponse::build(StatusCode::GATEWAY_TIMEOUT)
            .body("Sweep timed out after 60 seconds"),
    }
}

async fn analyze_compare(
    http_req: HttpRequest,
    state: web::Data<AppState>,
    req: web::Json<CompareRequest>,
) -> impl Responder {
    let start = std::time::Instant::now();
    tracing::info!(
        "[COMPARE] Request received with {} configs",
        req.configs.len()
    );

    let _user_id = match auth_any(&http_req, &state).await {
        Ok(id) => id,
        Err(resp) => return resp,
    };
    let _ = check_api_key_scope(&http_req, &state, "analyze");

    // Limit the number of configs to prevent abuse
    if req.configs.len() > 8 {
        return HttpResponse::build(StatusCode::BAD_REQUEST)
            .body("Maximum 8 hardware configurations for comparison");
    }

    let db = neurax_hardware_db::HardwareDatabase::new();
    let configs = req.configs.clone();
    let topology = req.topology.clone();

    // Run all analyses in a blocking task to avoid blocking the async runtime
    let result = actix_web::rt::task::spawn_blocking(move || {
        let mut results = Vec::with_capacity(configs.len());

        for cfg in &configs {
            let label = format!(
                "{} × {} @ {}",
                cfg.gpu_count.unwrap_or(1),
                cfg.hardware,
                cfg.precision.as_deref().unwrap_or("fp16")
            );

            // Look up GPU spec from hardware database
            let gpu_spec = db.get_gpu_or_fallback(&cfg.hardware);

            // Clone the topology and override hardware section
            let mut topology = topology.clone();

            // `neurax_parser::HardwareConfig` deserializes `hardware.gpus: [...]`
            // (a list of GPU objects), never a flat `hardware.name` /
            // `hardware.tflops_fp16`. Writing those fields directly onto the
            // `hardware` object — as this used to do — produced JSON the
            // parser's `RawHardware` struct has no field for, so it was
            // silently dropped and every comparison kept re-analyzing the
            // project's own GPU regardless of which one was requested (every
            // config in a comparison came back identical). The override has
            // to land inside `hardware.gpus[0]` to actually reach the parser.
            let gpu_entry = serde_json::json!({
                "name": cfg.hardware,
                "count": cfg.gpu_count.unwrap_or(1),
                "memory_gb": cfg.gpu_memory_gb.unwrap_or(gpu_spec.memory_gb),
                "tflops_fp16": gpu_spec.tflops_fp16,
                "tflops_fp32": gpu_spec.tflops_fp32,
                "memory_bandwidth_gb_s": gpu_spec.memory_bandwidth_gbs,
                "tensor_cores": gpu_spec.tensor_cores,
                "nvlink": gpu_spec.nvlink,
            });
            if let Some(hw) = topology.get_mut("hardware") {
                if let Some(hw_obj) = hw.as_object_mut() {
                    hw_obj.insert("gpus".to_string(), serde_json::json!([gpu_entry]));
                }
            } else if let Some(obj) = topology.as_object_mut() {
                obj.insert(
                    "hardware".to_string(),
                    serde_json::json!({ "gpus": [gpu_entry] }),
                );
            }

            // Override precision if specified
            if let Some(ref precision) = cfg.precision {
                if let Some(training) = topology.get_mut("training") {
                    if let Some(training_obj) = training.as_object_mut() {
                        training_obj.insert(
                            "precision".to_string(),
                            serde_json::Value::String(precision.clone()),
                        );
                    }
                } else {
                    if let Some(obj) = topology.as_object_mut() {
                        obj.insert(
                            "training".to_string(),
                            serde_json::json!({ "precision": precision }),
                        );
                    }
                }
            }

            // Override batch size if specified
            if let Some(batch_size) = cfg.batch_size {
                if let Some(training) = topology.get_mut("training") {
                    if let Some(training_obj) = training.as_object_mut() {
                        training_obj.insert(
                            "batch_size".to_string(),
                            serde_json::Value::Number(serde_json::Number::from(batch_size)),
                        );
                    }
                } else {
                    if let Some(obj) = topology.as_object_mut() {
                        obj.insert(
                            "training".to_string(),
                            serde_json::json!({ "batch_size": batch_size }),
                        );
                    }
                }
            }

            // Parse and run analysis
            let input = match serde_json::to_string(&topology) {
                Ok(v) => v,
                Err(e) => {
                    results.push(CompareResultItem {
                        label,
                        hardware: cfg.hardware.clone(),
                        precision: cfg.precision.clone().unwrap_or_else(|| "fp16".to_string()),
                        batch_size: cfg.batch_size.unwrap_or(1),
                        gpu_count: cfg.gpu_count.unwrap_or(1),
                        report: None,
                        error: Some(format!("Failed to serialize topology: {}", e)),
                    });
                    continue;
                }
            };

            let config = match neurax_parser::parse_model_config(&input) {
                Ok(c) => c,
                Err(e) => {
                    results.push(CompareResultItem {
                        label,
                        hardware: cfg.hardware.clone(),
                        precision: cfg.precision.clone().unwrap_or_else(|| "fp16".to_string()),
                        batch_size: cfg.batch_size.unwrap_or(1),
                        gpu_count: cfg.gpu_count.unwrap_or(1),
                        report: None,
                        error: Some(format!("Parse error: {}", e)),
                    });
                    continue;
                }
            };

            let result = neurax_core::run_analysis(config);
            match result {
                Ok(analysis_result) => {
                    results.push(CompareResultItem {
                        label,
                        hardware: cfg.hardware.clone(),
                        precision: cfg.precision.clone().unwrap_or_else(|| "fp16".to_string()),
                        batch_size: cfg.batch_size.unwrap_or(1),
                        gpu_count: cfg.gpu_count.unwrap_or(1),
                        report: Some(analysis_result.report),
                        error: None,
                    });
                }
                Err(e) => {
                    results.push(CompareResultItem {
                        label,
                        hardware: cfg.hardware.clone(),
                        precision: cfg.precision.clone().unwrap_or_else(|| "fp16".to_string()),
                        batch_size: cfg.batch_size.unwrap_or(1),
                        gpu_count: cfg.gpu_count.unwrap_or(1),
                        report: None,
                        error: Some(e.to_string()),
                    });
                }
            }
        }

        results
    })
    .await;

    let elapsed = start.elapsed();
    match result {
        Ok(results) => {
            tracing::info!(
                "[COMPARE] Completed {} configs in {}ms",
                results.len(),
                elapsed.as_millis()
            );
            HttpResponse::Ok().json(CompareResponse { results })
        }
        Err(e) => {
            tracing::error!(
                "[COMPARE] Task join error after {}ms: {}",
                elapsed.as_millis(),
                e
            );
            HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR)
                .body("Comparison task failed unexpectedly")
        }
    }
}

async fn time_machine(http_req: HttpRequest, req: web::Json<TimeMachineRequest>) -> impl Responder {
    let start = std::time::Instant::now();
    tracing::info!("[TIMEMACHINE] Request received");

    if let Err(resp) = require_verified_email(&http_req).await {
        return resp;
    }

    let input = match serde_json::to_string(&req.topology) {
        Ok(v) => v,
        Err(e) => return HttpResponse::build(StatusCode::BAD_REQUEST).body(e.to_string()),
    };

    let config = match neurax_parser::parse_model_config(&input) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("[TIMEMACHINE] Parse failed: {}", e);
            return HttpResponse::build(StatusCode::BAD_REQUEST).body(e.to_string());
        }
    };

    let params = req.params.clone();
    let result = actix_web::rt::task::spawn_blocking(move || neurax_core::run_analysis(config));
    let timeout_result = actix_web::rt::time::timeout(Duration::from_secs(60), result).await;

    match timeout_result {
        Ok(Ok(Ok(analysis_result))) => {
            let report = &analysis_result.report;
            let projection = neurax_ir::report::project_time_machine(
                &report.metrics,
                &report.recommendations,
                report.confidence_score,
                &params,
            );
            tracing::info!("[TIMEMACHINE] Success in {}ms", start.elapsed().as_millis());
            HttpResponse::Ok().json(TimeMachineResponse { projection })
        }
        Ok(Ok(Err(e))) => HttpResponse::build(StatusCode::BAD_REQUEST).body(e.to_string()),
        Ok(Err(_)) => HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR)
            .body("Time Machine task failed unexpectedly"),
        Err(_) => HttpResponse::build(StatusCode::GATEWAY_TIMEOUT).body("Time Machine timed out"),
    }
}

async fn health() -> impl Responder {
    HttpResponse::Ok().json(HealthResponse { status: "ok" })
}

async fn plugin_validate(req: web::Json<PluginValidateRequest>) -> impl Responder {
    // Plugin validation is a stub for now - just validate it's valid JSON
    if serde_json::to_string(&req.plugin).is_err() {
        return HttpResponse::build(StatusCode::BAD_REQUEST).body("Invalid JSON");
    }

    HttpResponse::Ok().json(PluginValidateResponse { ok: true })
}

async fn get_presets() -> impl Responder {
    let presets = presets::get_all_presets_metadata();
    HttpResponse::Ok().json(presets)
}

async fn get_preset(path: web::Path<String>) -> impl Responder {
    let id = path.into_inner();
    match presets::get_preset_by_id(&id) {
        Some(p) => HttpResponse::Ok().json(p),
        None => HttpResponse::NotFound().body("Preset not found"),
    }
}

// ─── Streaming Analysis Endpoints ──────────────────────────────────

/// POST /analyze/stream — Start a streaming analysis job, returns job_id immediately
async fn analyze_stream_start(
    http_req: HttpRequest,
    req: web::Json<AnalyzeStreamRequest>,
    state: web::Data<AppState>,
) -> impl Responder {
    let user = match require_verified_email(&http_req).await {
        Ok(u) => u,
        Err(resp) => return resp,
    };

    let job_id = uuid::Uuid::new_v4().to_string();
    let created_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    // Create broadcast channel for SSE events
    let (tx, _rx) = broadcast::channel::<String>(256);
    state.channels.insert(job_id.clone(), tx);

    // Insert job info
    let view_token = uuid::Uuid::new_v4().to_string();
    state.jobs.insert(
        job_id.clone(),
        JobInfo {
            job_id: job_id.clone(),
            user_id: user.id.clone(),
            view_token: view_token.clone(),
            status: "running".to_string(),
            created_at_ms: created_at,
            completed_at_ms: None,
            error: None,
        },
    );

    let job_id_clone = job_id.clone();
    let topology = req.topology.clone();
    let state_inner = state.into_inner();

    // Spawn the analysis in a background task
    actix_web::rt::spawn(async move {
        let input = match serde_json::to_string(&topology) {
            Ok(v) => v,
            Err(e) => {
                // Send error event
                if let Some(tx) = state_inner.channels.get(&job_id_clone) {
                    let event = serde_json::json!({
                        "type": "Failed",
                        "data": { "job_id": job_id_clone, "error": e.to_string(), "phase": "parse" }
                    });
                    let _ = tx.send(event.to_string());
                }
                // Update job status
                if let Some(mut job) = state_inner.jobs.get_mut(&job_id_clone) {
                    job.status = "failed".to_string();
                    job.error = Some(e.to_string());
                }
                return;
            }
        };

        let config = match neurax_parser::parse_model_config(&input) {
            Ok(c) => c,
            Err(e) => {
                if let Some(tx) = state_inner.channels.get(&job_id_clone) {
                    let event = serde_json::json!({
                        "type": "Failed",
                        "data": { "job_id": job_id_clone, "error": e.to_string(), "phase": "parse" }
                    });
                    let _ = tx.send(event.to_string());
                }
                if let Some(mut job) = state_inner.jobs.get_mut(&job_id_clone) {
                    job.status = "failed".to_string();
                    job.error = Some(e.to_string());
                }
                return;
            }
        };

        // Create emitter that broadcasts events
        let (event_sender, event_receiver) =
            tokio::sync::broadcast::channel::<neurax_core::streaming::AnalysisEvent>(256);
        // Spawn a task that forwards AnalysisEvents to the SSE string channel
        {
            let channels_clone = state_inner.channels.clone();
            let job_id_forward = job_id_clone.clone();
            actix_web::rt::spawn(async move {
                let mut rx = event_receiver;
                loop {
                    match rx.recv().await {
                        Ok(event) => {
                            let event_json = serde_json::to_string(&event).unwrap_or_default();
                            if let Some(tx) = channels_clone.get(&job_id_forward) {
                                let _ = tx.send(event_json);
                            }
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                    }
                }
            });
        }
        let emitter = neurax_core::streaming::SharedEmitter::new(
            neurax_core::streaming::BroadcastEmitter::from_sender(event_sender),
        );

        // Clone job_id for use after spawn_blocking
        let job_id_result = job_id_clone.clone();
        // Run analysis in blocking context
        let result = actix_web::rt::task::spawn_blocking(move || {
            neurax_core::streaming::run_analysis_streaming_fallible(config, emitter, &job_id_clone)
        })
        .await;

        match result {
            Ok(Ok(analysis_result)) => {
                // Store the result
                let report_value = match analysis_result.to_json() {
                    Ok(json_str) => serde_json::from_str::<serde_json::Value>(&json_str)
                        .unwrap_or_else(|_| serde_json::json!({"error": "parse failed"})),
                    Err(e) => serde_json::json!({"error": e.to_string()}),
                };

                state_inner
                    .results
                    .insert(job_id_result.clone(), report_value);

                // Update job status
                if let Some(mut job) = state_inner.jobs.get_mut(&job_id_result) {
                    job.status = "completed".to_string();
                    job.completed_at_ms = Some(
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis() as u64,
                    );
                }
            }
            Ok(Err(e)) => {
                if let Some(mut job) = state_inner.jobs.get_mut(&job_id_result) {
                    job.status = "failed".to_string();
                    job.error = Some(e.to_string());
                }
            }
            Err(e) => {
                if let Some(mut job) = state_inner.jobs.get_mut(&job_id_result) {
                    job.status = "failed".to_string();
                    job.error = Some(format!("Task join error: {}", e));
                }
            }
        }

        // Clean up channel after a delay (allow clients to drain events)
        let channels = state_inner.channels.clone();
        let job_id_cleanup = job_id_result.clone();
        actix_web::rt::spawn(async move {
            actix_web::rt::time::sleep(Duration::from_secs(30)).await;
            channels.remove(&job_id_cleanup);
        });
    });

    HttpResponse::Accepted().json(AnalyzeStreamResponse { job_id, view_token })
}

/// GET /analyze/stream/{job_id} — SSE endpoint for streaming analysis events
/// Auth: requires ?token=<view_token> query parameter
async fn analyze_stream_events(
    path: web::Path<String>,
    query: web::Query<std::collections::HashMap<String, String>>,
    state: web::Data<AppState>,
) -> HttpResponse {
    let job_id = path.into_inner();

    // Auth via view token
    let token = match query.get("token") {
        Some(t) => t.clone(),
        None => return HttpResponse::Unauthorized().body("Missing view token"),
    };

    // Check job exists and token matches
    let owned = match state.jobs.get(&job_id) {
        Some(job) => job.view_token == token,
        None => return HttpResponse::NotFound().body("Job not found"),
    };
    if !owned {
        return HttpResponse::Forbidden().body("Invalid token");
    }

    // Both branches below need to hand the client a Completed+Result pair
    // for a job that already finished, without a live subscriber to relay it.
    let immediate_result_body = || {
        let result = state.results.get(&job_id)?;
        let completed = serde_json::json!({
            "type": "Completed",
            "data": { "job_id": job_id, "total_ms": 0 }
        });
        let result_event = serde_json::json!({
            "type": "Result",
            "data": result.value()
        });
        Some(format!("data: {completed}\n\ndata: {result_event}\n\n"))
    };

    // Get or create a receiver.
    //
    // A `tokio::sync::broadcast` channel does not replay past sends to a
    // subscriber that joins late — and NEURAX's whole point is that analysis
    // is fast (well under 50ms for most models), so it routinely finishes
    // and broadcasts its Completed/Result events before this handler's GET
    // request even arrives, let alone subscribes. A subscriber that joins
    // after the fact would then sit on `rx.recv().await` forever: nothing
    // left to receive, the connection never closes, and the client's
    // "Analyzing…" state has nothing to end it.
    //
    // The fix has to subscribe *before* checking `state.results`, not after:
    // checking first and subscribing second leaves exactly the same race in
    // the gap between the two — the job can finish and broadcast in that
    // gap too. Subscribing first means every event sent from this point
    // forward is guaranteed to be captured by `rx`, so checking results
    // immediately afterward can only miss a job that finished *before* the
    // subscribe — which is fine, because a finished job's data is already
    // sitting in `state.results` for exactly that check to find.
    let rx = match state.channels.get(&job_id) {
        Some(tx) => tx.subscribe(),
        None => {
            // No channel at all — either this job never existed, or it's
            // old enough that the 30s cleanup already removed it. Either
            // way, `state.results` is the only place left to look.
            if let Some(body) = immediate_result_body() {
                return HttpResponse::Ok()
                    .content_type("text/event-stream")
                    .insert_header(("Cache-Control", "no-cache"))
                    .insert_header(("Connection", "keep-alive"))
                    .body(body);
            }
            return HttpResponse::NotFound().body("Job stream expired");
        }
    };

    if let Some(body) = immediate_result_body() {
        return HttpResponse::Ok()
            .content_type("text/event-stream")
            .insert_header(("Cache-Control", "no-cache"))
            .insert_header(("Connection", "keep-alive"))
            .body(body);
    }

    // Stream events via SSE.
    //
    // `rx.recv()` alone is not a reliable way to learn that the job is
    // done: a `tokio::sync::broadcast::Sender::send()` made while this
    // handler had zero subscribers — perfectly possible up until the
    // `subscribe()` call above completes — silently drops that message
    // rather than queuing it, and a receiver that subscribes even one
    // instant after "Completed" was sent has no way to see it, no matter
    // how the subscribe-vs-check ordering above is arranged. `state.jobs`
    // and `state.results`, on the other hand, are updated in place and
    // stay correct regardless of when anyone looks — so a periodic check
    // of them, racing against `rx.recv()`, is what actually closes this:
    // even a subscriber that missed every single broadcast event notices
    // the job finished within one tick and ends the stream, instead of
    // waiting forever for a message that already came and went.
    let state_inner = state.into_inner();
    let job_id_poll = job_id.clone();
    let stream = async_stream::stream! {
        let mut rx = rx;
        let mut poll = tokio::time::interval(Duration::from_millis(50));
        poll.tick().await; // the first tick fires immediately; consume it
        loop {
            tokio::select! {
                recv = rx.recv() => {
                    match recv {
                        Ok(event_json) => {
                            let event: serde_json::Value = match serde_json::from_str(&event_json) {
                                Ok(v) => v,
                                Err(_) => continue,
                            };
                            yield Ok::<_, actix_web::Error>(actix_web::web::Bytes::from(format!("data: {}\n\n", event_json)));

                            let event_type = event.get("type").and_then(|v| v.as_str()).unwrap_or("");
                            if event_type == "Failed" {
                                break;
                            }
                            if event_type == "Completed" {
                                // `Completed` is emitted by the analysis
                                // engine itself, inside `spawn_blocking`,
                                // strictly before that call returns and this
                                // handler's caller writes to `state.results`
                                // — so the result can genuinely not be there
                                // yet at this exact instant. Do not break:
                                // fall through to the next loop iteration,
                                // where `poll.tick()` (at most 50ms away)
                                // checks `state.results` again and sends
                                // Result — and terminates the stream — the
                                // moment it's actually there, without this
                                // branch needing its own retry loop.
                                if let Some(result) = state_inner.results.get(&job_id_poll) {
                                    let result_json = serde_json::json!({
                                        "type": "Result",
                                        "data": result.value()
                                    });
                                    yield Ok(actix_web::web::Bytes::from(format!("data: {}\n\n", result_json)));
                                    break;
                                }
                            }
                        }
                        Err(broadcast::error::RecvError::Lagged(n)) => {
                            yield Ok(actix_web::web::Bytes::from(format!("data: {{\"type\":\"Lagged\",\"data\":{{\"count\":{}}}}}\n\n", n)));
                        }
                        Err(broadcast::error::RecvError::Closed) => {
                            if let Some(result) = state_inner.results.get(&job_id_poll) {
                                let result_json = serde_json::json!({
                                    "type": "Result",
                                    "data": result.value()
                                });
                                yield Ok(actix_web::web::Bytes::from(format!("data: {}\n\n", result_json)));
                            }
                            break;
                        }
                    }
                }
                _ = poll.tick() => {
                    if let Some(result) = state_inner.results.get(&job_id_poll) {
                        let result_json = serde_json::json!({
                            "type": "Result",
                            "data": result.value()
                        });
                        yield Ok(actix_web::web::Bytes::from(format!("data: {}\n\n", result_json)));
                        break;
                    }
                    if let Some(job) = state_inner.jobs.get(&job_id_poll) {
                        if job.status == "failed" {
                            let failed_json = serde_json::json!({
                                "type": "Failed",
                                "data": { "job_id": job_id_poll, "error": job.error.clone().unwrap_or_default(), "phase": "unknown" }
                            });
                            yield Ok(actix_web::web::Bytes::from(format!("data: {}\n\n", failed_json)));
                            break;
                        }
                    }
                }
            }
        }
    };

    HttpResponse::Ok()
        .content_type("text/event-stream")
        .insert_header(("Cache-Control", "no-cache"))
        .insert_header(("Connection", "keep-alive"))
        .streaming(stream)
}

/// GET /analyze/result/{job_id} — Get the final result of a streaming analysis
async fn analyze_result(
    path: web::Path<String>,
    req: HttpRequest,
    state: web::Data<AppState>,
) -> impl Responder {
    let user = match require_verified_email(&req).await {
        Ok(u) => u,
        Err(resp) => return resp,
    };
    let job_id = path.into_inner();

    // Verify job ownership
    match state.jobs.get(&job_id) {
        Some(job) if job.user_id != user.id => {
            return HttpResponse::Forbidden().body("Access denied");
        }
        None => return HttpResponse::NotFound().body("Job not found"),
        _ => {}
    }

    match state.jobs.get(&job_id) {
        Some(job) => {
            if job.status == "running" {
                return HttpResponse::Accepted().json(serde_json::json!({
                    "status": "running",
                    "job_id": job_id,
                }));
            }
            if job.status == "failed" {
                return HttpResponse::build(StatusCode::BAD_REQUEST).json(serde_json::json!({
                    "status": "failed",
                    "job_id": job_id,
                    "error": job.error,
                }));
            }
            // Completed — return result
            match state.results.get(&job_id) {
                Some(result) => HttpResponse::Ok().json(serde_json::json!({
                    "status": "completed",
                    "job_id": job_id,
                    "report": result.value(),
                })),
                None => HttpResponse::NotFound().body("Result not found"),
            }
        }
        None => HttpResponse::NotFound().body("Job not found"),
    }
}

/// GET /analyze/status/{job_id} — Get the status of a streaming analysis job
async fn analyze_status(
    path: web::Path<String>,
    req: HttpRequest,
    state: web::Data<AppState>,
) -> impl Responder {
    let user = match require_verified_email(&req).await {
        Ok(u) => u,
        Err(resp) => return resp,
    };
    let job_id = path.into_inner();

    match state.jobs.get(&job_id) {
        Some(job) if job.user_id != user.id => {
            return HttpResponse::Forbidden().body("Access denied");
        }
        None => return HttpResponse::NotFound().body("Job not found"),
        _ => {}
    }

    match state.jobs.get(&job_id) {
        Some(job) => HttpResponse::Ok().json(serde_json::json!({
            "job_id": job.job_id,
            "status": job.status,
            "created_at_ms": job.created_at_ms,
            "completed_at_ms": job.completed_at_ms,
            "error": job.error,
        })),
        None => HttpResponse::NotFound().body("Job not found"),
    }
}

// ─── Project CRUD Handlers ──────────────────────────────────────────

async fn projects_list(http_req: HttpRequest, state: web::Data<AppState>) -> impl Responder {
    let user = match require_verified_email(&http_req).await {
        Ok(u) => u,
        Err(resp) => return resp,
    };

    let projects: Vec<Project> = state
        .projects
        .iter()
        .filter(|entry| entry.key().user_id == user.id)
        .map(|entry| entry.value().clone())
        .collect();

    HttpResponse::Ok().json(ProjectListResponse { projects })
}

async fn projects_create(
    http_req: HttpRequest,
    state: web::Data<AppState>,
    req: web::Json<CreateProjectRequest>,
) -> impl Responder {
    let user = match require_verified_email(&http_req).await {
        Ok(u) => u,
        Err(resp) => return resp,
    };

    // Limit projects per user (max 50 on free tier)
    let user_count = state
        .projects
        .iter()
        .filter(|entry| entry.key().user_id == user.id)
        .count();

    if user_count >= 50 {
        return HttpResponse::build(StatusCode::FORBIDDEN)
            .body("Project limit reached (max 50). Upgrade your plan for more.");
    }

    let now = chrono::Utc::now().to_rfc3339();
    let project = Project {
        id: uuid::Uuid::new_v4().to_string(),
        user_id: user.id.clone(),
        name: req.name.clone(),
        description: req.description.clone(),
        architecture: req.architecture.clone(),
        canvas: req.canvas.clone(),
        hardware_config: req.hardware_config.clone(),
        last_analysis: req.last_analysis.clone(),
        created_at: now.clone(),
        updated_at: now,
    };

    let key = ProjectKey {
        user_id: user.id,
        id: project.id.clone(),
    };

    state.projects.insert(key, project.clone());

    HttpResponse::Created().json(ProjectResponse { project })
}

async fn projects_get(
    http_req: HttpRequest,
    state: web::Data<AppState>,
    path: web::Path<String>,
) -> impl Responder {
    let user = match require_verified_email(&http_req).await {
        Ok(u) => u,
        Err(resp) => return resp,
    };

    let project_id = path.into_inner();
    let key = ProjectKey {
        user_id: user.id,
        id: project_id,
    };

    match state.projects.get(&key) {
        Some(entry) => HttpResponse::Ok().json(ProjectResponse {
            project: entry.value().clone(),
        }),
        None => HttpResponse::NotFound().body("Project not found"),
    }
}

async fn projects_update(
    http_req: HttpRequest,
    state: web::Data<AppState>,
    path: web::Path<String>,
    req: web::Json<UpdateProjectRequest>,
) -> impl Responder {
    let user = match require_verified_email(&http_req).await {
        Ok(u) => u,
        Err(resp) => return resp,
    };

    let project_id = path.into_inner();
    let key = ProjectKey {
        user_id: user.id.clone(),
        id: project_id,
    };

    let mut entry = match state.projects.get_mut(&key) {
        Some(e) => e,
        None => return HttpResponse::NotFound().body("Project not found"),
    };

    if let Some(name) = &req.name {
        entry.value_mut().name = name.clone();
    }
    if let Some(desc) = &req.description {
        entry.value_mut().description = Some(desc.clone());
    }
    if let Some(arch) = &req.architecture {
        entry.value_mut().architecture = Some(arch.clone());
    }
    if let Some(canvas) = &req.canvas {
        entry.value_mut().canvas = canvas.clone();
    }
    if let Some(hw) = &req.hardware_config {
        entry.value_mut().hardware_config = Some(hw.clone());
    }
    if let Some(analysis) = &req.last_analysis {
        entry.value_mut().last_analysis = Some(analysis.clone());
    }
    entry.value_mut().updated_at = chrono::Utc::now().to_rfc3339();

    let updated = entry.value().clone();
    drop(entry);

    HttpResponse::Ok().json(ProjectResponse { project: updated })
}

async fn projects_delete(
    http_req: HttpRequest,
    state: web::Data<AppState>,
    path: web::Path<String>,
) -> impl Responder {
    let user = match require_verified_email(&http_req).await {
        Ok(u) => u,
        Err(resp) => return resp,
    };

    let project_id = path.into_inner();
    let key = ProjectKey {
        user_id: user.id,
        id: project_id,
    };

    match state.projects.remove(&key) {
        Some(_) => HttpResponse::Ok().json(serde_json::json!({"deleted": true})),
        None => HttpResponse::NotFound().body("Project not found"),
    }
}

// ─── Public Shares ──────────────────────────────────────────────────
//
// A share is a published, read-only analysis snapshot: the growth-loop
// mechanic of turning "I analysed a model" into a link someone else can
// open with no account and no install. Anonymous by design, on both ends —
// creating and viewing need no session — so ownership for deletion is
// proven by possessing `edit_token`, the same model link-sharing tools like
// Pastebin use, not by being logged in as the right user.

const SHARE_ID_ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

/// Short and URL-friendly on purpose — this is meant to be pasted into a
/// tweet or a Slack message, not to be a UUID.
fn generate_share_id() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..10)
        .map(|_| SHARE_ID_ALPHABET[rng.gen_range(0..SHARE_ID_ALPHABET.len())] as char)
        .collect()
}

/// A bearer credential, not a lookup key, so it is long and opaque like the
/// API keys generated above rather than URL-friendly like the share id.
fn generate_edit_token() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..32)
        .map(|_| format!("{:02x}", rng.gen::<u8>()))
        .collect()
}

/// How much of the original design a public share carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShareMode {
    /// Numbers only — the full analysis report, no topology. Safe by
    /// default: nothing about *how* the model is built is disclosed.
    Card,
    /// The report plus the full node/connection graph, scrubbed of
    /// free-text labels, so a viewer can open it in NEURAX and edit it.
    Full,
}

/// A published, read-only analysis snapshot.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Share {
    pub id: String,
    pub mode: ShareMode,
    /// Chosen by the sharer at publish time — never the document's own
    /// name, which may be an internal project codename.
    pub display_name: String,
    pub family: Option<String>,
    /// The full `AnalysisResult` snapshot from the frontend, frozen at
    /// share time so the link keeps showing what it showed when it was
    /// published even if the original document later changes. Opaque JSON
    /// here on purpose — its shape is owned by the frontend, same as
    /// `AppState::results`.
    pub report: serde_json::Value,
    /// `{ nodes, connections, groups }`, present only for `ShareMode::Full`
    /// and scrubbed server-side before storage — see `scrub_design`. Never
    /// trust the client to have already redacted this itself.
    pub design: Option<serde_json::Value>,
    pub created_at: String,
    #[serde(skip_serializing)]
    pub edit_token: String,
    pub view_count: u64,
}

#[derive(Debug, serde::Deserialize)]
struct CreateShareRequest {
    mode: ShareMode,
    display_name: String,
    family: Option<String>,
    report: serde_json::Value,
    design: Option<serde_json::Value>,
}

#[derive(serde::Serialize)]
struct CreateShareResponse {
    id: String,
    edit_token: String,
}

#[derive(serde::Serialize)]
struct ShareResponse {
    share: Share,
}

/// Strips fields that could carry a user's free-text notes rather than the
/// architecture itself: node and group labels (often internal project
/// codenames) become generic, type-based names, and any custom
/// hyperparameters attached at the model level are dropped outright rather
/// than merely hidden. Numeric/structural parameters (hidden_size,
/// num_layers, kernel_size, ...) are left untouched — sharing those on
/// purpose is the entire point of `ShareMode::Full`.
fn scrub_design(design: &mut serde_json::Value) {
    if let Some(nodes) = design.get_mut("nodes").and_then(|v| v.as_array_mut()) {
        for node in nodes {
            if let Some(obj) = node.as_object_mut() {
                let type_label = obj
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Block")
                    .to_string();
                obj.insert(
                    "name".to_string(),
                    serde_json::Value::String(format!("{type_label} block")),
                );
            }
        }
    }
    if let Some(groups) = design.get_mut("groups").and_then(|v| v.as_array_mut()) {
        for (i, group) in groups.iter_mut().enumerate() {
            if let Some(obj) = group.as_object_mut() {
                obj.insert(
                    "name".to_string(),
                    serde_json::Value::String(format!("Group {}", i + 1)),
                );
            }
        }
    }
    if let Some(obj) = design.as_object_mut() {
        obj.remove("customParams");
    }
}

async fn shares_create(
    state: web::Data<AppState>,
    req: web::Json<CreateShareRequest>,
) -> impl Responder {
    if req.display_name.trim().is_empty() {
        return HttpResponse::build(StatusCode::BAD_REQUEST).body("display_name is required");
    }

    // A Card share never carries a topology, regardless of what the client
    // sent — enforced server-side, not just by client intent. A Full share
    // is scrubbed server-side too, for the same reason.
    let design = match req.mode {
        ShareMode::Full => req.design.clone().map(|mut d| {
            scrub_design(&mut d);
            d
        }),
        ShareMode::Card => None,
    };

    let id = loop {
        let candidate = generate_share_id();
        if !state.shares.contains_key(&candidate) {
            break candidate;
        }
    };
    let edit_token = generate_edit_token();

    let share = Share {
        id: id.clone(),
        mode: req.mode,
        display_name: req.display_name.trim().to_string(),
        family: req.family.clone(),
        report: req.report.clone(),
        design,
        created_at: chrono::Utc::now().to_rfc3339(),
        edit_token: edit_token.clone(),
        view_count: 0,
    };

    state.shares.insert(id.clone(), share);

    HttpResponse::Created().json(CreateShareResponse { id, edit_token })
}

async fn shares_get(state: web::Data<AppState>, path: web::Path<String>) -> impl Responder {
    let id = path.into_inner();
    match state.shares.get_mut(&id) {
        Some(mut entry) => {
            entry.view_count += 1;
            HttpResponse::Ok().json(ShareResponse {
                share: entry.value().clone(),
            })
        }
        None => HttpResponse::NotFound().body("Share not found"),
    }
}

/// A filename derived from the share's display name, safe on every platform
/// NEURAX ships to — same character rules as the desktop app's own
/// `suggestedFileName` in `neuraxFile.ts`, so a link and a local save behave
/// the same way. Kept here rather than shared with the frontend because the
/// two run in different languages; a mismatch would only ever affect the
/// cosmetic filename, never the file's content.
fn share_download_filename(display_name: &str) -> String {
    let cleaned: String = display_name
        .trim()
        .chars()
        .filter(|c| !c.is_control() && !"<>:\"/\\|?*".contains(*c))
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join("-");
    let cleaned = cleaned.trim_matches(|c| c == '.' || c == '-');
    let name = if cleaned.is_empty() {
        "untitled-design"
    } else {
        cleaned
    };
    format!("{}.json", &name[..name.len().min(80)])
}

/// The raw, unwrapped content behind a share — what a URL-based download
/// should return, as opposed to `shares_get`'s `{ share: ... }` API envelope.
/// Card shares download just the report; Full shares bundle the (already
/// scrubbed) design alongside it, since both are needed to actually rebuild
/// the architecture locally.
async fn shares_download(state: web::Data<AppState>, path: web::Path<String>) -> impl Responder {
    let id = path.into_inner();
    let Some(mut entry) = state.shares.get_mut(&id) else {
        return HttpResponse::NotFound().body("Share not found");
    };
    entry.view_count += 1;

    let filename = share_download_filename(&entry.display_name);
    let body = match entry.design.clone() {
        Some(design) => serde_json::json!({
            "schema_version": "1.0.0",
            "display_name": entry.display_name,
            "family": entry.family,
            "report": entry.report,
            "design": design,
        }),
        None => serde_json::json!({
            "schema_version": "1.0.0",
            "display_name": entry.display_name,
            "family": entry.family,
            "report": entry.report,
        }),
    };

    HttpResponse::Ok()
        .content_type("application/json")
        .insert_header((
            header::CONTENT_DISPOSITION,
            format!("attachment; filename=\"{filename}\""),
        ))
        .json(body)
}

async fn shares_delete(
    state: web::Data<AppState>,
    http_req: HttpRequest,
    path: web::Path<String>,
) -> impl Responder {
    let id = path.into_inner();
    let provided_token = http_req
        .headers()
        .get("X-Edit-Token")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    match state.shares.get(&id) {
        Some(entry) => {
            // Constant-time comparison: `edit_token` is a bearer credential,
            // same reasoning as the webhook signature check elsewhere in
            // this file.
            let matches: bool = entry
                .edit_token
                .as_bytes()
                .ct_eq(provided_token.as_bytes())
                .into();
            if !matches {
                return HttpResponse::build(StatusCode::FORBIDDEN).body("Invalid edit token");
            }
        }
        None => return HttpResponse::NotFound().body("Share not found"),
    }

    state.shares.remove(&id);
    HttpResponse::NoContent().finish()
}

// ─── Credits ────────────────────────────────────────────────────────

/// Per-user credit tracking stored in AppState
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CreditInfo {
    pub user_id: String,
    /// Credits used this billing period
    pub used: u32,
    /// Credits limit for this billing period (based on plan)
    pub limit: u32,
    /// Plan tier
    pub plan: String,
    /// Billing period start (ISO 8601)
    pub period_start: String,
    /// Billing period end (ISO 8601)
    pub period_end: String,
}

/// Plan credit limits
fn plan_credit_limit(plan: &str) -> u32 {
    match plan {
        "free" => 10,
        "essential" => 100,
        "architect" => 1000,
        "elite" => u32::MAX, // unlimited
        _ => 10,
    }
}

#[derive(Debug, serde::Serialize)]
struct CreditsResponse {
    credits: CreditInfo,
}

async fn credits_get(http_req: HttpRequest, state: web::Data<AppState>) -> impl Responder {
    let user = match get_supabase_user(&http_req).await {
        Ok(u) => u,
        Err(resp) => return resp,
    };

    // Determine plan
    let plan = if noauth_enabled() {
        env::var("NEURAX_MOCK_PLAN")
            .ok()
            .and_then(|p| normalize_plan_tier(&p))
            .unwrap_or_else(|| "elite".to_string())
    } else {
        // Try to get plan from profile
        match fetch_user_profile(&user.id).await {
            Ok(profile) => {
                if let Some(override_plan) = profile
                    .plan_override
                    .as_deref()
                    .and_then(normalize_plan_tier)
                {
                    override_plan
                } else {
                    fetch_active_subscription_plan(&user.id)
                        .await
                        .ok()
                        .flatten()
                        .unwrap_or_else(|| "free".to_string())
                }
            }
            Err(_) => "free".to_string(),
        }
    };

    let limit = plan_credit_limit(&plan);

    // Get or create credit tracking entry
    let now = chrono::Utc::now();
    let period_start = now.with_day0(0).unwrap_or(now).to_rfc3339();
    let period_end = {
        let next_month = (now.month() % 12) + 1;
        now.with_month(next_month).unwrap_or(now).to_rfc3339()
    };

    // Get or create the credit tracking entry, then read it back.
    // Hold a single mutable reference for both: taking the lock twice lets a
    // concurrent writer evict the entry in between, and `unwrap()` on the
    // second lookup would panic and take down the request handler.
    let credit_info = {
        let mut entry = state
            .credits
            .entry(user.id.clone())
            .or_insert_with(|| CreditInfo {
                user_id: user.id.clone(),
                used: 0,
                limit,
                plan: plan.clone(),
                period_start,
                period_end,
            });
        entry.limit = limit;
        entry.plan = plan.clone();
        entry.clone()
    };

    HttpResponse::Ok().json(CreditsResponse {
        credits: credit_info,
    })
}

/// Increment credit usage for a user. Returns false if limit exceeded.
#[allow(dead_code)]
fn increment_credits(state: &AppState, user_id: &str, plan: &str) -> bool {
    let limit = plan_credit_limit(plan);
    // Insert-and-read under one lock. Re-acquiring it to `unwrap()` a second
    // lookup would panic if a concurrent writer evicted the entry in between,
    // and the check-then-increment must be atomic or two requests racing at the
    // limit boundary can both be admitted.
    let mut entry = state.credits.entry(user_id.to_string()).or_insert_with(|| {
        let now = chrono::Utc::now();
        CreditInfo {
            user_id: user_id.to_string(),
            used: 0,
            limit,
            plan: plan.to_string(),
            period_start: now.to_rfc3339(),
            period_end: now.to_rfc3339(),
        }
    });

    if entry.used >= entry.limit && entry.limit != u32::MAX {
        return false;
    }
    entry.used += 1;
    true
}

// ─── Compliance Config ──────────────────────────────────────────────

/// Kept as an enum rather than a free string so the compiler — not a reader
/// of the JSON — enforces the fixed set. A typo or a new status value the
/// frontend doesn't recognise used to fall through to the UI's "upcoming"
/// fallback, mislabeling it as on its way rather than flagging it as
/// unrecognised; that can no longer happen once this only serialises one of
/// four known lowercase strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
enum ComplianceStatus {
    /// Legally in force today.
    Active,
    /// A change is agreed and dated but not yet legally in force.
    Upcoming,
    /// A change is proposed but not yet formally adopted.
    ///
    /// Not constructed by today's dataset — the one entry this used to cover
    /// (the Annex III deferral) was formally adopted in July 2026 and is now
    /// `Upcoming`. Kept in the enum because regulatory changes routinely
    /// pass through this state before adoption; removing it would just mean
    /// re-adding it the next time one does.
    #[allow(dead_code)]
    Uncertain,
    /// A rule that used to apply and no longer does — kept rather than
    /// deleted so a reader who remembers it finds out it stopped, instead of
    /// finding nothing.
    Repealed,
}

#[derive(Debug, serde::Serialize)]
struct ComplianceRegulation {
    name: String,
    year: u32,
    limit: Option<f64>,
    unit: Option<String>,
    status: ComplianceStatus,
    description: String,
    region: String,
}

#[derive(Debug, serde::Serialize)]
struct ComplianceConfig {
    regulations: Vec<ComplianceRegulation>,
    thresholds: ComplianceThresholds,
    recommendations: Vec<String>,
    /// The date this dataset was last checked against primary sources, so the
    /// frontend can say how fresh it is rather than imply the regulatory
    /// picture is settled. It isn't: three of the six entries below changed
    /// materially within the twelve months before this date.
    verified_as_of: String,
}

#[derive(Debug, serde::Serialize)]
struct ComplianceThresholds {
    /// Cumulative *training* compute, in FLOPs, above which a general-purpose
    /// AI model is presumed to carry systemic risk under EU AI Act Article 51
    /// (Regulation (EU) 2024/1689). This is a training-time total, not a
    /// per-request figure — the Act defines no per-request compute limit.
    /// Verified via EU AI Act Article 55 guidance, August 2025.
    systemic_risk_training_flops: f64,
    /// Recommended point to double-check CSRD scope. CSRD does not itself set
    /// an emissions-volume trigger — scope is by company size (see the CSRD
    /// regulation entry) — so this is operational guidance, not a legal cite:
    /// a training run at this scale is large enough that a company already in
    /// CSRD's scope should have carbon reporting ready for it.
    carbon_report_tonnes: f64,
    /// Recommended training-cost point to trigger a budget review. Operational
    /// guidance, not a legal threshold.
    cost_review_usd: f64,
}

async fn compliance_config() -> impl Responder {
    HttpResponse::Ok().json(get_compliance_data())
}

// ─── API Key Management ─────────────────────────────────────────────

#[derive(Debug, serde::Deserialize)]
struct CreateApiKeyRequest {
    name: String,
    #[serde(default)]
    scopes: Vec<String>,
}

#[derive(Debug, serde::Serialize)]
struct CreateApiKeyResponse {
    api_key: ApiKeyInfo,
    /// The raw key is only shown once at creation
    key: String,
}

#[derive(Debug, serde::Serialize)]
struct ListApiKeysResponse {
    keys: Vec<ApiKeyInfo>,
}

async fn api_keys_create(
    req: HttpRequest,
    state: web::Data<AppState>,
    body: web::Json<CreateApiKeyRequest>,
) -> impl Responder {
    let user = match get_supabase_user(&req).await {
        Ok(u) => u,
        Err(resp) => return resp,
    };

    // Validate scopes
    let valid_scopes = [
        "analyze",
        "inference",
        "compare",
        "export",
        "projects",
        "agent",
        "all",
    ];
    let scopes: Vec<String> = body
        .scopes
        .iter()
        .filter(|s| valid_scopes.contains(&s.as_str()))
        .cloned()
        .collect();
    let scopes = if scopes.is_empty() {
        vec!["all".to_string()]
    } else {
        scopes
    };

    // Limit to 10 API keys per user
    let user_key_count = state
        .api_keys
        .iter()
        .filter(|e| e.value().user_id == user.id)
        .count();
    if user_key_count >= 10 {
        return HttpResponse::build(StatusCode::BAD_REQUEST).body("Maximum 10 API keys per user");
    }

    let raw_key = generate_api_key();
    let now = chrono::Utc::now().to_rfc3339();

    let api_key_info = ApiKeyInfo {
        key: raw_key.clone(),
        user_id: user.id.clone(),
        name: body.name.clone(),
        created_at: now,
        last_used_at: None,
        active: true,
        scopes: scopes.clone(),
    };

    state.api_keys.insert(raw_key.clone(), api_key_info.clone());

    HttpResponse::Ok().json(CreateApiKeyResponse {
        api_key: api_key_info,
        key: raw_key,
    })
}

async fn api_keys_list(req: HttpRequest, state: web::Data<AppState>) -> impl Responder {
    let user = match get_supabase_user(&req).await {
        Ok(u) => u,
        Err(resp) => return resp,
    };

    let keys: Vec<ApiKeyInfo> = state
        .api_keys
        .iter()
        .filter(|e| e.value().user_id == user.id)
        .map(|e| e.value().clone())
        .collect();

    HttpResponse::Ok().json(ListApiKeysResponse { keys })
}

async fn api_keys_revoke(
    req: HttpRequest,
    state: web::Data<AppState>,
    path: web::Path<String>,
) -> impl Responder {
    let user = match get_supabase_user(&req).await {
        Ok(u) => u,
        Err(resp) => return resp,
    };

    let key_id = path.into_inner();

    // Find the key by matching user_id (key_id could be the key itself or a short identifier)
    let mut found = false;
    for mut entry in state.api_keys.iter_mut() {
        if entry.value().user_id == user.id
            && (entry.key() == &key_id || entry.value().key == key_id)
        {
            entry.value_mut().active = false;
            found = true;
            break;
        }
    }

    if found {
        HttpResponse::Ok().json(serde_json::json!({"revoked": true}))
    } else {
        HttpResponse::NotFound().body("API key not found")
    }
}

async fn api_keys_delete(
    req: HttpRequest,
    state: web::Data<AppState>,
    path: web::Path<String>,
) -> impl Responder {
    let user = match get_supabase_user(&req).await {
        Ok(u) => u,
        Err(resp) => return resp,
    };

    let key_id = path.into_inner();

    // Find and remove the key
    let mut found_key: Option<String> = None;
    for entry in state.api_keys.iter() {
        if entry.value().user_id == user.id
            && (entry.key() == &key_id || entry.value().key == key_id)
        {
            found_key = Some(entry.key().clone());
            break;
        }
    }

    match found_key {
        Some(k) => {
            state.api_keys.remove(&k);
            HttpResponse::Ok().json(serde_json::json!({"deleted": true}))
        }
        None => HttpResponse::NotFound().body("API key not found"),
    }
}

// ─── Agent Control Endpoints ─────────────────────────────────────────
// These endpoints accept API key auth and provide programmatic access
// for the agent system to control the entire frontend.

/// POST /agent/analyze — Run analysis and return full report (blocking)
async fn agent_analyze(
    req: HttpRequest,
    state: web::Data<AppState>,
    body: web::Json<AnalyzeRequest>,
) -> impl Responder {
    // Auth: API key or JWT
    let user_id = match auth_any(&req, &state).await {
        Ok(id) => id,
        Err(resp) => return resp,
    };
    match check_api_key_scope(&req, &state, "analyze") {
        Ok(_) => {}
        Err(resp) => return resp,
    };

    let input = match serde_json::to_string(&body.topology) {
        Ok(v) => v,
        Err(e) => return HttpResponse::build(StatusCode::BAD_REQUEST).body(e.to_string()),
    };

    let config = match neurax_parser::parse_model_config(&input) {
        Ok(c) => c,
        Err(e) => return HttpResponse::build(StatusCode::BAD_REQUEST).body(e.to_string()),
    };

    let analysis_result = match web::block(move || neurax_core::run_analysis(config)).await {
        Ok(Ok(result)) => result,
        Ok(Err(e)) => {
            tracing::error!("[AGENT_ANALYZE] Analysis failed: {}", e);
            return HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR).body(e.to_string());
        }
        Err(_) => {
            return HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR)
                .body("Analysis task failed")
        }
    };

    // Cache the result for the agent to read back
    let report_json = analysis_result.to_json().unwrap_or_default();
    state.user_analyses.insert(
        user_id,
        serde_json::from_str(&report_json).unwrap_or(serde_json::Value::Null),
    );

    HttpResponse::Ok().body(report_json)
}

/// POST /agent/inference — Run inference simulation and return full report
async fn agent_inference(
    req: HttpRequest,
    state: web::Data<AppState>,
    body: web::Json<serde_json::Value>,
) -> impl Responder {
    let user_id = match auth_any(&req, &state).await {
        Ok(id) => id,
        Err(resp) => return resp,
    };
    match check_api_key_scope(&req, &state, "inference") {
        Ok(_) => {}
        Err(resp) => return resp,
    };

    let topology = match body.get("topology") {
        Some(t) => t.clone(),
        None => return HttpResponse::build(StatusCode::BAD_REQUEST).body("Missing topology"),
    };

    let input = match serde_json::to_string(&topology) {
        Ok(v) => v,
        Err(e) => return HttpResponse::build(StatusCode::BAD_REQUEST).body(e.to_string()),
    };

    let config = match neurax_parser::parse_model_config(&input) {
        Ok(c) => c,
        Err(e) => return HttpResponse::build(StatusCode::BAD_REQUEST).body(e.to_string()),
    };

    // Extract inference params from request or use defaults
    let params: neurax_ir::inference::InferenceParams = body
        .get("params")
        .and_then(|p| serde_json::from_value(p.clone()).ok())
        .unwrap_or_default();
    let inference_report = neurax_ir::inference::InferencePass::run(&params);

    // Also run analysis to get the full report
    let analysis_result = match web::block(move || neurax_core::run_analysis(config)).await {
        Ok(Ok(result)) => result,
        Ok(Err(e)) => {
            tracing::error!("[AGENT_INFERENCE] Analysis failed: {}", e);
            return HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR).body(e.to_string());
        }
        Err(_) => {
            return HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR)
                .body("Analysis task failed")
        }
    };

    let report_json = analysis_result.to_json().unwrap_or_default();
    state.user_inferences.insert(
        user_id,
        serde_json::from_str(&report_json).unwrap_or(serde_json::Value::Null),
    );

    HttpResponse::Ok().json(serde_json::json!({
        "report": serde_json::from_str::<serde_json::Value>(&report_json).unwrap_or(serde_json::Value::Null),
        "inference": inference_report,
    }))
}

/// GET /agent/audit — Audit a model: run analysis + inference + compliance check
async fn agent_audit(
    req: HttpRequest,
    state: web::Data<AppState>,
    body: web::Json<serde_json::Value>,
) -> impl Responder {
    let user_id = match auth_any(&req, &state).await {
        Ok(id) => id,
        Err(resp) => return resp,
    };
    match check_api_key_scope(&req, &state, "agent") {
        Ok(_) => {}
        Err(resp) => return resp,
    };

    let topology = match body.get("topology") {
        Some(t) => t.clone(),
        None => return HttpResponse::build(StatusCode::BAD_REQUEST).body("Missing topology"),
    };

    let input = match serde_json::to_string(&topology) {
        Ok(v) => v,
        Err(e) => return HttpResponse::build(StatusCode::BAD_REQUEST).body(e.to_string()),
    };

    let config = match neurax_parser::parse_model_config(&input) {
        Ok(c) => c,
        Err(e) => return HttpResponse::build(StatusCode::BAD_REQUEST).body(e.to_string()),
    };

    // Run analysis
    let analysis_result = match web::block(move || neurax_core::run_analysis(config)).await {
        Ok(Ok(result)) => result,
        Ok(Err(e)) => {
            tracing::error!("[AGENT_AUDIT] Analysis failed: {}", e);
            return HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR).body(e.to_string());
        }
        Err(_) => {
            return HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR)
                .body("Analysis task failed")
        }
    };

    // Run inference
    let params = neurax_ir::inference::InferenceParams::default();
    let inference_report = neurax_ir::inference::InferencePass::run(&params);

    // Get compliance config
    let compliance = get_compliance_data();

    // Serialize report for JSON extraction
    let report_json_str = analysis_result
        .to_json()
        .unwrap_or_else(|_| "{}".to_string());
    let report_val: serde_json::Value =
        serde_json::from_str(&report_json_str).unwrap_or(serde_json::Value::Null);

    // Cache results
    state
        .user_analyses
        .insert(user_id.clone(), report_val.clone());
    state.user_inferences.insert(
        user_id,
        serde_json::to_value(&inference_report).unwrap_or(serde_json::Value::Null),
    );

    // Build audit summary
    let mut audit_issues: Vec<serde_json::Value> = vec![];
    let mut audit_score: f64 = 100.0;

    // Check diagnostics from report
    if let Some(diagnostics) = report_val.get("diagnostics").cloned() {
        if let Some(diags) = diagnostics.as_array() {
            for d in diags {
                let severity = d.get("severity").and_then(|s| s.as_str()).unwrap_or("info");
                let msg = d.get("message").and_then(|m| m.as_str()).unwrap_or("");
                let code = d.get("code").and_then(|c| c.as_str()).unwrap_or("");
                audit_issues.push(serde_json::json!({
                    "category": "diagnostic",
                    "severity": severity,
                    "code": code,
                    "message": msg,
                }));
                if severity == "error" {
                    audit_score -= 10.0;
                } else if severity == "warning" {
                    audit_score -= 3.0;
                }
            }
        }
    }

    // Check compliance thresholds
    let total_params = report_val
        .get("architecture")
        .and_then(|a| a.get("total_parameters"))
        .and_then(|p| p.as_f64())
        .unwrap_or(0.0);
    let total_flops = report_val
        .get("compute")
        .and_then(|c| c.get("total_flops_forward"))
        .and_then(|f| f.as_f64())
        .unwrap_or(0.0);

    // A per-forward-pass GFLOPs figure used to be checked against a
    // "high_risk_gflops" threshold framed as an EU AI Act limit. Neither half
    // of that comparison was real: the Act sets no per-request compute limit
    // at all — its one quantitative threshold is 10²⁵ FLOPs of *cumulative
    // training* compute (see `ComplianceThresholds::systemic_risk_training_flops`)
    // — and `total_flops` here is `total_flops_forward`, a single pass, which
    // could never approach that figure regardless. Comparing the two would
    // silently always pass rather than checking anything real, so the check
    // is removed rather than wired to a threshold it cannot meaningfully be
    // compared against. A genuine version of this check needs the model's
    // total training-run compute, which the compiler does not currently
    // surface to this endpoint.

    // Check inference stability
    let stability_score = inference_report.stability_index.score;
    if stability_score < 0.5 {
        audit_issues.push(serde_json::json!({
            "category": "inference",
            "severity": "warning",
            "code": "LOW_STABILITY",
            "message": format!("Inference stability index is {:.1}/100 — model may produce inconsistent outputs", stability_score * 100.0),
        }));
        audit_score -= 10.0;
    }

    audit_score = audit_score.max(0.0);

    HttpResponse::Ok().json(serde_json::json!({
        "audit_score": audit_score,
        "audit_grade": match audit_score {
            s if s >= 90.0 => "A",
            s if s >= 75.0 => "B",
            s if s >= 60.0 => "C",
            s if s >= 40.0 => "D",
            _ => "F",
        },
        "issues": audit_issues,
        "report": report_val,
        "inference": inference_report,
        "compliance": compliance,
        "total_parameters": total_params,
        "total_flops_forward": total_flops,
    }))
}

/// POST /agent/carbon — Get carbon/cost analysis for a model
async fn agent_carbon(
    req: HttpRequest,
    state: web::Data<AppState>,
    body: web::Json<serde_json::Value>,
) -> impl Responder {
    let _user_id = match auth_any(&req, &state).await {
        Ok(id) => id,
        Err(resp) => return resp,
    };
    match check_api_key_scope(&req, &state, "agent") {
        Ok(_) => {}
        Err(resp) => return resp,
    };

    let topology = match body.get("topology") {
        Some(t) => t.clone(),
        None => return HttpResponse::build(StatusCode::BAD_REQUEST).body("Missing topology"),
    };

    let input = match serde_json::to_string(&topology) {
        Ok(v) => v,
        Err(e) => return HttpResponse::build(StatusCode::BAD_REQUEST).body(e.to_string()),
    };

    let config = match neurax_parser::parse_model_config(&input) {
        Ok(c) => c,
        Err(e) => return HttpResponse::build(StatusCode::BAD_REQUEST).body(e.to_string()),
    };

    let analysis_result = match web::block(move || neurax_core::run_analysis(config)).await {
        Ok(Ok(result)) => result,
        Ok(Err(e)) => {
            tracing::error!("[AGENT_CARBON] Analysis failed: {}", e);
            return HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR).body(e.to_string());
        }
        Err(_) => {
            return HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR)
                .body("Analysis task failed")
        }
    };

    // Extract carbon/cost metrics from report
    let report_json_str = analysis_result
        .to_json()
        .unwrap_or_else(|_| "{}".to_string());
    let report_val: serde_json::Value =
        serde_json::from_str(&report_json_str).unwrap_or(serde_json::Value::Null);
    let cost = report_val
        .get("cost")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let training_hours = cost
        .get("training_hours")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let training_cost_usd = cost
        .get("training_cost_usd")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let co2_tonnes = cost
        .get("co2_tonnes")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let energy_kwh = cost
        .get("energy_kwh")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    let compliance = get_compliance_data();

    HttpResponse::Ok().json(serde_json::json!({
        "carbon": {
            "co2_tonnes": co2_tonnes,
            "energy_kwh": energy_kwh,
            "training_hours": training_hours,
            "training_cost_usd": training_cost_usd,
        },
        "compliance": {
            "exceeds_carbon_threshold": co2_tonnes > compliance.thresholds.carbon_report_tonnes,
            "carbon_report_tonnes": compliance.thresholds.carbon_report_tonnes,
            "exceeds_cost_threshold": training_cost_usd > compliance.thresholds.cost_review_usd,
            "cost_review_usd": compliance.thresholds.cost_review_usd,
        },
        "recommendations": vec![
            if co2_tonnes > compliance.thresholds.carbon_report_tonnes {
                format!("⚠️ CO₂ emissions ({:.2}t) exceed CSRD reporting threshold ({:.1}t)", co2_tonnes, compliance.thresholds.carbon_report_tonnes)
            } else {
                format!("✅ CO₂ emissions ({:.2}t) below CSRD threshold ({:.1}t)", co2_tonnes, compliance.thresholds.carbon_report_tonnes)
            },
            if training_cost_usd > compliance.thresholds.cost_review_usd {
                format!("⚠️ Training cost (${:.0}) exceeds review threshold (${:.0})", training_cost_usd, compliance.thresholds.cost_review_usd)
            } else {
                format!("✅ Training cost (${:.0}) within budget (${:.0})", training_cost_usd, compliance.thresholds.cost_review_usd)
            },
        ],
        "optimization_tips": vec![
            "Consider FP8 or INT8 quantization to reduce inference cost by 2-4x".to_string(),
            "Use gradient checkpointing to reduce peak memory by 30-60%".to_string(),
            "Consider tensor parallelism for models > 13B parameters".to_string(),
        ],
    }))
}

/// GET /agent/compliance — Get compliance configuration
async fn agent_compliance(req: HttpRequest, state: web::Data<AppState>) -> impl Responder {
    let _user_id = match auth_any(&req, &state).await {
        Ok(id) => id,
        Err(resp) => return resp,
    };
    match check_api_key_scope(&req, &state, "agent") {
        Ok(_) => {}
        Err(resp) => return resp,
    };

    let compliance = get_compliance_data();
    HttpResponse::Ok().json(compliance)
}

/// GET /agent/results — Get cached analysis results for the authenticated user
async fn agent_results(req: HttpRequest, state: web::Data<AppState>) -> impl Responder {
    let user_id = match auth_any(&req, &state).await {
        Ok(id) => id,
        Err(resp) => return resp,
    };
    match check_api_key_scope(&req, &state, "analyze") {
        Ok(_) => {}
        Err(resp) => return resp,
    };

    let analysis = state
        .user_analyses
        .get(&user_id)
        .map(|e| e.value().clone())
        .unwrap_or(serde_json::Value::Null);
    let inference = state
        .user_inferences
        .get(&user_id)
        .map(|e| e.value().clone())
        .unwrap_or(serde_json::Value::Null);

    HttpResponse::Ok().json(serde_json::json!({
        "analysis": analysis,
        "inference": inference,
    }))
}

/// GET /agent/projects — List user's projects (for agent to load saved models)
async fn agent_projects(req: HttpRequest, state: web::Data<AppState>) -> impl Responder {
    let user_id = match auth_any(&req, &state).await {
        Ok(id) => id,
        Err(resp) => return resp,
    };
    match check_api_key_scope(&req, &state, "projects") {
        Ok(_) => {}
        Err(resp) => return resp,
    };

    let projects: Vec<Project> = state
        .projects
        .iter()
        .filter(|e| e.value().user_id == user_id)
        .map(|e| e.value().clone())
        .collect();

    HttpResponse::Ok().json(serde_json::json!({
        "projects": projects,
    }))
}

// ─── Agent memory (neurax-agent's step-by-step loop) ───────────────────
//
// Root-level, not `/agent/*` — see `agent_memory`'s own module doc for why
// (no service-level credential exists for `neurax-agent` to present a
// scope-gated route with). Scoped by `project_id` alone, not `user_id` —
// same doc, same reason: no real per-user identity reaches this service
// from `neurax-agent` today.

#[derive(Debug, Deserialize)]
struct ProjectIdQuery {
    project_id: String,
}

async fn memory_core_get(query: web::Query<ProjectIdQuery>) -> impl Responder {
    match agent_memory::get_core_preferences(&query.project_id).await {
        Ok(preferences) => HttpResponse::Ok().json(serde_json::json!({ "preferences": preferences })),
        Err(resp) => resp,
    }
}

#[derive(Debug, Deserialize)]
struct AddPreferenceRequest {
    project_id: String,
    preference: String,
}

async fn memory_core_add_preference(body: web::Json<AddPreferenceRequest>) -> impl Responder {
    match agent_memory::add_core_preference(&body.project_id, &body.preference).await {
        Ok(()) => HttpResponse::Ok().json(serde_json::json!({ "ok": true })),
        Err(resp) => resp,
    }
}

#[derive(Debug, Deserialize)]
struct ArchivalSearchQuery {
    project_id: String,
    #[serde(default)]
    query: String,
    #[serde(default = "default_archival_limit")]
    limit: u32,
}

fn default_archival_limit() -> u32 {
    5
}

async fn memory_archival_search(query: web::Query<ArchivalSearchQuery>) -> impl Responder {
    match agent_memory::search_archival(&query.project_id, &query.query, query.limit).await {
        Ok(entries) => HttpResponse::Ok().json(serde_json::json!({ "entries": entries })),
        Err(resp) => resp,
    }
}

#[derive(Debug, Deserialize)]
struct AddArchivalRequest {
    project_id: String,
    content: String,
}

async fn memory_archival_add(body: web::Json<AddArchivalRequest>) -> impl Responder {
    match agent_memory::add_archival_entry(&body.project_id, &body.content).await {
        Ok(()) => HttpResponse::Ok().json(serde_json::json!({ "ok": true })),
        Err(resp) => resp,
    }
}

#[derive(Debug, Deserialize)]
struct ConversationQuery {
    project_id: String,
    #[serde(default = "default_conversation_limit")]
    limit: u32,
}

fn default_conversation_limit() -> u32 {
    8
}

#[derive(Debug, serde::Serialize)]
struct ConversationTurnOut {
    role: String,
    content: String,
}

async fn memory_conversation_get(query: web::Query<ConversationQuery>) -> impl Responder {
    match agent_memory::get_recent_conversation(&query.project_id, query.limit).await {
        Ok(turns) => HttpResponse::Ok().json(serde_json::json!({
            "turns": turns
                .into_iter()
                .map(|(role, content)| ConversationTurnOut { role, content })
                .collect::<Vec<_>>(),
        })),
        Err(resp) => resp,
    }
}

#[derive(Debug, Deserialize)]
struct AppendConversationRequest {
    project_id: String,
    turns: Vec<ConversationTurnIn>,
}

#[derive(Debug, Deserialize)]
struct ConversationTurnIn {
    role: String,
    content: String,
}

async fn memory_conversation_append(body: web::Json<AppendConversationRequest>) -> impl Responder {
    let turns: Vec<(String, String)> = body
        .turns
        .iter()
        .map(|t| (t.role.clone(), t.content.clone()))
        .collect();
    match agent_memory::append_conversation_turns(&body.project_id, &turns).await {
        Ok(()) => HttpResponse::Ok().json(serde_json::json!({ "ok": true })),
        Err(resp) => resp,
    }
}

/// Helper to get compliance data
/// Checked against primary and legal-tracker sources on this date. Regulatory
/// text moves: three of these six entries changed materially in the twelve
/// months before this check (an enforcement date arriving, an executive order
/// being revoked, a bill dying in a prorogued parliament) — this is not a
/// one-time fact-check, it needs redoing periodically.
const COMPLIANCE_VERIFIED_AS_OF: &str = "2026-08-14";

fn get_compliance_data() -> ComplianceConfig {
    let regulations = vec![
        ComplianceRegulation {
            name: "EU AI Act — GPAI Systemic Risk (Art. 51–55)".to_string(),
            year: 2025,
            limit: Some(1e25),
            unit: Some("cumulative training FLOPs".to_string()),
            status: ComplianceStatus::Active,
            description: "General-purpose AI models trained with cumulative compute above 10²⁵ FLOPs are presumed to carry systemic risk: technical documentation, adversarial testing, 72-hour incident reporting and energy-efficiency disclosure became legally binding 2 Aug 2025. The AI Office's enforcement powers activated 2 Aug 2026 — both dates have now passed.".to_string(),
            region: "EU".to_string(),
        },
        ComplianceRegulation {
            name: "EU AI Act — High-Risk Systems (Annex III)".to_string(),
            year: 2027,
            limit: None,
            unit: None,
            status: ComplianceStatus::Upcoming,
            description: "Obligations for high-risk AI systems (biometrics, critical infrastructure, employment, law enforcement) are use-case based, not compute-based — there is no FLOPs threshold. The original deadline was 2 Aug 2026. The Digital Omnibus on AI (Regulation (EU) 2026/1744), published in the Official Journal 24 Jul 2026 and in force since 27 Jul 2026, formally deferred stand-alone Annex III systems to 2 Dec 2027 — that date is now the binding one.".to_string(),
            region: "EU".to_string(),
        },
        ComplianceRegulation {
            name: "Carbon Reporting (CSRD, post-Omnibus I)".to_string(),
            year: 2026,
            limit: None,
            unit: None,
            status: ComplianceStatus::Active,
            description: "The Omnibus I directive (EU 2026/470), published in the Official Journal 26 Feb 2026 and in force since 18 Mar 2026, narrowed CSRD's scope to companies with over 1,000 employees and over €450M annual turnover — roughly 90% of the companies previously in scope are now excluded. Check scope under the new thresholds before assuming a disclosure obligation applies.".to_string(),
            region: "EU".to_string(),
        },
        ComplianceRegulation {
            name: "Digital Services Act".to_string(),
            year: 2024,
            limit: None,
            unit: None,
            status: ComplianceStatus::Active,
            description: "Transparency and algorithmic-accountability obligations for large online platforms, in force for very large platforms since Feb 2023 and broadly since 17 Feb 2024. Not an AI-training-compute disclosure regime — that obligation is the EU AI Act's, not the DSA's.".to_string(),
            region: "EU".to_string(),
        },
        ComplianceRegulation {
            name: "US AI Executive Order 14110".to_string(),
            year: 2025,
            limit: None,
            unit: None,
            status: ComplianceStatus::Repealed,
            description: "Biden's EO 14110 (Oct 2023), which required reporting AI models trained above ~10²⁶ FLOPs to the federal government, was revoked 20 Jan 2025 and replaced by EO 14179 (\"Removing Barriers to American Leadership in AI\"). The federal reporting requirement no longer applies; no directly comparable replacement has been issued.".to_string(),
            region: "US".to_string(),
        },
        ComplianceRegulation {
            name: "Canada AIDA (Bill C-27)".to_string(),
            year: 2025,
            limit: None,
            unit: None,
            status: ComplianceStatus::Repealed,
            description: "The Artificial Intelligence and Data Act, proposed as Part 3 of Bill C-27, died on the order paper when Parliament was prorogued in Jan 2025. Canada has no federal AI-specific legislation in force as of this check; a successor bill is expected but had not been introduced.".to_string(),
            region: "Canada".to_string(),
        },
    ];

    let thresholds = ComplianceThresholds {
        systemic_risk_training_flops: 1e25,
        carbon_report_tonnes: 50.0,
        cost_review_usd: 100_000.0,
    };

    let recommendations = vec![
        "Check cumulative training compute against the EU AI Act's 10²⁵ FLOPs systemic-risk threshold — it is a training-time total, not a per-request figure".to_string(),
        "The EU AI Act high-risk (Annex III) deadline was formally deferred to 2 Dec 2027 by the Digital Omnibus on AI (Regulation (EU) 2026/1744, in force since 27 Jul 2026) — the original 2 Aug 2026 date no longer applies".to_string(),
        "Confirm CSRD scope under the post-Omnibus thresholds (>1,000 employees and >€450M turnover) before preparing carbon disclosure".to_string(),
        "The US EO 14110 training-compute reporting requirement was repealed Jan 2025 — do not plan around it".to_string(),
        "FP8/INT8 quantization lowers serving compute and cost regardless of which compliance regime applies".to_string(),
    ];

    ComplianceConfig {
        regulations,
        thresholds,
        recommendations,
        verified_as_of: COMPLIANCE_VERIFIED_AS_OF.to_string(),
    }
}

#[cfg(test)]
mod compliance_tests {
    use super::*;

    /// This data was previously years out of date in a way that would
    /// actively mislead a compliance decision: the EU AI Act's real 10²⁵-FLOP
    /// systemic-risk threshold measures cumulative *training* compute, and no
    /// version of the Act sets a per-request GFLOPs limit at all — verified
    /// against EU AI Act Article 55 guidance, August 2025. Nothing here should
    /// mention a per-request figure again.
    #[test]
    fn no_regulation_cites_a_per_request_gflops_limit() {
        let config = get_compliance_data();
        for reg in &config.regulations {
            let unit = reg.unit.as_deref().unwrap_or("");
            assert!(
                !unit.to_lowercase().contains("request"),
                "{} cites a per-request unit ({unit:?}) — the Act has no such limit",
                reg.name,
            );
        }
    }

    /// A rule that stopped applying must say so, not sit alongside active
    /// ones with no visible difference. EO 14110 was revoked 20 Jan 2025;
    /// Bill C-27 / AIDA died at prorogation the same month — verified against
    /// Federal Register and LEGISinfo records.
    #[test]
    fn repealed_rules_are_marked_repealed_not_active() {
        let config = get_compliance_data();
        for name in ["US AI Executive Order 14110", "Canada AIDA (Bill C-27)"] {
            let reg = config
                .regulations
                .iter()
                .find(|r| r.name == name)
                .unwrap_or_else(|| panic!("expected a regulation named {name}"));
            assert_eq!(
                reg.status,
                ComplianceStatus::Repealed,
                "{name} should be marked repealed"
            );
        }
    }

    /// The one number in this dataset with real legal weight: EU AI Act
    /// Article 51's systemic-risk threshold is 10²⁵ FLOPs of training
    /// compute, not the invented "300 GFLOPs" this used to carry.
    #[test]
    fn systemic_risk_threshold_is_the_real_ten_to_the_25() {
        let config = get_compliance_data();
        assert_eq!(config.thresholds.systemic_risk_training_flops, 1e25);
    }

    /// Every entry names its jurisdiction — a compliance timeline that
    /// doesn't say which country a rule is from is not actionable.
    #[test]
    fn every_regulation_names_a_region() {
        let config = get_compliance_data();
        for reg in &config.regulations {
            assert!(!reg.region.is_empty(), "{} has no region", reg.name);
        }
    }

    /// A freshness date must travel with the data — regulatory text moves,
    /// and a reader needs to know how old this check is, not just trust it.
    #[test]
    fn carries_a_verified_as_of_date() {
        let config = get_compliance_data();
        let verified = chrono::NaiveDate::parse_from_str(&config.verified_as_of, "%Y-%m-%d")
            .unwrap_or_else(|e| {
                panic!(
                    "verified_as_of {:?} is not a real date: {e}",
                    config.verified_as_of
                )
            });
        // Regulatory text moves; a dataset that hasn't been re-checked in over
        // a year is stale enough that this should fail loudly rather than
        // silently keep shipping last year's compliance picture.
        let today = chrono::Utc::now().date_naive();
        let age_days = (today - verified).num_days();
        assert!(
            (0..=365).contains(&age_days),
            "compliance data was verified {age_days} days ago (on {verified}) — re-verify against \
             primary sources and bump COMPLIANCE_VERIFIED_AS_OF",
        );
    }
}

// ─── Library surface ────────────────────────────────────────────────
//
// The routing table below is the single definition of NEURAX's HTTP API. It is
// mounted two ways:
//
//   * `neurax-service` (the binary in `src/main.rs`) binds it to a public
//     address — this is the deployment the web app talks to.
//   * `neurax-desktop` mounts the very same table in-process on a loopback
//     port, so the desktop build runs the production API rather than a
//     reimplementation of it that could drift.
//
// Anything that must differ between the two — the bind address, the CORS
// origins — is a parameter of `ServerConfig`, never a fork of the routes.

/// Where and how to expose the API.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Address to bind, e.g. `0.0.0.0:9098` or `127.0.0.1:0` for an
    /// OS-assigned port.
    pub bind_addr: String,
    /// Browser origins allowed to call the API.
    pub allowed_origins: Vec<String>,
}

/// The origins the hosted web app is served from during development.
pub fn default_web_origins() -> Vec<String> {
    [
        "http://localhost:8080",
        "http://localhost:8081",
        "http://localhost:8082",
        "https://localhost:8080",
        "https://localhost:8082",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8081",
        "http://127.0.0.1:8082",
        "https://127.0.0.1:8080",
        "https://127.0.0.1:8081",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

impl Default for ServerConfig {
    fn default() -> Self {
        let bind_addr = env::var("NEURAX_BIND").unwrap_or_else(|_| {
            env::var("PORT")
                .map(|p| format!("0.0.0.0:{p}"))
                .unwrap_or_else(|_| "0.0.0.0:9098".to_string())
        });
        Self {
            bind_addr,
            allowed_origins: default_web_origins(),
        }
    }
}

/// Load `.env` and install the tracing subscriber.
///
/// Safe to call once per process; a second call leaves the existing subscriber
/// in place rather than panicking, which matters for the desktop build where
/// the shell may already have initialised logging.
pub fn init_runtime() {
    if dotenvy::dotenv().is_err() {
        dotenvy::from_filename("neurax-service/.env").ok();
    }
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

/// Origins the desktop webview presents itself as.
///
/// Tauri serves the bundled UI over a custom scheme, which differs by platform:
/// `tauri://localhost` on Linux and macOS, `http://tauri.localhost` on Windows.
pub const DESKTOP_ORIGINS: [&str; 2] = ["tauri://localhost", "http://tauri.localhost"];

/// Build the CORS layer for the given origins.
fn build_cors(allowed_origins: &[String]) -> Cors {
    let mut cors = Cors::default()
        .allowed_methods(vec!["GET", "POST", "PUT", "DELETE", "OPTIONS"])
        .allowed_headers(vec![
            header::CONTENT_TYPE,
            header::AUTHORIZATION,
            HeaderName::from_static("x-api-key"),
        ])
        // A page in a browser cannot forge a `tauri://` origin — the browser
        // sets the Origin header itself — so honouring that scheme costs the
        // hosted deployment nothing and spares the desktop build a per-platform
        // origin list.
        .allowed_origin_fn(|origin, _req| origin.as_bytes().starts_with(b"tauri://"))
        .max_age(3600);
    for origin in allowed_origins {
        cors = cors.allowed_origin(origin);
    }
    cors
}

/// Build the `App` factory shared by every way of starting the service.
macro_rules! neurax_app {
    ($origins:expr, $state:expr) => {
        App::new()
            // Origin is logged because CORS failures are otherwise invisible
            // here: a rejected request still reaches a handler, and the only
            // trace is an error inside the browser. Knowing which origin
            // asked is the whole diagnosis.
            .wrap(middleware::Logger::new(
                "%a \"%r\" %s %b origin=%{Origin}i %T",
            ))
            .wrap(middleware::Compress::default())
            .wrap(build_cors(&$origins))
            .app_data(
                web::JsonConfig::default()
                    .limit(10 * 1024 * 1024) // 10 MB max payload
                    .error_handler(|err, _req| {
                        let msg = err.to_string();
                        actix_web::error::InternalError::from_response(
                            err,
                            HttpResponse::build(StatusCode::BAD_REQUEST).body(msg),
                        )
                        .into()
                    }),
            )
            .app_data(web::Data::new($state.clone()))
            .configure(configure_routes)
    };
}

/// Mount every NEURAX route onto an `App`.
///
/// Kept as a free function taking `ServiceConfig` so both entry points share
/// one routing table; adding an endpoint here reaches web and desktop at once.
pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.route("/me", web::get().to(me))
        .route("/billing/checkout", web::post().to(billing_checkout))
        .route("/billing/portal", web::post().to(billing_portal))
        .route("/stripe/webhook", web::post().to(stripe_webhook))
        .route("/health", web::get().to(health))
        .route("/hardware", web::get().to(hardware_list))
        .route("/hardware/detect", web::get().to(hardware_detect))
        .route("/hardware/measure", web::post().to(hardware_measure))
        .route("/model/verify", web::post().to(model_verify))
        .route("/training/runs", web::get().to(training_list))
        .route("/training/runs", web::post().to(training_start))
        .route("/training/runs/{id}", web::get().to(training_get))
        .route("/training/runs/{id}/control", web::post().to(training_control))
        .route("/plugin/validate", web::post().to(plugin_validate))
        .route("/presets", web::get().to(get_presets))
        .route("/presets/{id}", web::get().to(get_preset))
        .route("/analyze", web::post().to(analyze))
        .route("/sweep", web::post().to(sweep))
        .route("/analyze/compare", web::post().to(analyze_compare))
        .route("/analyze/stream", web::post().to(analyze_stream_start))
        .route(
            "/analyze/stream/{job_id}",
            web::get().to(analyze_stream_events),
        )
        .route("/analyze/result/{job_id}", web::get().to(analyze_result))
        .route("/analyze/status/{job_id}", web::get().to(analyze_status))
        .route("/timemachine", web::post().to(time_machine))
        .route("/inference/simulate", web::post().to(inference_simulate))
        .route("/export/onnx", web::post().to(export_onnx))
        .route("/export/github", web::post().to(export_github))
        .route("/projects", web::get().to(projects_list))
        .route("/projects", web::post().to(projects_create))
        .route("/projects/{id}", web::get().to(projects_get))
        .route("/projects/{id}", web::put().to(projects_update))
        .route("/projects/{id}", web::delete().to(projects_delete))
        // ─── Public Shares (no auth: anonymous by design) ────────
        .route("/shares", web::post().to(shares_create))
        .route("/shares/{id}", web::get().to(shares_get))
        .route("/shares/{id}", web::delete().to(shares_delete))
        .route("/shares/{id}/download", web::get().to(shares_download))
        .route("/credits", web::get().to(credits_get))
        .route("/compliance/config", web::get().to(compliance_config))
        // ─── API Key Management ─────────────────────────────────
        .route("/api-keys", web::post().to(api_keys_create))
        .route("/api-keys", web::get().to(api_keys_list))
        .route("/api-keys/{key_id}/revoke", web::post().to(api_keys_revoke))
        .route("/api-keys/{key_id}", web::delete().to(api_keys_delete))
        // ─── Agent Control Endpoints (API key auth) ─────────────
        .route("/agent/analyze", web::post().to(agent_analyze))
        .route("/agent/inference", web::post().to(agent_inference))
        .route("/agent/compare", web::post().to(analyze_compare))
        .route("/agent/audit", web::post().to(agent_audit))
        .route("/agent/carbon", web::post().to(agent_carbon))
        .route("/agent/compliance", web::get().to(agent_compliance))
        .route("/agent/results", web::get().to(agent_results))
        .route("/agent/projects", web::get().to(agent_projects))
        .route("/memory/core", web::get().to(memory_core_get))
        .route("/memory/core/preference", web::post().to(memory_core_add_preference))
        .route("/memory/archival", web::get().to(memory_archival_search))
        .route("/memory/archival", web::post().to(memory_archival_add))
        .route("/memory/conversation", web::get().to(memory_conversation_get))
        .route("/memory/conversation", web::post().to(memory_conversation_append));
}

/// Build the configured `HttpServer` and return it alongside the addresses it
/// actually bound.
///
/// The bound addresses are returned rather than assumed because the desktop
/// build asks for port 0 and needs to learn which port the OS handed out.
pub fn build_server(
    config: &ServerConfig,
    app_state: AppState,
) -> std::io::Result<(actix_web::dev::Server, Vec<std::net::SocketAddr>)> {
    spawn_job_retention_sweeper(&app_state);
    let origins = config.allowed_origins.clone();
    let server =
        HttpServer::new(move || neurax_app!(origins, app_state)).bind(&config.bind_addr)?;
    let addrs = server.addrs();
    Ok((server.run(), addrs))
}

/// Serve the API on a listener the caller has already bound.
///
/// The desktop build binds `127.0.0.1:0` on its main thread so it can read the
/// OS-assigned port before the window exists, then hands the listener here. It
/// must be called from inside an actix runtime.
pub fn serve_on_listener(
    listener: std::net::TcpListener,
    allowed_origins: Vec<String>,
    app_state: AppState,
) -> std::io::Result<actix_web::dev::Server> {
    spawn_job_retention_sweeper(&app_state);
    let server =
        HttpServer::new(move || neurax_app!(allowed_origins, app_state)).listen(listener)?;
    Ok(server.run())
}

/// Bind and serve until the process is stopped. Used by the standalone binary.
pub async fn run_server(config: ServerConfig) -> std::io::Result<()> {
    let state = AppState::new();

    // Same persistence the desktop application uses, so a self-hosted
    // deployment behaves identically. Opt-in, because several replicas sharing
    // one file would overwrite each other; that case wants a database.
    if let Some(path) = persistence::configured_path() {
        persistence::attach(&state, &path);
    }

    let (server, addrs) = build_server(&config, state)?;
    for addr in &addrs {
        tracing::info!("[STARTUP] Neurax service listening on {addr}");
    }
    server.await
}
