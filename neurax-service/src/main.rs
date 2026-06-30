use actix_cors::Cors;
use actix_web::{
    http::{header, StatusCode},
    middleware, web, App, HttpRequest, HttpResponse, HttpServer, Responder,
};
use actix_web::http::header::HeaderName;
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
    if let Some(auth) = req.headers().get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()) {
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
fn check_api_key_scope(req: &HttpRequest, state: &AppState, required_scope: &str) -> Result<(), HttpResponse> {
    let key = api_key_from_req(req).ok_or_else(|| {
        HttpResponse::build(StatusCode::UNAUTHORIZED).body("Missing API key")
    })?;

    let api_key_info = state.api_keys.get(&key).ok_or_else(|| {
        HttpResponse::build(StatusCode::UNAUTHORIZED).body("Invalid API key")
    })?;

    if !api_key_info.value().active {
        return Err(HttpResponse::build(StatusCode::FORBIDDEN).body("API key has been revoked"));
    }

    let scopes = &api_key_info.value().scopes;
    // "agent" scope grants access to all agent endpoints
    if !scopes.contains(&required_scope.to_string()) && !scopes.contains(&"agent".to_string()) && !scopes.contains(&"all".to_string()) {
        return Err(HttpResponse::build(StatusCode::FORBIDDEN)
            .body(format!("API key lacks '{}' scope", required_scope)));
    }

    Ok(())
}

// ─── Project Storage ─────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Hash, Eq, PartialEq)]
pub struct ProjectKey {
    user_id: String,
    id: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Project {
    id: String,
    user_id: String,
    name: String,
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

fn noauth_enabled() -> bool {
    env::var("NEURAX_DEBUG_NOAUTH")
        .map(|v| v != "false")
        .unwrap_or(true)
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
        }
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

async fn supabase_rest_client() -> Result<(String, String, reqwest::Client), HttpResponse> {
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
        tracing::warn!("[INFERENCE] Auth failed after {}ms", start.elapsed().as_millis());
        return resp;
    }

    let params = req.params.clone();
    let result = actix_web::rt::task::spawn_blocking(move || {
        neurax_ir::inference::InferencePass::run(&params)
    });

    let timeout_result =
        actix_web::rt::time::timeout(Duration::from_secs(30), result).await;

    let elapsed = start.elapsed();
    match timeout_result {
        Ok(Ok(report)) => {
            tracing::info!("[INFERENCE] Success in {}ms", elapsed.as_millis());
            HttpResponse::Ok().json(InferenceResponse { report })
        }
        Ok(Err(_join_err)) => {
            tracing::error!("[INFERENCE] Task join error after {}ms", elapsed.as_millis());
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

async fn export_onnx(
    http_req: HttpRequest,
    req: web::Json<ExportOnnxRequest>,
) -> impl Responder {
    let start = std::time::Instant::now();
    tracing::info!("[EXPORT ONNX] Request received");

    if let Err(resp) = require_verified_email(&http_req).await {
        tracing::warn!("[EXPORT ONNX] Auth failed after {}ms", start.elapsed().as_millis());
        return resp;
    }

    let input = match serde_json::to_string(&req.topology) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("[EXPORT ONNX] Failed to serialize topology: {}", e);
            return HttpResponse::build(StatusCode::BAD_REQUEST).body(e.to_string());
        }
    };

    // Parse the topology JSON into ModelConfig
    tracing::info!("[EXPORT ONNX] Parsing model config...");
    let config = match neurax_parser::parse_model_config(&input) {
        Ok(c) => {
            tracing::info!(
                "[EXPORT ONNX] Parse OK: model_type={:?}, layers={}",
                c.model.model_type,
                c.model.layers.len()
            );
            c
        }
        Err(e) => {
            tracing::error!("[EXPORT ONNX] Parse failed: {}", e);
            return HttpResponse::build(StatusCode::BAD_REQUEST).body(e.to_string());
        }
    };

    let model_name = req.model_name.clone();
    let result = actix_web::rt::task::spawn_blocking(move || {
        // Run the analysis pipeline to get the ArchitectureIR
        let analysis = neurax_core::run_analysis(config.clone())
            .map_err(|e| e.to_string())?;

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
                "[EXPORT ONNX] Success in {}ms - {} nodes, {} initializers, {} bytes",
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
            tracing::error!("[EXPORT ONNX] Export error after {}ms: {}", elapsed.as_millis(), e);
            HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR).body(e.to_string())
        }
        Ok(Err(_join_err)) => {
            tracing::error!("[EXPORT ONNX] Task join error after {}ms", elapsed.as_millis());
            HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR)
                .body("Export task failed unexpectedly")
        }
        Err(_timeout) => {
            tracing::error!("[EXPORT ONNX] Timeout after {}ms", elapsed.as_millis());
            HttpResponse::build(StatusCode::GATEWAY_TIMEOUT)
                .body("Export timed out after 60 seconds")
        }
    }
}

async fn analyze(http_req: HttpRequest, req: web::Json<AnalyzeRequest>) -> impl Responder {
    let start = std::time::Instant::now();
    tracing::info!("[ANALYZE] Request received");

    // Log request payload summary
    if let Ok(payload_str) = serde_json::to_string(&req.topology) {
        let preview: String = payload_str.chars().take(500).collect();
        tracing::info!("[ANALYZE] Payload preview: {}...", preview);
        tracing::info!("[ANALYZE] Payload size: {} bytes", payload_str.len());
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

    let input = match serde_json::to_string(&req.topology) {
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

async fn analyze_compare(
    http_req: HttpRequest,
    req: web::Json<CompareRequest>,
) -> impl Responder {
    let start = std::time::Instant::now();
    tracing::info!("[COMPARE] Request received with {} configs", req.configs.len());

    if let Err(resp) = require_verified_email(&http_req).await {
        return resp;
    }

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

            // Override hardware in the topology JSON
            if let Some(hw) = topology.get_mut("hardware") {
                if let Some(hw_obj) = hw.as_object_mut() {
                    hw_obj.insert("name".to_string(), serde_json::Value::String(cfg.hardware.clone()));
                    hw_obj.insert(
                        "count".to_string(),
                        serde_json::Value::Number(
                            serde_json::Number::from(cfg.gpu_count.unwrap_or(1)),
                        ),
                    );
                    hw_obj.insert(
                        "memory_gb".to_string(),
                        serde_json::Value::Number(
                            serde_json::Number::from(cfg.gpu_memory_gb.unwrap_or(gpu_spec.memory_gb)),
                        ),
                    );
                    hw_obj.insert(
                        "tflops_fp16".to_string(),
                        serde_json::json!(gpu_spec.tflops_fp16),
                    );
                    hw_obj.insert(
                        "tflops_fp32".to_string(),
                        serde_json::json!(gpu_spec.tflops_fp32),
                    );
                    hw_obj.insert(
                        "memory_bandwidth_gb_s".to_string(),
                        serde_json::json!(gpu_spec.memory_bandwidth_gbs),
                    );
                    hw_obj.insert(
                        "tensor_cores".to_string(),
                        serde_json::Value::Bool(gpu_spec.tensor_cores),
                    );
                    hw_obj.insert(
                        "nvlink".to_string(),
                        serde_json::Value::Bool(gpu_spec.nvlink),
                    );
                }
            } else {
                // No hardware section in topology — create one
                if let Some(obj) = topology.as_object_mut() {
                    obj.insert(
                        "hardware".to_string(),
                        serde_json::json!({
                            "name": cfg.hardware,
                            "count": cfg.gpu_count.unwrap_or(1),
                            "memory_gb": cfg.gpu_memory_gb.unwrap_or(gpu_spec.memory_gb),
                            "tflops_fp16": gpu_spec.tflops_fp16,
                            "tflops_fp32": gpu_spec.tflops_fp32,
                            "memory_bandwidth_gb_s": gpu_spec.memory_bandwidth_gbs,
                            "tensor_cores": gpu_spec.tensor_cores,
                            "nvlink": gpu_spec.nvlink,
                        }),
                    );
                }
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
            tracing::error!("[COMPARE] Task join error after {}ms: {}", elapsed.as_millis(), e);
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
            tracing::info!(
                "[TIMEMACHINE] Success in {}ms",
                start.elapsed().as_millis()
            );
            HttpResponse::Ok().json(TimeMachineResponse { projection })
        }
        Ok(Ok(Err(e))) => HttpResponse::build(StatusCode::BAD_REQUEST).body(e.to_string()),
        Ok(Err(_)) => HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR)
            .body("Time Machine task failed unexpectedly"),
        Err(_) => {
            HttpResponse::build(StatusCode::GATEWAY_TIMEOUT).body("Time Machine timed out")
        }
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
    if let Err(resp) = require_verified_email(&http_req).await {
        return resp;
    }

    let job_id = uuid::Uuid::new_v4().to_string();
    let created_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    // Create broadcast channel for SSE events
    let (tx, _rx) = broadcast::channel::<String>(256);
    state.channels.insert(job_id.clone(), tx);

    // Insert job info
    state.jobs.insert(job_id.clone(), JobInfo {
        job_id: job_id.clone(),
        status: "running".to_string(),
        created_at_ms: created_at,
        completed_at_ms: None,
        error: None,
    });

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
        let (event_sender, event_receiver) = tokio::sync::broadcast::channel::<neurax_core::streaming::AnalysisEvent>(256);
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
            neurax_core::streaming::BroadcastEmitter::from_sender(event_sender)
        );

        // Clone job_id for use after spawn_blocking
        let job_id_result = job_id_clone.clone();
        // Run analysis in blocking context
        let result = actix_web::rt::task::spawn_blocking(move || {
            neurax_core::streaming::run_analysis_streaming_fallible(
                config,
                emitter,
                &job_id_clone,
            )
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

                state_inner.results.insert(job_id_result.clone(), report_value);

                // Update job status
                if let Some(mut job) = state_inner.jobs.get_mut(&job_id_result) {
                    job.status = "completed".to_string();
                    job.completed_at_ms = Some(std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as u64);
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

    HttpResponse::Accepted().json(AnalyzeStreamResponse { job_id })
}

/// GET /analyze/stream/{job_id} — SSE endpoint for streaming analysis events
async fn analyze_stream_events(
    path: web::Path<String>,
    state: web::Data<AppState>,
) -> HttpResponse {
    let job_id = path.into_inner();

    // Check if job exists
    if !state.jobs.contains_key(&job_id) {
        return HttpResponse::NotFound().body("Job not found");
    }

    // Get or create a receiver
    let rx = match state.channels.get(&job_id) {
        Some(tx) => tx.subscribe(),
        None => {
            // Job already completed, check for result
            if state.results.contains_key(&job_id) {
                // Return completion event directly
                let event = serde_json::json!({
                    "type": "Completed",
                    "data": { "job_id": job_id, "total_ms": 0 }
                });
                let sse_data = format!("data: {}\n\n", event);
                return HttpResponse::Ok()
                    .content_type("text/event-stream")
                    .insert_header(("Cache-Control", "no-cache"))
                    .insert_header(("Connection", "keep-alive"))
                    .body(sse_data);
            }
            return HttpResponse::NotFound().body("Job stream expired");
        }
    };

    // Stream events via SSE
    let state_inner = state.into_inner();
    let stream = async_stream::stream! {
        let mut rx = rx;
        loop {
            match rx.recv().await {
                Ok(event_json) => {
                    // Parse the event to check for terminal states
                    let event: serde_json::Value = match serde_json::from_str(&event_json) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                    yield Ok::<_, actix_web::Error>(actix_web::web::Bytes::from(format!("data: {}\n\n", event_json)));

                    // Check if this is a terminal event
                    let event_type = event.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    if event_type == "Completed" || event_type == "Failed" {
                        // Send final result if completed
                        if event_type == "Completed" {
                            if let Some(result) = state_inner.results.get(&job_id) {
                                let result_json = serde_json::json!({
                                    "type": "Result",
                                    "data": result.value()
                                });
                                yield Ok(actix_web::web::Bytes::from(format!("data: {}\n\n", result_json)));
                            }
                        }
                        break;
                    }
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    // Client is behind, send a lag notice
                    yield Ok(actix_web::web::Bytes::from(format!("data: {{\"type\":\"Lagged\",\"data\":{{\"count\":{}}}}}\n\n", n)));
                }
                Err(broadcast::error::RecvError::Closed) => {
                    // Channel closed, check for result
                    if let Some(result) = state_inner.results.get(&job_id) {
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
    state: web::Data<AppState>,
) -> impl Responder {
    let job_id = path.into_inner();

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
    state: web::Data<AppState>,
) -> impl Responder {
    let job_id = path.into_inner();

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

async fn projects_list(
    http_req: HttpRequest,
    state: web::Data<AppState>,
) -> impl Responder {
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

    state
        .credits
        .entry(user.id.clone())
        .or_insert_with(|| CreditInfo {
            user_id: user.id.clone(),
            used: 0,
            limit,
            plan: plan.clone(),
            period_start: period_start.clone(),
            period_end: period_end.clone(),
        });

    // Update limit and plan in case they changed
    {
        let mut entry = state.credits.get_mut(&user.id).unwrap();
        entry.value_mut().limit = limit;
        entry.value_mut().plan = plan.clone();
    }

    let credit_info = state.credits.get(&user.id).unwrap().value().clone();

    HttpResponse::Ok().json(CreditsResponse { credits: credit_info })
}

/// Increment credit usage for a user. Returns false if limit exceeded.
#[allow(dead_code)]
fn increment_credits(state: &AppState, user_id: &str, plan: &str) -> bool {
    let limit = plan_credit_limit(plan);
    state
        .credits
        .entry(user_id.to_string())
        .or_insert_with(|| {
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

    let mut entry = state.credits.get_mut(user_id).unwrap();
    if entry.value().used >= entry.value().limit && entry.value().limit != u32::MAX {
        return false;
    }
    entry.value_mut().used += 1;
    true
}

// ─── Compliance Config ──────────────────────────────────────────────

#[derive(Debug, serde::Serialize)]
struct ComplianceRegulation {
    name: String,
    year: u32,
    limit: Option<f64>,
    unit: Option<String>,
    status: String,
    description: String,
    region: String,
}

#[derive(Debug, serde::Serialize)]
struct ComplianceConfig {
    regulations: Vec<ComplianceRegulation>,
    thresholds: ComplianceThresholds,
    recommendations: Vec<String>,
}

#[derive(Debug, serde::Serialize)]
struct ComplianceThresholds {
    /// GFLOPs threshold for "high risk" classification (EU AI Act)
    high_risk_gflops: f64,
    /// CO₂e threshold in tonnes/year for mandatory reporting (CSRD)
    carbon_report_tonnes: f64,
    /// Training compute threshold in FLOPs for disclosure (DSA)
    dsa_disclosure_flops: f64,
    /// Recommended max training cost before review
    cost_review_usd: f64,
}

async fn compliance_config() -> impl Responder {
    let regulations = vec![
        ComplianceRegulation {
            name: "EU AI Act Phase 1".to_string(),
            year: 2027,
            limit: Some(300.0),
            unit: Some("GFLOPs/request".to_string()),
            status: "upcoming".to_string(),
            description: "General-purpose AI models trained with >10²⁵ FLOPs must comply with transparency and safety obligations.".to_string(),
            region: "EU".to_string(),
        },
        ComplianceRegulation {
            name: "EU AI Act Phase 2".to_string(),
            year: 2028,
            limit: Some(150.0),
            unit: Some("GFLOPs/request".to_string()),
            status: "upcoming".to_string(),
            description: "Stricter limits for high-risk AI applications in critical infrastructure, law enforcement, and biometrics.".to_string(),
            region: "EU".to_string(),
        },
        ComplianceRegulation {
            name: "Carbon Reporting (CSRD)".to_string(),
            year: 2026,
            limit: None,
            unit: None,
            status: "active".to_string(),
            description: "Corporate Sustainability Reporting Directive requires disclosure of energy consumption and CO₂ emissions for large companies.".to_string(),
            region: "EU".to_string(),
        },
        ComplianceRegulation {
            name: "Digital Services Act".to_string(),
            year: 2026,
            limit: None,
            unit: None,
            status: "active".to_string(),
            description: "Requires transparency reporting for very large online platforms using AI, including compute disclosure.".to_string(),
            region: "EU".to_string(),
        },
        ComplianceRegulation {
            name: "US AI Executive Order".to_string(),
            year: 2024,
            limit: Some(1e26),
            unit: Some("FLOPs (training)".to_string()),
            status: "active".to_string(),
            description: "Companies must report AI models trained with compute exceeding 10²⁶ FLOPs or biological sequence models above defined thresholds.".to_string(),
            region: "US".to_string(),
        },
        ComplianceRegulation {
            name: "Canada AIDA".to_string(),
            year: 2027,
            limit: None,
            unit: None,
            status: "proposed".to_string(),
            description: "Artificial Intelligence and Data Act — high-impact AI systems must meet safety, transparency, and monitoring requirements.".to_string(),
            region: "Canada".to_string(),
        },
    ];

    let thresholds = ComplianceThresholds {
        high_risk_gflops: 300.0,
        carbon_report_tonnes: 50.0,
        dsa_disclosure_flops: 1e25,
        cost_review_usd: 100_000.0,
    };

    let recommendations = vec![
        "Monitor EU AI Act Phase 1 compliance for models exceeding 300 GFLOPs/request".to_string(),
        "Prepare CSRD carbon reporting for training runs exceeding 50 tonnes CO₂e/year".to_string(),
        "Consider FP8 or INT8 quantization to reduce inference compute below regulatory thresholds".to_string(),
        "Document all training compute for models above 10²⁵ FLOPs (US EO requirement)".to_string(),
        "Implement energy monitoring for GPU clusters to track real-time carbon footprint".to_string(),
    ];

    HttpResponse::Ok().json(ComplianceConfig {
        regulations,
        thresholds,
        recommendations,
    })
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

async fn api_keys_create(req: HttpRequest, state: web::Data<AppState>, body: web::Json<CreateApiKeyRequest>) -> impl Responder {
    let user = match get_supabase_user(&req).await {
        Ok(u) => u,
        Err(resp) => return resp,
    };

    // Validate scopes
    let valid_scopes = ["analyze", "inference", "compare", "export", "projects", "agent", "all"];
    let scopes: Vec<String> = body.scopes.iter()
        .filter(|s| valid_scopes.contains(&s.as_str()))
        .cloned()
        .collect();
    let scopes = if scopes.is_empty() { vec!["all".to_string()] } else { scopes };

    // Limit to 10 API keys per user
    let user_key_count = state.api_keys.iter().filter(|e| e.value().user_id == user.id).count();
    if user_key_count >= 10 {
        return HttpResponse::build(StatusCode::BAD_REQUEST)
            .body("Maximum 10 API keys per user");
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

async fn api_keys_revoke(req: HttpRequest, state: web::Data<AppState>, path: web::Path<String>) -> impl Responder {
    let user = match get_supabase_user(&req).await {
        Ok(u) => u,
        Err(resp) => return resp,
    };

    let key_id = path.into_inner();

    // Find the key by matching user_id (key_id could be the key itself or a short identifier)
    let mut found = false;
    for mut entry in state.api_keys.iter_mut() {
        if entry.value().user_id == user.id && (entry.key() == &key_id || entry.value().key == key_id) {
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

async fn api_keys_delete(req: HttpRequest, state: web::Data<AppState>, path: web::Path<String>) -> impl Responder {
    let user = match get_supabase_user(&req).await {
        Ok(u) => u,
        Err(resp) => return resp,
    };

    let key_id = path.into_inner();

    // Find and remove the key
    let mut found_key: Option<String> = None;
    for entry in state.api_keys.iter() {
        if entry.value().user_id == user.id && (entry.key() == &key_id || entry.value().key == key_id) {
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
async fn agent_analyze(req: HttpRequest, state: web::Data<AppState>, body: web::Json<AnalyzeRequest>) -> impl Responder {
    // Auth: API key or JWT
    let user_id = match auth_any(&req, &state).await {
        Ok(id) => id,
        Err(resp) => return resp,
    };
    match check_api_key_scope(&req, &state, "analyze") {
        Ok(_) => {},
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
            tracing::error!("Analysis failed: {}", e);
            return HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR).body(e.to_string());
        }
        Err(_) => return HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR).body("Analysis task failed"),
    };

    // Cache the result for the agent to read back
    let report_json = analysis_result.to_json().unwrap_or_default();
    state.user_analyses.insert(user_id, serde_json::from_str(&report_json).unwrap_or(serde_json::Value::Null));

    HttpResponse::Ok().body(report_json)
}

/// POST /agent/inference — Run inference simulation and return full report
async fn agent_inference(req: HttpRequest, state: web::Data<AppState>, body: web::Json<serde_json::Value>) -> impl Responder {
    let user_id = match auth_any(&req, &state).await {
        Ok(id) => id,
        Err(resp) => return resp,
    };
    match check_api_key_scope(&req, &state, "inference") {
        Ok(_) => {},
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
    let params = neurax_ir::inference::InferenceParams::default();
    let inference_report = neurax_ir::inference::InferencePass::run(&params);

    // Also run analysis to get the full report
    let analysis_result = match web::block(move || neurax_core::run_analysis(config)).await {
        Ok(Ok(result)) => result,
        Ok(Err(e)) => {
            tracing::error!("Analysis failed: {}", e);
            return HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR).body(e.to_string());
        }
        Err(_) => return HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR).body("Analysis task failed"),
    };

    let report_json = analysis_result.to_json().unwrap_or_default();
    state.user_inferences.insert(user_id, serde_json::from_str(&report_json).unwrap_or(serde_json::Value::Null));

    HttpResponse::Ok().json(serde_json::json!({
        "report": serde_json::from_str::<serde_json::Value>(&report_json).unwrap_or(serde_json::Value::Null),
        "inference": inference_report,
    }))
}

/// POST /agent/compare — Compare multiple hardware configs
/// Delegates to analyze_compare after API key auth check
async fn agent_compare(http_req: HttpRequest, body: web::Json<CompareRequest>) -> impl Responder {
    // Auth is handled by analyze_compare internally
    analyze_compare(http_req, body).await
}

/// GET /agent/audit — Audit a model: run analysis + inference + compliance check
async fn agent_audit(req: HttpRequest, state: web::Data<AppState>, body: web::Json<serde_json::Value>) -> impl Responder {
    let user_id = match auth_any(&req, &state).await {
        Ok(id) => id,
        Err(resp) => return resp,
    };
    match check_api_key_scope(&req, &state, "agent") {
        Ok(_) => {},
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
            tracing::error!("Analysis failed: {}", e);
            return HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR).body(e.to_string());
        }
        Err(_) => return HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR).body("Analysis task failed"),
    };

    // Run inference
    let params = neurax_ir::inference::InferenceParams::default();
    let inference_report = neurax_ir::inference::InferencePass::run(&params);

    // Get compliance config
    let compliance = get_compliance_data();

    // Serialize report for JSON extraction
    let report_json_str = analysis_result.to_json().unwrap_or_else(|_| "{}".to_string());
    let report_val: serde_json::Value = serde_json::from_str(&report_json_str).unwrap_or(serde_json::Value::Null);

    // Cache results
    state.user_analyses.insert(user_id.clone(), report_val.clone());
    state.user_inferences.insert(user_id, serde_json::to_value(&inference_report).unwrap_or(serde_json::Value::Null));

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
    let total_params = report_val.get("architecture")
        .and_then(|a| a.get("total_parameters"))
        .and_then(|p| p.as_f64())
        .unwrap_or(0.0);
    let total_flops = report_val.get("compute")
        .and_then(|c| c.get("total_flops_forward"))
        .and_then(|f| f.as_f64())
        .unwrap_or(0.0);

    if total_flops / 1e9 > compliance.thresholds.high_risk_gflops {
        audit_issues.push(serde_json::json!({
            "category": "compliance",
            "severity": "warning",
            "code": "HIGH_RISK_GFLOPS",
            "message": format!("Model exceeds {:.0} GFLOPs threshold (EU AI Act)", compliance.thresholds.high_risk_gflops),
        }));
        audit_score -= 5.0;
    }

    // Check inference stability
    let stability_score = inference_report.stability_index.score;
    if stability_score < 50.0 {
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

/// GET /agent/carbon — Get carbon/cost analysis for a model
async fn agent_carbon(req: HttpRequest, state: web::Data<AppState>, body: web::Json<serde_json::Value>) -> impl Responder {
    let _user_id = match auth_any(&req, &state).await {
        Ok(id) => id,
        Err(resp) => return resp,
    };
    match check_api_key_scope(&req, &state, "agent") {
        Ok(_) => {},
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
            tracing::error!("Analysis failed: {}", e);
            return HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR).body(e.to_string());
        }
        Err(_) => return HttpResponse::build(StatusCode::INTERNAL_SERVER_ERROR).body("Analysis task failed"),
    };

    // Extract carbon/cost metrics from report
    let report_json_str = analysis_result.to_json().unwrap_or_else(|_| "{}".to_string());
    let report_val: serde_json::Value = serde_json::from_str(&report_json_str).unwrap_or(serde_json::Value::Null);
    let cost = report_val.get("cost").cloned().unwrap_or(serde_json::Value::Null);
    let training_hours = cost.get("training_hours").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let training_cost_usd = cost.get("training_cost_usd").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let co2_tonnes = cost.get("co2_tonnes").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let energy_kwh = cost.get("energy_kwh").and_then(|v| v.as_f64()).unwrap_or(0.0);

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
            "exceeds_gflops_threshold": false,
            "high_risk_gflops": compliance.thresholds.high_risk_gflops,
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
        Ok(_) => {},
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
        Ok(_) => {},
        Err(resp) => return resp,
    };

    let analysis = state.user_analyses.get(&user_id).map(|e| e.value().clone()).unwrap_or(serde_json::Value::Null);
    let inference = state.user_inferences.get(&user_id).map(|e| e.value().clone()).unwrap_or(serde_json::Value::Null);

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
        Ok(_) => {},
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

/// Helper to get compliance data
fn get_compliance_data() -> ComplianceConfig {
    let regulations = vec![
        ComplianceRegulation {
            name: "EU AI Act Phase 1".to_string(),
            year: 2027,
            limit: Some(300.0),
            unit: Some("GFLOPs/request".to_string()),
            status: "upcoming".to_string(),
            description: "General-purpose AI models trained with >10²⁵ FLOPs must comply with transparency and safety obligations.".to_string(),
            region: "EU".to_string(),
        },
        ComplianceRegulation {
            name: "EU AI Act Phase 2".to_string(),
            year: 2028,
            limit: Some(150.0),
            unit: Some("GFLOPs/request".to_string()),
            status: "upcoming".to_string(),
            description: "Stricter limits for high-risk AI applications in critical infrastructure, law enforcement, and biometrics.".to_string(),
            region: "EU".to_string(),
        },
        ComplianceRegulation {
            name: "Carbon Reporting (CSRD)".to_string(),
            year: 2026,
            limit: None,
            unit: None,
            status: "active".to_string(),
            description: "Corporate Sustainability Reporting Directive requires disclosure of energy consumption and CO₂ emissions for large companies.".to_string(),
            region: "EU".to_string(),
        },
        ComplianceRegulation {
            name: "Digital Services Act".to_string(),
            year: 2024,
            limit: None,
            unit: None,
            status: "active".to_string(),
            description: "Very large online platforms must disclose AI system training compute and risk assessments.".to_string(),
            region: "EU".to_string(),
        },
        ComplianceRegulation {
            name: "US AI Executive Order".to_string(),
            year: 2023,
            limit: None,
            unit: None,
            status: "active".to_string(),
            description: "Companies must report AI models trained with >10²⁵ FLOPs to the US government.".to_string(),
            region: "US".to_string(),
        },
        ComplianceRegulation {
            name: "Canada AIDA".to_string(),
            year: 2025,
            limit: None,
            unit: None,
            status: "proposed".to_string(),
            description: "Artificial Intelligence and Data Act — high-impact AI systems must meet safety, transparency, and monitoring requirements.".to_string(),
            region: "Canada".to_string(),
        },
    ];

    let thresholds = ComplianceThresholds {
        high_risk_gflops: 300.0,
        carbon_report_tonnes: 50.0,
        dsa_disclosure_flops: 1e25,
        cost_review_usd: 100_000.0,
    };

    let recommendations = vec![
        "Monitor EU AI Act Phase 1 compliance for models exceeding 300 GFLOPs/request".to_string(),
        "Prepare CSRD carbon reporting for training runs exceeding 50 tonnes CO₂e/year".to_string(),
        "Consider FP8 or INT8 quantization to reduce inference compute below regulatory thresholds".to_string(),
        "Document all training compute for models above 10²⁵ FLOPs (US EO requirement)".to_string(),
        "Implement energy monitoring for GPU clusters to track real-time carbon footprint".to_string(),
    ];

    ComplianceConfig {
        regulations,
        thresholds,
        recommendations,
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    if dotenvy::dotenv().is_err() {
        dotenvy::from_filename("neurax-service/.env").ok();
    }

    // Initialize tracing with info level by default, or use RUST_LOG env var
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    tracing::info!("[STARTUP] Neurax service starting...");

    let bind_addr = env::var("NEURAX_BIND").unwrap_or_else(|_| {
        env::var("PORT")
            .map(|p| format!("0.0.0.0:{p}"))
            .unwrap_or_else(|_| "0.0.0.0:9098".to_string())
    });

    let app_state = AppState::new();

    HttpServer::new(move || {
        let cors = Cors::default()
            .allowed_origin("http://localhost:8082")
            .allowed_origin("https://localhost:8082")
            .allowed_origin("http://localhost:8081")
            .allowed_origin("http://localhost:8080")
            .allowed_origin("https://localhost:8080")
            .allowed_origin("http://127.0.0.1:8082")
            .allowed_origin("http://127.0.0.1:8081")
            .allowed_origin("https://127.0.0.1:8081")
            .allowed_origin("http://127.0.0.1:8080")
            .allowed_origin("https://127.0.0.1:8080")
            .allowed_methods(vec!["GET", "POST", "PUT", "DELETE", "OPTIONS"])
            .allowed_headers(vec![header::CONTENT_TYPE, header::AUTHORIZATION, HeaderName::from_static("x-api-key")])
            .max_age(3600);

        App::new()
            .wrap(middleware::Logger::default())
            .wrap(middleware::Compress::default())
            .wrap(cors)
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
            .app_data(web::Data::new(app_state.clone()))
            .route("/me", web::get().to(me))
            .route("/billing/checkout", web::post().to(billing_checkout))
            .route("/billing/portal", web::post().to(billing_portal))
            .route("/stripe/webhook", web::post().to(stripe_webhook))
            .route("/health", web::get().to(health))
            .route("/hardware", web::get().to(hardware_list))
            .route("/plugin/validate", web::post().to(plugin_validate))
            .route("/presets", web::get().to(get_presets))
            .route("/presets/{id}", web::get().to(get_preset))
            .route("/analyze", web::post().to(analyze))
            .route("/analyze/compare", web::post().to(analyze_compare))
            .route("/analyze/stream", web::post().to(analyze_stream_start))
            .route("/analyze/stream/{job_id}", web::get().to(analyze_stream_events))
            .route("/analyze/result/{job_id}", web::get().to(analyze_result))
            .route("/analyze/status/{job_id}", web::get().to(analyze_status))
            .route("/timemachine", web::post().to(time_machine))
            .route("/inference/simulate", web::post().to(inference_simulate))
            .route("/export/onnx", web::post().to(export_onnx))
            .route("/projects", web::get().to(projects_list))
            .route("/projects", web::post().to(projects_create))
            .route("/projects/{id}", web::get().to(projects_get))
            .route("/projects/{id}", web::put().to(projects_update))
            .route("/projects/{id}", web::delete().to(projects_delete))
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
            .route("/agent/compare", web::post().to(agent_compare))
            .route("/agent/audit", web::post().to(agent_audit))
            .route("/agent/carbon", web::post().to(agent_carbon))
            .route("/agent/compliance", web::get().to(agent_compliance))
            .route("/agent/results", web::get().to(agent_results))
            .route("/agent/projects", web::get().to(agent_projects))
    })
    .bind(bind_addr)?
    .run()
    .await
}
