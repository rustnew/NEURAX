use actix_cors::Cors;
use actix_web::{
    http::{header, StatusCode},
    middleware, web, App, HttpRequest, HttpResponse, HttpServer, Responder,
};
use hmac::{Hmac, Mac};
use serde::Deserialize;
use sha2::Sha256;
use std::env;
use std::time::Duration;
use subtle::ConstantTimeEq;
use tracing_subscriber::EnvFilter;

mod presets;

#[derive(Debug, serde::Deserialize)]
struct AnalyzeRequest {
    topology: serde_json::Value,
}

#[derive(Debug, serde::Serialize)]
struct AnalyzeResponse {
    report: neurax_ir::report::ReportIR,
}

#[derive(Debug, serde::Serialize)]
struct HealthResponse {
    status: &'static str,
}

#[derive(Debug, serde::Serialize)]
struct HardwareEntry {
    name: String,
    peak_ops_per_s_fp16: f64,
    mem_bw_gbps: f64,
    vram_bytes: u64,
}

#[derive(Debug, serde::Deserialize)]
struct PluginValidateRequest {
    plugin: serde_json::Value,
}

#[derive(Debug, serde::Serialize)]
struct PluginValidateResponse {
    ok: bool,
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

async fn health() -> impl Responder {
    HttpResponse::Ok().json(HealthResponse { status: "ok" })
}

async fn hardware_list() -> impl Responder {
    let names = ["H100", "A100", "RTX4090", "RTX4080", "RTX3090"];
    let out: Vec<HardwareEntry> = names
        .iter()
        .map(|n| {
            let db = neurax_hardware_db::HardwareDatabase::new();
            let hw = db.get_gpu_or_fallback(n);
            HardwareEntry {
                name: hw.name.clone(),
                peak_ops_per_s_fp16: hw.tflops_fp16 * 1e12,
                mem_bw_gbps: hw.memory_bandwidth_gbs,
                vram_bytes: (hw.memory_gb as u64) * 1024 * 1024 * 1024,
            }
        })
        .collect();
    HttpResponse::Ok().json(out)
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

    HttpServer::new(|| {
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
            .allowed_methods(vec!["GET", "POST", "OPTIONS"])
            .allowed_headers(vec![header::CONTENT_TYPE, header::AUTHORIZATION])
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
    })
    .bind(bind_addr)?
    .run()
    .await
}
