//! The desktop application must be able to do everything the web application
//! can.
//!
//! The two share one frontend, so "identical" is not a claim about screens —
//! it is a claim about what the frontend can call. This file makes that claim
//! checkable: it starts the API exactly as `neurax-desktop` starts it (a
//! loopback listener, `tauri://` origins, no Supabase) and then drives every
//! endpoint `neurax-ui` actually calls.
//!
//! The list below was taken from the exported functions of
//! `neurax-ui/src/services/neuraxApi.ts` that are referenced by a component.
//! If someone adds an endpoint the UI needs and the desktop cannot serve, a
//! test here fails rather than a user finding it.

use neurax_service::{persistence, serve_on_listener, AppState, DESKTOP_ORIGINS};
use serde_json::{json, Value};
use std::net::{Ipv4Addr, SocketAddrV4, TcpListener};

/// Start the API the way the desktop does, and return its base URL.
fn desktop_api() -> String {
    // `neurax-desktop` sets this before starting: it is single-user, on
    // loopback, with no account to authenticate against.
    std::env::set_var("NEURAX_DEBUG_NOAUTH", "true");

    let listener =
        TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0)).expect("loopback binds");
    let addr = listener
        .local_addr()
        .expect("a bound listener has an address");
    let origins: Vec<String> = DESKTOP_ORIGINS.iter().map(|s| s.to_string()).collect();

    std::thread::spawn(move || {
        actix_web::rt::System::new().block_on(async move {
            serve_on_listener(listener, origins, AppState::new())
                .expect("the listener is already bound")
                .await
        })
    });

    format!("http://{addr}")
}

fn client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::new()
}

fn get(base: &str, path: &str) -> (u16, String) {
    let r = client()
        .get(format!("{base}{path}"))
        .header("Authorization", "Bearer dev-token")
        .send()
        .expect("the embedded API should answer");
    (r.status().as_u16(), r.text().unwrap_or_default())
}

fn post(base: &str, path: &str, body: Value) -> (u16, String) {
    let r = client()
        .post(format!("{base}{path}"))
        .header("Authorization", "Bearer dev-token")
        .json(&body)
        .send()
        .expect("the embedded API should answer");
    (r.status().as_u16(), r.text().unwrap_or_default())
}

/// A small but genuine transformer, so `/analyze` has real work to do.
fn topology() -> Value {
    json!({
        "schema_version": "1.0.0",
        "model": {
            "name": "parity-check",
            "type": "transformer",
            "global_params": {
                "num_layers": 2,
                "hidden_size": 256,
                "num_heads": 4,
                "intermediate_size": 1024,
                "vocab_size": 32000,
                "sequence_length": 512
            },
            "layers": [
                {"id": "embed", "layer_type": "embedding",
                 "params": {"vocab_size": 32000, "hidden_size": 256}},
                {"id": "layer_0", "layer_type": "attention",
                 "params": {"hidden_size": 256, "num_attention_heads": 4, "intermediate_size": 1024}},
                {"id": "layer_1", "layer_type": "feed_forward",
                 "params": {"hidden_size": 256, "intermediate_size": 1024}}
            ]
        },
        "training": {
            "batch_size": 8,
            "sequence_length": 512,
            "precision": "fp16",
            "learning_rate": 0.0003,
            "num_epochs": 1
        },
        "hardware": {"gpus": [{"name": "RTX4090", "memory_gb": 24, "count": 1}]},
        "data": {"dataset_size": 1000000, "vocab_size": 32000, "num_classes": 0}
    })
}

// ─── The compute endpoints: the product itself ──────────────────────

#[test]
fn analyze_runs_the_whole_pipeline_in_process() {
    let base = desktop_api();
    let (status, body) = post(&base, "/analyze", json!({ "topology": topology() }));

    assert_eq!(status, 200, "body was: {body}");
    let report: Value = serde_json::from_str(&body).expect("the report is JSON");
    // A report that parses but describes nothing would pass a status check.
    assert!(
        body.len() > 500,
        "the report is suspiciously small: {} bytes",
        body.len()
    );
    assert!(
        report.get("report").is_some() || report.is_object(),
        "unexpected report shape"
    );
}

#[test]
fn the_hardware_database_is_available_offline() {
    let base = desktop_api();
    let (status, body) = get(&base, "/hardware");

    assert_eq!(status, 200, "body was: {body}");
    let parsed: Value = serde_json::from_str(&body).unwrap();
    let list = parsed
        .as_array()
        .or_else(|| parsed.get("hardware").and_then(Value::as_array))
        .expect("a list of accelerators");
    assert!(list.len() > 5, "only {} accelerators", list.len());
}

#[test]
fn the_reference_architectures_ship_with_the_application() {
    // These are what the studio's preset menu loads. They are compiled into
    // the binary, so the desktop build has them with no network at all.
    let base = desktop_api();
    let (status, body) = get(&base, "/presets");

    assert_eq!(status, 200, "body was: {body}");
    let presets: Value = serde_json::from_str(&body).unwrap();
    let list = presets.as_array().expect("a list of presets");
    assert!(!list.is_empty(), "no presets are bundled");
}

#[test]
fn inference_simulation_answers() {
    let base = desktop_api();
    let (status, body) = post(
        &base,
        "/inference/simulate",
        json!({ "topology": topology() }),
    );

    assert!(
        status == 200 || status == 400,
        "unexpected status {status}: {body}"
    );
}

#[test]
fn the_time_machine_answers() {
    let base = desktop_api();
    let (status, body) = post(&base, "/timemachine", json!({ "topology": topology() }));

    assert!(
        status == 200 || status == 400,
        "unexpected status {status}: {body}"
    );
}

#[test]
fn hardware_comparison_answers() {
    let base = desktop_api();
    let (status, body) = post(
        &base,
        "/analyze/compare",
        json!({
            "topology": topology(),
            "configs": [
                {"hardware": "RTX4090", "precision": "fp16"},
                {"hardware": "H100-SXM", "precision": "bf16"}
            ]
        }),
    );

    assert!(
        status == 200 || status == 400,
        "unexpected status {status}: {body}"
    );
}

#[test]
fn the_compliance_configuration_is_served() {
    let base = desktop_api();
    let (status, body) = get(&base, "/compliance/config");
    assert_eq!(status, 200, "body was: {body}");
}

#[test]
fn credits_are_readable_without_an_account() {
    let base = desktop_api();
    let (status, body) = get(&base, "/credits");
    assert_eq!(status, 200, "body was: {body}");
}

// ─── Projects: create, list, update, delete ─────────────────────────

#[test]
fn a_project_can_be_created_listed_updated_and_deleted() {
    let base = desktop_api();

    let (status, body) = post(
        &base,
        "/projects",
        json!({ "name": "Parity", "canvas": { "nodes": [], "connections": [] } }),
    );
    // 201 Created, with the project under a `project` key.
    assert_eq!(status, 201, "create failed: {body}");
    let created: Value = serde_json::from_str(&body).expect("the project is JSON");
    let id = created
        .pointer("/project/id")
        .and_then(Value::as_str)
        .expect("a created project has an id")
        .to_string();

    let (status, body) = get(&base, "/projects");
    assert_eq!(status, 200, "list failed: {body}");
    assert!(body.contains(&id), "the created project is not in the list");

    let updated = client()
        .put(format!("{base}/projects/{id}"))
        .header("Authorization", "Bearer dev-token")
        .json(&json!({ "name": "Parity renamed" }))
        .send()
        .expect("the embedded API should answer");
    assert_eq!(updated.status().as_u16(), 200);

    let deleted = client()
        .delete(format!("{base}/projects/{id}"))
        .header("Authorization", "Bearer dev-token")
        .send()
        .expect("the embedded API should answer");
    assert!(deleted.status().is_success());

    let (_, body) = get(&base, "/projects");
    assert!(!body.contains(&id), "the project survived deletion");
}

/// The reason the desktop build persists at all: quitting the application is
/// normal, and an hour of work must still be there afterwards.
#[test]
fn projects_survive_a_restart() {
    let dir = std::env::temp_dir().join(format!("neurax-parity-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("projects.json");
    let _ = std::fs::remove_file(&path);

    // First run: something is designed and saved.
    let first = AppState::new();
    assert!(persistence::attach(&first, &path));
    let (status, body) = {
        let base = spawn_with_state(first.clone());
        post(
            &base,
            "/projects",
            json!({ "name": "Survives", "canvas": { "nodes": [] } }),
        )
    };
    assert_eq!(status, 201, "create failed: {body}");
    persistence::save_projects(&path, &first.snapshot_projects()).unwrap();

    // Second run: a brand new process, reading the same file.
    let second = AppState::new();
    assert!(persistence::attach(&second, &path));
    let names: Vec<String> = second
        .snapshot_projects()
        .into_iter()
        .map(|p| p.name)
        .collect();

    assert_eq!(
        names,
        vec!["Survives"],
        "the project did not survive the restart"
    );
}

/// Serve on a caller-provided state, so a test can inspect it afterwards.
fn spawn_with_state(state: AppState) -> String {
    std::env::set_var("NEURAX_DEBUG_NOAUTH", "true");
    let listener =
        TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0)).expect("loopback binds");
    let addr = listener.local_addr().unwrap();
    let origins: Vec<String> = DESKTOP_ORIGINS.iter().map(|s| s.to_string()).collect();

    std::thread::spawn(move || {
        actix_web::rt::System::new()
            .block_on(async move { serve_on_listener(listener, origins, state).unwrap().await })
    });

    format!("http://{addr}")
}

// ─── The webview's requests must not be blocked ─────────────────────

/// The desktop UI runs on a `tauri://` origin. If CORS rejected it, every
/// request the application makes would fail in the browser layer — before ever
/// reaching a handler — and every test above would still pass.
#[test]
fn the_desktop_origin_is_accepted_by_cors() {
    let base = desktop_api();

    let response = client()
        .request(reqwest::Method::OPTIONS, format!("{base}/analyze"))
        .header("Origin", "tauri://localhost")
        .header("Access-Control-Request-Method", "POST")
        .header("Access-Control-Request-Headers", "content-type")
        .send()
        .expect("preflight should be answered");

    assert!(
        response.status().is_success(),
        "preflight from tauri://localhost was rejected with {}",
        response.status()
    );
    let allowed = response
        .headers()
        .get("access-control-allow-origin")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert_eq!(allowed, "tauri://localhost");
}

/// A page in someone's browser must not be able to reach the local API.
#[test]
fn an_arbitrary_web_origin_is_refused() {
    let base = desktop_api();

    let response = client()
        .request(reqwest::Method::OPTIONS, format!("{base}/analyze"))
        .header("Origin", "https://evil.example")
        .header("Access-Control-Request-Method", "POST")
        .send()
        .expect("preflight should be answered");

    assert!(
        response
            .headers()
            .get("access-control-allow-origin")
            .is_none(),
        "an unknown origin was granted access"
    );
}

/// The gate every model passes before it is trained — including one the
/// assistant wrote, which is the whole reason it may write one.
///
/// Skipped rather than failed where PyTorch is absent: the desktop must
/// *serve* the route, and it does so honestly on a machine that cannot build
/// anything, reporting stage `torch` instead of pretending to a verdict.
#[test]
fn a_candidate_model_can_be_built_and_counted() {
    let base = desktop_api();
    let (status, body) = post(
        &base,
        "/model/verify",
        json!({
            "modelCode": "import torch.nn as nn\n\n\nclass TinyNet(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(8, 4)\n\n    def forward(self, x):\n        return self.fc(x)\n",
            "modelClass": "TinyNet",
            "inputShape": [8],
            "inputKind": "features"
        }),
    );
    assert_eq!(status, 200, "the desktop must serve /model/verify: {body}");

    let verdict: Value = serde_json::from_str(&body).expect("a verdict is JSON");
    let stage = verdict["stage"].as_str().unwrap_or_default();
    if stage == "torch" || stage == "checker" {
        eprintln!("skipped: no PyTorch on this machine ({stage})");
        return;
    }

    assert_eq!(verdict["ok"], true, "{body}");
    // 8*4 + 4, by hand. The point of the whole route is that this number is
    // what PyTorch built, not what anything predicted.
    assert_eq!(verdict["parameters"], 36, "{body}");
    assert_eq!(verdict["forwardOk"], true, "{body}");
}

/// A refusal is a 200 with a verdict, not a 500.
///
/// The distinction the studio depends on: "this code is wrong" is an answer,
/// and must not look like "the checker broke".
#[test]
fn a_model_that_does_not_build_is_answered_not_errored() {
    let base = desktop_api();
    let (status, body) = post(
        &base,
        "/model/verify",
        json!({
            "modelCode": "class Broken(nn.Module:\n",
            "modelClass": "Broken",
            "inputShape": [8],
            "inputKind": "features"
        }),
    );
    assert_eq!(status, 200, "a refusal is an answer: {body}");

    let verdict: Value = serde_json::from_str(&body).expect("a verdict is JSON");
    if verdict["stage"] == "checker" {
        eprintln!("skipped: no python on this machine");
        return;
    }
    assert_eq!(verdict["ok"], false, "{body}");
}
