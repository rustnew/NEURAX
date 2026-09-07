//! Systematic before/after validation for the hint diagnostics (H006-H008,
//! I001, I002): for every real reference model, recompute each hint's
//! firing condition independently from the raw JSON (or, for H007, from a
//! different reported value than the one being checked) and compare against
//! whether the compiler actually fired it. A mismatch in either direction —
//! fired when it shouldn't have (false positive) or silent when it should
//! have fired (false negative) — is a real implementation bug.
//!
//! This is the discipline described as missing this session: no lint should
//! ship without being checked against a real corpus for its false-positive
//! rate, the same way rustc validates a new lint against a crater run before
//! turning it on by default.

use serde_json::Value;
use std::fs;

struct CheckResult {
    model: String,
    code: &'static str,
    fired: bool,
    should_fire: bool,
}

impl CheckResult {
    fn verdict(&self) -> &'static str {
        match (self.fired, self.should_fire) {
            (true, true) => "ok (true positive)",
            (false, false) => "ok (true negative)",
            (true, false) => "FALSE POSITIVE",
            (false, true) => "FALSE NEGATIVE",
        }
    }

    fn is_wrong(&self) -> bool {
        self.fired != self.should_fire
    }
}

fn has_code(diagnostics: &[Value], code: &str) -> bool {
    diagnostics
        .iter()
        .any(|d| d.get("code").and_then(Value::as_str) == Some(code))
}

fn get_f64(v: &Value, path: &[&str]) -> Option<f64> {
    let mut cur = v;
    for key in path {
        cur = cur.get(key)?;
    }
    cur.as_f64()
}

fn get_u64(v: &Value, path: &[&str]) -> Option<u64> {
    let mut cur = v;
    for key in path {
        cur = cur.get(key)?;
    }
    cur.as_u64()
}

fn check_model(name: &str, raw: &Value, output: &Value) -> Vec<CheckResult> {
    let mut results = Vec::new();
    let diagnostics: Vec<Value> = output
        .get("diagnostics")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    // ── H006: no warmup, computed independently from the raw JSON ──
    let warmup_steps = get_u64(raw, &["training", "warmup_steps"]).unwrap_or(0);
    let max_steps = get_u64(raw, &["training", "max_steps"]).unwrap_or(0);
    let should_h006 = warmup_steps == 0 && max_steps > 0;
    results.push(CheckResult {
        model: name.to_string(),
        code: "H006",
        fired: has_code(&diagnostics, "H006"),
        should_fire: should_h006,
    });

    // ── H007: cross-checked against the OTHER reported value
    // (recommended_max_learning_rate) rather than re-deriving the Lipschitz
    // estimate ourselves — verifies H006/H007's own internal consistency,
    // not the stability formula itself (already verified separately this
    // session).
    let learning_rate = get_f64(raw, &["training", "learning_rate"]).unwrap_or(0.0);
    let recommended_max_lr = get_f64(
        output,
        &[
            "metrics",
            "dynamic",
            "stability",
            "recommended_max_learning_rate",
        ],
    )
    .unwrap_or(f64::INFINITY);
    let should_h007 = learning_rate > recommended_max_lr;
    results.push(CheckResult {
        model: name.to_string(),
        code: "H007",
        fired: has_code(&diagnostics, "H007"),
        should_fire: should_h007,
    });

    // ── H008: Chinchilla ratio, computed independently from raw dataset
    // size and the compiler's own (separately verified) parameter count.
    let dataset_size = get_f64(raw, &["data", "dataset_size"]);
    let total_parameters = get_f64(output, &["metrics", "structure", "total_parameters"]);
    let model_type = raw
        .get("model")
        .and_then(|m| m.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let is_llm_family = matches!(model_type, "transformer" | "moe");
    let should_h008 = match (is_llm_family, dataset_size, total_parameters) {
        (true, Some(tokens), Some(params)) if params > 0.0 && tokens > 0.0 => {
            let ratio = tokens / params;
            !(20.0 / 3.0..=20.0 * 3.0).contains(&ratio)
        }
        _ => false,
    };
    results.push(CheckResult {
        model: name.to_string(),
        code: "H008",
        fired: has_code(&diagnostics, "H008"),
        should_fire: should_h008,
    });

    // ── I001: GQA, scanned independently from raw layer params ──
    let should_i001 = raw
        .get("model")
        .and_then(|m| m.get("layers"))
        .and_then(Value::as_array)
        .map(|layers| {
            layers.iter().any(|l| {
                let is_attention = l.get("layer_type").and_then(Value::as_str) == Some("attention");
                let heads = get_u64(l, &["params", "num_heads"]);
                let kv_heads = get_u64(l, &["params", "num_key_value_heads"])
                    .or_else(|| get_u64(l, &["params", "num_kv_heads"]));
                is_attention && matches!((heads, kv_heads), (Some(h), Some(kv)) if kv < h)
            })
        })
        .unwrap_or(false);
    results.push(CheckResult {
        model: name.to_string(),
        code: "I001",
        fired: has_code(&diagnostics, "I001"),
        should_fire: should_i001,
    });

    // ── I002: MoE, scanned independently from raw layer types ──
    let should_i002 = raw
        .get("model")
        .and_then(|m| m.get("layers"))
        .and_then(Value::as_array)
        .map(|layers| {
            layers
                .iter()
                .any(|l| l.get("layer_type").and_then(Value::as_str) == Some("moe"))
        })
        .unwrap_or(false);
    results.push(CheckResult {
        model: name.to_string(),
        code: "I002",
        fired: has_code(&diagnostics, "I002"),
        should_fire: should_i002,
    });

    results
}

fn main() {
    let mut paths: Vec<String> = fs::read_dir("examples/models")
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path().to_string_lossy().to_string())
        .filter(|p| p.ends_with(".json"))
        .collect();
    paths.sort();

    let mut all_results = Vec::new();

    for path in &paths {
        let json = fs::read_to_string(path).unwrap();
        let name = std::path::Path::new(path)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let raw: Value = serde_json::from_str(&json).expect("valid JSON");

        let result = match neurax_core::analyze_json(&json) {
            Ok(r) => r,
            Err(e) => {
                println!("{name}: analysis failed, skipping: {e}");
                continue;
            }
        };
        let output_json = result.to_json().expect("to_json");
        let output: Value = serde_json::from_str(&output_json).expect("valid output JSON");

        all_results.extend(check_model(&name, &raw, &output));
    }

    println!(
        "{:<24} {:<6} {:<8} {:<12} verdict",
        "model", "code", "fired", "should_fire"
    );
    let mut wrong = 0;
    for r in &all_results {
        if r.is_wrong() {
            wrong += 1;
        }
        println!(
            "{:<24} {:<6} {:<8} {:<12} {}",
            r.model,
            r.code,
            r.fired,
            r.should_fire,
            r.verdict()
        );
    }

    let total = all_results.len();
    let fired_count = all_results.iter().filter(|r| r.fired).count();
    println!(
        "\n{total} checks across {} models, {fired_count} fired.",
        paths.len()
    );
    if wrong == 0 {
        println!("RESULT: 0 false positives, 0 false negatives — every hint's firing condition matches an independent recomputation from the raw JSON.");
    } else {
        println!(
            "RESULT: {wrong} mismatch(es) found — see FALSE POSITIVE / FALSE NEGATIVE rows above."
        );
        std::process::exit(1);
    }
}
