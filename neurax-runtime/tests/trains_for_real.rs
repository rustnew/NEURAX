//! Does it actually train?
//!
//! Every other test in this crate checks the bookkeeping — directories, state
//! files, interrupted detection. None of them would notice if the harness
//! never took a single step, which is the one thing the whole crate exists to
//! do. This test starts a real Python process, trains a real model, and reads
//! back what it wrote.
//!
//! It is `#[ignore]`d by default because it needs PyTorch, which not every
//! machine that builds NEURAX has. Run it with:
//!
//! ```text
//! cargo test -p neurax-runtime -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::{Duration, Instant};

use neurax_runtime::*;

/// A model small enough to train in a second on a CPU, and real enough that
/// its parameter count is worth checking: 8 inputs → 16 hidden → 4 classes is
/// 8*16+16 + 16*4+4 = 212 parameters, by hand.
const TINY_MODEL: &str = r#"
import torch.nn as nn


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
"#;

fn request(name: &str, epochs: u64, steps_per_epoch: u64) -> StartRequest {
    StartRequest {
        name: name.into(),
        model_code: TINY_MODEL.into(),
        model_class: "TinyNet".into(),
        dataset_path: None,
        // Eight features per sample, matching TinyNet's first layer. This is
        // the field whose absence the first run of these tests exposed.
        input_shape: vec![8],
        input_kind: "features".into(),
        vocab_size: None,
        num_classes: 4,
        epochs,
        batch_size: 4,
        learning_rate: 1e-2,
        precision: "fp32".into(),
        steps_per_epoch,
        checkpoint_every_steps: 5,
        predictions: serde_json::json!({ "parameters": 212 }),
    }
}

/// Wait until `predicate` holds, or give up. Returns whether it held, so a
/// failure reports what the run was actually doing rather than a bare timeout.
fn wait_for(dir: &PathBuf, timeout: Duration, predicate: impl Fn(&RunState) -> bool) -> Option<RunState> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Ok(state) = read_state_reconciled(dir) {
            if predicate(&state) {
                return Some(state);
            }
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    None
}

fn diagnose(dir: &PathBuf) -> String {
    let state = std::fs::read_to_string(dir.join("state.json")).unwrap_or_default();
    let log = std::fs::read_to_string(dir.join("log.txt")).unwrap_or_default();
    format!("state.json:\n{state}\n\nlog.txt:\n{log}")
}

#[test]
#[ignore = "needs python3 with PyTorch"]
fn a_run_trains_reports_and_finishes() {
    let scratch = tempfile::tempdir().unwrap();
    std::env::set_var("NEURAX_RUNS_DIR", scratch.path());

    let created = create_run(&request("End To End", 2, 6)).unwrap();
    let dir = PathBuf::from(&created.directory);
    assert_eq!(created.total_steps, 12);

    let started = start_run(&created).unwrap();
    assert!(started.pid.is_some(), "a started run records the process that owns it");

    let finished = wait_for(&dir, Duration::from_secs(90), |s| {
        matches!(s.status, RunStatus::Finished | RunStatus::Failed)
    })
    .unwrap_or_else(|| panic!("the run never finished.\n{}", diagnose(&dir)));

    assert_eq!(finished.status, RunStatus::Finished, "run failed.\n{}", diagnose(&dir));
    assert_eq!(finished.step, 12, "every planned step should have been taken");

    // The parameter count, reported before the first step. This is the whole
    // argument for the Accuracy view: 212 is arithmetic, checkable by hand.
    let built = read_model_built(&dir).expect("model_built.json should exist");
    assert_eq!(
        built["parameters"].as_u64(),
        Some(212),
        "8*16+16 + 16*4+4 = 212, and the harness must report what PyTorch actually built"
    );
    assert!(built["torchVersion"].is_string());

    // One line per step, in order, with the fields the Live view draws.
    let steps = read_steps(&dir, 0).unwrap();
    assert_eq!(steps.len(), 12, "one line per step");
    for (i, step) in steps.iter().enumerate() {
        assert_eq!(step["step"].as_u64(), Some(i as u64 + 1), "steps are in order");
        assert!(step["loss"].as_f64().is_some_and(|v| v.is_finite()), "loss must be a real number");
        assert!(step["stepTimeMs"].as_f64().is_some_and(|v| v > 0.0));
        assert!(step["dataTimeMs"].as_f64().is_some());
        assert!(step["samplesPerSec"].as_f64().is_some_and(|v| v > 0.0));
    }
    // Epochs are derived from the same steps-per-epoch the plan was built on.
    assert_eq!(steps[0]["epoch"].as_u64(), Some(0));
    assert_eq!(steps[11]["epoch"].as_u64(), Some(1));

    // Incremental reads are what the studio uses to reattach mid-run.
    assert_eq!(read_steps(&dir, 10).unwrap().len(), 2);

    // Checkpoints, and the honest claim about what they can restore.
    let checkpoints = list_checkpoints(&dir);
    assert!(!checkpoints.is_empty(), "a finished run leaves a checkpoint");
    let last = checkpoints.last().unwrap();
    assert_eq!(last["step"].as_u64(), Some(12));
    assert_eq!(last["exactResume"].as_bool(), Some(true));
    assert!(last["sizeBytes"].as_u64().unwrap_or(0) > 0);
}

#[test]
#[ignore = "needs python3 with PyTorch"]
fn a_run_stops_when_asked_and_keeps_what_it_had() {
    let scratch = tempfile::tempdir().unwrap();
    std::env::set_var("NEURAX_RUNS_DIR", scratch.path());

    // Long enough that it is certainly still running when we ask it to stop.
    let created = create_run(&request("Stoppable", 200, 50)).unwrap();
    let dir = PathBuf::from(&created.directory);
    start_run(&created).unwrap();

    // Let it get past the first few steps, so stopping is interrupting real
    // work rather than racing the process's own startup.
    let running = wait_for(&dir, Duration::from_secs(90), |s| s.step >= 3)
        .unwrap_or_else(|| panic!("the run never reached step 3.\n{}", diagnose(&dir)));
    assert_eq!(running.status, RunStatus::Running);

    write_control(&dir, Control::Stop).unwrap();

    let stopped = wait_for(&dir, Duration::from_secs(60), |s| {
        matches!(s.status, RunStatus::Finished | RunStatus::Failed)
    })
    .unwrap_or_else(|| panic!("the run ignored the stop request.\n{}", diagnose(&dir)));

    assert_eq!(stopped.status, RunStatus::Finished, "a requested stop is not a failure");
    assert!(stopped.step >= 3, "the steps it took are kept, not discarded");
    assert!(stopped.step < 10_000, "it stopped rather than running to completion");

    // Stopping writes a checkpoint: the work done is recoverable, which is
    // the difference between stopping and losing.
    assert!(
        !list_checkpoints(&dir).is_empty(),
        "stopping must leave the run resumable.\n{}",
        diagnose(&dir)
    );
    assert_eq!(read_steps(&dir, 0).unwrap().len() as u64, stopped.step);
}

#[test]
#[ignore = "needs python3 with PyTorch"]
fn a_resumed_run_continues_instead_of_starting_over() {
    let scratch = tempfile::tempdir().unwrap();
    std::env::set_var("NEURAX_RUNS_DIR", scratch.path());

    let created = create_run(&request("Resumable", 3, 4)).unwrap();
    let dir = PathBuf::from(&created.directory);
    start_run(&created).unwrap();
    wait_for(&dir, Duration::from_secs(90), |s| s.status == RunStatus::Finished)
        .unwrap_or_else(|| panic!("first pass never finished.\n{}", diagnose(&dir)));

    let first_steps = read_steps(&dir, 0).unwrap().len();
    assert_eq!(first_steps, 12);

    // Restarting the same directory must pick up from the checkpoint, not
    // retrain from step 1 — the whole promise of an interrupted run.
    let reread = read_state(&dir).unwrap();
    start_run(&reread).unwrap();
    let again = wait_for(&dir, Duration::from_secs(60), |s| {
        matches!(s.status, RunStatus::Finished | RunStatus::Failed)
    })
    .unwrap_or_else(|| panic!("the resumed run never settled.\n{}", diagnose(&dir)));

    assert_eq!(again.status, RunStatus::Finished);
    assert_eq!(
        read_steps(&dir, 0).unwrap().len(),
        first_steps,
        "a resumed run that was already complete adds no steps — it does not retrain"
    );
}

#[test]
#[ignore = "needs python3 with PyTorch"]
fn a_model_that_cannot_be_built_fails_with_the_reason() {
    let scratch = tempfile::tempdir().unwrap();
    std::env::set_var("NEURAX_RUNS_DIR", scratch.path());

    let mut req = request("Broken", 1, 2);
    req.model_code = "import torch.nn as nn\n\n\nclass Broken(nn.Module):\n    def __init__(self):\n        raise ValueError('this design cannot be built')\n".into();
    req.model_class = "Broken".into();

    let created = create_run(&req).unwrap();
    let dir = PathBuf::from(&created.directory);
    start_run(&created).unwrap();

    let failed = wait_for(&dir, Duration::from_secs(60), |s| s.status == RunStatus::Failed)
        .unwrap_or_else(|| panic!("a broken model should fail, not hang.\n{}", diagnose(&dir)));

    let reason = failed.error.unwrap_or_default();
    assert!(
        reason.contains("this design cannot be built"),
        "the failure must carry the reason, not just the fact: got {reason:?}"
    );
}

// ─── Verification ───────────────────────────────────────────────────────────
//
// The gate a candidate model passes before anything is trained. It matters
// most for code NEURAX did not write itself: the assistant can express designs
// the deterministic translator cannot, and the only thing that makes that
// trustworthy is being held to the same check.

#[test]
#[ignore = "needs python3 with PyTorch"]
fn a_correct_model_reports_the_parameter_count_it_really_has() {
    let verdict = verify_model(&VerifyRequest {
        model_code: TINY_MODEL.into(),
        model_class: "TinyNet".into(),
        input_shape: vec![8],
        input_kind: "features".into(),
        vocab_size: None,
    })
    .expect("the checker should run");

    assert!(verdict.ok, "verification failed: {verdict:?}");
    // 8*16+16 + 16*4+4 = 212, by hand.
    assert_eq!(verdict.parameters, Some(212));
    assert_eq!(verdict.class_name.as_deref(), Some("TinyNet"));
    assert!(verdict.forward_ok, "one batch should go through: {:?}", verdict.forward_error);
    assert_eq!(verdict.output_shape, Some(vec![2, 4]), "batch of 2, four classes out");
}

#[test]
#[ignore = "needs python3 with PyTorch"]
fn a_model_that_does_not_import_is_reported_at_that_stage() {
    let verdict = verify_model(&VerifyRequest {
        // Genuinely unparseable. (`this is not python` would *not* be: it
        // parses as an `is not` comparison and fails later, at name lookup.)
        model_code: "import torch.nn as nn\n\nclass Broken(nn.Module:\n".into(),
        model_class: "Anything".into(),
        input_shape: vec![8],
        input_kind: "features".into(),
        vocab_size: None,
    })
    .expect("the checker should run");

    assert!(!verdict.ok);
    assert_eq!(verdict.stage, "import");
    assert!(verdict.error.unwrap_or_default().contains("SyntaxError"));
}

#[test]
#[ignore = "needs python3 with PyTorch"]
fn a_model_that_builds_but_cannot_take_its_input_is_caught_before_training() {
    // The failure the parameter check cannot see: right weights, wrong shapes.
    // A run would die on its first batch; this costs a second.
    let verdict = verify_model(&VerifyRequest {
        model_code: TINY_MODEL.into(),
        model_class: "TinyNet".into(),
        // TinyNet's first layer takes 8 features, not an image.
        input_shape: vec![3, 32, 32],
        input_kind: "image".into(),
        vocab_size: None,
    })
    .expect("the checker should run");

    assert!(verdict.ok, "it builds — the model itself is fine");
    assert_eq!(verdict.parameters, Some(212));
    assert!(!verdict.forward_ok, "but the declared input does not fit it");
    assert!(verdict.forward_error.is_some());
}

#[test]
#[ignore = "needs python3 with PyTorch"]
fn a_model_that_prints_is_still_understood() {
    // Model code that announces itself in `__init__` is ordinary — every
    // second published implementation does it — and it used to sit in front of
    // the verdict on stdout and make it unparseable. A chatty model would have
    // been reported as "the checker did not answer", which is both wrong and
    // impossible to act on.
    let verdict = verify_model(&VerifyRequest {
        model_code: "import torch.nn as nn\n\n\nclass Chatty(nn.Module):\n    def __init__(self):\n        super().__init__()\n        print('building Chatty')\n        print('{\"not\": \"a verdict\"}')\n        self.fc = nn.Linear(8, 4)\n\n    def forward(self, x):\n        return self.fc(x)\n".into(),
        model_class: "Chatty".into(),
        input_shape: vec![8],
        input_kind: "features".into(),
        vocab_size: None,
    })
    .expect("the checker should run");

    assert!(verdict.ok, "printing is not a fault: {verdict:?}");
    assert_eq!(verdict.parameters, Some(36));
    assert!(verdict.forward_ok);
}

#[test]
#[ignore = "needs python3 with PyTorch"]
fn an_embedding_is_fed_indices_rather_than_floats() {
    // `inputKind` is not decoration: an embedding takes integers, and a float
    // tensor fails in a way that reads like a shape bug.
    let verdict = verify_model(&VerifyRequest {
        model_code: "import torch.nn as nn\n\n\nclass Emb(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.e = nn.Embedding(100, 16)\n\n    def forward(self, x):\n        return self.e(x)\n".into(),
        model_class: "Emb".into(),
        input_shape: vec![12],
        input_kind: "tokens".into(),
        vocab_size: Some(100),
    })
    .expect("the checker should run");

    assert!(verdict.ok && verdict.forward_ok, "{verdict:?}");
    assert_eq!(verdict.parameters, Some(1600));
    assert_eq!(verdict.output_shape, Some(vec![2, 12, 16]));
}
