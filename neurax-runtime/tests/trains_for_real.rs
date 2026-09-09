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
