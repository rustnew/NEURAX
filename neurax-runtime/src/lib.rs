//! A training run, and why it is a directory.
//!
//! NEURAX's compiler never executes a model, and that stays true — `sweep.rs`
//! still says so. This crate is the other side of the seam: the design leaves
//! NEURAX as PyTorch, a separate process takes it, and what comes back is
//! measurement rather than derivation.
//!
//! ## A run is a directory, not a process
//!
//! Everything about a run lives on disk, under one directory:
//!
//! ```text
//! ~/neurax/runs/<id>/
//!   state.json        what the run is, and how far it got
//!   control.json      what a human has asked it to do
//!   model.py          the design, as PyTorch
//!   train.py          the harness that trains it
//!   steps.jsonl       one line per step, appended
//!   model_built.json  the real parameter count, written before step 1
//!   log.txt           stdout and stderr
//!   checkpoints/
//! ```
//!
//! That is what makes the product's central promise true: closing the studio
//! does not stop a run, and reopening it does not lose one. A run held in
//! memory, or identified by a process handle, could not survive either. The
//! studio is a viewer onto a directory.
//!
//! ## Control by file, not by signal
//!
//! Pause and stop are written into `control.json`, which the training loop
//! reads between steps. Signals were the obvious alternative and are worse
//! here: `SIGSTOP` freezes a process mid-kernel with GPU memory still held and
//! no checkpoint written, and a custom handler has to be installed by the very
//! Python process we are trying to keep simple. A file is inspectable after
//! the fact, survives the studio, and lets the loop stop at a step boundary —
//! which is the only place stopping is clean.
//!
//! ## `interrupted` is inferred, never reported
//!
//! Nothing writes "interrupted": it is what the studio concludes when
//! `state.json` says `running` and the process that claimed it is gone. A
//! closed laptop, a killed terminal, a power cut all look like this, and none
//! of them gets the chance to write anything. Detecting it is why the state
//! file records a pid *and* the run id — pids are reused, and attaching to
//! whoever now holds 48211 would be worse than noticing nothing.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

mod process;

pub use process::is_running;

/// The Python harness. Embedded rather than generated as a string in Rust so
/// it is a real file that can be read, linted and run by hand — the first
/// thing anyone debugging a failed run will want to do.
const TRAIN_PY: &str = include_str!("train.py");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RunStatus {
    Running,
    Paused,
    Interrupted,
    Finished,
    Failed,
}

/// What `state.json` holds. Field names match the studio's `RunState`
/// exactly, because this file *is* the contract — the studio reads it through
/// the service without a translation layer that could drift.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunState {
    pub id: String,
    pub name: String,
    pub status: RunStatus,
    pub started_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_heartbeat: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pid: Option<u32>,
    pub step: u64,
    pub total_steps: u64,
    /// Optimizer steps in one pass over the training split. Absent for a
    /// stream with no defined end, where an epoch is not a thing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steps_per_epoch: Option<u64>,
    pub directory: String,
    /// Why a run failed, when it did. Absent otherwise — an empty string here
    /// would render as a failure with no explanation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// What a human has asked the run to do. Read by the training loop between
/// steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Control {
    #[default]
    Run,
    Pause,
    Stop,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ControlFile {
    pub command: Control,
    pub written_at: String,
}

/// Everything needed to start a run. Assembled by the studio, which is the
/// only place that holds all of it: the design (as PyTorch), the data, and
/// what the analysis predicted.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StartRequest {
    pub name: String,
    /// The model, as generated PyTorch. Produced by the studio's own
    /// `modelCodeGen`, which is verified against the analysis — regenerating
    /// it here would be a second source of truth for the thing the whole
    /// Accuracy view exists to check.
    pub model_code: String,
    /// The class to instantiate from `model_code`.
    pub model_class: String,
    /// Where the training data is. Absent runs on synthetic data, which is
    /// how the harness is smoke-tested without a dataset.
    #[serde(default)]
    pub dataset_path: Option<String>,
    /// One sample's shape, without the batch dimension: `[3, 224, 224]` for
    /// images, `[8]` for a tabular design, `[seq]` for a sequence model.
    ///
    /// Required, and it was not always. The harness previously defaulted to
    /// `[3, 224, 224]` when this was absent — which it always was, because
    /// the field did not exist — so every run fed 224×224 images to whatever
    /// the design actually was. A tabular model expecting 8 features got a
    /// 2688×224 matrix and died in its first `nn.Linear`. The end-to-end test
    /// found it on the first run; nothing else would have, because the
    /// bookkeeping was all correct.
    pub input_shape: Vec<u64>,
    /// What the first layer consumes.
    ///
    /// `tokens` for an embedding, which takes integer indices in `[0,
    /// vocab)`; `image` for `[C, H, W]` floats; `features` for a plain
    /// vector. The shape alone is not enough: feeding a float tensor to an
    /// embedding is the wrong *dtype*, and fails in a way that reads like a
    /// shape bug.
    #[serde(default = "default_input_kind")]
    pub input_kind: String,
    /// The vocabulary token indices must stay inside. Only for `tokens`.
    #[serde(default)]
    pub vocab_size: Option<u64>,
    /// Classes the model predicts. Used to generate synthetic targets, and to
    /// check a folder dataset has the number of classes the design expects.
    pub num_classes: u64,
    pub epochs: u64,
    pub batch_size: u64,
    pub learning_rate: f64,
    pub precision: String,
    /// Steps in one pass over the data, from the dataset profile. The step
    /// budget is derived from this and `epochs`, never stated directly — a
    /// round step count is a machine talking to itself.
    pub steps_per_epoch: u64,
    pub checkpoint_every_steps: u64,
    /// What NEURAX predicted, carried into the run so the Accuracy view can
    /// compare without the studio having to still be open.
    #[serde(default)]
    pub predictions: serde_json::Value,
}

fn default_input_kind() -> String {
    "features".to_string()
}

#[derive(Debug)]
pub enum RuntimeError {
    Io(std::io::Error),
    Json(serde_json::Error),
    NotFound(String),
    Spawn(String),
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::Json(e) => write!(f, "{e}"),
            Self::NotFound(id) => write!(f, "no run named {id}"),
            Self::Spawn(m) => write!(f, "could not start the training process: {m}"),
        }
    }
}
impl std::error::Error for RuntimeError {}
impl From<std::io::Error> for RuntimeError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<serde_json::Error> for RuntimeError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

type Result<T> = std::result::Result<T, RuntimeError>;

#[cfg(test)]
thread_local! {
    /// Per-thread runs root, for tests only.
    ///
    /// The first version of these tests set `NEURAX_RUNS_DIR` with
    /// `std::env::set_var`, which is process-global: run in parallel, every
    /// test pointed at whichever directory was set last, and a test asserting
    /// "a hundred runs, a hundred directories" found other tests' runs in its
    /// listing. It passed alone and failed together — the worst kind of test,
    /// because the failure looks like a product bug.
    ///
    /// A thread-local gives each test its own root with no coordination. It is
    /// deliberately test-only: the service is one process serving concurrent
    /// requests against one runs directory, and a per-thread root there would
    /// be a bug, not a feature.
    static TEST_RUNS_ROOT: std::cell::RefCell<Option<PathBuf>> = const { std::cell::RefCell::new(None) };
}

/// Where runs live.
pub fn runs_root() -> PathBuf {
    #[cfg(test)]
    {
        if let Some(dir) = TEST_RUNS_ROOT.with(|r| r.borrow().clone()) {
            return dir;
        }
    }
    if let Some(dir) = std::env::var_os("NEURAX_RUNS_DIR") {
        return PathBuf::from(dir);
    }
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    home.join("neurax").join("runs")
}

/// A readable, sortable run id.
///
/// Date first so a directory listing is chronological, then the name, then a
/// millisecond suffix.
///
/// The suffix is *not* a uniqueness guarantee, and an earlier version of this
/// claimed it was — "two runs in the same millisecond is not a thing" — which
/// its own test disproved on the first attempt. Two runs created back to back
/// produce the same id, and the second would have opened the first's
/// directory and overwritten its state, its model and its checkpoints.
///
/// Uniqueness is therefore established where it can actually be enforced: by
/// creating the directory, and only accepting the id if it did not exist
/// before. See `create_run`.
fn run_id(name: &str) -> String {
    let now = chrono::Utc::now();
    let slug: String = name
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .trim_matches('-')
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    let slug = slug.chars().take(40).collect::<String>();
    format!(
        "{}-{}-{:x}",
        now.format("%Y-%m-%d"),
        if slug.is_empty() { "run".into() } else { slug },
        now.timestamp_millis() & 0xffffff
    )
}

/// Create the run directory and everything in it, without starting anything.
///
/// Separate from `start` so the directory is complete and inspectable before
/// a process exists: if the spawn fails, what was going to run is still on
/// disk to look at, rather than vanishing with the error.
pub fn create_run(req: &StartRequest) -> Result<RunState> {
    let root = runs_root();
    std::fs::create_dir_all(&root)?;

    // Claim a directory rather than trust a name.
    //
    // `create_dir` fails with `AlreadyExists` instead of silently succeeding,
    // which turns "is this id free?" from a guess into an answer from the
    // filesystem — and closes the window between checking and creating, where
    // two concurrent starts would both see a free name.
    let base = run_id(&req.name);
    let mut id = base.clone();
    let mut dir = root.join(&id);
    for attempt in 1..1000 {
        match std::fs::create_dir(&dir) {
            Ok(()) => break,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                id = format!("{base}-{attempt}");
                dir = root.join(&id);
            }
            Err(e) => return Err(e.into()),
        }
    }
    if !dir.exists() {
        return Err(RuntimeError::Spawn(
            "could not find a free run directory after 1000 attempts".into(),
        ));
    }
    std::fs::create_dir_all(dir.join("checkpoints"))?;

    std::fs::write(dir.join("model.py"), &req.model_code)?;
    std::fs::write(dir.join("train.py"), TRAIN_PY)?;
    std::fs::write(dir.join("request.json"), serde_json::to_string_pretty(req)?)?;
    write_control(&dir, Control::Run)?;

    let total_steps = req.steps_per_epoch.saturating_mul(req.epochs).max(1);
    let state = RunState {
        id: id.clone(),
        name: req.name.clone(),
        status: RunStatus::Running,
        started_at: chrono::Utc::now().to_rfc3339(),
        last_heartbeat: None,
        pid: None,
        step: 0,
        total_steps,
        steps_per_epoch: Some(req.steps_per_epoch),
        directory: dir.to_string_lossy().to_string(),
        error: None,
    };
    write_state(&dir, &state)?;
    Ok(state)
}

/// Start the training process, detached.
///
/// Detached is the whole point: the process must outlive the service that
/// spawned it, or "closing NEURAX does not stop your run" is false the first
/// time someone quits the app. stdout and stderr go to `log.txt` rather than
/// to a pipe nobody is reading — a pipe with no reader fills and blocks the
/// child, which would hang a run at whatever step filled the buffer.
pub fn start_run(state: &RunState) -> Result<RunState> {
    let dir = PathBuf::from(&state.directory);
    let log = std::fs::File::create(dir.join("log.txt"))?;
    let log_err = log.try_clone()?;

    let child = std::process::Command::new(python_bin())
        .arg("-u") // unbuffered, or steps.jsonl lags the run by a buffer
        .arg(dir.join("train.py"))
        .current_dir(&dir)
        .stdout(std::process::Stdio::from(log))
        .stderr(std::process::Stdio::from(log_err))
        .stdin(std::process::Stdio::null())
        .spawn()
        .map_err(|e| RuntimeError::Spawn(format!("{} ({e})", python_bin())))?;

    let mut started = state.clone();
    started.pid = Some(child.id());
    write_state(&dir, &started)?;
    Ok(started)
}

fn python_bin() -> String {
    std::env::var("NEURAX_PYTHON").unwrap_or_else(|_| "python3".to_string())
}

pub fn write_state(dir: &Path, state: &RunState) -> Result<()> {
    // Written to a temporary file and renamed, so a reader never sees a
    // half-written state. A run's state file is read by the studio while the
    // training process is writing it, and a torn read there would show a run
    // as having no status at all.
    let tmp = dir.join("state.json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(state)?)?;
    std::fs::rename(tmp, dir.join("state.json"))?;
    Ok(())
}

pub fn read_state(dir: &Path) -> Result<RunState> {
    let text = std::fs::read_to_string(dir.join("state.json"))?;
    Ok(serde_json::from_str(&text)?)
}

pub fn write_control(dir: &Path, command: Control) -> Result<()> {
    let file = ControlFile {
        command,
        written_at: chrono::Utc::now().to_rfc3339(),
    };
    let tmp = dir.join("control.json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(&file)?)?;
    std::fs::rename(tmp, dir.join("control.json"))?;
    Ok(())
}

/// Read a run's state, correcting the one thing the file cannot know.
///
/// A state file saying `running` is only true while the process that wrote it
/// is alive. When it is not, the run was interrupted — and the file will
/// never say so itself, because whatever ended it did not give it the chance.
pub fn read_state_reconciled(dir: &Path) -> Result<RunState> {
    let mut state = read_state(dir)?;
    if matches!(state.status, RunStatus::Running | RunStatus::Paused) {
        let alive = state.pid.map(process::is_running).unwrap_or(false);
        if !alive {
            state.status = RunStatus::Interrupted;
        }
    }
    Ok(state)
}

/// Every run on this machine, newest first.
pub fn list_runs() -> Vec<RunState> {
    let root = runs_root();
    let Ok(entries) = std::fs::read_dir(&root) else {
        return Vec::new();
    };
    let mut runs: Vec<RunState> = entries
        .flatten()
        .filter(|e| e.path().is_dir())
        // A directory with no readable state file is not a run — a partial
        // creation, or something else that happens to live here. Skipped
        // rather than reported as a broken run.
        .filter_map(|e| read_state_reconciled(&e.path()).ok())
        .collect();
    runs.sort_by(|a, b| b.started_at.cmp(&a.started_at));
    runs
}

pub fn run_dir(id: &str) -> Result<PathBuf> {
    // The id is a single path component by construction, but it arrives over
    // HTTP, so it is checked rather than trusted: anything with a separator
    // or a parent reference could reach outside the runs directory.
    if id.is_empty() || id.contains('/') || id.contains('\\') || id.contains("..") {
        return Err(RuntimeError::NotFound(id.to_string()));
    }
    let dir = runs_root().join(id);
    if dir.join("state.json").exists() {
        Ok(dir)
    } else {
        Err(RuntimeError::NotFound(id.to_string()))
    }
}

/// Steps recorded since a byte offset, and the offset to ask from next time.
///
/// The offset is the whole design, and the first version did not have one: it
/// read the entire file and filtered by step number *after* parsing every
/// line. A studio polling a 100 000-step run once a second would parse
/// 100 000 JSON objects a second at the end of it, and throw away all but the
/// newest — linear work per poll, quadratic over a run, growing exactly as the
/// run gets more interesting to watch.
///
/// Seeking past what the caller already has makes each poll cost the new data
/// only. It is also what a log tail does, for the same reason.
///
/// A partially written final line is normal — the training process appends as
/// it goes and a reader can arrive mid-write — so the returned offset stops at
/// the last complete line, and the next call picks the rest up whole.
pub fn read_steps_from(dir: &Path, from_byte: u64) -> Result<(Vec<serde_json::Value>, u64)> {
    use std::io::{Read, Seek, SeekFrom};

    let path = dir.join("steps.jsonl");
    if !path.exists() {
        return Ok((Vec::new(), 0));
    }
    let mut file = std::fs::File::open(&path)?;
    let len = file.metadata()?.len();

    // A shorter file than the offset means it was truncated or replaced — a
    // fresh run in a reused directory. Start over rather than seek past the
    // end and report nothing forever.
    let start = if from_byte > len { 0 } else { from_byte };
    file.seek(SeekFrom::Start(start))?;

    let mut text = String::new();
    file.take(len - start).read_to_string(&mut text)?;

    let mut out = Vec::new();
    let mut consumed = start;
    for line in text.split_inclusive('\n') {
        if !line.ends_with('\n') {
            // The last line has no terminator yet: it is still being written.
            break;
        }
        consumed += line.len() as u64;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
            out.push(value);
        }
    }
    Ok((out, consumed))
}

/// Steps after a given step number.
///
/// Kept for callers that think in steps rather than bytes — chiefly tests,
/// where "the run should have taken twelve steps" is the readable assertion.
/// It reads the whole file, so it is not what a live view should use.
pub fn read_steps(dir: &Path, after: u64) -> Result<Vec<serde_json::Value>> {
    let (all, _) = read_steps_from(dir, 0)?;
    Ok(all
        .into_iter()
        .filter(|v| v.get("step").and_then(|s| s.as_u64()).unwrap_or(0) > after)
        .collect())
}

/// The parameter count the process reported before its first step, if it has.
pub fn read_model_built(dir: &Path) -> Option<serde_json::Value> {
    std::fs::read_to_string(dir.join("model_built.json"))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
}

/// Checkpoints on disk, oldest first.
pub fn list_checkpoints(dir: &Path) -> Vec<serde_json::Value> {
    let Ok(entries) = std::fs::read_dir(dir.join("checkpoints")) else {
        return Vec::new();
    };
    let mut out: Vec<serde_json::Value> = entries
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "pt"))
        .filter_map(|e| {
            let path = e.path();
            let meta = e.metadata().ok()?;
            let stem = path.file_stem()?.to_string_lossy().to_string();
            let step: u64 = stem.rsplit('_').next()?.parse().ok()?;
            // `exact_resume` is a claim about what is inside the file, so it
            // is read from the sidecar the harness writes rather than assumed
            // from the filename.
            let exact = std::fs::read_to_string(path.with_extension("json"))
                .ok()
                .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
                .and_then(|v| v.get("exactResume").and_then(|b| b.as_bool()))
                .unwrap_or(false);
            Some(serde_json::json!({
                "step": step,
                "path": format!("checkpoints/{}", path.file_name()?.to_string_lossy()),
                "sizeBytes": meta.len(),
                "savedAt": chrono::DateTime::<chrono::Utc>::from(meta.modified().ok()?).to_rfc3339(),
                "exactResume": exact,
            }))
        })
        .collect();
    out.sort_by_key(|v| v.get("step").and_then(|s| s.as_u64()).unwrap_or(0));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A runs directory of this test's own, isolated from every other test
    /// running beside it. The returned guard must be held: dropping it
    /// deletes the directory.
    fn scratch() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        TEST_RUNS_ROOT.with(|r| *r.borrow_mut() = Some(dir.path().to_path_buf()));
        dir
    }

    fn request() -> StartRequest {
        StartRequest {
            name: "Test Run".into(),
            model_code: "import torch.nn as nn\nclass M(nn.Module):\n    pass\n".into(),
            model_class: "M".into(),
            dataset_path: None,
            input_shape: vec![8],
            input_kind: "features".into(),
            vocab_size: None,
            num_classes: 4,
            epochs: 2,
            batch_size: 4,
            learning_rate: 1e-3,
            precision: "fp32".into(),
            steps_per_epoch: 5,
            checkpoint_every_steps: 5,
            predictions: serde_json::json!({}),
        }
    }

    #[test]
    fn a_run_is_a_directory_with_everything_in_it() {
        let _scratch = scratch();
        let state = create_run(&request()).unwrap();
        let dir = PathBuf::from(&state.directory);

        for file in ["state.json", "control.json", "model.py", "train.py", "request.json"] {
            assert!(dir.join(file).exists(), "{file} should have been written");
        }
        assert!(dir.join("checkpoints").is_dir());
        // The step budget is derived from epochs and the data, never stated.
        assert_eq!(state.total_steps, 10);
        assert_eq!(state.steps_per_epoch, Some(5));
    }

    #[test]
    fn a_run_whose_process_is_gone_reads_as_interrupted() {
        let _scratch = scratch();
        let mut state = create_run(&request()).unwrap();
        let dir = PathBuf::from(&state.directory);

        // A pid that cannot be running: the state file still says `running`,
        // which is exactly what a killed terminal or a power cut leaves.
        state.pid = Some(u32::MAX - 1);
        state.status = RunStatus::Running;
        write_state(&dir, &state).unwrap();

        let read = read_state_reconciled(&dir).unwrap();
        assert_eq!(read.status, RunStatus::Interrupted, "nothing writes `interrupted` — it is inferred");
    }

    #[test]
    fn a_finished_run_is_not_reinterpreted_as_interrupted() {
        let _scratch = scratch();
        let mut state = create_run(&request()).unwrap();
        let dir = PathBuf::from(&state.directory);
        state.status = RunStatus::Finished;
        state.pid = Some(u32::MAX - 1);
        write_state(&dir, &state).unwrap();
        assert_eq!(read_state_reconciled(&dir).unwrap().status, RunStatus::Finished);
    }

    #[test]
    fn reading_from_an_offset_returns_only_new_lines() {
        let _scratch = scratch();
        let state = create_run(&request()).unwrap();
        let dir = PathBuf::from(&state.directory);
        let path = dir.join("steps.jsonl");

        std::fs::write(&path, "{\"step\":1}\n{\"step\":2}\n").unwrap();
        let (first, offset) = read_steps_from(&dir, 0).unwrap();
        assert_eq!(first.len(), 2);
        assert_eq!(offset, std::fs::metadata(&path).unwrap().len());

        // Nothing new: no work, no repeats.
        let (none, same) = read_steps_from(&dir, offset).unwrap();
        assert!(none.is_empty());
        assert_eq!(same, offset);

        // Append, and only the appended line comes back.
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(b"{\"step\":3}\n").unwrap();
        let (new, _) = read_steps_from(&dir, offset).unwrap();
        assert_eq!(new.len(), 1);
        assert_eq!(new[0]["step"], 3);
    }

    #[test]
    fn a_torn_last_line_is_not_consumed_until_it_is_whole() {
        let _scratch = scratch();
        let state = create_run(&request()).unwrap();
        let dir = PathBuf::from(&state.directory);
        let path = dir.join("steps.jsonl");

        // The process is mid-write: line 2 has no newline yet.
        std::fs::write(&path, "{\"step\":1}\n{\"step\":2, \"lo").unwrap();
        let (partial, offset) = read_steps_from(&dir, 0).unwrap();
        assert_eq!(partial.len(), 1, "an unterminated line is not read");

        // It finishes writing; the next poll gets the line whole, once.
        std::fs::write(&path, "{\"step\":1}\n{\"step\":2,\"loss\":1.0}\n").unwrap();
        let (rest, _) = read_steps_from(&dir, offset).unwrap();
        assert_eq!(rest.len(), 1);
        assert_eq!(rest[0]["step"], 2);
    }

    #[test]
    fn an_offset_past_the_end_starts_over_instead_of_reporting_nothing() {
        let _scratch = scratch();
        let state = create_run(&request()).unwrap();
        let dir = PathBuf::from(&state.directory);
        std::fs::write(dir.join("steps.jsonl"), "{\"step\":1}\n").unwrap();
        // A directory reused by a fresh run: the file is shorter than the
        // offset a previous reader held.
        let (steps, _) = read_steps_from(&dir, 10_000).unwrap();
        assert_eq!(steps.len(), 1);
    }

    #[test]
    fn steps_are_read_incrementally_and_survive_a_torn_last_line() {
        let _scratch = scratch();
        let state = create_run(&request()).unwrap();
        let dir = PathBuf::from(&state.directory);
        std::fs::write(
            dir.join("steps.jsonl"),
            "{\"step\":1,\"loss\":2.0}\n{\"step\":2,\"loss\":1.5}\n{\"step\":3,\"lo",
        )
        .unwrap();

        let all = read_steps(&dir, 0).unwrap();
        assert_eq!(all.len(), 2, "the half-written line is skipped, not fatal");
        let after = read_steps(&dir, 1).unwrap();
        assert_eq!(after.len(), 1);
        assert_eq!(after[0]["step"], 2);
    }

    #[test]
    fn a_run_id_cannot_escape_the_runs_directory() {
        let _scratch = scratch();
        for bad in ["../etc", "a/b", "..", ""] {
            assert!(run_dir(bad).is_err(), "{bad:?} should be rejected");
        }
    }

    #[test]
    fn control_round_trips() {
        let _scratch = scratch();
        let state = create_run(&request()).unwrap();
        let dir = PathBuf::from(&state.directory);
        write_control(&dir, Control::Pause).unwrap();
        let text = std::fs::read_to_string(dir.join("control.json")).unwrap();
        let file: ControlFile = serde_json::from_str(&text).unwrap();
        assert_eq!(file.command, Control::Pause);
    }

    #[test]
    fn run_ids_are_readable_and_chronological() {
        let id = run_id("My Model");
        assert!(id.starts_with(&chrono::Utc::now().format("%Y-%m-%d").to_string()));
        assert!(id.contains("my-model"));
    }

    #[test]
    fn two_runs_started_together_never_share_a_directory() {
        // This is the bug the first version of `run_id` had: the suffix is a
        // millisecond, two `create_run` calls in a row land in the same one,
        // and the second run would have overwritten the first's state, model
        // and checkpoints. Uniqueness comes from claiming the directory, not
        // from the name being unlikely to repeat.
        let _scratch = scratch();
        let a = create_run(&request()).unwrap();
        let b = create_run(&request()).unwrap();
        assert_ne!(a.id, b.id);
        assert_ne!(a.directory, b.directory);
        assert!(PathBuf::from(&a.directory).join("state.json").exists());
        assert!(PathBuf::from(&b.directory).join("state.json").exists());
        // And the first run's own state still describes the first run.
        assert_eq!(read_state(&PathBuf::from(&a.directory)).unwrap().id, a.id);
    }

    #[test]
    fn a_hundred_runs_all_get_their_own_directory() {
        let _scratch = scratch();
        let ids: std::collections::HashSet<String> =
            (0..100).map(|_| create_run(&request()).unwrap().id).collect();
        assert_eq!(ids.len(), 100, "every run must own its directory");
        assert_eq!(list_runs().len(), 100);
    }
}
