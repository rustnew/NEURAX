//! What this machine can actually *run*, as opposed to what it contains.
//!
//! Detection answers "is there a GPU". This module answers the question that
//! decides whether a run starts: is there a driver, did it load, is there a
//! compute runtime, and does PyTorch see the accelerator?
//!
//! The distinction is not academic. The machine this was written on has
//! `nvidia-smi` installed and no working driver — a card, if it had one, would
//! be present and unusable. "Installed", "detected", "responsive" and "usable"
//! are four different states, and a tool that reports the first as though it
//! were the fourth will confidently plan a run that cannot start.
//!
//! Everything here is read by asking the tools themselves, with a short
//! timeout, and every field is optional. A machine with no Python is a normal
//! machine; it simply cannot run a training process yet, and that is a fact to
//! report rather than an error to raise.

use std::process::Command;

use serde::{Deserialize, Serialize};

/// How far a capability has been established.
///
/// The ladder matters more than the labels: a precision the hardware claims is
/// not a precision the runtime can use, and neither is proof that a kernel
/// will run. Anything below `Usable` should not be planned against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Support {
    /// Nothing on this machine offers it.
    Absent,
    /// Present on disk, but nothing has confirmed it works.
    Installed,
    /// The tool answered when asked.
    Responsive,
    /// A runtime reported it can use it.
    Usable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeInfo {
    pub name: String,
    pub version: Option<String>,
    pub support: Support,
    /// What was asked, and what came back — so a reader can check the claim
    /// rather than take it.
    pub evidence: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoftwareEnvironment {
    pub os: String,
    pub kernel: Option<String>,
    /// Python, and whether it can import torch — the pair that decides whether
    /// a training process can start at all.
    pub python: Option<RuntimeInfo>,
    pub torch: Option<RuntimeInfo>,
    /// The GPU compute stack, if any part of it is present.
    pub cuda: Option<RuntimeInfo>,
    pub rocm: Option<RuntimeInfo>,
    pub driver: Option<RuntimeInfo>,
    /// Precisions a real runtime says it can use, not ones a datasheet claims.
    pub precisions: Vec<String>,
    /// True when a training run could be started right now — on *something*,
    /// which on many machines means the CPU.
    pub can_train: bool,
    /// What stops training outright.
    pub blockers: Vec<String>,
    /// What is degraded but not disqualifying.
    ///
    /// Separate from `blockers`, and the distinction was a real bug: an NVIDIA
    /// driver that is installed and unresponsive was reported as a blocker, so
    /// a laptop with no NVIDIA card at all — but with the tooling installed —
    /// was told it could not train, while PyTorch sat there working perfectly
    /// well on its CPU. A degraded accelerator is a warning; only the absence
    /// of a runtime stops a run.
    pub warnings: Vec<String>,
}

/// Run a command with its arguments, returning trimmed stdout on success.
fn ask(bin: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(bin).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

fn python_bin() -> String {
    std::env::var("NEURAX_PYTHON").unwrap_or_else(|_| "python3".to_string())
}

/// Ask PyTorch about itself.
///
/// One interpreter launch answers every question that matters — version,
/// whether it sees an accelerator, and which precisions it will actually
/// accept — because asking the runtime is the only way to know. A driver
/// version and a datasheet together still cannot say whether `bfloat16` works
/// on this install.
fn probe_torch() -> Option<(String, bool, Vec<String>, String)> {
    let script = r#"
import json
try:
    import torch
except Exception as exc:
    print(json.dumps({"error": str(exc)}))
    raise SystemExit(0)

cuda = bool(torch.cuda.is_available())
precisions = ["fp32"]
# Asked of the runtime rather than assumed from the device name: a build
# without the right kernels will refuse a dtype the hardware supports.
for name, dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
    try:
        torch.zeros(2, dtype=dtype, device="cuda" if cuda else "cpu") + 1
        precisions.append(name)
    except Exception:
        pass
print(json.dumps({
    "version": torch.__version__,
    "cuda": cuda,
    "cudaVersion": getattr(torch.version, "cuda", None),
    "precisions": precisions,
}))
"#;
    let out = Command::new(python_bin()).arg("-c").arg(script).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    if value.get("error").is_some() {
        return None;
    }
    Some((
        value["version"].as_str()?.to_string(),
        value["cuda"].as_bool().unwrap_or(false),
        value["precisions"]
            .as_array()?
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect(),
        text,
    ))
}

pub fn detect(os: String) -> SoftwareEnvironment {
    let kernel = ask("uname", &["-r"]);

    let python = ask(&python_bin(), &["--version"]).map(|v| RuntimeInfo {
        name: "Python".into(),
        version: v.split_whitespace().nth(1).map(str::to_string),
        support: Support::Responsive,
        evidence: format!("{} --version → {v}", python_bin()),
    });

    let torch_probe = probe_torch();
    let torch = torch_probe.as_ref().map(|(version, _, _, evidence)| RuntimeInfo {
        name: "PyTorch".into(),
        version: Some(version.clone()),
        support: Support::Usable,
        evidence: evidence.clone(),
    });

    // The driver is asked separately from the runtime, because a present but
    // unloaded driver is the case that looks like a working GPU and is not.
    let driver_out = Command::new("nvidia-smi")
        .args(["--query-gpu=driver_version", "--format=csv,noheader"])
        .output();
    let driver = match driver_out {
        Ok(out) if out.status.success() => Some(RuntimeInfo {
            name: "NVIDIA driver".into(),
            version: String::from_utf8_lossy(&out.stdout).trim().lines().next().map(str::to_string),
            support: Support::Responsive,
            evidence: "nvidia-smi answered".into(),
        }),
        Ok(out) => Some(RuntimeInfo {
            name: "NVIDIA driver".into(),
            version: None,
            support: Support::Installed,
            evidence: format!(
                "nvidia-smi is installed but did not answer: {}",
                String::from_utf8_lossy(&out.stderr).lines().next().unwrap_or("no detail").trim()
            ),
        }),
        Err(_) => None,
    };

    let torch_sees_cuda = torch_probe.as_ref().map(|(_, cuda, _, _)| *cuda).unwrap_or(false);
    let cuda = ask("nvcc", &["--version"])
        .map(|v| RuntimeInfo {
            name: "CUDA toolkit".into(),
            version: v.split("release ").nth(1).and_then(|s| s.split(',').next()).map(str::to_string),
            support: if torch_sees_cuda { Support::Usable } else { Support::Installed },
            evidence: "nvcc --version".into(),
        })
        .or_else(|| {
            torch_probe.as_ref().and_then(|(_, cuda, _, _)| {
                cuda.then(|| RuntimeInfo {
                    name: "CUDA runtime".into(),
                    version: None,
                    support: Support::Usable,
                    evidence: "torch.cuda.is_available() → True".into(),
                })
            })
        });

    let rocm = ask("rocminfo", &["--version"]).map(|_| RuntimeInfo {
        name: "ROCm".into(),
        version: None,
        support: Support::Responsive,
        evidence: "rocminfo answered".into(),
    });

    let precisions = torch_probe
        .as_ref()
        .map(|(_, _, p, _)| p.clone())
        .unwrap_or_default();

    // Only what genuinely prevents a run. Everything else is a warning.
    let mut blockers = Vec::new();
    if python.is_none() {
        blockers.push("No Python interpreter was found, so no training process can be started.".into());
    } else if torch.is_none() {
        blockers.push("Python is available but cannot import PyTorch, which the training harness needs.".into());
    }

    let mut warnings = Vec::new();
    if let Some(d) = &driver {
        if d.support == Support::Installed {
            warnings.push(
                "An NVIDIA driver is installed but did not answer. Any NVIDIA card on this machine \
                 is currently unusable; training will fall back to the CPU."
                    .into(),
            );
        }
    }
    if torch.is_some() && !torch_sees_cuda && driver.is_some() {
        warnings.push("PyTorch does not see a CUDA device, so runs will use the CPU.".into());
    }

    SoftwareEnvironment {
        os,
        kernel,
        can_train: blockers.is_empty() && torch.is_some(),
        python,
        torch,
        cuda,
        rocm,
        driver,
        precisions,
        blockers,
        warnings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_reports_a_usable_shape_whatever_is_installed() {
        let env = detect("test-os".into());
        assert_eq!(env.os, "test-os");
        // Either it can train and says nothing is blocking, or it cannot and
        // names at least one reason. Never both, never neither.
        assert_eq!(env.can_train, env.blockers.is_empty() && env.torch.is_some());
    }

    #[test]
    fn a_present_but_unresponsive_driver_is_named_without_blocking_cpu_training() {
        // The bug this guards: an unresponsive NVIDIA driver was a blocker, so
        // a laptop with the tooling installed and no NVIDIA card was told it
        // could not train — while PyTorch worked fine on its CPU.
        let env = detect("test-os".into());
        if let Some(driver) = &env.driver {
            if driver.support == Support::Installed {
                assert!(
                    env.warnings.iter().any(|w| w.contains("did not answer")),
                    "a driver that did not answer must be named",
                );
                assert!(
                    !env.blockers.iter().any(|b| b.contains("did not answer")),
                    "a degraded accelerator must not stop a CPU run",
                );
            }
        }
    }

    #[test]
    fn a_machine_with_a_working_runtime_can_train() {
        let env = detect("test-os".into());
        if env.torch.is_some() {
            assert!(
                env.can_train,
                "PyTorch imports here, so training is possible; blockers were {:?}",
                env.blockers
            );
        }
    }

    #[test]
    fn every_reported_capability_carries_its_evidence() {
        let env = detect("test-os".into());
        for runtime in [&env.python, &env.torch, &env.cuda, &env.rocm, &env.driver]
            .into_iter()
            .flatten()
        {
            assert!(
                !runtime.evidence.trim().is_empty(),
                "{} was reported with no evidence behind it",
                runtime.name
            );
        }
    }
}
