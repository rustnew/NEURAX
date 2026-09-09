//! What machine is this, and what can it actually do?
//!
//! NEURAX has always answered the first question by asking the user: pick a
//! chip from a catalogue of 38. That was reasonable while the compiler was a
//! calculator you pointed at a hypothetical, and it is wrong now that the
//! product's claim is "know what this costs *here*". A catalogue answer is a
//! guess about someone else's machine.
//!
//! This crate answers both questions from the machine itself, and the split
//! between them is deliberate:
//!
//!  - **Detection is free and always runs.** Reading `/proc`, a `statvfs` and
//!    `nvidia-smi` costs milliseconds and touches nothing. It yields what is
//!    installed: cores, memory, disk, accelerators.
//!  - **Measurement costs a second and runs once.** How many FLOP/s a chip
//!    actually sustains, and how fast its memory really is, cannot be read
//!    from anywhere — the numbers vendors publish are ceilings no kernel
//!    reaches, and for the integrated GPU in a laptop there is often no
//!    published number at all. So they are measured, and the result is
//!    cached, keyed to the CPU it was measured on.
//!
//! ## Why measuring is not a contradiction
//!
//! `sweep.rs` says NEURAX never runs a model, and that stays true. This does
//! not run a model; it runs a matrix multiply and a memory sweep to calibrate
//! the *machine constant* the analytical formulas divide by. That is the same
//! role the hardware database plays for a recognised card — a table of
//! constants — except measured on the hardware in front of you rather than
//! copied from a vendor's slide. The compiler still never executes the design.
//!
//! ## Every field can be absent
//!
//! A container with no `/proc/cpuinfo`, a machine whose driver will not
//! answer, a locked-down disk: all are normal, and none is an error. The
//! shapes below say so — a profile with an empty `gpus` and an unknown CPU
//! model is still a valid profile, and the studio is expected to render it
//! rather than refuse it. Confidently wrong numbers are the failure mode this
//! codebase keeps removing, and inventing a GPU is the worst version of it.

use serde::{Deserialize, Serialize};

pub mod bench;
mod platform;

pub use bench::{measure_cached, measure_compute, ComputeMeasurement};

/// How a figure came to be known. Travels with the numbers so nothing
/// downstream has to guess how much to trust them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provenance {
    /// Read from the machine — a core count, a memory size, a driver string.
    Detected,
    /// Measured on this machine by `neurax-probe`.
    Measured,
    /// Taken from `neurax-hardware-db` for a part it recognises.
    Published,
    /// Derived from something else, and worth saying so.
    Estimated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CpuInfo {
    pub model: String,
    pub vendor: String,
    /// Physical cores. Falls back to the thread count when the machine does
    /// not expose the topology — better an equal number than a zero.
    pub cores: usize,
    pub threads: usize,
    pub architecture: String,
    /// Nominal clock, when the machine publishes one. Absent is common.
    pub base_mhz: Option<u32>,
}

/// A compute backend an accelerator can be reached through. `Cpu` is a real
/// member, not a placeholder: a machine with no accelerator is a machine
/// NEURAX should still analyse for, and for a great many users it is the only
/// machine they have.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ComputeBackend {
    Cuda,
    Rocm,
    Metal,
    Vulkan,
    Cpu,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuInfo {
    /// The name the driver reports, verbatim. Not normalised, because the
    /// exact string is what a spec lookup has to match against and what the
    /// user recognises.
    pub name: String,
    pub vendor: String,
    /// True when a spec for this part exists. Set by the caller that owns the
    /// database, not here — this crate reads hardware, it does not look
    /// anything up.
    pub recognised: bool,
    /// Whether the part is soldered beside the CPU and shares its memory. It
    /// changes the reading of every VRAM figure: an integrated GPU's "VRAM"
    /// is system RAM it is borrowing.
    pub integrated: bool,
    pub backend: ComputeBackend,
    pub vram_total_bytes: Option<u64>,
    /// What is free right now, which is what governs whether a design starts.
    pub vram_free_bytes: Option<u64>,
    pub driver_version: Option<String>,
    pub temperature_c: Option<f32>,
    pub utilisation_pct: Option<f32>,
    pub power_watts: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MachineProfile {
    pub detected_at: String,
    pub os: String,
    pub hostname: Option<String>,
    pub cpu: CpuInfo,
    pub ram_total_bytes: u64,
    pub ram_available_bytes: u64,
    pub disk_available_bytes: u64,
    /// Empty on a machine with no accelerator — a normal state, and the one
    /// this crate was first tested on.
    pub gpus: Vec<GpuInfo>,
    /// Measured throughput, when a measurement has been taken. `None` until
    /// one has; the studio asks for it explicitly rather than paying the cost
    /// on every page load.
    pub compute: Option<ComputeMeasurement>,
    /// Anything the probe could not read, named. A profile is never rejected
    /// for being partial, but the gaps are stated rather than hidden behind a
    /// plausible default.
    pub notes: Vec<String>,
}

impl MachineProfile {
    /// The accelerator the analysis should target, or `None` when the machine
    /// has only a CPU.
    ///
    /// Prefers a discrete part over an integrated one: a laptop with both is
    /// asking about the discrete card, and an integrated GPU sharing system
    /// RAM is almost never the thing you train on.
    pub fn primary_gpu(&self) -> Option<&GpuInfo> {
        self.gpus
            .iter()
            .find(|g| !g.integrated)
            .or_else(|| self.gpus.first())
    }

    /// True when nothing but the CPU is available to compute on.
    pub fn is_cpu_only(&self) -> bool {
        self.gpus.iter().all(|g| g.integrated) || self.gpus.is_empty()
    }
}

/// Read this machine. Never fails: what cannot be read is recorded in `notes`
/// and left absent, because a partial profile is worth more than an error.
pub fn detect() -> MachineProfile {
    let mut notes = Vec::new();
    let cpu = platform::cpu(&mut notes);
    let (ram_total, ram_available) = platform::memory(&mut notes);
    let disk_available = platform::disk_available(&mut notes);
    let gpus = platform::gpus(&mut notes);

    MachineProfile {
        detected_at: chrono::Utc::now().to_rfc3339(),
        os: platform::os_name(),
        hostname: platform::hostname(),
        cpu,
        ram_total_bytes: ram_total,
        ram_available_bytes: ram_available,
        disk_available_bytes: disk_available,
        gpus,
        compute: None,
        notes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Detection has to work on whatever machine the tests run on — a CI
    /// container with no GPU and a stubbed `/proc` included — so this asserts
    /// the shape and the invariants, never a specific machine.
    #[test]
    fn detection_always_returns_a_usable_profile() {
        let profile = detect();
        assert!(profile.cpu.threads >= 1, "a machine running this test has at least one thread");
        assert!(profile.cpu.cores >= 1, "cores never reports zero, even when the topology is unreadable");
        assert!(
            profile.cpu.cores <= profile.cpu.threads,
            "physical cores cannot exceed threads: got {} cores, {} threads",
            profile.cpu.cores,
            profile.cpu.threads
        );
        assert!(!profile.detected_at.is_empty());
    }

    #[test]
    fn a_machine_with_no_discrete_gpu_is_cpu_only() {
        let profile = detect();
        // Not an assertion about *this* machine — an assertion that the two
        // helpers agree with each other whatever it finds.
        match profile.primary_gpu() {
            Some(gpu) if !gpu.integrated => assert!(!profile.is_cpu_only()),
            _ => assert!(profile.is_cpu_only()),
        }
    }

    #[test]
    fn available_memory_never_exceeds_total() {
        let profile = detect();
        if profile.ram_total_bytes > 0 {
            assert!(profile.ram_available_bytes <= profile.ram_total_bytes);
        }
    }
}
