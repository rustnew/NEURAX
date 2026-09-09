//! Reading the machine, per platform.
//!
//! Linux is implemented against `/proc`, `/sys` and `statvfs` directly rather
//! than through a crate that abstracts all three. That is a deliberate trade:
//! this is the one place in NEURAX whose whole job is to tell the truth about
//! the hardware, and a dependency that silently reports a plausible default
//! where the kernel reported nothing would defeat the purpose. Reading the
//! files means every fallback in this file is visible and commented.
//!
//! Other platforms return what they can and record the rest in `notes`. A
//! macOS or Windows machine gets a real CPU and memory reading through the
//! portable paths; its accelerators are left empty and said to be, which is
//! the honest state until those code paths exist.

use std::process::Command;

use crate::{ComputeBackend, CpuInfo, GpuInfo, MemoryState};

// ─── CPU ────────────────────────────────────────────────────────────────────

pub(crate) fn cpu(notes: &mut Vec<String>) -> CpuInfo {
    // `num_cpus` reports logical processors on every platform, and it is the
    // one number always available — so it is the floor everything else is
    // reconciled against.
    let threads = num_cpus::get().max(1);
    let mut model = String::new();
    let mut vendor = String::new();
    let mut base_mhz = None;
    let mut cores = 0usize;
    let mut features: Vec<String> = Vec::new();

    #[cfg(target_os = "linux")]
    {
        if let Ok(text) = std::fs::read_to_string("/proc/cpuinfo") {
            // Physical cores are counted as distinct (physical id, core id)
            // pairs. Counting `cpu cores` alone is wrong on a dual-socket
            // machine, where the field repeats per socket.
            let mut seen = std::collections::HashSet::new();
            let (mut physical_id, mut core_id) = (None, None);

            for line in text.lines() {
                let Some((key, value)) = line.split_once(':') else {
                    // A blank line ends one processor's block.
                    if line.trim().is_empty() {
                        if let (Some(p), Some(c)) = (physical_id.take(), core_id.take()) {
                            seen.insert((p, c));
                        }
                    }
                    continue;
                };
                let (key, value) = (key.trim(), value.trim());
                match key {
                    "model name" if model.is_empty() => model = value.to_string(),
                    "vendor_id" if vendor.is_empty() => vendor = value.to_string(),
                    "physical id" => physical_id = value.parse::<u32>().ok(),
                    "core id" => core_id = value.parse::<u32>().ok(),
                    "cpu MHz" if base_mhz.is_none() => {
                        base_mhz = value.parse::<f64>().ok().map(|v| v.round() as u32)
                    }
                    // Only the sets that change how fast a kernel can go. The
                    // full flags line is over a hundred entries, almost none
                    // of which anything here would act on.
                    "flags" | "Features" if features.is_empty() => {
                        const INTERESTING: &[&str] = &[
                            "sse4_2", "avx", "avx2", "fma", "avx512f", "avx512bw", "avx512vnni",
                            "amx_bf16", "amx_int8", "neon", "asimd", "sve",
                        ];
                        features = value
                            .split_whitespace()
                            .filter(|f| INTERESTING.contains(f))
                            .map(str::to_string)
                            .collect();
                    }
                    _ => {}
                }
            }
            if let (Some(p), Some(c)) = (physical_id, core_id) {
                seen.insert((p, c));
            }
            cores = seen.len();
        } else {
            notes.push("/proc/cpuinfo could not be read; core topology is unknown.".into());
        }
    }

    // A container that hides the topology, or a platform with no code path
    // here, leaves `cores` at zero. Zero cores describes no machine, so the
    // thread count stands in — an equal number is a known overestimate, and a
    // visible one, where a zero would silently divide something later.
    if cores == 0 || cores > threads {
        if cores > threads {
            notes.push(format!(
                "Reported {cores} physical cores on {threads} threads, which cannot be; using the thread count."
            ));
        }
        cores = threads;
    }

    if model.is_empty() {
        model = "Unknown CPU".to_string();
        notes.push("The CPU model could not be read.".into());
    }

    CpuInfo {
        model,
        features,
        vendor: if vendor.is_empty() { "unknown".into() } else { vendor },
        cores,
        threads,
        architecture: std::env::consts::ARCH.to_string(),
        base_mhz,
    }
}

// ─── Memory ─────────────────────────────────────────────────────────────────

/// Memory as the kernel reports it.
///
/// `MemAvailable` rather than `MemTotal - used`: most of what Linux counts as
/// used is reclaimable page cache, and subtracting it refuses runs that would
/// have been fine. On the machine this was written on the difference is
/// 3.2 GB free against 5.9 GB available — a run sized on the first number
/// would be turned away with nearly three gigabytes going spare.
pub(crate) fn memory(notes: &mut Vec<String>) -> MemoryState {
    let empty = MemoryState {
        total_bytes: 0,
        available_bytes: 0,
        free_bytes: 0,
        reclaimable_bytes: 0,
        swap_total_bytes: 0,
        swap_used_bytes: 0,
    };

    #[cfg(target_os = "linux")]
    {
        let Ok(text) = std::fs::read_to_string("/proc/meminfo") else {
            notes.push("/proc/meminfo could not be read; memory is unknown.".into());
            return empty;
        };
        let read = |key: &str| -> u64 {
            text.lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<u64>().ok())
                .map(|kb| kb * 1024)
                .unwrap_or(0)
        };
        let total = read("MemTotal:");
        let free = read("MemFree:");
        let available = {
            let a = read("MemAvailable:");
            if a > 0 { a } else { free }
        };
        let swap_total = read("SwapTotal:");
        return MemoryState {
            total_bytes: total,
            available_bytes: available.min(total.max(available)),
            free_bytes: free,
            reclaimable_bytes: read("Cached:") + read("SReclaimable:"),
            swap_total_bytes: swap_total,
            swap_used_bytes: swap_total.saturating_sub(read("SwapFree:")),
        };
    }
    #[cfg(not(target_os = "linux"))]
    {
        notes.push("Memory detection is not implemented for this platform yet.".into());
        empty
    }
}

// ─── Disk ───────────────────────────────────────────────────────────────────

/// Bytes free where a training run would write. Checkpoints are the reason
/// this is here: a run that writes 24 × 742 MB needs to know before it starts,
/// not at step 8 000.
pub(crate) fn disk_available(notes: &mut Vec<String>) -> u64 {
    #[cfg(unix)]
    {
        let path = std::env::var("HOME").unwrap_or_else(|_| "/".to_string());
        let Ok(c_path) = std::ffi::CString::new(path.clone()) else {
            notes.push("The home path could not be checked for free space.".into());
            return 0;
        };
        // SAFETY: `statvfs` writes into a zeroed struct we own, and the path
        // is a valid NUL-terminated C string that outlives the call.
        unsafe {
            let mut stat: libc::statvfs = std::mem::zeroed();
            if libc::statvfs(c_path.as_ptr(), &mut stat) == 0 {
                // `f_bavail` is what a non-root process may use, which is the
                // honest number here; `f_bfree` includes the reserved blocks
                // only root can touch.
                return stat.f_bavail as u64 * stat.f_frsize as u64;
            }
        }
        notes.push(format!("Free space on {path} could not be read."));
    }
    #[cfg(not(unix))]
    {
        notes.push("Disk detection is not implemented for this platform yet.".into());
    }
    0
}

// ─── Identity ───────────────────────────────────────────────────────────────

pub(crate) fn os_name() -> String {
    #[cfg(target_os = "linux")]
    {
        if let Ok(text) = std::fs::read_to_string("/etc/os-release") {
            if let Some(line) = text.lines().find(|l| l.starts_with("PRETTY_NAME=")) {
                return line
                    .trim_start_matches("PRETTY_NAME=")
                    .trim_matches('"')
                    .to_string();
            }
        }
    }
    format!("{} {}", std::env::consts::OS, std::env::consts::ARCH)
}

pub(crate) fn hostname() -> Option<String> {
    std::fs::read_to_string("/etc/hostname")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

// ─── Accelerators ───────────────────────────────────────────────────────────

/// Every accelerator the machine will admit to.
///
/// Two passes, in order of how much they know:
///
///  1. `nvidia-smi`, which reports the name, both memory figures, the driver
///     and live telemetry. When it answers, nothing else is needed.
///  2. `/sys/class/drm`, which exists on any Linux with a graphics driver and
///     yields the PCI vendor — enough to say "there is an Intel graphics
///     processor here" without pretending to know its throughput.
///
/// A machine where both come back empty gets an empty vector, and that is a
/// complete answer: it means compute happens on the CPU.
pub(crate) fn gpus(notes: &mut Vec<String>) -> Vec<GpuInfo> {
    let mut found = nvidia(notes);
    if found.is_empty() {
        found.extend(drm_devices(notes));
    }
    if found.is_empty() {
        notes.push("No accelerator was detected; this machine computes on its CPU.".into());
    }
    found
}

fn nvidia(notes: &mut Vec<String>) -> Vec<GpuInfo> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.free,driver_version,temperature.gpu,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ])
        .output();

    let Ok(output) = output else {
        // Not an error worth reporting: most machines have no NVIDIA tooling,
        // and saying so on every detection would be noise.
        return Vec::new();
    };
    if !output.status.success() {
        // This one *is* worth reporting. `nvidia-smi` present but failing
        // usually means a driver that did not load — a card that exists and
        // cannot be used, which is very different from having no card.
        let err = String::from_utf8_lossy(&output.stderr);
        if err.contains("couldn't communicate") || err.contains("Driver/library") {
            notes.push(
                "nvidia-smi is installed but the NVIDIA driver did not answer, so any NVIDIA card \
                 on this machine is currently unusable."
                    .into(),
            );
        }
        return Vec::new();
    }

    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| {
            let f: Vec<&str> = line.split(',').map(str::trim).collect();
            if f.is_empty() || f[0].is_empty() {
                return None;
            }
            let mib = |i: usize| f.get(i).and_then(|v| v.parse::<u64>().ok()).map(|m| m * 1024 * 1024);
            let num = |i: usize| f.get(i).and_then(|v| v.parse::<f32>().ok());
            Some(GpuInfo {
                name: f[0].to_string(),
                vendor: "NVIDIA".into(),
                recognised: false, // resolved against the spec database by the caller
                integrated: false,
                backend: ComputeBackend::Cuda,
                vram_total_bytes: mib(1),
                vram_free_bytes: mib(2),
                driver_version: f.get(3).filter(|v| !v.is_empty()).map(|v| v.to_string()),
                temperature_c: num(4),
                utilisation_pct: num(5),
                power_watts: num(6),
            })
        })
        .collect()
}

/// PCI vendor ids, for the devices `/sys/class/drm` exposes.
fn vendor_name(id: &str) -> Option<(&'static str, ComputeBackend, bool)> {
    match id.trim() {
        "0x10de" => Some(("NVIDIA", ComputeBackend::Cuda, false)),
        "0x1002" => Some(("AMD", ComputeBackend::Rocm, false)),
        // Intel's discrete Arc parts share this vendor id with the integrated
        // graphics in every laptop chip, and nothing in sysfs distinguishes
        // them without a device-id table. Treated as integrated, which is the
        // safe direction: it means "shares system memory, do not size a run
        // against dedicated VRAM", and being wrong that way costs nothing.
        "0x8086" => Some(("Intel", ComputeBackend::Vulkan, true)),
        _ => None,
    }
}

fn drm_devices(notes: &mut Vec<String>) -> Vec<GpuInfo> {
    let Ok(entries) = std::fs::read_dir("/sys/class/drm") else {
        return Vec::new();
    };
    let mut out = Vec::new();

    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        // `card0` is a device; `card0-HDMI-A-1` is one of its connectors.
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let device = entry.path().join("device");
        let Ok(vendor_id) = std::fs::read_to_string(device.join("vendor")) else {
            continue;
        };
        let Some((vendor, backend, integrated)) = vendor_name(&vendor_id) else {
            continue;
        };

        // A readable model name needs a PCI id table, which belongs to the
        // system rather than to NEURAX. `lspci` has one; where it is missing,
        // the vendor alone is stated rather than a made-up model.
        let model = lspci_name(&device).unwrap_or_else(|| format!("{vendor} graphics"));

        out.push(GpuInfo {
            name: model,
            vendor: vendor.into(),
            recognised: false,
            integrated,
            backend,
            // An integrated part has no VRAM of its own: it borrows system
            // memory, which is already reported as RAM. Publishing a
            // dedicated-VRAM figure for it would be double counting.
            vram_total_bytes: None,
            vram_free_bytes: None,
            driver_version: None,
            temperature_c: None,
            utilisation_pct: None,
            power_watts: None,
        });
    }

    if out.iter().any(|g| g.integrated) {
        notes.push(
            "The graphics processor found is integrated: it shares system memory with the CPU, so \
             it has no VRAM budget of its own."
                .into(),
        );
    }
    out
}

/// The device's marketing name, via `lspci` when it is installed.
fn lspci_name(device_path: &std::path::Path) -> Option<String> {
    // `/sys/class/drm/card0/device` is a symlink whose final component is the
    // PCI address `lspci` indexes by.
    let addr = std::fs::canonicalize(device_path).ok()?;
    let slot = addr.file_name()?.to_string_lossy().to_string();
    // lspci wants `00:02.0`; sysfs gives `0000:00:02.0`.
    let slot = slot.splitn(2, ':').nth(1).unwrap_or(&slot).to_string();

    let output = Command::new("lspci").args(["-mm", "-s", &slot]).output().ok()?;
    if !output.status.success() {
        return None;
    }
    // `-mm` quotes each field: slot "Class" "Vendor" "Device" ...
    let line = String::from_utf8_lossy(&output.stdout);
    let fields: Vec<&str> = line.split('"').collect();
    let vendor = fields.get(3)?.trim();
    let device = fields.get(5)?.trim();
    if device.is_empty() {
        return None;
    }
    let vendor_short = vendor.split_whitespace().next().unwrap_or(vendor);
    Some(format!("{vendor_short} {device}"))
}
