//! What this machine can actually sustain.
//!
//! The hardware database holds published peaks for 38 accelerators. Two things
//! are wrong with relying on it alone. The first is coverage: the machine this
//! was written on has an integrated Intel graphics processor and a laptop
//! CPU, and neither has an entry — so every latency figure would have been
//! unavailable. The second is that published peaks are ceilings no real kernel
//! reaches; dividing by them makes every prediction optimistic by a factor
//! that varies per machine.
//!
//! So the constants are measured here. Two of them, because they are the two
//! the analytical model divides by:
//!
//!  - **Sustained FLOP/s**, from a blocked single-precision matrix multiply.
//!    A GEMM is the right probe because it is what the model is mostly made
//!    of — attention and feed-forward layers are matrix multiplies — and
//!    because it is compute-bound, so it measures arithmetic rather than
//!    memory.
//!  - **Memory bandwidth**, from a STREAM-style triad (`a = b + k·c`). Chosen
//!    over a plain copy because the triad is the standard, is read-two-write-
//!    one like a real kernel, and cannot be optimised into a `memcpy`.
//!
//! ## Keeping the measurement honest
//!
//! A benchmark is easy to make lie. Three things guard against it:
//!
//!  - **The compiler must not delete the work.** Both kernels consume their
//!    result into a checksum that is returned, so nothing can be optimised
//!    away as dead.
//!  - **The buffer must not fit in cache.** The triad allocates well past any
//!    last-level cache, or it measures the cache and reports a number several
//!    times too high.
//!  - **A short run must not be reported as precise.** Each kernel is run
//!    repeatedly and the *best* iteration is taken, because the fastest run is
//!    the one least disturbed by everything else on the machine — and the
//!    elapsed time is reported so a suspiciously quick measurement is visible.
//!
//! This does not run a model. It calibrates the machine constant the formulas
//! use, which is the same role the database plays for a recognised card.

use std::time::Instant;

use serde::{Deserialize, Serialize};

/// What a measurement found. Carries how it was obtained, so a consumer can
/// tell a measured number from a published one.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ComputeMeasurement {
    /// What was measured — the CPU model, since that is what these kernels
    /// ran on. A GPU measurement needs a compute API and is not implemented.
    pub device: String,
    /// Sustained single-precision throughput, in GFLOP/s.
    pub gflops_f32: f64,
    /// Sustained memory bandwidth, in GB/s.
    pub memory_bandwidth_gbs: f64,
    /// Threads the measurement used.
    pub threads: usize,
    pub measured_at: String,
    /// How long the whole measurement took, so a result that was too quick to
    /// be meaningful can be seen for what it is.
    pub duration_ms: u64,
    pub method: String,
}

/// Matrix side length. 1024³ is ~2.1 GFLOP per pass — long enough that a
/// single run is tens of milliseconds even on a fast machine, so the timer
/// resolution and scheduler noise stop mattering, and small enough that the
/// three matrices together are 12 MB.
const N: usize = 1024;
/// Triad buffer length, per array. 8M floats × 3 arrays = 96 MB, comfortably
/// past any last-level cache.
const STREAM_LEN: usize = 8 << 20;
const GEMM_REPEATS: usize = 3;
const STREAM_REPEATS: usize = 5;

/// Where a measurement is remembered between runs.
///
/// Keyed by nothing in the filename — the CPU model is stored *inside* and
/// checked on read, so a cache written on one machine and copied to another
/// is detected and ignored rather than silently believed.
fn cache_path() -> Option<std::path::PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))?;
    Some(base.join("neurax").join("machine-compute.json"))
}

/// Measure once, then remember.
///
/// The measurement costs about half a second. That is nothing once and
/// unacceptable on every page load, so the studio calls this: it returns the
/// stored figure when it was taken on the same CPU, and measures otherwise.
///
/// A cache is a place for a stale answer to hide, so two things invalidate it:
/// a different CPU model, and a different measurement method. The method
/// string is part of the stored record precisely so that improving the kernel
/// — which happened three times while writing it — does not leave every
/// existing user on the old, lower number forever.
pub fn measure_cached(device: &str) -> ComputeMeasurement {
    let path = cache_path();

    if let Some(path) = path.as_ref() {
        if let Ok(text) = std::fs::read_to_string(path) {
            if let Ok(stored) = serde_json::from_str::<ComputeMeasurement>(&text) {
                if stored.device == device && stored.method == method_string(num_cpus::get().max(1)) {
                    return stored;
                }
            }
        }
    }

    let fresh = measure_compute(device);
    if let Some(path) = path {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(text) = serde_json::to_string_pretty(&fresh) {
            // A cache that cannot be written is not an error worth failing
            // for — the measurement is already in hand.
            let _ = std::fs::write(&path, text);
        }
    }
    fresh
}

/// The exact description of how a measurement was taken. Part of the cached
/// record so a changed kernel invalidates it.
fn method_string(threads: usize) -> String {
    format!(
        "register-blocked f32 GEMM {N}x{N}x{N} × {GEMM_REPEATS} ({}), STREAM triad {} MB × {STREAM_REPEATS}, best of each, {threads} threads",
        isa_name(),
        STREAM_LEN * 3 * 4 / (1024 * 1024)
    )
}

/// Measure this machine. Takes roughly half a second and touches only memory.
pub fn measure_compute(device: &str) -> ComputeMeasurement {
    let started = Instant::now();
    let threads = num_cpus::get().max(1);

    let (gflops, _checksum_a) = measure_gemm(threads);
    let (bandwidth, _checksum_b) = measure_bandwidth(threads);

    ComputeMeasurement {
        device: device.to_string(),
        gflops_f32: gflops,
        memory_bandwidth_gbs: bandwidth,
        threads,
        measured_at: chrono::Utc::now().to_rfc3339(),
        duration_ms: started.elapsed().as_millis() as u64,
        method: method_string(threads),
    }
}

/// Blocked f32 GEMM. Returns `(GFLOP/s, checksum)`; the checksum exists so the
/// work cannot be eliminated as dead.
fn measure_gemm(threads: usize) -> (f64, f32) {
    // Deterministic, non-trivial inputs. Zeros would be as valid arithmetically
    // and would let a sufficiently clever compiler skip the multiply.
    let a: Vec<f32> = (0..N * N).map(|i| ((i % 97) as f32) * 0.01 + 0.5).collect();
    let b: Vec<f32> = (0..N * N).map(|i| ((i % 89) as f32) * 0.013 + 0.25).collect();

    let mut best = f64::MAX;
    let mut checksum = 0.0f32;

    for _ in 0..GEMM_REPEATS {
        let start = Instant::now();
        let c = gemm(&a, &b, threads);
        let elapsed = start.elapsed().as_secs_f64();
        // Every run's result is folded in, so no iteration is removable.
        checksum += c[0] + c[c.len() - 1];
        if elapsed < best {
            best = elapsed;
        }
    }

    // 2 FLOP per multiply-accumulate.
    let flops = 2.0 * (N as f64).powi(3);
    ((flops / best) / 1e9, checksum)
}

/// The GEMM kernel.
///
/// A straightforward triple loop measures the loop, not the machine. The first
/// version of this reported 16.8 GFLOP/s on a chip whose arithmetic units can
/// issue an order of magnitude more — which would have made every predicted
/// latency roughly ten times too slow, because the real run uses a library
/// (oneDNN, MKL) that does reach a large fraction of peak.
///
/// Two changes close most of that gap without hand-written intrinsics:
///
///  - **Register blocking.** Four rows of `C` are accumulated at once against
///    one row of `B`. A single accumulator chain stalls on FMA latency —
///    four independent chains keep the pipeline fed, which is the same reason
///    every real GEMM micro-kernel accumulates a tile rather than a scalar.
///  - **A contiguous inner loop over `j`.** Both `B` and `C` are walked
///    along their rows, which is what lets the compiler auto-vectorise; a
///    loop striding down a column measures cache misses instead.
///
/// What this still is not is a tuned BLAS, and the figure it produces should
/// be read as "what a good ordinary kernel sustains here", which is the
/// honest constant to divide by — closer to a real run than the vendor's peak,
/// and not a claim to have reimplemented MKL.
fn gemm(a: &[f32], b: &[f32], threads: usize) -> Vec<f32> {
    /// Rows of `C` in a tile — four independent FMA chains.
    const MR: usize = 4;
    /// Columns of `C` in a tile. Sixteen floats is two AVX2 registers per row,
    /// so the whole 4×16 accumulator lives in eight registers and never
    /// touches memory during the k loop.
    const NR: usize = 16;
    /// Depth tile, so the panel of `B` being reused stays in L2.
    const KC: usize = 256;

    let mut c = vec![0.0f32; N * N];
    let blocks = N.div_ceil(MR);
    let blocks_per_thread = blocks.div_ceil(threads);
    let rows_per_thread = blocks_per_thread * MR;

    std::thread::scope(|scope| {
        for (t, chunk) in c.chunks_mut(rows_per_thread * N).enumerate() {
            let row0 = t * rows_per_thread;
            scope.spawn(move || {
                let rows = chunk.len() / N;
                for kk in (0..N).step_by(KC) {
                    let k_end = (kk + KC).min(N);
                    for jj in (0..N).step_by(NR) {
                        let mut i = 0;
                        while i + MR <= rows {
                            micro_kernel::<MR, NR>(a, b, chunk, row0, i, jj, kk, k_end);
                            i += MR;
                        }
                        // The remainder, when the row count is not a multiple
                        // of MR. At most three rows, so correctness matters
                        // and speed does not.
                        while i < rows {
                            for k in kk..k_end {
                                let aik = a[(row0 + i) * N + k];
                                let b_row = &b[k * N + jj..k * N + jj + NR];
                                let c_row = &mut chunk[i * N + jj..i * N + jj + NR];
                                for j in 0..NR {
                                    c_row[j] += aik * b_row[j];
                                }
                            }
                            i += 1;
                        }
                    }
                }
            });
        }
    });
    c
}

/// The innermost loop, dispatched on what the CPU can actually issue.
///
/// This is where the measurement was going wrong. Rust compiles for the
/// baseline x86-64 target unless told otherwise, which is SSE2: four floats
/// per instruction, no fused multiply-add. The chip this was written on
/// supports AVX2 and FMA — eight floats, multiply and add in one — so the
/// kernel was leaving a factor of about eight on the table, and reporting the
/// result as the machine's capacity. It measured what the *compiler default*
/// allows, not what the hardware can do.
///
/// The fix is runtime dispatch rather than a build flag. `-C
/// target-cpu=native` would produce a binary that crashes on any older
/// machine, which is unacceptable for something shipped; `is_x86_feature_
/// detected!` asks the CPU in front of us and picks the widest kernel it
/// supports. Neither variant contains an intrinsic — `#[target_feature]` is
/// enough for the compiler to vectorise the same loop with the wider
/// instruction set.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn micro_kernel_inner<const MR: usize, const NR: usize>(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    row0: usize,
    i: usize,
    jj: usize,
    kk: usize,
    k_end: usize,
) {
    // The whole tile is a local array, so it stays in registers across the
    // entire k loop instead of being re-read from `C` on every step. This is
    // the difference between 0.4 and 2 FLOP per byte moved, and it is the
    // reason a real GEMM has a micro-kernel at all.
    let mut acc = [[0.0f32; NR]; MR];
    for k in kk..k_end {
        let b_tile = &b[k * N + jj..k * N + jj + NR];
        for (r, acc_row) in acc.iter_mut().enumerate() {
            let ar = a[(row0 + i + r) * N + k];
            for j in 0..NR {
                acc_row[j] += ar * b_tile[j];
            }
        }
    }
    for (r, acc_row) in acc.iter().enumerate() {
        let c_row = &mut c[(i + r) * N + jj..(i + r) * N + jj + NR];
        for j in 0..NR {
            c_row[j] += acc_row[j];
        }
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn micro_kernel_avx2<const MR: usize, const NR: usize>(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    row0: usize,
    i: usize,
    jj: usize,
    kk: usize,
    k_end: usize,
) {
    micro_kernel_inner::<MR, NR>(a, b, c, row0, i, jj, kk, k_end);
}

#[allow(clippy::too_many_arguments)]
fn micro_kernel<const MR: usize, const NR: usize>(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    row0: usize,
    i: usize,
    jj: usize,
    kk: usize,
    k_end: usize,
) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: guarded by the runtime check immediately above.
            unsafe { micro_kernel_avx2::<MR, NR>(a, b, c, row0, i, jj, kk, k_end) };
            return;
        }
    }
    micro_kernel_inner::<MR, NR>(a, b, c, row0, i, jj, kk, k_end);
}

/// The widest instruction set this CPU actually supports, for the method
/// string — so a low figure can be told apart from a low figure *on a
/// machine that only has SSE2*.
fn isa_name() -> &'static str {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx512f") {
            return "avx512f";
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return "avx2+fma";
        }
        if is_x86_feature_detected!("avx") {
            return "avx";
        }
        return "sse2";
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        "baseline"
    }
}

/// STREAM triad: `a[i] = b[i] + k * c[i]`. Returns `(GB/s, checksum)`.
fn measure_bandwidth(threads: usize) -> (f64, f32) {
    let mut a = vec![0.0f32; STREAM_LEN];
    let b: Vec<f32> = (0..STREAM_LEN).map(|i| (i % 31) as f32).collect();
    let c: Vec<f32> = (0..STREAM_LEN).map(|i| (i % 17) as f32).collect();

    let mut best = f64::MAX;
    let mut checksum = 0.0f32;

    for r in 0..STREAM_REPEATS {
        // A different scalar each pass, so no iteration is a repeat the
        // compiler could hoist.
        let k = 1.0 + r as f32 * 0.5;
        let start = Instant::now();
        let chunk = STREAM_LEN.div_ceil(threads);
        std::thread::scope(|scope| {
            for ((ac, bc), cc) in a.chunks_mut(chunk).zip(b.chunks(chunk)).zip(c.chunks(chunk)) {
                scope.spawn(move || {
                    for i in 0..ac.len() {
                        ac[i] = bc[i] + k * cc[i];
                    }
                });
            }
        });
        let elapsed = start.elapsed().as_secs_f64();
        checksum += a[0] + a[STREAM_LEN - 1];
        if elapsed < best {
            best = elapsed;
        }
    }

    // Three arrays touched per element: two read, one written.
    let bytes = (STREAM_LEN * 3 * std::mem::size_of::<f32>()) as f64;
    ((bytes / best) / 1e9, checksum)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The measurement has to be plausible on any machine CI runs on, so the
    /// bounds are wide — what is asserted is that it produced a real,
    /// finite, positive figure rather than a zero or an infinity from a
    /// divide by a zero-length interval.
    #[test]
    fn measures_something_finite_and_positive() {
        let m = measure_compute("test");
        assert!(m.gflops_f32.is_finite() && m.gflops_f32 > 0.0, "GFLOP/s was {}", m.gflops_f32);
        assert!(
            m.memory_bandwidth_gbs.is_finite() && m.memory_bandwidth_gbs > 0.0,
            "bandwidth was {}",
            m.memory_bandwidth_gbs
        );
        assert!(m.threads >= 1);
        assert!(!m.method.is_empty());
    }

    /// A CPU that appeared to sustain more than a datacenter GPU would mean
    /// the timing loop, not the hardware, is being measured — the classic
    /// symptom of a kernel optimised away or a buffer that fits in cache.
    #[test]
    fn does_not_report_impossible_throughput() {
        let m = measure_compute("test");
        assert!(
            m.gflops_f32 < 100_000.0,
            "{} GFLOP/s from a CPU GEMM means the work was elided, not that the machine is fast",
            m.gflops_f32
        );
        assert!(
            m.memory_bandwidth_gbs < 10_000.0,
            "{} GB/s means the triad fitted in cache",
            m.memory_bandwidth_gbs
        );
    }

    #[test]
    fn gemm_computes_the_right_product() {
        // Correctness of the kernel is what makes its timing meaningful: a
        // GEMM that skips work would be fast and wrong.
        let a: Vec<f32> = (0..N * N).map(|i| ((i % 97) as f32) * 0.01 + 0.5).collect();
        let b: Vec<f32> = (0..N * N).map(|i| ((i % 89) as f32) * 0.013 + 0.25).collect();
        let c = gemm(&a, &b, 4);

        // One element, computed independently.
        let (i, j) = (3usize, 7usize);
        let expected: f32 = (0..N).map(|k| a[i * N + k] * b[k * N + j]).sum();
        let got = c[i * N + j];
        assert!(
            (got - expected).abs() / expected.abs() < 1e-4,
            "blocked GEMM disagreed with the direct sum: {got} vs {expected}"
        );
    }
}
