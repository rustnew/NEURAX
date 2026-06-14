//! # Newtypes pour la sécurité de type
//!
//! Ces types empêchent les confusions entre différentes unités physiques.
//! Le compilateur Rust garantit qu'on ne peut pas mélanger FLOPs et Bytes.
//!
//! ## Principe
//! ```rust
//! use neurax_core::{FLOPs, Bytes};
//! let flops = FLOPs(1.6e12);
//! let bytes = Bytes(16_000_000_000);
//! // flops + bytes // ❌ Erreur de compilation !
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, Sub};

// ═══════════════════════════════════════════════════════════════════════════
// FLOPs — Opérations à virgule flottante
// ═══════════════════════════════════════════════════════════════════════════

/// Nombre d'opérations à virgule flottante (FLOPs).
/// Type distinct pour éviter la confusion avec Bytes ou Latency.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default, Serialize, Deserialize)]
#[repr(transparent)]
pub struct FLOPs(pub f64);

impl FLOPs {
    /// Crée un nouveau compteur de FLOPs.
    #[inline(always)]
    pub const fn new(value: f64) -> Self {
        Self(value)
    }

    /// Retourne la valeur brute.
    #[inline(always)]
    pub const fn get(self) -> f64 {
        self.0
    }

    /// Convertit en GFLOPs (10^9).
    #[inline(always)]
    pub fn to_gflops(self) -> f64 {
        self.0 / 1e9
    }

    /// Convertit en TFLOPs (10^12).
    #[inline(always)]
    pub fn to_tflops(self) -> f64 {
        self.0 / 1e12
    }

    /// Convertit en PFLOPs (10^15).
    #[inline(always)]
    pub fn to_pflops(self) -> f64 {
        self.0 / 1e15
    }

    /// FLOPs nuls.
    pub const ZERO: FLOPs = FLOPs(0.0);
}

impl Add for FLOPs {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign for FLOPs {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl Sub for FLOPs {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl Mul<f64> for FLOPs {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f64) -> Self::Output {
        Self(self.0 * rhs)
    }
}

impl Div<f64> for FLOPs {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f64) -> Self::Output {
        Self(self.0 / rhs)
    }
}

impl fmt::Display for FLOPs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 >= 1e15 {
            write!(f, "{:.2} PFLOPs", self.to_pflops())
        } else if self.0 >= 1e12 {
            write!(f, "{:.2} TFLOPs", self.to_tflops())
        } else if self.0 >= 1e9 {
            write!(f, "{:.2} GFLOPs", self.to_gflops())
        } else if self.0 >= 1e6 {
            write!(f, "{:.2} MFLOPs", self.0 / 1e6)
        } else if self.0 >= 1e3 {
            write!(f, "{:.2} kFLOPs", self.0 / 1e3)
        } else {
            write!(f, "{:.2} FLOPs", self.0)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Bytes — Taille mémoire
// ═══════════════════════════════════════════════════════════════════════════

/// Taille en bytes (mémoire, stockage).
/// Type distinct pour éviter la confusion avec FLOPs ou Latency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Bytes(pub u64);

impl Bytes {
    /// Crée une nouvelle taille en bytes.
    #[inline(always)]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Retourne la valeur brute.
    #[inline(always)]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Convertit en KB (10^3).
    #[inline(always)]
    pub fn to_kb(self) -> f64 {
        self.0 as f64 / 1e3
    }

    /// Convertit en MB (10^6).
    #[inline(always)]
    pub fn to_mb(self) -> f64 {
        self.0 as f64 / 1e6
    }

    /// Convertit en GB (10^9).
    #[inline(always)]
    pub fn to_gb(self) -> f64 {
        self.0 as f64 / 1e9
    }

    /// Convertit en TB (10^12).
    #[inline(always)]
    pub fn to_tb(self) -> f64 {
        self.0 as f64 / 1e12
    }

    /// Convertit en KiB (2^10).
    #[inline(always)]
    pub fn to_kib(self) -> f64 {
        self.0 as f64 / 1024.0
    }

    /// Convertit en MiB (2^20).
    #[inline(always)]
    pub fn to_mib(self) -> f64 {
        self.0 as f64 / (1024.0 * 1024.0)
    }

    /// Convertit en GiB (2^30).
    #[inline(always)]
    pub fn to_gib(self) -> f64 {
        self.0 as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Bytes nuls.
    pub const ZERO: Bytes = Bytes(0);
}

impl Add for Bytes {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign for Bytes {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl Sub for Bytes {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0.saturating_sub(rhs.0))
    }
}

impl Mul<u64> for Bytes {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: u64) -> Self::Output {
        Self(self.0 * rhs)
    }
}

impl Div<u64> for Bytes {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: u64) -> Self::Output {
        Self(self.0 / rhs)
    }
}

impl fmt::Display for Bytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 >= (1 << 40) {
            write!(f, "{:.2} TiB", self.to_gib() / 1024.0)
        } else if self.0 >= (1 << 30) {
            write!(f, "{:.2} GiB", self.to_gib())
        } else if self.0 >= (1 << 20) {
            write!(f, "{:.2} MiB", self.to_mib())
        } else if self.0 >= (1 << 10) {
            write!(f, "{:.2} KiB", self.to_kib())
        } else {
            write!(f, "{} B", self.0)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// LatencyMs — Temps en millisecondes
// ═══════════════════════════════════════════════════════════════════════════

/// Latence en millisecondes.
/// Type distinct pour éviter la confusion avec FLOPs ou Bytes.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default, Serialize, Deserialize)]
#[repr(transparent)]
pub struct LatencyMs(pub f64);

impl LatencyMs {
    /// Crée une nouvelle latence en ms.
    #[inline(always)]
    pub const fn new(value: f64) -> Self {
        Self(value)
    }

    /// Retourne la valeur brute en ms.
    #[inline(always)]
    pub const fn get(self) -> f64 {
        self.0
    }

    /// Convertit en microsecondes.
    #[inline(always)]
    pub fn to_us(self) -> f64 {
        self.0 * 1e3
    }

    /// Convertit en secondes.
    #[inline(always)]
    pub fn to_seconds(self) -> f64 {
        self.0 / 1e3
    }

    /// Convertit en minutes.
    #[inline(always)]
    pub fn to_minutes(self) -> f64 {
        self.0 / (1e3 * 60.0)
    }

    /// Latence nulle.
    pub const ZERO: LatencyMs = LatencyMs(0.0);
}

impl Add for LatencyMs {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign for LatencyMs {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl Sub for LatencyMs {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl Mul<f64> for LatencyMs {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f64) -> Self::Output {
        Self(self.0 * rhs)
    }
}

impl Div<f64> for LatencyMs {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f64) -> Self::Output {
        Self(self.0 / rhs)
    }
}

impl fmt::Display for LatencyMs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 >= 60000.0 {
            write!(f, "{:.2} min", self.to_minutes())
        } else if self.0 >= 1000.0 {
            write!(f, "{:.2} s", self.to_seconds())
        } else if self.0 >= 1.0 {
            write!(f, "{:.2} ms", self.0)
        } else if self.0 >= 0.001 {
            write!(f, "{:.2} µs", self.to_us())
        } else {
            write!(f, "{:.2} ns", self.to_us() * 1e3)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TokensPerSec — Débit de tokens
// ═══════════════════════════════════════════════════════════════════════════

/// Débit en tokens par seconde.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default, Serialize, Deserialize)]
#[repr(transparent)]
pub struct TokensPerSec(pub f64);

impl TokensPerSec {
    /// Crée un nouveau débit.
    #[inline(always)]
    pub const fn new(value: f64) -> Self {
        Self(value)
    }

    /// Retourne la valeur brute.
    #[inline(always)]
    pub const fn get(self) -> f64 {
        self.0
    }

    /// Débit nul.
    pub const ZERO: TokensPerSec = TokensPerSec(0.0);
}

impl fmt::Display for TokensPerSec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 >= 1e6 {
            write!(f, "{:.2} M tok/s", self.0 / 1e6)
        } else if self.0 >= 1e3 {
            write!(f, "{:.2} k tok/s", self.0 / 1e3)
        } else {
            write!(f, "{:.2} tok/s", self.0)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ParamCount — Nombre de paramètres
// ═══════════════════════════════════════════════════════════════════════════

/// Nombre de paramètres d'un modèle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
#[repr(transparent)]
pub struct ParamCount(pub u64);

impl ParamCount {
    /// Crée un nouveau compteur de paramètres.
    #[inline(always)]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Retourne la valeur brute.
    #[inline(always)]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Convertit en millions (M).
    #[inline(always)]
    pub fn to_millions(self) -> f64 {
        self.0 as f64 / 1e6
    }

    /// Convertit en milliards (B).
    #[inline(always)]
    pub fn to_billions(self) -> f64 {
        self.0 as f64 / 1e9
    }

    /// Compteur nul.
    pub const ZERO: ParamCount = ParamCount(0);
}

impl Add for ParamCount {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign for ParamCount {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl fmt::Display for ParamCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 >= 1_000_000_000 {
            write!(f, "{:.2}B", self.to_billions())
        } else if self.0 >= 1_000_000 {
            write!(f, "{:.2}M", self.to_millions())
        } else if self.0 >= 1_000 {
            write!(f, "{:.2}K", self.0 as f64 / 1e3)
        } else {
            write!(f, "{}", self.0)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flops_display() {
        assert_eq!(format!("{}", FLOPs(1.6e12)), "1.60 TFLOPs");
        assert_eq!(format!("{}", FLOPs(1.6e9)), "1.60 GFLOPs");
        assert_eq!(format!("{}", FLOPs(1.6e6)), "1.60 MFLOPs");
    }

    #[test]
    fn test_bytes_display() {
        assert_eq!(format!("{}", Bytes(1_073_741_824)), "1.00 GiB");
        assert_eq!(format!("{}", Bytes(1_048_576)), "1.00 MiB");
        assert_eq!(format!("{}", Bytes(1024)), "1.00 KiB");
    }

    #[test]
    fn test_latency_display() {
        assert_eq!(format!("{}", LatencyMs(1500.0)), "1.50 s");
        assert_eq!(format!("{}", LatencyMs(50.0)), "50.00 ms");
        assert_eq!(format!("{}", LatencyMs(0.5)), "500.00 µs");
    }

    #[test]
    fn test_param_count_display() {
        assert_eq!(format!("{}", ParamCount(7_000_000_000)), "7.00B");
        assert_eq!(format!("{}", ParamCount(125_000_000)), "125.00M");
    }

    #[test]
    fn test_type_safety() {
        // Ces lignes ne compilent pas - c'est le but !
        // let flops = FLOPs(1e9);
        // let bytes = Bytes(1_000_000);
        // let _ = flops + bytes; // Error: mismatched types
    }

    #[test]
    fn test_arithmetic() {
        let mut flops = FLOPs(1e9);
        flops += FLOPs(1e9);
        assert_eq!(flops, FLOPs(2e9));

        let bytes = Bytes(1024) + Bytes(1024);
        assert_eq!(bytes, Bytes(2048));
    }
}
