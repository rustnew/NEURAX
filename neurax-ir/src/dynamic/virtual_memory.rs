//! Virtual Memory Pass
//! 
//! Models memory fragmentation and predicts savings from virtualization.
//! 
//! Metrics produced:
//! - M36: fragmentation_overhead_gb
//! - M37: fragmentation_pct
//! - M38: defrag_savings_gb
//! - M39: virtual_savings_gb
//! - M40: virtual_savings_pct
//! - M41: peak_vram_with_defrag_gb
//! - M42: peak_vram_with_virtual_gb

use serde::{Deserialize, Serialize};

use crate::memory::MemoryMetrics;
use crate::dynamic::types::CalibrationData;

/// Virtual Memory Analysis Pass
#[derive(Debug, Clone, Default)]
pub struct VirtualMemoryPass {
    calibration: CalibrationData,
}

/// Metrics from virtual memory analysis (M36-M42)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VirtualMemoryMetrics {
    /// M36 : Fragmentation overhead réelle
    pub fragmentation_overhead_gb:       f64,
    /// M37 : % de fragmentation estimé
    pub fragmentation_pct:               f64,
    /// M38 : Gain potentiel par défragmentation seule
    pub defrag_savings_gb:               f64,
    /// M39 : Gain potentiel par virtualisation complète
    pub virtual_savings_gb:              f64,
    /// M40 : % de gain par virtualisation
    pub virtual_savings_pct:             f64,
    /// M41 : Pic VRAM effectif avec défragmentation
    pub peak_vram_with_defrag_gb:        f64,
    /// M42 : Pic VRAM effectif avec virtualisation complète
    pub peak_vram_with_virtual_gb:       f64,
    /// Stratégie recommandée
    pub recommended_strategy:           AllocationStrategy,
    /// Confiance de l'estimation
    pub confidence:                      f64,
}

/// Memory allocation strategy recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationStrategy {
    NoAction,
    EnableCompaction,
    EnableFlashAttention,
    EnableVirtualMemory,
}

impl Default for AllocationStrategy {
    fn default() -> Self { Self::NoAction }
}

impl VirtualMemoryPass {
    pub fn new() -> Self { Self::default() }
    
    pub fn run(&self, mem: &MemoryMetrics) -> VirtualMemoryMetrics {
        let frag_pct = self.estimate_fragmentation_pct(mem);
        let naive_gb = mem.peak_vram_gb();
        let defrag_gb = naive_gb * (1.0 - frag_pct / 100.0 * 0.5);
        let virtual_gb = naive_gb * (1.0 - frag_pct / 100.0 * 0.75);
        
        let frag_overhead_gb = naive_gb - defrag_gb;
        let virtual_savings_gb = naive_gb - virtual_gb;
        let virtual_savings_pct = if naive_gb > 0.0 { virtual_savings_gb / naive_gb * 100.0 } else { 0.0 };
        
        let strategy = match frag_pct as u32 {
            0..=4  => AllocationStrategy::NoAction,
            5..=14 => AllocationStrategy::EnableCompaction,
            15..=29 => AllocationStrategy::EnableFlashAttention,
            _       => AllocationStrategy::EnableVirtualMemory,
        };
        
        let confidence = if self.calibration.has_fragmentation_data() { 0.85 } else { 0.65 };
        
        VirtualMemoryMetrics {
            fragmentation_overhead_gb: frag_overhead_gb,
            fragmentation_pct: frag_pct,
            defrag_savings_gb: naive_gb - defrag_gb,
            virtual_savings_gb,
            virtual_savings_pct,
            peak_vram_with_defrag_gb: defrag_gb,
            peak_vram_with_virtual_gb: virtual_gb,
            recommended_strategy: strategy,
            confidence,
        }
    }
    
    fn estimate_fragmentation_pct(&self, mem: &MemoryMetrics) -> f64 {
        let params_gb = if mem.parameter_memory_bytes > 0 { mem.parameter_memory_bytes as f64 / 1e9 } else { 0.001 };
        let activations_gb = mem.activation_memory_bytes as f64 / 1e9;
        let ratio = activations_gb / params_gb;
        let peak_gb = mem.peak_vram_gb();
        let size_factor = (peak_gb / 10.0).min(2.0);
        
        if peak_gb > 0.0 {
            let activation_ratio = if peak_gb > 0.0 { activations_gb / peak_gb } else { 0.0 };
            if let Some(calibrated) = self.calibration.get_fragmentation(peak_gb, activation_ratio) {
                return calibrated;
            }
        }
        
        let base: f64 = match (ratio as u32, size_factor as u32) {
            (0..=1, 0..=1) => 8.5,
            (0..=1, 2)     => 10.5,
            (2..=4, 0..=1) => 11.5,
            (2..=4, 2)     => 13.5,
            _              => 16.0,
        };
        base.min(40.0_f64)
    }
    
    pub fn validate(&self, metrics: &VirtualMemoryMetrics, naive_peak: f64) -> Vec<String> {
        let mut diags = vec![];
        if metrics.peak_vram_with_virtual_gb > naive_peak {
            diags.push("BUG: virtual peak > naive peak".to_string());
        }
        if !(0.0..=50.0).contains(&metrics.fragmentation_pct) {
            diags.push(format!("Fragmentation {:.1}% out of [0, 50%]", metrics.fragmentation_pct));
        }
        diags
    }
}
