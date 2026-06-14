//! Memory metrics utilities

use super::{MemoryMetrics, OomRisk};

/// Format memory size for display
pub fn format_memory(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

/// Get memory breakdown as percentages
pub fn memory_breakdown(metrics: &MemoryMetrics) -> MemoryBreakdown {
    let total = metrics.peak_vram_bytes as f64;
    
    MemoryBreakdown {
        params_pct: if total > 0.0 { metrics.parameter_memory_bytes as f64 / total * 100.0 } else { 0.0 },
        activations_pct: if total > 0.0 { metrics.activation_memory_bytes as f64 / total * 100.0 } else { 0.0 },
        gradients_pct: if total > 0.0 { metrics.gradient_memory_bytes as f64 / total * 100.0 } else { 0.0 },
        optimizer_pct: if total > 0.0 { metrics.optimizer_state_bytes as f64 / total * 100.0 } else { 0.0 },
    }
}

#[derive(Debug, Clone)]
pub struct MemoryBreakdown {
    pub params_pct: f64,
    pub activations_pct: f64,
    pub gradients_pct: f64,
    pub optimizer_pct: f64,
}

/// Get OOM risk description
pub fn oom_risk_description(risk: OomRisk) -> &'static str {
    match risk {
        OomRisk::Safe => "Memory usage is safe (<80% of GPU VRAM)",
        OomRisk::Warning => "Memory usage is high (80-95% of GPU VRAM). Consider gradient checkpointing.",
        OomRisk::Critical => "Memory usage is critical (95-100% of GPU VRAM). OOM likely.",
        OomRisk::Overflow => "Memory exceeds GPU VRAM. Model will not fit.",
    }
}
