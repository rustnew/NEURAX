//! Dynamic Feedback Module

use crate::memory::MemoryMetrics;
use crate::{Diagnostic, DiagnosticCode, Severity, DiagnosticCategory};
use super::types::DynamicResults;
use super::virtual_memory::AllocationStrategy;

/// Apply dynamic feedback to memory metrics
pub fn apply_dynamic_feedback(
    mem: &mut MemoryMetrics,
    dynamic: &DynamicResults,
) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    
    // AMPLIFICATION 1 : Couche instable → recalcul en fp32
    if let Some(ref sta) = dynamic.stability {
        for (layer_id, &margin) in &sta.stability_margin_by_layer {
            if margin < 0.2 {
                diagnostics.push(Diagnostic {
                    severity: Severity::Warning,
                    category: DiagnosticCategory::ArchitectureInefficiency,
                    code: DiagnosticCode::W005,
                    message: format!(
                        "Layer '{}' is unstable (margin={:.2}). Memory recalculated assuming fp32.",
                        layer_id, margin
                    ),
                    layer_id: Some(layer_id.clone()),
                    suggestion: Some("Consider using fp32 for this layer.".to_string()),
                    precision_impact: 0.3,
                });
            }
        }
        
        // Apply fp32 fallback overhead
        if sta.fp32_fallback_memory_overhead_gb > 0.0 {
            mem.peak_vram_bytes += (sta.fp32_fallback_memory_overhead_gb * 1e9) as u64;
        }
    }
    
    // AMPLIFICATION 2 : Déséquilibre MoE → extension liveness
    if let Some(ref bps) = dynamic.behavioral {
        if bps.expert_load_imbalance > 0.5 {
            let extension_factor = 1.0 + bps.expert_load_imbalance * 0.3;
            let new_activation = (mem.activation_memory_bytes as f64 * extension_factor) as u64;
            mem.activation_memory_bytes = new_activation;
            
            diagnostics.push(Diagnostic {
                severity: Severity::Warning,
                category: DiagnosticCategory::ArchitectureInefficiency,
                code: DiagnosticCode::W006,
                message: format!(
                    "MoE load imbalance = {:.0}%. Activation memory increased by {:.0}%.",
                    bps.expert_load_imbalance * 100.0,
                    (extension_factor - 1.0) * 100.0
                ),
                layer_id: None,
                suggestion: Some("Consider adjusting MoE routing.".to_string()),
                precision_impact: 0.2,
            });
        }
    }
    
    // AMPLIFICATION 3 : Fragmentation élevée → recommandation Flash Attention
    if let Some(ref vm) = dynamic.virtual_memory {
        if vm.fragmentation_pct > 20.0 {
            diagnostics.push(Diagnostic {
                severity: Severity::Hint,
                category: DiagnosticCategory::ArchitectureInefficiency,
                code: DiagnosticCode::H002,
                message: format!(
                    "Memory fragmentation = {:.1}%. Enabling Flash Attention reduces fragmentation.",
                    vm.fragmentation_pct
                ),
                layer_id: None,
                suggestion: Some("Enable flash_attention in training config.".to_string()),
                precision_impact: 0.0,
            });
        }
        
        match vm.recommended_strategy {
            AllocationStrategy::EnableCompaction => {
                diagnostics.push(Diagnostic {
                    severity: Severity::Hint,
                    category: DiagnosticCategory::Configuration,
                    code: DiagnosticCode::H002,
                    message: "Consider enabling PyTorch memory compaction.".to_string(),
                    layer_id: None,
                    suggestion: None,
                    precision_impact: 0.0,
                });
            }
            AllocationStrategy::EnableFlashAttention => {
                diagnostics.push(Diagnostic {
                    severity: Severity::Hint,
                    category: DiagnosticCategory::Configuration,
                    code: DiagnosticCode::H002,
                    message: "Consider enabling Flash Attention.".to_string(),
                    layer_id: None,
                    suggestion: None,
                    precision_impact: 0.0,
                });
            }
            AllocationStrategy::EnableVirtualMemory => {
                diagnostics.push(Diagnostic {
                    severity: Severity::Hint,
                    category: DiagnosticCategory::Configuration,
                    code: DiagnosticCode::H002,
                    message: "High fragmentation. Consider virtual memory management.".to_string(),
                    layer_id: None,
                    suggestion: None,
                    precision_impact: 0.0,
                });
            }
            AllocationStrategy::NoAction => {}
        }
    }
    
    diagnostics
}

/// Calculate dynamic confidence score
pub fn calculate_dynamic_confidence(dynamic: &DynamicResults) -> f64 {
    let mut confidence = 1.0;
    let mut count = 0;
    
    if let Some(ref vm) = dynamic.virtual_memory {
        confidence *= vm.confidence;
        count += 1;
    }
    
    if let Some(ref sta) = dynamic.stability {
        confidence *= sta.confidence;
        count += 1;
    }
    
    if let Some(ref bps) = dynamic.behavioral {
        confidence *= bps.prediction_confidence;
        count += 1;
    }
    
    if count > 0 { confidence.powf(1.0 / count as f64) } else { 1.0 }
}
