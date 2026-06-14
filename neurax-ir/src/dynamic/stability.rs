//! Stability Analysis Pass
//! 
//! Predicts training stability via Lyapunov exponents and chaos analysis.
//! 
//! Metrics produced:
//! - M43: stability_margin_by_layer
//! - M44: lyapunov_exponent_mean
//! - M45: chaos_index
//! - M46: high_risk_layers
//! - M47: fp32_required_pct
//! - M48: global_robustness_score
//! - M49: fp32_fallback_memory_overhead_gb

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::graph::{GraphIR, GraphNode};
use crate::memory::MemoryMetrics;
use neurax_parser::LayerType;

/// Stability Analysis Pass
#[derive(Debug, Clone, Default)]
pub struct StabilityAnalysisPass;

/// Metrics from stability analysis (M43-M49)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StabilityMetrics {
    /// M43 : Marge de stabilité par couche (layer_id → [0,1])
    pub stability_margin_by_layer:   HashMap<String, f64>,
    /// M44 : Exposant de Lyapunov moyen (> 0 = tendance au chaos)
    pub lyapunov_exponent_mean:      f64,
    /// M45 : Indice de chaos global [0,1]
    pub chaos_index:                  f64,
    /// M46 : Couches à risque (margin < 0.2)
    pub high_risk_layers:            Vec<String>,
    /// M47 : % de couches nécessitant fp32
    pub fp32_required_pct:           f64,
    /// M48 : Score de robustesse global [0,1]
    pub global_robustness_score:     f64,
    /// M49 : Mémoire supplémentaire si fp32 forcé
    pub fp32_fallback_memory_overhead_gb: f64,
    /// Confiance de l'analyse
    pub confidence:                   f64,
}

impl StabilityAnalysisPass {
    pub fn new() -> Self { Self::default() }
    
    pub fn run(&self, graph: &GraphIR, mem: &MemoryMetrics) -> StabilityMetrics {
        let mut margins: HashMap<String, f64> = HashMap::new();
        let mut lyapunovs: Vec<f64> = Vec::new();
        let mut high_risk: Vec<String> = Vec::new();
        let mut fp32_count: u32 = 0;
        let depth = graph.metrics.graph_depth;
        
        for node_idx in &graph.topo_order {
            let node = &graph.dag[*node_idx];
            let layer_id = node.layer_id.clone();
            
            let lipschitz = self.estimate_lipschitz(node, depth);
            let epsilon_init = 1e-5f64;
            let epsilon_after = epsilon_init * lipschitz;
            
            let lyapunov = if epsilon_after > 0.0 {
                (epsilon_after / epsilon_init).ln()
            } else {
                0.0
            };
            lyapunovs.push(lyapunov);
            
            let margin = 1.0 / (1.0 + lyapunov.max(0.0));
            margins.insert(layer_id.clone(), margin);
            
            if margin < 0.2 {
                high_risk.push(layer_id.clone());
                fp32_count += 1;
            }
        }
        
        let n = lyapunovs.len() as f64;
        let lyap_mean = if n > 0.0 { lyapunovs.iter().sum::<f64>() / n } else { 0.0 };
        let chaos_index = 1.0 / (1.0 + (-lyap_mean).exp());
        let robustness = if n > 0.0 { margins.values().sum::<f64>() / n } else { 1.0 };
        let fp32_pct = if n > 0.0 { fp32_count as f64 / n * 100.0 } else { 0.0 };
        
        let params_gb = mem.parameter_memory_bytes as f64 / 1e9;
        let fp32_overhead = if fp32_count > 0 && n > 0.0 && params_gb > 0.0 {
            let params_per_layer = params_gb / n;
            params_per_layer * fp32_count as f64
        } else { 0.0 };
        
        StabilityMetrics {
            stability_margin_by_layer: margins,
            lyapunov_exponent_mean: lyap_mean,
            chaos_index,
            high_risk_layers: high_risk,
            fp32_required_pct: fp32_pct,
            global_robustness_score: robustness,
            fp32_fallback_memory_overhead_gb: fp32_overhead,
            confidence: 0.70,
        }
    }
    
    fn estimate_lipschitz(&self, node: &GraphNode, graph_depth: usize) -> f64 {
        let base = match &node.layer_type {
            LayerType::Attention => {
                let seq = 2048.0; // default seq_len
                (seq / 512.0_f64).sqrt() * 2.0
            }
            LayerType::Mlp => 1.5,
            LayerType::Normalization => 0.8,
            LayerType::Conv => 1.8,
            LayerType::MoE => {
                let experts = 8.0;
                let top_k = 2.0;
                1.5 + (1.0 - top_k / experts) * 1.5
            }
            LayerType::Embedding => 0.3,
            LayerType::MambaBlock => {
                let d_state = 16.0;
                1.0 + (d_state / 64.0_f64).sqrt() * 0.5
            }
            _ => 1.0,
        };
        
        let depth_factor = if graph_depth > 0 {
            1.0 + (node.param_count as f64 / 1e9 / graph_depth as f64).min(1.0) * 0.3
        } else { 1.0 };
        
        (base * depth_factor).min(10.0)
    }
    
    pub fn validate(&self, metrics: &StabilityMetrics) -> Vec<String> {
        let mut diags = vec![];
        if !(0.0..=1.0).contains(&metrics.chaos_index) {
            diags.push("chaos_index hors [0,1]".to_string());
        }
        for (id, &m) in &metrics.stability_margin_by_layer {
            if !(0.0..=1.0).contains(&m) {
                diags.push(format!("stability_margin hors [0,1] pour couche '{}'", id));
            }
        }
        diags
    }
}
