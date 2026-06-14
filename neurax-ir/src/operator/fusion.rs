//! Fusion detection for operation merging per tuning.md §13
//! 
//! Detects patterns of operations that can be fused together
//! for improved memory access and latency

use serde::Serialize;

/// Fusion pattern type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum FusionPattern {
    /// Linear + Activation (e.g., Linear + ReLU)
    LinearActivation,
    /// Conv + BatchNorm + ReLU
    ConvBnActivation,
    /// Attention + Residual
    AttentionResidual,
    /// MLP block (Linear + Activation + Linear)
    MlpBlock,
    /// LayerNorm + Linear
    LayerNormLinear,
    /// Embedding + LayerNorm
    EmbeddingLayerNorm,
    /// Softmax + Dropout
    SoftmaxDropout,
    /// Multiple elementwise ops
    ElementwiseChain,
}

impl FusionPattern {
    /// Get memory bytes reduction from fusion
    pub fn bytes_reduction_ratio(&self) -> f64 {
        match self {
            // Writing intermediate output eliminated
            Self::LinearActivation => 0.5,
            Self::ConvBnActivation => 0.33,
            Self::AttentionResidual => 0.5,
            Self::MlpBlock => 0.33,
            Self::LayerNormLinear => 0.5,
            Self::EmbeddingLayerNorm => 0.5,
            Self::SoftmaxDropout => 0.5,
            Self::ElementwiseChain => 0.5 * self.elementwise_chain_length() as f64,
        }
    }
    
    /// Get latency speedup from fusion
    pub fn latency_speedup(&self) -> f64 {
        match self {
            // Kernel launch overhead eliminated + better cache utilization
            Self::LinearActivation => 1.15,
            Self::ConvBnActivation => 1.25,
            Self::AttentionResidual => 1.10,
            Self::MlpBlock => 1.20,
            Self::LayerNormLinear => 1.12,
            Self::EmbeddingLayerNorm => 1.10,
            Self::SoftmaxDropout => 1.08,
            Self::ElementwiseChain => 1.05 * self.elementwise_chain_length() as f64,
        }
    }
    
    /// FLOPs remain unchanged by fusion
    pub fn flops_unchanged(&self) -> bool {
        true
    }
    
    fn elementwise_chain_length(&self) -> usize {
        // Default chain length for elementwise
        2
    }
}

/// Detected fusion opportunity
#[derive(Debug, Clone, Serialize)]
pub struct FusionOpportunity {
    /// Pattern type
    pub pattern: FusionPattern,
    /// Layer IDs involved
    pub layer_ids: Vec<String>,
    /// Memory bytes saved
    pub bytes_saved: u64,
    /// Latency improvement factor
    pub latency_factor: f64,
    /// Confidence in detection (0.0-1.0)
    pub confidence: f64,
}

/// Fusion detector for operation patterns
pub struct FusionDetector {
    /// Minimum confidence threshold for reporting
    min_confidence: f64,
}

impl Default for FusionDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl FusionDetector {
    pub fn new() -> Self {
        Self { min_confidence: 0.70 }
    }
    
    /// Detect fusion opportunities in a layer sequence
    pub fn detect(&self, layers: &[LayerInfo]) -> Vec<FusionOpportunity> {
        let mut opportunities = Vec::new();
        
        if layers.len() < 2 {
            return opportunities;
        }
        
        // Detect Linear + Activation
        for i in 0..layers.len().saturating_sub(1) {
            if let Some(opp) = self.detect_linear_activation(&layers[i], &layers[i + 1]) {
                if opp.confidence >= self.min_confidence {
                    opportunities.push(opp);
                }
            }
        }
        
        // Detect Conv + BN + Activation
        for i in 0..layers.len().saturating_sub(2) {
            if let Some(opp) = self.detect_conv_bn_activation(&layers[i], &layers[i + 1], &layers[i + 2]) {
                if opp.confidence >= self.min_confidence {
                    opportunities.push(opp);
                }
            }
        }
        
        // Detect MLP block (Linear + Act + Linear)
        for i in 0..layers.len().saturating_sub(2) {
            if let Some(opp) = self.detect_mlp_block(&layers[i], &layers[i + 1], &layers[i + 2]) {
                if opp.confidence >= self.min_confidence {
                    opportunities.push(opp);
                }
            }
        }
        
        // Detect Attention + Residual
        for i in 0..layers.len().saturating_sub(1) {
            if let Some(opp) = self.detect_attention_residual(&layers[i], &layers[i + 1]) {
                if opp.confidence >= self.min_confidence {
                    opportunities.push(opp);
                }
            }
        }
        
        // Detect LayerNorm + Linear
        for i in 0..layers.len().saturating_sub(1) {
            if let Some(opp) = self.detect_layernorm_linear(&layers[i], &layers[i + 1]) {
                if opp.confidence >= self.min_confidence {
                    opportunities.push(opp);
                }
            }
        }
        
        opportunities
    }
    
    fn detect_linear_activation(&self, l1: &LayerInfo, l2: &LayerInfo) -> Option<FusionOpportunity> {
        if l1.layer_type == "Linear" && is_activation(&l2.layer_type) {
            let bytes_saved = (l1.output_bytes as f64 * FusionPattern::LinearActivation.bytes_reduction_ratio()) as u64;
            Some(FusionOpportunity {
                pattern: FusionPattern::LinearActivation,
                layer_ids: vec![l1.id.clone(), l2.id.clone()],
                bytes_saved,
                latency_factor: FusionPattern::LinearActivation.latency_speedup(),
                confidence: 0.95,
            })
        } else {
            None
        }
    }
    
    fn detect_conv_bn_activation(&self, l1: &LayerInfo, l2: &LayerInfo, l3: &LayerInfo) -> Option<FusionOpportunity> {
        if l1.layer_type == "Conv2d" && l2.layer_type == "BatchNorm" && is_activation(&l3.layer_type) {
            let total_bytes = l1.output_bytes + l2.output_bytes;
            let bytes_saved = (total_bytes as f64 * FusionPattern::ConvBnActivation.bytes_reduction_ratio()) as u64;
            Some(FusionOpportunity {
                pattern: FusionPattern::ConvBnActivation,
                layer_ids: vec![l1.id.clone(), l2.id.clone(), l3.id.clone()],
                bytes_saved,
                latency_factor: FusionPattern::ConvBnActivation.latency_speedup(),
                confidence: 0.90,
            })
        } else {
            None
        }
    }
    
    fn detect_mlp_block(&self, l1: &LayerInfo, l2: &LayerInfo, l3: &LayerInfo) -> Option<FusionOpportunity> {
        if l1.layer_type == "Linear" && is_activation(&l2.layer_type) && l3.layer_type == "Linear" {
            let total_bytes = l1.output_bytes + l2.output_bytes;
            let bytes_saved = (total_bytes as f64 * FusionPattern::MlpBlock.bytes_reduction_ratio()) as u64;
            Some(FusionOpportunity {
                pattern: FusionPattern::MlpBlock,
                layer_ids: vec![l1.id.clone(), l2.id.clone(), l3.id.clone()],
                bytes_saved,
                latency_factor: FusionPattern::MlpBlock.latency_speedup(),
                confidence: 0.85,
            })
        } else {
            None
        }
    }
    
    fn detect_attention_residual(&self, l1: &LayerInfo, l2: &LayerInfo) -> Option<FusionOpportunity> {
        if l1.layer_type == "Attention" && l2.layer_type == "Residual" {
            let bytes_saved = (l1.output_bytes as f64 * FusionPattern::AttentionResidual.bytes_reduction_ratio()) as u64;
            Some(FusionOpportunity {
                pattern: FusionPattern::AttentionResidual,
                layer_ids: vec![l1.id.clone(), l2.id.clone()],
                bytes_saved,
                latency_factor: FusionPattern::AttentionResidual.latency_speedup(),
                confidence: 0.80,
            })
        } else {
            None
        }
    }
    
    fn detect_layernorm_linear(&self, l1: &LayerInfo, l2: &LayerInfo) -> Option<FusionOpportunity> {
        if (l1.layer_type == "LayerNorm" || l1.layer_type == "RmsNorm") && l2.layer_type == "Linear" {
            let bytes_saved = (l1.output_bytes as f64 * FusionPattern::LayerNormLinear.bytes_reduction_ratio()) as u64;
            Some(FusionOpportunity {
                pattern: FusionPattern::LayerNormLinear,
                layer_ids: vec![l1.id.clone(), l2.id.clone()],
                bytes_saved,
                latency_factor: FusionPattern::LayerNormLinear.latency_speedup(),
                confidence: 0.90,
            })
        } else {
            None
        }
    }
}

/// Layer information for fusion detection
#[derive(Debug, Clone)]
pub struct LayerInfo {
    pub id: String,
    pub layer_type: String,
    pub output_bytes: u64,
    pub flops: u64,
}

impl LayerInfo {
    pub fn new(id: String, layer_type: String, output_bytes: u64, flops: u64) -> Self {
        Self { id, layer_type, output_bytes, flops }
    }
}

/// Check if layer type is an activation
fn is_activation(layer_type: &str) -> bool {
    matches!(layer_type, "ReLU" | "GELU" | "SiLU" | "Sigmoid" | "Tanh" | "Swish")
}

/// Fusion analysis result
#[derive(Debug, Clone, Serialize, Default)]
pub struct FusionAnalysis {
    /// All detected opportunities
    pub opportunities: Vec<FusionOpportunity>,
    /// Total bytes saved by all fusions
    pub total_bytes_saved: u64,
    /// Average latency improvement
    pub avg_latency_factor: f64,
    /// Number of fusion patterns detected
    pub patterns_detected: usize,
}

impl FusionAnalysis {
    pub fn from_opportunities(opportunities: Vec<FusionOpportunity>) -> Self {
        let total_bytes_saved = opportunities.iter().map(|o| o.bytes_saved).sum();
        let avg_latency_factor = if opportunities.is_empty() {
            1.0
        } else {
            opportunities.iter().map(|o| o.latency_factor).sum::<f64>() / opportunities.len() as f64
        };
        
        Self {
            patterns_detected: opportunities.len(),
            total_bytes_saved,
            avg_latency_factor,
            opportunities,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fusion_pattern_properties() {
        let pattern = FusionPattern::LinearActivation;
        assert_eq!(pattern.bytes_reduction_ratio(), 0.5);
        assert!(pattern.latency_speedup() > 1.0);
        assert!(pattern.flops_unchanged());
    }
    
    #[test]
    fn test_detect_linear_activation() {
        let detector = FusionDetector::new();
        let layers = vec![
            LayerInfo::new("l1".into(), "Linear".into(), 1024 * 1024, 1000),
            LayerInfo::new("l2".into(), "GELU".into(), 1024 * 1024, 100),
        ];
        
        let opportunities = detector.detect(&layers);
        assert_eq!(opportunities.len(), 1);
        assert_eq!(opportunities[0].pattern, FusionPattern::LinearActivation);
    }
    
    #[test]
    fn test_detect_mlp_block() {
        let detector = FusionDetector::new();
        let layers = vec![
            LayerInfo::new("l1".into(), "Linear".into(), 1024 * 1024, 1000),
            LayerInfo::new("l2".into(), "GELU".into(), 1024 * 1024, 100),
            LayerInfo::new("l3".into(), "Linear".into(), 1024 * 1024, 1000),
        ];
        
        let opportunities = detector.detect(&layers);
        assert!(!opportunities.is_empty());
        assert!(opportunities.iter().any(|o| o.pattern == FusionPattern::MlpBlock));
    }
    
    #[test]
    fn test_fusion_analysis() {
        let opportunities = vec![
            FusionOpportunity {
                pattern: FusionPattern::LinearActivation,
                layer_ids: vec!["l1".into(), "l2".into()],
                bytes_saved: 1024,
                latency_factor: 1.15,
                confidence: 0.95,
            },
        ];
        
        let analysis = FusionAnalysis::from_opportunities(opportunities);
        assert_eq!(analysis.total_bytes_saved, 1024);
        assert_eq!(analysis.patterns_detected, 1);
    }
}
