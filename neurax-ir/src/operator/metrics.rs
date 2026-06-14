//! Operator metrics

use super::{OperatorIR, OperatorMetrics, OpType};

/// Get FLOPs breakdown by operation type
pub fn get_flops_by_type(op_ir: &OperatorIR) -> std::collections::HashMap<String, f64> {
    let mut flops_by_type = std::collections::HashMap::new();
    
    for op in &op_ir.operations {
        let type_str = op.op_type.as_str().to_string();
        let entry = flops_by_type.entry(type_str).or_insert(0.0);
        *entry += op.flops;
    }
    
    flops_by_type
}

/// Get top FLOPs-consuming layers
pub fn get_top_flops_layers(metrics: &OperatorMetrics, top_n: usize) -> Vec<(String, f64)> {
    let mut layers: Vec<_> = metrics.flops_per_layer.iter()
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    layers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    layers.into_iter().take(top_n).collect()
}

/// Calculate percentage of tensor-core eligible FLOPs
pub fn tensor_core_eligible_percentage(op_ir: &OperatorIR) -> f64 {
    let total: f64 = op_ir.operations.iter().map(|op| op.flops).sum();
    if total == 0.0 {
        return 0.0;
    }
    
    let eligible: f64 = op_ir.operations.iter()
        .filter(|op| matches!(op.op_type, 
            OpType::MatMul | OpType::BatchedMatMul | 
            OpType::Conv2D | OpType::Attention |
            OpType::AttentionScores | OpType::AttentionOutput |
            OpType::Linear
        ))
        .map(|op| op.flops)
        .sum();
    
    eligible / total
}
