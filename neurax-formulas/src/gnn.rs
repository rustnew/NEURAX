//! Graph Neural Network formulas

/// Compute FLOPs for Graph Convolutional Network (GCN) layer
///
/// # Arguments
/// * `num_nodes` - Number of nodes in the graph
/// * `in_features` - Input feature dimension
/// * `out_features` - Output feature dimension
/// * `num_edges` - Number of edges (for message passing)
pub fn gcn_flops(
    num_nodes: usize,
    in_features: usize,
    out_features: usize,
    num_edges: usize,
) -> f64 {
    // Linear transformation: [N, in] × [in, out] → [N, out]
    let linear_flops = 2.0 * num_nodes as f64 * in_features as f64 * out_features as f64;
    
    // Aggregation: sum over neighbors for each node
    // Each edge contributes one addition
    let agg_flops = num_edges as f64 * out_features as f64;
    
    // Normalization (degree-based)
    let norm_flops = 2.0 * num_nodes as f64 * out_features as f64;
    
    linear_flops + agg_flops + norm_flops
}

/// Compute FLOPs for Graph Attention Network (GAT) layer
pub fn gat_flops(
    num_nodes: usize,
    in_features: usize,
    out_features: usize,
    num_edges: usize,
    num_heads: usize,
) -> f64 {
    let head_dim = out_features / num_heads;
    
    // Per-head attention
    let per_head = {
        // Linear projections for Q, K (or just one for source)
        let proj = 2.0 * num_nodes as f64 * in_features as f64 * head_dim as f64;
        
        // Attention scores for each edge
        let attn = 2.0 * num_edges as f64 * head_dim as f64;
        
        // Softmax over neighbors
        let softmax = 5.0 * num_edges as f64;
        
        // Message aggregation
        let agg = num_edges as f64 * head_dim as f64;
        
        proj + attn + softmax + agg
    };
    
    per_head * num_heads as f64
}

/// Compute FLOPs for message passing neural network
pub fn mpnn_flops(
    num_nodes: usize,
    num_edges: usize,
    node_features: usize,
    edge_features: usize,
    message_dim: usize,
) -> f64 {
    // Message function: edge + source + target features → message
    let msg_flops = num_edges as f64 * (2.0 * node_features as f64 + edge_features as f64) * message_dim as f64 * 2.0;
    
    // Aggregation: sum messages per node
    let agg_flops = num_edges as f64 * message_dim as f64;
    
    // Update function: GRU or MLP
    let update_flops = 2.0 * num_nodes as f64 * (node_features as f64 + message_dim as f64) * node_features as f64;
    
    msg_flops + agg_flops + update_flops
}

/// Compute parameters for GCN layer
pub fn gcn_params(in_features: usize, out_features: usize, bias: bool) -> u64 {
    let weight = in_features * out_features;
    let bias_params = if bias { out_features } else { 0 };
    (weight + bias_params) as u64
}

/// Compute parameters for GAT layer
pub fn gat_params(in_features: usize, out_features: usize, num_heads: usize, bias: bool) -> u64 {
    let head_dim = out_features / num_heads;
    let weight = in_features * num_heads * head_dim;
    let attn_src = num_heads * head_dim;
    let attn_dst = num_heads * head_dim;
    let bias_params = if bias { out_features } else { 0 };
    (weight + attn_src + attn_dst + bias_params) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcn_flops() {
        let flops = gcn_flops(1000, 128, 256, 5000);
        assert!(flops > 0.0);
    }

    #[test]
    fn test_gcn_params() {
        let params = gcn_params(128, 256, true);
        assert_eq!(params, 128 * 256 + 256);
    }
}
