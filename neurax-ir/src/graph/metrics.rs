//! Graph metrics calculation

use super::GraphIR;
use petgraph::graph::NodeIndex;

/// Find parallel paths in the graph
pub fn find_parallel_paths(_graph: &GraphIR) -> Vec<Vec<String>> {
    // For now, return empty - this is for non-sequential models
    // TODO: Implement proper parallel path detection for DAGs
    vec![]
}

/// Calculate critical path (longest path in DAG)
pub fn calculate_critical_path(graph: &GraphIR) -> Vec<String> {
    if graph.topo_order.is_empty() {
        return vec![];
    }
    
    // Find the node with maximum depth and trace back
    let mut path = vec![];
    let mut current: Option<NodeIndex> = graph.topo_order.last().copied();
    
    while let Some(idx) = current {
        if let Some(node) = graph.dag.node_weight(idx) {
            path.push(node.layer_id.clone());
        }
        
        // Get predecessor with maximum depth
        let preds: Vec<_> = graph.dag.neighbors_directed(idx, petgraph::Direction::Incoming).collect();
        if preds.is_empty() {
            break;
        }
        current = Some(preds[0]); // Simplified - take first predecessor
    }
    
    path.reverse();
    path
}

/// Calculate graph statistics
pub fn calculate_graph_stats(graph: &GraphIR) -> GraphStats {
    GraphStats {
        node_count: graph.dag.node_count(),
        edge_count: graph.dag.edge_count(),
        density: calculate_density(graph),
        average_degree: calculate_avg_degree(graph),
    }
}

#[derive(Debug, Clone, Default)]
pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub density: f64,
    pub average_degree: f64,
}

fn calculate_density(graph: &GraphIR) -> f64 {
    let n = graph.dag.node_count() as f64;
    if n <= 1.0 {
        return 0.0;
    }
    let e = graph.dag.edge_count() as f64;
    let max_edges = n * (n - 1.0) / 2.0; // For DAG
    e / max_edges
}

fn calculate_avg_degree(graph: &GraphIR) -> f64 {
    if graph.dag.node_count() == 0 {
        return 0.0;
    }
    let total_degree: usize = graph.dag.node_indices()
        .map(|n| graph.dag.neighbors_undirected(n).count())
        .sum();
    total_degree as f64 / graph.dag.node_count() as f64
}
