//! Graph IR structures

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::algo::{is_cyclic_directed, toposort};
use std::collections::HashMap;
use neurax_parser::LayerType;

/// Graph IR - dialecte du graphe computationnel
#[derive(Debug, Clone)]
pub struct GraphIR {
    /// DAG des opérations
    pub dag: DiGraph<GraphNode, GraphEdge>,
    /// Index des noeuds par ID de layer
    pub node_index: HashMap<String, NodeIndex>,
    /// Ordre topologique précalculé
    pub topo_order: Vec<NodeIndex>,
    /// Métriques calculées
    pub metrics: GraphMetrics,
    pub metrics_done: bool,
}

impl Default for GraphIR {
    fn default() -> Self {
        Self {
            dag: DiGraph::new(),
            node_index: HashMap::new(),
            topo_order: Vec::new(),
            metrics: GraphMetrics::default(),
            metrics_done: false,
        }
    }
}

/// Node in the computation graph
#[derive(Debug, Clone)]
pub struct GraphNode {
    pub layer_id: String,
    pub layer_type: LayerType,
    pub flops_approx: f64,
    pub input_shapes: Vec<Vec<usize>>,
    pub output_shape: Vec<usize>,
    pub param_count: u64,
}

/// Edge in the computation graph
#[derive(Debug, Clone)]
pub struct GraphEdge {
    /// Tensor shape transported on this edge
    pub tensor_shape: Vec<usize>,
    pub dtype: String,
    pub size_bytes: u64,
}

/// Graph metrics (Métriques 3-5)
#[derive(Debug, Clone, Default)]
pub struct GraphMetrics {
    /// Métrique 3: Profondeur du graphe (chemin critique)
    pub graph_depth: usize,
    /// Métrique 4: Nombre total d'opérations
    pub total_operations: usize,
    /// Métrique 5: Nombre de tenseurs intermédiaires
    pub total_intermediate_tensors: usize,
    /// Chemins parallèles détectés
    pub parallel_paths: Vec<Vec<String>>,
    /// Longueur du chemin critique
    pub critical_path_length: usize,
    /// Nombre d'arêtes
    pub edge_count: usize,
}

impl GraphMetrics {
    pub fn is_valid(&self) -> bool {
        self.total_operations > 0 && self.graph_depth > 0
    }
}

impl GraphIR {
    /// Create a new empty GraphIR
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Add a node to the graph
    pub fn add_node(&mut self, node: GraphNode) -> NodeIndex {
        let id = node.layer_id.clone();
        let idx = self.dag.add_node(node);
        self.node_index.insert(id, idx);
        idx
    }
    
    /// Add an edge between two nodes
    pub fn add_edge(&mut self, from: NodeIndex, to: NodeIndex, edge: GraphEdge) {
        self.dag.add_edge(from, to, edge);
    }
    
    /// Get node by layer ID
    pub fn get_node(&self, layer_id: &str) -> Option<&GraphNode> {
        self.node_index.get(layer_id).and_then(|idx| self.dag.node_weight(*idx))
    }
    
    /// Check if graph has cycles
    pub fn has_cycle(&self) -> bool {
        is_cyclic_directed(&self.dag)
    }
    
    /// Compute topological order
    pub fn compute_topo_order(&mut self) -> Result<Vec<NodeIndex>, String> {
        if self.has_cycle() {
            return Err("Graph has cycles".to_string());
        }
        
        match toposort(&self.dag, None) {
            Ok(order) => {
                self.topo_order = order.clone();
                Ok(order)
            }
            Err(_) => Err("Topological sort failed".to_string()),
        }
    }
    
    /// Calculate graph depth (longest path)
    pub fn calculate_depth(&self) -> usize {
        if self.topo_order.is_empty() {
            return 0;
        }
        
        // Use DP to find longest path
        let mut max_depth = HashMap::<NodeIndex, usize>::new();
        
        for &node in &self.topo_order {
            let mut max_pred_depth = 0;
            for pred in self.dag.neighbors_directed(node, petgraph::Direction::Incoming) {
                if let Some(&depth) = max_depth.get(&pred) {
                    max_pred_depth = max_pred_depth.max(depth);
                }
            }
            max_depth.insert(node, max_pred_depth + 1);
        }
        
        max_depth.values().copied().max().unwrap_or(0)
    }
}
