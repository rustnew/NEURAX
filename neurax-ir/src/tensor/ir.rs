//! Tensor IR structures

use std::collections::HashMap;

/// Tensor IR - dialecte de la propagation de formes
#[derive(Debug, Clone)]
pub struct TensorIR {
    /// Tous les tenseurs du modèle avec leurs shapes
    pub tensors: HashMap<TensorId, TensorInfo>,
    /// Mapping layer → tenseurs d'entrée/sortie
    pub layer_tensors: HashMap<String, LayerTensors>,
    pub metrics: TensorMetrics,
    pub metrics_done: bool,
}

impl Default for TensorIR {
    fn default() -> Self {
        Self {
            tensors: HashMap::new(),
            layer_tensors: HashMap::new(),
            metrics: TensorMetrics::default(),
            metrics_done: false,
        }
    }
}

/// Tensor identifier
pub type TensorId = String;

/// Tensor information
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub id: TensorId,
    pub shape: Shape,
    pub dtype: String,
    /// Taille en octets = prod(shape) × dtype.bytes()
    pub size_bytes: u64,
    /// Layer qui produit ce tenseur
    pub produced_by: String,
    /// Layers qui consomment ce tenseur
    pub consumed_by: Vec<String>,
}

/// Shape with optional symbolic dimensions
#[derive(Debug, Clone, PartialEq)]
pub struct Shape(pub Vec<Dim>);

impl Shape {
    pub fn known(shape: Vec<usize>) -> Self {
        Shape(shape.into_iter().map(Dim::Known).collect())
    }
    
    pub fn symbolic(shape: Vec<Dim>) -> Self {
        Shape(shape)
    }
    
    /// Check if all dimensions are known
    pub fn is_fully_known(&self) -> bool {
        self.0.iter().all(|d| matches!(d, Dim::Known(_)))
    }
    
    /// Get concrete shape if fully known
    pub fn to_concrete(&self) -> Option<Vec<usize>> {
        self.0.iter().map(|d| match d {
            Dim::Known(n) => Some(*n),
            _ => None,
        }).collect()
    }
    
    /// Calculate number of elements
    pub fn num_elements(&self) -> Option<usize> {
        self.to_concrete().map(|s| s.iter().product())
    }
    
    /// Calculate size in bytes
    pub fn size_bytes(&self, dtype: &str) -> u64 {
        let elements = self.num_elements().unwrap_or(0);
        let bytes_per_elem = match dtype {
            "fp64" | "float64" => 8,
            "fp32" | "float32" => 4,
            "fp16" | "float16" => 2,
            "bf16" | "bfloat16" => 2,
            "int8" => 1,
            _ => 4,
        };
        (elements * bytes_per_elem) as u64
    }
}

impl Default for Shape {
    fn default() -> Self {
        Shape(vec![])
    }
}

/// Dimension type
#[derive(Debug, Clone, PartialEq)]
pub enum Dim {
    /// Known concrete dimension
    Known(usize),
    /// Symbolic dimension (e.g., "batch", "seq")
    Symbolic(String),
    /// Dynamic dimension (unknown at compile time)
    Dynamic,
}

/// Tensors for a single layer
#[derive(Debug, Clone, Default)]
pub struct LayerTensors {
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
}

/// Tensor metrics (Métriques 11, 7 partiel, 10 partiel)
#[derive(Debug, Clone, Default)]
pub struct TensorMetrics {
    /// Mémoire des activations (partie de Métrique 7)
    pub activation_memory_bytes: u64,
    /// Bande passante mémoire requise (Métrique 10)
    pub memory_bandwidth_required: f64,
    /// Distribution des tailles de tenseurs (Métrique 11)
    pub tensor_size_distribution: TensorSizeDistribution,
    /// Nombre total de tenseurs
    pub total_tensor_count: usize,
    /// Plus grand tenseur
    pub largest_tensor_bytes: u64,
    /// Plus grand tenseur ID
    pub largest_tensor_id: Option<String>,
    /// Ratio of resolved dimensions (ShapeInferenceGate)
    pub resolution_ratio: f32,
    /// Number of unresolved symbolic/dynamic dimensions
    pub unresolved_dim_count: usize,
    /// Total dimension count
    pub total_dim_count: usize,
}

impl TensorMetrics {
    pub fn is_valid(&self) -> bool {
        self.total_tensor_count > 0
    }
}

/// Tensor size distribution
#[derive(Debug, Clone, Default)]
pub struct TensorSizeDistribution {
    pub tiny: usize,      // < 1KB
    pub small: usize,     // 1KB - 1MB
    pub medium: usize,    // 1MB - 100MB
    pub large: usize,     // 100MB - 1GB
    pub huge: usize,      // > 1GB
}

impl TensorSizeDistribution {
    pub fn classify(size_bytes: u64) -> &'static str {
        match size_bytes {
            0..=1024 => "tiny",
            1025..=1_048_576 => "small",
            1_048_577..=104_857_600 => "medium",
            104_857_601..=1_073_741_824 => "large",
            _ => "huge",
        }
    }
    
    pub fn add(&mut self, size_bytes: u64) {
        match size_bytes {
            0..=1024 => self.tiny += 1,
            1025..=1_048_576 => self.small += 1,
            1_048_577..=104_857_600 => self.medium += 1,
            104_857_601..=1_073_741_824 => self.large += 1,
            _ => self.huge += 1,
        }
    }
}
