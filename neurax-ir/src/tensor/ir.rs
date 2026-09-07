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

#[allow(clippy::derivable_impls)]
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
        self.0
            .iter()
            .map(|d| match d {
                Dim::Known(n) => Some(*n),
                _ => None,
            })
            .collect()
    }

    /// Calculate number of elements
    ///
    /// Returns `None` when the shape is not fully concrete, and saturates
    /// rather than wrapping if the dimensions multiply beyond `usize` — a
    /// wrapped element count would silently understate tensor memory.
    pub fn num_elements(&self) -> Option<usize> {
        self.to_concrete()
            .map(|s| s.iter().fold(1usize, |acc, d| acc.saturating_mul(*d)))
    }

    /// Calculate size in bytes
    /// Byte size of this shape at `dtype`.
    ///
    /// Delegates to `neurax_formulas::dtype_bytes` — the project's one table of
    /// storage widths — rather than keeping a second one here. The copy this
    /// replaces had drifted in the way a duplicated table always eventually
    /// does: it had no `int4` arm, so int4 fell through to `_ => 4` and every
    /// *output* tensor was sized at fp32 width, while input tensors on the
    /// lines above already used `dtype_bytes` and correctly took 0.5. Across
    /// all 106 reference templates that mix reported int4 activations at
    /// 0.54–0.99x fp32 instead of 0.125x — larger, on every single model, than
    /// the same design at int8.
    ///
    /// The integer `bytes_per_elem` was the deeper half of the fault: a width
    /// of half a byte cannot be expressed in it at all, so no `int4` arm could
    /// have been added without this change. int4 packs two values per byte
    /// (GPTQ/AWQ/QLoRA-NF4/GGUF Q4 all store it that way), which is exactly
    /// why `dtype_bytes` returns `f64`.
    pub fn size_bytes(&self, dtype: &str) -> u64 {
        let elements = self.num_elements().unwrap_or(0);
        (elements as f64 * neurax_formulas::dtype_bytes(dtype)).round() as u64
    }
}

#[allow(clippy::derivable_impls)]
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
    pub tiny: usize,   // < 1KB
    pub small: usize,  // 1KB - 1MB
    pub medium: usize, // 1MB - 100MB
    pub large: usize,  // 100MB - 1GB
    pub huge: usize,   // > 1GB
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
