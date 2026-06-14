//! Parallelism IR structures


/// Parallelism IR - dialecte de l'analyse de scalabilité
#[derive(Debug, Clone)]
pub struct ParallelismIR {
    pub strategies: Vec<ParallelStrategy>,
    pub optimal_strategy: ParallelStrategy,
    pub metrics: ParallelismMetrics,
    pub metrics_done: bool,
}

impl Default for ParallelismIR {
    fn default() -> Self {
        Self {
            strategies: Vec::new(),
            optimal_strategy: ParallelStrategy::default(),
            metrics: ParallelismMetrics::default(),
            metrics_done: false,
        }
    }
}

/// Parallel strategy
#[derive(Debug, Clone, PartialEq)]
pub enum ParallelStrategy {
    DataParallel {
        num_gpus: u32,
        efficiency: f64,
    },
    ModelParallel {
        splits: Vec<LayerRange>,
    },
    PipelineParallel {
        stages: u32,
        micro_batches: u32,
        bubble_ratio: f64,
    },
    Hybrid {
        dp: u32,
        tp: u32,
        pp: u32,
    },
    ZeRO {
        stage: u8,
        memory_per_gpu: u64,
    },
    TensorParallel {
        tp_degree: u32,
    },
}

impl Default for ParallelStrategy {
    fn default() -> Self {
        Self::DataParallel {
            num_gpus: 1,
            efficiency: 1.0,
        }
    }
}

/// Layer range for model parallel
#[derive(Debug, Clone, PartialEq)]
pub struct LayerRange {
    pub start_layer: String,
    pub end_layer: String,
}

/// Parallelism metrics (Métriques 25-28, 33-36)
#[derive(Debug, Clone, Default)]
pub struct ParallelismMetrics {
    /// Métrique 33: Efficacité data parallel (0.0–1.0)
    pub data_parallel_efficiency: f64,
    /// Métrique 34: Model parallel faisable
    pub model_parallel_feasible: bool,
    /// Métrique 35: Pipeline parallel stages
    pub pipeline_stages: Option<u32>,
    /// Métrique 36: Communication overhead (fraction du temps total)
    pub communication_overhead: f64,
    /// GPU optimal count
    pub optimal_gpu_count: u32,
    /// Mémoire par GPU après partitionnement
    pub memory_per_gpu_bytes: u64,
    /// Courbe d'efficacité de scaling
    pub scaling_efficiency_curve: Vec<(u32, f64)>,
    /// Temps de communication All-Reduce
    pub allreduce_time_ms: f64,
    /// Temps de calcul
    pub compute_time_ms: f64,
}

impl ParallelismMetrics {
    pub fn is_valid(&self) -> bool {
        self.optimal_gpu_count > 0 && self.data_parallel_efficiency >= 0.0
    }
}
