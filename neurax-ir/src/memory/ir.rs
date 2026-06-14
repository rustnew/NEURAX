//! Memory IR structures


/// Memory IR - dialecte de la simulation mémoire
#[derive(Debug, Clone)]
pub struct MemoryIR {
    /// Intervalles de liveness pour chaque tenseur
    pub liveness: Vec<LivenessInterval>,
    /// Simulation temporelle d'occupation mémoire
    pub memory_timeline: Vec<MemorySnapshot>,
    pub metrics: MemoryMetrics,
    pub metrics_done: bool,
    /// Total parameters from ArchitectureIR for consistent memory calculation
    pub total_parameters: u64,
}

impl Default for MemoryIR {
    fn default() -> Self {
        Self {
            liveness: Vec::new(),
            memory_timeline: Vec::new(),
            metrics: MemoryMetrics::default(),
            metrics_done: false,
            total_parameters: 0,
        }
    }
}

/// Intervalle de vie d'un tenseur [start_step, end_step]
#[derive(Debug, Clone)]
pub struct LivenessInterval {
    pub tensor_id: String,
    pub size_bytes: u64,
    pub start_step: usize,
    pub end_step: usize,
}

/// Snapshot de mémoire à un instant donné
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    pub step: usize,
    pub live_tensors: Vec<String>,
    pub total_memory: u64,
}

/// Memory metrics (Métriques 12-19, 25, 26-32)
#[derive(Debug, Clone, Default)]
pub struct MemoryMetrics {
    /// Métrique 12: Mémoire des paramètres
    pub parameter_memory_bytes: u64,
    /// Métrique 13: Mémoire des activations
    pub activation_memory_bytes: u64,
    /// Métrique 14: Mémoire des gradients
    pub gradient_memory_bytes: u64,
    /// Métrique 15: États de l'optimiseur
    pub optimizer_state_bytes: u64,
    /// Métrique 16: Peak VRAM
    pub peak_vram_bytes: u64,
    /// Métrique 17: Bande passante mémoire requise
    pub memory_bandwidth_req: f64,
    /// Métrique 18: Distribution des tailles de tenseurs
    pub tensor_size_dist: TensorSizeDist,
    /// Métrique 19: Fragmentation estimée
    pub fragmentation_estimate: f64,
    /// Métrique 25: Batch size maximum
    pub max_batch_size_fit: u32,
    /// Risque OOM
    pub oom_risk: OomRisk,
    /// GPU VRAM disponible
    pub gpu_vram_bytes: u64,
}

impl MemoryMetrics {
    pub fn is_valid(&self) -> bool {
        self.peak_vram_bytes > 0 && self.parameter_memory_bytes > 0
    }
    
    /// Convertir en GB
    pub fn peak_vram_gb(&self) -> f64 {
        self.peak_vram_bytes as f64 / 1e9
    }
    
    /// GPU VRAM en GB
    pub fn gpu_vram_gb(&self) -> f64 {
        self.gpu_vram_bytes as f64 / 1e9
    }
}

/// Distribution des tailles de tenseurs
#[derive(Debug, Clone, Default)]
pub struct TensorSizeDist {
    pub tiny: usize,
    pub small: usize,
    pub medium: usize,
    pub large: usize,
    pub huge: usize,
}

/// Risque OOM
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OomRisk {
    /// peak < 80% VRAM
    #[default]
    Safe,
    /// 80% < peak < 95% VRAM
    Warning,
    /// 95% < peak < 100% VRAM
    Critical,
    /// peak > VRAM
    Overflow,
}

impl OomRisk {
    pub fn from_ratio(peak: u64, vram: u64) -> Self {
        if vram == 0 {
            return Self::Critical;
        }
        let ratio = peak as f64 / vram as f64;
        if ratio < 0.8 {
            Self::Safe
        } else if ratio < 0.95 {
            Self::Warning
        } else if ratio < 1.0 {
            Self::Critical
        } else {
            Self::Overflow
        }
    }
    
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Safe => "safe",
            Self::Warning => "warning",
            Self::Critical => "critical",
            Self::Overflow => "overflow",
        }
    }
}
