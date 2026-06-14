//! Memory fragmentation model per tuning.md §18
//! 
//! Dynamically estimates memory fragmentation based on allocation patterns

use serde::Serialize;

/// Allocation strategy affects fragmentation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
pub enum AllocationStrategy {
    /// Best-fit allocation (lowest fragmentation)
    BestFit,
    /// First-fit allocation (medium fragmentation)
    #[default]
    FirstFit,
    /// PyTorch caching allocator (reduces fragmentation)
    PyTorchCachingAllocator,
    /// CUDA memory pool (modern, lowest fragmentation)
    CudaPool,
}

impl AllocationStrategy {
    /// Base fragmentation factor for this strategy
    pub fn base_fragmentation(&self) -> f64 {
        match self {
            Self::BestFit => 0.05,
            Self::FirstFit => 0.15,
            Self::PyTorchCachingAllocator => 0.08,
            Self::CudaPool => 0.03,
        }
    }
}

/// Memory fragmentation model
#[derive(Debug, Clone, Serialize)]
pub struct FragmentationModel {
    /// Allocation strategy used
    pub strategy: AllocationStrategy,
    /// Estimated fragmentation ratio (0.0-1.0)
    pub fragmentation_ratio: f64,
    /// Wasted memory in bytes
    pub wasted_bytes: u64,
    /// Factors contributing to fragmentation
    pub factors: Vec<FragmentationFactor>,
}

impl Default for FragmentationModel {
    fn default() -> Self {
        Self {
            strategy: AllocationStrategy::default(),
            fragmentation_ratio: 0.1,
            wasted_bytes: 0,
            factors: Vec::new(),
        }
    }
}

/// Factors that increase fragmentation
#[derive(Debug, Clone, Serialize)]
pub enum FragmentationFactor {
    /// Many small allocations
    SmallAllocations { count: u32, impact: f64 },
    /// Variable tensor sizes
    VariableSizes { variance: f64, impact: f64 },
    /// In-place operations creating gaps
    InPlaceOps { count: u32, impact: f64 },
    /// Gradient checkpointing memory churn
    GradientCheckpointing { enabled: bool, impact: f64 },
    /// MoE expert swapping
    MoeExpertSwapping { num_experts: u32, impact: f64 },
    /// Activation offloading
    ActivationOffloading { enabled: bool, impact: f64 },
}

/// Fragmentation estimator
pub struct FragmentationEstimator {
    strategy: AllocationStrategy,
}

impl FragmentationEstimator {
    pub fn new(strategy: AllocationStrategy) -> Self {
        Self { strategy }
    }
    
    /// Estimate fragmentation for a model configuration
    pub fn estimate(
        &self,
        peak_memory_bytes: u64,
        num_layers: u32,
        has_gradient_checkpointing: bool,
        num_experts: u32,
        activation_offloading: bool,
    ) -> FragmentationModel {
        let mut fragmentation = self.strategy.base_fragmentation();
        let mut factors = Vec::new();
        
        // Factor 1: Number of layers affects allocation pattern
        let layer_factor = if num_layers > 100 {
            0.05
        } else if num_layers > 50 {
            0.03
        } else if num_layers > 24 {
            0.02
        } else {
            0.0
        };
        if layer_factor > 0.0 {
            fragmentation += layer_factor;
            factors.push(FragmentationFactor::SmallAllocations {
                count: num_layers,
                impact: layer_factor,
            });
        }
        
        // Factor 2: Gradient checkpointing creates memory churn
        if has_gradient_checkpointing {
            let impact = match self.strategy {
                AllocationStrategy::PyTorchCachingAllocator => 0.02, // Caching helps
                AllocationStrategy::CudaPool => 0.01,
                _ => 0.05,
            };
            fragmentation += impact;
            factors.push(FragmentationFactor::GradientCheckpointing {
                enabled: true,
                impact,
            });
        }
        
        // Factor 3: MoE expert swapping
        if num_experts > 0 {
            let impact = if num_experts > 16 {
                0.08
            } else if num_experts > 8 {
                0.05
            } else {
                0.03
            };
            fragmentation += impact;
            factors.push(FragmentationFactor::MoeExpertSwapping {
                num_experts,
                impact,
            });
        }
        
        // Factor 4: Activation offloading
        if activation_offloading {
            fragmentation += 0.04;
            factors.push(FragmentationFactor::ActivationOffloading {
                enabled: true,
                impact: 0.04,
            });
        }
        
        // Cap fragmentation at reasonable levels
        let max_frag = match self.strategy {
            AllocationStrategy::CudaPool => 0.15,
            AllocationStrategy::PyTorchCachingAllocator => 0.20,
            _ => 0.30,
        };
        fragmentation = fragmentation.min(max_frag);
        
        let wasted_bytes = (peak_memory_bytes as f64 * fragmentation) as u64;
        
        FragmentationModel {
            strategy: self.strategy,
            fragmentation_ratio: fragmentation,
            wasted_bytes,
            factors,
        }
    }
    
    /// Get effective memory after accounting for fragmentation
    pub fn effective_memory(&self, peak_bytes: u64, model: &FragmentationModel) -> u64 {
        peak_bytes + model.wasted_bytes
    }
}

/// Memory allocation pattern for analysis
#[derive(Debug, Clone)]
pub struct AllocationPattern {
    /// Size of each allocation
    pub sizes: Vec<u64>,
    /// Lifetime of each allocation (in ops)
    pub lifetimes: Vec<u32>,
    /// Whether allocation is persistent
    pub persistent: Vec<bool>,
}

impl AllocationPattern {
    /// Analyze pattern for fragmentation prediction
    pub fn analyze(&self) -> PatternAnalysis {
        let total = self.sizes.len();
        if total == 0 {
            return PatternAnalysis::default();
        }
        
        // Size variance
        let mean_size = self.sizes.iter().sum::<u64>() as f64 / total as f64;
        let variance = if mean_size > 0.0 {
            self.sizes.iter()
                .map(|s| (*s as f64 - mean_size).powi(2))
                .sum::<f64>() / total as f64
        } else {
            0.0
        };
        let std_dev = variance.sqrt();
        let cv = if mean_size > 0.0 { std_dev / mean_size } else { 0.0 };
        
        // Small allocation ratio
        let small_threshold = mean_size * 0.25;
        let small_count = self.sizes.iter()
            .filter(|s| **s < small_threshold as u64)
            .count() as u32;
        
        // Short-lived ratio
        let median_lifetime = {
            let mut sorted = self.lifetimes.clone();
            sorted.sort();
            sorted[total / 2]
        };
        let short_lived = self.lifetimes.iter()
            .filter(|l| **l < median_lifetime / 2)
            .count() as u32;
        
        PatternAnalysis {
            size_cv: cv,
            small_allocation_ratio: small_count as f64 / total as f64,
            short_lived_ratio: short_lived as f64 / total as f64,
            total_allocations: total as u32,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct PatternAnalysis {
    /// Coefficient of variation for sizes
    pub size_cv: f64,
    /// Ratio of small allocations
    pub small_allocation_ratio: f64,
    /// Ratio of short-lived allocations
    pub short_lived_ratio: f64,
    /// Total number of allocations
    pub total_allocations: u32,
}

impl PatternAnalysis {
    /// Predict fragmentation from pattern analysis
    pub fn predicted_fragmentation(&self, strategy: AllocationStrategy) -> f64 {
        let base = strategy.base_fragmentation();
        
        // High size variance increases fragmentation
        let variance_factor = self.size_cv * 0.1;
        
        // Many small allocations increase fragmentation
        let small_factor = self.small_allocation_ratio * 0.08;
        
        // Many short-lived allocations can increase fragmentation
        let churn_factor = self.short_lived_ratio * 0.05;
        
        let total = base + variance_factor + small_factor + churn_factor;
        
        // Cap based on strategy
        let max = match strategy {
            AllocationStrategy::CudaPool => 0.15,
            AllocationStrategy::PyTorchCachingAllocator => 0.20,
            _ => 0.30,
        };
        
        total.min(max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fragmentation_estimator() {
        let estimator = FragmentationEstimator::new(AllocationStrategy::PyTorchCachingAllocator);
        let model = estimator.estimate(
            80 * 1024 * 1024 * 1024, // 80GB
            96,
            true,
            0,
            false,
        );
        
        assert!(model.fragmentation_ratio > 0.0);
        assert!(model.fragmentation_ratio < 0.30);
        assert!(model.wasted_bytes > 0);
    }
    
    #[test]
    fn test_moe_fragmentation() {
        let estimator = FragmentationEstimator::new(AllocationStrategy::FirstFit);
        let model = estimator.estimate(
            80 * 1024 * 1024 * 1024,
            32,
            false,
            16,
            false,
        );
        
        // MoE should increase fragmentation
        assert!(model.fragmentation_ratio > 0.15);
    }
    
    #[test]
    fn test_cuda_pool_lowest_fragmentation() {
        let estimator = FragmentationEstimator::new(AllocationStrategy::CudaPool);
        let model = estimator.estimate(
            80 * 1024 * 1024 * 1024,
            96,
            true,
            0,
            true,
        );
        
        // CUDA pool should have lowest fragmentation
        assert!(model.fragmentation_ratio < 0.15);
    }
    
    #[test]
    fn test_allocation_pattern_analysis() {
        let pattern = AllocationPattern {
            sizes: vec![1000, 2000, 500, 3000, 1500],
            lifetimes: vec![10, 20, 5, 30, 15],
            persistent: vec![false, false, true, false, false],
        };
        
        let analysis = pattern.analyze();
        assert!(analysis.size_cv > 0.0);
        assert!(analysis.total_allocations == 5);
    }
}
