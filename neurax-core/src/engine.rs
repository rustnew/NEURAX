//! IR Passer Engine - Sequential orchestration

use ahash::AHashMap as HashMap;
use std::sync::Arc;
use parking_lot::Mutex;
use std::time::Duration;

/// Metrics store - thread-safe accumulator
pub struct MetricsStore {
    metrics: Arc<Mutex<HashMap<String, f64>>>,
}

impl MetricsStore {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    pub fn insert(&self, key: String, value: f64) {
        self.metrics.lock().insert(key, value);
    }
    
    pub fn get(&self, key: &str) -> Option<f64> {
        self.metrics.lock().get(key).copied()
    }
    
    pub fn all(&self) -> HashMap<String, f64> {
        self.metrics.lock().clone()
    }
}

impl Default for MetricsStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Pass timing record
#[derive(Debug, Clone)]
pub struct PassTiming {
    pub pass_name: String,
    pub duration_ms: u64,
    pub success: bool,
}

/// Engine statistics
#[derive(Debug, Clone, Default)]
pub struct EngineStats {
    pub total_time_ms: u64,
    pub pass_timings: Vec<PassTiming>,
    pub passes_completed: usize,
    pub passes_failed: usize,
}

/// IR Passer Engine
pub struct IrPasserEngine {
    stats: EngineStats,
    metrics_store: MetricsStore,
}

impl IrPasserEngine {
    pub fn new() -> Self {
        Self {
            stats: EngineStats::default(),
            metrics_store: MetricsStore::new(),
        }
    }
    
    /// Record a pass timing
    pub fn record_pass(&mut self, name: &str, duration: Duration, success: bool) {
        self.stats.pass_timings.push(PassTiming {
            pass_name: name.to_string(),
            duration_ms: duration.as_millis() as u64,
            success,
        });
        if success {
            self.stats.passes_completed += 1;
        } else {
            self.stats.passes_failed += 1;
        }
    }
    
    /// Get engine statistics
    pub fn stats(&self) -> &EngineStats {
        &self.stats
    }
    
    /// Get metrics store
    pub fn metrics(&self) -> &MetricsStore {
        &self.metrics_store
    }
    
    /// Set total time
    pub fn set_total_time(&mut self, duration: Duration) {
        self.stats.total_time_ms = duration.as_millis() as u64;
    }
}

impl Default for IrPasserEngine {
    fn default() -> Self {
        Self::new()
    }
}
