//! Liveness analysis

use super::{LivenessInterval, MemorySnapshot};

/// Find peak memory from liveness intervals
pub fn find_peak_memory(intervals: &[LivenessInterval]) -> u64 {
    if intervals.is_empty() {
        return 0;
    }
    
    let max_step = intervals.iter().map(|l| l.end_step).max().unwrap_or(0);
    
    (0..=max_step)
        .map(|step| {
            intervals.iter()
                .filter(|l| l.start_step <= step && l.end_step >= step)
                .map(|l| l.size_bytes)
                .sum()
        })
        .max()
        .unwrap_or(0)
}

/// Find tensors alive at a given step
pub fn find_live_tensors(intervals: &[LivenessInterval], step: usize) -> Vec<&LivenessInterval> {
    intervals.iter()
        .filter(|l| l.start_step <= step && l.end_step >= step)
        .collect()
}

/// Calculate memory savings from gradient checkpointing
pub fn checkpoint_savings(intervals: &[LivenessInterval], checkpoint_steps: &[usize]) -> u64 {
    // With checkpointing, we only keep activations at checkpoint boundaries
    let mut saved = 0u64;
    
    for interval in intervals {
        // If tensor is not at a checkpoint boundary, it can be recomputed
        let is_checkpointed = checkpoint_steps.contains(&interval.start_step) 
            || checkpoint_steps.contains(&interval.end_step);
        
        if !is_checkpointed {
            saved += interval.size_bytes;
        }
    }
    
    saved
}

/// Analyze memory fragmentation
pub fn analyze_fragmentation(snapshots: &[MemorySnapshot]) -> f64 {
    if snapshots.is_empty() {
        return 0.0;
    }
    
    // Fragmentation = 1 - (min_memory / max_memory)
    let peak = snapshots.iter().map(|s| s.total_memory).max().unwrap_or(0);
    let min = snapshots.iter().map(|s| s.total_memory).min().unwrap_or(0);
    
    if peak == 0 {
        return 0.0;
    }
    
    1.0 - (min as f64 / peak as f64)
}
