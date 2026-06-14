//! NEURAX Dynamic Predictive System
//! 
//! This module implements the dynamic analysis passes that extend the static
//! pipeline with predictive capabilities:
//! 
//! - **VirtualMemoryPass**: Models memory fragmentation and virtualization savings
//! - **StabilityAnalysisPass**: Predicts training stability via Lyapunov exponents
//! - **BehavioralSynthesisPass**: Infers runtime behaviors (MoE imbalance, cache locality)
//! 
//! These passes run in parallel after the static pipeline and provide
//! cross-pass feedback to enrich the final report.

pub mod virtual_memory;
pub mod stability;
pub mod behavioral;
pub mod types;
pub mod feedback;
pub mod evaluation;

pub use virtual_memory::{VirtualMemoryPass, VirtualMemoryMetrics, AllocationStrategy};
pub use stability::{StabilityAnalysisPass, StabilityMetrics};
pub use behavioral::{BehavioralSynthesisPass, BehavioralMetrics};
pub use types::{DynamicResults, DynamicConfig};
pub use feedback::apply_dynamic_feedback;
pub use evaluation::{run_full_evaluation, EvaluationReport};
