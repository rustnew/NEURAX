//! Parallelism IR - Dialecte de l'analyse de scalabilité

mod ir;
mod pass;
mod strategies;
mod metrics;

pub use ir::*;
pub use pass::*;
pub use strategies::*;
pub use metrics::*;
