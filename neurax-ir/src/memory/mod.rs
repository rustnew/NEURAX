//! Memory IR - Dialecte de la simulation mémoire

mod ir;
mod pass;
mod liveness;
mod metrics;
mod fragmentation;

pub use ir::*;
pub use pass::*;
pub use liveness::*;
pub use metrics::*;
pub use fragmentation::*;
