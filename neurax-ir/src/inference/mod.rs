//! Inference Dialect — simulation analytique du comportement d'inférence.
//!
//! Expose :
//! - `InferenceParams`  : paramètres de contrôle fournis par le frontend
//! - `InferenceReport`  : rapport complet (10 widgets)
//! - `InferencePass`    : calcul analytique du rapport

pub mod ir;
pub mod pass;

pub use ir::{
    HallucinationRisk, InferenceParams, InferenceReport, RiskLevel, RiskOverview,
    RouterStability, SamplingVolatility, StabilityIndex, StabilityLevel,
};
pub use pass::InferencePass;
