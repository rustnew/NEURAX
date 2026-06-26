//! Export module — binary format exporters
//!
//! Currently supports ONNX protobuf export.

pub mod onnx;

pub use onnx::{export_onnx, OnnxExportResult};