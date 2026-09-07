//! Algebraic invariants the metric system must hold for *every* model.
//!
//! The example-based tests beside this file each pin one design to one expected
//! figure. That is what catches a wrong formula, and it is why
//! `published_model_accuracy.rs` exists. It is not what catches a formula that
//! is right on the shapes someone happened to write down and wrong everywhere
//! else — which is exactly how int4 activations came to be reported *larger*
//! than int8 on all 106 reference templates while every hand-written test
//! stayed green.
//!
//! These are the relationships that hold regardless of the architecture:
//! storage width scales memory exactly, per-layer parameters sum to the total,
//! FLOPs are linear in batch, depth never reduces size. `proptest` searches for
//! a counter-example instead of trusting a fixture to contain one.
//!
//! `proptest` has been a declared dev-dependency of this crate, of
//! `neurax-formulas` and of the workspace root since before this file, with
//! zero `proptest!` blocks anywhere in the repository.

use neurax_core::analyze_json;
use proptest::prelude::*;

/// Storage widths, and the exact bytes-per-parameter each one means.
const WIDTHS: [(&str, f64); 5] = [
    ("fp32", 4.0),
    ("fp16", 2.0),
    ("bf16", 2.0),
    ("int8", 1.0),
    ("int4", 0.5),
];

fn model_json(
    family: &str,
    layers: &str,
    hidden: usize,
    batch: usize,
    seq: usize,
    precision: &str,
) -> String {
    format!(
        r#"{{
            "schema_version": "1.0",
            "model": {{
                "name": "Invariant-Check",
                "type": "{family}",
                "layers": [{layers}],
                "global_params": {{ "hidden_size": {hidden}, "sequence_length": {seq} }}
            }},
            "training": {{ "batch_size": {batch}, "precision": "{precision}", "optimizer": "none" }},
            "hardware": {{ "gpus": [{{ "name": "A100-80GB", "count": 1 }}] }},
            "data": {{ "dtype": "{precision}" }}
        }}"#
    )
}

/// A transformer stack of `depth` attention+MLP pairs at `hidden` width.
fn transformer_layers(depth: usize, hidden: usize) -> String {
    (0..depth)
        .map(|i| {
            format!(
                r#"{{"id": "attn{i}", "layer_type": "attention", "params": {{"hidden_size": {hidden}, "num_heads": 8}}}},
                   {{"id": "mlp{i}", "layer_type": "mlp", "params": {{"hidden_size": {hidden}, "intermediate_size": {}}}}}"#,
                hidden * 4
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn analyse(depth: usize, hidden: usize, batch: usize, seq: usize, precision: &str) -> neurax_core::AnalysisResult {
    let json = model_json("transformer", &transformer_layers(depth, hidden), hidden, batch, seq, precision);
    analyze_json(&json).expect("a well-formed model must analyse")
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// Weight memory is the parameter count times the storage width — exactly,
    /// to within the half byte an odd count cannot store.
    #[test]
    fn weight_memory_is_parameters_times_storage_width(
        depth in 1usize..5,
        hidden in prop::sample::select(vec![128usize, 256, 512, 1024]),
        width_idx in 0usize..5,
    ) {
        let (precision, bytes) = WIDTHS[width_idx];
        let report = analyse(depth, hidden, 1, 128, precision);
        let params = report.memory.total_parameters as f64;
        let observed = report.memory.metrics.parameter_memory_bytes as f64;

        prop_assert!(
            (observed - params * bytes).abs() <= 0.5,
            "{precision}: {observed} bytes for {params} parameters, expected {}",
            params * bytes
        );
    }

    /// Halving the storage width halves the activation memory. This is the
    /// property int4 violated on every architecture: a second, hand-rolled
    /// width table in `Shape::size_bytes` had no `int4` arm, so output tensors
    /// took fp32 width while input tensors took 0.5.
    #[test]
    fn narrower_storage_never_needs_more_activation_memory(
        depth in 1usize..5,
        hidden in prop::sample::select(vec![256usize, 512, 1024]),
        seq in prop::sample::select(vec![128usize, 512, 2048]),
    ) {
        let mut previous: Option<(&str, u64)> = None;
        // Widest to narrowest.
        for (precision, _) in [WIDTHS[0], WIDTHS[1], WIDTHS[3], WIDTHS[4]] {
            let activations = analyse(depth, hidden, 1, seq, precision)
                .memory
                .metrics
                .activation_memory_bytes;
            if let Some((wider, wider_bytes)) = previous {
                prop_assert!(
                    activations <= wider_bytes,
                    "{precision} needs {activations} bytes of activations, more than {wider}'s {wider_bytes}"
                );
            }
            previous = Some((precision, activations));
        }
    }

    /// Every layer's parameters are counted once, and only once, in the total.
    #[test]
    fn per_layer_parameters_sum_to_the_total(
        depth in 1usize..6,
        hidden in prop::sample::select(vec![128usize, 512, 1024]),
    ) {
        let report = analyse(depth, hidden, 1, 128, "fp16");
        let summed: u64 = report.arch.metrics.params_per_layer.values().sum();
        let total = report.memory.total_parameters;

        prop_assert_eq!(
            summed, total,
            "per-layer parameters sum to {} but the total says {}", summed, total
        );
    }

    /// Doubling the batch doubles the work. A metric that ignores batch, or
    /// applies it twice, breaks here on every shape rather than on the one a
    /// fixture happened to use.
    #[test]
    fn flops_are_linear_in_batch_size(
        depth in 1usize..4,
        hidden in prop::sample::select(vec![256usize, 512]),
        batch in prop::sample::select(vec![1usize, 2, 4, 8]),
    ) {
        let single = analyse(depth, hidden, batch, 128, "fp16").compute.metrics.total_flops;
        let double = analyse(depth, hidden, batch * 2, 128, "fp16").compute.metrics.total_flops;

        prop_assert!(single > 0.0, "a real model must do some work");
        let ratio = double / single;
        prop_assert!(
            (ratio - 2.0).abs() < 1e-6,
            "doubling the batch scaled FLOPs by {ratio}, not 2"
        );
    }

    /// Depth and width only ever add parameters. A formula that silently drops
    /// a layer, or reads a width it then ignores, is not monotone.
    #[test]
    fn depth_and_width_never_reduce_the_parameter_count(
        depth in 1usize..5,
        hidden in prop::sample::select(vec![128usize, 256, 512]),
    ) {
        let base = analyse(depth, hidden, 1, 128, "fp16").memory.total_parameters;
        let deeper = analyse(depth + 1, hidden, 1, 128, "fp16").memory.total_parameters;
        let wider = analyse(depth, hidden * 2, 1, 128, "fp16").memory.total_parameters;

        prop_assert!(deeper > base, "adding a layer did not add parameters ({base} -> {deeper})");
        prop_assert!(wider > base, "doubling the width did not add parameters ({base} -> {wider})");
    }
}
