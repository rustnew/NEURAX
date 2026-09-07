//! int4 quantization was reported at the same memory footprint as int8 —
//! `dtype_bytes("int4")` returned 1 (a whole byte) instead of 0.5 (two
//! values packed per byte, as GPTQ/AWQ/QLoRA-NF4/GGUF Q4 — the most-cited
//! LLM quantization scheme in practice — actually store it). A model
//! quantized to int4 specifically to fit smaller hardware was reported as
//! needing twice the memory it actually does.

use neurax_core::analyze_json;

fn model_json(precision: &str) -> String {
    format!(
        r#"
        {{
            "schema_version": "1.0",
            "model": {{
                "name": "Int4-Check",
                "type": "transformer",
                "layers": [
                    {{"id": "attn", "layer_type": "attention", "params": {{"hidden_size": 4096, "num_heads": 32}}}},
                    {{"id": "mlp", "layer_type": "mlp", "params": {{"hidden_size": 4096, "intermediate_size": 11008}}}}
                ],
                "global_params": {{ "hidden_size": 4096 }}
            }},
            "training": {{ "batch_size": 1, "precision": "{precision}", "optimizer": "none" }},
            "hardware": {{ "gpus": [{{ "name": "A100-80GB", "count": 1 }}] }},
            "data": {{ "dtype": "{precision}" }}
        }}
        "#
    )
}

#[test]
fn int4_is_half_the_memory_of_int8() {
    let int8 = analyze_json(&model_json("int8")).expect("analysis should succeed");
    let int4 = analyze_json(&model_json("int4")).expect("analysis should succeed");

    let int8_bytes = int8.memory.metrics.parameter_memory_bytes;
    let int4_bytes = int4.memory.metrics.parameter_memory_bytes;

    assert!(int8_bytes > 0);
    assert_eq!(
        int4_bytes,
        int8_bytes / 2,
        "int4 should store exactly half the bytes int8 does for the same parameter count"
    );
}

/// int4 activations were *larger* than int8 on every one of the 106 reference
/// templates — 0.54-0.99x fp32 where 0.125x is correct.
///
/// `dtype_bytes` was only half the storage-width story: `Shape::size_bytes`
/// kept a second, hand-rolled table with no `int4` arm and an integer
/// `bytes_per_elem` that could not have held one. Output tensors took `_ => 4`
/// (fp32 width) while input tensors, sized through `dtype_bytes` on the lines
/// above, correctly took 0.5 — so a design "quantized to int4 to fit smaller
/// hardware" reported activation memory close to its fp32 figure.
#[test]
fn int4_activations_are_half_of_int8_not_larger() {
    let at = |precision: &str| {
        let json = model_json(precision);
        let report = analyze_json(&json).expect("analysis should succeed");
        (
            report.memory.metrics.activation_memory_bytes,
            report.tensor.metrics.largest_tensor_bytes,
        )
    };

    let (act32, big32) = at("fp32");
    let (act8, big8) = at("int8");
    let (act4, big4) = at("int4");

    assert!(act32 > 0 && big32 > 0, "the fixture must produce activations");

    // int8 is one byte per element, int4 half of one.
    assert_eq!(act8 * 4, act32, "int8 activations should be a quarter of fp32");
    assert_eq!(
        act4 * 8,
        act32,
        "int4 activations should be an eighth of fp32, got {act4} against {act32}"
    );
    assert!(
        act4 < act8,
        "int4 must not need more activation memory than int8 ({act4} vs {act8})"
    );
    assert_eq!(big4 * 2, big8, "the largest tensor should halve from int8 to int4");
}
