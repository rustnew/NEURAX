//! Regression tests for the "Optimize" pillar's flagship output —
//! `generate_recommendations()` — whose impact estimates used to be
//! hardcoded strings unrelated to the model actually analyzed ("Save ~70%
//! on GPU costs", "Improve efficiency to ~90%", "Potential 2-3x speedup"),
//! rather than values derived from that model's own metrics. A 4-layer
//! model and a 96-layer one received the identical "gradient checkpointing"
//! savings estimate despite the real technique scaling very differently
//! with depth.
//!
//! Each test here proves a specific recommendation's number now reacts to
//! the model it was generated for, rather than being a fixed constant.

fn model_json(num_layers: u32, batch_size: u32) -> String {
    format!(
        r#"{{
            "schema_version": "1.0.0",
            "model": {{ "name": "test", "type": "transformer",
                "global_params": {{
                    "num_layers": {num_layers}, "hidden_size": 4096, "num_heads": 32,
                    "intermediate_size": 16384, "vocab_size": 32000, "sequence_length": 4096
                }},
                "layers": [
                    {{"id": "embed", "layer_type": "embedding", "params": {{"vocab_size": 32000, "hidden_size": 4096}}}},
                    {{"id": "attn", "layer_type": "attention", "params": {{"hidden_size": 4096, "num_attention_heads": 32, "intermediate_size": 16384}}}},
                    {{"id": "mlp", "layer_type": "mlp", "params": {{"hidden_size": 4096, "intermediate_size": 16384}}}}
                ]
            }},
            "training": {{"batch_size": {batch_size}, "sequence_length": 4096, "precision": "bf16", "max_steps": 100}},
            "hardware": {{"gpus": [{{"name": "A100-SXM", "count": 1}}]}},
            "data": {{"dataset_size": 1000000000, "vocab_size": 32000, "num_classes": 0}}
        }}"#
    )
}

fn analyze(num_layers: u32, batch_size: u32) -> neurax_core::AnalysisResult {
    let config =
        neurax_parser::parse_model_config(&model_json(num_layers, batch_size)).expect("parses");
    neurax_core::run_analysis(config).expect("analyzes")
}

fn model_json_with_gpus(num_layers: u32, num_gpus: u32) -> String {
    format!(
        r#"{{
            "schema_version": "1.0.0",
            "model": {{ "name": "test", "type": "transformer",
                "global_params": {{
                    "num_layers": {num_layers}, "hidden_size": 4096, "num_heads": 32,
                    "intermediate_size": 16384, "vocab_size": 32000, "sequence_length": 4096
                }},
                "layers": [
                    {{"id": "embed", "layer_type": "embedding", "params": {{"vocab_size": 32000, "hidden_size": 4096}}}},
                    {{"id": "attn", "layer_type": "attention", "params": {{"hidden_size": 4096, "num_attention_heads": 32, "intermediate_size": 16384}}}},
                    {{"id": "mlp", "layer_type": "mlp", "params": {{"hidden_size": 4096, "intermediate_size": 16384}}}}
                ]
            }},
            "training": {{"batch_size": 32, "sequence_length": 4096, "precision": "bf16", "max_steps": 100}},
            "hardware": {{"gpus": [{{"name": "A100-SXM", "count": {num_gpus}}}]}},
            "data": {{"dataset_size": 1000000000, "vocab_size": 32000, "num_classes": 0}}
        }}"#
    )
}

fn analyze_with_gpus(num_layers: u32, num_gpus: u32) -> neurax_core::AnalysisResult {
    let config = neurax_parser::parse_model_config(&model_json_with_gpus(num_layers, num_gpus))
        .expect("parses");
    neurax_core::run_analysis(config).expect("analyzes")
}

fn checkpointing_recommendation(r: &neurax_core::AnalysisResult) -> Option<String> {
    r.report
        .recommendations
        .iter()
        .find(|rec| rec.title == "Enable Gradient Checkpointing")
        .map(|rec| rec.impact.clone())
}

fn hybrid_parallelism_recommendation(r: &neurax_core::AnalysisResult) -> Option<String> {
    r.report
        .recommendations
        .iter()
        .find(|rec| rec.title == "Use Hybrid Parallelism")
        .map(|rec| rec.impact.clone())
}

fn bandwidth_json(gpu_name: &str) -> String {
    format!(
        r#"{{
            "schema_version": "1.0.0",
            "model": {{ "name": "test", "type": "transformer",
                "global_params": {{
                    "num_layers": 4, "hidden_size": 512, "num_heads": 8,
                    "intermediate_size": 2048, "vocab_size": 500000, "sequence_length": 128
                }},
                "layers": [
                    {{"id": "embed", "layer_type": "embedding", "params": {{"vocab_size": 500000, "hidden_size": 512}}}}
                ]
            }},
            "training": {{"batch_size": 1, "sequence_length": 128, "precision": "fp16", "max_steps": 1}},
            "hardware": {{"gpus": [{{"name": "{gpu_name}", "count": 1}}]}},
            "data": {{"dataset_size": 1000000, "vocab_size": 500000, "num_classes": 0}}
        }}"#
    )
}

fn bandwidth_recommendation_impact(gpu_name: &str) -> Option<String> {
    let config = neurax_parser::parse_model_config(&bandwidth_json(gpu_name)).expect("parses");
    let r = neurax_core::run_analysis(config).expect("analyzes");
    r.report
        .recommendations
        .iter()
        .find(|rec| rec.title == "Consider Higher Bandwidth GPU")
        .map(|rec| rec.impact.clone())
}

#[test]
fn gradient_checkpointing_savings_scale_with_real_depth_not_a_flat_constant() {
    // Both models are pushed over the 80% VRAM threshold (same batch), but
    // differ only in depth. The real technique (Chen et al. 2016) frees
    // 1 - 1/sqrt(L) of activation memory — a shallow model should show a
    // meaningfully smaller savings percentage than a deep one, not the
    // same flat estimate.
    let shallow = analyze(16, 128);
    let deep = analyze(96, 128);

    let shallow_impact = checkpointing_recommendation(&shallow)
        .expect("shallow model should trigger the memory recommendation");
    let deep_impact = checkpointing_recommendation(&deep)
        .expect("deep model should trigger the memory recommendation");

    assert_ne!(
        shallow_impact, deep_impact,
        "a 16-layer and a 96-layer model got the identical gradient-checkpointing \
         savings estimate — the real sqrt(L) technique should scale with depth"
    );

    // 16 layers: 1 - 1/sqrt(16) = 75%. 96 layers: 1 - 1/sqrt(96) ~= 89.8%.
    assert!(
        shallow_impact.contains("75%"),
        "expected ~75% for a 16-layer model, got: {shallow_impact}"
    );
    assert!(
        deep_impact.contains("90%"),
        "expected ~90% for a 96-layer model, got: {deep_impact}"
    );
}

#[test]
fn hybrid_parallelism_target_reacts_to_the_models_own_current_efficiency() {
    // Two GPU counts that land on different current data_parallel_efficiency
    // values (calculate_dp_efficiency, parallelism/pass.rs) should produce
    // different "from X% to Y%" reports, not the same flat "~90%" target
    // regardless of where the model actually starts.
    let low_gpus = analyze_with_gpus(32, 64);
    let high_gpus = analyze_with_gpus(32, 512);

    let low_impact = hybrid_parallelism_recommendation(&low_gpus)
        .expect("64-GPU config should trigger the parallelism recommendation");
    let high_impact = hybrid_parallelism_recommendation(&high_gpus)
        .expect("512-GPU config should trigger the parallelism recommendation");

    assert_ne!(
        low_impact, high_impact,
        "different GPU counts produced different current efficiencies but the \
         identical recommendation text — the target should be computed from \
         each model's own starting point, not a flat constant"
    );
    // The target must never be reported below the model's own current
    // efficiency (a real risk with a flat absolute target like the old
    // "~90%": nonsensical for a model already above 90%).
    for impact in [&low_impact, &high_impact] {
        assert!(
            impact.starts_with("Improve efficiency from ~"),
            "expected the current efficiency to be reported alongside the \
             target, got: {impact}"
        );
    }
}

#[test]
fn higher_bandwidth_gpu_speedup_is_the_real_bandwidth_ratio() {
    // A memory-bound model on a slower GPU (V100, 900 GB/s) should see a
    // larger claimed speedup than the identical model on a faster one
    // (A100-SXM, 2039 GB/s) — both measured against whichever GPU the
    // database actually reports as fastest, not a hardcoded "H100, 2-3x".
    let v100_impact = bandwidth_recommendation_impact("V100")
        .expect("V100 should be memory-bound for this tiny-compute, huge-embedding model");
    let a100_impact = bandwidth_recommendation_impact("A100-SXM")
        .expect("A100-SXM should also be memory-bound for this model");

    assert_ne!(
        v100_impact, a100_impact,
        "a slower and a faster starting GPU produced the identical speedup \
         claim — it should scale with the real bandwidth ratio"
    );

    let extract_speedup = |s: &str| -> f64 {
        let after = s.split('~').nth(1).expect("has a ~X.Yx figure");
        after
            .split('x')
            .next()
            .expect("has an x suffix")
            .parse()
            .expect("parses as a float")
    };
    let v100_speedup = extract_speedup(&v100_impact);
    let a100_speedup = extract_speedup(&a100_impact);
    assert!(
        v100_speedup > a100_speedup,
        "V100 (900 GB/s) is further from the fastest database entry than \
         A100-SXM (2039 GB/s), so its claimed speedup should be larger: \
         got v100={v100_speedup}, a100={a100_speedup}"
    );
}

fn cost_json(max_steps: u32) -> String {
    format!(
        r#"{{
            "schema_version": "1.0.0",
            "model": {{ "name": "test", "type": "transformer",
                "global_params": {{
                    "num_layers": 32, "hidden_size": 4096, "num_heads": 32,
                    "intermediate_size": 16384, "vocab_size": 32000, "sequence_length": 2048
                }},
                "layers": [
                    {{"id": "embed", "layer_type": "embedding", "params": {{"vocab_size": 32000, "hidden_size": 4096}}}},
                    {{"id": "attn", "layer_type": "attention", "params": {{"hidden_size": 4096, "num_attention_heads": 32, "intermediate_size": 16384}}}},
                    {{"id": "mlp", "layer_type": "mlp", "params": {{"hidden_size": 4096, "intermediate_size": 16384}}}}
                ]
            }},
            "training": {{"batch_size": 8, "sequence_length": 2048, "precision": "bf16", "max_steps": {max_steps}}},
            "hardware": {{"gpus": [{{"name": "A100-SXM", "count": 8}}]}},
            "data": {{"dataset_size": 1000000000, "vocab_size": 32000, "num_classes": 0}}
        }}"#
    )
}

fn cost_recommendation(max_steps: u32) -> (f64, Option<String>) {
    let config = neurax_parser::parse_model_config(&cost_json(max_steps)).expect("parses");
    let r = neurax_core::run_analysis(config).expect("analyzes");
    let impact = r
        .report
        .recommendations
        .iter()
        .find(|rec| rec.title == "Optimize Training Duration")
        .map(|rec| rec.impact.clone());
    (r.cost.metrics.training_cost_usd, impact)
}

#[test]
fn spot_instance_savings_scale_with_the_models_real_training_cost() {
    // Below the $10k trigger: no recommendation at all.
    let (small_cost, small_impact) = cost_recommendation(500_000);
    assert!(
        small_cost < 10_000.0,
        "expected a small run under the trigger, got ${small_cost}"
    );
    assert!(
        small_impact.is_none(),
        "a run under the $10k trigger should not get this recommendation, got: {small_impact:?}"
    );

    // Above it: the dollar figure must reflect this specific run's real
    // cost (~60% of it, the conservative spot-discount constant), not a
    // flat "up to 70%" untethered to any amount.
    let (big_cost, big_impact) = cost_recommendation(5_000_000);
    assert!(
        big_cost > 10_000.0,
        "expected a run over the trigger, got ${big_cost}"
    );
    let big_impact = big_impact.expect("a run over $10k should get this recommendation");

    let expected_savings = big_cost * 0.60;
    assert!(
        big_impact.contains(&format!("{:.0}", expected_savings)),
        "expected the savings figure to be ~60% of this run's real ${big_cost:.0} \
         cost (~${expected_savings:.0}), got: {big_impact}"
    );
}

#[test]
fn already_on_the_fastest_gpu_gets_no_bandwidth_recommendation() {
    // Recommending a switch to a GPU that isn't actually faster than the
    // one already configured is a nonsensical suggestion. Whatever the
    // database's current fastest entry is, a model already on it must not
    // receive this recommendation.
    //
    // Asked of the database rather than named here. This test used to hardcode
    // "GH200" while claiming to be about "whatever the current fastest entry
    // is" — so adding the B200 (7.7 TB/s against the GH200's 4.8) broke it,
    // and the break said nothing about the recommendation being wrong. The
    // property is real; the name was never part of it.
    let db = neurax_hardware_db::HardwareDatabase::new();
    let fastest = db
        .list_gpus()
        .into_iter()
        .max_by(|a, b| {
            a.memory_bandwidth_gbs
                .partial_cmp(&b.memory_bandwidth_gbs)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("the database ships with GPUs in it");

    let impact = bandwidth_recommendation_impact(&fastest.name);
    assert!(
        impact.is_none(),
        "expected no bandwidth recommendation when already on {} ({} GB/s), the \
         fastest GPU in the database, got: {impact:?}",
        fastest.name,
        fastest.memory_bandwidth_gbs,
    );
}
