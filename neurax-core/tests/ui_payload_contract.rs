use neurax_core::run_analysis;

#[test]
fn frontend_topology_payload_still_produces_ui_report_json() {
    let json = r#"{
        "schema_version": "1.0",
        "model": {
            "name": "UIContractTest",
            "type": "transformer",
            "global_params": {
                "hidden_size": 768,
                "num_layers": 2,
                "vocab_size": 32000,
                "sequence_length": 128,
                "num_heads": 12,
                "ffn_dim": 3072
            },
            "layers": [
                {
                    "id": "embed",
                    "layer_type": "embedding",
                    "input_shape": [1, 128],
                    "output_shape": [1, 128, 768],
                    "params": {
                        "vocab_size": 32000,
                        "embedding_dim": 768
                    }
                },
                {
                    "id": "attn",
                    "layer_type": "attention",
                    "input_shape": [1, 128, 768],
                    "output_shape": [1, 128, 768],
                    "params": {
                        "d_model": 768,
                        "num_heads": 12
                    }
                },
                {
                    "id": "ffn",
                    "layer_type": "mlp",
                    "input_shape": [1, 128, 768],
                    "output_shape": [1, 128, 768],
                    "params": {
                        "hidden_size": 768,
                        "intermediate_size": 3072,
                        "activation": "gelu"
                    }
                },
                {
                    "id": "norm",
                    "layer_type": "normalization",
                    "input_shape": [1, 128, 768],
                    "output_shape": [1, 128, 768],
                    "params": {
                        "hidden_size": 768
                    }
                },
                {
                    "id": "head",
                    "layer_type": "dense",
                    "input_shape": [1, 128, 768],
                    "output_shape": [1, 128, 32000],
                    "params": {
                        "in_features": 768,
                        "out_features": 32000,
                        "vocab_size": 32000
                    }
                }
            ]
        },
        "training": {
            "batch_size": 1,
            "precision": "fp16"
        },
        "hardware": {
            "gpus": [
                {
                    "name": "RTX4090",
                    "count": 1
                }
            ]
        }
    }"#;

    let config = neurax_parser::parse_model_config(json).expect("frontend payload should parse");
    let analysis = run_analysis(config).expect("frontend payload should analyze");

    assert_eq!(analysis.report.metadata.model_name, "UIContractTest");
    assert!(analysis.report.metrics.total_parameters > 0);
    assert!(analysis.report.metrics.total_flops > 0.0);

    let report_json = serde_json::to_value(&analysis.report).expect("report should serialize for the service response");
    assert!(report_json.get("metrics").is_some());
    assert!(report_json.get("diagnostics").is_some());
    assert!(report_json.get("warnings").is_some());
}
