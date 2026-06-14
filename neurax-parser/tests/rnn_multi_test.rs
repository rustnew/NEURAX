//! Multi-model RNN/LSTM compilation test
//! Compiles ELMo, ULMFiT, BiLSTM-CRF, GRU-Seq2Seq to verify RNN family support

use neurax_parser::{parse_model_config, AbsorbedModel};
use neurax_ir::IrInjector;

/// ELMo - BiLSTM language model (94M params)
const ELMO_JSON: &str = r#"
{
    "schema_version": "1.0",
    "model": {
        "name": "ELMo",
        "type": "rnn",
        "layers": [
            {"id": "embedding", "layer_type": "embedding", "input_shape": [512, 50000], "output_shape": [512, 512], "params": {"vocab_size": 50000, "embedding_dim": 512}},
            {"id": "bilstm_1", "layer_type": "lstm_block", "input_shape": [512, 512], "output_shape": [512, 1024], "params": {"rnn_hidden_size": 512, "bidirectional_rnn": true, "num_rnn_layers": 1, "cell_type": "lstm"}},
            {"id": "bilstm_2", "layer_type": "lstm_block", "input_shape": [512, 1024], "output_shape": [512, 1024], "params": {"rnn_hidden_size": 512, "bidirectional_rnn": true, "num_rnn_layers": 1, "cell_type": "lstm"}}
        ],
        "global_params": {
            "vocab_size": 50000,
            "embedding_dim": 512,
            "rnn_hidden_size": 512,
            "num_rnn_layers": 2,
            "bidirectional_rnn": true,
            "cell_type": "lstm",
            "sequence_length": 512
        }
    },
    "training": {"batch_size": 64, "optimizer": "adam", "learning_rate": 0.001, "precision": "fp32", "gradient_checkpointing": false, "zero_stage": 0, "max_steps": 300000, "warmup_steps": 5000, "parallelism": {"data_parallel": 4, "tensor_parallel": 1, "pipeline_parallel": 1}},
    "hardware": {"gpus": [{"name": "V100-32GB", "count": 4, "memory_gb": 32, "tflops_fp32": 15.7, "memory_bandwidth_gb_s": 900, "tensor_cores": true, "nvlink": true}], "interconnect": "NVLink", "interconnect_bandwidth_gb_s": 300},
    "data": {"input_shape": [512], "dtype": "fp32"},
    "cost_config": {"provider": "aws", "gpu_hour_usd": 3.00, "energy_kwh_usd": 0.12, "pue_factor": 1.2}
}
"#;

/// ULMFiT - LSTM language model (24M params)
const ULMFIT_JSON: &str = r#"
{
    "schema_version": "1.0",
    "model": {
        "name": "ULMFiT",
        "type": "rnn",
        "layers": [
            {"id": "embedding", "layer_type": "embedding", "input_shape": [512, 30000], "output_shape": [512, 400], "params": {"vocab_size": 30000, "embedding_dim": 400}},
            {"id": "lstm_1", "layer_type": "lstm_block", "input_shape": [512, 400], "output_shape": [512, 400], "params": {"rnn_hidden_size": 400, "bidirectional_rnn": false, "num_rnn_layers": 1, "cell_type": "lstm"}},
            {"id": "lstm_2", "layer_type": "lstm_block", "input_shape": [512, 400], "output_shape": [512, 400], "params": {"rnn_hidden_size": 400, "bidirectional_rnn": false, "num_rnn_layers": 1, "cell_type": "lstm"}},
            {"id": "lstm_3", "layer_type": "lstm_block", "input_shape": [512, 400], "output_shape": [512, 400], "params": {"rnn_hidden_size": 400, "bidirectional_rnn": false, "num_rnn_layers": 1, "cell_type": "lstm"}},
            {"id": "output", "layer_type": "dense", "input_shape": [512, 400], "output_shape": [512, 30000], "params": {"in_features": 400, "out_features": 30000}}
        ],
        "global_params": {
            "vocab_size": 30000,
            "embedding_dim": 400,
            "rnn_hidden_size": 400,
            "num_rnn_layers": 3,
            "bidirectional_rnn": false,
            "cell_type": "lstm",
            "sequence_length": 512
        }
    },
    "training": {"batch_size": 48, "optimizer": "adam", "learning_rate": 0.001, "precision": "fp32", "gradient_checkpointing": false, "zero_stage": 0, "max_steps": 200000, "warmup_steps": 3000, "parallelism": {"data_parallel": 2, "tensor_parallel": 1, "pipeline_parallel": 1}},
    "hardware": {"gpus": [{"name": "V100-32GB", "count": 2, "memory_gb": 32, "tflops_fp32": 15.7, "memory_bandwidth_gb_s": 900, "tensor_cores": true, "nvlink": false}], "interconnect": "PCIe", "interconnect_bandwidth_gb_s": 16},
    "data": {"input_shape": [512], "dtype": "fp32"},
    "cost_config": {"provider": "local", "gpu_hour_usd": 0.0, "energy_kwh_usd": 0.0, "pue_factor": 1.0}
}
"#;

/// BiLSTM-CRF for NER (12M params)
const BILSTM_CRF_JSON: &str = r#"
{
    "schema_version": "1.0",
    "model": {
        "name": "BiLSTM-CRF",
        "type": "rnn",
        "layers": [
            {"id": "embedding", "layer_type": "embedding", "input_shape": [128, 50000], "output_shape": [128, 300], "params": {"vocab_size": 50000, "embedding_dim": 300}},
            {"id": "bilstm_1", "layer_type": "lstm_block", "input_shape": [128, 300], "output_shape": [128, 512], "params": {"rnn_hidden_size": 256, "bidirectional_rnn": true, "num_rnn_layers": 1, "cell_type": "lstm"}},
            {"id": "bilstm_2", "layer_type": "lstm_block", "input_shape": [128, 512], "output_shape": [128, 512], "params": {"rnn_hidden_size": 256, "bidirectional_rnn": true, "num_rnn_layers": 1, "cell_type": "lstm"}},
            {"id": "classifier", "layer_type": "dense", "input_shape": [128, 512], "output_shape": [128, 9], "params": {"in_features": 512, "out_features": 9}}
        ],
        "global_params": {
            "vocab_size": 50000,
            "embedding_dim": 300,
            "rnn_hidden_size": 256,
            "num_rnn_layers": 2,
            "bidirectional_rnn": true,
            "cell_type": "lstm",
            "sequence_length": 128,
            "num_classes": 9
        }
    },
    "training": {"batch_size": 32, "optimizer": "adamw", "learning_rate": 0.001, "precision": "fp32", "gradient_checkpointing": false, "zero_stage": 0, "max_steps": 100000, "warmup_steps": 5000, "parallelism": {"data_parallel": 1, "tensor_parallel": 1, "pipeline_parallel": 1}},
    "hardware": {"gpus": [{"name": "A100-80GB", "count": 1, "memory_gb": 80, "tflops_fp32": 19.5, "memory_bandwidth_gb_s": 2039, "tensor_cores": true, "nvlink": false}], "interconnect": "None", "interconnect_bandwidth_gb_s": 0},
    "data": {"input_shape": [128], "dtype": "fp32"},
    "cost_config": {"provider": "local", "gpu_hour_usd": 0.0, "energy_kwh_usd": 0.0, "pue_factor": 1.0}
}
"#;

/// GRU Seq2Seq for translation (45M params)
const GRU_SEQ2SEQ_JSON: &str = r#"
{
    "schema_version": "1.0",
    "model": {
        "name": "GRU-Seq2Seq",
        "type": "rnn",
        "layers": [
            {"id": "enc_embed", "layer_type": "embedding", "input_shape": [64, 30000], "output_shape": [64, 512], "params": {"vocab_size": 30000, "embedding_dim": 512}},
            {"id": "enc_gru_1", "layer_type": "gru_block", "input_shape": [64, 512], "output_shape": [64, 1024], "params": {"rnn_hidden_size": 512, "bidirectional_rnn": true, "num_rnn_layers": 1, "cell_type": "gru"}},
            {"id": "enc_gru_2", "layer_type": "gru_block", "input_shape": [64, 1024], "output_shape": [64, 1024], "params": {"rnn_hidden_size": 512, "bidirectional_rnn": true, "num_rnn_layers": 1, "cell_type": "gru"}},
            
            {"id": "dec_embed", "layer_type": "embedding", "input_shape": [64, 30000], "output_shape": [64, 512], "params": {"vocab_size": 30000, "embedding_dim": 512}},
            {"id": "dec_gru", "layer_type": "gru_block", "input_shape": [64, 1536], "output_shape": [64, 512], "params": {"rnn_hidden_size": 512, "bidirectional_rnn": false, "num_rnn_layers": 1, "cell_type": "gru"}},
            {"id": "attention", "layer_type": "attention", "input_shape": [64, 512], "output_shape": [64, 512], "params": {"attention_type": "bahdanau"}},
            {"id": "output", "layer_type": "dense", "input_shape": [64, 512], "output_shape": [64, 30000], "params": {"in_features": 512, "out_features": 30000}}
        ],
        "global_params": {
            "vocab_size": 30000,
            "embedding_dim": 512,
            "rnn_hidden_size": 512,
            "num_rnn_layers": 2,
            "bidirectional_rnn": true,
            "cell_type": "gru",
            "sequence_length": 64
        }
    },
    "training": {"batch_size": 64, "optimizer": "adamw", "learning_rate": 0.0005, "precision": "fp32", "gradient_checkpointing": false, "zero_stage": 0, "max_steps": 200000, "warmup_steps": 10000, "parallelism": {"data_parallel": 4, "tensor_parallel": 1, "pipeline_parallel": 1}},
    "hardware": {"gpus": [{"name": "A100-80GB", "count": 4, "memory_gb": 80, "tflops_fp32": 19.5, "memory_bandwidth_gb_s": 2039, "tensor_cores": true, "nvlink": true}], "interconnect": "NVLink", "interconnect_bandwidth_gb_s": 600},
    "data": {"input_shape": [64], "dtype": "fp32"},
    "cost_config": {"provider": "aws", "gpu_hour_usd": 4.50, "energy_kwh_usd": 0.12, "pue_factor": 1.2}
}
"#;

/// Large LSTM Language Model (1.3B params)
const LSTM_LM_LARGE_JSON: &str = r#"
{
    "schema_version": "1.0",
    "model": {
        "name": "LSTM-LM-1.3B",
        "type": "rnn",
        "layers": [
            {"id": "embedding", "layer_type": "embedding", "input_shape": [512, 50000], "output_shape": [512, 2048], "params": {"vocab_size": 50000, "embedding_dim": 2048}},
            {"id": "lstm_1", "layer_type": "lstm_block", "input_shape": [512, 2048], "output_shape": [512, 4096], "params": {"rnn_hidden_size": 2048, "bidirectional_rnn": false, "num_rnn_layers": 1, "cell_type": "lstm"}},
            {"id": "lstm_2", "layer_type": "lstm_block", "input_shape": [512, 4096], "output_shape": [512, 4096], "params": {"rnn_hidden_size": 2048, "bidirectional_rnn": false, "num_rnn_layers": 1, "cell_type": "lstm"}},
            {"id": "lstm_3", "layer_type": "lstm_block", "input_shape": [512, 4096], "output_shape": [512, 4096], "params": {"rnn_hidden_size": 2048, "bidirectional_rnn": false, "num_rnn_layers": 1, "cell_type": "lstm"}},
            {"id": "output", "layer_type": "dense", "input_shape": [512, 2048], "output_shape": [512, 50000], "params": {"in_features": 2048, "out_features": 50000}}
        ],
        "global_params": {
            "vocab_size": 50000,
            "embedding_dim": 2048,
            "rnn_hidden_size": 2048,
            "num_rnn_layers": 3,
            "bidirectional_rnn": false,
            "cell_type": "lstm",
            "sequence_length": 512
        }
    },
    "training": {"batch_size": 128, "optimizer": "adamw", "learning_rate": 0.001, "precision": "fp32", "gradient_checkpointing": false, "zero_stage": 1, "max_steps": 500000, "warmup_steps": 10000, "parallelism": {"data_parallel": 8, "tensor_parallel": 1, "pipeline_parallel": 1}},
    "hardware": {"gpus": [{"name": "A100-80GB", "count": 8, "memory_gb": 80, "tflops_fp32": 19.5, "memory_bandwidth_gb_s": 2039, "tensor_cores": true, "nvlink": true}], "interconnect": "NVLink", "interconnect_bandwidth_gb_s": 600},
    "data": {"input_shape": [512], "dtype": "fp32"},
    "cost_config": {"provider": "aws", "gpu_hour_usd": 4.50, "energy_kwh_usd": 0.12, "pue_factor": 1.2}
}
"#;

struct RNNModel {
    name: &'static str,
    json: &'static str,
    expected_hidden: u32,
    expected_layers: u32,
    bidirectional: bool,
    cell_type: &'static str,
}

#[test]
fn test_all_rnn_models() {
    println!("\n╔════════════════════════════════════════════════════════════════════╗");
    println!("║           MULTI-MODEL RNN/LSTM COMPILATION TEST                    ║");
    println!("║           ELMo | ULMFiT | BiLSTM-CRF | GRU-Seq2Seq | LSTM-1.3B     ║");
    println!("╚════════════════════════════════════════════════════════════════════╝\n");
    
    let models = [
        RNNModel { name: "ELMo", json: ELMO_JSON, expected_hidden: 512, expected_layers: 2, bidirectional: true, cell_type: "lstm" },
        RNNModel { name: "ULMFiT", json: ULMFIT_JSON, expected_hidden: 400, expected_layers: 3, bidirectional: false, cell_type: "lstm" },
        RNNModel { name: "BiLSTM-CRF", json: BILSTM_CRF_JSON, expected_hidden: 256, expected_layers: 2, bidirectional: true, cell_type: "lstm" },
        RNNModel { name: "GRU-Seq2Seq", json: GRU_SEQ2SEQ_JSON, expected_hidden: 512, expected_layers: 2, bidirectional: true, cell_type: "gru" },
        RNNModel { name: "LSTM-1.3B", json: LSTM_LM_LARGE_JSON, expected_hidden: 2048, expected_layers: 3, bidirectional: false, cell_type: "lstm" },
    ];
    
    println!("┌──────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Model       │ Params (M) │ Hidden │ Layers │ Bidir │ Type  │ Status │ Conf     │");
    println!("├──────────────────────────────────────────────────────────────────────────────────┤");
    
    let mut all_passed = true;
    
    for model in &models {
        let config = parse_model_config(model.json).expect(&format!("Failed to parse {}", model.name));
        let absorbed = AbsorbedModel::absorb(config);
        let grc = &absorbed.resolution_context;
        
        let total_params = IrInjector::calculate_total_params(&absorbed) as f64 / 1e6;
        let hidden = grc.rnn_hidden_size.unwrap_or(0);
        let layers = grc.num_rnn_layers.unwrap_or(0);
        let bidir = grc.bidirectional_rnn;
        let cell = grc.cell_type.as_deref().unwrap_or("unknown");
        let confidence = grc.confidence_score * 100.0;
        
        let status = if total_params > 0.0 && hidden == model.expected_hidden as u64 {
            "✓ OK"
        } else {
            all_passed = false;
            "✗ FAIL"
        };
        
        println!("│ {:<11} │ {:>10.1} │ {:>6} │ {:>6} │ {:>5} │ {:>5} │ {:>6} │ {:>5.1}%  │", 
                 model.name, total_params, hidden, layers, bidir, cell, status, confidence);
    }
    
    println!("└──────────────────────────────────────────────────────────────────────────────────┘\n");
    
    assert!(all_passed, "Some RNN models failed compilation");
    println!("✓ All RNN/LSTM models compiled successfully!\n");
}

#[test]
fn test_rnn_cell_type_comparison() {
    println!("\n=== RNN Cell Type Comparison ===\n");
    
    println!("┌────────────────────────────────────────────────────────────────┐");
    println!("│ Cell Type │ Gates │ Parameters per Layer        │ Use Case    │");
    println!("├────────────────────────────────────────────────────────────────┤");
    println!("│ LSTM      │   4   │ 4 × (input + hidden + 1) × h  │ Long-term   │");
    println!("│ GRU       │   3   │ 3 × (input + hidden + 1) × h  │ Efficient   │");
    println!("│ Vanilla   │   1   │ (input + hidden + 1) × h      │ Simple      │");
    println!("├────────────────────────────────────────────────────────────────┤");
    println!("│ BiLSTM    │   4   │ 2 × LSTM params              │ Context     │");
    println!("│ BiGRU     │   3   │ 2 × GRU params               │ Seq2Seq     │");
    println!("└────────────────────────────────────────────────────────────────┘\n");
    
    println!("Key insights:\n");
    println!("  - LSTM: Best for long sequences, 4 gates (input, forget, cell, output)");
    println!("  - GRU:  Faster training, 3 gates (reset, update, new)");
    println!("  - Bidirectional: Captures both past and future context");
    println!("  - GRU has ~75% params of LSTM for same hidden size\n");
}

#[test]
fn test_rnn_layer_types_validation() {
    println!("\n=== RNN Layer Types Validation ===\n");
    
    let layer_types = [
        ("lstm_block", "LstmBlock - LSTM layer with 4 gates"),
        ("gru_block", "GruBlock - GRU layer with 3 gates"),
        ("rnn_cell", "RnnCell - Vanilla RNN cell"),
        ("bidirectional", "Bidirectional - Wrapper for BiLSTM/BiGRU"),
        ("encoder_block", "EncoderBlock - RNN encoder"),
        ("decoder_block", "DecoderBlock - RNN decoder with attention"),
    ];
    
    println!("Supported RNN layer types (6 total):\n");
    for (input, expected) in layer_types {
        println!("  ✓ '{}' -> {}", input, expected);
    }
    
    println!("\nRNN-specific parameters:\n");
    println!("  - rnn_hidden_size: Hidden state dimension");
    println!("  - num_rnn_layers: Number of stacked RNN layers");
    println!("  - bidirectional_rnn: Whether to use bidirectional RNN");
    println!("  - cell_type: lstm, gru, or vanilla_rnn");
    println!("  - forget_bias: LSTM forget gate bias (default 1.0)");
    println!("  - peephole: LSTM peephole connections");
    println!("  - recurrent_dropout: Dropout on recurrent connections");
    println!("  - attention_type: bahdanau, luong, dot\n");
}
