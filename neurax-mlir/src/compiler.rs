//! High-level NEURAX → MLIR compilation entry point.
//!
//! Exposes [`compile_model_to_mlir`], which lowers a parsed
//! [`neurax_parser::ModelConfig`] into textual MLIR using the NEURAX
//! dialects. This is the library API consumed by `neurax-cli`'s
//! `compile` command so the full MLIR backend is wired into the
//! `neurax_full` pipeline.

use melior::ir::Location;

use crate::dialects::{
    ArchitectureDialect, ComputeDialect, CostDialect, HardwareDialect, MemoryDialect,
    OperatorDialect, ParallelismDialect, ReportDialect, TensorDialect,
};
use neurax_parser::{LayerType, ModelConfig};

/// Compile a parsed model configuration to textual MLIR.
///
/// Produces one MLIR operation per architectural element (model, layers,
/// operators, hardware, parallelism, compute, memory, cost, report) using
/// the NEURAX MLIR dialects, then concatenates their textual forms.
pub fn compile_model_to_mlir(
    context: &melior::Context,
    config: &ModelConfig,
) -> Result<String, String> {
    let mut operations = Vec::new();
    let location = Location::unknown(context);

    // 1. Architecture model operation
    let model_name = config.model.name.as_deref().unwrap_or("unnamed_model");
    let model_type = config.model.model_type.as_str();

    let model_op = ArchitectureDialect::model(context, model_name, model_type, location)
        .map_err(|e| format!("Failed to create model op: {e:?}"))?;
    operations.push(model_op.to_string());

    // 2. Global params
    let global_params = &config.model.global_params;
    let num_layers = global_params
        .num_layers
        .unwrap_or(config.model.layers.len() as u64) as i64;

    let mut params_vec: Vec<(&str, i64)> = Vec::new();
    if let Some(seq_len) = global_params.sequence_length {
        params_vec.push(("sequence_length", seq_len as i64));
    }
    if let Some(vocab) = global_params.vocab_size {
        params_vec.push(("vocab_size", vocab as i64));
    }
    if let Some(embed_dim) = global_params.embedding_dim {
        params_vec.push(("embedding_dim", embed_dim as i64));
    }

    if !params_vec.is_empty() {
        let global_op = ArchitectureDialect::global_params(context, &params_vec, location)
            .map_err(|e| format!("Failed to create global_params op: {e:?}"))?;
        operations.push(global_op.to_string());
    }

    // 3. Layer operations
    for layer in &config.model.layers {
        let layer_op =
            ArchitectureDialect::layer(context, &layer.id, layer.layer_type.as_str(), location)
                .map_err(|e| format!("Failed to create layer op for {}: {e:?}", layer.id))?;
        operations.push(layer_op.to_string());

        match layer.layer_type {
            LayerType::Attention => {
                let hidden_size = layer.params.hidden_size.unwrap_or(768) as i64;
                let num_heads = layer.params.num_heads.unwrap_or(12) as i64;

                let param_count = hidden_size * hidden_size;
                let flops = 4.0 * (hidden_size as f64).powi(2);

                let attn_op = OperatorDialect::attention(
                    context,
                    hidden_size,
                    num_heads,
                    param_count,
                    flops,
                    location,
                )
                .map_err(|e| format!("Failed to create attention op: {e:?}"))?;
                operations.push(attn_op.to_string());
            }
            LayerType::Mlp => {
                let hidden_size = layer.params.hidden_size.unwrap_or(768) as i64;
                let intermediate_size = layer.params.intermediate_size.unwrap_or(3072) as i64;

                let param_count = hidden_size * intermediate_size * 3;
                let flops = 3.0 * hidden_size as f64 * intermediate_size as f64;

                let mlp_op = OperatorDialect::matmul(context, param_count, flops, location)
                    .map_err(|e| format!("Failed to create mlp op: {e:?}"))?;
                operations.push(mlp_op.to_string());
            }
            LayerType::MoE => {
                let hidden_size = layer.params.hidden_size.unwrap_or(768) as i64;
                let num_experts = layer.params.num_experts.unwrap_or(8) as i64;
                let top_k = layer.params.top_k.unwrap_or(2) as i64;

                let param_count = hidden_size * hidden_size * num_experts;
                let flops = 2.0 * hidden_size as f64 * hidden_size as f64 * top_k as f64;

                let moe_op = OperatorDialect::moe(
                    context,
                    hidden_size,
                    num_experts,
                    top_k,
                    param_count,
                    flops,
                    location,
                )
                .map_err(|e| format!("Failed to create moe op: {e:?}"))?;
                operations.push(moe_op.to_string());
            }
            LayerType::Embedding => {
                let vocab_size = layer.params.vocab_size.unwrap_or(50000) as i64;
                let embed_dim = layer.params.embedding_dim.unwrap_or(768) as i64;

                let tensor_shape = vec![vocab_size, embed_dim];
                let tensor_op = TensorDialect::tensor_info(
                    context,
                    &format!("{}_weights", layer.id),
                    &tensor_shape,
                    "f32",
                    vocab_size * embed_dim * 4,
                    &layer.id,
                    location,
                )
                .map_err(|e| format!("Failed to create tensor op: {e:?}"))?;
                operations.push(tensor_op.to_string());
            }
            LayerType::Dense => {
                let in_features = layer.params.hidden_size.unwrap_or(768) as i64;
                let out_features = layer
                    .params
                    .intermediate_size
                    .unwrap_or(layer.output_shape.last().copied().unwrap_or(768) as usize)
                    as i64;

                let param_count = in_features * out_features;
                let flops = 2.0 * in_features as f64 * out_features as f64;

                let dense_op = OperatorDialect::matmul(context, param_count, flops, location)
                    .map_err(|e| format!("Failed to create dense op: {e:?}"))?;
                operations.push(dense_op.to_string());
            }
            _ => {}
        }
    }

    // 4. Architecture metrics
    let total_params = calculate_total_params(config);
    let metrics_op = ArchitectureDialect::metrics(context, total_params, num_layers, location)
        .map_err(|e| format!("Failed to create metrics op: {e:?}"))?;
    operations.push(metrics_op.to_string());

    // 5. Hardware configuration
    if let Some(gpu) = config.hardware.primary_gpu() {
        let hw_op = HardwareDialect::gpu(
            context,
            &gpu.name,
            gpu.memory_gb as i64,
            gpu.tflops_fp16,
            gpu.memory_bandwidth_gbs,
            location,
        )
        .map_err(|e| format!("Failed to create hw op: {e:?}"))?;
        operations.push(hw_op.to_string());
    }

    // 6. Parallelism configuration
    let par = &config.training.parallelism;
    if par.data_parallel > 0 || par.tensor_parallel > 0 || par.pipeline_parallel > 0 {
        let par_op = ParallelismDialect::hybrid(
            context,
            par.data_parallel as i64,
            par.tensor_parallel as i64,
            par.pipeline_parallel as i64,
            location,
        )
        .map_err(|e| format!("Failed to create parallelism op: {e:?}"))?;
        operations.push(par_op.to_string());
    }

    // 7. Compute metrics
    let total_flops = calculate_total_flops(config);
    let compute_op = ComputeDialect::flops(context, total_flops, total_flops * 2.0, location)
        .map_err(|e| format!("Failed to create compute op: {e:?}"))?;
    operations.push(compute_op.to_string());

    // 8. Memory metrics (approximate)
    let param_memory = total_params * 4; // FP32
    let activation_memory = calculate_activation_memory(config);
    let gradient_memory = param_memory;
    let optimizer_memory = param_memory * 2; // Adam states
    let peak_vram = param_memory + activation_memory + gradient_memory + optimizer_memory;

    let mem_op = MemoryDialect::metrics(
        context,
        param_memory,
        activation_memory,
        gradient_memory,
        optimizer_memory,
        peak_vram,
        1,
        location,
    )
    .map_err(|e| format!("Failed to create memory op: {e:?}"))?;
    operations.push(mem_op.to_string());

    // 9. Cost analysis
    let training_hours = estimate_training_hours(config);
    let gpu_hours = training_hours * config.hardware.total_gpu_count() as f64;
    let training_cost = gpu_hours * config.cost_config.gpu_hour_usd;

    let cost_op =
        CostDialect::training_cost(context, training_hours, training_cost, gpu_hours, location)
            .map_err(|e| format!("Failed to create cost op: {e:?}"))?;
    operations.push(cost_op.to_string());

    // 10. Final report
    let report_op = ReportDialect::report(context, model_name, model_type, "1.0", 0, location)
        .map_err(|e| format!("Failed to create report op: {e:?}"))?;
    operations.push(report_op.to_string());

    // Format output
    let mut output = String::new();
    output.push_str(&format!("// MLIR for model: {model_name}\n"));
    output.push_str(&format!("// Type: {model_type}\n"));
    output.push_str(&format!("// Layers: {num_layers}\n"));
    output.push_str(&format!("// Parameters: {total_params}\n\n"));

    for op in operations {
        output.push_str(&op);
        output.push('\n');
    }

    Ok(output)
}

fn calculate_total_params(config: &ModelConfig) -> i64 {
    let num_layers = config
        .model
        .global_params
        .num_layers
        .unwrap_or(config.model.layers.len() as u64) as i64;
    let hidden = config
        .model
        .global_params
        .extra
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(config.model.global_params.embedding_dim.unwrap_or(768) as u64)
        as i64;
    let vocab = config.model.global_params.vocab_size.unwrap_or(50000) as i64;

    let intermediate = config
        .model
        .global_params
        .extra
        .get("intermediate_size")
        .and_then(|v| v.as_u64())
        .unwrap_or((hidden * 4) as u64) as i64;
    let num_heads = config
        .model
        .global_params
        .extra
        .get("num_attention_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(32) as i64;
    let num_kv_heads = config
        .model
        .global_params
        .extra
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(num_heads as u64) as i64;
    let head_dim = hidden / num_heads.max(1);

    let has_moe = config
        .model
        .layers
        .iter()
        .any(|l| l.layer_type == LayerType::MoE);

    let attn_params = hidden * hidden + hidden * head_dim * num_kv_heads * 2 + hidden * hidden;

    let mlp_params = if has_moe {
        let mut expert_params = hidden * intermediate * 3;
        let mut num_experts = 8i64;
        let mut shared_experts = 0i64;

        for layer in &config.model.layers {
            if layer.layer_type == LayerType::MoE {
                num_experts = layer.params.num_experts.unwrap_or(8) as i64;
                shared_experts = layer.params.shared_experts.unwrap_or(0) as i64;
                expert_params = layer.params.hidden_size.unwrap_or(hidden as usize) as i64
                    * layer.params.intermediate_size.unwrap_or(intermediate as usize) as i64
                    * 3;
                break;
            }
        }
        expert_params * num_experts + expert_params * shared_experts + hidden * num_experts
    } else {
        hidden * intermediate * 3
    };

    let embedding_params = vocab * hidden;
    let layernorm_params = hidden * 2 * num_layers;
    let final_ln_params = hidden;
    let lm_head_params = hidden * vocab;

    embedding_params
        + (attn_params + mlp_params) * num_layers
        + layernorm_params
        + final_ln_params
        + lm_head_params
}

fn calculate_total_flops(config: &ModelConfig) -> f64 {
    let mut total = 0.0;

    for layer in &config.model.layers {
        let seq = config.model.global_params.sequence_length.unwrap_or(2048) as f64;
        match layer.layer_type {
            LayerType::Attention => {
                let hidden = layer.params.hidden_size.unwrap_or(768) as f64;
                total += 4.0 * hidden * hidden * seq;
            }
            LayerType::Mlp => {
                let hidden = layer.params.hidden_size.unwrap_or(768) as f64;
                let intermediate = layer.params.intermediate_size.unwrap_or(3072) as f64;
                total += 3.0 * hidden * intermediate * seq;
            }
            LayerType::MoE => {
                let hidden = layer.params.hidden_size.unwrap_or(768) as f64;
                let top_k = layer.params.top_k.unwrap_or(2) as f64;
                total += 2.0 * hidden * hidden * seq * top_k;
            }
            LayerType::Dense => {
                let in_f = layer.params.hidden_size.unwrap_or(768) as f64;
                let out_f = layer.params.intermediate_size.unwrap_or(768) as f64;
                total += 2.0 * in_f * out_f * seq;
            }
            _ => {}
        }
    }

    let num_layers = config
        .model
        .global_params
        .num_layers
        .unwrap_or(config.model.layers.len() as u64);
    let json_layers = config.model.layers.len().max(1) as u64;

    if num_layers > json_layers {
        total *= num_layers as f64 / json_layers as f64;
    }

    total
}

fn calculate_activation_memory(config: &ModelConfig) -> i64 {
    let hidden = config.model.global_params.embedding_dim.unwrap_or(768) as i64;
    let seq = config.model.global_params.sequence_length.unwrap_or(2048) as i64;
    let batch = config.training.batch_size as i64;
    let num_layers = config
        .model
        .global_params
        .num_layers
        .unwrap_or(config.model.layers.len() as u64) as i64;

    batch * seq * hidden * num_layers * 2 * 2
}

fn estimate_training_hours(config: &ModelConfig) -> f64 {
    let total_params = calculate_total_params(config) as f64;
    let steps = config.training.max_steps as f64;
    let batch = config.training.batch_size as f64;

    let tokens_per_step =
        batch * config.model.global_params.sequence_length.unwrap_or(2048) as f64;
    let total_training_flops = 6.0 * total_params * tokens_per_step * steps;

    let gpu_tflops = config
        .hardware
        .primary_gpu()
        .map(|g| g.tflops_fp16)
        .unwrap_or(100.0)
        * 1e12;

    let num_gpus = config.hardware.total_gpu_count() as f64;
    let efficiency = 0.3;

    if gpu_tflops <= 0.0 || num_gpus <= 0.0 {
        return 0.0;
    }

    let time_seconds = total_training_flops / (gpu_tflops * num_gpus * efficiency);
    time_seconds / 3600.0
}
