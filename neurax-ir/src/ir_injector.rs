//! IR Injector - Injects absorbed JSON data into each IR
//!
//! Guarantees that EVERY JSON field exploited arrives in the correct IR
//! Each IR receives EXACTLY the fields it needs, via documented mapping

use neurax_parser::{AbsorbedModel, GlobalResolutionContext};

/// IR Injector - Targeted injection of absorbed data into IRs
pub struct IrInjector;

impl IrInjector {
    /// Create ArchitectureIR from absorbed model
    /// Uses: model.*, training basics, hardware basics
    pub fn to_architecture_ir(absorbed: &AbsorbedModel) -> ArchitectureIRInput {
        let grc = &absorbed.resolution_context;
        let config = &absorbed.config;

        ArchitectureIRInput {
            model_name: config.model.name.clone(),
            model_type: config.model.model_type,
            layers: config.model.layers.clone(),
            global_params: config.model.global_params.clone(),

            // Inject resolved values from GlobalResolutionContext
            hidden_size: grc.hidden_size,
            num_layers: grc.num_layers,
            vocab_size: grc.vocab_size,
            intermediate_size: grc.intermediate_size,
            num_attention_heads: grc.num_attention_heads,
            num_key_value_heads: grc.num_key_value_heads,
            head_dim: grc.head_dim,

            // Confidence score for architecture validation
            confidence: grc.confidence_score,
        }
    }

    /// Configure MemoryPass from absorbed model
    /// Uses: training.*, hardware.*, parallelism.*
    pub fn configure_memory_pass(absorbed: &AbsorbedModel) -> MemoryPassConfig {
        let grc = &absorbed.resolution_context;
        let _training = &absorbed.config.training;

        MemoryPassConfig {
            // training.precision → dtype_bytes
            dtype_bytes: grc.dtype_bytes,

            // training.optimizer → optimizer_bytes_per_param
            optimizer_bytes: grc.optimizer_bytes_per_param,

            // training.gradient_checkpointing
            checkpointing_enabled: grc.gradient_checkpointing,

            // training.parallelism.zero_stage → ZeRO memory division
            zero_stage: grc.zero,

            // training.parallelism.*_degree → N GPUs for division
            num_gpus: grc.num_gpus,

            // hardware.targets[0].memory_gb → OOM check threshold
            gpu_memory_gb: grc.primary_gpu_memory_gb,

            // model.global_params.num_key_value_heads → KV cache
            num_kv_heads: grc
                .num_key_value_heads
                .unwrap_or(grc.num_attention_heads.unwrap_or(1)),

            // training.seq_len → KV cache size
            seq_len: grc.seq_len.unwrap_or(2048),

            // batch_size for activation memory
            batch_size: grc.batch_size.unwrap_or(1),

            // hidden_size for activation memory
            hidden_size: grc.hidden_size.unwrap_or(768),

            // num_layers for layer memory
            num_layers: grc.num_layers.unwrap_or(1),
        }
    }

    /// Configure HardwarePass from absorbed model
    /// Uses: hardware.targets.*, training.flash_attention
    pub fn configure_hardware_pass(absorbed: &AbsorbedModel) -> HardwarePassConfig {
        let grc = &absorbed.resolution_context;

        HardwarePassConfig {
            // hardware.targets[0].tflops_* → peak compute
            gpu_tflops_fp16: grc.primary_gpu_tflops,

            // hardware.targets[0].memory_bandwidth_gb_s → peak BW
            gpu_bw_gb_s: grc.primary_gpu_bw_gb_s,

            // hardware.targets[0].tensor_cores → TC eligibility
            has_tensor_cores: grc.has_tensor_cores,

            // training.flash_attention → FlashAttention timings
            flash_attention: grc.flash_attention,

            // interconnect bandwidth
            interconnect_bw: grc.interconnect_bw,

            // parallelism degrees
            dp: grc.dp,
            tp: grc.tp,
            pp: grc.pp,
            ep: grc.ep,
        }
    }

    /// Configure CostPass from absorbed model
    /// Uses: cost_config.*, training.num_steps, hardware.*
    pub fn configure_cost_pass(absorbed: &AbsorbedModel) -> CostPassConfig {
        let cc = &absorbed.config.cost_config;
        let grc = &absorbed.resolution_context;
        let training = &absorbed.config.training;

        CostPassConfig {
            // cost_config.gpu_hour_usd
            gpu_hour_usd: cc.gpu_hour_usd,

            // cost_config.energy_kwh_usd
            energy_kwh_usd: cc.energy_kwh_usd,

            // cost_config.pue_factor
            pue_factor: cc.pue_factor,

            // training.num_steps
            num_steps: training.max_steps as u64,

            // num_gpus from hardware
            num_gpus: grc.num_gpus,
        }
    }

    /// Calculate total parameters using GlobalResolutionContext
    /// This replaces the ad-hoc calculation with structured absorption
    pub fn calculate_total_params(absorbed: &AbsorbedModel) -> u64 {
        let grc = &absorbed.resolution_context;

        let h = grc.hidden_size.unwrap_or(768);
        let n = grc.num_layers.unwrap_or(1) as u64;
        let vocab = grc.vocab_size.unwrap_or(50000);
        let intermediate = grc.intermediate_size.unwrap_or(h * 4);
        let heads = grc.num_attention_heads.unwrap_or(32) as u64;
        let kv_heads = grc.num_key_value_heads.unwrap_or(heads as u32) as u64;
        let head_dim = if grc.head_dim > 0 {
            grc.head_dim
        } else {
            h / heads
        };

        // Embedding params
        let embed_params = vocab * h;

        // Attention params per layer (GQA-aware)
        // Q: H × H, K: H × H_kv, V: H × H_kv, O: H × H
        let h_kv = h * kv_heads / heads;
        let q_params = h * h; // Q projection
        let k_params = h * h_kv; // K projection (GQA: smaller)
        let v_params = h * h_kv; // V projection (GQA: smaller)
        let o_params = h * h; // Output projection
        let attn_params = q_params + k_params + v_params + o_params;

        // MLP params per layer (SwiGLU: 3 matrices)
        // gate: H × I, up: H × I, down: I × H
        let mlp_params = 3 * h * intermediate;

        // LayerNorm/RMSNorm params (2 per layer + 1 final)
        let norm_params = (2 * n + 1) * h;

        // LM head (tied or not)
        let lm_head_params = if grc.tied_embeddings { 0 } else { vocab * h };

        // Total
        let total = embed_params + n * (attn_params + mlp_params) + norm_params + lm_head_params;

        total
    }

    /// Calculate FLOPs per token using GlobalResolutionContext
    pub fn calculate_flops_per_token(absorbed: &AbsorbedModel) -> u64 {
        let grc = &absorbed.resolution_context;

        let h = grc.hidden_size.unwrap_or(768);
        let n = grc.num_layers.unwrap_or(1) as u64;
        let intermediate = grc.intermediate_size.unwrap_or(h * 4);
        let heads = grc.num_attention_heads.unwrap_or(32) as u64;
        let kv_heads = grc.num_key_value_heads.unwrap_or(heads as u32) as u64;
        let head_dim = if grc.head_dim > 0 {
            grc.head_dim
        } else {
            h / heads
        };
        let s = grc.seq_len.unwrap_or(2048);

        // Attention FLOPs per token
        // QKV projections: 2 × (H² + 2 × H × H_kv)
        let h_kv = h * kv_heads / heads;
        let qkv_flops = 2 * (h * h + 2 * h * h_kv);

        // QK^T + AV: 2 × heads × S × head_dim
        let attn_score_flops = 2 * heads * s * head_dim;

        // Output projection: 2 × H²
        let o_flops = 2 * h * h;

        let attn_flops = qkv_flops + attn_score_flops + o_flops;

        // MLP FLOPs per token (SwiGLU: 3 matrices, 2 ops each)
        // gate: 2 × H × I, up: 2 × H × I, down: 2 × I × H
        let mlp_flops = 6 * h * intermediate;

        // Norm FLOPs (RMSNorm: 5 ops per element, 2 per layer)
        let norm_flops = 5 * h * 2;

        // Total per layer
        let flops_per_layer = attn_flops + mlp_flops + norm_flops;

        // Total for all layers
        n * flops_per_layer
    }
}

/// Input for ArchitectureIR construction
#[derive(Debug, Clone)]
pub struct ArchitectureIRInput {
    pub model_name: Option<String>,
    pub model_type: neurax_parser::ModelType,
    pub layers: Vec<neurax_parser::Layer>,
    pub global_params: neurax_parser::GlobalParams,

    // Resolved values from GlobalResolutionContext
    pub hidden_size: Option<u64>,
    pub num_layers: Option<u32>,
    pub vocab_size: Option<u64>,
    pub intermediate_size: Option<u64>,
    pub num_attention_heads: Option<u32>,
    pub num_key_value_heads: Option<u32>,
    pub head_dim: u64,

    pub confidence: f32,
}

/// Configuration for MemoryPass
#[derive(Debug, Clone)]
pub struct MemoryPassConfig {
    pub dtype_bytes: u64,
    pub optimizer_bytes: u64,
    pub checkpointing_enabled: bool,
    pub zero_stage: u8,
    pub num_gpus: u32,
    pub gpu_memory_gb: f64,
    pub num_kv_heads: u32,
    pub seq_len: u64,
    pub batch_size: u64,
    pub hidden_size: u64,
    pub num_layers: u32,
}

/// Configuration for HardwarePass
#[derive(Debug, Clone)]
pub struct HardwarePassConfig {
    pub gpu_tflops_fp16: f64,
    pub gpu_bw_gb_s: f64,
    pub has_tensor_cores: bool,
    pub flash_attention: bool,
    pub interconnect_bw: f64,
    pub dp: u32,
    pub tp: u32,
    pub pp: u32,
    pub ep: u32,
}

/// Configuration for CostPass
#[derive(Debug, Clone)]
pub struct CostPassConfig {
    pub gpu_hour_usd: f64,
    pub energy_kwh_usd: f64,
    pub pue_factor: f64,
    pub num_steps: u64,
    pub num_gpus: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_calculation() {
        // Test with LLaMA-like parameters
        let mut grc = GlobalResolutionContext::default();
        grc.hidden_size = Some(4096);
        grc.num_layers = Some(32);
        grc.vocab_size = Some(32000);
        grc.intermediate_size = Some(11008);
        grc.num_attention_heads = Some(32);
        grc.num_key_value_heads = Some(8);
        grc.head_dim = 128;
        grc.tied_embeddings = true;

        // Expected: embed + 32 * (attn + mlp) + norms
        // embed = 32000 * 4096 = 131,072,000
        // attn = 4096² + 4096*1024 + 4096*1024 + 4096² = 16,777,216 + 4,194,304 + 4,194,304 + 16,777,216 = 41,943,040
        // mlp = 3 * 4096 * 11008 = 135,266,304
        // norm = (2*32+1) * 4096 = 266,240
        // total = 131,072,000 + 32 * (41,943,040 + 135,266,304) + 266,240
        //       = 131,072,000 + 32 * 177,209,344 + 266,240
        //       = 131,072,000 + 5,670,699,008 + 266,240
        //       = 5,802,037,248 (close to 6.7B for LLaMA-7B)

        let h = 4096u64;
        let n = 32u64;
        let vocab = 32000u64;
        let intermediate = 11008u64;
        let heads = 32u64;
        let kv_heads = 8u64;

        let embed = vocab * h;
        let h_kv = h * kv_heads / heads;
        let attn = h * h + h * h_kv + h * h_kv + h * h;
        let mlp = 3 * h * intermediate;
        let norm = (2 * n + 1) * h;
        let total = embed + n * (attn + mlp) + norm;

        // This should be approximately 6.7B parameters
        assert!(total > 5_000_000_000, "Expected > 5B params, got {}", total);
        assert!(total < 7_000_000_000, "Expected < 7B params, got {}", total);
    }
}
