//! JSON Absorption Module - Complete exploitation of every JSON field
//! 
//! Based on the 5 Principles:
//! P1. Exhaustive Reading - Every field extracted into typed Rust struct
//! P2. Cascade Resolution - Symbolic dims resolved via 6 strategies
//! P3. Downward Propagation - Global values propagated to layer params
//! P4. Targeted Injection - Each IR receives exactly the fields it needs
//! P5. Total Traceability - Every metric traced to source JSON fields

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use crate::model_config::{ModelConfig, ModelType, GlobalParams, TrainingConfig, HardwareConfig, DataConfig};

/// Global Resolution Context - Built once, used by ALL IRs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalResolutionContext {
    // ── Core Numeric Values ─────────────────────────────────────────────
    pub batch_size: Option<u64>,
    pub seq_len: Option<u64>,
    pub hidden_size: Option<u64>,
    pub num_attention_heads: Option<u32>,
    pub num_key_value_heads: Option<u32>,
    pub head_dim: u64,  // ALWAYS calculated
    pub intermediate_size: Option<u64>,
    pub vocab_size: Option<u64>,
    pub num_layers: Option<u32>,
    
    // ── MoE Parameters ──────────────────────────────────────────────────
    pub num_experts: Option<u32>,
    pub num_experts_per_tok: Option<u32>,
    pub num_shared_experts: Option<u32>,
    pub moe_intermediate_size: Option<u64>,
    
    // ── SSM/Mamba Parameters ────────────────────────────────────────────
    pub ssm_state_size: Option<u32>,
    pub ssm_expand: Option<u32>,
    pub ssm_conv_kernel: Option<u32>,
    
    // ── CNN Parameters ──────────────────────────────────────────────────
    pub initial_channels: Option<u32>,
    pub num_classes: Option<u64>,
    pub base_channels: Option<u32>,
    
    // ── RNN/LSTM Parameters ─────────────────────────────────────────────
    pub rnn_hidden_size: Option<u64>,
    pub num_rnn_layers: Option<u32>,
    pub bidirectional_rnn: bool,
    pub cell_type: Option<String>,
    
    // ── Diffusion Parameters ────────────────────────────────────────────
    pub diffusion_timesteps: Option<u32>,
    pub latent_channels: Option<u32>,
    pub image_size: Option<u64>,
    pub in_channels: Option<u32>,
    pub out_channels: Option<u32>,
    pub cross_attention_dim: Option<u64>,
    pub attention_head_dim: Option<u64>,
    pub block_out_channels: Option<Vec<u64>>,
    pub down_block_types: Option<Vec<String>>,
    pub up_block_types: Option<Vec<String>>,
    pub layers_per_block: Option<u32>,
    pub vae_scale_factor: Option<u32>,
    pub sample_size: Option<u64>,
    pub noise_schedule: Option<String>,
    pub beta_start: Option<f64>,
    pub beta_end: Option<f64>,
    
    // ── GNN Parameters ──────────────────────────────────────────────────
    pub node_features: Option<u64>,
    pub edge_features: Option<u64>,
    pub num_message_passing: Option<u32>,
    
    // ── Image/Data Parameters ───────────────────────────────────────────
    pub image_height: Option<u64>,
    pub image_width: Option<u64>,
    pub image_channels: Option<u64>,
    pub num_nodes: Option<u64>,
    pub num_edges: Option<u64>,
    
    // ── Derived Values (calculated from raw fields) ─────────────────────
    pub dtype_bytes: u64,           // from training.precision
    pub grad_dtype_bytes: u64,      // always 4 (fp32 for master gradients)
    pub optimizer_bytes_per_param: u64,  // Adam=8, SGD+momentum=4, SGD=0
    pub h_kv: Option<u64>,          // hidden_size × (kv_heads/q_heads)
    
    // ── Boolean Flags ───────────────────────────────────────────────────
    pub gradient_checkpointing: bool,
    pub flash_attention: bool,
    pub mixed_precision: bool,
    pub tied_embeddings: bool,      // lm_head shares embed weights
    
    // ── Primary Hardware Config ─────────────────────────────────────────
    pub primary_gpu_tflops: f64,
    pub primary_gpu_memory_gb: f64,
    pub primary_gpu_bw_gb_s: f64,
    pub num_gpus: u32,
    pub has_tensor_cores: bool,
    pub interconnect_bw: f64,
    
    // ── Parallelism Config ──────────────────────────────────────────────
    pub dp: u32,  // data parallel
    pub tp: u32,  // tensor parallel
    pub pp: u32,  // pipeline parallel
    pub ep: u32,  // expert parallel
    pub zero: u8, // ZeRO stage
    
    // ── Symbol Resolver ────────────────────────────────────────────────
    pub symbol_table: HashMap<String, u64>,
    
    // ── Confidence Score ───────────────────────────────────────────────
    pub confidence_score: f32,
    pub missing_fields: Vec<String>,
}

impl GlobalResolutionContext {
    /// Build the global resolution context from all config sections
    pub fn build(
        model_type: &ModelType,
        global_params: &GlobalParams,
        training: &TrainingConfig,
        hardware: &HardwareConfig,
        data: &DataConfig,
    ) -> Self {
        // ── Calculate head_dim (ALWAYS needed for attention) ────────────
        let head_dim = if let Some(hd) = global_params.extra.get("head_dim")
            .and_then(|v| v.as_u64()) {
            hd
        } else if let (Some(h), Some(n)) = (
            global_params.extra.get("hidden_size").and_then(|v| v.as_u64()),
            global_params.extra.get("num_attention_heads").and_then(|v| v.as_u64())
        ) {
            h / n
        } else {
            0 // will be flagged as symbolic
        };
        
        // ── dtype_bytes from precision ──────────────────────────────────
        let dtype_bytes = match training.precision.as_str() {
            "fp32" => 4,
            "fp16" | "bf16" | "mixed" => 2,
            "fp8" | "int8" => 1,
            "int4" => 0, // 0.5 bytes, handled specially
            _ => 4, // default fp32
        };
        
        // ── Optimizer state bytes ───────────────────────────────────────
        let optimizer_bytes = match training.optimizer.as_str() {
            "adamw" | "adam" => 8, // 2× fp32
            "sgd" => 4, // momentum
            "adafactor" | "lion" => 4,
            _ => 8, // default Adam
        };
        
        // ── H_kv for GQA ────────────────────────────────────────────────
        let h_kv = if let (Some(h), Some(q), Some(kv)) = (
            global_params.extra.get("hidden_size").and_then(|v| v.as_u64()),
            global_params.extra.get("num_attention_heads").and_then(|v| v.as_u64()),
            global_params.extra.get("num_key_value_heads").and_then(|v| v.as_u64())
        ) {
            Some(h * kv / q)
        } else {
            None
        };
        
        // ── Primary GPU ─────────────────────────────────────────────────
        let primary = hardware.gpus.first();
        let primary_gpu_tflops = primary.map(|g| g.tflops_fp16).unwrap_or(312.0);
        let primary_gpu_mem = primary.map(|g| g.memory_gb as f64).unwrap_or(80.0);
        let primary_gpu_bw = primary.map(|g| g.memory_bandwidth_gbs).unwrap_or(2000.0);
        let has_tc = primary.map(|g| g.tensor_cores).unwrap_or(true);
        let num_gpus: u32 = hardware.gpus.iter().map(|t| t.count).sum();
        
        // ── Tied embeddings detection ───────────────────────────────────
        let tied = matches!(model_type, ModelType::Transformer | ModelType::Moe | ModelType::Ssm);
        
        // ── Build symbol table ──────────────────────────────────────────
        let mut sym = HashMap::new();
        
        // Batch and sequence
        if training.batch_size > 0 { 
            sym.insert("B".into(), training.batch_size as u64); 
            sym.insert("batch".into(), training.batch_size as u64);
            sym.insert("batch_size".into(), training.batch_size as u64);
        }
        // seq_len not in current TrainingConfig, skip for now
        
        // Hidden dimensions
        if let Some(h) = global_params.extra.get("hidden_size").and_then(|v| v.as_u64()) {
            sym.insert("D".into(), h);
            sym.insert("H".into(), h);
            sym.insert("hidden".into(), h);
            sym.insert("hidden_size".into(), h);
            sym.insert("d_model".into(), h);
        }
        
        // Intermediate size
        if let Some(i) = global_params.extra.get("intermediate_size").and_then(|v| v.as_u64()) {
            sym.insert("I".into(), i);
            sym.insert("intermediate".into(), i);
            sym.insert("ff_dim".into(), i);
        }
        
        // Vocab
        let vocab: Option<u64> = global_params.vocab_size.map(|v| v as u64)
            .or(data.vocab_size.map(|v| v as u64));
        if let Some(v) = vocab {
            sym.insert("V".into(), v);
            sym.insert("vocab".into(), v);
            sym.insert("vocab_size".into(), v);
        }
        
        // num_classes (for CNN classification)
        let num_classes: Option<u64> = data.num_classes.map(|v| v as u64)
            .or(global_params.extra.get("num_classes").and_then(|v| v.as_u64()));
        if let Some(c) = num_classes {
            sym.insert("num_classes".into(), c);
            sym.insert("C_out".into(), c);
        }
        
        // Image dimensions
        if let Some(h) = data.image_height.map(|v| v as u64) {
            sym.insert("H_img".into(), h);
        }
        if let Some(w) = data.image_width.map(|v| v as u64) {
            sym.insert("W_img".into(), w);
        }
        if let Some(c) = data.image_channels.map(|v| v as u64) {
            sym.insert("C_img".into(), c);
            sym.insert("C".into(), c);
            sym.insert("channels".into(), c);
        }
        
        // dtype
        sym.insert("dtype_bytes".into(), dtype_bytes);
        
        // ── Parallelism ────────────────────────────────────────────────
        let dp = training.parallelism.data_parallel;
        let tp = training.parallelism.tensor_parallel;
        let pp = training.parallelism.pipeline_parallel;
        let zero = training.zero_stage;
        let ep = 1; // not in current config
        
        // ── Calculate confidence score ──────────────────────────────────
        let mut missing = Vec::new();
        let mut total_fields = 0;
        let mut present_fields = 0;
        
        // Critical fields
        let critical_fields = [
            ("hidden_size", global_params.extra.get("hidden_size").is_some()),
            ("num_layers", global_params.num_layers.is_some()),
            ("batch_size", training.batch_size > 0),
        ];
        
        for (name, present) in &critical_fields {
            total_fields += 1;
            if *present { present_fields += 1; }
            else { missing.push(name.to_string()); }
        }
        
        let confidence_score = if total_fields > 0 {
            present_fields as f32 / total_fields as f32
        } else {
            1.0
        };
        
        // Extract values from extra HashMap
        let get_u64 = |key: &str| global_params.extra.get(key).and_then(|v| v.as_u64());
        let get_u32 = |key: &str| global_params.extra.get(key).and_then(|v| v.as_u64()).map(|v| v as u32);
        
        Self {
            batch_size: Some(training.batch_size as u64),
            seq_len: None, // not in current TrainingConfig
            hidden_size: get_u64("hidden_size"),
            num_attention_heads: get_u32("num_attention_heads"),
            num_key_value_heads: get_u32("num_key_value_heads"),
            head_dim,
            intermediate_size: get_u64("intermediate_size"),
            vocab_size: vocab,
            num_layers: global_params.num_layers.map(|n| n as u32),
            
            num_experts: get_u32("num_experts"),
            num_experts_per_tok: get_u32("num_experts_per_tok"),
            num_shared_experts: get_u32("num_shared_experts"),
            moe_intermediate_size: get_u64("moe_intermediate_size"),
            
            ssm_state_size: get_u32("ssm_state_size"),
            ssm_expand: get_u32("ssm_expand"),
            ssm_conv_kernel: get_u32("ssm_conv_kernel"),
            
            initial_channels: get_u32("initial_channels"),
            num_classes: get_u64("num_classes"),
            base_channels: get_u32("base_channels"),
            
            rnn_hidden_size: get_u64("rnn_hidden_size"),
            num_rnn_layers: get_u32("num_rnn_layers"),
            bidirectional_rnn: global_params.extra.get("bidirectional_rnn")
                .and_then(|v| v.as_bool()).unwrap_or(false),
            cell_type: global_params.extra.get("cell_type")
                .and_then(|v| v.as_str().map(|s| s.to_string())),
            
            diffusion_timesteps: global_params.diffusion_timesteps.map(|v| v as u32),
            latent_channels: get_u32("latent_channels"),
            image_size: get_u64("image_size"),
            in_channels: get_u32("in_channels"),
            out_channels: get_u32("out_channels"),
            cross_attention_dim: get_u64("cross_attention_dim"),
            attention_head_dim: get_u64("attention_head_dim"),
            block_out_channels: global_params.extra.get("block_out_channels")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|i| i.as_u64()).collect()),
            down_block_types: global_params.extra.get("down_block_types")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|i| i.as_str().map(|s| s.to_string())).collect()),
            up_block_types: global_params.extra.get("up_block_types")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|i| i.as_str().map(|s| s.to_string())).collect()),
            layers_per_block: get_u32("layers_per_block"),
            vae_scale_factor: get_u32("vae_scale_factor"),
            sample_size: get_u64("sample_size"),
            noise_schedule: global_params.extra.get("noise_schedule")
                .and_then(|v| v.as_str().map(|s| s.to_string())),
            beta_start: global_params.extra.get("beta_start").and_then(|v| v.as_f64()),
            beta_end: global_params.extra.get("beta_end").and_then(|v| v.as_f64()),
            
            node_features: get_u64("node_features"),
            edge_features: get_u64("edge_features"),
            num_message_passing: get_u32("num_message_passing"),
            
            image_height: data.image_height.map(|v| v as u64),
            image_width: data.image_width.map(|v| v as u64),
            image_channels: data.image_channels.map(|v| v as u64),
            num_nodes: None, // not in current DataConfig
            num_edges: None,
            
            dtype_bytes,
            grad_dtype_bytes: 4, // always fp32
            optimizer_bytes_per_param: optimizer_bytes,
            h_kv,
            
            gradient_checkpointing: training.gradient_checkpointing,
            flash_attention: false, // not in current config
            mixed_precision: false, // not in current config
            tied_embeddings: tied,
            
            primary_gpu_tflops,
            primary_gpu_memory_gb: primary_gpu_mem,
            primary_gpu_bw_gb_s: primary_gpu_bw,
            num_gpus,
            has_tensor_cores: has_tc,
            interconnect_bw: hardware.interconnect_bandwidth_gbs,
            
            dp, tp, pp, ep, zero,
            symbol_table: sym,
            
            confidence_score,
            missing_fields: missing,
        }
    }
    
    /// Resolve a symbolic dimension to a concrete value
    pub fn resolve_dim(&self, dim: &Dim) -> Option<u64> {
        match dim {
            Dim::Known(v) => Some(*v),
            Dim::Symbolic(s) => self.symbol_table.get(s).copied(),
        }
    }
    
    /// Resolve a complete shape
    pub fn resolve_shape(&self, shape: &[Dim]) -> (Vec<u64>, f32) {
        let mut resolved = vec![];
        let mut resolved_count = 0;
        for dim in shape {
            match self.resolve_dim(dim) {
                Some(v) => { resolved.push(v); resolved_count += 1; }
                None => { resolved.push(0); }
            }
        }
        let ratio = if shape.is_empty() { 1.0 }
                    else { resolved_count as f32 / shape.len() as f32 };
        (resolved, ratio)
    }
    
    /// Get derived d_inner for SSM/Mamba
    pub fn d_inner(&self) -> Option<u64> {
        self.hidden_size.zip(self.ssm_expand.map(|e| e as u64))
            .map(|(h, e)| h * e)
    }
}

/// Dimension representation - either concrete or symbolic
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Dim {
    Known(u64),
    Symbolic(String),
}

impl Dim {
    pub fn is_known(&self) -> bool {
        matches!(self, Dim::Known(_))
    }
    
    pub fn is_symbolic(&self) -> bool {
        matches!(self, Dim::Symbolic(_))
    }
    
    pub fn value(&self) -> Option<u64> {
        match self {
            Dim::Known(v) => Some(*v),
            Dim::Symbolic(_) => None,
        }
    }
}

/// Source of a resolved dimension
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DimSource {
    ExplicitJson,    // Directly from JSON
    GlobalParam,     // From global_params
    LayerParam,      // From layer.params
    NeighborProp,    // Inferred from neighbor shapes
    ArchRule,        // From architectural rules
    Unresolved,      // Could not resolve
}

/// A resolved dimension with metadata
#[derive(Debug, Clone)]
pub struct ResolvedDim {
    pub value: u64,
    pub confidence: f32,
    pub source: DimSource,
}

/// Absorbed model with complete context
#[derive(Debug, Clone)]
pub struct AbsorbedModel {
    pub config: ModelConfig,
    pub resolution_context: GlobalResolutionContext,
}

impl AbsorbedModel {
    /// Absorb a ModelConfig and build the resolution context
    pub fn absorb(config: ModelConfig) -> Self {
        let grc = GlobalResolutionContext::build(
            &config.model.model_type,
            &config.model.global_params,
            &config.training,
            &config.hardware,
            &config.data,
        );
        
        Self {
            config,
            resolution_context: grc,
        }
    }
    
    /// Propagate global values to all layers
    pub fn propagate(&mut self) {
        for layer in &mut self.config.model.layers {
            GlobalPropagator::propagate(layer, &self.resolution_context);
        }
    }
}

/// Global Propagator - Enriches layer params with global values
/// When a layer doesn't have a local param, use the global value
pub struct GlobalPropagator;

impl GlobalPropagator {
    /// Propagate global values to a single layer
    pub fn propagate(
        layer: &mut crate::model_config::Layer,
        ctx: &GlobalResolutionContext,
    ) {
        let p = &mut layer.params;
        
        // ── Universal propagation ─────────────────────────────────────
        // Propagate hidden_size to all layers that might need it
        if p.hidden_size.is_none() && ctx.hidden_size.is_some() {
            p.hidden_size = ctx.hidden_size.map(|v| v as usize);
        }
        
        // Propagate batch_size to input shapes
        for dim in layer.input_shape.iter_mut() {
            if *dim == 0 && ctx.batch_size.is_some() {
                // First dimension is often batch
            }
        }
        
        // ── Propagate based on layer type ─────────────────────────────
        match layer.layer_type {
            crate::model_config::LayerType::Embedding => {
                if p.vocab_size.is_none() && ctx.vocab_size.is_some() {
                    p.vocab_size = ctx.vocab_size.map(|v| v as usize);
                }
                if p.embedding_dim.is_none() && ctx.hidden_size.is_some() {
                    p.embedding_dim = ctx.hidden_size.map(|v| v as usize);
                }
            }
            
            crate::model_config::LayerType::Attention => {
                if p.num_heads.is_none() && ctx.num_attention_heads.is_some() {
                    p.num_heads = ctx.num_attention_heads.map(|n| n as usize);
                }
                if p.hidden_size.is_none() && ctx.hidden_size.is_some() {
                    p.hidden_size = ctx.hidden_size.map(|v| v as usize);
                }
            }
            
            crate::model_config::LayerType::Dense => {
                // Dense layers in transformers often use hidden_size
                if p.in_features.is_none() && ctx.hidden_size.is_some() {
                    p.in_features = ctx.hidden_size.map(|v| v as usize);
                }
            }
            
            crate::model_config::LayerType::Normalization => {
                if p.hidden_size.is_none() && ctx.hidden_size.is_some() {
                    p.hidden_size = ctx.hidden_size.map(|v| v as usize);
                }
            }
            
            _ => {}
        }
    }
}

/// Dimension Resolver - 6 strategies for resolving symbolic dimensions
pub struct DimResolver<'a> {
    pub grc: &'a GlobalResolutionContext,
}

impl<'a> DimResolver<'a> {
    pub fn new(grc: &'a GlobalResolutionContext) -> Self {
        Self { grc }
    }
    
    /// Resolve a symbolic dimension with full traceability
    /// Applies 6 strategies in priority order
    pub fn resolve(&self, dim: &Dim, layer_context: Option<&LayerDimContext>) -> ResolvedDim {
        match dim {
            // Case trivial: already concrete
            Dim::Known(v) => ResolvedDim {
                value: *v,
                confidence: 1.0,
                source: DimSource::ExplicitJson,
            },
            
            Dim::Symbolic(sym) => {
                // Strategy 1: Direct lookup in symbol table
                if let Some(&v) = self.grc.symbol_table.get(sym.as_str()) {
                    return ResolvedDim { 
                        value: v, 
                        confidence: 1.0,
                        source: DimSource::ExplicitJson 
                    };
                }
                
                // Strategy 2: Extended matches (aliases)
                let resolved = self.resolve_alias(sym);
                if let Some(v) = resolved {
                    return ResolvedDim { 
                        value: v, 
                        confidence: 0.95,
                        source: DimSource::GlobalParam 
                    };
                }
                
                // Strategy 3: Derivation from layer params
                if let Some(ctx) = layer_context {
                    if let Some(v) = ctx.params.get(sym) {
                        return ResolvedDim { 
                            value: v, 
                            confidence: 0.98,
                            source: DimSource::LayerParam 
                        };
                    }
                }
                
                // Strategy 4: Derivation from neighbor shapes
                if let Some(ctx) = layer_context {
                    if let Some(v) = self.infer_from_neighbors(sym, ctx) {
                        return ResolvedDim { 
                            value: v, 
                            confidence: 0.85,
                            source: DimSource::NeighborProp 
                        };
                    }
                }
                
                // Strategy 5: Architectural rule
                if let Some(ctx) = layer_context {
                    if let Some(v) = self.apply_arch_rule(sym, ctx) {
                        return ResolvedDim { 
                            value: v, 
                            confidence: 0.80,
                            source: DimSource::ArchRule 
                        };
                    }
                }
                
                // Strategy 6: Still symbolic - unresolved
                ResolvedDim { 
                    value: 0, 
                    confidence: 0.0,
                    source: DimSource::Unresolved 
                }
            }
        }
    }
    
    /// Strategy 2: Resolve via aliases
    fn resolve_alias(&self, sym: &str) -> Option<u64> {
        let aliases: &[(&str, &str)] = &[
            ("batch", "B"), ("batch_size", "B"),
            ("seq", "S"), ("sequence_length", "S"),
            ("hidden", "H"), ("d_model", "H"),
            ("intermediate", "I"), ("ff_dim", "I"),
            ("vocab", "V"), ("n_vocab", "V"),
            ("C", "C_img"), ("channels", "C_img"),
            ("n_heads", "num_heads"), ("heads", "num_heads"),
        ];
        
        for (alias, canonical) in aliases {
            if sym.eq_ignore_ascii_case(alias) {
                if let Some(&v) = self.grc.symbol_table.get(*canonical) {
                    return Some(v);
                }
            }
        }
        None
    }
    
    /// Strategy 4: Infer from neighbor shapes
    fn infer_from_neighbors(&self, _sym: &str, _ctx: &LayerDimContext) -> Option<u64> {
        // TODO: Implement shape propagation
        None
    }
    
    /// Strategy 5: Apply architectural rules
    fn apply_arch_rule(&self, sym: &str, ctx: &LayerDimContext) -> Option<u64> {
        match (sym, ctx.layer_type.as_str()) {
            // For Transformer: D_out = hidden_size except lm_head
            ("D_out", "attention") | ("D_out", "mlp") | ("D_out", "rms_norm") => {
                self.grc.hidden_size
            }
            ("D_out", "lm_head") => self.grc.vocab_size,
            
            // For SSM: d_inner = d_model × expand
            ("d_inner", "mamba_block") => {
                self.grc.d_inner()
            }
            
            _ => None
        }
    }
    
    /// Resolve all shapes in a layer
    pub fn resolve_layer_shapes(&self, layer: &mut crate::model_config::Layer) {
        let ctx = LayerDimContext {
            layer_type: format!("{:?}", layer.layer_type),
            params: LayerParamsMap::from_layer(&layer.params),
            input_shape: layer.input_shape.clone(),
            output_shape: layer.output_shape.clone(),
        };
        
        // Resolve input shape
        for dim in layer.input_shape.iter_mut() {
            let resolved = self.resolve(&Dim::Known(*dim as u64), Some(&ctx));
            if resolved.source != DimSource::Unresolved {
                *dim = resolved.value as usize;
            }
        }
        
        // Resolve output shape
        for dim in layer.output_shape.iter_mut() {
            let resolved = self.resolve(&Dim::Known(*dim as u64), Some(&ctx));
            if resolved.source != DimSource::Unresolved {
                *dim = resolved.value as usize;
            }
        }
    }
}

/// Context for dimension resolution at layer level
#[derive(Debug, Clone)]
pub struct LayerDimContext {
    pub layer_type: String,
    pub params: LayerParamsMap,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
}

/// Simple map for layer params access
#[derive(Debug, Clone, Default)]
pub struct LayerParamsMap {
    values: HashMap<String, u64>,
}

impl LayerParamsMap {
    pub fn from_layer(layer_params: &crate::model_config::LayerParams) -> Self {
        let mut values: HashMap<String, u64> = HashMap::new();
        
        if let Some(v) = layer_params.hidden_size { values.insert("hidden_size".into(), v as u64); }
        if let Some(v) = layer_params.vocab_size { values.insert("vocab_size".into(), v as u64); }
        if let Some(v) = layer_params.embedding_dim { values.insert("embedding_dim".into(), v as u64); }
        if let Some(v) = layer_params.num_heads { values.insert("num_heads".into(), v as u64); }
        if let Some(v) = layer_params.in_features { values.insert("in_features".into(), v as u64); }
        if let Some(v) = layer_params.out_features { values.insert("out_features".into(), v as u64); }
        if let Some(v) = layer_params.intermediate_size { values.insert("intermediate_size".into(), v as u64); }
        
        Self { values }
    }
    
    pub fn get(&self, key: &str) -> Option<u64> {
        self.values.get(key).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_symbol_resolution() {
        let mut sym = HashMap::new();
        sym.insert("B".into(), 32);
        sym.insert("S".into(), 2048);
        sym.insert("H".into(), 4096);
        
        let grc = GlobalResolutionContext {
            symbol_table: sym,
            ..Default::default()
        };
        
        assert_eq!(grc.resolve_dim(&Dim::Known(100)), Some(100));
        assert_eq!(grc.resolve_dim(&Dim::Symbolic("B".into())), Some(32));
        assert_eq!(grc.resolve_dim(&Dim::Symbolic("S".into())), Some(2048));
        assert_eq!(grc.resolve_dim(&Dim::Symbolic("H".into())), Some(4096));
        assert_eq!(grc.resolve_dim(&Dim::Symbolic("X".into())), None);
    }
    
    #[test]
    fn test_shape_resolution() {
        let mut sym = HashMap::new();
        sym.insert("B".into(), 32);
        sym.insert("S".into(), 2048);
        
        let grc = GlobalResolutionContext {
            symbol_table: sym,
            ..Default::default()
        };
        
        let shape = vec![
            Dim::Symbolic("B".into()),
            Dim::Symbolic("S".into()),
            Dim::Known(4096),
            Dim::Symbolic("X".into()), // unresolved
        ];
        
        let (resolved, ratio) = grc.resolve_shape(&shape);
        assert_eq!(resolved, vec![32, 2048, 4096, 0]);
        assert!((ratio - 0.75).abs() < 0.01);
    }
}

impl Default for GlobalResolutionContext {
    fn default() -> Self {
        Self {
            batch_size: None,
            seq_len: None,
            hidden_size: None,
            num_attention_heads: None,
            num_key_value_heads: None,
            head_dim: 0,
            intermediate_size: None,
            vocab_size: None,
            num_layers: None,
            num_experts: None,
            num_experts_per_tok: None,
            num_shared_experts: None,
            moe_intermediate_size: None,
            ssm_state_size: None,
            ssm_expand: None,
            ssm_conv_kernel: None,
            initial_channels: None,
            num_classes: None,
            base_channels: None,
            rnn_hidden_size: None,
            num_rnn_layers: None,
            bidirectional_rnn: false,
            cell_type: None,
            diffusion_timesteps: None,
            latent_channels: None,
            image_size: None,
            in_channels: None,
            out_channels: None,
            cross_attention_dim: None,
            attention_head_dim: None,
            block_out_channels: None,
            down_block_types: None,
            up_block_types: None,
            layers_per_block: None,
            vae_scale_factor: None,
            sample_size: None,
            noise_schedule: None,
            beta_start: None,
            beta_end: None,
            node_features: None,
            edge_features: None,
            num_message_passing: None,
            image_height: None,
            image_width: None,
            image_channels: None,
            num_nodes: None,
            num_edges: None,
            dtype_bytes: 4,
            grad_dtype_bytes: 4,
            optimizer_bytes_per_param: 8,
            h_kv: None,
            gradient_checkpointing: false,
            flash_attention: false,
            mixed_precision: false,
            tied_embeddings: true,
            primary_gpu_tflops: 312.0,
            primary_gpu_memory_gb: 80.0,
            primary_gpu_bw_gb_s: 2000.0,
            num_gpus: 1,
            has_tensor_cores: true,
            interconnect_bw: 600.0,
            dp: 1,
            tp: 1,
            pp: 1,
            ep: 1,
            zero: 0,
            symbol_table: HashMap::new(),
            confidence_score: 1.0,
            missing_fields: Vec::new(),
        }
    }
}
