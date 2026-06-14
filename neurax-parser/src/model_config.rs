//! Internal model configuration structures

use crate::error::ParserError;
use crate::schema::*;
use std::collections::HashMap;

/// Parsed and validated model configuration
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub schema_version: String,
    pub model: Model,
    pub training: TrainingConfig,
    pub hardware: HardwareConfig,
    pub data: DataConfig,
    pub metrics_config: MetricsConfig,
    pub cost_config: CostConfig,
}

impl ModelConfig {
    /// Convert raw config to validated config
    pub fn from_raw(raw: RawModelConfig) -> Result<Self, ParserError> {
        // Validate schema version
        if raw.schema_version.is_empty() {
            return Err(ParserError::schema_validation("schema_version is required"));
        }

        Ok(Self {
            schema_version: raw.schema_version,
            model: Model::from_raw(raw.model)?,
            training: TrainingConfig::from_raw(raw.training),
            hardware: HardwareConfig::from_raw(raw.hardware),
            data: DataConfig::from_raw(raw.data),
            metrics_config: MetricsConfig::from_raw(raw.metrics_config),
            cost_config: CostConfig::from_raw(raw.cost_config),
        })
    }
}

/// Model definition
#[derive(Debug, Clone)]
pub struct Model {
    pub name: Option<String>,
    pub model_type: ModelType,
    pub layers: Vec<Layer>,
    pub global_params: GlobalParams,
}

impl Model {
    pub fn from_raw(raw: RawModel) -> Result<Self, ParserError> {
        let model_type = ModelType::from_str(&raw.model_type)?;
        let layers: Result<Vec<_>, _> = raw.layers.into_iter().map(Layer::from_raw).collect();

        Ok(Self {
            name: raw.name,
            model_type,
            layers: layers?,
            global_params: GlobalParams::from_raw(raw.global_params),
        })
    }
}

/// Model type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelType {
    Transformer,
    Cnn,
    Moe,
    Diffusion,
    Gnn,
    Rnn,
    Ssm,    // State Space Models (Mamba, S4, H3, etc.)
    Gan,    // Generative Adversarial Networks
    Hybrid, // Multi-architecture models (ViT, DiT, etc.)
}

impl ModelType {
    pub fn from_str(s: &str) -> Result<Self, ParserError> {
        match s.to_lowercase().as_str() {
            "transformer" => Ok(Self::Transformer),
            "cnn" | "convolutional" => Ok(Self::Cnn),
            "moe" | "mixture_of_experts" => Ok(Self::Moe),
            "diffusion" => Ok(Self::Diffusion),
            "gnn" | "graph_neural_network" => Ok(Self::Gnn),
            "rnn" | "recurrent" => Ok(Self::Rnn),
            "ssm" | "state_space" | "mamba" | "state_space_model" => Ok(Self::Ssm),
            "gan" | "generative_adversarial" | "adversarial" => Ok(Self::Gan),
            "hybrid" | "multi_architecture" => Ok(Self::Hybrid),
            _ => Err(ParserError::InvalidModelType(s.to_string())),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Transformer => "transformer",
            Self::Cnn => "cnn",
            Self::Moe => "moe",
            Self::Diffusion => "diffusion",
            Self::Gnn => "gnn",
            Self::Rnn => "rnn",
            Self::Ssm => "ssm",
            Self::Gan => "gan",
            Self::Hybrid => "hybrid",
        }
    }
}

/// Layer type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerType {
    Embedding,
    Attention,
    Mlp,
    Conv,
    Dense,
    Normalization,
    Pooling,
    MoE,
    // CNN Modern Layer Types
    ResidualBlock,
    Mbconv,
    Inception,
    DenseBlock,
    ConvnextBlock,
    ShuffleUnit,
    C2f,
    Detection,
    Transition,
    // State Space Model Layer Types
    MambaBlock,
    S4Block,
    H3Block,
    StateSpace,
    RwkvBlock,
    RetentionBlock,
    // GAN Layer Types
    GeneratorBlock,
    DiscriminatorBlock,
    StyleMod,
    AdaIN,            // Adaptive Instance Normalization
    MinibatchStd,     // Minibatch discrimination
    PixelNorm,        // Pixel-wise normalization
    SelfAttention,    // Self-attention for GANs
    SpectralNorm,     // Spectral normalization
    ProgressiveBlock, // Progressive growing
    // LSTM/RNN Layer Types
    LstmBlock,
    GruBlock,
    RnnCell,
    Bidirectional,
    EncoderBlock,
    DecoderBlock,
    // Diffusion Model Layer Types
    UnetBlock,
    TimeEmbedding,
    CrossAttention,
    DownBlock,
    UpBlock,
    MidBlock,
    ResnetBlock,
    TimestepBlock,
    ConditionBlock,
    NoisePredictor,
    VaeEncoder,
    VaeDecoder,
    // Custom layer with user-defined equations
    Custom,
}

impl LayerType {
    pub fn from_str(s: &str) -> Result<Self, ParserError> {
        match s.to_lowercase().as_str() {
            "embedding" => Ok(Self::Embedding),
            "positional_encoding" | "positional_embed" | "pos_embed" | "position_embedding" => {
                Ok(Self::Embedding)
            } // Treated as embedding
            "attention" | "self_attention" | "multi_head_attention" | "transformer" => {
                Ok(Self::Attention)
            }
            "mlp" | "ffn" | "feed_forward" => Ok(Self::Mlp),
            "conv" | "conv2d" | "convolution" => Ok(Self::Conv),
            "dense" | "linear" => Ok(Self::Dense),
            "normalization" | "layer_norm" | "batch_norm" | "rms_norm" => Ok(Self::Normalization),
            "pooling" | "max_pool" | "avg_pool" => Ok(Self::Pooling),
            "moe" | "mixture_of_experts" => Ok(Self::MoE),
            // CNN Modern architectures
            "residual_block" | "res_block" | "residual" => Ok(Self::ResidualBlock),
            "mbconv" | "inverted_residual" | "mobile_bottleneck" => Ok(Self::Mbconv),
            "inception" | "inception_module" => Ok(Self::Inception),
            "dense_block" | "denseblock" => Ok(Self::DenseBlock),
            "convnext_block" | "convnext" => Ok(Self::ConvnextBlock),
            "shuffle_unit" | "shuffleunit" | "channel_shuffle" => Ok(Self::ShuffleUnit),
            "resnext_block" | "resnext" => Ok(Self::ResidualBlock), // ResNeXt uses residual blocks
            "c2f" | "csp_stage" => Ok(Self::C2f),
            "detection" | "detect" | "yolo_head" => Ok(Self::Detection),
            "transition" | "transition_layer" => Ok(Self::Transition),
            // State Space Models
            "mamba_block" | "mamba" | "mamba_layer" => Ok(Self::MambaBlock),
            "s4_block" | "s4" | "structured_state_space" => Ok(Self::S4Block),
            "h3_block" | "h3" | "hungry_hungry_hippos" => Ok(Self::H3Block),
            "state_space" | "ssm_layer" | "state_space_layer" => Ok(Self::StateSpace),
            "rwkv_block" | "rwkv" | "receptance_weighted_key_value" => Ok(Self::RwkvBlock),
            "retention_block" | "retention" => Ok(Self::RetentionBlock),
            // GAN architectures
            "generator_block" | "generator" | "gen_block" => Ok(Self::GeneratorBlock),
            "discriminator_block" | "discriminator" | "disc_block" => Ok(Self::DiscriminatorBlock),
            "style_mod" | "style_module" | "stylemod" => Ok(Self::StyleMod),
            "adain" | "adaptive_instance_norm" | "adaptive_instance" => Ok(Self::AdaIN),
            "minibatch_std" | "minibatch_discrim" | "minibatch" => Ok(Self::MinibatchStd),
            "pixel_norm" | "pixelwise_norm" | "pixelnorm" => Ok(Self::PixelNorm),
            "gan_attention" | "gan_self_attention" => Ok(Self::SelfAttention),
            "spectral_norm" | "spectralnorm" | "spectral" => Ok(Self::SpectralNorm),
            "progressive_block" | "progressive" | "prog_block" => Ok(Self::ProgressiveBlock),
            // LSTM/RNN architectures
            "lstm_block" | "lstm" | "lstm_layer" => Ok(Self::LstmBlock),
            "gru_block" | "gru" | "gru_layer" => Ok(Self::GruBlock),
            "rnn_cell" | "rnn" | "vanilla_rnn" => Ok(Self::RnnCell),
            "bidirectional" | "bilstm" | "bigru" | "bi_rnn" => Ok(Self::Bidirectional),
            "encoder_block" | "encoder" | "enc_layer" => Ok(Self::EncoderBlock),
            "decoder_block" | "decoder" | "dec_layer" => Ok(Self::DecoderBlock),
            // Diffusion architectures
            "unet_block" | "unet" | "unet_layer" => Ok(Self::UnetBlock),
            "time_embedding" | "time_embed" | "timestep_embedding" => Ok(Self::TimeEmbedding),
            "cross_attention" | "cross_attn" | "conditioned_attention" => Ok(Self::CrossAttention),
            "down_block" | "downsample_block" | "downsampling" => Ok(Self::DownBlock),
            "up_block" | "upsample_block" | "upsampling" => Ok(Self::UpBlock),
            "mid_block" | "middle_block" | "midblock" => Ok(Self::MidBlock),
            "resnet_block" | "resnet" | "res_block_diff" => Ok(Self::ResnetBlock),
            "timestep_block" | "timestep" | "time_block" => Ok(Self::TimestepBlock),
            "condition_block" | "condition" | "conditioning" => Ok(Self::ConditionBlock),
            "noise_predictor" | "noise_pred" | "denoiser" => Ok(Self::NoisePredictor),
            "vae_encoder" | "vae_enc" | "encoder_vae" => Ok(Self::VaeEncoder),
            "vae_decoder" | "vae_dec" | "decoder_vae" => Ok(Self::VaeDecoder),
            // Custom layer
            "custom" | "custom_layer" | "user_defined" => Ok(Self::Custom),
            _ => Err(ParserError::InvalidLayerType(s.to_string())),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::Attention => "attention",
            Self::Mlp => "mlp",
            Self::Conv => "conv",
            Self::Dense => "dense",
            Self::Normalization => "normalization",
            Self::Pooling => "pooling",
            Self::MoE => "moe",
            Self::ResidualBlock => "residual_block",
            Self::Mbconv => "mbconv",
            Self::Inception => "inception",
            Self::DenseBlock => "dense_block",
            Self::ConvnextBlock => "convnext_block",
            Self::ShuffleUnit => "shuffle_unit",
            Self::C2f => "c2f",
            Self::Detection => "detection",
            Self::Transition => "transition",
            // State Space Models
            Self::MambaBlock => "mamba_block",
            Self::S4Block => "s4_block",
            Self::H3Block => "h3_block",
            Self::StateSpace => "state_space",
            Self::RwkvBlock => "rwkv_block",
            Self::RetentionBlock => "retention_block",
            // GAN Layer Types
            Self::GeneratorBlock => "generator_block",
            Self::DiscriminatorBlock => "discriminator_block",
            Self::StyleMod => "style_mod",
            Self::AdaIN => "adain",
            Self::MinibatchStd => "minibatch_std",
            Self::PixelNorm => "pixel_norm",
            Self::SelfAttention => "self_attention",
            Self::SpectralNorm => "spectral_norm",
            Self::ProgressiveBlock => "progressive_block",
            // LSTM/RNN Layer Types
            Self::LstmBlock => "lstm_block",
            Self::GruBlock => "gru_block",
            Self::RnnCell => "rnn_cell",
            Self::Bidirectional => "bidirectional",
            Self::EncoderBlock => "encoder_block",
            Self::DecoderBlock => "decoder_block",
            // Diffusion Layer Types
            Self::UnetBlock => "unet_block",
            Self::TimeEmbedding => "time_embedding",
            Self::CrossAttention => "cross_attention",
            Self::DownBlock => "down_block",
            Self::UpBlock => "up_block",
            Self::MidBlock => "mid_block",
            Self::ResnetBlock => "resnet_block",
            Self::TimestepBlock => "timestep_block",
            Self::ConditionBlock => "condition_block",
            Self::NoisePredictor => "noise_predictor",
            Self::VaeEncoder => "vae_encoder",
            Self::VaeDecoder => "vae_decoder",
            // Custom
            Self::Custom => "custom",
        }
    }
}

/// Layer definition
#[derive(Debug, Clone)]
pub struct Layer {
    pub id: String,
    pub layer_type: LayerType,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub params: LayerParams,
    pub custom_equations: Option<CustomEquations>,
}

impl Layer {
    pub fn from_raw(raw: RawLayer) -> Result<Self, ParserError> {
        Ok(Self {
            id: raw.id,
            layer_type: LayerType::from_str(&raw.layer_type)?,
            input_shape: raw.input_shape,
            output_shape: raw.output_shape,
            params: LayerParams::from_raw(raw.params),
            custom_equations: raw.custom_equations.map(CustomEquations::from_raw),
        })
    }
}

/// Layer parameters
#[derive(Debug, Clone, Default)]
pub struct LayerParams {
    // Common parameters
    pub hidden_size: Option<usize>,
    pub num_heads: Option<usize>,
    pub head_dim: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub vocab_size: Option<usize>,
    pub embedding_dim: Option<usize>,

    // Dense/Linear parameters
    pub in_features: Option<usize>,
    pub out_features: Option<usize>,

    // Conv parameters
    pub kernel_size: Option<usize>,
    pub kernel_h: Option<usize>,
    pub kernel_w: Option<usize>,
    pub stride: Option<usize>,
    pub padding: Option<usize>,
    pub in_channels: Option<usize>,
    pub out_channels: Option<usize>,

    // Attention parameters
    pub causal: bool,
    pub num_kv_heads: Option<usize>,

    // MLP parameters
    pub gated: bool,
    pub activation: Option<String>,

    // MoE parameters
    pub num_experts: Option<usize>,
    pub top_k: Option<usize>,
    pub shared_experts: Option<usize>,

    // CNN Modern architecture parameters
    pub num_blocks: Option<usize>,
    pub num_layers: Option<usize>, // Number of layers in a block
    pub expand_ratio: Option<usize>,
    pub cardinality: Option<usize>,      // ResNeXt groups
    pub bottleneck: bool,                // ResNet bottleneck
    pub bottleneck_width: Option<usize>, // ResNeXt bottleneck width
    pub groups: Option<usize>,           // Grouped convolutions
    pub se: bool,                        // Squeeze-and-Excitation
    pub h_swish: bool,                   // MobileNet-V3 activation
    pub pool_type: Option<String>,       // max, avg, global
    pub num_classes: Option<usize>,
    pub num_anchors: Option<usize>,     // YOLO
    pub growth_rate: Option<usize>,     // DenseNet
    pub compression: Option<f64>,       // DenseNet transition
    pub mlp_ratio: Option<f64>,         // MLP ratio for ConvNeXt
    pub num_bottlenecks: Option<usize>, // Number of bottlenecks in C2f

    // State Space Model parameters
    pub state_dim: Option<usize>,        // Mamba state dimension (N)
    pub expansion_factor: Option<usize>, // Mamba expansion factor (usually 2)
    pub dt_rank: Option<usize>,          // Delta rank for Mamba
    pub conv_kernel_size: Option<usize>, // Conv1d kernel in Mamba
    pub inner_rank: Option<usize>,       // Inner dimension rank
    pub time_step: Option<f64>,          // S4 time step (dt)
    pub ssm_type: Option<String>,        // "mamba", "s4", "h3", "rwkv"
    pub bidirectional: bool,             // Bidirectional SSM
    pub use_conv1d: bool,                // Use Conv1d in SSM
    pub gate_activation: Option<String>, // SiLU, tanh, etc.
    pub dt_min: Option<f64>,             // Mamba dt min
    pub dt_max: Option<f64>,             // Mamba dt max
    pub dt_init_floor: Option<f64>,      // Mamba dt init floor

    // GAN parameters
    pub latent_dim: Option<usize>,     // Latent vector dimension (z)
    pub style_dim: Option<usize>,      // Style dimension (w)
    pub w_dim: Option<usize>,          // Intermediate style dimension
    pub spectral_norm: bool,           // Apply spectral normalization
    pub feature_matching: bool,        // Feature matching loss
    pub progressive: bool,             // Progressive growing
    pub alpha: Option<f64>,            // Progressive growing alpha
    pub truncation: Option<f64>,       // Style truncation trick
    pub mapping_layers: Option<usize>, // Mapping network layers
    pub num_channels: Option<usize>,   // Output channels (3 for RGB)
    pub resolution: Option<usize>,     // Output resolution
    pub base_channels: Option<usize>,  // Base feature maps
    pub max_channels: Option<usize>,   // Max feature maps
    pub minibatch_std_group: Option<usize>, // Minibatch std group size
    pub label_dim: Option<usize>,      // Conditional label dimension

    // LSTM/RNN parameters
    pub rnn_hidden_size: Option<usize>, // Hidden state size
    pub num_rnn_layers: Option<usize>,  // Number of RNN layers
    pub bidirectional_rnn: bool,        // Bidirectional RNN
    pub cell_type: Option<String>,      // "lstm", "gru", "vanilla"
    pub forget_bias: Option<f64>,       // LSTM forget gate bias
    pub peephole: bool,                 // LSTM peephole connections
    pub recurrent_dropout: Option<f64>, // Recurrent dropout
    pub zoneout: Option<f64>,           // Zoneout probability
    pub seq_len: Option<usize>,         // Sequence length
    pub vocab_size_rnn: Option<usize>,  // RNN vocab size
    pub attention_type: Option<String>, // "bahdanau", "luong", "dot"
    pub beam_width: Option<usize>,      // Beam search width

    // Diffusion model parameters
    pub diffusion_timesteps: Option<usize>, // Number of diffusion steps (T)
    pub noise_schedule: Option<String>,     // "linear", "cosine", "sqrt"
    pub beta_start: Option<f64>,            // Starting beta
    pub beta_end: Option<f64>,              // Ending beta
    pub image_size: Option<usize>,          // Image resolution
    pub in_channels_diff: Option<usize>,    // Input channels (latent)
    pub out_channels_diff: Option<usize>,   // Output channels (latent)
    pub latent_channels: Option<usize>,     // VAE latent channels
    pub down_block_types: Option<Vec<String>>, // DownBlock types
    pub up_block_types: Option<Vec<String>>, // UpBlock types
    pub block_out_channels: Option<Vec<usize>>, // Channels per block
    pub layers_per_block: Option<usize>,    // ResNet layers per block
    pub cross_attention_dim: Option<usize>, // Cross-attention dimension
    pub attention_head_dim: Option<usize>,  // Attention head dimension
    pub norm_num_groups: Option<usize>,     // GroupNorm groups
    pub resnet_time_scale_shift: Option<String>, // "default", "scale_shift"
    pub vae_scale_factor: Option<usize>,    // VAE scale factor (8 for SD)
    pub sample_size: Option<usize>,         // Sample size for generation
    pub conditioning_channels: Option<usize>, // Conditioning input channels
    pub projection_class_embeddings_input_dim: Option<usize>, // Class embedding dim
    pub class_embed_type: Option<String>,   // "timestep", "identity", "projection"
    pub addition_embed_type: Option<String>, // "text_time", "text", "image"
    pub time_embedding_dim: Option<usize>,  // Time embedding dimension
    pub transformer_layers_per_block: Option<usize>, // Transformer layers in block
    pub num_attention_heads_diff: Option<usize>, // Number of attention heads

    // General
    pub dropout: Option<f64>,
    pub bias: bool,
    pub param_count: Option<u64>, // Explicit param count for custom layers

    // Extra parameters
    pub extra: HashMap<String, serde_json::Value>,
}

impl LayerParams {
    pub fn from_raw(raw: RawLayerParams) -> Self {
        Self {
            hidden_size: raw
                .get_usize("hidden_size")
                .or_else(|| raw.get_usize("d_model")),
            num_heads: raw.get_usize("num_heads"),
            head_dim: raw.get_usize("head_dim"),
            intermediate_size: raw
                .get_usize("intermediate_size")
                .or_else(|| raw.get_usize("hidden_features")),
            vocab_size: raw.get_usize("vocab_size"),
            embedding_dim: raw.get_usize("embedding_dim"),
            // Dense/Linear parameters
            in_features: raw.get_usize("in_features"),
            out_features: raw.get_usize("out_features"),
            kernel_size: raw.get_usize("kernel_size"),
            kernel_h: raw.get_usize("kernel_h"),
            kernel_w: raw.get_usize("kernel_w"),
            stride: raw
                .get_usize("stride")
                .or_else(|| raw.get_usize("stride_h")),
            padding: raw.get_usize("padding"),
            in_channels: raw.get_usize("in_channels"),
            out_channels: raw.get_usize("out_channels"),
            causal: raw.get_bool("causal").unwrap_or(false),
            num_kv_heads: raw.get_usize("num_kv_heads"),
            gated: raw.get_bool("gated").unwrap_or(false),
            activation: raw.get_string("activation"),
            num_experts: raw.get_usize("num_experts"),
            top_k: raw.get_usize("top_k"),
            shared_experts: raw.get_usize("shared_experts"),
            // CNN Modern architecture parameters
            num_blocks: raw.get_usize("num_blocks"),
            num_layers: raw.get_usize("num_layers"),
            expand_ratio: raw.get_usize("expand_ratio"),
            cardinality: raw.get_usize("cardinality"),
            bottleneck: raw.get_bool("bottleneck").unwrap_or(false),
            bottleneck_width: raw.get_usize("bottleneck_width"),
            groups: raw.get_usize("groups"),
            se: raw.get_bool("se").unwrap_or(false),
            h_swish: raw.get_bool("h_swish").unwrap_or(false),
            pool_type: raw.get_string("pool_type"),
            num_classes: raw.get_usize("num_classes"),
            num_anchors: raw.get_usize("num_anchors"),
            growth_rate: raw.get_usize("growth_rate"),
            compression: raw.get_f64("compression"),
            mlp_ratio: raw.get_f64("mlp_ratio"),
            num_bottlenecks: raw.get_usize("num_bottlenecks"),
            // State Space Model parameters
            state_dim: raw
                .get_usize("state_dim")
                .or_else(|| raw.get_usize("d_state")),
            expansion_factor: raw
                .get_usize("expansion_factor")
                .or_else(|| raw.get_usize("expand")),
            dt_rank: raw.get_usize("dt_rank"),
            conv_kernel_size: raw.get_usize("conv_kernel_size"),
            inner_rank: raw.get_usize("inner_rank"),
            time_step: raw.get_f64("time_step"),
            ssm_type: raw.get_string("ssm_type"),
            bidirectional: raw.get_bool("bidirectional").unwrap_or(false),
            use_conv1d: raw.get_bool("use_conv1d").unwrap_or(true),
            gate_activation: raw.get_string("gate_activation"),
            dt_min: raw.get_f64("dt_min"),
            dt_max: raw.get_f64("dt_max"),
            dt_init_floor: raw.get_f64("dt_init_floor"),
            // GAN parameters
            latent_dim: raw.get_usize("latent_dim"),
            style_dim: raw.get_usize("style_dim"),
            w_dim: raw.get_usize("w_dim"),
            spectral_norm: raw.get_bool("spectral_norm").unwrap_or(false),
            feature_matching: raw.get_bool("feature_matching").unwrap_or(false),
            progressive: raw.get_bool("progressive").unwrap_or(false),
            alpha: raw.get_f64("alpha"),
            truncation: raw.get_f64("truncation"),
            mapping_layers: raw.get_usize("mapping_layers"),
            num_channels: raw.get_usize("num_channels"),
            resolution: raw.get_usize("resolution"),
            base_channels: raw.get_usize("base_channels"),
            max_channels: raw.get_usize("max_channels"),
            minibatch_std_group: raw.get_usize("minibatch_std_group"),
            label_dim: raw.get_usize("label_dim"),
            // LSTM/RNN parameters
            rnn_hidden_size: raw.get_usize("rnn_hidden_size"),
            num_rnn_layers: raw.get_usize("num_rnn_layers"),
            bidirectional_rnn: raw.get_bool("bidirectional_rnn").unwrap_or(false),
            cell_type: raw.get_string("cell_type"),
            forget_bias: raw.get_f64("forget_bias"),
            peephole: raw.get_bool("peephole").unwrap_or(false),
            recurrent_dropout: raw.get_f64("recurrent_dropout"),
            zoneout: raw.get_f64("zoneout"),
            seq_len: raw.get_usize("seq_len"),
            vocab_size_rnn: raw.get_usize("vocab_size_rnn"),
            attention_type: raw.get_string("attention_type"),
            beam_width: raw.get_usize("beam_width"),
            // Diffusion parameters
            diffusion_timesteps: raw.get_usize("diffusion_timesteps"),
            noise_schedule: raw.get_string("noise_schedule"),
            beta_start: raw.get_f64("beta_start"),
            beta_end: raw.get_f64("beta_end"),
            image_size: raw.get_usize("image_size"),
            in_channels_diff: raw.get_usize("in_channels"),
            out_channels_diff: raw.get_usize("out_channels"),
            latent_channels: raw.get_usize("latent_channels"),
            down_block_types: raw.get_string_vec("down_block_types"),
            up_block_types: raw.get_string_vec("up_block_types"),
            block_out_channels: raw.get_usize_vec("block_out_channels"),
            layers_per_block: raw.get_usize("layers_per_block"),
            cross_attention_dim: raw.get_usize("cross_attention_dim"),
            attention_head_dim: raw.get_usize("attention_head_dim"),
            norm_num_groups: raw.get_usize("norm_num_groups"),
            resnet_time_scale_shift: raw.get_string("resnet_time_scale_shift"),
            vae_scale_factor: raw.get_usize("vae_scale_factor"),
            sample_size: raw.get_usize("sample_size"),
            conditioning_channels: raw.get_usize("conditioning_channels"),
            projection_class_embeddings_input_dim: raw
                .get_usize("projection_class_embeddings_input_dim"),
            class_embed_type: raw.get_string("class_embed_type"),
            addition_embed_type: raw.get_string("addition_embed_type"),
            time_embedding_dim: raw.get_usize("time_embedding_dim"),
            transformer_layers_per_block: raw.get_usize("transformer_layers_per_block"),
            num_attention_heads_diff: raw.get_usize("num_attention_heads"),
            // General
            dropout: raw.get_f64("dropout"),
            bias: raw.get_bool("bias").unwrap_or(true),
            param_count: raw.get_u64("param_count"),
            extra: raw.extra,
        }
    }
}

/// Custom equations for experimental layers
#[derive(Debug, Clone)]
pub struct CustomEquations {
    pub flops_forward: Option<String>,
    pub memory_activation: Option<String>,
    pub gradient: Option<String>,
    pub extra: HashMap<String, String>,
}

impl CustomEquations {
    pub fn from_raw(raw: RawCustomEquations) -> Self {
        Self {
            flops_forward: raw.flops_forward,
            memory_activation: raw.memory_activation,
            gradient: raw.gradient,
            extra: raw.extra,
        }
    }
}

/// Global parameters
#[derive(Debug, Clone, Default)]
pub struct GlobalParams {
    pub sequence_length: Option<usize>,
    pub vocab_size: Option<usize>,
    pub embedding_dim: Option<usize>,
    pub num_experts: Option<usize>,
    pub diffusion_timesteps: Option<usize>,
    pub graph_message_dim: Option<usize>,
    /// Total number of transformer layers in the full model (may be > number of JSON layers)
    pub num_layers: Option<u64>,
    /// Number of dense (non-MoE) layers in MoE models like DeepSeek-V3
    pub num_dense_layers: Option<u64>,
    // RNN/LSTM global parameters
    pub rnn_hidden_size: Option<usize>,
    pub num_rnn_layers: Option<usize>,
    pub bidirectional_rnn: Option<bool>,
    pub cell_type: Option<String>,
    // Diffusion global parameters
    pub image_size: Option<usize>,
    pub in_channels: Option<usize>,
    pub out_channels: Option<usize>,
    pub latent_channels: Option<usize>,
    pub cross_attention_dim: Option<usize>,
    pub attention_head_dim: Option<usize>,
    pub block_out_channels: Option<Vec<usize>>,
    pub down_block_types: Option<Vec<String>>,
    pub up_block_types: Option<Vec<String>>,
    pub layers_per_block: Option<usize>,
    pub vae_scale_factor: Option<usize>,
    pub sample_size: Option<usize>,
    pub noise_schedule: Option<String>,
    pub beta_start: Option<f64>,
    pub beta_end: Option<f64>,
    pub extra: HashMap<String, serde_json::Value>,
}

impl GlobalParams {
    pub fn from_raw(raw: RawGlobalParams) -> Self {
        // Also accept num_layers from the extra flat map (for backwards compat)
        let num_layers = raw
            .num_layers
            .or_else(|| raw.extra.get("num_layers").and_then(|v| v.as_u64()));
        let num_dense_layers = raw
            .num_dense_layers
            .or_else(|| raw.extra.get("num_dense_layers").and_then(|v| v.as_u64()));

        // Extract RNN/LSTM global params from extra
        let rnn_hidden_size = raw
            .extra
            .get("rnn_hidden_size")
            .and_then(|v| v.as_u64().map(|n| n as usize));
        let num_rnn_layers = raw
            .extra
            .get("num_rnn_layers")
            .and_then(|v| v.as_u64().map(|n| n as usize));
        let bidirectional_rnn = raw.extra.get("bidirectional_rnn").and_then(|v| v.as_bool());
        let cell_type = raw
            .extra
            .get("cell_type")
            .and_then(|v| v.as_str().map(|s| s.to_string()));

        // Extract Diffusion global params from extra
        let image_size = raw
            .extra
            .get("image_size")
            .and_then(|v| v.as_u64().map(|n| n as usize));
        let in_channels = raw
            .extra
            .get("in_channels")
            .and_then(|v| v.as_u64().map(|n| n as usize));
        let out_channels = raw
            .extra
            .get("out_channels")
            .and_then(|v| v.as_u64().map(|n| n as usize));
        let latent_channels = raw
            .extra
            .get("latent_channels")
            .and_then(|v| v.as_u64().map(|n| n as usize));
        let cross_attention_dim = raw
            .extra
            .get("cross_attention_dim")
            .and_then(|v| v.as_u64().map(|n| n as usize));
        let attention_head_dim = raw
            .extra
            .get("attention_head_dim")
            .and_then(|v| v.as_u64().map(|n| n as usize));
        let block_out_channels = raw
            .extra
            .get("block_out_channels")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|i| i.as_u64().map(|n| n as usize))
                    .collect()
            });
        let down_block_types = raw
            .extra
            .get("down_block_types")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|i| i.as_str().map(|s| s.to_string()))
                    .collect()
            });
        let up_block_types = raw
            .extra
            .get("up_block_types")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|i| i.as_str().map(|s| s.to_string()))
                    .collect()
            });
        let layers_per_block = raw
            .extra
            .get("layers_per_block")
            .and_then(|v| v.as_u64().map(|n| n as usize));
        let vae_scale_factor = raw
            .extra
            .get("vae_scale_factor")
            .and_then(|v| v.as_u64().map(|n| n as usize));
        let sample_size = raw
            .extra
            .get("sample_size")
            .and_then(|v| v.as_u64().map(|n| n as usize));
        let noise_schedule = raw
            .extra
            .get("noise_schedule")
            .and_then(|v| v.as_str().map(|s| s.to_string()));
        let beta_start = raw.extra.get("beta_start").and_then(|v| v.as_f64());
        let beta_end = raw.extra.get("beta_end").and_then(|v| v.as_f64());

        Self {
            sequence_length: raw.sequence_length,
            vocab_size: raw.vocab_size,
            embedding_dim: raw.embedding_dim,
            num_experts: raw.num_experts,
            diffusion_timesteps: raw.diffusion_timesteps,
            graph_message_dim: raw.graph_message_dim,
            num_layers,
            num_dense_layers,
            rnn_hidden_size,
            num_rnn_layers,
            bidirectional_rnn,
            cell_type,
            image_size,
            in_channels,
            out_channels,
            latent_channels,
            cross_attention_dim,
            attention_head_dim,
            block_out_channels,
            down_block_types,
            up_block_types,
            layers_per_block,
            vae_scale_factor,
            sample_size,
            noise_schedule,
            beta_start,
            beta_end,
            extra: raw.extra,
        }
    }
}

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub optimizer: String,
    pub learning_rate: f64,
    pub precision: String,
    pub gradient_checkpointing: bool,
    pub zero_stage: u8,
    pub max_steps: usize,
    pub warmup_steps: usize,
    pub parallelism: ParallelismConfig,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            optimizer: "adamw".to_string(),
            learning_rate: 0.001,
            precision: "fp32".to_string(),
            gradient_checkpointing: false,
            zero_stage: 0,
            max_steps: 0,
            warmup_steps: 0,
            parallelism: ParallelismConfig::default(),
        }
    }
}

impl TrainingConfig {
    pub fn from_raw(raw: RawTraining) -> Self {
        Self {
            batch_size: raw.batch_size,
            optimizer: raw.optimizer.unwrap_or_else(|| "adamw".to_string()),
            learning_rate: raw.learning_rate.unwrap_or(0.001),
            precision: raw.precision,
            gradient_checkpointing: raw.gradient_checkpointing,
            zero_stage: raw.zero_stage,
            max_steps: raw.max_steps,
            warmup_steps: raw.warmup_steps,
            parallelism: ParallelismConfig::from_raw(raw.parallelism),
        }
    }
}

/// Parallelism configuration
#[derive(Debug, Clone, Default)]
pub struct ParallelismConfig {
    pub data_parallel: u32,
    pub tensor_parallel: u32,
    pub pipeline_parallel: u32,
}

impl ParallelismConfig {
    pub fn from_raw(raw: RawParallelism) -> Self {
        Self {
            data_parallel: raw.data_parallel,
            tensor_parallel: raw.tensor_parallel,
            pipeline_parallel: raw.pipeline_parallel,
        }
    }

    pub fn total_gpus(&self) -> u32 {
        self.data_parallel * self.tensor_parallel * self.pipeline_parallel
    }
}

/// Hardware configuration
#[derive(Debug, Clone, Default)]
pub struct HardwareConfig {
    pub gpus: Vec<GpuConfig>,
    pub interconnect: String,
    pub interconnect_bandwidth_gbs: f64,
}

impl HardwareConfig {
    pub fn from_raw(raw: RawHardware) -> Self {
        Self {
            gpus: raw.gpus.into_iter().map(GpuConfig::from_raw).collect(),
            interconnect: raw.interconnect.unwrap_or_default(),
            interconnect_bandwidth_gbs: raw.interconnect_bandwidth_gb_s.unwrap_or(64.0),
        }
    }

    pub fn total_gpu_count(&self) -> u32 {
        self.gpus.iter().map(|g| g.count).sum()
    }

    pub fn primary_gpu(&self) -> Option<&GpuConfig> {
        self.gpus.first()
    }
}

/// GPU configuration
#[derive(Debug, Clone)]
pub struct GpuConfig {
    pub name: String,
    pub count: u32,
    pub memory_gb: u64,
    pub tflops_fp16: f64,
    pub tflops_fp32: f64,
    pub tflops_fp8: f64,
    pub memory_bandwidth_gbs: f64,
    pub tensor_cores: bool,
    pub nvlink: bool,
}

impl GpuConfig {
    pub fn from_raw(raw: RawGpu) -> Self {
        Self {
            name: raw.name,
            count: raw.count,
            memory_gb: raw.memory_gb.unwrap_or(40),
            tflops_fp16: raw.tflops_fp16.unwrap_or(100.0),
            tflops_fp32: raw.tflops_fp32.unwrap_or(20.0),
            tflops_fp8: raw.tflops_fp8.unwrap_or(200.0),
            memory_bandwidth_gbs: raw.memory_bandwidth_gb_s.unwrap_or(1000.0),
            tensor_cores: raw.tensor_cores.unwrap_or(true),
            nvlink: raw.nvlink.unwrap_or(false),
        }
    }
}

/// Data configuration
#[derive(Debug, Clone, Default)]
pub struct DataConfig {
    pub input_shape: Vec<usize>,
    pub dtype: String,
    pub vocab_size: Option<usize>,
    pub num_classes: Option<usize>,
    pub image_channels: Option<usize>,
    pub image_height: Option<usize>,
    pub image_width: Option<usize>,
}

impl DataConfig {
    pub fn from_raw(raw: RawData) -> Self {
        Self {
            input_shape: raw.input_shape,
            dtype: raw.dtype,
            vocab_size: raw.vocab_size,
            num_classes: raw.num_classes,
            image_channels: raw.image_channels,
            image_height: raw.image_height,
            image_width: raw.image_width,
        }
    }
}

/// Metrics configuration
#[derive(Debug, Clone, Default)]
pub struct MetricsConfig {
    pub calculate_all: bool,
    pub groups: MetricGroups,
}

impl MetricsConfig {
    pub fn from_raw(raw: RawMetricsConfig) -> Self {
        Self {
            calculate_all: raw.calculate_all,
            groups: MetricGroups::from_raw(raw.groups),
        }
    }
}

/// Metric groups
#[derive(Debug, Clone, Default)]
pub struct MetricGroups {
    pub structure: bool,
    pub compute: bool,
    pub memory: bool,
    pub training: bool,
    pub parallelism: bool,
    pub hardware: bool,
    pub performance: bool,
    pub cost: bool,
}

impl MetricGroups {
    pub fn from_raw(raw: RawMetricGroups) -> Self {
        Self {
            structure: raw.structure,
            compute: raw.compute,
            memory: raw.memory,
            training: raw.training,
            parallelism: raw.parallelism,
            hardware: raw.hardware,
            performance: raw.performance,
            cost: raw.cost,
        }
    }
}

/// Cost configuration
#[derive(Debug, Clone, Default)]
pub struct CostConfig {
    pub provider: String,
    pub gpu_hour_usd: f64,
    pub energy_kwh_usd: f64,
    pub pue_factor: f64,
}

impl CostConfig {
    pub fn from_raw(raw: RawCostConfig) -> Self {
        Self {
            provider: raw.provider.unwrap_or_default(),
            gpu_hour_usd: raw.gpu_hour_usd.unwrap_or(3.0),
            energy_kwh_usd: raw.energy_kwh_usd.unwrap_or(0.12),
            pue_factor: raw.pue_factor,
        }
    }
}
