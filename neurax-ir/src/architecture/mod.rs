//! Architecture IR - Premier dialecte du pipeline NEURAX

mod ir;
mod pass;
mod metrics;

pub use ir::*;
pub use pass::*;
pub use metrics::*;

use neurax_parser::{Layer, LayerType};
use neurax_formulas::*;

/// Calculate parameters for a single layer
pub fn calculate_layer_params(layer: &Layer) -> u64 {
    match layer.layer_type {
        LayerType::Embedding => {
            let vocab = layer.params.vocab_size.unwrap_or(50000);
            let dim = layer.params.embedding_dim.unwrap_or(layer.params.hidden_size.unwrap_or(512));
            embedding::embedding_params(vocab, dim)
        }
        LayerType::Attention => {
            let hidden = layer.params.hidden_size.unwrap_or(512);
            let heads = layer.params.num_heads.unwrap_or(8);
            let kv_heads = layer.params.num_kv_heads.unwrap_or(heads);
            if kv_heads == heads {
                attention::attention_params(hidden, heads, layer.params.bias)
            } else {
                attention::gqa_params(hidden, heads, kv_heads, layer.params.bias)
            }
        }
        LayerType::Mlp => {
            let hidden = layer.params.hidden_size.unwrap_or(512);
            let intermediate = layer.params.intermediate_size.unwrap_or(4 * hidden);
            if layer.params.gated {
                mlp::gated_mlp_params(hidden, intermediate, layer.params.bias)
            } else {
                mlp::mlp_params(hidden, intermediate, layer.params.bias)
            }
        }
        LayerType::Dense => {
            let in_features = layer.params.in_features
                .or(layer.params.in_channels)
                .or(layer.params.hidden_size)
                .unwrap_or(512);
            let out_features = layer.params.out_features
                .or(layer.params.out_channels)
                .or(layer.params.hidden_size)
                .unwrap_or(512);
            let bias = if layer.params.bias { out_features } else { 0 };
            (in_features * out_features + bias) as u64
        }
        LayerType::Conv => {
            let in_ch = layer.params.in_channels.unwrap_or(3);
            let out_ch = layer.params.out_channels.unwrap_or(64);
            let kh = layer.params.kernel_h.unwrap_or(layer.params.kernel_size.unwrap_or(3));
            let kw = layer.params.kernel_w.unwrap_or(layer.params.kernel_size.unwrap_or(3));
            let groups = layer.params.groups.unwrap_or(1);
            conv::conv2d_params(in_ch, out_ch, kh, kw, groups, layer.params.bias)
        }
        LayerType::Normalization => {
            let hidden = layer.params.hidden_size.unwrap_or(512);
            // Check if it's RMSNorm or LayerNorm
            if layer.params.activation.as_deref() == Some("rms") {
                normalization::rms_norm_params(hidden)
            } else {
                normalization::layer_norm_params(hidden, true)
            }
        }
        LayerType::Pooling => 0,
        LayerType::MoE => {
            let hidden = layer.params.hidden_size.unwrap_or(512);
            let intermediate = layer.params.intermediate_size.unwrap_or(4 * hidden);
            let num_experts = layer.params.num_experts.unwrap_or(8);
            let shared_experts = layer.params.shared_experts.unwrap_or(1);
            // Each expert is a gated MLP (2 matrices for up/down projection)
            let expert_params = mlp::gated_mlp_params(hidden, intermediate, layer.params.bias);
            // Router + all experts + shared experts
            moe::moe_params_with_shared(hidden, intermediate, num_experts, shared_experts, expert_params)
        }
        // CNN layer types - use dedicated formulas
        LayerType::ResidualBlock => {
            let in_ch = layer.params.in_channels.unwrap_or(64);
            let out_ch = layer.params.out_channels.unwrap_or(64);
            let stride = layer.params.stride.unwrap_or(1);
            cnn_blocks::resnet_basic_block_params(in_ch, out_ch, stride, layer.params.bias)
        }
        LayerType::Mbconv => {
            let in_ch = layer.params.in_channels.unwrap_or(32);
            let out_ch = layer.params.out_channels.unwrap_or(16);
            let expand = layer.params.expansion_factor.unwrap_or(6);
            let kernel = layer.params.kernel_size.unwrap_or(3);
            let stride = layer.params.stride.unwrap_or(1);
            cnn_blocks::mbconv_params(in_ch, out_ch, expand, kernel, stride, layer.params.bias)
        }
        LayerType::Inception => {
            let in_ch = layer.params.in_channels.unwrap_or(288);
            let out_1x1 = layer.params.out_channels.unwrap_or(64);
            cnn_blocks::inception_module_params(
                in_ch, out_1x1,
                out_1x1 / 2, out_1x1,  // 3x3 branch
                out_1x1 / 8, out_1x1 / 2,  // 5x5 branch
                out_1x1,  // pool branch
                layer.params.bias
            )
        }
        LayerType::DenseBlock => {
            let in_ch = layer.params.in_channels.unwrap_or(64);
            let growth = layer.params.growth_rate.unwrap_or(32);
            let num_layers = layer.params.num_layers.unwrap_or(4);
            cnn_blocks::dense_block_params(in_ch, growth, num_layers, 4, layer.params.bias)
        }
        LayerType::ConvnextBlock => {
            let channels = layer.params.hidden_size.unwrap_or(96);
            let mlp_ratio = layer.params.mlp_ratio.unwrap_or(4.0);
            cnn_blocks::convnext_block_params(channels, mlp_ratio, layer.params.bias)
        }
        LayerType::ShuffleUnit => {
            let in_ch = layer.params.in_channels.unwrap_or(64);
            let out_ch = layer.params.out_channels.unwrap_or(64);
            let groups = layer.params.groups.unwrap_or(2);
            let stride = layer.params.stride.unwrap_or(1);
            cnn_blocks::shuffle_unit_params(in_ch, out_ch, groups, stride, layer.params.bias)
        }
        LayerType::C2f => {
            let in_ch = layer.params.in_channels.unwrap_or(64);
            let out_ch = layer.params.out_channels.unwrap_or(64);
            let num_bn = layer.params.num_bottlenecks.unwrap_or(3);
            cnn_blocks::c2f_block_params(in_ch, out_ch, num_bn, true, layer.params.bias)
        }
        LayerType::Detection | LayerType::Transition => {
            // Detection heads and transition layers are typically simple convs
            let in_ch = layer.params.in_channels.unwrap_or(256);
            let out_ch = layer.params.out_channels.unwrap_or(256);
            let kernel = layer.params.kernel_size.unwrap_or(3);
            conv::conv2d_params(in_ch, out_ch, kernel, kernel, 1, layer.params.bias)
        }
        // State Space Model layer types - use dedicated formulas
        LayerType::MambaBlock => {
            let hidden = layer.params.hidden_size.unwrap_or(512);
            let state_dim = layer.params.state_dim.unwrap_or(16);
            let expansion = layer.params.expansion_factor.unwrap_or(2);
            ssm::mamba_params(hidden, state_dim, expansion)
        }
        LayerType::S4Block | LayerType::H3Block | LayerType::StateSpace 
        | LayerType::RwkvBlock | LayerType::RetentionBlock => {
            let hidden = layer.params.hidden_size.unwrap_or(512);
            let state_dim = layer.params.state_dim.unwrap_or(16);
            let expansion = layer.params.expansion_factor.unwrap_or(2);
            ssm::mamba_params(hidden, state_dim, expansion)
        }
        // GAN layer types
        LayerType::GeneratorBlock | LayerType::DiscriminatorBlock => {
            let in_ch = layer.params.in_channels.unwrap_or(64);
            let out_ch = layer.params.out_channels.unwrap_or(64);
            let kh = layer.params.kernel_size.unwrap_or(3);
            conv::conv2d_params(in_ch, out_ch, kh, kh, 1, layer.params.bias)
        }
        LayerType::StyleMod => {
            // Style modulation: affine transform per channel
            let channels = layer.params.out_channels.unwrap_or(512);
            (channels * 2) as u64 // scale + bias per channel
        }
        LayerType::AdaIN => {
            // Adaptive Instance Norm: no params, uses style input
            0
        }
        LayerType::MinibatchStd => 0, // No params
        LayerType::PixelNorm => 0,    // No params
        LayerType::SelfAttention => {
            let channels = layer.params.out_channels.unwrap_or(512);
            attention::attention_params(channels, channels / 64, false)
        }
        LayerType::SpectralNorm => {
            // Spectral norm adds one vector per weight matrix
            let in_ch = layer.params.in_channels.unwrap_or(64);
            let out_ch = layer.params.out_channels.unwrap_or(64);
            (in_ch * out_ch / out_ch) as u64 // u vector
        }
        LayerType::ProgressiveBlock => {
            let in_ch = layer.params.in_channels.unwrap_or(64);
            let out_ch = layer.params.out_channels.unwrap_or(64);
            let kh = layer.params.kernel_size.unwrap_or(3);
            conv::conv2d_params(in_ch, out_ch, kh, kh, 1, layer.params.bias)
        }
        // LSTM/RNN layer types - use dedicated formulas
        LayerType::LstmBlock => {
            let hidden = layer.params.rnn_hidden_size.unwrap_or(512);
            let input_size = layer.params.hidden_size.unwrap_or(hidden);
            let bidir_mult = if layer.params.bidirectional_rnn { 2 } else { 1 };
            rnn::lstm_params(hidden, input_size, true) * bidir_mult
        }
        LayerType::GruBlock => {
            let hidden = layer.params.rnn_hidden_size.unwrap_or(512);
            let input_size = layer.params.hidden_size.unwrap_or(hidden);
            let bidir_mult = if layer.params.bidirectional_rnn { 2 } else { 1 };
            rnn::gru_params(hidden, input_size, true) * bidir_mult
        }
        LayerType::RnnCell => {
            let hidden = layer.params.rnn_hidden_size.unwrap_or(512);
            let input_size = layer.params.hidden_size.unwrap_or(hidden);
            rnn::rnn_params(hidden, input_size, true)
        }
        LayerType::Bidirectional => {
            let hidden = layer.params.rnn_hidden_size.unwrap_or(512);
            let input_size = layer.params.hidden_size.unwrap_or(hidden);
            rnn::lstm_params(hidden, input_size, true) * 2
        }
        LayerType::EncoderBlock | LayerType::DecoderBlock => {
            let hidden = layer.params.rnn_hidden_size.unwrap_or(512);
            let input_size = layer.params.hidden_size.unwrap_or(hidden);
            rnn::lstm_params(hidden, input_size, true)
        }
        // Diffusion layer types - use dedicated formulas
        LayerType::UnetBlock | LayerType::ResnetBlock => {
            let in_ch = layer.params.in_channels_diff.unwrap_or(layer.params.in_channels.unwrap_or(320));
            let out_ch = layer.params.out_channels_diff.unwrap_or(layer.params.out_channels.unwrap_or(320));
            // UNet ResNet block: 2 convs + 2 norms + skip
            cnn_blocks::resnet_basic_block_params(in_ch, out_ch, 1, layer.params.bias)
        }
        LayerType::TimeEmbedding | LayerType::TimestepBlock => {
            let channels = layer.params.hidden_size.unwrap_or(320);
            // Time embedding: Linear + SiLU + Linear
            mlp::mlp_params(channels, channels * 4, true)
        }
        LayerType::CrossAttention => {
            let hidden = layer.params.hidden_size.unwrap_or(320);
            let heads = layer.params.num_heads.unwrap_or(8);
            // Cross attention: Q from hidden, K,V from conditioning
            attention::attention_params(hidden, heads, true)
        }
        LayerType::DownBlock | LayerType::UpBlock | LayerType::MidBlock => {
            let in_ch = layer.params.in_channels_diff.unwrap_or(layer.params.in_channels.unwrap_or(320));
            let out_ch = layer.params.out_channels_diff.unwrap_or(layer.params.out_channels.unwrap_or(320));
            cnn_blocks::resnet_basic_block_params(in_ch, out_ch, 1, layer.params.bias)
        }
        LayerType::ConditionBlock => {
            let hidden = layer.params.hidden_size.unwrap_or(320);
            mlp::mlp_params(hidden, hidden * 4, true)
        }
        LayerType::NoisePredictor => {
            let channels = layer.params.out_channels_diff.unwrap_or(4);
            // Final conv to predict noise
            conv::conv2d_params(channels, channels, 3, 3, 1, false)
        }
        LayerType::VaeEncoder | LayerType::VaeDecoder => {
            let in_ch = layer.params.in_channels.unwrap_or(3);
            let out_ch = layer.params.out_channels_diff.unwrap_or(4);
            // VAE encoder/decoder: multiple convs (simplified)
            conv::conv2d_params(in_ch, out_ch, 3, 3, 1, false)
        }
        // Custom layer - use param_count if provided, else estimate from shapes
        LayerType::Custom => {
            // Try to estimate from custom equations or use a default
            layer.params.param_count.unwrap_or(0)
        }
    }
}
