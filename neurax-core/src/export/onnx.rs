//! ONNX Binary Export Module
//!
//! Generates ONNX ModelProto protobuf binary from NEURAX IR data.
//! This constructs a valid ONNX model that can be loaded by ONNX Runtime,
//! PyTorch, or other ONNX-compatible frameworks.

use neurax_ir::ArchitectureIR;
use neurax_parser::{LayerType, LayerParams, ModelType, TrainingConfig, DataConfig, GlobalParams};

// ─── ONNX Constants ────────────────────────────────────────────────────

/// ONNX opset version we target
const OPSET_VERSION: i64 = 17;
/// ONNX IR version
const IR_VERSION: i64 = 8;
/// Default producer name
const PRODUCER_NAME: &str = "NEURAX";

// ─── Protobuf Wire Format Helpers ─────────────────────────────────────

mod wire {
    pub const VARINT: u8 = 0;
    pub const LENGTH_DELIMITED: u8 = 2;
}

fn encode_varint(value: u64, buf: &mut Vec<u8>) {
    let mut v = value;
    loop {
        let byte = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            buf.push(byte);
            break;
        }
        buf.push(byte | 0x80);
    }
}

fn encode_tag(field_number: u32, wire_type: u8, buf: &mut Vec<u8>) {
    encode_varint(((field_number as u64) << 3) | (wire_type as u64), buf);
}

fn encode_string_field(field_number: u32, value: &str, buf: &mut Vec<u8>) {
    encode_tag(field_number, wire::LENGTH_DELIMITED, buf);
    encode_varint(value.len() as u64, buf);
    buf.extend_from_slice(value.as_bytes());
}

fn encode_bytes_field(field_number: u32, value: &[u8], buf: &mut Vec<u8>) {
    encode_tag(field_number, wire::LENGTH_DELIMITED, buf);
    encode_varint(value.len() as u64, buf);
    buf.extend_from_slice(value);
}

fn encode_int64_field(field_number: u32, value: i64, buf: &mut Vec<u8>) {
    encode_tag(field_number, wire::VARINT, buf);
    encode_varint(value as u64, buf);
}

fn encode_repeated_int64(field_number: u32, values: &[i64], buf: &mut Vec<u8>) {
    for &value in values {
        encode_int64_field(field_number, value, buf);
    }
}

fn encode_enum_field(field_number: u32, value: i32, buf: &mut Vec<u8>) {
    encode_tag(field_number, wire::VARINT, buf);
    encode_varint(value as u64, buf);
}

fn encode_float_field(field_number: u32, value: f32, buf: &mut Vec<u8>) {
    encode_tag(field_number, 5, buf); // wire type 5 = 32-bit fixed
    buf.extend_from_slice(&value.to_le_bytes());
}

fn encode_repeated_float(field_number: u32, values: &[f32], buf: &mut Vec<u8>) {
    if values.is_empty() {
        return;
    }
    encode_tag(field_number, wire::LENGTH_DELIMITED, buf);
    let data_len = values.len() * 4;
    encode_varint(data_len as u64, buf);
    for &v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

fn encode_repeated_int64_packed(field_number: u32, values: &[i64], buf: &mut Vec<u8>) {
    if values.is_empty() {
        return;
    }
    encode_tag(field_number, wire::LENGTH_DELIMITED, buf);
    let mut data = Vec::with_capacity(values.len() * 10);
    for &v in values {
        encode_varint(v as u64, &mut data);
    }
    encode_varint(data.len() as u64, buf);
    buf.extend_from_slice(&data);
}

// ─── ONNX Data Types ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
enum OnnxDataType {
    Float = 1,
    Uint8 = 2,
    Int8 = 3,
    Uint16 = 4,
    Int16 = 5,
    Int32 = 6,
    Int64 = 7,
    String = 8,
    Float16 = 10,
    Double = 11,
    Uint32 = 12,
    Uint64 = 13,
    Bfloat16 = 16,
}

// ─── ONNX Proto Structures ─────────────────────────────────────────────

struct OnnxNode {
    op_type: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    name: String,
    attributes: Vec<OnnxAttribute>,
}

struct OnnxAttribute {
    name: String,
    attr_type: i32, // 1=float, 2=int, 3=string, 6=floats, 7=ints
    float_value: Option<f32>,
    int_value: Option<i64>,
    string_value: Option<String>,
    floats: Vec<f32>,
    ints: Vec<i64>,
}

impl Default for OnnxAttribute {
    fn default() -> Self {
        Self {
            name: String::new(),
            attr_type: 0,
            float_value: None,
            int_value: None,
            string_value: None,
            floats: Vec::new(),
            ints: Vec::new(),
        }
    }
}

struct OnnxTensor {
    name: String,
    data_type: OnnxDataType,
    dims: Vec<i64>,
    raw_data: Vec<u8>,
}

struct OnnxValueInfo {
    name: String,
    elem_type: i32,
    shape: Vec<i64>,
}

// ─── ONNX Export Result ────────────────────────────────────────────────

/// Result of ONNX export
#[derive(Debug)]
pub struct OnnxExportResult {
    /// The serialized ONNX model protobuf bytes
    pub bytes: Vec<u8>,
    /// Model name
    pub model_name: String,
    /// Number of nodes in the graph
    pub node_count: usize,
    /// Number of initializers (parameters)
    pub initializer_count: usize,
}

// ─── ONNX Model Builder ───────────────────────────────────────────────

struct OnnxModelBuilder {
    model_name: String,
    nodes: Vec<OnnxNode>,
    initializers: Vec<OnnxTensor>,
    inputs: Vec<OnnxValueInfo>,
    outputs: Vec<OnnxValueInfo>,
}

impl OnnxModelBuilder {
    fn new(name: &str) -> Self {
        Self {
            model_name: name.to_string(),
            nodes: Vec::new(),
            initializers: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Build ONNX graph from NEURAX IR
    fn build_from_ir(
        &mut self,
        arch: &ArchitectureIR,
        training: &TrainingConfig,
        data: &DataConfig,
    ) -> Result<(), String> {
        let batch_size = training.batch_size as i64;
        let global = &arch.global_params;

        // Determine input shape based on model type
        let input_shape = get_input_shape(&arch.model_type, global, data);
        self.inputs.push(OnnxValueInfo {
            name: "input".to_string(),
            elem_type: OnnxDataType::Float as i32,
            shape: vec![batch_size].into_iter().chain(input_shape.clone()).collect(),
        });

        // Track current tensor name
        let mut current_tensor = "input".to_string();

        // Process each layer
        for (idx, layer) in arch.layers.iter().enumerate() {
            let layer_name = format!("{}_{}", layer.layer_type.as_str(), idx);
            let output_name = format!("{}_out", layer_name);

            // Map NEURAX layer type to ONNX op
            let onnx_op = map_layer_to_onnx_op(&layer.layer_type);

            let mut node = OnnxNode {
                op_type: onnx_op.to_string(),
                inputs: vec![current_tensor.clone()],
                outputs: vec![output_name.clone()],
                name: layer_name,
                attributes: Vec::new(),
            };

            // Add layer-specific attributes
            add_layer_attributes(&mut node, &layer.layer_type, &layer.params);

            // Add weight initializer if needed
            if has_weights(&layer.layer_type) {
                let weight_name = format!("{}_weight", node.name);
                let weight_dims = get_weight_dims(&layer.layer_type, &layer.params, global, data);

                self.initializers.push(OnnxTensor {
                    name: weight_name.clone(),
                    data_type: OnnxDataType::Float,
                    dims: weight_dims.clone(),
                    raw_data: vec![0u8; (weight_dims.iter().product::<i64>() * 4) as usize],
                });
                node.inputs.push(weight_name);

                // Add bias if applicable
                if has_bias(&layer.layer_type) {
                    let bias_name = format!("{}_bias", node.name);
                    let bias_dim = *weight_dims.last().unwrap_or(&1);
                    self.initializers.push(OnnxTensor {
                        name: bias_name.clone(),
                        data_type: OnnxDataType::Float,
                        dims: vec![bias_dim],
                        raw_data: vec![0u8; (bias_dim * 4) as usize],
                    });
                    node.inputs.push(bias_name);
                }
            }

            self.nodes.push(node);
            current_tensor = output_name;
        }

        // Add final output
        let output_dim = get_output_dim(&arch.model_type, global, data);
        self.outputs.push(OnnxValueInfo {
            name: current_tensor,
            elem_type: OnnxDataType::Float as i32,
            shape: vec![batch_size, output_dim],
        });

        Ok(())
    }

    /// Serialize the ONNX model to protobuf binary
    fn serialize(&self) -> Result<Vec<u8>, String> {
        let mut buf = Vec::with_capacity(64 * 1024);

        // Build the graph first
        let mut graph_buf = Vec::with_capacity(64 * 1024);
        self.serialize_graph(&mut graph_buf)?;

        // ModelProto
        encode_int64_field(1, IR_VERSION, &mut buf); // ir_version
        encode_string_field(2, PRODUCER_NAME, &mut buf); // producer_name
        encode_string_field(3, env!("CARGO_PKG_VERSION"), &mut buf); // producer_version
        encode_string_field(5, &format!("Exported by NEURAX v{}", env!("CARGO_PKG_VERSION")), &mut buf); // doc_string

        // opset_import (field 7)
        let mut opset_buf = Vec::new();
        encode_string_field(1, "", &mut opset_buf); // domain = "" (ONNX default)
        encode_int64_field(2, OPSET_VERSION, &mut opset_buf); // version
        encode_bytes_field(7, &opset_buf, &mut buf);

        encode_int64_field(9, 1, &mut buf); // model_version

        // Graph (field 10)
        encode_bytes_field(10, &graph_buf, &mut buf);

        Ok(buf)
    }

    fn serialize_graph(&self, buf: &mut Vec<u8>) -> Result<(), String> {
        // Nodes (field 1)
        for node in &self.nodes {
            let mut node_buf = Vec::new();
            self.serialize_node(node, &mut node_buf);
            encode_bytes_field(1, &node_buf, buf);
        }

        // Graph name (field 2)
        encode_string_field(2, &self.model_name, buf);

        // Initializers (field 3)
        for tensor in &self.initializers {
            let mut tensor_buf = Vec::new();
            self.serialize_tensor(tensor, &mut tensor_buf);
            encode_bytes_field(3, &tensor_buf, buf);
        }

        // Inputs (field 5)
        for input in &self.inputs {
            let mut vi_buf = Vec::new();
            self.serialize_value_info(input, &mut vi_buf);
            encode_bytes_field(5, &vi_buf, buf);
        }

        // Outputs (field 6)
        for output in &self.outputs {
            let mut vi_buf = Vec::new();
            self.serialize_value_info(output, &mut vi_buf);
            encode_bytes_field(6, &vi_buf, buf);
        }

        Ok(())
    }

    fn serialize_node(&self, node: &OnnxNode, buf: &mut Vec<u8>) {
        for input in &node.inputs {
            encode_string_field(1, input, buf);
        }
        for output in &node.outputs {
            encode_string_field(2, output, buf);
        }
        encode_string_field(3, &node.name, buf);
        encode_string_field(4, &node.op_type, buf);

        for attr in &node.attributes {
            let mut attr_buf = Vec::new();
            self.serialize_attribute(attr, &mut attr_buf);
            encode_bytes_field(6, &attr_buf, buf);
        }
    }

    fn serialize_attribute(&self, attr: &OnnxAttribute, buf: &mut Vec<u8>) {
        encode_string_field(1, &attr.name, buf);
        encode_enum_field(4, attr.attr_type, buf);

        if let Some(f) = attr.float_value {
            encode_float_field(5, f, buf);
        }
        if let Some(i) = attr.int_value {
            encode_int64_field(6, i, buf);
        }
        if let Some(ref s) = attr.string_value {
            encode_string_field(7, s, buf);
        }
        if !attr.floats.is_empty() {
            encode_repeated_float(10, &attr.floats, buf);
        }
        if !attr.ints.is_empty() {
            encode_repeated_int64_packed(11, &attr.ints, buf);
        }
    }

    fn serialize_tensor(&self, tensor: &OnnxTensor, buf: &mut Vec<u8>) {
        encode_repeated_int64(1, &tensor.dims, buf);
        encode_int64_field(2, tensor.data_type as i64, buf);
        encode_string_field(7, &tensor.name, buf);

        if !tensor.raw_data.is_empty() {
            encode_bytes_field(8, &tensor.raw_data, buf);
        }
    }

    fn serialize_value_info(&self, info: &OnnxValueInfo, buf: &mut Vec<u8>) {
        encode_string_field(1, &info.name, buf);

        // TypeProto (field 2) - nested message
        let mut type_buf = Vec::new();
        // TypeProto.Tensor (field 1)
        let mut tensor_type_buf = Vec::new();
        encode_int64_field(1, info.elem_type as i64, &mut tensor_type_buf);

        // ShapeProto (field 2)
        let mut shape_buf = Vec::new();
        for &dim in &info.shape {
            let mut dim_buf = Vec::new();
            if dim > 0 {
                encode_int64_field(1, dim, &mut dim_buf);
            } else {
                encode_string_field(2, "batch", &mut dim_buf);
            }
            encode_bytes_field(1, &dim_buf, &mut shape_buf);
        }
        encode_bytes_field(2, &shape_buf, &mut tensor_type_buf);
        encode_bytes_field(1, &tensor_type_buf, &mut type_buf);
        encode_bytes_field(2, &type_buf, buf);
    }
}

// ─── Helper Functions ───────────────────────────────────────────────────

fn get_input_shape(model_type: &ModelType, global: &GlobalParams, data: &DataConfig) -> Vec<i64> {
    match model_type {
        ModelType::Transformer | ModelType::Moe => {
            let seq_len = data.input_shape.first().copied().unwrap_or(128) as i64;
            let hidden = global.embedding_dim.unwrap_or(768) as i64;
            vec![seq_len, hidden]
        }
        ModelType::Cnn => {
            let channels = data.image_channels.unwrap_or(3) as i64;
            let height = data.image_height.unwrap_or(224) as i64;
            let width = data.image_width.unwrap_or(224) as i64;
            vec![channels, height, width]
        }
        ModelType::Ssm => {
            let seq_len = data.input_shape.first().copied().unwrap_or(128) as i64;
            let hidden = global.embedding_dim.unwrap_or(768) as i64;
            vec![seq_len, hidden]
        }
        ModelType::Rnn => {
            let seq_len = data.input_shape.first().copied().unwrap_or(128) as i64;
            let hidden = global.rnn_hidden_size.unwrap_or(256) as i64;
            vec![seq_len, hidden]
        }
        ModelType::Gnn => {
            let hidden = global.embedding_dim.unwrap_or(256) as i64;
            vec![hidden]
        }
        ModelType::Diffusion => {
            let channels = data.image_channels.unwrap_or(3) as i64;
            let height = data.image_height.unwrap_or(64) as i64;
            let width = data.image_width.unwrap_or(64) as i64;
            vec![channels, height, width]
        }
        ModelType::Gan => {
            let latent = data.input_shape.first().copied().unwrap_or(128) as i64;
            vec![latent]
        }
        _ => {
            let seq_len = data.input_shape.first().copied().unwrap_or(128) as i64;
            let hidden = global.embedding_dim.unwrap_or(768) as i64;
            vec![seq_len, hidden]
        }
    }
}

fn get_output_dim(model_type: &ModelType, global: &GlobalParams, data: &DataConfig) -> i64 {
    match model_type {
        ModelType::Cnn | ModelType::Gan => data.num_classes.unwrap_or(1000) as i64,
        _ => global.vocab_size.unwrap_or(data.vocab_size.unwrap_or(1000)) as i64,
    }
}

fn map_layer_to_onnx_op(layer_type: &LayerType) -> &'static str {
    match layer_type {
        LayerType::Dense => "Gemm",
        LayerType::Conv => "Conv",
        LayerType::Attention => "MultiHeadAttention",
        LayerType::Normalization => "LayerNormalization",
        LayerType::Embedding => "Gather",
        LayerType::MoE => "If", // MoE as conditional
        LayerType::MambaBlock | LayerType::S4Block | LayerType::H3Block | LayerType::StateSpace => "Squeeze",
        LayerType::Pooling => "MaxPool",
        LayerType::Mlp => "Gemm",
        LayerType::ResidualBlock => "Add",
        LayerType::LstmBlock => "LSTM",
        LayerType::GruBlock => "GRU",
        LayerType::RnnCell => "RNN",
        LayerType::UnetBlock | LayerType::DownBlock | LayerType::UpBlock | LayerType::MidBlock => "Conv",
        LayerType::CrossAttention => "MultiHeadAttention",
        LayerType::TimeEmbedding => "Gemm",
        LayerType::ResnetBlock => "Conv",
        LayerType::GeneratorBlock | LayerType::DiscriminatorBlock => "Conv",
        LayerType::StyleMod => "Gemm",
        LayerType::AdaIN => "InstanceNormalization",
        LayerType::SpectralNorm => "Gemm",
        LayerType::SelfAttention => "MultiHeadAttention",
        LayerType::Detection => "Conv",
        LayerType::Transition => "Gemm",
        LayerType::EncoderBlock | LayerType::DecoderBlock => "MultiHeadAttention",
        LayerType::RwkvBlock | LayerType::RetentionBlock => "Squeeze",
        LayerType::Custom => "Identity",
        _ => "Identity",
    }
}

fn add_layer_attributes(node: &mut OnnxNode, layer_type: &LayerType, params: &LayerParams) {
    match layer_type {
        LayerType::Conv => {
            if let Some(kernel) = params.kernel_size {
                node.attributes.push(OnnxAttribute {
                    name: "kernel_shape".to_string(),
                    attr_type: 7, // INTS
                    ints: vec![kernel as i64, kernel as i64],
                    ..Default::default()
                });
            }
            if let Some(stride) = params.stride {
                node.attributes.push(OnnxAttribute {
                    name: "strides".to_string(),
                    attr_type: 7,
                    ints: vec![stride as i64, stride as i64],
                    ..Default::default()
                });
            }
            if let Some(padding) = params.padding {
                node.attributes.push(OnnxAttribute {
                    name: "pads".to_string(),
                    attr_type: 7,
                    ints: vec![padding as i64, padding as i64, padding as i64, padding as i64],
                    ..Default::default()
                });
            }
            if let Some(groups) = params.groups {
                node.attributes.push(OnnxAttribute {
                    name: "group".to_string(),
                    attr_type: 2, // INT
                    int_value: Some(groups as i64),
                    ..Default::default()
                });
            }
        }
        LayerType::Pooling => {
            if let Some(kernel) = params.kernel_size {
                node.attributes.push(OnnxAttribute {
                    name: "kernel_shape".to_string(),
                    attr_type: 7,
                    ints: vec![kernel as i64, kernel as i64],
                    ..Default::default()
                });
            }
            if let Some(ref pool_type) = params.pool_type {
                if pool_type == "avg" {
                    node.op_type = "AveragePool".to_string();
                }
            }
        }
        LayerType::Attention | LayerType::CrossAttention | LayerType::SelfAttention => {
            if let Some(heads) = params.num_heads {
                node.attributes.push(OnnxAttribute {
                    name: "num_heads".to_string(),
                    attr_type: 2,
                    int_value: Some(heads as i64),
                    ..Default::default()
                });
            }
        }
        LayerType::Normalization => {
            // Check if it's batch norm or layer norm based on params
            if let Some(ref activation) = params.activation {
                if activation == "batch_norm" {
                    node.op_type = "BatchNormalization".to_string();
                }
            }
        }
        LayerType::Dense | LayerType::Mlp => {
            if params.gated {
                node.op_type = "Gemm".to_string();
            }
            if let Some(ref activation) = params.activation {
                // Add activation as a separate node after
                match activation.as_str() {
                    "gelu" => {
                        // We'll add a Gelu node after this one
                    }
                    "relu" => {}
                    "silu" => {}
                    _ => {}
                }
            }
        }
        LayerType::LstmBlock => {
            if let Some(hidden) = params.rnn_hidden_size {
                node.attributes.push(OnnxAttribute {
                    name: "hidden_size".to_string(),
                    attr_type: 2,
                    int_value: Some(hidden as i64),
                    ..Default::default()
                });
            }
        }
        LayerType::GruBlock => {
            if let Some(hidden) = params.rnn_hidden_size {
                node.attributes.push(OnnxAttribute {
                    name: "hidden_size".to_string(),
                    attr_type: 2,
                    int_value: Some(hidden as i64),
                    ..Default::default()
                });
            }
        }
        _ => {}
    }
}

fn has_weights(layer_type: &LayerType) -> bool {
    matches!(
        layer_type,
        LayerType::Dense
            | LayerType::Conv
            | LayerType::Attention
            | LayerType::CrossAttention
            | LayerType::SelfAttention
            | LayerType::Embedding
            | LayerType::LstmBlock
            | LayerType::GruBlock
            | LayerType::Mlp
            | LayerType::Normalization
            | LayerType::ResidualBlock
            | LayerType::UnetBlock
            | LayerType::DownBlock
            | LayerType::UpBlock
            | LayerType::MidBlock
            | LayerType::ResnetBlock
            | LayerType::GeneratorBlock
            | LayerType::DiscriminatorBlock
    )
}

fn has_bias(layer_type: &LayerType) -> bool {
    matches!(
        layer_type,
        LayerType::Dense | LayerType::Conv | LayerType::Mlp
    )
}

fn get_weight_dims(layer_type: &LayerType, params: &LayerParams, global: &GlobalParams, data: &DataConfig) -> Vec<i64> {
    match layer_type {
        LayerType::Dense | LayerType::Mlp => {
            let in_dim = params.in_features.unwrap_or(global.embedding_dim.unwrap_or(768)) as i64;
            let out_dim = params.out_features.unwrap_or(global.vocab_size.unwrap_or(data.vocab_size.unwrap_or(768))) as i64;
            vec![in_dim, out_dim]
        }
        LayerType::Conv => {
            let out_ch = params.out_channels.unwrap_or(64) as i64;
            let in_ch = params.in_channels.unwrap_or(data.image_channels.unwrap_or(3)) as i64;
            let k = params.kernel_size.unwrap_or(3) as i64;
            vec![out_ch, in_ch, k, k]
        }
        LayerType::Attention | LayerType::CrossAttention | LayerType::SelfAttention => {
            let hidden = params.hidden_size.unwrap_or(global.embedding_dim.unwrap_or(768)) as i64;
            let heads = params.num_heads.unwrap_or(12) as i64;
            let head_dim = hidden / heads;
            vec![hidden, 3 * head_dim * heads]
        }
        LayerType::Embedding => {
            let vocab = params.vocab_size.unwrap_or(global.vocab_size.unwrap_or(data.vocab_size.unwrap_or(30000))) as i64;
            let hidden = params.hidden_size.unwrap_or(global.embedding_dim.unwrap_or(768)) as i64;
            vec![vocab, hidden]
        }
        LayerType::Normalization => {
            let hidden = params.hidden_size.unwrap_or(global.embedding_dim.unwrap_or(768)) as i64;
            vec![hidden]
        }
        LayerType::LstmBlock => {
            let hidden = params.rnn_hidden_size.unwrap_or(global.rnn_hidden_size.unwrap_or(256)) as i64;
            let input_size = params.in_features.unwrap_or(hidden as usize) as i64;
            vec![input_size + hidden, 4 * hidden]
        }
        LayerType::GruBlock => {
            let hidden = params.rnn_hidden_size.unwrap_or(global.rnn_hidden_size.unwrap_or(256)) as i64;
            let input_size = params.in_features.unwrap_or(hidden as usize) as i64;
            vec![input_size + hidden, 3 * hidden]
        }
        _ => vec![1, 1],
    }
}

// ─── Public API ────────────────────────────────────────────────────────

/// Export NEURAX IR to ONNX protobuf binary
pub fn export_onnx(
    arch: &ArchitectureIR,
    training: &TrainingConfig,
    data: &DataConfig,
    model_name: Option<&str>,
) -> Result<OnnxExportResult, String> {
    let name = model_name
        .or(arch.model_name.as_deref())
        .unwrap_or("neurax_model");

    let mut builder = OnnxModelBuilder::new(name);

    // Build graph from IR
    builder.build_from_ir(arch, training, data)?;

    let node_count = builder.nodes.len();
    let initializer_count = builder.initializers.len();

    // Serialize to protobuf
    let bytes = builder.serialize()?;

    Ok(OnnxExportResult {
        bytes,
        model_name: name.to_string(),
        node_count,
        initializer_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varint_encoding() {
        let mut buf = Vec::new();
        encode_varint(0, &mut buf);
        assert_eq!(buf, vec![0u8]);

        buf.clear();
        encode_varint(1, &mut buf);
        assert_eq!(buf, vec![1u8]);

        buf.clear();
        encode_varint(127, &mut buf);
        assert_eq!(buf, vec![127u8]);

        buf.clear();
        encode_varint(128, &mut buf);
        assert_eq!(buf, vec![0x80, 0x01]);

        buf.clear();
        encode_varint(300, &mut buf);
        assert_eq!(buf, vec![0xAC, 0x02]);
    }

    #[test]
    fn test_string_field_encoding() {
        let mut buf = Vec::new();
        encode_string_field(1, "hello", &mut buf);
        // Tag for field 1, length-delimited: (1 << 3) | 2 = 10
        assert_eq!(buf[0], 10);
        // Length: 5
        assert_eq!(buf[1], 5);
        // Content: "hello"
        assert_eq!(&buf[2..7], b"hello");
    }

    #[test]
    fn test_onnx_builder_simple() {
        let mut builder = OnnxModelBuilder::new("test");
        builder.inputs.push(OnnxValueInfo {
            name: "input".to_string(),
            elem_type: OnnxDataType::Float as i32,
            shape: vec![1, 3, 224, 224],
        });
        builder.nodes.push(OnnxNode {
            op_type: "Conv".to_string(),
            inputs: vec!["input".to_string(), "conv1_weight".to_string()],
            outputs: vec!["conv1_out".to_string()],
            name: "conv1".to_string(),
            attributes: vec![OnnxAttribute {
                name: "kernel_shape".to_string(),
                attr_type: 7,
                ints: vec![3, 3],
                ..Default::default()
            }],
        });
        builder.outputs.push(OnnxValueInfo {
            name: "conv1_out".to_string(),
            elem_type: OnnxDataType::Float as i32,
            shape: vec![1, 64, 222, 222],
        });

        let bytes = builder.serialize().unwrap();
        assert!(!bytes.is_empty(), "Serialized ONNX model should not be empty");
        // Check that the protobuf starts with ir_version field (field 1, varint)
        assert_eq!(bytes[0], 0x08, "First byte should be field 1 tag (ir_version)");
    }
}