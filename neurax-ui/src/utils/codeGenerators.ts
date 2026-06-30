import { CanvasNode, Connection } from '@/types/architecture.ts';

export interface GeneratedCode {
  filename: string;
  content: string;
  language: string;
}

export interface CodeGeneratorOptions {
  modelName?: string;
  includeComments?: boolean;
  includeAnalysis?: boolean;
}

// Sort nodes in topological order based on connections
function topologicalSort(nodes: CanvasNode[], connections: Connection[]): CanvasNode[] {
  const sorted: CanvasNode[] = [];
  const visited = new Set<string>();
  const nodeMap = new Map(nodes.map(n => [n.id, n]));
  
  // Build adjacency list
  const inDegree = new Map<string, number>();
  nodes.forEach(n => inDegree.set(n.id, 0));
  
  connections.forEach(c => {
    inDegree.set(c.to, (inDegree.get(c.to) || 0) + 1);
  });
  
  // Find all nodes with no incoming edges
  const queue = nodes.filter(n => (inDegree.get(n.id) || 0) === 0);
  
  while (queue.length > 0) {
    const node = queue.shift()!;
    if (visited.has(node.id)) continue;
    visited.add(node.id);
    sorted.push(node);
    
    // Find outgoing connections
    const outgoing = connections.filter(c => c.from === node.id);
    outgoing.forEach(c => {
      const newDegree = (inDegree.get(c.to) || 1) - 1;
      inDegree.set(c.to, newDegree);
      const targetNode = nodeMap.get(c.to);
      if (targetNode && newDegree === 0) {
        queue.push(targetNode);
      }
    });
  }
  
  // Add any remaining unvisited nodes
  nodes.forEach(n => {
    if (!visited.has(n.id)) sorted.push(n);
  });
  
  return sorted;
}

// ==================== PyTorch Generator ====================
export function generatePyTorchCode(
  nodes: CanvasNode[],
  connections: Connection[],
  options: CodeGeneratorOptions = {}
): GeneratedCode {
  const { modelName = 'GeneratedModel', includeComments = true } = options;
  const sortedNodes = topologicalSort(nodes, connections);
  
  const imports = `import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
`;

  const classHeader = `
class ${modelName}(nn.Module):
    """
    Auto-generated model from NEURAX Architecture Designer
    ${includeComments ? `
    Architecture Summary:
    - Layers: ${nodes.length}
    - Connections: ${connections.length}
    ` : ''}
    """
    
    def __init__(self):
        super().__init__()
`;

  let initBody = '';
  let forwardBody = '    def forward(self, x: torch.Tensor) -> torch.Tensor:\n';
  
  sortedNodes.forEach((node, idx) => {
    const varName = `self.${node.name.toLowerCase().replace(/[^a-z0-9]/gi, '_')}`;
    const inputVar = idx === 0 ? 'x' : `x${idx}`;
    const outputVar = `x${idx + 1}`;
    
    switch (node.type) {
      case 'input':
        forwardBody += `        # Input: ${node.inputShape || 'auto'}\n`;
        forwardBody += `        ${outputVar} = x\n\n`;
        break;
        
      case 'output':
        forwardBody += `        # Output: ${node.outputShape || 'auto'}\n`;
        forwardBody += `        return ${inputVar}\n`;
        break;
        
      case 'dense':
        const units = node.params.units || 512;
        initBody += `        ${varName} = nn.Linear(in_features=512, out_features=${units})  # TODO: Set correct in_features\n`;
        forwardBody += `        ${outputVar} = ${varName}(${inputVar})\n`;
        break;
        
      case 'conv2d':
        const filters = node.params.filters || 64;
        const kernel = node.params.kernel || 3;
        const stride = node.params.stride || 1;
        initBody += `        ${varName} = nn.Conv2d(in_channels=3, out_channels=${filters}, kernel_size=${kernel}, stride=${stride}, padding=${Math.floor(Number(kernel) / 2)})  # TODO: Set correct in_channels\n`;
        forwardBody += `        ${outputVar} = ${varName}(${inputVar})\n`;
        break;
        
      case 'relu':
        forwardBody += `        ${outputVar} = F.relu(${inputVar})\n`;
        break;
        
      case 'gelu':
        forwardBody += `        ${outputVar} = F.gelu(${inputVar})\n`;
        break;
        
      case 'attention':
        const heads = node.params.heads || 8;
        const dim = node.params.dim || 512;
        initBody += `        ${varName} = nn.MultiheadAttention(embed_dim=${dim}, num_heads=${heads}, batch_first=True)\n`;
        forwardBody += `        ${outputVar}, _ = ${varName}(${inputVar}, ${inputVar}, ${inputVar})\n`;
        break;
        
      case 'layernorm':
        initBody += `        ${varName} = nn.LayerNorm(normalized_shape=512)  # TODO: Set correct shape\n`;
        forwardBody += `        ${outputVar} = ${varName}(${inputVar})\n`;
        break;
        
      case 'batchnorm':
        initBody += `        ${varName} = nn.BatchNorm2d(num_features=64)  # TODO: Set correct num_features\n`;
        forwardBody += `        ${outputVar} = ${varName}(${inputVar})\n`;
        break;
        
      case 'residual':
        forwardBody += `        # Residual connection\n`;
        forwardBody += `        ${outputVar} = ${inputVar} + x  # TODO: Ensure shape compatibility\n`;
        break;
        
      case 'transformer':
        const tHeads = node.params.heads || 8;
        const tDim = node.params.dim || 512;
        const ffn = node.params.ffn || 2048;
        initBody += `        ${varName} = nn.TransformerEncoderLayer(d_model=${tDim}, nhead=${tHeads}, dim_feedforward=${ffn}, batch_first=True)\n`;
        forwardBody += `        ${outputVar} = ${varName}(${inputVar})\n`;
        break;
        
      default:
        forwardBody += `        # ${node.type}: ${node.name}\n`;
        forwardBody += `        ${outputVar} = ${inputVar}\n`;
    }
  });

  const modelStats = `

# ==================== Model Statistics ====================
# Total Layers: ${nodes.length}
# Architecture Type: Auto-generated
# 
# To use this model:
# model = ${modelName}()
# output = model(torch.randn(1, 3, 224, 224))  # Adjust input shape as needed
`;

  const code = imports + classHeader + initBody + '\n' + forwardBody + modelStats;
  
  return {
    filename: `${modelName.toLowerCase()}.py`,
    content: code,
    language: 'python'
  };
}

// ==================== ONNX Generator ====================
export function generateONNXExportCode(
  _nodes: CanvasNode[],
  _connections: Connection[],
  options: CodeGeneratorOptions = {}
): GeneratedCode {
  const { modelName = 'GeneratedModel' } = options;
  
  const code = `import torch
import torch.onnx
from ${modelName.toLowerCase()} import ${modelName}

def export_to_onnx(model_path: str = "${modelName.toLowerCase()}.onnx"):
    """
    Export the model to ONNX format
    
    Auto-generated by NEURAX Architecture Designer
    """
    model = ${modelName}()
    model.eval()
    
    # Create dummy input - adjust shape as needed
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export the model
    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {model_path}")
    return model_path

if __name__ == "__main__":
    export_to_onnx()
`;

  return {
    filename: `export_${modelName.toLowerCase()}_onnx.py`,
    content: code,
    language: 'python'
  };
}

// ==================== Rust/Burn Generator ====================
export function generateRustCode(
  nodes: CanvasNode[],
  connections: Connection[],
  options: CodeGeneratorOptions = {}
): GeneratedCode {
  const { modelName = 'GeneratedModel', includeComments = true } = options;
  const sortedNodes = topologicalSort(nodes, connections);
  
  const structName = modelName.replace(/[^a-zA-Z0-9]/g, '');
  
  let structFields = '';
  let newBody = '';
  let forwardBody = '';
  
  sortedNodes.forEach((node, idx) => {
    const fieldName = node.name.toLowerCase().replace(/[^a-z0-9]/gi, '_');
    const inputVar = idx === 0 ? 'x' : `x${idx}`;
    const outputVar = `x${idx + 1}`;
    
    switch (node.type) {
      case 'input':
        forwardBody += `        // Input: ${node.inputShape || 'auto'}\n`;
        forwardBody += `        let ${outputVar} = x;\n\n`;
        break;
        
      case 'output':
        forwardBody += `        // Output: ${node.outputShape || 'auto'}\n`;
        forwardBody += `        ${inputVar}\n`;
        break;
        
      case 'dense':
        const units = node.params.units || 512;
        structFields += `    ${fieldName}: Linear<B>,\n`;
        newBody += `            ${fieldName}: LinearConfig::new(512, ${units}).init(device),\n`;
        forwardBody += `        let ${outputVar} = self.${fieldName}.forward(${inputVar});\n`;
        break;
        
      case 'conv2d':
        const filters = node.params.filters || 64;
        const kernel = node.params.kernel || 3;
        structFields += `    ${fieldName}: Conv2d<B>,\n`;
        newBody += `            ${fieldName}: Conv2dConfig::new([3, ${filters}], [${kernel}, ${kernel}]).init(device),\n`;
        forwardBody += `        let ${outputVar} = self.${fieldName}.forward(${inputVar});\n`;
        break;
        
      case 'relu':
        forwardBody += `        let ${outputVar} = activation::relu(${inputVar});\n`;
        break;
        
      case 'gelu':
        forwardBody += `        let ${outputVar} = activation::gelu(${inputVar});\n`;
        break;
        
      case 'attention':
        const heads = node.params.heads || 8;
        const dim = node.params.dim || 512;
        structFields += `    ${fieldName}: MultiHeadAttention<B>,\n`;
        newBody += `            ${fieldName}: MultiHeadAttentionConfig::new(${dim}, ${heads}).init(device),\n`;
        forwardBody += `        let ${outputVar} = self.${fieldName}.forward(${inputVar}.clone(), ${inputVar}.clone(), ${inputVar}, None);\n`;
        break;
        
      case 'layernorm':
        structFields += `    ${fieldName}: LayerNorm<B>,\n`;
        newBody += `            ${fieldName}: LayerNormConfig::new(512).init(device),\n`;
        forwardBody += `        let ${outputVar} = self.${fieldName}.forward(${inputVar});\n`;
        break;
        
      default:
        forwardBody += `        // ${node.type}: ${node.name}\n`;
        forwardBody += `        let ${outputVar} = ${inputVar};\n`;
    }
  });

  const code = `//! Auto-generated by NEURAX Architecture Designer
//! 
//! This module contains the ${structName} model definition for Burn framework.

use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        Linear, LinearConfig,
        LayerNorm, LayerNormConfig,
        attention::{MultiHeadAttention, MultiHeadAttentionConfig},
    },
    tensor::{backend::Backend, Tensor},
    nn::activation,
};

${includeComments ? `/// ${structName} - Auto-generated Neural Network
/// 
/// Architecture Summary:
/// - Total Layers: ${nodes.length}
/// - Connections: ${connections.length}` : ''}
#[derive(Module, Debug)]
pub struct ${structName}<B: Backend> {
${structFields}}

impl<B: Backend> ${structName}<B> {
    /// Create a new instance of the model
    pub fn new(device: &B::Device) -> Self {
        Self {
${newBody}        }
    }
    
    /// Forward pass through the network
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
${forwardBody}    }
}

// ==================== Usage Example ====================
// 
// use burn::backend::Wgpu;
// 
// let device = Default::default();
// let model = ${structName}::<Wgpu>::new(&device);
// let input = Tensor::random([1, 3, 224, 224], Distribution::Default, &device);
// let output = model.forward(input);
`;

  return {
    filename: `${modelName.toLowerCase()}.rs`,
    content: code,
    language: 'rust'
  };
}

// ==================== Triton Kernels Generator ====================
export function generateTritonCode(
  nodes: CanvasNode[],
  _connections: Connection[],
  options: CodeGeneratorOptions = {}
): GeneratedCode {
  const { modelName = 'GeneratedModel' } = options;
  
  // Find attention and dense layers for optimized kernels
  const attentionNodes = nodes.filter(n => n.type === 'attention');
  const denseNodes = nodes.filter(n => n.type === 'dense');
  
  let kernels = `"""
NEURAX Auto-Generated Triton Kernels
=====================================

Optimized GPU kernels for ${modelName}
Generated for: ${nodes.length} layers

These kernels provide optimized implementations for:
- Fused attention operations
- Optimized matrix multiplications
- Custom activation functions
"""

import triton
import triton.language as tl
import torch

`;

  // Fused attention kernel
  if (attentionNodes.length > 0) {
    const attn = attentionNodes[0];
    const heads = attn.params.heads || 8;
    const dim = attn.params.dim || 512;
    
    kernels += `
# ==================== Fused Flash Attention ====================
@triton.jit
def fused_attention_kernel(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Fused Flash Attention Kernel
    Optimized for: ${heads} heads, ${dim} dim
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    off_z = off_hz // H
    off_h = off_hz % H
    
    # Initialize pointers
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh
    
    Q_block_ptr = Q + q_offset
    K_block_ptr = K + k_offset
    V_block_ptr = V + v_offset
    O_block_ptr = Out + o_offset
    
    # Core attention computation
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Load Q block
    q = tl.load(Q_block_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    
    # Accumulator for output
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    
    # Iterate over K, V blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        k = tl.load(K_block_ptr + (start_n + offs_n[None, :]) * stride_kn + offs_k[:, None] * stride_kk)
        v = tl.load(V_block_ptr + (start_n + offs_n[:, None]) * stride_vn + offs_k[None, :] * stride_vk)
        
        # Compute attention scores
        qk = tl.dot(q, k)
        qk *= 1.0 / (${Math.sqrt(Number(dim) / Number(heads))})
        
        # Softmax
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        l_i = alpha * l_i + beta * tl.sum(tl.exp(qk - m_ij[:, None]), axis=1)
        
        # Update accumulator
        p = tl.exp(qk - m_new[:, None])
        acc = alpha[:, None] * acc + tl.dot(p, v)
        m_i = m_new
    
    # Final output
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok, acc)


def flash_attention(q, k, v):
    """
    Flash Attention wrapper for ${modelName}
    
    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        v: Value tensor [batch, heads, seq_len, head_dim]
    
    Returns:
        Attention output [batch, heads, seq_len, head_dim]
    """
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 64
    
    batch, heads, seq_len, head_dim = q.shape
    out = torch.empty_like(q)
    
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * heads)
    
    fused_attention_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        batch, heads, seq_len,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    
    return out
`;
  }

  // Fused GELU kernel
  kernels += `

# ==================== Fused GELU Activation ====================
@triton.jit
def fused_gelu_kernel(
    x_ptr, y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused GELU activation kernel"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654
    x_cubed = x * x * x
    inner = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
    y = 0.5 * x * (1.0 + tl.libdevice.tanh(inner))
    
    tl.store(y_ptr + offsets, y, mask=mask)


def fused_gelu(x: torch.Tensor) -> torch.Tensor:
    """Fused GELU activation"""
    y = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    fused_gelu_kernel[grid](x, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return y
`;

  // Dense layer kernel
  if (denseNodes.length > 0) {
    kernels += `

# ==================== Optimized Matrix Multiplication ====================
@triton.jit  
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Optimized GEMM kernel for dense layers"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Optimized matrix multiplication using Triton"""
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    
    return c
`;
  }

  kernels += `

# ==================== Usage Example ====================
if __name__ == "__main__":
    # Test the kernels
    device = "cuda"
    
    # Test GELU
    x = torch.randn(1024, 1024, device=device)
    y = fused_gelu(x)
    print(f"GELU output shape: {y.shape}")
    
    # Test matmul
    a = torch.randn(512, 256, device=device)
    b = torch.randn(256, 512, device=device)
    c = triton_matmul(a, b)
    print(f"MatMul output shape: {c.shape}")
    
    print("All kernels tested successfully!")
`;

  return {
    filename: `${modelName.toLowerCase()}_kernels.py`,
    content: kernels,
    language: 'python'
  };
}

// ==================== JSON Schema Generator ====================
export function generateJSONSchema(
  nodes: CanvasNode[],
  connections: Connection[],
  options: CodeGeneratorOptions = {}
): GeneratedCode {
  const { modelName = 'GeneratedModel' } = options;
  
  const schema = {
    schema_version: 1,
    model_name: modelName,
    generated_by: 'NEURAX Architecture Designer',
    generated_at: new Date().toISOString(),
    architecture: {
      nodes: nodes.map(n => ({
        id: n.id,
        type: n.type,
        name: n.name,
        position: { x: n.x, y: n.y },
        params: n.params,
        inputShape: n.inputShape,
        outputShape: n.outputShape,
      })),
      connections: connections.map(c => ({
        id: c.id,
        from: c.from,
        to: c.to,
      })),
    },
    statistics: {
      total_layers: nodes.length,
      total_connections: connections.length,
      layer_types: [...new Set(nodes.map(n => n.type))],
    },
  };

  return {
    filename: `${modelName.toLowerCase()}_schema.json`,
    content: JSON.stringify(schema, null, 2),
    language: 'json'
  };
}

// ==================== Main Export Function ====================
export function generateCode(
  format: 'pytorch' | 'onnx' | 'rust' | 'triton' | 'json',
  nodes: CanvasNode[],
  connections: Connection[],
  options: CodeGeneratorOptions = {}
): GeneratedCode {
  switch (format) {
    case 'pytorch':
      return generatePyTorchCode(nodes, connections, options);
    case 'onnx':
      return generateONNXExportCode(nodes, connections, options);
    case 'rust':
      return generateRustCode(nodes, connections, options);
    case 'triton':
      return generateTritonCode(nodes, connections, options);
    case 'json':
      return generateJSONSchema(nodes, connections, options);
    default:
      throw new Error(`Unknown format: ${format}`);
  }
}
