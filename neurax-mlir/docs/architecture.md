# Neurax Compiler — Zero to Hero Architecture Guide

> A complete technical reference covering the Neurax analytic compiler and MLIR code-generation backend, from first principles to full system design.

---

## Table of Contents

1. [What Is Neurax?](#1-what-is-neurax)
2. [Motivation & Problem Space](#2-motivation--problem-space)
3. [Workspace Structure](#3-workspace-structure)
4. [The Input Format — Model JSON](#4-the-input-format--model-json)
5. [Layer 1 — `neurax-parser`](#5-layer-1--neurax-parser)
6. [Layer 2 — `neurax-ir` — The 10-Pass Analytic Pipeline](#6-layer-2--neurax-ir--the-10-pass-analytic-pipeline)
   - [Shared Infrastructure](#shared-infrastructure)
   - [Pass 1 — Architecture IR](#pass-1--architecture-ir)
   - [Pass 2 — Graph IR](#pass-2--graph-ir)
   - [Pass 3 — Tensor IR](#pass-3--tensor-ir)
   - [Pass 4 — Operator IR](#pass-4--operator-ir)
   - [Pass 5 — Compute IR](#pass-5--compute-ir)
   - [Pass 6 — Memory IR](#pass-6--memory-ir)
   - [Pass 7 — Parallelism IR](#pass-7--parallelism-ir)
   - [Pass 8 — Hardware IR](#pass-8--hardware-ir)
   - [Pass 9 — Cost IR](#pass-9--cost-ir)
   - [Pass 10 — Report IR](#pass-10--report-ir)
7. [Layer 3 — `neurax-core` — Pipeline Orchestrator](#7-layer-3--neurax-core--pipeline-orchestrator)
8. [Supporting Libraries](#8-supporting-libraries)
   - [`neurax-formulas`](#neurax-formulas)
   - [`neurax-hardware-db`](#neurax-hardware-db)
9. [User Interfaces](#9-user-interfaces)
   - [`neurax-cli`](#neurax-cli)
   - [`neurax-tui`](#neurax-tui)
10. [Layer 4 — `neurax-mlir` — The Code Generation Backend](#10-layer-4--neurax-mlir--the-code-generation-backend)
11. [Full End-to-End Data Flow](#11-full-end-to-end-data-flow)
12. [Diagnostic System](#12-diagnostic-system)
13. [Comparisons with Peer Tools](#13-comparisons-with-peer-tools)
14. [Quality Assessment](#14-quality-assessment)
15. [Gap Analysis & Roadmap](#15-gap-analysis--roadmap)
16. [The Full Vision](#16-the-full-vision)

---

## 1. What Is Neurax?

**Neurax** is a **Universal Analytic Compiler for AI Architectures** written in Rust. It takes a JSON description of a neural network and statically computes — without ever executing the model — a comprehensive set of metrics covering:

- Parameter counts and architecture statistics
- FLOPs (forward, backward, total per step)
- Memory footprint (parameters, activations, gradients, optimizer state, peak VRAM)
- Tensor shapes and data flow
- Parallelism strategy recommendations
- Hardware performance via the Roofline model
- Training cost in USD and CO₂

It is both:
1. An **analytic compiler** — transforms a declarative model spec into quantified performance metrics via 10 IR passes
2. A **code generator** (in progress) — lowers those IR structures into MLIR operations that can be compiled to native binary code for any hardware target

---

## 2. Motivation & Problem Space

Training or deploying a large model without upfront analysis leads to:
- Out-of-memory crashes discovered only after hours of setup
- Suboptimal parallelism that wastes 30–60% of GPU utilization
- Unexpected training costs in the hundreds of thousands of dollars
- No systematic way to compare model architectures before committing to a run

Existing tools are either **too narrow** (only transformers, or only memory, or only one GPU vendor) or **require running the model** first (making them useless for planning).

Neurax's design goal: **predict everything before a single GPU is provisioned**, for any model type, on any hardware.

---

## 3. Workspace Structure

```
Neurax-IR/                      ← Cargo workspace root
├── neurax-parser/              ← JSON ingestion & strong-typed config
├── neurax-ir/                  ← 10 IR dialects (core analysis engine)
├── neurax-core/                ← Pipeline orchestrator
├── neurax-formulas/            ← Per-architecture FLOPs formulas
├── neurax-hardware-db/         ← GPU/CPU/interconnect spec database
├── neurax-cli/                 ← Binary: `neurax` command-line tool
├── neurax-tui/                 ← Binary: Ratatui terminal UI
└── neurax-mlir/                ← MLIR dialect definitions & code gen
```

**Dependency graph:**

```
neurax-parser
    ▲
neurax-hardware-db
    ▲
neurax-formulas  ──┐
    ▲              │
neurax-ir  ────────┤
    ▲              │
neurax-core ───────┘
    ▲
neurax-cli / neurax-tui

neurax-mlir (independent, bridges to neurax-ir outputs via integration.rs)
```

---

## 4. The Input Format — Model JSON

Users describe their model in a structured JSON file. Here is a minimal transformer example:

```json
{
  "schema_version": "1.0",
  "model": {
    "name": "MyLLM-7B",
    "type": "transformer",
    "global_params": {
      "hidden_size": 4096,
      "num_attention_heads": 32,
      "num_key_value_heads": 8,
      "intermediate_size": 14336,
      "num_layers": 32,
      "vocab_size": 128000
    },
    "layers": [
      { "id": "embed", "layer_type": "embedding", "params": { "vocab_size": 128000, "embedding_dim": 4096 } },
      { "id": "attn_0", "layer_type": "attention", "params": { "hidden_size": 4096, "num_heads": 32, "num_kv_heads": 8 } },
      { "id": "mlp_0", "layer_type": "mlp", "params": { "hidden_size": 4096, "intermediate_size": 14336, "gated": true } }
    ]
  },
  "training": {
    "batch_size": 4,
    "precision": "bf16",
    "gradient_checkpointing": false,
    "zero_stage": 2,
    "parallelism": { "data_parallel": 8, "tensor_parallel": 4, "pipeline_parallel": 2 }
  },
  "hardware": {
    "gpus": [{ "name": "H100-SXM", "count": 64, "memory_gb": 80 }],
    "interconnect": "NVLink4"
  },
  "cost_config": {
    "provider": "AWS",
    "gpu_hour_usd": 4.10,
    "energy_kwh_usd": 0.12
  }
}
```

**Supported model types:** `transformer`, `cnn`, `moe`, `diffusion`, `gnn`, `rnn`, `ssm` (Mamba/S4/H3/RWKV), `gan`

**Supported layer types (50+):** attention, mlp, conv, embedding, normalization, pooling, moe, residual_block, mbconv, inception, dense_block, convnext_block, shuffle_unit, c2f, detection, mamba_block, s4_block, h3_block, rwkv_block, retention_block, generator_block, discriminator_block, adain, style_mod, lstm_block, gru_block, rnn_cell, bidirectional, encoder_block, decoder_block, custom, and more.

---

## 5. Layer 1 — `neurax-parser`

**Purpose:** Parse raw JSON → strongly-typed Rust structures with full validation.

### Key Structs

| Struct | Description |
|---|---|
| `ModelConfig` | Root: `Model + TrainingConfig + HardwareConfig + DataConfig + CostConfig` |
| `Model` | `ModelType + Vec<Layer> + GlobalParams` |
| `Layer` | `id + LayerType + input_shape + output_shape + LayerParams + Option<CustomEquations>` |
| `LayerParams` | ~50 fields covering all architectures: attention heads, conv kernels, SSM state dims, GAN style dims, LSTM cell params, etc. |
| `CustomEquations` | User-defined formulas: `flops_forward`, `memory_activation`, `gradient` (string expressions evaluated by `evalexpr`) |
| `TrainingConfig` | Batch size, optimizer, precision, ZeRO stage, `ParallelismConfig` (DP×TP×PP) |
| `HardwareConfig` | `Vec<GpuConfig>` — name, count, VRAM, TFLOPS per precision, NVLink, bandwidth |
| `CostConfig` | Provider, GPU-hour USD, energy kWh USD, PUE factor |

### Validation Pipeline

```
Raw JSON bytes
    │ serde_json deserialization
    ▼
RawModelConfig  (unvalidated, all fields optional/string)
    │ schema.rs — field presence, type coercions
    ▼
schema_validator.rs — cross-field validation
    │ coherence.rs — shape consistency, parameter ranges
    ▼
ModelConfig  (fully validated, strongly typed)
```

The parser rejects invalid configs **before** any expensive analysis begins.

---

## 6. Layer 2 — `neurax-ir` — The 10-Pass Analytic Pipeline

This is the heart of Neurax. The crate defines **10 IR dialects** in a strict sequential pipeline. Each dialect is a Rust module with 4 files:

- `ir.rs` — data structures for this dialect's output
- `pass.rs` — computation logic implementing `IrPass`
- `metrics.rs` — typed metrics structs
- `mod.rs` — re-exports

### Shared Infrastructure

#### `NeuraxContext`
Created once before any pass runs:

```rust
pub struct NeuraxContext {
    pub config: Arc<ModelConfig>,          // Shared parsed config
    pub gpu_db: Arc<HardwareDatabase>,     // Shared GPU spec database
    pub compute_config: ComputeConfig,     // Thread count, timeout
    pub diagnostics: Arc<Mutex<Vec<Diagnostic>>>,    // Thread-safe collector
    metrics_store: Arc<Mutex<HashMap<String, f64>>>, // Inter-pass messaging
}
```

#### `IrPass` Trait
Every dialect implements this contract:

```rust
trait IrPass {
    type Input;
    type Output;
    type Metrics;

    fn build(&self, input: &Self::Input, ctx: &NeuraxContext) -> Result<Output>;
    fn compute_metrics(&self, output: &mut Output, ctx: &NeuraxContext) -> Result<Metrics>;
    fn validate(&self, output: &Output, metrics: &Metrics) -> Result<()>;
}
```

---

### Pass 1 — Architecture IR

**Input:** `ModelConfig`
**Output:** `ArchitectureIR`

```
ModelConfig
    │
    ├─ Enumerate all layers → Vec<LayerDef>
    ├─ Calculate param_count per layer
    ├─ Scale to full model size via global_params.num_layers
    ├─ Detect MoE, GQA, special architectures
    └─ Compute total_parameters
```

**Key Metrics (M1–M2):**

| Metric | Description |
|---|---|
| `total_parameters` | Sum across all layers × model scale factor |
| `num_layers` | Effective full model depth |
| `params_per_layer` | `HashMap<layer_id, param_count>` |
| `layers_by_type` | Census of layer types |
| `model_type_info` | String description |

**Special handling:** When the JSON contains only a *sample* of layers (e.g., one transformer block representing 32 identical ones), `global_params.num_layers` is used to scale correctly.

---

### Pass 2 — Graph IR

**Input:** `ArchitectureIR`
**Output:** `GraphIR` — a `petgraph::DiGraph<GraphNode, GraphEdge>`

```
ArchitectureIR
    │
    ├─ Create a GraphNode per layer
    ├─ Connect with GraphEdge (carrying tensor shape + dtype + size_bytes)
    ├─ Detect and reject cycles (diagnostic E005)
    ├─ Compute topological order (Kahn's algorithm)
    └─ DP longest-path → graph_depth (critical path)
```

**Key Metrics (M3–M5):**

| Metric | Description |
|---|---|
| `graph_depth` | Longest path from input to output |
| `total_operations` | Number of nodes |
| `critical_path_length` | Minimum sequential depth |
| `parallel_paths` | Detected branches (residual connections, MoE routing) |
| `edge_count` | Total data dependencies |

---

### Pass 3 — Tensor IR

**Input:** `GraphIR`
**Output:** `TensorIR`

Maps every edge in the graph to a named `TensorInfo` with shape, dtype, size in bytes, producer layer, and consumer layers. The `Shape` type supports three dimension kinds:

```rust
enum Dim {
    Known(usize),      // Concrete: e.g. 4096
    Symbolic(String),  // Named: e.g. "batch", "seq"
    Dynamic,           // Unknown at compile time
}
```

**Key Metrics (M9–M12):**

| Metric | Description |
|---|---|
| `activation_memory_bytes` | Sum of all intermediate tensor sizes |
| `resolution_ratio` | Fraction of dims that are fully `Known` |
| `unresolved_dim_count` | Symbolic/Dynamic dims remaining |
| `largest_tensor_id` | Heaviest single tensor |
| `tensor_size_distribution` | Histogram: tiny/small/medium/large/huge |

A `resolution_ratio < 1.0` triggers diagnostic `W002` (symbolic dimensions remaining) which lowers the analysis confidence score.

---

### Pass 4 — Operator IR

**Input:** `(TensorIR, ArchitectureIR)`
**Output:** `OperatorIR`

This pass **decomposes every layer into atomic operations (`AtomOp`)**. It is the bridge between high-level layers and actual computation counts.

**24+ AtomOp types:**

```
Core:          MatMul, BatchedMatMul, Conv2D, DepthwiseConv2D, Linear
Attention:     Attention, FlashAttention, GroupedQueryAttention, MultiQueryAttention,
               AttentionScores, AttentionOutput, Softmax
Normalization: LayerNorm, BatchNorm, RMSNorm, GroupNorm
Embeddings:    Embedding, TokenEmbedding, PositionalEmbedding, RotaryEmbedding
MoE:           MoE, MoERouter, MoEExpertGroup
SSM/Mamba:     SsmStateUpdate, MambaConv1d, S4Block, H3Block
RNN:           LstmCell, GruCell, RnnCell
Fine-tuning:   LoRALinear
Activations:   Add, Mul, Div, ReLU, GELU, SiLU, Tanh, Sigmoid
Pooling:       Pooling(Max/Avg)
Tensor ops:    Reshape, Transpose, Concat, Split
Custom:        Custom (evaluated via neurax-formulas::custom)
```

Each `AtomOp` carries:
```rust
pub struct AtomOp {
    pub op_type: OpType,
    pub flops: f64,            // Analytically computed
    pub param_count: u64,
    pub activation_memory: u64,
    pub is_custom: bool,
}
```

The per-architecture FLOPs formulas come from `neurax-formulas`.

---

### Pass 5 — Compute IR

**Input:** `OperatorIR`
**Output:** `ComputeIR`

Aggregates all `AtomOp.flops` into the full compute picture.

**Key Metrics (M13–M18):**

| Metric | Description |
|---|---|
| `forward_flops` | Sum of all AtomOp forward FLOPs |
| `backward_flops` | ≈ 2× forward (per-op backward formula in roadmap) |
| `optimizer_flops` | Adam/AdamW momentum + variance updates |
| `total_step_flops` | forward + backward + optimizer |
| `flops_per_token` | For autoregressive models |
| `arithmetic_intensity` | FLOPs / bytes_accessed — key roofline input |
| `complexity_class` | O(n), O(n log n), O(n²), O(n³) — attention detection |
| `macs` | `total_flops / 2` |

The `complexity_class` is determined by the dominant `OpType`:
- Standard `Attention` → `O(n²)` (quadratic in sequence length)
- `FlashAttention` → upgraded to `O(n log n)`
- `Conv2D` → `O(n)` (linear in spatial dimensions)

---

### Pass 6 — Memory IR

**Input:** `(ComputeIR, TensorIR, ArchitectureIR)`
**Output:** `MemoryIR`

The most complex pass. Simulates the full memory lifecycle of a training step.

**Memory components:**

```
Peak VRAM = Parameters + Activations + Gradients + Optimizer State
                                                  + Intra-op buffers (roadmap)
```

| Component | Formula |
|---|---|
| Parameters | `total_params × bytes_per_element` |
| Activations | Tensor liveness simulation across the graph |
| Gradients | `= Parameters` (same size as weights) |
| Optimizer State | Adam: `2 × Parameters` (momentum + variance); SGD: 0 |

The pass also constructs a **memory timeline**: at each execution step, which tensors are live and what is the total allocation. This enables pinpointing the exact operation that causes peak VRAM.

**OOM Risk classification:**

```
peak / VRAM_available:
  < 80%  → Safe
  80–95% → Warning (W005 diagnostic)
  95–99% → Critical
  > 100% → Overflow (E001 diagnostic — blocks analysis)
```

**Key Metrics (M19–M25):**

| Metric | Description |
|---|---|
| `peak_vram_bytes` | Maximum concurrent memory usage |
| `parameter_memory_bytes` | Weights only |
| `activation_memory_bytes` | Intermediate feature maps |
| `gradient_memory_bytes` | Backprop gradients |
| `optimizer_state_bytes` | Optimizer accumulators |
| `max_batch_size_fit` | Largest batch that fits without OOM |
| `oom_risk` | Safe / Warning / Critical / Overflow |

---

### Pass 7 — Parallelism IR

**Input:** `(MemoryIR, GraphIR)`
**Output:** `ParallelismIR`

Evaluates six parallelism strategies and selects the optimal one:

| Strategy | When optimal |
|---|---|
| `DataParallel { num_gpus, efficiency }` | Model fits on 1 GPU; scale batch size |
| `TensorParallel { tp_degree }` | Large matmuls that split cleanly across GPUs |
| `PipelineParallel { stages, micro_batches, bubble_ratio }` | Very deep models, bandwidth-limited |
| `Hybrid { dp, tp, pp }` | Large models requiring 3D parallelism (GPT-4 style) |
| `ZeRO { stage, memory_per_gpu }` | ZeRO-1/2/3 for memory efficiency in DP |
| `ModelParallel { splits }` | Custom layer-range partitioning |

**Key Metrics (M26–M30):**

| Metric | Description |
|---|---|
| `data_parallel_efficiency` | Linear scaling ratio (Amdahl-adjusted) |
| `communication_overhead` | Fraction of step time spent on AllReduce |
| `optimal_gpu_count` | Recommended minimum GPU count |
| `pipeline_stages` | Optimal PP stages |
| `allreduce_time_ms` | Time for gradient synchronization |
| `scaling_efficiency_curve` | `Vec<(num_gpus, efficiency)>` |

---

### Pass 8 — Hardware IR

**Input:** `(ComputeIR, MemoryIR, ParallelismIR)`
**Output:** `HardwareIR`

Implements the **Industrial Roofline Model** — a 4-level hierarchy:

| Level | What it models |
|---|---|
| Classic | Peak TFLOPS vs. Peak HBM bandwidth |
| Calibrated | Efficiency per op type (attention ≠ matmul ≠ conv) |
| Memory Hierarchy | L1 SRAM / L2 / HBM separately |
| Industrial | Compute-memory overlap, kernel launch overhead |

The GPU profile is resolved from `neurax-hardware-db` using the name in the JSON:

```rust
GpuProfile {
    tflops_fp32, tflops_fp16, tflops_bf16, tflops_int8, tflops_fp8,
    hbm_bandwidth_gb_s, l2_bandwidth_tb_s, sram_bandwidth_tb_s,
    l2_cache_mb, sram_per_sm_kb, num_sms,
    nvlink_bandwidth_gb_s, tdp_watts, ...
}
```

Per-layer `LatencyEstimate` is computed:
```
t_compute  = flops / (peak_tflops × efficiency)
t_memory   = bytes / bandwidth[cache_level]
t_effective = max(t_compute, t_memory) + overlap_reduction
t_total    = t_effective + kernel_launch_overhead
```

**Key Metrics (M31–M35):**

| Metric | Description |
|---|---|
| `latency_ms` | Estimated time per training step |
| `throughput_tokens_per_s` | Tokens processed per second |
| `gpu_utilization` | Effective TFLOPS / peak TFLOPS |
| `bottleneck` | `compute-bound` / `memory-bound` / `balanced` |
| `roofline_position` | 0.0 = fully memory-bound, 1.0 = fully compute-bound |

> **Note:** Passes 7 and 8 run **in parallel** using `rayon::join()` since they are independent given passes 1–6. Hardware is then re-run once with the actual parallelism data for final accuracy.

---

### Pass 9 — Cost IR

**Input:** `(HardwareIR, ParallelismIR)`
**Output:** `CostIR`

Translates hardware metrics into real-world financial and environmental costs.

**Formulas:**
```
training_time_hours = (total_steps × latency_ms) / 3_600_000
gpu_hours_total     = training_time_hours × total_gpus
training_cost_usd   = gpu_hours_total × gpu_hour_usd
energy_kwh          = (total_gpus × tdp_watts × training_time_hours × pue) / 1000
co2_kg              = energy_kwh × co2_per_kwh (0.233 kg/kWh EU average)
cost_per_token      = training_cost_usd / (total_steps × batch_size × seq_len)
```

**Key Metrics (M36–M40):**

| Metric | Description |
|---|---|
| `training_cost_usd` | Total GPU rental cost |
| `training_time_hours` | Wall-clock training duration |
| `energy_kwh` | Total electricity consumed |
| `co2_kg` | Carbon footprint |
| `cost_per_million_tokens_usd` | Inference economics |
| `monthly_inference_cost_usd` | Sustained inference budget |

---

### Pass 10 — Report IR

**Input:** All 9 previous IRs
**Output:** `ReportIR`

The final consolidation pass. Assembles all 43 metrics into a single `AllMetrics` struct, collects all diagnostics fired during previous passes, generates `Recommendation` objects, and computes a `confidence_score`.

**Confidence score** degrades with:
- Unresolved symbolic dimensions (`resolution_ratio < 1.0`)
- Custom layers without formulas
- Missing global params that required estimation

Output formats:
- **Markdown** — human-readable report with tables and sections
- **JSON** — machine-parsable with full metric tree

---

## 7. Layer 3 — `neurax-core` — Pipeline Orchestrator

`neurax-core` exposes the single public API:

```rust
pub fn run_analysis(config: ModelConfig) -> Result<AnalysisResult, NeuraxError>
```

Internally it sequences all 10 `IrPass` executions, handling the `rayon::join()` for passes 7+8, then wraps everything in:

```rust
pub struct AnalysisResult {
    pub arch: ArchitectureIR,
    pub graph: GraphIR,
    pub tensor: TensorIR,
    pub operator: OperatorIR,
    pub compute: ComputeIR,
    pub memory: MemoryIR,
    pub parallelism: ParallelismIR,
    pub hardware: HardwareIR,
    pub cost: CostIR,
    pub report: ReportIR,
    pub analysis_time_ms: u64,
}
```

The `IrPassExt` blanket trait adds a `run()` method that chains `build → compute_metrics → validate`:

```rust
impl<T: IrPass> IrPassExt for T {}
```

Convenience helpers in `engine.rs` / `runner.rs`:
- `analyze_file(path)` — reads JSON from disk, runs pipeline
- `analyze_json(json_str)` — from string
- `validate_json(json_str)` — parse-only, no analysis
- `get_model_summary(config)` — quick model overview without full pipeline

---

## 8. Supporting Libraries

### `neurax-formulas`

12 formula modules, one per architecture family:

| Module | Covers |
|---|---|
| `attention.rs` | Standard MHA: `4bsh²`; GQA: `2bsh(h_kv/h_q + 1)`; FlashAttention tiled |
| `mlp.rs` | Dense FFN: `2bsh·d_ff`; Gated (SwiGLU): `3bsh·d_ff` |
| `conv.rs` | Conv2D: `2·C_in·C_out·K²·H_out·W_out·B`; Depthwise; Grouped |
| `embedding.rs` | `vocab_size × embedding_dim` params; fwd = table lookup |
| `normalization.rs` | LayerNorm: `5n`; RMSNorm: `4n`; BatchNorm: `6n` |
| `moe.rs` | Router: `B·S·H·E`; Expert: `top_k × mlp_flops`; Load balancing overhead |
| `ssm.rs` | Mamba: `O(B·L·D·N)` linear; S4: structured convolution; H3; RWKV |
| `rnn.rs` | LSTM: `8·B·T·H²`; GRU: `6·B·T·H²`; Vanilla RNN: `2·B·T·H²` |
| `diffusion.rs` | U-Net per timestep: `∑(conv_flops_per_block)`; timestep conditioning |
| `gnn.rs` | Message passing: `E·D²` (edges × feature dim²) |
| `custom.rs` | AST evaluator via `evalexpr`; validates formula safety |

Custom formula evaluation is sandboxed — only whitelisted variable names (`B`, `S`, `H`, `D`, etc.) are allowed.

### `neurax-hardware-db`

Static database of 20+ GPU specifications loaded at startup:

| GPU | VRAM | FP16 TFLOPS | FP8 TFLOPS | BW (GB/s) | NVLink |
|---|---|---|---|---|---|
| NVIDIA H200 | 141 GB | 989 | 3958 | 4800 | ✅ |
| NVIDIA H100-SXM | 80 GB | 989 | 3958 | 3352 | ✅ |
| NVIDIA H100-PCIe | 80 GB | 756 | 3026 | 2000 | ❌ |
| NVIDIA A100-SXM | 80 GB | 312 | — | 2039 | ✅ |
| NVIDIA A100-PCIe | 40 GB | 312 | — | 1555 | ❌ |
| NVIDIA RTX 4090 | 24 GB | 165 | 660 | 1008 | ❌ |
| NVIDIA L40S | 48 GB | 362 | 1448 | 864 | ❌ |
| NVIDIA V100 | 32 GB | 125 | — | 900 | ✅ |
| NVIDIA T4 | 16 GB | 65 | — | 300 | ❌ |
| …and 11 more | | | | | |

Also includes CPU specs (AMD EPYC, Intel Xeon) and interconnects (NVLink3/4, PCIe4/5, InfiniBand-NDR).

---

## 9. User Interfaces

### `neurax-cli`

The `neurax` binary provides three commands:

```bash
# Full analysis — outputs Markdown or JSON report
neurax analyze model.json [-o report.md] [-f json|markdown]

# Parse validation only — fast schema check
neurax validate model.json

# Quick one-line summary
neurax summary model.json
```

### `neurax-tui`

A full **Ratatui** terminal UI (`ratatui` + `crossterm`):

```
┌─────────────────────────────────────────────────────────┐
│ NEURAX — AI Architecture Analyzer              v0.1.0   │
├──────────┬──────────┬──────────┬──────────────────────  │
│ Overview │ Compute  │ Memory   │ Hardware               │
├──────────┴──────────┴──────────┴────────────────────    │
│  [Model Selector]      │  [Metrics Panel]               │
│  > Llama-3-70B         │  Parameters: 70.6B             │
│    GPT-4-Turbo         │  Peak VRAM: 147.3 GB           │
│    Mistral-7B          │  FLOPs/step: 1.83e15           │
│    Mamba-2.8B          │  Bottleneck: memory-bound      │
│                        │  Training cost: $124,000       │
│─────────────────────── │─────────────────────────────── │
│  [Comparison vs Real]  │  [Recommendations]             │
│  Our: 147GB / Ref:142G │  H001: Enable grad checkp.     │
│  +3.5% variance        │  H005: Consider ZeRO-3         │
└────────────────────────┴───────────────────────────────-┘
```

**Key bindings:** `↑↓/jk` model select · `Enter` compile · `Tab/1-4` switch tabs · `r` refresh · `q` quit

---

## 10. Layer 4 — `neurax-mlir` — The Code Generation Backend

### Position in the Stack

`neurax-mlir` is the **second compiler stage** — it sits after `neurax-core` and receives computed IR metrics, then encodes them as **MLIR operations** using the `melior` crate (Rust bindings to LLVM's MLIR C API).

```
AnalysisResult  (from neurax-core)
       │
       ▼ integration.rs
NeuraxModule (MLIR in-memory module)
       │
       ▼ passes/  (dialect lowering)
Standard MLIR dialects (linalg, arith, affine, gpu, nvvm...)
       │
       ▼ LLVM backend
PTX / x86 / ARM / WASM binary
```

### MLIR Dialects Defined

`neurax-mlir` defines **13 MLIR dialects** mirroring the Neurax IR:

| Dialect | Namespace | Key Operations |
|---|---|---|
| `arch` | `mlir::neurax::arch` | `arch.model`, `arch.layer`, `arch.global_params`, `arch.metrics`, `arch.repeat`, `arch.layer_pattern` |
| `graph` | `mlir::neurax::graph` | `graph.node`, `graph.edge`, `graph.dag` |
| `tensor` | `mlir::neurax::tensor` | `tensor.info`, `tensor.shape`, `tensor.liveness` |
| `operator` | `mlir::neurax::operator` | `op.matmul`, `op.attention`, `op.gqa`, `op.conv2d`, `op.mamba`, `op.lstm`, ... |
| `compute` | `mlir::neurax::compute` | `compute.flops`, `compute.complexity`, `compute.intensity` |
| `memory` | `mlir::neurax::memory` | `mem.peak_vram`, `mem.liveness_interval`, `mem.oom_risk` |
| `hardware` | `mlir::neurax::hardware` | `hw.gpu_profile`, `hw.roofline`, `hw.latency` |
| `parallelism` | `mlir::neurax::par` | `par.data`, `par.tensor`, `par.pipeline`, `par.hybrid`, `par.zero` |
| `cost` | `mlir::neurax::cost` | `cost.training`, `cost.energy`, `cost.inference` |
| `report` | `mlir::neurax::report` | `report.final`, `report.metric`, `report.diagnostic` |
| `training` | `mlir::neurax::training` | `train.optimizer`, `train.precision`, `train.schedule` |
| `data` | `mlir::neurax::data` | `data.input`, `data.pipeline` |
| `optimization` | `mlir::neurax::opt` | `opt.fuse`, `opt.tile`, `opt.quantize` |

### TableGen Definitions (`.td` files)

The `include/` directory contains LLVM TableGen specifications for each dialect. For example, `ArchitectureOps.td` defines:

```tablegen
def ModelOp : Arch_Op<"model"> {
  let arguments = (ins StrAttr:$name, StrAttr:$model_type);
  let regions = (region AnyRegion:$layers);
}

def LayerOp : Arch_Op<"layer"> {
  let arguments = (ins StrAttr:$id, StrAttr:$layer_type,
                   OptionalAttr<I64ArrayAttr>:$input_shape,
                   OptionalAttr<I64ArrayAttr>:$output_shape);
}

def ArchMetricsOp : Arch_Op<"metrics"> {
  let arguments = (ins I64Attr:$total_parameters, I64Attr:$num_layers);
}
```

These `.td` files are the **ground-truth schema** for the MLIR layer. They can be processed by `mlir-tblgen` to auto-generate C++ dialect implementations.

### MLIR Passes

Six lowering passes defined in `src/passes/`:

| Pass | Transforms |
|---|---|
| `ArchitecturePass` | `arch.*` → `func.func` + `arith.*` |
| `ComputePass` | `compute.flops` → `arith.mulf` chains |
| `MemoryPass` | `mem.*` → buffer allocation ops |
| `HardwarePass` | `hw.roofline` → target-specific performance hints |
| `ParallelismPass` | `par.*` → `gpu.launch` + collectives |
| `CostPass` | `cost.*` → metadata annotations |

### Current State

`neurax-mlir` is **architecturally sound but not yet wired** to `neurax-core`. The `integration.rs` functions take raw scalar values (not `AnalysisResult`), so the bridge must be built. The Cargo.toml for `neurax-mlir` has no dependency on `neurax-ir` or `neurax-core`.

---

## 11. Full End-to-End Data Flow

```
┌─────────────────────────────────────────────────┐
│                   JSON File                      │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│  neurax-parser — schema validation, coherence   │
│  Output: ModelConfig (fully typed)              │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│  neurax-core::run_analysis()                    │
│                                                 │
│  Pass 1  ArchitecturePass  → ArchitectureIR     │
│      ↓                                          │
│  Pass 2  GraphPass         → GraphIR (DAG)      │
│      ↓                                          │
│  Pass 3  TensorPass        → TensorIR (shapes)  │
│      ↓                                          │
│  Pass 4  OperatorPass      → OperatorIR (atoms) │
│      ↓                                          │
│  Pass 5  ComputePass       → ComputeIR (FLOPs)  │
│      ↓                                          │
│  Pass 6  MemoryPass        → MemoryIR (VRAM)    │
│      ↓             ↓                            │
│  Pass 7  ParallelismPass   Pass 8  HardwarePass  │
│  (rayon::join — run in parallel)                │
│      ↓             ↓                            │
│  Pass 8  HardwarePass (re-run with P7 data)     │
│      ↓                                          │
│  Pass 9  CostPass          → CostIR (USD/CO₂)  │
│      ↓                                          │
│  Pass 10 ReportPass        → ReportIR (43 metrics)│
│                                                 │
│  Output: AnalysisResult                         │
└──────────────────────┬──────────────────────────┘
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
  ┌────────────────┐    ┌──────────────────────┐
  │  neurax-cli    │    │  neurax-mlir          │
  │  Markdown/JSON │    │  NeuraxModule (MLIR)  │
  │  report file   │    │  → lowering passes    │
  └────────────────┘    │  → LLVM IR            │
                        │  → native binary      │
  ┌────────────────┐    └──────────────────────┘
  │  neurax-tui    │
  │  Ratatui panels│
  └────────────────┘
```

---

## 12. Diagnostic System

All passes emit structured diagnostics collected in `NeuraxContext`:

### Code Classes

| Code | Severity | Meaning |
|---|---|---|
| `E001` | Error | OOM risk — peak VRAM exceeds GPU capacity |
| `E002` | Error | Shape gate blocked — insufficient dimension resolution |
| `E003` | Error | Custom formula evaluation failed |
| `E004` | Error | Unsupported layer type |
| `E005` | Error | Cycle detected in computation graph |
| `W001` | Warning | Custom layer without formula — estimation used |
| `W002` | Warning | Symbolic dimensions remaining in shapes |
| `W003` | Warning | ZeRO not recommended for this config |
| `W004` | Warning | Flash Attention not enabled |
| `W005` | Warning | Memory close to GPU limit (>80%) |
| `W006` | Warning | Inefficient parallelism strategy |
| `I001` | Info | GQA detected |
| `I002` | Info | MoE detected |
| `I003` | Info | Flash Attention detected |
| `H001` | Hint | Enable gradient checkpointing |
| `H002` | Hint | Enable Flash Attention |
| `H003` | Hint | Consider INT8 quantization |
| `H004` | Hint | Increase micro-batches for PP |
| `H005` | Hint | ZeRO-3 recommended |

Every diagnostic includes a `message`, `suggestion` (actionable fix), `layer_id` (where it occurred), and `precision_impact` (0.0–1.0 — how much this degrades analysis accuracy).

---

## 13. Comparisons with Peer Tools

### Feature Matrix

| Feature | **Neurax** | PyTorch Profiler | DeepSpeed Estimator | Apache TVM | XLA / JAX | IREE |
|---|---|---|---|---|---|---|
| **Static analysis (no GPU needed)** | ✅ | ❌ Needs runtime | ⚠️ Partial | ❌ | ❌ | ❌ |
| **Multi-architecture support** | ✅ 8 types | ⚠️ PyTorch only | ⚠️ Transformer | ✅ | ✅ | ✅ |
| **Memory breakdown** | ✅ 5 components | ⚠️ Activation only | ✅ | ❌ | ❌ | ❌ |
| **Parallelism recommendation** | ✅ 6 strategies | ❌ | ✅ ZeRO only | ❌ | ⚠️ | ❌ |
| **Cost & CO₂ estimation** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Hardware roofline model** | ✅ 4-level | ❌ | ❌ | ⚠️ Basic | ❌ | ❌ |
| **20+ GPU profiles** | ✅ | ⚠️ Measured | ❌ | ❌ | ❌ | ❌ |
| **Diagnostic system** | ✅ Structured | ⚠️ Ad hoc | ⚠️ | ❌ | ❌ | ❌ |
| **Custom layer formulas** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Code generation** | 🔧 In progress | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Multi-target compilation** | 🔧 | ❌ | ❌ | ✅ | ✅ CUDA/TPU | ✅ |
| **Interpretable output** | ✅ Markdown/JSON | ⚠️ Traces | ⚠️ | ❌ | ❌ | ❌ |

### Positioning

```
                    Static Analysis Completeness
                    LOW ◄─────────────────► HIGH
                         │               │
     Code Gen      HIGH  │  TVM          │  ← Neurax (target)
     Capability          │  XLA          │
                         │  IREE         │
                    LOW   │               │  Neurax (current)
                          │               │
                          DeepSpeed Est.  PyTorch Profiler
```

**Neurax's unique value proposition:** It is the only tool that combines _full static analysis before any GPU is provisioned_ with a _structured MLIR code generation pathway_ for _any ML architecture type_.

### vs. PyTorch Profiler
PyTorch Profiler requires executing the model — you need GPU access, a loaded dataset, and a Python environment. It gives you accurate timing traces but nothing about _pre-run planning_. Neurax answers the question PyTorch Profiler cannot: "Should I even start training this configuration?"

### vs. DeepSpeed's Memory Estimator
Excellent for ZeRO memory estimation, but only covers Transformer models and memory. No FLOPs, no hardware modeling, no cost, no code generation.

### vs. Apache TVM
TVM is a code generation compiler (input: ONNX/TensorFlow, output: optimized binary) with no analytic layer. It doesn't tell you anything about the model's properties before compilation. Neurax's analytic stage would make an excellent pre-pass for TVM, informing tile sizes and parallelism strategies.

### vs. XLA / JAX
XLA is Google's compiler for TPU/GPU optimization — JIT-based, tightly coupled to JAX/TensorFlow. Excellent for code generation, nothing for pre-run planning or multi-model comparison.

---

## 14. Quality Assessment

| Dimension | Score | Notes |
|---|---|---|
| **IR Architecture Design** | 9.5/10 | Textbook pipeline, clean IrPass trait, proper separation |
| **Model Type Coverage** | 9/10 | 8 families, 50+ layer types, including SSM and GAN |
| **Formula Accuracy (forward FLOPs)** | 8.5/10 | Very good; GQA correct; diffusion/SSM could be deeper |
| **Formula Accuracy (backward FLOPs)** | 6/10 | 2× forward is a rough estimate; per-op backward not done |
| **Memory Modeling** | 7.5/10 | 4 components accurate; intra-op buffers missing |
| **Hardware Modeling** | 8.5/10 | Industrial Roofline is sophisticated and rare |
| **Parallelism Analysis** | 8/10 | All 6 strategies; Amdahl/bubble ratio well-handled |
| **Diagnostic System** | 8.5/10 | Production-grade structured codes and suggestions |
| **Code Generation (MLIR)** | 4/10 | Well-designed but disconnected; passes not yet lowering |
| **Test Coverage** | 5/10 | Unit tests exist; no ground-truth benchmark fixtures |
| **Overall** | **8.2/10** | Exceptional analytic engine; code-gen backend needs wiring |

---

## 15. Gap Analysis & Roadmap

### Critical Gaps

1. **`neurax-mlir` ↔ `neurax-core` bridge missing**
   > `integration.rs` takes raw scalars, not `AnalysisResult`. Need a `From<AnalysisResult> for NeuraxModule` conversion and a dependency edge in Cargo.

2. **Backward FLOPs by operator**
   > Currently `backward = 2 × forward`. Need per-op formulas (attention backward has different structure from matmul backward, especially with GQA).

3. **Intra-op buffer tracking in VRAM**
   > Flash Attention tiling buffers, Conv2D im2col workspace, MoE gating temp buffers. Can underestimate peak VRAM by 5–20%.

4. **Shape inference (not shape propagation)**
   > Currently users must provide both `input_shape` and `output_shape`. True compiler-style inference would derive output shapes automatically.

5. **Ground-truth test fixtures**
   > No benchmark suite comparing Neurax's predictions against measured GPU results. Without this, calibration cannot be validated or improved.

### Roadmap (Suggested Priority)

| Priority | Task |
|---|---|
| P0 | Wire `neurax-mlir` to `AnalysisResult` — the core vision gap |
| P1 | Per-op backward FLOPs in `neurax-formulas` |
| P1 | Intra-op buffer tracking in `MemoryPass` |
| P2 | Ground-truth fixtures in `neurax-tests` for top 10 open models |
| P2 | Shape inference propagation (`TensorPass`) |
| P3 | MLIR lowering passes → linalg/gpu dialects |
| P3 | LLVM backend emission (PTX / x86) |
| P4 | Multi-target compilation pipeline |
| P4 | Plugin/extension mechanism for custom hardware backends |

---

## 16. The Full Vision

When complete, Neurax will cover the full lifecycle from model description to deployed binary:

```
┌────────────────────────────────────────────────────────────────┐
│  Step 1: PLAN  (neurax analyze model.json)                     │
│  → 43 metrics before any GPU is provisioned                    │
│  → "Your model needs 3× more VRAM than your cluster has"       │
│  → "Enabling ZeRO-3 + GC reduces VRAM 68%. Cost: $80K"        │
└──────────────────────────────┬─────────────────────────────────┘
                               │
┌──────────────────────────────▼─────────────────────────────────┐
│  Step 2: OPTIMIZE  (neurax-mlir lowering passes)               │
│  → Guided by Neurax's own metrics:                             │
│     - Use arithmetic_intensity to decide op fusion             │
│     - Use tile sizes from roofline cache analysis              │
│     - Auto-insert AllReduce from optimal_gpu_count             │
│     - Emit INT8 kernels when H003 hint was fired               │
└──────────────────────────────┬─────────────────────────────────┘
                               │
┌──────────────────────────────▼─────────────────────────────────┐
│  Step 3: DEPLOY  (LLVM backend)                                │
│  → PTX → cubin for NVIDIA GPUs                                 │
│  → AMDGPU IR for ROCm                                          │
│  → x86 / ARM for CPU inference                                 │
│  → WebAssembly for browser                                     │
│  → Custom ASIC via new backend dialect                         │
│                                                                │
│  Output: standalone binary, no Python, no CUDA toolkit         │
└────────────────────────────────────────────────────────────────┘

The key architectural insight that makes Neurax different from every other compiler in this space: **the analytic pass and the code generation pass share the same IR**. The roofline ridge point that `HardwareIR` computes is the exact number controlling tile sizes in the MLIR lowering pass. The optimal parallelism strategy from `ParallelismIR` is the input to the `par.hybrid` MLIR op that generates collective communication. **The analysis informs the compilation** — not as heuristics, but as first-class IR attributes.

This is the vision of Neurax as a full compiler system.
