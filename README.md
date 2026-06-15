<div align="center">

<h1>⚡ NEURAX</h1>

### The Pre‑Flight Compiler for Artificial Intelligence

**_Know the cost, memory, speed and feasibility of any AI model — in milliseconds, before a single GPU spins up._**

<br/>

<!-- Core -->
![Rust](https://img.shields.io/badge/Rust-2021-000000?style=for-the-badge&logo=rust&logoColor=white)
![MLIR](https://img.shields.io/badge/MLIR-LLVM%2018-2C2C32?style=for-the-badge&logo=llvm&logoColor=white)
![LLVM](https://img.shields.io/badge/LLVM-18-262D3A?style=for-the-badge&logo=llvm&logoColor=white)
![License](https://img.shields.io/badge/License-Proprietary-blue?style=for-the-badge)

<!-- Backend / Service -->
![Actix Web](https://img.shields.io/badge/Actix_Web-4-000000?style=flat-square&logo=rust)
![Tokio](https://img.shields.io/badge/Tokio-async-15883E?style=flat-square)
![Supabase](https://img.shields.io/badge/Supabase-Auth-3ECF8E?style=flat-square&logo=supabase&logoColor=white)
![Stripe](https://img.shields.io/badge/Stripe-Billing-635BFF?style=flat-square&logo=stripe&logoColor=white)

<!-- Front-end -->
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?style=flat-square&logo=typescript&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-8-646CFF?style=flat-square&logo=vite&logoColor=white)
![Tailwind CSS](https://img.shields.io/badge/Tailwind-3-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white)
![shadcn/ui](https://img.shields.io/badge/shadcn%2Fui-Radix-000000?style=flat-square)

<!-- Agent -->
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-SSE-009688?style=flat-square&logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Agent-1C3C3C?style=flat-square&logo=langchain&logoColor=white)

<!-- Status -->
![Tests](https://img.shields.io/badge/tests-126%20passing-success?style=flat-square)
![Backend](https://img.shields.io/badge/compiler-neurax--mlir-orange?style=flat-square)
![Build](https://img.shields.io/badge/workspace-green-success?style=flat-square)

</div>

---

> ## 🎯 Mission
> **NEURAX exists to revolutionize AI research by making the *economics and physics* of any neural network knowable in advance.**
> Today, the cost of a model is discovered the hard way — after weeks of engineering, a launched cluster job, and a budget already spent. NEURAX turns that gamble into a **prediction**: design an architecture, and instantly see exactly what it will cost, how much memory it will demand, how fast it will run, and whether it will even fit — *before you write the training loop.*

---

## Table of Contents

1. [The Story](#1-the-story)
2. [The Problem NEURAX Solves](#2-the-problem-neurax-solves)
3. [How NEURAX Changes AI Research](#3-how-neurax-changes-ai-research)
4. [What NEURAX Is](#4-what-neurax-is)
5. [How It Works — The IR Pipeline](#5-how-it-works--the-ir-pipeline)
6. [The MLIR Compiler Backend](#6-the-mlir-compiler-backend)
7. [NEURAX vs. XLA / TVM / IREE / TensorRT](#7-neurax-vs-xla--tvm--iree--tensorrt)
8. [The Interfaces](#8-the-interfaces)
9. [The Complete Service & API](#9-the-complete-service--api)
10. [Pricing & Plans](#10-pricing--plans)
11. [The 35+ Metrics](#11-the-35-metrics)
12. [Repository Layout](#12-repository-layout)
13. [Installation](#13-installation)
14. [Building](#14-building)
15. [Usage](#15-usage)
16. [The Universal Model JSON](#16-the-universal-model-json)
17. [Technology Stack](#17-technology-stack)
18. [Project Status & Roadmap](#18-project-status--roadmap)
19. [License](#19-license)

---

## 1. The Story

Every breakthrough in modern AI is shadowed by a brutal, invisible question: **"What will it cost to find out if this works?"**

GPT‑3 cost an estimated **\$4.6M** to train. A single failed configuration on a large cluster can burn tens of thousands of dollars in hours. Researchers and engineers routinely:

- write a model,
- spin up expensive GPUs,
- watch it crash with `CUDA out of memory`,
- shrink the batch size, try again, crash again,
- and only *then* — days later — learn that the architecture was never going to fit.

This is the opposite of science. It's trial‑and‑error with a price tag.

**NEURAX was born from a simple conviction:** the cost, memory, speed and feasibility of a neural network are not mysteries to be discovered by *running* it — they are **physical quantities that can be computed**, the same way a structural engineer computes whether a bridge will stand before pouring concrete.

So we built a **compiler that doesn't run your model — it understands it.** Feed NEURAX a description of any architecture and it lowers it through a real multi‑dialect IR pipeline, applies first‑principles analytical models calibrated against real hardware, and hands you a complete engineering report in **milliseconds**.

---

## 2. The Problem NEURAX Solves

| The question every AI team asks | When they usually find out | With NEURAX |
|---|---|---|
| **Will it fit in VRAM?** | After the OOM crash | Before writing code |
| **How fast / how much throughput?** | After profiling a live run | Instantly, from the roofline |
| **How much will training cost?** | After the cloud invoice | In GPU‑hours, \$, kWh, CO₂ — upfront |
| **Where is the bottleneck?** | After deep profiling | Compute‑ vs memory‑bound, per layer |
| **What parallelism strategy?** | After trial and error | Recommended DP/TP/PP + efficiency |

The cost of answering these questions late is measured in **weeks of engineer time and millions of dollars of compute**. NEURAX collapses it to a single command.

---

## 3. How NEURAX Changes AI Research

NEURAX shifts model design from *"build → run → discover"* to **"design → predict → decide"**:

- 🔬 **Explore 100× more architectures.** Sweep hundreds of design variants analytically in the time it takes to launch one training job.
- 💸 **Budget before you burn.** Get a dollar/energy/CO₂ figure for a run *before* requesting the cluster — turn capacity planning into a calculation.
- 🚫 **Kill dead‑ends instantly.** Catch infeasible (OOM) or hopelessly inefficient configurations in milliseconds, not days.
- 🌍 **Make AI accountable.** Report the carbon and energy footprint of a model at design time, enabling greener research.
- 🎓 **Teach the physics of deep learning.** A transparent, formula‑driven engine that shows *why* a model is compute‑bound or memory‑bound.
- 🧭 **Democratize scale.** Let small teams reason about 70B‑ and 175B‑parameter models without owning a supercomputer.

---

## 4. What NEURAX Is

NEURAX is an **analytical compiler** for neural network architectures. Like a traditional compiler it has a front‑end (parser), a multi‑stage **intermediate representation (IR)**, optimization/analysis passes, and a code‑generation backend — but instead of emitting machine code to be *executed*, it emits a complete **engineering report** of the model's behaviour on target hardware, **plus** real MLIR for downstream lowering.

```
            ┌──────────────┐      ┌────────────────────────────────┐      ┌──────────────┐
  model.json│   PARSER     │ AST  │     10-DIALECT IR PIPELINE      │  IR  │   REPORT     │  35+ metrics
 ──────────▶│ typed config │─────▶│ arch ▸ graph ▸ tensor ▸ op ▸    │─────▶│  JSON / MD   │──────────▶
            └──────────────┘      │ compute ▸ memory ▸ parallelism  │      └──────────────┘
                                  │ ▸ hardware ▸ cost ▸ report      │              │
                                  └────────────────────────────────┘              ▼
                                                  │                        ┌──────────────┐
                                                  └───────────────────────▶│  NEURAX-MLIR │  model.mlir
                                                       code generation      │   LLVM 18    │──────────▶
                                                                            └──────────────┘
```

It supports **Transformers, Mixture‑of‑Experts, CNNs, RNN/LSTM, State‑Space Models (Mamba/RWKV), Diffusion (U‑Net/SDXL), GNNs and custom architectures** through one universal JSON schema.

---

## 5. How It Works — The IR Pipeline

NEURAX is built like a production compiler. The parsed model is lowered through **ten IR dialects**, each a *pass* that enriches the program with new analysis. Each dialect owns a clearly defined slice of the metrics.

| # | Dialect | Responsibility |
|---|---------|----------------|
| 1 | **Architecture** | Layer inventory, parameter counting, model topology |
| 2 | **Graph** | Builds the computation DAG, topological order, critical path / depth |
| 3 | **Tensor** | Shape propagation (incl. symbolic dims), activation tensor sizing |
| 4 | **Operator** | Lowers layers to atomic ops (MatMul, Attention, Conv…) |
| 5 | **Compute** | Forward / backward / optimizer FLOPs, arithmetic intensity |
| 6 | **Memory** | Liveness analysis, peak VRAM, fragmentation, max batch that fits |
| 7 | **Parallelism** | Data / tensor / pipeline strategy, all‑reduce cost, scaling efficiency |
| 8 | **Hardware** | Roofline model, compute‑ vs memory‑bound, tensor‑core utilization |
| 9 | **Cost** | GPU‑hours, USD, energy (kWh), CO₂ from a pricing / energy database |
| 10 | **Report** | Aggregation, diagnostics (OOM, bottleneck) and recommendations |

Each pass is independently unit‑tested and the whole pipeline is trivially extensible to new architecture families.

---

## 6. The MLIR Compiler Backend

`neurax-mlir` is the project's **canonical code‑generation compiler**, selected for its depth and structure. It is built on **MLIR** through the `melior` Rust bindings against **LLVM 18**, and provides:

- **15 custom MLIR dialects** — `arch`, `graph`, `tensor`, `operator`, `compute`, `memory`, `parallelism`, `hardware`, `cost`, `report`, `virt`, `training`, `data`, `optimization`, `utils`.
- **Lowering passes**, including LLVM lowering and a code‑generation pass.
- **Multi‑target backends** — CPU, CUDA, ROCm, Metal, Vulkan — plus **IREE** integration.
- **TableGen (ODS)** dialect definitions alongside the Rust implementation.

It consumes the typed model from the parser and emits textual MLIR. The public entry point:

```rust
neurax_mlir::compile_model_to_mlir(context: &melior::Context, config: &ModelConfig)
    -> Result<String, String>
```

is wired directly into the CLI's `compile` command (behind the `mlir` feature), so one command turns a model JSON into a `model.mlir` module.

> **Note:** the analytical pipeline (metrics) is always available and needs no system dependencies. The MLIR code‑generation layer is **feature‑gated** (`--features mlir`) and requires LLVM 18.

---

## 7. NEURAX vs. XLA / TVM / IREE / TensorRT

NEURAX is **not a competitor** to runtime kernel compilers — it operates **one level upstream**. XLA, TVM, IREE and TensorRT take a model and make it *run fast on hardware*. **NEURAX tells you whether you should run it at all, what it will cost, and whether it fits — before any of those tools are even invoked.**

| Capability | **NEURAX** | XLA | TVM | IREE | TensorRT |
|---|:---:|:---:|:---:|:---:|:---:|
| **Predicts cost (\$ / GPU‑h)** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Predicts energy & CO₂** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Predicts peak VRAM / OOM** | ✅ | ⚠️ partial | ⚠️ partial | ⚠️ partial | ⚠️ partial |
| **Predicts latency & throughput** | ✅ (analytical) | via run | via run/tuning | via run | via run |
| **Recommends parallelism (DP/TP/PP)** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Works without a GPU / without running** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Answer time** | **milliseconds** | minutes–hours | minutes–hours (autotune) | minutes | minutes |
| **Generates MLIR** | ✅ | ✅ | ✅ (Relay/Relax) | ✅ | ❌ |
| **Optimizes & emits runnable kernels** | 🚧 roadmap | ✅ | ✅ | ✅ | ✅ |
| **Architecture‑family aware (MoE, SSM, Diffusion…)** | ✅ | ⚠️ | ⚠️ | ⚠️ | ⚠️ |

**In one line:** *XLA/TVM/IREE/TensorRT make a model **fast**; NEURAX tells you if it's **worth running** and what it will **cost** — and it shares their MLIR foundation, making it a natural pre‑flight layer in front of them.*

---

## 8. The Interfaces

NEURAX meets every kind of user where they are — from a terminal to a polished web canvas.

### 🖥️ `neurax` — Command‑Line Interface
The engineer's tool: `analyze`, `validate`, `summary`, and the full `compile` pipeline (LLVM IR + **real MLIR** + metrics).

### 📊 Terminal UI (`neurax-tui`)
A **Ratatui** dashboard to browse bundled architectures and view live analysis without leaving the terminal.

### 🎨 Web UI (`neurax-ui`)
A modern **React 18 + TypeScript + Vite** application with a **visual architecture canvas** — drag‑and‑drop layers, connect blocks, and watch metrics update in real time. Built with **shadcn/ui** (Radix primitives), **TailwindCSS**, **Recharts** for live charts, **TanStack Query**, **react‑hook‑form + Zod**, and **Supabase** auth.

### 🤖 AI Planning Agent (`neurax-agent`)
A **FastAPI + LangChain** co‑pilot. Send it the canvas snapshot and a natural‑language request (with an adjustable *creativity* dial); it streams suggestions over **Server‑Sent Events**, proposes architecture edits from a curated block catalogue, validates topology, and lays out the result. This is NEURAX as a **design partner**, not just an analyzer.

### 🌐 HTTP Service (`neurax-service`)
The **actix‑web** API that powers the web UI and integrations — analysis, hardware catalogue, presets, authentication and billing (see below).

---

## 9. The Complete Service & API

`neurax-service` is a production **actix‑web** server (default `0.0.0.0:9098`) with CORS, gzip compression, a 10 MB payload limit, a 60‑second analysis timeout, **Supabase** authentication and **Stripe** billing.

| Method | Endpoint | Auth | Purpose |
|---|---|:---:|---|
| `GET` | `/health` | — | Liveness probe |
| `POST` | `/analyze` | ✅ verified email | Run the full analytical pipeline on a topology, return the `ReportIR` |
| `GET` | `/hardware` | — | GPU catalogue (H100, A100, RTX 4090/4080/3090) with peak FLOPs, bandwidth, VRAM |
| `GET` | `/presets` · `/presets/{id}` | — | Ready‑made model presets |
| `POST` | `/plugin/validate` | — | Validate a plugin/extension payload |
| `GET` | `/me` | ✅ | Current user + active plan tier |
| `POST` | `/billing/checkout` | ✅ | Create a Stripe Checkout session for a plan/interval |
| `POST` | `/billing/portal` | ✅ | Open the Stripe customer billing portal |
| `POST` | `/stripe/webhook` | HMAC | Stripe events (HMAC‑SHA256 verified, idempotent subscription sync) |

**Security & correctness highlights:** bearer‑token auth via Supabase, email‑verification gating on analysis, constant‑time webhook signature verification, idempotent webhook processing, and plan resolution that respects subscription status and admin overrides.

---

## 10. Pricing & Plans

NEURAX ships as a **SaaS** with four tiers, billed **monthly or annually** through Stripe (`free`, `essential`, `architect`, `elite`). Subscription state is synced from Stripe webhooks into Supabase and surfaced via `/me`.

| | 🆓 **Free** | 🚀 **Essential** | 🏗️ **Architect** | 👑 **Elite** |
|---|:---:|:---:|:---:|:---:|
| **For** | Students & explorers | Individual researchers | Teams & labs | Enterprises |
| **Analyses / month** | Limited | Generous | High | Unlimited |
| **All 35+ metrics** | ✅ | ✅ | ✅ | ✅ |
| **Visual canvas UI** | ✅ | ✅ | ✅ | ✅ |
| **AI planning agent** | Preview | ✅ | ✅ | ✅ (priority) |
| **MLIR compile export** | — | ✅ | ✅ | ✅ |
| **Cost / energy / CO₂ reports** | Basic | ✅ | ✅ + history | ✅ + history |
| **Parallelism advisor** | — | ✅ | ✅ | ✅ |
| **API access** | — | Limited | ✅ | ✅ (higher limits) |
| **Support** | Community | Email | Priority | Dedicated / SLA |
| **Billing** | — | Monthly / Annual | Monthly / Annual | Monthly / Annual / Custom |

> 💡 **Annual billing is discounted** (each tier exposes `*_MONTHLY` and `*_ANNUAL` Stripe prices). Exact price points are configured in Stripe and shown on the in‑app pricing page; the table above describes positioning and feature differentiation. The economic value proposition is simple: **one avoided OOM crash or one cancelled dead‑end training run pays for the subscription many times over.**

---

## 11. The 35+ Metrics

A NEURAX report covers, among others:

- **Architecture** — total / trainable parameters, layer count, params by family.
- **Compute** — forward, backward and optimizer FLOPs; FLOPs/token; **incremental‑decode FLOPs** (KV‑cache); arithmetic intensity; top‑10 most expensive layers.
- **Memory** — parameter / activation / gradient / optimizer memory; peak VRAM; fragmentation estimate; maximum batch size that fits a given GPU.
- **Hardware** — latency (ms/token), throughput (tokens/s), roofline bottleneck classification, tensor‑core utilization.
- **Parallelism** — recommended data / tensor / pipeline configuration and scaling efficiency.
- **Cost** — training hours, GPU‑hours, USD, cost per million tokens, energy (kWh) and CO₂ estimate.
- **Dynamic (M36–M55)** — virtual‑memory defrag savings, stability/chaos indices, and behavioral metrics (expert load imbalance, cache locality, numerical sensitivity…).
- **Diagnostics** — automatic OOM / bottleneck warnings and optimization recommendations (gradient checkpointing, more GPUs, precision changes…).

Every one of these metrics is surfaced in the web UI's Metrics Dashboard and per‑layer charts.

---

## 12. Repository Layout

Everything lives in a single, self‑contained workspace.

```
neurax_full/
├── Cargo.toml              # Rust workspace (members + shared dependencies)
├── README.md               # You are here
├── DESIGN.md               # Architecture & design notes
│
├── neurax-parser/          # JSON ingestion → strongly-typed ModelConfig
├── neurax-formulas/        # Per-architecture FLOPs / parameter formulas
├── neurax-hardware-db/     # GPU / CPU / interconnect spec database
├── neurax-ir/              # The 10 IR dialects (analytical engine)
├── neurax-core/            # Pipeline orchestrator + IR backend abstraction
├── neurax-mlir/            # ★ MLIR compiler backend (15 dialects, LLVM 18)
│
├── neurax-cli/             # `neurax` command-line tool  (analyze / compile / validate)
├── neurax-tui/             # Ratatui terminal user interface
├── neurax-service/         # actix-web HTTP API (auth, billing, analysis)
├── neurax-ui/              # React 18 + TypeScript + Vite web front-end (visual canvas)
├── neurax-agent/           # Python / FastAPI / LangChain architecture-planning agent
│
├── models/                 # Sample model definitions + reference outputs
├── test_models/            # 20 curated architectures for testing
└── examples/models/        # Models bundled into the TUI
```

**Internal dependency graph (Rust):**

```
neurax-cli ─┬─▶ neurax-core ─┬─▶ neurax-ir ──▶ neurax-formulas
            │                ├─▶ neurax-parser
            │                └─▶ neurax-hardware-db
            └─▶ neurax-mlir ──▶ neurax-parser          (feature "mlir")
neurax-tui / neurax-service ──▶ neurax-core
neurax-ui  ──HTTP──▶ neurax-service ◀──HTTP── neurax-agent
```

---

## 13. Installation

### Prerequisites

- **Rust** (edition 2021) — install via [rustup](https://rustup.rs).
- **LLVM 18** — *only required for the MLIR backend* (`--features mlir`).
- **Node.js / Bun** — for the web UI. **Python 3.11+** — for the agent.

On Debian/Ubuntu:

```bash
sudo apt install llvm-18 llvm-18-dev libmlir-18-dev mlir-18-tools
```

Set the environment so the `melior` / `mlir-sys` build scripts find LLVM 18:

```bash
export LLVM_SYS_180_PREFIX=/usr/lib/llvm-18
export MLIR_SYS_180_PREFIX=/usr/lib/llvm-18
export TABLEGEN_180_PREFIX=/usr/lib/llvm-18
export PATH="/usr/lib/llvm-18/bin:$PATH"
```

---

## 14. Building

**Analytical engine only** (no LLVM needed):

```bash
cargo build -p neurax-cli            # the `neurax` CLI, analysis features
```

**Full build, including the MLIR compiler** (requires the LLVM 18 env above):

```bash
cargo build --workspace --features neurax-cli/mlir
```

Run the test suites:

```bash
cargo test -p neurax-core --lib
cargo test -p neurax-mlir            # 118 tests (needs LLVM 18 env)
```

---

## 15. Usage

### Command‑line (`neurax`)

```bash
# Full analytical report (Markdown or JSON)
neurax analyze test_models/01_gpt2_small.json --format markdown

# Validate a model JSON against the schema
neurax validate test_models/04_mixtral_8x7b.json

# Quick one-line summary
neurax summary test_models/10_deepseek_v3.json

# Full compilation pipeline → emits model.mlir, model.ll, metrics.json …
#   (the MLIR step requires the binary built with --features mlir)
neurax compile test_models/01_gpt2_small.json -o ./out
```

Example `compile` output:

```
[2/6] Analyzing model architecture...
      ✓ Analysis completed in 3 ms
      • Total params: 208.94M (0.2089B)
[5/6] Generating native code...
      ✓ LLVM IR generated: 47 lines
      ✓ MLIR generated: 91 lines      ← real MLIR via neurax-mlir
✅ Compilation complete!
```

### The other surfaces

```bash
cargo run -p neurax-tui                         # interactive terminal UI
cargo run -p neurax-service                     # HTTP API on :9098
cd neurax-ui && npm install && npm run dev      # web canvas (Vite, :8081)
cd neurax-agent && pip install -r requirements.txt && python app.py   # AI agent
```

---

## 16. The Universal Model JSON

A model is one JSON document: global parameters, an ordered list of layers, the training config, the target hardware, and a cost config.

```json
{
  "model": {
    "name": "GPT2-Small",
    "model_type": "transformer",
    "global_params": {
      "sequence_length": 1024,
      "vocab_size": 50257,
      "embedding_dim": 768,
      "num_layers": 12
    },
    "layers": [
      { "id": "embed", "layer_type": "embedding",
        "params": { "vocab_size": 50257, "embedding_dim": 768 } },
      { "id": "blk",   "layer_type": "attention",
        "params": { "hidden_size": 768, "num_heads": 12 } },
      { "id": "ffn",   "layer_type": "mlp",
        "params": { "hidden_size": 768, "intermediate_size": 3072 } }
    ]
  },
  "training": {
    "batch_size": 8, "precision": "fp16", "optimizer": "adamw",
    "parallelism": { "data_parallel": 1, "tensor_parallel": 1, "pipeline_parallel": 1 }
  },
  "hardware": { "gpus": [ { "name": "A100", "memory_gb": 80, "tflops_fp16": 312 } ] }
}
```

See `test_models/` for **20 complete, ready‑to‑analyze examples** spanning every supported family.

---

## 17. Technology Stack

| Layer | Technology |
|-------|------------|
| **Core engine** | Rust 2021, `rayon`, `petgraph`, `evalexpr` |
| **Compiler backend** | MLIR via `melior`, LLVM 18, TableGen ODS, IREE |
| **HTTP service** | actix‑web 4, Tokio, Supabase (auth), Stripe (billing), HMAC‑SHA256 |
| **Terminal UI** | Ratatui + crossterm |
| **Web UI** | React 18, TypeScript, Vite 8, TailwindCSS, shadcn/ui (Radix), Recharts, TanStack Query, Zod |
| **Planning agent** | Python, FastAPI, LangChain, Server‑Sent Events |

---

## 18. Project Status & Roadmap

**Status**
- ✅ Analytical pipeline (10 dialects, 35+ metrics) — operational.
- ✅ `neurax-mlir` integrated as the canonical compiler backend; **118 tests pass**.
- ✅ End‑to‑end `neurax compile` emits real MLIR.
- ✅ Full workspace builds green (`--features neurax-cli/mlir`).
- ✅ Single self‑contained folder; the legacy Pliron backend fully removed.
- ✅ Web UI, AI agent, and SaaS service (auth + Stripe billing) in place.

**Roadmap**
- 🚧 Lower NEURAX‑MLIR all the way to runnable kernels (closing the loop with IREE).
- 🚧 Migrate the remaining `neurax-core` integration fixtures into the workspace.
- 🚧 Expand the architecture catalogue and hardware database.
- 🚧 Public benchmark suite validating predictions against measured runs.

---

## 19. License

Proprietary — © NEURAX. All rights reserved unless a separate license file states otherwise.

<div align="center">
<br/>

**NEURAX** — _See the cost of intelligence, before you pay for it._

</div>
