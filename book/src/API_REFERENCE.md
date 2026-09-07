# NEURAX API Reference

`neurax-service` is a production **actix‑web** HTTP server (default `0.0.0.0:9098`) exposing **42+ REST routes** with CORS, gzip compression, and authentication via Supabase JWT or API keys.

## Base URL

```
http://localhost:9098
```

## Authentication

Two authentication methods are supported:

### 1. Supabase JWT (Web UI Users)

Pass a valid Supabase access token in the `Authorization` header:

```http
Authorization: Bearer <supabase-jwt>
```

### 2. API Keys (Programmatic Access)

Pass an API key (prefixed with `nrx_`) in either the `X-API-Key` header or the `Authorization: Bearer` header:

```http
X-API-Key: nrx_<64-hex-chars>
```

### API Key Scopes

| Scope | Access |
|---|---|
| `analyze` | Analysis, comparison, streaming, presets, hardware, time machine |
| `inference` | Inference simulation |
| `compare` | Multi‑hardware comparison |
| `export` | ONNX and GitHub export |
| `projects` | Project CRUD operations |
| `agent` | Agent control endpoints (grants access to all agent endpoints) |
| `all` | Full access to all endpoints |

### Error Codes

| Code | Meaning |
|---|---|
| `401 Unauthorized` | Missing or invalid API key / JWT |
| `403 Forbidden` | API key lacks required scope |
| `400 Bad Request` | Invalid request body or parameters |
| `404 Not Found` | Resource not found |
| `408 Request Timeout` | Analysis timed out (60s) |
| `500 Internal Server Error` | Unexpected server error |
| `502 Bad Gateway` | Downstream service (Supabase/Stripe) unavailable |
| `504 Gateway Timeout` | Downstream service timed out |

---

## Endpoint Summary

### System

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | None | Health check |
| `GET` | `/me` | JWT | Get current user profile and subscription plan |

### Analysis

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/analyze` | JWT/API Key | Run full 11-phase analytical pipeline synchronously |
| `POST` | `/analyze/stream` | JWT/API Key | Start streaming analysis (SSE) |
| `GET` | `/analyze/stream/{job_id}` | JWT/API Key | Stream SSE events for a running job |
| `GET` | `/analyze/result/{job_id}` | JWT/API Key | Retrieve completed analysis result |
| `GET` | `/analyze/status/{job_id}` | JWT/API Key | Check job status |
| `POST` | `/analyze/compare` | JWT/API Key | Compare up to 8 hardware configurations |
| `POST` | `/sweep` | JWT/API Key | Grid-search batch_size x zero_stage x gpu_count x precision for the best feasible training config |

### Inference

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/inference/simulate` | JWT/API Key | Predict inference stability, hallucination risk, sampling volatility |

### Time Machine

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/timemachine` | JWT/API Key | Multi‑year cost/carbon projection |

### Export

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/export/onnx` | JWT/API Key | Binary ONNX protobuf export |
| `POST` | `/export/github` | JWT/API Key | Push model files to GitHub, optionally create PR |

### Presets

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/presets` | None | List all reference architecture presets |
| `GET` | `/presets/{id}` | None | Get a specific preset by ID |

### Hardware

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/hardware` | None | List all hardware specifications (21 GPUs, CPUs, interconnects) |

### Projects

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/projects` | JWT | List user projects |
| `POST` | `/projects` | JWT | Create a new project |
| `GET` | `/projects/{id}` | JWT | Get a specific project |
| `PUT` | `/projects/{id}` | JWT | Update a project |
| `DELETE` | `/projects/{id}` | JWT | Delete a project |

### Credits & Billing

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/credits` | JWT | Get usage balance and plan limits |
| `POST` | `/billing/checkout` | JWT | Create Stripe checkout session |
| `POST` | `/billing/portal` | JWT | Create Stripe customer portal session |
| `POST` | `/stripe/webhook` | None | Stripe webhook endpoint |

### Compliance

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/compliance/config` | JWT | Get regulatory compliance configuration (EU AI Act, CSRD, DSA) |

### API Keys

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/api-keys` | JWT | List all API keys |
| `POST` | `/api-keys` | JWT | Create a new API key |
| `POST` | `/api-keys/{key_id}/revoke` | JWT | Revoke an API key |
| `DELETE` | `/api-keys/{key_id}` | JWT | Delete an API key |

### Agent Control (API Key Auth)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/agent/analyze` | API Key | Agent‑initiated analysis |
| `POST` | `/agent/inference` | API Key | Agent‑initiated inference simulation |
| `POST` | `/agent/compare` | API Key | Agent‑initiated comparison |
| `POST` | `/agent/audit` | API Key | Agent‑initiated audit |
| `POST` | `/agent/carbon` | API Key | Agent‑initiated carbon calculation |
| `GET` | `/agent/compliance` | API Key | Agent‑initiated compliance check |
| `GET` | `/agent/results` | API Key | Agent‑initiated results retrieval |
| `GET` | `/agent/projects` | API Key | Agent‑initiated project listing |

### Plugin

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/plugin/validate` | None | Validate a plugin architecture |

---

## Endpoint Details

### `GET /health`

Health check endpoint. No authentication required.

**Response** `200 OK`:

```json
{
  "status": "ok"
}
```

---

### `POST /analyze`

Run the full 11-phase analytical pipeline on a model configuration.

**Request Body**:

```json
{
  "topology": {
    "schema_version": "1.0",
    "model": {
      "name": "MyModel",
      "type": "transformer",
      "global_params": {
        "num_layers": 12,
        "sequence_length": 2048,
        "vocab_size": 50257,
        "embedding_dim": 768
      },
      "layers": [
        {
          "id": "layer_1",
          "layer_type": "embedding",
          "params": {
            "vocab_size": 50257,
            "embedding_dim": 768
          }
        }
      ]
    },
    "training": {
      "batch_size": 128,
      "max_steps": 100000
    },
    "hardware": {
      "gpus": [
        {"name": "A100-SXM", "count": 8}
      ]
    }
  }
}
```

**Response** `200 OK`:

```json
{
  "report": {
    "model_name": "MyModel",
    "total_parameters": 125000000,
    "total_flops": 2.5e17,
    "peak_vram_bytes": 42000000000,
    "training_cost_usd": 45000.00,
    "training_time_hours": 120.5,
    "energy_kwh": 15000.0,
    "co2_kg": 4500.0,
    "metrics": { },
    "diagnostics": [ ],
    "phase_timeline": [ ]
  }
}
```

**Error Responses**:

| Code | Message |
|---|---|
| `400` | `Analysis error: <details>` — Invalid model config |
| `504` | `Analysis timed out after 60 seconds` |
| `500` | `Analysis task failed unexpectedly` |

---

### `POST /sweep`

Grid-search over `batch_size x zero_stage x gpu_count x precision` for a model configuration. Each point in the grid is a full, real call to the same analytical pipeline `/analyze` uses — nothing here is approximated or cached from a smaller run. Points whose `peak_vram_bytes` would not fit the configured GPU are marked infeasible and never selected as `best`. Capped at 512 combinations per request; a larger requested grid is rejected with `400` rather than silently truncated.

**Request Body**:

```json
{
  "topology": { "...": "same shape as /analyze's topology" },
  "candidates": {
    "batch_sizes": [1, 8, 32, 64],
    "zero_stages": [0, 1, 2, 3],
    "gpu_counts": [1, 4, 8],
    "precisions": ["fp16", "bf16", "fp8"]
  },
  "objective": "max_throughput"
}
```

`candidates` is optional — any field left out is filled in with the compiler's own defaults for that model (`SweepCandidates::defaults_for`). `objective` is one of `max_throughput`, `min_cost`, `min_latency`, `max_batch_size` (default: `max_throughput`).

**Response** `200 OK`:

```json
{
  "result": {
    "points": [
      {
        "batch_size": 32,
        "zero_stage": 2,
        "gpu_count": 8,
        "precision": "bf16",
        "feasible": true,
        "peak_vram_gb": 62.4,
        "throughput_tokens_per_s": 184320.0,
        "latency_ms": 173.9,
        "training_cost_usd": 0.0421
      }
    ],
    "best": { "...": "the single SweepPoint that wins the requested objective, or null if none were feasible" }
  }
}
```

`points` includes every evaluated combination, feasible or not — so a caller can see the whole feasibility frontier, not just the winner. `best` is `null` when no candidate fit the GPU's VRAM at all.

**Error Responses**:

| Code | Message |
|---|---|
| `400` | `Analysis error: <details>` — Invalid model config |
| `400` | `Sweep grid too large (<n> combinations, max 512) — narrow the candidate lists` |
| `504` | Sweep timed out after 60 seconds |

---

### `POST /analyze/stream`

Start a streaming analysis job. Returns a job ID for SSE streaming.

**Request Body** (same as `/analyze`).

**Response** `200 OK`:

```json
{
  "job_id": "abc123-...",
  "status": "running"
}
```

---

### `GET /analyze/stream/{job_id}`

Stream Server‑Sent Events for a running analysis job.

**Response** `text/event-stream`:

```
event: phase
data: {"phase": "Architecture", "status": "running", "progress": 10}

event: metric
data: {"name": "total_parameters", "value": 125000000, "unit": "params"}

event: done
data: {"job_id": "abc123-...", "status": "completed"}
```

---

### `GET /analyze/result/{job_id}`

Retrieve the completed analysis report for a job.

**Response** `200 OK`:

```json
{
  "job_id": "abc123-...",
  "status": "completed",
  "report": { }
}
```

---

### `POST /analyze/compare`

Compare up to 8 hardware configurations for the same model.

**Request Body**:

```json
{
  "topology": { },
  "configs": [
    {
      "hardware": "H100-SXM",
      "gpu_count": 8,
      "precision": "fp16",
      "gpu_memory_gb": 80,
      "gpu_bandwidth_gbs": 3352.0
    },
    {
      "hardware": "A100-SXM",
      "gpu_count": 8,
      "precision": "bf16",
      "gpu_memory_gb": 80,
      "gpu_bandwidth_gbs": 2039.0
    }
  ]
}
```

**Response** `200 OK`:

```json
{
  "results": [
    {
      "label": "8 × H100-SXM @ fp16",
      "report": { }
    },
    {
      "label": "8 × A100-SXM @ bf16",
      "report": { }
    }
  ]
}
```

---

### `POST /inference/simulate`

Simulate inference behavior for a model configuration.

**Request Body**:

```json
{
  "topology": { },
  "params": {
    "model_name": "MyModel",
    "num_layers": 12,
    "d_model": 768,
    "num_heads": 12,
    "seq_len": 2048,
    "vocab_size": 50257,
    "batch_size": 1,
    "precision": "fp16",
    "hardware": "A100-SXM"
  }
}
```

**Response** `200 OK`:

```json
{
  "report": {
    "model_name": "MyModel",
    "stability_index": 0.85,
    "hallucination_risk": 0.12,
    "sampling_volatility": 0.05,
    "latency_ms": 12.5,
    "throughput_tokens_per_s": 800.0
  }
}
```

---

### `POST /timemachine`

Multi‑year cost/carbon projection.

**Request Body**:

```json
{
  "topology": { },
  "years": 5,
  "hardware_growth_rate": 0.15,
  "energy_cost_per_kwh": 0.12,
  "carbon_intensity_g_per_kwh": 475
}
```

**Response** `200 OK`:

```json
{
  "projections": [
    {
      "year": 2026,
      "training_cost_usd": 45000.0,
      "energy_kwh": 15000.0,
      "co2_kg": 4500.0
    }
  ],
    "compliance": { }
}
```

---

### `POST /export/onnx`

Export a model topology to ONNX binary format.

**Request Body**:

```json
{
  "topology": { },
  "model_name": "MyModel"
}
```

**Response** `200 OK`:

```json
{
  "data": "<base64-encoded-onnx-protobuf>",
  "model_name": "MyModel",
  "node_count": 12
}
```

---

### `POST /export/github`

Push model files to a GitHub repository and optionally create a pull request.

**Request Body**:

```json
{
  "topology": { },
  "github_token": "<personal-access-token>",
  "owner": "username",
  "repo": "repo-name",
  "branch": "main",
  "create_pr": true,
  "pr_title": "Add MyModel architecture",
  "pr_body": "Auto-generated by NEURAX"
}
```

**Response** `200 OK`:

```json
{
  "success": true,
  "commit_sha": "abc123...",
  "pr_url": "https://github.com/rustnew/NEURAX/pull/1"
}
```

---

### `GET /projects`

List all projects for the authenticated user.

**Response** `200 OK`:

```json
[
  {
    "id": "proj_123",
    "name": "My Project",
    "description": "A transformer model",
    "topology": { },
    "created_at": "2026-07-24T10:00:00Z",
    "updated_at": "2026-07-24T10:00:00Z"
  }
]
```

---

### `POST /projects`

Create a new project.

**Request Body**:

```json
{
  "name": "My Project",
  "description": "A transformer model",
  "topology": { }
}
```

**Response** `200 OK`:

```json
{
  "id": "proj_123",
  "name": "My Project",
  "description": "A transformer model",
  "topology": { },
  "created_at": "2026-07-24T10:00:00Z",
  "updated_at": "2026-07-24T10:00:00Z"
}
```

---

### `GET /credits`

Get the current user's credit balance and plan information.

**Response** `200 OK`:

```json
{
  "credits": {
    "used": 150,
    "limit": 1000,
    "plan": "elite",
    "period_start": "2026-07-01T00:00:00Z",
    "period_end": "2026-08-01T00:00:00Z"
  }
}
```

---

### `GET /compliance/config`

Get regulatory compliance configuration (EU AI Act, CSRD, DSA).

**Response** `200 OK`:

```json
{
  "eu_ai_act": { },
  "csrd": { },
  "dsa": { }
}
```

---

### API Key Management

#### `POST /api-keys`

Create a new API key.

**Request Body**:

```json
{
  "name": "My API Key",
  "scopes": ["analyze", "inference", "export"]
}
```

**Response** `200 OK`:

```json
{
  "key": "nrx_abc123...",
  "name": "My API Key",
  "user_id": "user_123",
  "created_at": "2026-07-24T10:00:00Z",
  "active": true,
  "scopes": ["analyze", "inference", "export"]
}
```

#### `GET /api-keys`

List all API keys for the current user.

#### `POST /api-keys/{key_id}/revoke`

Revoke (deactivate) an API key.

#### `DELETE /api-keys/{key_id}`

Delete (permanently remove) an API key.

---

### `POST /plugin/validate`

Validate a plugin architecture specification.

**Request Body**:

```json
{
  "topology": { }
}
```

**Response** `200 OK`:

```json
{
  "valid": true,
  "warnings": [],
  "errors": []
}
```

---

## Python Agent API

The `neurax-agent` (FastAPI, port 8099) provides a separate API for AI‑driven architecture design.

### `POST /runs`

Start a new agent run. The agent uses an LLM (OpenAI or Anthropic) to plan, validate, and materialize an architecture.

**Request Body**:

```json
{
  "user_message": "Design a transformer model with 12 layers for text classification",
  "snapshot": {
    "family": "transformer",
    "nodes": [],
    "connections": [],
    "groups": [],
    "allowed_layer_types": [],
    "allowed_families": ["transformer", "cnn", "moe"],
    "catalogue_id": null,
    "catalogue": [],
    "missing_mandatory_fields": [],
    "hw_config": {},
    "analysis_warnings": []
  },
  "creativity": 0.3
}
```

**Response** `200 OK`:

```json
{
  "run_id": "abc123-..."
}
```

### `GET /runs/{run_id}/events`

Stream Server‑Sent Events for an agent run.

**Response** `text/event-stream`:

```
event: assistant
data: {"content": "I'll design a transformer architecture for you..."}

event: tool
data: {"name": "add_node", "args": {"id": "input", "type": "input", "params": {}}}

event: done
data: {}
```

### `GET /health`

Agent health check.

**Response** `200 OK`:

```json
{
  "status": "ok"
}
```

---

## CLI Interface

The `neurax` CLI provides command‑line access to the core analysis pipeline.

```bash
# Analyze a model and generate a report
neurax analyze model.json -o report.md

# Analyze with JSON output
neurax analyze model.json -f json -o report.json

# Validate a JSON model configuration
neurax validate model.json

# Show a quick summary of the model
neurax summary model.json

# Show version
neurax version
```


---

## MCP Server

The `neurax-mcp` package provides a Model Context Protocol server that exposes NEURAX capabilities to MCP‑compatible clients (e.g., Claude Desktop).

### Available Tools

| Tool | Description |
|---|---|
| `analyze_architecture` | Analyze a neural network architecture |
| `list_templates` | List available reference templates |
| `get_template` | Get a specific template |
| `list_hardware` | List supported hardware |
| `estimate_training_cost` | Estimate training cost for a model |
| `get_compliance_config` | Get regulatory compliance configuration |
| `get_credits` | Get credit balance information |
| `get_user_info` | Get user profile information |
| `health_check` | Check NEURAX service health |
