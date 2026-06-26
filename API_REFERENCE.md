# NEURAX API Reference

Base URL: `http://localhost:9098`

All endpoints require authentication via Supabase JWT token in the `Authorization: Bearer <token>` header, unless `NEURAX_DEBUG_NOAUTH=true` is set.

---

## Authentication

### GET /me

Returns the authenticated user's profile and subscription status.

**Response:**
```json
{
  "id": "uuid",
  "email": "user@example.com",
  "plan": "free|essential|architect|elite",
  "subscription_status": "active|inactive|past_due"
}
```

---

## Analysis

### POST /analyze

Run a full analysis on a model topology.

**Request:**
```json
{
  "topology": { ... },
  "env": {
    "hardware": "A100-SXM",
    "precision": "fp16",
    "batch_size": 64,
    "seq_len": 2048,
    "confidence_min": 0.8,
    "no_variants": false,
    "parallel_scan": false
  }
}
```

**Response:** Full analysis report (ReportIR JSON) with 35+ metrics across all 10 IR dialects.

---

### POST /analyze/stream

Start a streaming analysis job. Returns a job ID, then streams progress events via SSE.

**Request:** Same as `/analyze`

**Response:**
```json
{
  "job_id": "uuid"
}
```

### GET /analyze/stream/{job_id}

SSE endpoint for streaming analysis events. Connect via EventSource.

**Event types:**
- `Started` — Analysis began
- `PhaseStarted` — A pipeline phase started
- `PhaseCompleted` — A pipeline phase completed
- `Progress` — Progress update with percentage
- `Diagnostic` — Diagnostic message
- `Completed` — Analysis finished
- `Failed` — Analysis failed
- `Result` — Final analysis report

### GET /analyze/result/{job_id}

Get the final result of a completed analysis job.

**Response:** Full analysis report (ReportIR JSON)

### GET /analyze/status/{job_id}

Get the status of an analysis job.

**Response:**
```json
{
  "job_id": "uuid",
  "status": "running|completed|failed",
  "progress": 0.75
}
```

---

### POST /analyze/compare

Compare the same model across multiple hardware configurations.

**Request:**
```json
{
  "topology": { ... },
  "configs": [
    {
      "hardware": "A100-SXM",
      "precision": "fp16",
      "batch_size": 64,
      "gpu_count": 1,
      "gpu_memory_gb": 80
    },
    {
      "hardware": "H100-SXM",
      "precision": "bf16",
      "batch_size": 32,
      "gpu_count": 4,
      "gpu_memory_gb": 80
    }
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "label": "A100-SXM fp16 b64 g1",
      "hardware": "A100-SXM",
      "precision": "fp16",
      "batch_size": 64,
      "gpu_count": 1,
      "report": { ... },
      "error": null
    }
  ]
}
```

Maximum 8 configurations per request.

---

## Inference Simulation

### POST /inference/simulate

Simulate inference behavior for a model.

**Request:**
```json
{
  "topology": { ... },
  "params": {
    "batch_size": 1,
    "seq_len": 128,
    "precision": "fp16",
    "hardware": "A100-SXM",
    "gpu_count": 1
  }
}
```

**Response:** InferenceReport with 10 widgets: latency, throughput, memory, batching, kv_cache, compute_utilization, bandwidth, token_generation, comparative, recommendations.

---

## Time Machine

### POST /timemachine

Project model costs over time with growth assumptions.

**Request:**
```json
{
  "topology": { ... },
  "projection": {
    "months": 24,
    "growth_rate": 0.1,
    "confidence": 0.8
  }
}
```

**Response:** TimeMachineReport with monthly projections.

---

## ONNX Export

### POST /export/onnx

Export a model topology as ONNX protobuf binary.

**Request:**
```json
{
  "topology": { ... },
  "model_name": "my-model"
}
```

**Response:**
```json
{
  "data": "base64-encoded-onnx-binary",
  "model_name": "my-model",
  "node_count": 42,
  "initializer_count": 15,
  "size_bytes": 12345
}
```

The `data` field is a base64-encoded ONNX ModelProto. Decode and save as `.onnx` file.

---

## Projects

### GET /projects

List all projects for the authenticated user.

**Response:**
```json
{
  "projects": [
    {
      "id": "uuid",
      "user_id": "uuid",
      "name": "My Project",
      "description": "...",
      "architecture": "transformer",
      "canvas": { ... },
      "hardware_config": { ... },
      "last_analysis": { ... },
      "created_at": "2026-01-01T00:00:00Z",
      "updated_at": "2026-01-01T00:00:00Z"
    }
  ]
}
```

### POST /projects

Create a new project. Maximum 50 projects per user.

**Request:**
```json
{
  "name": "My Project",
  "description": "Optional description",
  "architecture": "transformer",
  "canvas": { ... },
  "hardware_config": { ... },
  "last_analysis": null
}
```

### GET /projects/{id}

Get a single project by ID.

### PUT /projects/{id}

Update a project. All fields are optional (partial update).

**Request:**
```json
{
  "name": "Updated Name",
  "description": "Updated description"
}
```

### DELETE /projects/{id}

Delete a project.

**Response:**
```json
{
  "deleted": true
}
```

---

## Hardware

### GET /hardware

List all available GPUs with full specifications.

**Response:**
```json
{
  "gpus": [
    {
      "name": "A100-SXM",
      "manufacturer": "NVIDIA",
      "memory_gb": 80,
      "memory_bandwidth_gbs": 2039,
      "tflops_fp64": 19.5,
      "tflops_fp32": 19.5,
      "tflops_fp16": 312,
      "tflops_bf16": 312,
      "tflops_int8": 624,
      "tflops_fp8": 1248,
      "tensor_cores": true,
      "nvlink": true,
      "nvlink_bandwidth_gbs": 600,
      "tdp_watts": 400,
      "launch_year": 2020
    }
  ]
}
```

---

## Presets

### GET /presets

List all model presets.

### GET /presets/{id}

Get a specific model preset by ID.

---

## Credits

### GET /credits

Get the authenticated user's credit balance.

**Response:**
```json
{
  "credits": {
    "user_id": "uuid",
    "used": 5,
    "limit": 10,
    "plan": "free",
    "period_start": "2026-06-01",
    "period_end": "2026-07-01"
  }
}
```

Credit limits per plan:
- Free: 10/month
- Essential: 100/month
- Architect: 1000/month
- Elite: Unlimited

---

## Compliance

### GET /compliance/config

Get compliance regulations and thresholds.

**Response:**
```json
{
  "regulations": [
    {
      "name": "EU AI Act - Phase 1",
      "year": 2025,
      "limit": 10,
      "unit": "GFLOPS",
      "status": "active",
      "description": "...",
      "region": "EU"
    }
  ],
  "thresholds": {
    "high_risk_gflops": 10,
    "carbon_report_tonnes": 1000,
    "dsa_disclosure_flops": 1e15,
    "cost_review_usd": 100000
  },
  "recommendations": [
    "Monitor EU AI Act Phase 2 requirements...",
    "..."
  ]
}
```

---

## Billing

### POST /billing/checkout

Create a Stripe checkout session for subscription upgrade.

**Request:**
```json
{
  "plan": "essential|architect|elite",
  "interval": "monthly|yearly"
}
```

**Response:**
```json
{
  "url": "https://checkout.stripe.com/..."
}
```

### POST /billing/portal

Create a Stripe billing portal session for managing subscriptions.

**Response:**
```json
{
  "url": "https://billing.stripe.com/..."
}
```

### POST /stripe/webhook

Stripe webhook endpoint for subscription events. Requires `STRIPE_WEBHOOK_SECRET` environment variable.

---

## Health

### GET /health

Health check endpoint. No authentication required.

**Response:**
```json
{
  "status": "ok"
}
```

---

## Plugin Validation

### POST /plugin/validate

Validate a model topology JSON without running analysis.

**Request:**
```json
{
  "topology": { ... }
}
```

**Response:**
```json
{
  "valid": true,
  "errors": [],
  "warnings": []
}
```

---

## Error Responses

All endpoints return consistent error responses:

```json
{
  "error": "Error message",
  "code": 400
}
```

Common HTTP status codes:
- `200` — Success
- `400` — Bad request (invalid input)
- `401` — Unauthorized (missing or invalid token)
- `403` — Forbidden (insufficient permissions)
- `404` — Not found
- `429` — Rate limited
- `500` — Internal server error

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEURAX_PORT` | Server port | `9098` |
| `NEURAX_DEBUG_NOAUTH` | Skip authentication | `false` |
| `NEURAX_SUPABASE_URL` | Supabase project URL | — |
| `NEURAX_SUPABASE_ANON_KEY` | Supabase anon key | — |
| `NEURAX_SUPABASE_JWT_SECRET` | JWT verification secret | — |
| `STRIPE_SECRET_KEY` | Stripe API secret key | — |
| `STRIPE_WEBHOOK_SECRET` | Stripe webhook signing secret | — |
| `NEURAX_ESSENTIAL_PRICE_ID` | Stripe price ID for Essential plan | — |
| `NEURAX_ARCHITECT_PRICE_ID` | Stripe price ID for Architect plan | — |
| `NEURAX_ELITE_PRICE_ID` | Stripe price ID for Elite plan | — |