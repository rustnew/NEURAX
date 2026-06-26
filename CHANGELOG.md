# NEURAX Changelog

All notable changes to the NEURAX project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.5.0] - 2026-06-26

### Added

#### Phase 1: Inference Intelligence
- Inference simulation dialect (`neurax-ir/src/inference/`) with 10 compute methods
- `POST /inference/simulate` endpoint for inference behavior prediction
- Frontend `InferenceIntelligence` component with `BehaviorDashboard` (10 widgets)
- Frontend `InferenceControls` component for simulation parameter configuration
- 11 unit tests for inference pass

#### Phase 2: Streaming SSE
- `AnalysisEventEmitter` trait and `BroadcastEmitter` in `neurax-core/src/streaming.rs`
- `run_analysis_streaming()` with progressive event emission for all 11 pipeline phases
- `POST /analyze/stream` endpoint — starts streaming analysis job, returns job_id
- `GET /analyze/stream/{job_id}` — SSE endpoint for real-time analysis progress
- `GET /analyze/result/{job_id}` — retrieve final analysis result
- `GET /analyze/status/{job_id}` — check job status
- Frontend `analyzeStream()` API client with EventSource support
- Frontend `handleRunAnalysisStream()` with progressive compilation updates
- Automatic fallback from SSE to synchronous analysis on error

#### Phase 3: Comparison
- `POST /analyze/compare` endpoint for multi-hardware comparison (up to 8 configs)
- `CompareHardwareConfig`, `CompareRequest`, `CompareResultItem`, `CompareResponse` types
- `HardwareDetailEntry` with full GPU specs (20 GPUs from HardwareDatabase)
- Frontend `compareAnalyses()` API client
- Frontend `ComparisonCharts` component with:
  - Config editor (add/remove hardware configs)
  - GPU selector with full spec database
  - Results comparison table (params, FLOPs, VRAM, latency, throughput, cost, CO₂)
  - Visual comparison bars
  - Current config highlighting

#### Phase 4: Cloud Projects
- `GET /projects` — list user projects
- `POST /projects` — create project (max 50 per user)
- `GET /projects/{id}` — get single project
- `PUT /projects/{id}` — update project (partial)
- `DELETE /projects/{id}` — delete project
- `Project`, `CreateProjectRequest`, `UpdateProjectRequest` types
- Frontend project management in `TopNav` (save, load, delete projects)
- Frontend `listProjects()`, `createProject()`, `getProject()`, `updateProject()`, `deleteProject()` API methods

#### Phase 5: ONNX Binary Export
- `neurax-core/src/export/onnx.rs` — full ONNX protobuf binary generation from IR
- Protobuf wire format encoding (varint, string, bytes, int64, float, repeated fields)
- 40+ NEURAX LayerType → ONNX op mappings
- Weight initializer generation with correct dimensions
- Input shape inference based on model type
- `POST /export/onnx` endpoint with base64-encoded response
- Frontend `exportOnnx()` API client
- Frontend `ExportPanel` ONNX download with loading state and success toast

#### Phase 6: Production
- `GET /credits` — user credit balance with plan-based limits
- `GET /compliance/config` — regulatory compliance data (EU AI Act, CSRD, DSA, US AI EO, Canada AIDA)
- `ComplianceRegulation`, `ComplianceThresholds`, `ComplianceConfig` types
- Frontend `ComplianceView` with real API data and fallback
- Frontend credit display in `AIChatDrawer`
- `Dockerfile` — multi-stage Rust build (debian:bookworm-slim)
- `Dockerfile.ui` — multi-stage Node build (nginx:alpine)
- `Dockerfile.agent` — Python 3.11-slim with uvicorn
- `docker-compose.yml` — 3-service orchestration with healthchecks
- `.dockerignore`

#### Phase 7: Tests & Documentation
- Created missing model JSON files: `gpt2_small.json`, `gpt2_medium.json`, `llama_8b.json`, `llama_70b.json`, `gpt4.json`
- Fixed all broken `include_str!` paths in neurax-core tests (was referencing `../../../../Neurax-IR/models/`)
- Fixed `read_to_string` paths for runtime file loading
- `API_REFERENCE.md` — complete API documentation for all 24 endpoints
- `DEPLOYMENT.md` — deployment guide with Docker, reverse proxy, scaling, troubleshooting
- `CHANGELOG.md` — this file

### Changed
- `HardwareDetail` interface updated with full GPU spec fields (was: name, peak_ops_per_s_fp16, mem_bw_gbps, vram_bytes; now: name, manufacturer, memory_gb, memory_bandwidth_gbs, tflops_fp64/32/16/bf16/int8/fp8, tensor_cores, nvlink, nvlink_bandwidth_gbs, tdp_watts, launch_year)
- `hardware_list()` endpoint now returns all 20 GPUs from HardwareDatabase (was: 5 hardcoded)
- `handleRunAnalysis` refactored to use `parseAnalysisReport()` helper (reduced ~420 lines of inline parsing)
- `SimulationWorkspace` now accepts `topology` prop for comparison feature
- `deriveHardwareProjections()` updated for new `HardwareDetail` field names
- CORS allowed methods expanded to include PUT and DELETE

### Fixed
- neurax-core test compilation errors (broken model file paths)
- TypeScript errors in `ComparisonCharts.tsx`, `Index.tsx`, `ExportPanel.tsx`

## [0.4.0] - 2026-06-19

### Added
- Initial web UI with React 18 + TypeScript + Vite
- Canvas-based model designer with drag-and-drop
- Architecture presets (Transformer, CNN, SSM, MoE, Diffusion, GNN, RNN, GAN)
- Hardware configuration panel
- Analysis dashboard with 35+ metrics
- AI Chat Drawer with LangChain agent integration
- Time Machine workspace for cost projection
- Stripe billing integration (checkout, portal, webhooks)
- Supabase authentication

## [0.3.0] - 2026-06-12

### Added
- 10 IR dialect pipeline (Architecture → Graph → Tensor → Operator → Compute → Memory → Parallelism → Hardware → Cost → Report)
- MLIR backend with 15 custom dialects via melior bindings
- CLI tool (analyze, compile, validate, summary commands)
- TUI (terminal UI) with ratatui
- Hardware database with 20 GPU specs
- Per-architecture FLOPs/parameter formulas
- 32 validation rules for JSON input

## [0.2.0] - 2026-06-05

### Added
- JSON parser with strongly-typed ModelConfig
- Core analysis pipeline
- Basic report generation

## [0.1.0] - 2026-05-29

### Added
- Initial project structure
- Workspace with 10 crates
- Basic CI setup