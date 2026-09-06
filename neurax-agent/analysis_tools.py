"""The compiler-facing half of the agent's tool vocabulary.

`neurax-mcp` already exposes read/analysis operations over MCP for
external clients (Claude Desktop, etc.), but `neurax-agent`'s own LLM loop
(`agent_graph.py`) never called any of them — it only ever emitted
canvas-mutation tool calls, with budget-checking happening as deterministic
glue between planning attempts rather than something the model itself could
decide to do. This module is what closes that gap: those operations,
callable from the step-by-step loop.

Three of them (`analyze_architecture`, `check_budget`,
`find_optimal_hyperparameters`) are thin wrappers around
`budget_check.py`'s already-tested `measure_and_check`/
`optimize_hyperparameters` — real, working code this file reuses rather than
reimplements. The others have no existing neurax-agent equivalent, so
they call `neurax-service` directly, formatted the same way
`neurax-mcp/neurax_mcp/server.py`'s tool handlers already do (that
formatting is good and proven; only the plumbing around it changes here).

No shared package was introduced for this: `neurax-mcp` and `neurax-agent`
install and deploy independently, and `budget_check.py`'s own
`LAYER_TYPE_MAP` is already the more complete of the two that exist in this
codebase (see its own module comment) — `spec_to_topology` reuses that one,
so this module has nothing left to duplicate. `neurax-mcp` is unaffected by
any of this; it keeps its own copy for its own external callers.

Every function returns a plain string: the same shape a `tool_result` needs
to carry back into `agent_graph.py`'s history for the next `plan_step` call
to read, matching the plain-text results `run_controller_step`'s prompt
already expects for its tool-execution feedback.

Routing: these calls hit `neurax-service`'s root endpoints (`/analyze`,
`/sweep`, ...) — the same ones `budget_check.py` already calls — not the
`/agent/*` API-key-scoped prefix. `neurax-agent` has no service-level API
key or credential of its own anywhere in this codebase today (verified by
grep before writing this file); routing through `/agent/*` without one to
present would just fail every call. Moving these calls to `/agent/*` is a
real, still-open improvement, gated on `neurax-agent` actually being issued
a service credential — not something to fake with an unused env var now.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import httpx

from budget_check import measure_and_check, optimize_hyperparameters, spec_to_topology
from requirements import DeploymentBudget
from topology_validator import ArchSpec

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] [NEURAX-ANALYSIS] %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

NEURAX_SERVICE_URL = os.environ.get("NEURAX_SERVICE_URL", "http://127.0.0.1:9098")

#: Names an `execute_tool` dispatch (agent_graph.py) checks against before
#: falling back to snapshot mutation — the single source of truth for "is
#: this an analysis tool" so that list can't silently drift from what's
#: actually implemented below.
ANALYSIS_TOOL_NAMES = frozenset({
    "analyze_architecture",
    "check_budget",
    "find_optimal_hyperparameters",
    "get_hardware_list",
    "get_presets",
    "get_preset",
    "estimate_training_cost",
    "get_compliance_config",
})


def snapshot_to_spec(snapshot: dict[str, Any]) -> ArchSpec:
    """The one place a raw canvas snapshot (the shape `_apply_tool_to_snapshot`
    reads and writes) becomes an `ArchSpec` (the shape `spec_to_topology`
    needs). `ArchSpec.from_dict` already accepts exactly this: `nodes` with
    `id`/`type`/`params`, `edges` reading either `from`/`to` or
    `from_id`/`to_id` — the snapshot's own `connections` list uses the
    former, which `from_dict` already handles."""
    return ArchSpec.from_dict({
        "family": snapshot.get("family", ""),
        "nodes": snapshot.get("nodes") or [],
        "edges": snapshot.get("connections") or [],
        "hw_config": snapshot.get("hw_config") or {},
    })


def format_size(num_bytes: float) -> str:
    if not num_bytes:
        return "0 B"
    for unit, scale in (("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024)):
        if num_bytes >= scale:
            return f"{num_bytes / scale:.2f} {unit} ({num_bytes:,.0f} bytes)"
    return f"{num_bytes:,.0f} bytes"


def _format_metrics_block(metrics: dict[str, Any]) -> str:
    """Ported from `neurax-mcp`'s `_format_metrics`, adapted to take the
    metrics dict directly — `measure_and_check` already unwraps
    `report.metrics` from the raw HTTP response, so there is no
    `results.get("metrics", ...)` layer left to peel here."""
    if not metrics:
        return "(no metrics returned)"

    lines = [f"- Total Parameters: {metrics.get('total_parameters', 0):,}"]
    lines.append(f"- Model Size: {format_size(metrics.get('parameter_memory_bytes', 0))}")
    lines.append(f"- Total FLOPs: {metrics.get('total_flops', 0):.2e}")
    lines.append(f"- Peak VRAM: {format_size(metrics.get('peak_vram_bytes', 0))}")

    latency = metrics.get("latency_ms", 0)
    if latency:
        lines.append(f"- Estimated Latency: {latency:.2f} ms")
    throughput = metrics.get("throughput_tokens_per_s", 0)
    if throughput:
        lines.append(f"- Throughput: {throughput:.2f} tok/s")

    training = [
        ("training_cost_usd", "Training Cost", "${:,.2f}"),
        ("training_time_hours", "Training Time", "{:,.2f} hours"),
        ("energy_kwh", "Energy", "{:,.2f} kWh"),
        ("co2_kg", "CO2", "{:,.2f} kg"),
    ]
    reported = [
        (label, fmt.format(metrics[key]))
        for key, label, fmt in training
        if metrics.get(key, 0)
    ]
    if reported:
        for label, value in reported:
            lines.append(f"- {label}: {value}")
    else:
        lines.append("- Training cost: not applicable (no training budget given)")

    return "\n".join(lines)


async def analyze_architecture(snapshot: dict[str, Any]) -> str:
    """Compile the current canvas and report its real metrics — no budget,
    just "what does this design actually cost". Reuses `measure_and_check`
    with an empty `DeploymentBudget` so the only real work (compiling, HTTP,
    error handling) happens in one already-tested place."""
    spec = snapshot_to_spec(snapshot)
    report = await measure_and_check(spec, DeploymentBudget(), snapshot.get("hw_config"))

    if report.error:
        return f"Could not analyze the design: {report.error}"

    lines = ["NEURAX Analysis Results:", _format_metrics_block(report.metrics)]
    blocking = report.blocking_diagnostics()
    if blocking:
        lines.append("")
        lines.append("Compiler diagnostics (this design will not run as-is):")
        for d in blocking[:5]:
            lines.append(f"  - {d.get('message', '')}")
    notes = report.notes_text()
    if notes:
        lines.append("")
        lines.append(notes)
    return "\n".join(lines)


async def check_budget(
    snapshot: dict[str, Any],
    max_size_mb: Optional[float] = None,
    max_vram_gb: Optional[float] = None,
    max_latency_ms: Optional[float] = None,
    max_parameters: Optional[float] = None,
) -> str:
    """Compile the current canvas and check it against explicit limits —
    `measure_and_check`'s own `.summary()` already formats this exactly
    right (per-check PASS/FAIL with headroom, a verdict line); nothing here
    re-derives that."""
    spec = snapshot_to_spec(snapshot)
    budget = DeploymentBudget(
        max_size_bytes=(max_size_mb * 1024 ** 2) if max_size_mb is not None else None,
        max_vram_bytes=(max_vram_gb * 1024 ** 3) if max_vram_gb is not None else None,
        max_latency_ms=max_latency_ms,
        max_parameters=max_parameters,
    )
    report = await measure_and_check(spec, budget, snapshot.get("hw_config"))
    return report.summary()


async def find_optimal_hyperparameters(
    snapshot: dict[str, Any],
    objective: str = "max_throughput",
    candidates: Optional[dict[str, Any]] = None,
) -> str:
    """Search batch_size x zero_stage x gpu_count x precision for the best
    feasible training configuration — real compiler analyses, no training,
    no GPU. Reuses `optimize_hyperparameters`; `.summary()` already formats
    the result."""
    spec = snapshot_to_spec(snapshot)
    report = await optimize_hyperparameters(
        spec, snapshot.get("hw_config"), objective=objective, candidates=candidates
    )
    return report.summary()


async def _get(path: str, timeout_s: float = 30.0) -> Any:
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        response = await client.get(f"{NEURAX_SERVICE_URL}{path}")
        response.raise_for_status()
        return response.json()


async def get_hardware_list() -> str:
    try:
        result = await _get("/hardware")
    except httpx.HTTPStatusError as e:
        return f"Backend error ({e.response.status_code}): {e.response.text[:500]}"
    except httpx.RequestError:
        return f"Cannot reach NEURAX backend at {NEURAX_SERVICE_URL}. Is the server running?"

    gpus = result if isinstance(result, list) else []
    lines = ["Available NEURAX Hardware:", ""]
    for gpu in gpus:
        lines.append(
            f"- {gpu.get('name', 'Unknown')}: {gpu.get('memory_gb', '?')}GB, "
            f"{gpu.get('tflops_fp16', '?')} TFLOPs (FP16), "
            f"{gpu.get('memory_bandwidth_gbs', '?')} GB/s"
        )
    return "\n".join(lines)


async def get_presets() -> str:
    try:
        presets = await _get("/presets")
    except httpx.HTTPStatusError as e:
        return f"Backend error ({e.response.status_code}): {e.response.text[:500]}"
    except httpx.RequestError:
        return f"Cannot reach NEURAX backend at {NEURAX_SERVICE_URL}. Is the server running?"

    lines = ["Available Architecture Presets:", ""]
    for p in presets or []:
        desc = str(p.get("description", ""))[:100]
        lines.append(f"- {p.get('id', '?')}: {p.get('name', '?')} ({p.get('family', '?')}) — {desc}")
    return "\n".join(lines)


async def get_preset(preset_id: str) -> str:
    try:
        result = await _get(f"/presets/{preset_id}")
        return json.dumps(result, indent=2)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return f"Preset '{preset_id}' not found."
        return f"Backend error ({e.response.status_code}): {e.response.text[:500]}"
    except httpx.RequestError:
        return f"Cannot reach NEURAX backend at {NEURAX_SERVICE_URL}. Is the server running?"


#: Rough hourly on-demand cost, USD, per GPU — a fallback estimate only;
#: `find_optimal_hyperparameters`'s real /sweep call is the authoritative
#: cost figure whenever a canvas/topology is available. This is for a bare
#: "parameters + tokens" question with no architecture on the canvas at all.
_GPU_HOURLY_COST = {"A100": 2.5, "H100": 4.0, "V100": 1.5, "T4": 0.5, "A10G": 1.0}


async def estimate_training_cost(
    parameters: float,
    tokens: float,
    gpu_type: str = "A100",
    gpu_count: int = 1,
    hours: Optional[float] = None,
) -> str:
    hourly = _GPU_HOURLY_COST.get(gpu_type, 2.0)
    if hours is None:
        flops = parameters * tokens * 2
        hours = flops / (gpu_count * 300e12 * 3600)
    cost = hours * gpu_count * hourly

    return (
        "Training Cost Estimate:\n"
        f"- Model Size: {parameters / 1e9:.2f}B parameters\n"
        f"- Training Tokens: {tokens / 1e9:.2f}B tokens\n"
        f"- GPU: {gpu_type} x {gpu_count}\n"
        f"- Estimated Hours: {hours:.2f} hours\n"
        f"- Hourly Cost: ${hourly:.2f}/GPU-hour\n"
        f"- Total Cost: ${cost:.2f}"
    )


async def get_compliance_config() -> str:
    try:
        result = await _get("/compliance/config")
        return json.dumps(result, indent=2)
    except httpx.HTTPStatusError as e:
        return f"Backend error ({e.response.status_code}): {e.response.text[:500]}"
    except httpx.RequestError:
        return f"Cannot reach NEURAX backend at {NEURAX_SERVICE_URL}. Is the server running?"





async def dispatch(name: str, args: dict[str, Any], snapshot: dict[str, Any]) -> str:
    """Single entry point `agent_graph.py::execute_tool` calls for any name
    in `ANALYSIS_TOOL_NAMES` — keeps the argument-unpacking for all 11 tools
    in one place instead of spread across the graph module."""
    if name == "analyze_architecture":
        return await analyze_architecture(snapshot)
    if name == "check_budget":
        return await check_budget(
            snapshot,
            max_size_mb=args.get("max_size_mb"),
            max_vram_gb=args.get("max_vram_gb"),
            max_latency_ms=args.get("max_latency_ms"),
            max_parameters=args.get("max_parameters"),
        )
    if name == "find_optimal_hyperparameters":
        return await find_optimal_hyperparameters(
            snapshot,
            objective=args.get("objective", "max_throughput"),
            candidates=args.get("candidates"),
        )
    if name == "get_hardware_list":
        return await get_hardware_list()
    if name == "get_presets":
        return await get_presets()
    if name == "get_preset":
        return await get_preset(str(args.get("preset_id", "")))
    if name == "estimate_training_cost":
        return await estimate_training_cost(
            parameters=float(args.get("parameters", 0)),
            tokens=float(args.get("tokens", 0)),
            gpu_type=str(args.get("gpu_type", "A100")),
            gpu_count=int(args.get("gpu_count", 1)),
            hours=args.get("hours"),
        )
    if name == "get_compliance_config":
        return await get_compliance_config()
    raise ValueError(f"Unknown analysis tool: {name}")
