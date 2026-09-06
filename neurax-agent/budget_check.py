"""Measure a planned architecture and check it against the client's budget.

Structural validation answers *is this a well-formed graph*. It cannot answer
*is this the model the client asked for*: a topology can be perfectly valid and
still be fifty times too large for a phone. That question is settled by compiling
the design and comparing the result against the stated limits, which is what this
module does.

When a design misses, the report names the measurement and the gap so the next
planning pass has something concrete to work from rather than being told to "try
smaller".
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from requirements import DeploymentBudget

logger = logging.getLogger(__name__)

NEURAX_SERVICE_URL = os.environ.get("NEURAX_SERVICE_URL", "http://127.0.0.1:9098")

# Canvas block names translated to the layer kinds the compiler understands.
#
# The compiler rejects an unknown `layer_type` outright, so a block missing from
# this table does not merely go uncounted — it fails the whole analysis with a
# 400. Every block the catalogue offers therefore needs an entry.
#
# Three kinds of mapping appear here:
#   * a true equivalent, where the compiler models the same operator;
#   * `custom`, for operators the compiler has no built-in formula for (graph
#     convolutions, spiking neurons) — the established path in this codebase,
#     and the one its own GNN tests use, since a custom layer can carry its cost
#     as an equation;
#   * `custom` again for parameter-free element-wise steps (activations,
#     dropout, reshapes), which contribute no weights and whose cost is folded
#     into the layer they follow.
LAYER_TYPE_MAP = {
    # ── Sequence / attention ──────────────────────────────────────────────
    "token_embedding": "embedding",
    "embedding": "embedding",
    "pos_embed": "positional_embed",
    "positional_encoding": "positional_embed",
    "rope": "positional_embed",
    "alibi": "positional_embed",
    "mha_attention": "attention",
    "mha": "attention",
    "attention": "attention",
    "gqa": "attention",
    "flash_attention": "attention",
    "self_attention": "attention",
    "cross_attention": "cross_attention",
    "bahdanau_attention": "attention",
    "ffn_standard": "mlp",
    "ffn_gated": "mlp",
    "ffn": "mlp",
    "mlp": "mlp",
    "swiglu": "mlp",
    "lm_head": "dense",
    "classification_head": "dense",
    "linear": "dense",
    "linear_projection": "dense",
    "dense": "dense",
    "output": "dense",
    "input": "embedding",

    # ── Normalisation (carries scale/shift parameters) ────────────────────
    "layer_norm": "normalization",
    "layernorm": "normalization",
    "rmsnorm": "normalization",
    "batchnorm": "normalization",
    "groupnorm": "normalization",
    "instancenorm": "normalization",
    "graphnorm": "normalization",
    "pixelnorm": "pixel_norm",
    "spectral_norm": "spectral_norm",

    # ── Convolution / pooling ─────────────────────────────────────────────
    "conv2d": "conv",
    "conv1d": "conv",
    "depthwise_conv2d": "conv",
    "conv_transpose2d": "conv",
    "max_pool": "pooling",
    "avg_pool": "pooling",
    "global_pool": "pooling",
    "global_mean_pool": "pooling",
    "global_max_pool": "pooling",
    "global_add_pool": "pooling",
    "downsample": "pooling",
    "upsample": "pooling",
    "residual_block": "residual_block",
    "se_block": "custom",

    # ── Mixture of experts ────────────────────────────────────────────────
    "moe_block": "moe",
    "expert": "moe",
    "gate": "custom",
    "router_softmax": "custom",
    "expert_combine": "custom",

    # ── State space ───────────────────────────────────────────────────────
    "mamba_block": "mamba_block",
    "s4_block": "s4_block",

    # ── Recurrent ─────────────────────────────────────────────────────────
    "lstm": "lstm",
    "gru": "gru",
    "bilstm": "bilstm",
    "bigru": "bigru",

    # ── Generative ────────────────────────────────────────────────────────
    "timestep_embedding": "timestep_embedding",
    "unet_block": "unet_block",
    "noise_scheduler": "custom",

    # ── Graph and spiking: no built-in formula, carried as custom ─────────
    "gcn_conv": "custom",
    "gat_conv": "custom",
    "sage_conv": "custom",
    "edge_conv": "custom",
    "lif_neuron": "custom",
    "leaky_neuron": "custom",
    "synaptic_layer": "custom",
    "rate_encoder": "custom",
    "latency_encoder": "custom",

    # ── Parameter-free element-wise steps ─────────────────────────────────
    "relu": "custom",
    "gelu": "custom",
    "silu": "custom",
    "tanh": "custom",
    "sigmoid": "custom",
    "leaky_relu": "custom",
    "dropout": "custom",
    "flatten": "custom",
    "add": "residual",
    "residual": "residual",
    "concat": "custom",
    "merge": "custom",
    "custom": "custom",
}


@dataclass
class BudgetCheck:
    """Outcome of comparing one measurement against one limit."""

    label: str
    measured: float
    limit: float
    unit: str
    fits: bool

    def describe(self) -> str:
        verdict = "PASS" if self.fits else "FAIL"
        gap = self.limit - self.measured
        relation = "headroom" if self.fits else "over by"
        return (
            f"[{verdict}] {self.label}: {self.measured:,.4g} {self.unit} "
            f"against a limit of {self.limit:,.4g} {self.unit} "
            f"({relation} {abs(gap):,.4g} {self.unit})"
        )


@dataclass
class BudgetReport:
    """What the compiler measured, and whether it satisfies the client."""

    fits: bool
    checks: list[BudgetCheck] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    #: Diagnostics the compiler emitted about the design itself, as opposed to
    #: whether it meets the client's budget. A model can fit a size limit
    #: perfectly and still be one the compiler says will not start.
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def blocking_diagnostics(self) -> list[dict[str, Any]]:
        """Diagnostics that describe a design which will not work.

        Warnings about, say, an unusual head count are information. A peak VRAM
        that exceeds the target GPU is a design that cannot run, and no budget
        check would catch it — the model may be well under the size limit and
        still never start.

        W007 (a layer's declared input shape disagreeing with what the layer
        before it produces) is included here even though the compiler reports
        it at "warning" severity — it stays a warning there because a linear
        list is all the compiler ever sees, so it cannot always tell a real
        mismatch from a merge point it has no representation for. But on this
        side, a design an agent just planned has no legitimate reason to
        disagree with itself layer-to-layer, so it is treated as blocking:
        this is what lets the planner catch and fix a mismatched hidden size
        before the client ever sees it, instead of only after materializing.
        """
        return [
            d
            for d in self.diagnostics
            if str(d.get("severity", "")).lower() in {"critical", "error"}
            or str(d.get("code", "")).upper() == "W007"
        ]

    def notable_diagnostics(self) -> list[dict[str, Any]]:
        """Non-blocking observations worth surfacing to the client.

        `blocking_diagnostics()` feeds the *planner* — only things that stop a
        design from working. This is for the *client*: real, already-computed
        signals (GQA/MoE detected, no LR warmup, learning_rate past the
        Lipschitz-based stability estimate, tokens-per-parameter far from
        Chinchilla-optimal) that were reaching `self.diagnostics` but nothing
        ever read past `blocking_diagnostics()`'s filter, so a design could
        clear every budget check and still never mention e.g. "your
        learning_rate is above the estimated stability bound" even though the
        compiler had already said so.
        """
        return [
            d
            for d in self.diagnostics
            if str(d.get("severity", "")).lower() in {"hint", "info"}
        ]

    def notes_text(self) -> str:
        """Just the formatted hint/info block, or "" when there is none —
        shared by `summary()` and by a caller that wants to mention these
        even when no budget was stated at all (in which case `summary()`'s
        own "No budget stated" line would be noise).
        """
        notes = self.notable_diagnostics()
        if not notes:
            return ""
        lines = ["Notes:"]
        for d in notes:
            message = str(d.get("message", "")).strip()
            if not message:
                continue
            suggestion = str(d.get("suggestion") or "").strip()
            lines.append(f"  - {message}" + (f" {suggestion}" if suggestion else ""))
        return "\n".join(lines)

    def summary(self) -> str:
        if self.error:
            return f"Could not measure the design: {self.error}"
        if not self.checks:
            lines = ["No budget stated; the design was measured but not constrained."]
        else:
            lines = [check.describe() for check in self.checks]
            lines.append(
                "The design fits every stated budget."
                if self.fits
                else "The design exceeds at least one budget."
            )

        notes = self.notes_text()
        if notes:
            lines.append("")
            lines.append(notes)

        return "\n".join(lines)

    def planner_feedback(self) -> list[str]:
        """Concrete instructions for the next planning pass.

        Phrased as measurements and required reductions, because a planner told
        only that it "failed" tends to make an arbitrary change; told that the
        model is 12.4 MB against a 1 MB limit, it can size the fix.
        """
        messages: list[str] = []

        # Blocking diagnostics are reported whether or not the budget was met.
        # Returning early on `fits` used to hide them completely: a design that
        # came in under the size limit was delivered without anyone mentioning
        # that the compiler said it would not run.
        for diagnostic in self.blocking_diagnostics():
            message = str(diagnostic.get("message", "")).strip()
            if not message:
                continue
            suggestion = str(diagnostic.get("suggestion") or "").strip()
            code = str(diagnostic.get("code") or "").strip()
            prefix = f"[{code}] " if code else ""
            messages.append(
                f"{prefix}The compiler rejects this design: {message}"
                + (f" {suggestion}" if suggestion else "")
            )

        if self.fits or self.error:
            return messages

        for check in self.checks:
            if check.fits:
                continue
            factor = check.measured / check.limit if check.limit else float("inf")
            messages.append(
                f"{check.label} is {check.measured:,.4g} {check.unit} but must be at most "
                f"{check.limit:,.4g} {check.unit} — roughly {factor:.1f}x too large. "
                f"Reduce it by shrinking hidden/embedding dimensions (parameters grow with "
                f"the square of hidden size), cutting the number of repeated blocks, "
                f"reducing the vocabulary, or narrowing the dtype."
            )
        return messages


#: Image-shaped families ([batch, channels, height, width]) — the compiler
#: reads their entry shape from `data.image_*`, the same fields
#: The canvas's own hw_config already carries these for these families.
_IMAGE_SHAPE_FAMILIES = {"cnn", "vit", "gan", "diffusion"}


def spec_to_topology(
    spec: Any,
    hw_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate a planned ArchSpec into the compiler's model config."""
    # The design's own settings win over the panel's: the agent chose them for
    # this model, so they are what must be measured. Measuring the panel's
    # precision while the agent asked for int8 would check a different model.
    hw = {**(hw_config or {}), **(getattr(spec, "hw_config", None) or {})}
    precision = hw.get("precision") or "fp16"
    family = str(getattr(spec, "family", "") or "").lower()

    # No shape guessing here anymore — send the real graph (`connections`,
    # matching the wire field the compiler now understands) and real
    # per-node params, and let the compiler's own shape-inference engine
    # (conv/pool arithmetic, hidden-dim threading, ...) do what it already
    # does for the frontend. This used to hand-thread a hidden dimension
    # through sequence-family nodes only, because at the time the compiler
    # had no graph to walk and no shape engine of its own for any family —
    # both now exist, and this pre-flight check gets the same real answer
    # the frontend would compute post-materialization, before that instead
    # of after it.
    layers = []
    for node in spec.nodes:
        raw_type = str(getattr(node, "type", "")).lower()
        params = dict(getattr(node, "params", {}) or {})
        layer = {
            "id": getattr(node, "id", f"layer_{len(layers)}"),
            "layer_type": LAYER_TYPE_MAP.get(raw_type, raw_type),
            "params": params,
        }
        equations = dict(getattr(node, "custom_equations", {}) or {})
        if equations:
            layer["custom_equations"] = equations
        layers.append(layer)

    connections = [
        {"from": getattr(edge, "from_id", ""), "to": getattr(edge, "to_id", "")}
        for edge in getattr(spec, "edges", None) or []
    ]

    global_params: dict[str, Any] = {}
    for key, source in (
        ("hidden_size", "hiddenDim"),
        ("sequence_length", "seqLen"),
        ("vocab_size", "vocabSize"),
        ("num_heads", "numHeads"),
    ):
        value = hw.get(source)
        if value:
            global_params[key] = value

    data: dict[str, Any] = {"dtype": precision}
    if family in _IMAGE_SHAPE_FAMILIES:
        for key, source in (
            ("image_channels", "inChannels"),
            ("image_height", "imgHeight"),
            ("image_width", "imgWidth"),
        ):
            value = hw.get(source)
            if value:
                data[key] = value

    return {
        "schema_version": "1.0.0",
        "model": {
            "name": "AgentDesign",
            "type": spec.family or "transformer",
            "global_params": global_params,
            "layers": layers,
            "connections": connections,
        },
        "training": {
            "batch_size": hw.get("batchSize") or 1,
            "precision": precision,
        },
        "hardware": {
            "gpus": [
                {
                    "name": hw.get("hardware") or "T4",
                    "count": hw.get("gpuCount") or 1,
                }
            ]
        },
        "data": data,
    }


async def measure_and_check(
    spec: Any,
    budget: DeploymentBudget,
    hw_config: dict[str, Any] | None = None,
    timeout_s: float = 30.0,
) -> BudgetReport:
    """Compile the design and report whether it satisfies the budget."""
    topology = spec_to_topology(spec, hw_config)

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(
                f"{NEURAX_SERVICE_URL}/analyze", json={"topology": topology}
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPStatusError as exc:
        # The compiler explains *why* it rejected a design in the body; without
        # it the caller sees a bare "400" and cannot act on it.
        detail = exc.response.text.strip() or str(exc)
        logger.warning("compiler rejected the design: %s", detail)
        return BudgetReport(fits=True, error=detail)
    except Exception as exc:  # network, timeout, ...
        logger.warning("budget check could not reach the compiler: %s", exc)
        return BudgetReport(fits=True, error=str(exc))

    report = payload.get("report", payload)
    metrics = report.get("metrics", {})
    # The compiler's own verdict on the design. Read here rather than discarded,
    # because the planner needs it: a design can satisfy every stated budget and
    # still be one the compiler reports as unable to start.
    diagnostics = report.get("diagnostics", []) or []
    if not metrics:
        return BudgetReport(
            fits=True, error="analysis returned no metrics", diagnostics=diagnostics
        )

    checks: list[BudgetCheck] = []

    def add(label: str, measured: float, limit: Optional[float], unit: str, scale: float = 1.0):
        if limit is None:
            return
        checks.append(
            BudgetCheck(
                label=label,
                measured=measured / scale,
                limit=limit / scale,
                unit=unit,
                fits=measured <= limit,
            )
        )

    add("Model size", metrics.get("parameter_memory_bytes", 0),
        budget.max_size_bytes, "MB", 1024 ** 2)
    add("Parameters", metrics.get("total_parameters", 0),
        budget.max_parameters, "params")
    add("Latency", metrics.get("latency_ms", 0),
        budget.max_latency_ms, "ms")
    add("Peak VRAM", metrics.get("peak_vram_bytes", 0),
        budget.max_vram_bytes, "GB", 1024 ** 3)

    return BudgetReport(
        fits=all(check.fits for check in checks),
        checks=checks,
        metrics=metrics,
        diagnostics=diagnostics,
    )


#: Storage widths, narrowest last — the order the loop tries them in.
_PRECISION_LADDER = ["fp32", "bf16", "fp16", "int8"]

#: Bytes per parameter for each storage width.
_PRECISION_BYTES = {"fp32": 4, "float32": 4, "bf16": 2, "bfloat16": 2,
                    "fp16": 2, "float16": 2, "fp8": 1, "int8": 1}


def suggest_precision(current: str, overshoot_factor: float) -> Optional[str]:
    """Narrower dtype that could absorb an overshoot, if one exists.

    Precision is the cheapest lever available: it changes storage width without
    touching the architecture, and fp32 to int8 is a factor of four. It is only
    worth proposing when it can plausibly close the gap on its own.
    """
    current_bytes = _PRECISION_BYTES.get((current or "").lower())
    if current_bytes is None:
        return None

    for candidate in _PRECISION_LADDER:
        candidate_bytes = _PRECISION_BYTES[candidate]
        if candidate_bytes >= current_bytes:
            continue
        if current_bytes / candidate_bytes >= overshoot_factor:
            return candidate
    return None


async def narrow_precision_to_fit(
    spec: Any,
    budget: DeploymentBudget,
    hw_config: dict[str, Any] | None = None,
    report: Optional[BudgetReport] = None,
) -> tuple[Optional[str], Optional[BudgetReport]]:
    """Try narrower storage widths until the design fits, or run out of them.

    Storage width is the one lever that costs nothing to pull: it changes how
    weights are stored without touching the architecture the client described,
    and fp32 to int8 divides model size by four. Applying it here — measuring
    each step rather than predicting it — settles the question without spending
    a planning attempt on a model that may or may not follow the instruction.

    Returns the precision that worked and its measurement, or ``(None, None)``
    when no available width is enough and the architecture itself must shrink.
    """
    if report is not None and report.fits:
        return None, None

    current = (
        (getattr(spec, "hw_config", None) or {}).get("precision")
        or (hw_config or {}).get("precision")
        or "fp16"
    )
    current_bytes = _PRECISION_BYTES.get(current.lower())
    if current_bytes is None:
        return None, None

    original = dict(getattr(spec, "hw_config", None) or {})

    for candidate in _PRECISION_LADDER:
        if _PRECISION_BYTES[candidate] >= current_bytes:
            continue

        spec.hw_config = {**original, "precision": candidate}
        candidate_report = await measure_and_check(spec, budget, hw_config)

        if candidate_report.error:
            break
        if candidate_report.fits:
            logger.info("narrowed precision %s -> %s to meet the budget", current, candidate)
            return candidate, candidate_report

    # Nothing worked; leave the design as the planner wrote it.
    spec.hw_config = original
    return None, None


@dataclass
class SweepReport:
    """Outcome of asking NEURAX for the fastest/cheapest feasible training
    configuration for a design, via the /sweep endpoint (see
    neurax_core::sweep). Mirrors BudgetReport's shape deliberately — same
    module, same "measure, don't guess" discipline, same failure handling.
    """

    best: Optional[dict[str, Any]] = None
    points_evaluated: int = 0
    feasible_count: int = 0
    error: Optional[str] = None

    def summary(self) -> str:
        if self.error:
            return f"Could not search hyperparameters: {self.error}"
        if self.best is None:
            return (
                f"No feasible training configuration found among "
                f"{self.points_evaluated} candidates evaluated — this design "
                "doesn't fit the configured GPU's VRAM at any swept batch size."
            )
        b = self.best
        return (
            f"Searched {self.points_evaluated} configurations "
            f"({self.feasible_count} feasible). Best: batch_size={b.get('batch_size')}, "
            f"zero_stage={b.get('zero_stage')}, precision={b.get('precision')} — "
            f"{b.get('throughput_tokens_per_s', 0):,.0f} tok/s, "
            f"${b.get('training_cost_usd', 0):,.2f}, "
            f"{b.get('peak_vram_gb', 0):.2f} GB peak VRAM."
        )


async def optimize_hyperparameters(
    spec: Any,
    hw_config: dict[str, Any] | None = None,
    objective: str = "max_throughput",
    candidates: dict[str, Any] | None = None,
    timeout_s: float = 30.0,
) -> SweepReport:
    """Search batch_size x zero_stage x precision for the design's best
    feasible training configuration. Reuses spec_to_topology exactly like
    measure_and_check — the sweep endpoint takes the same wire topology, it
    just evaluates many hyperparameter combinations against it instead of one.

    `candidates` (batch_sizes/zero_stages/gpu_counts/precisions) narrows the
    sweep to specific values instead of the backend's own default range —
    added for `analysis_tools.find_optimal_hyperparameters`, whose MCP
    counterpart already exposed this; every existing caller leaves it unset
    and gets exactly the previous behavior.
    """
    topology = spec_to_topology(spec, hw_config)
    payload: dict[str, Any] = {"topology": topology, "objective": objective}
    if candidates:
        payload["candidates"] = candidates

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(
                f"{NEURAX_SERVICE_URL}/sweep",
                json=payload,
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text.strip() or str(exc)
        logger.warning("sweep rejected the design: %s", detail)
        return SweepReport(error=detail)
    except Exception as exc:  # network, timeout, ...
        logger.warning("hyperparameter sweep could not reach the compiler: %s", exc)
        return SweepReport(error=str(exc))

    result = payload.get("result", payload)
    points = result.get("points", []) or []
    return SweepReport(
        best=result.get("best"),
        points_evaluated=len(points),
        feasible_count=sum(1 for p in points if p.get("feasible")),
    )
