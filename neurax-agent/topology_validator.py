"""
Topology Validator — single authoritative constraint enforcement.

Validates a declarative ArchSpec before any tool calls are emitted.
All constraint logic lives here — no duplication across other modules.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Import shared merge-block set (single source of truth, no circular dependency)
from constants import MERGE_BLOCK_TYPES

CONSTRAINTS_FILE = Path(__file__).parent / "block_constraints.json"

# ── Lazy-loaded fanin limits from block_constraints.json ──────────────────────
_fanin_limits: dict[str, int] | None = None


def _load_fanin_limits() -> dict[str, int]:
    """Load max_inputs per block type from block_constraints.json (cached)."""
    global _fanin_limits
    if _fanin_limits is not None:
        return _fanin_limits

    _fanin_limits = {}
    if not CONSTRAINTS_FILE.exists():
        logger.warning(f"Constraints file not found: {CONSTRAINTS_FILE}")
        return _fanin_limits

    try:
        with open(CONSTRAINTS_FILE) as f:
            data = json.load(f)
        blocks = data.get("blocks", {})
        for btype, constraint in blocks.items():
            _fanin_limits[btype.lower()] = int(constraint.get("max_inputs", 1))
        logger.info(f"✅ Loaded fanin limits for {len(_fanin_limits)} block types")
    except Exception as e:
        logger.error(f"Failed to load fanin limits: {e}")

    return _fanin_limits


def get_max_inputs(block_type: str) -> int:
    """Return max inputs for a block type. -1 = unlimited, 1 = single input."""
    limits = _load_fanin_limits()
    return limits.get(block_type.lower(), 1)  # default: single input


def is_fanin_capable(block_type: str) -> bool:
    """Return True if this block can accept multiple inputs."""
    return get_max_inputs(block_type) == -1


# ── ArchSpec data structures ──────────────────────────────────────────────────

@dataclass
class ArchNode:
    id: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)
    #: Cost equations for a block the compiler has no built-in formula for.
    #: Graph convolutions, spiking neurons and genuinely novel operators all
    #: compile as custom layers, and a custom layer without equations counts as
    #: zero parameters — which the compiler rejects as an empty model.
    custom_equations: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArchEdge:
    from_id: str
    to_id: str


@dataclass
class ArchSpec:
    family: str
    nodes: list[ArchNode]
    edges: list[ArchEdge]
    rationale: str = ""
    #: Platform-level settings the design needs (precision, batch size, target
    #: GPU, sequence length, ...). Kept beside the graph because they change the
    #: model's cost as much as the blocks do — precision alone moves size 4x.
    hw_config: dict[str, Any] = field(default_factory=dict)
    #: Training hyperparameters (optimizer, warmup, schedule, parallelism, ...).
    hyperparams: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArchSpec":
        nodes = [
            ArchNode(
                id=str(n.get("id", "")),
                type=str(n.get("type", "")),
                params=n.get("params") if isinstance(n.get("params"), dict) else {},
                custom_equations=(
                    n.get("custom_equations")
                    if isinstance(n.get("custom_equations"), dict)
                    else {}
                ),
            )
            for n in (data.get("nodes") or [])
        ]
        edges = [
            ArchEdge(
                from_id=str(e.get("from") or e.get("from_id", "")),
                to_id=str(e.get("to") or e.get("to_id", "")),
            )
            for e in (data.get("edges") or [])
        ]
        return cls(
            family=str(data.get("family", "")),
            nodes=nodes,
            edges=edges,
            rationale=str(data.get("rationale", "")),
            hw_config=data.get("hw_config") if isinstance(data.get("hw_config"), dict) else {},
            hyperparams=data.get("hyperparams") if isinstance(data.get("hyperparams"), dict) else {},
        )


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ── Graph helpers ─────────────────────────────────────────────────────────────

def _build_adjacency(spec: ArchSpec) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Build forward (fwd) and reverse (rev) adjacency maps."""
    node_ids = {n.id for n in spec.nodes}
    fwd: dict[str, list[str]] = {n.id: [] for n in spec.nodes}
    rev: dict[str, list[str]] = {n.id: [] for n in spec.nodes}

    for edge in spec.edges:
        if edge.from_id in node_ids and edge.to_id in node_ids:
            fwd[edge.from_id].append(edge.to_id)
            rev[edge.to_id].append(edge.from_id)

    return fwd, rev


def _has_cycle(spec: ArchSpec) -> bool:
    """Kahn's algorithm — return True if there is a cycle."""
    _, rev = _build_adjacency(spec)
    fwd, _ = _build_adjacency(spec)
    in_degree = {n.id: len(rev[n.id]) for n in spec.nodes}
    queue = [nid for nid, d in in_degree.items() if d == 0]
    visited = 0
    while queue:
        cur = queue.pop()
        visited += 1
        for nxt in fwd.get(cur, []):
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)
    return visited != len(spec.nodes)


def _has_io_path(spec: ArchSpec) -> bool:
    """Return True if there exists at least one path from any input node to any output node."""
    fwd, _ = _build_adjacency(spec)
    input_ids = {n.id for n in spec.nodes if n.type == "input"}
    output_ids = {n.id for n in spec.nodes if n.type == "output"}

    if not input_ids or not output_ids:
        return False

    seen: set[str] = set()
    stack = list(input_ids)
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if cur in output_ids:
            return True
        for nxt in fwd.get(cur, []):
            if nxt not in seen:
                stack.append(nxt)
    return False


def _count_incoming(spec: ArchSpec) -> dict[str, int]:
    """Return incoming edge count per node id."""
    counts: dict[str, int] = {n.id: 0 for n in spec.nodes}
    for edge in spec.edges:
        if edge.to_id in counts:
            counts[edge.to_id] += 1
    return counts


# ── Deterministic fan-in repair ─────────────────────────────────────────────────
#
# Structural merge blocks, ranked by how safe they are to insert without being
# asked: a plain sum or concatenation changes nothing about what the branches
# compute. `lm_head`, `gate`, `moe_block` and the rest of MERGE_BLOCK_TYPES are
# fan-in capable too, but each *is* a specific operation with its own meaning —
# inserting one where the design never asked for it would silently change what
# the model does, not just how its edges are drawn.
_AUTO_REPAIR_MERGE_PRIORITY: tuple[str, ...] = (
    "residual_add", "add", "concat", "merge", "skip_connection",
)


def auto_repair_fanin_violations(
    spec: ArchSpec, catalogue: list[dict[str, Any]]
) -> Optional[ArchSpec]:
    """Insert a merge node ahead of any single-input node fed by more than one edge.

    A multi-task or multimodal design routinely produces more than one branch
    that needs to land on one downstream node — most often several heads
    converging on `output`. The planner is told how to avoid this (the fan-in
    error names the exact fix), but a model that gets it wrong once tends to
    get it wrong the same way on retry, spending every attempt on an edit any
    graph library would apply mechanically. Doing it here first means a retry
    is only spent when the design is wrong in some way this cannot fix.

    Returns a repaired copy, or None if no catalogue block is available to
    merge with — every family carries no such block, and there's nothing to
    insert.
    """
    available = [b.get("type") for b in catalogue if b.get("type") in MERGE_BLOCK_TYPES]
    merge_type = next((t for t in _AUTO_REPAIR_MERGE_PRIORITY if t in available), None)
    if merge_type is None:
        return None

    incoming = _count_incoming(spec)
    violations = [
        n for n in spec.nodes
        if get_max_inputs(n.type) == 1 and incoming.get(n.id, 0) > 1
    ]
    if not violations:
        return None

    existing_ids = {n.id for n in spec.nodes}
    nodes = list(spec.nodes)
    edges = list(spec.edges)

    for node in violations:
        merge_id = f"{node.id}_merge_auto"
        suffix = 2
        while merge_id in existing_ids:
            merge_id = f"{node.id}_merge_auto{suffix}"
            suffix += 1
        existing_ids.add(merge_id)

        nodes.append(ArchNode(id=merge_id, type=merge_type, params={}))
        redirected = False
        for i, edge in enumerate(edges):
            if edge.to_id == node.id:
                edges[i] = ArchEdge(from_id=edge.from_id, to_id=merge_id)
                redirected = True
        if redirected:
            edges.append(ArchEdge(from_id=merge_id, to_id=node.id))

    return ArchSpec(
        family=spec.family,
        nodes=nodes,
        edges=edges,
        rationale=spec.rationale,
        hw_config=spec.hw_config,
        hyperparams=spec.hyperparams,
    )


# ── Main validation function ──────────────────────────────────────────────────

def validate_arch_spec(
    spec: ArchSpec,
    catalogue: list[dict[str, Any]],
    constraints: dict[str, Any],
) -> ValidationResult:
    """
    Validate a declarative ArchSpec before materialisation.

    Args:
        spec:        The architecture spec built from a canvas snapshot
        catalogue:   Family-specific block catalogue (list of {type, name, defaultParams})
        constraints: Family constraints from catalogue.json ({requiredBlocks, incompatibleBlocks})

    Returns:
        ValidationResult with valid=True/False and detailed error messages.
    """
    errors: list[str] = []
    warnings: list[str] = []

    allowed_types: set[str] = {str(b.get("type", "")) for b in catalogue} | {"input", "output"}
    incompatible: set[str] = set(constraints.get("incompatibleBlocks", []))
    required: list[str] = constraints.get("requiredBlocks", ["input", "output"])

    node_ids = {n.id for n in spec.nodes}
    node_type_map = {n.id: n.type for n in spec.nodes}

    # 1. Node IDs must be unique
    seen_ids: set[str] = set()
    for node in spec.nodes:
        if not node.id:
            errors.append("Node has empty id")
        elif node.id in seen_ids:
            errors.append(f"Duplicate node id: '{node.id}'")
        seen_ids.add(node.id)

    # 2. All types must be in the family catalogue
    for node in spec.nodes:
        if node.type not in allowed_types:
            errors.append(f"Block type '{node.type}' is not in the {spec.family} catalogue")

    # 3. No incompatible blocks
    for node in spec.nodes:
        if node.type in incompatible:
            errors.append(
                f"Block '{node.id}' (type='{node.type}') is incompatible with family '{spec.family}'"
            )

    # 4. Required blocks must be present
    present_types = {n.type for n in spec.nodes}
    for req in required:
        if req not in present_types:
            errors.append(f"Required block '{req}' is missing from the architecture")

    # 5. Must have at least one input and one output
    if "input" not in present_types:
        errors.append("Architecture has no input node")
    if "output" not in present_types:
        errors.append("Architecture has no output node")

    # 6. Edge endpoints must exist
    for edge in spec.edges:
        if edge.from_id not in node_ids:
            errors.append(f"Edge references unknown node '{edge.from_id}'")
        if edge.to_id not in node_ids:
            errors.append(f"Edge references unknown node '{edge.to_id}'")
        if edge.from_id == edge.to_id:
            errors.append(f"Self-loop on node '{edge.from_id}'")

    # 7. No cycles (skip if edges are already invalid)
    if not errors:
        if _has_cycle(spec):
            errors.append("Architecture contains a cycle — all connections must flow forward")

    # 8. Input → Output path must exist
    if not errors:
        if not _has_io_path(spec):
            errors.append("No path exists from input to output — check your connections")

    # 9. Fan-in constraints: single-input nodes must have ≤ 1 incoming edge
    if not errors:
        # Use shared MERGE_BLOCK_TYPES for a consistent, helpful error message
        available_merge = [b.get("type") for b in catalogue if b.get("type") in MERGE_BLOCK_TYPES]
        merge_hint = ", ".join(available_merge[:3]) if available_merge else "concat or residual"
        
        incoming = _count_incoming(spec)
        for node in spec.nodes:
            in_count = incoming.get(node.id, 0)
            max_in = get_max_inputs(node.type)
            if max_in != -1 and in_count > max_in:
                errors.append(
                    f"Node '{node.id}' ({node.type}) has {in_count} incoming connections "
                    f"but only accepts {max_in}. "
                    f"Route multiple inputs through a {merge_hint} node first."
                )

    # 10. Orphan warnings (connected to input or output but not both)
    if not errors:
        fwd, rev = _build_adjacency(spec)
        input_ids = {n.id for n in spec.nodes if n.type == "input"}
        output_ids = {n.id for n in spec.nodes if n.type == "output"}

        from_inputs: set[str] = set()
        stack = list(input_ids)
        seen: set[str] = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            from_inputs.add(cur)
            for nxt in fwd.get(cur, []):
                stack.append(nxt)

        to_outputs: set[str] = set()
        stack = list(output_ids)
        seen = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            to_outputs.add(cur)
            for nxt in rev.get(cur, []):
                stack.append(nxt)

        main_flow = from_inputs & to_outputs
        for node in spec.nodes:
            if node.type in ("input", "output"):
                continue
            if node.id not in main_flow:
                errors.append(
                    f"Node '{node.id}' ({node.type}) is not on any path from input to output. "
                    "Ensure all blocks are connected in a single directed chain/graph."
                )

    # 11. Transformer/MoE families need real sublayers, not just normalization.
    #
    # A prompt-only instruction to "keep the attention and feed-forward
    # sublayers" is advisory — a smaller model can still ignore it under
    # competing instructions, and silently produce a stack of nothing but
    # LayerNorm blocks (observed live: asking to enlarge an existing GPT-2
    # produced exactly this, with total_parameters actually *dropping*
    # despite a larger requested hidden size and depth). Catching it here
    # makes it a hard validation failure the existing retry loop already
    # knows how to act on, the same way it already catches a bad num_heads.
    ATTENTION_TYPES_BY_FAMILY = {
        "transformer": {"mha", "gqa", "mqa", "mla"},
        "moe": {"mha", "gqa", "mqa", "mla"},
    }
    FEEDFORWARD_TYPES_BY_FAMILY = {
        "transformer": {"ffn", "swiglu"},
        "moe": {"ffn", "swiglu", "expert", "moe_block"},
    }
    fam_key = str(spec.family or "").lower()
    if fam_key in ATTENTION_TYPES_BY_FAMILY:
        if not (present_types & ATTENTION_TYPES_BY_FAMILY[fam_key]):
            errors.append(
                f"A '{spec.family}' architecture has no attention block "
                f"(expected one of {sorted(ATTENTION_TYPES_BY_FAMILY[fam_key])}) — "
                "every decoder/encoder layer needs one, not just normalization."
            )
        if not (present_types & FEEDFORWARD_TYPES_BY_FAMILY[fam_key]):
            errors.append(
                f"A '{spec.family}' architecture has no feed-forward block "
                f"(expected one of {sorted(FEEDFORWARD_TYPES_BY_FAMILY[fam_key])}) — "
                "every decoder/encoder layer needs one, not just normalization."
            )

    # 12. Attention head count must evenly divide the hidden size.
    #
    # The compiler backend already rejects this at analyze time — this
    # duplicates that check here so the agent's own retry loop can catch and
    # self-correct it before ever materializing, instead of only finding out
    # once the design is already on the user's canvas. Concretely: enlarging
    # an existing model's hidden_size without also updating num_heads (e.g.
    # keeping GPT-2's 12 heads after raising d_model from 768 to 1024) isn't
    # evenly divisible — 1024 / 12 isn't an integer head_dim.
    for node in spec.nodes:
        p = node.params if isinstance(node.params, dict) else {}
        hidden = p.get("hidden_size") if p.get("hidden_size") is not None else p.get("d_model")
        heads = p.get("num_heads") if p.get("num_heads") is not None else p.get("n_heads")
        if isinstance(hidden, (int, float)) and isinstance(heads, (int, float)) and heads:
            if int(hidden) % int(heads) != 0:
                errors.append(
                    f"Node '{node.id}': hidden_size ({int(hidden)}) is not evenly "
                    f"divisible by num_heads ({int(heads)}) — pick a head count "
                    f"(or hidden size) where hidden_size % num_heads == 0."
                )

    # 13. Dimension params must never be 0.
    #
    # Live failure this catches: a from-scratch CNN build left the first
    # conv's in_channels at 0 (the prompt told the planner to inherit it
    # from HW Config, which a fresh build has nothing in yet) — the
    # compiler then shaped every single downstream layer as a 0-sized
    # tensor and rejected the whole design. Padding/stride/dropout/eps are
    # legitimately 0 sometimes; a channel, feature, or vocab count never is.
    ZERO_INVALID_DIM_KEYS = (
        "in_channels", "out_channels", "in_features", "out_features",
        "hidden_size", "d_model", "vocab_size", "num_heads", "n_heads",
        "kernel_size",
    )
    for node in spec.nodes:
        p = node.params if isinstance(node.params, dict) else {}
        for key in ZERO_INVALID_DIM_KEYS:
            if key in p and isinstance(p[key], (int, float)) and p[key] == 0:
                errors.append(
                    f"Node '{node.id}': '{key}' is 0 — every layer it flows "
                    f"into will be shaped as an empty tensor. Set a real "
                    f"value (e.g. 3 for RGB in_channels on the first conv)."
                )

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
