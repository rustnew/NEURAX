"""
Architecture Planner — Phase 1 of the new declarative agent pipeline.

Makes a single structured LLM call to produce a complete ArchSpec (nodes + edges)
for the requested user architecture. Temperature is controlled by the `creativity`
parameter (0.0 = canonical/deterministic, 1.0 = experimental/research-grade).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional, List
from pydantic import BaseModel, Field

from topology_validator import ArchSpec, ArchNode, ArchEdge
from constants import MERGE_BLOCK_TYPES
from langchain_runner import make_chat_model

logger = logging.getLogger(__name__)

class StrategyItem(BaseModel):
    id: str = Field(description="Unique step ID, e.g. '1', 'setup'")
    text: str = Field(description="High-level description of the design step")

class StrategyPlan(BaseModel):
    items: List[StrategyItem] = Field(description="ordered list of strategy steps")

# ── Temperature mapping ───────────────────────────────────────────────────────

def _creativity_to_temperature(creativity: float) -> float:
    """
    Map creativity [0.0, 1.0] → LLM temperature.

    Presets:
        0.0  → 0.00   Canon      — fully deterministic, same result every time
        0.3  → 0.27   Balanced   — small variation on known patterns
        0.7  → 0.63   Creative   — novel combos, hybrid arrangements
        1.0  → 0.90   Research   — experimental (capped at 0.9 to avoid incoherence)
    """
    return round(min(0.9, max(0.0, creativity) * 0.9), 3)


# ── Family architecture templates ─────────────────────────────────────────────

FAMILY_TEMPLATES: dict[str, str] = {
    "cnn": (
        "Typical CNN flow: Input → Conv2D (×N, increasing filters) → Pool → BN → "
        "Flatten → Dense → ClassificationHead → Output. "
        "For skip connections, route through a 'residual' merge block. "
        "For multi-branch CNNs (Inception-style): Split into parallel conv branches, "
        "then MERGE all branches with 'concat' BEFORE any pooling or dense layer. "
        "Pattern: Input → [Branch1: Conv→Conv, Branch2: Conv→Pool, ...] → concat → Pool → Dense → Output. "
        "CRITICAL: Never connect multiple branches directly to a single-input block (pool, conv, dense). "
        "Always use concat/merge/add to combine parallel branches first. "
        "EDGE EXAMPLE for multi-branch: "
        '[{{"from": "input", "to": "conv1"}}, {{"from": "input", "to": "conv2"}}, '
        '{{"from": "conv1", "to": "concat"}}, {{"from": "conv2", "to": "concat"}}, '
        '{{"from": "concat", "to": "pool"}}, {{"from": "pool", "to": "output"}}]. '
        "Every node except 'input' must have exactly ONE incoming edge (unless it is a merge block)."
    ),
    "transformer": (
        "Typical Transformer flow: Input → Embedding → PositionalEncoding → "
        "[LayerNorm → Attention → Residual → LayerNorm → FFN → Residual] (×N layers) → "
        "LMHead or ClassificationHead → Output."
    ),
    "moe": (
        "MoE flow: Input → Embedding → Attention → Gate → "
        "[Expert (×K experts, each from Gate output)] → ExpertCombine (fan-in from all experts) → "
        "LMHead → Output. Each Expert is a separate node. "
        "CRITICAL: Gate connects to EACH expert. EACH expert connects to expert_combine. "
        "expert_combine connects to the next layer (lm_head or output). "
        "For DeepSeek-style: Use multiple MoE layers with attention between them."
    ),
    "ssm": (
        "SSM / Mamba flow: Input → Embedding → "
        "[MambaBlock or S4Block] (×N, stacked sequentially) → Dense → LMHead → Output."
    ),
    "rnn": (
        "RNN flow: Input → Embedding → LSTM or GRU (×N layers) → Dense → "
        "ClassificationHead → Output."
    ),
    "gnn": (
        "GNN flow: Input → GCNConv or GATConv (×N layers) → GlobalMeanPool → "
        "Dense → Output."
    ),
    "gan": (
        "GAN Generator: Input (latent z) → Dense → Reshape → "
        "[ConvTranspose2D → BN → ReLU] (×N) → Output. "
        "GAN Discriminator: Input (image) → [Conv2D → BN → LeakyReLU] (×N) → Flatten → Dense → Output."
    ),
    "diffusion": (
        "Diffusion UNet: Input → TimestepEmbedding → "
        "[Downsample ResBlock (×N encoder levels)] → [Mid ResBlock] → "
        "[Upsample + Skip from encoder (×N decoder levels)] → Output. "
        "Use unet_block or residual_block + concat for skip merges."
    ),
    "snn": (
        "SNN flow: Input → Dense → LIF Neuron (×N layers) → Dense → Output."
    ),
    "experimental": (
        "Experimental / custom flow: freely combine any catalogue blocks. "
        "Ensure input → ... → output path exists."
    ),
}


# ── Pydantic models for LLM structured output (module-level for stable schema) ──

class _ArchNode(BaseModel):
    id: str = Field(description="Unique node identifier, e.g. 'conv1', 'attn_1'")
    type: str = Field(description="Block type — MUST be in the provided catalogue")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Block hyperparameters (use defaultParams from catalogue as baseline)"
    )


class _ArchEdge(BaseModel):
    # NOTE: use from_id/to_id (no alias) to avoid JSON schema issues with
    # reserved keyword 'from' breaking tool-calling schemas on some LLMs.
    from_id: str = Field(description="Source node id")
    to_id: str = Field(description="Target node id")


class _ArchSpecOut(BaseModel):
    family: str = Field(description="Selected architecture family")
    nodes: list[_ArchNode] = Field(description="All nodes in the architecture, in topological order")
    edges: list[_ArchEdge] = Field(description="All directed connections (from_id → to_id)")
    rationale: str = Field(
        default="",
        description="1-2 sentence explanation of the design choices made"
    )


def _parse_json_fallback(raw: str, family: str) -> _ArchSpecOut:
    """Parse raw LLM text as JSON when structured output fails.

    Handles markdown fences, 'from'/'to' edge keys, and truncated JSON
    (attempts minimal brace-repair before raising).
    """
    import re

    text = raw.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
    text = text.strip()

    start = text.find('{')
    if start == -1:
        raise ValueError("No JSON object found in LLM response")

    json_text = text[start:]

    def _normalise(data: dict) -> dict:
        raw_edges = data.get("edges", [])
        normalised = []
        for e in raw_edges:
            if isinstance(e, dict):
                from_id = e.get("from_id") or e.get("from", "")
                to_id = e.get("to_id") or e.get("to", "")
                if from_id and to_id:
                    normalised.append({"from_id": from_id, "to_id": to_id})
        data["edges"] = normalised
        if not data.get("family"):
            data["family"] = family
        return data

    # 1. Try as-is (complete JSON)
    try:
        return _ArchSpecOut.model_validate(_normalise(json.loads(json_text)))
    except json.JSONDecodeError:
        pass

    # 2. Extract largest balanced object
    depth = 0
    end = -1
    for i, ch in enumerate(json_text):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end != -1:
        try:
            return _ArchSpecOut.model_validate(_normalise(json.loads(json_text[:end])))
        except Exception:
            pass

    # 3. Truncation repair: strip trailing partial tokens then close open structures
    trimmed = re.sub(r'[,\s]*$', '', json_text)
    trimmed = re.sub(r',?\s*"[^"]*$', '', trimmed)   # remove trailing unclosed key
    open_braces = max(0, trimmed.count('{') - trimmed.count('}'))
    open_brackets = max(0, trimmed.count('[') - trimmed.count(']'))
    repaired = trimmed + ']' * open_brackets + '}' * open_braces
    try:
        data = _ArchSpecOut.model_validate(_normalise(json.loads(repaired)))
        logger.warning("JSON was truncated; parsed successfully after repair")
        return data
    except Exception as repair_err:
        raise ValueError(
            f"Could not parse or repair JSON fallback. "
            f"Error: {repair_err}. "
            f"Raw (first 300 chars): {raw[:300]!r}"
        ) from repair_err


# (legacy shim — kept so any cached reference doesn't break at import time)
def _make_arch_spec_schema():
    return _ArchSpecOut







# ── Main planner function ─────────────────────────────────────────────────────

async def plan_strategy(
    user_message: str,
    family: str,
    hw_config: Optional[dict[str, Any]] = None,
) -> list[StrategyItem]:
    """Phase 0: Generate a high-level orchestration plan/strategy."""
    hw_desc = json.dumps(hw_config or {}, indent=2)
    
    prompt = f"""You are Neurax, a consultative AI architect.
The user wants to building a solution for: "{user_message}"
Family: {family.upper()}
Hardware: {hw_desc}

Your task is to provide a high-level orchestration strategy (4-6 steps). 
The strategy should bridge the gap between their business objective and the technical neural network.

Strategy Guidelines:
1. **Analyze Domain**: Start by interpreting the specific business domain (e.g., medical imaging, text sentiment, fraud detection).
2. **Backbone Design**: Plan the core architecture backbone (e.g., feature extraction, sequence modeling) that fits the data type.
3. **Head & Task Adaptation**: Address how the model will output the specific results the business needs (e.g., classification, prediction).
4. **Validation**: Ensure the final step includes verifying the architecture against the business constraints.

Example for a "Factory Defect Detector":
1. Analyze image-based defect detection requirements for factory environments.
2. Design a robust CNN backbone for high-resolution feature extraction.
3. Integrate specialized pooling and normalization layers to handle varying lighting conditions.
4. Implement a precise classification head to distinguish between 'Pass' and 'Fail' categories.
5. Review the full topology to ensure real-time inference compatibility on the target hardware.

Return exactly 4-6 concise, business-aware steps."""

    llm = make_chat_model(temperature=0.0)
    structured = llm.with_structured_output(StrategyPlan)
    
    messages = [
        {"role": "system", "content": "You are a concise technical architect."},
        {"role": "user", "content": prompt},
    ]
    
    res = await structured.ainvoke(messages)
    return res.items

async def plan_architecture(
    user_message: str,
    family: str,
    catalogue: list[dict[str, Any]],
    constraints: dict[str, Any],
    creativity: float = 0.0,
    hw_config: Optional[dict[str, Any]] = None,
    strategy: Optional[list[str]] = None,
    previous_errors: Optional[list[str]] = None,
) -> ArchSpec:
    """
    Generate a complete architecture specification from a user request.

    Args:
        user_message:     The user's natural-language architecture description
        family:           Selected architecture family (e.g. 'cnn', 'transformer')
        catalogue:        Family-specific block catalogue (type, name, defaultParams)
        constraints:      Family constraints from catalogue.json
        creativity:       Float [0.0, 1.0] controlling LLM temperature
        hw_config:        Hardware/training configuration (inChannels, seqLen, etc.)
        previous_errors:  Validation errors from a previous attempt (for retry)

    Returns:
        ArchSpec with nodes, edges, and rationale
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    temperature = _creativity_to_temperature(creativity)
    logger.info(f"🎨 Planning architecture | family={family} | creativity={creativity} | temp={temperature}")

    # Use a generous token budget so Haiku doesn't truncate mid-JSON.
    llm = make_chat_model(temperature=temperature, max_tokens=4096)
    structured = llm.with_structured_output(_ArchSpecOut)

    # Build catalogue description — cap at 40 blocks and 5 params each to stay
    # well under Haiku's context window and avoid output truncation.
    def _fmt_block(b: dict[str, Any]) -> str:
        btype = b.get("type", "?")
        bname = b.get("name", btype)
        dp = b.get("defaultParams") or {}
        max_in = b.get("maxInputs", 1)
        parts = [f"  - {btype} ({bname})"]
        if max_in == -1:
            parts.append("[fan-in]")
        if isinstance(dp, dict) and dp:
            params_str = ", ".join(f"{k}={v}" for k, v in list(dp.items())[:5])
            parts.append(f"params: {{{params_str}}}")
        return " ".join(parts)

    catalogue_desc = "\n".join(_fmt_block(b) for b in catalogue[:40])
    incompatible = constraints.get("incompatibleBlocks", [])
    required = constraints.get("requiredBlocks", ["input", "output"])
    template = FAMILY_TEMPLATES.get(family.lower(), FAMILY_TEMPLATES["experimental"])

    # Detect available merge/fan-in blocks from the catalogue for the prompt
    available_merge_blocks = [b.get("type") for b in catalogue if b.get("type") in MERGE_BLOCK_TYPES]
    merge_hint = ", ".join(available_merge_blocks) if available_merge_blocks else "residual (if available)"

    # Creativity mode hint
    if creativity >= 0.8:
        creativity_hint = (
            "\nCREATIVITY MODE: RESEARCH — Feel free to propose novel or hybrid arrangements. "
            "You can combine blocks in unconventional ways, add auxiliary branches, "
            "or use blocks not in the typical template — as long as every block type is in the catalogue."
        )
    elif creativity >= 0.5:
        creativity_hint = (
            "\nCREATIVITY MODE: CREATIVE — You may vary the architecture meaningfully. "
            "Adjust layer counts, add skip connections, try less common block orderings."
        )
    elif creativity >= 0.2:
        creativity_hint = (
            "\nCREATIVITY MODE: BALANCED — Follow the standard template but adapt scale and structure "
            "to the user's specific request."
        )
    else:
        creativity_hint = (
            "\nCREATIVITY MODE: CANONICAL — Follow the reference template closely. "
            "Use the standard architecture pattern for this family."
        )

    hw_config_desc = json.dumps(hw_config or {}, indent=2)
    error_section = ""
    if previous_errors:
        error_section = (
            "\n\n⚠️ PREVIOUS ATTEMPT FAILED — Fix ALL of these errors:\n"
            + "\n".join(f"  - {e}" for e in previous_errors)
        )

    strategy_section = ""
    if strategy:
        strategy_section = "\n## Approved Strategy (FOLLOW THIS):\n" + "\n".join(f"- {s}" for s in strategy)

    system = f"""You are Neurax, a consultative AI architect. 
Your task is to interpret a user's business requirements and generate a professional neural architecture specification.

## Your Role: Business Domain Interpreter
You translate non-technical business needs (e.g., "classify documents", "detect defects", "forecast sales") into a precise graph of neural network blocks.
1. **Identify the Data Domain**: Is it pixels (images), tokens (text), graphs (networks), or streams (sequences)?
2. **Select the Functional Backbone**: Choose the core structure (CNN, Transformer, etc.) that best solves the business problem.
3. **Optimize for Hardware**: Ensure the scale of the model matches the training environment provided below.

## Family: {family.upper()}
## Design Principles
{template}{creativity_hint}{strategy_section}

## Block Catalogue (ONLY use types from this list)
{catalogue_desc}

## Training Environment (HW Config)
{hw_config_desc}

## Rules
1. Every node `id` must be unique (use descriptive names: conv1, attn_1, expert_0, etc.)
2. Every node `type` MUST be exactly as listed in the catalogue above
3. NEVER use these incompatible block types: {incompatible}
4. Required blocks (must include): {required}
5. There MUST be an `input` node and an `output` node
6. Every processing node MUST be part of the main flow: input → ... → output. No orphans or disconnected sub-graphs allowed.
7. Nodes with max_inputs=1 (not marked fan-in capable) can only have ONE incoming edge
8. For multiple branches merging: use one of these available merge blocks [{merge_hint}] BEFORE the target
9. Set `params` using defaultParams from the catalogue as a starting baseline, then adjust to the user's needs AND the Training Environment:
   - Match `in_channels` for the input layer to the HW Config's `inChannels`
   - Match `out_features` or `num_classes` for the head to the HW Config's `numClasses`
   - Match `hidden_size` or `d_model` if specified in HW Config
   - Ensure `stride`, `padding`, and `kernel_size` are technically sound
10. **Interpretive Rationale**: In your `rationale`, explain in clear, professional language *how* this architecture addresses the user's business goal. Avoid overly dense jargon where possible.
11. The `edges` list MUST be COMPLETE — every connection must be explicitly listed:
    - For MoE: Gate → Expert_1, Gate → Expert_2, ..., Expert_1 → Expert_n...
    - For parallel branches: ALL branches must have explicit edges to their merge node
    - NO IMPLICIT CONNECTIONS — if it is not in edges, it will not be connected
    - EDGE FORMAT: Each edge is {{"from_id": "source_id", "to_id": "target_id"}}
12. Every processing node MUST have at least one incoming AND one outgoing edge (no orphans)
13. CHAIN CONNECTIVITY: Ensure a continuous path from input to output:
    - input → first_layer → second_layer → ... → output
    - NO GAPS: Every node must be reachable from input AND must reach output
14. MANDATORY PARAMETERS: For every node, you MUST provide appropriate values for these mandatory fields if present in the block's catalogue entry:
    - CNN: in_channels, out_channels, kernel_size, stride, padding
    - Transformer/SSM: d_model, n_heads, d_state, seq_len, vocab_size
    - Linear/Dense: in_features, out_features
    - Classification: num_classes{error_section}

Output a COMPLETE JSON spec with all nodes and edges. You MUST populate the `nodes` and `edges` arrays.

EXAMPLE output structure (use this as a template — replace with real architecture):
{{
  "family": "{family}",
  "nodes": [
    {{"id": "input", "type": "input", "params": {{}}}},
    {{"id": "conv1", "type": "conv2d", "params": {{"in_channels": 3, "out_channels": 32, "kernel_size": 3}}}},
    {{"id": "output", "type": "output", "params": {{}}}}
  ],
  "edges": [
    {{"from_id": "input", "to_id": "conv1"}},
    {{"from_id": "conv1", "to_id": "output"}}
  ],
  "rationale": "Brief explanation here."
}}"""

    user_tmpl = f"""## User Request
{user_message}

Generate the complete architecture specification now."""

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=user_tmpl),
    ]

    out: _ArchSpecOut
    try:
        out = await structured.ainvoke(messages)
        # Validate that essential fields were actually populated (Haiku sometimes
        # returns only {'family': '...'} when the schema is too demanding)
        if not out.nodes or not out.edges:
            raise ValueError(
                f"Structured output incomplete: nodes={len(out.nodes)}, edges={len(out.edges)}. "
                "Falling back to JSON mode."
            )
    except Exception as struct_err:
        logger.warning(
            f"⚠️  Structured output failed ({struct_err}); retrying in JSON mode..."
        )
        # JSON-mode fallback: ask the LLM to output raw JSON without tool-calling
        json_system = system + (
            "\n\nIMPORTANT: Output ONLY a single raw JSON object with keys "
            '"family", "nodes", "edges", "rationale". '
            "No markdown fences, no prose — just the JSON object."
        )
        json_messages = [
            SystemMessage(content=json_system),
            HumanMessage(content=user_tmpl),
        ]
        raw_response = await llm.ainvoke(json_messages)
        raw_text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
        out = _parse_json_fallback(raw_text, family)
        logger.info("✅ JSON fallback parsed successfully")

    logger.info(f"✅ Spec generated: {[n.id for n in out.nodes[:6]]}...")

    # Convert to ArchSpec dataclass
    nodes = [
        ArchNode(
            id=n.id,
            type=n.type,
            params=n.params if isinstance(n.params, dict) else {},
        )
        for n in out.nodes
    ]
    edges = [
        ArchEdge(from_id=e.from_id, to_id=e.to_id)
        for e in out.edges
    ]

    return ArchSpec(
        family=out.family or family,
        nodes=nodes,
        edges=edges,
        rationale=out.rationale or "",
    )
