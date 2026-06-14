"""LangChain-based helpers for structured orchestration."""

import json
import logging
import os
from typing import Any, Optional

from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] [NEURAX-LLM] %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class _SetFamilyArgs(BaseModel):
    family: str


class _ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class _FamilySelection(BaseModel):
    """Family selection response - must include family name."""
    family: str = Field(
        description="The selected architecture family name. MUST be exactly one of the allowed families provided in the prompt."
    )


def make_chat_model(temperature: float = 0.0, max_tokens: int = 2048):
    timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))

    llm_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    llm_api_key = os.getenv("LLM_API_KEY", "").strip()
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    llm_model = os.getenv("LLM_MODEL", "").strip() or os.getenv("LLAMA_MODEL", "")

    # Auto-detect provider if not explicitly set
    if not llm_provider:
        if llm_model:
            m_lower = llm_model.lower()
            if m_lower.startswith("claude-") or "anthropic" in m_lower:
                llm_provider = "anthropic"
            elif m_lower.startswith("gpt-") or m_lower.startswith("o1-"):
                llm_provider = "openai"

        if not llm_provider:
            if anthropic_api_key and not llm_api_key:
                llm_provider = "anthropic"
            else:
                llm_provider = "openai"

    if llm_provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
            logger.info(f"Using Anthropic provider with model: {llm_model or 'claude-3-5-sonnet-20240620'}")
            return ChatAnthropic(
                model=llm_model or "claude-3-5-sonnet-20240620",
                anthropic_api_key=anthropic_api_key,
                temperature=temperature,
                timeout=timeout,
                max_tokens=max_tokens,
            )
        except ImportError:
            logger.error(
                "langchain-anthropic is not installed. "
                "Run: pip install langchain-anthropic  "
                "Falling back to OpenAI with gpt-4o-mini."
            )
            # CRITICAL: reset model so we don't send a claude name to OpenAI's API
            llm_model = "gpt-4o-mini"
            llm_provider = "openai"

    # OpenAI-compatible provider (OpenAI or local LLM servers)
    from langchain_openai import ChatOpenAI
    llm_base_url = os.getenv("LLM_BASE_URL", "").strip()
    llama_base_url = os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:8080").strip()

    # Only redirect to a custom base_url if explicitly overridden or no real key exists.
    # Never apply the local llama URL when we have a real OpenAI API key.
    base_url: Optional[str] = None
    if llm_base_url and "api.openai.com" not in llm_base_url:
        base_url = llm_base_url.rstrip("/")
    elif not llm_api_key:
        base_url = llama_base_url.rstrip("/")

    openai_model = llm_model or "gpt-4o-mini"
    logger.info(f"Using OpenAI provider with model: {openai_model}")

    return ChatOpenAI(
        model=openai_model,
        base_url=base_url,
        api_key=llm_api_key or "EMPTY",
        temperature=temperature,
        timeout=timeout,
        max_tokens=max_tokens,
    )


async def select_family(
    *,
    user_message: str,
    allowed_families: list[str],
    catalogue: list[dict[str, Any]],
    current_family: Optional[str],
    max_retries: int = 3,
) -> str:
    """Select the best architecture family using LangChain structured output."""
    from langchain_core.prompts import ChatPromptTemplate

    # Normalize allowed families
    allowed = [str(x).strip() for x in allowed_families if str(x).strip()]
    if not allowed:
        raise ValueError("allowed_families is empty")

    # Build lookup for case-insensitive matching
    allowed_lower_to_orig = {a.lower(): a for a in allowed}

    # Build catalogue by family - show block types and their capabilities
    blocks_by_family: dict[str, list[dict[str, Any]]] = {f: [] for f in allowed}
    for item in catalogue:
        fam = item.get("family")
        item_type = str(item.get("type", "")).lower()
        
        # Infer family from block type/name if not explicit
        if not fam:
            if any(kw in item_type for kw in ("moe", "expert", "router")):
                fam = "moe"
            elif any(kw in item_type for kw in ("conv", "pool", "stem", "backbone")):
                fam = "cnn"
            elif any(kw in item_type for kw in ("attention", "embedding", "transformer")):
                fam = "transformer"
            elif any(kw in item_type for kw in ("graph", "gcn", "gat", "sage")):
                fam = "gnn"
            elif any(kw in item_type for kw in ("diffusion", "unet", "vae", "denois")):
                fam = "diffusion"
            elif any(kw in item_type for kw in ("ssm", "mamba", "state")):
                fam = "ssm"
            else:
                fam = item_type

        if fam in blocks_by_family:
            blocks_by_family[fam].append({
                "type": item.get("type"),
                "name": item.get("name"),
                "category": item.get("category"),
            })

    # Build a capability-focused description
    catalogue_desc = ""
    for fam, blocks in blocks_by_family.items():
        if not blocks:
            continue
        types = list(set(str(b.get("type", "")) for b in blocks if b.get("type")))
        categories = list(set(str(b.get("category", "")) for b in blocks if b.get("category")))
        catalogue_desc += f"\n  {fam}:\n"
        catalogue_desc += f"    blocks: {', '.join(types[:12])}\n"
        if categories:
            catalogue_desc += f"    capabilities: {', '.join(categories[:5])}\n"


    llm = make_chat_model()
    structured = llm.with_structured_output(_FamilySelection)

    # Architecture-agnostic family selection prompt
    system_template = """You are Neurax, a consultative AI architect. Your task is to interpret a user's business requirements and select the most appropriate neural architecture family.

## Your Role
You bridge the gap between business needs (e.g., "detect factory defects", "forecast sales", "analyze customer sentiment") and technical implementations.

## Business Domain Mapping
Use these associations as a guide when the user provides non-technical requirements:
- **Computer Vision (CNN)**: Image classification, "detecting [objects/defects]", "analyzing photos", "visual quality control".
- **Natural Language / Sequences (Transformer)**: Chatbots, "analyzing text", "translating documents", "summarizing meetings", "sentiment analysis".
- **Time Series / Financial (SSM/RNN)**: "Stock prediction", "sales forecasting", "sensor telemetry analysis", "fraud detection in transaction streams".
- **Specialized Reasoning (MoE/Transformer)**: "Expert reasoning systems", "large-scale general intelligence", "multi-task optimization".
- **Graph/Network Data (GNN)**: "Social network analysis", "drug discovery (molecular graphs)", "recommendation systems (user-item graphs)", "supply chain optimization".
- **Generative Media (Diffusion/GAN)**: "Creating realistic images", "generating artwork", "image restoration", "synthetic data generation".

## Selection Principles
1. **Business Objective First**: Identify the core problem. Is it seeing, reading, predicting, or creating?
2. **Consultative Approach**: Select the family that offers the best "backbone" for that specific business domain.
3. **Handle Ambiguity**: If the request is broad, select the most versatile family (usually Transformer or CNN) that fits the likely data type.

## Available Families
{families_list}

## Family Capabilities
{catalogue_desc}

## Output
Return ONLY a JSON object with the selected family name. The family MUST be exactly one from the available families list."""

    user_template = """User request: {user_message}

Current family: {current_family}

Based on the request and available family capabilities, which family is most appropriate?"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", user_template),
    ])

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            messages = prompt.format_messages(
                families_list=", ".join(allowed),
                catalogue_desc=catalogue_desc or "  (no blocks available)",
                user_message=user_message,
                current_family=current_family or "None set",
            )
            out: _FamilySelection = await structured.ainvoke(messages)
            
            fam_raw = str(out.family or "").strip()
            if not fam_raw:
                raise ValueError(f"Family is empty in response: {out.model_dump()}")
            fam = allowed_lower_to_orig.get(fam_raw.lower())
            if fam is None:
                logger.error(f"❌ INVALID FAMILY: '{fam_raw}' not in {allowed}")
                raise ValueError(f"Selected family '{fam_raw}' not in allowed_families: {allowed}")
            
            logger.info(f"✅ FAMILY SELECTED: '{fam}'")
            return fam
        except Exception as e:
            last_err = e
            logger.error(f"❌ FAMILY SELECTION FAILED (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                # Strengthen the prompt on retry
                system_template += (
                    "\n\nCRITICAL: You MUST return ONLY valid JSON with the exact family name from: "
                    f"{', '.join(allowed)}"
                )
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_template),
                    ("human", user_template),
                ])

    logger.error(f"❌ FAMILY SELECTION FAILED after {max_retries} retries: {last_err}")
    raise ValueError(f"Failed to select family after {max_retries} retries: {last_err}")


class _ControllerStep(BaseModel):
    """Single controller step with assistant message and tool call."""
    assistant: str = Field(description="Short user-facing reason (1-2 sentences) for THIS step")
    tool: _ToolCall


async def run_controller_step(
    *,
    user_message: str,
    snapshot: dict[str, Any],
    history: list[dict[str, Any]],
    max_retries: int = 2,
) -> dict[str, Any]:
    """Run a single controller step using LangChain structured output."""
    from langchain_core.prompts import ChatPromptTemplate

    llm = make_chat_model()
    structured = llm.with_structured_output(_ControllerStep)

    # Extract snapshot data
    allowed_families = snapshot.get("allowed_families") or []
    # Use filtered catalogue if available (family-specific blocks only)
    catalogue = snapshot.get("_filtered_catalogue") or snapshot.get("catalogue") or []
    missing_fields = snapshot.get("missing_mandatory_fields") or []
    analysis_warnings = snapshot.get("analysis_warnings") or []
    current_family = snapshot.get("family")
    nodes = snapshot.get("nodes") or []
    connections = snapshot.get("connections") or []
    hw_config = snapshot.get("hw_config") or {}

    # Build detailed catalogue with all parameters
    def _fmt_block(item: dict[str, Any]) -> str:
        btype = item.get("type", "unknown")
        bname = item.get("name", "Unknown")
        category = item.get("category", "")
        dp = item.get("defaultParams") or {}
        mandatory = item.get("mandatoryParams") or []
        max_inputs = item.get("maxInputs", 1)
        
        parts = [f"{btype} ({bname})"]
        if category:
            parts.append(f"[{category}]")
        # Show input constraint: maxInputs=1 means single input, -1 means unlimited
        if max_inputs == 1:
            parts.append("maxIn:1")
        else:
            parts.append("maxIn:∞")
        if isinstance(dp, dict) and dp:
            params_str = ", ".join(f"{k}={v}" for k, v in list(dp.items())[:6])
            parts.append(f"defaults:{{{params_str}}}")
        if isinstance(mandatory, list) and mandatory:
            parts.append(f"required:{mandatory[:4]}")
        return f"  - {' '.join(parts)}"

    catalogue_desc = "\n".join([_fmt_block(item) for item in catalogue[:100]]) if catalogue else "  (no catalogue provided)"

    # Build warnings description with actionable info
    warnings_desc = ""
    if analysis_warnings:
        for w in analysis_warnings[:15]:
            wtype = w.get("type", "unknown")
            wcode = w.get("code", "unknown")
            wmsg = w.get("message", "No message")
            wnode = w.get("nodeId") or w.get("node_id", "")
            warnings_desc += f"  - [{wtype}] {wcode}: {wmsg}"
            if wnode:
                warnings_desc += f" (node: {wnode})"
            warnings_desc += "\n"
    else:
        warnings_desc = "  (no warnings)"

    # Format history
    history_text = ""
    for h in history[-4:]:
        role = h.get("role", "user")
        content = h.get("content", "")
        # Truncate long history entries
        if len(content) > 200:
            content = content[:200] + "..."
        history_text += f"\n{role.upper()}: {content}\n"

    # Build node summary (include ALL nodes so LLM knows exact IDs)
    node_summary = ""
    existing_types = []
    for n in nodes:
        if isinstance(n, dict):
            ntype = str(n.get("type", ""))
            nid = str(n.get("id", ""))
            if ntype not in ("input", "output"):
                existing_types.append(ntype)
            node_summary += f"  - {nid}: {ntype}\n"

    # Build node input status (show which nodes have inputs and can accept more)
    # Count incoming connections per node
    incoming_count: dict[str, int] = {}
    incoming_from: dict[str, list[str]] = {}
    for c in connections:
        if isinstance(c, dict):
            to_id = str(c.get("to") or c.get("to_id") or "")
            from_id = str(c.get("from") or c.get("from_id") or "")
            if to_id:
                incoming_count[to_id] = incoming_count.get(to_id, 0) + 1
                if to_id not in incoming_from:
                    incoming_from[to_id] = []
                incoming_from[to_id].append(from_id)

    # Build input status per node
    input_status_lines = []
    for n in nodes:
        if isinstance(n, dict):
            nid = str(n.get("id", ""))
            ntype = str(n.get("type", ""))
            if ntype == "input":
                continue  # Skip input nodes
            
            # Get maxInputs: check catalogue first, then use known defaults
            max_in = None
            for item in catalogue:
                if item.get("type") == ntype:
                    max_in = item.get("maxInputs", 1)
                    break
            
            # Fallback for types not in catalogue (input/output always have maxInputs=1)
            if max_in is None:
                # These types always have max 1 input
                single_input_types = {"output", "layernorm", "rmsnorm", "batchnorm", "groupnorm", 
                                      "dropout", "flatten", "dense", "conv2d", "attention", "mha", "gqa",
                                      "embedding", "positional_encoding", "ffn", "swiglu", "pool", 
                                      "max_pool", "avg_pool", "global_pool", "classification_head",
                                      "transformer_layer", "transformer_block", "decoder_block", "encoder_block",
                                      "expert", "gate", "router_softmax", "moe_block", "residual_block",
                                      "unet_block", "downsample", "upsample", "timestep_embedding",
                                      "s4_block", "mamba_block", "lstm", "gru", "bilstm", "lstm_cell", "gru_cell",
                                      "lif_neuron", "gcn_conv", "gat_conv", "sage_conv", "global_mean_pool"}
                if ntype in single_input_types:
                    max_in = 1
                else:
                    # Default to unlimited for unknown types (residual, concat, etc.)
                    max_in = -1
            
            curr_in = incoming_count.get(nid, 0)
            sources = incoming_from.get(nid, [])
            
            if max_in == 1:
                if curr_in >= 1:
                    input_status_lines.append(f"  - {nid}: FULL (1/1 input, from: {sources[0]})")
                else:
                    input_status_lines.append(f"  - {nid}: available (0/1 inputs)")
            else:
                input_status_lines.append(f"  - {nid}: can accept more ({curr_in} inputs, unlimited)")

    input_status_desc = "\n".join(input_status_lines) if input_status_lines else "  (no processing nodes yet)"

    # Architecture-agnostic controller prompt
    system_template = """You are Neurax, an expert neural architecture designer. You construct models step-by-step using the available tools.

## Design Philosophy
You are building a computational graph that transforms input data to output predictions. Think like an architect:
- Every block serves a purpose
- Data must flow logically
- Parameters must be concrete, not symbolic

## Available Tools
- `set_family`: Set the architecture family (cnn, transformer, moe, gnn, diffusion, ssm, etc.)
- `add_node`: Add a block to the canvas (args: layer_type, node_id, x, y)
- `connect`: Wire blocks together (args: from_id, to_id)
- `disconnect`: Remove a connection (args: from_id, to_id) - use to rewire when a node is FULL
- `set_node_params`: Set block hyperparameters (args: node_id, updates)
- `set_hw_config`: Set global config (args: updates) - use for batchSize, numClasses, seqLen, etc.
- `done`: Finalize the architecture

## Construction Principles

### 1. Understand the Request
Parse the user's request for:
- **Task type**: classification, generation, detection, etc.
- **Data domain**: images, text, graphs, sequences
- **Scale indicators**: "X layers", "Y experts", "Z classes"
- **Architecture hints**: named blocks, specific patterns

### 2. CATALOGUE IS YOUR SOURCE OF TRUTH
**CRITICAL: You can ONLY use blocks listed in the Block Catalogue below.**
- Each block shows: type, name, default parameters, and required parameters
- Use `layer_type` from the catalogue when calling `add_node`
- Do NOT invent block types - only use what's in the catalogue
- Think sequentially: what block from the catalogue enables the next transformation?

### 3. Build Incrementally
Start with Input/Output, then add processing blocks between them. Each step should:
- Add ONE block OR connect nodes OR set parameters
- Be justified by the user's request or fixing an error

### 4. Parameter Inference
When adding blocks, immediately set their parameters:
- Use `defaultParams` from the catalogue as baseline
- Adjust based on user's specifications (e.g., "64 channels" → outChannels=64)
- Ensure all mandatory parameters have concrete values

### 5. Topology Rules
- Every processing block must be on a path from Input to Output
- No orphan blocks (disconnected from the main flow)
- Use `connect` to wire blocks in sequence
- Data flows forward only: input → processing → output

### 6. Building Strategy
For each step, reason about:
1. **What does the user want?** Parse the request for task type, data domain, scale, and constraints
2. **What exists already?** Review the current nodes and connections
3. **What's missing?** Identify gaps in the data flow from input to output
4. **What's next?** Choose the single most impactful action:
   - Add a missing node that enables the next transformation
   - Connect two existing nodes to extend the data path
   - Set parameters on a node that lacks required values

Think incrementally: each step adds ONE piece to the puzzle. Build the path from input to output one node and one connection at a time.

### 6. Error Recovery
When analysis_warnings show errors:
- Read the error message and affected node
- Determine which parameter is missing/invalid
- Use `set_node_params` to fix it

## Current Context
- Family: {current_family}
- Available families: {families_list}
- Missing global params: {missing_fields}

## Block Catalogue
{catalogue_desc}

## Output Format
Return JSON with:
- `assistant`: Brief explanation of this step (1-2 sentences)
- `tool`: Object with `name` and `args`"""

    user_template = """## User Request
{user_message}

## Current Architecture State
Nodes: {node_count} total
{node_summary}
Connections: {connection_count}
{connection_summary}

## Node Input Status (CRITICAL - check before connecting!)
{input_status_desc}
- Nodes marked FULL cannot accept more inputs.
- To merge multiple paths into a FULL node, use a merge block (residual, concat, etc.) first.

## Global Config
{hw_config}

## Analysis Warnings
{warnings_desc}

## Recent Actions
{history_text}

---

What is the next step to progress toward a complete architecture?"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", user_template),
    ])

    last_err = None
    for _ in range(max_retries):
        try:
            # Build connection summary
            connection_summary = ""
            for c in connections:
                if isinstance(c, dict):
                    from_id = c.get("from") or c.get("from_id")
                    to_id = c.get("to") or c.get("to_id")
                    if from_id and to_id:
                        connection_summary += f"  - {from_id} → {to_id}\n"
            if not connection_summary:
                connection_summary = "  (no connections yet)"
            
            messages = prompt.format_messages(
                families_list=", ".join(str(f) for f in allowed_families[:15]),
                catalogue_desc=catalogue_desc,
                user_message=user_message,
                current_family=current_family or "none",
                node_count=len(nodes),
                node_summary=node_summary or "  (no nodes yet)",
                connection_count=len(connections),
                connection_summary=connection_summary,
                input_status_desc=input_status_desc,
                hw_config=json.dumps(hw_config, indent=2) if hw_config else "  (empty)",
                missing_fields=", ".join(str(f) for f in missing_fields[:8]) if missing_fields else "none",
                warnings_desc=warnings_desc,
                history_text=history_text or "(no history)",
            )
            out = await structured.ainvoke(messages)
            
            tool_name = out.tool.name
            tool_args = out.tool.args
            
            # Log specific tool actions
            if tool_name == "add_node":
                logger.info(f"🔧 TOOL: add_node | type={tool_args.get('layer_type')} | id={tool_args.get('node_id')}")
            elif tool_name == "connect":
                logger.info(f"🔧 TOOL: connect | {tool_args.get('from_id')} → {tool_args.get('to_id')}")
            elif tool_name == "set_node_params":
                logger.info(f"🔧 TOOL: set_node_params | node={tool_args.get('node_id')} | updates={list(tool_args.get('updates', {}).keys())}")
            elif tool_name == "set_hw_config":
                logger.info(f"🔧 TOOL: set_hw_config | updates={list(tool_args.get('updates', {}).keys())}")
            elif tool_name == "done":
                logger.info(f"🏁 TOOL: done - Finalizing architecture")
            else:
                logger.info(f"🔧 TOOL: {tool_name} | args={tool_args}")
            
            return {"assistant": out.assistant, "tool": {"name": tool_name, "args": tool_args}}
        except Exception as e:
            last_err = e
            logger.error(f"❌ CONTROLLER STEP FAILED: {e}")

    logger.error(f"❌ CONTROLLER STEP FAILED after {max_retries} retries: {last_err}")
    raise ValueError(f"Controller step failed after {max_retries}: {last_err}")
