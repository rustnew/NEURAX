"""Suggestion functions for agent actions."""
import logging
import re
from typing import Any, Optional
from graph_utils import (
    _node_type, _graph_sets, _reachable, _main_flow_nodes, _orphan_nodes,
    _has_orphans, _unique_node_id
)

logger = logging.getLogger(__name__)

# Catalogue cache (shared with config module)
_catalogue_cache: dict[str, list[dict[str, Any]]] = {}
_CATALOGUE_CACHE_MAX = 32


def _rehydrate_catalogue(snapshot: dict[str, Any]) -> None:
    cat = snapshot.get("catalogue")
    cat_id = snapshot.get("catalogue_id")

    if isinstance(cat, list) and cat:
        if isinstance(cat_id, str) and cat_id:
            _catalogue_cache[cat_id] = cat
            # bounded cache
            while len(_catalogue_cache) > _CATALOGUE_CACHE_MAX:
                _catalogue_cache.pop(next(iter(_catalogue_cache)))
        return

    if isinstance(cat_id, str) and cat_id:
        cached = _catalogue_cache.get(cat_id)
        if cached:
            snapshot["catalogue"] = cached


def _suggest_attach_orphan(snapshot: dict[str, Any]) -> Optional[dict[str, Any]]:
    nodes: list[dict[str, Any]] = list(snapshot.get("nodes") or [])
    inputs, outputs, fwd, rev = _graph_sets(snapshot)
    
    if not inputs or not outputs:
        return None

    # Main flow nodes are those reachable from input AND can reach output
    flow = _main_flow_nodes(snapshot)
    
    # Identify nodes that are NOT in the main flow
    non_flow_nodes = []
    for n in nodes:
        nid = str(n.get("id") or "")
        if not nid or _node_type(n) in ("input", "output"):
            continue
        if nid not in flow:
            non_flow_nodes.append(nid)

    if not non_flow_nodes:
        return None

    # Pick the first non-flow node to fix
    target_id = non_flow_nodes[0]
    
    # Check if it's reachable from input
    from_input = _reachable(inputs, fwd)
    is_reachable = target_id in from_input
    
    if is_reachable:
        # It's reachable from input but doesn't reach output (Dead-end)
        # Connect it to the output to complete the path
        out_id = next(iter(outputs))
        print(f"[ORPHAN FIX] Connecting dead-end node {target_id} to output {out_id}")
        return {"name": "connect", "args": {"from_id": target_id, "to_id": out_id}}
    else:
        # It's NOT reachable from input (Orphan)
        # Connect the "tail" of the current flow (or an input) to it
        tail: Optional[str] = None
        if flow:
            # Find a node in flow that has no outgoing edges to other flow nodes (excluding output)
            for nid in flow:
                n = next((x for x in nodes if str(x.get("id")) == nid), None)
                if not n or _node_type(n) == "output":
                    continue
                # If this node doesn't connect to any other non-output node in the flow
                outs_in_flow = [x for x in fwd.get(nid, set()) if x in flow and _node_type(next(y for y in nodes if str(y.get("id")) == x)) != "output"]
                if not outs_in_flow:
                    tail = nid
                    break
        
        source_id = tail if tail else next(iter(inputs))
        return {"name": "connect", "args": {"from_id": source_id, "to_id": target_id}}

    return None


def _suggest_family(snapshot: dict[str, Any], user_message: str) -> Optional[dict[str, Any]]:
    # Removed hardcoded family inference - let the LLM decide based on context
    # The LLM has access to allowed_families in the prompt and can pick appropriately
    return None


def _suggest_hw_defaults(snapshot: dict[str, Any], user_message: str) -> Optional[dict[str, Any]]:
    missing_mandatory = snapshot.get("missing_mandatory_fields") or []
    if not isinstance(missing_mandatory, list) or not missing_mandatory:
        return None

    hw = snapshot.get("hw_config")
    if not isinstance(hw, dict):
        hw = {}

    missing_set = {str(x) for x in missing_mandatory}
    updates: dict[str, Any] = {}

    # ── Common minimal defaults (all families) ───────────────────────────────
    if "hardware" in missing_set and not str(hw.get("hardware") or "").strip():
        updates["hardware"] = "cpu"
    if "precision" in missing_set and not str(hw.get("precision") or "").strip():
        updates["precision"] = "fp32"
    if "batchSize" in missing_set and (hw.get("batchSize") is None or float(hw.get("batchSize") or 0) <= 0):
        updates["batchSize"] = 1

    current_family = str(snapshot.get("family") or "").lower().strip()

    # ── Family-specific mandatory fields ─────────────────────────────────────
    if current_family in ("cnn", "vit"):
        if "inChannels" in missing_set and (hw.get("inChannels") is None or float(hw.get("inChannels") or 0) <= 0):
            updates["inChannels"] = 3
        # Opportunistic fills
        if hw.get("imgHeight") is None or float(hw.get("imgHeight") or 0) <= 0:
            updates["imgHeight"] = 224
        if hw.get("imgWidth") is None or float(hw.get("imgWidth") or 0) <= 0:
            updates["imgWidth"] = 224
        if hw.get("numClasses") is None:
            updates["numClasses"] = 1000

    elif current_family in ("transformer", "rnn"):
        if "seqLen" in missing_set and (hw.get("seqLen") is None or int(hw.get("seqLen") or 0) <= 0):
            updates["seqLen"] = 2048
        if "numLayers" in missing_set and (hw.get("numLayers") is None or int(hw.get("numLayers") or 0) <= 0):
            updates["numLayers"] = 6
        if "vocabSize" in missing_set and (hw.get("vocabSize") is None or int(hw.get("vocabSize") or 0) <= 0):
            updates["vocabSize"] = 32000
        # Opportunistic fills
        if hw.get("dModel") is None or int(hw.get("dModel") or 0) <= 0:
            updates["dModel"] = 768
        if hw.get("nHeads") is None or int(hw.get("nHeads") or 0) <= 0:
            updates["nHeads"] = 12
        if hw.get("numClasses") is None:
            updates["numClasses"] = 32000  # vocab-size default for LM head

    elif current_family == "moe":
        if "seqLen" in missing_set and (hw.get("seqLen") is None or int(hw.get("seqLen") or 0) <= 0):
            updates["seqLen"] = 2048
        if "numLayers" in missing_set and (hw.get("numLayers") is None or int(hw.get("numLayers") or 0) <= 0):
            updates["numLayers"] = 6
        if "vocabSize" in missing_set and (hw.get("vocabSize") is None or int(hw.get("vocabSize") or 0) <= 0):
            updates["vocabSize"] = 32000
        if "numExperts" in missing_set and (hw.get("numExperts") is None or int(hw.get("numExperts") or 0) <= 0):
            updates["numExperts"] = 8
        if "topK" in missing_set and (hw.get("topK") is None or int(hw.get("topK") or 0) <= 0):
            updates["topK"] = 2
        # Opportunistic fills
        if hw.get("dModel") is None or int(hw.get("dModel") or 0) <= 0:
            updates["dModel"] = 768
        if hw.get("nHeads") is None or int(hw.get("nHeads") or 0) <= 0:
            updates["nHeads"] = 12
        if hw.get("numClasses") is None:
            updates["numClasses"] = 32000

    elif current_family == "ssm":
        if "dState" in missing_set and (hw.get("dState") is None or int(hw.get("dState") or 0) <= 0):
            updates["dState"] = 16
        # Opportunistic fills
        if hw.get("dModel") is None or int(hw.get("dModel") or 0) <= 0:
            updates["dModel"] = 512
        if hw.get("vocabSize") is None or int(hw.get("vocabSize") or 0) <= 0:
            updates["vocabSize"] = 32000
        if hw.get("seqLen") is None or int(hw.get("seqLen") or 0) <= 0:
            updates["seqLen"] = 2048

    elif current_family == "diffusion":
        if "numDenoisingSteps" in missing_set and (hw.get("numDenoisingSteps") is None or int(hw.get("numDenoisingSteps") or 0) <= 0):
            updates["numDenoisingSteps"] = 1000
        if "inChannels" in missing_set and (hw.get("inChannels") is None or int(hw.get("inChannels") or 0) <= 0):
            updates["inChannels"] = 4   # latent diffusion typical
        # Opportunistic fills
        if hw.get("imgSize") is None or int(hw.get("imgSize") or 0) <= 0:
            updates["imgSize"] = 256

    elif current_family == "gnn":
        if "numNodes" in missing_set and (hw.get("numNodes") is None or int(hw.get("numNodes") or 0) <= 0):
            updates["numNodes"] = 100
        if "numEdges" in missing_set and (hw.get("numEdges") is None or int(hw.get("numEdges") or 0) <= 0):
            updates["numEdges"] = 500
        if "nodeFeatDim" in missing_set and (hw.get("nodeFeatDim") is None or int(hw.get("nodeFeatDim") or 0) <= 0):
            updates["nodeFeatDim"] = 64
        # Opportunistic fills
        if hw.get("numClasses") is None:
            updates["numClasses"] = 10

    elif current_family == "gan":
        # Opportunistic fills
        if hw.get("latentDim") is None or int(hw.get("latentDim") or 0) <= 0:
            updates["latentDim"] = 512
        if hw.get("imgChannels") is None or int(hw.get("imgChannels") or 0) <= 0:
            updates["imgChannels"] = 3
        if hw.get("imgSize") is None or int(hw.get("imgSize") or 0) <= 0:
            updates["imgSize"] = 128

    elif current_family == "snn":
        # Opportunistic fills
        if hw.get("timesteps") is None or int(hw.get("timesteps") or 0) <= 0:
            updates["timesteps"] = 100
        if hw.get("threshold") is None or float(hw.get("threshold") or 0) <= 0:
            updates["threshold"] = 1.0

    if not updates:
        return None
    return {"name": "set_hw_config", "args": {"updates": updates}}


def _count_non_io_nodes(snapshot: dict[str, Any]) -> int:
    nodes = snapshot.get("nodes") or []
    if not isinstance(nodes, list):
        return 0
    c = 0
    for n in nodes:
        if not isinstance(n, dict):
            continue
        t = str(n.get("type") or "")
        if t and t not in ("input", "output"):
            c += 1
    return c


def _suggest_moe_architecture(snapshot: dict[str, Any], user_message: str) -> Optional[dict[str, Any]]:
    """Suggest MoE-specific blocks including multiple experts.
    
    Parses user request for 'X experts' and adds that many expert_linear blocks.
    """
    family = str(snapshot.get("family") or "").lower()
    if family != "moe":
        return None
    
    nodes = snapshot.get("nodes") or []
    if not isinstance(nodes, list):
        return None
    
    # Count existing expert blocks
    existing_experts = sum(
        1 for n in nodes 
        if isinstance(n, dict) and "expert" in str(n.get("type", "")).lower()
    )
    
    # Parse number of experts from user message
    import re
    match = re.search(r"(\d+)\s*expert", user_message.lower())
    target_experts = int(match.group(1)) if match else 8
    
    # If we already have enough experts, don't add more
    if existing_experts >= target_experts:
        return None
    
    # Add next expert block
    existing_ids = {str(n.get("id")) for n in nodes if isinstance(n, dict) and n.get("id")}
    expert_id = _unique_node_id(existing_ids, f"expert_{existing_experts + 1}")
    
    # Position after router, before combine
    max_x = max([float(n.get("x") or 0.0) for n in nodes if isinstance(n, dict)], default=300.0)
    avg_y = sum([float(n.get("y") or 0.0) for n in nodes if isinstance(n, dict)]) / max(1, len([n for n in nodes if isinstance(n, dict)]))
    
    return {
        "name": "add_node",
        "args": {
            "layer_type": "expert_linear",
            "node_id": expert_id,
            "x": max_x + 150,
            "y": avg_y + (existing_experts * 80),  # Stagger vertically
        }
    }


def _suggest_vision_backbone(snapshot: dict[str, Any], user_message: str) -> Optional[dict[str, Any]]:
    # Only suggest vision backbone if current family is explicitly CNN
    current_family = str(snapshot.get("family") or "")
    if current_family != "cnn":
        return None

    # If user asks for detection/classification and we have no backbone layers yet,
    # bootstrap with a lightweight stem so we don't end up with Input -> Head directly.
    if _count_non_io_nodes(snapshot) > 0:
        return None

    allowed = snapshot.get("allowed_layer_types") or []
    if not isinstance(allowed, list) or "stem_block" not in {str(x) for x in allowed}:
        return None

    nodes = snapshot.get("nodes") or []
    if not isinstance(nodes, list):
        return None

    existing_ids = {str(n.get("id")) for n in nodes if isinstance(n, dict) and n.get("id")}
    node_id = _unique_node_id(existing_ids, "stem")

    # Place near the left side so connect-chain ordering works well.
    try:
        min_x = min([float(n.get("x") or 0.0) for n in nodes if isinstance(n, dict)], default=0.0)
        avg_y = sum([float(n.get("y") or 0.0) for n in nodes if isinstance(n, dict)]) / max(
            1,
            len([n for n in nodes if isinstance(n, dict)]),
        )
    except Exception:
        min_x = 0.0
        avg_y = 0.0

    return {"name": "add_node", "args": {"layer_type": "stem_block", "node_id": node_id, "x": min_x + 260, "y": avg_y}}


def _suggest_input_shape_for_family(snapshot: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Update input block shape to match the architecture family."""
    family = str(snapshot.get("family") or "").lower()
    nodes: list[dict[str, Any]] = list(snapshot.get("nodes") or [])

    input_node = None
    for n in nodes:
        if isinstance(n, dict) and str(n.get("type")) == "input":
            input_node = n
            break
    if not input_node:
        return None

    node_id = str(input_node.get("id"))
    params = input_node.get("params") if isinstance(input_node.get("params"), dict) else {}

    # Define expected shapes per family
    family_shapes = {
        "cnn": {"shape": "[B, 3, 224, 224]", "in_channels": 3},  # Image: RGB 224x224
        "transformer": {"shape": "[B, seq_len]", "vocab_size": 32000},  # Text
        "vit": {"shape": "[B, 3, 224, 224]", "in_channels": 3},  # Vision transformer
        "diffusion": {"shape": "[B, 3, 256, 256]", "in_channels": 3},  # Image generation
        "gnn": {"shape": "[B, num_nodes, num_features]", "in_features": 64},  # Graph
        "moe": {"shape": "[B, seq_len]", "vocab_size": 32000},  # Text with MoE
        "ssm": {"shape": "[B, seq_len]", "vocab_size": 32000},  # Sequence
        "rnn": {"shape": "[B, seq_len]", "vocab_size": 32000},  # Sequence
    }

    expected = family_shapes.get(family)
    if not expected:
        return None

    current_shape = str(params.get("shape") or "")
    expected_shape = expected.get("shape", "")

    # Only update if shape is different or not set
    if current_shape != expected_shape:
        updates: dict[str, Any] = {"shape": expected_shape}
        # Also set any family-specific params
        for k, v in expected.items():
            if k != "shape" and k not in params:
                updates[k] = v
        return {"name": "set_node_params", "args": {"node_id": node_id, "updates": updates}}

    return None


def _suggest_add_io(snapshot: dict[str, Any]) -> Optional[dict[str, Any]]:
    nodes = snapshot.get("nodes") or []
    if not isinstance(nodes, list):
        return None

    has_input = any(isinstance(n, dict) and str(n.get("type")) == "input" for n in nodes)
    has_output = any(isinstance(n, dict) and str(n.get("type")) == "output" for n in nodes)

    existing_ids = {str(n.get("id")) for n in nodes if isinstance(n, dict) and n.get("id")}
    max_x = 0.0
    max_y = 0.0
    for n in nodes:
        if not isinstance(n, dict):
            continue
        try:
            x = float(n.get("x") or 0.0)
            y = float(n.get("y") or 0.0)
        except Exception:
            x = 0.0
            y = 0.0
        if x >= max_x:
            max_x = x
            max_y = y

    if not has_input:
        node_id = _unique_node_id(existing_ids, "input_node")
        return {"name": "add_node", "args": {"layer_type": "input", "node_id": node_id, "x": max_x - 260, "y": max_y}}

    if not has_output:
        node_id = _unique_node_id(existing_ids, "output_node")
        return {"name": "add_node", "args": {"layer_type": "output", "node_id": node_id, "x": max_x + 260, "y": max_y}}

    return None


def _suggest_connect(snapshot: dict[str, Any]) -> Optional[dict[str, Any]]:
    nodes = snapshot.get("nodes") or []
    conns = snapshot.get("connections") or []
    if not isinstance(nodes, list) or not isinstance(conns, list):
        return None

    input_nodes = [n for n in nodes if isinstance(n, dict) and str(n.get("type")) == "input" and n.get("id")]
    output_nodes = [n for n in nodes if isinstance(n, dict) and str(n.get("type")) == "output" and n.get("id")]
    if not input_nodes or not output_nodes:
        return None

    input_id = str(input_nodes[0].get("id"))
    output_id = str(output_nodes[0].get("id"))

    from_ids = {str(c.get("from")) for c in conns if isinstance(c, dict) and c.get("from")}
    to_ids = {str(c.get("to")) for c in conns if isinstance(c, dict) and c.get("to")}

    non_io_ids = [
        str(n.get("id"))
        for n in nodes
        if isinstance(n, dict)
        and n.get("id")
        and str(n.get("type")) not in ("input", "output")
    ]

    if input_id not in from_ids:
        target = non_io_ids[0] if non_io_ids else output_id
        if target and target != input_id:
            return {"name": "connect", "args": {"from_id": input_id, "to_id": target}}

    if output_id not in to_ids:
        source = None
        for nid in reversed(non_io_ids):
            if nid != output_id:
                source = nid
                break
        source = source or input_id
        if source and source != output_id:
            return {"name": "connect", "args": {"from_id": source, "to_id": output_id}}

    return None


def _suggest_connect_chain(snapshot: dict[str, Any]) -> Optional[dict[str, Any]]:
    nodes = snapshot.get("nodes") or []
    conns = snapshot.get("connections") or []
    if not isinstance(nodes, list) or not isinstance(conns, list):
        return None

    input_nodes = [n for n in nodes if isinstance(n, dict) and str(n.get("type")) == "input" and n.get("id")]
    output_nodes = [n for n in nodes if isinstance(n, dict) and str(n.get("type")) == "output" and n.get("id")]
    if not input_nodes or not output_nodes:
        return None

    input_id = str(input_nodes[0].get("id"))
    output_id = str(output_nodes[0].get("id"))

    non_io = [
        n
        for n in nodes
        if isinstance(n, dict)
        and n.get("id")
        and str(n.get("type")) not in ("input", "output")
    ]
    non_io_sorted = sorted(non_io, key=lambda n: float(n.get("x") or 0.0))
    chain_ids = [input_id, *[str(n.get("id")) for n in non_io_sorted], output_id]

    existing: set[tuple[str, str]] = set()
    for c in conns:
        if not isinstance(c, dict):
            continue
        a = c.get("from")
        b = c.get("to")
        if not a or not b:
            continue
        existing.add((str(a), str(b)))

    for a, b in zip(chain_ids, chain_ids[1:]):
        if a and b and a != b and (a, b) not in existing:
            return {"name": "connect", "args": {"from_id": a, "to_id": b}}

    return None


# Node types that can accept multiple inputs (fan-in)
FAN_IN_CAPABLE_TYPES = {
    "concat", "residual_add", "skip_connection",
    "router_softmax", "lm_head", "expert_combine",
    "output_combination", "moe_block"
}


def _suggest_fanin_fix(snapshot: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Suggest adding a merge node when a connect was rejected for fan-in.
    
    If the last tool rejection was a fan-in issue, propose:
    1. Add a fan-in capable node (residual_add or concat)
    2. Rewire: disconnect existing input → merge, connect rejected source → merge, connect merge → target
    """
    last_rej = snapshot.get("_last_tool_rejection")
    if not isinstance(last_rej, dict):
        return None
    if str(last_rej.get("reason") or "") != "fan_in":
        return None
    
    to_id = str(last_rej.get("args", {}).get("to_id") or "")
    from_id = str(last_rej.get("args", {}).get("from_id") or "")
    existing_inputs = list(last_rej.get("existing_inputs") or [])
    
    if not to_id or not from_id:
        return None
    
    nodes = snapshot.get("nodes") or []
    if not isinstance(nodes, list):
        return None
    
    # Find the target node to get its position
    target_node = None
    for n in nodes:
        if isinstance(n, dict) and str(n.get("id")) == to_id:
            target_node = n
            break
    
    if not target_node:
        return None
    
    # Check if a fan-in capable block type exists in allowed_layer_types
    allowed = set(snapshot.get("allowed_layer_types") or [])
    merge_type = None
    for t in ["residual_add", "concat", "skip_connection"]:
        if t in allowed:
            merge_type = t
            break
    
    if not merge_type:
        # No merge node available - can't auto-fix
        return None
    
    # Generate unique ID for merge node
    existing_ids = {str(n.get("id")) for n in nodes if isinstance(n, dict) and n.get("id")}
    merge_id = _unique_node_id(existing_ids, f"{merge_type}_1")
    
    # Position merge node slightly before target
    try:
        target_x = float(target_node.get("x") or 0)
        target_y = float(target_node.get("y") or 0)
    except Exception:
        target_x = 0
        target_y = 0
    
    merge_x = target_x - 100
    merge_y = target_y + 50  # Slight offset for visual clarity
    
    # Return add_node for merge block
    return {
        "name": "add_node",
        "args": {
            "layer_type": merge_type,
            "node_id": merge_id,
            "x": merge_x,
            "y": merge_y,
        },
        # Include wiring hints for agent_runner to use in feedback
        "_wiring_hint": {
            "merge_id": merge_id,
            "existing_inputs": existing_inputs,
            "rejected_from": from_id,
            "target": to_id,
        }
    }


def _suggest_fix_from_warnings(snapshot: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Batch-fix all missing hyperparameter errors for each affected node.

    Returns a single set_node_params tool call that patches ALL missing params
    on the first erroring node, so we don't waste one loop iteration per field.
    """
    warnings = snapshot.get("analysis_warnings")
    if not isinstance(warnings, list) or not warnings:
        return None

    nodes: list[dict[str, Any]] = list(snapshot.get("nodes") or [])

    def find_node(node_id: str) -> Optional[dict[str, Any]]:
        for n in nodes:
            if str(n.get("id")) == node_id:
                return n
        return None

    def _default_val(key: str, node: Optional[dict[str, Any]], family: str) -> Any:
        """Return a safe minimal default for a missing hyperparameter."""
        k = key.lower()
        fam = family.lower()
        # Vision
        if k == "inchannels":     return 3
        if k in {"outchannels", "channels", "num_channels"}: return 64
        if k in {"kernel", "kernel_size", "kernelsize"}:    return 3
        if k in {"stride"}:       return 1
        if k in {"padding"}:      return 0
        if k in {"groups"}:       return 1
        # Language / sequence
        if k in {"vocab_size", "vocabsize"}:                return 32000
        if k in {"d_model", "dmodel", "embed_dim", "hidden_size"}: 
            return 768 if fam in ("transformer", "moe", "rnn") else 512
        if k in {"n_heads", "nheads", "num_heads"}:        return 12
        if k in {"seq_len", "seqlen", "max_seq_len"}:     return 512
        if k in {"d_ff", "dff", "ffn_dim", "ffn_hidden"}:return 2048
        if k in {"d_state", "dstate"}:                     return 16
        # Classification / detection
        if k in {"num_classes", "numclasses"}:             return 10
        # Diffusion
        if k in {"timesteps", "time_steps", "num_timesteps"}: return 1000
        # GAN
        if k in {"latent_dim", "latentdim", "z_dim"}:    return 512
        # Generic
        if k in {"dim", "size", "features", "hidden"}:    return 64
        if k in {"dropout", "drop_rate"}:                 return 0.1
        return 1

    family = str(snapshot.get("family") or "").lower()

    # ── Group all missing-hyperparameter errors by node_id ───────────────────
    # node_id → {key: safe_default}
    per_node: dict[str, dict[str, Any]] = {}

    for w in warnings:
        if not isinstance(w, dict):
            continue
        if str(w.get("type") or "").lower() != "error":
            continue
        code = str(w.get("code") or "")
        msg = str(w.get("message") or "")
        node_id = str(w.get("nodeId") or w.get("node_id") or "")

        if not node_id:
            continue
        if code != "E_MISSING_HYPERPARAMETER" and "Mandatory Hyperparameter" not in msg:
            continue

        m = re.search(r"Mandatory Hyperparameter:\s*([A-Za-z0-9_]+)", msg)
        if not m:
            continue
        key = m.group(1)

        n = find_node(node_id)
        if n is None:
            continue
        params = n.get("params") if isinstance(n.get("params"), dict) else {}
        if key in params and params.get(key) not in (None, 0, "", False):
            continue

        if node_id not in per_node:
            per_node[node_id] = {}
        per_node[node_id][key] = _default_val(key, n, family)

    if not per_node:
        return None

    # Fix the first affected node — emit all its missing keys at once
    first_node_id = next(iter(per_node))
    updates = per_node[first_node_id]
    return {"name": "set_node_params", "args": {"node_id": first_node_id, "updates": updates}}
