"""Snapshot manipulation - applying tools to canvas state."""
import json
import logging
from typing import Any, Optional
from topology_validator import get_max_inputs

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] [NEURAX-SNAPSHOT] %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _apply_tool_to_snapshot(snapshot: dict[str, Any], tool: dict[str, Any]) -> dict[str, Any]:
    name = str(tool.get("name") or "")
    args = tool.get("args") or {}
    
    nodes: list[dict[str, Any]] = list(snapshot.get("nodes") or [])
    conns: list[dict[str, Any]] = list(snapshot.get("connections") or [])

    # Best-effort: clear previous tool outcome metadata
    snapshot.pop("_last_tool_rejection", None)
    
    initial_node_count = len(nodes)
    initial_conn_count = len(conns)

    def find_node(node_id: str) -> Optional[dict[str, Any]]:
        for n in nodes:
            if str(n.get("id")) == node_id:
                return n
        return None

    def node_type(node_id: str) -> str:
        n = find_node(node_id)
        if not n:
            return ""
        return str(n.get("type") or "")

    def in_degree(node_id: str) -> int:
        d = 0
        for c in conns:
            if not isinstance(c, dict):
                continue
            if str(c.get("to") or c.get("to_id") or "") == node_id:
                d += 1
        return d

    def allows_fanin(node_id: str) -> bool:
        """True if this node can accept more than one incoming edge."""
        t = node_type(node_id)
        return get_max_inputs(t) == -1  # -1 means unlimited fan-in

    def max_fanin(node_id: str) -> int:
        """Returns the max allowed incoming edges for this node (-1 = unlimited)."""
        t = node_type(node_id)
        return get_max_inputs(t)

    if name == "clear_canvas":
        # Wipe all nodes, edges, and cached analysis so the build starts from blank.
        logger.info(f"🧹 CLEAR CANVAS: removed {initial_node_count} nodes, {initial_conn_count} connections")
        snapshot["nodes"] = []
        snapshot["connections"] = []
        snapshot["groups"] = []
        snapshot["analysis_warnings"] = []
        snapshot["missing_mandatory_fields"] = list(snapshot.get("missing_mandatory_fields") or [])

    if name == "set_family":
        fam = str(args.get("family") or "")
        if fam:
            old_family = snapshot.get("family", "none")
            snapshot["family"] = fam
            logger.info(f"🏷️ SET FAMILY: '{old_family}' → '{fam}'")

    if name == "set_hw_config":
        updates = args.get("updates")
        if isinstance(updates, dict):
            hw = snapshot.get("hw_config")
            if not isinstance(hw, dict):
                hw = {}
            hw.update(updates)
            snapshot["hw_config"] = hw
            logger.info(f"⚙️ SET HW_CONFIG: {list(updates.keys())}")
            # Only prune the fields that were actually set — leave remaining mandatory
            # fields visible so the agent continues filling them on subsequent steps.
            updated_keys = set(updates.keys())
            prev_missing = list(snapshot.get("missing_mandatory_fields") or [])
            snapshot["missing_mandatory_fields"] = [
                f for f in prev_missing if f not in updated_keys
            ]

    if name == "add_node":
        layer_type = str(args.get("layer_type") or "")
        node_id = str(args.get("node_id") or "")
        x = float(args.get("x") or 0)
        y = float(args.get("y") or 0)
        
        # ── AUTO-GENERATE UNIQUE ID if proposed ID already exists ──
        existing_ids = {str(n.get("id")) for n in nodes if n.get("id")}
        actual_id = node_id
        if node_id and node_id in existing_ids:
            # Generate unique ID: conv2 -> conv2_2, conv2_2 -> conv2_3, etc.
            i = 2
            while f"{node_id}_{i}" in existing_ids:
                i += 1
            actual_id = f"{node_id}_{i}"
            logger.info(f"🔄 AUTO-RENAMED: '{node_id}' → '{actual_id}' (duplicate detected)")
        
        if actual_id:
            nodes.append({"id": actual_id, "type": layer_type, "name": layer_type, "x": x, "y": y, "params": {}})
            logger.info(f"➕ ADD NODE: id='{actual_id}' type='{layer_type}' pos=({x:.0f}, {y:.0f})")
            # Store the actual ID used so agent_runner can feed it back to LLM
            snapshot["_last_added_node_id"] = actual_id
        else:
            logger.warning(f"⚠️ ADD NODE FAILED: invalid node_id")

    elif name == "set_node_params":
        node_id = str(args.get("node_id") or "")
        updates = args.get("updates")
        if node_id and isinstance(updates, dict):
            n = find_node(node_id)
            if n is not None:
                params = n.get("params")
                if not isinstance(params, dict):
                    params = {}
                params.update(updates)
                n["params"] = params
                logger.info(f"📝 SET PARAMS: node='{node_id}' updates={list(updates.keys())}")

                # Best-effort pruning: if snapshot contains warnings about missing params for this node,
                # drop any warning whose message mentions one of the updated keys.
                aw = snapshot.get("analysis_warnings")
                if isinstance(aw, list) and updates:
                    keys = {str(k) for k in updates.keys()}

                    def _keep(w: Any) -> bool:
                        if not isinstance(w, dict):
                            return True
                        w_node = str(w.get("nodeId") or w.get("node_id") or "")
                        if w_node != node_id:
                            return True
                        msg = str(w.get("message") or "")
                        for k in keys:
                            if k and k in msg:
                                return False
                        return True

                    snapshot["analysis_warnings"] = [w for w in aw if _keep(w)]
            else:
                logger.warning(f"⚠️ SET PARAMS FAILED: node='{node_id}' not found")

    elif name == "move_node":
        node_id = str(args.get("node_id") or "")
        x = float(args.get("x") or 0)
        y = float(args.get("y") or 0)
        n = find_node(node_id)
        if n:
            n["x"] = x
            n["y"] = y

    elif name == "connect":
        from_id = str(args.get("from_id") or "")
        to_id = str(args.get("to_id") or "")
        if from_id and to_id and from_id != to_id and find_node(from_id) and find_node(to_id):
            # Check for cycles: would this create a backward path?
            def would_create_cycle(src: str, dst: str) -> bool:
                """Check if dst can reach src (meaning adding src→dst would create cycle)."""
                visited = set()
                stack = [dst]
                while stack:
                    current = stack.pop()
                    if current == src:
                        return True
                    if current in visited:
                        continue
                    visited.add(current)
                    for c in conns:
                        if not isinstance(c, dict):
                            continue
                        c_from = str(c.get("from") or c.get("from_id") or "")
                        if c_from == current:
                            c_to = str(c.get("to") or c.get("to_id") or "")
                            if c_to:
                                stack.append(c_to)
                return False
            
            if would_create_cycle(from_id, to_id):
                msg = f"would create cycle ({from_id} → {to_id})"
                logger.warning(f"⚠️ CONNECT REJECTED: {msg}")
                snapshot["_last_tool_rejection"] = {
                    "tool": name,
                    "args": {"from_id": from_id, "to_id": to_id},
                    "reason": "cycle",
                    "message": msg,
                }
            # Enforce fan-in limits: reject only when a node already has >= its max allowed inputs
            elif (lambda mx: mx != -1 and in_degree(to_id) >= mx)(max_fanin(to_id)):
                existing_inputs: list[str] = []
                for c in conns:
                    if not isinstance(c, dict):
                        continue
                    c_to = str(c.get("to") or c.get("to_id") or "")
                    if c_to != to_id:
                        continue
                    c_from = str(c.get("from") or c.get("from_id") or "")
                    if c_from:
                        existing_inputs.append(c_from)

                # Generate suggestion for merge block
                suggestion = (
                    f"Use a 'merge' or 'concat' block to combine multiple inputs before '{to_id}'. "
                    f"Add a merge node, connect {existing_inputs} -> merge -> {to_id}."
                )
                msg = (
                    f"fan-in to '{to_id}' (type={node_type(to_id)}) not allowed; "
                    f"existing_inputs={existing_inputs}. {suggestion}"
                )
                logger.warning(f"⚠️ CONNECT REJECTED: {msg}")
                snapshot["_last_tool_rejection"] = {
                    "tool": name,
                    "args": {"from_id": from_id, "to_id": to_id},
                    "reason": "fan_in",
                    "message": msg,
                    "to_type": node_type(to_id),
                    "existing_inputs": existing_inputs,
                    "suggestion": suggestion,
                    "suggested_fix": {
                        "action": "add_merge",
                        "description": f"Add merge block to combine inputs before {to_id}",
                        "steps": [
                            f"Add a 'merge' block",
                            f"Disconnect: {existing_inputs} -> {to_id}",
                            f"Connect: {existing_inputs} -> merge",
                            f"Connect: merge -> {to_id}"
                        ]
                    }
                }
            else:
                conns.append({"from": from_id, "to": to_id})
                logger.info(f"🔗 CONNECT: '{from_id}' → '{to_id}'")
        else:
            msg = f"from='{from_id}' to='{to_id}' (invalid nodes or same id)"
            logger.warning(f"⚠️ CONNECT FAILED: {msg}")
            snapshot["_last_tool_rejection"] = {
                "tool": name,
                "args": {"from_id": from_id, "to_id": to_id},
                "reason": "invalid",
                "message": msg,
            }

    elif name == "disconnect":
        from_id = str(args.get("from_id") or "")
        to_id = str(args.get("to_id") or "")
        if from_id and to_id:
            # Find and remove the connection
            removed = False
            new_conns = []
            for c in conns:
                if not isinstance(c, dict):
                    new_conns.append(c)
                    continue
                c_from = str(c.get("from") or c.get("from_id") or "")
                c_to = str(c.get("to") or c.get("to_id") or "")
                if c_from == from_id and c_to == to_id:
                    removed = True
                    logger.info(f"✂️ DISCONNECT: '{from_id}' → '{to_id}'")
                else:
                    new_conns.append(c)
            conns = new_conns
            if not removed:
                logger.warning(f"⚠️ DISCONNECT FAILED: connection '{from_id}' → '{to_id}' not found")
        else:
            logger.warning(f"⚠️ DISCONNECT FAILED: missing from_id or to_id")

    snapshot["nodes"] = nodes
    snapshot["connections"] = conns
    
    return snapshot
