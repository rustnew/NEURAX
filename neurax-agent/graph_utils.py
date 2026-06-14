"""Graph utilities for analyzing canvas topology."""
from typing import Any, Optional


def _node_type(n: dict[str, Any]) -> str:
    return str(n.get("type") or "").lower()


def _graph_sets(snapshot: dict[str, Any]) -> tuple[set[str], set[str], dict[str, set[str]], dict[str, set[str]]]:
    nodes: list[dict[str, Any]] = list(snapshot.get("nodes") or [])
    conns: list[dict[str, Any]] = list(snapshot.get("connections") or [])

    node_ids = {str(n.get("id")) for n in nodes if n.get("id")}
    fwd: dict[str, set[str]] = {nid: set() for nid in node_ids}
    rev: dict[str, set[str]] = {nid: set() for nid in node_ids}

    for c in conns:
        if not isinstance(c, dict):
            continue
        a = str(c.get("from") or c.get("from_id") or "")
        b = str(c.get("to") or c.get("to_id") or "")
        if not a or not b or a == b:
            continue
        if a not in node_ids or b not in node_ids:
            continue
        fwd[a].add(b)
        rev[b].add(a)

    inputs = {str(n.get("id")) for n in nodes if n.get("id") and _node_type(n) == "input"}
    outputs = {str(n.get("id")) for n in nodes if n.get("id") and _node_type(n) == "output"}
    return inputs, outputs, fwd, rev


def _reachable(starts: set[str], adj: dict[str, set[str]]) -> set[str]:
    seen: set[str] = set()
    stack = list(starts)
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for nxt in adj.get(cur, set()):
            if nxt not in seen:
                stack.append(nxt)
    return seen


def _main_flow_nodes(snapshot: dict[str, Any]) -> set[str]:
    inputs, outputs, fwd, rev = _graph_sets(snapshot)
    if not inputs or not outputs:
        return set()
    from_in = _reachable(inputs, fwd)
    to_out = _reachable(outputs, rev)
    return from_in.intersection(to_out)


def _orphan_nodes(snapshot: dict[str, Any]) -> list[str]:
    nodes: list[dict[str, Any]] = list(snapshot.get("nodes") or [])
    flow = _main_flow_nodes(snapshot)
    out: list[str] = []
    for n in nodes:
        nid = str(n.get("id") or "")
        if not nid:
            continue
        t = _node_type(n)
        if t in {"input", "output"}:
            continue
        if nid not in flow:
            out.append(nid)
    return out


def _has_orphans(snapshot: dict[str, Any]) -> bool:
    return len(_orphan_nodes(snapshot)) > 0


def _has_input_to_output_path(snapshot: dict[str, Any]) -> bool:
    nodes = snapshot.get("nodes") or []
    conns = snapshot.get("connections") or []
    if not isinstance(nodes, list) or not isinstance(conns, list):
        return False

    input_ids = [str(n.get("id")) for n in nodes if str(n.get("type")) == "input" and n.get("id")]
    output_ids = {str(n.get("id")) for n in nodes if str(n.get("type")) == "output" and n.get("id")}
    if not input_ids or not output_ids:
        return False

    adj: dict[str, list[str]] = {}
    for c in conns:
        if not isinstance(c, dict):
            continue
        a = c.get("from")
        b = c.get("to")
        if not a or not b:
            continue
        a_id = str(a)
        b_id = str(b)
        adj.setdefault(a_id, []).append(b_id)

    seen: set[str] = set()
    stack: list[str] = list(input_ids)
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if cur in output_ids:
            return True
        for nxt in adj.get(cur, []):
            if nxt not in seen:
                stack.append(nxt)
    return False


def _unique_node_id(existing_ids: set[str], base: str) -> str:
    if base not in existing_ids:
        return base
    i = 2
    while f"{base}_{i}" in existing_ids:
        i += 1
    return f"{base}_{i}"


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
