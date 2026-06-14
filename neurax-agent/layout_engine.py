"""
Layout Engine — deterministic topological-sort-based position assignment.

Given a validated ArchSpec, assigns pixel (x, y) coordinates to every
node with zero collisions. Parallel branches at the same depth are
staggered vertically.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any

from topology_validator import ArchSpec

logger = logging.getLogger(__name__)

# Canvas pixel spacing constants
SPACING_X: float = 100.0   # horizontal pixels per topology depth level
SPACING_Y: float = 80.0    # vertical pixels per parallel branch slot


def _compute_depths(spec: ArchSpec) -> dict[str, int]:
    """
    Compute the depth (topological level) of each node via BFS from input nodes.
    Nodes at depth 0 are input nodes. Each subsequent layer increments depth by 1.
    For nodes with multiple incoming edges, depth = max(predecessor depths) + 1.
    """
    node_ids = {n.id for n in spec.nodes}

    # Build adjacency
    fwd: dict[str, list[str]] = {n.id: [] for n in spec.nodes}
    in_degree: dict[str, int] = {n.id: 0 for n in spec.nodes}

    for edge in spec.edges:
        if edge.from_id in node_ids and edge.to_id in node_ids:
            fwd[edge.from_id].append(edge.to_id)
            in_degree[edge.to_id] += 1

    # Kahn's BFS for topological depth
    depths: dict[str, int] = {n.id: 0 for n in spec.nodes}
    queue: deque[str] = deque(nid for nid, d in in_degree.items() if d == 0)

    while queue:
        cur = queue.popleft()
        for nxt in fwd.get(cur, []):
            new_depth = depths[cur] + 1
            if new_depth > depths[nxt]:
                depths[nxt] = new_depth
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    return depths


def _topological_order(spec: ArchSpec) -> list[str]:
    """Return node IDs in topological order (input nodes first)."""
    node_ids = {n.id for n in spec.nodes}
    fwd: dict[str, list[str]] = {n.id: [] for n in spec.nodes}
    in_degree: dict[str, int] = {n.id: 0 for n in spec.nodes}

    for edge in spec.edges:
        if edge.from_id in node_ids and edge.to_id in node_ids:
            fwd[edge.from_id].append(edge.to_id)
            in_degree[edge.to_id] += 1

    # BFS order — input nodes first, then by discovery
    order: list[str] = []
    queue: deque[str] = deque()

    # Seed with input nodes first so they always appear at x=0
    for n in spec.nodes:
        if n.type == "input":
            queue.append(n.id)

    # Add remaining source nodes (should not happen in valid specs, but be safe)
    for nid, d in in_degree.items():
        if d == 0 and nid not in {n.id for n in spec.nodes if n.type == "input"}:
            queue.append(nid)

    visited: set[str] = set()
    while queue:
        cur = queue.popleft()
        if cur in visited:
            continue
        visited.add(cur)
        order.append(cur)
        for nxt in fwd.get(cur, []):
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    # Append any unreachable nodes at the end (orphans, handled by validator)
    for n in spec.nodes:
        if n.id not in visited:
            order.append(n.id)

    return order


def assign_positions(spec: ArchSpec) -> dict[str, tuple[float, float]]:
    """
    Assign pixel (x, y) coordinates to every node in the spec.

    Strategy:
    - Compute topological depth for each node
    - x = depth × SPACING_X   (nodes at same depth share same x column)
    - y = slot × SPACING_Y     (parallel nodes at same depth staggered vertically)
    - Input node(s) always at y=0 in their column
    - Output node(s) always at y=0 in their column (final column)

    Returns:
        dict mapping node_id → (x, y) in canvas pixel coordinates
    """
    if not spec.nodes:
        return {}

    depths = _compute_depths(spec)
    topo_order = _topological_order(spec)

    # Count nodes per depth to centre them vertically
    nodes_at_depth: dict[int, list[str]] = {}
    for nid in topo_order:
        d = depths.get(nid, 0)
        nodes_at_depth.setdefault(d, []).append(nid)

    positions: dict[str, tuple[float, float]] = {}

    for depth, node_ids_at_depth in nodes_at_depth.items():
        n = len(node_ids_at_depth)
        # Centre the column vertically around y=0
        total_height = (n - 1) * SPACING_Y
        start_y = -total_height / 2

        for slot, nid in enumerate(node_ids_at_depth):
            x = depth * SPACING_X
            y = start_y + slot * SPACING_Y
            positions[nid] = (round(x, 1), round(y, 1))

    return positions


def positions_summary(positions: dict[str, tuple[float, float]]) -> str:
    """Human-readable summary of assigned positions (for logging)."""
    lines = [f"  {nid}: ({x:.0f}, {y:.0f})" for nid, (x, y) in sorted(positions.items())]
    return "\n".join(lines)
