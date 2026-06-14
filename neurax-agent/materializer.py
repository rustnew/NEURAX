"""
Materializer — Phase 3 of the declarative agent pipeline.

Converts a validated ArchSpec + positions map into an ordered stream of
canvas tool call dicts. The sequence is deterministic and ordered so that
all `add_node` calls precede `connect` calls (no forward reference issues).
"""
from __future__ import annotations

import logging
from typing import AsyncGenerator

from topology_validator import ArchSpec
from layout_engine import assign_positions

logger = logging.getLogger(__name__)


def _tool(name: str, args: dict) -> dict:
    return {"name": name, "args": args}


async def materialize(
    spec: ArchSpec,
    positions: dict[str, tuple[float, float]] | None = None,
) -> AsyncGenerator[dict, None]:
    """
    Yield canvas tool call dicts for a validated ArchSpec.

    Tool call order:
        1. clear_canvas
        2. set_family
        3. add_node × N   (all nodes first, in topological order)
        4. set_node_params × N  (params for each node that has them)
        5. connect × M    (all edges after all nodes exist)
        6. done

    Args:
        spec:      Validated ArchSpec
        positions: Optional pre-computed positions. If None, layout_engine is called.

    Yields:
        dict with keys "name" and "args" — compatible with snapshot_ops._apply_tool_to_snapshot
    """
    if positions is None:
        positions = assign_positions(spec)

    logger.info(f"🔧 Materializing: {len(spec.nodes)} nodes, {len(spec.edges)} edges")

    # 1. Clear canvas
    yield _tool("clear_canvas", {})

    # 2. Set family
    yield _tool("set_family", {"family": spec.family})

    # 3. Add all nodes (in topological order from positions keys)
    # Sort nodes so input comes first, output comes last
    def _node_sort_key(node):
        if node.type == "input":
            return (0, node.id)
        if node.type == "output":
            return (2, node.id)
        return (1, positions.get(node.id, (0, 0))[0])  # sort by x position

    sorted_nodes = sorted(spec.nodes, key=_node_sort_key)

    for node in sorted_nodes:
        x, y = positions.get(node.id, (0.0, 0.0))
        yield _tool("add_node", {
            "layer_type": node.type,
            "node_id": node.id,
            "x": x,
            "y": y,
        })

        # 4. Set params immediately after adding node
        if node.params:
            yield _tool("set_node_params", {
                "node_id": node.id,
                "updates": node.params,
            })

    # 5. Connect all edges
    for edge in spec.edges:
        yield _tool("connect", {
            "from_id": edge.from_id,
            "to_id": edge.to_id,
        })

    # 6. Done
    logger.info(f"✅ Materialization complete")
    yield _tool("done", {})
