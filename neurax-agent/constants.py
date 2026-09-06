"""
Shared constants for the Neurax Agent pipeline.

This module has NO internal imports to avoid circular import issues.
"""

import json
from pathlib import Path

# Block types that can accept multiple incoming edges (fan-in capable merge nodes).
# Used by topology_validator.py (validation, and picking a block to
# auto-insert when a design needs one).
#
# Derived from block_constraints.json's `merge_capable_types` — the file
# `topology_validator` actually validates a design against — rather than a
# second hand-written list here. The two had already drifted in opposite
# directions (this list had `unet_block` but not `output_combination`; a
# third, separate list in catalogue_store.py had the reverse), so the same
# block type could be treated as merge-capable in one place and not another.
_CONSTRAINTS_FILE = Path(__file__).parent / "block_constraints.json"


def _load_merge_block_types() -> frozenset[str]:
    try:
        with open(_CONSTRAINTS_FILE) as f:
            data = json.load(f)
        types = data.get("merge_capable_types") or []
        if types:
            return frozenset(types)
    except Exception:
        pass
    # Fallback if the file is missing or malformed — keeps the pipeline
    # running with the last known-good set rather than an empty one.
    return frozenset({
        "concat", "merge", "add", "residual", "residual_add", "skip_connection",
        "expert_combine", "gate", "lm_head", "moe_block", "unet_block",
        "router_softmax", "output_combination",
    })


MERGE_BLOCK_TYPES: frozenset[str] = _load_merge_block_types()
