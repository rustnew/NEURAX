"""
Shared constants for the Neurax Agent pipeline.

This module has NO internal imports to avoid circular import issues.
"""

# Block types that can accept multiple incoming edges (fan-in capable merge nodes).
# Used by both arch_planner.py (prompt generation) and topology_validator.py (validation).
MERGE_BLOCK_TYPES: frozenset[str] = frozenset({
    "concat",
    "merge",
    "add",
    "residual",
    "residual_add",
    "skip_connection",
    "expert_combine",
    "gate",
    "lm_head",
    "moe_block",
    "unet_block",
    "router_softmax",
})
