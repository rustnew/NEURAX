#!/usr/bin/env python3
"""
Local catalogue store for Neurax agent.

Catalogue is organized by family as key, matching the frontend display.
Each family contains:
- description
- taskHints (keywords for family selection)
- defaultInputShape
- blocks (with type, name, defaultParams)
- macroBlocks
- presets
- constraints

Usage:
    from catalogue_store import get_families, get_catalogue_for_family
    families = get_families()
    cnn_catalogue = get_catalogue_for_family("cnn")
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Path to catalogue file
_CATALOGUE_PATH = Path(__file__).parent / "catalogue.json"

# In-memory cache
_catalogue_cache: dict[str, Any] | None = None


def _load_catalogue() -> dict[str, Any]:
    """Load catalogue from JSON file."""
    global _catalogue_cache
    
    if _catalogue_cache is not None:
        return _catalogue_cache
    
    if not _CATALOGUE_PATH.exists():
        logger.warning(f"⚠️ Catalogue file not found: {_CATALOGUE_PATH}")
        return {}
    
    try:
        with open(_CATALOGUE_PATH, 'r') as f:
            _catalogue_cache = json.load(f)
        logger.info(f"📦 Loaded catalogue for {len(_catalogue_cache)} families")
        return _catalogue_cache
    except Exception as e:
        logger.error(f"❌ Error loading catalogue: {e}")
        return {}


def get_families() -> list[str]:
    """Get list of available families."""
    catalogue = _load_catalogue()
    return list(catalogue.keys())


def get_family_info(family: str) -> dict[str, Any] | None:
    """Get full info for a family (description, taskHints, blocks, etc.)."""
    catalogue = _load_catalogue()
    return catalogue.get(family.lower())


# Block types that can accept multiple inputs (fan-in allowed)
# -1 means unlimited inputs, 1 means single input only
FAN_IN_CAPABLE_TYPES = {
    "concat", "residual_add", "skip_connection", "residual", "merge", "add",
    "router_softmax", "gate", "lm_head", "expert_combine",
    "output_combination", "moe_block"
}


def _get_max_inputs(block_type: str) -> int:
    """Get max inputs for a block type. -1 = unlimited, 1 = single input."""
    return -1 if block_type.lower() in FAN_IN_CAPABLE_TYPES else 1


def get_catalogue_for_family(family: str) -> list[dict[str, Any]]:
    """Get blocks for a specific family (format compatible with agent)."""
    family_data = get_family_info(family)
    if not family_data:
        return []
    
    blocks = family_data.get("blocks", [])
    
    # Ensure each block has the expected format
    formatted_blocks = []
    for block in blocks:
        block_type = block.get("type", "")
        formatted_blocks.append({
            "type": block_type,
            "name": block.get("name", block.get("type")),
            "defaultParams": block.get("defaultParams", {}),
            "maxInputs": _get_max_inputs(block_type),
            "family": family.lower()
        })
    
    return formatted_blocks


def get_all_blocks() -> list[dict[str, Any]]:
    """Get all blocks from all families (flattened)."""
    catalogue = _load_catalogue()
    all_blocks = []
    
    for family, data in catalogue.items():
        for block in data.get("blocks", []):
            block_type = block.get("type", "")
            all_blocks.append({
                "type": block_type,
                "name": block.get("name", block.get("type")),
                "defaultParams": block.get("defaultParams", {}),
                "maxInputs": _get_max_inputs(block_type),
                "family": family
            })
    
    return all_blocks


def get_task_hints_for_family(family: str) -> list[str]:
    """Get task hints for family selection."""
    family_data = get_family_info(family)
    if not family_data:
        return []
    return family_data.get("taskHints", [])


def get_default_input_shape(family: str) -> str:
    """Get default input shape for a family."""
    family_data = get_family_info(family)
    if not family_data:
        return "[B, ...]"
    return family_data.get("defaultInputShape", "[B, ...]")


def get_family_constraints(family: str) -> dict[str, Any]:
    """Get constraints for a family."""
    family_data = get_family_info(family)
    if not family_data:
        return {"requiredBlocks": ["input", "output"], "incompatibleBlocks": []}
    return family_data.get("constraints", {"requiredBlocks": ["input", "output"], "incompatibleBlocks": []})


def get_presets_for_family(family: str) -> list[dict[str, Any]]:
    """Get presets for a family."""
    family_data = get_family_info(family)
    if not family_data:
        return []
    return family_data.get("presets", [])


def get_macro_blocks_for_family(family: str) -> list[dict[str, Any]]:
    """Get macro blocks for a family."""
    family_data = get_family_info(family)
    if not family_data:
        return []
    return family_data.get("macroBlocks", [])


def find_best_family_for_task(task_description: str) -> str | None:
    """Find the best family for a task based on task hints."""
    catalogue = _load_catalogue()
    task_lower = task_description.lower()
    
    best_family = None
    best_score = 0
    
    for family, data in catalogue.items():
        task_hints = data.get("taskHints", [])
        score = sum(1 for hint in task_hints if hint.lower() in task_lower)
        
        if score > best_score:
            best_score = score
            best_family = family
    
    return best_family


# Backwards compatibility - get_catalogue returns all blocks
def get_catalogue() -> list[dict[str, Any]]:
    """Get all blocks (backwards compatible)."""
    return get_all_blocks()


# Test the module
if __name__ == "__main__":
    print("=== Available Families ===")
    print(get_families())
    
    print("\n=== CNN Blocks ===")
    cnn_blocks = get_catalogue_for_family("cnn")
    for b in cnn_blocks[:5]:
        print(f"  - {b['type']}: {b['name']}")
    print(f"  ... ({len(cnn_blocks)} total)")
    
    print("\n=== CNN Task Hints ===")
    print(get_task_hints_for_family("cnn"))
    
    print("\n=== CNN Default Input Shape ===")
    print(get_default_input_shape("cnn"))
    
    print("\n=== Find Best Family ===")
    print(find_best_family_for_task("Build a CNN for image classification with 10 classes"))
