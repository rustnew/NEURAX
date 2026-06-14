"""Automated tests for suggestions.py — checks MANDATORY_FIELDS alignment and topology helpers."""
import pytest
from suggestions import _suggest_hw_defaults, _suggest_fix_from_warnings
from snapshot_ops import _apply_tool_to_snapshot


# ────────────────────────────────────────────────────────────────────────────
# _suggest_hw_defaults — must use MANDATORY_FIELDS keys exactly
# ────────────────────────────────────────────────────────────────────────────

def test_hw_defaults_transformer_mandatory_keys():
    """Fills seqLen / numLayers / vocabSize — the actual MANDATORY_FIELDS keys."""
    snapshot = {
        "family": "transformer",
        "hw_config": {"hardware": "RTX4090", "precision": "fp16", "batchSize": 1},
        "missing_mandatory_fields": ["seqLen", "numLayers", "vocabSize"],
    }
    result = _suggest_hw_defaults(snapshot, "build a gpt-2 transformer")
    assert result is not None, "Expected set_hw_config for transformer with missing mandatory fields"
    updates = result["args"]["updates"]
    assert "seqLen" in updates,   f"seqLen missing from {updates}"
    assert "numLayers" in updates, f"numLayers missing from {updates}"
    assert "vocabSize" in updates, f"vocabSize missing from {updates}"
    assert updates["seqLen"] > 0
    assert updates["numLayers"] > 0
    assert updates["vocabSize"] > 0


def test_hw_defaults_moe_mandatory_keys():
    """Fills seqLen / numLayers / vocabSize / numExperts / topK for MoE."""
    snapshot = {
        "family": "moe",
        "hw_config": {"hardware": "RTX4090", "precision": "fp16", "batchSize": 1},
        "missing_mandatory_fields": ["seqLen", "numLayers", "vocabSize", "numExperts", "topK"],
    }
    result = _suggest_hw_defaults(snapshot, "build a mixture-of-experts model")
    assert result is not None
    updates = result["args"]["updates"]
    for key in ("seqLen", "numLayers", "vocabSize", "numExperts", "topK"):
        assert key in updates, f"{key} missing from {updates}"
        assert updates[key] > 0


def test_hw_defaults_gnn_mandatory_keys():
    """Fills numNodes / numEdges / nodeFeatDim for GNN."""
    snapshot = {
        "family": "gnn",
        "hw_config": {"hardware": "RTX4090", "precision": "fp32", "batchSize": 1},
        "missing_mandatory_fields": ["numNodes", "numEdges", "nodeFeatDim"],
    }
    result = _suggest_hw_defaults(snapshot, "build a graph neural network")
    assert result is not None
    updates = result["args"]["updates"]
    for key in ("numNodes", "numEdges", "nodeFeatDim"):
        assert key in updates, f"{key} missing from {updates}"


def test_hw_defaults_diffusion_mandatory_keys():
    """Fills numDenoisingSteps / inChannels for Diffusion."""
    snapshot = {
        "family": "diffusion",
        "hw_config": {"hardware": "A100", "precision": "fp16", "batchSize": 1},
        "missing_mandatory_fields": ["numDenoisingSteps", "inChannels"],
    }
    result = _suggest_hw_defaults(snapshot, "build a diffusion model")
    assert result is not None
    updates = result["args"]["updates"]
    assert "numDenoisingSteps" in updates
    assert "inChannels" in updates


def test_hw_defaults_ssm_mandatory_keys():
    """Fills dState (the only mandatory SSM field)."""
    snapshot = {
        "family": "ssm",
        "hw_config": {"hardware": "RTX4090", "precision": "bf16", "batchSize": 1},
        "missing_mandatory_fields": ["dState"],
    }
    result = _suggest_hw_defaults(snapshot, "build a mamba SSM")
    assert result is not None
    updates = result["args"]["updates"]
    assert "dState" in updates and updates["dState"] > 0


def test_hw_defaults_cnn_mandatory_keys():
    """Fills inChannels for CNN."""
    snapshot = {
        "family": "cnn",
        "hw_config": {"hardware": "RTX4090", "precision": "fp32", "batchSize": 1},
        "missing_mandatory_fields": ["inChannels"],
    }
    result = _suggest_hw_defaults(snapshot, "build a resnet")
    assert result is not None
    updates = result["args"]["updates"]
    assert "inChannels" in updates


def test_hw_defaults_returns_none_when_nothing_missing():
    """Returns None when missing_mandatory_fields is empty."""
    snapshot = {
        "family": "transformer",
        "hw_config": {"hardware": "RTX4090", "precision": "fp16", "batchSize": 1},
        "missing_mandatory_fields": [],
    }
    assert _suggest_hw_defaults(snapshot, "build a transformer") is None


# ────────────────────────────────────────────────────────────────────────────
# snapshot_ops — partial pruning of missing_mandatory_fields
# ────────────────────────────────────────────────────────────────────────────

def test_partial_hw_config_prunes_only_updated_keys():
    """After set_hw_config with seqLen only, numLayers / vocabSize stay in missing list."""
    snapshot = {
        "family": "transformer",
        "hw_config": {"hardware": "RTX4090", "precision": "fp16", "batchSize": 1},
        "missing_mandatory_fields": ["seqLen", "numLayers", "vocabSize"],
        "nodes": [],
        "connections": [],
    }
    tool = {"name": "set_hw_config", "args": {"updates": {"seqLen": 2048}}}
    result = _apply_tool_to_snapshot(snapshot, tool)
    remaining = result["missing_mandatory_fields"]
    assert "seqLen" not in remaining, "seqLen should be pruned after update"
    assert "numLayers" in remaining, "numLayers should still be missing"
    assert "vocabSize" in remaining, "vocabSize should still be missing"


def test_full_hw_config_clears_all():
    """After filling all mandatory fields, missing_mandatory_fields becomes empty."""
    snapshot = {
        "family": "transformer",
        "hw_config": {"hardware": "RTX4090", "precision": "fp16", "batchSize": 1},
        "missing_mandatory_fields": ["seqLen", "numLayers", "vocabSize"],
        "nodes": [],
        "connections": [],
    }
    tool = {"name": "set_hw_config", "args": {"updates": {"seqLen": 2048, "numLayers": 12, "vocabSize": 50257}}}
    result = _apply_tool_to_snapshot(snapshot, tool)
    assert result["missing_mandatory_fields"] == []


# ────────────────────────────────────────────────────────────────────────────
# _suggest_fix_from_warnings — batching and guard logic
# ────────────────────────────────────────────────────────────────────────────

def test_suggest_fix_from_warnings_batching():
    """All missing fields for the same node batched into ONE set_node_params call."""
    snapshot = {
        "family": "transformer",
        "nodes": [{"id": "node-1", "type": "mha_attention", "params": {}}],
        "analysis_warnings": [
            {"type": "error", "code": "E_MISSING_HYPERPARAMETER",
             "message": "Mandatory Hyperparameter: d_model", "nodeId": "node-1"},
            {"type": "error", "code": "E_MISSING_HYPERPARAMETER",
             "message": "Mandatory Hyperparameter: n_heads", "nodeId": "node-1"},
        ],
    }
    result = _suggest_fix_from_warnings(snapshot)
    assert result is not None
    assert result["name"] == "set_node_params"
    assert result["args"]["node_id"] == "node-1"
    updates = result["args"]["updates"]
    assert "d_model" in updates and "n_heads" in updates


def test_suggest_fix_from_warnings_no_errors():
    snapshot = {"family": "transformer", "nodes": [], "analysis_warnings": []}
    assert _suggest_fix_from_warnings(snapshot) is None


def test_suggest_fix_from_warnings_already_set():
    """Does not re-patch params already set to a non-zero value."""
    snapshot = {
        "family": "transformer",
        "nodes": [{"id": "n1", "type": "mha_attention", "params": {"d_model": 512}}],
        "analysis_warnings": [
            {"type": "error", "code": "E_MISSING_HYPERPARAMETER",
             "message": "Mandatory Hyperparameter: d_model", "nodeId": "n1"},
        ],
    }
    assert _suggest_fix_from_warnings(snapshot) is None
