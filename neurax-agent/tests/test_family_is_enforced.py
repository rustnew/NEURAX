"""The agent may only work in a family that exists end to end.

`set_family` accepted any string at all, and nothing downstream caught it:
`get_catalogue_for_family` returns no blocks for an unknown family and
`get_family_constraints` falls back to a permissive default, so the loop kept
reasoning about a family the canvas never adopted — the studio validates
`set_family` and drops it, so the agent's snapshot said one thing and the real
canvas another, silently, until `/analyze` refused the design with
`Invalid model type`.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import agent_graph
import catalogue_store
import snapshot_ops

SUPPORTED = sorted(catalogue_store.get_families())


def _set_family(family, snapshot=None):
    snapshot = dict(snapshot or {"family": "cnn", "nodes": [], "connections": []})
    return snapshot_ops._apply_tool_to_snapshot(
        snapshot, {"name": "set_family", "args": {"family": family}}
    )


def test_the_eight_supported_families_are_accepted():
    assert len(SUPPORTED) == 8
    for family in SUPPORTED:
        out = _set_family(family)
        assert out["family"] == family
        assert "_last_tool_rejection" not in out


@pytest.mark.parametrize("family", ["snn", "rl", "multimodal", "experimental", "quantum"])
def test_anything_else_is_rejected_and_the_family_is_left_alone(family):
    out = _set_family(family)

    # Not silently applied, and not silently ignored either.
    assert out["family"] == "cnn", "an unsupported family must not replace the current one"
    rejection = out.get("_last_tool_rejection")
    assert rejection is not None, f"'{family}' should have been rejected"
    assert rejection["reason"] == "unknown_family"
    # The message has to name the real options, or the model has nothing to
    # correct towards on its next step.
    for supported in SUPPORTED:
        assert supported in rejection["message"]


def test_the_caller_s_own_declared_list_wins_over_the_catalogue():
    # `Index.tsx::agentGetSnapshot` sends `allowed_families`; a caller that
    # supports fewer families must not have the catalogue overrule it.
    snapshot = {"family": "cnn", "nodes": [], "connections": [], "allowed_families": ["cnn", "moe"]}

    assert _set_family("moe", snapshot)["family"] == "moe"

    out = _set_family("transformer", snapshot)
    assert out["family"] == "cnn"
    assert out["_last_tool_rejection"]["reason"] == "unknown_family"


def test_the_rejection_reaches_the_model_as_its_own_failed_step():
    """A rejection nobody reads is the same as no rejection at all."""
    state = {
        "run_id": "r", "user_message": "build an snn", "history": [],
        "snapshot": {"family": "cnn", "nodes": [], "connections": []},
        "credentials": None, "search_api_key": None, "project_id": None,
        "core_memory": [], "plan_items": [], "mode": "creation",
        "allowed_tools": agent_graph.MODE_TOOL_GRANTS["creation"],
        "step_count": 0, "max_steps": 10, "max_expensive_calls": 3,
        "tool_call_counts": {}, "started_at": 0.0, "timeout_seconds": 60.0,
        "last_tool": {"name": "set_family", "args": {"family": "snn"}},
        "stop_reason": None, "events": [],
    }
    update = asyncio.run(agent_graph.execute_tool(state))

    said = "\n".join(h["content"] for h in update["history"] if h["role"] == "system")
    assert "rejected" in said and "snn" in said
    assert update["snapshot"]["family"] == "cnn"
    # And it never travels onward as canvas state.
    assert "_last_tool_rejection" not in update["snapshot"]


def test_a_block_outside_the_family_still_blocks_done():
    """The other half of the same rule, already enforced: family-foreign blocks."""
    snapshot = {
        "family": "cnn",
        "nodes": [
            {"id": "in", "type": "input", "params": {}},
            {"id": "lif", "type": "lif_neuron", "params": {}},
            {"id": "out", "type": "output", "params": {}},
        ],
        "connections": [{"from": "in", "to": "lif"}, {"from": "lif", "to": "out"}],
        "catalogue": catalogue_store.get_catalogue_for_family("cnn"),
    }
    errors = agent_graph._validate_canvas_coherence(snapshot)
    assert any("lif_neuron" in e and "catalogue" in e for e in errors)
