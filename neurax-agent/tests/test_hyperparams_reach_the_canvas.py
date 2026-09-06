"""What the agent sets as training config has to arrive on the canvas.

`initialize_hyperparams` derives a whole training configuration from the
design and takes no arguments, so the `tool` event it produced carried an
empty `args` — the studio had nothing to apply even once it handled the tool.
The agent's own snapshot gained a configuration the canvas never saw.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_graph


def _state(**over):
    state = {
        "run_id": "r", "user_message": "m",
        "snapshot": {"family": "transformer", "nodes": [{"id": "a", "type": "input"}],
                     "connections": [], "hw_config": {"batchSize": 16}},
        "history": [], "credentials": None, "search_api_key": None, "project_id": None,
        "core_memory": [], "plan_items": [], "mode": "creation",
        "allowed_tools": agent_graph.MODE_TOOL_GRANTS["creation"],
        "step_count": 0, "max_steps": 10, "max_expensive_calls": 3,
        "tool_call_counts": {}, "started_at": 0.0, "timeout_seconds": 60.0,
        "last_tool": None, "stop_reason": None, "events": [],
    }
    state.update(over)
    return state


def _tool_event(update):
    return next(e for e in update["events"] if e["event"] == "tool")


def test_initialize_hyperparams_emits_the_values_it_derived():
    state = _state(last_tool={"name": "initialize_hyperparams", "args": {}})
    update = asyncio.run(agent_graph.execute_tool(state))

    updates = _tool_event(update)["data"]["args"]["updates"]
    # The frontend applies `updates`; an empty one is the bug this covers.
    assert updates, "the derived configuration must travel with the event"
    assert updates["batch_size"] == 16, "derived from the canvas's own hw_config"
    assert "learning_rate" in updates and "warmup_steps" in updates
    # And it really did land on the agent's own snapshot too.
    assert update["snapshot"]["hyperparams"] == updates


def test_set_hyperparams_passes_the_models_own_updates_through():
    state = _state(last_tool={"name": "set_hyperparams", "args": {"updates": {"learning_rate": 3e-4}}})
    update = asyncio.run(agent_graph.execute_tool(state))

    # Already explicit, so it is forwarded unchanged — one shape for both
    # tools, the same `{updates: ...}` the frontend already reads.
    assert _tool_event(update)["data"]["args"]["updates"] == {"learning_rate": 3e-4}
    assert update["snapshot"]["hyperparams"]["learning_rate"] == 3e-4


def test_both_tools_are_granted_where_the_agent_can_use_them():
    # A tool the frontend now honours must actually be reachable.
    assert "initialize_hyperparams" in agent_graph.MODE_TOOL_GRANTS["creation"]
    assert "set_hyperparams" in agent_graph.MODE_TOOL_GRANTS["optimization"]
    # And never in the read-only mode.
    assert "set_hyperparams" not in agent_graph.MODE_TOOL_GRANTS["explanation"]
