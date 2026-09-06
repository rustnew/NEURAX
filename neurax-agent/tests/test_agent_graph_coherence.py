"""`agent_graph.py`'s coherence gate on `done` — the real gap this closes:

The old 3-phase pipeline planned a whole `ArchSpec` and ran
`topology_validator.validate_arch_spec` on it before a single tool call is
emitted. The new step-by-step loop emits tool calls one at a time with no
such gate anywhere — before this, the model could call `done` on a design
with a block wired to nothing, or no path from input to output at all, and
nothing would say so. `execute_tool`'s `done` handling now runs the same
validator before accepting it.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_graph


def _drain(q):
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    return events


def _run_with_fixed_snapshot(monkeypatch, snapshot, mode, max_steps=5):
    """The fake planner always proposes `done` — a real agent's plan, just
    collapsed to the one call this file is testing the reaction to."""
    async def fake(**kw):
        return {"assistant": "finishing", "tool": {"name": "done", "args": {}}}

    monkeypatch.setattr(agent_graph, "run_controller_step", fake)

    async def go():
        q: asyncio.Queue = asyncio.Queue()
        await agent_graph.run_agent_graph(
            "test-run", q, "build it", snapshot, mode=mode, max_steps=max_steps
        )
        return _drain(q)

    return asyncio.run(go())


_ORPHAN_SNAPSHOT = {
    "nodes": [
        {"id": "in", "type": "input"},
        {"id": "out", "type": "output"},
        {"id": "stray", "type": "conv2d"},  # connected to nothing
    ],
    "connections": [{"from": "in", "to": "out"}],
}

_COHERENT_SNAPSHOT = {
    "nodes": [
        {"id": "in", "type": "input"},
        {"id": "conv", "type": "conv2d"},
        {"id": "out", "type": "output"},
    ],
    "connections": [{"from": "in", "to": "conv"}, {"from": "conv", "to": "out"}],
}

_CYCLE_SNAPSHOT = {
    "nodes": [
        {"id": "in", "type": "input"},
        {"id": "a", "type": "conv2d"},
        {"id": "b", "type": "conv2d"},
        {"id": "out", "type": "output"},
    ],
    "connections": [
        {"from": "in", "to": "a"}, {"from": "a", "to": "b"},
        {"from": "b", "to": "a"},  # a <-> b cycle
        {"from": "b", "to": "out"},
    ],
}


def test_done_is_refused_when_a_block_is_connected_to_nothing(monkeypatch):
    events = _run_with_fixed_snapshot(monkeypatch, _ORPHAN_SNAPSHOT, mode="creation")
    result_events = [e for e in events if e["event"] == "tool_result" and e["data"]["tool"] == "done"]
    assert result_events, "the refused 'done' must be narrated back, not silently swallowed"
    assert "isn't coherent" in result_events[0]["data"]["content"].lower()
    assert "stray" in result_events[0]["data"]["content"]


def test_done_is_refused_on_a_cycle(monkeypatch):
    events = _run_with_fixed_snapshot(monkeypatch, _CYCLE_SNAPSHOT, mode="creation")
    result_events = [e for e in events if e["event"] == "tool_result" and e["data"]["tool"] == "done"]
    assert result_events
    assert "cycle" in result_events[0]["data"]["content"].lower()


def test_done_is_accepted_on_a_genuinely_coherent_design(monkeypatch):
    events = _run_with_fixed_snapshot(monkeypatch, _COHERENT_SNAPSHOT, mode="creation")
    tool_events = [e for e in events if e["event"] == "tool"]
    # Exactly one 'done' tool event, no refusal — the run finishes on the
    # very first step, the same as before this gate existed.
    assert len(tool_events) == 1
    assert tool_events[0]["data"]["name"] == "done"
    assert events[-1]["event"] == "done"


def test_an_empty_canvas_is_never_checked_nothing_was_built_yet(monkeypatch):
    events = _run_with_fixed_snapshot(
        monkeypatch, {"nodes": [], "connections": []}, mode="creation"
    )
    tool_events = [e for e in events if e["event"] == "tool"]
    assert len(tool_events) == 1
    assert tool_events[0]["data"]["name"] == "done"


def test_optimization_mode_is_never_gated_on_coherence_it_could_not_have_caused(monkeypatch):
    # optimization mode has no add_node/connect grant at all — a pre-existing
    # orphan block is not something this run could have introduced or fixed,
    # so 'done' must not be blocked on it here.
    events = _run_with_fixed_snapshot(monkeypatch, _ORPHAN_SNAPSHOT, mode="optimization")
    tool_events = [e for e in events if e["event"] == "tool"]
    assert len(tool_events) == 1
    assert tool_events[0]["data"]["name"] == "done"


def test_explanation_mode_is_never_gated_on_coherence_either(monkeypatch):
    events = _run_with_fixed_snapshot(monkeypatch, _ORPHAN_SNAPSHOT, mode="explanation")
    tool_events = [e for e in events if e["event"] == "tool"]
    assert len(tool_events) == 1
    assert tool_events[0]["data"]["name"] == "done"


def test_a_refused_done_does_not_mutate_the_snapshot(monkeypatch):
    async def fake(**kw):
        return {"assistant": "finishing", "tool": {"name": "done", "args": {}}}

    monkeypatch.setattr(agent_graph, "run_controller_step", fake)

    snapshot = {k: v for k, v in _ORPHAN_SNAPSHOT.items()}
    snapshot["nodes"] = [dict(n) for n in _ORPHAN_SNAPSHOT["nodes"]]
    snapshot["connections"] = [dict(c) for c in _ORPHAN_SNAPSHOT["connections"]]
    before = (list(snapshot["nodes"]), list(snapshot["connections"]))

    async def go():
        q: asyncio.Queue = asyncio.Queue()
        await agent_graph.run_agent_graph("r", q, "build it", snapshot, mode="creation", max_steps=2)

    asyncio.run(go())
    assert (snapshot["nodes"], snapshot["connections"]) == before


def test_the_run_still_stops_at_the_step_limit_if_the_model_never_fixes_it(monkeypatch):
    # A model that keeps calling `done` on an unfixed incoherent design must
    # not loop forever — the existing step ceiling has to still apply. A
    # refused `done` produces a `tool_result` (informational), not a `tool`
    # event — the same shape a "not available in this mode" refusal already
    # uses; it never actually ran, so there is nothing to report as ran.
    events = _run_with_fixed_snapshot(monkeypatch, _ORPHAN_SNAPSHOT, mode="creation", max_steps=3)
    refusals = [e for e in events if e["event"] == "tool_result" and e["data"]["tool"] == "done"]
    assert len(refusals) == 3
    assert events[-1]["event"] == "done"


def test_fixing_the_design_after_a_refusal_lets_done_succeed(monkeypatch):
    calls = {"n": 0}

    async def fake(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"assistant": "finishing", "tool": {"name": "done", "args": {}}}
        if calls["n"] == 2:
            # "Fixing" the design the simplest possible way: remove the
            # block that was connected to nothing, rather than rewiring it
            # in (which would need a merge block — 'out' already has its
            # one allowed input from 'in' in this fixture, a different,
            # unrelated concern this test isn't about).
            return {"assistant": "removing the stray block", "tool": {"name": "delete_node", "args": {"node_id": "stray"}}}
        return {"assistant": "finishing for real", "tool": {"name": "done", "args": {}}}

    monkeypatch.setattr(agent_graph, "run_controller_step", fake)

    snapshot = {
        "nodes": [dict(n) for n in _ORPHAN_SNAPSHOT["nodes"]],
        "connections": [dict(c) for c in _ORPHAN_SNAPSHOT["connections"]],
    }

    async def go():
        q: asyncio.Queue = asyncio.Queue()
        await agent_graph.run_agent_graph("r", q, "build it", snapshot, mode="creation", max_steps=5)
        return _drain(q)

    events = asyncio.run(go())
    tool_events = [e for e in events if e["event"] == "tool"]
    # The first 'done' is refused (no 'tool' event, see above); 'delete_node'
    # really runs; the second 'done' really finishes the run.
    assert [e["data"]["name"] for e in tool_events] == ["delete_node", "done"]
    refusals = [e for e in events if e["event"] == "tool_result" and e["data"]["tool"] == "done"]
    assert len(refusals) == 1, "only the first (refused) done produces a tool_result"
