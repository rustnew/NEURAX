"""`analysis_tools.py` — the compiler-facing tools the step-by-step loop
(`agent_graph.py`) can call.

Three tools (`analyze_architecture`, `check_budget`,
`find_optimal_hyperparameters`) reuse `budget_check.py`'s already-tested
`measure_and_check`/`optimize_hyperparameters` — those tests hit the real
`neurax-service` and skip when it's down (`test_budget_check.py`'s own
convention); this file follows the same pattern for the same reason: the one
thing actually worth checking there is that the real endpoint understands
what gets sent, which no mock can tell you.

The remaining tools have no existing neurax-agent equivalent and are
tested here against a faked `httpx.AsyncClient` — deterministic regardless
of whether a real backend is running, matching `test_credentials.py`'s style
for `make_chat_model`.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import pytest

import analysis_tools


# ─── snapshot_to_spec / formatting (pure, no I/O) ──────────────────────────

def test_snapshot_to_spec_reads_nodes_connections_and_hw_config():
    snapshot = {
        "family": "cnn",
        "nodes": [{"id": "c1", "type": "conv2d", "params": {"out_channels": 16}}],
        "connections": [{"from": "c1", "to": "c1"}],
        "hw_config": {"precision": "int8"},
    }
    spec = analysis_tools.snapshot_to_spec(snapshot)
    assert spec.family == "cnn"
    assert spec.nodes[0].id == "c1"
    assert spec.nodes[0].params == {"out_channels": 16}
    assert spec.edges[0].from_id == "c1" and spec.edges[0].to_id == "c1"
    assert spec.hw_config == {"precision": "int8"}


def test_format_size_reports_both_the_unit_and_the_exact_byte_count():
    assert analysis_tools.format_size(0) == "0 B"
    text = analysis_tools.format_size(2 * 1024 ** 2)
    assert "MB" in text and "2,097,152" in text


# ─── analyze_architecture / check_budget / find_optimal_hyperparameters ───
# Real-service style: hit neurax-service for real, skip if it's not up.

_TINY_SNAPSHOT = {
    "family": "transformer",
    "nodes": [
        {"id": "emb", "type": "token_embedding", "params": {"vocab_size": 2000, "hidden_size": 64}},
        {"id": "attn", "type": "mha_attention", "params": {"hidden_size": 64, "num_heads": 4}},
        {"id": "ffn", "type": "ffn_standard", "params": {"hidden_size": 64, "intermediate_size": 128}},
        {"id": "head", "type": "lm_head", "params": {"in_features": 64, "out_features": 10}},
    ],
    "connections": [
        {"from": "emb", "to": "attn"}, {"from": "attn", "to": "ffn"}, {"from": "ffn", "to": "head"},
    ],
    "hw_config": {"hardware": "T4", "gpuCount": 1, "precision": "int8", "batchSize": 1},
}


def _skip_if_unreachable(text: str):
    if "Cannot reach NEURAX backend" in text or "could not" in text.lower():
        pytest.skip(f"compiler unavailable: {text}")


def test_analyze_architecture_reports_real_metrics():
    text = asyncio.run(analysis_tools.analyze_architecture(_TINY_SNAPSHOT))
    _skip_if_unreachable(text)
    assert "Total Parameters" in text


def test_check_budget_reports_pass_fail_against_a_stated_limit():
    text = asyncio.run(analysis_tools.check_budget(_TINY_SNAPSHOT, max_size_mb=1))
    _skip_if_unreachable(text)
    assert "PASS" in text or "FAIL" in text


def test_check_budget_with_no_limits_says_unconstrained():
    text = asyncio.run(analysis_tools.check_budget(_TINY_SNAPSHOT))
    _skip_if_unreachable(text)
    assert "not constrained" in text.lower()


def test_find_optimal_hyperparameters_returns_a_sweep_summary():
    text = asyncio.run(analysis_tools.find_optimal_hyperparameters(_TINY_SNAPSHOT))
    _skip_if_unreachable(text)
    assert "configurations" in text.lower() or "no feasible" in text.lower()


# ─── estimate_training_cost (pure function, no I/O) ────────────────────────

def test_estimate_training_cost_computes_a_positive_cost():
    text = asyncio.run(analysis_tools.estimate_training_cost(
        parameters=7e9, tokens=1e12, gpu_type="H100", gpu_count=8,
    ))
    assert "Total Cost: $" in text
    assert "7.00B parameters" in text


def test_estimate_training_cost_honours_explicit_hours_over_the_flop_estimate():
    text = asyncio.run(analysis_tools.estimate_training_cost(
        parameters=1e9, tokens=1e9, gpu_type="A100", gpu_count=1, hours=10,
    ))
    assert "Estimated Hours: 10.00 hours" in text


# ─── The standalone HTTP tools: faked httpx.AsyncClient ───────────────────

class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "http://test")
            raise httpx.HTTPStatusError(
                "error", request=request, response=httpx.Response(self.status_code, text=self.text, request=request)
            )

    def json(self):
        return self._payload


class _FakeAsyncClient:
    def __init__(self, payload=None, status_code=200, raise_connect_error=False):
        self._payload = payload
        self._status_code = status_code
        self._raise_connect_error = raise_connect_error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, *a, **kw):
        if self._raise_connect_error:
            raise httpx.ConnectError("connection refused", request=httpx.Request("GET", url))
        return _FakeResponse(self._payload, self._status_code)


def _patch_client(monkeypatch, **kwargs):
    monkeypatch.setattr(analysis_tools.httpx, "AsyncClient", lambda **_: _FakeAsyncClient(**kwargs))


def test_get_hardware_list_formats_each_gpu(monkeypatch):
    _patch_client(monkeypatch, payload=[
        {"name": "H100-SXM", "memory_gb": 80, "tflops_fp16": 989, "memory_bandwidth_gbs": 3350},
    ])
    text = asyncio.run(analysis_tools.get_hardware_list())
    assert "H100-SXM" in text and "80GB" in text


def test_get_presets_lists_id_name_family(monkeypatch):
    _patch_client(monkeypatch, payload=[
        {"id": "gpt2-small", "name": "GPT-2 Small", "family": "transformer", "description": "A small GPT-2."},
    ])
    text = asyncio.run(analysis_tools.get_presets())
    assert "gpt2-small" in text and "GPT-2 Small" in text


def test_get_preset_returns_json_for_a_known_id(monkeypatch):
    _patch_client(monkeypatch, payload={"id": "gpt2-small", "family": "transformer"})
    text = asyncio.run(analysis_tools.get_preset("gpt2-small"))
    assert '"id": "gpt2-small"' in text


def test_get_preset_reports_not_found_on_404(monkeypatch):
    _patch_client(monkeypatch, payload={"detail": "not found"}, status_code=404)
    text = asyncio.run(analysis_tools.get_preset("does-not-exist"))
    assert "not found" in text.lower()


def test_get_compliance_config_returns_json(monkeypatch):
    _patch_client(monkeypatch, payload={"eu_ai_act": True})
    text = asyncio.run(analysis_tools.get_compliance_config())
    assert "eu_ai_act" in text





def test_a_connection_error_produces_a_readable_message_not_a_crash(monkeypatch):
    _patch_client(monkeypatch, raise_connect_error=True)
    text = asyncio.run(analysis_tools.get_hardware_list())
    assert "Cannot reach" in text


def test_a_backend_error_status_is_reported_with_its_code(monkeypatch):
    _patch_client(monkeypatch, payload={"detail": "boom"}, status_code=500)
    text = asyncio.run(analysis_tools.get_compliance_config())
    assert "500" in text


# ─── dispatch() routing ────────────────────────────────────────────────────

def test_every_registered_tool_is_one_a_mode_can_actually_reach():
    """A count is not the invariant that matters; reachability is.

    `get_credits`, `get_user_info` and `health_check` were implemented,
    routed by `dispatch` and described to nobody: no entry in any of
    `MODE_TOOL_GRANTS`'s four sets, so `execute_tool` would have refused
    them had the model ever named one. This asserts the property that was
    actually violated, rather than restoring a number that only counted
    dead code.
    """
    import agent_graph

    granted = set().union(*agent_graph.MODE_TOOL_GRANTS.values())
    unreachable = analysis_tools.ANALYSIS_TOOL_NAMES - granted
    assert not unreachable, f"registered but no mode grants them: {sorted(unreachable)}"


def test_dispatch_routes_estimate_training_cost(monkeypatch):
    text = asyncio.run(analysis_tools.dispatch(
        "estimate_training_cost",
        {"parameters": 1e9, "tokens": 1e9, "gpu_type": "A100"},
        snapshot={},
    ))
    assert "Total Cost: $" in text


def test_dispatch_routes_get_hardware_list(monkeypatch):
    _patch_client(monkeypatch, payload=[])
    text = asyncio.run(analysis_tools.dispatch("get_hardware_list", {}, snapshot={}))
    assert "Available NEURAX Hardware" in text


def test_dispatch_raises_on_an_unknown_tool_name():
    with pytest.raises(ValueError):
        asyncio.run(analysis_tools.dispatch("not_a_real_tool", {}, snapshot={}))
