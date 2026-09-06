"""The agent must drive the platform, not just draw a graph.

Blocks alone do not make the model the client asked for: precision, batch size,
sequence length and the training hyperparameters change its cost as much as the
architecture does. The agent has to set them, and the budget check has to measure
what the agent actually chose.
"""
import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from topology_validator import ArchSpec
from budget_check import spec_to_topology, measure_and_check, suggest_precision
from snapshot_ops import _apply_tool_to_snapshot
from requirements import extract_budget

MB = 1024 ** 2

SPEC = ArchSpec.from_dict({
    "family": "transformer",
    "nodes": [
        {"id": "emb", "type": "token_embedding",
         "params": {"vocab_size": 2000, "hidden_size": 64}},
        {"id": "attn", "type": "mha_attention", "params": {"hidden_size": 64, "num_heads": 4}},
        {"id": "head", "type": "lm_head", "params": {"in_features": 64, "out_features": 10}},
    ],
    "edges": [{"from": "emb", "to": "attn"}, {"from": "attn", "to": "head"}],
    "hw_config": {"precision": "int8", "batchSize": 1, "hardware": "T4", "seqLen": 64},
    "hyperparams": {"learning_rate": 3e-4, "warmup_steps": 500, "optimizer": "adamw"},
})



def test_spec_carries_platform_settings_and_hyperparameters():
    assert SPEC.hw_config["precision"] == "int8"
    assert SPEC.hyperparams["optimizer"] == "adamw"





def test_budget_measures_the_precision_the_agent_chose():
    # The panel says fp32; the agent chose int8 for this design. The measurement
    # must follow the agent, or it checks a model nobody asked for.
    topo = spec_to_topology(SPEC, {"precision": "fp32"})
    assert topo["training"]["precision"] == "int8"


def test_precision_lever_is_offered_only_when_it_can_close_the_gap():
    # fp32 -> int8 is 4x, enough for a 3x overshoot.
    assert suggest_precision("fp32", 3.0) == "int8"
    # A 10x overshoot cannot be fixed by dtype alone from fp32.
    assert suggest_precision("fp32", 10.0) is None
    # Already at the narrowest width: nothing left to give.
    assert suggest_precision("int8", 2.0) is None


def test_measured_design_respects_the_clients_size_budget():
    budget = extract_budget("un modele pour telephone, moins de 1 mega")
    report = asyncio.run(measure_and_check(SPEC, budget, {}))
    if report.error:
        pytest.skip(f"compiler unavailable: {report.error}")
    assert report.fits, report.summary()
    params = report.metrics["total_parameters"]
    size = report.metrics["parameter_memory_bytes"]
    assert size == params, "int8 must store one byte per parameter"
