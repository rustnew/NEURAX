"""The assistant may write the model file — and must be told what happened.

NEURAX's translator turns a canvas into PyTorch layer by layer, or refuses. The
designs it refuses were simply untrainable, and the assistant can write those.
What makes that trustworthy is not the assistant: it is the studio building the
file with PyTorch, comparing the parameter count it really has against the
analysis, and pushing one batch through it. This file covers the agent's half —
that the tool is reachable, and that the verdict comes back in words the model
can act on. Refusing without saying why would be no better than not checking.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_graph import MODE_TOOL_GRANTS
from langchain_runner import (
    ALL_TOOL_DESCRIPTIONS,
    CANVAS_TOOL_DESCRIPTIONS,
    _describe_assistant_model,
)


def test_the_tool_is_described_and_names_its_arguments():
    description = CANVAS_TOOL_DESCRIPTIONS["write_model_code"]
    # An agent that cannot tell which arguments a tool takes calls it wrong.
    assert "code" in description and "class_name" in description
    # And it must know the code is checked, or it will not bother being right.
    assert "refused" in description.lower()


def test_the_modes_that_build_architectures_may_write_the_model():
    assert "write_model_code" in MODE_TOOL_GRANTS["creation"]
    assert "write_model_code" in MODE_TOOL_GRANTS["research"]


def test_the_read_only_mode_may_not():
    # `explanation` mutates nothing. Writing code the studio then executes is
    # the least read-only thing in the product.
    assert "write_model_code" not in MODE_TOOL_GRANTS["explanation"]


def test_every_granted_tool_is_a_real_one():
    # The same invariant agent_graph asserts at import time, restated here so
    # the failure names this file rather than an import.
    for mode, grant in MODE_TOOL_GRANTS.items():
        unknown = grant - set(ALL_TOOL_DESCRIPTIONS)
        assert not unknown, f"{mode} grants undescribed tools: {unknown}"


def test_no_code_written_says_what_would_run_instead():
    described = _describe_assistant_model(None)
    assert "translator" in described


def test_a_refusal_carries_the_reason_and_asks_for_a_fix():
    described = _describe_assistant_model({
        "class_name": "MyNet",
        "accepted": False,
        "stage": "import",
        "reason": "The file does not import. SyntaxError: invalid syntax",
        "built_parameters": None,
        "analyzed_parameters": 1_000_000,
    })
    assert "REFUSED" in described
    assert "SyntaxError" in described, "the agent cannot fix what it is not told"
    assert "write_model_code again" in described


def test_a_parameter_disagreement_shows_both_numbers():
    # The failure the whole gate exists for. "It disagrees" is not actionable;
    # "12,000,000 against 9,000,000" is.
    described = _describe_assistant_model({
        "class_name": "MyNet",
        "accepted": False,
        "stage": "verified",
        "reason": "PyTorch builds 12,000,000 parameters, but NEURAX analysed 9,000,000.",
        "built_parameters": 12_000_000,
        "analyzed_parameters": 9_000_000,
    })
    assert "12,000,000" in described and "9,000,000" in described


def test_an_acceptance_says_the_run_will_use_it():
    described = _describe_assistant_model({
        "class_name": "MyNet",
        "accepted": True,
        "stage": "verified",
        "reason": "Built by PyTorch: 1,000,000 parameters.",
        "built_parameters": 1_000_000,
        "analyzed_parameters": 1_000_000,
    })
    assert "accepted" in described
    assert "not the translator" in described


def test_a_verdict_about_an_old_design_is_marked_stale():
    # Otherwise the agent reads "accepted" for a model nobody is looking at.
    described = _describe_assistant_model({
        "class_name": "MyNet",
        "accepted": True,
        "stage": "verified",
        "reason": "Built by PyTorch: 1,000,000 parameters.",
        "built_parameters": 1_000_000,
        "analyzed_parameters": 1_000_000,
        "stale": True,
    })
    assert "changed since" in described
    assert "Send it again" in described
