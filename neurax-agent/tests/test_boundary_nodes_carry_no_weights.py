"""`input` and `output` declare shapes. They must not be counted as layers.

`LAYER_TYPE_MAP` mapped `input` onto `embedding` and `output` onto `dense`,
both with empty params, so the compiler applied its own defaults to them: a
50,000-token vocabulary at width 512, and a 512-wide dense layer. Every
analysis the agent ran carried 25,862,656 parameters that were not in the
design.

It was on every design without exception — `input` and `output` are in
`requiredBlocks` for all eight families, so `validate_arch_spec` refuses to let
the agent call `done` without them. A CNN whose real weight count is 2,442 was
reported at 25,865,098, and `check_budget` against any real size limit failed
on phantom weight.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analysis_tools
from budget_check import SHAPE_ONLY_TYPES, spec_to_topology

CONV = {"id": "c", "type": "conv2d",
        "params": {"in_channels": 3, "out_channels": 64, "kernel_size": 3}}
HEAD = {"id": "h", "type": "classification_head",
        "params": {"in_features": 64, "out_features": 10}}


def _topology(nodes, edges):
    return spec_to_topology(analysis_tools.snapshot_to_spec(
        {"family": "cnn", "nodes": nodes, "connections": edges, "hw_config": {}}
    ))


def test_boundary_nodes_never_reach_the_compiler_as_layers():
    topo = _topology(
        [{"id": "in", "type": "input", "params": {}}, CONV, HEAD,
         {"id": "out", "type": "output", "params": {}}],
        [{"from": "in", "to": "c"}, {"from": "c", "to": "h"}, {"from": "h", "to": "out"}],
    )
    assert [l["id"] for l in topo["model"]["layers"]] == ["c", "h"]


def test_the_graph_is_bridged_not_broken():
    # Dropping a node must not orphan its neighbours: the real layers stay
    # connected, and no edge names a node the compiler was never sent.
    topo = _topology(
        [{"id": "in", "type": "input", "params": {}}, CONV, HEAD,
         {"id": "out", "type": "output", "params": {}}],
        [{"from": "in", "to": "c"}, {"from": "c", "to": "h"}, {"from": "h", "to": "out"}],
    )
    ids = {l["id"] for l in topo["model"]["layers"]}
    assert topo["model"]["connections"] == [{"from": "c", "to": "h"}]
    for conn in topo["model"]["connections"]:
        assert conn["from"] in ids and conn["to"] in ids


def test_a_shape_only_node_in_the_middle_joins_its_neighbours():
    # The bridge is written generally rather than special-cased on position,
    # so a boundary marker anywhere still leaves the path intact.
    topo = _topology(
        [CONV, {"id": "mid", "type": "input", "params": {}}, HEAD],
        [{"from": "c", "to": "mid"}, {"from": "mid", "to": "h"}],
    )
    assert topo["model"]["connections"] == [{"from": "c", "to": "h"}]


def test_the_design_weighs_what_its_real_layers_weigh():
    with_boundaries = _topology(
        [{"id": "in", "type": "input", "params": {}}, CONV, HEAD,
         {"id": "out", "type": "output", "params": {}}],
        [{"from": "in", "to": "c"}, {"from": "c", "to": "h"}, {"from": "h", "to": "out"}],
    )
    without = _topology([CONV, HEAD], [{"from": "c", "to": "h"}])
    assert with_boundaries["model"]["layers"] == without["model"]["layers"]


def test_the_map_no_longer_claims_a_boundary_node_has_weights():
    from budget_check import LAYER_TYPE_MAP

    for name in SHAPE_ONLY_TYPES:
        assert name not in LAYER_TYPE_MAP, (
            f"'{name}' declares a shape; mapping it to a layer type is what "
            "produced the phantom parameters"
        )
