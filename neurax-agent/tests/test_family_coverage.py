"""Every architecture family must actually compile.

The compiler rejects an unknown `layer_type` with a 400, so a block the agent
can place but the compiler cannot read fails the entire analysis — the budget
then goes unverified and the client receives a design nobody measured. These
tests build one representative model per family out of that family's own
catalogue blocks and require it to compile.

One fixture per family, and exactly the eight families that exist end to end:
`neurax-parser`'s `ModelType::from_str` accepts these and no others, and
`Index.tsx`'s `ALL_ARCHITECTURE_FAMILIES`, `plugins.ts`'s `ArchitectureFamily`
and `catalogue.json` all agree on the same eight. Fixtures for `snn`,
`multimodal` and `experimental` used to sit here too and could never pass:
the compiler answers `Invalid model type` for all three, and no catalogue
entry exists to validate their blocks against.
"""
import sys, os, json, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from topology_validator import ArchSpec
from budget_check import measure_and_check, spec_to_topology, LAYER_TYPE_MAP
from requirements import DeploymentBudget

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATALOGUE = json.load(open(os.path.join(HERE, "catalogue.json")))

HW = {"hardware": "T4", "gpuCount": 1, "precision": "fp16", "batchSize": 1, "seqLen": 128}


def _spec(family, nodes, edges=None):
    edges = edges or [{"from": nodes[i]["id"], "to": nodes[i + 1]["id"]}
                      for i in range(len(nodes) - 1)]
    return ArchSpec.from_dict({"family": family, "nodes": nodes, "edges": edges,
                               "hw_config": HW})


# One representative design per family, built from that family's own blocks.
REFERENCE_MODELS = {
    "transformer": [
        {"id": "emb", "type": "embedding", "params": {"vocab_size": 32000, "hidden_size": 256}},
        {"id": "attn", "type": "mha", "params": {"hidden_size": 256, "num_heads": 8}},
        {"id": "ffn", "type": "ffn", "params": {"hidden_size": 256, "intermediate_size": 1024}},
        {"id": "norm", "type": "rmsnorm", "params": {"hidden_size": 256}},
        {"id": "head", "type": "lm_head", "params": {"in_features": 256, "out_features": 32000}},
    ],
    "cnn": [
        {"id": "c1", "type": "conv2d", "params": {"in_channels": 3, "out_channels": 64, "kernel_size": 7}},
        # Activation isn't its own block — it's a param on the block it
        # applies to.
        {"id": "bn", "type": "batchnorm", "params": {"hidden_size": 64, "activation": "relu"}},
        {"id": "pool", "type": "max_pool", "params": {"kernel_size": 2}},
        {"id": "fc", "type": "dense", "params": {"in_features": 64, "out_features": 1000}},
    ],
    "moe": [
        {"id": "emb", "type": "embedding", "params": {"vocab_size": 32000, "hidden_size": 256}},
        {"id": "gate", "type": "gate", "params": {"hidden_size": 256}},
        {"id": "moe", "type": "moe_block",
         "params": {"hidden_size": 256, "intermediate_size": 1024, "num_experts": 8, "top_k": 2}},
        {"id": "head", "type": "lm_head", "params": {"in_features": 256, "out_features": 32000}},
    ],
    "ssm": [
        {"id": "emb", "type": "embedding", "params": {"vocab_size": 32000, "hidden_size": 256}},
        {"id": "mamba", "type": "mamba_block",
         "params": {"hidden_size": 256, "state_dim": 16, "expansion_factor": 2}},
        {"id": "norm", "type": "rmsnorm", "params": {"hidden_size": 256}},
        {"id": "head", "type": "lm_head", "params": {"in_features": 256, "out_features": 32000}},
    ],
    "rnn": [
        {"id": "emb", "type": "embedding", "params": {"vocab_size": 20000, "hidden_size": 256}},
        {"id": "lstm", "type": "lstm", "params": {"hidden_size": 256, "rnn_hidden_size": 256}},
        {"id": "head", "type": "classification_head", "params": {"in_features": 256, "out_features": 5}},
    ],
    "gnn": [
        {"id": "g1", "type": "gcn_conv", "params": {"hidden_size": 128},
         "custom_equations": {"flops_forward": "2 * B * S * H * H", "params": "H * H"}},
        {"id": "norm", "type": "graphnorm", "params": {"hidden_size": 128}},
        {"id": "pool", "type": "global_mean_pool", "params": {}},
        {"id": "head", "type": "classification_head", "params": {"in_features": 128, "out_features": 10}},
    ],
    "gan": [
        {"id": "z", "type": "dense", "params": {"in_features": 100, "out_features": 4096}},
        {"id": "up", "type": "conv_transpose2d", "params": {"in_channels": 256, "out_channels": 128, "kernel_size": 4}},
        {"id": "bn", "type": "batchnorm", "params": {"hidden_size": 128, "activation": "tanh"}},
    ],
    "diffusion": [
        {"id": "t", "type": "timestep_embedding", "params": {"hidden_size": 256}},
        {"id": "unet", "type": "unet_block", "params": {"in_channels": 64, "out_channels": 128}},
        {"id": "attn", "type": "mha", "params": {"hidden_size": 256, "num_heads": 8}},
        {"id": "out", "type": "conv2d", "params": {"in_channels": 128, "out_channels": 3, "kernel_size": 3}},
    ],
}


def test_every_catalogue_block_maps_to_a_compiler_type():
    """A block the agent can place must be one the compiler can read."""
    accepted = _accepted_layer_types()
    unmapped = []
    for family, entry in CATALOGUE.items():
        for block in entry.get("blocks", []):
            kind = block.get("type", "")
            if LAYER_TYPE_MAP.get(kind, kind) not in accepted:
                unmapped.append(f"{family}:{kind}")
    assert unmapped == [], f"blocks with no compiler equivalent: {unmapped}"


def _accepted_layer_types():
    import re
    source = open(os.path.join(HERE, "..", "neurax-parser", "src", "model_config.rs")).read()
    start = source.index("impl LayerType")
    end = source.index("pub fn as_str", start)
    accepted = set()
    for match in re.finditer(
        r'^\s*((?:"[a-z0-9_]+"\s*\|\s*)*"[a-z0-9_]+")\s*=>', source[start:end], re.M
    ):
        accepted.update(re.findall(r'"([a-z0-9_]+)"', match.group(1)))
    return accepted


@pytest.mark.parametrize("family", sorted(REFERENCE_MODELS))
def test_a_reference_model_compiles_for_every_family(family):
    spec = _spec(family, REFERENCE_MODELS[family])
    report = asyncio.run(measure_and_check(spec, DeploymentBudget(), HW))
    if report.error and "Connect" in str(report.error):
        pytest.skip("compiler unavailable")
    assert not report.error, f"{family} failed to compile: {report.error}"
    assert report.metrics.get("total_parameters", 0) > 0, \
        f"{family} compiled but reported no parameters"


@pytest.mark.parametrize("family", sorted(REFERENCE_MODELS))
def test_reference_models_use_only_their_own_family_blocks(family):
    """Guards the fixtures themselves against drifting from the catalogue."""
    catalogue_blocks = {b["type"] for b in CATALOGUE[family].get("blocks", [])}
    used = {n["type"] for n in REFERENCE_MODELS[family]}
    foreign = used - catalogue_blocks
    assert foreign == set(), f"{family} fixture uses blocks outside its catalogue: {foreign}"


def test_the_families_are_the_same_eight_everywhere():
    """One list of families, agreed on by four independent places.

    They had drifted: `presets.rs` served `snn` and `rl` presets — six designs
    a user could load and the compiler would then refuse with
    `Invalid model type`, since `ModelType::from_str` has never accepted
    either. This asserts the agreement rather than any one list, so the next
    family added anywhere has to be added everywhere.
    """
    import re

    root = os.path.join(HERE, "..")

    parser = open(os.path.join(root, "neurax-parser", "src", "model_config.rs")).read()
    body = parser[parser.index("pub fn from_str"):parser.index("pub fn as_str")]
    compiler = {
        # The first alternative in each arm is the canonical name; the rest
        # are aliases ("convolutional", "mamba", ...), not separate families.
        re.findall(r'"([a-z_]+)"', arm)[0]
        for arm in re.findall(r'^\s*("[a-z_]+"(?:\s*\|\s*"[a-z_]+")*)\s*=>', body, re.M)
    }

    ui = open(os.path.join(root, "neurax-ui", "src", "types", "plugins.ts")).read()
    ui_block = ui[ui.index("export type ArchitectureFamily"):ui.index("export interface ArchitectureFamilyConfig")]
    studio = set(re.findall(r"'([a-z]+)'", ui_block))

    catalogue = set(CATALOGUE)

    presets = open(os.path.join(root, "neurax-service", "src", "presets.rs")).read()
    served = set(re.findall(r'family: "([a-z_]+)"', presets))

    assert compiler == studio == catalogue == served == set(REFERENCE_MODELS), (
        f"families disagree — compiler={sorted(compiler)}, studio={sorted(studio)}, "
        f"catalogue={sorted(catalogue)}, presets={sorted(served)}, "
        f"fixtures={sorted(REFERENCE_MODELS)}"
    )
