import asyncio
import os
import json
from dotenv import load_dotenv
load_dotenv()

from agent_runner import _run_agent
from arch_planner import plan_architecture, plan_strategy
from topology_validator import validate_arch_spec
from layout_engine import assign_positions
from materializer import materialize
from catalogue_store import get_catalogue_for_family

async def verify_pipeline():
    user_message = "A simple CNN with 3 conv layers and a classification head"
    family = "cnn"
    catalogue = get_catalogue_for_family(family)
    constraints = {"requiredBlocks": ["input", "output"], "incompatibleBlocks": []}
    hw_config = {"inChannels": 3, "numClasses": 10, "imgHeight": 224, "imgWidth": 224, "batchSize": 32}

    print(f"--- Phase 0: Strategy Planning ---")
    strategy = await plan_strategy(user_message, family, hw_config)
    for item in strategy:
        print(f"  [{item.id}] {item.text}")

    print(f"\n--- Phase 1: Architecture Planning (Creativity: 0.0) ---")
    spec = await plan_architecture(
        user_message, 
        family, 
        catalogue, 
        constraints, 
        creativity=0.0, 
        hw_config=hw_config,
        strategy=[s.text for s in strategy]
    )
    print(f"Rationale: {spec.rationale}")
    print(f"Nodes: {[n.id for n in spec.nodes]}")

    result = validate_arch_spec(spec, catalogue, constraints)
    print(f"Validation: {'Valid' if result.valid else 'Invalid'}")
    if not result.valid:
        print(f"Errors: {result.errors}")
        return

    positions = assign_positions(spec)
    print(f"Layout: Produced positions for {len(positions)} nodes")
    
    # Verify spacing
    input_pos = positions.get("input")
    output_pos = positions.get("output")
    if input_pos and output_pos:
        print(f"Input Pos: {input_pos}")
        print(f"Output Pos: {output_pos}")
        print(f"Total Width: {output_pos[0] - input_pos[0]} px")

    print("\n--- Materializing Tool Calls ---")
    async for tool in materialize(spec, positions):
        print(f"  Tool: {tool['name']} -> {json.dumps(tool['args'])}")

    print(f"\n--- Testing Creativity Variance (Creativity: 0.9) ---")
    spec1 = await plan_architecture(user_message, family, catalogue, constraints, creativity=0.9, hw_config=hw_config)
    spec2 = await plan_architecture(user_message, family, catalogue, constraints, creativity=0.9, hw_config=hw_config)
    
    print(f"Spec 1 Nodes: {[n.id for n in spec1.nodes]}")
    print(f"Spec 2 Nodes: {[n.id for n in spec2.nodes]}")
    
    if [n.id for n in spec1.nodes] != [n.id for n in spec2.nodes]:
        print("✅ Creativity confirmed: different specs generated for same prompt at high temperature")
    else:
        print("ℹ️ Specs were identical (LLM choice), try again for statistical variance if needed")

if __name__ == "__main__":
    asyncio.run(verify_pipeline())
