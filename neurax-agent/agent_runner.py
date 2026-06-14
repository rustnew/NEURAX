"""Agent orchestration - main run loop using the 3-phase declarative pipeline."""
import asyncio
import logging
import os
from typing import Any, Optional

from langchain_runner import select_family
from snapshot_ops import _apply_tool_to_snapshot
from catalogue_store import get_catalogue_for_family, get_all_blocks, get_family_constraints
from suggestions import _rehydrate_catalogue

# New 3-phase modules
from arch_planner import plan_architecture, plan_strategy
from topology_validator import validate_arch_spec, ArchSpec
from layout_engine import assign_positions
from materializer import materialize

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] [NEURAX-AGENT] %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)


def _event(event_type: str, data: Any) -> dict[str, Any]:
    """Helper to format SSE events."""
    return {"event": event_type, "data": data}


async def _run_agent(
    run_id: str,
    q: "asyncio.Queue[dict[str, Any]]",
    user_message: str,
    snapshot: dict[str, Any],
    _runs: dict[str, "asyncio.Queue[dict[str, Any]]"],
    creativity: float = 0.0,
) -> None:
    """
    Main agent orchestration using the 3-phase declarative pipeline:
    1. Phase 1: Planning (LLM generates a full ArchSpec)
    2. Phase 2: Validation & Layout (Pure Python correctness check + auto-layout)
    3. Phase 3: Materialization (Stream tool calls to the canvas)
    """
    logger.info(f"🚀 AGENT STARTED (3-PHASE) | run_id={run_id} | creativity={creativity}")
    logger.info(f"   User message: {user_message[:100]}...")

    try:
        # ── 0. Setup & Family Selection ──
        _rehydrate_catalogue(snapshot)
        
        current_family = str(snapshot.get("family") or "").strip()
        allowed_families = list(snapshot.get("allowed_families") or [])
        selected_family = current_family

        if allowed_families:
            try:
                selected_family = await select_family(
                    user_message=user_message,
                    allowed_families=allowed_families,
                    catalogue=list(snapshot.get("catalogue") or []),
                    current_family=current_family or None,
                    max_retries=3,
                )
            except Exception as e:
                logger.error(f"❌ Family selection failed: {e}")
                selected_family = current_family or allowed_families[0]

            if selected_family != current_family:
                logger.info(f"🔄 Family changed: {current_family} → {selected_family}")
                await q.put(_event("assistant", {"content": f"I've selected the '{selected_family}' family for this architecture."}))
                # Apply set_family immediately
                tool = {"name": "set_family", "args": {"family": selected_family}}
                await q.put(_event("tool", tool))
                snapshot = _apply_tool_to_snapshot(snapshot, tool)
        
        # Ensure we have the right catalogue for the selected family
        family_catalogue = get_catalogue_for_family(selected_family)
        if not family_catalogue:
            family_catalogue = get_all_blocks()
        
        # Family-specific constraints (from catalogue_store / catalogue.json)
        constraints = get_family_constraints(selected_family)

        # ── Phase 0: Orchestration Planning ──
        logger.info("📋 Phase 0: Generating orchestration strategy...")
        strategy_items = await plan_strategy(
            user_message=user_message,
            family=selected_family,
            hw_config=snapshot.get("hw_config")
        )
        
        # Emit initial plan
        plan_data = [{"id": item.id, "text": item.text, "status": "pending"} for item in strategy_items]
        if plan_data:
            plan_data[0]["status"] = "in_progress"
        await q.put(_event("plan", {"items": plan_data}))

        def _update_plan(idx: int, status: str):
            if idx < len(plan_data):
                plan_data[idx]["status"] = status
                # If we mark one as done, mark next as in_progress
                if status == "done" and idx + 1 < len(plan_data):
                    plan_data[idx+1]["status"] = "in_progress"
            return _event("plan", {"items": plan_data})

        # ── 1. Phase 1: Architecture Planning (Structured LLM Spec) ──
        await q.put(_event("assistant", {"content": "Designing the architecture specification..."}))
        
        spec: Optional[ArchSpec] = None
        validation_result = None
        previous_errors: list[str] = []

        # Max 3 attempts to get a valid spec from the LLM
        for attempt in range(1, 4):
            logger.info(f"📋 Planning attempt {attempt}/3...")
            try:
                spec = await plan_architecture(
                    user_message=user_message,
                    family=selected_family,
                    catalogue=family_catalogue,
                    constraints=constraints,
                    creativity=creativity,
                    hw_config=snapshot.get("hw_config"),
                    strategy=[item.text for item in strategy_items],
                    previous_errors=previous_errors if previous_errors else None
                )
                
                # ── 2. Phase 2: Validation ──
                validation_result = validate_arch_spec(spec, family_catalogue, constraints)
                
                if validation_result.valid:
                    logger.info(f"✅ Spec validated successfully on attempt {attempt}")
                    if spec.rationale:
                        await q.put(_event("assistant", {"content": spec.rationale}))
                    await q.put(_update_plan(0, "done")) # Mark first step (planning) as done
                    break
                else:
                    previous_errors = validation_result.errors
                    logger.warning(f"⚠️ Validation failed on attempt {attempt}: {previous_errors}")
                    if attempt < 3:
                        await q.put(_event("assistant", {"content": f"Refining design (fix: {previous_errors[0]})..."}))
            except Exception as e:
                logger.error(f"❌ Planning attempt {attempt} failed: {e}")
                if attempt == 3:
                    raise

        if not validation_result or not validation_result.valid:
            error_details = "; ".join(previous_errors) if previous_errors else "Unknown error"
            raise ValueError(f"Could not generate a valid architecture: {error_details}")

        # ── 3. Phase 2e: Layout ──
        logger.info("📐 Computing optimal layout...")
        positions = assign_positions(spec)

        # ── 4. Phase 3: Materialization ──
        logger.info("🔧 streaming tool calls to canvas...")
        # Mark middle steps as done during materialization (simplification)
        for i in range(1, len(plan_data) - 1):
             await q.put(_update_plan(i, "done"))

        count = 0
        async for tool_call in materialize(spec, positions):
            await q.put(_event("tool", tool_call))
            snapshot = _apply_tool_to_snapshot(snapshot, tool_call)
            count += 1
            # Slight delay to make the UI feel "alive" as it builds
            if tool_call["name"] != "clear_canvas":
                await asyncio.sleep(0.05)

        # Mark final step as done
        await q.put(_update_plan(len(plan_data) - 1, "done"))
        logger.info(f"✨ Architecture materialized with {count} tool calls")

    except Exception as e:
        logger.error(f"💥 Agent Run Failed: {e}", exc_info=True)
        await q.put(_event("error", {"message": str(e)}))
        await q.put(_event("assistant", {"content": f"I encountered an error while building the architecture: {str(e)}"}))
    finally:
        await q.put(_event("done", {}))
