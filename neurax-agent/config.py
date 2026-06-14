"""Shared configuration and state for neurax-agent."""
import asyncio
from typing import Any

# Runtime state
_runs: dict[str, asyncio.Queue[dict[str, Any]]] = {}
_catalogue_cache: dict[str, list[dict[str, Any]]] = {}
_CATALOGUE_CACHE_MAX = 32


def _sse_event(event: str, data: dict[str, Any]) -> str:
    import json
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
