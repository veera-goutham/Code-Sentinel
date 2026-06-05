"""
orchestrator.py — Multi-Agent Orchestrator

Dispatches all five Code Sentinel agents in parallel using a LangGraph
StateGraph, collects their results into a unified report dict, and isolates
failures so one broken agent never blocks the others.
"""
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Annotated, Any

from langgraph.graph import END, START, StateGraph
from typing import TypedDict

from agents import documentation, lineage, performance, security, test_gen

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("./cache/reviews")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _script_hash(code: str, agents: list[str]) -> str:
    key = code.strip() + "|" + ",".join(sorted(agents))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _cache_load(cache_key: str) -> dict | None:
    cache_file = _CACHE_DIR / f"{cache_key}.json"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to read cache %s: %s", cache_key, e)
        return None


def _cache_save(cache_key: str, result: dict) -> None:
    cache_file = _CACHE_DIR / f"{cache_key}.json"
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
    except Exception as e:
        logger.warning("Failed to write cache %s: %s", cache_key, e)


# ---------------------------------------------------------------------------
# LangGraph state and graph (built once at module load)
# ---------------------------------------------------------------------------

class _State(TypedDict):
    script: str
    config: dict
    memory_context: str
    active_agents: set
    performance: Any
    documentation: Any
    security: Any
    test_gen: Any
    lineage: Any
    errors: Annotated[dict, lambda a, b: {**a, **b}]


_AGENT_NAMES: set[str] = {"performance", "documentation", "security", "test_gen", "lineage"}

_AGENT_FNS: dict[str, Any] = {
    "performance":   performance.run,
    "documentation": documentation.run,
    "security":      security.run,
    "test_gen":      test_gen.run,
    "lineage":       lineage.run,
}


def _make_node(name: str, fn: Any):
    def _node(state: _State) -> dict:
        if name not in state["active_agents"]:
            return {"errors": {name: "Not selected"}}
        try:
            return {name: fn(state["script"], state["config"], state["memory_context"])}
        except Exception as exc:
            return {"errors": {name: str(exc)}}
    _node.__name__ = name
    return _node


def _build_graph():
    builder = StateGraph(_State)
    for name, fn in _AGENT_FNS.items():
        builder.add_node(name, _make_node(name, fn))
        builder.add_edge(START, name)
        builder.add_edge(name, END)
    return builder.compile()


_GRAPH = _build_graph()


def review_script(
    script: str,
    config: dict,
    selected_agents: set[str] | None = None,
    memory_context: str = "",
) -> dict:
    """
    Run agents against *script* in parallel and return a unified report.

    Args:
        script: Raw PySpark / Python shell source code of the Glue job.
        config: Dict with keys worker_type, number_of_workers, timeout,
                job_type (all optional / may be None).
        selected_agents: Set of agent names to run. If None, all 5 run.
                         Unselected agents get result=None and
                         errors[name]="Not selected".

    Returns:
        Dict with keys performance, documentation, security, test_gen,
        lineage (each the agent's result dict or None), errors, and _meta.
    """
    active = selected_agents if selected_agents is not None else set(_AGENT_NAMES)

    cache_key = _script_hash(script, sorted(active))
    cached = _cache_load(cache_key)
    if cached is not None:
        logger.info("Cache hit for %s", cache_key)
        return cached

    t_start = time.perf_counter()

    final_state = _GRAPH.invoke({
        "script": script,
        "config": config,
        "memory_context": memory_context,
        "active_agents": active,
        "performance": None,
        "documentation": None,
        "security": None,
        "test_gen": None,
        "lineage": None,
        "errors": {},
    })

    total_seconds = time.perf_counter() - t_start

    errors: dict[str, str] = final_state.get("errors", {})
    agents_failed = sum(1 for msg in errors.values() if msg != "Not selected")

    results = {
        "performance":   final_state.get("performance"),
        "documentation": final_state.get("documentation"),
        "security":      final_state.get("security"),
        "test_gen":      final_state.get("test_gen"),
        "lineage":       final_state.get("lineage"),
        "errors":        errors,
        "_meta": {
            "total_seconds": round(total_seconds, 2),
            "agents_run":    len(active),
            "agents_failed": agents_failed,
        },
    }

    _cache_save(cache_key, results)
    return results
