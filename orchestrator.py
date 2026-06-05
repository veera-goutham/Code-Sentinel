"""
orchestrator.py — Multi-Agent Orchestrator

Dispatches all five Code Sentinel agents in parallel using a thread pool,
collects their results into a unified report dict, and isolates failures so
one broken agent never blocks the others.
"""
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from agents import documentation, lineage, performance, security, test_gen

logger = logging.getLogger(__name__)

_AGENTS: dict[str, Any] = {
    "performance":   performance.run,
    "documentation": documentation.run,
    "security":      security.run,
    "test_gen":      test_gen.run,
    "lineage":       lineage.run,
}

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
    active = selected_agents if selected_agents is not None else set(_AGENTS)

    cache_key = _script_hash(script, sorted(active))
    cached = _cache_load(cache_key)
    if cached is not None:
        logger.info("Cache hit for %s", cache_key)
        return cached

    results: dict[str, Any] = {
        "performance":   None,
        "documentation": None,
        "security":      None,
        "test_gen":      None,
        "lineage":       None,
        "errors":        {},
    }

    # Mark skipped agents immediately
    for name in _AGENTS:
        if name not in active:
            results["errors"][name] = "Not selected"

    t_start = time.perf_counter()

    agents_to_run = {name: fn for name, fn in _AGENTS.items() if name in active}
    if agents_to_run:
        with ThreadPoolExecutor(max_workers=len(agents_to_run)) as pool:
            futures = {
                pool.submit(fn, script, config, memory_context): name
                for name, fn in agents_to_run.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as exc:
                    results["errors"][name] = str(exc)

    total_seconds = time.perf_counter() - t_start
    # Only count truly failed agents, not skipped ones
    agents_failed = sum(
        1 for _, msg in results["errors"].items() if msg != "Not selected"
    )

    results["_meta"] = {
        "total_seconds": round(total_seconds, 2),
        "agents_run":    len(active),
        "agents_failed": agents_failed,
    }

    _cache_save(cache_key, results)
    return results
