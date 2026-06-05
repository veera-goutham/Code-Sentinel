"""
assistant.py — Contextual review assistant for Code Sentinel.

No Streamlit imports — UI-agnostic and testable standalone.
"""
import logging
import os
from typing import Generator

import boto3

logger = logging.getLogger(__name__)

MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-4-5-20250929-v1:0")
REGION = os.getenv("AWS_BEDROCK_REGION", "ap-south-2")
MAX_TOKENS = 1500
MAX_HISTORY_PAIRS = 10
MAX_SCRIPT_CHARS_IN_CONTEXT = 6000

_client = None


def _get_client():
    """Singleton boto3 bedrock-runtime client."""
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime", region_name=REGION)
    return _client


def build_system_prompt(
    job_name: str,
    paradigm: str | None,
    original_script: str | None,
    review_result: dict,
    similar_reviews: list[dict],
) -> str:
    """
    Build the system prompt that gives the assistant full context of the
    current review.
    """
    lines = [
        "You are Code Sentinel's review assistant.",
        "The user has just reviewed a data pipeline. Your job is to help them",
        "understand the agent findings, explore organizational memory, and",
        "reason about whether to approve or reject. Answer concisely and",
        "accurately. If asked about something outside this review's scope,",
        "give general guidance but note that it's not specific to the team's",
        "history.",
        "",
        "=== CURRENT REVIEW ===",
        f"Job: {job_name}",
        f"Detected paradigm: {paradigm or 'unknown'}",
        "",
        f"Original script (truncated to {MAX_SCRIPT_CHARS_IN_CONTEXT} chars):",
        "```",
        (original_script or "")[:MAX_SCRIPT_CHARS_IN_CONTEXT] or "(not available)",
        "```",
        "",
        "=== AGENT FINDINGS ===",
    ]

    # Performance section
    perf = review_result.get("performance")
    if perf is not None:
        lines.append("** Performance **")
        changes = perf.get("summary_of_changes") or []
        for change in changes:
            lines.append(f"- {change}")
        savings = perf.get("estimated_savings") or {}
        before = savings.get("before_per_run_usd")
        after = savings.get("after_per_run_usd")
        if before is not None and after is not None:
            lines.append(f"Estimated savings: before ${before}/run → after ${after}/run")
        lines.append("")

    # Security section
    sec = review_result.get("security")
    if sec is not None:
        lines.append("** Security **")
        risk = sec.get("overall_risk")
        if risk:
            lines.append(f"Risk level: {risk}")
        for finding in (sec.get("findings") or []):
            sev = finding.get("severity", "")
            category = finding.get("category", "")
            issue = finding.get("issue", "")
            lines.append(f"- [{sev}] {category}: {issue}")
        lines.append("")

    # Documentation section
    doc = review_result.get("documentation")
    if doc is not None:
        lines.append("** Documentation **")
        summary = doc.get("business_summary", "")
        if summary:
            lines.append(summary)
        lines.append("(Full documentation available in the Documentation tab.)")
        lines.append("")

    # Lineage section
    lin = review_result.get("lineage")
    if lin is not None:
        lines.append("** Lineage **")
        inputs = lin.get("input_tables") or []
        outputs = lin.get("output_tables") or []
        dep = lin.get("dependency_summary", "")
        if inputs:
            lines.append(f"Sources: {', '.join(t.get('name', '') for t in inputs)}")
        if outputs:
            lines.append(f"Sinks: {', '.join(t.get('name', '') for t in outputs)}")
        if dep:
            lines.append(dep)
        lines.append("")

    # Organizational memory section
    lines.append("=== ORGANIZATIONAL MEMORY ===")
    if similar_reviews:
        for m in similar_reviews:
            decision = m.get("decision", "")
            jname = m.get("job_name", "")
            ts = m.get("timestamp", "")[:10]
            sim_pct = int(m.get("similarity", 0) * 100)
            reason = m.get("reason", "")
            findings_summary = m.get("findings_summary", "")
            lines.append(f"{decision} · {jname} · {ts} · similarity {sim_pct}%")
            if decision == "REJECTED" and reason:
                lines.append(f"  Reason: {reason}")
            if findings_summary:
                lines.append(f"  Findings: {findings_summary}")
    else:
        lines.append("No similar past reviews found in organizational memory.")

    lines += [
        "",
        "Use this context to answer questions specifically about THIS review",
        "and what the team has decided in the past. When citing past reviews,",
        "name them explicitly.",
    ]

    return "\n".join(lines)


def chat_stream(
    messages: list[dict],
    system_prompt: str,
) -> Generator[str, None, None]:
    """
    Generator that yields text chunks from Bedrock Converse stream API.

    messages format:
      [{"role": "user", "content": [{"text": "..."}]},
       {"role": "assistant", "content": [{"text": "..."}]}, ...]

    On any exception, yields a single error string and stops. Never raises.
    """
    try:
        client = _get_client()
        response = client.converse_stream(
            modelId=MODEL_ID,
            messages=messages,
            system=[{"text": system_prompt}],
            inferenceConfig={"maxTokens": MAX_TOKENS, "temperature": 0.3},
        )
        stream = response.get("stream")
        for event in stream:
            delta = event.get("contentBlockDelta")
            if delta:
                text = delta.get("delta", {}).get("text", "")
                if text:
                    yield text
    except Exception as e:
        logger.warning("chat_stream error: %s", e)
        yield f"⚠️ Assistant error: {e}"


def trim_history(
    history: list[dict],
    max_pairs: int = MAX_HISTORY_PAIRS,
) -> list[dict]:
    """
    Keep only the last max_pairs * 2 messages. If the resulting slice
    starts with an assistant message (which can happen when slicing
    past a pair boundary), drop that orphan so the conversation begins
    with a user message — Bedrock requires this.
    Never drop an orphan user message at the start; that orphan IS
    the current question.
    """
    if not history:
        return []
    trimmed = history[-(max_pairs * 2):]
    if trimmed and trimmed[0].get("role") == "assistant":
        trimmed = trimmed[1:]
    return trimmed
