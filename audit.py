"""
audit.py — Audit Trail

Persists every Approve / Reject decision to a local JSONL file so that
every action taken on a Glue script is traceable across sessions.
"""
import json
import os

AUDIT_LOG_PATH = "code_sentinel_audit.jsonl"


def log_decision(record: dict) -> None:
    """Append a decision record (one JSON line) to the audit log."""
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def load_decisions(limit: int = 50) -> list[dict]:
    """Return the most recent *limit* decisions, newest first."""
    if not os.path.exists(AUDIT_LOG_PATH):
        return []
    with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    records: list[dict] = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records[-limit:][::-1]  # newest first
