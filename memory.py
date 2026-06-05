"""
memory.py — ChromaDB-backed Organizational Memory for Code Sentinel

Stores past review decisions as vector embeddings so future reviews can
retrieve relevant context and inject it into agent prompts.

No Streamlit imports — this module is UI-agnostic.
"""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "code_sentinel_memory"
TITAN_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBED_DIMS = 1024
MAX_EMBED_CHARS = 30000
SIMILARITY_THRESHOLD = 0.50
TOP_K = 3

_CHROMA_UNAVAILABLE = False

try:
    import chromadb
except ImportError:
    logger.warning(
        "chromadb not installed — memory features disabled. "
        "Run: pip install 'chromadb>=0.4.22'"
    )
    _CHROMA_UNAVAILABLE = True

# Module-level singleton so we don't re-instantiate on every call
_collection = None


def get_embedding(text: str) -> Optional[list[float]]:
    """Call Bedrock Titan Text Embeddings V2 and return the embedding vector."""
    try:
        import boto3
        from dotenv import load_dotenv
        load_dotenv()

        region = os.getenv("AWS_BEDROCK_REGION", "ap-south-2")
        client = boto3.client("bedrock-runtime", region_name=region)

        body = json.dumps({
            "inputText": text[:MAX_EMBED_CHARS],
            "dimensions": EMBED_DIMS,
            "normalize": True,
        })
        response = client.invoke_model(
            modelId=TITAN_MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        return result["embedding"]
    except Exception as exc:
        logger.warning("get_embedding failed: %s", exc)
        return None


def get_collection():
    """Return a singleton ChromaDB collection configured for cosine distance."""
    global _collection
    if _CHROMA_UNAVAILABLE:
        return None
    if _collection is not None:
        return _collection
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        return _collection
    except Exception as exc:
        logger.warning("get_collection failed: %s", exc)
        return None


def _build_entry_id(job_name: str, timestamp: str, decision: str) -> str:
    """Return a stable Chroma document ID."""
    return f"{job_name}__{timestamp}__{decision}"


def store_memory(
    record: dict,
    script_content: Optional[str],
    findings_summary: Optional[str],
) -> bool:
    """
    Store a review decision as a vector memory in ChromaDB.

    Called from the Approve and Reject handlers after log_decision.
    Returns True on success, False on failure. Never raises.
    """
    try:
        collection = get_collection()
        if collection is None:
            return False

        job_name = str(record.get("job_name", "unknown"))
        decision = str(record.get("decision", "UNKNOWN"))
        timestamp = str(record.get("timestamp", datetime.now(timezone.utc).isoformat()))
        reason = str(record.get("reason") or "")

        has_script_embedding = False
        embedding: Optional[list[float]] = None
        doc_text: Optional[str] = None

        if script_content:
            embedding = get_embedding(script_content)
            if embedding is not None:
                has_script_embedding = True
                doc_text = script_content[:2000]

        if embedding is None:
            synthetic = f"Job: {job_name}\nDecision: {decision}\nReason: {reason}"
            embedding = get_embedding(synthetic)
            doc_text = synthetic

        if embedding is None:
            logger.warning(
                "store_memory: both script and synthetic embedding failed for %s", job_name
            )
            return False

        agents_run = record.get("agents_run") or []
        agents_run_str = (
            ",".join(agents_run) if isinstance(agents_run, list) else str(agents_run)
        )

        savings = record.get("savings_per_run_usd")
        savings_float = float(savings) if savings is not None else 0.0

        metadata: dict = {
            "job_name": job_name,
            "decision": decision,
            "timestamp": timestamp,
            "reason": reason,
            "source": str(record.get("source", "")),
            "agents_run": agents_run_str,
            "savings_per_run_usd": savings_float,
            "findings_summary": str(findings_summary) if findings_summary else "",
            "has_script_embedding": has_script_embedding,
            "s3_backup_path": str(record.get("s3_backup_path") or ""),
        }

        entry_id = _build_entry_id(job_name, timestamp, decision)
        collection.upsert(
            ids=[entry_id],
            embeddings=[embedding],
            documents=[doc_text],
            metadatas=[metadata],
        )
        logger.info("store_memory: stored entry %s", entry_id)
        return True
    except Exception as exc:
        logger.warning("store_memory failed: %s", exc)
        return False


def find_similar_reviews(
    script_content: str,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Query ChromaDB for past reviews similar to the given script.

    Returns a list of dicts sorted by similarity desc, newest first within tiers.
    Returns [] on any failure. Never raises.
    """
    try:
        collection = get_collection()
        if collection is None:
            return []

        total = collection.count()
        if total == 0:
            return []

        embedding = get_embedding(script_content)
        if embedding is None:
            return []

        n_results = min(top_k, total)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["metadatas", "distances"],
        )

        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        all_matches: list[dict] = []
        for meta, dist in zip(metadatas, distances):
            # ChromaDB cosine space: distance = 1 - cosine_similarity
            similarity = 1.0 - float(dist)
            if similarity < SIMILARITY_THRESHOLD:
                continue
            all_matches.append({
                "job_name": meta.get("job_name", ""),
                "decision": meta.get("decision", ""),
                "timestamp": meta.get("timestamp", ""),
                "reason": meta.get("reason", ""),
                "findings_summary": meta.get("findings_summary", ""),
                "similarity": round(similarity, 4),
                "agents_run": meta.get("agents_run", ""),
                "_has_script": meta.get("has_script_embedding", False),
            })

        # Prefer script-embedded matches; fall back to metadata-only if nothing else
        script_matches = [m for m in all_matches if m.get("_has_script")]
        candidates = script_matches if script_matches else all_matches

        # Sort: highest similarity first, then newest timestamp first (ISO strings sort chronologically)
        candidates.sort(key=lambda x: x["timestamp"], reverse=True)
        candidates.sort(key=lambda x: x["similarity"], reverse=True)

        # Strip internal key before returning
        return [{k: v for k, v in m.items() if k != "_has_script"} for m in candidates]
    except Exception as exc:
        logger.warning("find_similar_reviews failed: %s", exc)
        return []


def build_memory_context(similar: list[dict]) -> str:
    """
    Format similar reviews into a string for prepending to agent prompts.

    Returns empty string if list is empty.
    """
    if not similar:
        return ""

    lines = [
        "=== ORGANIZATIONAL MEMORY ===",
        "The following relevant past reviews from this team were found.",
        "Consider these decisions when making your recommendations.",
        "",
    ]

    for i, m in enumerate(similar, 1):
        decision = m.get("decision", "")
        job_name = m.get("job_name", "")
        timestamp = m.get("timestamp", "")[:10]
        similarity_pct = int(m.get("similarity", 0) * 100)
        agents_run = m.get("agents_run", "")
        reason = m.get("reason", "")
        findings = m.get("findings_summary", "")

        lines.append(
            f"[{i}] {decision} · {job_name} · {timestamp} · similarity {similarity_pct}%"
        )
        lines.append(f"    Agents: {agents_run}")
        if decision == "REJECTED" and reason:
            lines.append(f"    Reason: {reason}")
        if findings:
            lines.append(f"    Findings: {findings}")
        lines.append("")

    lines += [
        "=== END MEMORY ===",
        "Use this context to:",
        "- Avoid recommending changes the team has rejected for valid business reasons",
        "- Apply patterns from past approved optimizations where applicable",
        "- Surface relevant past concerns proactively",
    ]

    return "\n".join(lines)


def backfill_from_audit_log() -> int:
    """
    Backfill ChromaDB from the audit log on first startup.

    Only runs when the collection is empty. Returns count of entries added.
    Never raises.
    """
    try:
        collection = get_collection()
        if collection is None:
            return 0

        if collection.count() > 0:
            return 0

        audit_path = "code_sentinel_audit.jsonl"
        if not os.path.exists(audit_path):
            return 0

        with open(audit_path, "r", encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]

        if not lines:
            return 0

        logger.info("Backfilling %d entries from audit log...", len(lines))
        count = 0

        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            decision = record.get("decision", "")
            s3_backup_path = record.get("s3_backup_path") or ""
            script_content: Optional[str] = None

            if decision == "APPROVED" and s3_backup_path:
                try:
                    from aws_helpers import get_s3_client
                    parsed = urlparse(s3_backup_path)
                    bucket = parsed.netloc
                    key = parsed.path.lstrip("/")
                    obj = get_s3_client().get_object(Bucket=bucket, Key=key)
                    script_content = obj["Body"].read().decode("utf-8")
                except Exception as exc:
                    logger.warning(
                        "Backfill S3 fetch failed for %s: %s", s3_backup_path, exc
                    )

            if store_memory(record, script_content, None):
                count += 1

        logger.info("Backfill complete: %d/%d entries added.", count, len(lines))
        return count
    except Exception as exc:
        logger.warning("backfill_from_audit_log failed: %s", exc)
        return 0


def get_memory_stats() -> dict:
    """Return counts of stored memories by decision type."""
    try:
        collection = get_collection()
        if collection is None:
            return {"total": 0, "approved": 0, "rejected": 0, "with_script_embedding": 0}

        total = collection.count()
        if total == 0:
            return {"total": 0, "approved": 0, "rejected": 0, "with_script_embedding": 0}

        all_results = collection.get(include=["metadatas"])
        metadatas = all_results.get("metadatas", [])

        approved = sum(1 for m in metadatas if m.get("decision") == "APPROVED")
        rejected = sum(1 for m in metadatas if m.get("decision") == "REJECTED")
        with_script = sum(
            1 for m in metadatas if m.get("has_script_embedding") is True
        )

        return {
            "total": total,
            "approved": approved,
            "rejected": rejected,
            "with_script_embedding": with_script,
        }
    except Exception as exc:
        logger.warning("get_memory_stats failed: %s", exc)
        return {"total": 0, "approved": 0, "rejected": 0, "with_script_embedding": 0}
