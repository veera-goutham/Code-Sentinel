"""
aws_helpers.py — Shared AWS Client Factory

Centralises boto3 client creation for Glue, S3, and Bedrock Runtime so that
all agents share the same cached clients. Reads AWS_GLUE_REGION,
AWS_BEDROCK_REGION, and BEDROCK_MODEL_ID from environment variables (via
.env). Streamlit's @st.cache_resource decorator is used where the module is
imported inside a Streamlit app; a plain module-level singleton is used in
non-Streamlit contexts (e.g. CLI, tests).

Extracted from the client-creation helpers at the top of app.py.
"""
import json
import os
from functools import lru_cache
from urllib.parse import urlparse

import boto3
from dotenv import load_dotenv

load_dotenv()

GLUE_REGION: str = os.getenv("AWS_GLUE_REGION", "ap-south-2")
BEDROCK_REGION: str = os.getenv("AWS_BEDROCK_REGION", "ap-south-2")
BEDROCK_MODEL_ID: str = os.getenv(
    "BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
)


# ---------------------------------------------------------------------------
# Client factories — lru_cache(maxsize=None) gives a per-process singleton
# without any Streamlit dependency.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def get_glue_client():
    return boto3.client("glue", region_name=GLUE_REGION)


@lru_cache(maxsize=None)
def get_s3_client():
    return boto3.client("s3", region_name=GLUE_REGION)


@lru_cache(maxsize=None)
def get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)


# ---------------------------------------------------------------------------
# Glue data-access helpers
# ---------------------------------------------------------------------------

def list_glue_jobs() -> list[str]:
    """Return a sorted list of Glue job names in the configured region."""
    response = get_glue_client().list_jobs()
    return sorted(response.get("JobNames", []))


def get_glue_job_details(job_name: str) -> dict:
    """Fetch the full config dict for a single Glue job."""
    return get_glue_client().get_job(JobName=job_name)["Job"]


def fetch_script_from_s3(s3_uri: str) -> str:
    """Download an S3 object by s3:// URI and return its content as UTF-8 text."""
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    obj = get_s3_client().get_object(Bucket=bucket, Key=key)
    return obj["Body"].read().decode("utf-8")


# ---------------------------------------------------------------------------
# Bedrock helper
# ---------------------------------------------------------------------------

def invoke_claude(
    prompt: str,
    max_tokens: int = 4000,
    temperature: float = 0.2,
) -> str:
    """
    Send a single-turn prompt to Claude via Bedrock and return the response text.

    Callers that expect JSON should pass the result to extract_json().
    """
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    response = get_bedrock_client().invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"].strip()


# ---------------------------------------------------------------------------
# JSON extraction utility
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# S3 write / copy helpers (used by the Approve flow in app.py)
# ---------------------------------------------------------------------------

def overwrite_s3_object(s3_uri: str, content: str) -> None:
    """Replace an S3 object's content with the given UTF-8 string."""
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    ext = key.rsplit(".", 1)[-1].lower() if "." in key else ""
    content_type = {"py": "text/x-python", "sql": "text/plain"}.get(ext, "text/plain")
    get_s3_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=content.encode("utf-8"),
        ContentType=content_type,
    )


def backup_s3_object(source_uri: str, backup_prefix: str) -> str:
    """
    Copy the object at source_uri to <backup_prefix>/<filename>.
    Returns the full destination s3:// URI.
    """
    parsed_src = urlparse(source_uri)
    src_bucket = parsed_src.netloc
    src_key = parsed_src.path.lstrip("/")
    filename = src_key.rsplit("/", 1)[-1]

    parsed_dst = urlparse(backup_prefix)
    dst_bucket = parsed_dst.netloc
    dst_key = parsed_dst.path.lstrip("/") + "/" + filename

    get_s3_client().copy_object(
        Bucket=dst_bucket,
        Key=dst_key,
        CopySource={"Bucket": src_bucket, "Key": src_key},
    )
    return f"s3://{dst_bucket}/{dst_key}"


def upload_s3_artifact(bucket: str, key: str, content: str | bytes) -> str:
    """
    Upload a string or bytes to s3://<bucket>/<key>.
    ContentType is inferred from the file extension.
    Returns the full s3:// URI.
    """
    _TYPES = {
        ".md":   "text/markdown",
        ".json": "application/json",
        ".py":   "text/x-python",
        ".mmd":  "text/plain",
        ".sql":  "text/plain",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    ext = "." + key.rsplit(".", 1)[-1].lower() if "." in key else ""
    content_type = _TYPES.get(ext, "text/plain")
    body = content.encode("utf-8") if isinstance(content, str) else content
    get_s3_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType=content_type,
    )
    return f"s3://{bucket}/{key}"


# ---------------------------------------------------------------------------
# JSON extraction utility
# ---------------------------------------------------------------------------

def extract_json(claude_text: str) -> dict:
    """
    Parse JSON from a Claude response, stripping ```json fences if present.

    Raises ValueError if the text cannot be parsed as JSON.
    """
    text = claude_text.strip()
    if text.startswith("```"):
        # Drop the opening fence line and the closing fence
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Claude response is not valid JSON: {exc}\n\nRaw text:\n{claude_text}") from exc
