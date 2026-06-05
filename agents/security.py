"""
security.py — Security Review Agent

Scans a job script for security and compliance issues: hardcoded
credentials, unmasked PII, injection vectors, unsafe deserialization,
and insecure write destinations. Reports findings with severity levels
and remediation guidance. Does NOT modify the script.

Injects paradigm-aware context so findings are relevant to the actual
execution model of the script being reviewed.
"""
from aws_helpers import extract_json, invoke_claude

_SECURITY_PARADIGM_CONTEXTS: dict[str, tuple[str, str]] = {
    "pyspark_glue": (
        "This is an AWS Glue PySpark script running on distributed workers.",
        "Focus on: Spark SQL injection via spark.sql() f-strings, Glue IAM role over-permission, "
        "S3 encryption on Glue output paths, sensitive values in Glue logger/CloudWatch.",
    ),
    "pyspark": (
        "This is a standalone PySpark script (not Glue).",
        "Focus on: Spark SQL injection, unsafe serialization in distributed contexts, "
        "hardcoded credentials passed to SparkSession config.",
    ),
    "pandas": (
        "This is a pandas/Python script — NOT a distributed Spark job.",
        "Focus on: hardcoded credentials, PII in dataframe columns written to disk, "
        "SQL injection in cursor.execute() calls, unsafe file deserialization. "
        "Do NOT recommend PySpark or Glue-specific fixes.",
    ),
    "plain_python": (
        "This is plain Python without a dataframes library.",
        "Focus on: hardcoded credentials, shell injection, eval/exec misuse, "
        "unsafe deserialization (pickle, yaml). "
        "Do NOT recommend PySpark, Glue, or pandas-specific fixes.",
    ),
    "sql": (
        "This is a SQL script.",
        "Focus on: SQL injection patterns, overly permissive queries, "
        "hardcoded credentials or secrets in comments or string literals.",
    ),
}

_SECURITY_PARADIGM_PREFIX = """\
PARADIGM CONTEXT:
{paradigm_description}
{paradigm_focus}

"""

_PROMPT_TEMPLATE = """\
You are a cloud security engineer specializing in AWS Glue ETL pipelines \
and PySpark data processing. Your task is to perform a thorough security \
and compliance review of the script below.

Job configuration:
- Job type    : {job_type}
- Worker type : {worker_type}
- Workers     : {num_workers}
- Timeout     : {timeout} minutes

Script:
```python
{script}
```

Scan the script for every issue in the following checklist:

CREDENTIALS & SECRETS
- Hardcoded AWS access key IDs or secret access keys (patterns like \
AKIA*, hard-coded strings passed to boto3)
- Hardcoded API keys, passwords, tokens, or private keys in any form
- Hardcoded database connection strings containing usernames or passwords
- Secrets embedded in comments

PII EXPOSURE
- Column names or variables suggesting PII being written without masking: \
email, phone, mobile, aadhaar, aadhar, pan, dob, date_of_birth, ssn, \
address, full_name, first_name, last_name, ip_address, passport, \
credit_card, bank_account
- PII column values passed to print(), logging calls, or st.write()
- PII written to destinations that appear unencrypted or publicly accessible

INJECTION
- SQL injection via string concatenation or f-strings passed to \
spark.sql(), sqlContext.sql(), or cursor.execute()
- Shell injection via subprocess with untrusted input
- Unsafe use of eval() or exec() on external data

UNSAFE DESERIALIZATION
- pickle.loads() or joblib.load() called on data from S3 or external sources
- yaml.load() without Loader=yaml.SafeLoader

INSECURE WRITE DESTINATIONS
- Writes to S3 paths containing "-public-", "/public/", \
"open-data", or "public-bucket" in the path or bucket name
- Missing ServerSideEncryption flags on sensitive S3 puts

IAM & PERMISSIONS
- Hardcoded ARNs for IAM roles being assumed via sts.assume_role()
- Wildcard resource ("*") in inline policy documents embedded in the script

LOGGING & OBSERVABILITY
- Sensitive values (keys, tokens, PII fields) appearing in log messages \
or exception strings that would surface in CloudWatch

For each issue found, record:
- The approximate line number (1-based) in the original script, or null \
if the issue is structural and not tied to a specific line
- The exact category from this fixed list: \
"credentials", "pii_exposure", "injection", "encryption", "logging", \
"iam", "other"
- A one-sentence description of the problem
- A one-sentence recommended fix

Also list every column name or variable name in the script that appears to \
contain PII (use the PII field list above as your guide).

Determine the overall risk level:
- CRITICAL : any finding with severity CRITICAL
- HIGH     : highest finding is HIGH
- MEDIUM   : highest finding is MEDIUM
- LOW      : highest finding is LOW
- NONE     : no findings at all

Respond with ONLY a JSON object — no markdown fences, no prose outside JSON:
{{
  "findings": [
    {{
      "severity": "<CRITICAL | HIGH | MEDIUM | LOW>",
      "category": "<credentials | pii_exposure | injection | encryption | logging | iam | other>",
      "line_number": <integer or null>,
      "issue": "<one sentence describing the problem>",
      "suggested_fix": "<one sentence describing the fix>"
    }}
  ],
  "pii_columns_detected": ["<column or variable name>"],
  "overall_risk": "<CRITICAL | HIGH | MEDIUM | LOW | NONE>",
  "summary": "<one paragraph summarising the overall security posture of \
this script, the most critical issues, and the top priority remediation steps>"
}}

Rules:
- If no findings exist, return an empty list [] for findings, not null.
- If no PII columns are detected, return an empty list [] for \
pii_columns_detected, not null.
- overall_risk must equal the highest severity across all findings, \
or "NONE" if findings is empty.
- Do not report the same issue twice at different line numbers; \
pick the first occurrence and note if it recurs.
- Do not invent findings not supported by the script text.
"""


def _detect_paradigm(code: str) -> str:
    """Detect the execution paradigm of the script (mirrors performance.py)."""
    lowered = code.lower()
    if "from awsglue" in lowered or "gluecontext" in lowered:
        return "pyspark_glue"
    if "from pyspark" in lowered or "sparksession" in lowered:
        return "pyspark"
    if "import pandas" in lowered or "from pandas" in lowered:
        return "pandas"
    if lowered.strip().startswith("select ") or "create table" in lowered:
        return "sql"
    return "plain_python"


def _build_prompt(script: str, config: dict) -> str:
    paradigm = _detect_paradigm(script)
    description, focus = _SECURITY_PARADIGM_CONTEXTS[paradigm]
    paradigm_prefix = _SECURITY_PARADIGM_PREFIX.format(
        paradigm_description=description,
        paradigm_focus=focus,
    )
    worker_type = config.get("worker_type") or "n/a (Python shell)"
    num_workers = config.get("number_of_workers") or "n/a"
    timeout = config.get("timeout") or "n/a"
    job_type = config.get("job_type") or "glueetl"
    return paradigm_prefix + _PROMPT_TEMPLATE.format(
        job_type=job_type,
        worker_type=worker_type,
        num_workers=num_workers,
        timeout=timeout,
        script=script,
    )


def run(script: str, config: dict, memory_context: str = "") -> dict:
    """
    Scan *script* for security and compliance issues.

    Args:
        script: Raw PySpark / Python shell source code of the Glue job.
        config: Dict with keys worker_type, number_of_workers, timeout,
                job_type (all optional / may be None).
        memory_context: Optional past-review context prepended to the prompt.

    Returns:
        Dict with keys findings, pii_columns_detected, overall_risk, summary.
    """
    prompt = _build_prompt(script, config)
    if memory_context:
        prompt = memory_context + "\n\n" + prompt
    raw = invoke_claude(prompt, max_tokens=4000, temperature=0)
    result = extract_json(raw)
    # Guarantee list types even if Claude returns null for empty collections
    result.setdefault("findings", [])
    result.setdefault("pii_columns_detected", [])
    if result["findings"] is None:
        result["findings"] = []
    if result["pii_columns_detected"] is None:
        result["pii_columns_detected"] = []
    return result
