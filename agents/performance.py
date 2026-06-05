"""
performance.py — Performance Optimization Agent

Analyzes an AWS Glue job script and its worker configuration, identifies
performance and cost anti-patterns, and produces an optimized rewrite with
a config recommendation and ballpark cost-savings estimate.

Preserves the original code's paradigm (pyspark_glue, pyspark, pandas,
plain_python, sql) — never converts between execution models.
"""
from aws_helpers import extract_json, invoke_claude

_PARADIGM_RULES: dict[str, str] = {
    "pyspark_glue": (
        "This is an AWS Glue PySpark script. Keep all awsglue and pyspark imports. "
        "Optimize using broadcast joins, partition pruning, predicate pushdown, caching strategy."
    ),
    "pyspark": (
        "This is a standalone PySpark script (not Glue). Keep pyspark imports only. "
        "Do NOT add awsglue imports."
    ),
    "pandas": (
        "This is a pandas script. Optimize using vectorization, chunking, dtype downcasting, "
        "query() over boolean indexing. Do NOT convert to PySpark. "
        "Do NOT add pyspark or awsglue imports."
    ),
    "plain_python": (
        "This is plain Python (no dataframes library). Optimize using algorithmic improvements, "
        "list comprehensions, generators, built-in functions. "
        "Do NOT add pandas, pyspark, or awsglue imports unless they exist in the original."
    ),
    "sql": (
        "This is SQL. Optimize using better joins, indexes, partition predicates, CTE refactoring. "
        "Output must remain valid SQL."
    ),
}

_PARADIGM_PREFIX = """\
CRITICAL CONSTRAINT — PARADIGM PRESERVATION:
{paradigm_instruction}
The optimized code MUST use the same libraries, frameworks, and execution
model as the original. Do NOT rewrite from one paradigm to another.
If the original is pandas, do NOT produce PySpark. If the original is
plain Python, do NOT add Spark/Glue.

"""

_PROMPT_TEMPLATE = """\
You are an expert performance engineer specializing in cost optimization \
for data processing workloads on AWS.

Analyze the script below and its worker configuration. Identify \
every performance and cost anti-pattern from this checklist:

1. Missing broadcast hints on joins with small dimension tables
2. SELECT * patterns that pull excessive columns before aggregation or writes
3. Cross joins or accidental cartesian products — require explicit join keys
4. Python UDFs that can be replaced with native Spark/SQL functions
5. Missing partition predicate pushdown on partitioned data sources
6. Repeated reads of the same source that could be cached or broadcast
7. Worker config over-provisioning relative to the apparent job complexity

Rewrite the script fixing every identified issue. Then recommend an updated \
worker configuration if the job appears over- or under-provisioned.

Estimate the cost per run before and after your changes using these reference \
rates for ap-south-2:
- G.1X worker: $0.44 per DPU-hour (each G.1X worker = 1 DPU)
- G.2X worker: $0.88 per DPU-hour (each G.2X worker = 2 DPU)
- Python shell: $0.44 per DPU-hour, max 1 DPU
- Estimate actual runtime based on the job's workload, NOT the timeout \
field. The timeout is a SAFETY MAXIMUM (often 2880 min), never the \
expected runtime. Use these realistic ranges for ap-south-2 Glue ETL:
  * Small daily aggregation (under 100 GB scanned): 10-25 minutes
  * Medium ETL with joins (100-500 GB scanned): 25-60 minutes
  * Heavy reprocessing (over 500 GB scanned): 60-180 minutes
  * Python shell jobs: 1-10 minutes
- Total cost per run = DPU-count × DPU-rate × (estimated_runtime_min / 60).
- For after_per_run_usd, account for optimizations (broadcast joins, \
  column pruning, predicate pushdown) typically reducing runtime 30-70%; \
  also use the recommended worker count, not the original.
- before_per_run_usd should land in the $1.50-$15 range for typical daily \
  ETL. If the script is trivial (under 20 lines, no real transforms), \
  before_per_run_usd should be under $1.
- The estimated_savings.rationale must state runtime in minutes before and \
  after, e.g.: "Original ~45 min × 10 G.2X workers = $6.60; optimized \
  broadcast join + column pruning drops to ~12 min × 5 G.1X workers = $0.44."

Current worker configuration:
- Job type    : {job_type}
- Worker type : {worker_type}
- Workers     : {num_workers}
- Timeout     : {timeout} minutes

Original script:
```python
{script}
```

Respond with ONLY a JSON object — no markdown fences, no prose outside JSON:
{{
  "optimized_code": "<full rewritten script as a single string>",
  "config_recommendation": {{
    "worker_type": "<G.1X | G.2X | unchanged>",
    "number_of_workers": <integer or "unchanged">,
    "rationale": "<one sentence explaining the recommendation>"
  }},
  "summary_of_changes": [
    "<bullet 1 — what was changed and why>",
    "<bullet 2>",
    "<bullet 3>"
  ],
  "estimated_savings": {{
    "before_per_run_usd": <float>,
    "after_per_run_usd": <float>,
    "rationale": "<one sentence describing how the estimate was reached>"
  }}
}}

Rules:
- summary_of_changes must have between 3 and 6 entries.
- If the job is a Python shell job (worker_type is n/a), recommend \
MaxCapacity changes instead of worker_type/number_of_workers; set \
worker_type to "n/a" and number_of_workers to "unchanged".
- Do not invent data about the script that is not present. If a pattern \
from the checklist is not applicable, skip it.
"""


def _detect_paradigm(code: str) -> str:
    """
    Return one of: 'pyspark_glue', 'pyspark', 'pandas', 'plain_python', 'sql'
    based on imports and constructs in the source. Conservative — when in
    doubt, return the most narrow paradigm.
    """
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


def _build_prompt(script: str, config: dict, paradigm: str) -> str:
    paradigm_instruction = _PARADIGM_RULES[paradigm]
    prefix = _PARADIGM_PREFIX.format(paradigm_instruction=paradigm_instruction)
    worker_type = config.get("worker_type") or "n/a (Python shell)"
    num_workers = config.get("number_of_workers") or "n/a"
    timeout = config.get("timeout") or "n/a"
    job_type = config.get("job_type") or "glueetl"
    return prefix + _PROMPT_TEMPLATE.format(
        job_type=job_type,
        worker_type=worker_type,
        num_workers=num_workers,
        timeout=timeout,
        script=script,
    )


def run(script: str, config: dict, memory_context: str = "") -> dict:
    """
    Analyze *script* for performance anti-patterns and return an optimized
    rewrite with a config recommendation and cost-savings estimate.

    Args:
        script: Raw source code of the job.
        config: Dict with keys worker_type, number_of_workers, timeout,
                job_type (all optional / may be None).
        memory_context: Optional past-review context prepended to the prompt.

    Returns:
        Dict with keys optimized_code, config_recommendation,
        summary_of_changes, estimated_savings, detected_paradigm.
    """
    paradigm = _detect_paradigm(script)
    prompt = _build_prompt(script, config, paradigm)
    if memory_context:
        prompt = memory_context + "\n\n" + prompt
    raw = invoke_claude(prompt, max_tokens=16000, temperature=0)
    result = extract_json(raw)
    result["detected_paradigm"] = paradigm
    return result
