"""
lineage.py — Data Lineage Agent

Parses a Glue job script to extract data lineage: source tables/S3 paths,
transformation steps, and sink destinations. Produces a Mermaid flowchart
and a structured node inventory. Does NOT modify the script.
"""
from aws_helpers import extract_json, invoke_claude

_PROMPT_TEMPLATE = """\
You are a data engineer specializing in AWS Glue ETL pipelines. \
Your task is to extract the complete data lineage from the Glue job \
script below and represent it as a Mermaid diagram plus structured metadata.

Job configuration:
- Job type    : {job_type}
- Worker type : {worker_type}
- Workers     : {num_workers}
- Timeout     : {timeout} minutes

Script:
```python
{script}
```

Extract the data lineage by identifying:

INPUTS — locate every data source read in the script:
- spark.read.parquet(), spark.read.csv(), spark.read.json(), \
  spark.read.format().load()
- glueContext.create_dynamic_frame.from_catalog() or from_options()
- boto3 s3.get_object() or s3.download_file() calls
- Any JDBC or catalog reads

TRANSFORMATIONS — identify each distinct transformation step in order:
- Joins (join(), crossJoin(), merge())
- Aggregations (groupBy(), agg(), pivot())
- Filters and projections (filter(), where(), select(), drop(), withColumn())
- Deduplication (dropDuplicates(), distinct())
- Unions (union(), unionByName())
- Custom UDF applications
- Schema changes / casts

OUTPUTS — locate every data sink written in the script:
- .write.parquet(), .write.csv(), .write.json(), .write.format().save()
- glueContext.write_dynamic_frame.from_options()
- boto3 s3.put_object() or s3.upload_file() calls
- Any JDBC or catalog writes

MERMAID DIAGRAM RULES
Build a graph LR flowchart using these exact node shapes:
- Source / sink table  : NodeId["Label"]
- Join operation       : NodeId((join))
- Aggregation          : NodeId[[aggregate]]
- Filter / transform   : NodeId{{"transform"}}
- Union                : NodeId((union))

NODE LABEL QUOTING (Mermaid v10 strict syntax):
- ALWAYS wrap source/sink labels in straight double-quotes: NodeId["label here"]
- Any label containing slashes, colons, dots, equals signs, or spaces MUST \
  be quoted. If in doubt, quote it.
- Do NOT use <br/> or <br> inside labels. Labels must be single-line.
- For S3 URIs in labels: strip the s3:// prefix and shorten to \
  bucket/last-folder only — never include full paths or partition keys.
  GOOD: T1["transactions (retail-warehouse/transactions)"]
  BAD:  T1[s3://retail-warehouse/transactions/dt=2026-05-30/]
- If you need to show both a name and a location, use " - " as separator \
  on a single line:
  GOOD: T1["transactions - retail-warehouse/transactions"]
  BAD:  T1[transactions<br/>s3://retail-warehouse/transactions/]

GOOD example (valid Mermaid v10):
  graph LR
  T1["transactions (retail-warehouse)"]
  S1["store lookup - dim/store"]
  J1((join))
  A1[["aggregate by region"]]
  O1["daily output - warehouse/daily-agg"]
  T1 --> J1
  S1 --> J1
  J1 --> A1
  A1 --> O1

BAD example (will fail Mermaid v10 — do NOT produce this):
  graph LR
  T1[transactions<br/>s3://retail-warehouse/transactions/dt=2026-05-30/]
  T1 --> J1

Other requirements:
- Every source node must have at least one outgoing edge.
- Every sink node must have at least one incoming edge.
- Transformation nodes must appear between their input and output nodes.
- Node IDs must be short plain alphanumeric tokens (T1, J1, A1, O1, etc.) \
  — no spaces, no special characters, no underscores needed.
- Edges use --> with no labels and no quotes: T1 --> J1
- Do not add a title line or subgraph wrappers; start directly with \
  "graph LR".

ESTIMATED SIZE
For each input table, infer "small" / "medium" / "large" only if the \
script contains a comment or variable name that strongly implies it \
(e.g. "# lookup table", "dim_", "# ~1M rows"). Otherwise use null.

DEPENDENCY SUMMARY
Write one sentence describing what downstream systems or teams likely \
depend on this job's outputs, inferred from destination paths, table \
names, or comments in the script.

Respond with ONLY a JSON object — no markdown fences, no prose outside JSON:
{{
  "mermaid_diagram": "<complete Mermaid graph as a single string — \
escape all newlines as \\n; must start with 'graph LR\\n'>",
  "input_tables": [
    {{
      "name": "<DataFrame variable name or catalog table name>",
      "source": "<S3 URI, catalog database.table, or JDBC URL>",
      "estimated_size": "<small | medium | large | null>"
    }}
  ],
  "output_tables": [
    {{
      "name": "<DataFrame variable name or catalog table name>",
      "destination": "<S3 URI, catalog database.table, or JDBC URL>"
    }}
  ],
  "transformations": [
    "<ordered bullet 1 — name and purpose of this transformation step>",
    "<ordered bullet 2>",
    "<ordered bullet 3>"
  ],
  "dependency_summary": "<one sentence>"
}}

Rules:
- mermaid_diagram must start with "graph LR\\n" (newline escaped).
- input_tables must have at least one entry; if no explicit read is \
  found, infer from variable names and note it with source "inferred".
- output_tables must have at least one entry; same fallback applies.
- transformations must have between 3 and 6 entries.
- estimated_size must be the string "null" when not inferable — \
  do NOT use JSON null for this field; use the string "null".
- Do not invent sources or destinations not present in the script.
"""


def _build_prompt(script: str, config: dict) -> str:
    worker_type = config.get("worker_type") or "n/a (Python shell)"
    num_workers = config.get("number_of_workers") or "n/a"
    timeout = config.get("timeout") or "n/a"
    job_type = config.get("job_type") or "glueetl"
    return _PROMPT_TEMPLATE.format(
        job_type=job_type,
        worker_type=worker_type,
        num_workers=num_workers,
        timeout=timeout,
        script=script,
    )


def run(script: str, config: dict, memory_context: str = "") -> dict:
    """
    Extract data lineage from *script* and return a Mermaid diagram
    plus structured node metadata.

    Args:
        script: Raw PySpark / Python shell source code of the Glue job.
        config: Dict with keys worker_type, number_of_workers, timeout,
                job_type (all optional / may be None).
        memory_context: Optional past-review context prepended to the prompt.

    Returns:
        Dict with keys mermaid_diagram, input_tables, output_tables,
        transformations, dependency_summary.
    """
    prompt = _build_prompt(script, config)
    if memory_context:
        prompt = memory_context + "\n\n" + prompt
    raw = invoke_claude(prompt, max_tokens=4000, temperature=0)
    result = extract_json(raw)
    # Guarantee list types and normalize string "null" -> None for estimated_size
    result.setdefault("input_tables", [])
    result.setdefault("output_tables", [])
    result.setdefault("transformations", [])
    if result["input_tables"] is None:
        result["input_tables"] = []
    if result["output_tables"] is None:
        result["output_tables"] = []
    if result["transformations"] is None:
        result["transformations"] = []
    for table in result["input_tables"]:
        if table.get("estimated_size") == "null":
            table["estimated_size"] = None
    return result
