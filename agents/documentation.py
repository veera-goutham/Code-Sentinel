"""
documentation.py — Documentation Generation Agent

Reads a Glue job script and produces business-readable documentation:
a plain-English summary, input/output inventory, transformation bullets,
operational notes, and a ready-to-paste Python module docstring.
Does NOT modify the script.
"""
from aws_helpers import extract_json, invoke_claude

_PROMPT_TEMPLATE = """\
You are a technical writer specializing in AWS Glue ETL pipelines. \
Your audience is a mix of business analysts and data engineers — \
write for the business analyst first, engineer second. \
When technical jargon is unavoidable, follow it with a brief parenthetical \
explanation.

Read the Glue job script below and produce documentation for it.

Job configuration:
- Job type    : {job_type}
- Worker type : {worker_type}
- Workers     : {num_workers}
- Timeout     : {timeout} minutes

Script:
```python
{script}
```

Produce a JSON object with the following fields. \
Respond with ONLY the JSON — no markdown fences, no prose outside JSON.

{{
  "business_summary": "<2–3 sentence plain-English description of what this \
job does, what business problem it solves, and roughly how often it matters. \
No code references.>",

  "inputs": [
    {{
      "source": "<S3 URI, Glue catalog table name, or JDBC source>",
      "description": "<what data is in this source and why it is needed>"
    }}
  ],

  "outputs": [
    {{
      "destination": "<S3 URI, Glue catalog table name, or target system>",
      "description": "<what data is written here and who consumes it>"
    }}
  ],

  "transformation_logic": [
    "<bullet 1 — describe a distinct step in the data flow in plain English>",
    "<bullet 2>",
    "<bullet 3>"
  ],

  "operational_notes": [
    "<note 1 — scheduling hints, upstream dependencies, retry behaviour, \
or known failure modes inferred from the script>",
    "<note 2>"
  ],

  "generated_docstring": "<a complete Python module-level docstring \
(triple-quoted style, rendered as a plain string value here) following \
this template exactly:\n\nJob Name: <inferred from comments, filename \
references, or job variables in the code; if unknown write Unknown>\n\n\
Purpose: <one sentence>\n\nInputs:\n- <source>: <description>\n\n\
Outputs:\n- <destination>: <description>\n\nSchedule: \
<if inferable from comments or trigger references, else 'not specified'>\n\n\
Owner: <if inferable from comments or tags, else 'not specified'>"
}}

Rules:
- inputs and outputs must each have at least one entry. \
If a source or destination cannot be determined, use \
"unknown — not specified in script" as the source/destination value.
- transformation_logic must have between 3 and 6 entries.
- operational_notes must have between 2 and 4 entries.
- Do not invent facts not present in or strongly implied by the script.
- generated_docstring is a plain string value inside JSON — \
escape newlines as \\n, do NOT use actual newlines inside the JSON string value.
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
    Generate business-readable documentation for *script*.

    Args:
        script: Raw PySpark / Python shell source code of the Glue job.
        config: Dict with keys worker_type, number_of_workers, timeout,
                job_type (all optional / may be None).
        memory_context: Optional past-review context prepended to the prompt.

    Returns:
        Dict with keys business_summary, inputs, outputs,
        transformation_logic, operational_notes, generated_docstring.
    """
    prompt = _build_prompt(script, config)
    if memory_context:
        prompt = memory_context + "\n\n" + prompt
    raw = invoke_claude(prompt, max_tokens=4000, temperature=0)
    return extract_json(raw)
