"""
test_gen.py — Test Generation Agent

Takes a Glue job script and generates a complete pytest test suite for it:
unit tests for individual transformations, schema-validation assertions, and
mocked SparkSession / boto3 calls. Does NOT modify the original script.
"""
from aws_helpers import extract_json, invoke_claude

_PROMPT_TEMPLATE = """\
You are a senior Python test engineer specializing in AWS Glue ETL pipelines \
and PySpark data processing. Your task is to generate a complete pytest test \
suite for the Glue job script below.

Job configuration:
- Job type    : {job_type}
- Worker type : {worker_type}
- Workers     : {num_workers}
- Timeout     : {timeout} minutes

Script:
```python
{script}
```

Generate a pytest file that thoroughly tests the logic in this script. \
Follow these rules:

STRUCTURE
- Begin with all necessary imports (pytest, unittest.mock, pyspark, chispa, \
  etc. — include only what the generated tests actually use).
- Group tests with descriptive function names: test_<function_or_step>_<scenario>.
- Each test function must have a single-line docstring stating exactly what \
  it verifies.

COVERAGE STRATEGY
Prioritise in this order:
1. Happy-path test for every distinct transformation or business-logic \
   function identifiable in the script.
2. One edge-case test per function: empty DataFrame / empty input, missing \
   or null column values, zero-row result sets.
3. One negative test where applicable: invalid input type, schema mismatch, \
   missing required argument.

MOCKING RULES — PySpark jobs (job_type = "glueetl"):
- Use pytest-mock (the `mocker` fixture) or unittest.mock.patch.
- Mock spark.read.parquet, spark.read.csv, spark.read.format, \
  glueContext.create_dynamic_frame, and any boto3 client calls.
- Use chispa's assert_df_equality for DataFrame comparisons where chispa \
  would be a natural fit; fall back to collected row comparison otherwise.
- Provide a local SparkSession fixture using pyspark.sql.SparkSession.builder \
  with master("local[1]") — name it `spark`.

MOCKING RULES — Python shell jobs (job_type = "pythonshell"):
- Use unittest.mock.patch for all boto3 client methods.
- Test pure-Python functions directly without spinning up Spark.
- Mock S3 GetObject / PutObject calls with MagicMock return values that \
  include a "Body" key with a BytesIO / mock read() method.

ASSERTIONS
- Always assert on the actual returned value or DataFrame content, \
  not just that a mock was called.
- For DataFrame tests: assert on row count AND at least one column value.
- For dict/JSON outputs: assert every key in the expected shape is present.

OUTPUT FORMAT
Respond with ONLY a JSON object — no markdown fences, no prose outside JSON:
{{
  "pytest_code": "<complete Python test file as a single JSON string — \
escape all newlines as \\n, escape all double-quotes inside the code as \\\";\
the file must start with the import block and be syntactically valid Python>",
  "test_count": <integer — number of def test_ functions in pytest_code>,
  "coverage_notes": [
    "<bullet 1 — what this suite covers>",
    "<bullet 2 — what it does NOT cover and why>"
  ],
  "fixtures_needed": [
    "<name and brief description of any pytest fixture the user must wire up \
before the tests will pass, e.g. 'spark — local SparkSession (already \
included in generated code)' or 'glue_context — mock GlueContext pointing \
at local catalog'>"
  ]
}}

Rules:
- pytest_code must be a complete, syntactically valid Python file. \
  If pasted into test_job.py it must import cleanly (fixtures may need \
  wiring up, but no NameErrors should occur from missing imports).
- test_count must equal the exact number of `def test_` functions \
  present in pytest_code.
- coverage_notes must have between 2 and 4 entries.
- fixtures_needed must list every fixture referenced in pytest_code \
  that is NOT defined within the file itself; if all fixtures are \
  self-contained, return an empty list [].
- Do not invent logic not present in the script; if a function cannot \
  be tested in isolation (e.g. it is a monolithic top-level script with \
  no callable functions), generate tests that exercise it via subprocess \
  or by importing the module with mocked entry points and note this in \
  coverage_notes.
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
    Generate a pytest test suite for *script*.

    Args:
        script: Raw PySpark / Python shell source code of the Glue job.
        config: Dict with keys worker_type, number_of_workers, timeout,
                job_type (all optional / may be None).
        memory_context: Optional past-review context prepended to the prompt.

    Returns:
        Dict with keys pytest_code, test_count, coverage_notes,
        fixtures_needed.
    """
    prompt = _build_prompt(script, config)
    if memory_context:
        prompt = memory_context + "\n\n" + prompt
    raw = invoke_claude(prompt, max_tokens=4000, temperature=0.3)
    result = extract_json(raw)
    # Guarantee list types even if Claude returns null for empty collections
    result.setdefault("coverage_notes", [])
    result.setdefault("fixtures_needed", [])
    if result["coverage_notes"] is None:
        result["coverage_notes"] = []
    if result["fixtures_needed"] is None:
        result["fixtures_needed"] = []
    return result
