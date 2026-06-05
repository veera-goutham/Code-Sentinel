# Code Sentinel

AI-powered code review and documentation for AWS Glue data pipelines — catches performance waste, security risks, and data lineage gaps in seconds.

Upload a PySpark/SQL/Jupyter script or point it at a live Glue job. Five parallel AI agents analyse the code and return an optimised rewrite, a Word doc, a security report, and a Mermaid lineage diagram — all in under 30 seconds.

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (v24+)
- An AWS account with Bedrock model access enabled for `claude-sonnet-4-5` and `amazon.titan-embed-text-v2` *(optional — the UI loads and the Upload path works without credentials)*

---

## Quick Start

```bash
cp .env.example .env
# Edit .env and fill in your AWS credentials (optional — UI loads without them)

# Create the audit log file so Docker can bind-mount it as a file (not a directory)
touch code_sentinel_audit.jsonl   # Linux/Mac
# On Windows PowerShell: New-Item -ItemType File code_sentinel_audit.jsonl

docker compose up --build
# Open http://localhost:8501
```

---

## Running Without AWS Credentials

You can explore the UI without any AWS credentials:

- The Streamlit sidebar loads normally, showing Service Status dots (Bedrock and S3/Glue will be red).
- The **Upload a file** path works fully — drop in any `.py`, `.sql`, or `.ipynb` script and run a review.
- The **From AWS Glue** path will show a friendly error ("Failed to list Glue jobs") and stop — no Python traceback.
- ChromaDB memory is local and always available.

---

## Architecture

```
Streamlit UI
    └── orchestrator.py
            └── ThreadPoolExecutor (5 agents in parallel)
                    ├── performance_agent  → optimised rewrite + cost estimate
                    ├── documentation_agent → Word doc generation
                    ├── security_agent     → PII + compliance scan
                    ├── lineage_agent      → Mermaid data-flow diagram
                    └── test_gen_agent     → hidden from UI by design
                            ↓
                    AWS Bedrock: Claude Sonnet 4.5 (review) + Titan Embed V2 (memory)
                    ChromaDB: organisational memory (vector search over past decisions)
                    S3: approved-script overwrite + backup + doc upload
                    JSONL: local audit trail (code_sentinel_audit.jsonl)
```

---

## Demo Jobs

The following sample Glue jobs are pre-loaded in the demo AWS account (`ap-south-2`):

| Job | Description |
|-----|-------------|
| `etl_customer_orders` | Ingests raw order events from S3, joins with customer dim |
| `transform_product_events` | Normalises product click-stream data for the analytics layer |
| `aggregate_daily_sales` | Rolls up transactional data to daily sales summaries |
| `process_user_clickstream` | Sessionises raw clickstream and computes engagement metrics |
| `sync_inventory_data` | Syncs inventory deltas from the warehouse feed into the data lake |

---

## Tech Stack

- **Python 3.12**, **Streamlit** — UI framework
- **boto3** — AWS SDK (Glue, S3, Bedrock)
- **AWS Bedrock** — Claude Sonnet 4.5 (code review & chat), Titan Embed V2 (memory embeddings)
- **AWS Glue + S3** — script source, optimised-script write-back, doc storage
- **ChromaDB** — local vector store for organisational memory
- **python-docx** — Word document generation

---

## Submission

**Ganit GenAI Ideathon 2026**
Submitted by: **Veera Goutham Katam**
Mentor: **Amogh Shetty**
