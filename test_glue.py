"""
test_glue.py — Sanity check #1
Lists your Glue jobs to confirm AWS credentials + region + permissions are working.

Expected output:
    Found 5 Glue jobs in ap-south-2:
      - customer_daily_agg
      - customer_lookup
      - inventory_pipeline
      - pii_masking_etl
      - sales_dim_load
"""
import os
import boto3
from dotenv import load_dotenv

load_dotenv()

region = os.getenv("AWS_GLUE_REGION", "ap-south-2")

print(f"Connecting to AWS Glue in region: {region}\n")

glue = boto3.client("glue", region_name=region)

response = glue.list_jobs()
job_names = response.get("JobNames", [])

print(f"Found {len(job_names)} Glue jobs in {region}:")
for name in sorted(job_names):
    print(f"  - {name}")

if not job_names:
    print("\n⚠️  No jobs found. Check that you're in the right AWS region.")
    print("   Your jobs were created in Hyderabad (ap-south-2).")
