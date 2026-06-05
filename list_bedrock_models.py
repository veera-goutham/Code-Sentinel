"""
list_bedrock_models.py — Discovery utility
Lists all Anthropic models AND inference profiles available in your Bedrock region.
Use the output to pick the right BEDROCK_MODEL_ID for .env.
"""
import os
import boto3
from dotenv import load_dotenv

load_dotenv()

region = os.getenv("AWS_BEDROCK_REGION", "ap-south-2")
print(f"Bedrock region: {region}\n")

bedrock = boto3.client("bedrock", region_name=region)

# ------------------------------------------------------------------
# 1. Foundation models (the "raw" model IDs)
# ------------------------------------------------------------------
print("=" * 70)
print("ANTHROPIC FOUNDATION MODELS in this region")
print("=" * 70)
try:
    resp = bedrock.list_foundation_models(byProvider="anthropic")
    models = resp.get("modelSummaries", [])
    if not models:
        print("  (none found in this region)")
    for m in models:
        model_id = m.get("modelId", "")
        name = m.get("modelName", "")
        inference_types = m.get("inferenceTypesSupported", [])
        status = m.get("modelLifecycle", {}).get("status", "")
        on_demand = "ON_DEMAND" in inference_types

        marker = "DIRECT  " if on_demand else "PROFILE "
        print(f"  [{marker}] {model_id}")
        print(f"             name: {name}   status: {status}")
        print()
except Exception as e:
    print(f"  Error: {e}\n")

# ------------------------------------------------------------------
# 2. Inference profiles (newer Claude models need these)
# ------------------------------------------------------------------
print("=" * 70)
print("INFERENCE PROFILES in this region")
print("=" * 70)
try:
    resp = bedrock.list_inference_profiles()
    profiles = resp.get("inferenceProfileSummaries", [])
    if not profiles:
        print("  (no inference profiles in this region)")
    for p in profiles:
        name = p.get("inferenceProfileName", "")
        pid = p.get("inferenceProfileId", "")
        if "claude" in name.lower() or "anthropic" in pid.lower() or "claude" in pid.lower():
            print(f"  {pid}")
            print(f"     name: {name}")
            print()
except Exception as e:
    print(f"  Error: {e}\n")

print("=" * 70)
print("HOW TO USE")
print("=" * 70)
print("- [DIRECT]  IDs: copy into .env as-is — they work with invoke_model.")
print("- [PROFILE] IDs: you need the matching inference profile ID instead.")
print("- For our build, ANY Claude 3.5 Sonnet or newer is fine.")
print()
print("Recommended priority order:")
print("  1. anthropic.claude-3-5-sonnet-20241022-v2:0 (direct, reliable)")
print("  2. Any 'apac.' / 'global.' inference profile for Sonnet 4 or 4.5")
print("  3. anthropic.claude-3-5-sonnet-20240620-v1:0 (older direct)")