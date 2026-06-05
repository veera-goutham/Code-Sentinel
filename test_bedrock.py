"""
test_bedrock.py — Sanity check #2
Calls Claude Sonnet 4.5 on Bedrock with a hello-world prompt.

Expected output:
    Claude says: <some friendly response>

If you get AccessDeniedException → model access isn't enabled in this region.
If you get ResourceNotFoundException → model ID typo, or model not available here.
If you get ExpiredTokenException → AWS credentials issue.
"""
import os
import json
import boto3
from dotenv import load_dotenv

load_dotenv()

region = os.getenv("AWS_BEDROCK_REGION", "ap-south-2")
model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-sonnet-4-5-20250929-v1:0")

print(f"Calling Bedrock in region: {region}")
print(f"Model: {model_id}\n")

bedrock = boto3.client("bedrock-runtime", region_name=region)

body = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 200,
    "messages": [
        {
            "role": "user",
            "content": "Say hello in one short sentence and confirm you're Claude Sonnet 4.5 running on AWS Bedrock.",
        }
    ],
}

response = bedrock.invoke_model(
    modelId=model_id,
    body=json.dumps(body),
    contentType="application/json",
    accept="application/json",
)

result = json.loads(response["body"].read())
text = result["content"][0]["text"]
print(f"Claude says: {text}")
