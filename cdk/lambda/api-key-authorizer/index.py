"""
API Gateway Lambda Authorizer — accepts API key from either:
  - x-api-key header (native API GW / Anthropic SDK default)
  - Authorization: Bearer <key> header (OpenAI SDK default)

Validates the key against API Gateway usage plan keys, then returns
an IAM policy allowing the request. Returns the key as usageIdentifierKey
so per-key throttling and quotas are still enforced.
"""

import boto3
import logging
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Cache the valid keys in-memory for the Lambda container lifetime
# (avoids API GW list call on every request within same container)
_cached_keys: dict[str, str] = {}  # key_value -> key_id


def _load_keys() -> dict[str, str]:
    global _cached_keys
    if _cached_keys:
        return _cached_keys
    client = boto3.client("apigateway", region_name=os.environ.get("AWS_REGION", "us-west-2"))
    paginator = client.get_paginator("get_api_keys")
    keys = {}
    for page in paginator.paginate(includeValues=True):
        for key in page["items"]:
            if key.get("enabled", True):
                keys[key["value"]] = key["id"]
    _cached_keys = keys
    logger.info(f"Loaded {len(keys)} API keys")
    return keys


def handler(event, context):
    logger.info(f"Authorizer invoked for: {event.get('methodArn')}")

    # Extract key from either header (case-insensitive)
    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}

    api_key = None

    # Prefer x-api-key, fall back to Authorization: Bearer
    if headers.get("x-api-key"):
        api_key = headers["x-api-key"].strip()
        logger.info("Using x-api-key header")
    elif headers.get("authorization", "").lower().startswith("bearer "):
        api_key = headers["authorization"][7:].strip()  # strip "Bearer "
        logger.info("Using Authorization: Bearer header")

    if not api_key:
        logger.warning("No API key found in request headers")
        raise Exception("Unauthorized")

    # Validate against known keys
    valid_keys = _load_keys()
    if api_key not in valid_keys:
        # Invalidate cache and retry once (key may have been recently created)
        global _cached_keys
        _cached_keys = {}
        valid_keys = _load_keys()

    if api_key not in valid_keys:
        logger.warning("Invalid API key presented")
        raise Exception("Unauthorized")

    key_id = valid_keys[api_key]
    logger.info(f"Authorized key ID: {key_id}")

    method_arn = event["methodArn"]
    # Wildcard the resource so cached policy covers all methods/paths on this API
    arn_parts = method_arn.split(":")
    api_part = arn_parts[5].split("/")
    region = arn_parts[3]
    account = arn_parts[4]
    api_id = api_part[0]
    stage = api_part[1]
    wildcard_arn = f"arn:aws:execute-api:{region}:{account}:{api_id}/{stage}/*/*"

    return {
        "principalId": key_id,
        "policyDocument": {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Action": "execute-api:Invoke",
                    "Effect": "Allow",
                    "Resource": wildcard_arn,
                }
            ],
        },
        "usageIdentifierKey": api_key,  # keeps per-key throttling/quotas working
    }
