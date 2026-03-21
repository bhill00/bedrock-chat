"""OpenAI-compatible /v1/chat/completions endpoint.

Translates OpenAI API format to Bedrock Converse API calls using the existing
bedrock-chat infrastructure. Supports streaming (SSE) and non-streaming modes,
per-request inference parameters, and accurate token usage reporting.
"""

import json
import logging
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.bedrock import compose_args_for_converse_api, calculate_price
from app.repositories.models.conversation import (
    SimpleMessageModel,
    TextContentModel,
)
from app.repositories.models.custom_bot import GenerationParamsModel
from app.routes.schemas.conversation import type_model_name
from app.routes.schemas.openai_compat import OpenAIChatCompletionRequest
from app.utils import get_bedrock_runtime_client

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter(prefix="/v1", tags=["openai_compat"])

# Aliases: OpenAI/Anthropic model names → bedrock-chat type_model_name values.
# Clients can use either the alias or the native name directly.
MODEL_ALIASES: dict[str, str] = {
    # OpenAI drop-in aliases
    "gpt-4": "claude-v4.5-sonnet",
    "gpt-4o": "claude-v4.5-sonnet",
    "gpt-3.5-turbo": "claude-v4.5-haiku",
    # Anthropic API-style names
    "claude-sonnet-4-5": "claude-v4.5-sonnet",
    "claude-sonnet-4": "claude-v4-sonnet",
    "claude-opus-4-5": "claude-v4.5-opus",
    "claude-opus-4": "claude-v4-opus",
    "claude-haiku-4-5": "claude-v4.5-haiku",
    "claude-haiku-3-5": "claude-v3.5-haiku",
    # Bedrock model IDs (pass-through to native name)
    "us.anthropic.claude-sonnet-4-6": "claude-v4.5-sonnet",
    "us.anthropic.claude-opus-4-6-v1": "claude-v4.5-opus",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": "claude-v4.5-haiku",
}

# All valid type_model_name values (duplicated here to validate without importing Literal)
VALID_MODEL_NAMES = {
    "claude-v4-opus", "claude-v4.1-opus", "claude-v4.5-opus",
    "claude-v4-sonnet", "claude-v4.5-sonnet", "claude-v4.5-haiku",
    "claude-v3.5-sonnet", "claude-v3.5-sonnet-v2", "claude-v3.7-sonnet",
    "claude-v3.5-haiku", "claude-v3-haiku", "claude-v3-opus",
    "mistral-7b-instruct", "mixtral-8x7b-instruct",
    "mistral-large", "mistral-large-2",
    "amazon-nova-pro", "amazon-nova-lite", "amazon-nova-micro",
    "deepseek-r1",
    "llama3-3-70b-instruct", "llama3-2-1b-instruct",
    "llama3-2-3b-instruct", "llama3-2-11b-instruct",
    "llama3-2-90b-instruct",
    "gpt-oss-20b", "gpt-oss-120b",
}


def resolve_model(model: str) -> type_model_name:
    """Resolve an OpenAI/Anthropic/Bedrock model name to a bedrock-chat type_model_name."""
    resolved = MODEL_ALIASES.get(model, model)
    if resolved not in VALID_MODEL_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model}. Use one of: {sorted(VALID_MODEL_NAMES)} or aliases: {sorted(MODEL_ALIASES.keys())}",
        )
    return resolved  # type: ignore


def openai_messages_to_simple(
    messages: list,
) -> tuple[list[str], list[SimpleMessageModel]]:
    """Convert OpenAI messages array to (system_instructions, SimpleMessageModel list).

    Separates system messages into instructions and converts user/assistant
    messages to the SimpleMessageModel format that compose_args_for_converse_api expects.
    """
    system_parts: list[str] = []
    simple_messages: list[SimpleMessageModel] = []

    for msg in messages:
        if msg.role == "system":
            if isinstance(msg.content, str):
                system_parts.append(msg.content)
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        system_parts.append(block["text"])
                    elif isinstance(block, str):
                        system_parts.append(block)

        elif msg.role in ("user", "assistant"):
            if isinstance(msg.content, str):
                content = [TextContentModel(content_type="text", body=msg.content)]
            elif isinstance(msg.content, list):
                content = []
                for block in msg.content:
                    if isinstance(block, str):
                        content.append(TextContentModel(content_type="text", body=block))
                    elif isinstance(block, dict) and block.get("type") == "text":
                        content.append(
                            TextContentModel(content_type="text", body=block["text"])
                        )
                    # TODO: image_url support could be added here
            else:
                content = [TextContentModel(content_type="text", body=str(msg.content or ""))]

            simple_messages.append(
                SimpleMessageModel(role=msg.role, content=content)
            )

    return system_parts, simple_messages


def build_generation_params(req: OpenAIChatCompletionRequest) -> GenerationParamsModel:
    """Build GenerationParamsModel from per-request OpenAI parameters."""
    return GenerationParamsModel(
        max_tokens=req.max_tokens or 4096,
        temperature=req.temperature if req.temperature is not None else 1.0,
        top_p=req.top_p if req.top_p is not None else 0.999,
        top_k=req.top_k if req.top_k is not None else 250,
        stop_sequences=([req.stop] if isinstance(req.stop, str) else req.stop) or [],
    )


# ---------------------------------------------------------------------------
# Non-streaming endpoint
# ---------------------------------------------------------------------------
@router.post("/chat/completions")
def chat_completions(request: Request, req: OpenAIChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    current_user = getattr(request.state, "current_user", None)
    model_name = resolve_model(req.model)
    request_id = uuid.uuid4().hex[:8]

    if req.stream:
        return StreamingResponse(
            stream_openai_response(req, model_name, request_id),
            media_type="text/event-stream",
        )

    # Non-streaming: invoke Bedrock and return full response
    instructions, messages = openai_messages_to_simple(req.messages)
    generation_params = build_generation_params(req)

    args = compose_args_for_converse_api(
        messages=messages,
        model=model_name,
        instructions=instructions,
        generation_params=generation_params,
        stream=False,
    )

    client = get_bedrock_runtime_client()
    try:
        response = client.converse(**args)
    except Exception as e:
        logger.error(f"Bedrock error: {e}")
        raise HTTPException(status_code=502, detail=f"Bedrock error: {e}")

    # Extract text from response
    reply_text = ""
    for block in response.get("output", {}).get("message", {}).get("content", []):
        if "text" in block:
            reply_text += block["text"]

    usage = response.get("usage", {})
    input_tokens = usage.get("inputTokens", 0)
    output_tokens = usage.get("outputTokens", 0)

    logger.info(
        "USAGE endpoint=openai model=%s input_tokens=%d output_tokens=%d total_tokens=%d user=%s",
        model_name, input_tokens, output_tokens, input_tokens + output_tokens,
        current_user.id if current_user else "anonymous",
    )

    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Streaming endpoint (SSE)
# ---------------------------------------------------------------------------
async def stream_openai_response(
    req: OpenAIChatCompletionRequest,
    model_name: type_model_name,
    request_id: str,
):
    """Stream Bedrock Converse API response as OpenAI SSE chunks."""
    chunk_id = f"chatcmpl-{request_id}"
    created = int(time.time())

    instructions, messages = openai_messages_to_simple(req.messages)
    generation_params = build_generation_params(req)

    args = compose_args_for_converse_api(
        messages=messages,
        model=model_name,
        instructions=instructions,
        generation_params=generation_params,
        stream=True,
    )

    # Role chunk
    yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': req.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

    client = get_bedrock_runtime_client()
    try:
        response = client.converse_stream(**args)

        input_tokens = 0
        output_tokens = 0

        for event in response["stream"]:
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    text = delta["text"]
                    yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': None}]})}\n\n"

            elif "metadata" in event:
                usage = event["metadata"].get("usage", {})
                input_tokens = usage.get("inputTokens", 0)
                output_tokens = usage.get("outputTokens", 0)

            elif "messageStop" in event:
                logger.info(
                    "USAGE endpoint=openai_stream model=%s input_tokens=%d output_tokens=%d total_tokens=%d",
                    model_name, input_tokens, output_tokens, input_tokens + output_tokens,
                )
                yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': req.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': input_tokens, 'completion_tokens': output_tokens, 'total_tokens': input_tokens + output_tokens}})}\n\n"

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Models list
# ---------------------------------------------------------------------------
@router.get("/models")
def list_models(request: Request):
    """List available models in OpenAI format."""
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "created": 0,
                "owned_by": "bedrock-chat",
            }
            for name in sorted(VALID_MODEL_NAMES)
        ],
    }
