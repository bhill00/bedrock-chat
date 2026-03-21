"""OpenAI and Anthropic compatible API endpoints.

Designed to work with the Published Bot API infrastructure. When a bot is
published, its API endpoint gains standard API endpoints alongside the existing
/conversation endpoint:

  POST /v1/chat/completions  — OpenAI-compatible (streaming + non-streaming)
  POST /v1/messages          — Anthropic-compatible (streaming + non-streaming)
  GET  /v1/models            — List available models

The bot's settings (instruction, model, guardrails, generation_params) serve
as defaults, with per-request overrides for temperature, max_tokens, top_p,
and stop sequences. Guardrails are always applied (not overridable).
"""

import json
import logging
import os
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.bedrock import compose_args_for_converse_api
from app.repositories.models.conversation import (
    SimpleMessageModel,
    TextContentModel,
)
from app.repositories.models.custom_bot import GenerationParamsModel
from app.routes.schemas.conversation import type_model_name
from app.routes.schemas.openai_compat import (
    AnthropicMessagesRequest,
    OpenAIChatCompletionRequest,
)
from app.user import User
from app.utils import get_bedrock_runtime_client

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter(prefix="/v1", tags=["openai_compat"])

PUBLISHED_API_ID = os.environ.get("PUBLISHED_API_ID", None)

# Aliases: OpenAI/Anthropic model names → bedrock-chat type_model_name values.
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


def _get_bot_context() -> tuple:
    """Load bot settings when running as a published API.

    Returns (instruction, generation_params, guardrail) from the bot config.
    If not in published API mode, returns (None, None, None).
    """
    if not PUBLISHED_API_ID:
        return None, None, None

    try:
        from app.usecases.bot import fetch_bot

        user = User.from_published_api_id(PUBLISHED_API_ID)
        bot_id = user.id.split("#")[1] if "#" in user.id else user.id
        _, bot = fetch_bot(user, bot_id)
        return bot.instruction, bot.generation_params, bot.bedrock_guardrails
    except Exception as e:
        logger.warning(f"Failed to load bot context: {e}")
        return None, None, None


def openai_messages_to_simple(
    messages: list,
) -> tuple[list[str], list[SimpleMessageModel]]:
    """Convert OpenAI messages array to (system_instructions, SimpleMessageModel list)."""
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
            else:
                content = [TextContentModel(content_type="text", body=str(msg.content or ""))]

            simple_messages.append(
                SimpleMessageModel(role=msg.role, content=content)
            )

    return system_parts, simple_messages


def build_generation_params(
    req: OpenAIChatCompletionRequest,
    bot_params: GenerationParamsModel | None = None,
) -> GenerationParamsModel:
    """Build GenerationParamsModel with bot defaults + per-request overrides.

    Bot settings provide the base. Per-request params override when explicitly set.
    This lets the bot owner set safe defaults while the API consumer can tune
    temperature/max_tokens for their specific use case.
    """
    if bot_params:
        return GenerationParamsModel(
            max_tokens=req.max_tokens if req.max_tokens is not None else bot_params.max_tokens,
            temperature=req.temperature if req.temperature is not None else bot_params.temperature,
            top_p=req.top_p if req.top_p is not None else bot_params.top_p,
            top_k=req.top_k if req.top_k is not None else bot_params.top_k,
            stop_sequences=(
                ([req.stop] if isinstance(req.stop, str) else req.stop)
                if req.stop is not None
                else bot_params.stop_sequences
            ),
        )

    return GenerationParamsModel(
        max_tokens=req.max_tokens or 4096,
        temperature=req.temperature if req.temperature is not None else 1.0,
        top_p=req.top_p if req.top_p is not None else 0.999,
        top_k=req.top_k if req.top_k is not None else 250,
        stop_sequences=([req.stop] if isinstance(req.stop, str) else req.stop) or [],
    )


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------
@router.post("/chat/completions")
def chat_completions(request: Request, req: OpenAIChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint.

    When running as a Published Bot API:
    - Bot's instruction is prepended to system messages
    - Bot's generation_params serve as defaults (overridable per-request)
    - Bot's guardrails are always applied (not overridable)

    When running in the main app:
    - Uses only the parameters provided in the request
    - Requires Cognito JWT auth
    """
    current_user = getattr(request.state, "current_user", None)
    model_name = resolve_model(req.model)
    request_id = uuid.uuid4().hex[:8]

    # Load bot context (instruction, params, guardrails) if in published API mode
    bot_instruction, bot_gen_params, bot_guardrail = _get_bot_context()

    if req.stream:
        return StreamingResponse(
            stream_openai_response(
                req, model_name, request_id,
                bot_instruction, bot_gen_params, bot_guardrail,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming
    instructions, messages = openai_messages_to_simple(req.messages)

    # Prepend bot instruction to system instructions
    if bot_instruction:
        instructions.insert(0, bot_instruction)

    generation_params = build_generation_params(req, bot_gen_params)

    args = compose_args_for_converse_api(
        messages=messages,
        model=model_name,
        instructions=instructions,
        generation_params=generation_params,
        guardrail=bot_guardrail,
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
        "USAGE endpoint=openai model=%s input_tokens=%d output_tokens=%d total_tokens=%d user=%s bot=%s",
        model_name, input_tokens, output_tokens, input_tokens + output_tokens,
        current_user.id if current_user else "anonymous",
        PUBLISHED_API_ID or "none",
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
# Streaming
# ---------------------------------------------------------------------------
async def stream_openai_response(
    req: OpenAIChatCompletionRequest,
    model_name: type_model_name,
    request_id: str,
    bot_instruction: str | None = None,
    bot_gen_params: GenerationParamsModel | None = None,
    bot_guardrail=None,
):
    """Stream Bedrock Converse API response as OpenAI SSE chunks."""
    chunk_id = f"chatcmpl-{request_id}"
    created = int(time.time())

    instructions, messages = openai_messages_to_simple(req.messages)

    # Prepend bot instruction
    if bot_instruction:
        instructions.insert(0, bot_instruction)

    generation_params = build_generation_params(req, bot_gen_params)

    args = compose_args_for_converse_api(
        messages=messages,
        model=model_name,
        instructions=instructions,
        generation_params=generation_params,
        guardrail=bot_guardrail,
        stream=True,
    )

    # Role chunk
    yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': req.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

    client = get_bedrock_runtime_client()
    try:
        response = client.converse_stream(**args)

        input_tokens = 0
        output_tokens = 0

        # Bedrock Converse API event order: contentBlockDelta* → messageStop → metadata
        # We defer the final "stop" chunk until after metadata so token counts are accurate.
        stop_received = False

        for event in response["stream"]:
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    text = delta["text"]
                    yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': None}]})}\n\n"

            elif "messageStop" in event:
                stop_received = True

            elif "metadata" in event:
                usage = event["metadata"].get("usage", {})
                input_tokens = usage.get("inputTokens", 0)
                output_tokens = usage.get("outputTokens", 0)

        # Emit final chunk with accurate token counts after all events processed
        if stop_received:
            logger.info(
                "USAGE endpoint=openai_stream model=%s input_tokens=%d output_tokens=%d total_tokens=%d bot=%s",
                model_name, input_tokens, output_tokens, input_tokens + output_tokens,
                PUBLISHED_API_ID or "none",
            )
            yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': req.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': input_tokens, 'completion_tokens': output_tokens, 'total_tokens': input_tokens + output_tokens}})}\n\n"

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

    yield "data: [DONE]\n\n"


# ===========================================================================
# Anthropic-compatible endpoint: POST /v1/messages
# ===========================================================================
def anthropic_messages_to_simple(
    req: AnthropicMessagesRequest,
) -> tuple[list[str], list[SimpleMessageModel]]:
    """Convert Anthropic messages request to (instructions, SimpleMessageModel list).

    Anthropic format differs from OpenAI:
    - System prompt is a separate `system` field, not in messages
    - Messages array only contains user/assistant turns
    """
    instructions: list[str] = []
    if req.system:
        instructions.append(req.system)

    simple_messages: list[SimpleMessageModel] = []
    for msg in req.messages:
        if isinstance(msg.content, str):
            content = [TextContentModel(content_type="text", body=msg.content)]
        elif isinstance(msg.content, list):
            content = []
            for block in msg.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    content.append(
                        TextContentModel(content_type="text", body=block["text"])
                    )
                elif isinstance(block, str):
                    content.append(TextContentModel(content_type="text", body=block))
        else:
            content = [TextContentModel(content_type="text", body=str(msg.content))]

        simple_messages.append(SimpleMessageModel(role=msg.role, content=content))

    return instructions, simple_messages


def build_generation_params_anthropic(
    req: AnthropicMessagesRequest,
    bot_params: GenerationParamsModel | None = None,
) -> GenerationParamsModel:
    """Build GenerationParamsModel from Anthropic request params + bot defaults."""
    if bot_params:
        return GenerationParamsModel(
            max_tokens=req.max_tokens,  # Required in Anthropic API
            temperature=req.temperature if req.temperature is not None else bot_params.temperature,
            top_p=req.top_p if req.top_p is not None else bot_params.top_p,
            top_k=req.top_k if req.top_k is not None else bot_params.top_k,
            stop_sequences=req.stop_sequences if req.stop_sequences is not None else bot_params.stop_sequences,
        )

    return GenerationParamsModel(
        max_tokens=req.max_tokens,
        temperature=req.temperature if req.temperature is not None else 1.0,
        top_p=req.top_p if req.top_p is not None else 0.999,
        top_k=req.top_k if req.top_k is not None else 250,
        stop_sequences=req.stop_sequences or [],
    )


@router.post("/messages")
def anthropic_messages(request: Request, req: AnthropicMessagesRequest):
    """Anthropic-compatible /v1/messages endpoint.

    Same bot integration as /v1/chat/completions — bot instruction, guardrails,
    and generation_params apply. Key differences from the OpenAI endpoint:
    - System prompt is a separate `system` field
    - `max_tokens` is required
    - Response uses `content[]` with `type`/`text` blocks
    - Streaming uses Anthropic SSE event types
    - Supports `top_k` natively
    """
    current_user = getattr(request.state, "current_user", None)
    model_name = resolve_model(req.model)
    request_id = uuid.uuid4().hex[:8]

    bot_instruction, bot_gen_params, bot_guardrail = _get_bot_context()

    if req.stream:
        return StreamingResponse(
            stream_anthropic_response(
                req, model_name, request_id,
                bot_instruction, bot_gen_params, bot_guardrail,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming
    instructions, messages = anthropic_messages_to_simple(req)
    if bot_instruction:
        instructions.insert(0, bot_instruction)

    generation_params = build_generation_params_anthropic(req, bot_gen_params)

    args = compose_args_for_converse_api(
        messages=messages,
        model=model_name,
        instructions=instructions,
        generation_params=generation_params,
        guardrail=bot_guardrail,
        stream=False,
    )

    client = get_bedrock_runtime_client()
    try:
        response = client.converse(**args)
    except Exception as e:
        logger.error(f"Bedrock error: {e}")
        raise HTTPException(status_code=502, detail=f"Bedrock error: {e}")

    # Build Anthropic response format
    content_blocks = []
    for block in response.get("output", {}).get("message", {}).get("content", []):
        if "text" in block:
            content_blocks.append({"type": "text", "text": block["text"]})

    usage = response.get("usage", {})
    input_tokens = usage.get("inputTokens", 0)
    output_tokens = usage.get("outputTokens", 0)

    logger.info(
        "USAGE endpoint=anthropic model=%s input_tokens=%d output_tokens=%d total_tokens=%d user=%s bot=%s",
        model_name, input_tokens, output_tokens, input_tokens + output_tokens,
        current_user.id if current_user else "anonymous",
        PUBLISHED_API_ID or "none",
    )

    return {
        "id": f"msg_{request_id}",
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": req.model,
        "stop_reason": response.get("stopReason", "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Anthropic streaming
# ---------------------------------------------------------------------------
async def stream_anthropic_response(
    req: AnthropicMessagesRequest,
    model_name: type_model_name,
    request_id: str,
    bot_instruction: str | None = None,
    bot_gen_params: GenerationParamsModel | None = None,
    bot_guardrail=None,
):
    """Stream Bedrock Converse API response in Anthropic SSE format.

    Translates Converse API events to Anthropic streaming event types:
    - message_start, content_block_start, content_block_delta,
      content_block_stop, message_delta, message_stop
    """
    instructions, messages = anthropic_messages_to_simple(req)
    if bot_instruction:
        instructions.insert(0, bot_instruction)

    generation_params = build_generation_params_anthropic(req, bot_gen_params)

    args = compose_args_for_converse_api(
        messages=messages,
        model=model_name,
        instructions=instructions,
        generation_params=generation_params,
        guardrail=bot_guardrail,
        stream=True,
    )

    client = get_bedrock_runtime_client()
    try:
        response = client.converse_stream(**args)

        input_tokens = 0
        output_tokens = 0
        block_index = 0

        # message_start
        yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': f'msg_{request_id}', 'type': 'message', 'role': 'assistant', 'content': [], 'model': req.model, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"

        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

        for event in response["stream"]:
            if "contentBlockStart" in event:
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

            elif "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': delta['text']}})}\n\n"

            elif "contentBlockStop" in event:
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                block_index += 1

            elif "messageStop" in event:
                stop_reason = event["messageStop"].get("stopReason", "end_turn")

            elif "metadata" in event:
                usage = event["metadata"].get("usage", {})
                input_tokens = usage.get("inputTokens", 0)
                output_tokens = usage.get("outputTokens", 0)

        # message_delta with usage
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}})}\n\n"

        # message_stop
        logger.info(
            "USAGE endpoint=anthropic_stream model=%s input_tokens=%d output_tokens=%d total_tokens=%d bot=%s",
            model_name, input_tokens, output_tokens, input_tokens + output_tokens,
            PUBLISHED_API_ID or "none",
        )
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

    except Exception as e:
        logger.error(f"Anthropic streaming error: {e}")
        yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': str(e)}})}\n\n"


# ---------------------------------------------------------------------------
# GET /v1/models
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
