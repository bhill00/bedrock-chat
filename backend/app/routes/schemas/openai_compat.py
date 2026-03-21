"""Pydantic models for OpenAI and Anthropic compatible API endpoints."""

from pydantic import BaseModel
from typing import Optional, Union


# ---------------------------------------------------------------------------
# OpenAI /v1/chat/completions
# ---------------------------------------------------------------------------
class OpenAIMessage(BaseModel):
    role: str
    content: Union[str, list, None] = None
    name: Optional[str] = None


class OpenAIChatCompletionRequest(BaseModel):
    model: str
    messages: list[OpenAIMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, list[str]]] = None
    # Accept but ignore these OpenAI-specific fields
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    n: Optional[int] = None
    user: Optional[str] = None


# ---------------------------------------------------------------------------
# Anthropic /v1/messages
# ---------------------------------------------------------------------------
class AnthropicMessage(BaseModel):
    role: str
    content: Union[str, list]


class AnthropicMessagesRequest(BaseModel):
    model: str
    messages: list[AnthropicMessage]
    system: Optional[str] = None
    max_tokens: int = 4096
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stream: Optional[bool] = False
    stop_sequences: Optional[list[str]] = None
    metadata: Optional[dict] = None
