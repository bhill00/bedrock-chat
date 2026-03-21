"""Pydantic models for OpenAI-compatible /v1/chat/completions endpoint."""

from pydantic import BaseModel, Field
from typing import Optional, Union


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
