"""The provider-independent LLM contract used by the agent core."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from agent.llm.models import LLMResponse, Message, ModelConfig, ModelInfo, StreamChunk, ToolSpec

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class LLMProvider(Protocol):
    """Everything the agent needs from a model backend.

    Implementations raise only ``agent.llm.errors.LLMError`` subclasses.
    New providers implement this protocol and register with a
    ``ProviderRegistry``; the orchestration core is never modified.
    """

    name: str
    config: ModelConfig

    async def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse: ...

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamChunk]: ...

    async def generate_structured(
        self,
        messages: list[Message],
        schema: type[T],
    ) -> T: ...

    def model_info(self) -> ModelInfo: ...

    async def health_check(self) -> bool: ...

    async def aclose(self) -> None: ...
