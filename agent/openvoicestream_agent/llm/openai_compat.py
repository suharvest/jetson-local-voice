"""OpenAI-compatible streaming chat backend."""
from __future__ import annotations

from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from .base import LLMBackend


class OpenAICompatBackend(LLMBackend):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        default_params: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.default_params = dict(default_params or {})
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def stream(
        self, messages: list[dict[str, str]], **kw: Any
    ) -> AsyncIterator[str]:
        params: dict[str, Any] = {**self.default_params, **kw}
        extra_body = params.pop("extra_body", None)
        request_kwargs: dict[str, Any] = {
            "model": params.pop("model", self.model),
            "messages": messages,
            "stream": True,
        }
        if extra_body:
            request_kwargs["extra_body"] = extra_body
        # Forward any remaining caller params (temperature, max_tokens...).
        request_kwargs.update(params)

        response = await self.client.chat.completions.create(**request_kwargs)
        async for chunk in response:
            try:
                delta = chunk.choices[0].delta.content
            except (IndexError, AttributeError):
                delta = None
            if delta:
                yield delta

    async def aclose(self) -> None:
        try:
            await self.client.close()
        except Exception:  # pragma: no cover - best effort
            pass
