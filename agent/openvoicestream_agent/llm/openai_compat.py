"""OpenAI-compatible streaming chat backend."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI

from .base import LLMBackend

logger = logging.getLogger(__name__)


class LLMStreamError(Exception):
    """Raised when an upstream OpenAI-compatible server signals failure
    mid-stream (typically via an SSE chunk with ``finish_reason="error"``)
    while still returning HTTP 200. Not a subclass of ``APIError`` so the
    retry logic deliberately leaves it alone — by the time we see it the
    caller has already received partial tokens and a transparent retry
    would emit duplicates."""


def _is_retryable(exc: BaseException) -> bool:
    """True when ``exc`` is a transient upstream failure worth retrying."""
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIError):
        status = getattr(exc, "status_code", None)
        if isinstance(status, int) and 500 <= status < 600:
            return True
    return False


class OpenAICompatBackend(LLMBackend):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        default_params: dict[str, Any] | None = None,
        retry_on_transient: int = 1,
        retry_backoff_s: float = 0.5,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.default_params = dict(default_params or {})
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        # Number of *additional* attempts after the first failure. The
        # default of 1 means: 1 initial try + 1 retry = up to 2 attempts.
        self._retry_on_transient = max(0, int(retry_on_transient))
        self._retry_backoff_s = max(0.0, float(retry_backoff_s))

    async def _do_stream(
        self, messages: list[dict[str, str]], **kw: Any
    ) -> AsyncIterator[str]:
        """One full attempt at the upstream call. Yields text deltas.

        Detects upstream SSE error frames (``finish_reason="error"``)
        emitted mid-stream and converts them into ``LLMStreamError`` so
        the caller sees a real exception instead of a silent EOS.
        """
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

        # Reset per-call cache metrics; EdgeLLMBackend overrides stream and
        # checks this attribute after the iterator drains.
        self.last_cache_metrics = None
        response = await self.client.chat.completions.create(**request_kwargs)
        async for chunk in response:
            # Surface prefix-cache metrics if the server attached them.
            # edge-llm-chat-service places them on the final chunk envelope
            # (top-level or under `metadata` / `cache_metrics`).
            try:
                me = getattr(chunk, "model_extra", None) or {}
                if not isinstance(me, dict):
                    me = {}
                cm = (
                    getattr(chunk, "cache_metrics", None)
                    or me.get("cache_metrics")
                    or (me.get("metadata") or {}).get("cache_metrics")
                )
                if cm is None and isinstance(chunk, dict):
                    cm = (
                        chunk.get("cache_metrics")
                        or (chunk.get("metadata") or {}).get("cache_metrics")
                    )
                if cm is not None:
                    if hasattr(cm, "model_dump"):
                        cm = cm.model_dump()
                    elif hasattr(cm, "dict"):
                        cm = cm.dict()
                    self.last_cache_metrics = cm
            except Exception:  # pragma: no cover - defensive
                pass
            # Detect upstream "I gave up" SSE frame. edge-llm's
            # api_server.py emits `finish_reason="error"` with an empty
            # delta when streaming generation blows up after HTTP 200.
            try:
                choice0 = chunk.choices[0] if chunk.choices else None
                finish_reason = getattr(choice0, "finish_reason", None)
            except (IndexError, AttributeError):
                choice0 = None
                finish_reason = None
            if finish_reason == "error":
                raise LLMStreamError(
                    "upstream emitted finish_reason=error mid-stream"
                )
            try:
                delta = chunk.choices[0].delta.content
            except (IndexError, AttributeError):
                delta = None
            if delta:
                yield delta

    async def stream(
        self, messages: list[dict[str, str]], **kw: Any
    ) -> AsyncIterator[str]:
        """Public stream entry — wraps ``_do_stream`` with transient-failure
        retry. We only retry while we have *not yet yielded a token*: once
        the caller sees any text we must not duplicate it on a re-attempt.

        ``LLMStreamError`` is never retried — by definition it surfaces
        after the upstream has already produced (or tried to produce)
        output, so a retry would risk double-speak.

        Internal protocol — ``_retry_disabled`` (kwarg, not user-facing):
            When ``True``, the A3 transient-failure retry loop is bypassed
            (single attempt). EdgeLLMBackend's A4 fallback path passes this
            so we don't end up with A3-retry × A4-retry = 4 upstream calls
            when prefix_cache failures arrive as 5xx. ``_`` prefix marks it
            as an internal contract between EdgeLLM and the base class —
            callers should not rely on it.
        """
        retry_disabled = bool(kw.pop("_retry_disabled", False))
        attempts = 1 if retry_disabled else (1 + self._retry_on_transient)
        last_exc: BaseException | None = None
        for attempt in range(attempts):
            gen = self._do_stream(messages, **kw)
            try:
                first = await gen.__anext__()
            except StopAsyncIteration:
                # Empty stream — treat as a successful (if uninteresting)
                # call. Drain the generator's aclose() for hygiene.
                await gen.aclose()
                return
            except BaseException as e:  # noqa: BLE001
                # Failure before any token was yielded → safe to retry.
                await gen.aclose()
                if _is_retryable(e) and attempt < attempts - 1:
                    last_exc = e
                    logger.warning(
                        "LLM transient failure on attempt %d/%d: %r — "
                        "retrying in %.2fs",
                        attempt + 1, attempts, e, self._retry_backoff_s,
                    )
                    await asyncio.sleep(self._retry_backoff_s)
                    continue
                raise
            # First token in hand. From here on, do NOT retry — yield
            # everything through the caller as-is, including any failure.
            try:
                yield first
                async for tok in gen:
                    yield tok
            finally:
                await gen.aclose()
            return
        # Loop fell through without a return — means we exhausted retries
        # entirely on the connect/first-token path. Re-raise the last
        # observed exception.
        if last_exc is not None:  # pragma: no cover - defensive
            raise last_exc

    async def aclose(self) -> None:
        try:
            await self.client.close()
        except Exception:  # pragma: no cover - best effort
            pass
