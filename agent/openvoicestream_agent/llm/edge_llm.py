"""edge-llm-chat-service backend (OpenAI-compatible + prefix-cache hooks)."""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from openai import APIError

from ..session import Session
from .openai_compat import LLMStreamError, OpenAICompatBackend

logger = logging.getLogger(__name__)


class _PrefixCacheError(Exception):
    """Raised internally when an upstream failure is identified as
    prefix_cache-specific. We never let this leak out of EdgeLLMBackend —
    it is the trigger for the no-prefix retry path."""


# Substrings that indicate the upstream rejected (only) the prefix_cache
# request. Kept conservative so unrelated 4xx/5xx still propagate to A3
# (transient retry) / A1+A5 (fail-fast).
_PREFIX_CACHE_MARKERS = (
    "prefix_cache",
    "prefix cache",
    "kv cache",
    "kv_cache",
    "kv mismatch",
    "prefix_messages",
)


def _is_prefix_cache_failure(exc: BaseException) -> bool:
    """Heuristic: True when ``exc`` looks like a prefix_cache-only failure.

    Upstream (tensorrt-edge-llm api_server.py L207-L224) returns
    ``JSONResponse(status_code=400, content={"error": str(exc)})`` when
    ``_build_prefix_formatted_request`` raises. That surfaces here as an
    ``APIError`` whose ``str(exc)`` contains the original ValueError text
    (e.g. "prefix_cache requires prefix_messages or at least two messages").
    Mid-stream prefix-cache blowups arrive as ``LLMStreamError``; we apply
    the same substring check.
    """
    msg = str(exc).lower()
    return any(marker in msg for marker in _PREFIX_CACHE_MARKERS)


class EdgeLLMBackend(OpenAICompatBackend):
    """Adds edge-llm's ``save_system_prompt_kv_cache`` / ``prefix_cache``
    flags, plus an A4 fallback: if the upstream rejects ``prefix_cache``
    we retry the *same* call without it and pin the session so subsequent
    calls also skip prefix_cache (until ``session.reset()`` clears it).

    Important: we deliberately do NOT reuse ``session.cache_warmed`` as
    the disable flag. A successful drain always re-sets ``cache_warmed``,
    which would loop us straight back into ``prefix_cache=True`` on the
    next turn. ``session.prefix_cache_disabled`` is an independent latch.
    """

    def _build_extra_body(self, session: Session | None) -> dict[str, Any]:
        use_prefix_cache = (
            session is not None
            and session.cache_warmed
            and not session.prefix_cache_disabled
        )
        if use_prefix_cache:
            return {
                "prefix_cache": True,
                "return_cache_metrics": True,
            }
        # Cold path OR warm-but-disabled path. Both ask edge-llm to cache
        # the system prompt KV (cheap, no prefix-formatting risk) and to
        # report cache metrics so the dashboard stays informative.
        return {
            "save_system_prompt_kv_cache": True,
            "return_cache_metrics": True,
        }

    def _disable_prefix_cache(
        self, session: Session | None, exc: BaseException
    ) -> None:
        """Latch ``prefix_cache_disabled`` and notify the event bus."""
        if session is None:
            return
        already = session.prefix_cache_disabled
        session.prefix_cache_disabled = True
        if already:
            return
        bus = getattr(session, "event_bus", None)
        if bus is None:
            return
        try:
            bus.emit(
                "on_prefix_cache_disabled",
                {
                    "reason": str(exc),
                    "sid": getattr(session, "sid", None),
                },
            )
        except Exception:  # pragma: no cover - defensive
            logger.debug(
                "event_bus emit on_prefix_cache_disabled failed",
                exc_info=True,
            )

    async def _stream_once(
        self,
        messages: list[dict[str, str]],
        session: Session | None,
        caller_kw: dict[str, Any],
        *,
        disable_inner_retry: bool = False,
    ) -> AsyncIterator[str]:
        """One pass through the base streamer with our cache flags injected.

        ``disable_inner_retry`` is set when invoked from the A4 fallback
        path so that the A3 retry inside ``OpenAICompatBackend.stream`` is
        bypassed for this call — otherwise a prefix_cache failure that
        surfaces as 5xx would trigger A3 retry (1 retry) and *then* the
        A4 fallback would also trigger A3 retry, for a worst case of 4
        upstream calls per turn.
        """
        kw = dict(caller_kw)
        cache_flags = self._build_extra_body(session)
        caller_extra = dict(kw.pop("extra_body", None) or {})
        cache_flags.update(caller_extra)
        kw["extra_body"] = cache_flags
        if disable_inner_retry:
            kw["_retry_disabled"] = True
        async for delta in super().stream(messages, **kw):
            yield delta

    async def stream(  # type: ignore[override]
        self,
        messages: list[dict[str, str]],
        session: Session | None = None,
        **kw: Any,
    ) -> AsyncIterator[str]:
        used_prefix_cache = (
            session is not None
            and session.cache_warmed
            and not session.prefix_cache_disabled
        )

        yielded_any = False
        try:
            async for delta in self._stream_once(messages, session, kw):
                yielded_any = True
                yield delta
        except (APIError, LLMStreamError) as exc:
            if used_prefix_cache and _is_prefix_cache_failure(exc):
                # Always latch the flag so future turns skip prefix_cache,
                # even when we can't safely retry this turn (mid-stream
                # failure → tokens already shipped → retry would duplicate).
                self._disable_prefix_cache(session, exc)
                if yielded_any:
                    # Bubble up; A3 won't retry mid-stream errors either.
                    raise
                logger.warning(
                    "prefix_cache failed (%s); retrying without prefix_cache",
                    exc,
                )
                # Immediate retry — same messages, but _build_extra_body
                # now sees prefix_cache_disabled=True and omits the flag.
                # Disable inner A3 retry: the base class already retried
                # this call once before raising; doing so again here would
                # double-count attempts (worst case 4 upstream hits/turn).
                async for delta in self._stream_once(
                    messages, session, kw, disable_inner_retry=True
                ):
                    yield delta
                if session is not None:
                    session.cache_warmed = True
                return
            raise

        if session is not None:
            session.cache_warmed = True
