"""Per-conversation Session state.

History is normally append-only (edge-llm's prefix cache rewards stable
prefixes), but edge-llm engines have a hard ``max_seq_len`` ceiling
(e.g. 3072 tokens for engines-3072). Past that point the server crashes
with a CUDA fatal abort. We therefore support an *optional* token-aware
trim: when ``max_input_tokens`` is set, the oldest *full turns* are
dropped before the prompt is shipped to the LLM, while always preserving
the system prompt + the most recent user/assistant pair.

When ``max_input_tokens`` is ``None`` (the default), behaviour matches
the original invariant — no trimming, prefix cache stays warm.
"""
from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


# Module-level tokenizer cache so we only pay HF load cost once per
# model name across the whole process (and across Session instances).
_TOKENIZER_CACHE: dict[str, Any] = {}


# Sentinel used to cache "tokenizer load failed; use char/4 fallback".
_FALLBACK = object()


def _get_tokenizer(model_name: str) -> Any:
    """Lazy, cached HuggingFace tokenizer loader.

    Returns ``_FALLBACK`` (and warns once) when transformers isn't
    installed or the model can't be loaded — callers must handle this
    by falling back to the char/4 estimator. We deliberately don't
    raise: a missing tokenizer should degrade gracefully, not crash
    the voice loop.
    """
    if model_name not in _TOKENIZER_CACHE:
        try:
            from transformers import AutoTokenizer  # local import: heavy
            _TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
        except Exception as e:
            logger.warning(
                "tokenizer load failed for %r (%s); using char/4 estimate",
                model_name, e,
            )
            _TOKENIZER_CACHE[model_name] = _FALLBACK
    return _TOKENIZER_CACHE[model_name]


def _fallback_estimate(text: str) -> int:
    """Conservative char-based estimator.

    Use ``ceil(chars * 1.5)`` rather than ``chars // 4`` because Chinese
    text averages ~1.5 tokens per character (not 0.25). A too-low estimate
    causes long Chinese prompts to slip past the trim/guard and crash the
    engine on ``max_seq_len`` overflow.
    """
    return max(1, math.ceil(len(text) * 1.5))


def _count_tokens(tokenizer: Any, text: str) -> int:
    """Best-effort token count. Falls back to a conservative char estimate."""
    if tokenizer is _FALLBACK:
        return _fallback_estimate(text)
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return len(ids)
    except Exception:  # pragma: no cover - defensive
        return _fallback_estimate(text)


@dataclass
class Session:
    sid: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: float = field(default_factory=time.time)
    history: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    locale: str = "zh"
    cache_warmed: bool = False
    # Token-aware trim controls (A2). None disables trimming entirely
    # and preserves the original append-only invariant.
    max_input_tokens: int | None = None
    tokenizer_model: str = "Qwen/Qwen3-4B-AWQ"
    # Optional override for tokenization (used by tests to avoid pulling
    # a real model from HF). Signature: (text: str) -> int.
    token_counter: Callable[[str], int] | None = None
    # A4 placeholder: when True, the LLM backend should skip prefix
    # cache hints. This task only wires the field; logic lands in A4.
    prefix_cache_disabled: bool = False
    # Optional EventBus injected by the app. Plain attribute (not a
    # dataclass field) so tests can construct Sessions cheaply without
    # importing the EventBus type.
    event_bus: Any = None

    def reset(self) -> None:
        """Clear conversation history and per-session cache latches.

        Called when a new dialogue starts (mode switch / explicit clear /
        user-driven reset). Both ``cache_warmed`` and the A4
        ``prefix_cache_disabled`` flag get reset so the next turn can
        re-warm normally — otherwise a session that ever hit a prefix-cache
        failure would skip prefix_cache forever, even across logical
        conversations.
        """
        self.history.clear()
        self.cache_warmed = False
        self.prefix_cache_disabled = False

    def add_user(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})

    def messages(self, system_prompt: str) -> list[dict[str, str]]:
        """Return OpenAI-format messages with system prompt prepended.

        When ``max_input_tokens`` is set, oldest turns are dropped so the
        total token budget stays within ``max_input_tokens * 0.75``
        (25% margin reserved for the response).
        """
        base: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            *self.history,
        ]
        if self.max_input_tokens is None:
            return base
        return self._trim_to_budget(base, self.max_input_tokens)

    # ── internal ────────────────────────────────────────────────────

    def _count(self, text: str) -> int:
        if self.token_counter is not None:
            return self.token_counter(text)
        tok = _get_tokenizer(self.tokenizer_model)
        return _count_tokens(tok, text)

    def _msg_tokens(self, msg: dict[str, str]) -> int:
        # Add a small per-message overhead (~4 tokens) to mirror
        # OpenAI's chat template framing — keeps the estimate honest
        # without needing the full chat template here.
        return self._count(msg.get("content", "")) + 4

    def _trim_to_budget(
        self, messages: list[dict[str, str]], max_tokens: int
    ) -> list[dict[str, str]]:
        """Drop oldest *whole turns* until the prompt fits the budget.

        Invariants:
          * messages[0] (system prompt) always kept.
          * Latest user+assistant turn always kept.
          * Turns are dropped as pairs (user+assistant) — never half.
          * Budget is ``max_tokens * 0.75`` (25% margin for response).
        """
        budget = int(max_tokens * 0.75)
        if not messages:
            return messages

        system = messages[0]
        rest = list(messages[1:])  # may be odd-length while a turn is in flight
        if not rest:
            return messages

        # Detect a trailing single user message (turn in progress: assistant
        # hasn't replied yet). Pin it to the tail; only the completed
        # prefix is grouped into (user, assistant) turns.
        trailing: list[dict[str, str]] = []
        if len(rest) % 2 == 1:
            trailing = [rest[-1]]
            rest = rest[:-1]

        # Group remaining history into (user, assistant) turns. Skip any
        # leading orphan assistant defensively.
        turns: list[list[dict[str, str]]] = []
        i = 0
        while i < len(rest) and rest[i].get("role") != "user":
            i += 1
        while i + 1 < len(rest):
            turns.append([rest[i], rest[i + 1]])
            i += 2

        sys_tokens = self._msg_tokens(system)
        trailing_tokens = sum(self._msg_tokens(m) for m in trailing)
        turn_tokens = [sum(self._msg_tokens(m) for m in t) for t in turns]
        total = sys_tokens + trailing_tokens + sum(turn_tokens)

        if total <= budget:
            return messages

        # Drop from the front (oldest) until under budget OR only one
        # turn remains — the latest completed turn must stay.
        kept_turns = list(turns)
        dropped = 0
        while total > budget and len(kept_turns) > 1:
            removed = kept_turns.pop(0)
            total -= sum(self._msg_tokens(m) for m in removed)
            dropped += 1

        kept_count = len(kept_turns)
        approx_tokens = total

        if dropped > 0:
            logger.warning(
                "session trimmed: dropped %d turns, kept %d turns, ~%d tokens",
                dropped, kept_count, approx_tokens,
            )
            # MED-2: trimming changes the prefix that the upstream KV cache
            # is keyed against. If we leave ``cache_warmed=True`` the next
            # turn will send ``prefix_cache=True`` and the server-side KV
            # cache (built for the pre-trim history) will mismatch — A4
            # would then catch the resulting failure and pin the session
            # to no-prefix-cache for the rest of the conversation. That's
            # a wasted upstream round-trip *and* a permanent latch from
            # an avoidable mismatch. Clearing the flag here means the
            # very next call goes cold (saves the system prompt KV again)
            # without ever attempting the doomed prefix_cache path.
            #
            # We deliberately do NOT touch ``prefix_cache_disabled``: that
            # latch tracks "the engine reject prefix_cache for this
            # session" — independent of trim. Conflating them would mean
            # a single trim event permanently disables prefix_cache for
            # the rest of the dialogue, which we don't want.
            if self.cache_warmed:
                logger.info(
                    "session trim invalidates upstream KV cache, "
                    "clearing cache_warmed (prefix_cache_disabled untouched)"
                )
                self.cache_warmed = False
            bus = getattr(self, "event_bus", None)
            if bus is not None:
                try:
                    bus.emit(
                        "on_session_trimmed",
                        {
                            "dropped_turns": dropped,
                            "kept_turns": kept_count,
                            "approx_tokens": approx_tokens,
                            "budget": budget,
                            "max_input_tokens": max_tokens,
                            "sid": self.sid,
                        },
                    )
                except Exception:  # pragma: no cover - defensive
                    logger.debug(
                        "event_bus emit on_session_trimmed failed", exc_info=True
                    )

        rebuilt: list[dict[str, str]] = [system]
        for t in kept_turns:
            rebuilt.extend(t)
        rebuilt.extend(trailing)
        return rebuilt
