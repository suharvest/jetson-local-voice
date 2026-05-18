"""Tests for Session.messages token-aware trimming (Task #18 A2)."""
from __future__ import annotations

import logging

import pytest

from openvoicestream_agent.event_bus import EventBus
from openvoicestream_agent.session import Session


# A trivial "tokenizer" that estimates ~1 token per 4 chars. This avoids
# pulling a real Qwen tokenizer at test time (CI / offline runs).
def _fake_counter(text: str) -> int:
    return max(1, len(text) // 4)


SYSTEM = "system prompt"


def _add_turn(session: Session, user_text: str, assistant_text: str) -> None:
    session.add_user(user_text)
    session.add_assistant(assistant_text)


# ── 1. trim disabled when max_input_tokens is None ──────────────────


def test_no_trim_when_max_is_none() -> None:
    session = Session(max_input_tokens=None, token_counter=_fake_counter)
    for i in range(1000):
        _add_turn(session, f"hello {i}", f"hi {i}")
    msgs = session.messages(SYSTEM)
    # system + 1000 user + 1000 assistant
    assert len(msgs) == 1 + 2000
    assert msgs[0] == {"role": "system", "content": SYSTEM}


# ── 2. oldest turns dropped under pressure ──────────────────────────


def test_trim_drops_oldest_turns() -> None:
    session = Session(max_input_tokens=200, token_counter=_fake_counter)
    # Each turn ~ 40 chars -> ~10 tokens + overhead. Make many.
    for i in range(50):
        _add_turn(session, f"user message number {i:03d}", f"assistant reply {i:03d}")
    msgs = session.messages(SYSTEM)
    # The earliest turns must be gone.
    contents = [m["content"] for m in msgs]
    assert "user message number 000" not in contents
    assert "assistant reply 000" not in contents
    # And the most recent turn must remain.
    assert "user message number 049" in contents
    assert "assistant reply 049" in contents


# ── 3. system prompt always preserved at index 0 ────────────────────


def test_system_prompt_preserved() -> None:
    session = Session(max_input_tokens=50, token_counter=_fake_counter)
    for i in range(20):
        _add_turn(session, f"long user content here {i}" * 4, f"long reply {i}" * 4)
    msgs = session.messages(SYSTEM)
    assert msgs[0] == {"role": "system", "content": SYSTEM}


# ── 4. latest turn always preserved ─────────────────────────────────


def test_latest_turn_preserved() -> None:
    session = Session(max_input_tokens=20, token_counter=_fake_counter)
    for i in range(30):
        _add_turn(session, f"u{i}" * 50, f"a{i}" * 50)
    msgs = session.messages(SYSTEM)
    # Latest user + assistant should be the last two messages.
    assert msgs[-2]["role"] == "user"
    assert msgs[-1]["role"] == "assistant"
    assert "u29" in msgs[-2]["content"]
    assert "a29" in msgs[-1]["content"]


# ── 5. drops whole turns, never odd halves ──────────────────────────


def test_drops_whole_turns_not_single_messages() -> None:
    session = Session(max_input_tokens=100, token_counter=_fake_counter)
    for i in range(30):
        _add_turn(session, f"user-{i}" * 8, f"asst-{i}" * 8)
    msgs = session.messages(SYSTEM)
    # Skip the system entry and check alternation in the remaining.
    history = msgs[1:]
    # Length must be even (no half-turn left behind).
    assert len(history) % 2 == 0, f"odd history left: {len(history)}"
    for idx in range(0, len(history), 2):
        assert history[idx]["role"] == "user", f"role mismatch at {idx}: {history[idx]}"
        assert history[idx + 1]["role"] == "assistant", (
            f"role mismatch at {idx+1}: {history[idx+1]}"
        )


# ── 6. on_session_trimmed event fires when trimming happens ─────────


def test_trim_emits_event() -> None:
    bus = EventBus()
    received: list[dict] = []
    bus.subscribe("on_session_trimmed", lambda d: received.append(d))

    session = Session(
        max_input_tokens=100,
        token_counter=_fake_counter,
        event_bus=bus,
    )
    for i in range(30):
        _add_turn(session, f"user message {i}" * 5, f"assistant reply {i}" * 5)
    _ = session.messages(SYSTEM)
    assert len(received) == 1, f"expected one event, got {len(received)}"
    evt = received[0]
    assert evt["dropped_turns"] > 0
    assert evt["kept_turns"] >= 1
    assert evt["sid"] == session.sid


def test_no_event_when_under_budget() -> None:
    bus = EventBus()
    received: list[dict] = []
    bus.subscribe("on_session_trimmed", lambda d: received.append(d))

    session = Session(
        max_input_tokens=10_000,
        token_counter=_fake_counter,
        event_bus=bus,
    )
    _add_turn(session, "short", "short")
    _ = session.messages(SYSTEM)
    assert received == []


# ── 7. budget is max * 0.75 (25% margin) ────────────────────────────


def test_25_percent_margin() -> None:
    """Tokens kept after trim must fit under max * 0.75, not max itself."""
    max_tokens = 400
    session = Session(max_input_tokens=max_tokens, token_counter=_fake_counter)
    # Each turn ~120 chars -> 30 tokens content + 4 overhead per message.
    for i in range(20):
        _add_turn(
            session,
            "u" * 120 + f"{i}",
            "a" * 120 + f"{i}",
        )
    msgs = session.messages(SYSTEM)

    def msg_cost(m: dict) -> int:
        return _fake_counter(m["content"]) + 4

    total = sum(msg_cost(m) for m in msgs)
    # Must fit the 75% budget (one full turn may put us slightly over the
    # 0.75 line because trim stops once dropping more would violate the
    # "keep latest turn" rule — but for this fixture the latest turn alone
    # fits well under budget, so we should be at or below it).
    budget = int(max_tokens * 0.75)
    assert total <= budget, f"total {total} exceeds budget {budget}"
    # And must definitely be below the full max_tokens too.
    assert total < max_tokens


# ── 8. trim emits a warning log line ────────────────────────────────


def test_trim_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    session = Session(max_input_tokens=80, token_counter=_fake_counter)
    for i in range(20):
        _add_turn(session, f"user-{i}" * 10, f"asst-{i}" * 10)
    with caplog.at_level(logging.WARNING, logger="openvoicestream_agent.session"):
        _ = session.messages(SYSTEM)
    msgs = [r.message for r in caplog.records if "session trimmed" in r.message]
    assert msgs, f"expected 'session trimmed' warning, got: {caplog.text}"


# ── 9. trailing in-progress user (no assistant yet) survives trim ───


def test_fallback_uses_conservative_estimate(monkeypatch) -> None:
    """When the HF tokenizer can't be loaded, the char-based fallback must
    be conservative (~1.5 tokens/char) so long Chinese prompts don't slip
    past the trim and crash the engine on max_seq_len overflow.
    """
    import math

    from openvoicestream_agent import session as session_mod

    # Force tokenizer load to fail (cache the _FALLBACK sentinel).
    session_mod._TOKENIZER_CACHE.clear()
    session_mod._TOKENIZER_CACHE["dummy-model"] = session_mod._FALLBACK

    # 100-char "Chinese" content; conservative estimate: ceil(100 * 1.5) = 150.
    assert session_mod._fallback_estimate("你" * 100) == 150
    assert session_mod._count_tokens(session_mod._FALLBACK, "你" * 100) == 150

    # And the Session._count path should pick this up (no token_counter override).
    s = Session(
        max_input_tokens=None, tokenizer_model="dummy-model", token_counter=None
    )
    assert s._count("你" * 100) == 150
    session_mod._TOKENIZER_CACHE.clear()


def test_fallback_trim_uses_conservative_budget(monkeypatch) -> None:
    """A 100-char history under the fallback estimator should reflect a
    ~150-token cost (not ~25). This means trim decisions made with the
    fallback estimator are at least as aggressive as with a real tokenizer.
    """
    import math

    from openvoicestream_agent import session as session_mod

    session_mod._TOKENIZER_CACHE.clear()
    session_mod._TOKENIZER_CACHE["dummy-model"] = session_mod._FALLBACK

    s = Session(max_input_tokens=None, tokenizer_model="dummy-model")
    cost = s._msg_tokens({"role": "user", "content": "你" * 100})
    # 150 (content) + 4 (overhead) = 154
    assert cost == 154, (
        f"fallback msg cost should be 154 (1.5/char + overhead), got {cost}"
    )
    session_mod._TOKENIZER_CACHE.clear()


# ── MED-2: trim clears cache_warmed (avoid wasted prefix_cache call) ──


def test_trim_clears_cache_warmed() -> None:
    """When trimming happens, cache_warmed must be reset to False so the
    next call doesn't waste a round-trip with prefix_cache=True against a
    server-side KV cache that no longer matches our history."""
    session = Session(max_input_tokens=100, token_counter=_fake_counter)
    session.cache_warmed = True  # pretend we already warmed
    for i in range(30):
        _add_turn(session, f"user message {i}" * 5, f"assistant reply {i}" * 5)
    _ = session.messages(SYSTEM)
    assert session.cache_warmed is False, (
        "trim should clear cache_warmed (upstream KV cache no longer valid)"
    )


def test_trim_does_not_set_prefix_cache_disabled() -> None:
    """MED-2: trim must NOT touch prefix_cache_disabled — that latch is
    reserved for engine-level prefix_cache rejection, not history mutation.
    Conflating the two would permanently disable prefix_cache for the rest
    of the dialogue after the very first trim."""
    session = Session(max_input_tokens=100, token_counter=_fake_counter)
    session.cache_warmed = True
    assert session.prefix_cache_disabled is False  # baseline
    for i in range(30):
        _add_turn(session, f"user message {i}" * 5, f"assistant reply {i}" * 5)
    _ = session.messages(SYSTEM)
    assert session.prefix_cache_disabled is False, (
        "trim must not poison prefix_cache_disabled"
    )


def test_no_trim_no_cache_warmed_change() -> None:
    """When the history fits the budget, cache_warmed must be preserved
    (no spurious cache invalidation)."""
    session = Session(max_input_tokens=10_000, token_counter=_fake_counter)
    session.cache_warmed = True
    _add_turn(session, "short", "short")
    _ = session.messages(SYSTEM)
    assert session.cache_warmed is True, (
        "no trim happened — cache_warmed must stay True"
    )


def test_trim_with_cache_warmed_false_is_no_op() -> None:
    """Defensive: trimming when cache_warmed was already False should
    not flip anything and not log the (now-redundant) cache-invalidation
    info message."""
    session = Session(max_input_tokens=100, token_counter=_fake_counter)
    assert session.cache_warmed is False
    for i in range(30):
        _add_turn(session, f"user message {i}" * 5, f"assistant reply {i}" * 5)
    _ = session.messages(SYSTEM)
    assert session.cache_warmed is False  # unchanged


def test_trailing_user_message_preserved_across_trim() -> None:
    """If the assistant reply is still streaming when messages() is
    called, the last message is a lone user turn — it must not get
    paired into a phantom turn or dropped."""
    session = Session(max_input_tokens=100, token_counter=_fake_counter)
    for i in range(20):
        _add_turn(session, f"user-{i}" * 8, f"asst-{i}" * 8)
    session.add_user("brand new question still in flight")
    msgs = session.messages(SYSTEM)
    assert msgs[-1]["role"] == "user"
    assert msgs[-1]["content"] == "brand new question still in flight"
