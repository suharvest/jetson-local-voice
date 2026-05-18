"""AppMode framework: protocol, ModeManager, built-in modes."""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from openvoicestream_agent import Config, Session
from openvoicestream_agent.app_mode import AppMode, ModeContext, ModeManager
from openvoicestream_agent.modes import (
    ChatMode,
    InterpreterMode,
    MonologueMode,
    TranscribeMode,
)


class FakeSLV:
    def __init__(self) -> None:
        self.text_frames: list[str] = []
        self.flushed: int = 0

    async def send_text(self, text: str) -> None:
        self.text_frames.append(text)

    async def flush_tts(self) -> None:
        self.flushed += 1


class FakeLLM:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.calls: list[list[dict]] = []
        self.kwargs: list[dict[str, Any]] = []
        self.last_cache_metrics: dict | None = None

    async def stream(self, messages, **kw):
        self.calls.append(list(messages))
        self.kwargs.append(dict(kw))
        for t in self.tokens:
            yield t

    async def aclose(self) -> None:
        pass


def _make_ctx(cfg=None, llm=None, slv=None, session=None, broadcast=None):
    cfg = cfg or Config(system_prompt="SYS")
    llm = llm or FakeLLM(["hi"])
    slv = slv or FakeSLV()
    session = session or Session()
    broadcasts: list[tuple] = []

    async def _br(name, *args):
        broadcasts.append((name, args))

    bc = broadcast or _br

    events = type("E", (), {"emit": lambda *a, **k: None})()
    ctx = ModeContext(
        config=cfg, slv=slv, llm=llm, session=session, audio=None,
        events=events, broadcast=bc,
    )
    return ctx, broadcasts


# ── AppMode protocol -------------------------------------------------


class _StubMode(AppMode):
    name = "stub"
    display_name = "Stub"
    icon = "•"
    description = "test"

    def __init__(self) -> None:
        self.entered = 0
        self.exited = 0
        self.utterances: list[str] = []
        self.assistant_done = 0
        self.idle_ticks: list[float] = []

    async def enter(self, ctx):
        self.entered += 1

    async def exit(self, ctx):
        self.exited += 1

    async def on_user_utterance(self, ctx, text):
        self.utterances.append(text)

    async def on_assistant_done(self, ctx):
        self.assistant_done += 1

    async def on_session_idle(self, ctx, idle_seconds):
        self.idle_ticks.append(idle_seconds)


@pytest.mark.asyncio
async def test_appmode_subclass_lifecycle():
    m = _StubMode()
    ctx, _ = _make_ctx()
    await m.enter(ctx)
    await m.on_user_utterance(ctx, "hello")
    await m.on_assistant_done(ctx)
    await m.on_session_idle(ctx, 1.5)
    await m.exit(ctx)
    assert m.entered == 1
    assert m.exited == 1
    assert m.utterances == ["hello"]
    assert m.assistant_done == 1
    assert m.idle_ticks == [1.5]
    # preprocess default is pass-through.
    assert m.preprocess_user_text("hi") == "hi"


# ── ModeManager ------------------------------------------------------


@pytest.mark.asyncio
async def test_mode_manager_register_switch_list():
    ctx, broadcasts = _make_ctx()
    mgr = ModeManager(lambda: ctx)
    a, b = _StubMode(), _StubMode()
    b.name = "stub2"
    b.display_name = "Stub2"
    mgr.register(a)
    mgr.register(b)
    # duplicate name rejected
    with pytest.raises(ValueError):
        mgr.register(_StubMode())
    # current before start raises
    with pytest.raises(RuntimeError):
        _ = mgr.current
    await mgr.start("stub")
    assert mgr.current_name == "stub"
    assert a.entered == 1 and b.entered == 0
    listing = mgr.list_all()
    assert [m["name"] for m in listing] == ["stub", "stub2"]
    assert listing[0]["current"] is True and listing[1]["current"] is False
    # switch to second
    await mgr.switch("stub2")
    assert mgr.current_name == "stub2"
    assert a.exited == 1 and b.entered == 1
    # broadcast was fired for mode_change
    names = [n for n, _ in broadcasts]
    assert "on_mode_change" in names
    # no-op when switching to same
    await mgr.switch("stub2")
    assert b.entered == 1
    # unknown name -> KeyError
    with pytest.raises(KeyError):
        await mgr.switch("nope")


@pytest.mark.asyncio
async def test_mode_manager_start_fallback_when_default_missing():
    ctx, _ = _make_ctx()
    mgr = ModeManager(lambda: ctx)
    only = _StubMode()
    only.name = "only"
    mgr.register(only)
    await mgr.start("does-not-exist")
    assert mgr.current_name == "only"


# ── ChatMode ---------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_mode_invokes_default_dialogue_turn():
    ctx, broadcasts = _make_ctx(llm=FakeLLM(["a", "b"]))
    cm = ChatMode()
    await cm.on_user_utterance(ctx, "hello")
    assert ctx.slv.text_frames == ["a", "b"]
    assert ctx.slv.flushed == 1
    assert ctx.session.history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "ab"},
    ]
    # System prompt is the configured default (SYS).
    assert ctx.llm.calls[0][0] == {"role": "system", "content": "SYS"}


# ── InterpreterMode --------------------------------------------------


@pytest.mark.asyncio
async def test_interpreter_mode_clears_history_each_turn():
    cfg = Config(system_prompt="SYS")
    llm = FakeLLM(["EN."])
    ctx, _ = _make_ctx(cfg=cfg, llm=llm)
    # Seed prior history.
    ctx.session.add_user("旧")
    ctx.session.add_assistant("old")
    im = InterpreterMode()
    # We need ModeContext bound to a manager so system prompt resolution
    # walks through interpreter.system_prompt (overrides app SYS).
    mgr = ModeManager(lambda: ctx)
    mgr.register(im)
    await mgr.start("interpreter")
    await im.on_user_utterance(ctx, "你好")
    # History only contains this turn — prior was cleared.
    assert ctx.session.history == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "EN."},
    ]
    # System prompt is interpreter's, not Config's SYS.
    assert ctx.llm.calls[0][0]["role"] == "system"
    assert "real-time interpreter" in ctx.llm.calls[0][0]["content"].lower()


# ── MonologueMode ----------------------------------------------------


@pytest.mark.asyncio
async def test_monologue_mode_enter_starts_task_exit_cancels():
    ctx, _ = _make_ctx()
    mm = MonologueMode()
    mm.interval_s = 10.0  # never fire during this test
    await mm.enter(ctx)
    assert mm._task is not None and not mm._task.done()
    # Ignores user utterances.
    await mm.on_user_utterance(ctx, "hi")
    assert ctx.slv.text_frames == []
    await mm.exit(ctx)
    assert mm._task is None or mm._task.done()


# ── TranscribeMode ---------------------------------------------------


@pytest.mark.asyncio
async def test_transcribe_mode_broadcasts_no_llm_no_history():
    ctx, broadcasts = _make_ctx()
    tm = TranscribeMode()
    await tm.on_user_utterance(ctx, "hello world")
    assert ctx.slv.text_frames == []
    assert ctx.session.history == []
    assert ctx.llm.calls == []
    # Broadcast must have included on_transcribed.
    names = [n for n, _ in broadcasts]
    assert "on_transcribed" in names
    # Find the payload.
    payload = next(args for n, args in broadcasts if n == "on_transcribed")
    assert payload[0] == {"text": "hello world"}


# ── produces_tts contract --------------------------------------------


def test_appmode_default_produces_tts_true():
    """AppMode subclasses default to produces_tts=True (most modes talk)."""
    assert AppMode.produces_tts is True
    assert ChatMode.produces_tts is True
    assert InterpreterMode.produces_tts is True
    # MonologueMode keeps the default True (broadcast loop produces
    # TTS); it ignores user input via preprocess_user_text instead.
    assert MonologueMode.produces_tts is True


def test_transcribe_mode_declares_no_tts():
    """TranscribeMode must opt out so the dispatcher restores IDLE
    after each utterance — otherwise FSM gets stuck in THINKING."""
    assert TranscribeMode.produces_tts is False


@pytest.mark.asyncio
async def test_multi_mode_app_restores_idle_for_silent_mode():
    """Regression: TranscribeMode + MonologueMode produce no
    TTS, so the SPEAKING→IDLE path never fires. MultiModeApp must
    detect this and restore IDLE itself, otherwise the FSM stays in
    THINKING after the first utterance and the agent appears dead."""
    from openvoicestream_agent.state import ConvState
    from apps.multi_mode.app import MultiModeApp

    cfg = Config(pipeline_mode="always_on")
    app = MultiModeApp.__new__(MultiModeApp)
    # Manually wire just what we need (skip BaseApp __init__ to avoid
    # spinning up audio/slv/llm clients).
    app.config = cfg
    app.plugins = []
    app._state = ConvState.THINKING  # simulate BaseApp pre-set
    app._sleep_task = None

    ctx_holder = {}
    def _factory():
        return ctx_holder["ctx"]
    app.modes = ModeManager(_factory)
    app.modes.register(TranscribeMode())
    app.modes._current = app.modes.get("transcribe")

    # ModeContext fake — broadcast just records.
    broadcasts: list[tuple] = []
    async def _br(name, *args):
        broadcasts.append((name, args))
    ctx_holder["ctx"] = ModeContext(
        config=cfg, slv=None, llm=None, session=Session(),
        audio=None, events=type("E", (), {"emit": lambda *a, **k: None})(),
        broadcast=_br,
    )
    app.broadcast = _br
    # _make_mode_ctx normally reads self.slv etc.; skip that wiring.
    app._make_mode_ctx = lambda: ctx_holder["ctx"]

    # Drive the utterance — TranscribeMode produces no TTS.
    await app.on_user_utterance("hello world")
    # FSM must NOT be stuck in THINKING.
    assert app._state == ConvState.IDLE, (
        f"transcribe mode left FSM in {app._state.value}; expected idle"
    )
    # And the transcribe broadcast still fired.
    assert any(n == "on_transcribed" for n, _ in broadcasts)


@pytest.mark.asyncio
async def test_multi_mode_app_restores_idle_when_preprocess_drops():
    """Monologue.preprocess returns None; dispatcher must restore IDLE
    instead of leaving the FSM in THINKING."""
    from openvoicestream_agent.state import ConvState
    from apps.multi_mode.app import MultiModeApp

    cfg = Config(pipeline_mode="always_on")
    app = MultiModeApp.__new__(MultiModeApp)
    app.config = cfg
    app.plugins = []
    app._state = ConvState.THINKING
    app._sleep_task = None

    ctx_holder = {}
    def _factory():
        return ctx_holder["ctx"]
    app.modes = ModeManager(_factory)
    app.modes.register(MonologueMode())
    app.modes._current = app.modes.get("monologue")

    async def _br(name, *args):
        pass
    ctx_holder["ctx"] = ModeContext(
        config=cfg, slv=None, llm=None, session=Session(),
        audio=None, events=type("E", (), {"emit": lambda *a, **k: None})(),
        broadcast=_br,
    )
    app.broadcast = _br

    await app.on_user_utterance("ignored")
    assert app._state == ConvState.IDLE


def test_monologue_preprocess_drops_user_input():
    """MonologueMode drops user input via preprocess so MultiModeApp
    takes the 'dropped' branch (which now restores IDLE)."""
    mm = MonologueMode()
    assert mm.preprocess_user_text("hello") is None


# ── system-prompt resolution order -----------------------------------


@pytest.mark.asyncio
async def test_mode_overrides_take_precedence_over_mode_class():
    cfg = Config(
        system_prompt="GLOBAL",
        mode_overrides={"chat": {"system_prompt": "OVERRIDE"}},
    )
    llm = FakeLLM(["x"])
    ctx, _ = _make_ctx(cfg=cfg, llm=llm)
    mgr = ModeManager(lambda: ctx)
    mgr.register(ChatMode())
    await mgr.start("chat")
    await mgr.current.on_user_utterance(ctx, "hi")
    assert llm.calls[0][0] == {"role": "system", "content": "OVERRIDE"}


@pytest.mark.asyncio
async def test_explicit_system_prompt_override_wins():
    cfg = Config(
        system_prompt="GLOBAL",
        mode_overrides={"chat": {"system_prompt": "MO"}},
    )
    llm = FakeLLM(["x"])
    ctx, _ = _make_ctx(cfg=cfg, llm=llm)
    mgr = ModeManager(lambda: ctx)
    mgr.register(ChatMode())
    await mgr.start("chat")
    await ctx.run_default_dialogue_turn("hi", system_prompt_override="EXPLICIT")
    assert llm.calls[0][0] == {"role": "system", "content": "EXPLICIT"}


@pytest.mark.asyncio
async def test_interpreter_uses_mode_override_for_system_prompt():
    """Regression: InterpreterMode used to pass `self.system_prompt` as an
    explicit override which masked dashboard edits. Now it should walk
    the resolver and pick up mode_overrides["interpreter"]["system_prompt"]."""
    cfg = Config(
        system_prompt="GLOBAL",
        mode_overrides={"interpreter": {"system_prompt": "TEST OVERRIDE"}},
    )
    llm = FakeLLM(["x"])
    ctx, _ = _make_ctx(cfg=cfg, llm=llm)
    mgr = ModeManager(lambda: ctx)
    mgr.register(InterpreterMode())
    await mgr.start("interpreter")
    await mgr.current.on_user_utterance(ctx, "hi")
    assert llm.calls[0][0] == {"role": "system", "content": "TEST OVERRIDE"}


@pytest.mark.asyncio
async def test_empty_string_system_prompt_override_is_honoured():
    """Regression: explicit empty-string override must be
    preserved — truthiness checks would silently fall through to the
    class default or global system_prompt."""
    cfg = Config(
        system_prompt="GLOBAL",
        mode_overrides={"chat": {"system_prompt": ""}},
    )
    llm = FakeLLM(["x"])
    ctx, _ = _make_ctx(cfg=cfg, llm=llm)
    mgr = ModeManager(lambda: ctx)
    mgr.register(ChatMode())
    await mgr.start("chat")
    await mgr.current.on_user_utterance(ctx, "hi")
    assert llm.calls[0][0] == {"role": "system", "content": ""}


@pytest.mark.asyncio
async def test_mode_temperature_override_passes_to_llm():
    cfg = Config(
        system_prompt="GLOBAL",
        mode_overrides={"chat": {"temperature": 0.2}},
    )
    llm = FakeLLM(["x"])
    ctx, _ = _make_ctx(cfg=cfg, llm=llm)
    mgr = ModeManager(lambda: ctx)
    mgr.register(ChatMode())
    await mgr.start("chat")
    await mgr.current.on_user_utterance(ctx, "hi")
    assert llm.kwargs[0]["temperature"] == 0.2
