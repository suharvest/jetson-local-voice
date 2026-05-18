"""Config dataclass + YAML loader with ${VAR} env substitution."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("openvoicestream-agent requires PyYAML (uv add pyyaml)") from exc


_ENV_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def _default_slv_config() -> dict[str, Any]:
    return {
        "asr_language": "zh",
        "tts_language": "zh",
        "tts_voice": "default",
        "tts_speed": 1.0,
        "sample_rate": 16000,
        "vad": "silero",
        "vad_silence_ms": 400,
        "multi_utterance": True,
    }


@dataclass
class Config:
    """Top-level agent config."""

    slv_url: str = "ws://localhost:8621/v2v/stream"
    slv_config: dict[str, Any] = field(default_factory=_default_slv_config)
    llm_backend: str = "edge_llm"
    llm_base_url: str = "http://localhost:8000/v1"
    llm_api_key: str = "EMPTY"
    llm_model: str = "qwen2.5-3b-instruct"
    system_prompt: str = "You are a helpful, concise voice assistant."
    audio_input_device: str | int | None = None
    audio_output_device: str | int | None = None
    audio_input_sample_rate: int = 16000
    audio_output_sample_rate: int = 24000
    log_level: str = "INFO"
    metadata: dict[str, Any] = field(default_factory=dict)
    # Client-side VAD (replaces server VAD when slv_config.vad == "none").
    # backend: "silero" | "energy" | "auto" | "off"
    client_vad_backend: str = "auto"
    client_vad_threshold: float | None = None
    client_vad_speech_min_ms: int = 200
    client_vad_silence_ms: int = 600
    # SLV closes the WS when asr_eos arrives (even in multi_utterance), so
    # firing it from the client requires reconnect-per-turn. Off by default:
    # we keep the persistent WS and let Paraformer's CIF endpoint detect
    # utterance boundaries instead. Enable only if Paraformer endpoints
    # arrive too late or not at all for your audio.
    client_vad_drive_eos: bool = False
    # Stop-intent recognition: when the ASR final exactly matches one of
    # these (after normalisation), abort current TTS, drop the turn, and
    # transition state→IDLE without consulting the LLM. Chinese strings
    # match the whole utterance; English strings match case-insensitive
    # whole-utterance or word-boundary prefix.
    stop_words: list[str] = field(default_factory=lambda: [
        "停", "停下", "停下来", "别说了", "闭嘴", "安静",
        "stop", "shut up", "be quiet", "silence",
    ])
    # AppMode framework: which mode the app boots into, plus per-mode
    # overrides keyed by mode name, e.g.
    #   mode_overrides: {chat: {system_prompt: "..."}, interpreter: {...}}
    default_mode: str = "chat"
    mode_overrides: dict[str, Any] = field(default_factory=dict)
    # pipeline_mode: controls HOW user audio enters the agent. Orthogonal
    # to AppMode (which controls what the agent DOES with the text).
    #   always_on     — current behaviour. Mic always streams; client VAD
    #                   drives turn boundaries.
    #   wake_word     — agent boots SLEEPING. A WakeSource plugin (HTTP,
    #                   MQTT, serial, local keyword spotter) fires
    #                   app.wake() → state→IDLE for one turn, then
    #                   auto-sleep after sleep_timeout_s of IDLE.
    #   push_to_talk  — agent boots SLEEPING. POST /api/control/ptt/start
    #                   wakes + jumps to LISTENING; POST /api/control/ptt/end
    #                   sends asr_eos + sleeps.
    pipeline_mode: str = "always_on"
    sleep_timeout_s: float = 30.0
    wake_sources: list[str] = field(default_factory=lambda: ["http"])
    # In push_to_talk mode, optionally disable the client-VAD silence
    # detector — relying entirely on the explicit ptt/end signal for EOS.
    # Default True since PTT users typically don't want VAD second-guessing.
    push_to_talk_no_vad_silence: bool = True
    # LLM 防卡死超时（秒）
    # llm_first_token_timeout_s: 发请求 → 首 token 的最长等待
    # llm_stream_idle_timeout_s: 流式过程中两 token 间最长间隔
    llm_first_token_timeout_s: float = 15.0
    llm_stream_idle_timeout_s: float = 30.0
    # Transparent retry for transient upstream LLM failures (network
    # resets, 5xx, connect timeouts) that happen *before any token has
    # been yielded*. Once the model has started speaking we never retry
    # — that would duplicate audio. Set to 0 to disable.
    llm_retry_on_transient: int = 1
    llm_retry_backoff_s: float = 0.5
    # LLM availability probe + circuit breaker (combined state machine —
    # see plugins/llm_availability.py). The probe hits a real
    # /v1/chat/completions with max_tokens=1, not /v1/models, so a server
    # that returns metadata but fails on inference still flips to DOWN.
    llm_availability_enabled: bool = True
    llm_availability_probe_interval_s: float = 30.0
    llm_availability_probe_timeout_s: float = 5.0
    llm_availability_failures_to_down: int = 3
    # MED-3: consecutive "unknown" probe results (timeout / connect error)
    # before we transition to UNKNOWN state. UNKNOWN surfaces a grey dot
    # on the dashboard (vs HEALTHY's green) so operators notice a network
    # partition or hung server instead of mistakenly trusting a stale
    # "everything is fine" indicator. Set to a large number to disable.
    llm_availability_unknowns_to_unknown_state: int = 3
    # Session history trim (A2). When set, the oldest turns are dropped
    # before the prompt is shipped to the LLM so total input tokens stay
    # below this ceiling. Default 3000 leaves a small margin under the
    # engines-3072 build's max_seq_len. Set to None to disable (matches
    # the original append-only invariant).
    session_max_input_tokens: int | None = 3000
    # Tokenizer used to estimate prompt size. Default matches the most
    # common edge-llm engine; override per-deployment if your engine
    # ships a different vocabulary.
    session_tokenizer_model: str = "Qwen/Qwen3-4B-AWQ"
    # Path the config was loaded from (set by `load_config`); used by
    # the dashboard's per-mode override editor to persist changes back
    # to disk. None when the Config was constructed in code.
    _source_path: Path | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        allowed = {"always_on", "wake_word", "push_to_talk"}
        if self.pipeline_mode not in allowed:
            raise ValueError(
                f"pipeline_mode must be one of {sorted(allowed)}; got {self.pipeline_mode!r}"
            )
        if not (isinstance(self.llm_first_token_timeout_s, (int, float))
                and self.llm_first_token_timeout_s > 0):
            raise ValueError(
                f"llm_first_token_timeout_s must be a positive number; "
                f"got {self.llm_first_token_timeout_s!r}"
            )
        if not (isinstance(self.llm_stream_idle_timeout_s, (int, float))
                and self.llm_stream_idle_timeout_s > 0):
            raise ValueError(
                f"llm_stream_idle_timeout_s must be a positive number; "
                f"got {self.llm_stream_idle_timeout_s!r}"
            )
        if not (isinstance(self.llm_retry_on_transient, int)
                and self.llm_retry_on_transient >= 0):
            raise ValueError(
                f"llm_retry_on_transient must be a non-negative int; "
                f"got {self.llm_retry_on_transient!r}"
            )
        if not (isinstance(self.llm_retry_backoff_s, (int, float))
                and self.llm_retry_backoff_s >= 0):
            raise ValueError(
                f"llm_retry_backoff_s must be a non-negative number; "
                f"got {self.llm_retry_backoff_s!r}"
            )


def _expand_env(value: Any) -> Any:
    """Recursively expand ${VAR} / ${VAR:-default} in strings."""
    if isinstance(value, str):
        def repl(m: re.Match[str]) -> str:
            var, default = m.group(1), m.group(2)
            return os.environ.get(var, default if default is not None else "")

        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_config(path: str | Path) -> Config:
    """Load YAML config, apply env substitution, return a Config."""
    p = Path(path).expanduser()
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw = _expand_env(raw)
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping; got {type(raw).__name__}")

    # SLV config sub-block: merge with defaults so users don't have to
    # restate every key.
    slv_cfg = _default_slv_config()
    slv_cfg.update(raw.get("slv_config", {}) or {})
    # Force the framework invariant: persistent WS across utterances.
    slv_cfg["multi_utterance"] = True

    fields = {k: v for k, v in raw.items() if k != "slv_config"}
    cfg = Config(slv_config=slv_cfg, **fields)
    cfg._source_path = p
    return cfg
