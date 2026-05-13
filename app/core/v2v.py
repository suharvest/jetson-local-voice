"""V2V WebSocket protocol + sentence buffering.

Used by:
- WS /v2v/stream — unified ASR + TTS + VAD + barge-in endpoint
- (optionally exposed as TTS-only / ASR-only by which config keys the
  client supplies)

Protocol spec: docs/api/v2v-stream.md
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator, Optional


# ────────────────────────────────────────────────────────────────────────
# Client → Server JSON message types
# ────────────────────────────────────────────────────────────────────────
CLIENT_CONFIG     = "config"        # initial setup, must be first message
CLIENT_TEXT       = "text"          # streaming text input for TTS
CLIENT_ASR_EOS    = "asr_eos"       # manually finalize ASR (overrides VAD)
CLIENT_TTS_FLUSH  = "tts_flush"     # flush remaining TTS buffer
CLIENT_ABORT      = "abort"         # barge-in: cancel current TTS

# ────────────────────────────────────────────────────────────────────────
# Server → Client JSON message types
# ────────────────────────────────────────────────────────────────────────
SERVER_ASR_PARTIAL        = "asr_partial"
SERVER_ASR_ENDPOINT       = "asr_endpoint"       # VAD detected end of speech
SERVER_ASR_FINAL          = "asr_final"
SERVER_TTS_STARTED        = "tts_started"        # first audio frame about to ship
SERVER_TTS_SENTENCE_DONE  = "tts_sentence_done"  # one sentence finished
SERVER_TTS_DONE           = "tts_done"           # flush complete, no more audio
SERVER_ERROR              = "error"


# ────────────────────────────────────────────────────────────────────────
# Sentence buffering for streaming TTS input
# ────────────────────────────────────────────────────────────────────────

# Sentence-ending punctuation: CJK forms always end a sentence; ASCII
# punct only ends a sentence if followed by whitespace or end-of-buffer.
# Note: this heuristic correctly handles "3.14" (no space after .) and
# "U.S.A." (when used mid-sentence), but still over-splits abbreviations
# like "Dr. Smith arrived." into ["Dr.", "Smith arrived."]. For streaming
# TTS the over-split is acceptable (brief pause, no quality loss). If
# your LLM emits abbreviation-heavy English, expect a slightly choppier
# cadence — proper abbreviation handling would require nltk.sent_tokenize
# or a maintained abbreviation list, both heavyweight for streaming.
_SENTENCE_END_RE = re.compile(r"[。！？；\n]+|[!?.](?=\s|$)")

DEFAULT_MIN_SENTENCE_CHARS = 2
DEFAULT_MAX_BUFFER_CHARS   = 200


@dataclass
class SentenceBuffer:
    """Accumulates streaming text and emits complete sentences.

    Used to bridge a token-streaming source (LLM) to a sentence-batched
    sink (TTS engine). Designed to maximize streaming latency: emits as
    soon as a sentence boundary is seen, falls back to a hard flush at
    MAX_BUFFER_CHARS so a punctuation-less LLM doesn't stall forever.

        buf = SentenceBuffer()
        for token in llm_tokens:
            for sentence in buf.add(token):
                tts.synthesize(sentence)
        for sentence in buf.flush():     # at end-of-stream
            tts.synthesize(sentence)
    """

    min_chars:    int = DEFAULT_MIN_SENTENCE_CHARS
    max_buffer:   int = DEFAULT_MAX_BUFFER_CHARS
    _buf:         str = field(default="", init=False, repr=False)

    def add(self, chunk: str) -> Iterator[str]:
        """Append `chunk`, yield any sentences now complete."""
        if not chunk:
            return
        self._buf += chunk
        while True:
            sentence = self._extract_next_sentence()
            if sentence is None:
                return
            yield sentence

    def flush(self) -> Iterator[str]:
        """Yield any remaining text as a final sentence (regardless of
        punctuation or min-length)."""
        leftover = self._buf.strip()
        self._buf = ""
        if leftover:
            yield leftover

    def is_empty(self) -> bool:
        return not self._buf.strip()

    # ----- internals -----

    def _extract_next_sentence(self) -> Optional[str]:
        """Find the smallest prefix of `_buf` that ends at a sentence
        boundary AND has at least `min_chars` characters. Pops it from
        the buffer and returns it, or returns None if no such prefix is
        ready (in which case the buffer is unchanged, unless it grew
        past `max_buffer` and we forced a flush)."""
        pos = 0
        while True:
            m = _SENTENCE_END_RE.search(self._buf, pos)
            if m is None:
                # No more sentence boundary in scope. Force-flush if
                # the buffer has grown past the safety threshold.
                if len(self._buf) >= self.max_buffer:
                    out = self._buf.strip()
                    self._buf = ""
                    return out or None
                return None
            end = m.end()
            prefix = self._buf[:end]
            if len(prefix.strip()) >= self.min_chars:
                self._buf = self._buf[end:]
                return prefix.strip()
            # Too short — try the next boundary out.
            pos = end
