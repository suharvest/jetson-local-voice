# `WS /v2v/stream` — unified ASR + TTS + VAD + barge-in

Single bi-directional WebSocket. The first JSON frame the client sends
is a `config` that decides which features light up:

| Config key | Effect |
|---|---|
| `asr_language` set | ASR is enabled. Client may send PCM binary frames; server emits `asr_partial` / `asr_endpoint` / `asr_final` JSON. |
| `tts_language` set | TTS is enabled. Client may send `text` JSON frames; server emits PCM binary chunks + `tts_started` / `tts_sentence_done` / `tts_done` JSON. |
| Both set | Full V2V duplex. Binary in both directions: client → ASR input, server → TTS output. |
| `vad` | Server-side VAD backend (default `silero` if ASR enabled, `none` otherwise). |
| `vad_silence_ms` | How long silence to trigger auto `asr_endpoint`. Default 500 ms. |
| `multi_utterance` | If `true`, the session stays open across utterances; each VAD/backend endpoint emits a mid-session `asr_final` with `session_complete: false` and the loop keeps listening. Default `false` (single-utterance, current behaviour). |

Existing `/asr/stream` and `/tts/stream` endpoints stay unchanged for
backward compatibility. The new endpoint adds capability without
breaking anything.

## Protocol

### Client → Server JSON message types

```
{"type":"config",
 "asr_language":"zh",          // omit to disable ASR
 "tts_language":"zh",          // omit to disable TTS
 "tts_voice":"default",        // optional; backend-specific
 "tts_speed":1.0,              // optional; some backends only
 "sample_rate":16000,          // PCM sample rate
 "vad":"silero",               // "silero" | "webrtcvad" | "none"
 "vad_silence_ms":500,
 "multi_utterance":false}      // see "End-of-utterance semantics" below

{"type":"text", "text":"<incremental text chunk>"}
{"type":"asr_eos"}             // manually finalize ASR (overrides VAD)
{"type":"tts_flush"}           // flush remaining TTS buffer
{"type":"abort"}               // barge-in: cancel current TTS, drop queue
```

Plus: **binary frames** = int16 PCM at `sample_rate` (mono), feeding ASR.

### Server → Client JSON message types

```
{"type":"asr_partial",       "text":"...", "is_stable":false}
{"type":"asr_endpoint"}                              // VAD detected end of speech
{"type":"asr_final",         "text":"..."}           // single-utterance mode: one per session
                                                     // multi_utterance mode adds:
                                                     //   "session_complete": false   // mid-session boundary
                                                     //   "session_complete": true,   // session-end final
                                                     //   "duplicate_of_streamed": bool
{"type":"tts_started",       "sentence":"..."}       // about to ship audio
{"type":"tts_sentence_done", "sentence":"..."}       // one sentence finished
{"type":"tts_done"}                                  // tts_flush honored, no more audio
{"type":"error",             "error":"..."}
```

Plus: **binary frames** = int16 PCM TTS output. The first binary frame is
a 4-byte little-endian uint32 = sample rate (matches `/tts/stream`).

## Sentence boundaries (TTS input)

The server buffers text until it sees a sentence-ending punctuation
(`。！？；\n` always; ASCII `.!?` only when followed by whitespace or
end-of-buffer, so "3.14" stays intact). Minimum sentence length is
2 characters (so "Hi.", "OK." pass through). If no punctuation arrives
within 200 characters, a force-flush kicks in.

`tts_flush` always flushes the remainder, regardless of punctuation.

**Abbreviation handling**: when the `tts_language` config maps to one
of pysbd's 22 supported languages (zh/en/ja/ko/es/fr/de/it/pt/ru/ar/
hi/...), the splitter uses pysbd's rule-based segmenter — abbreviations
("Dr. Smith", "U.S.A.", "Ph.D."), inline numbers ("$3.14"), and URLs
("example.com") all stay intact. Unsupported languages fall back to a
simple punctuation regex that over-splits abbreviations into separate
sentences (still works, just slightly choppier TTS cadence).

## Barge-in

Two triggers, same effect (cancel the in-flight TTS synth task + drop
queued sentences):

1. **Client-initiated**: send `{"type":"abort"}` when the local
   audio playback indicates the user is interrupting.
2. **Server-initiated**: if VAD detects `speech_start` while a TTS
   synth is in flight, the synth is cancelled automatically.

After barge-in, ASR continues to receive audio normally and emit
partials. The next text frame from the client starts a new TTS
sequence (the sentence buffer is per-connection, persistent across
barge-ins).

## End-of-utterance semantics

**Single-utterance (default, `multi_utterance: false`)**

- VAD-driven: server emits `asr_endpoint`, then `asr_final`. The audio
  buffer is closed; further binary frames are ignored until the
  client opens a new WebSocket.
- Client-driven: send `{"type":"asr_eos"}` to override VAD and force
  finalize immediately.

If the client sends nothing (binary or eos) and VAD is disabled, the
ASR stream stays open indefinitely (or until WS disconnect).

**Multi-utterance (`multi_utterance: true`)**

Same WS session carries an unbounded sequence of utterances. On each
VAD or backend-detected end-of-speech the server emits:

```
{"type":"asr_endpoint"}
{"type":"asr_final", "text":"<this utterance>", "session_complete": false}
```

The audio buffer stays open; the next utterance is recognized in the
same stream (backends reset per-utterance state automatically on the
next chunk of speech). The session terminates on:

- **Client `asr_eos`** — server runs `finalize()`, emits a final
  `asr_final` with `session_complete: true`. If the final text matches
  the last mid-session final the client already received, the frame
  also carries `duplicate_of_streamed: true` so the client can dedupe.
- **WS disconnect** — same behaviour, but no closing frame is delivered.

Clients should treat mid-session `asr_final` frames as the canonical
transcript of each utterance; the closing frame only delivers any
trailing audio that arrived after the last VAD endpoint and is usually
a duplicate.

## Customer example (Python, asyncio)

Demo: take a Chinese sentence from a streaming LLM, speak it
synchronously to the user while the user can talk back and barge in.

```python
import asyncio, json, struct, websockets

async def v2v_session(transcribe_partial_cb, llm_token_stream):
    uri = "ws://device:8000/v2v/stream"
    async with websockets.connect(uri) as ws:
        # 1. config
        await ws.send(json.dumps({
            "type": "config",
            "asr_language": "Chinese",
            "tts_language": "zh",
            "vad": "silero",
            "vad_silence_ms": 500,
        }))

        sample_rate = None        # parsed from first binary frame

        async def upstream():
            # Stream mic PCM up to the server
            async for pcm_chunk in mic.read_chunks(16000):
                await ws.send(pcm_chunk)         # binary

        async def llm_to_tts():
            # When ASR final arrives we'll start pulling LLM tokens
            # and forwarding them as text frames to TTS.
            pass   # set up below in main loop

        async def downstream():
            nonlocal sample_rate
            llm_task = None
            async for msg in ws:
                if isinstance(msg, bytes):
                    if sample_rate is None:
                        sample_rate = struct.unpack("<I", msg[:4])[0]
                        speaker.open(sample_rate)
                        continue
                    speaker.write(msg)
                    continue
                evt = json.loads(msg)
                t = evt["type"]
                if t == "asr_partial":
                    transcribe_partial_cb(evt["text"])
                elif t == "asr_final":
                    # Got the user's full utterance. Kick off LLM
                    # → TTS forwarding.
                    user_text = evt["text"]
                    llm_task = asyncio.create_task(
                        forward_tokens_to_tts(ws, user_text, llm_token_stream))
                elif t == "tts_done":
                    break

        async def forward_tokens_to_tts(ws, prompt, stream_fn):
            async for token in stream_fn(prompt):
                await ws.send(json.dumps({"type":"text", "text": token}))
            await ws.send(json.dumps({"type":"tts_flush"}))

        await asyncio.gather(upstream(), downstream())
```

Customer integrates their LLM of choice in `stream_fn(prompt)`.

## Minimal modes

**TTS-only** (feed an LLM stream into TTS, no microphone):

```python
await ws.send(json.dumps({"type":"config", "tts_language":"zh"}))
for token in llm_stream():
    await ws.send(json.dumps({"type":"text", "text": token}))
await ws.send(json.dumps({"type":"tts_flush"}))
async for msg in ws:
    if isinstance(msg, bytes): play(msg)
    elif json.loads(msg)["type"] == "tts_done": break
```

**ASR-only with VAD** (drop-in replacement for `/asr/stream?vad=silero`):

```python
await ws.send(json.dumps({"type":"config",
                          "asr_language":"Chinese",
                          "vad":"silero"}))
async for pcm in mic.read_chunks(16000):
    await ws.send(pcm)         # never need to send {} or b""
async for msg in ws:
    evt = json.loads(msg)
    if evt["type"] == "asr_final":
        return evt["text"]
```

## Operational notes

- **VAD model load**: silero ONNX is loaded once per process, shared
  across every WS connection. First connection pays ~500 ms init.
- **Concurrent connections**: each connection gets its own
  `SileroVADSession` state, its own `SentenceBuffer`, and its own
  ASR/TTS streams. Coordinator policy (concurrent / serialized) still
  applies per device profile.
- **No persistent dialogue state**: each WS is one utterance round
  (audio → text → text → audio). Customer holds LLM context.
- **HTTPS / auth not included** — see operational hardening section in
  `docs/perf-test-runbook.md`.

## Connection lifetime

- **ASR-only mode**: server emits `asr_final` and closes the WS after
  the VAD or client `asr_eos` triggers finalization.
- **TTS-only mode**: server processes `text` frames as long as the
  client sends them; ships audio until `tts_flush` empties the queue,
  then emits `tts_done` and closes.
- **V2V mode**: server waits for *both* ASR work (final emitted) and
  TTS work (flush received + queue drained) to complete before
  closing. The client should always send `tts_flush` after the LLM
  stream ends, otherwise the TTS task lives forever (waiting for
  more text) even though no more audio will come.
- WS disconnect from the client side immediately cancels all in-flight
  tasks (synth threads are signaled to break out of their generators).
