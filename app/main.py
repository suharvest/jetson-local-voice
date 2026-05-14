"""FastAPI speech service: ASR + TTS with pluggable backends."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Jetson Speech Service", version="2.0.0")


class TTSRequest(BaseModel):
    text: str
    sid: int | None = None
    speed: float | None = None
    pitch: float | None = None
    language: str | None = None


class CloneRequest(BaseModel):
    text: str
    speaker_embedding_b64: str  # base64-encoded speaker embedding
    language: str | None = None


class CloneStreamRequest(BaseModel):
    text: str
    speaker_embedding_b64: str  # base64-encoded speaker embedding
    language: str | None = None
    streaming_profile: str | None = None
    first_chunk_frames: int | None = None
    chunk_frames: int | None = None


_asr_backend = None

# Dedicated single-thread executor for streaming TTS (T3 fix).
# Default asyncio executor spawns multiple worker threads; each new thread
# observes a cold CUDA per-thread context for the C++ TRT engine, which
# inflates streaming prefill from ~16ms (warm) to 33-122ms (cold) under
# any concurrency. Pinning streaming TTS to a single worker keeps the
# CUDA context warm across all requests.
_tts_stream_executor: ThreadPoolExecutor | None = None

# Dedicated single-thread executor for streaming ASR.  Without this,
# concurrent WS connections dispatch ASR work to separate IO threads,
# each racing on _ASR_CUDA_STREAM (process-global singleton) leading to
# CUDA Graph capture failures.  One worker serialises all ASR ops on a
# consistent thread with a warm CUDA context.
_asr_executor: ThreadPoolExecutor | None = None


def _get_asr_backend():
    return _asr_backend


def _get_tts_stream_executor() -> ThreadPoolExecutor:
    global _tts_stream_executor
    if _tts_stream_executor is None:
        _tts_stream_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tts-stream"
        )
    return _tts_stream_executor


def _get_asr_executor() -> ThreadPoolExecutor:
    global _asr_executor
    if _asr_executor is None:
        _asr_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="asr-stream"
        )
    return _asr_executor

@app.on_event("startup")
async def startup():
    global _asr_backend

    try:
        from app.core.profile_loader import apply_profile_from_env, current_profile
        apply_profile_from_env()
    except Exception as exc:
        logger.error("Failed to apply Seeed Local Voice profile: %s", exc)
        raise

    # Initialise the execution coordinator from the loaded profile's
    # execution_policy block. Default to concurrent (no lock) when the
    # profile does not declare one — matches the previous behaviour.
    from app.core.coordinator import init_coordinator, get_coordinator
    init_coordinator(current_profile().get("execution_policy", {"mode": "concurrent"}))

    # Log language mode configuration
    language_mode = os.environ.get("LANGUAGE_MODE", "zh_en")
    logger.info("=" * 60)
    logger.info("LANGUAGE_MODE: %s", language_mode)
    if language_mode == "multilanguage":
        logger.info("  → Using Qwen3 TTS + ASR (52 languages, voice cloning)")
    else:
        logger.info("  → Using Sherpa TTS + ASR (zh/en mode)")
    logger.info("=" * 60)

    from app.core import model_downloader
    model_dir = os.environ.get("MODEL_DIR", "/opt/models")
    model_downloader.ensure_models(language_mode, model_dir)

    # Resolve any TRT engines declared by the active profile. Must run
    # AFTER model_downloader (ONNX inputs may be needed for fallback
    # compile) and BEFORE any backend module is imported by the factories
    # (backends read engine paths from env vars at module import time).
    try:
        from app.core.engine_resolver import resolve_all
        resolved = resolve_all(current_profile())
        if resolved:
            logger.info("engine_resolver: resolved %d engine(s)", len(resolved))
            for env_var, path in resolved.items():
                logger.info("  %s → %s", env_var, path)
    except Exception as exc:
        logger.error("engine_resolver failed: %s", exc)
        raise

    # ASR backend (load before TTS to avoid ORT session conflicts)
    # Note: create_asr_backend() will auto-select based on LANGUAGE_MODE
    try:
        from app.core.asr_backend import create_asr_backend
        _asr_backend = create_asr_backend()  # Let it auto-detect from LANGUAGE_MODE
        logger.info("Pre-loading ASR (%s)...", _asr_backend.name)
        _asr_backend.preload()
        logger.info("ASR backend: %s (capabilities: %s)",
                     _asr_backend.name, [c.value for c in _asr_backend.capabilities])

        # Warm up ASR executor thread so its CUDA per-thread context is
        # initialised before the first streaming request.  Without this the
        # very first accept_waveform pays a cold-context tax on encoder.
        # SKIP_ASR_WARMUP=1 skips this on memory-constrained devices (Nano 8GB):
        # saves ~300-400 MB at startup, costs ~100ms one-time cold-context tax
        # on the very first ASR request.
        if os.environ.get("SKIP_ASR_WARMUP", "").lower() in ("1", "true", "yes"):
            logger.info("ASR streaming warmup skipped (SKIP_ASR_WARMUP set).")
        else:
            _asyncio = __import__("asyncio")
            _executor = _get_asr_executor()

            def _warm_asr():
                # Some backends (e.g. SherpaASRBackend) don't expose a
                # transcribe_audio convenience method; their warmup is
                # implicit in preload(). Skip silently to avoid log noise.
                if not hasattr(_asr_backend, "transcribe_audio"):
                    logger.info(
                        "ASR warmup skipped: %s has no transcribe_audio (preload already warmed).",
                        type(_asr_backend).__name__,
                    )
                    return
                try:
                    import numpy as _np
                    silence = _np.zeros(16000, dtype=_np.float32)
                    _asr_backend.transcribe_audio(silence)
                    logger.info("ASR streaming executor warmed up (1 thread, CUDA primed).")
                except Exception as exc:
                    logger.warning("ASR warm-up failed: %s", exc)

            await _asyncio.get_event_loop().run_in_executor(_executor, _warm_asr)
    except Exception as e:
        logger.warning("ASR backend failed: %s", e)

    from app.core import tts_service
    if not tts_service.is_configured():
        logger.info("ASR-only mode: profile declares no tts_backend; TTS endpoints will return 503.")
    elif os.environ.get("LAZY_TTS", "").lower() in ("1", "true", "yes"):
        logger.info("TTS preload skipped (LAZY_TTS set); will load on first request.")
    else:
        logger.info("Pre-loading TTS model...")
        tts_service.preload()

    # Warm up the dedicated streaming-TTS executor thread so its CUDA
    # per-thread context is initialized before the first /tts/stream
    # request lands. Without this, the very first streaming request
    # pays a ~30ms cold-context tax on prefill.
    # Skip when LAZY_TTS or ASR-only — TTS not loaded yet, can't warm what isn't there.
    if not tts_service.is_configured():
        pass  # ASR-only mode, no TTS warmup
    elif os.environ.get("LAZY_TTS", "").lower() in ("1", "true", "yes"):
        logger.info("TTS streaming warmup skipped (LAZY_TTS).")
    else:
      try:
        from app.core.tts_backend import TTSCapability
        if tts_service.has_capability(TTSCapability.STREAMING):
            backend = tts_service.get_backend()
            executor = _get_tts_stream_executor()

            def _warm_stream():
                try:
                    # Run one tiny streaming synthesis on the executor
                    # thread to materialize CUDA context state.
                    stream_kwargs = {}
                    profile = os.environ.get("EDGE_LLM_TTS_WARMUP_STREAMING_PROFILE")
                    if profile:
                        stream_kwargs["streaming_profile"] = profile
                    warmup_text = os.environ.get("EDGE_LLM_TTS_WARMUP_TEXT", "你好")
                    for _ in backend.generate_streaming(warmup_text, **stream_kwargs):
                        pass
                except Exception as exc:  # pragma: no cover
                    logger.warning("TTS streaming warm-up failed: %s", exc)

            import asyncio as _asyncio
            await _asyncio.get_event_loop().run_in_executor(executor, _warm_stream)
            logger.info("TTS streaming executor warmed up (1 thread, CUDA primed).")
      except Exception as exc:  # pragma: no cover
        logger.warning("TTS streaming executor warm-up skipped: %s", exc)

    # Register backend getters with the coordinator so 'exclusive' policy can
    # call unload() on the dormant slot. Lambdas resolve lazily so they cope
    # with backends loaded after this point (LAZY_TTS).
    try:
        from app.core import tts_service as _tts_service_mod
        coord = get_coordinator()
        coord.register_backend("asr", lambda: _asr_backend)
        coord.register_backend("tts", lambda: _tts_service_mod._backend)
    except Exception as exc:  # pragma: no cover
        logger.warning("Coordinator backend registration skipped: %s", exc)

    logger.info("Speech service ready.")


# ── Health & Capabilities ────────────────────────────────────────

@app.get("/health")
async def health():
    from app.core import tts_service

    result = {
        "tts": tts_service.is_ready(),
        "tts_backend": tts_service.backend_name() if tts_service.is_ready() else None,
        "tts_capabilities": [c.value for c in tts_service.capabilities()] if tts_service.is_ready() else [],
    }

    # ASR
    try:
        from app.core.asr_backend import create_asr_backend
        asr_be = _get_asr_backend()
        result["asr"] = asr_be.is_ready() if asr_be else False
        result["asr_backend"] = asr_be.name if asr_be and asr_be.is_ready() else None
        result["asr_capabilities"] = [c.value for c in asr_be.capabilities] if asr_be and asr_be.is_ready() else []
        if asr_be and asr_be.is_ready() and hasattr(asr_be, "providers"):
            result["asr_providers"] = asr_be.providers
    except Exception:
        result["asr"] = False
        result["asr_backend"] = None
        result["asr_capabilities"] = []

    return result


@app.get("/asr/capabilities")
async def asr_capabilities():
    """Return ASR backend info and supported capabilities."""
    asr_be = _get_asr_backend()
    if not asr_be or not asr_be.is_ready():
        return JSONResponse({"error": "ASR not ready"}, status_code=503)
    caps = {
        "backend": asr_be.name,
        "capabilities": [c.value for c in asr_be.capabilities],
        "sample_rate": asr_be.sample_rate,
    }
    if hasattr(asr_be, "providers"):
        caps["providers"] = asr_be.providers
    return caps


@app.get("/tts/capabilities")
async def tts_capabilities():
    """Return TTS backend info and supported capabilities."""
    from app.core import tts_service
    if not tts_service.is_ready():
        return JSONResponse({"error": "TTS not ready"}, status_code=503)
    return {
        "backend": tts_service.backend_name(),
        "capabilities": [c.value for c in tts_service.capabilities()],
        "sample_rate": tts_service.get_sample_rate(),
    }


# ── TTS ──────────────────────────────────────────────────────────

@app.post("/tts")
async def tts(req: TTSRequest):
    from app.core import tts_service
    from app.core.coordinator import get_coordinator

    async with get_coordinator().acquire("tts"):
        wav_bytes, meta = tts_service.synthesize(
            text=req.text,
            speaker_id=req.sid,
            speed=req.speed,
            pitch_shift=req.pitch,
            language=req.language,
        )
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Audio-Duration": str(meta.get("duration", meta.get("duration_s", 0))),
            "X-Inference-Time": str(meta.get("inference_time", meta.get("inference_time_s", 0))),
            "X-RTF": str(meta.get("rtf", 0)),
        },
    )


@app.options("/tts/stream")
async def tts_stream_options():
    return Response(status_code=200)


@app.post("/tts/stream")
async def tts_stream(req: TTSRequest):
    """Stream TTS as raw PCM: first 4 bytes = sample_rate (uint32 LE), then int16 PCM chunks."""
    import asyncio
    import struct
    from app.core import tts_service
    from app.core.tts_backend import TTSCapability

    if not tts_service.has_capability(TTSCapability.STREAMING):
        return JSONResponse(
            {"error": "Streaming not supported by current backend",
             "required_capability": "streaming"},
            status_code=501,
        )

    sr = tts_service.get_sample_rate()
    backend = tts_service.get_backend()
    from app.core.coordinator import get_coordinator

    # Sentence-level streaming: split the request text into sentences (via
    # pysbd when the language is supported, regex fallback otherwise) and
    # call the TTS backend per sentence. The first audio chunk of the
    # first sentence reaches the client as soon as the first sentence's
    # model warmup + KV prep is done — for big TTS models (Qwen3 voice
    # clone, ~6700ms total for a 16s clip on Nano) this halves the
    # perceived first-audio latency on long inputs. Single-sentence
    # inputs (the common case) see zero change in behavior.
    from app.core.v2v import SentenceBuffer
    sbuf = SentenceBuffer(language=req.language)
    sentences = list(sbuf.add(req.text or "")) + list(sbuf.flush())
    if not sentences:
        # Empty text — preserve original behavior: send SR header then
        # close with no audio chunks.
        async def empty():
            async with get_coordinator().acquire("tts"):
                yield struct.pack("<I", sr)
        return StreamingResponse(empty(), media_type="application/octet-stream")

    async def stream():
        async with get_coordinator().acquire("tts"):
            yield struct.pack("<I", sr)
            loop = asyncio.get_event_loop()
            for sentence in sentences:
                queue: asyncio.Queue[bytes | None] = asyncio.Queue()

                def _run(text=sentence):
                    try:
                        for chunk in backend.generate_streaming(
                            text,
                            speaker_id=req.sid,
                            speed=req.speed,
                            pitch_shift=req.pitch,
                            language=req.language,
                        ):
                            loop.call_soon_threadsafe(queue.put_nowait, chunk)
                    finally:
                        loop.call_soon_threadsafe(queue.put_nowait, None)

                loop.run_in_executor(_get_tts_stream_executor(), _run)

                while True:
                    chunk = await queue.get()
                    if chunk is None:
                        break
                    yield chunk

    return StreamingResponse(stream(), media_type="application/octet-stream")


# ── Voice Clone ───��──────────────────────────────────────────────

@app.post("/tts/clone")
async def tts_clone(req: CloneRequest):
    """Synthesize with voice cloning. Requires voice_clone capability."""
    import base64
    from app.core import tts_service
    from app.core.tts_backend import TTSCapability

    if not tts_service.has_capability(TTSCapability.VOICE_CLONE):
        return JSONResponse(
            {"error": "Voice cloning not supported by current backend",
             "required_capability": "voice_clone",
             "backend": tts_service.backend_name()},
            status_code=501,
        )

    try:
        speaker_embedding = base64.b64decode(req.speaker_embedding_b64)
    except Exception:
        return JSONResponse({"error": "Invalid base64 speaker_embedding_b64"}, status_code=400)

    from app.core.coordinator import get_coordinator
    async with get_coordinator().acquire("tts"):
        wav_bytes, meta = tts_service.clone_voice(
            text=req.text,
            speaker_embedding=speaker_embedding,
            language=req.language,
        )
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Audio-Duration": str(meta.get("duration", meta.get("duration_s", 0))),
            "X-Inference-Time": str(meta.get("inference_time", meta.get("inference_time_s", 0))),
            "X-RTF": str(meta.get("rtf", 0)),
        },
    )


@app.post("/tts/clone/embedding")
async def tts_extract_embedding(file: UploadFile = File(...)):
    """Extract speaker embedding from reference audio WAV.

    Returns base64-encoded speaker embedding that can be reused
    across multiple /tts/clone calls.
    """
    import base64
    from app.core import tts_service
    from app.core.tts_backend import TTSCapability

    if not tts_service.has_capability(TTSCapability.VOICE_CLONE):
        return JSONResponse(
            {"error": "Voice cloning not supported by current backend",
             "required_capability": "voice_clone",
             "backend": tts_service.backend_name()},
            status_code=501,
        )

    audio_bytes = await file.read()
    from app.core.coordinator import get_coordinator
    async with get_coordinator().acquire("tts"):
        embedding = tts_service.extract_speaker_embedding(audio_bytes)
    return {
        "speaker_embedding_b64": base64.b64encode(embedding).decode(),
        "embedding_size": len(embedding),
    }


@app.post("/tts/clone/stream")
async def tts_clone_stream(req: CloneStreamRequest):
    """Stream TTS with voice cloning.

    Returns raw PCM: first 4 bytes = sample_rate (uint32 LE), then int16 PCM chunks.
    Requires voice_clone capability.
    """
    import asyncio
    import struct
    import base64
    from app.core import tts_service
    from app.core.tts_backend import TTSCapability

    if not tts_service.has_capability(TTSCapability.VOICE_CLONE):
        return JSONResponse(
            {"error": "Voice cloning not supported by current backend",
             "required_capability": "voice_clone",
             "backend": tts_service.backend_name()},
            status_code=501,
        )

    if not tts_service.has_capability(TTSCapability.STREAMING):
        return JSONResponse(
            {"error": "Streaming not supported by current backend",
             "required_capability": "streaming"},
            status_code=501,
        )

    try:
        speaker_embedding = base64.b64decode(req.speaker_embedding_b64)
    except Exception:
        return JSONResponse({"error": "Invalid base64 speaker_embedding_b64"}, status_code=400)

    sr = tts_service.get_sample_rate()
    backend = tts_service.get_backend()
    from app.core.coordinator import get_coordinator

    async def stream():
        async with get_coordinator().acquire("tts"):
            yield struct.pack("<I", sr)
            loop = asyncio.get_event_loop()
            queue: asyncio.Queue[bytes | None] = asyncio.Queue()
            stream_kwargs = {
                "speaker_embedding": speaker_embedding,
                "language": req.language,
            }
            if req.first_chunk_frames is not None:
                stream_kwargs["first_chunk_frames"] = req.first_chunk_frames
            if req.chunk_frames is not None:
                stream_kwargs["chunk_frames"] = req.chunk_frames
            if req.streaming_profile is not None:
                stream_kwargs["streaming_profile"] = req.streaming_profile

            def _run():
                try:
                    for chunk in backend.generate_streaming(req.text, **stream_kwargs):
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            loop.run_in_executor(_get_tts_stream_executor(), _run)

            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk

    return StreamingResponse(stream(), media_type="application/octet-stream")


# ── ASR ──────────────────────────────────────────────────────────

@app.post("/asr")
async def asr(
    file: UploadFile = File(...),
    language: str = Query("auto"),
):
    audio_bytes = await file.read()

    from app.core.coordinator import get_coordinator
    asr_be = _get_asr_backend()
    if asr_be and asr_be.is_ready():
        async with get_coordinator().acquire("asr"):
            result = asr_be.transcribe(audio_bytes, language=language)
        return {
            "text": result.text,
            "language": result.language,
            "backend": asr_be.name,
            **result.meta,
        }
    else:
        return JSONResponse(
            status_code=503,
            content={"error": "ASR backend not available"},
        )


@app.websocket("/asr/stream")
async def asr_stream(
    ws: WebSocket,
    language: str = "auto",
    sample_rate: int = 16000,
    vad: Optional[str] = None,           # opt-in: "silero" | "webrtcvad" | "none"
    vad_silence_ms: int = 500,
):
    """Streaming ASR via WebSocket.

    Client sends: raw int16 PCM bytes
    Client sends: empty bytes b"" to signal end
    Server sends: JSON {"text": "...", "is_final": bool, "is_stable": bool}

    Optional server-side VAD: pass ?vad=silero (or webrtcvad) to have the
    server auto-finalize on silence — client no longer needs to send the
    empty b"" frame. ``vad_silence_ms`` controls the silence threshold
    (default 500 ms). Without the query param this endpoint behaves
    exactly as before (forced EOS by client).

    Requires an ASR backend with STREAMING capability.
    """
    import asyncio
    import numpy as np
    from app.core.asr_backend import ASRCapability

    await ws.accept()

    # Choose backend: prefer ASR backend with STREAMING, fall back to sherpa
    asr_be = _get_asr_backend()
    use_backend_stream = (
        asr_be is not None
        and asr_be.is_ready()
        and asr_be.has_capability(ASRCapability.STREAMING)
    )

    # Lazy-init server-side VAD only if requested. Reuses the shared
    # singleton model from app.core.vad — no extra model load per
    # connection.
    vad_session = None
    if vad and vad not in ("none", "off", "disabled"):
        try:
            from app.core import vad as vad_mod
            vad_session = vad_mod.create_vad(
                vad, sample_rate=sample_rate, silence_ms=vad_silence_ms
            )
        except Exception as e:
            logger.warning("VAD '%s' init failed (%s); falling back to forced-EOS", vad, e)
            vad_session = None

    if use_backend_stream:
        from app.core.coordinator import get_coordinator
        async with get_coordinator().acquire("asr"):
            await _asr_stream_backend(ws, asr_be, language, sample_rate, vad_session)
    else:
        await ws.send_json({"error": "no streaming ASR available"})
        await ws.close()


async def _asr_stream_backend(
    ws: WebSocket,
    asr_be,
    language: str,
    sample_rate: int,
    vad_session=None,
):
    """Streaming ASR using ASR backend (accumulate-then-transcribe).

    Supports a ``reset`` control command: the client may send a JSON text
    message ``{"command": "reset"}`` at any time.  This discards the
    current stream and creates a fresh one without closing the WebSocket.
    """
    import asyncio
    import json as _json
    import numpy as np

    stream = asr_be.create_stream(language=language)
    logger.info("ASR stream opened (backend=%s)", asr_be.name)

    try:
        while True:
            msg = await ws.receive()

            # ── Text message: control command ──
            if "text" in msg and msg["text"]:
                try:
                    cmd = _json.loads(msg["text"])
                except (ValueError, TypeError):
                    continue
                if cmd.get("command") == "reset":
                    stream = asr_be.create_stream(language=language)
                    await ws.send_json({
                        "type": "reset",
                        "text": "",
                        "is_final": True,
                        "is_stable": True,
                        "reset": True,
                    })
                    logger.debug("ASR stream reset by client command (backend=%s)", asr_be.name)
                elif cmd.get("command") == "end_utterance":
                    _loop = asyncio.get_event_loop()
                    final_text = await _loop.run_in_executor(_get_asr_executor(), stream.force_endpoint)
                    await ws.send_json({
                        "type": "final",
                        "text": final_text,
                        "is_final": True,
                        "is_stable": True,
                    })
                    logger.debug("ASR utterance endpoint forced (backend=%s)", asr_be.name)
                continue

            # ── Binary message: audio data ──
            data = msg.get("bytes", b"")
            if data is None:
                # WebSocket disconnect frame — no bytes key
                break

            if len(data) == 0:
                # End of audio — pre-encode tail, then decode
                _loop = asyncio.get_event_loop()
                await _loop.run_in_executor(_get_asr_executor(), stream.prepare_finalize)
                final_text = await _loop.run_in_executor(_get_asr_executor(), stream.finalize)
                await ws.send_json({
                    "type": "final",
                    "text": final_text,
                    "is_final": True,
                    "is_stable": True,
                })
                break

            # Buffer audio (run in thread to avoid blocking event loop)
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            _loop = asyncio.get_event_loop()
            await _loop.run_in_executor(_get_asr_executor(), stream.accept_waveform, sample_rate, samples)

            # Server-side VAD endpoint detection (opt-in via ?vad=)
            if vad_session is not None:
                from app.core.vad import VADSession
                event = vad_session.process(samples)
                if event == VADSession.SPEECH_END:
                    await _loop.run_in_executor(_get_asr_executor(), stream.prepare_finalize)
                    final_text = await _loop.run_in_executor(_get_asr_executor(), stream.finalize)
                    await ws.send_json({
                        "type": "final",
                        "text": final_text,
                        "is_final": True,
                        "is_stable": True,
                        "endpoint": "vad",
                    })
                    break

            # Check for partial results
            partial_text, is_endpoint = stream.get_partial()
            if partial_text:
                if is_endpoint:
                    await ws.send_json({
                        "type": "final",
                        "text": partial_text,
                        "is_final": True,
                        "is_stable": True,
                    })
                else:
                    await ws.send_json({
                        "type": "partial",
                        "text": partial_text,
                        "is_final": False,
                        "is_stable": False,
                    })

    except WebSocketDisconnect:
        logger.debug("ASR stream client disconnected (backend=%s)", asr_be.name)
    except Exception as e:
        logger.error("ASR stream error (backend=%s): %s", asr_be.name, e, exc_info=True)
        # Surface the error to the client as a structured frame instead of a
        # silent close. Without this the client's next ws.recv() returns "",
        # json.loads("") raises JSONDecodeError, and the real cause is hidden.
        try:
            await ws.send_json({
                "type": "error",
                "error": f"{type(e).__name__}: {e}",
                "backend": asr_be.name,
                "is_final": True,
                "is_stable": True,
            })
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────
# Unified V2V WebSocket: ASR + TTS + VAD + barge-in
# ──────────────────────────────────────────────────────────────────────

@app.websocket("/v2v/stream")
async def v2v_stream(ws: WebSocket):
    """Unified bi-directional WebSocket: speech in, partials + audio out.

    Client may enable any subset of features via the first ``config``
    JSON frame. See ``docs/api/v2v-stream.md`` for the protocol spec.
    Minimum viable patterns:

      TTS-only (LLM token stream → audio):
        send {"type":"config", "tts_language":"zh"}
        send {"type":"text", "text":"..."} repeatedly
        send {"type":"tts_flush"}; await binary chunks + tts_done

      ASR-only (mic → text, with auto VAD endpoint):
        send {"type":"config", "asr_language":"zh", "vad":"silero"}
        send PCM binary chunks
        await asr_partial / asr_endpoint / asr_final

      V2V (full duplex):
        config with both asr_language + tts_language
        interleave binary (mic) with text (LLM tokens)
        receive partials, endpoints, audio
        send {"type":"abort"} to barge-in
    """
    import asyncio
    import json as _json
    import struct
    import numpy as np

    from app.core import tts_service, v2v as v2v_proto
    from app.core.asr_backend import ASRCapability
    from app.core.tts_backend import TTSCapability
    from app.core import vad as vad_mod
    from app.core.coordinator import get_coordinator

    coord = get_coordinator()

    await ws.accept()

    # ── Stage 1: receive initial config ─────────────────────────────
    try:
        first_msg = await ws.receive()
    except WebSocketDisconnect:
        return
    cfg_text = first_msg.get("text", "")
    if not cfg_text:
        await ws.close(code=1003); return
    try:
        cfg = _json.loads(cfg_text)
    except (ValueError, TypeError):
        await ws.close(code=1003); return
    if cfg.get("type") != v2v_proto.CLIENT_CONFIG:
        await ws.send_json({"type": v2v_proto.SERVER_ERROR,
                            "error": "first message must be a config frame"})
        await ws.close(code=1003); return

    asr_language    = cfg.get("asr_language")  # e.g. "zh" / "Chinese" / "en" / None
    tts_language    = cfg.get("tts_language")
    tts_voice       = cfg.get("tts_voice")
    tts_speed       = cfg.get("tts_speed")
    sample_rate     = int(cfg.get("sample_rate", 16000))
    vad_backend     = cfg.get("vad", "silero" if asr_language else "none")
    vad_silence_ms  = int(cfg.get("vad_silence_ms", 500))
    multi_utterance = bool(cfg.get("multi_utterance", False))

    if not asr_language and not tts_language:
        await ws.send_json({"type": v2v_proto.SERVER_ERROR,
                            "error": "config must enable asr_language and/or tts_language"})
        await ws.close(code=1003); return

    # ── Stage 2: bring up the backends ──────────────────────────────
    asr_be = None
    asr_stream = None
    vad = None
    if asr_language:
        asr_be = _get_asr_backend()
        if asr_be is None or not asr_be.is_ready() or not asr_be.has_capability(ASRCapability.STREAMING):
            await ws.send_json({"type": v2v_proto.SERVER_ERROR,
                                "error": "asr_language requested but no streaming ASR backend ready"})
            await ws.close(code=1011); return
        try:
            asr_stream = asr_be.create_stream(language=asr_language)
        except Exception as e:
            await ws.send_json({"type": v2v_proto.SERVER_ERROR, "error": f"asr stream init: {e}"})
            await ws.close(code=1011); return
        # VAD init runs in executor: silero ONNX first-load takes ~500ms and
        # would otherwise stall the event loop. ValueError (e.g. unsupported
        # sample rate) is a hard config error → reject and close. Other init
        # failures fall back to no-VAD with a warning.
        try:
            _loop_init = asyncio.get_event_loop()
            vad = await _loop_init.run_in_executor(
                None,
                lambda: vad_mod.create_vad(vad_backend, sample_rate=sample_rate, silence_ms=vad_silence_ms),
            )
        except ValueError as e:
            await ws.send_json({"type": v2v_proto.SERVER_ERROR, "error": f"VAD config: {e}"})
            await ws.close(code=1003); return
        except Exception as e:
            logger.warning("v2v VAD init (%s) failed: %s — running without VAD", vad_backend, e)
            vad = None

    tts_be = None
    tts_buffer = None
    if tts_language:
        if not tts_service.is_ready() or not tts_service.has_capability(TTSCapability.STREAMING):
            await ws.send_json({"type": v2v_proto.SERVER_ERROR,
                                "error": "tts_language requested but no streaming TTS backend ready"})
            await ws.close(code=1011); return
        tts_be = tts_service.get_backend()
        tts_buffer = v2v_proto.SentenceBuffer(language=tts_language)

    logger.info("v2v stream opened (asr=%s tts=%s vad=%s)",
                asr_language or "off", tts_language or "off", vad_backend if asr_language else "off")

    # ── Stage 3: per-connection state + write serialization ─────────
    send_lock = asyncio.Lock()
    state = {
        "asr_eos":          False,   # set when VAD endpoint or client asr_eos
        "vad_endpoint":     False,   # set ONLY when VAD detected speech-end
                                     # (so we emit asr_endpoint frame in the
                                     # VAD case but not on client-driven eos)
        "vad_endpoint_pending": False,  # multi_utterance: VAD speech-end fired;
                                        # asr_out_task should emit endpoint+final
                                        # but keep listening for the next utterance
        "tts_flush":        False,   # set when client tts_flush
        "current_tts_task": None,    # running TTS synth task (cancellable)
        "current_tts_stop": None,    # threading.Event to signal synth thread
                                     # to stop on barge-in (avoids orphan
                                     # synth blocking the TTS executor)
        "tts_started":      False,   # tts_started frame sent for current sentence
        "client_closed":    False,
    }
    tts_q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    async def send_json(payload):
        async with send_lock:
            try:
                await ws.send_json(payload)
            except Exception:
                state["client_closed"] = True

    async def send_bytes(data):
        async with send_lock:
            try:
                await ws.send_bytes(data)
            except Exception:
                state["client_closed"] = True

    async def send_error(msg):
        await send_json({"type": v2v_proto.SERVER_ERROR, "error": msg})

    # ── Stage 4: tasks ──────────────────────────────────────────────

    async def dispatcher():
        """Receive incoming binary (audio) + text (control) frames."""
        try:
            while not state["client_closed"]:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    state["client_closed"] = True
                    break
                # binary → ASR input
                data = msg.get("bytes")
                if data:
                    if not asr_stream:
                        continue  # ignored in TTS-only mode
                    # Bug #2 fix: after asr_eos, ignore further audio so we
                    # don't race finalize. The protocol doc explicitly says
                    # "further binary frames are ignored until the client
                    # opens a new WebSocket".
                    if state["asr_eos"]:
                        continue
                    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    if vad is not None:
                        event = vad.process(samples)
                        if event == vad_mod.VADSession.SPEECH_START:
                            # auto barge-in: cancel any in-flight TTS
                            t = state["current_tts_task"]
                            if t is not None and not t.done():
                                t.cancel()
                            stop = state["current_tts_stop"]
                            if stop is not None:
                                stop.set()
                        elif event == vad_mod.VADSession.SPEECH_END:
                            if multi_utterance:
                                # Mid-session: emit an utterance boundary; do
                                # NOT terminate the session — keep accepting
                                # audio for the next utterance.
                                state["vad_endpoint_pending"] = True
                            else:
                                state["asr_eos"] = True
                                state["vad_endpoint"] = True
                    async with coord.acquire("asr"):
                        await loop.run_in_executor(
                            _get_asr_executor(), asr_stream.accept_waveform, sample_rate, samples
                        )
                    continue
                # text → JSON control
                text = msg.get("text", "")
                if not text:
                    continue
                try:
                    payload = _json.loads(text)
                except (ValueError, TypeError):
                    continue
                typ = payload.get("type")
                if typ == v2v_proto.CLIENT_TEXT and tts_buffer is not None:
                    for sentence in tts_buffer.add(payload.get("text", "")):
                        await tts_q.put(sentence)
                elif typ == v2v_proto.CLIENT_TTS_FLUSH:
                    if tts_buffer is not None:
                        for sentence in tts_buffer.flush():
                            await tts_q.put(sentence)
                    state["tts_flush"] = True
                elif typ == v2v_proto.CLIENT_ASR_EOS:
                    state["asr_eos"] = True
                elif typ == v2v_proto.CLIENT_ABORT:
                    t = state["current_tts_task"]
                    if t is not None and not t.done():
                        t.cancel()
                    stop = state["current_tts_stop"]
                    if stop is not None:
                        stop.set()
                    # Drain queue so flush doesn't replay queued sentences
                    while not tts_q.empty():
                        try: tts_q.get_nowait()
                        except asyncio.QueueEmpty: break
                    if asr_stream is not None:
                        try: asr_stream.cancel_and_finalize()
                        except Exception: pass
        except WebSocketDisconnect:
            state["client_closed"] = True

    async def asr_out_task():
        """Poll ASR stream for partials, emit endpoint + final.

        Single-utterance (default): asr_endpoint + asr_final emitted once,
        then session ends. Client-driven asr_eos skips asr_endpoint.

        Multi-utterance (config multi_utterance=True): on each VAD or
        backend endpoint, emit asr_endpoint + asr_final with
        session_complete=false, then keep listening. Session terminates
        only on client asr_eos / disconnect, which sends a closing
        asr_final with session_complete=true (and duplicate_of_streamed
        if the text matches the last mid-session final).
        """
        backend_endpoint = False        # single-utterance only
        last_streamed_final = None      # multi-utterance: last text emitted as a
                                        # mid-session final, for dedup on close
        while not state["asr_eos"] and not state["client_closed"]:
            try:
                async with coord.acquire("asr"):
                    partial, is_endpoint = await loop.run_in_executor(
                        _get_asr_executor(), asr_stream.get_partial
                    )
            except Exception:
                partial, is_endpoint = "", False
            if partial:
                await send_json({"type": v2v_proto.SERVER_ASR_PARTIAL,
                                 "text": partial, "is_stable": bool(is_endpoint)})

            # Endpoint sources: backend (is_endpoint from get_partial) or VAD
            # (vad_endpoint_pending, multi only). Single-mode VAD also sets
            # asr_eos so we never reach the "pending" path there.
            endpoint_fired = is_endpoint or state["vad_endpoint_pending"]
            if endpoint_fired and multi_utterance:
                await send_json({"type": v2v_proto.SERVER_ASR_ENDPOINT})
                text = partial or ""
                await send_json({"type": v2v_proto.SERVER_ASR_FINAL,
                                 "text": text,
                                 "session_complete": False})
                last_streamed_final = text
                state["vad_endpoint_pending"] = False
                # Backend stream is_endpoint will auto-reset on the next audio
                # chunk via _check_new_utterance_resume; loop continues.
                await asyncio.sleep(0.05)
                continue

            if is_endpoint:
                state["asr_eos"] = True
                backend_endpoint = True
                break
            await asyncio.sleep(0.05)

        if state["client_closed"]:
            return
        # Only emit asr_endpoint for VAD- or backend-detected endpoints,
        # not when the client manually requested asr_eos.
        if not multi_utterance and (state["vad_endpoint"] or backend_endpoint):
            await send_json({"type": v2v_proto.SERVER_ASR_ENDPOINT})
        try:
            async with coord.acquire("asr"):
                final_text = await loop.run_in_executor(_get_asr_executor(), asr_stream.finalize)
        except Exception as e:
            await send_error(f"asr finalize: {e}")
            return

        if multi_utterance:
            duplicate = (final_text or "") == (last_streamed_final or "")
            await send_json({"type": v2v_proto.SERVER_ASR_FINAL,
                             "text": final_text or "",
                             "session_complete": True,
                             "duplicate_of_streamed": duplicate})
        else:
            await send_json({"type": v2v_proto.SERVER_ASR_FINAL,
                             "text": final_text or ""})

    async def tts_out_task():
        """Drain sentence queue → synthesize → emit audio.

        Sends the 4-byte sample-rate header on first successful synth
        (NOT first attempted synth — so a cancelled-mid-flight first
        sentence doesn't leave the client with a header but no audio).
        """
        sr_header_sent = False
        while not state["client_closed"]:
            # Exit when client said flush and the queue is drained.
            if state["tts_flush"] and tts_q.empty():
                break
            try:
                sentence = await asyncio.wait_for(tts_q.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            audio_queue: asyncio.Queue = asyncio.Queue()
            # Likely #1 fix: signal the synth thread to stop mid-iteration
            # on barge-in (single-thread TTS executor would otherwise be
            # blocked by the orphaned generator until it completes).
            import threading as _threading
            stop_event = _threading.Event()
            state["current_tts_stop"] = stop_event

            def _run_synth(s):
                try:
                    stream_kwargs = {"language": tts_language}
                    if tts_voice is not None:    stream_kwargs["voice"] = tts_voice
                    if tts_speed is not None:    stream_kwargs["speed"] = tts_speed
                    for chunk in tts_be.generate_streaming(s, **stream_kwargs):
                        if stop_event.is_set():
                            break
                        loop.call_soon_threadsafe(audio_queue.put_nowait, chunk)
                except Exception as e:
                    loop.call_soon_threadsafe(audio_queue.put_nowait, ("__error__", str(e)))
                finally:
                    loop.call_soon_threadsafe(audio_queue.put_nowait, None)

            async def drain():
                nonlocal sr_header_sent
                # Coord lock per-sentence: cheap on concurrent profiles;
                # serializes sentences against ASR on serialized profiles.
                async with coord.acquire("tts"):
                    if not sr_header_sent:
                        sr = tts_service.get_sample_rate() if hasattr(tts_service, "get_sample_rate") else 16000
                        await send_bytes(struct.pack("<I", sr))
                        sr_header_sent = True
                    await send_json({"type": v2v_proto.SERVER_TTS_STARTED, "sentence": sentence})
                    loop.run_in_executor(_get_tts_stream_executor(), _run_synth, sentence)
                    state["tts_started"] = True
                    while True:
                        item = await audio_queue.get()
                        if item is None:
                            break
                        if isinstance(item, tuple) and item[0] == "__error__":
                            await send_error(f"tts: {item[1]}")
                            break
                        await send_bytes(item)
                    await send_json({"type": v2v_proto.SERVER_TTS_SENTENCE_DONE, "sentence": sentence})

            task = asyncio.create_task(drain())
            state["current_tts_task"] = task
            try:
                await task
            except asyncio.CancelledError:
                # Barge-in: tell the synth thread to break out of the
                # generator loop, then drain any chunks it produced
                # before noticing the flag.
                stop_event.set()
                try:
                    while True:
                        item = audio_queue.get_nowait()
                        if item is None: break
                except asyncio.QueueEmpty:
                    pass
            finally:
                state["current_tts_task"] = None
                state["current_tts_stop"] = None
        if not state["client_closed"]:
            await send_json({"type": v2v_proto.SERVER_TTS_DONE})

    # ── Stage 5: orchestrate ────────────────────────────────────────
    # Bug #3 fix: dispatcher loops on ws.receive() forever (only exits
    # on disconnect). If we asyncio.gather all three, the server hangs
    # after asr_final / tts_done. Spawn work tasks separately, wait for
    # them, then cancel the dispatcher.
    dispatcher_task = asyncio.create_task(dispatcher())
    work_tasks = []
    if asr_stream is not None:
        work_tasks.append(asyncio.create_task(asr_out_task()))
    if tts_be is not None:
        work_tasks.append(asyncio.create_task(tts_out_task()))

    try:
        if work_tasks:
            await asyncio.gather(*work_tasks, return_exceptions=False)
        else:
            # No work tasks (shouldn't happen — config rejected earlier),
            # just keep the dispatcher running until the client closes.
            await dispatcher_task
    except Exception as e:
        logger.error("v2v stream error: %s", e, exc_info=True)
        try:
            await send_error(f"{type(e).__name__}: {e}")
        except Exception:
            pass
    finally:
        if not dispatcher_task.done():
            dispatcher_task.cancel()
            try:
                await dispatcher_task
            except (asyncio.CancelledError, Exception):
                pass
        for t in work_tasks:
            if not t.done():
                t.cancel()
        # Tell the synth thread to bail (if running) so the TTS executor
        # frees up for the next connection.
        stop = state["current_tts_stop"]
        if stop is not None:
            stop.set()
        try:
            await ws.close()
        except Exception:
            pass
        logger.info("v2v stream closed")
