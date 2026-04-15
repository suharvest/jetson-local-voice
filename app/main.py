"""FastAPI speech service: ASR + TTS with pluggable backends."""

from __future__ import annotations

import logging
import os

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
    first_chunk_frames: int | None = None
    chunk_frames: int | None = None


_asr_backend = None

def _get_asr_backend():
    return _asr_backend

@app.on_event("startup")
async def startup():
    global _asr_backend

    # Log language mode configuration
    language_mode = os.environ.get("LANGUAGE_MODE", "zh_en")
    logger.info("=" * 60)
    logger.info("LANGUAGE_MODE: %s", language_mode)
    if language_mode == "multilanguage":
        logger.info("  → Using Qwen3 TTS + ASR (52 languages, voice cloning)")
    else:
        logger.info("  → Using Sherpa TTS + ASR (zh/en mode)")
    logger.info("=" * 60)

    import model_downloader
    model_dir = os.environ.get("MODEL_DIR", "/opt/models")
    model_downloader.ensure_models(language_mode, model_dir)

    # ASR backend (load before TTS to avoid ORT session conflicts)
    # Note: create_asr_backend() will auto-select based on LANGUAGE_MODE
    try:
        from asr_backend import create_asr_backend
        _asr_backend = create_asr_backend()  # Let it auto-detect from LANGUAGE_MODE
        logger.info("Pre-loading ASR (%s)...", _asr_backend.name)
        _asr_backend.preload()
        logger.info("ASR backend: %s (capabilities: %s)",
                     _asr_backend.name, [c.value for c in _asr_backend.capabilities])
    except Exception as e:
        logger.warning("ASR backend failed: %s", e)

    import tts_service
    logger.info("Pre-loading TTS model...")
    tts_service.preload()

    logger.info("Speech service ready.")


# ── Health & Capabilities ────────────────────────────────────────

@app.get("/health")
async def health():
    import tts_service

    result = {
        "tts": tts_service.is_ready(),
        "tts_backend": tts_service.backend_name() if tts_service.is_ready() else None,
        "tts_capabilities": [c.value for c in tts_service.capabilities()] if tts_service.is_ready() else [],
    }

    # ASR
    try:
        from asr_backend import create_asr_backend
        asr_be = _get_asr_backend()
        result["asr"] = asr_be.is_ready() if asr_be else False
        result["asr_backend"] = asr_be.name if asr_be and asr_be.is_ready() else None
        result["asr_capabilities"] = [c.value for c in asr_be.capabilities] if asr_be and asr_be.is_ready() else []
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
    return {
        "backend": asr_be.name,
        "capabilities": [c.value for c in asr_be.capabilities],
        "sample_rate": asr_be.sample_rate,
    }


@app.get("/tts/capabilities")
async def tts_capabilities():
    """Return TTS backend info and supported capabilities."""
    import tts_service
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
    import tts_service

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
            "X-Audio-Duration": str(meta["duration"]),
            "X-Inference-Time": str(meta["inference_time"]),
            "X-RTF": str(meta["rtf"]),
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
    import tts_service
    from tts_backend import TTSCapability

    if not tts_service.has_capability(TTSCapability.STREAMING):
        return JSONResponse(
            {"error": "Streaming not supported by current backend",
             "required_capability": "streaming"},
            status_code=501,
        )

    sr = tts_service.get_sample_rate()
    backend = tts_service.get_backend()

    async def stream():
        yield struct.pack("<I", sr)
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        def _run():
            try:
                for chunk in backend.generate_streaming(
                    req.text,
                    speaker_id=req.sid,
                    speed=req.speed,
                    pitch_shift=req.pitch,
                    language=req.language,
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(None, _run)

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
    import tts_service
    from tts_backend import TTSCapability

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

    wav_bytes, meta = tts_service.clone_voice(
        text=req.text,
        speaker_embedding=speaker_embedding,
        language=req.language,
    )
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Audio-Duration": str(meta["duration"]),
            "X-Inference-Time": str(meta["inference_time"]),
            "X-RTF": str(meta["rtf"]),
        },
    )


@app.post("/tts/clone/embedding")
async def tts_extract_embedding(file: UploadFile = File(...)):
    """Extract speaker embedding from reference audio WAV.

    Returns base64-encoded speaker embedding that can be reused
    across multiple /tts/clone calls.
    """
    import base64
    import tts_service
    from tts_backend import TTSCapability

    if not tts_service.has_capability(TTSCapability.VOICE_CLONE):
        return JSONResponse(
            {"error": "Voice cloning not supported by current backend",
             "required_capability": "voice_clone",
             "backend": tts_service.backend_name()},
            status_code=501,
        )

    audio_bytes = await file.read()
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
    import tts_service
    from tts_backend import TTSCapability

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

    async def stream():
        yield struct.pack("<I", sr)
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        def _run():
            try:
                for chunk in backend.generate_streaming(
                    req.text,
                    speaker_embedding=speaker_embedding,
                    language=req.language,
                    first_chunk_frames=req.first_chunk_frames or 10,
                    chunk_frames=req.chunk_frames or 25,
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(None, _run)

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

    asr_be = _get_asr_backend()
    if asr_be and asr_be.is_ready():
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
):
    """Streaming ASR via WebSocket.

    Client sends: raw int16 PCM bytes
    Client sends: empty bytes b"" to signal end
    Server sends: JSON {"text": "...", "is_final": bool, "is_stable": bool}

    Requires an ASR backend with STREAMING capability.
    """
    import asyncio
    import numpy as np
    from asr_backend import ASRCapability

    await ws.accept()

    # Choose backend: prefer ASR backend with STREAMING, fall back to sherpa
    asr_be = _get_asr_backend()
    use_backend_stream = (
        asr_be is not None
        and asr_be.is_ready()
        and asr_be.has_capability(ASRCapability.STREAMING)
    )

    if use_backend_stream:
        await _asr_stream_backend(ws, asr_be, language, sample_rate)
    else:
        await ws.send_json({"error": "no streaming ASR available"})
        await ws.close()


async def _asr_stream_backend(
    ws: WebSocket,
    asr_be,
    language: str,
    sample_rate: int,
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
                        "text": "",
                        "is_final": True,
                        "is_stable": True,
                        "reset": True,
                    })
                    logger.debug("ASR stream reset by client command (backend=%s)", asr_be.name)
                continue

            # ── Binary message: audio data ──
            data = msg.get("bytes", b"")
            if data is None:
                # WebSocket disconnect frame — no bytes key
                break

            if len(data) == 0:
                # End of audio — pre-encode tail, then decode
                await asyncio.to_thread(stream.prepare_finalize)
                final_text = await asyncio.to_thread(stream.finalize)
                await ws.send_json({
                    "text": final_text,
                    "is_final": True,
                    "is_stable": True,
                })
                break

            # Buffer audio
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            stream.accept_waveform(sample_rate, samples)

            # Check for partial results (V2 feature, no-op by default)
            partial_text, is_endpoint = stream.get_partial()
            if partial_text:
                await ws.send_json({
                    "text": partial_text,
                    "is_final": is_endpoint,
                    "is_stable": False,
                })

    except WebSocketDisconnect:
        logger.debug("ASR stream client disconnected (backend=%s)", asr_be.name)
    except Exception as e:
        logger.error("ASR stream error (backend=%s): %s", asr_be.name, e)
    finally:
        try:
            await ws.close()
        except Exception:
            pass


