"""FastAPI TTS+ASR service for RK3576 with pluggable backends.

Select backend via TTS_BACKEND / ASR_BACKEND env vars.

API-compatible with jetson-voice:
  POST /tts         — JSON {"text": "...", "sid": 0, "speed": 1.0} -> WAV
  POST /tts/stream  — streaming TTS (PCM chunks)
  POST /asr         — multipart upload -> {"text": ..., "language": ...}
  WS   /asr/stream  — streaming ASR (int16 PCM frames -> JSON)
  GET  /health      — health check
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import struct

import numpy as np
from fastapi import FastAPI, File, Query, UploadFile, WebSocket
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RK3576 Speech Service", version="3.0.0")

_backend = None
_asr_backend = None


class TTSRequest(BaseModel):
    text: str
    sid: int | None = None
    speed: float | None = None
    pitch: float | None = None


@app.on_event("startup")
async def startup():
    global _backend, _asr_backend

    # --- TTS (required) ---
    from tts_backend import create_backend

    backend_name = os.environ.get("TTS_BACKEND", "qwen3_rknn")
    logger.info("Loading TTS backend: %s", backend_name)

    try:
        _backend = create_backend(backend_name)
        _backend.preload()
        logger.info("TTS backend '%s' ready.", _backend.name)
    except Exception as e:
        logger.error("Failed to load TTS backend '%s': %s", backend_name, e)
        raise

    # --- ASR (optional) ---
    asr_backend_name = os.environ.get("ASR_BACKEND", "")
    if asr_backend_name:
        logger.info("Loading ASR backend: %s", asr_backend_name)
        try:
            from asr_backend import create_asr_backend
            _asr_backend = create_asr_backend(asr_backend_name)
            _asr_backend.preload()
            logger.info("ASR backend '%s' ready.", _asr_backend.name)
        except Exception as e:
            logger.error("Failed to load ASR backend '%s': %s — ASR disabled", asr_backend_name, e)
            _asr_backend = None
    else:
        logger.info("ASR_BACKEND not set — ASR disabled.")


@app.get("/health")
async def health():
    from asr_backend import ASRCapability
    asr_ready = _asr_backend.is_ready() if _asr_backend else False
    return {
        "tts": _backend.is_ready() if _backend else False,
        "tts_backend": _backend.name if _backend and _backend.is_ready() else None,
        "asr": asr_ready,
        "asr_backend": _asr_backend.name if asr_ready else None,
        "streaming_asr": asr_ready and _asr_backend.has_capability(ASRCapability.STREAMING),
    }


@app.post("/tts")
async def tts(req: TTSRequest):
    if not _backend or not _backend.is_ready():
        return JSONResponse({"error": "TTS not ready"}, status_code=503)

    loop = asyncio.get_event_loop()
    wav_bytes, meta = await loop.run_in_executor(
        None,
        lambda: _backend.synthesize(
            text=req.text,
            speaker_id=req.sid or 0,
            speed=req.speed,
            pitch_shift=req.pitch,
        ),
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
    """Allow clients to probe for streaming support."""
    return Response(status_code=200)


@app.post("/tts/stream")
async def tts_stream(req: TTSRequest):
    """Stream TTS as raw PCM: first 4 bytes = sample_rate (uint32 LE), then int16 PCM chunks."""
    if not _backend or not _backend.is_ready():
        return JSONResponse({"error": "TTS not ready"}, status_code=503)

    audio_queue: queue.Queue[bytes | None] = queue.Queue()
    sr = _backend.get_sample_rate()

    def _generate():
        wav_bytes, meta = _backend.synthesize(
            text=req.text,
            speaker_id=req.sid or 0,
            speed=req.speed,
            pitch_shift=req.pitch,
        )
        import io
        import soundfile as sf
        import numpy as np

        buf = io.BytesIO(wav_bytes)
        data, _ = sf.read(buf, dtype="int16")
        chunk_size = 4800  # 400ms at 12kHz
        for i in range(0, len(data), chunk_size):
            audio_queue.put(data[i: i + chunk_size].tobytes())
        audio_queue.put(None)

    async def stream():
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _generate)
        yield struct.pack("<I", sr)
        while True:
            chunk = await loop.run_in_executor(None, audio_queue.get)
            if chunk is None:
                break
            yield chunk

    return StreamingResponse(stream(), media_type="application/octet-stream")


# ---------------------------------------------------------------------------
# ASR routes
# ---------------------------------------------------------------------------

@app.post("/asr")
async def asr(
    file: UploadFile = File(...),
    language: str = Query("auto"),
):
    """Transcribe an audio file (WAV, FLAC, MP3, …).

    Returns JSON: {"text": "...", "language": "...", "backend": "..."}
    """
    if not _asr_backend or not _asr_backend.is_ready():
        return JSONResponse({"error": "ASR not ready"}, status_code=503)

    audio_bytes = await file.read()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: _asr_backend.transcribe(audio_bytes, language=language),
    )
    return {
        "text": result.text,
        "language": result.language,
        "backend": _asr_backend.name,
        **result.meta,
    }


@app.websocket("/asr/stream")
async def asr_stream(
    ws: WebSocket,
    language: str = "auto",
    sample_rate: int = 16000,
):
    """Streaming ASR over WebSocket.

    Client sends raw int16 PCM frames (at `sample_rate`).
    Server sends JSON objects: {"text": "...", "is_final": bool}.
    Send an empty frame (0 bytes) or close the connection to finalize.
    """
    await ws.accept()

    if not _asr_backend or not _asr_backend.is_ready():
        await ws.send_json({"error": "ASR not ready"})
        await ws.close()
        return

    loop = asyncio.get_event_loop()
    stream = _asr_backend.create_stream(language=language)

    try:
        while True:
            data = await ws.receive_bytes()
            if not data:
                # Empty frame signals end of audio
                break

            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            await loop.run_in_executor(
                None, stream.accept_waveform, sample_rate, samples
            )

            partial, is_endpoint = stream.get_partial()
            if partial:
                await ws.send_json({"text": partial, "is_final": False})

    except Exception as exc:
        logger.debug("ASR stream error: %s", exc)

    finally:
        try:
            final_text = await loop.run_in_executor(None, stream.finalize)
            await ws.send_json({"text": final_text, "is_final": True})
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass
