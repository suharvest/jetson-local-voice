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


@app.on_event("startup")
async def startup():
    import model_downloader
    mode = os.environ.get("LANGUAGE_MODE", "zh_en")
    model_dir = os.environ.get("MODEL_DIR", "/opt/models")
    model_downloader.ensure_models(mode, model_dir)

    import tts_service
    logger.info("Pre-loading TTS model...")
    tts_service.preload()

    try:
        import streaming_asr_service
        streaming_asr_service.preload()
    except Exception as e:
        logger.info(f"Streaming ASR not available: {e}")

    logger.info("Speech service ready.")


# ── Health & Capabilities ────────────────────────────────────────

@app.get("/health")
async def health():
    import asr_service, tts_service

    result = {
        "asr": asr_service.is_ready(),
        "tts": tts_service.is_ready(),
        "tts_backend": tts_service.backend_name() if tts_service.is_ready() else None,
        "tts_capabilities": [c.value for c in tts_service.capabilities()] if tts_service.is_ready() else [],
    }
    try:
        import streaming_asr_service
        result["streaming_asr"] = streaming_asr_service.is_ready()
    except ImportError:
        result["streaming_asr"] = False
    return result


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

        def _gen():
            return list(backend.generate_streaming(
                req.text,
                speaker_id=req.sid,
                speed=req.speed,
                pitch_shift=req.pitch,
            ))

        chunks = await loop.run_in_executor(None, _gen)
        for chunk in chunks:
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


# ── ASR ──────────────────────────────────────────────────────────

@app.post("/asr")
async def asr(
    file: UploadFile = File(...),
    language: str = Query("auto"),
):
    import asr_service
    audio_bytes = await file.read()
    text = asr_service.transcribe_audio(audio_bytes, language=language)
    return {"text": text}


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
    """
    import asyncio
    import numpy as np

    await ws.accept()

    try:
        import streaming_asr_service
    except ImportError:
        await ws.send_json({"error": "streaming ASR not available"})
        await ws.close()
        return

    stream = streaming_asr_service.create_stream()
    prev_text = ""

    try:
        while True:
            data = await ws.receive_bytes()

            if len(data) == 0:
                final_text = await asyncio.to_thread(
                    streaming_asr_service.finalize, stream
                )
                await ws.send_json({
                    "text": final_text,
                    "is_final": True,
                    "is_stable": True,
                })
                break

            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            text, is_endpoint = await asyncio.to_thread(
                streaming_asr_service.feed_and_decode,
                stream, samples, sample_rate
            )

            if is_endpoint:
                await ws.send_json({
                    "text": text,
                    "is_final": True,
                    "is_stable": True,
                })
                stream = streaming_asr_service.create_stream()
                prev_text = ""
            elif text and text != prev_text:
                is_stable = text.startswith(prev_text) if prev_text else False
                await ws.send_json({
                    "text": text,
                    "is_final": False,
                    "is_stable": is_stable,
                })
                prev_text = text

    except WebSocketDisconnect:
        logger.debug("ASR stream client disconnected")
    except Exception as e:
        logger.error(f"ASR stream error: {e}")
    finally:
        try:
            await ws.close()
        except Exception:
            pass
