"""FastAPI TTS service for RK3576 with Qwen3-TTS.

API-compatible with jetson-voice:
  POST /tts         — JSON {"text": "...", "sid": 0, "speed": 1.0} -> WAV
  POST /tts/stream  — streaming TTS (PCM chunks)
  GET  /health      — health check
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import struct

from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RK3576 TTS Service", version="1.0.0")


class TTSRequest(BaseModel):
    text: str
    sid: int | None = None
    speed: float | None = None
    pitch: float | None = None


@app.on_event("startup")
async def startup():
    import tts_service

    logger.info("Loading TTS models...")
    tts_service.preload()
    logger.info("TTS service ready.")


@app.get("/health")
async def health():
    import tts_service

    return {
        "tts": tts_service.is_ready(),
        "asr": False,
        "streaming_asr": False,
    }


@app.post("/tts")
async def tts(req: TTSRequest):
    import tts_service

    loop = asyncio.get_event_loop()
    wav_bytes, meta = await loop.run_in_executor(
        None,
        lambda: tts_service.synthesize(
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
    import tts_service
    import numpy as np

    audio_queue: queue.Queue[bytes | None] = queue.Queue()
    sr = tts_service.get_sample_rate()

    def _generate():
        wav_bytes, meta = tts_service.synthesize(
            text=req.text,
            speaker_id=req.sid or 0,
            speed=req.speed,
        )
        # Convert WAV to raw PCM int16
        import io
        import soundfile as sf

        buf = io.BytesIO(wav_bytes)
        data, _ = sf.read(buf, dtype="int16")
        # Send in chunks
        chunk_size = 4800  # 400ms at 12kHz
        for i in range(0, len(data), chunk_size):
            audio_queue.put(data[i : i + chunk_size].tobytes())
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
