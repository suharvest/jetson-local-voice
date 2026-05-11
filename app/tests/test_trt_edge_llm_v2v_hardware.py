import os

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_TRT_EDGELLM_HARDWARE_TESTS") != "1",
    reason="requires Jetson TRT-Edge-LLM engines and workers",
)


def test_qwen3_tts_asr_round_trip_nihao():
    from app.backends.jetson.trt_edge_llm_asr import TRTEdgeLLMASRBackend
    from app.backends.jetson.trt_edge_llm_tts import TRTEdgeLLMTTSBackend

    text = "你好。"
    tts = TRTEdgeLLMTTSBackend()
    tts.preload()
    wav, meta = tts.synthesize(text, max_audio_length=80)

    assert wav
    assert meta["duration_s"] > 0

    asr = TRTEdgeLLMASRBackend()
    asr.preload()
    result = asr.transcribe(wav)

    assert "你好" in result.text
