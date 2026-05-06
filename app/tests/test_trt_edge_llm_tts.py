import json
import subprocess
import io
import wave


def _make_wav_bytes(frame_count: int, sample_rate: int = 24000) -> bytes:
    payload = b"\x00\x00" * frame_count
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(payload)
    return out.getvalue()


def test_one_shot_tts_passes_code_predictor_dir(monkeypatch):
    import backends.trt_edge_llm_tts as tts_mod

    captured = {}
    monkeypatch.setenv("EDGE_LLM_TTS_WORKER", "0")
    monkeypatch.setattr(tts_mod, "TTS_BINARY", "/tmp/qwen3_tts_inference")
    monkeypatch.setattr(tts_mod, "TTS_TALKER_DIR", "/models/talker")
    monkeypatch.setattr(tts_mod, "TTS_CODE_PREDICTOR_DIR", "/models/code_predictor")
    monkeypatch.setattr(tts_mod, "TTS_CODE2WAV_DIR", "/models/code2wav")
    monkeypatch.setattr(tts_mod, "TTS_TOKENIZER_DIR", "/models/tokenizer")

    def fake_run_binary(binary, args, timeout):
        captured["binary"] = binary
        captured["args"] = args
        input_path = args[args.index("--inputFile") + 1]
        with open(input_path) as f:
            captured["input"] = json.load(f)
        output_path = args[args.index("--outputFile") + 1]
        audio_dir = args[args.index("--outputAudioDir") + 1]
        audio_path = f"{audio_dir}/audio_req0.wav"
        with open(audio_path, "wb") as f:
            f.write(b"RIFFtest")
        with open(output_path, "w") as f:
            json.dump(
                {
                    "responses": [
                        {
                            "audio_file": audio_path,
                            "audio_duration_ms": 10,
                            "audio_samples": 240,
                        }
                    ]
                },
                f,
            )
        return subprocess.CompletedProcess([binary] + args, 0, "", "")

    monkeypatch.setattr(tts_mod, "run_binary", fake_run_binary)

    backend = tts_mod.TRTEdgeLLMTTSBackend()
    backend._ready = True
    wav, _ = backend.synthesize("你好", max_audio_length=8)

    assert wav == b"RIFFtest"
    assert captured["binary"] == "/tmp/qwen3_tts_inference"
    assert captured["args"][
        captured["args"].index("--codePredictorEngineDir") + 1
    ] == "/models/code_predictor"
    assert captured["input"]["codec_eos_logit_offset"] == 0
    assert captured["input"]["talker_top_k"] == 50
    assert captured["input"]["talker_top_p"] == 1.0
    assert captured["input"]["predictor_temperature"] == 0.9
    assert captured["input"]["predictor_top_k"] == 50
    assert captured["input"]["predictor_top_p"] == 1.0
    assert captured["input"]["min_audio_length"] == 30


def test_split_tts_text_handles_cjk_and_latin(monkeypatch):
    import backends.trt_edge_llm_tts as tts_mod

    zh = "你好，很高兴认识你。今天我们来测试一下语音合成的稳定性，看看这段稍微长一点的中文是不是能清楚自然地读出来。"
    zh_parts = tts_mod._split_tts_text(zh, max_chars=24)

    assert len(zh_parts) > 1
    assert "".join(zh_parts) == zh
    assert max(len(part) for part in zh_parts) <= 32
    assert all(part not in "。！？!?；;，,、：" for part in zh_parts)

    punctuated = "真的吗？可以的，请继续！不过，逗号也要保留。"
    punctuated_parts = tts_mod._split_tts_text(punctuated, max_chars=8)

    assert "".join(punctuated_parts) == punctuated
    assert any(part.endswith("？") for part in punctuated_parts)
    assert any(part.endswith("！") for part in punctuated_parts)
    assert any("，" in part for part in punctuated_parts)
    assert all(part not in "。！？!?；;，,、：" for part in punctuated_parts)

    en = "Hello, this is a longer text for validating that product-side segmentation also works for English input without relying on Chinese punctuation."
    en_parts = tts_mod._split_tts_text(en, max_chars=48)

    assert len(en_parts) > 1
    assert " ".join(en_parts).replace("  ", " ") == en
    assert max(len(part) for part in en_parts) <= 48


def test_split_tts_text_preserves_common_punctuation_and_grammar():
    import backends.trt_edge_llm_tts as tts_mod

    cases = [
        ("中文", "真的吗？可以的，请继续！不过，逗号、顿号、冒号：都要保留。", 8, ""),
        ("中文引号", "他说：“今天很好，可以继续。”然后停了一下。", 10, ""),
        ("英文", "Really? Yes, please continue! However, commas, semicolons; and colons: must stay.", 28, " "),
        ("英文缩写", "Dr. Smith said, \"Let's test TTS, ASR, and V2V.\" It worked.", 32, " "),
        ("混合", "EdgeLLM 可以跑 TTS/ASR，对吗？Yes, it can.", 12, ""),
    ]

    punctuation = set("。！？!?；;，,、：:.\"'“”‘’()（）")
    for _, text, max_chars, joiner in cases:
        parts = tts_mod._split_tts_text(text, max_chars=max_chars)
        reconstructed = joiner.join(parts).replace("  ", " ") if joiner else "".join(parts)

        assert reconstructed == text
        assert len(parts) > 1
        assert all(part.strip() for part in parts)
        assert all(not set(part).issubset(punctuation) for part in parts)

    zh_parts = tts_mod._split_tts_text(cases[0][1], max_chars=8)
    assert any(part.endswith("？") for part in zh_parts)
    assert any(part.endswith("！") for part in zh_parts)
    assert any("，" in part for part in zh_parts)

    en_parts = tts_mod._split_tts_text(cases[2][1], max_chars=28)
    assert any(part.endswith("?") for part in en_parts)
    assert any(part.endswith("!") for part in en_parts)
    assert any("," in part for part in en_parts)

    abbrev_parts = tts_mod._split_tts_text(cases[3][1], max_chars=32)
    assert all(part != "Dr." for part in abbrev_parts)
    assert "Dr. Smith" in " ".join(abbrev_parts)

    decimal = "Version 3.14 works. Version 4.0 also works!"
    decimal_parts = tts_mod._split_tts_text(decimal, max_chars=24)
    assert "3.14" in " ".join(decimal_parts)
    assert "4.0" in " ".join(decimal_parts)


def test_segmented_tts_concatenates_one_shot_wavs(monkeypatch):
    import backends.trt_edge_llm_tts as tts_mod

    calls = []
    monkeypatch.setenv("EDGE_LLM_TTS_WORKER", "0")
    monkeypatch.setattr(tts_mod, "TTS_BINARY", "/tmp/qwen3_tts_inference")
    monkeypatch.setattr(tts_mod, "TTS_TALKER_DIR", "/models/talker")
    monkeypatch.setattr(tts_mod, "TTS_CODE_PREDICTOR_DIR", "/models/code_predictor")
    monkeypatch.setattr(tts_mod, "TTS_CODE2WAV_DIR", "/models/code2wav")
    monkeypatch.setattr(tts_mod, "TTS_TOKENIZER_DIR", "/models/tokenizer")

    def fake_run_binary(binary, args, timeout):
        input_path = args[args.index("--inputFile") + 1]
        with open(input_path) as f:
            input_data = json.load(f)
        calls.append((args, input_data))
        output_path = args[args.index("--outputFile") + 1]
        audio_dir = args[args.index("--outputAudioDir") + 1]
        audio_path = f"{audio_dir}/audio_req0.wav"
        with open(audio_path, "wb") as f:
            f.write(_make_wav_bytes(240))
        with open(output_path, "w") as f:
            json.dump(
                {
                    "responses": [
                        {
                            "audio_file": audio_path,
                            "audio_duration_ms": 10,
                            "audio_samples": 240,
                        }
                    ]
                },
                f,
            )
        return subprocess.CompletedProcess([binary] + args, 0, "", "")

    monkeypatch.setattr(tts_mod, "run_binary", fake_run_binary)

    backend = tts_mod.TRTEdgeLLMTTSBackend()
    backend._ready = True
    text = "你好，很高兴认识你。今天我们来测试一下语音合成的稳定性，看看这段稍微长一点的中文是不是能清楚自然地读出来。"
    wav, meta = backend.synthesize(text, max_audio_length=64, segment_max_chars=24)

    assert len(calls) > 1
    assert meta["segmented"] is True
    assert meta["segment_count"] == len(calls)
    assert meta["samples"] == 240 * len(calls)
    assert calls[0][1]["codec_eos_logit_offset"] == 0
    assert calls[0][1]["talker_top_k"] == 50
    assert calls[0][1]["talker_top_p"] == 1.0
    assert calls[0][1]["predictor_top_k"] == 50
    assert calls[0][1]["predictor_top_p"] == 1.0
    assert calls[0][1]["min_audio_length"] == 30
    with wave.open(io.BytesIO(wav), "rb") as reader:
        assert reader.getframerate() == 24000
        assert reader.getnframes() == 240 * len(calls)
