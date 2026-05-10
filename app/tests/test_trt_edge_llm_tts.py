import json
import subprocess
import io
import wave
import types


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
    assert meta["samples"] > 240 * len(calls)
    assert meta["segment_pauses_ms"] == [120, 80]
    assert calls[0][1]["codec_eos_logit_offset"] == 0
    assert calls[0][1]["talker_top_k"] == 50
    assert calls[0][1]["talker_top_p"] == 1.0
    assert calls[0][1]["predictor_top_k"] == 50
    assert calls[0][1]["predictor_top_p"] == 1.0
    assert calls[0][1]["min_audio_length"] == 30
    with wave.open(io.BytesIO(wav), "rb") as reader:
        assert reader.getframerate() == 24000
        assert reader.getnframes() == meta["samples"]


def test_cjk_default_segmentation_prefers_sentence_boundary(monkeypatch):
    import backends.trt_edge_llm_tts as tts_mod

    monkeypatch.delenv("EDGE_LLM_TTS_CJK_SEGMENT_MAX_CHARS", raising=False)
    text = "你好，今天我们继续验证语音合成的稳定性。这个版本应该保持清晰自然，不应该出现逐渐变沙、吞音或者明显的噪声积累。"
    parts = tts_mod._split_tts_text(text)

    assert parts == [
        "你好，今天我们继续验证语音合成的稳定性。",
        "这个版本应该保持清晰自然，不应该出现逐渐变沙、吞音或者明显的噪声积累。",
    ]


def test_product_backend_bypasses_generic_segmentation(monkeypatch):
    import backends.trt_edge_llm_tts as tts_mod

    calls = []

    class FakeProductBackend:
        def synthesize(self, text, **kwargs):
            calls.append((text, kwargs.get("seed")))
            return _make_wav_bytes(240), {"backend": "product_explicit_kv"}

    monkeypatch.setenv("JETSON_VOICE_TTS_SEED", "42")
    backend = tts_mod.TRTEdgeLLMTTSBackend()
    backend._ready = True
    backend._product_backend = FakeProductBackend()

    text = "你好，今天我们继续验证语音合成的稳定性。这个版本应该保持清晰自然，不应该出现逐渐变沙、吞音或者明显的噪声积累。"
    _, meta = backend.synthesize(text, seed=42)

    assert meta["backend"] == "product_explicit_kv"
    assert calls == [(text, 42)]


def test_qwen3_trt_caps_trt_vocoder_frames_and_passes_seed(monkeypatch):
    import backends.qwen3_trt as qwen3_mod

    captured = {}

    class FakeTokenizer:
        def encode(self, text):
            return types.SimpleNamespace(ids=[1, 2, 3])

    class FakeEngine:
        def synthesize(self, **kwargs):
            captured.update(kwargs)
            return {
                "wav_bytes": _make_wav_bytes(240),
                "duration": 0.01,
                "rtf": 0.5,
                "n_frames": 100,
                "per_step_ms": 1.0,
            }

    monkeypatch.setenv("TTS_VOCODER_TRT", "1")
    monkeypatch.setenv("TTS_TRT_VOCODER_MAX_FRAMES", "100")
    backend = qwen3_mod.Qwen3TRTBackend()
    backend._ready = True
    backend._tokenizer = FakeTokenizer()
    backend._engine = FakeEngine()

    _, meta = backend.synthesize("你好", max_audio_length=200, seed=42)

    assert captured["max_frames"] == 100
    assert captured["seed"] == 42
    assert meta["seed"] == 42


def test_qwen3_trt_collects_streaming_for_long_offline_requests(monkeypatch):
    import backends.qwen3_trt as qwen3_mod

    class FakeTokenizer:
        def encode(self, text):
            return types.SimpleNamespace(ids=list(range(60)))

    class FakeEngine:
        def synthesize(self, **kwargs):
            raise AssertionError("long offline requests should use streaming collection")

    monkeypatch.setenv("TTS_VOCODER_TRT", "1")
    monkeypatch.setenv("TTS_TRT_VOCODER_MAX_FRAMES", "100")
    backend = qwen3_mod.Qwen3TRTBackend()
    backend._ready = True
    backend._tokenizer = FakeTokenizer()
    backend._engine = FakeEngine()

    calls = []

    def fake_streaming(text, **kwargs):
        calls.append((text, kwargs))
        yield b"\x01\x00" * 240
        yield b"\x02\x00" * 240

    monkeypatch.setattr(backend, "generate_streaming", fake_streaming)

    wav, meta = backend.synthesize("这是一段比较长的文本", max_audio_length=200, seed=42)

    assert calls[0][1]["max_frames"] == 200
    assert calls[0][1]["seed"] == 42
    assert meta["offline_collected_streaming"] is True
    assert meta["samples"] == 480
    with wave.open(io.BytesIO(wav), "rb") as reader:
        assert reader.getframerate() == 24000
        assert reader.getnframes() == 480


def test_qwen3_trt_product_segments_cjk_punctuation(monkeypatch):
    import backends.qwen3_trt as qwen3_mod

    class FakeTokenizer:
        def encode(self, text):
            return types.SimpleNamespace(ids=[1, 2, 3])

    class FakeEngine:
        def __init__(self):
            self.calls = []

        def synthesize(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "wav_bytes": _make_wav_bytes(240),
                "duration": 0.01,
                "rtf": 0.5,
                "n_frames": 10,
                "per_step_ms": 1.0,
            }

    fake_engine = FakeEngine()
    monkeypatch.setenv("TTS_VOCODER_TRT", "1")
    monkeypatch.setenv("TTS_TRT_VOCODER_MAX_FRAMES", "100")
    monkeypatch.setenv("QWEN3_TTS_PRODUCT_SEGMENT_TEXT", "1")
    backend = qwen3_mod.Qwen3TRTBackend()
    backend._ready = True
    backend._tokenizer = FakeTokenizer()
    backend._engine = fake_engine

    wav, meta = backend.synthesize("今天天气很好，我们一起测试语音合成。", max_audio_length=100, seed=42)

    assert meta["product_segmented"] is True
    assert [call["text"] for call in fake_engine.calls] == ["今天天气很好，", "我们一起测试语音合成。"]
    assert all(call["seed"] == 42 for call in fake_engine.calls)
    assert meta["segment_pauses_ms"] == [120]
    with wave.open(io.BytesIO(wav), "rb") as reader:
        assert reader.getnframes() == 480 + int(24000 * 0.12)


def test_qwen3_trt_product_segmentation_keeps_ascii_words_and_punctuation():
    import backends.qwen3_trt as qwen3_mod

    text = "今天我们继续验证千问语音合成在 Jetson 上的稳定性。"
    parts = qwen3_mod._split_product_tts_text(text, max_chars=20)

    assert "".join(parts) == text
    assert parts == ["今天我们继续验证千问语音合成在 ", "Jetson 上的稳定性。"]
    assert all(part not in "。！？!?；;，,、：" for part in parts)
    assert all("Jets" != part and "on 上的稳定性。" != part for part in parts)


def test_product_explicit_kv_backend_is_selected_explicitly(monkeypatch, tmp_path):
    import backends.trt_edge_llm_tts as tts_mod

    calls = []

    class FakeProductBackend:
        def preload(self):
            calls.append(("preload", None))

        def synthesize(self, text, **kwargs):
            calls.append(("synthesize", text, kwargs))
            return b"wav", {"backend": "product_explicit_kv"}

    fake_module = types.SimpleNamespace(Qwen3TRTBackend=FakeProductBackend)
    monkeypatch.setenv("JETSON_VOICE_TTS_BACKEND", "product_explicit_kv")
    monkeypatch.setenv("JETSON_VOICE_TTS_MODEL_BASE", str(tmp_path / "models" / "qwen3-tts"))
    monkeypatch.setenv("JETSON_VOICE_TTS_NATIVE_MODULE_DIR", str(tmp_path / "app_overlay"))
    monkeypatch.setattr(tts_mod.importlib, "import_module", lambda name: fake_module)
    monkeypatch.setattr(tts_mod.importlib, "reload", lambda module: module)

    backend = tts_mod.TRTEdgeLLMTTSBackend()
    backend.preload()
    wav, meta = backend.synthesize("你好")

    assert backend.is_ready()
    assert wav == b"wav"
    assert meta["backend"] == "product_explicit_kv"
    assert calls[0] == ("preload", None)
    assert calls[1][0] == "synthesize"
    assert calls[1][1] == "你好"


def test_old_native_fallback_env_no_longer_changes_backend(monkeypatch, tmp_path):
    import backends.trt_edge_llm_tts as tts_mod

    required = [
        tmp_path / "worker",
        tmp_path / "plugin.so",
        tmp_path / "talker" / "config.json",
        tmp_path / "talker" / "llm.engine",
        tmp_path / "tokenizer" / "tokenizer.json",
    ]
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    monkeypatch.setenv("EDGE_LLM_TTS_NATIVE_FALLBACK", "1")
    monkeypatch.delenv("JETSON_VOICE_TTS_BACKEND", raising=False)
    monkeypatch.delenv("EDGE_LLM_TTS_BACKEND", raising=False)
    monkeypatch.setattr(tts_mod, "TTS_WORKER_BINARY", str(tmp_path / "worker"))
    monkeypatch.setattr(tts_mod, "PLUGIN_PATH", str(tmp_path / "plugin.so"))
    monkeypatch.setattr(tts_mod, "TTS_TALKER_DIR", str(tmp_path / "talker"))
    monkeypatch.setattr(tts_mod, "TTS_TOKENIZER_DIR", str(tmp_path / "tokenizer"))
    monkeypatch.setattr(tts_mod.TRTEdgeLLMTTSBackend, "_ensure_worker", lambda self: None)

    backend = tts_mod.TRTEdgeLLMTTSBackend()
    backend.preload()

    assert backend.is_ready()
    assert backend._product_backend is None


def test_edgellm_worker_defaults_match_dual_resident_streaming_profile(monkeypatch):
    import backends.trt_edge_llm_tts as tts_mod

    captured = {}

    class FakeWorker:
        stdin = None
        stdout = None

    monkeypatch.delenv("EDGE_LLM_TTS_CUDA_GRAPH", raising=False)
    monkeypatch.delenv("EDGE_LLM_TTS_FIRST_CHUNK_FRAMES", raising=False)
    monkeypatch.delenv("EDGE_LLM_TTS_CHUNK_FRAMES", raising=False)
    monkeypatch.delenv("EDGE_LLM_TTS_MAX_CHUNK_FRAMES", raising=False)
    monkeypatch.delenv("EDGE_LLM_TTS_CODE2WAV_CONTEXT_FRAMES", raising=False)
    monkeypatch.delenv("EDGE_LLM_TTS_STATEFUL_CODE2WAV", raising=False)
    monkeypatch.delenv("QWEN3_TTS_CP_DECODE_CUDA_GRAPH", raising=False)
    monkeypatch.delenv("QWEN3_TTS_ACTIVE_CP_GROUPS", raising=False)
    monkeypatch.delenv("EDGE_LLM_TTS_PERF_PROFILE", raising=False)

    backend = tts_mod.TRTEdgeLLMTTSBackend()
    env = backend._worker_env()

    def fake_ensure_worker():
        backend._worker = FakeWorker()
        backend._worker.stdin = types.SimpleNamespace(
            write=lambda data: captured.setdefault("request", json.loads(data)),
            flush=lambda: None,
        )
        backend._worker.stdout = types.SimpleNamespace(
            readline=lambda: json.dumps({"event": "done", "ok": True}) + "\n"
        )

    monkeypatch.setattr(backend, "_ensure_worker", fake_ensure_worker)
    backend._ready = True

    assert list(backend.generate_streaming("你好")) == []
    assert env["EDGE_LLM_TTS_CUDA_GRAPH"] == "0"
    assert env["EDGE_LLM_TTS_STATEFUL_CODE2WAV"] == "1"
    assert env["EDGE_LLM_TTS_CODE2WAV_CONTEXT_FRAMES"] == "0"
    assert env["QWEN3_TTS_CP_DECODE_CUDA_GRAPH"] == "1"
    assert env["QWEN3_TTS_ACTIVE_CP_GROUPS"] == "13"
    assert env["QWEN3_TTS_VOCAB_PRUNED"] == "0"
    assert captured["request"]["first_chunk_frames"] == 7
    assert captured["request"]["chunk_frames"] == 10
    assert captured["request"]["max_chunk_frames"] == 10
    assert captured["request"]["adaptive_chunks"] is False


def test_edgellm_worker_legacy_v2v_profile_uses_first_frame_fast_window(monkeypatch):
    import backends.trt_edge_llm_tts as tts_mod

    captured = {}

    class FakeWorker:
        stdin = None
        stdout = None

    monkeypatch.delenv("EDGE_LLM_TTS_FIRST_CHUNK_FRAMES", raising=False)
    monkeypatch.delenv("EDGE_LLM_TTS_CHUNK_FRAMES", raising=False)
    monkeypatch.delenv("EDGE_LLM_TTS_MAX_CHUNK_FRAMES", raising=False)
    monkeypatch.setenv("EDGE_LLM_TTS_STATEFUL_CODE2WAV", "0")

    backend = tts_mod.TRTEdgeLLMTTSBackend()

    def fake_ensure_worker():
        backend._worker = FakeWorker()
        backend._worker.stdin = types.SimpleNamespace(
            write=lambda data: captured.setdefault("request", json.loads(data)),
            flush=lambda: None,
        )
        backend._worker.stdout = types.SimpleNamespace(
            readline=lambda: json.dumps({"event": "done", "ok": True}) + "\n"
        )

    monkeypatch.setattr(backend, "_ensure_worker", fake_ensure_worker)
    backend._ready = True

    assert list(backend.generate_streaming("你好", streaming_profile="v2v")) == []
    assert captured["request"]["first_chunk_frames"] == 1
    assert captured["request"]["chunk_frames"] == 97
    assert captured["request"]["max_chunk_frames"] == 97
    assert captured["request"]["adaptive_chunks"] is False


def test_edgellm_worker_official_profile_uses_upstream_like_streaming_defaults(monkeypatch):
    import backends.trt_edge_llm_tts as tts_mod

    captured = {}

    class FakeWorker:
        stdin = None
        stdout = None

    monkeypatch.setenv("EDGE_LLM_QWEN3_PROFILE", "official")
    monkeypatch.delenv("EDGE_LLM_TTS_STATEFUL_CODE2WAV", raising=False)
    monkeypatch.delenv("EDGE_LLM_TTS_FIRST_CHUNK_FRAMES", raising=False)
    monkeypatch.delenv("EDGE_LLM_TTS_CHUNK_FRAMES", raising=False)
    monkeypatch.delenv("EDGE_LLM_TTS_MAX_CHUNK_FRAMES", raising=False)
    monkeypatch.delenv("EDGE_LLM_TTS_CODE2WAV_CONTEXT_FRAMES", raising=False)
    monkeypatch.delenv("QWEN3_TTS_CP_DECODE_CUDA_GRAPH", raising=False)
    monkeypatch.delenv("QWEN3_TTS_ACTIVE_CP_GROUPS", raising=False)
    monkeypatch.setattr(tts_mod, "qwen3_highperf_enabled", lambda: False)

    backend = tts_mod.TRTEdgeLLMTTSBackend()
    env = backend._worker_env()

    def fake_ensure_worker():
        backend._worker = FakeWorker()
        backend._worker.stdin = types.SimpleNamespace(
            write=lambda data: captured.setdefault("request", json.loads(data)),
            flush=lambda: None,
        )
        backend._worker.stdout = types.SimpleNamespace(
            readline=lambda: json.dumps({"event": "done", "ok": True}) + "\n"
        )

    monkeypatch.setattr(backend, "_ensure_worker", fake_ensure_worker)
    backend._ready = True

    assert list(backend.generate_streaming("你好")) == []
    assert env["EDGE_LLM_TTS_STATEFUL_CODE2WAV"] == "0"
    assert env["EDGE_LLM_TTS_CODE2WAV_CONTEXT_FRAMES"] == "3"
    assert "QWEN3_TTS_CP_DECODE_CUDA_GRAPH" not in env
    assert "QWEN3_TTS_ACTIVE_CP_GROUPS" not in env
    assert captured["request"]["first_chunk_frames"] == 50
    assert captured["request"]["chunk_frames"] == 97
    assert captured["request"]["max_chunk_frames"] == 97


def test_edgellm_worker_stateful_profile_uses_small_continuous_chunks(monkeypatch):
    import backends.trt_edge_llm_tts as tts_mod

    captured = {}

    class FakeWorker:
        stdin = None
        stdout = None

    monkeypatch.setenv("EDGE_LLM_TTS_STATEFUL_CODE2WAV", "1")
    monkeypatch.delenv("EDGE_LLM_TTS_FIRST_CHUNK_FRAMES", raising=False)
    monkeypatch.delenv("EDGE_LLM_TTS_CHUNK_FRAMES", raising=False)
    monkeypatch.delenv("EDGE_LLM_TTS_MAX_CHUNK_FRAMES", raising=False)
    monkeypatch.delenv("EDGE_LLM_TTS_CODE2WAV_CONTEXT_FRAMES", raising=False)
    monkeypatch.delenv("QWEN3_TTS_CP_DECODE_CUDA_GRAPH", raising=False)

    backend = tts_mod.TRTEdgeLLMTTSBackend()
    env = backend._worker_env()

    def fake_ensure_worker():
        backend._worker = FakeWorker()
        backend._worker.stdin = types.SimpleNamespace(
            write=lambda data: captured.setdefault("request", json.loads(data)),
            flush=lambda: None,
        )
        backend._worker.stdout = types.SimpleNamespace(
            readline=lambda: json.dumps({"event": "done", "ok": True}) + "\n"
        )

    monkeypatch.setattr(backend, "_ensure_worker", fake_ensure_worker)
    backend._ready = True

    assert list(backend.generate_streaming("你好")) == []
    assert env["EDGE_LLM_TTS_CODE2WAV_CONTEXT_FRAMES"] == "0"
    assert env["QWEN3_TTS_CP_DECODE_CUDA_GRAPH"] == "1"
    assert env["QWEN3_TTS_ACTIVE_CP_GROUPS"] == "13"
    assert captured["request"]["first_chunk_frames"] == 7
    assert captured["request"]["chunk_frames"] == 10
    assert captured["request"]["max_chunk_frames"] == 10
    assert captured["request"]["adaptive_chunks"] is False


def test_edgellm_worker_stateful_balanced_profile_uses_cp13(monkeypatch):
    import backends.trt_edge_llm_tts as tts_mod

    captured = {}

    class FakeWorker:
        stdin = None
        stdout = None

    monkeypatch.setenv("EDGE_LLM_TTS_STATEFUL_CODE2WAV", "1")
    monkeypatch.setenv("EDGE_LLM_TTS_PERF_PROFILE", "balanced")
    monkeypatch.delenv("QWEN3_TTS_ACTIVE_CP_GROUPS", raising=False)
    monkeypatch.delenv("EDGE_LLM_TTS_FIRST_CHUNK_FRAMES", raising=False)

    backend = tts_mod.TRTEdgeLLMTTSBackend()
    env = backend._worker_env()

    def fake_ensure_worker():
        backend._worker = FakeWorker()
        backend._worker.stdin = types.SimpleNamespace(
            write=lambda data: captured.setdefault("request", json.loads(data)),
            flush=lambda: None,
        )
        backend._worker.stdout = types.SimpleNamespace(
            readline=lambda: json.dumps({"event": "done", "ok": True}) + "\n"
        )

    monkeypatch.setattr(backend, "_ensure_worker", fake_ensure_worker)
    backend._ready = True

    assert list(backend.generate_streaming("你好")) == []
    assert env["QWEN3_TTS_CP_DECODE_CUDA_GRAPH"] == "1"
    assert env["QWEN3_TTS_ACTIVE_CP_GROUPS"] == "13"
    assert captured["request"]["first_chunk_frames"] == 6


def test_edgellm_worker_stateful_fast_profile_uses_first4(monkeypatch):
    import backends.trt_edge_llm_tts as tts_mod

    captured = {}

    class FakeWorker:
        stdin = None
        stdout = None

    monkeypatch.setenv("EDGE_LLM_TTS_STATEFUL_CODE2WAV", "1")
    monkeypatch.setenv("EDGE_LLM_TTS_PERF_PROFILE", "fast")
    monkeypatch.delenv("EDGE_LLM_TTS_FIRST_CHUNK_FRAMES", raising=False)

    backend = tts_mod.TRTEdgeLLMTTSBackend()
    env = backend._worker_env()

    def fake_ensure_worker():
        backend._worker = FakeWorker()
        backend._worker.stdin = types.SimpleNamespace(
            write=lambda data: captured.setdefault("request", json.loads(data)),
            flush=lambda: None,
        )
        backend._worker.stdout = types.SimpleNamespace(
            readline=lambda: json.dumps({"event": "done", "ok": True}) + "\n"
        )

    monkeypatch.setattr(backend, "_ensure_worker", fake_ensure_worker)
    backend._ready = True

    assert list(backend.generate_streaming("你好")) == []
    assert env["QWEN3_TTS_ACTIVE_CP_GROUPS"] == "13"
    assert captured["request"]["first_chunk_frames"] == 4


def test_edgellm_worker_stateful_respects_explicit_cp_groups(monkeypatch):
    import backends.trt_edge_llm_tts as tts_mod

    monkeypatch.setenv("EDGE_LLM_TTS_STATEFUL_CODE2WAV", "1")
    monkeypatch.setenv("EDGE_LLM_TTS_PERF_PROFILE", "balanced")
    monkeypatch.setenv("QWEN3_TTS_ACTIVE_CP_GROUPS", "14")

    backend = tts_mod.TRTEdgeLLMTTSBackend()
    env = backend._worker_env()

    assert env["QWEN3_TTS_ACTIVE_CP_GROUPS"] == "14"
