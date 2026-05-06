import importlib
import sys


def _reload_ipc(monkeypatch):
    monkeypatch.delitem(sys.modules, "backends.trt_edge_llm_ipc", raising=False)
    import backends.trt_edge_llm_ipc as ipc

    return importlib.reload(ipc)


def test_worker_paths_prefer_voice_native_build(monkeypatch, tmp_path):
    voice_root = tmp_path / "voice"
    worker_dir = voice_root / "build" / "edgellm_voice_worker" / "workers"
    worker_dir.mkdir(parents=True)
    tts_worker = worker_dir / "qwen3_tts_worker"
    asr_worker = worker_dir / "qwen3_asr_worker"
    tts_worker.touch()
    asr_worker.touch()

    monkeypatch.setenv("JETSON_VOICE_BASE", str(voice_root))
    monkeypatch.setenv("EDGE_LLM_BASE", str(tmp_path / "edge"))
    monkeypatch.setenv("EDGE_LLM_BUILD_DIR", "build_sm87")
    monkeypatch.delenv("EDGE_LLM_TTS_WORKER_BIN", raising=False)
    monkeypatch.delenv("EDGE_LLM_ASR_WORKER_BIN", raising=False)

    ipc = _reload_ipc(monkeypatch)

    assert ipc.TTS_WORKER_BINARY == str(tts_worker)
    assert ipc.ASR_WORKER_BINARY == str(asr_worker)


def test_worker_paths_fall_back_to_edgellm_examples(monkeypatch, tmp_path):
    edge_root = tmp_path / "edge"
    monkeypatch.setenv("JETSON_VOICE_BASE", str(tmp_path / "voice"))
    monkeypatch.setenv("EDGE_LLM_BASE", str(edge_root))
    monkeypatch.setenv("EDGE_LLM_BUILD_DIR", "build_sm87")
    monkeypatch.delenv("EDGE_LLM_TTS_WORKER_BIN", raising=False)
    monkeypatch.delenv("EDGE_LLM_ASR_WORKER_BIN", raising=False)

    ipc = _reload_ipc(monkeypatch)

    assert ipc.TTS_WORKER_BINARY == str(edge_root / "build_sm87" / "examples" / "omni" / "qwen3_tts_worker")
    assert ipc.ASR_WORKER_BINARY == str(edge_root / "build_sm87" / "examples" / "llm" / "qwen3_asr_worker")


def test_tts_code_predictor_dir_can_be_overridden(monkeypatch, tmp_path):
    cp_dir = tmp_path / "hf_export" / "code_predictor"
    monkeypatch.setenv("EDGE_LLM_TTS_CP_DIR", str(cp_dir))

    ipc = _reload_ipc(monkeypatch)

    assert ipc.TTS_CODE_PREDICTOR_DIR == str(cp_dir)
