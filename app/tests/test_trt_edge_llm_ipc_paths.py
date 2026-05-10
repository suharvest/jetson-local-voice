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


def test_tts_code_predictor_prefers_bf16_io_dir_when_present(monkeypatch, tmp_path):
    bf16_io_dir = tmp_path / "cp_bf16_io"
    default_talker = tmp_path / "runtime" / "engines" / "talker"
    bf16_io_dir.mkdir(parents=True)
    default_talker.mkdir(parents=True)
    monkeypatch.delenv("EDGE_LLM_TTS_CP_DIR", raising=False)
    monkeypatch.setenv("EDGE_LLM_TTS_CP_BF16_IO_DIR", str(bf16_io_dir))
    monkeypatch.setenv("EDGE_LLM_TTS_TALKER_DIR", str(default_talker))

    ipc = _reload_ipc(monkeypatch)

    assert ipc.TTS_CODE_PREDICTOR_DIR == str(bf16_io_dir)


def test_tts_code_predictor_uses_official_dir_in_official_profile(monkeypatch, tmp_path):
    bf16_io_dir = tmp_path / "cp_bf16_io"
    default_talker = tmp_path / "runtime" / "engines" / "talker"
    bf16_io_dir.mkdir(parents=True)
    default_talker.mkdir(parents=True)
    monkeypatch.delenv("EDGE_LLM_TTS_CP_DIR", raising=False)
    monkeypatch.setenv("EDGE_LLM_QWEN3_PROFILE", "official")
    monkeypatch.setenv("EDGE_LLM_TTS_CP_BF16_IO_DIR", str(bf16_io_dir))
    monkeypatch.setenv("EDGE_LLM_TTS_TALKER_DIR", str(default_talker))

    ipc = _reload_ipc(monkeypatch)

    assert ipc.TTS_CODE_PREDICTOR_DIR == str(tmp_path / "runtime" / "engines" / "code_predictor")


def test_tts_vocab_pruned_selects_configured_talker_dir(monkeypatch, tmp_path):
    full_dir = tmp_path / "talker_full"
    pruned_dir = tmp_path / "talker_pruned"
    monkeypatch.setenv("EDGE_LLM_TTS_FULL_TALKER_DIR", str(full_dir))
    monkeypatch.setenv("EDGE_LLM_TTS_PRUNED_TALKER_DIR", str(pruned_dir))
    monkeypatch.setenv("EDGE_LLM_TTS_VOCAB_PRUNED", "1")
    monkeypatch.delenv("EDGE_LLM_TTS_TALKER_DIR", raising=False)

    ipc = _reload_ipc(monkeypatch)

    assert ipc.TTS_TALKER_DIR == str(pruned_dir)

    monkeypatch.setenv("EDGE_LLM_TTS_VOCAB_PRUNED", "0")
    ipc = _reload_ipc(monkeypatch)

    assert ipc.TTS_TALKER_DIR == str(full_dir)


def test_tts_vocab_pruned_defaults_to_full_talker_dir(monkeypatch, tmp_path):
    full_dir = tmp_path / "talker_full"
    pruned_dir = tmp_path / "talker_pruned"
    monkeypatch.setenv("EDGE_LLM_TTS_FULL_TALKER_DIR", str(full_dir))
    monkeypatch.setenv("EDGE_LLM_TTS_PRUNED_TALKER_DIR", str(pruned_dir))
    monkeypatch.delenv("EDGE_LLM_TTS_VOCAB_PRUNED", raising=False)
    monkeypatch.delenv("QWEN3_TTS_VOCAB_PRUNED", raising=False)
    monkeypatch.delenv("EDGE_LLM_TTS_TALKER_DIR", raising=False)

    ipc = _reload_ipc(monkeypatch)

    assert ipc.TTS_TALKER_DIR == str(full_dir)


def test_tts_talker_dir_override_wins_over_vocab_pruned(monkeypatch, tmp_path):
    explicit_dir = tmp_path / "explicit"
    pruned_dir = tmp_path / "talker_pruned"
    monkeypatch.setenv("EDGE_LLM_TTS_TALKER_DIR", str(explicit_dir))
    monkeypatch.setenv("EDGE_LLM_TTS_PRUNED_TALKER_DIR", str(pruned_dir))
    monkeypatch.setenv("EDGE_LLM_TTS_VOCAB_PRUNED", "1")

    ipc = _reload_ipc(monkeypatch)

    assert ipc.TTS_TALKER_DIR == str(explicit_dir)


def test_asr_vocab_pruned_selects_configured_engine_dir(monkeypatch, tmp_path):
    full_dir = tmp_path / "asr_full"
    pruned_dir = tmp_path / "asr_pruned"
    monkeypatch.setenv("EDGE_LLM_ASR_FULL_ENGINE_DIR", str(full_dir))
    monkeypatch.setenv("EDGE_LLM_ASR_PRUNED_ENGINE_DIR", str(pruned_dir))
    monkeypatch.setenv("EDGE_LLM_ASR_VOCAB_PRUNED", "1")
    monkeypatch.delenv("EDGE_LLM_ASR_ENGINE_DIR", raising=False)

    ipc = _reload_ipc(monkeypatch)

    assert ipc.ASR_ENGINE_DIR == str(pruned_dir)

    monkeypatch.setenv("EDGE_LLM_ASR_VOCAB_PRUNED", "0")
    ipc = _reload_ipc(monkeypatch)

    assert ipc.ASR_ENGINE_DIR == str(full_dir)


def test_asr_vocab_pruned_defaults_to_full_engine_dir(monkeypatch, tmp_path):
    full_dir = tmp_path / "asr_full"
    pruned_dir = tmp_path / "asr_pruned"
    monkeypatch.setenv("EDGE_LLM_ASR_FULL_ENGINE_DIR", str(full_dir))
    monkeypatch.setenv("EDGE_LLM_ASR_PRUNED_ENGINE_DIR", str(pruned_dir))
    monkeypatch.delenv("EDGE_LLM_ASR_VOCAB_PRUNED", raising=False)
    monkeypatch.delenv("EDGE_LLM_ASR_ENGINE_DIR", raising=False)

    ipc = _reload_ipc(monkeypatch)

    assert ipc.ASR_ENGINE_DIR == str(full_dir)


def test_asr_engine_dir_override_wins_over_vocab_pruned(monkeypatch, tmp_path):
    explicit_dir = tmp_path / "explicit_asr"
    pruned_dir = tmp_path / "asr_pruned"
    monkeypatch.setenv("EDGE_LLM_ASR_ENGINE_DIR", str(explicit_dir))
    monkeypatch.setenv("EDGE_LLM_ASR_PRUNED_ENGINE_DIR", str(pruned_dir))
    monkeypatch.setenv("EDGE_LLM_ASR_VOCAB_PRUNED", "1")

    ipc = _reload_ipc(monkeypatch)

    assert ipc.ASR_ENGINE_DIR == str(explicit_dir)
