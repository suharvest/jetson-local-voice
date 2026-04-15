"""Unit tests for create_asr_backend() factory selection logic.

Backend modules (sherpa_onnx, CUDA deps) are mocked via sys.modules so
the factory can be imported without hardware present.
"""

import sys
import types
import importlib
from unittest.mock import MagicMock


def _make_mock_backend_class(name: str):
    """Return a trivial mock class that looks like an ASRBackend subclass."""
    cls = MagicMock(name=name)
    instance = MagicMock()
    cls.return_value = instance
    return cls


def _inject_mock_sherpa(monkeypatch):
    """Inject a fake backends.sherpa_asr module into sys.modules."""
    mock_cls = _make_mock_backend_class("SherpaASRBackend")
    mock_mod = types.ModuleType("backends.sherpa_asr")
    mock_mod.SherpaASRBackend = mock_cls

    # Also need the parent package present
    if "backends" not in sys.modules:
        parent = types.ModuleType("backends")
        monkeypatch.setitem(sys.modules, "backends", parent)
    monkeypatch.setitem(sys.modules, "backends.sherpa_asr", mock_mod)
    return mock_cls


def _inject_mock_qwen3(monkeypatch):
    """Inject a fake backends.qwen3_asr module into sys.modules."""
    mock_cls = _make_mock_backend_class("Qwen3ASRBackend")
    mock_mod = types.ModuleType("backends.qwen3_asr")
    mock_mod.Qwen3ASRBackend = mock_cls

    if "backends" not in sys.modules:
        parent = types.ModuleType("backends")
        monkeypatch.setitem(sys.modules, "backends", parent)
    monkeypatch.setitem(sys.modules, "backends.qwen3_asr", mock_mod)

    # Ensure jetson_qwen3_speech is NOT importable so the local fallback is used
    monkeypatch.setitem(sys.modules, "jetson_qwen3_speech", None)
    return mock_cls


def _load_factory(monkeypatch):
    """Remove asr_backend from sys.modules cache and re-import it fresh.

    This is necessary because create_asr_backend uses lazy imports inside the
    function body, but the module itself must be reloaded to ensure we get a
    clean function reference unaffected by any previous import-cache state.
    """
    # Also stub out numpy so asr_backend.py can import without the real package
    if "numpy" not in sys.modules:
        monkeypatch.setitem(sys.modules, "numpy", MagicMock())

    monkeypatch.delitem(sys.modules, "asr_backend", raising=False)
    import asr_backend
    importlib.reload(asr_backend)
    return asr_backend.create_asr_backend


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_default_is_sherpa(monkeypatch):
    """No env vars set → backend defaults to SherpaASRBackend."""
    sherpa_cls = _inject_mock_sherpa(monkeypatch)
    monkeypatch.delenv("LANGUAGE_MODE", raising=False)
    monkeypatch.delenv("ASR_BACKEND", raising=False)

    create_asr_backend = _load_factory(monkeypatch)
    result = create_asr_backend()

    sherpa_cls.assert_called_once()
    assert result is sherpa_cls.return_value


def test_multilanguage_selects_qwen3(monkeypatch):
    """LANGUAGE_MODE=multilanguage → backend resolves to Qwen3ASRBackend."""
    qwen3_cls = _inject_mock_qwen3(monkeypatch)
    monkeypatch.setenv("LANGUAGE_MODE", "multilanguage")
    monkeypatch.delenv("ASR_BACKEND", raising=False)

    create_asr_backend = _load_factory(monkeypatch)
    result = create_asr_backend()

    qwen3_cls.assert_called_once()
    assert result is qwen3_cls.return_value


def test_explicit_backend_name(monkeypatch):
    """Passing backend_name='sherpa' explicitly → SherpaASRBackend regardless of env."""
    sherpa_cls = _inject_mock_sherpa(monkeypatch)
    monkeypatch.setenv("LANGUAGE_MODE", "multilanguage")  # would pick qwen3 if auto

    create_asr_backend = _load_factory(monkeypatch)
    result = create_asr_backend(backend_name="sherpa")

    sherpa_cls.assert_called_once()
    assert result is sherpa_cls.return_value


def test_unknown_backend_raises(monkeypatch):
    """Passing an unrecognised backend_name → ValueError."""
    monkeypatch.delenv("LANGUAGE_MODE", raising=False)
    monkeypatch.delenv("ASR_BACKEND", raising=False)

    create_asr_backend = _load_factory(monkeypatch)

    import pytest
    with pytest.raises(ValueError, match="Unknown ASR backend"):
        create_asr_backend(backend_name="nonexistent")
