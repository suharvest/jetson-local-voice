"""Unit tests for the registry-driven create_asr_backend() factory.

The factory consults app.core.profile_loader.current_profile() for the
``asr_backend`` key and importlib-loads the matching module. Each test
injects a fake backend module via sys.modules and patches current_profile()
to return the right key.
"""

import importlib
import sys
import types
from unittest.mock import MagicMock

import pytest


def _make_mock_backend_class(name: str):
    cls = MagicMock(name=name)
    cls.return_value = MagicMock()
    return cls


def _inject_module(monkeypatch, dotted: str, class_name: str):
    """Inject a fake module at `dotted` exposing `class_name` and return the class mock.

    Walks the dotted path so intermediate packages exist in sys.modules.
    """
    parts = dotted.split(".")
    for i in range(1, len(parts)):
        prefix = ".".join(parts[:i])
        if prefix not in sys.modules:
            monkeypatch.setitem(sys.modules, prefix, types.ModuleType(prefix))
    mock_cls = _make_mock_backend_class(class_name)
    mod = types.ModuleType(dotted)
    setattr(mod, class_name, mock_cls)
    monkeypatch.setitem(sys.modules, dotted, mod)
    return mock_cls


def _patch_profile(monkeypatch, profile_dict):
    """Patch app.core.profile_loader.current_profile() to return profile_dict."""
    from app.core import profile_loader
    monkeypatch.setattr(profile_loader, "_CURRENT_PROFILE", profile_dict, raising=False)


@pytest.fixture
def fresh_factory(monkeypatch):
    """Stub numpy if absent (asr_backend imports it at module scope), reload the
    factory module, and return create_asr_backend."""
    if "numpy" not in sys.modules:
        monkeypatch.setitem(sys.modules, "numpy", MagicMock())
    from app.core import asr_backend
    importlib.reload(asr_backend)
    return asr_backend.create_asr_backend


def test_profile_selects_trt_edge_llm(monkeypatch, fresh_factory):
    cls = _inject_module(
        monkeypatch, "app.backends.jetson.trt_edge_llm_asr", "TRTEdgeLLMASRBackend"
    )
    _patch_profile(monkeypatch, {"asr_backend": "jetson.trt_edge_llm"})

    result = fresh_factory()

    cls.assert_called_once()
    assert result is cls.return_value


def test_profile_selects_paraformer(monkeypatch, fresh_factory):
    cls = _inject_module(
        monkeypatch, "app.backends.jetson.paraformer_trt", "ParaformerTRTBackend"
    )
    _patch_profile(monkeypatch, {"asr_backend": "jetson.paraformer_trt"})

    result = fresh_factory()

    cls.assert_called_once()
    assert result is cls.return_value


def test_profile_selects_sherpa(monkeypatch, fresh_factory):
    cls = _inject_module(
        monkeypatch, "app.backends.cpu.sherpa_asr", "SherpaASRBackend"
    )
    _patch_profile(monkeypatch, {"asr_backend": "cpu.sherpa_asr"})

    result = fresh_factory()

    cls.assert_called_once()
    assert result is cls.return_value


def test_missing_asr_backend_raises(monkeypatch, fresh_factory):
    _patch_profile(monkeypatch, {})
    with pytest.raises(ValueError, match="asr_backend"):
        fresh_factory()


def test_unknown_asr_backend_raises(monkeypatch, fresh_factory):
    _patch_profile(monkeypatch, {"asr_backend": "nonexistent.backend"})
    with pytest.raises(ValueError, match="Unknown asr_backend"):
        fresh_factory()
