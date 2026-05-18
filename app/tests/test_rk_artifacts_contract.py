import pytest

from app.core.rk_artifacts import RKArtifactError, _validate_runtime_contract


def test_rk_runtime_contract_accepts_expected_env(monkeypatch):
    monkeypatch.setenv("TTS_BACKEND", "matcha_rknn")
    monkeypatch.setenv("MATCHA_USE_ORT", "1")
    monkeypatch.setenv("VOCOS_FRAMES", "256")

    _validate_runtime_contract(
        {
            "runtime_contract": {
                "env": {
                    "TTS_BACKEND": "matcha_rknn",
                    "MATCHA_USE_ORT": "1",
                    "VOCOS_FRAMES": "256",
                }
            }
        }
    )


def test_rk_runtime_contract_rejects_shape_drift(monkeypatch):
    monkeypatch.setenv("TTS_BACKEND", "matcha_rknn")
    monkeypatch.setenv("MATCHA_USE_ORT", "0")
    monkeypatch.setenv("VOCOS_FRAMES", "256")

    with pytest.raises(RKArtifactError, match="MATCHA_USE_ORT"):
        _validate_runtime_contract(
            {
                "runtime_contract": {
                    "env": {
                        "TTS_BACKEND": "matcha_rknn",
                        "MATCHA_USE_ORT": "1",
                        "VOCOS_FRAMES": "256",
                    }
                }
            }
        )
