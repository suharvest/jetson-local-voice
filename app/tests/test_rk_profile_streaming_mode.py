import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _profile_env(name: str) -> dict[str, str]:
    data = json.loads((ROOT / "configs" / "profiles" / f"{name}.json").read_text())
    return data["env"]


def test_rk_profiles_do_not_shadow_true_streaming():
    profiles = [
        "rk3576-default",
        "rk3576-multilang",
        "rk3588-default",
        "rk3588-multilang",
    ]
    for name in profiles:
        env = _profile_env(name)
        assert env["QWEN3_ASR_STREAM_TRUE"] == "1"
        assert env["QWEN3_ASR_CHUNK_CONFIRM"] == "0"
