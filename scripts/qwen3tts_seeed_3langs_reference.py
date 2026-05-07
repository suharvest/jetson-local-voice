#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
from pathlib import Path

BASE = Path("/tmp/qwen3tts_ref_0507_from_nano")
WORK = Path("/tmp/qwen3tts_seeed_3langs_0507")
IN_DIR = WORK / "in"
OUT_DIR = WORK / "out"

TEXTS = {
    "zh": ("chinese", "欢迎了解矽递科技（Seeed Studio）对话式 AI 解决方案。"),
    "en": ("english", "Welcome to learn about Seeed Studio conversational AI solutions."),
    "ja": ("japanese", "Seeed Studioの対話型AIソリューションへようこそ。"),
}

COMMON = {
    "batch_size": 1,
    "apply_chat_template": True,
    "add_generation_prompt": True,
    "enable_thinking": False,
    "talker_temperature": 0.9,
    "talker_top_k": 40,
    "talker_top_p": 0.8,
    "predictor_temperature": 0.9,
    "predictor_top_k": 40,
    "predictor_top_p": 0.8,
    "repetition_penalty": 1.05,
    "codec_eos_logit_offset": 0.0,
    "max_audio_length": 128,
    "min_audio_length": 30,
}


def main() -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    IN_DIR.mkdir(parents=True)
    OUT_DIR.mkdir(parents=True)

    binary = BASE / "build/examples/omni/qwen3_tts_inference"
    plugin = BASE / "build/libNvInfer_edgellm_plugin.so"
    required = [
        binary,
        plugin,
        BASE / "talker",
        BASE / "cp_product_unified_0506",
        BASE / "code2wav",
        Path("/home/harvest/voice_test/models/qwen3-tts/engines/talker_decode_bf16.engine"),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("Missing required Qwen3-TTS runtime paths:\n" + "\n".join(missing))

    env = os.environ.copy()
    env.update(
        {
            "QWEN3_TTS_SEED": "42",
            "QWEN3_TTS_DIRECT_TALKER_ENGINE": "/home/harvest/voice_test/models/qwen3-tts/engines/talker_decode_bf16.engine",
            "QWEN3_TTS_HOST_TEXT_PROJECTION": "1",
            "EDGELLM_PLUGIN_PATH": str(plugin),
        }
    )

    for key, (language, text) in TEXTS.items():
        request = dict(COMMON)
        request["requests"] = [
            {
                "messages": [{"role": "user", "content": text}],
                "speaker": "",
                "language": language,
            }
        ]
        input_file = IN_DIR / f"{key}.json"
        input_file.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")

        one_out = OUT_DIR / key
        audio_dir = one_out / "audio"
        audio_dir.mkdir(parents=True)
        result_json = one_out / "result.json"
        log_file = one_out / "run.log"

        cmd = [
            str(binary),
            f"--talkerEngineDir={BASE / 'talker'}",
            f"--codePredictorEngineDir={BASE / 'cp_product_unified_0506'}",
            f"--code2wavEngineDir={BASE / 'code2wav'}",
            "--tokenizerDir=/home/harvest/qwen3-tts-trt-edge-llm-export",
            f"--inputFile={input_file}",
            f"--outputFile={result_json}",
            f"--outputAudioDir={audio_dir}",
            "--dumpProfile",
        ]
        with log_file.open("w", encoding="utf-8") as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

        src = audio_dir / "audio_req0.wav"
        dst = OUT_DIR / f"{key}_qwen3tts_seeed_conversational_ai.wav"
        shutil.copy2(src, dst)
        print(dst)


if __name__ == "__main__":
    main()
