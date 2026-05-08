#!/usr/bin/env python3
"""Run a single qwen3_tts_worker streaming smoke request.

This harness exists because the worker is an stdin/stdout JSON process and its
stderr must be drained while waiting for the JSON ready/event lines.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import threading
import time


def drain_stderr(proc: subprocess.Popen[str], lines: list[str]) -> None:
    assert proc.stderr is not None
    for line in proc.stderr:
        lines.append(line)
        if "[JV_MEM]" in line:
            print(line.rstrip(), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    parser.add_argument("--talker-dir", required=True)
    parser.add_argument("--talker-engine", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--code2wav-dir", required=True)
    parser.add_argument("--cp-dir", required=True)
    parser.add_argument("--plugin-path", required=True)
    parser.add_argument("--text", default="你好")
    parser.add_argument("--language", default="chinese")
    parser.add_argument("--max-audio-length", type=int, default=50)
    parser.add_argument("--min-audio-length", type=int, default=10)
    args = parser.parse_args()

    env = os.environ.copy()
    env["EDGELLM_PLUGIN_PATH"] = args.plugin_path
    env["EDGE_LLM_TTS_LAZY_CODE2WAV"] = "1"
    env["EDGE_LLM_TTS_CUDA_GRAPH"] = "0"

    cmd = [
        args.worker,
        "--talkerEngineDir",
        args.talker_dir,
        "--qwen3TtsTalkerBackend",
        "qwen3_tts_explicit_kv",
        "--qwen3TtsTalkerEngine",
        args.talker_engine,
        "--tokenizerDir",
        args.tokenizer_dir,
        "--code2wavEngineDir",
        args.code2wav_dir,
        "--codePredictorEngineDir",
        args.cp_dir,
        "--codePredictorBackend",
        "qwen3_tts_native",
        "--qwen3TtsTextProjection",
        "host_fp32",
        "--qwen3TtsPromptKvCache",
        "0",
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )
    stderr_lines: list[str] = []
    threading.Thread(target=drain_stderr, args=(proc, stderr_lines), daemon=True).start()
    assert proc.stdin is not None and proc.stdout is not None

    ready_line = proc.stdout.readline()
    if not ready_line:
        raise RuntimeError("worker exited before ready: " + "".join(stderr_lines)[-2000:])
    print(ready_line.rstrip(), flush=True)
    ready = json.loads(ready_line)
    if ready.get("event") != "ready":
        raise RuntimeError(f"unexpected ready event: {ready}")

    req = {
        "id": "smoke",
        "text": args.text,
        "language": args.language,
        "stream": True,
        "stream_only": True,
        "chunk_transport": "base64",
        "chunk_format": "pcm_s16le",
        "first_chunk_frames": 25,
        "chunk_frames": 25,
        "adaptive_chunks": False,
        "max_audio_length": args.max_audio_length,
        "min_audio_length": args.min_audio_length,
        "talker_temperature": 0.9,
        "talker_top_k": 40,
        "talker_top_p": 0.8,
        "predictor_temperature": 0.9,
        "predictor_top_k": 40,
        "predictor_top_p": 0.8,
        "repetition_penalty": 1.05,
        "codec_eos_logit_offset": 0.0,
    }
    start = time.time()
    proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
    proc.stdin.flush()

    chunks = 0
    pcm_bytes = 0
    first_chunk_s = None
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("worker exited during request: " + "".join(stderr_lines)[-2000:])
        print(line.rstrip(), flush=True)
        event = json.loads(line)
        if not event.get("ok", False):
            print("stderr_tail_begin", flush=True)
            print("".join(stderr_lines)[-4000:], flush=True)
            print("stderr_tail_end", flush=True)
            raise RuntimeError(f"worker error: {event}")
        if event.get("event") == "chunk":
            chunks += 1
            if first_chunk_s is None:
                first_chunk_s = time.time() - start
            pcm_bytes += len(base64.b64decode(event.get("audio_b64", "")))
        if event.get("event") == "done":
            break

    print(
        json.dumps(
            {
                "summary": "ok",
                "chunks": chunks,
                "pcm_bytes": pcm_bytes,
                "first_chunk_wall_s": first_chunk_s,
                "total_wall_s": time.time() - start,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


if __name__ == "__main__":
    main()
