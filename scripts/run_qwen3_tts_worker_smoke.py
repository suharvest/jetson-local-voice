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
import wave


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
    parser.add_argument("--first-chunk-frames", type=int, default=25)
    parser.add_argument("--chunk-frames", type=int, default=25)
    parser.add_argument("--max-chunk-frames", type=int, default=25)
    parser.add_argument("--chunk-growth-frames", type=int, default=0)
    parser.add_argument("--adaptive-chunks", action="store_true")
    parser.add_argument("--async-code2wav", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--preload-code2wav", action="store_true")
    parser.add_argument("--print-stderr", action="store_true")
    parser.add_argument("--output-pcm", default="")
    parser.add_argument("--output-wav", default="")
    parser.add_argument("--quiet-chunks", action="store_true")
    parser.add_argument("--print-chunk-meta", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--talker-temperature", type=float, default=0.9)
    parser.add_argument("--talker-top-k", type=int, default=40)
    parser.add_argument("--talker-top-p", type=float, default=0.8)
    parser.add_argument("--predictor-temperature", type=float, default=0.9)
    parser.add_argument("--predictor-top-k", type=int, default=40)
    parser.add_argument("--predictor-top-p", type=float, default=0.8)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    args = parser.parse_args()

    env = os.environ.copy()
    env["EDGELLM_PLUGIN_PATH"] = args.plugin_path
    env["EDGE_LLM_TTS_LAZY_CODE2WAV"] = "0" if args.preload_code2wav else "1"
    env["EDGE_LLM_TTS_CUDA_GRAPH"] = "1" if args.cuda_graph else "0"

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

    summaries = []
    total_runs = max(1, args.repeat + args.warmup_runs)
    for run_idx in range(total_runs):
        measured_idx = run_idx - args.warmup_runs
        is_warmup = run_idx < args.warmup_runs
        req_id = f"warmup_{run_idx + 1}" if is_warmup else f"smoke_{measured_idx + 1}"
        req = {
            "id": req_id,
            "text": args.text,
            "language": args.language,
            "stream": True,
            "stream_only": True,
            "async_code2wav": args.async_code2wav,
            "chunk_transport": "base64",
            "chunk_format": "pcm_s16le",
            "first_chunk_frames": args.first_chunk_frames,
            "chunk_frames": args.chunk_frames,
            "adaptive_chunks": args.adaptive_chunks,
            "max_chunk_frames": args.max_chunk_frames,
            "chunk_growth_frames": args.chunk_growth_frames,
            "max_audio_length": args.max_audio_length,
            "min_audio_length": args.min_audio_length,
            "talker_temperature": args.talker_temperature,
            "talker_top_k": args.talker_top_k,
            "talker_top_p": args.talker_top_p,
            "predictor_temperature": args.predictor_temperature,
            "predictor_top_k": args.predictor_top_k,
            "predictor_top_p": args.predictor_top_p,
            "repetition_penalty": args.repetition_penalty,
            "codec_eos_logit_offset": 0.0,
        }
        start = time.time()
        proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        proc.stdin.flush()

        chunks = 0
        pcm_bytes = 0
        pcm_parts: list[bytes] = []
        first_chunk_s = None
        while True:
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("worker exited during request: " + "".join(stderr_lines)[-2000:])
            event = json.loads(line)
            if args.print_chunk_meta and event.get("event") == "chunk":
                meta = dict(event)
                if "audio_b64" in meta:
                    meta["audio_b64_bytes"] = len(meta.pop("audio_b64"))
                print(json.dumps(meta, ensure_ascii=False), flush=True)
            elif not (args.quiet_chunks and event.get("event") == "chunk"):
                print(line.rstrip(), flush=True)
            if not event.get("ok", False):
                print("stderr_tail_begin", flush=True)
                print("".join(stderr_lines)[-4000:], flush=True)
                print("stderr_tail_end", flush=True)
                raise RuntimeError(f"worker error: {event}")
            if event.get("event") == "chunk":
                chunks += 1
                if first_chunk_s is None:
                    first_chunk_s = time.time() - start
                chunk = base64.b64decode(event.get("audio_b64", ""))
                pcm_parts.append(chunk)
                pcm_bytes += len(chunk)
            if event.get("event") == "done":
                break

        pcm = b"".join(pcm_parts)
        output_pcm = args.output_pcm
        output_wav = args.output_wav
        if total_runs > 1:
            if output_pcm:
                root, ext = os.path.splitext(output_pcm)
                output_pcm = f"{root}.{req_id}{ext or '.pcm'}"
            if output_wav:
                root, ext = os.path.splitext(output_wav)
                output_wav = f"{root}.{req_id}{ext or '.wav'}"
        if output_pcm and not is_warmup:
            with open(output_pcm, "wb") as f:
                f.write(pcm)
        if output_wav and not is_warmup:
            with wave.open(output_wav, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(pcm)

        summary = {
            "summary": "ok",
            "id": req_id,
            "warmup": is_warmup,
            "chunks": chunks,
            "pcm_bytes": pcm_bytes,
            "first_chunk_wall_s": first_chunk_s,
            "total_wall_s": time.time() - start,
            "output_pcm": "" if is_warmup else output_pcm,
            "output_wav": "" if is_warmup else output_wav,
        }
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    if len(summaries) > 1:
        print(json.dumps({"summary": "all", "runs": summaries}, ensure_ascii=False), flush=True)
    if args.print_stderr and stderr_lines:
        print("stderr_begin", flush=True)
        print("".join(stderr_lines), flush=True)
        print("stderr_end", flush=True)
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


if __name__ == "__main__":
    main()
