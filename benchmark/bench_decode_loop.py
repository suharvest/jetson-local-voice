#!/usr/bin/env python3
"""
Benchmark: Qwen3-TTS 0.6B INT8 decode loop performance.
Tests each component SEQUENTIALLY to avoid OOM on 15GB Jetson.
"""

import time
import json
import gc
import numpy as np
import os

import onnxruntime as ort

MODEL_DIR = "/tmp/qwen3-tts-bench/model-int8"
HIDDEN_SIZE = 1024
NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
NUM_CODE_GROUPS = 16
CODEC_HZ = 12.5
WARMUP = 3
RUNS = 10


def get_mem_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0


def create_session(name, providers):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    path = f"{MODEL_DIR}/{name}"
    return ort.InferenceSession(path, sess_options=opts, providers=providers)


def stats(times):
    arr = np.array(times) * 1000
    return {"mean": round(float(np.mean(arr)), 1),
            "std": round(float(np.std(arr)), 1),
            "p50": round(float(np.median(arr)), 1),
            "p95": round(float(np.percentile(arr, 95)), 1),
            "min": round(float(np.min(arr)), 1)}


def bench(label, session, feed, warmup=WARMUP, runs=RUNS):
    for _ in range(warmup):
        session.run(None, feed)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = session.run(None, feed)
        times.append(time.perf_counter() - t0)
    s = stats(times)
    print(f"  {label}: {s['mean']:.1f}ms ± {s['std']:.1f} (p50={s['p50']:.1f}, p95={s['p95']:.1f})")
    return s, out


def main():
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        providers = ["CPUExecutionProvider"]

    print("=" * 65)
    print("Qwen3-TTS 0.6B INT8 — Streaming Decode Benchmark (sequential)")
    print("=" * 65)
    print(f"ORT {ort.__version__} | Provider: {providers[0]} | RSS: {get_mem_mb():.0f}MB")
    print()

    results = {}

    # === 1. TEXT PROJECTION ===
    print("[1/5] TEXT PROJECTION")
    sess = create_session("text_project_q.onnx", providers)
    for n in [20, 50, 100]:
        ids = np.random.randint(0, 150000, (1, n), dtype=np.int64)
        s, _ = bench(f"seq={n}", sess, {"input_ids": ids})
        results[f"text_proj_{n}"] = s
    del sess; gc.collect()
    print(f"  [mem] {get_mem_mb():.0f}MB\n")

    # === 2. PREFILL (TTFA) ===
    print("[2/5] TALKER PREFILL (Time-to-First-Audio)")
    sess = create_session("talker_prefill_q.onnx", providers)
    for n in [20, 50, 100]:
        embeds = np.random.randn(1, n, HIDDEN_SIZE).astype(np.float32)
        mask = np.ones((1, n), dtype=np.int64)
        s, out = bench(f"seq={n}", sess, {"inputs_embeds": embeds, "attention_mask": mask})
        results[f"prefill_{n}"] = s

    # Keep KV cache from seq=50 for decode test
    output_names = [o.name for o in sess.get_outputs()]
    kv50 = {}
    embeds50 = np.random.randn(1, 50, HIDDEN_SIZE).astype(np.float32)
    mask50 = np.ones((1, 50), dtype=np.int64)
    out50 = sess.run(None, {"inputs_embeds": embeds50, "attention_mask": mask50})
    for i, name in enumerate(output_names):
        if "present_key" in name or "present_value" in name:
            kv50[name] = out50[i]
    del sess, out, out50; gc.collect()
    print(f"  [mem] {get_mem_mb():.0f}MB\n")

    # === 3. DECODE STEP ===
    print("[3/5] TALKER DECODE (per-step, need < 80ms for real-time)")
    sess = create_session("talker_decode_q.onnx", providers)
    input_names = [i.name for i in sess.get_inputs()]

    for past_len in [50, 100, 200]:
        embeds = np.random.randn(1, 1, HIDDEN_SIZE).astype(np.float32)
        mask = np.ones((1, past_len + 1), dtype=np.int64)
        feed = {"inputs_embeds": embeds, "attention_mask": mask}
        for name in input_names:
            if name.startswith("past_key_"):
                idx = name.replace("past_key_", "")
                pk = f"present_key_{idx}"
                if pk in kv50 and past_len == 50:
                    feed[name] = kv50[pk]
                else:
                    feed[name] = np.zeros((1, NUM_KV_HEADS, past_len, HEAD_DIM), dtype=np.float32)
            elif name.startswith("past_value_"):
                idx = name.replace("past_value_", "")
                pv = f"present_value_{idx}"
                if pv in kv50 and past_len == 50:
                    feed[name] = kv50[pv]
                else:
                    feed[name] = np.zeros((1, NUM_KV_HEADS, past_len, HEAD_DIM), dtype=np.float32)

        s, _ = bench(f"past={past_len}", sess, feed)
        rt = "✅" if s['mean'] < 80 else "❌"
        print(f"    {rt} {'real-time OK' if s['mean'] < 80 else 'TOO SLOW'}")
        results[f"decode_{past_len}"] = s

    del sess, kv50; gc.collect()
    print(f"  [mem] {get_mem_mb():.0f}MB\n")

    # === 4. CODE PREDICTOR (15 sub-codes) ===
    print("[4/5] CODE PREDICTOR (15 sub-codes per talker step)")
    sess = create_session("code_predictor_q.onnx", providers)
    embeds = np.random.randn(1, 1, HIDDEN_SIZE).astype(np.float32)

    # Bench 15-step loop
    for _ in range(WARMUP):
        for step in range(15):
            sess.run(None, {"inputs_embeds": embeds, "generation_step": np.array([step], dtype=np.int64)})
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        for step in range(15):
            sess.run(None, {"inputs_embeds": embeds, "generation_step": np.array([step], dtype=np.int64)})
        times.append(time.perf_counter() - t0)
    s = stats(times)
    print(f"  15 steps: {s['mean']:.1f}ms ± {s['std']:.1f} (per step: {s['mean']/15:.1f}ms)")
    results["code_pred_15"] = s

    # Also bench single step
    for _ in range(WARMUP):
        sess.run(None, {"inputs_embeds": embeds, "generation_step": np.array([0], dtype=np.int64)})
    t1 = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        sess.run(None, {"inputs_embeds": embeds, "generation_step": np.array([0], dtype=np.int64)})
        t1.append(time.perf_counter() - t0)
    s1 = stats(t1)
    print(f"  1 step:   {s1['mean']:.1f}ms")
    results["code_pred_1"] = s1

    del sess; gc.collect()
    print(f"  [mem] {get_mem_mb():.0f}MB\n")

    # === 5. VOCODER ===
    print("[5/5] VOCODER (codes → audio)")
    sess = create_session("tokenizer12hz_decode_q.onnx", providers)
    for nf in [10, 25, 50]:
        codes = np.random.randint(0, 2048, (1, nf, NUM_CODE_GROUPS), dtype=np.int64)
        s, out = bench(f"frames={nf} ({nf/CODEC_HZ:.1f}s)", sess, {"audio_codes": codes})
        audio_dur = nf / CODEC_HZ
        rtf = s['mean'] / 1000 / audio_dur
        print(f"    RTF={rtf:.4f}")
        results[f"vocoder_{nf}"] = {**s, "audio_s": audio_dur, "rtf": round(rtf, 4)}
    del sess; gc.collect()
    print(f"  [mem] {get_mem_mb():.0f}MB\n")

    # === SUMMARY ===
    print("=" * 65)
    print("STREAMING PERFORMANCE ESTIMATE")
    print("=" * 65)

    prefill = results.get("prefill_50", {}).get("mean", 0)
    decode = results.get("decode_100", {}).get("mean", 0)
    code_pred = results.get("code_pred_15", {}).get("mean", 0)
    text_proj = results.get("text_proj_50", {}).get("mean", 0)
    vocoder = results.get("vocoder_25", {}).get("mean", 0)

    per_step = decode + code_pred
    audio_per_step = 1000 / CODEC_HZ  # 80ms

    print(f"\nTypical sentence (~50 tokens → ~25 frames → ~2s audio):")
    print(f"  Text projection:    {text_proj:.0f}ms")
    print(f"  Prefill (TTFA):     {prefill:.0f}ms")
    print(f"  Decode/step:        {decode:.0f}ms (talker) + {code_pred:.0f}ms (code_pred) = {per_step:.0f}ms")
    print(f"  Audio/step:         {audio_per_step:.0f}ms")
    print(f"  Vocoder (25 fr):    {vocoder:.0f}ms")

    decode_rtf = per_step / audio_per_step
    if decode_rtf < 1.0:
        print(f"\n  ✅ REAL-TIME STREAMING OK — decode RTF = {decode_rtf:.2f}x")
    else:
        print(f"\n  ❌ TOO SLOW — decode RTF = {decode_rtf:.2f}x (need < 1.0)")

    ttfa = text_proj + prefill + decode + code_pred
    print(f"\n  Estimated TTFA:     ~{ttfa:.0f}ms")
    print(f"  Current Matcha:     ~60ms (zh_en) / ~130ms (en)")
    print(f"  Ratio vs Matcha:    {ttfa/60:.1f}x (zh) / {ttfa/130:.1f}x (en)")

    batch_2s = text_proj + prefill + per_step * 25 + vocoder
    print(f"\n  Batch 2s audio:     ~{batch_2s:.0f}ms")
    print(f"  Current Matcha 2s:  ~150ms")

    print(f"\n{'='*65}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
