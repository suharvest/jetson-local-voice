#!/usr/bin/env python3
"""
Benchmark: Qwen3-TTS 0.6B via ONNX Runtime (no PyTorch)
Runs inside the existing jetson-voice container.
"""

import os
import sys
import time
import json
import subprocess
import numpy as np

WORK_DIR = "/tmp/qwen3-tts-bench"
MODEL_REPO = "elbruno/Qwen3-TTS-12Hz-0.6B-Base-ONNX"
SAMPLE_RATE = 24000

# Test sentences
TEST_SENTENCES = [
    ("zh", "你好，欢迎使用语音合成系统。今天天气真不错。"),
    ("en", "Hello, welcome to the voice synthesis system. The weather is nice today."),
    ("zh", "这是一个关于人工智能的测试，我们正在验证语音合成的性能表现。"),
]


def install_deps():
    """Install missing Python dependencies."""
    deps = ["tokenizers", "librosa", "huggingface_hub"]
    for dep in deps:
        try:
            __import__(dep)
        except ImportError:
            print(f"[setup] Installing {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "-q"])


def download_model():
    """Download model files from HuggingFace."""
    from huggingface_hub import snapshot_download

    model_dir = os.path.join(WORK_DIR, "model")
    if os.path.exists(model_dir) and len(os.listdir(model_dir)) > 5:
        print(f"[setup] Model already downloaded at {model_dir}")
        return model_dir

    print(f"[setup] Downloading {MODEL_REPO}...")
    model_dir = snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=model_dir,
        local_dir_use_symlinks=False,
    )
    print(f"[setup] Model downloaded to {model_dir}")
    return model_dir


def find_onnx_files(model_dir):
    """Find and catalog all ONNX files."""
    onnx_files = {}
    for root, dirs, files in os.walk(model_dir):
        for f in files:
            if f.endswith(".onnx"):
                rel = os.path.relpath(os.path.join(root, f), model_dir)
                onnx_files[f] = os.path.join(root, f)
                size_mb = os.path.getsize(os.path.join(root, f)) / 1024 / 1024
                print(f"  {rel}: {size_mb:.1f} MB")
    return onnx_files


def check_onnx_model_info(onnx_files):
    """Print input/output info for each ONNX model."""
    import onnxruntime as ort

    print("\n[info] ONNX model I/O signatures:")
    for name, path in sorted(onnx_files.items()):
        try:
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            inputs = [(i.name, i.shape, i.type) for i in sess.get_inputs()]
            outputs = [(o.name, o.shape, o.type) for o in sess.get_outputs()]
            print(f"\n  {name}:")
            for inp in inputs:
                print(f"    IN:  {inp[0]} {inp[1]} ({inp[2]})")
            for out in outputs:
                print(f"    OUT: {out[0]} {out[1]} ({out[2]})")
            del sess
        except Exception as e:
            print(f"  {name}: ERROR loading - {e}")


def try_basic_synthesis(model_dir, onnx_files, provider="CUDAExecutionProvider"):
    """Attempt basic synthesis using the ONNX pipeline."""
    import onnxruntime as ort

    providers = [provider, "CPUExecutionProvider"]
    print(f"\n[bench] Provider: {provider}")

    # Check if there's an inference script in the repo
    for script_name in ["synthesize.py", "inference.py", "run.py", "tts.py"]:
        script_path = os.path.join(model_dir, script_name)
        if os.path.exists(script_path):
            print(f"[info] Found inference script: {script_name}")
            # Try to run it
            try:
                result = subprocess.run(
                    [sys.executable, script_path, "--help"],
                    capture_output=True, text=True, cwd=model_dir, timeout=30
                )
                print(f"  Help output: {result.stdout[:500]}")
            except Exception as e:
                print(f"  Could not run: {e}")

    # Load key models and measure load time
    results = {}

    # 1. Text embedding model
    text_embed_path = None
    for name, path in onnx_files.items():
        if "text_embed" in name.lower():
            text_embed_path = path
            break

    if text_embed_path:
        t0 = time.time()
        text_embed_sess = ort.InferenceSession(text_embed_path, providers=providers)
        load_time = time.time() - t0
        actual_provider = text_embed_sess.get_providers()[0]
        print(f"  text_embedding loaded in {load_time:.2f}s (provider: {actual_provider})")
        results["text_embed_load"] = load_time

        # Quick inference test
        test_ids = np.array([[151644, 77091, 198]], dtype=np.int64)  # <|im_start|>assistant\n
        t0 = time.time()
        out = text_embed_sess.run(None, {"input_ids": test_ids})
        inf_time = time.time() - t0
        print(f"  text_embedding inference: {inf_time*1000:.1f}ms, output shape: {out[0].shape}")
        results["text_embed_inf"] = inf_time
        del text_embed_sess

    # 2. Speech tokenizer decoder (vocoder)
    decoder_path = None
    for name, path in onnx_files.items():
        if "speech_tokenizer_decoder" in name.lower() or "decoder" in name.lower():
            decoder_path = path
            break

    if decoder_path:
        t0 = time.time()
        decoder_sess = ort.InferenceSession(decoder_path, providers=providers)
        load_time = time.time() - t0
        actual_provider = decoder_sess.get_providers()[0]
        print(f"  decoder loaded in {load_time:.2f}s (provider: {actual_provider})")
        results["decoder_load"] = load_time

        # Quick decode test with random codes
        test_codes = np.random.randint(0, 1024, size=(1, 50, 16), dtype=np.int64)
        input_name = decoder_sess.get_inputs()[0].name
        t0 = time.time()
        try:
            audio_out = decoder_sess.run(None, {input_name: test_codes})
            inf_time = time.time() - t0
            audio_shape = audio_out[0].shape
            duration = audio_shape[-1] / SAMPLE_RATE
            print(f"  decoder inference: {inf_time*1000:.1f}ms, output shape: {audio_shape}")
            print(f"  decoded audio: {duration:.2f}s @ {SAMPLE_RATE}Hz")
            results["decoder_inf"] = inf_time
            results["decoder_audio_dur"] = duration
        except Exception as e:
            print(f"  decoder inference failed: {e}")
        del decoder_sess

    # 3. Talker model (main autoregressive model)
    talker_path = None
    for name, path in onnx_files.items():
        if "talker" in name.lower():
            talker_path = path
            break

    if talker_path:
        size_mb = os.path.getsize(talker_path) / 1024 / 1024
        print(f"\n  talker model: {size_mb:.1f} MB")
        t0 = time.time()
        try:
            talker_sess = ort.InferenceSession(talker_path, providers=providers)
            load_time = time.time() - t0
            actual_provider = talker_sess.get_providers()[0]
            print(f"  talker loaded in {load_time:.2f}s (provider: {actual_provider})")
            results["talker_load"] = load_time
            results["talker_size_mb"] = size_mb

            # Show inputs
            for inp in talker_sess.get_inputs():
                print(f"    IN: {inp.name} {inp.shape}")

            del talker_sess
        except Exception as e:
            print(f"  talker load failed: {e}")
            results["talker_error"] = str(e)

    # 4. Speaker encoder
    spk_path = None
    for name, path in onnx_files.items():
        if "speaker_encoder" in name.lower():
            spk_path = path
            break

    if spk_path:
        t0 = time.time()
        spk_sess = ort.InferenceSession(spk_path, providers=providers)
        load_time = time.time() - t0
        print(f"  speaker_encoder loaded in {load_time:.2f}s")
        results["spk_encoder_load"] = load_time
        del spk_sess

    return results


def check_repo_scripts(model_dir):
    """Check for any inference scripts or READMEs in the downloaded model."""
    print("\n[info] Repository contents:")
    for item in sorted(os.listdir(model_dir)):
        path = os.path.join(model_dir, item)
        if os.path.isfile(path):
            size = os.path.getsize(path)
            if size < 1024 * 1024:  # < 1MB, likely a script/config
                print(f"  {item} ({size} bytes)")
        elif os.path.isdir(path):
            n_files = len(os.listdir(path))
            print(f"  {item}/ ({n_files} files)")


def measure_memory():
    """Measure current GPU/system memory usage."""
    try:
        # Jetson uses tegrastats or /sys for memory
        with open("/sys/kernel/debug/nvmap/iovmm/clients", "r") as f:
            content = f.read()
            print(f"\n[mem] GPU memory clients (first 500 chars):\n{content[:500]}")
    except:
        pass

    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"[mem] System RAM: {mem.used/1024**3:.1f}/{mem.total/1024**3:.1f} GB ({mem.percent}%)")
    except ImportError:
        # fallback
        with open("/proc/meminfo") as f:
            lines = f.readlines()[:3]
            for l in lines:
                print(f"[mem] {l.strip()}")


def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    print("=" * 60)
    print("Qwen3-TTS 0.6B ONNX Benchmark")
    print("=" * 60)

    # Step 1: Install deps
    install_deps()

    import onnxruntime as ort
    print(f"\n[env] ONNX Runtime: {ort.__version__}")
    print(f"[env] Providers: {ort.get_available_providers()}")
    print(f"[env] Python: {sys.version}")

    measure_memory()

    # Step 2: Download model
    model_dir = download_model()
    check_repo_scripts(model_dir)

    # Step 3: Catalog ONNX files
    print("\n[info] ONNX model files:")
    onnx_files = find_onnx_files(model_dir)

    if not onnx_files:
        print("[ERROR] No ONNX files found! Checking subdirectories...")
        for d in os.listdir(model_dir):
            subdir = os.path.join(model_dir, d)
            if os.path.isdir(subdir):
                print(f"  {d}/: {os.listdir(subdir)[:10]}")
        return

    # Step 4: Check model info
    check_onnx_model_info(onnx_files)

    # Step 5: Benchmark with CUDA
    print("\n" + "=" * 60)
    print("CUDA Benchmark")
    print("=" * 60)
    cuda_results = {}
    if "CUDAExecutionProvider" in ort.get_available_providers():
        cuda_results = try_basic_synthesis(model_dir, onnx_files, "CUDAExecutionProvider")
    else:
        print("[WARN] CUDA not available, skipping")

    # Step 6: Benchmark with CPU
    print("\n" + "=" * 60)
    print("CPU Benchmark")
    print("=" * 60)
    cpu_results = try_basic_synthesis(model_dir, onnx_files, "CPUExecutionProvider")

    # Step 7: Memory after loading
    measure_memory()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(json.dumps({"cuda": cuda_results, "cpu": cpu_results}, indent=2, default=str))
    print("\n[NOTE] This benchmark tests model loading and component inference.")
    print("Full end-to-end TTS requires the autoregressive pipeline (tokenize -> embed -> generate -> decode).")
    print(f"Model directory: {model_dir}")


if __name__ == "__main__":
    main()
