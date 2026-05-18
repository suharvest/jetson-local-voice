"""F5-TTS ONNX optimization sweep on Jetson Orin NX.

Tests: TensorRT, CUDA+TF32, CUDA Graph, graph optimization levels.
"""

import re
import time
import os
import numpy as np
import onnxruntime
import soundfile as sf

MODEL_DIR = os.environ.get("F5_MODEL_DIR", "/opt/models/f5-tts-onnx")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.txt")
ONNX_A = os.path.join(MODEL_DIR, "F5_Preprocess.onnx")
ONNX_B = os.path.join(MODEL_DIR, "F5_Transformer.onnx")
ONNX_C = os.path.join(MODEL_DIR, "F5_Decode.onnx")

RANDOM_SEED = 9527
NFE_STEP = 32
SPEED = 1.0
MAX_THREADS = 4
DEVICE_ID = 0
MODEL_SAMPLE_RATE = 24000
HOP_LENGTH = 256

REF_AUDIO = os.environ.get("F5_REF_AUDIO", "/opt/models/f5-tts-onnx/basic_ref_zh.wav")
REF_TEXT = "对，这就是我，万人敬仰的太乙真人。"
GEN_TEXT = "你好，我是你的智能助手，很高兴认识你。"


def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        return {char.rstrip("\n"): i for i, char in enumerate(f)}


def convert_char_to_pinyin(text_list, polyphone=True):
    import jieba
    from pypinyin import lazy_pinyin, Style
    if not jieba.dt.initialized:
        jieba.default_logger.setLevel(50)
        jieba.initialize()
    custom_trans = str.maketrans({";": ",", "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"})
    def is_chinese(c):
        return "\u3100" <= c <= "\u9fff"
    final = []
    for text in text_list:
        chars = []
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(seg.encode("utf-8"))
            if seg_byte_len == len(seg):
                if chars and seg_byte_len > 1 and chars[-1] not in " :'\"":
                    chars.append(" ")
                chars.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c):
                        chars.append(" ")
                        chars.append(seg_[i])
            else:
                for c in seg:
                    if ord(c) < 256:
                        chars.extend(c)
                    elif is_chinese(c):
                        chars.append(" ")
                        chars.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                    else:
                        chars.append(c)
        final.append(chars)
    return final


def list_str_to_idx_numpy(text, vocab_char_map, padding_value=-1):
    idx_lists = [[vocab_char_map.get(c, 0) for c in t] for t in text]
    max_len = max(len(t) for t in idx_lists)
    result = np.full((len(idx_lists), max_len), padding_value, dtype=np.int32)
    for i, idx in enumerate(idx_lists):
        result[i, : len(idx)] = idx
    return result


def prepare_inputs(vocab):
    from pydub import AudioSegment
    audio_seg = AudioSegment.from_file(REF_AUDIO).set_channels(1).set_frame_rate(MODEL_SAMPLE_RATE)
    audio = np.array(audio_seg.get_array_of_samples(), dtype=np.float32)
    max_val = np.max(np.abs(audio))
    audio = (audio * (32767.0 / max_val) if max_val > 0 else audio).astype(np.int16)
    audio_len = len(audio)
    audio = audio.reshape(1, 1, -1)

    zh_pause_punc = r"。，、；：？！"
    ref_text_len = len(REF_TEXT.encode("utf-8")) + 3 * len(re.findall(zh_pause_punc, REF_TEXT))
    gen_text_len = len(GEN_TEXT.encode("utf-8")) + 3 * len(re.findall(zh_pause_punc, GEN_TEXT))
    ref_audio_len = audio_len // HOP_LENGTH + 1
    max_duration = np.array([ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / SPEED)], dtype=np.int64)

    gen_text_pinyin = convert_char_to_pinyin([REF_TEXT + GEN_TEXT])
    text_ids = list_str_to_idx_numpy(gen_text_pinyin, vocab)
    time_step = np.array([0], dtype=np.int32)

    print(f"  audio: {audio.shape}, text_ids: {text_ids.shape}, max_dur: {max_duration}")
    return audio, text_ids, max_duration, time_step


def run_benchmark(label, sess_A, sess_B, sess_C, audio, text_ids, max_duration, time_step, nfe=32):
    """Run full pipeline and return timing dict."""
    in_A = [i.name for i in sess_A.get_inputs()]
    out_A = [o.name for o in sess_A.get_outputs()]
    in_B = [i.name for i in sess_B.get_inputs()]
    out_B = [o.name for o in sess_B.get_outputs()]
    in_C = [i.name for i in sess_C.get_inputs()]
    out_C = [o.name for o in sess_C.get_outputs()]

    total_start = time.time()

    # Stage A
    t0 = time.time()
    a_out = sess_A.run(out_A, {in_A[0]: audio, in_A[1]: text_ids, in_A[2]: max_duration})
    noise = a_out[0]
    ref_signal_len = a_out[7]
    t_A = time.time() - t0

    # Stage B with IO binding
    t0 = time.time()
    ts = time_step.copy()
    inputs_ort = [
        onnxruntime.OrtValue.ortvalue_from_numpy(noise, "cuda", DEVICE_ID),
        onnxruntime.OrtValue.ortvalue_from_numpy(a_out[1], "cuda", DEVICE_ID),
        onnxruntime.OrtValue.ortvalue_from_numpy(a_out[2], "cuda", DEVICE_ID),
        onnxruntime.OrtValue.ortvalue_from_numpy(a_out[3], "cuda", DEVICE_ID),
        onnxruntime.OrtValue.ortvalue_from_numpy(a_out[4], "cuda", DEVICE_ID),
        onnxruntime.OrtValue.ortvalue_from_numpy(a_out[5], "cuda", DEVICE_ID),
        onnxruntime.OrtValue.ortvalue_from_numpy(a_out[6], "cuda", DEVICE_ID),
        onnxruntime.OrtValue.ortvalue_from_numpy(ts, "cuda", DEVICE_ID),
    ]
    iob = sess_B.io_binding()
    for i in range(8):
        iob.bind_ortvalue_input(in_B[i], inputs_ort[i])
    iob.bind_ortvalue_output(out_B[0], inputs_ort[0])
    iob.bind_ortvalue_output(out_B[1], inputs_ort[7])

    step_times = []
    for step in range(nfe):
        t_step = time.time()
        sess_B.run_with_iobinding(iob)
        step_times.append(time.time() - t_step)

    noise_out = onnxruntime.OrtValue.numpy(iob.get_outputs()[0])
    t_B = time.time() - t0

    # Stage C
    t0 = time.time()
    generated = sess_C.run(out_C, {in_C[0]: noise_out, in_C[1]: ref_signal_len})[0]
    t_C = time.time() - t0

    total = time.time() - total_start
    gen_dur = generated.reshape(-1).shape[0] / MODEL_SAMPLE_RATE

    return {
        "label": label,
        "nfe": nfe,
        "t_A": t_A,
        "t_B": t_B,
        "t_C": t_C,
        "total": total,
        "gen_dur": gen_dur,
        "rtf": total / gen_dur,
        "per_step_avg": np.mean(step_times),
        "per_step_min": np.min(step_times),
        "per_step_max": np.max(step_times),
        "step_times": step_times,
    }


def make_session(model_path, provider, opts_override=None):
    sess_opts = onnxruntime.SessionOptions()
    sess_opts.log_severity_level = 4
    sess_opts.inter_op_num_threads = MAX_THREADS
    sess_opts.intra_op_num_threads = MAX_THREADS
    sess_opts.enable_cpu_mem_arena = True
    sess_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    sess_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    if opts_override:
        for k, v in opts_override.items():
            setattr(sess_opts, k, v)

    if provider == "TensorrtExecutionProvider":
        provider_options = [{
            "device_id": DEVICE_ID,
            "trt_max_workspace_size": str(4 * 1024 * 1024 * 1024),
            "trt_fp16_enable": "1",
            "trt_engine_cache_enable": "1",
            "trt_engine_cache_path": "/tmp/trt_cache",
        }]
        providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
    elif provider == "CUDAExecutionProvider":
        provider_options = [{
            "device_id": DEVICE_ID,
            "gpu_mem_limit": str(8 * 1024 * 1024 * 1024),
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "cudnn_conv_use_max_workspace": "1",
            "do_copy_in_default_stream": "1",
            "cudnn_conv1d_pad_to_nc1d": "1",
            "enable_cuda_graph": "0",
            "use_tf32": "0",
        }]
        providers = ["CUDAExecutionProvider"]
    elif provider == "CUDA+TF32":
        provider_options = [{
            "device_id": DEVICE_ID,
            "gpu_mem_limit": str(8 * 1024 * 1024 * 1024),
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "cudnn_conv_use_max_workspace": "1",
            "do_copy_in_default_stream": "1",
            "cudnn_conv1d_pad_to_nc1d": "1",
            "enable_cuda_graph": "0",
            "use_tf32": "1",
        }]
        providers = ["CUDAExecutionProvider"]
    elif provider == "CUDA+TF32+CUDAGraph":
        provider_options = [{
            "device_id": DEVICE_ID,
            "gpu_mem_limit": str(8 * 1024 * 1024 * 1024),
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "cudnn_conv_use_max_workspace": "1",
            "do_copy_in_default_stream": "1",
            "cudnn_conv1d_pad_to_nc1d": "1",
            "enable_cuda_graph": "1",
            "use_tf32": "1",
        }]
        providers = ["CUDAExecutionProvider"]
    else:
        provider_options = None
        providers = [provider]

    return onnxruntime.InferenceSession(
        model_path, sess_options=sess_opts,
        providers=providers, provider_options=provider_options,
    )


def main():
    onnxruntime.set_seed(RANDOM_SEED)
    print("=" * 70)
    print("F5-TTS ONNX Optimization Sweep — Jetson Orin NX")
    print("=" * 70)

    vocab = load_vocab(VOCAB_PATH)
    audio, text_ids, max_duration, time_step = prepare_inputs(vocab)

    # ── Configs to test ──
    configs = [
        {
            "label": "1. CUDA (baseline)",
            "provider_B": "CUDAExecutionProvider",
        },
        {
            "label": "2. CUDA + TF32",
            "provider_B": "CUDA+TF32",
        },
        {
            "label": "3. CUDA + TF32 + CUDAGraph",
            "provider_B": "CUDA+TF32+CUDAGraph",
        },
        {
            "label": "4. TensorRT (FP16)",
            "provider_B": "TensorrtExecutionProvider",
        },
    ]

    results = []
    for cfg in configs:
        label = cfg["label"]
        provider_B = cfg["provider_B"]
        print(f"\n{'='*70}")
        print(f"Config: {label}")
        print(f"{'='*70}")

        try:
            # A and C always on CUDA (small models, not the bottleneck)
            print("  Loading models...")
            t0 = time.time()
            sess_A = make_session(ONNX_A, "CUDAExecutionProvider")
            sess_B = make_session(ONNX_B, provider_B)
            sess_C = make_session(ONNX_C, "CUDAExecutionProvider")
            load_time = time.time() - t0
            print(f"  Models loaded in {load_time:.1f}s")
            print(f"  Transformer provider: {sess_B.get_providers()[0]}")

            # Warmup run
            print("  Warmup run...")
            run_benchmark("warmup", sess_A, sess_B, sess_C, audio, text_ids, max_duration, time_step, nfe=2)

            # Actual benchmark
            print("  Benchmark runs...")
            for nfe in [8, 16, 32]:
                r = run_benchmark(label, sess_A, sess_B, sess_C, audio, text_ids, max_duration, time_step, nfe=nfe)
                results.append(r)
                print(f"    NFE={nfe:2d}: total={r['total']*1000:.0f}ms, "
                      f"transformer={r['t_B']*1000:.0f}ms ({r['per_step_avg']*1000:.0f}ms/step), "
                      f"RTF={r['rtf']:.3f}, audio={r['gen_dur']:.1f}s")

            del sess_A, sess_B, sess_C
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # ── Summary ──
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<35} {'NFE':>3} {'Total':>8} {'Trans':>8} {'Step':>8} {'RTF':>6} {'Audio':>6}")
    print("-" * 80)
    for r in results:
        print(f"{r['label']:<35} {r['nfe']:>3} {r['total']*1000:>7.0f}ms {r['t_B']*1000:>7.0f}ms "
              f"{r['per_step_avg']*1000:>7.1f}ms {r['rtf']:>6.3f} {r['gen_dur']:>5.1f}s")


if __name__ == "__main__":
    main()
