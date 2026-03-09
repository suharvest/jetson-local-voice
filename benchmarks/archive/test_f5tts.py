"""F5-TTS ONNX benchmark on Jetson Orin NX.

No torch dependency — uses numpy for all tensor ops.
Three-stage pipeline: Preprocess (CPU) -> Transformer (CUDA, 32 NFE steps) -> Decode (CPU)
"""

import re
import time
import sys
import os
import numpy as np
import onnxruntime
import soundfile as sf

# ── Paths ──────────────────────────────────────────────────────────────
MODEL_DIR = os.environ.get("F5_MODEL_DIR", "/opt/models/f5-tts-onnx")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.txt")
ONNX_A = os.path.join(MODEL_DIR, "F5_Preprocess.onnx")
ONNX_B = os.path.join(MODEL_DIR, "F5_Transformer.onnx")
ONNX_C = os.path.join(MODEL_DIR, "F5_Decode.onnx")
OUTPUT_WAV = "/tmp/f5tts_output.wav"

# ── Config ─────────────────────────────────────────────────────────────
RANDOM_SEED = 9527
NFE_STEP = 32          # flow matching steps (fewer = faster but lower quality)
FUSE_NFE = 1
SPEED = 1.0
MAX_THREADS = 4
DEVICE_ID = 0
MODEL_SAMPLE_RATE = 24000
HOP_LENGTH = 256

# ── Reference audio + text ─────────────────────────────────────────────
# Use the bundled Chinese reference from f5_tts package, or provide your own
REF_AUDIO = os.environ.get("F5_REF_AUDIO", "")
REF_TEXT = os.environ.get("F5_REF_TEXT", "对，这就是我，万人敬仰的太乙真人。")
GEN_TEXT = os.environ.get("F5_GEN_TEXT", "你好，我是你的智能助手，很高兴认识你。今天天气真不错，我们聊聊吧。")


# ── Vocab ──────────────────────────────────────────────────────────────
def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        return {char.rstrip("\n"): i for i, char in enumerate(f)}


# ── Chinese text to pinyin (replaces torch-dependent original) ─────────
def convert_char_to_pinyin(text_list, polyphone=True):
    import jieba
    from pypinyin import lazy_pinyin, Style

    if not jieba.dt.initialized:
        jieba.default_logger.setLevel(50)
        jieba.initialize()

    custom_trans = str.maketrans(
        {";": ",", "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"}
    )

    def is_chinese(c):
        return "\u3100" <= c <= "\u9fff"

    final = []
    for text in text_list:
        chars = []
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(seg.encode("utf-8"))
            if seg_byte_len == len(seg):
                # ASCII segment
                if chars and seg_byte_len > 1 and chars[-1] not in " :'\"":
                    chars.append(" ")
                chars.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):
                # Pure Chinese segment
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
                        chars.extend(
                            lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True)
                        )
                    else:
                        chars.append(c)
        final.append(chars)
    return final


def list_str_to_idx_numpy(text, vocab_char_map, padding_value=-1):
    """Convert list of char lists to padded numpy int32 array (replaces torch version)."""
    idx_lists = [[vocab_char_map.get(c, 0) for c in t] for t in text]
    max_len = max(len(t) for t in idx_lists)
    result = np.full((len(idx_lists), max_len), padding_value, dtype=np.int32)
    for i, idx in enumerate(idx_lists):
        result[i, : len(idx)] = idx
    return result


def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    scale = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * scale).astype(np.int16)


def find_ref_audio():
    """Find a usable reference audio file."""
    if REF_AUDIO and os.path.exists(REF_AUDIO):
        return REF_AUDIO
    # Try f5_tts package bundled examples
    candidates = [
        "/usr/local/lib/python3.10/dist-packages/f5_tts/infer/examples/basic/basic_ref_zh.wav",
        "/opt/models/f5-tts-onnx/basic_ref_zh.wav",
        "/tmp/basic_ref_zh.wav",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def create_sine_ref_audio(path="/tmp/f5tts_ref.wav", duration=3.0):
    """Create a simple sine wave as fallback reference audio."""
    t = np.linspace(0, duration, int(MODEL_SAMPLE_RATE * duration), dtype=np.float32)
    audio = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    sf.write(path, audio, MODEL_SAMPLE_RATE)
    print(f"[WARN] No reference audio found. Created sine wave at {path}")
    return path


def main():
    print("=" * 60)
    print("F5-TTS ONNX Benchmark — Jetson Orin NX")
    print("=" * 60)

    # ── Check files ──
    for f in [VOCAB_PATH, ONNX_A, ONNX_B, ONNX_C]:
        if not os.path.exists(f):
            print(f"[ERROR] Missing: {f}")
            sys.exit(1)

    # ── Load vocab ──
    t0 = time.time()
    vocab = load_vocab(VOCAB_PATH)
    print(f"Vocab loaded: {len(vocab)} tokens ({time.time()-t0:.3f}s)")

    # ── Reference audio ──
    ref_audio_path = find_ref_audio()
    if ref_audio_path is None:
        ref_audio_path = create_sine_ref_audio()
    print(f"Reference audio: {ref_audio_path}")

    # Load and preprocess reference audio
    from pydub import AudioSegment

    audio_seg = (
        AudioSegment.from_file(ref_audio_path).set_channels(1).set_frame_rate(MODEL_SAMPLE_RATE)
    )
    audio = np.array(audio_seg.get_array_of_samples(), dtype=np.float32)
    audio = normalize_to_int16(audio)
    audio_len = len(audio)
    audio = audio.reshape(1, 1, -1)
    print(f"Reference audio: {audio_len} samples ({audio_len/MODEL_SAMPLE_RATE:.2f}s)")

    # ── Text processing ──
    zh_pause_punc = r"。，、；：？！"
    ref_text_len = len(REF_TEXT.encode("utf-8")) + 3 * len(
        re.findall(zh_pause_punc, REF_TEXT)
    )
    gen_text_len = len(GEN_TEXT.encode("utf-8")) + 3 * len(
        re.findall(zh_pause_punc, GEN_TEXT)
    )
    ref_audio_len = audio_len // HOP_LENGTH + 1
    max_duration = np.array(
        [ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / SPEED)],
        dtype=np.int64,
    )
    print(f"Estimated max_duration: {max_duration[0]} frames ({max_duration[0]*HOP_LENGTH/MODEL_SAMPLE_RATE:.2f}s audio)")

    t0 = time.time()
    gen_text_pinyin = convert_char_to_pinyin([REF_TEXT + GEN_TEXT])
    text_ids = list_str_to_idx_numpy(gen_text_pinyin, vocab)
    time_step = np.array([0], dtype=np.int32)
    print(f"Text processing: {time.time()-t0:.3f}s, text_ids shape: {text_ids.shape}")

    # ── ONNX Sessions ──
    onnxruntime.set_seed(RANDOM_SEED)
    sess_opts = onnxruntime.SessionOptions()
    sess_opts.log_severity_level = 4
    sess_opts.inter_op_num_threads = MAX_THREADS
    sess_opts.intra_op_num_threads = MAX_THREADS
    sess_opts.enable_cpu_mem_arena = True
    sess_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    sess_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    cuda_opts = [
        {
            "device_id": DEVICE_ID,
            "gpu_mem_limit": 8 * 1024 * 1024 * 1024,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "cudnn_conv_use_max_workspace": "1",
            "do_copy_in_default_stream": "1",
            "cudnn_conv1d_pad_to_nc1d": "1",
            "enable_cuda_graph": "0",
            "use_tf32": "0",
        }
    ]
    providers = ["CUDAExecutionProvider"]
    provider_options = cuda_opts

    print(f"\nLoading ALL ONNX models on CUDA...")
    t0 = time.time()
    sess_A = onnxruntime.InferenceSession(
        ONNX_A, sess_options=sess_opts, providers=providers, provider_options=provider_options,
    )
    t_a = time.time() - t0
    print(f"  Preprocess model: {t_a:.2f}s (provider: {sess_A.get_providers()[0]})")

    t0 = time.time()
    sess_opts_gpu = onnxruntime.SessionOptions()
    sess_opts_gpu.log_severity_level = 4
    sess_opts_gpu.inter_op_num_threads = MAX_THREADS
    sess_opts_gpu.intra_op_num_threads = MAX_THREADS
    sess_opts_gpu.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_B = onnxruntime.InferenceSession(
        ONNX_B,
        sess_options=sess_opts_gpu,
        providers=providers,
        provider_options=provider_options,
    )
    t_b = time.time() - t0
    print(f"  Transformer model: {t_b:.2f}s (provider: {sess_B.get_providers()[0]})")

    t0 = time.time()
    sess_C = onnxruntime.InferenceSession(
        ONNX_C, sess_options=sess_opts, providers=providers, provider_options=provider_options,
    )
    t_c = time.time() - t0
    print(f"  Decode model: {t_c:.2f}s (provider: {sess_C.get_providers()[0]})")
    print(f"  Total model load: {t_a+t_b+t_c:.2f}s")

    # Print model I/O info
    print("\nModel I/O:")
    for name, sess in [("Preprocess", sess_A), ("Transformer", sess_B), ("Decode", sess_C)]:
        inputs = [(i.name, i.shape, i.type) for i in sess.get_inputs()]
        outputs = [(o.name, o.shape, o.type) for o in sess.get_outputs()]
        print(f"  {name}: {len(inputs)} inputs -> {len(outputs)} outputs")

    in_A = [i.name for i in sess_A.get_inputs()]
    out_A = [o.name for o in sess_A.get_outputs()]
    in_B = [i.name for i in sess_B.get_inputs()]
    out_B = [o.name for o in sess_B.get_outputs()]
    in_C = [i.name for i in sess_C.get_inputs()]
    out_C = [o.name for o in sess_C.get_outputs()]

    # ── Inference (all stages on CUDA with IO binding) ──
    print(f"\n{'='*60}")
    print(f"Generating: \"{GEN_TEXT}\"")
    print(f"NFE steps: {NFE_STEP}, all stages on CUDA")
    print(f"{'='*60}")

    total_start = time.time()

    # Stage A: Preprocess (CUDA with IO binding)
    t0 = time.time()
    audio_ort = onnxruntime.OrtValue.ortvalue_from_numpy(audio, "cuda", DEVICE_ID)
    text_ids_ort = onnxruntime.OrtValue.ortvalue_from_numpy(text_ids, "cuda", DEVICE_ID)
    max_dur_ort = onnxruntime.OrtValue.ortvalue_from_numpy(max_duration, "cuda", DEVICE_ID)

    iob_A = sess_A.io_binding()
    iob_A.bind_ortvalue_input(in_A[0], audio_ort)
    iob_A.bind_ortvalue_input(in_A[1], text_ids_ort)
    iob_A.bind_ortvalue_input(in_A[2], max_dur_ort)
    for name in out_A:
        iob_A.bind_output(name, "cuda", DEVICE_ID)
    sess_A.run_with_iobinding(iob_A)
    a_outputs = iob_A.get_outputs()
    t_preprocess = time.time() - t0

    # Get numpy copy for shape reporting
    noise_np = a_outputs[0].numpy()
    print(f"\nStage A (Preprocess on CUDA): {t_preprocess*1000:.1f}ms")
    print(f"  noise shape: {noise_np.shape}")

    # Stage B: Transformer (NFE steps with IO binding for CUDA)
    t0 = time.time()
    noise_ort = a_outputs[0]  # keep on GPU
    time_step_ort = onnxruntime.OrtValue.ortvalue_from_numpy(time_step, "cuda", DEVICE_ID)

    io_binding = sess_B.io_binding()
    io_binding.bind_ortvalue_input(in_B[0], noise_ort)
    for i in range(1, 7):
        io_binding.bind_ortvalue_input(in_B[i], a_outputs[i])  # rope/cat tensors stay on GPU
    io_binding.bind_ortvalue_input(in_B[7], time_step_ort)
    io_binding.bind_ortvalue_output(out_B[0], noise_ort)
    io_binding.bind_ortvalue_output(out_B[1], time_step_ort)

    t_transfer = time.time() - t0
    print(f"\nStage B (Transformer on CUDA): GPU binding: {t_transfer*1000:.1f}ms")

    step_times = []
    t0 = time.time()
    for step in range(0, NFE_STEP, FUSE_NFE):
        t_step = time.time()
        sess_B.run_with_iobinding(io_binding)
        step_times.append(time.time() - t_step)

    t_transformer = time.time() - t0
    print(f"  {NFE_STEP} steps: {t_transformer*1000:.1f}ms total")
    print(f"  Per step: {np.mean(step_times)*1000:.1f}ms avg, {np.min(step_times)*1000:.1f}ms min, {np.max(step_times)*1000:.1f}ms max")

    # Stage C: Decode (CUDA with IO binding)
    t0 = time.time()
    noise_out_ort = io_binding.get_outputs()[0]
    ref_signal_len_ort = a_outputs[7]  # ref_signal_len from preprocess, on GPU

    iob_C = sess_C.io_binding()
    iob_C.bind_ortvalue_input(in_C[0], noise_out_ort)
    iob_C.bind_ortvalue_input(in_C[1], ref_signal_len_ort)
    iob_C.bind_output(out_C[0], "cuda", DEVICE_ID)
    sess_C.run_with_iobinding(iob_C)
    generated = iob_C.get_outputs()[0].numpy()
    t_decode = time.time() - t0
    print(f"\nStage C (Decode on CUDA): {t_decode*1000:.1f}ms")

    total_time = time.time() - total_start
    gen_samples = generated.reshape(-1).shape[0]
    gen_duration = gen_samples / MODEL_SAMPLE_RATE

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Generated audio: {gen_samples} samples ({gen_duration:.2f}s)")
    print(f"Total inference: {total_time*1000:.1f}ms")
    print(f"  Preprocess:  {t_preprocess*1000:.1f}ms")
    print(f"  GPU transfer: {t_transfer*1000:.1f}ms")
    print(f"  Transformer: {t_transformer*1000:.1f}ms ({NFE_STEP} steps)")
    print(f"  Decode:      {t_decode*1000:.1f}ms")
    print(f"RTF: {total_time/gen_duration:.3f} (1.0 = realtime)")
    print(f"Latency per second of audio: {total_time/gen_duration*1000:.0f}ms")

    # Save output
    sf.write(OUTPUT_WAV, generated.reshape(-1), MODEL_SAMPLE_RATE, format="WAV")
    print(f"\nSaved to: {OUTPUT_WAV}")

    # ── Benchmark: try fewer NFE steps ──
    print(f"\n{'='*60}")
    print("NFE step sweep (quality vs speed tradeoff)")
    print(f"{'='*60}")
    for nfe in [8, 16, 24, 32]:
        # Re-run preprocess to reset noise
        a_out = sess_A.run(out_A, {in_A[0]: audio, in_A[1]: text_ids, in_A[2]: max_duration})
        noise2 = a_out[0]
        time_step2 = np.array([0], dtype=np.int32)

        inputs2 = [
            onnxruntime.OrtValue.ortvalue_from_numpy(noise2, "cuda", DEVICE_ID),
            onnxruntime.OrtValue.ortvalue_from_numpy(a_out[1], "cuda", DEVICE_ID),
            onnxruntime.OrtValue.ortvalue_from_numpy(a_out[2], "cuda", DEVICE_ID),
            onnxruntime.OrtValue.ortvalue_from_numpy(a_out[3], "cuda", DEVICE_ID),
            onnxruntime.OrtValue.ortvalue_from_numpy(a_out[4], "cuda", DEVICE_ID),
            onnxruntime.OrtValue.ortvalue_from_numpy(a_out[5], "cuda", DEVICE_ID),
            onnxruntime.OrtValue.ortvalue_from_numpy(a_out[6], "cuda", DEVICE_ID),
            onnxruntime.OrtValue.ortvalue_from_numpy(time_step2, "cuda", DEVICE_ID),
        ]
        outputs2 = [inputs2[0], inputs2[-1]]
        iob = sess_B.io_binding()
        for i in range(len(inputs2)):
            iob.bind_ortvalue_input(name=in_B[i], ortvalue=inputs2[i])
        for i in range(len(outputs2)):
            iob.bind_ortvalue_output(name=out_B[i], ortvalue=outputs2[i])

        t0 = time.time()
        for _ in range(nfe):
            sess_B.run_with_iobinding(iob)
        t_nfe = time.time() - t0
        print(f"  NFE={nfe:2d}: {t_nfe*1000:.0f}ms transformer, {t_nfe/gen_duration:.3f} RTF")

    # ── Benchmark: pseudo-streaming (sentence splitting) ──
    print(f"\n{'='*60}")
    print("Pseudo-streaming test (sentence-level chunking)")
    print(f"{'='*60}")
    sentences = re.split(r'(?<=[。！？，,.])', GEN_TEXT)
    sentences = [s.strip() for s in sentences if s.strip()]
    print(f"Sentences: {sentences}")
    for i, sent in enumerate(sentences):
        gen_len = len(sent.encode("utf-8")) + 3 * len(re.findall(zh_pause_punc, sent))
        md = np.array(
            [ref_audio_len + int(ref_audio_len / ref_text_len * gen_len / SPEED)],
            dtype=np.int64,
        )
        pinyin = convert_char_to_pinyin([REF_TEXT + sent])
        tids = list_str_to_idx_numpy(pinyin, vocab)

        t0 = time.time()
        a_out = sess_A.run(out_A, {in_A[0]: audio, in_A[1]: tids, in_A[2]: md})
        n, rc, rs, rkc, rks, cmt, cmtd, rsl = a_out
        ts = np.array([0], dtype=np.int32)
        ins = [
            onnxruntime.OrtValue.ortvalue_from_numpy(n, "cuda", DEVICE_ID),
            onnxruntime.OrtValue.ortvalue_from_numpy(rc, "cuda", DEVICE_ID),
            onnxruntime.OrtValue.ortvalue_from_numpy(rs, "cuda", DEVICE_ID),
            onnxruntime.OrtValue.ortvalue_from_numpy(rkc, "cuda", DEVICE_ID),
            onnxruntime.OrtValue.ortvalue_from_numpy(rks, "cuda", DEVICE_ID),
            onnxruntime.OrtValue.ortvalue_from_numpy(cmt, "cuda", DEVICE_ID),
            onnxruntime.OrtValue.ortvalue_from_numpy(cmtd, "cuda", DEVICE_ID),
            onnxruntime.OrtValue.ortvalue_from_numpy(ts, "cuda", DEVICE_ID),
        ]
        outs = [ins[0], ins[-1]]
        iob = sess_B.io_binding()
        for j in range(len(ins)):
            iob.bind_ortvalue_input(name=in_B[j], ortvalue=ins[j])
        for j in range(len(outs)):
            iob.bind_ortvalue_output(name=out_B[j], ortvalue=outs[j])
        for _ in range(NFE_STEP):
            sess_B.run_with_iobinding(iob)
        n_out = onnxruntime.OrtValue.numpy(iob.get_outputs()[0])
        gen = sess_C.run(out_C, {in_C[0]: n_out, in_C[1]: rsl})[0]
        elapsed = time.time() - t0
        dur = gen.reshape(-1).shape[0] / MODEL_SAMPLE_RATE
        print(f"  [{i}] \"{sent}\" -> {dur:.2f}s audio in {elapsed*1000:.0f}ms (TTFB would be {elapsed*1000:.0f}ms)")


if __name__ == "__main__":
    main()
