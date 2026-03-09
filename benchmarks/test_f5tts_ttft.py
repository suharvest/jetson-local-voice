"""F5-TTS TTFT (Time To First Token) benchmark on Jetson Orin NX.

Measures: how fast can we get the first audio chunk to the speaker?
Strategy: split text into sentences, generate shortest first.
Tests: CUDA vs TensorRT, various NFE steps.
"""

import re, time, os
import numpy as np
import onnxruntime as ort

MODEL_DIR = "/opt/models/f5-tts-onnx"
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.txt")
ONNX_A = os.path.join(MODEL_DIR, "F5_Preprocess.onnx")
ONNX_B_CUDA = os.path.join(MODEL_DIR, "F5_Transformer_fixed.onnx")
ONNX_B_TRT = os.path.join(MODEL_DIR, "F5_Transformer_trt.onnx")
ONNX_C = os.path.join(MODEL_DIR, "F5_Decode.onnx")

SPEED = 1.0
DEVICE_ID = 0
MODEL_SAMPLE_RATE = 24000
HOP_LENGTH = 256

REF_AUDIO = os.environ.get("F5_REF_AUDIO", os.path.join(MODEL_DIR, "basic_ref_zh.wav"))
REF_TEXT = "对，这就是我，万人敬仰的太乙真人。"

# Realistic conversation responses to test TTFT
TEST_TEXTS = [
    "好的。",
    "没问题。",
    "你好，很高兴认识你。",
    "今天天气真不错，我们出去走走吧。",
    "你好，我是你的智能助手，很高兴认识你。今天天气真不错，我们聊聊吧。",
]


def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        return {char.rstrip("\n"): i for i, char in enumerate(f)}


def convert_char_to_pinyin(text_list):
    import jieba
    from pypinyin import lazy_pinyin, Style
    if not jieba.dt.initialized:
        jieba.default_logger.setLevel(50)
        jieba.initialize()
    tr = str.maketrans({";": ",", "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"})
    def is_zh(c): return "\u3100" <= c <= "\u9fff"
    result = []
    for text in text_list:
        chars = []
        text = text.translate(tr)
        for seg in jieba.cut(text):
            sbl = len(seg.encode("utf-8"))
            if sbl == len(seg):
                if chars and sbl > 1 and chars[-1] not in " :'\"": chars.append(" ")
                chars.extend(seg)
            elif sbl == 3 * len(seg):
                for i, c in enumerate(seg):
                    if is_zh(c):
                        chars.append(" ")
                        chars.append(lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)[i])
            else:
                for c in seg:
                    if ord(c) < 256: chars.extend(c)
                    elif is_zh(c): chars.append(" "); chars.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                    else: chars.append(c)
        result.append(chars)
    return result


def text_to_ids(text, vocab, pad=-1):
    idx = [[vocab.get(c, 0) for c in t] for t in text]
    ml = max(len(t) for t in idx)
    r = np.full((len(idx), ml), pad, dtype=np.int32)
    for i, row in enumerate(idx): r[i, :len(row)] = row
    return r


def load_ref_audio():
    from pydub import AudioSegment
    seg = AudioSegment.from_file(REF_AUDIO).set_channels(1).set_frame_rate(MODEL_SAMPLE_RATE)
    audio = np.array(seg.get_array_of_samples(), dtype=np.float32)
    mx = np.max(np.abs(audio))
    return (audio * (32767.0 / mx) if mx > 0 else audio).astype(np.int16).reshape(1, 1, -1)


def make_session(path, provider="cuda"):
    opts = ort.SessionOptions()
    opts.log_severity_level = 4
    opts.inter_op_num_threads = 4
    opts.intra_op_num_threads = 4
    if provider == "trt":
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        return ort.InferenceSession(path, sess_options=opts,
            providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"],
            provider_options=[
                {"device_id": str(DEVICE_ID), "trt_max_workspace_size": str(4*1024**3),
                 "trt_fp16_enable": "True", "trt_engine_cache_enable": "True",
                 "trt_engine_cache_path": "/tmp/trt_f5_cache"},
                {"device_id": str(DEVICE_ID)}])
    else:
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(path, sess_options=opts,
            providers=["CUDAExecutionProvider"],
            provider_options=[{"device_id": str(DEVICE_ID), "use_tf32": "1",
                               "cudnn_conv_algo_search": "EXHAUSTIVE",
                               "cudnn_conv_use_max_workspace": "1",
                               "enable_cuda_graph": "0"}])


def measure_ttft(sess_A, sess_B, sess_C, ref_audio, vocab, gen_text, nfe):
    """Measure TTFT: time from start to first audio samples available.

    For F5-TTS, TTFT = preprocess + NFE steps + decode (no streaming possible).
    For pseudo-streaming, TTFT = time to generate first sentence chunk.
    """
    zh_p = r"。，、；：？！"
    ref_audio_len = ref_audio.shape[-1]
    ral_frames = ref_audio_len // HOP_LENGTH + 1
    rtl = len(REF_TEXT.encode("utf-8")) + 3 * len(re.findall(zh_p, REF_TEXT))

    in_A = [i.name for i in sess_A.get_inputs()]
    out_A = [o.name for o in sess_A.get_outputs()]
    in_B = [i.name for i in sess_B.get_inputs()]
    out_B = [o.name for o in sess_B.get_outputs()]
    in_C = [i.name for i in sess_C.get_inputs()]
    out_C = [o.name for o in sess_C.get_outputs()]

    # ── Full text TTFT (no splitting) ──
    gtl = len(gen_text.encode("utf-8")) + 3 * len(re.findall(zh_p, gen_text))
    max_dur = np.array([ral_frames + int(ral_frames / rtl * gtl / SPEED)], dtype=np.int64)
    pinyin = convert_char_to_pinyin([REF_TEXT + gen_text])
    tids = text_to_ids(pinyin, vocab)
    ts = np.array([0], dtype=np.int32)

    t_start = time.time()

    a_out = sess_A.run(out_A, {in_A[0]: ref_audio, in_A[1]: tids, in_A[2]: max_dur})
    t_preprocess = time.time() - t_start

    ins = [ort.OrtValue.ortvalue_from_numpy(a_out[i], "cuda", DEVICE_ID) for i in range(7)]
    ins.append(ort.OrtValue.ortvalue_from_numpy(ts, "cuda", DEVICE_ID))
    iob = sess_B.io_binding()
    for i in range(8): iob.bind_ortvalue_input(in_B[i], ins[i])
    iob.bind_ortvalue_output(out_B[0], ins[0])
    iob.bind_ortvalue_output(out_B[1], ins[7])
    for _ in range(nfe):
        sess_B.run_with_iobinding(iob)
    t_transformer = time.time() - t_start - t_preprocess

    noise_out = ort.OrtValue.numpy(iob.get_outputs()[0])
    gen = sess_C.run(out_C, {in_C[0]: noise_out, in_C[1]: a_out[7]})[0]
    t_total = time.time() - t_start

    audio_dur = gen.reshape(-1).shape[0] / MODEL_SAMPLE_RATE
    full_ttft = t_total  # no streaming, TTFT = total

    # ── First-sentence TTFT (pseudo-streaming) ──
    sentences = re.split(r'(?<=[。！？，,.\!?])', gen_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    first_sent = sentences[0] if sentences else gen_text

    gtl1 = len(first_sent.encode("utf-8")) + 3 * len(re.findall(zh_p, first_sent))
    max_dur1 = np.array([ral_frames + int(ral_frames / rtl * gtl1 / SPEED)], dtype=np.int64)
    pinyin1 = convert_char_to_pinyin([REF_TEXT + first_sent])
    tids1 = text_to_ids(pinyin1, vocab)
    ts1 = np.array([0], dtype=np.int32)

    t_start2 = time.time()
    a_out1 = sess_A.run(out_A, {in_A[0]: ref_audio, in_A[1]: tids1, in_A[2]: max_dur1})
    ins1 = [ort.OrtValue.ortvalue_from_numpy(a_out1[i], "cuda", DEVICE_ID) for i in range(7)]
    ins1.append(ort.OrtValue.ortvalue_from_numpy(ts1, "cuda", DEVICE_ID))
    iob1 = sess_B.io_binding()
    for i in range(8): iob1.bind_ortvalue_input(in_B[i], ins1[i])
    iob1.bind_ortvalue_output(out_B[0], ins1[0])
    iob1.bind_ortvalue_output(out_B[1], ins1[7])
    for _ in range(nfe):
        sess_B.run_with_iobinding(iob1)
    n1 = ort.OrtValue.numpy(iob1.get_outputs()[0])
    g1 = sess_C.run(out_C, {in_C[0]: n1, in_C[1]: a_out1[7]})[0]
    chunk_ttft = time.time() - t_start2
    chunk_dur = g1.reshape(-1).shape[0] / MODEL_SAMPLE_RATE

    return {
        "text": gen_text,
        "first_sent": first_sent,
        "nfe": nfe,
        "full_ttft_ms": full_ttft * 1000,
        "full_audio_dur": audio_dur,
        "chunk_ttft_ms": chunk_ttft * 1000,
        "chunk_audio_dur": chunk_dur,
        "n_sentences": len(sentences),
    }


def main():
    ort.set_seed(9527)
    print("=" * 75)
    print("F5-TTS TTFT Benchmark — Jetson Orin NX")
    print("=" * 75)

    vocab = load_vocab(VOCAB_PATH)
    ref_audio = load_ref_audio()
    print(f"Ref audio: {ref_audio.shape[-1]} samples ({ref_audio.shape[-1]/MODEL_SAMPLE_RATE:.1f}s)")

    # Load sessions
    configs = []

    print("\n--- Loading CUDA sessions ---")
    sess_A = make_session(ONNX_A, "cuda")
    sess_C = make_session(ONNX_C, "cuda")

    sess_B_cuda = make_session(ONNX_B_CUDA, "cuda")
    print(f"  CUDA Transformer: {sess_B_cuda.get_providers()[0]}")
    configs.append(("CUDA", sess_B_cuda))

    if os.path.exists(ONNX_B_TRT):
        print("\n--- Loading TensorRT session (engine build may take minutes) ---")
        try:
            sess_B_trt = make_session(ONNX_B_TRT, "trt")
            trt_provider = sess_B_trt.get_providers()[0]
            print(f"  TRT Transformer: {trt_provider}")
            if trt_provider == "TensorrtExecutionProvider":
                configs.append(("TRT", sess_B_trt))
            else:
                print("  TRT fell back to non-TRT provider, skipping")
        except Exception as e:
            print(f"  TRT failed: {e}")

    # Warmup
    print("\n--- Warmup ---")
    for label, sess_B in configs:
        measure_ttft(sess_A, sess_B, sess_C, ref_audio, vocab, "你好。", 2)
        print(f"  {label} warmup done")

    # Benchmark
    results = []
    for nfe in [8, 16, 32]:
        for label, sess_B in configs:
            for text in TEST_TEXTS:
                r = measure_ttft(sess_A, sess_B, sess_C, ref_audio, vocab, text, nfe)
                r["provider"] = label
                results.append(r)

    # Print results
    print(f"\n{'='*75}")
    print("TTFT RESULTS (lower = better)")
    print(f"{'='*75}")

    for nfe in [8, 16, 32]:
        print(f"\n--- NFE = {nfe} ---")
        print(f"{'Provider':<6} {'Text':<30} {'FullTTFT':>9} {'1stChunkTTFT':>13} {'1stChunk':>20} {'FullAudio':>9}")
        print("-" * 95)
        for r in results:
            if r["nfe"] != nfe: continue
            short_text = r["text"][:28] + ".." if len(r["text"]) > 30 else r["text"]
            first_s = r["first_sent"][:16] + ".." if len(r["first_sent"]) > 18 else r["first_sent"]
            print(f"{r['provider']:<6} {short_text:<30} {r['full_ttft_ms']:>8.0f}ms "
                  f"{r['chunk_ttft_ms']:>12.0f}ms "
                  f"(\"{first_s}\" {r['chunk_audio_dur']:.1f}s) "
                  f"{r['full_audio_dur']:>7.1f}s")

    # Summary: best achievable TTFT
    print(f"\n{'='*75}")
    print("SUMMARY: Best achievable TTFT per text length")
    print(f"{'='*75}")
    for text in TEST_TEXTS:
        best = min((r for r in results if r["text"] == text), key=lambda r: r["chunk_ttft_ms"])
        short = text[:40] + ".." if len(text) > 42 else text
        print(f"  \"{short}\"")
        print(f"    Best TTFT: {best['chunk_ttft_ms']:.0f}ms "
              f"(NFE={best['nfe']}, {best['provider']}, "
              f"first chunk=\"{best['first_sent']}\" {best['chunk_audio_dur']:.1f}s)")

    # Compare with Kokoro reference
    print(f"\n{'='*75}")
    print("REFERENCE: Kokoro TTS (current production)")
    print(f"{'='*75}")
    print("  Kokoro RTF: ~0.133 (7.5x realtime)")
    print("  Kokoro TTFT for '你好': ~130ms")
    print("  Kokoro TTFT for full sentence: ~1s")


if __name__ == "__main__":
    main()
