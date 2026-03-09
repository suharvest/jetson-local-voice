"""F5-TTS TensorRT benchmark on Jetson Orin NX."""

import re, time, os
import numpy as np
import onnxruntime as ort
import soundfile as sf

MODEL_DIR = "/opt/models/f5-tts-onnx"
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.txt")
ONNX_A = os.path.join(MODEL_DIR, "F5_Preprocess.onnx")
ONNX_B_CUDA = os.path.join(MODEL_DIR, "F5_Transformer_fixed.onnx")
ONNX_B_TRT = os.path.join(MODEL_DIR, "F5_Transformer_trt.onnx")
ONNX_C = os.path.join(MODEL_DIR, "F5_Decode.onnx")

RANDOM_SEED = 9527
SPEED = 1.0
MAX_THREADS = 4
DEVICE_ID = 0
MODEL_SAMPLE_RATE = 24000
HOP_LENGTH = 256

REF_AUDIO = os.environ.get("F5_REF_AUDIO", os.path.join(MODEL_DIR, "basic_ref_zh.wav"))
REF_TEXT = "对，这就是我，万人敬仰的太乙真人。"
GEN_TEXT = "你好，我是你的智能助手，很高兴认识你。"


def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        return {char.rstrip("\n"): i for i, char in enumerate(f)}


def convert_char_to_pinyin(text_list):
    import jieba
    from pypinyin import lazy_pinyin, Style
    if not jieba.dt.initialized:
        jieba.default_logger.setLevel(50)
        jieba.initialize()
    custom_trans = str.maketrans({";": ",", "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"})
    def is_chinese(c): return "\u3100" <= c <= "\u9fff"
    final = []
    for text in text_list:
        chars = []
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            sbl = len(seg.encode("utf-8"))
            if sbl == len(seg):
                if chars and sbl > 1 and chars[-1] not in " :'\"": chars.append(" ")
                chars.extend(seg)
            elif sbl == 3 * len(seg):
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c): chars.append(" "); chars.append(seg_[i])
            else:
                for c in seg:
                    if ord(c) < 256: chars.extend(c)
                    elif is_chinese(c): chars.append(" "); chars.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                    else: chars.append(c)
        final.append(chars)
    return final


def list_str_to_idx(text, vocab, pad=-1):
    idx_lists = [[vocab.get(c, 0) for c in t] for t in text]
    ml = max(len(t) for t in idx_lists)
    r = np.full((len(idx_lists), ml), pad, dtype=np.int32)
    for i, idx in enumerate(idx_lists):
        r[i, :len(idx)] = idx
    return r


def prepare_inputs(vocab):
    from pydub import AudioSegment
    audio_seg = AudioSegment.from_file(REF_AUDIO).set_channels(1).set_frame_rate(MODEL_SAMPLE_RATE)
    audio = np.array(audio_seg.get_array_of_samples(), dtype=np.float32)
    mx = np.max(np.abs(audio))
    audio = (audio * (32767.0 / mx) if mx > 0 else audio).astype(np.int16).reshape(1, 1, -1)
    audio_len = audio.shape[-1]

    zh_p = r"。，、；：？！"
    rtl = len(REF_TEXT.encode("utf-8")) + 3 * len(re.findall(zh_p, REF_TEXT))
    gtl = len(GEN_TEXT.encode("utf-8")) + 3 * len(re.findall(zh_p, GEN_TEXT))
    ral = audio_len // HOP_LENGTH + 1
    max_dur = np.array([ral + int(ral / rtl * gtl / SPEED)], dtype=np.int64)

    pinyin = convert_char_to_pinyin([REF_TEXT + GEN_TEXT])
    text_ids = list_str_to_idx(pinyin, vocab)
    return audio, text_ids, max_dur


def make_cuda_session(path):
    opts = ort.SessionOptions()
    opts.log_severity_level = 4
    opts.inter_op_num_threads = MAX_THREADS
    opts.intra_op_num_threads = MAX_THREADS
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=opts,
        providers=["CUDAExecutionProvider"],
        provider_options=[{"device_id": str(DEVICE_ID), "use_tf32": "1",
                           "cudnn_conv_algo_search": "EXHAUSTIVE",
                           "cudnn_conv_use_max_workspace": "1"}])


def make_trt_session(path):
    opts = ort.SessionOptions()
    opts.log_severity_level = 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(path, sess_options=opts,
        providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"],
        provider_options=[
            {"device_id": str(DEVICE_ID),
             "trt_max_workspace_size": str(4*1024*1024*1024),
             "trt_fp16_enable": "True",
             "trt_engine_cache_enable": "True",
             "trt_engine_cache_path": "/tmp/trt_f5_cache"},
            {"device_id": str(DEVICE_ID)}
        ])


def benchmark(label, sess_A, sess_B, sess_C, audio, text_ids, max_dur, nfe):
    in_A = [i.name for i in sess_A.get_inputs()]
    out_A = [o.name for o in sess_A.get_outputs()]
    in_B = [i.name for i in sess_B.get_inputs()]
    out_B = [o.name for o in sess_B.get_outputs()]
    in_C = [i.name for i in sess_C.get_inputs()]
    out_C = [o.name for o in sess_C.get_outputs()]

    t0 = time.time()
    a_out = sess_A.run(out_A, {in_A[0]: audio, in_A[1]: text_ids, in_A[2]: max_dur})
    t_A = time.time() - t0

    ts = np.array([0], dtype=np.int32)
    t0 = time.time()
    ins = [ort.OrtValue.ortvalue_from_numpy(a_out[i], "cuda", DEVICE_ID) for i in range(7)]
    ins.append(ort.OrtValue.ortvalue_from_numpy(ts, "cuda", DEVICE_ID))
    iob = sess_B.io_binding()
    for i in range(8): iob.bind_ortvalue_input(in_B[i], ins[i])
    iob.bind_ortvalue_output(out_B[0], ins[0])
    iob.bind_ortvalue_output(out_B[1], ins[7])

    step_times = []
    for _ in range(nfe):
        st = time.time()
        sess_B.run_with_iobinding(iob)
        step_times.append(time.time() - st)
    noise_out = ort.OrtValue.numpy(iob.get_outputs()[0])
    t_B = time.time() - t0

    t0 = time.time()
    gen = sess_C.run(out_C, {in_C[0]: noise_out, in_C[1]: a_out[7]})[0]
    t_C = time.time() - t0

    dur = gen.reshape(-1).shape[0] / MODEL_SAMPLE_RATE
    total = t_A + t_B + t_C
    avg_step = np.mean(step_times)
    print(f"  [{label}] NFE={nfe:2d}: total={total*1000:.0f}ms, "
          f"A={t_A*1000:.0f}ms B={t_B*1000:.0f}ms({avg_step*1000:.0f}ms/step) C={t_C*1000:.0f}ms, "
          f"RTF={total/dur:.3f}, audio={dur:.1f}s")
    return total, dur, avg_step


def main():
    ort.set_seed(RANDOM_SEED)
    print("=" * 70)
    print("F5-TTS: CUDA vs TensorRT Benchmark")
    print("=" * 70)

    vocab = load_vocab(VOCAB_PATH)
    audio, text_ids, max_dur = prepare_inputs(vocab)

    # === Test 1: CUDA baseline (fixed model) ===
    print("\n--- Loading CUDA sessions ---")
    sess_A = make_cuda_session(ONNX_A)
    sess_B_cuda = make_cuda_session(ONNX_B_CUDA)
    sess_C = make_cuda_session(ONNX_C)
    print(f"  Transformer provider: {sess_B_cuda.get_providers()[0]}")

    # Warmup
    benchmark("warmup", sess_A, sess_B_cuda, sess_C, audio, text_ids, max_dur, 2)

    print("\n--- CUDA Results ---")
    for nfe in [8, 16, 32]:
        benchmark("CUDA", sess_A, sess_B_cuda, sess_C, audio, text_ids, max_dur, nfe)

    del sess_B_cuda

    # === Test 2: TensorRT (decomposed model) ===
    print("\n--- Loading TensorRT sessions (first run builds engine, may take minutes) ---")
    t0 = time.time()
    sess_B_trt = make_trt_session(ONNX_B_TRT)
    load_time = time.time() - t0
    active = sess_B_trt.get_providers()
    print(f"  Loaded in {load_time:.1f}s, providers: {active}")

    # Warmup
    benchmark("warmup", sess_A, sess_B_trt, sess_C, audio, text_ids, max_dur, 2)

    print("\n--- TensorRT Results ---")
    for nfe in [8, 16, 32]:
        benchmark("TRT", sess_A, sess_B_trt, sess_C, audio, text_ids, max_dur, nfe)

    del sess_B_trt

    # === Test 3: CUDA on decomposed model (to compare fairly) ===
    print("\n--- CUDA on decomposed model (same ops as TRT version) ---")
    sess_B_cuda2 = make_cuda_session(ONNX_B_TRT)
    benchmark("warmup", sess_A, sess_B_cuda2, sess_C, audio, text_ids, max_dur, 2)
    for nfe in [8, 16, 32]:
        benchmark("CUDA-decomp", sess_A, sess_B_cuda2, sess_C, audio, text_ids, max_dur, nfe)


if __name__ == "__main__":
    main()
