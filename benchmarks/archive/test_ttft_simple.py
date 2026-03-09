"""F5-TTS TTFT benchmark — single session, lean."""
import re, time, os, numpy as np, onnxruntime as ort

MODEL_DIR = "/opt/models/f5-tts-onnx"
SR = 24000; HOP = 256
REF_TEXT = "对，这就是我，万人敬仰的太乙真人。"

def load_vocab():
    with open(os.path.join(MODEL_DIR, "vocab.txt"), "r") as f:
        return {c.rstrip("\n"): i for i, c in enumerate(f)}

def to_pinyin(texts):
    import jieba; from pypinyin import lazy_pinyin, Style
    if not jieba.dt.initialized: jieba.default_logger.setLevel(50); jieba.initialize()
    tr = str.maketrans({";": ","})
    def iz(c): return "\u3100" <= c <= "\u9fff"
    r = []
    for t in texts:
        cs = []; t = t.translate(tr)
        for seg in jieba.cut(t):
            sbl = len(seg.encode("utf-8"))
            if sbl == len(seg):
                if cs and sbl > 1 and cs[-1] not in " :'\"": cs.append(" ")
                cs.extend(seg)
            elif sbl == 3*len(seg):
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if iz(c): cs.append(" "); cs.append(seg_[i])
            else:
                for c in seg:
                    if ord(c) < 256: cs.extend(c)
                    elif iz(c): cs.append(" "); cs.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                    else: cs.append(c)
        r.append(cs)
    return r

def to_ids(text, vocab, pad=-1):
    il = [[vocab.get(c, 0) for c in t] for t in text]
    ml = max(len(t) for t in il)
    r = np.full((len(il), ml), pad, dtype=np.int32)
    for i, row in enumerate(il): r[i, :len(row)] = row
    return r

def load_ref():
    from pydub import AudioSegment
    s = AudioSegment.from_file(os.path.join(MODEL_DIR, "basic_ref_zh.wav")).set_channels(1).set_frame_rate(SR)
    a = np.array(s.get_array_of_samples(), dtype=np.float32)
    mx = np.max(np.abs(a))
    return (a * (32767.0 / mx) if mx > 0 else a).astype(np.int16).reshape(1, 1, -1)

vocab = load_vocab()
ref = load_ref()
ral = ref.shape[-1] // HOP + 1
rtl = len(REF_TEXT.encode("utf-8"))

ort.set_seed(9527)
opts = ort.SessionOptions()
opts.log_severity_level = 4
opts.inter_op_num_threads = 4
opts.intra_op_num_threads = 4
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
cuda_po = [{"device_id": "0", "use_tf32": "1", "cudnn_conv_algo_search": "EXHAUSTIVE", "cudnn_conv_use_max_workspace": "1"}]

print("Loading models...")
sA = ort.InferenceSession(os.path.join(MODEL_DIR, "F5_Preprocess.onnx"), sess_options=opts, providers=["CUDAExecutionProvider"], provider_options=cuda_po)
sB = ort.InferenceSession(os.path.join(MODEL_DIR, "F5_Transformer_fixed.onnx"), sess_options=opts, providers=["CUDAExecutionProvider"], provider_options=cuda_po)
sC = ort.InferenceSession(os.path.join(MODEL_DIR, "F5_Decode.onnx"), sess_options=opts, providers=["CUDAExecutionProvider"], provider_options=cuda_po)
print("  A:", sA.get_providers()[0], " B:", sB.get_providers()[0], " C:", sC.get_providers()[0])

inA = [i.name for i in sA.get_inputs()]; outA = [o.name for o in sA.get_outputs()]
inB = [i.name for i in sB.get_inputs()]; outB = [o.name for o in sB.get_outputs()]
inC = [i.name for i in sC.get_inputs()]; outC = [o.name for o in sC.get_outputs()]

def measure(gen_text, nfe):
    gtl = len(gen_text.encode("utf-8"))
    md = np.array([ral + int(ral / rtl * gtl / 1.0)], dtype=np.int64)
    tids = to_ids(to_pinyin([REF_TEXT + gen_text]), vocab)
    ts = np.array([0], dtype=np.int32)

    t0 = time.time()
    ao = sA.run(outA, {inA[0]: ref, inA[1]: tids, inA[2]: md})
    t_pre = time.time() - t0

    t1 = time.time()
    ins = [ort.OrtValue.ortvalue_from_numpy(ao[i], "cuda", 0) for i in range(7)]
    ins.append(ort.OrtValue.ortvalue_from_numpy(ts, "cuda", 0))
    iob = sB.io_binding()
    for i in range(8): iob.bind_ortvalue_input(inB[i], ins[i])
    iob.bind_ortvalue_output(outB[0], ins[0])
    iob.bind_ortvalue_output(outB[1], ins[7])
    for _ in range(nfe): sB.run_with_iobinding(iob)
    t_trans = time.time() - t1

    t2 = time.time()
    no = ort.OrtValue.numpy(iob.get_outputs()[0])
    g = sC.run(outC, {inC[0]: no, inC[1]: ao[7]})[0]
    t_dec = time.time() - t2

    total = time.time() - t0
    dur = g.reshape(-1).shape[0] / SR
    return total, dur, t_pre, t_trans, t_dec

# Warmup
measure("你好。", 2)
measure("你好。", 2)
print("Warmup done\n")

tests = [
    "好的。",
    "没问题。",
    "你好，很高兴认识你。",
    "今天天气真不错，我们出去走走吧。",
    "你好，我是你的智能助手，很高兴认识你。今天天气真不错，我们聊聊吧。",
]

sep = "=" * 80
print(sep)
print("F5-TTS TTFT Benchmark (CUDA FP16) on Jetson Orin NX")
print(sep)
header = f"{'NFE':>3} {'Text':<42} {'TTFT':>8} {'Pre':>6} {'Trans':>7} {'Dec':>5} {'Audio':>6} {'RTF':>5}"
print(header)
print("-" * 80)

for nfe in [8, 16, 32]:
    for t in tests:
        total, dur, tp, tt, td = measure(t, nfe)
        short = t[:40] + ".." if len(t) > 42 else t
        print(f"{nfe:>3} {short:<42} {total*1000:>7.0f}ms {tp*1000:>5.0f}ms {tt*1000:>6.0f}ms {td*1000:>4.0f}ms {dur:>5.2f}s {total/dur:>5.2f}")
    print()

# Pseudo-streaming: first sentence chunk
print(sep)
print("Pseudo-streaming: TTFT = time to generate FIRST sentence chunk")
print(sep)

long_text = "你好，我是你的智能助手，很高兴认识你。今天天气真不错，我们聊聊吧。"
sents = [s.strip() for s in re.split(r'(?<=[。！？，,.])', long_text) if s.strip()]
print(f"Full: \"{long_text}\"")
print(f"Chunks: {sents}\n")

for nfe in [8, 16, 32]:
    e1, d1, _, _, _ = measure(sents[0], nfe)
    ef, df, _, _, _ = measure(long_text, nfe)
    print(f"  NFE={nfe:2d}: 1st chunk \"{sents[0]}\" TTFT={e1*1000:.0f}ms ({d1:.1f}s)  |  Full TTFT={ef*1000:.0f}ms ({df:.1f}s)")

print(f"\n{sep}")
print("Reference: Kokoro TTS TTFT ~130ms for short, ~1s for sentence")
print(sep)
