#!/usr/bin/env python3
"""
MTE Streaming ASR Test

Tests the MTE streaming encoder by processing audio in 39-frame chunks:
  1. Compute fbank features
  2. For each 39-frame chunk: encoder_embed -> [16, 256] post-embed
  3. MTE streaming encoder: [16, 256] -> [8, 512] per chunk
  4. Accumulate all encoder output
  5. Greedy search with decoder + joiner
  6. Compare with ONNX reference

Usage:
  python test_mte_stream.py                    # default wav
  python test_mte_stream.py /path/to/test.wav
  python test_mte_stream.py --compare          # also run ONNX reference for comparison
"""

import sys
import os
import time
import wave
import numpy as np

# ---------- Paths ----------
WEIGHT_DIR = "/tmp/mte/weights"
EMBED_DIR = "/tmp/mte/weights/encoder_embed"
ENGINE_DIR = "/tmp/mte/engine"
LIB_PATH = os.path.join(ENGINE_DIR, "libzipformer_encoder.so")

ENCODER_ONNX = "/home/cat/zipformer-onnx/encoder-epoch-99-avg-1.onnx"
DECODER_RKNN = "/home/cat/zipformer-rknn/decoder.rknn"
JOINER_ONNX = "/home/cat/zipformer-onnx/joiner-epoch-99-avg-1.onnx"
TOKENS_TXT = "/home/cat/zipformer-rknn/tokens.txt"
TEST_WAV = "/home/cat/zipformer-rknn/test_wavs/0.wav"
CPU_REF_PATH = "/tmp/mte/encoder_out_cpu_reference.npy"

BLANK_ID = 0
CONTEXT_SIZE = 2
CHUNK_SIZE = 39  # fbank frames per chunk
EMBED_OUT_T = 16  # post-embed frames per chunk (after Conv2d subsampling)


# ============================================================
# SwooshR activation
# ============================================================

def swooshr(x):
    z = np.clip(x - 1.0, -80.0, 80.0)
    return x * (1.0 / (1.0 + np.exp(-z)))


# ============================================================
# Conv2d (im2col)
# ============================================================

def conv2d(x, weight, bias, stride=(1,1), padding=(0,0,0,0)):
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = weight.shape
    pt, pl, pb, pr = padding
    sH, sW = stride

    if pt > 0 or pb > 0 or pl > 0 or pr > 0:
        x = np.pad(x, ((0,0), (0,0), (pt,pb), (pl,pr)), mode='constant')

    _, _, H_pad, W_pad = x.shape
    H_out = (H_pad - kH) // sH + 1
    W_out = (W_pad - kW) // sW + 1

    col = np.zeros((N, C_in * kH * kW, H_out * W_out), dtype=x.dtype)
    idx = 0
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * sH
            w_start = j * sW
            patch = x[:, :, h_start:h_start+kH, w_start:w_start+kW]
            col[:, :, idx] = patch.reshape(N, -1)
            idx += 1

    w_col = weight.reshape(C_out, -1)
    out = np.zeros((N, C_out, H_out * W_out), dtype=x.dtype)
    for n in range(N):
        out[n] = w_col @ col[n]

    out = out.reshape(N, C_out, H_out, W_out)
    out += bias[None, :, None, None]
    return out


# ============================================================
# Encoder Embed
# ============================================================

class EncoderEmbed:
    def __init__(self, embed_dir):
        self.conv0_w = np.fromfile(f'{embed_dir}/conv0_weight.fp32.bin', dtype=np.float32).reshape(8, 1, 3, 3)
        self.conv0_b = np.fromfile(f'{embed_dir}/conv0_bias.fp32.bin', dtype=np.float32)
        self.conv3_w = np.fromfile(f'{embed_dir}/conv3_weight.fp32.bin', dtype=np.float32).reshape(32, 8, 3, 3)
        self.conv3_b = np.fromfile(f'{embed_dir}/conv3_bias.fp32.bin', dtype=np.float32)
        self.conv6_w = np.fromfile(f'{embed_dir}/conv6_weight.fp32.bin', dtype=np.float32).reshape(128, 32, 3, 3)
        self.conv6_b = np.fromfile(f'{embed_dir}/conv6_bias.fp32.bin', dtype=np.float32)

        linear_int8 = np.fromfile(f'{embed_dir}/linear.int8.bin', dtype=np.int8).reshape(2432, 256)
        scales = np.fromfile(f'{embed_dir}/linear.scales.bin', dtype=np.float32)
        self.linear_w = linear_int8.astype(np.float32) * scales[None, :]
        self.linear_b = np.fromfile(f'{embed_dir}/linear_bias.fp32.bin', dtype=np.float32)

    def __call__(self, features):
        """features: [T, 80] -> [T_sub, 256]"""
        T, F = features.shape
        assert F == 80
        x = features.reshape(1, 1, T, F).astype(np.float32)

        x = conv2d(x, self.conv0_w, self.conv0_b, stride=(1,1), padding=(0,1,0,1))
        x = swooshr(x)
        x = conv2d(x, self.conv3_w, self.conv3_b, stride=(2,2), padding=(0,0,0,0))
        x = swooshr(x)
        x = conv2d(x, self.conv6_w, self.conv6_b, stride=(1,2), padding=(0,0,0,0))
        x = swooshr(x)

        x = x.transpose(0, 2, 1, 3)
        N, T_sub, C, freq = x.shape
        x = x.reshape(N, T_sub, C * freq)
        out = x @ self.linear_w + self.linear_b
        return out[0]


# ============================================================
# Audio / Features / Tokens
# ============================================================

def read_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        sr = w.getframerate()
        data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0, sr

def compute_fbank(samples, sr):
    import kaldi_native_fbank as knf
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0.0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = float(sr)
    opts.mel_opts.num_bins = 80
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(float(sr), samples.tolist())
    fbank.input_finished()
    n = fbank.num_frames_ready
    features = np.zeros((n, 80), dtype=np.float32)
    for i in range(n):
        features[i] = fbank.get_frame(i)
    return features

def load_tokens(path):
    id2token = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                id2token[int(parts[1])] = parts[0]
    return id2token

def tokens_to_text(token_ids, id2token):
    pieces = [id2token.get(tid, f"<{tid}>") for tid in token_ids]
    text = "".join(pieces).replace("\u2581", " ").strip()
    return text


# ============================================================
# Greedy Search
# ============================================================

def greedy_search(encoder_out, joiner_sess, decoder, id2token, use_rknn=True):
    """Greedy search with RKNN decoder + ONNX joiner."""
    T, enc_dim = encoder_out.shape
    hyp = [BLANK_ID] * CONTEXT_SIZE

    for t in range(T):
        decoder_input = np.array([hyp[-CONTEXT_SIZE:]], dtype=np.int64)

        if use_rknn:
            dec_out = decoder.inference(inputs=[decoder_input])[0]
        else:
            dec_out = decoder.run(None, {"y": decoder_input})[0]

        dec_out = np.asarray(dec_out, dtype=np.float32)

        enc_frame = encoder_out[t:t+1].reshape(1, enc_dim).astype(np.float32)
        dec_frame = dec_out.reshape(1, -1).astype(np.float32)

        joiner_out = joiner_sess.run(None, {
            "encoder_out": enc_frame,
            "decoder_out": dec_frame
        })[0]

        token_id = int(np.argmax(joiner_out.squeeze()))
        if token_id != BLANK_ID:
            hyp.append(token_id)

    return hyp[CONTEXT_SIZE:]


# ============================================================
# ONNX streaming encoder (reference)
# ============================================================

def run_onnx_streaming(features):
    """Run ONNX encoder in streaming mode, chunk by chunk."""
    import onnxruntime as ort

    sess = ort.InferenceSession(ENCODER_ONNX, providers=["CPUExecutionProvider"])
    inputs_meta = sess.get_inputs()
    output_names = [o.name for o in sess.get_outputs()]

    # Initialize states
    states = {}
    for inp in inputs_meta:
        if inp.name == "x":
            continue
        shape = [s if isinstance(s, int) else 1 for s in inp.shape]
        dtype = np.float32 if "float" in inp.type else np.int64
        states[inp.name] = np.zeros(shape, dtype=dtype)

    n_frames = features.shape[0]
    n_chunks = (n_frames + CHUNK_SIZE - 1) // CHUNK_SIZE
    all_out = []

    for i in range(n_chunks):
        start = i * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, n_frames)
        chunk = features[start:end]
        if chunk.shape[0] < CHUNK_SIZE:
            chunk = np.concatenate([chunk, np.zeros((CHUNK_SIZE - chunk.shape[0], 80), dtype=np.float32)])

        feeds = {"x": chunk.reshape(1, CHUNK_SIZE, 80)}
        feeds.update(states)

        results = sess.run(output_names, feeds)

        new_states = {}
        for name, data in zip(output_names, results):
            if name == "encoder_out":
                all_out.append(data.squeeze(0))
            else:
                new_states[name.replace("new_", "")] = data
        states = new_states

    return np.concatenate(all_out, axis=0)


# ============================================================
# MTE streaming encoder
# ============================================================

def run_mte_streaming(features, embed):
    """Run MTE encoder in streaming mode with KV/conv cache."""
    sys.path.insert(0, ENGINE_DIR)
    from zipformer_encoder_wrapper import ZipformerEncoderEngine

    t0 = time.time()
    enc = ZipformerEncoderEngine(WEIGHT_DIR, max_T=64, lib_path=LIB_PATH)
    init_ms = (time.time() - t0) * 1000
    print(f"  MTE init: {init_ms:.0f}ms")

    state = enc.create_state()

    n_frames = features.shape[0]
    n_chunks = (n_frames + CHUNK_SIZE - 1) // CHUNK_SIZE
    all_out = []

    total_embed_ms = 0
    total_enc_ms = 0

    for i in range(n_chunks):
        start = i * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, n_frames)
        chunk = features[start:end]
        if chunk.shape[0] < CHUNK_SIZE:
            chunk = np.concatenate([chunk, np.zeros((CHUNK_SIZE - chunk.shape[0], 80), dtype=np.float32)])

        # Encoder embed: [39, 80] -> [16, 256]
        t0 = time.time()
        post_embed = embed(chunk)
        embed_ms = (time.time() - t0) * 1000
        total_embed_ms += embed_ms

        assert post_embed.shape == (EMBED_OUT_T, 256), \
            f"Expected [{EMBED_OUT_T}, 256], got {post_embed.shape}"

        # MTE streaming encoder: [16, 256] -> [8, 512]
        t0 = time.time()
        enc_out = enc.run_chunk(state, post_embed)
        enc_ms = (time.time() - t0) * 1000
        total_enc_ms += enc_ms

        all_out.append(enc_out)
        print(f"  Chunk {i}/{n_chunks}: embed={embed_ms:.0f}ms, "
              f"enc={enc_ms:.0f}ms, out={enc_out.shape}")

    enc.destroy_state(state)
    enc.close()

    encoder_out = np.concatenate(all_out, axis=0)
    print(f"  Total: embed={total_embed_ms:.0f}ms, enc={total_enc_ms:.0f}ms")
    return encoder_out


# ============================================================
# Main
# ============================================================

def main():
    wav_path = TEST_WAV
    do_compare = False

    for arg in sys.argv[1:]:
        if arg == "--compare":
            do_compare = True
        elif not arg.startswith("--"):
            wav_path = arg

    print("=" * 60)
    print("MTE Streaming ASR Test")
    print("=" * 60)

    # Load tokens
    id2token = load_tokens(TOKENS_TXT)
    print(f"Tokens: {len(id2token)}")

    # Read audio & features
    samples, sr = read_wav(wav_path)
    print(f"Audio: {wav_path} ({len(samples)/sr:.2f}s, {sr}Hz)")

    t0 = time.time()
    features = compute_fbank(samples, sr)
    fbank_ms = (time.time() - t0) * 1000
    n_chunks = (features.shape[0] + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"Fbank: {features.shape} ({fbank_ms:.0f}ms), {n_chunks} chunks")

    # Load embed
    print("\nLoading encoder_embed...")
    embed = EncoderEmbed(EMBED_DIR)

    # ─── MTE streaming ───
    print("\n--- MTE Streaming Encoder ---")
    mte_out = run_mte_streaming(features, embed)
    print(f"MTE output: {mte_out.shape}")
    print(f"  range=[{mte_out.min():.4f}, {mte_out.max():.4f}]")
    print(f"  mean={mte_out.mean():.6f}, std={mte_out.std():.6f}")
    print(f"  NaN={np.isnan(mte_out).sum()}")

    # ─── ONNX reference ───
    onnx_out = None
    if do_compare:
        print("\n--- ONNX Streaming Encoder (reference) ---")
        t0 = time.time()
        onnx_out = run_onnx_streaming(features)
        onnx_ms = (time.time() - t0) * 1000
        print(f"ONNX output: {onnx_out.shape} ({onnx_ms:.0f}ms)")
        print(f"  range=[{onnx_out.min():.4f}, {onnx_out.max():.4f}]")
        print(f"  mean={onnx_out.mean():.6f}, std={onnx_out.std():.6f}")

    # ─── Compare ───
    if onnx_out is not None:
        T_cmp = min(mte_out.shape[0], onnx_out.shape[0])
        print(f"\n--- Comparison (T_cmp={T_cmp}) ---")
        m = mte_out[:T_cmp]
        o = onnx_out[:T_cmp]
        cos = np.dot(m.ravel(), o.ravel()) / (np.linalg.norm(m) * np.linalg.norm(o) + 1e-30)
        diff = np.abs(m - o)
        rmse = np.sqrt(np.mean(diff**2))
        print(f"  cos={cos:.6f}")
        print(f"  max_diff={diff.max():.4f}, rmse={rmse:.4f}")

        # Per-frame cosine similarity
        print("  Per-frame cos:")
        for t in range(min(T_cmp, 10)):
            fc = np.dot(m[t], o[t]) / (np.linalg.norm(m[t]) * np.linalg.norm(o[t]) + 1e-30)
            print(f"    frame {t}: cos={fc:.6f}")

    # ─── Compare with saved CPU reference ───
    if os.path.exists(CPU_REF_PATH):
        ref = np.load(CPU_REF_PATH)
        T_cmp = min(mte_out.shape[0], ref.shape[0])
        if T_cmp > 0:
            m, r = mte_out[:T_cmp], ref[:T_cmp]
            cos = np.dot(m.ravel(), r.ravel()) / (np.linalg.norm(m) * np.linalg.norm(r) + 1e-30)
            print(f"\nvs saved CPU reference: cos={cos:.8f} ({T_cmp} frames)")

    # ─── Greedy Search ───
    print("\n--- Greedy Search (MTE output) ---")
    import onnxruntime as ort
    joiner_sess = ort.InferenceSession(JOINER_ONNX, providers=["CPUExecutionProvider"])

    try:
        from rknnlite.api import RKNNLite
        decoder_rknn = RKNNLite(verbose=False)
        decoder_rknn.load_rknn(DECODER_RKNN)
        decoder_rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        use_rknn = True
        decoder = decoder_rknn
    except Exception:
        decoder = ort.InferenceSession(
            DECODER_RKNN.replace(".rknn", ".onnx"),
            providers=["CPUExecutionProvider"]
        )
        use_rknn = False

    t0 = time.time()
    token_ids = greedy_search(mte_out, joiner_sess, decoder, id2token, use_rknn=use_rknn)
    search_ms = (time.time() - t0) * 1000

    if use_rknn:
        decoder_rknn.release()

    text = tokens_to_text(token_ids, id2token)

    print(f"\n{'='*60}")
    print(f"ASR Result (MTE streaming):")
    print(f"  Tokens ({len(token_ids)}): {token_ids[:20]}{'...' if len(token_ids) > 20 else ''}")
    print(f"  Text: \"{text}\"")
    print(f"{'='*60}")
    print(f"\nTiming: fbank={fbank_ms:.0f}ms, encoder={0:.0f}ms, search={search_ms:.0f}ms")

    ref_text = "昨天是 MONDAY TODAY IS李八二 ZA AFTERMORROW是星期三"
    print(f"\nReference: \"{ref_text}\"")
    if text.strip() == ref_text.strip():
        print("RESULT: EXACT MATCH!")
    else:
        # Check partial match
        common = 0
        for c in text:
            if c in ref_text:
                common += 1
        print(f"RESULT: partial match (common chars: {common}/{len(ref_text)})")

    # Also run ONNX greedy search if we have ONNX output
    if onnx_out is not None:
        print("\n--- Greedy Search (ONNX output) ---")
        if use_rknn:
            decoder_rknn2 = RKNNLite(verbose=False)
            decoder_rknn2.load_rknn(DECODER_RKNN)
            decoder_rknn2.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
            decoder2 = decoder_rknn2
        else:
            decoder2 = decoder

        onnx_tokens = greedy_search(onnx_out, joiner_sess, decoder2, id2token, use_rknn=use_rknn)
        onnx_text = tokens_to_text(onnx_tokens, id2token)
        print(f"  ONNX text: \"{onnx_text}\"")

        if use_rknn:
            decoder_rknn2.release()


if __name__ == "__main__":
    main()
