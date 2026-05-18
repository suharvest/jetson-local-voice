#!/usr/bin/env python3
"""2×2 diagnosis matrix: ORT/TRT encoder × ORT/TRT decoder.

Identifies which component (encoder or decoder) causes ASR accuracy loss
by comparing all 4 combinations on the same audio input.

Usage (inside container):
    PARAFORMER_MODEL_DIR=/opt/models/paraformer-streaming \
    python3 /tmp/diag_2x2.py /tmp/tts_test.wav
"""
import sys, os, time, wave, numpy as np

MODEL_DIR = os.environ.get("PARAFORMER_MODEL_DIR", "/opt/models/paraformer-streaming")
ENGINE_DIR = os.path.join(MODEL_DIR, "engines")
sys.path.insert(0, "/opt/speech")
sys.path.insert(0, "/opt/speech/app")

os.environ.setdefault("PARAFORMER_MODEL_DIR", MODEL_DIR)
os.environ.setdefault("PARAFORMER_ENC_ENGINE", os.path.join(ENGINE_DIR, "paraformer_encoder_dp4_400.plan"))
os.environ.setdefault("PARAFORMER_DEC_ENGINE", os.path.join(ENGINE_DIR, "paraformer_decoder_fp16.plan"))
os.environ.setdefault("PARAFORMER_ENC_ONNX", os.path.join(MODEL_DIR, "encoder.onnx"))
os.environ.setdefault("PARAFORMER_DEC_ONNX", os.path.join(MODEL_DIR, "decoder-trt.onnx"))
os.environ.setdefault("PARAFORMER_TOKENS", os.path.join(MODEL_DIR, "tokens.txt"))

from app.backends.jetson.paraformer_trt import (
    ParaformerTRTBackend, compute_fbank, stack_frames,
    cif, decode_ids, load_tokens, SAMPLE_RATE,
)
import onnxruntime as ort


def load_audio(path):
    with wave.open(path, "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    return data


def run_encoder_ort(feats, enc_onnx_path):
    """Run encoder via ORT-CPU."""
    n_frames = feats.shape[0]
    if n_frames < 40:
        feats = np.pad(feats, ((0, 40 - n_frames), (0, 0)), mode="edge")
        orig_n = n_frames
        n_frames = 40
    else:
        orig_n = n_frames

    sess = ort.InferenceSession(enc_onnx_path, providers=["CPUExecutionProvider"])
    speech = np.ascontiguousarray(feats[np.newaxis, :].astype(np.float32))
    speech_len = np.array([n_frames], dtype=np.int32)
    enc, enc_len, alphas = sess.run(["enc", "enc_len", "alphas"],
                                     {"speech": speech, "speech_lengths": speech_len})
    if orig_n < n_frames:
        enc = enc[:, :orig_n, :]
        alphas = alphas[:, :orig_n]
    return enc, alphas


def run_decoder_ort(enc, enc_len, acoustic_embeds, ae_len, caches, dec_onnx_path):
    """Run decoder via ORT-CPU."""
    sess = ort.InferenceSession(dec_onnx_path, providers=["CPUExecutionProvider"])
    ort_in = {
        "enc": np.ascontiguousarray(enc),
        "enc_len": np.array([enc_len], dtype=np.int32),
        "acoustic_embeds": np.ascontiguousarray(acoustic_embeds[np.newaxis, :]),
        "acoustic_embeds_len": np.array([ae_len], dtype=np.int32),
        "pad_mask": np.ones((1, ae_len), dtype=np.float32),
        "enc_pad_mask": np.ones((1, enc_len), dtype=np.float32),
    }
    for i in range(16):
        ort_in[f"in_cache_{i}"] = np.ascontiguousarray(caches[i])
    outputs = sess.run(["sample_ids"] + [f"out_cache_{i}" for i in range(16)], ort_in)
    sample_ids = outputs[0][0]
    for i in range(16):
        caches[i][:] = outputs[1 + i]
    return sample_ids


def run_pipeline(audio, be, enc_mode, dec_mode, enc_onnx, dec_onnx):
    """Run full ASR pipeline with specified encoder/decoder modes."""
    feats = compute_fbank(audio)
    lfr = stack_frames(feats)
    n_total = lfr.shape[0]

    all_ids = []
    carry_w, carry_e = 0.0, np.zeros(512, dtype=np.float32)
    caches = [np.zeros((1, 512, 10), dtype=np.float32) for _ in range(16)]

    chunk_size = min(400, max(40, n_total))
    for start in range(0, n_total, chunk_size):
        chunk = lfr[start:start + chunk_size]
        if chunk.shape[0] < chunk_size:
            pad = np.zeros((chunk_size - chunk.shape[0], 560), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)

        # Encoder
        if enc_mode == "trt":
            enc, alphas = be._run_encoder_trt(chunk)
        else:
            enc, alphas = run_encoder_ort(chunk, enc_onnx)

        if enc is None:
            continue

        # CIF
        ae, carry_w, carry_e = cif(enc[0], alphas[0],
                                    carry_weight=carry_w, carry_embed=carry_e)
        if len(ae) == 0:
            continue

        # Decoder
        if dec_mode == "trt":
            ids = be._run_decoder(enc, chunk.shape[0], ae, len(ae), caches)
        else:
            ids = run_decoder_ort(enc, chunk.shape[0], ae, len(ae), caches, dec_onnx)

        if ids is not None:
            all_ids.extend(ids.tolist())

    # Flush tail
    if carry_w >= 0.5:
        ae = (carry_e / carry_w)[np.newaxis, :]
        dummy_enc = np.zeros((1, 1, 512), dtype=np.float32)
        if dec_mode == "trt":
            ids = be._run_decoder(dummy_enc, 1, ae, 1, caches)
        else:
            ids = run_decoder_ort(dummy_enc, 1, ae, 1, caches, dec_onnx)
        if ids is not None:
            all_ids.extend(ids.tolist())

    tokens = load_tokens(os.environ["PARAFORMER_TOKENS"])
    text = decode_ids(all_ids, tokens)
    return text, all_ids


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 diag_2x2.py <wav_file>")
        sys.exit(1)

    audio = load_audio(sys.argv[1])
    print(f"Audio: {len(audio)/SAMPLE_RATE:.1f}s, {len(audio)} samples")
    print(f"Expected text: {sys.argv[2] if len(sys.argv) > 2 else '?'}")

    enc_onnx = os.environ["PARAFORMER_ENC_ONNX"]
    dec_onnx = os.environ["PARAFORMER_DEC_ONNX"]

    be = ParaformerTRTBackend()
    be.preload()
    print(f"Backend providers: {be.providers}")

    results = {}
    for enc_mode in ["ort", "trt"]:
        for dec_mode in ["ort", "trt"]:
            label = f"enc={enc_mode} dec={dec_mode}"
            print(f"\n{'='*50}")
            print(f"Running: {label}")
            t0 = time.time()
            try:
                text, ids = run_pipeline(audio, be, enc_mode, dec_mode, enc_onnx, dec_onnx)
                elapsed = time.time() - t0
                results[label] = {"text": text, "ids": ids[:20], "n_ids": len(ids)}
                print(f"  Text: {text!r}")
                print(f"  Tokens: {len(ids)}, first 20: {ids[:20]}")
                print(f"  Time: {elapsed:.1f}s")
            except Exception as e:
                results[label] = {"error": str(e)}
                print(f"  ERROR: {e}")

    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    for label, r in results.items():
        if "error" in r:
            print(f"  {label}: ERROR - {r['error']}")
        else:
            print(f"  {label}: {r['n_ids']:3d} tokens → {r['text']!r}")


if __name__ == "__main__":
    main()
