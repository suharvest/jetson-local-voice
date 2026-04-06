#!/usr/bin/env python3
"""
Test MTE encoder vs ONNX streaming encoder output.
Runs ONNX encoder to get embed output, then feeds same embed to both
ONNX body and MTE, comparing per-chunk cosine similarity.
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, '/tmp/mte/engine')

WEIGHT_DIR = '/tmp/mte/weights'
ENGINE_DIR = '/tmp/mte/engine'
LIB_PATH = os.path.join(ENGINE_DIR, 'libzipformer_encoder.so')
ENCODER_ONNX = '/home/cat/zipformer-onnx/encoder-epoch-99-avg-1.onnx'
TEST_WAV = '/home/cat/zipformer-rknn/test_wavs/0.wav'


def cos_sim(a, b):
    a_f = a.flatten().astype(np.float64)
    b_f = b.flatten().astype(np.float64)
    return np.dot(a_f, b_f) / (np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-12)


def compute_fbank(samples, sr=16000):
    import kaldi_native_fbank as knf
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0.0
    opts.frame_opts.samp_freq = sr
    opts.frame_opts.frame_shift_ms = 10.0
    opts.frame_opts.frame_length_ms = 25.0
    opts.mel_opts.num_bins = 80
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(sr, samples.tolist())
    fbank.input_finished()
    n = fbank.num_frames_ready
    features = np.zeros((n, 80), dtype=np.float32)
    for i in range(n):
        features[i] = fbank.get_frame(i)
    return features


def main():
    import wave
    import onnxruntime as ort

    # Load audio
    with wave.open(TEST_WAV, 'rb') as wf:
        sr = wf.getframerate()
        samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    print(f"Audio: {TEST_WAV} ({len(samples)/sr:.2f}s)")

    # Compute fbank
    features = compute_fbank(samples, sr)
    print(f"Fbank: {features.shape}")

    # Load ONNX streaming encoder
    print("\nLoading ONNX streaming encoder...")
    sess = ort.InferenceSession(ENCODER_ONNX, providers=['CPUExecutionProvider'])
    input_names = [i.name for i in sess.get_inputs()]
    output_names = [o.name for o in sess.get_outputs()]

    # Initialize ONNX states
    N = 1
    onnx_states = {}
    type_map = {
        'tensor(float)': np.float32,
        'tensor(int64)': np.int64,
    }
    for inp in sess.get_inputs():
        name = inp.name
        if name == 'x':
            continue
        shape = []
        for d in inp.shape:
            if isinstance(d, int):
                shape.append(d)
            else:
                shape.append(N)
        dtype = type_map.get(inp.type, np.float32)
        onnx_states[name] = np.zeros(shape, dtype=dtype)

    # Load MTE engine
    from zipformer_encoder_wrapper import ZipformerEncoderEngine
    print("Loading MTE engine...")
    enc = ZipformerEncoderEngine(WEIGHT_DIR, max_T=64, lib_path=LIB_PATH)
    mte_state = enc.create_state()

    # Process chunks
    chunk_fbank = 39
    stride_fbank = 32
    n_chunks = (features.shape[0] - chunk_fbank) // stride_fbank + 1
    print(f"\nProcessing {n_chunks} chunks")
    print(f"{'Chunk':>5} {'ONNX shape':>12} {'MTE shape':>12} {'cos':>10} {'ONNX range':>24} {'MTE range':>24}")
    print("-" * 100)

    all_cos = []

    for ci in range(min(n_chunks, 30)):
        start = ci * stride_fbank
        end = start + chunk_fbank
        if end > features.shape[0]:
            break
        chunk = features[start:end]

        # Run ONNX
        feeds = {'x': chunk[np.newaxis, :, :]}  # [1, 39, 80]
        feeds.update(onnx_states)
        onnx_outs = sess.run(None, feeds)

        # Map outputs to names
        onnx_out_dict = dict(zip(output_names, onnx_outs))
        onnx_encoder_out = onnx_out_dict['encoder_out']  # [1, T_out, 512]

        # Update ONNX states for next chunk
        for name in onnx_states:
            new_name = 'new_' + name
            if new_name in onnx_out_dict:
                onnx_states[name] = onnx_out_dict[new_name]

        # Extract embed from ONNX output (we need the post-embed hidden states)
        # The ONNX model includes embed, so onnx_encoder_out is the final output.
        # For MTE, we need to extract the embed output separately.
        # Let's use the ONNX model with intermediate output extraction.
        # Actually, let's just compare the final encoder outputs.
        # But MTE takes post-embed [16, 256] as input, while ONNX takes fbank [1, 39, 80].
        # We need to extract the embed output from ONNX.

        # For now, let's use a separate approach: run MTE with random input
        # and just check that the output is reasonable (no NaN, reasonable range).
        # The proper comparison requires extracting ONNX intermediate tensors.

        onnx_out_2d = onnx_encoder_out.squeeze(0)  # [T_out, 512]

        # For MTE, we need the post-embed features. Let's extract from ONNX using
        # a modified model or use the standalone embed we have.
        # Since we don't have a clean way, let's just run MTE with the same
        # chunk through our own embed implementation.

        pass  # We'll handle this below

    # Since extracting embed from ONNX is complex, let's do a simpler test:
    # Run MTE with random input and verify no NaN, reasonable range
    print("\n\n=== MTE Sanity Test (random input) ===")
    enc.reset_state(mte_state)

    rng = np.random.RandomState(42)
    for ci in range(5):
        chunk = (rng.randn(16, 256) * 0.1).astype(np.float32)
        t0 = time.time()
        mte_out = enc.run_chunk(mte_state, chunk)
        ms = (time.time() - t0) * 1000
        has_nan = np.any(np.isnan(mte_out)) or np.any(np.isinf(mte_out))
        print(f"  Chunk {ci}: [{mte_out.shape[0]},512] "
              f"range=[{mte_out.min():.4f},{mte_out.max():.4f}] "
              f"mean={mte_out.mean():.4f} std={mte_out.std():.4f} "
              f"NaN={'YES' if has_nan else 'no'} ({ms:.1f}ms)")

    # Now compare ONNX vs MTE using the ONNX embed output
    # We need to hook into the ONNX model to get intermediate tensors
    print("\n\n=== ONNX vs MTE Comparison ===")
    print("Extracting embed outputs from ONNX model...")

    import onnx
    from onnx import helper

    # Load model and add intermediate output for embed
    model = onnx.load(ENCODER_ONNX)

    # The embed output in the ONNX graph is /Transpose_output_0 [T, 1, 256]
    # (NOT /encoder_embed/Transpose_output_0 which is mid-conv)
    embed_output_name = '/Transpose_output_0'
    print(f"  Using embed output: {embed_output_name}")

    print(f"  Embed output tensor: {embed_output_name}")

    # Add embed output to the model
    embed_out_value = helper.make_tensor_value_info(embed_output_name, onnx.TensorProto.FLOAT, None)
    model.graph.output.append(embed_out_value)

    # Create new session with modified model
    model_bytes = model.SerializeToString()
    sess2 = ort.InferenceSession(model_bytes, providers=['CPUExecutionProvider'])
    output_names2 = [o.name for o in sess2.get_outputs()]

    # Reset states
    onnx_states2 = {}
    for inp in sess2.get_inputs():
        name = inp.name
        if name == 'x':
            continue
        shape = [d if isinstance(d, int) else N for d in inp.shape]
        dtype = type_map.get(inp.type, np.float32)
        onnx_states2[name] = np.zeros(shape, dtype=dtype)

    enc.reset_state(mte_state)

    all_cos = []
    for ci in range(min(n_chunks, 30)):
        start = ci * stride_fbank
        end = start + chunk_fbank
        if end > features.shape[0]:
            break
        chunk = features[start:end]

        # Run ONNX with embed extraction
        feeds = {'x': chunk[np.newaxis, :, :]}
        feeds.update(onnx_states2)
        onnx_outs = sess2.run(None, feeds)
        onnx_out_dict = dict(zip(output_names2, onnx_outs))

        # Update states
        for name in onnx_states2:
            new_name = 'new_' + name
            if new_name in onnx_out_dict:
                onnx_states2[name] = onnx_out_dict[new_name]

        onnx_encoder_out = onnx_out_dict['encoder_out'].squeeze(0)  # [T, 512]
        embed_out = onnx_out_dict.get(embed_output_name)

        if embed_out is None:
            print(f"  Chunk {ci}: embed output not found!")
            continue

        # embed_out shape: [T_embed, 1, 256] -> squeeze to [T_embed, 256]
        embed_2d = embed_out.reshape(-1, embed_out.shape[-1])  # [T, 256]
        if embed_2d.ndim == 2 and embed_2d.shape[-1] == 256:
            # Run MTE with embed output
            t0 = time.time()
            mte_out = enc.run_chunk(mte_state, embed_2d)
            ms = (time.time() - t0) * 1000

            # Compare
            T_min = min(onnx_encoder_out.shape[0], mte_out.shape[0])
            if T_min > 0:
                cos = cos_sim(onnx_encoder_out[:T_min], mte_out[:T_min])
                all_cos.append(cos)
                print(f"  Chunk {ci:2d}: cos={cos:.6f} "
                      f"ONNX=[{onnx_encoder_out.min():.3f},{onnx_encoder_out.max():.3f}] "
                      f"MTE=[{mte_out.min():.3f},{mte_out.max():.3f}] "
                      f"({ms:.1f}ms)")
            else:
                print(f"  Chunk {ci:2d}: T_min=0, shapes: ONNX={onnx_encoder_out.shape} MTE={mte_out.shape}")
        else:
            print(f"  Chunk {ci:2d}: unexpected embed shape: {embed_out.shape}")

    if all_cos:
        print(f"\n  Mean cos: {np.mean(all_cos):.6f}")
        print(f"  Min  cos: {np.min(all_cos):.6f}")
        print(f"  Max  cos: {np.max(all_cos):.6f}")

    enc.destroy_state(mte_state)
    enc.close()
    print("\nDone.")


if __name__ == '__main__':
    main()
