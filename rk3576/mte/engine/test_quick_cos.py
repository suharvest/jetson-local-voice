#!/usr/bin/env python3
"""Quick cosine similarity test of MTE vs CPU reference."""
import sys, os, numpy as np
sys.path.insert(0, '/tmp/mte/engine')
from zipformer_encoder_wrapper import ZipformerEncoderEngine

lib = sys.argv[1] if len(sys.argv) > 1 else '/tmp/mte/engine/libzipformer_encoder_fp16.so'

# Redirect C stdout to /dev/null to avoid buffer overflow
import ctypes
libc = ctypes.CDLL(None)
devnull = os.open(os.devnull, os.O_WRONLY)
old_stdout = os.dup(1)
os.dup2(devnull, 1)

enc = ZipformerEncoderEngine('/tmp/mte/weights', 64, lib)
state = enc.create_state()

embed = np.load('/tmp/mte/embed_out_reference.npy').astype(np.float32)
ref = np.load('/tmp/mte/encoder_out_cpu_reference.npy')

outs = []
n_chunks = embed.shape[0] // 16
for i in range(n_chunks):
    out = enc.run_chunk(state, embed[i*16:(i+1)*16])
    outs.append(out)

enc.destroy_state(state)
enc.close()

# Restore stdout
os.dup2(old_stdout, 1)
os.close(devnull)

eo = np.concatenate(outs)
T = min(eo.shape[0], ref.shape[0])

ef = eo[:T].ravel().astype(np.float64)
rf = ref[:T].ravel().astype(np.float64)
cos = np.dot(ef, rf) / (np.linalg.norm(ef) * np.linalg.norm(rf) + 1e-12)

print(f"Overall cos={cos:.6f} (T={T})")
print(f"MTE: shape={eo.shape} range=[{eo[:T].min():.4f}, {eo[:T].max():.4f}]")
print(f"Ref: shape={ref.shape} range=[{ref[:T].min():.4f}, {ref[:T].max():.4f}]")

# Per-chunk
for i in range(min(10, T // 8)):
    t0, t1 = i * 8, min(i * 8 + 8, T)
    mc = eo[t0:t1].ravel().astype(np.float64)
    rc = ref[t0:t1].ravel().astype(np.float64)
    cc = np.dot(mc, rc) / (np.linalg.norm(mc) * np.linalg.norm(rc) + 1e-12)
    print(f"  Chunk {i}: cos={cc:.6f}")
