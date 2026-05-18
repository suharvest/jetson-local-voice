#!/usr/bin/env python3
"""Prune TTS text_embed_fp16.bin using ASR keep_ids (reconstructed from token_map.bin)."""
import numpy as np
import os

VOCAB = 151936
DIM = 2048
EMBED_PATH = "/home/harvest/voice_test/models/qwen3-tts/onnx/text_embed_fp16.bin"
OUT_PATH = "/home/harvest/voice_test/models/qwen3-tts/onnx/text_embed_fp16_pruned.bin"
MAP_PATH = "/home/harvest/voice_test/models/qwen3-tts/onnx/token_map.bin"
ASR_MAP_PATH = "/home/harvest/voice_test/models/qwen3-asr-v2/token_map.bin"

# Step 1: Reconstruct keep_ids from ASR token_map.bin
# token_map.bin = red2orig: 35666 keep IDs + 71 trailing padding zeros
# Token 0 is a valid BPE token (included in keep_set), not padding
asr_red2orig = np.fromfile(ASR_MAP_PATH, dtype=np.uint32)
print(f"ASR token_map.bin: {len(asr_red2orig)} entries")

# all unique values = 35666 (the 71 trailing zeros are duplicates of the valid 0 at index 0)
keep_set = set(int(x) for x in asr_red2orig)
print(f"Unique keep IDs: {len(keep_set)}")

assert len(keep_set) == 35666, f"Expected 35666 keep IDs, got {len(keep_set)}"
assert all(0 <= x < VOCAB for x in keep_set), "Some IDs out of range"

keep_sorted = sorted(keep_set)
print(f"keep_sorted range: [{keep_sorted[0]}, {keep_sorted[-1]}]")

# Step 2: Define extras (same 25 prompt extras from ASR)
extras = sorted([198, 10043, 10726, 11528, 17936, 21632, 25950, 27948, 28963,
                 35454, 36565, 42527, 43419, 44029, 45195, 47776, 52017, 52948,
                 56387, 63145, 73620, 77032, 77091, 96839, 151704])

for eid in extras:
    assert eid in keep_set, f"Extra ID {eid} not in keep_set"

# Step 3: Build new_layout = old_keep_sorted + extras
old_keep = sorted(set(keep_sorted) - set(extras))
print(f"old_keep (excl extras): {len(old_keep)} entries")
print(f"extras: {len(extras)} entries")

new_layout = np.array(old_keep + extras, dtype=np.uint32)
print(f"new_layout: {len(new_layout)} total = {len(old_keep)} + {len(extras)}")
assert len(new_layout) == 35666

# Step 4: Load full embed and prune
print(f"\nLoading {EMBED_PATH}...")
fsize = os.path.getsize(EMBED_PATH)
print(f"File size: {fsize} bytes ({fsize/(1024*1024):.1f} MB)")
full = np.fromfile(EMBED_PATH, dtype=np.float16)
assert full.shape[0] == VOCAB * DIM, f"Expected {VOCAB*DIM} elements, got {full.shape[0]}"
full = full.reshape(VOCAB, DIM)
print(f"Full embed shape: {full.shape}")

pruned = full[new_layout.astype(int)]
print(f"Pruned embed shape: {pruned.shape}")
print(f"Pruned size: {pruned.nbytes / (1024*1024):.1f} MB")

# Save pruned
pruned.tofile(OUT_PATH)
print(f"\nSaved: {OUT_PATH} ({os.path.getsize(OUT_PATH)} bytes)")

# Step 5: Verify bit-equality on random subset
orig2red = {int(v): i for i, v in enumerate(new_layout)}
rng = np.random.RandomState(42)
test_ids = rng.choice(new_layout, size=min(100, len(new_layout)), replace=False)
failures = []
for tid in test_ids:
    rid = orig2red[int(tid)]
    if not np.allclose(pruned[rid], full[int(tid)]):
        failures.append((tid, rid))

if failures:
    print(f"\nBIT-EQUALITY FAILED: {len(failures)} mismatches")
    for tid, rid in failures[:5]:
        diff = np.max(np.abs(pruned[rid].astype(np.float32) - full[tid].astype(np.float32)))
        print(f"  id={tid}, red={rid}, max_diff={diff}")
else:
    print(f"\nBIT-EQUALITY PASS: {len(test_ids)} random IDs verified")

# Step 6: Save tts_token_map.bin (red2orig for TTS)
np.asarray(new_layout, dtype=np.uint32).tofile(MAP_PATH)
print(f"\nSaved token_map.bin: {MAP_PATH} ({os.path.getsize(MAP_PATH)} bytes)")

# Summary
print(f"\n=== SUMMARY ===")
print(f"Full vocab: {VOCAB}")
print(f"Keep IDs: {len(keep_sorted)}")
print(f"New layout: {len(new_layout)}")
print(f"Pruned embed: {pruned.shape} = {pruned.nbytes/(1024*1024):.1f} MB")
print(f"Savings: {(full.nbytes - pruned.nbytes)/(1024*1024):.1f} MB")
