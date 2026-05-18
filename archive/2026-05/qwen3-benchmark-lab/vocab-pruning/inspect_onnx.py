"""Inspect ONNX for vocab-sized tensors (embed_tokens, lm_head)."""
import json, onnx, numpy as np, sys

BASE = "/home/harve/qwen3-vocab-pruning/source"

config = json.load(open(f"{BASE}/config.json"))
print("=== Config ===")
print(f"  vocab_size: {config.get('vocab_size')}")
print(f"  hidden_size: {config.get('hidden_size')}")

embed = np.fromfile(f"{BASE}/embed_tokens.bin", dtype=np.float16)
vocab = config.get("vocab_size", 151936)
hidden = config.get("hidden_size", 1024)
print(f"\n=== embed_tokens.bin ===")
print(f"  raw shape: {embed.shape}")
print(f"  expected: {vocab} * {hidden} = {vocab*hidden}")
print(f"  reshape to: ({vocab}, {hidden})")
embed_2d = embed.reshape(vocab, hidden)
print(f"  dtype: {embed_2d.dtype}")
print(f"  first few values: {embed_2d[0,:4]}")

print(f"\n=== Loading ONNX (this may take a moment for 1.2G)... ===")
sys.stdout.flush()
model = onnx.load(f"{BASE}/decoder_unified.onnx")
print(f"  ir_version: {model.ir_version}")
print(f"  producer: {model.producer_name} {model.producer_version}")
print(f"  nodes: {len(model.graph.node)}, initializers: {len(model.graph.initializer)}")

print(f"\n=== Vocab-sized initializers ===")
for i, init in enumerate(model.graph.initializer):
    shape = tuple(d.dim_value for d in init.dims)
    if len(shape) >= 2 and shape[0] >= 100000:
        print(f"  [#{i:3d}] {init.name!r:50s} shape={shape}")
    elif any(k in init.name.lower() for k in ["embed", "lm_head", "head", "output_proj"]):
        print(f"  [#{i:3d}] {init.name!r:50s} shape={shape}")

print(f"\n=== Graph inputs ===")
for inp in model.graph.input:
    shape = [str(d.dim_value) if d.dim_value > 0 else "?" for d in inp.type.tensor_type.shape.dim]
    print(f"  {inp.name!r}: [{','.join(shape)}]")

print(f"\n=== Graph outputs ===")
for out in model.graph.output:
    shape = [str(d.dim_value) if d.dim_value > 0 else "?" for d in out.type.tensor_type.shape.dim]
    print(f"  {out.name!r}: [{','.join(shape)}]")
