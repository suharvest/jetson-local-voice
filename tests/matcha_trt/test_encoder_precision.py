"""Task E — TRT vs ORT-CPU precision compare for Matcha encoder.

Same z0 noise (seed=42), same tokens, run encoder via:
  - ORT-CPU on /tmp/matcha_surgery/matcha_encoder_trt.onnx
  - TRT engine at /opt/models/matcha-icefall-zh-en/engines/matcha_encoder_bf16.engine

Compare mu / mask / z0_passthrough L2 to validate TRT encoder precision.
"""
import sys, os, json
sys.path.insert(0, "/opt/speech/app")
import numpy as np
import onnxruntime as ort
import tensorrt as trt
from cuda import cudart

SURGERY_DIR = "/tmp/matcha_surgery"
ENGINE_DIR  = "/opt/models/matcha-icefall-zh-en/engines"
SAMPLE_TEXTS = [
    ("zh_short",  "今天天气不错"),
    ("zh_long",   "重庆银行行长正在开会"),
    ("en_short",  "Hello world"),
    ("en_long",   "The quick brown fox jumps over the lazy dog"),
    ("mix",       "Wi-Fi 已连接"),
]

# Load tokenizer from backend
from app.backends.jetson.matcha_trt import MatchaTRTBackend
backend = MatchaTRTBackend.__new__(MatchaTRTBackend)
backend._lexicon = {}
backend._token_to_id = {}
backend._load_lexicon()


def chk(err):
    if err != 0:
        raise RuntimeError(f"cuda error: {err}")


# --- ORT-CPU encoder
ort_enc = ort.InferenceSession(f"{SURGERY_DIR}/matcha_encoder_trt.onnx",
                                providers=["CPUExecutionProvider"])
print("ORT inputs:", [(i.name, i.shape, i.type) for i in ort_enc.get_inputs()])
print("ORT outputs:", [(o.name, o.shape) for o in ort_enc.get_outputs()])


# --- TRT encoder
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(TRT_LOGGER)
with open(f"{ENGINE_DIR}/matcha_encoder_bf16.engine", "rb") as f:
    trt_enc = runtime.deserialize_cuda_engine(f.read())
trt_ctx = trt_enc.create_execution_context()
err, stream = cudart.cudaStreamCreate(); chk(err)


def trt_encoder_run(inputs):
    """inputs: dict of name → np.ndarray. Returns dict of output name → np.ndarray."""
    n_io = trt_enc.num_io_tensors
    bufs = {}
    for i in range(n_io):
        name = trt_enc.get_tensor_name(i)
        shape = trt_enc.get_tensor_shape(name)
        dtype = trt.nptype(trt_enc.get_tensor_dtype(name))
        if trt_enc.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            arr = inputs[name].astype(dtype)
            err, ptr = cudart.cudaMalloc(arr.nbytes); chk(err)
            err = cudart.cudaMemcpy(ptr, arr.ctypes.data, arr.nbytes,
                                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)[0]; chk(err)
            bufs[name] = (ptr, arr)
        else:
            host = np.zeros(shape, dtype=dtype)
            err, ptr = cudart.cudaMalloc(host.nbytes); chk(err)
            bufs[name] = (ptr, host)
        trt_ctx.set_tensor_address(name, bufs[name][0])
    trt_ctx.execute_async_v3(stream); cudart.cudaStreamSynchronize(stream)
    outputs = {}
    for i in range(n_io):
        name = trt_enc.get_tensor_name(i)
        if trt_enc.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            ptr, host = bufs[name]
            err = cudart.cudaMemcpy(host.ctypes.data, ptr, host.nbytes,
                                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)[0]; chk(err)
            outputs[name] = host
    for ptr, _ in bufs.values():
        cudart.cudaFree(ptr)
    return outputs


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-9))


print("\n=== Task E encoder precision (TRT BF16/native vs ORT-CPU FP32) ===")
print(f"{'tag':12s} {'tokens':>6s}  {'mu_L2':>8s}  {'mask_L2':>8s}  {'z0p_L2':>8s}  {'mu_max':>10s}")
results = []
rng = np.random.default_rng(42)
for tag, text in SAMPLE_TEXTS:
    tokens = backend._text_to_tokens(text)[:80]
    n = len(tokens)
    x = np.zeros((1, 80), dtype=np.int32); x[0, :n] = tokens
    x_length = np.array([n], dtype=np.int32)
    noise_scale = np.array([0.667], dtype=np.float32)
    length_scale = np.array([1.0], dtype=np.float32)
    z0_noise = (rng.standard_normal((1, 80, 600)).astype(np.float32) * 0.667).astype(np.float32)

    inputs = {
        "noise_scale": noise_scale, "length_scale": length_scale,
        "z0_noise": z0_noise, "x": x, "x_length": x_length,
    }
    ort_out = ort_enc.run(None, inputs)
    out_names = [o.name for o in ort_enc.get_outputs()]
    ort_dict = dict(zip(out_names, ort_out))

    trt_dict = trt_encoder_run(inputs)
    # Map TRT output names to ORT names (they may differ slightly)
    # Both should have keys like /Transpose_3_output_0 etc
    common = set(ort_dict.keys()) & set(trt_dict.keys())

    mu_name = next((k for k in common if "Transpose" in k or "mu" in k.lower()), None)
    mask_name = next((k for k in common if "Cast" in k or "mask" in k.lower()), None)
    z0p_name = next((k for k in common if "Mul" in k or "z0" in k.lower()), None)

    if not mu_name or not mask_name or not z0p_name:
        print(f"[{tag}] output name mismatch. ORT={list(ort_dict.keys())} TRT={list(trt_dict.keys())}")
        continue

    mu_l2 = rel_l2(trt_dict[mu_name], ort_dict[mu_name])
    mask_l2 = rel_l2(trt_dict[mask_name], ort_dict[mask_name])
    z0p_l2 = rel_l2(trt_dict[z0p_name], ort_dict[z0p_name])
    mu_max = float(np.abs(trt_dict[mu_name] - ort_dict[mu_name]).max())
    # Where is the divergence concentrated in time?
    diff = np.abs(trt_dict[mu_name] - ort_dict[mu_name])  # [1,80,600]
    per_t = diff.mean(axis=(0, 1))  # [600]
    mask_ort = ort_dict[mask_name][0, 0, :]
    valid_t = int(mask_ort.sum() + 0.5)
    inside_diff = per_t[:valid_t].mean() if valid_t > 0 else 0
    outside_diff = per_t[valid_t:].mean() if valid_t < 600 else 0
    print(f"{tag:12s} tok={n:3d} valid_t={valid_t:3d}  mu_L2={mu_l2:.4f}  mu_max={mu_max:.3f}  inside={inside_diff:.5f} outside={outside_diff:.5f}  tokens0..5={tokens[:6]}")
    results.append({
        "tag": tag, "text": text, "tokens": n,
        "mu_l2": mu_l2, "mask_l2": mask_l2, "z0p_l2": z0p_l2, "mu_max": mu_max,
    })

print("\n=== SUMMARY ===")
if results:
    mu_l2s = [r["mu_l2"] for r in results]
    print(f"mu L2:    mean={np.mean(mu_l2s):.6f}  max={np.max(mu_l2s):.6f}")
    print(f"spec target < 0.05; verdict: {'PASS' if max(mu_l2s) < 0.05 else 'FAIL'}")

with open("/tmp/task_e_results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Wrote /tmp/task_e_results.json")
