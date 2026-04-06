#!/usr/bin/env python3
"""
Python/numpy streaming reference implementation of the Zipformer encoder.

Traces the ONNX graph node-by-node and implements each operation exactly.
Validates against onnxruntime intermediate tensors at each step.

Architecture (from ONNX graph tracing):
  Per layer:
    1. x += ff1(x)                    -- feed_forward1 (no 0.5 scale in ONNX!)
    2. whiten = cumulative_mean(x)    -- BasicNorm with running mean
    3. x_attn_in = x + whiten_proj(whiten)  -- learned projection of whitened
    4. scores = self_attn(x_attn_in, pos_bias)  -- multi-head attention with pos
    5. x += out_proj(attn_out)        -- first attention output
    6. x += conv1(x)                  -- conv module 1
    7. x += ff2(x)                    -- feed_forward2
    8. v2 = whiten2_proj(x)           -- second value projection from post-ff2
    9. attn_out2 = reuse_softmax(scores, v2)  -- reuse softmax with V2
   10. x += out_proj2(attn_out2)      -- second attention output
   11. x += conv2(x)                  -- conv module 2
   12. x += ff3(x)                    -- feed_forward3
   13. x_normed = rms_norm(x)         -- norm_final
   14. bypass = x_normed - x_orig     -- bypass computation
   15. output = x_orig + bypass_scale * bypass

  NOTE: The FF scale of 0.5 is apparently baked into weights during export.
  NOTE: The V2 path reuses softmax scores from step 4.
  NOTE: whiten_proj and whiten2_proj are the "unmapped" MatMul weights.
"""
import numpy as np
import onnx
import onnx.numpy_helper
import onnxruntime as ort
import os
import json
import re

MODEL_PATH = "/tmp/sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16/encoder-epoch-99-avg-1.onnx"
WEIGHTS_DIR = "/tmp/jetson-voice-mte/rk3576/mte/weights"
INTERMEDIATES_DIR = "/tmp/jetson-voice-mte/rk3576/mte/reference/streaming_intermediates"

# Architecture constants
HIDDEN_DIM = 256
FFN_DIM = 768
KEY_DIM = 192
VAL_DIM = 96
POS_DIM = 16
IN_PROJ_DIM = KEY_DIM + KEY_DIM + VAL_DIM + POS_DIM  # 496
N_STACKS = 5
N_LAYERS = 2
NUM_HEADS = 4  # From Softmax shape [4, T_q, T_kv]
HEAD_DIM_K = KEY_DIM // NUM_HEADS  # 48
HEAD_DIM_V = VAL_DIM // NUM_HEADS  # 24
CONV_KERNEL = 31
PW1_DIM = 512
ENCODER_OUT_DIM = 512
LEFT_CTX = [64, 32, 16, 8, 32]
DS_FACTOR = [1, 2, 4, 8, 2]


def cos_sim(a, b):
    """Cosine similarity between two flattened arrays."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def load_ref(name):
    """Load reference intermediate tensor."""
    safe_name = name.replace("/", "__").replace(":", "_")
    path = os.path.join(INTERMEDIATES_DIR, f"{safe_name}.npy")
    if os.path.exists(path):
        return np.load(path)
    return None


def check(name, computed, tolerance="cos>0.999"):
    """Check computed tensor against ONNX reference."""
    ref = load_ref(name)
    if ref is None:
        print(f"  [{name}] NO REFERENCE")
        return
    cos = cos_sim(computed, ref)
    shape_match = computed.shape == ref.shape
    status = "OK" if cos > 0.999 else ("WARN" if cos > 0.99 else "FAIL")
    shape_str = f"shape={'MATCH' if shape_match else f'{computed.shape} vs {ref.shape}'}"
    print(f"  [{status}] {name}: cos={cos:.6f} {shape_str}")
    if not shape_match:
        print(f"    SHAPE MISMATCH: computed={computed.shape} ref={ref.shape}")
    return cos


def swooshr(x):
    """SwooshR activation: x * sigmoid(x - 1)"""
    return x * (1.0 / (1.0 + np.exp(-(x - 1.0))))


def swooshl(x):
    """SwooshL activation: x * sigmoid(x - 1) (same formula in ONNX export)"""
    return x * (1.0 / (1.0 + np.exp(-(x - 1.0))))


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def rms_norm(x, eps=6.1035e-05):
    """RMS normalization (BasicNorm / norm_final).
    From ONNX: x^2 -> mean -> + eps_adjusted -> pow(-0.5) -> x * scale
    The eps in ONNX graph appears as a large constant (~6.43) added to mean(x^2).
    Actually looking at the values:
      norm_final/Mul_output_0 = x^2
      norm_final/ReduceMean_output_0 = mean(x^2) ~ 0.046-0.080
      norm_final/Add_output_0 = mean(x^2) + 6.432362 ~ 6.478-6.513
      norm_final/Pow_output_0 = (mean(x^2) + 6.432362)^(-0.5) ~ 0.391-0.392

    So the "eps" is actually a large constant, not the typical 1e-5!
    This must be: RMSNorm with channel_size as the denominator offset.
    In icefall BasicNorm: x * (x.shape[-1] / (x^2.sum(-1) + eps))^0.5
    = x / sqrt(mean(x^2) + eps/dim)... let me check:
    = x * (dim / (sum(x^2) + eps))^0.5
    = x * (1 / (mean(x^2) + eps/dim))^0.5
    With dim=256, eps/dim would need to be ~6.43, so eps = 6.43*256 = 1646
    That's way too large. Let me re-examine.

    Actually: BasicNorm computes: x * (dim / (x^2.sum(-1, keepdim=True) + eps)).sqrt()
    = x * sqrt(dim) / sqrt(x^2.sum(-1) + eps)
    = x / sqrt(mean(x^2) + eps/dim) since mean(x^2) = sum(x^2)/dim

    In ONNX:
    step1: x^2
    step2: ReduceMean(x^2, axis=-1, keepdims=1) = mean(x^2)
    step3: mean(x^2) + CONSTANT
    step4: (mean(x^2) + CONSTANT)^(-0.5)
    step5: x * result

    The CONSTANT in the ONNX graph is what we need to find.
    From the output: Add_output_0 ~ 6.48 and ReduceMean ~ 0.047
    So CONSTANT ~ 6.48 - 0.047 = 6.433

    Wait, that's just 256 * 0.02513... Actually 6.4 ~= 256 * 0.025.
    Let me check: BasicNorm forward is: x * ((x.shape[-1]) / (x.square().sum(-1, keepdim=True) + eps)).sqrt()
    Rewriting: x * sqrt(dim / (sum(x^2) + eps))
    = x * sqrt(1 / (mean(x^2) + eps/dim))
    = x * (mean(x^2) + eps/dim)^(-0.5)

    So the constant = eps/dim. If eps=1646, then 1646/256 = 6.4296875
    But typical eps is much smaller. Let me check the icefall source...

    Actually looking at it more carefully, the ONNX export of BasicNorm uses:
    learn_eps: a learned parameter that's exponentiated.
    eps_value = exp(learn_eps) which can be any positive number.

    So the constant in the ONNX graph IS the learned eps.
    """
    # We need to extract the actual constant from the ONNX graph
    # For now, compute using the formula and match
    x_sq = x * x
    mean_sq = np.mean(x_sq, axis=-1, keepdims=True)
    # The constant from ONNX is approximately 6.432
    # We'll load it from the graph
    return x, mean_sq


class ZipformerWeights:
    """Load all weights from the extracted binary files."""

    def __init__(self, weights_dir):
        self.weights_dir = weights_dir
        self._cache = {}

    def _load_fp16(self, path):
        if path not in self._cache:
            data = np.fromfile(path, dtype=np.float16).astype(np.float32)
            self._cache[path] = data
        return self._cache[path]

    def _load_fp32(self, path):
        if path not in self._cache:
            data = np.fromfile(path, dtype=np.float32)
            self._cache[path] = data
        return self._cache[path]

    def get_matmul(self, stack, layer, name):
        """Get matmul weight and bias. Returns (weight [K,N], bias [N])."""
        layer_dir = os.path.join(self.weights_dir, f"stack{stack}_layer{layer}")
        w = self._load_fp16(os.path.join(layer_dir, f"{name}.fp16.bin"))
        b = self._load_fp32(os.path.join(layer_dir, f"{name}_bias.fp32.bin"))
        return w, b

    def get_matmul_weight_only(self, stack, layer, name):
        """Get matmul weight without bias (for unmapped weights)."""
        layer_dir = os.path.join(self.weights_dir, f"stack{stack}_layer{layer}")
        path = os.path.join(layer_dir, f"{name}.fp16.bin")
        return self._load_fp16(path)

    def get_conv(self, stack, layer, conv_idx):
        """Get conv module weights. Returns dict of weight arrays."""
        layer_dir = os.path.join(self.weights_dir, f"stack{stack}_layer{layer}")
        cn = f"conv{conv_idx + 1}"
        return {
            "pw1_weight": self._load_fp32(os.path.join(layer_dir, f"{cn}_pw1.fp32.bin")),
            "pw1_bias": self._load_fp32(os.path.join(layer_dir, f"{cn}_pw1_bias.fp32.bin")),
            "dw_weight": self._load_fp32(os.path.join(layer_dir, f"{cn}_dw.fp32.bin")),
            "dw_bias": self._load_fp32(os.path.join(layer_dir, f"{cn}_dw_bias.fp32.bin")),
            "pw2_weight": self._load_fp32(os.path.join(layer_dir, f"{cn}_pw2.fp32.bin")),
            "pw2_bias": self._load_fp32(os.path.join(layer_dir, f"{cn}_pw2_bias.fp32.bin")),
        }

    def get_bypass_scale(self, stack, layer):
        layer_dir = os.path.join(self.weights_dir, f"stack{stack}_layer{layer}")
        return self._load_fp32(os.path.join(layer_dir, "bypass_scale.fp32.bin"))[0]

    def get_encoder_proj(self):
        w = self._load_fp16(os.path.join(self.weights_dir, "encoder_proj.fp16.bin"))
        b = self._load_fp32(os.path.join(self.weights_dir, "encoder_proj_bias.fp32.bin"))
        return w, b

    def get_inter_stack(self, name):
        path = os.path.join(self.weights_dir, "inter_stack", f"{name}.fp32.bin")
        return self._load_fp32(path)


def extract_norm_eps_from_onnx():
    """Extract the learned eps constants from the ONNX model for each norm_final."""
    model = onnx.load(MODEL_PATH, load_external_data=False)
    graph = model.graph

    # Build init map
    init_data = {}
    for init in graph.initializer:
        init_data[init.name] = onnx.numpy_helper.to_array(init)

    # Find norm_final/Constant (the eps constant) and
    # the Mul_1 constant for whiten scaling
    constants = {}
    for i, node in enumerate(graph.node):
        if node.op_type == "Constant":
            name = node.output[0]
            for attr in node.attribute:
                if attr.name == "value":
                    tensor = onnx.numpy_helper.to_array(attr.t)
                    constants[name] = tensor

    # Find the norm_final eps values by looking at Add nodes in norm_final
    eps_values = []
    whiten_mul_constants = []

    for i, node in enumerate(graph.node):
        out = node.output[0] if node.output else ""

        # norm_final Add: adds eps to mean(x^2)
        if node.op_type == "Add" and "norm_final/Add_output" in out:
            for inp in node.input:
                if inp in constants:
                    val = constants[inp]
                    eps_values.append(float(val))
                    print(f"  norm_final eps at node {i}: {float(val):.6f}")

        # Whiten Mul_1: the scaling constant
        if node.op_type == "Mul" and "Mul_1_output" in out:
            for inp in node.input:
                if inp in constants:
                    val = constants[inp]
                    if val.size == 1:
                        whiten_mul_constants.append(float(val))
                        # print(f"  whiten mul constant at node {i}: {float(val):.6f}")

    return eps_values, whiten_mul_constants, constants


def extract_pos_encoding():
    """Extract the positional encoding tensor from the ONNX model."""
    # The pos encoding is a Slice from a precomputed table
    # /encoder_pos/Slice_output_0 shape [1, 95, 256]
    ref = load_ref("/encoder_pos/Slice_output_0")
    if ref is not None:
        return ref
    # If not available, we need to extract from the model
    return None


def linear(x, weight, bias=None):
    """Linear projection: y = x @ weight + bias.
    weight: [K, N], x: [..., K] -> [..., N]
    """
    K, N = weight.shape[0], weight.shape[-1]
    weight_2d = weight.reshape(K, N)
    orig_shape = x.shape
    x_2d = x.reshape(-1, K)
    out = x_2d @ weight_2d
    if bias is not None:
        out += bias
    out_shape = list(orig_shape[:-1]) + [N]
    return out.reshape(out_shape)


def ff_module(x, w_in, b_in, w_out, b_out):
    """Feed-forward module: out_proj(swooshr(in_proj(x)))"""
    # in_proj: [*, 256] -> [*, 768]
    in_dim = x.shape[-1]
    out_dim_in = w_in.size // in_dim
    h = linear(x, w_in.reshape(in_dim, out_dim_in), b_in)
    # SwooshR
    h = swooshr(h)
    # out_proj: [*, 768] -> [*, 256]
    in_dim_out = h.shape[-1]
    out_dim_out = w_out.size // in_dim_out
    out = linear(h, w_out.reshape(in_dim_out, out_dim_out), b_out)
    return out


def conv_module_streaming(x, conv_weights, conv_cache):
    """Conv module with streaming (causal) support.

    x: [T, 1, 256] (the current layer activations)
    conv_cache: [1, 256, 30] (previous frames for causal conv)
    Returns: (output [T, 1, 256], new_cache [1, 256, 30])
    """
    T = x.shape[0]
    # Transpose to conv format: [T, 1, 256] -> [1, 256, T]
    x_conv = x.transpose(1, 2, 0)  # [1, 256, T]

    # 1. Pointwise conv1: [1, 256, T] -> [1, 512, T]
    pw1_w = conv_weights["pw1_weight"].reshape(PW1_DIM, HIDDEN_DIM, 1)
    pw1_b = conv_weights["pw1_bias"].reshape(PW1_DIM, 1)
    pw1_out = np.zeros((1, PW1_DIM, T), dtype=np.float32)
    for t in range(T):
        for o in range(PW1_DIM):
            pw1_out[0, o, t] = np.dot(pw1_w[o, :, 0], x_conv[0, :, t]) + conv_weights["pw1_bias"][o]

    # 2. GLU: split [1, 512, T] -> [1, 256, T] * sigmoid([1, 256, T])
    a = pw1_out[:, :HIDDEN_DIM, :]
    b = pw1_out[:, HIDDEN_DIM:, :]
    glu_out = a * (1.0 / (1.0 + np.exp(-b)))  # [1, 256, T]

    # 3. Concat with cache for causal conv
    # conv_cache: [1, 256, 30], glu_out: [1, 256, T]
    padded = np.concatenate([conv_cache, glu_out], axis=2)  # [1, 256, 30+T]

    # New cache: last 30 frames
    new_cache = padded[:, :, -CONV_KERNEL + 1:]  # [1, 256, 30]

    # 4. Depthwise conv1d: kernel=31, groups=256, no padding (causal already padded)
    dw_w = conv_weights["dw_weight"].reshape(HIDDEN_DIM, 1, CONV_KERNEL)
    dw_b = conv_weights["dw_bias"]
    dw_out = np.zeros((1, HIDDEN_DIM, T), dtype=np.float32)
    for c in range(HIDDEN_DIM):
        for t in range(T):
            t_start = t  # in padded: position t to t+31
            val = 0.0
            for k in range(CONV_KERNEL):
                val += padded[0, c, t_start + k] * dw_w[c, 0, k]
            dw_out[0, c, t] = val + dw_b[c]

    # 5. SwooshL activation (same as SwooshR in ONNX export)
    dw_out = swooshl(dw_out)

    # 6. Pointwise conv2: [1, 256, T] -> [1, 256, T]
    pw2_w = conv_weights["pw2_weight"].reshape(HIDDEN_DIM, HIDDEN_DIM, 1)
    pw2_out = np.zeros((1, HIDDEN_DIM, T), dtype=np.float32)
    for t in range(T):
        for o in range(HIDDEN_DIM):
            pw2_out[0, o, t] = np.dot(pw2_w[o, :, 0], dw_out[0, :, t]) + conv_weights["pw2_bias"][o]

    # Transpose back: [1, 256, T] -> [T, 1, 256]
    output = pw2_out.transpose(2, 0, 1)  # [T, 1, 256]

    return output, new_cache


def run_layer_0(weights, pos_encoding, norm_eps, whiten_const):
    """Run stack 0, layer 0 and compare with ONNX intermediates."""
    stack, layer = 0, 0
    print(f"\n{'='*80}")
    print(f"Stack {stack} Layer {layer}")
    print(f"{'='*80}")

    # Load layer input (post-embed output)
    x = load_ref("/Transpose_output_0")  # [16, 1, 256]
    if x is None:
        print("ERROR: Cannot load layer input!")
        return
    T = x.shape[0]
    print(f"Layer input: shape={x.shape}")

    # Save original for bypass
    x_orig = x.copy()

    # === 1. Feed-Forward 1 ===
    print("\n--- FF1 ---")
    w_in, b_in = weights.get_matmul(stack, layer, "ff1_in")
    w_out, b_out = weights.get_matmul(stack, layer, "ff1_out")

    # in_proj
    ff1_in = linear(x, w_in.reshape(HIDDEN_DIM, FFN_DIM), b_in)
    check("/feed_forward1/in_proj/Add_output_0", ff1_in)

    # SwooshR
    ff1_act = swooshr(ff1_in)
    check("/feed_forward1/activation/Mul_output_0", ff1_act)

    # out_proj
    ff1_out = linear(ff1_act, w_out.reshape(FFN_DIM, HIDDEN_DIM), b_out)
    check("/feed_forward1/out_proj/Add_output_0", ff1_out)

    # Residual (NO 0.5 scale in ONNX!)
    x = x + ff1_out
    check("/Add_output_0", x)

    # === 2. Whiten (BasicNorm with running mean) ===
    print("\n--- Whiten ---")
    # ONNX whiten computation for first chunk (cached_len=0, cached_avg=0):
    # CumSum(x, axis=0) -> cumulative sum along T
    # Range(0, T) + cached_len -> frame indices [0,1,...,T-1] (first chunk)
    # reciprocal = 1 / (frame_index + 1) -> [1, 1/2, 1/3, ..., 1/T]
    # multiply by whiten_const (=1.0 from ONNX)
    # whitened = cumsum * reciprocal -> running average

    cumsum = np.cumsum(x, axis=0)
    check("/CumSum_output_0", cumsum)

    # Frame indices: for first chunk, cached_len=0, so indices are [1, 2, ..., T]
    # Actually from ONNX: Range(0, T+cached_len) + cached_len... let me check
    # /Range_output_0: Range(0, cast(cached_len + T), 1) = [0, 1, 2, ..., T-1]
    # /Add_3_output_0 = Range + cached_len = [0+0, 1+0, ..., (T-1)+0] = [0,1,...,T-1]
    # /Cast_2_output_0 = cast to float = [0.0, 1.0, ..., 15.0]
    # /Reciprocal_output_0 = 1 / [0,1,...,15] = [inf, 1.0, 0.5, ..., 1/15]
    # Wait, 1/0 = inf? No, let me re-check...
    # Actually: Add_3 = Range + cached_len, and Range starts from 0 when cached_len=0
    # BUT there might be an offset. Let me check the Reciprocal output:
    # /Reciprocal_output_0: range=[0.062500, 1.000000] = [1/16, 1/1]
    # So indices are [1, 2, ..., 16], not [0, 1, ..., 15]!
    # Check: 1/16 = 0.0625, 1/1 = 1.0. Yes!
    # So the formula is: index = Range(1, T+1) + cached_len

    cached_len = 0  # first chunk
    indices = np.arange(1, T + 1, dtype=np.float32) + cached_len  # [1, 2, ..., 16]
    reciprocal = 1.0 / indices  # [1, 0.5, 0.333, ..., 0.0625]
    check("/Reciprocal_output_0", reciprocal.reshape(T, 1))

    # Multiply by whiten constant (should be 1.0 for first layer)
    # /Mul_1_output_0: reciprocal * constant
    # From check: Mul_1 range matches Reciprocal, so constant is 1.0
    scaled_recip = reciprocal * whiten_const  # still [T]
    check("/Mul_1_output_0", scaled_recip.reshape(T, 1))

    # Whitened = cumsum * reciprocal (broadcast over dim 256)
    whitened = cumsum * scaled_recip.reshape(T, 1, 1)
    check("/Mul_2_output_0", whitened)

    # Whiten projection: whitened @ whiten_proj [256, 256]
    w_whiten = weights.get_matmul_weight_only(stack, layer, "attn_whiten")
    whiten_proj_out = linear(whitened, w_whiten.reshape(HIDDEN_DIM, HIDDEN_DIM))
    check("/proj/MatMul_output_0", whiten_proj_out)

    # Add whiten projection to residual
    x_attn_in = x + whiten_proj_out
    check("/Add_5_output_0", x_attn_in)

    # === 3. Self-Attention ===
    print("\n--- Self-Attention ---")

    # in_proj: [T, 1, 256] -> [T, 1, 496]
    w_attn_in, b_attn_in = weights.get_matmul(stack, layer, "attn_in")
    in_proj_out = linear(x_attn_in, w_attn_in.reshape(HIDDEN_DIM, IN_PROJ_DIM), b_attn_in)
    check("/in_proj/Add_output_0", in_proj_out)

    # Split: Q[T,1,192], K[T,1,192], V[T,1,96], pos[T,1,16]
    Q = in_proj_out[:, :, :KEY_DIM]      # [T, 1, 192]
    K = in_proj_out[:, :, KEY_DIM:2*KEY_DIM]  # [T, 1, 192]
    V = in_proj_out[:, :, 2*KEY_DIM:2*KEY_DIM+VAL_DIM]  # [T, 1, 96]
    pos = in_proj_out[:, :, 2*KEY_DIM+VAL_DIM:]  # [T, 1, 16]

    check("/Slice_output_0", Q)
    check("/Slice_1_output_0", K)
    check("/Slice_2_output_0", V)
    check("/Slice_3_output_0", pos)

    # Positional encoding: linear_pos maps pos_encoding [1, S, 256] -> [1, S, 16]
    # where S = total_left_ctx + T = 64 + 16 = 80... no, S = 95 from ref
    # Actually from ONNX: /encoder_pos/Slice_output_0 [1, 95, 256] -> linear_pos -> [1, 95, 16]
    w_pos = weights.get_matmul_weight_only(stack, layer, "attn_pos_bias")
    pos_proj = linear(pos_encoding, w_pos.reshape(HIDDEN_DIM, POS_DIM))
    check("/linear_pos/MatMul_output_0", pos_proj)

    # Multi-head attention
    # Reshape Q, K, V for multi-head:
    # Q: [T, 1, 192] -> [T, 1, 4, 48] -> [1, 4, T, 48]... actually let me check ONNX shapes
    # /Reshape_output_0 (Slice_output_0 reshaped): this is Q reshaped
    # From ONNX: Reshape to [T_q, N, num_heads, head_dim_k] then Transpose [1, 2, 0, 3]
    # -> [N, num_heads, T_q, head_dim_k]

    # Cached keys/values: for first chunk, cache is zeros
    # /Concat_output_0: [cached_key, K] along axis=0 -> [64+16, 1, 192] = [80, 1, 192]
    left_ctx = LEFT_CTX[stack]  # 64
    cached_key = np.zeros((left_ctx, 1, KEY_DIM), dtype=np.float32)
    cached_val = np.zeros((left_ctx, 1, VAL_DIM), dtype=np.float32)
    cached_val2 = np.zeros((left_ctx, 1, VAL_DIM), dtype=np.float32)

    # Concat cache + current
    K_full = np.concatenate([cached_key, K], axis=0)  # [80, 1, 192]
    V_full = np.concatenate([cached_val, V], axis=0)  # [80, 1, 96]

    # Slice to keep last left_ctx frames (after cache update)
    # /Slice_4_output_0: new_cached_key = Concat_output_0[-left_ctx:] = [64, 1, 192]
    # But for attention we use the full [80, 1, 192]
    T_kv = K_full.shape[0]  # 80

    # Reshape for multi-head:
    # Q: [T_q, N, 192] -> [T_q, N, num_heads, head_dim_k] -> transpose [N, num_heads, T_q, head_dim_k]
    Q_mh = Q.reshape(T, 1, NUM_HEADS, HEAD_DIM_K).transpose(1, 2, 0, 3)  # [1, 4, 16, 48]
    check("/Reshape_output_0", Q.reshape(T, 1, NUM_HEADS, HEAD_DIM_K))

    # pos: [T_q, N, 16] -> [T_q, N, num_heads, pos_head_dim=4] -> transpose [N, num_heads, T_q, pos_head_dim]
    pos_head_dim = POS_DIM // NUM_HEADS  # 4
    pos_mh = pos.reshape(T, 1, NUM_HEADS, pos_head_dim).transpose(1, 2, 0, 3)  # [1, 4, 16, 4]
    check("/Reshape_1_output_0", pos.reshape(T, 1, NUM_HEADS, pos_head_dim))

    # K_full: [T_kv, N, 192] -> [T_kv, N, num_heads, head_dim_k] -> transpose [N, num_heads, head_dim_k, T_kv]
    K_mh = K_full.reshape(T_kv, 1, NUM_HEADS, HEAD_DIM_K).transpose(1, 2, 3, 0)  # [1, 4, 48, 80]
    check("/Reshape_2_output_0", K_full.reshape(T_kv, 1, NUM_HEADS, HEAD_DIM_K))

    # V_full: [T_kv, N, 96] -> reshape for multi-head value
    # V: [T_kv, N, num_heads*val_head_dim] -> [T_kv, num_heads*val_head_dim, N] -> transpose
    # Actually from ONNX: V is reshaped to [T_kv, mul_heads*head_dim_v, N] then Transpose [1, 0, 2]
    # /Reshape_3_output_0: V_full -> reshape to [T_kv, num_heads*head_dim_v, N]
    # Hmm wait, V_full shape is [80, 1, 96]. The ONNX reshape is:
    # Concat_5 target shape: [T_kv, num_heads*head_dim_v, N] = [80, 96, 1]? No...
    # Let me check: V_full [80, 1, 96], and target shape uses N=1, dim=96
    # Actually: Reshape_3 target = [Add_13, Mul_8, Constant_79]
    # /Add_13_output_0 = Gather_21 + Gather_19 where Gather_21 = left_ctx shape[0], Gather_19 = T_q
    # Hmm this is getting complex. Let me just check from ONNX output.
    # /Reshape_3_output_0 -> check shape in intermediates

    # Compute Q @ K^T: [1, 4, 16, 48] @ [1, 4, 48, 80] = [1, 4, 16, 80]
    qk_scores = Q_mh @ K_mh  # [1, 4, 16, 80]
    check("/MatMul_1_output_0", qk_scores)

    # === Positional Bias (self-contained) ===
    # pos_proj: pos_encoding [1, S, 256] @ W_pos [256, 16] -> [1, S, 16]
    # where S = T_q + T_kv - 1 = 16 + 80 - 1 = 95
    S = pos_proj.shape[1]  # 95

    # Reshape pos_proj for multi-head: [1, S, 16] -> [1, S, 4, 4] -> transpose [0,2,3,1] -> [1, 4, 4, S]
    pos_proj_mh = pos_proj.reshape(1, S, NUM_HEADS, pos_head_dim).transpose(0, 2, 3, 1)  # [1, 4, 4, 95]

    # pos_query @ pos_proj_mh: [1, 4, 16, 4] @ [1, 4, 4, 95] = [1, 4, 16, 95]
    raw_pos_scores = pos_mh @ pos_proj_mh  # [1, 4, 16, 95]

    # Create relative position index matrix: index[q, kv] = (T_q - 1 - q) + kv
    # This maps relative positions to indices in the pos encoding table
    q_indices = np.arange(T - 1, -1, -1)  # [T_q-1, T_q-2, ..., 0]
    kv_indices = np.arange(T_kv)  # [0, 1, ..., T_kv-1]
    # Tile q_indices across all heads: [num_heads * T_q]
    q_tiled = np.tile(q_indices, NUM_HEADS)  # [64]
    # pos_index[i, j] = q_tiled[i] + kv_indices[j]
    pos_index = q_tiled[:, None] + kv_indices[None, :]  # [64, 80]

    # Flatten raw_pos_scores from [1, 4, 16, 95] to [64, 95]
    raw_flat = raw_pos_scores.reshape(NUM_HEADS * T, S)

    # Gather: extract pos_bias[i, j] = raw_flat[i, pos_index[i, j]]
    pos_bias_flat = np.take_along_axis(raw_flat, pos_index, axis=1)  # [64, 80]

    # Reshape to [1, 4, 16, 80]
    pos_bias = pos_bias_flat.reshape(1, NUM_HEADS, T, T_kv)
    check("/Reshape_6_output_0", pos_bias)

    # QK + pos_bias
    qk_with_pos = qk_scores + pos_bias  # [1, 4, 16, 80]
    check("/Add_8_output_0", qk_with_pos)

    # Reshape for softmax: [1, 4, 16, 80] -> [4, 16, 80]
    scores_pre_sm = qk_with_pos.reshape(NUM_HEADS, T, T_kv)

    # Softmax
    scores = softmax(scores_pre_sm, axis=-1)
    check("/Softmax_output_0", scores)

    # Weighted V: scores @ V
    # V_full [T_kv, 1, 96] -> reshape [T_kv, num_heads, head_dim_v] -> [num_heads, T_kv, head_dim_v]
    V_mh = V_full.reshape(T_kv, NUM_HEADS, HEAD_DIM_V).transpose(1, 0, 2)  # [4, 80, 24]
    attn_out = scores @ V_mh  # [4, 16, 24]
    check("/MatMul_2_output_0", attn_out)

    # Transpose and reshape attn output for out_proj
    # /Transpose_6_output_0 perm=[1,0,2] from MatMul_2 [4,16,24] -> [16,4,24]
    # /Reshape_8_output_0 -> [T_q, N, val_dim] = [16, 1, 96]
    attn_out_reshaped = attn_out.transpose(1, 0, 2).reshape(T, 1, VAL_DIM)  # [16, 1, 96]
    check("/Reshape_8_output_0", attn_out_reshaped)

    # out_proj: [T, 1, 96] -> [T, 1, 256]
    w_out, b_out = weights.get_matmul(stack, layer, "attn_out")
    attn_proj = linear(attn_out_reshaped, w_out.reshape(VAL_DIM, HIDDEN_DIM), b_out)
    check("/Add_9_output_0", attn_proj)

    # Attention residual
    x = x_attn_in + attn_proj
    check("/Add_10_output_0", x)

    # === 4. Conv Module 1 ===
    print("\n--- Conv1 ---")
    conv1_weights = weights.get_conv(stack, layer, 0)
    conv1_cache = np.zeros((1, HIDDEN_DIM, CONV_KERNEL - 1), dtype=np.float32)
    conv1_out, new_conv1_cache = conv_module_streaming(x, conv1_weights, conv1_cache)
    check("/Transpose_8_output_0", conv1_out)

    # Conv1 residual
    x = x + conv1_out
    check("/Add_11_output_0", x)

    # === 5. Feed-Forward 2 ===
    print("\n--- FF2 ---")
    w_in, b_in = weights.get_matmul(stack, layer, "ff2_in")
    w_out, b_out = weights.get_matmul(stack, layer, "ff2_out")
    ff2_out = ff_module(x, w_in, b_in, w_out, b_out)
    check("/feed_forward2/out_proj/Add_output_0", ff2_out)

    x = x + ff2_out
    check("/Add_12_output_0", x)

    # === 6. V2 path (whiten2 + reuse softmax) ===
    print("\n--- V2 path (whiten2 + reuse softmax) ---")
    # in_proj2: x [T, 1, 256] -> [T, 1, 96] using whiten2 weight
    w_whiten2 = weights.get_matmul_weight_only(stack, layer, "attn_whiten2")
    v2_proj = linear(x, w_whiten2.reshape(HIDDEN_DIM, VAL_DIM))
    check("/in_proj2/MatMul_output_0", v2_proj)

    # Concat with cached_val2
    V2_full = np.concatenate([cached_val2, v2_proj], axis=0)  # [80, 1, 96]

    # Reshape V2 for multi-head: [80, 1, 96] -> [4, 80, 24]
    V2_mh = V2_full.reshape(T_kv, NUM_HEADS, HEAD_DIM_V).transpose(1, 0, 2)  # [4, 80, 24]

    # Reuse softmax scores from step 3
    attn_out2 = scores @ V2_mh  # [4, 16, 80] @ [4, 80, 24] = [4, 16, 24]
    check("/MatMul_4_output_0", attn_out2)

    # Reshape: [4, 16, 24] -> [16, 4, 24] -> [16, 1, 96]
    attn_out2_reshaped = attn_out2.transpose(1, 0, 2).reshape(T, 1, VAL_DIM)

    # out_proj2: [T, 1, 96] -> [T, 1, 256]
    w_out2, b_out2 = weights.get_matmul(stack, layer, "attn_out2")
    attn_proj2 = linear(attn_out2_reshaped, w_out2.reshape(VAL_DIM, HIDDEN_DIM), b_out2)
    check("/out_proj2/Add_output_0", attn_proj2)

    # V2 residual
    x = x + attn_proj2
    check("/Add_14_output_0", x)

    # === 7. Conv Module 2 ===
    print("\n--- Conv2 ---")
    conv2_weights = weights.get_conv(stack, layer, 1)
    conv2_cache = np.zeros((1, HIDDEN_DIM, CONV_KERNEL - 1), dtype=np.float32)
    conv2_out, new_conv2_cache = conv_module_streaming(x, conv2_weights, conv2_cache)

    x = x + conv2_out
    check("/Add_15_output_0", x)

    # === 8. Feed-Forward 3 ===
    print("\n--- FF3 ---")
    w_in, b_in = weights.get_matmul(stack, layer, "ff3_in")
    w_out, b_out = weights.get_matmul(stack, layer, "ff3_out")
    ff3_out = ff_module(x, w_in, b_in, w_out, b_out)
    check("/feed_forward3/out_proj/Add_output_0", ff3_out)

    x = x + ff3_out
    check("/Add_16_output_0", x)

    # === 9. Norm Final + Bypass ===
    print("\n--- Norm Final + Bypass ---")

    # RMS norm: x * (mean(x^2) + eps)^(-0.5)
    x_sq = x * x
    check("/norm_final/Mul_output_0", x_sq)

    mean_sq = np.mean(x_sq, axis=-1, keepdims=True)
    check("/norm_final/ReduceMean_output_0", mean_sq)

    eps_const = norm_eps[0] if norm_eps else 6.432
    mean_sq_eps = mean_sq + eps_const
    check("/norm_final/Add_output_0", mean_sq_eps)

    scale = np.power(mean_sq_eps, -0.5)
    check("/norm_final/Pow_output_0", scale)

    x_normed = x * scale
    check("/norm_final/Mul_1_output_0", x_normed)

    # Bypass: output = x_orig + bypass_scale * (x_normed - x_orig)
    bypass = x_normed - x_orig
    check("/Sub_2_output_0", bypass)

    bypass_scale = weights.get_bypass_scale(stack, layer)
    print(f"  bypass_scale = {bypass_scale:.6f}")
    bypass_scaled = bypass * bypass_scale
    check("/Mul_10_output_0", bypass_scaled)

    output = x_orig + bypass_scaled
    check("/Add_17_output_0", output)

    return output


def main():
    print("Loading weights...")
    weights = ZipformerWeights(WEIGHTS_DIR)

    print("Extracting norm eps values from ONNX model...")
    norm_eps, whiten_consts, all_constants = extract_norm_eps_from_onnx()
    print(f"  norm_eps values: {norm_eps[:5]}...")
    print(f"  whiten constants: {whiten_consts[:5]}...")

    print("Loading positional encoding...")
    pos_encoding = extract_pos_encoding()
    if pos_encoding is not None:
        print(f"  pos_encoding shape: {pos_encoding.shape}")

    # Run layer 0
    whiten_const = whiten_consts[0] if whiten_consts else 1.0
    print(f"\nUsing whiten_const = {whiten_const}")
    output = run_layer_0(weights, pos_encoding, norm_eps, whiten_const)

    if output is not None:
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Layer 0 output shape: {output.shape}")
        print(f"Layer 0 output range: [{output.min():.6f}, {output.max():.6f}]")


if __name__ == "__main__":
    main()
