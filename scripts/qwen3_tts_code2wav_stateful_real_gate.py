#!/usr/bin/env python3
"""Stateful streaming reference gate for the product Qwen3-TTS Code2Wav.

This covers the actual tokenizer-12Hz decoder exported to ONNX, not the older
Qwen3-Omni standalone Code2Wav reference.  It reconstructs a PyTorch decoder
from the ONNX initializers, then compares full forward with an online chunked
forward that carries attention, causal-conv, and transposed-conv state.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
import importlib.util
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper
from safetensors import safe_open


Qwen3TTSTokenizerV2DecoderConfig = None
Qwen3TTSTokenizerV2CausalConvNet = None
Qwen3TTSTokenizerV2CausalTransConvNet = None
Qwen3TTSTokenizerV2ConvNeXtBlock = None
Qwen3TTSTokenizerV2Decoder = None
Qwen3TTSTokenizerV2DecoderDecoderBlock = None
Qwen3TTSTokenizerV2DecoderDecoderResidualUnit = None
SnakeBeta = None
apply_rotary_pos_emb = None


def _stub_package(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_qwen_tts_symbols(qwen_tts_root: Path) -> None:
    """Load tokenizer_12hz modules without executing qwen_tts package __init__."""

    global Qwen3TTSTokenizerV2DecoderConfig
    global Qwen3TTSTokenizerV2CausalConvNet
    global Qwen3TTSTokenizerV2CausalTransConvNet
    global Qwen3TTSTokenizerV2ConvNeXtBlock
    global Qwen3TTSTokenizerV2Decoder
    global Qwen3TTSTokenizerV2DecoderDecoderBlock
    global Qwen3TTSTokenizerV2DecoderDecoderResidualUnit
    global SnakeBeta
    global apply_rotary_pos_emb

    core_root = qwen_tts_root / "core"
    tokenizer_root = core_root / "tokenizer_12hz"
    _stub_package("qwen_tts", qwen_tts_root)
    _stub_package("qwen_tts.core", core_root)
    _stub_package("qwen_tts.core.tokenizer_12hz", tokenizer_root)

    cfg = _load_module(
        "qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2",
        tokenizer_root / "configuration_qwen3_tts_tokenizer_v2.py",
    )
    modeling = _load_module(
        "qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2",
        tokenizer_root / "modeling_qwen3_tts_tokenizer_v2.py",
    )
    Qwen3TTSTokenizerV2DecoderConfig = cfg.Qwen3TTSTokenizerV2DecoderConfig
    Qwen3TTSTokenizerV2CausalConvNet = modeling.Qwen3TTSTokenizerV2CausalConvNet
    Qwen3TTSTokenizerV2CausalTransConvNet = modeling.Qwen3TTSTokenizerV2CausalTransConvNet
    Qwen3TTSTokenizerV2ConvNeXtBlock = modeling.Qwen3TTSTokenizerV2ConvNeXtBlock
    Qwen3TTSTokenizerV2Decoder = modeling.Qwen3TTSTokenizerV2Decoder
    Qwen3TTSTokenizerV2DecoderDecoderBlock = modeling.Qwen3TTSTokenizerV2DecoderDecoderBlock
    Qwen3TTSTokenizerV2DecoderDecoderResidualUnit = modeling.Qwen3TTSTokenizerV2DecoderDecoderResidualUnit
    SnakeBeta = modeling.SnakeBeta
    apply_rotary_pos_emb = modeling.apply_rotary_pos_emb


@dataclass
class CausalConvState:
    history: torch.Tensor


@dataclass
class TransposeConvEmitState:
    raw: torch.Tensor
    emitted_trimmed: int
    input_offset: int


@dataclass
class AttentionState:
    k: torch.Tensor
    v: torch.Tensor
    positions: torch.Tensor


@dataclass
class TransformerLayerState:
    attn: AttentionState


def load_codes(path: Path, max_frames: int | None) -> torch.Tensor:
    with safe_open(path, framework="pt", device="cpu") as f:
        if "rvq_codes" not in f.keys():
            raise KeyError(f"{path} does not contain rvq_codes")
        codes = f.get_tensor("rvq_codes")
    if codes.ndim != 2:
        raise ValueError(f"Expected rvq_codes [T, Q], got {tuple(codes.shape)}")
    if max_frames is not None:
        codes = codes[:max_frames]
    return codes.transpose(0, 1).unsqueeze(0).to(torch.long).contiguous()


def load_onnx_initializers(onnx_path: Path) -> tuple[list[str], dict[str, torch.Tensor]]:
    model = onnx.load(str(onnx_path), load_external_data=True)
    names = []
    tensors = {}
    for init in model.graph.initializer:
        names.append(init.name)
        tensors[init.name] = torch.from_numpy(numpy_helper.to_array(init)).contiguous()
    return names, tensors


def snake_parameter_keys(config: Qwen3TTSTokenizerV2DecoderConfig) -> list[str]:
    keys: list[str] = []
    for layer_idx, _rate in enumerate(config.upsample_rates, start=1):
        keys.extend([f"decoder.{layer_idx}.block.0.alpha", f"decoder.{layer_idx}.block.0.beta"])
        for unit_idx in (2, 3, 4):
            keys.extend(
                [
                    f"decoder.{layer_idx}.block.{unit_idx}.act1.alpha",
                    f"decoder.{layer_idx}.block.{unit_idx}.act1.beta",
                    f"decoder.{layer_idx}.block.{unit_idx}.act2.alpha",
                    f"decoder.{layer_idx}.block.{unit_idx}.act2.beta",
                ]
            )
    final_idx = 1 + len(config.upsample_rates)
    keys.extend([f"decoder.{final_idx}.alpha", f"decoder.{final_idx}.beta"])
    return keys


def matmul_parameter_keys(config: Qwen3TTSTokenizerV2DecoderConfig) -> list[str]:
    keys = ["pre_transformer.input_proj.weight"]
    for layer_idx in range(config.num_hidden_layers):
        prefix = f"pre_transformer.layers.{layer_idx}"
        keys.extend(
            [
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            ]
        )
    keys.append("pre_transformer.output_proj.weight")
    for stage_idx in range(len(config.upsampling_ratios)):
        keys.extend([f"upsample.{stage_idx}.1.pwconv1.weight", f"upsample.{stage_idx}.1.pwconv2.weight"])
    return keys


def build_real_decoder(config_path: Path, onnx_path: Path, dtype: torch.dtype, verbose: bool = False):
    config = Qwen3TTSTokenizerV2DecoderConfig(**json.loads(config_path.read_text()))
    config._attn_implementation = "eager"
    if verbose:
        print({"event": "build_config_loaded"}, flush=True)
    model = Qwen3TTSTokenizerV2Decoder(config).eval()
    if verbose:
        print({"event": "build_model_instantiated", "state_tensors": len(model.state_dict())}, flush=True)
    state = model.state_dict()
    init_names, init = load_onnx_initializers(onnx_path)
    if verbose:
        print({"event": "build_onnx_loaded", "initializers": len(init_names)}, flush=True)

    loaded = set()
    for key in list(state.keys()):
        if key in init and tuple(init[key].shape) == tuple(state[key].shape):
            state[key] = init[key].to(dtype=state[key].dtype)
            loaded.add(key)
    if verbose:
        print({"event": "build_direct_loaded", "count": len(loaded)}, flush=True)

    matmul_names = [name for name in init_names if name.startswith("onnx::MatMul")]
    matmul_keys = matmul_parameter_keys(config)
    if verbose:
        print({"event": "build_matmul_plan", "names": len(matmul_names), "keys": len(matmul_keys)}, flush=True)
    if len(matmul_names) < len(matmul_keys):
        raise RuntimeError(f"Not enough ONNX MatMul initializers: {len(matmul_names)} < {len(matmul_keys)}")
    for key, name in zip(matmul_keys, matmul_names, strict=True):
        tensor = init[name].transpose(0, 1).contiguous()
        if tuple(tensor.shape) != tuple(state[key].shape):
            raise RuntimeError(f"Shape mismatch for {key}: ONNX {tuple(tensor.shape)} vs state {tuple(state[key].shape)}")
        state[key] = tensor.to(dtype=state[key].dtype)
        loaded.add(key)
    if verbose:
        print({"event": "build_matmul_loaded", "count": len(loaded)}, flush=True)

    exp_names = [name for name in init_names if name.startswith("onnx::Exp")]
    exp_keys = snake_parameter_keys(config)
    if verbose:
        print({"event": "build_exp_plan", "names": len(exp_names), "keys": len(exp_keys)}, flush=True)
    if len(exp_names) != len(exp_keys):
        raise RuntimeError(f"Unexpected SnakeBeta initializer count: {len(exp_names)} != {len(exp_keys)}")
    for key, name in zip(exp_keys, exp_names, strict=True):
        tensor = init[name].squeeze(0).squeeze(-1).contiguous()
        if tuple(tensor.shape) != tuple(state[key].shape):
            raise RuntimeError(f"Shape mismatch for {key}: ONNX {tuple(tensor.shape)} vs state {tuple(state[key].shape)}")
        state[key] = tensor.to(dtype=state[key].dtype)
        loaded.add(key)
    if verbose:
        print({"event": "build_exp_loaded", "count": len(loaded)}, flush=True)

    allowed_missing = {
        "pre_transformer.rotary_emb.inv_freq",
        "pre_transformer.rotary_emb.original_inv_freq",
        "quantizer.rvq_first.input_proj.weight",
        "quantizer.rvq_rest.input_proj.weight",
    }
    missing = [key for key in state if key not in loaded and key not in allowed_missing]
    if verbose:
        skipped = [key for key in state if key not in loaded and key in allowed_missing]
        print({"event": "build_missing_checked", "missing": missing[:8], "skipped": skipped}, flush=True)
    if missing:
        raise RuntimeError(f"Failed to load {len(missing)} state tensors, first missing: {missing[:8]}")
    model.load_state_dict(state, strict=True)
    if verbose:
        print({"event": "build_state_loaded"}, flush=True)
    return model.to(dtype=dtype).eval()


def causal_conv_step(module, x: torch.Tensor, state: CausalConvState) -> tuple[torch.Tensor, CausalConvState]:
    if module.conv.stride[0] != 1:
        raise NotImplementedError("Only stride=1 causal conv is supported")
    padded = torch.cat([state.history, x], dim=-1)
    y = module.conv(padded)
    history_len = module.padding
    next_history = padded[..., -history_len:].detach() if history_len else padded[..., :0].detach()
    return y, CausalConvState(next_history)


def causal_conv_stream_chunks(module, chunks: list[torch.Tensor]) -> list[torch.Tensor]:
    if not chunks:
        return []
    first = chunks[0]
    state = CausalConvState(
        torch.zeros(first.shape[0], first.shape[1], module.padding, dtype=first.dtype, device=first.device)
    )
    outs = []
    for part in chunks:
        y, state = causal_conv_step(module, part, state)
        outs.append(y)
    return outs


def transposed_conv_emit_step(
    module,
    x: torch.Tensor,
    state: TransposeConvEmitState,
    *,
    final: bool,
) -> tuple[torch.Tensor, TransposeConvEmitState]:
    stride = module.conv.stride[0]
    left_pad = module.left_pad
    right_pad = module.right_pad
    raw = F.conv_transpose1d(
        x,
        module.conv.weight,
        bias=None,
        stride=module.conv.stride,
        padding=module.conv.padding,
        output_padding=module.conv.output_padding,
        groups=module.conv.groups,
        dilation=module.conv.dilation,
    )
    start = state.input_offset * stride
    raw_len = max(state.raw.shape[-1], start + raw.shape[-1])
    if raw_len > state.raw.shape[-1]:
        grown = torch.zeros(
            state.raw.shape[0],
            state.raw.shape[1],
            raw_len,
            dtype=state.raw.dtype,
            device=state.raw.device,
        )
        grown[..., : state.raw.shape[-1]] = state.raw
        raw_accum = grown
    else:
        raw_accum = state.raw.clone()
    raw_accum[..., start : start + raw.shape[-1]] += raw

    next_input_offset = state.input_offset + x.shape[-1]
    if final:
        stable_trimmed_end = max(0, raw_accum.shape[-1] - right_pad - left_pad)
    else:
        stable_trimmed_end = max(0, next_input_offset * stride - left_pad)
    emit_start = state.emitted_trimmed
    emit_end = max(emit_start, stable_trimmed_end)
    y = raw_accum[..., left_pad + emit_start : left_pad + emit_end]
    if module.conv.bias is not None and y.shape[-1] > 0:
        y = y + module.conv.bias.view(1, -1, 1)
    return y, TransposeConvEmitState(raw_accum.detach(), emit_end, next_input_offset)


def transposed_conv_stream_chunks(module, chunks: list[torch.Tensor]) -> list[torch.Tensor]:
    if not chunks:
        return []
    first = chunks[0]
    state = TransposeConvEmitState(
        raw=torch.zeros(first.shape[0], module.conv.out_channels, 0, dtype=first.dtype, device=first.device),
        emitted_trimmed=0,
        input_offset=0,
    )
    outs = []
    for index, part in enumerate(chunks):
        y, state = transposed_conv_emit_step(module, part, state, final=index == len(chunks) - 1)
        if y.shape[-1] > 0:
            outs.append(y)
    return outs


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(bsz, kv_heads * n_rep, seq_len, head_dim)


def empty_attention_state(module, batch_size: int, dtype: torch.dtype, device: torch.device) -> AttentionState:
    kv_heads = module.config.num_key_value_heads
    return AttentionState(
        k=torch.empty(batch_size, kv_heads, 0, module.head_dim, dtype=dtype, device=device),
        v=torch.empty(batch_size, kv_heads, 0, module.head_dim, dtype=dtype, device=device),
        positions=torch.empty(0, dtype=torch.long, device=device),
    )


def attention_step(
    module,
    hidden_states: torch.Tensor,
    state: AttentionState,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    positions: torch.Tensor,
) -> tuple[torch.Tensor, AttentionState]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    q = module.q_norm(module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    k = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    v = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, *position_embeddings)

    all_k = torch.cat([state.k, k], dim=2)
    all_v = torch.cat([state.v, v], dim=2)
    all_pos = torch.cat([state.positions, positions], dim=0)
    k_for_attn = repeat_kv(all_k, module.num_key_value_groups)
    v_for_attn = repeat_kv(all_v, module.num_key_value_groups)

    q_pos = positions.unsqueeze(1)
    k_pos = all_pos.unsqueeze(0)
    valid = (k_pos > q_pos - module.sliding_window) & (k_pos <= q_pos)
    mask = torch.where(
        valid,
        torch.zeros(1, dtype=hidden_states.dtype, device=hidden_states.device),
        torch.tensor(torch.finfo(hidden_states.dtype).min, dtype=hidden_states.dtype, device=hidden_states.device),
    )
    attn = torch.matmul(q, k_for_attn.transpose(2, 3)) * module.scaling + mask.unsqueeze(0).unsqueeze(0)
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(attn, v_for_attn).transpose(1, 2).reshape(*input_shape, -1).contiguous()
    out = module.o_proj(out)

    keep = max(0, module.sliding_window - 1)
    return out, AttentionState(all_k[:, :, -keep:, :].detach(), all_v[:, :, -keep:, :].detach(), all_pos[-keep:].detach())


def transformer_layer_step(
    layer,
    hidden: torch.Tensor,
    state: TransformerLayerState,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    positions: torch.Tensor,
) -> tuple[torch.Tensor, TransformerLayerState]:
    residual = hidden
    h = layer.input_layernorm(hidden)
    h, attn_state = attention_step(layer.self_attn, h, state.attn, position_embeddings, positions)
    hidden = residual + layer.self_attn_layer_scale(h)
    residual = hidden
    hidden = layer.post_attention_layernorm(hidden)
    hidden = layer.mlp(hidden)
    hidden = residual + layer.mlp_layer_scale(hidden)
    return hidden, TransformerLayerState(attn_state)


def pre_transformer_stream_chunks(model, chunks: list[torch.Tensor]) -> list[torch.Tensor]:
    if not chunks:
        return []
    batch_size = chunks[0].shape[0]
    dtype = chunks[0].dtype
    device = chunks[0].device
    pre = model.pre_transformer
    states = [TransformerLayerState(empty_attention_state(layer.self_attn, batch_size, dtype, device)) for layer in pre.layers]
    outs = []
    offset = 0
    for part in chunks:
        hidden = pre.input_proj(part.transpose(1, 2))
        positions = torch.arange(offset, offset + hidden.shape[1], device=device)
        position_embeddings = pre.rotary_emb(hidden, positions.unsqueeze(0))
        for index, layer in enumerate(pre.layers):
            hidden, states[index] = transformer_layer_step(layer, hidden, states[index], position_embeddings, positions)
        hidden = pre.norm(hidden)
        hidden = pre.output_proj(hidden)
        outs.append(hidden.permute(0, 2, 1))
        offset += hidden.shape[1]
    return outs


def convnext_stream_chunks(block, chunks: list[torch.Tensor]) -> list[torch.Tensor]:
    conv = causal_conv_stream_chunks(block.dwconv, chunks)
    outs = []
    for residual, y in zip(chunks, conv, strict=True):
        y = y.permute(0, 2, 1)
        y = block.norm(y)
        y = block.pwconv1(y)
        y = block.act(y)
        y = block.pwconv2(y)
        y = block.gamma * y
        y = y.permute(0, 2, 1)
        outs.append(residual + y)
    return outs


def decoder_residual_unit_stream_chunks(
    unit,
    chunks: list[torch.Tensor],
) -> list[torch.Tensor]:
    y = [unit.act1(part) for part in chunks]
    y = causal_conv_stream_chunks(unit.conv1, y)
    y = [unit.act2(part) for part in y]
    y = causal_conv_stream_chunks(unit.conv2, y)
    return [residual + part for residual, part in zip(chunks, y, strict=True)]


def decoder_block_stream_chunks(block, chunks: list[torch.Tensor]) -> list[torch.Tensor]:
    chunks = [block.block[0](part) for part in chunks]
    chunks = transposed_conv_stream_chunks(block.block[1], chunks)
    for unit in block.block[2:]:
        chunks = decoder_residual_unit_stream_chunks(unit, chunks)
    return chunks


def code2wav_online(model, codes: torch.Tensor, chunk_frames: int) -> torch.Tensor:
    code_chunks = [codes[..., start : start + chunk_frames] for start in range(0, codes.shape[-1], chunk_frames)]
    chunks = [model.quantizer.decode(part) for part in code_chunks]
    chunks = causal_conv_stream_chunks(model.pre_conv, chunks)
    chunks = pre_transformer_stream_chunks(model, chunks)
    for blocks in model.upsample:
        chunks = transposed_conv_stream_chunks(blocks[0], chunks)
        chunks = convnext_stream_chunks(blocks[1], chunks)
    for block in model.decoder:
        if isinstance(block, Qwen3TTSTokenizerV2CausalConvNet):
            chunks = causal_conv_stream_chunks(block, chunks)
        elif isinstance(block, Qwen3TTSTokenizerV2DecoderDecoderBlock):
            chunks = decoder_block_stream_chunks(block, chunks)
        elif isinstance(block, SnakeBeta):
            chunks = [block(part) for part in chunks]
        else:
            raise TypeError(f"Unhandled decoder block: {type(block)}")
    return torch.cat(chunks, dim=-1).clamp(min=-1, max=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-dir", type=Path, default=Path("tmp_code2wav_onnx_real"))
    parser.add_argument("--codes", type=Path, default=Path("qwen3tts-listen-0506/clean-allprs-0507/short_cn/rvq_req0.safetensors"))
    parser.add_argument(
        "--qwen-tts-root",
        type=Path,
        default=Path("/Users/harvest/project/Qwen3-TTS/qwen_tts"),
        help="Path to the qwen_tts package directory; loaded by file path to avoid package __init__ side effects.",
    )
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--chunk-frames", type=int, default=3)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--tolerance", type=float, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    tolerance = args.tolerance if args.tolerance is not None else (3e-3 if dtype == torch.float16 else 2e-4)
    torch.set_grad_enabled(False)
    if args.verbose:
        print({"event": "start", "dtype": args.dtype}, flush=True)
    load_qwen_tts_symbols(args.qwen_tts_root)
    if args.verbose:
        print({"event": "symbols_loaded"}, flush=True)

    model = build_real_decoder(args.onnx_dir / "config.json", args.onnx_dir / "model.onnx", dtype=dtype, verbose=args.verbose)
    if args.verbose:
        print({"event": "model_loaded"}, flush=True)
    codes = load_codes(args.codes, args.max_frames)
    if args.verbose:
        print({"event": "codes_loaded", "shape": list(codes.shape)}, flush=True)
    full = model(codes).detach()
    if args.verbose:
        print({"event": "full_done", "shape": list(full.shape)}, flush=True)
    streamed = code2wav_online(model, codes, args.chunk_frames).detach()
    if args.verbose:
        print({"event": "stream_done", "shape": list(streamed.shape)}, flush=True)
    error = float((full - streamed).abs().max().item())
    print(
        {
            "status": "passed" if error <= tolerance else "failed",
            "codes_shape": list(codes.shape),
            "full_shape": list(full.shape),
            "stream_shape": list(streamed.shape),
            "chunk_frames": args.chunk_frames,
            "max_abs": error,
            "tolerance": tolerance,
        }
    )
    return 0 if error <= tolerance else 1


if __name__ == "__main__":
    raise SystemExit(main())
