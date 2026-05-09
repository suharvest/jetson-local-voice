#!/usr/bin/env python3
"""Reference probes for a stateful Qwen3 Code2Wav implementation.

This script is intentionally PyTorch-only.  It validates the streaming state
rules for individual Code2Wav building blocks before we export a stateful ONNX
or add a TensorRT runner.
"""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


EDGE_LLM_ROOT = Path(__file__).resolve().parents[1] / ".." / "tensorrt-edge-llm"
CODE2WAV_MODEL = (
    EDGE_LLM_ROOT
    / "experimental"
    / "llm_loader"
    / "models"
    / "qwen3_omni"
    / "modeling_qwen3_omni_code2wav.py"
).resolve()


def load_code2wav_module():
    spec = importlib.util.spec_from_file_location("qwen3_code2wav_model", CODE2WAV_MODEL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {CODE2WAV_MODEL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class CausalConvState:
    history: torch.Tensor


def causal_conv_step(module, x: torch.Tensor, state: CausalConvState) -> tuple[torch.Tensor, CausalConvState]:
    """Run one streaming step for CausalConv1d.

    The exported model pads ``padding`` zeros on the left for a full utterance.
    In streaming mode that left pad becomes explicit input history.  For stride
    1, which is what Qwen3 Code2Wav uses for its causal convs, this emits
    exactly one output sample per new input sample.
    """

    if module.conv.stride[0] != 1:
        raise NotImplementedError("Only stride=1 CausalConv1d is supported by this reference step")
    if module.padding == 0:
        y = module.conv(x)
        return y, CausalConvState(torch.empty_like(x[..., :0]))

    padded = torch.cat([state.history, x], dim=-1)
    y = module.conv(padded)
    next_history = padded[..., -module.padding :].detach()
    return y, CausalConvState(next_history)


def causal_conv_stream(module, x: torch.Tensor, chunks: list[int]) -> torch.Tensor:
    state = CausalConvState(torch.zeros(x.shape[0], x.shape[1], module.padding, dtype=x.dtype, device=x.device))
    outs = []
    offset = 0
    for chunk in chunks:
        part = x[..., offset : offset + chunk]
        if part.shape[-1] == 0:
            continue
        y, state = causal_conv_step(module, part, state)
        outs.append(y)
        offset += chunk
    if offset != x.shape[-1]:
        raise ValueError(f"Chunks cover {offset}, input length is {x.shape[-1]}")
    return torch.cat(outs, dim=-1)


@dataclass
class TransposeConvChunk:
    start: int
    raw: torch.Tensor


@dataclass
class AttentionState:
    k: torch.Tensor
    v: torch.Tensor
    positions: torch.Tensor


@dataclass
class TransformerLayerState:
    attn: AttentionState


@dataclass
class TransposeConvEmitState:
    raw: torch.Tensor
    emitted_trimmed: int
    input_offset: int


def transposed_conv_stream_collect(module, x: torch.Tensor, chunks: list[int]) -> torch.Tensor:
    """Reference streaming for CausalTransposeConv1d via overlap-add.

    ConvTranspose is the risky Code2Wav stateful piece.  This reference treats
    each chunk as contributing a raw transposed-conv segment placed at
    ``input_start * stride`` in the global raw output, then applies the same
    global left/right trim as the full module.  A production runner can emit
    only stable samples and keep the right-edge pending region as state.
    """

    stride = module.conv.stride[0]
    left_pad = module.left_pad
    right_pad = module.right_pad

    pieces: list[TransposeConvChunk] = []
    input_offset = 0
    raw_len = 0
    for chunk in chunks:
        part = x[..., input_offset : input_offset + chunk]
        if part.shape[-1] == 0:
            continue
        raw = F.conv_transpose1d(
            part,
            module.conv.weight,
            bias=None,
            stride=module.conv.stride,
            padding=module.conv.padding,
            output_padding=module.conv.output_padding,
            groups=module.conv.groups,
            dilation=module.conv.dilation,
        )
        start = input_offset * stride
        pieces.append(TransposeConvChunk(start=start, raw=raw))
        raw_len = max(raw_len, start + raw.shape[-1])
        input_offset += chunk
    if input_offset != x.shape[-1]:
        raise ValueError(f"Chunks cover {input_offset}, input length is {x.shape[-1]}")

    full_raw = torch.zeros(
        x.shape[0],
        module.conv.out_channels,
        raw_len,
        dtype=x.dtype,
        device=x.device,
    )
    for piece in pieces:
        end = piece.start + piece.raw.shape[-1]
        full_raw[..., piece.start : end] += piece.raw
    if module.conv.bias is not None:
        full_raw += module.conv.bias.view(1, -1, 1)

    right = full_raw.shape[-1] - right_pad
    return full_raw[..., left_pad:right]


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

    return y, TransposeConvEmitState(
        raw=raw_accum.detach(),
        emitted_trimmed=emit_end,
        input_offset=next_input_offset,
    )


def transposed_conv_stream_emit(module, x: torch.Tensor, chunks: list[int]) -> torch.Tensor:
    state = TransposeConvEmitState(
        raw=torch.zeros(x.shape[0], module.conv.out_channels, 0, dtype=x.dtype, device=x.device),
        emitted_trimmed=0,
        input_offset=0,
    )
    outs = []
    offset = 0
    for index, chunk in enumerate(chunks):
        part = x[..., offset : offset + chunk]
        if part.shape[-1] == 0:
            continue
        y, state = transposed_conv_emit_step(module, part, state, final=index == len(chunks) - 1)
        outs.append(y)
        offset += chunk
    if offset != x.shape[-1]:
        raise ValueError(f"Chunks cover {offset}, input length is {x.shape[-1]}")
    return torch.cat(outs, dim=-1)


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def report_probe(name: str, full: torch.Tensor, streamed: torch.Tensor) -> float:
    error = max_abs(full, streamed)
    print(
        {
            "probe": name,
            "full_shape": list(full.shape),
            "stream_shape": list(streamed.shape),
            "max_abs": error,
        }
    )
    return error


def attention_step(
    module,
    hidden_states: torch.Tensor,
    state: AttentionState,
    position_offset: int,
) -> tuple[torch.Tensor, AttentionState]:
    B, L, _ = hidden_states.shape
    dtype = hidden_states.dtype
    device = hidden_states.device

    q = module.q_proj(hidden_states).view(B, L, module.num_heads, module.head_dim).transpose(1, 2)
    k = module.k_proj(hidden_states).view(B, L, module.num_heads, module.head_dim).transpose(1, 2)
    v = module.v_proj(hidden_states).view(B, L, module.num_heads, module.head_dim).transpose(1, 2)

    positions = torch.arange(position_offset, position_offset + L, device=device)
    freqs = torch.outer(positions.float(), module.inv_freq)
    cos = freqs.cos().to(dtype).unsqueeze(0).unsqueeze(0)
    sin = freqs.sin().to(dtype).unsqueeze(0).unsqueeze(0)
    q = module._apply_rotary(q, cos, sin)
    k = module._apply_rotary(k, cos, sin)

    all_k = torch.cat([state.k, k], dim=2)
    all_v = torch.cat([state.v, v], dim=2)
    all_pos = torch.cat([state.positions, positions], dim=0)

    q_pos = positions.unsqueeze(1)
    k_pos = all_pos.unsqueeze(0)
    valid = (k_pos > q_pos - module.sliding_window) & (k_pos <= q_pos)
    mask = torch.where(
        valid,
        torch.zeros(1, dtype=dtype, device=device),
        torch.tensor(torch.finfo(dtype).min, dtype=dtype, device=device),
    )

    scale = torch.tensor(module.scale, dtype=q.dtype, device=device)
    attn = (q @ all_k.transpose(-2, -1)) * scale + mask.unsqueeze(0).unsqueeze(0)
    attn = F.softmax(attn, dim=-1)
    out = (attn @ all_v).transpose(1, 2).reshape(B, L, -1)
    out = module.o_proj(out)

    keep = max(0, module.sliding_window - 1)
    if keep:
        next_state = AttentionState(
            k=all_k[:, :, -keep:, :].detach(),
            v=all_v[:, :, -keep:, :].detach(),
            positions=all_pos[-keep:].detach(),
        )
    else:
        next_state = AttentionState(
            k=all_k[:, :, :0, :].detach(),
            v=all_v[:, :, :0, :].detach(),
            positions=all_pos[:0].detach(),
        )
    return out, next_state


def attention_stream(module, hidden_states: torch.Tensor, chunks: list[int]) -> torch.Tensor:
    B = hidden_states.shape[0]
    state = AttentionState(
        k=torch.empty(B, module.num_heads, 0, module.head_dim, dtype=hidden_states.dtype, device=hidden_states.device),
        v=torch.empty(B, module.num_heads, 0, module.head_dim, dtype=hidden_states.dtype, device=hidden_states.device),
        positions=torch.empty(0, dtype=torch.long, device=hidden_states.device),
    )
    outs = []
    offset = 0
    for chunk in chunks:
        part = hidden_states[:, offset : offset + chunk, :]
        if part.shape[1] == 0:
            continue
        y, state = attention_step(module, part, state, offset)
        outs.append(y)
        offset += chunk
    if offset != hidden_states.shape[1]:
        raise ValueError(f"Chunks cover {offset}, input length is {hidden_states.shape[1]}")
    return torch.cat(outs, dim=1)


def empty_attention_state(module, batch_size: int, dtype: torch.dtype, device: torch.device) -> AttentionState:
    return AttentionState(
        k=torch.empty(batch_size, module.num_heads, 0, module.head_dim, dtype=dtype, device=device),
        v=torch.empty(batch_size, module.num_heads, 0, module.head_dim, dtype=dtype, device=device),
        positions=torch.empty(0, dtype=torch.long, device=device),
    )


def transformer_layer_step(
    layer,
    x: torch.Tensor,
    state: TransformerLayerState,
    position_offset: int,
) -> tuple[torch.Tensor, TransformerLayerState]:
    residual = x
    h = layer.input_layernorm(x)
    h, attn_state = attention_step(layer.self_attn, h, state.attn, position_offset)
    x = residual + layer.self_attn_layer_scale(h)

    residual = x
    h = layer.post_attention_layernorm(x)
    h = layer.mlp_down_proj(F.silu(layer.mlp_gate_proj(h)) * layer.mlp_up_proj(h))
    x = residual + layer.mlp_layer_scale(h)
    return x, TransformerLayerState(attn=attn_state)


def pre_transformer_full(model, codes: torch.Tensor) -> torch.Tensor:
    hidden = model.code_embedding(codes + model.code_offset).mean(1)
    for layer in model.layers:
        hidden = layer(hidden)
    return model.norm(hidden)


def pre_transformer_stream(model, codes: torch.Tensor, chunks: list[int]) -> torch.Tensor:
    return torch.cat(pre_transformer_stream_chunks(model, codes, chunks), dim=1)


def pre_transformer_stream_chunks(model, codes: torch.Tensor, chunks: list[int]) -> list[torch.Tensor]:
    batch_size = codes.shape[0]
    dtype = model.code_embedding.weight.dtype
    device = codes.device
    states = [
        TransformerLayerState(attn=empty_attention_state(layer.self_attn, batch_size, dtype, device))
        for layer in model.layers
    ]

    outs: list[torch.Tensor] = []
    offset = 0
    for chunk in chunks:
        part = codes[..., offset : offset + chunk]
        if part.shape[-1] == 0:
            continue
        hidden = model.code_embedding(part + model.code_offset).mean(1)
        for index, layer in enumerate(model.layers):
            hidden, states[index] = transformer_layer_step(layer, hidden, states[index], offset)
        outs.append(model.norm(hidden))
        offset += chunk
    if offset != codes.shape[-1]:
        raise ValueError(f"Chunks cover {offset}, input length is {codes.shape[-1]}")
    return outs


def make_chunks(total: int, pattern: list[int]) -> list[int]:
    chunks = []
    offset = 0
    index = 0
    while offset < total:
        chunk = min(pattern[index % len(pattern)], total - offset)
        chunks.append(chunk)
        offset += chunk
        index += 1
    return chunks


def stream_convnext_block(block, x: torch.Tensor, chunks: list[int]) -> torch.Tensor:
    residual = x
    y = causal_conv_stream(block.dwconv, x, chunks)
    y = y.transpose(1, 2)
    y = block.norm(y)
    y = F.gelu(block.pwconv1(y))
    y = block.pwconv2(y)
    y = block.gamma * y
    y = y.transpose(1, 2)
    return residual + y


def causal_conv_stream_chunks(module, chunks: list[torch.Tensor]) -> list[torch.Tensor]:
    if not chunks:
        return []
    first = chunks[0]
    state = CausalConvState(
        torch.zeros(first.shape[0], first.shape[1], module.padding, dtype=first.dtype, device=first.device)
    )
    outs = []
    for part in chunks:
        if part.shape[-1] == 0:
            continue
        y, state = causal_conv_step(module, part, state)
        outs.append(y)
    return outs


def transposed_conv_stream_emit_chunks(module, chunks: list[torch.Tensor]) -> list[torch.Tensor]:
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
        if part.shape[-1] == 0:
            continue
        y, state = transposed_conv_emit_step(module, part, state, final=index == len(chunks) - 1)
        if y.shape[-1] > 0:
            outs.append(y)
    return outs


def stream_convnext_block_chunks(block, chunks: list[torch.Tensor]) -> list[torch.Tensor]:
    conv = causal_conv_stream_chunks(block.dwconv, chunks)
    outs = []
    for residual, y in zip(chunks, conv, strict=True):
        y = y.transpose(1, 2)
        y = block.norm(y)
        y = F.gelu(block.pwconv1(y))
        y = block.pwconv2(y)
        y = block.gamma * y
        y = y.transpose(1, 2)
        outs.append(residual + y)
    return outs


def stream_decoder_residual_unit(unit, x: torch.Tensor, chunks: list[int]) -> torch.Tensor:
    y = unit.act1(x)
    y = causal_conv_stream(unit.conv1, y, chunks)
    y = unit.act2(y)
    y = causal_conv_stream(unit.conv2, y, chunks)
    return x + y


def stream_decoder_residual_unit_chunks(unit, chunks: list[torch.Tensor]) -> list[torch.Tensor]:
    y_chunks = [unit.act1(part) for part in chunks]
    y_chunks = causal_conv_stream_chunks(unit.conv1, y_chunks)
    y_chunks = [unit.act2(part) for part in y_chunks]
    y_chunks = causal_conv_stream_chunks(unit.conv2, y_chunks)
    return [residual + y for residual, y in zip(chunks, y_chunks, strict=True)]


def stream_decoder_stage_collect(stage, x: torch.Tensor, chunks: list[int]) -> tuple[torch.Tensor, list[int]]:
    y = stage.block[0](x)
    y = transposed_conv_stream_collect(stage.block[1], y, chunks)
    chunks = make_chunks(y.shape[-1], [11, 3, 17, 5])
    for unit in stage.block[2:]:
        y = stream_decoder_residual_unit(unit, y, chunks)
    return y, chunks


def stream_decoder_stage_chunks(stage, chunks: list[torch.Tensor]) -> list[torch.Tensor]:
    chunks = [stage.block[0](part) for part in chunks]
    chunks = transposed_conv_stream_emit_chunks(stage.block[1], chunks)
    for unit in stage.block[2:]:
        chunks = stream_decoder_residual_unit_chunks(unit, chunks)
    return chunks


def convnet_stream_collect(module, model, hidden: torch.Tensor, chunks: list[int]) -> torch.Tensor:
    for stage in model.upsample:
        hidden = transposed_conv_stream_collect(stage[0], hidden, chunks)
        chunks = make_chunks(hidden.shape[-1], [9, 4, 13, 2])
        hidden = stream_convnext_block(stage[1], hidden, chunks)

    wav = hidden
    for block in model.decoder:
        if isinstance(block, module.CausalConv1d):
            wav = causal_conv_stream(block, wav, chunks)
        elif isinstance(block, module.DecoderStage):
            wav, chunks = stream_decoder_stage_collect(block, wav, chunks)
        elif isinstance(block, module.SnakeBeta):
            wav = block(wav)
        else:
            raise TypeError(f"Unhandled decoder block: {type(block)}")
    return wav


def convnet_stream_online(module, model, chunks: list[torch.Tensor]) -> list[torch.Tensor]:
    for stage in model.upsample:
        chunks = transposed_conv_stream_emit_chunks(stage[0], chunks)
        chunks = stream_convnext_block_chunks(stage[1], chunks)

    for block in model.decoder:
        if isinstance(block, module.CausalConv1d):
            chunks = causal_conv_stream_chunks(block, chunks)
        elif isinstance(block, module.DecoderStage):
            chunks = stream_decoder_stage_chunks(block, chunks)
        elif isinstance(block, module.SnakeBeta):
            chunks = [block(part) for part in chunks]
        else:
            raise TypeError(f"Unhandled decoder block: {type(block)}")
    return chunks


def code2wav_stream_collect(module, model, codes: torch.Tensor, chunks: list[int]) -> torch.Tensor:
    hidden = pre_transformer_stream(model, codes, chunks).permute(0, 2, 1)
    wav = convnet_stream_collect(module, model, hidden, chunks)
    return wav.clamp(min=-1, max=1)


def code2wav_stream_online(module, model, codes: torch.Tensor, chunks: list[int]) -> torch.Tensor:
    hidden_chunks = [part.permute(0, 2, 1) for part in pre_transformer_stream_chunks(model, codes, chunks)]
    wav_chunks = convnet_stream_online(module, model, hidden_chunks)
    return torch.cat(wav_chunks, dim=-1).clamp(min=-1, max=1)


def run_module_probes(dtype: torch.dtype, device: torch.device) -> int:
    m = load_code2wav_module()
    torch.manual_seed(7)
    tolerance = 2e-3 if dtype == torch.float16 else 1e-5
    errors = []

    conv = m.CausalConv1d(4, 6, kernel_size=7).to(device=device, dtype=dtype).eval()
    x = torch.randn(2, 4, 31, device=device, dtype=dtype)
    chunks = [1, 5, 8, 3, 14]
    full = conv(x)
    streamed = causal_conv_stream(conv, x, chunks)
    errors.append(report_probe("causal_conv", full, streamed))

    tconv = m.CausalTransposeConv1d(4, 3, kernel_size=10, stride=5).to(device=device, dtype=dtype).eval()
    x = torch.randn(2, 4, 29, device=device, dtype=dtype)
    chunks = [4, 7, 1, 9, 8]
    full = tconv(x)
    streamed = transposed_conv_stream_collect(tconv, x, chunks)
    errors.append(report_probe("transpose_conv_overlap_add", full, streamed))
    streamed = transposed_conv_stream_emit(tconv, x, chunks)
    errors.append(report_probe("transpose_conv_online_emit", full, streamed))

    attn = m.Attention(hidden_size=32, num_heads=4, head_dim=8, sliding_window=7, rope_theta=10000.0).to(
        device=device, dtype=dtype
    ).eval()
    x = torch.randn(2, 23, 32, device=device, dtype=dtype)
    chunks = [3, 1, 8, 4, 7]
    full = attn(x)
    streamed = attention_stream(attn, x, chunks)
    errors.append(report_probe("sliding_window_attention", full, streamed))

    config = {
        "hidden_size": 32,
        "num_quantizers": 4,
        "codebook_size": 64,
        "decoder_dim": 32,
        "upsample_rates": [2],
        "upsampling_ratios": [2],
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "intermediate_size": 64,
        "sliding_window": 7,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "layer_scale_initial_scale": 0.01,
    }
    model = m.Code2WavModel(config).to(device=device, dtype=dtype).eval()
    codes = torch.randint(0, config["codebook_size"], (2, config["num_quantizers"], 23), device=device)
    chunks = [2, 5, 1, 7, 8]
    full = pre_transformer_full(model, codes)
    streamed = pre_transformer_stream(model, codes, chunks)
    errors.append(report_probe("pre_transformer_stack", full, streamed))

    full = model(codes)
    streamed = code2wav_stream_collect(m, model, codes, chunks)
    errors.append(report_probe("code2wav_collect_full_model", full, streamed))
    streamed = code2wav_stream_online(m, model, codes, chunks)
    errors.append(report_probe("code2wav_online_full_model", full, streamed))

    worst = max(errors)
    if worst > tolerance:
        print({"status": "failed", "worst_max_abs": worst, "tolerance": tolerance})
        return 1
    print({"status": "passed", "worst_max_abs": worst, "tolerance": tolerance})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    device = torch.device(args.device)
    return run_module_probes(dtype, device)


if __name__ == "__main__":
    raise SystemExit(main())
