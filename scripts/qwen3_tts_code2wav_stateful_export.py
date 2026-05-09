#!/usr/bin/env python3
"""Prototype/export stateful ONNX interface for product Qwen3-TTS Code2Wav.

This script reuses the real-weight loader from
``qwen3_tts_code2wav_stateful_real_gate.py`` and builds an explicit
single-chunk stateful module:

    codes, position_offset, *_state_in -> waveform, *_state_out

It first verifies the explicit state interface against the already validated
online reference.  ONNX export is intentionally a separate flag so interface
validation can run quickly before creating a large graph.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import qwen3_tts_code2wav_stateful_real_gate as real_gate


@dataclass(frozen=True)
class StateSpec:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype


def _zeros(spec: StateSpec, device: torch.device) -> torch.Tensor:
    return torch.zeros(spec.shape, dtype=spec.dtype, device=device)


def _causal_conv_step(module, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if module.padding == 0:
        return module.conv(x), state
    padded = torch.cat([state, x], dim=-1)
    y = module.conv(padded)
    return y, padded[..., -module.padding :]


def _transposed_conv_step(module, x: torch.Tensor, pending: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    stride = module.conv.stride[0]
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
    if pending.shape[-1] > 0:
        raw = raw.clone()
        raw[..., : pending.shape[-1]] = raw[..., : pending.shape[-1]] + pending
    emit_len = x.shape[-1] * stride
    y = raw[..., :emit_len]
    if module.conv.bias is not None:
        y = y + module.conv.bias.view(1, -1, 1)
    next_pending = raw[..., emit_len:]
    return y, next_pending


def _transposed_conv_no_state(module, x: torch.Tensor) -> torch.Tensor:
    stride = module.conv.stride[0]
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
    y = raw[..., : x.shape[-1] * stride]
    if module.conv.bias is not None:
        y = y + module.conv.bias.view(1, -1, 1)
    return y


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(bsz, kv_heads * n_rep, seq_len, head_dim)


class StatefulCode2WavModule(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config
        self.keep = self.config.sliding_window - 1
        self.state_specs = self._build_state_specs()

    def _build_state_specs(self) -> list[StateSpec]:
        cfg = self.config
        specs: list[StateSpec] = []
        dtype = next(self.model.parameters()).dtype
        for layer_idx in range(cfg.num_hidden_layers):
            specs.append(StateSpec(f"attn_{layer_idx}_k", (1, cfg.num_key_value_heads, self.keep, cfg.head_dim), dtype))
            specs.append(StateSpec(f"attn_{layer_idx}_v", (1, cfg.num_key_value_heads, self.keep, cfg.head_dim), dtype))

        def add_conv(name: str, module) -> None:
            if module.padding == 0:
                return
            specs.append(StateSpec(name, (1, module.conv.in_channels, module.padding), dtype))

        def add_tconv(name: str, module) -> None:
            if module.right_pad == 0:
                return
            specs.append(StateSpec(name, (1, module.conv.out_channels, module.right_pad), dtype))

        add_conv("pre_conv_state", self.model.pre_conv)
        for idx, blocks in enumerate(self.model.upsample):
            add_tconv(f"upsample_{idx}_tconv_pending", blocks[0])
            add_conv(f"upsample_{idx}_convnext_dw_state", blocks[1].dwconv)

        for idx, block in enumerate(self.model.decoder):
            if isinstance(block, real_gate.Qwen3TTSTokenizerV2CausalConvNet):
                add_conv(f"decoder_{idx}_conv_state", block)
            elif isinstance(block, real_gate.Qwen3TTSTokenizerV2DecoderDecoderBlock):
                add_tconv(f"decoder_{idx}_tconv_pending", block.block[1])
                for unit_idx, unit in enumerate(block.block[2:], start=2):
                    add_conv(f"decoder_{idx}_unit_{unit_idx}_conv1_state", unit.conv1)
                    add_conv(f"decoder_{idx}_unit_{unit_idx}_conv2_state", unit.conv2)
        return specs

    def initial_states(self, device: torch.device) -> list[torch.Tensor]:
        return [_zeros(spec, device) for spec in self.state_specs]

    def _take(self, states: tuple[torch.Tensor, ...], cursor: int) -> tuple[torch.Tensor, int]:
        return states[cursor], cursor + 1

    def _attention_step(
        self,
        layer,
        hidden_states: torch.Tensor,
        k_state: torch.Tensor,
        v_state: torch.Tensor,
        position_offset: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        q = layer.self_attn.q_norm(layer.self_attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k = layer.self_attn.k_norm(layer.self_attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        seq_len = hidden_states.shape[1]
        positions = torch.arange(seq_len, device=hidden_states.device, dtype=torch.long) + position_offset.to(torch.long)
        position_embeddings = self.model.pre_transformer.rotary_emb(hidden_states, positions.unsqueeze(0))
        q, k = real_gate.apply_rotary_pos_emb(q, k, *position_embeddings)

        all_k = torch.cat([k_state, k], dim=2)
        all_v = torch.cat([v_state, v], dim=2)
        k_for_attn = _repeat_kv(all_k, layer.self_attn.num_key_value_groups)
        v_for_attn = _repeat_kv(all_v, layer.self_attn.num_key_value_groups)

        past_positions = torch.arange(
            -self.keep, 0, device=hidden_states.device, dtype=torch.long
        ) + position_offset.to(torch.long)
        all_positions = torch.cat([past_positions, positions], dim=0)
        q_pos = positions.unsqueeze(1)
        k_pos = all_positions.unsqueeze(0)
        valid = (k_pos >= 0) & (k_pos > q_pos - layer.self_attn.sliding_window) & (k_pos <= q_pos)
        mask = torch.where(
            valid,
            torch.zeros(1, dtype=hidden_states.dtype, device=hidden_states.device),
            torch.tensor(torch.finfo(hidden_states.dtype).min, dtype=hidden_states.dtype, device=hidden_states.device),
        )
        attn = torch.matmul(q, k_for_attn.transpose(2, 3)) * layer.self_attn.scaling + mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v_for_attn).transpose(1, 2).reshape(*input_shape, -1).contiguous()
        out = layer.self_attn.o_proj(out)
        return out, all_k[:, :, -self.keep :, :], all_v[:, :, -self.keep :, :]

    def _pre_transformer_step(
        self,
        hidden: torch.Tensor,
        position_offset: torch.Tensor,
        states: tuple[torch.Tensor, ...],
        cursor: int,
        outputs: list[torch.Tensor],
    ) -> tuple[torch.Tensor, int]:
        hidden = self.model.pre_transformer.input_proj(hidden.transpose(1, 2))
        for layer in self.model.pre_transformer.layers:
            k_state, cursor = self._take(states, cursor)
            v_state, cursor = self._take(states, cursor)
            residual = hidden
            h = layer.input_layernorm(hidden)
            h, next_k, next_v = self._attention_step(layer, h, k_state, v_state, position_offset)
            outputs.extend([next_k, next_v])
            hidden = residual + layer.self_attn_layer_scale(h)
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + layer.mlp_layer_scale(hidden)
        hidden = self.model.pre_transformer.norm(hidden)
        hidden = self.model.pre_transformer.output_proj(hidden)
        return hidden.permute(0, 2, 1), cursor

    def _convnext_step(
        self, block, hidden: torch.Tensor, states: tuple[torch.Tensor, ...], cursor: int, outputs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, int]:
        if block.dwconv.padding == 0:
            y = block.dwconv.conv(hidden)
        else:
            state, cursor = self._take(states, cursor)
            y, next_state = _causal_conv_step(block.dwconv, hidden, state)
            outputs.append(next_state)
        y = y.permute(0, 2, 1)
        y = block.norm(y)
        y = block.pwconv1(y)
        y = block.act(y)
        y = block.pwconv2(y)
        y = block.gamma * y
        y = y.permute(0, 2, 1)
        return hidden + y, cursor

    def _decoder_unit_step(
        self, unit, hidden: torch.Tensor, states: tuple[torch.Tensor, ...], cursor: int, outputs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, int]:
        residual = hidden
        hidden = unit.act1(hidden)
        if unit.conv1.padding == 0:
            hidden = unit.conv1.conv(hidden)
        else:
            state, cursor = self._take(states, cursor)
            hidden, next_state = _causal_conv_step(unit.conv1, hidden, state)
            outputs.append(next_state)
        hidden = unit.act2(hidden)
        if unit.conv2.padding == 0:
            hidden = unit.conv2.conv(hidden)
        else:
            state, cursor = self._take(states, cursor)
            hidden, next_state = _causal_conv_step(unit.conv2, hidden, state)
            outputs.append(next_state)
        return hidden + residual, cursor

    def forward(self, codes: torch.Tensor, position_offset: torch.Tensor, *states: torch.Tensor):
        outputs: list[torch.Tensor] = []
        cursor = 0

        hidden = self.model.quantizer.decode(codes)
        state, cursor = self._take(states, cursor + 2 * self.config.num_hidden_layers)
        hidden, next_state = _causal_conv_step(self.model.pre_conv, hidden, state)

        # Attention states are consumed inside pre_transformer, so prepend their outputs first.
        pre_outputs: list[torch.Tensor] = []
        hidden, _ = self._pre_transformer_step(hidden, position_offset, states, 0, pre_outputs)
        outputs.extend(pre_outputs)
        outputs.append(next_state)

        for blocks in self.model.upsample:
            if blocks[0].right_pad == 0:
                hidden = _transposed_conv_no_state(blocks[0], hidden)
            else:
                pending, cursor = self._take(states, cursor)
                hidden, next_pending = _transposed_conv_step(blocks[0], hidden, pending)
                outputs.append(next_pending)
            hidden, cursor = self._convnext_step(blocks[1], hidden, states, cursor, outputs)

        for block in self.model.decoder:
            if isinstance(block, real_gate.Qwen3TTSTokenizerV2CausalConvNet):
                if block.padding == 0:
                    hidden = block.conv(hidden)
                else:
                    state, cursor = self._take(states, cursor)
                    hidden, next_state = _causal_conv_step(block, hidden, state)
                    outputs.append(next_state)
            elif isinstance(block, real_gate.Qwen3TTSTokenizerV2DecoderDecoderBlock):
                hidden = block.block[0](hidden)
                if block.block[1].right_pad == 0:
                    hidden = _transposed_conv_no_state(block.block[1], hidden)
                else:
                    pending, cursor = self._take(states, cursor)
                    hidden, next_pending = _transposed_conv_step(block.block[1], hidden, pending)
                    outputs.append(next_pending)
                for unit in block.block[2:]:
                    hidden, cursor = self._decoder_unit_step(unit, hidden, states, cursor, outputs)
            elif isinstance(block, real_gate.SnakeBeta):
                hidden = block(hidden)
            else:
                raise TypeError(f"Unhandled decoder block: {type(block)}")

        waveform = hidden.clamp(min=-1, max=1)
        return (waveform, *outputs)


def run_explicit_state_gate(args) -> int:
    real_gate.load_qwen_tts_symbols(args.qwen_tts_root)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    model = real_gate.build_real_decoder(args.onnx_dir / "config.json", args.onnx_dir / "model.onnx", dtype=dtype)
    module = StatefulCode2WavModule(model).eval()
    codes = real_gate.load_codes(args.codes, args.max_frames)
    states = module.initial_states(codes.device)
    chunks = []
    offset = 0
    with torch.no_grad():
        for start in range(0, codes.shape[-1], args.chunk_frames):
            part = codes[..., start : start + args.chunk_frames]
            result = module(part, torch.tensor([offset], dtype=torch.long), *states)
            chunks.append(result[0])
            states = list(result[1:])
            offset += part.shape[-1]
        streamed = torch.cat(chunks, dim=-1)
        full = model(codes)
    err = float((full - streamed).abs().max().item())
    print(
        {
            "status": "passed" if err <= args.tolerance else "failed",
            "codes_shape": list(codes.shape),
            "full_shape": list(full.shape),
            "stream_shape": list(streamed.shape),
            "state_count": len(module.state_specs),
            "chunk_frames": args.chunk_frames,
            "max_abs": err,
            "tolerance": args.tolerance,
        },
        flush=True,
    )
    return 0 if err <= args.tolerance else 1


def print_state_layout(args) -> int:
    real_gate.load_qwen_tts_symbols(args.qwen_tts_root)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    model = real_gate.build_real_decoder(args.onnx_dir / "config.json", args.onnx_dir / "model.onnx", dtype=dtype)
    module = StatefulCode2WavModule(model).eval()
    layout = [
        {"name": spec.name, "input": f"{spec.name}_in", "output": f"{spec.name}_out", "shape": list(spec.shape), "dtype": str(spec.dtype)}
        for spec in module.state_specs
    ]
    print(json.dumps({"state_count": len(layout), "states": layout}, indent=2), flush=True)
    return 0


def export_onnx(args) -> int:
    real_gate.load_qwen_tts_symbols(args.qwen_tts_root)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    model = real_gate.build_real_decoder(args.onnx_dir / "config.json", args.onnx_dir / "model.onnx", dtype=dtype)
    module = StatefulCode2WavModule(model).eval()
    codes = torch.zeros((1, model.config.num_quantizers, args.chunk_frames), dtype=torch.long)
    position_offset = torch.zeros((1,), dtype=torch.long)
    states = module.initial_states(codes.device)
    input_names = ["codes", "position_offset"] + [f"{spec.name}_in" for spec in module.state_specs]
    output_names = ["waveform"] + [f"{spec.name}_out" for spec in module.state_specs]
    dynamic_axes = {"codes": {2: "chunk_frames"}, "waveform": {2: "waveform_len"}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module,
        (codes, position_offset, *states),
        str(args.output),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    print({"status": "exported", "output": str(args.output), "state_count": len(module.state_specs)}, flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--codes", type=Path, required=True)
    parser.add_argument("--qwen-tts-root", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--chunk-frames", type=int, default=4)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--tolerance", type=float, default=2e-4)
    parser.add_argument("--print-layout", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("qwen3_tts_code2wav_stateful.onnx"))
    args = parser.parse_args()
    if args.print_layout:
        return print_state_layout(args)
    if args.export:
        return export_onnx(args)
    return run_explicit_state_gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
