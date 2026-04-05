#!/usr/bin/env python3
"""Re-export talker_prefill + talker_decode with TorchScript (not dynamo).
Preserves attention_mask as ONNX input for sherpa-onnx C++ compatibility.
"""
import importlib, sys, types, os
# Mock torchaudio
spec = importlib.machinery.ModuleSpec("torchaudio", None)
ta = types.ModuleType("torchaudio"); ta.__spec__ = spec; ta.__path__ = []; ta.__version__ = "0.0.0"
sys.modules["torchaudio"] = ta
for sub in ["compliance", "compliance.kaldi", "_extension", "_extension.utils"]:
    m = types.ModuleType(f"torchaudio.{sub}")
    m.__spec__ = importlib.machinery.ModuleSpec(f"torchaudio.{sub}", None)
    m.__path__ = []
    sys.modules[f"torchaudio.{sub}"] = m

import torch, torch.nn as nn
import transformers.masking_utils

def _simple_causal_mask(config, input_embeds, attention_mask, cache_position, past_key_values, **kwargs):
    dtype = input_embeds.dtype; device = input_embeds.device
    batch, seq_len = input_embeds.shape[:2]
    if past_key_values is not None and hasattr(past_key_values, 'get_seq_length'):
        past_len = past_key_values.get_seq_length()
    else:
        past_len = 0
    total_len = past_len + seq_len
    if seq_len == 1:
        return None
    mask = torch.triu(torch.full((seq_len, total_len), float("-inf"), device=device, dtype=dtype), diagonal=past_len + 1)
    return mask.unsqueeze(0).unsqueeze(0).expand(batch, 1, -1, -1)

transformers.masking_utils.create_causal_mask = _simple_causal_mask

from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration
from transformers import AutoConfig, AutoModel
AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)

print("Loading model...")
model = AutoModel.from_pretrained("/tmp/qwen3-full", device_map="cpu", dtype=torch.float32, attn_implementation="eager")
model.eval()

talker = model.talker
N = talker.config.num_hidden_layers
D = talker.config.hidden_size
H = talker.config.num_key_value_heads
dh = getattr(talker.config, 'head_dim', 128)
OUT = "/tmp/qwen3-sherpa-v2"

# ==================== PREFILL ====================
print("Exporting talker_prefill...")

class Prefill(nn.Module):
    def __init__(self, talker):
        super().__init__()
        self.talker = talker
    def forward(self, inputs_embeds, attention_mask):
        out = self.talker.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                use_cache=True, return_dict=True)
        hidden = out.last_hidden_state
        logits = self.talker.codec_head(hidden[:, -1:, :])
        pkv = out.past_key_values.to_legacy_cache()
        return (logits, hidden) + tuple(t for kv in pkv for t in kv)

w = Prefill(talker).eval()
dummy_e = torch.randn(1, 10, D)
dummy_m = torch.ones(1, 10, dtype=torch.long)

kv_names = []
for i in range(N):
    kv_names += [f"past_key_{i}", f"past_value_{i}"]

torch.onnx.export(w, (dummy_e, dummy_m), f"{OUT}/talker_prefill_ts.onnx",
    input_names=["inputs_embeds", "attention_mask"],
    output_names=["logits", "last_hidden"] + kv_names,
    dynamic_axes={
        "inputs_embeds": {1: "T"}, "attention_mask": {1: "T"},
        "last_hidden": {1: "T"},
        **{n: {2: "T"} for n in kv_names},
    },
    opset_version=14, dynamo=False)
print("  Prefill exported")

# ==================== DECODE ====================
print("Exporting talker_decode...")

from transformers.cache_utils import DynamicCache

class Decode(nn.Module):
    def __init__(self, talker, num_layers):
        super().__init__()
        self.talker = talker
        self.num_layers = num_layers
    def forward(self, inputs_embeds, attention_mask, *past_kv_flat):
        cache = DynamicCache()
        for i in range(self.num_layers):
            cache.update(past_kv_flat[2*i], past_kv_flat[2*i+1], i)
        out = self.talker.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                past_key_values=cache, use_cache=True, return_dict=True)
        hidden = out.last_hidden_state
        logits = self.talker.codec_head(hidden)
        new_pkv = out.past_key_values.to_legacy_cache()
        return (logits, hidden) + tuple(t for kv in new_pkv for t in kv)

w2 = Decode(talker, N).eval()
T_past = 10
dummy_e2 = torch.randn(1, 1, D)
dummy_m2 = torch.ones(1, T_past + 1, dtype=torch.long)
dummy_kv = [torch.randn(1, H, T_past, dh) for _ in range(N * 2)]

in_kv = []; out_kv = []
for i in range(N):
    in_kv += [f"past_key_{i}", f"past_value_{i}"]
    out_kv += [f"new_past_key_{i}", f"new_past_value_{i}"]

torch.onnx.export(w2, (dummy_e2, dummy_m2, *dummy_kv), f"{OUT}/talker_decode_ts.onnx",
    input_names=["inputs_embeds", "attention_mask"] + in_kv,
    output_names=["logits", "last_hidden"] + out_kv,
    dynamic_axes={
        "attention_mask": {1: "full_len"},
        **{n: {2: "past_len"} for n in in_kv},
        **{n: {2: "new_len"} for n in out_kv},
    },
    opset_version=14, dynamo=False)
print("  Decode exported")

# ==================== EXTERNAL DATA ====================
import onnx
for name in ["talker_prefill_ts", "talker_decode_ts"]:
    print(f"  Converting {name} to external data...")
    m = onnx.load(f"{OUT}/{name}.onnx")
    onnx.save_model(m, f"{OUT}/{name}.onnx",
        save_as_external_data=True, all_tensors_to_one_file=True,
        location=f"{name}.onnx.data", size_threshold=1024)
    sz = os.path.getsize(f"{OUT}/{name}.onnx.data") / 1024 / 1024
    m2 = onnx.load(f"{OUT}/{name}.onnx")
    ins = [i.name for i in m2.graph.input][:4]
    ifs = sum(1 for n in m2.graph.node if n.op_type == "If")
    print(f"    {sz:.0f}MB, If={ifs}, inputs={ins}...")

print("\nDone!")
