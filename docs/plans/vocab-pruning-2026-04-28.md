# Vocab Pruning Spec: Qwen3-ASR-0.6B-v2 + Qwen3-TTS-12Hz-0.6B-Base

**Date**: 2026-04-28  
**Target**: Save ≥350MB GPU weight on Orin Nano 8GB (multilanguage mode)  
**Reference**: AtomGradient Swift-Qwen3-TTS — Qwen3-TTS-1.7B from 2.35GB→808MB, 152K→47K tokens, lossless  

---

## Repo Architecture Findings (from reading export scripts)

- **ASR**: `embed_tokens.bin` is a **separate FP16 binary** loaded by Python/C++ at runtime — it is NOT embedded in the ONNX graph. `lm_head` IS inside the ONNX (as an initializer). So ASR has two independent vocab-sized tensors to prune.
- **TTS Talker**: uses `inputs_embeds` (pre-embedded) and outputs a `codec_head` (vocab=2048 CP). The text-token `embed_tokens` is loaded externally before the ONNX call. `lm_head` style text-logits may not exist — Talker outputs codec logits, not text logits. Confirm before pruning `lm_head`.
- **TTS CP / Vocoder**: vocab=2048. Do NOT prune.

---

## 1. Workload Corpus Definition

**Method**:
- **ASR corpus** (audio→text, output tokens only matter): FLORES-200 dev/devtest (all 200 langs, ~50K sentences) + OPUS-100 sampled 1M lines from top 20 target languages + repo test WAVs transcripts.
- **TTS corpus** (text→audio, input tokens matter): same FLORES-200 + user dialog dataset (all `text` fields from production logs if available) + punctuation/number/URL/code-switching edge cases (manually curated ~5K lines).
- **Use separate corpora** for ASR and TTS — their active vocab surfaces differ (ASR output side = natural text; TTS input side = user-supplied text, more formatting-heavy).
- Target corpus size: ~5M tokens post-tokenization for each; beyond that marginal new token_ids plateau.

**Risk/Catch**: FLORES-200 over-represents academic prose; misses numerals, URLs, code snippets, and voice-clone reference text. Must manually augment. Over-inclusion from OPUS-100 is acceptable — erring toward keeping more tokens is safer than over-pruning.

**Effort**: 3h (download + tokenize + count script)

**Decision Point**: If production dialog logs exist, weight them 3× — they reflect actual OOV risk better than FLORES. If absent, add 10K lines of synthetic edge cases (phone numbers, addresses, currency, markdown headers).

---

## 2. Active Token Discovery Flow

**Method**:
```python
from transformers import AutoTokenizer
from collections import Counter
import json

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")  # same tokenizer for ASR
counts = Counter()
for line in corpus_lines:
    counts.update(tok.encode(line))

# Always-include special tokens
ALWAYS_INCLUDE = set(tok.all_special_ids)
# Qwen3-specific: lang tags, audio/pad tokens
# Expected special tokens to enumerate:
# - tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
# - "<|im_start|>", "<|im_end|>", "<|endoftext|>"
# - "<|zh|>", "<|en|>", "<|ja|>", "<|ko|>", ... all lang tags in added_tokens.json
# - "<|AUDIO|>", "<|audio_pad|>", "<|audio_bos|>", "<|audio_eos|>" (ASR)
# Check: len(tok.added_tokens_encoder) — expect 100-500 special tokens
for sid in tok.added_tokens_encoder.values():
    ALWAYS_INCLUDE.add(sid)

# Threshold: freq >= 2 across corpus (not just top-N)
FREQ_THRESHOLD = 2
keep_ids = ALWAYS_INCLUDE | {tid for tid, cnt in counts.items() if cnt >= FREQ_THRESHOLD}
# Expect ~45K-55K tokens kept
```

**Risk/Catch**: Qwen3 added_tokens may include audio-specific sentinel tokens not present in standard Qwen3 text tokenizer — must inspect `tokenizer_config.json` and `added_tokens.json` from HF cache. Missing even one lang-tag token will cause silent wrong decoding.

**Effort**: 2h

**Decision Point**: If kept set > 80K tokens (savings <150MB), raise threshold to freq ≥ 5 or top-60K. If kept set < 40K, lower to freq ≥ 1 (include any observed token). Target 47K±10K matches the AtomGradient result.

---

## 3. ONNX Surgery Steps

**Method**:

For **ASR embed_tokens.bin** (external binary, not ONNX):
```python
import numpy as np
embed = np.fromfile("embed_tokens.bin", dtype=np.float16).reshape(151936, 1024)
keep_ids_sorted = sorted(keep_ids)
pruned_embed = embed[keep_ids_sorted]  # shape: (N_kept, 1024)
pruned_embed.tofile("embed_tokens_pruned.bin")
```

For **ASR lm_head inside ONNX** (initializer):
```python
import onnx, numpy as np
model = onnx.load("asr_decoder.onnx")
for init in model.graph.initializer:
    if init.name == "lm_head.weight":  # confirm name via netron
        w = np.frombuffer(init.raw_data, dtype=np.float16).reshape(151936, 1024)
        pruned_w = w[keep_ids_sorted]
        init.CopyFrom(onnx.numpy_helper.from_array(pruned_w.astype(np.float16), name=init.name))
onnx.save(model, "asr_decoder_pruned.onnx")
```

For **TTS Talker embed_tokens** (external, verify first):
- Check `export_talker_no_if.py` — if `inputs_embeds` are pre-computed before ONNX call, then the embedding table is also external `.bin`. Apply same numpy slice.
- If `lm_head` exists in Talker ONNX (text logits, not codec): apply same ONNX surgery. If only codec_head (vocab=2048): skip, nothing to prune.

**Risk/Catch**: ONNX `lm_head` weight name may differ (`/lm_head/weight`, `decoder.lm_head.weight`, etc.). Use `netron` or `onnx.load` + iterate initializer names to confirm before hardcoding. Shape must be `[vocab, hidden]` not `[hidden, vocab]` — check transpose usage.

**Effort**: 4h

**Decision Point**: If lm_head is a `Gather` node rather than a `MatMul` over a full initializer, the surgery is more complex (need to replace Gather indices). In that case, fall back to exporting a fresh ONNX with `output_attentions=False` and the reduced tokenizer config.

---

## 4. Runtime Token Map Indirection

**Method**:
```python
keep_ids_sorted = sorted(keep_ids)
orig2red = {orig: red for red, orig in enumerate(keep_ids_sorted)}
red2orig = {red: orig for orig, red in orig2red.items()}

# Serialize to compact binary: 2x uint32 arrays, ~47K×4B×2 = ~376KB
import struct
with open("token_map.bin", "wb") as f:
    f.write(struct.pack(f"{len(keep_ids_sorted)}I", *keep_ids_sorted))  # red→orig lookup (array index = reduced_id)
```

**C++ integration points** (from reading `asr_pipeline.cpp`, `tts_pipeline.cpp`):
- **Input side** (tokenizer → model): after calling HF tokenizer (Python via pybind or C++ tokenizer), map each `orig_id → red_id` using `orig2red` hash map before building the embedding lookup into `embed_tokens_pruned.bin`.
- **Output side** (model → detokenize): after argmax/sample_topk_cpu over reduced logits (size N_kept), map result `red_id → orig_id` before passing to tokenizer decode.
- Both maps fit in L2 cache (~376KB). Use `std::unordered_map<int32_t, int32_t>` or a direct lookup array (`orig2red` as a 152K-element `int32_t` array initialized to -1, with valid entries filled).

**Risk/Catch**: C++ sample_topk currently operates on raw logit buffer of size 151936. After pruning, buffer is N_kept. The argmax result is a `reduced_id` — must remap before any token-specific logic (e.g., EOS check uses `tok.eos_token_id` = original ID, so compare against `orig2red[eos_token_id]` in the reduced space).

**Effort**: 3h (map generation) + 4h (C++ plumbing)

**Decision Point**: For Phase A Python-only validation, implement indirection purely in Python wrapper. Defer C++ changes to Phase C so Phase A/B can proceed without C++ build cycle.

---

## 5. Tied Weights Handling

**Method**:
- Qwen3 base models use tied embed/lm_head in PyTorch (`model.lm_head.weight is model.embed_tokens.weight == True`).
- When exported to ONNX via the scripts in this repo, the tie is **typically broken** — both tensors become separate initializers (ONNX has no concept of shared references). Verify: `grep -c "embed_tokens\|lm_head" <(python -c "import onnx; m=onnx.load('asr.onnx'); [print(i.name) for i in m.graph.initializer]")`.
- If both appear as separate initializers with identical data: prune both with the same `keep_ids_sorted`. No special handling needed.
- If only one appears (tied): prune that one; ONNX will use it for both Gather and MatMul.

**Risk/Catch**: After export, the ASR model keeps `embed_tokens` external (`.bin` file) — so it is already effectively untied from the ONNX `lm_head`. This is the expected case based on reading `export_qwen3_asr_unified.py`.

**Effort**: 0.5h (verification only)

**Decision Point**: If TTS Talker ONNX contains neither embed initializer nor lm_head initializer (pure codec-head model), the only pruning target is the external embed `.bin`. Accept ~205MB savings from that alone.

---

## 6. Validation Flow

**Method**:

**Step 1 — Tensor equivalence** (Python, WSL):
```python
# Feed same input_ids through original and pruned paths
orig_embed = full_embed[input_ids]          # (seq, 1024)
red_input_ids = [orig2red[t] for t in input_ids]
pruned_embed_out = pruned_embed[red_input_ids]  # (seq, 1024)
assert np.allclose(orig_embed, pruned_embed_out, atol=0)  # must be bit-identical
```

**Step 2 — ASR CER on 5 reference WAVs**:
- Run original ONNX pipeline → transcripts.
- Run pruned ONNX pipeline with indirection → transcripts.
- CER must be 0.00 (lossless by construction if indirection is correct).

**Step 3 — TTS log-mel L2**:
- Generate audio from same 3 prompts with original and pruned Talker.
- Compute log-mel spectrogram L2: `np.linalg.norm(mel_orig - mel_pruned)`.
- Threshold: < 1e-3 (FP16 accumulation epsilon). If > 1e-3, check for wrong slice indices.

**Risk/Catch**: Any difference > epsilon in Step 3 indicates an off-by-one in `keep_ids_sorted` indexing. Step 1 bit-equality catches this before running full TTS.

**Effort**: 3h

---

## 7. Per-Model Savings Estimate

| Tensor | Original | Pruned (47K) | Saving |
|--------|----------|--------------|--------|
| ASR embed_tokens.bin | 151936×1024×2B = **297MB** | 47K×1024×2B = **92MB** | **~205MB** |
| ASR lm_head (ONNX) | 151936×1024×2B = **297MB** | 47K×1024×2B = **92MB** | **~205MB** |
| TTS Talker embed_tokens.bin | ~297MB (if text vocab) | ~92MB | **~205MB** |
| TTS Talker lm_head (if present) | ~297MB | ~92MB | **~205MB** |
| TTS CP / Vocoder | vocab=2048, not pruned | — | 0 |

**Realistic total** (conservative — not all of the above may be GPU-resident simultaneously):
- ASR: 205MB + 205MB = **~410MB** ✓
- TTS Talker text embed only (if no text lm_head): **~205MB**
- Combined minimum: **~410MB** (ASR only already exceeds 350MB target)

**Caution**: if embed_tokens.bin is pinned to CPU RAM (not GPU), it doesn't help GPU OOM — verify with `torch.cuda.memory_summary()` that embed lookup happens on GPU.

---

## 8. Risk Points

| Risk | Severity | Mitigation |
|------|----------|-----------|
| OOV multilingual token | Medium | BPE fallback still works — rare word gets split into sub-tokens that are in reduced vocab. Worst case: slightly longer token sequence, no crash. |
| Voice-clone reference OOV | Low-Medium | Reference text is typically short (1-3 sentences). Add all tokens from a curated 50K-line "arbitrary reference" corpus to the keep set. |
| C++ argmax scope | Medium | EOS/BOS comparisons must use `orig2red[eos_id]` in reduced space. One missed comparison = silent wrong decoding. Write a unit test for each sentinel. |
| Tokenizer.json unchanged | Low | Do NOT replace tokenizer. Use indirection layer. Tokenizer stays at 152K; only model tensors are pruned. |
| Embed in CPU RAM only | High-risk if true | If embed lookup is CPU→GPU copy per step, pruning helps RAM not VRAM. Profile with `nvtx` before assuming GPU savings. |
| TTS Talker has no text lm_head | Medium | Confirmed via script reading: Talker outputs codec logits (vocab=2048). Text lm_head pruning may not apply. Savings from Talker = embed only (~205MB). |

---

## 9. Implementation Phases

**Phase A** (WSL, Python only, 1-2 days):
1. Download FLORES-200 + OPUS-100 sample → tokenize → build freq map → `keep_ids.json`
2. Slice `embed_tokens.bin` → `embed_tokens_pruned.bin`
3. ONNX surgery on `lm_head` in ASR decoder ONNX
4. Python indirection wrapper
5. Validation Steps 1-3

**Phase B** (Nano, 1 day):
1. Copy pruned ONNX + pruned embed bin to Nano
2. `trtexec` rebuild of ASR decoder TRT engine with reduced ONNX
3. `nvidia-smi` / `torch.cuda.memory_summary()` before/after comparison

**Phase C** (macOS/WSL C++, 1-2 days):
1. Load `token_map.bin` in C++ startup
2. Patch tokenizer input path (orig→red mapping)
3. Patch `sample_topk_cpu` / argmax output path (red→orig)
4. Patch all EOS/BOS/sentinel comparisons

**Phase D** (Nano, 1 day):
1. End-to-end multilanguage ASR test (5 langs, 5 WAVs each)
2. End-to-end TTS test (3 prompts, measure TTFT + audio quality)
3. Memory measurement: `free -h` + `nvidia-smi --query-gpu=memory.used`

**Total effort**: ~6-8 days engineering (Phase A is the highest-uncertainty gate).

---

## Final Assessment

**Probability ≥70% of saving ≥350MB**: **YES — ~80% confidence**

- ASR embed + ASR lm_head alone = ~410MB if both are GPU-resident. This already beats the 350MB target.
- Main uncertainty: whether ASR lm_head is truly a separate 297MB initializer in the exported ONNX, or is referenced differently. Reading `export_qwen3_asr_unified.py` suggests it is — but must verify with `onnx.load` on the actual file.

**Recommended corpus + threshold**:
- FLORES-200 devtest (all 200 langs) + OPUS-100 sampled 500K lines (top 20 langs) + 10K synthetic edge cases
- Threshold: **freq ≥ 2** (include token if seen at least twice across corpus)
- Always-include: all `tok.all_special_ids` + all `tok.added_tokens_encoder` values
- Expected result: 45K-55K kept tokens (~70% reduction in vocab size)

**Fallback — GGUF Q5_K_M path**:
- If ONNX surgery proves error-prone or C++ indirection causes subtle bugs, export the full Qwen3-ASR-0.6B-v2 to GGUF and run via `llama.cpp` on CPU (iGPU offload with `-ngl` layers).
- Q5_K_M at 0.6B ≈ 400-450MB total model size vs. original ~600MB FP16.
- Savings ~150-200MB, less than pruning, but zero code change risk.
- Latency penalty: llama.cpp CPU inference ~3-5× slower than TRT. Only viable if Nano can sustain RTF < 1.0 for ASR at this speed.
- Alternative fallback: INT8 quantize the embed + lm_head layers only (TRT `--int8` on those nodes) — saves ~50% of vocab tensor memory with minimal quality loss.

