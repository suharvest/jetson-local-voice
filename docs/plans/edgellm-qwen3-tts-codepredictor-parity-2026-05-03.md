# EdgeLLM Qwen3-TTS CodePredictor Parity Plan - 2026-05-03

## Goal

Bring Qwen3-TTS back to the official TensorRT-Edge-LLM CodePredictor path, identify the parity issue, and prepare a focused upstream PR.

The current special CodePredictor path is a correctness workaround. It should be treated as a diagnostic bridge, not the final integration target.

## Current Finding

On Orin Nano, the same prompt and same Talker/Code2Wav runtime produce different semantic output depending on the CodePredictor path.

Prompt:

```text
你好。
```

Generic EdgeLLM CodePredictor path:

```json
{"text":"是的，是的。","language":"Chinese"}
```

Special CodePredictor path:

```json
{"text":"你好。","language":"Chinese"}
```

The ASR result above was produced by the EdgeLLM ASR backend on the generated full WAV, not on a streaming chunk. This isolates the issue away from the new streaming chunk code.

## Working Hypothesis

This is not a Qwen3-TTS model bug. It is likely a parity issue in the EdgeLLM Qwen3-TTS CodePredictor adapter/export/runtime path.

The failure surface is the residual RVQ code generation path:

```text
Talker selected code_0
  -> CodePredictor prefill generates code_1
  -> CodePredictor decode generates code_2..code_k
  -> residual embeddings are materialized
  -> Talker residual connection consumes the RVQ frame
```

The generic path currently behaves like a normal decoder runner plus manually switched group embeddings/lm_heads. Qwen3-TTS CodePredictor is more specialized than a normal text decoder, so small mismatches in group indexing or KV/cache semantics can produce plausible audio with wrong text.

## Already Excluded

- Not caused by streaming chunking. The mismatch is visible in the final full WAV.
- Not caused only by CUDA graph capture. The failing generic run used `EDGE_LLM_TTS_CUDA_GRAPH=0`.
- Not primarily Code2Wav. The same Code2Wav engine is used by both paths.
- Not primarily tokenizer/template. The same tokenizer/template is used by both paths.

## Candidate Root Causes

1. CodePredictor prefill input mismatch
   - `talkerHiddenState`
   - `code_0` embedding
   - `small_to_mtp_projection`
   - prefill `cache_position`

2. Residual group off-by-one
   - `embeddingIdx = step - 2`
   - `lmHeadIdx = step - 1`
   - `generationStep`
   - active group count and zero-fill behavior

3. KV/cache metadata mismatch
   - `past_length`
   - `cache_position`
   - generic `LLMEngineRunner` assumptions for decode length
   - whether the exported engine expects explicit `gen_step`

4. Embedding/table source mismatch
   - `codec_embeddings.safetensors`
   - `embedding.safetensors`
   - `cp_embed_fp32.bin`
   - dtype or projection differences

5. Export mismatch
   - generic CodePredictor engine may not expose all Qwen3-TTS CP control inputs.
   - special CP engine may encode assumptions that the generic engine path does not reproduce.

## Parity Debug Plan

The next step is to compare RVQ codes, not audio.

Use a fixed prompt, seed, and sampling parameters:

```json
{
  "text": "你好。",
  "language": "chinese",
  "talker_temperature": 0.9,
  "talker_top_k": 50,
  "talker_top_p": 1.0,
  "repetition_penalty": 1.05,
  "max_audio_length": 80
}
```

Dump these values per generated audio frame:

```text
frame_index
code_0 from Talker
code_1 from CP prefill
code_2..code_k from CP decode
active group count
sampled logits top-k ids for each CP group
```

Compare three paths:

1. Reference PyTorch or old native TRT implementation.
2. Current special CP path.
3. Generic EdgeLLM CodePredictor path.

Interpretation:

- If `code_1` diverges first, focus on prefill input, `lm_head_0`, primary embedding, and prefill cache metadata.
- If `code_1` matches but `code_2+` diverges, focus on KV/cache position, `gen_step`, group embedding index, and lm_head index.
- If sampled ids match but residual connection diverges, focus on materialized embeddings in `mCodecHiddensBuffer`.

## Instrumentation To Add

Add debug-only dumps behind an environment variable, for example:

```bash
QWEN3_TTS_DUMP_CP=/tmp/cp_dump.jsonl
```

Each JSONL row should include:

```json
{
  "path": "generic",
  "frame": 0,
  "group": 1,
  "phase": "prefill",
  "sampled": 123,
  "top_ids": [123, 456, 789],
  "top_logits": [1.23, 1.11, 0.98],
  "embedding_idx": 0,
  "lm_head_idx": 0,
  "cache_position": [0, 1],
  "past_length": 0
}
```

Keep this instrumentation local/debug-only. The eventual upstream PR should contain the actual fix and a smaller validation hook or unit test if acceptable to maintainers.

## Upstream PR Shape

Preferred PR direction:

1. Keep the public Qwen3-TTS API unchanged.
2. Fix generic CodePredictor parity so special local env vars are not required.
3. Add a minimal regression test or deterministic RVQ-code parity check.
4. Keep streaming as a separate PR if the CP fix is large.

Do not upstream the current special CP workaround as the final solution unless investigation proves the generic abstraction is structurally incompatible with Qwen3-TTS CP.

## Current Related Commits

EdgeLLM local commits:

```text
b41a314 Support reduced vocab input embeddings
f4912f6 Add streaming chunks for Qwen3 TTS
```

Jetson voice docs:

```text
bde7e11 Document EdgeLLM TTS streaming design
```

## Open Questions

- Does the official exported CodePredictor engine include `gen_step` and `past_length` inputs?
- Are group-specific lm_heads bound in the same order as the reference model?
- Is `codec_embeddings.safetensors` exactly equivalent to the embedding table used by the special CP path?
- Does the generic `executeVanillaDecodingStep` update cache metadata in a way that matches Qwen3-TTS CP?
- Can the generic path be fixed cleanly, or does Qwen3-TTS CP need a first-class runtime path in EdgeLLM?
