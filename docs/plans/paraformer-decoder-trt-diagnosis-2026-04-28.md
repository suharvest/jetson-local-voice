# Paraformer Decoder TRT Build Failure — Root Cause Diagnosis (2026-04-28)

## 1. Root Cause Analysis

The decoder ONNX path from the M1 manifest is `/opt/models/paraformer-streaming/decoder.onnx`. The same manifest records the decoder as a 228,464,044 byte ONNX file with MD5 `4eb7c94ece0ad861f18ef56db5f72379`, 2,232 nodes, and no `Scan`/`Loop`/`If` control-flow nodes. Its real inputs are `enc`, `enc_len`, `acoustic_embeds`, `acoustic_embeds_len`, and `in_cache_0` through `in_cache_15`. The cache tensors are fixed-depth inputs with shape [`batch_size`, 512, 10]. The outputs are `logits`, `sample_ids`, and `out_cache_0` through `out_cache_15`.

Remote ONNX inspection through `fleet` was requested but was not available in this local sandbox. The alias invocation failed with `zsh:1: command not found: fleet`. The documented fallback via `uv run --project ~/project/_hub python ~/project/_hub/fleet.py` failed because `uv` attempted to open `/Users/harvest/.cache/uv/sdists-v9/.git`, outside the sandbox. Retrying with `UV_CACHE_DIR=/tmp/uv-cache` still returned `Error: [Errno 1] Operation not permitted`. Direct `python3 ~/project/_hub/fleet.py` failed because local Python does not have `paramiko`. Therefore the full requested ONNX inspection output was not available. A local fallback search found no local Paraformer decoder ONNX copy under this repo; it only found `/Users/harvest/project/jetson-voice/models/qwen3-asr-v2/decoder_step.onnx`, which is unrelated.

The structural analysis below therefore uses two evidence sources: the M1 manifest and bounded greps of the previous deepseek job log. The job log includes a cached Python TensorRT parse of `/tmp/decoder.onnx`, which exposed actual TRT layer/input/output names, and it includes the decoder `trtexec` failure.

The cached decoder parse shows the following concrete decoder I/O and first FSMN Conv layer:

```text
Input 12: in_cache_8 (-1, 512, 10)
Input 13: in_cache_9 (-1, 512, 10)
Input 14: in_cache_10 (-1, 512, 10)
Input 15: in_cache_11 (-1, 512, 10)
Input 16: in_cache_12 (-1, 512, 10)
Input 17: in_cache_13 (-1, 512, 10)
Input 18: in_cache_14 (-1, 512, 10)
Input 19: in_cache_15 (-1, 512, 10)
Output 0: logits (-1, -1, 8404)
Output 1: sample_ids (-1, -1)
Output 2: out_cache_0 (-1, 512, -1)
...
Output 17: out_cache_15 (-1, 512, -1)
Layer 186: /decoder/decoders.0/self_attn/fsmn_block/Conv [LayerType.CONVOLUTION]
```

This is the important architectural clue. The input caches are fixed to depth 10, but TRT sees the output cache time dimension as dynamic (`-1`). The first FSMN convolution is `/decoder/decoders.0/self_attn/fsmn_block/Conv`; the log later confirms TensorRT fails while building that exact layer.

The decoder build command in the evidence was:

```text
/usr/src/tensorrt/bin/trtexec --onnx=/tmp/decoder.onnx --minShapes=enc:1x1x512,enc_len:1,acoustic_embeds:1x1x512,acoustic_embeds_len:1 --optShapes=enc:1x20x512,enc_len:1,acoustic_embeds:1x10x512,acoustic_embeds_len:1 --maxShapes=enc:1x40x512,enc_len:1,acoustic_embeds:1x40x512,acoustic_embeds_len:1 --fp16 --memPoolSize=workspace:4096 --saveEngine=/tmp/paraformer-engines/paraformer_decoder_sp1_40.plan --verbose
```

The verbatim fatal `trtexec` snippet is:

```text
[04/27/2026-18:38:45] [V] [TRT] =============== Computing costs for /decoder/decoders.0/self_attn/fsmn_block/Conv
[04/27/2026-18:38:45] [V] [TRT] *************** Autotuning format combination: Float((MUL_ADD 512 E1 5120),E0,E0,1) where E0=(+ E1 10) E1=(BROADCAST_SIZE token_length (MAX 0 (- 0 (CAST_F32_TO_I (FLOOR (MUL_ADD_F -1 (CAST_I_TO_F32 (# 0 (VALUE /decoder/make_pad_mask/ReduceMax_output_0))) 0)))))) -> Float((* 512 E0),E0,E0,1) where E0=(BROADCAST_SIZE token_length (MAX 0 (- 0 (CAST_F32_TO_I (FLOOR (MUL_ADD_F -1 (CAST_I_TO_F32 (# 0 (VALUE /decoder/make_pad_mask/ReduceMax_output_0))) 0)))))) where E0=(+ E1 10) E1=(BROADCAST_SIZE token_length (MAX 0 (- 0 (CAST_F32_TO_I (FLOOR (MUL_ADD_F -1 (CAST_I_TO_F32 (# 0 (VALUE /decoder/make_pad_mask/ReduceMax_output_0))) 0)))))) ***************
[04/27/2026-18:38:45] [E] Error[2]: [convBaseBuilder.cpp::createConvolution::259] Error Code 2: Internal Error (Assertion isOpConsistent(convolution.get()) failed. Cask convolution isConsistent check failed.)
[04/27/2026-18:38:45] [E] Engine could not be created from network
[04/27/2026-18:38:45] [E] Building engine failed
[04/27/2026-18:38:45] [E] Failed to create engine from model or file.
[04/27/2026-18:38:45] [E] Engine set up failed
&&&& FAILED TensorRT.trtexec [TensorRT v100300] # /usr/src/tensorrt/bin/trtexec --onnx=/tmp/decoder.onnx --minShapes=enc:1x1x512,enc_len:1,acoustic_embeds:1x1x512,acoustic_embeds_len:1 --optShapes=enc:1x20x512,enc_len:1,acoustic_embeds:1x10x512,acoustic_embeds_len:1 --maxShapes=enc:1x40x512,enc_len:1,acoustic_embeds:1x40x512,acoustic_embeds_len:1 --fp16 --memPoolSize=workspace:4096 --saveEngine=/tmp/paraformer-engines/paraformer_decoder_sp1_40.plan --verbose
```

The same failure reproduced at `--builderOptimizationLevel=0`:

```text
[04/27/2026-18:43:58] [E] Error[2]: [convBaseBuilder.cpp::createConvolution::259] Error Code 2: Internal Error (Assertion isOpConsistent(convolution.get()) failed. Cask convolution isConsistent check failed.)
&&&& FAILED TensorRT.trtexec [TensorRT v100300] # /usr/src/tensorrt/bin/trtexec --onnx=/tmp/decoder.onnx --minShapes=enc:1x1x512,enc_len:1,acoustic_embeds:1x1x512,acoustic_embeds_len:1 --optShapes=enc:1x20x512,enc_len:1,acoustic_embeds:1x10x512,acoustic_embeds_len:1 --maxShapes=enc:1x40x512,enc_len:1,acoustic_embeds:1x40x512,acoustic_embeds_len:1 --builderOptimizationLevel=0 --saveEngine=/tmp/paraformer-engines/paraformer_decoder_opt0.plan
```

This rules out a simple high-optimization tactic issue. The failing op is a depthwise FSMN `Conv1d` inside the decoder, and the build-time shape expression for that conv is polluted by the dynamic `make_pad_mask` path:

```text
/decoder/make_pad_mask/ReduceMax [ReduceMax] outputs=['/decoder/make_pad_mask/ReduceMax_output_0'] inputs=['acoustic_embeds_len']
/decoder/make_pad_mask/Range [Range] outputs=['/decoder/make_pad_mask/Range_output_0'] inputs=['/decoder/make_pad_mask/Constant_output_0', '/decoder/make_pad_mask/Cast_output_0', '/decoder/make_pad_mask/Constant_1_output_0']
/decoder/make_pad_mask/Less [Less] outputs=['/decoder/make_pad_mask/Less_output_0'] inputs=['/decoder/make_pad_mask/Cast_1_output_0', '/decoder/make_pad_mask/Cast_2_output_0']
/decoder/make_pad_mask_1/Range [Range] outputs=['/decoder/make_pad_mask_1/Range_output_0'] inputs=['/decoder/make_pad_mask_1/Constant_output_0', '/decoder/make_pad_mask_1/Cast_output_0', '/decoder/make_pad_mask_1/Constant_1_output_0']
```

The root cause is not `If`, `Loop`, `Scan`, or `NonZero`; none are known from the manifest. The likely root cause is the interaction of dynamic `Range`/mask length expressions with the FSMN Conv input length. TensorRT’s Cask convolution consistency check receives a shape expression based on `BROADCAST_SIZE token_length` and a `MAX 0 ... ReduceMax(acoustic_embeds_len)` expression, then rejects the resulting convolution. The decoder’s `Range` and `Reshape` ops are blacklist-risk ops in the manifest, but the direct fatal op is `/decoder/decoders.0/self_attn/fsmn_block/Conv`.

There was also a memory warning:

```text
[04/27/2026-18:38:45] [W] [TRT] Tactic Device request: 4096MB Available: 307MB. Device memory is insufficient to use tactic.
```

This is a secondary pressure point, not the primary root cause, because the same Cask consistency assertion occurred even with builder optimization level 0.

## 2. Candidate Solutions

| Option | Description | Surgery ops | Effort | Risk | deepseek-ready? |
|--------|-------------|-------------|--------|------|-----------------|
| A | encoder-only TRT + decoder ORT-CUDA fallback | none | 0h | low | yes |
| B | decoder ONNX surgery: replace dynamic `make_pad_mask` `Range` path with static/bucketed mask path, then rebuild decoder for token buckets | ONNX node rewrites around `/decoder/make_pad_mask/*`, `/decoder/make_pad_mask_1/*`, and validate `/decoder/decoders.0/self_attn/fsmn_block/Conv` | 6-10h | medium | yes |
| C | split into 16 FSMN engines + Python scheduler | graph partition | 24-40h | high | no |
| D | Pad + single Conv, remove Slice/Range-driven cache length coupling | node substitution around `Concat`/`Conv`/cache outputs for each FSMN layer | 10-16h | medium-high | partial |

Option A is operationally safest. It avoids the decoder’s dynamic shape/Cask failure entirely and still preserves most of the likely ASR speedup if the encoder dominates cost. The backend must label the provider honestly as encoder TRT plus decoder ORT-CUDA, not full TRT.

Option B is the most plausible route to full decoder TRT. The actual failure expression depends on `/decoder/make_pad_mask/ReduceMax_output_0` and `/decoder/make_pad_mask/Range_output_0`. For a first cut, build fixed token buckets, for example token length 10, 20, and 40. Replace mask-generation subgraphs with bucket-shaped constants or `ConstantOfShape`/`Expand` from tensor `Shape` rather than scalar length values. Then verify that `/decoder/decoders.0/self_attn/fsmn_block/Conv` no longer receives a `BROADCAST_SIZE token_length` expression derived from `ReduceMax(acoustic_embeds_len)`.

Option C is technically possible but too invasive for M4. The decoder has 16 FSMN/cache layers; splitting them into separate engines creates a Python scheduler and cache ABI surface that must exactly match the original model.

Option D may be better than Option B if the FSMN cache export is the true trigger rather than only the mask path. It would rewrite each FSMN block as an explicitly padded, fixed-bucket depthwise Conv and explicitly slice the last 10 cache frames with constants. However, it touches every FSMN layer and has higher correctness risk.

## 3. Recommended Path + Rationale

Recommended path: Option A for immediate M4 retry, with Option B as a follow-on full-TRT experiment.

The evidence shows the decoder fails inside TensorRT engine creation, not during service integration. The fatal op is `/decoder/decoders.0/self_attn/fsmn_block/Conv`, and TensorRT’s own shape expression shows that `/decoder/make_pad_mask/ReduceMax_output_0` participates in the conv’s dynamic dimension. This makes the decoder ONNX a poor candidate for blind `trtexec` retry. Repeating builds with different optimization levels already failed.

Expected success rate for Option A is high, around 80-90%, because it avoids the failing graph section and only requires runtime plumbing. Expected success rate for Option B is moderate, around 50-65%, because it depends on replacing all mask/cache dynamic expressions without perturbing logits. The first full-TRT surgery should be validated by ORT parity before another long `trtexec` build.

## 4. deepseek M4 Retry Prompt Template

```text
Goal: complete Paraformer M4 with encoder TRT and decoder ORT-CUDA fallback first. Do not attempt full decoder TRT unless the fallback path passes.

Working dir on orin-nx: /home/recomputer/jetson-voice
Model paths:
- encoder ONNX: /opt/models/paraformer-streaming/encoder.onnx
- decoder ONNX: /opt/models/paraformer-streaming/decoder.onnx
- tokens: /opt/models/paraformer-streaming/tokens.txt

Hard evidence from diagnosis:
- decoder fatal TRT layer: /decoder/decoders.0/self_attn/fsmn_block/Conv
- decoder mask nodes involved in failing shape expression:
  - /decoder/make_pad_mask/ReduceMax
  - /decoder/make_pad_mask/Range
  - /decoder/make_pad_mask/Less
  - /decoder/make_pad_mask_1/Range
- decoder inputs include enc, enc_len, acoustic_embeds, acoustic_embeds_len, in_cache_0..in_cache_15
- in_cache_8..in_cache_15 parsed as (-1, 512, 10)
- outputs include logits (-1, -1, 8404), sample_ids (-1, -1), out_cache_0..15 (-1, 512, -1)
- verbatim TRT failure: Error[2] convBaseBuilder.cpp::createConvolution::259, Assertion isOpConsistent(convolution.get()) failed, Cask convolution isConsistent check failed.

Step limit: 35 bash steps total. Number each step in the raw log.

Required implementation:
1. Use existing or newly built encoder TRT engine only if it runs at the actual streaming chunk shape. Do not depend on a plan that only exists after failed verification unless you validate runtime enqueue at the intended shape.
2. Keep decoder in ONNX Runtime CUDA EP. If CUDA EP is unavailable, fail clearly or label CPU fallback explicitly.
3. Implement provider labels in logs/metrics: encoder=trt, decoder=ort_cuda or decoder=ort_cpu.
4. Start service with ASR_BACKEND=paraformer_trt and verify /asr/stream reset, end_utterance, partial, and final behavior.
5. Report chunk latency P50 for 400 ms chunks.

Optional full-decoder surgery only after fallback passes:
1. Copy decoder ONNX to a new file, never modify the original.
2. Locate exact nodes:
   - /decoder/make_pad_mask/ReduceMax
   - /decoder/make_pad_mask/Range
   - /decoder/make_pad_mask/Less
   - /decoder/make_pad_mask_1/Range
   - /decoder/decoders.0/self_attn/fsmn_block/Conv
3. Write a surgery script skeleton:
   - load ONNX
   - inspect consumers of /decoder/make_pad_mask/Cast_3_output_0 and /decoder/make_pad_mask_1/Cast_3_output_0
   - replace Range/Less mask generation with bucket-static all-ones masks for token buckets 10, 20, 40
   - make cache Slice starts/ends constants where possible
   - save decoder_bucket{10,20,40}.onnx
   - run onnx.checker and ORT parity against original for logits/sample_ids on random valid inputs
   - only then run trtexec on one bucket

Required EVIDENCE in final report:
- md5 and size of every engine and modified ONNX
- exact provider labels
- raw trtexec success/failure snippets
- ORT-vs-original parity numbers if surgery is attempted
- websocket transcript log with reset/end_utterance/partial/final
- latency summary and whether decoder is TRT or fallback

Forbidden:
- no docker rm, no docker compose down/up
- no editing original ONNX in place
- no deleting existing engines except files created in this run
- no hiding fallback provider labels
```
