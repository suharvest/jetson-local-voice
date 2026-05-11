# Qwen3-TTS Correctness Contract

Date: 2026-05-07

This is the executable contract for the Seeed Local Voice Qwen3-TTS product path.
If a path does not satisfy these conditions, it is not a correctness candidate
even if it can generate a WAV.

## Required Product Path

- Product backend mode must be `product_explicit_kv`.
- The active Python backend must load `backends.qwen3_trt` from the expected
  product overlay or repo path.
- The active native module must be the intended
  `qwen3_speech_engine*.so`; stale overlay modules are a hard failure.
- Generic text segmentation must not run before `product_explicit_kv`.
  Long text must stay in one Talker/CP request/session unless a separate
  state-transfer strategy is explicitly implemented.

## Precision And Engine Contract

- Talker decode uses the known-good explicit-KV TensorRT engine.
- Talker autoregressive loop boundary must expose FP32-equivalent state:
  `inputs_embeds`, `logits`, `last_hidden`, and Talker KV tensors are checked
  from the TensorRT engine metadata when TensorRT Python is available.
- CodePredictor uses the unified CP KV engine built in BF16 mode:
  `engines/cp_unified_bf16.engine`.
- CP engine I/O may still be FP32. The critical requirement is BF16-capable
  TensorRT compute/tactics inside the CP engine, because CP FP16 can overflow.
- Product config must keep `cp_active_groups == 15`, i.e. all residual
  codebooks. `cp_active_groups == 13` was a historical performance experiment;
  it zero-fills the last two residual groups and changes the later primary-code
  trajectory, so it is not the default correctness path.

## Sampling Contract

- Every product request has a request seed.
- The seed must reach both:
  - primary Talker codec sampling
  - CP residual-code sampling
- Streaming and offline entry points must both pass the seed to C++.
- For the product explicit-KV path, fixed-seed sampling must use the same
  uniform random stream and vocab-order top-k sampling semantics as the old
  Python reference path. Sampling over a sorted top-k list changes CP residual
  codes even when logits and seed are identical.

## Prefill And Text-Conditioning Contract

- The product explicit-KV path must use the old validated sherpa-style
  `prefill8` layout:
  role tokens, `codec_nothink`, `codec_think_bos`, `codec_think_eos`,
  `codec_pad`, and `codec_bos`.
- Do not insert the language codec token in this product path unless a new
  reference is generated and accepted. The 9-token layout changed the
  autoregressive trajectory and regressed listening quality.
- Every trailing text/eos addend must include `codec_pad`. Dropping
  `codec_pad` from trailing embeddings changes the Talker input distribution
  and was associated with swallowed tails and voice drift.

## Vocoder Contract

- The currently installed TRT vocoder engine has a fixed output tensor of
  `192000` samples, i.e. 8 seconds at 24 kHz.
- Short offline product requests may cap `max_frames` to
  `TTS_TRT_VOCODER_MAX_FRAMES` (default `100`) and use one-shot vocoder.
- Long offline product requests must not restart text segments. They should
  collect the product streaming/chunked vocoder path under the hood so Talker
  and CP state remains continuous through the whole utterance.
- Full-length long-form output requires one of:
  - a re-exported TRT vocoder engine with a longer output tensor
  - a quality-approved ORT vocoder path
  - a quality-approved streaming/chunked vocoder path

## Required Verification

Run this on the target Jetson before accepting a path:

```bash
PYTHONPATH=/tmp/jetson-voice-product-layer-0507/app:/home/harvest/voice_test/app_overlay \
SEEED_LOCAL_VOICE_TTS_BACKEND=product_explicit_kv \
SEEED_LOCAL_VOICE_TTS_MODEL_BASE=/home/harvest/voice_test/models/qwen3-tts \
SEEED_LOCAL_VOICE_TTS_NATIVE_MODULE_DIR=/home/harvest/voice_test/app_overlay \
python3 scripts/verify_qwen3_tts_contract.py \
  --run-sample \
  --output /tmp/qwen3_tts_contract.wav \
  --text "语音合成的稳定性。"
```

The script must print `CONTRACT_OK`. Any `CONTRACT_FAIL` means the path is not
the validated product path.
