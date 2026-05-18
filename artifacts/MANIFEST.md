# Runtime Artifacts Manifest

This directory documents large artifacts that are not meant to be committed to
git. Deployment scripts now prefer Hugging Face manifests and official upstream
model sources over checked-in binary blobs.

## Jetson Qwen3 TensorRT-EdgeLLM Artifacts

| Item | Location | Notes |
|---|---|---|
| Qwen3 ASR/TTS exported ONNX, TensorRT engines, and worker metadata | [`harvestsu/qwen3-edgellm-jetson-artifacts`](https://huggingface.co/harvestsu/qwen3-edgellm-jetson-artifacts) | Downloaded and SHA-256 verified by the Qwen3 high-performance reproduction flow. |
| Export/build/runtime scripts | [`third_party/qwen3-edgellm-jetson`](../third_party/qwen3-edgellm-jetson) | Pinned submodule; owns Jetson Qwen3 artifact generation. |
| Slim runtime workers bundled in this repo | [`deploy/jetson-workers`](../deploy/jetson-workers) | Copied into the Jetson image so runtime does not need a full build toolchain. |

Validated image:

- `sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:jetson-v1.12-highperf`
- Profile used for the 2026-05-17 release gate: `jetson-multilang-highperf`

The lightweight Jetson `zh_en` path still downloads official sherpa-onnx /
Matcha assets at first boot and does not require Qwen3 artifacts.

## Rockchip RK3576/RK3588 Artifacts

| Item | Location | Notes |
|---|---|---|
| RKNN/RKLLM generated artifacts | [`harvestsu/seeed-local-voice-rk-artifacts`](https://huggingface.co/harvestsu/seeed-local-voice-rk-artifacts) | Contains only generated files that are not already hosted by official sources. |
| Deploy manifest | [`deploy/artifacts/rk_manifest.json`](../deploy/artifacts/rk_manifest.json) | Source paths, sizes, SHA-256 hashes, runtime version hints, and artifact set names. |
| RK runtime sidecars | [`deploy/rk-runtime`](../deploy/rk-runtime) | Bundled into the RK image to reduce runtime drift across devices. |

Validated image:

- `sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:rk-v1.4-closedloop`

## Raspberry Pi / CPU Artifacts

Raspberry Pi uses ONNX through sherpa-onnx. It does not use precompiled engine
artifacts and therefore has no device-runtime compatibility matrix.

Validated image:

- `sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:rpi-v1.0-onnx`

## Legacy Local Files

The older files under `artifacts/engines/` and `artifacts/onnx/` describe the
first Matcha/Paraformer TensorRT split experiments:

- Paraformer encoder/decoder TensorRT plans for Orin Nano.
- Split Matcha estimator TensorRT engines.
- Surgery-produced ONNX files for those builds.

They are retained for historical reproducibility, but new deployment should use
the manifest-driven paths above.
