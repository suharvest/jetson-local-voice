# Matcha-TTS TRT 与 Paraformer Streaming ASR TRT 规格

## 1. 目标

| 组件 | 指标 | baseline (sherpa/ORT) | 目标 | 拒绝阈值 |
|---|---|---|---|---|
| Matcha | TTFT P50 | ~400ms | ≤200ms | >280ms |
| Matcha | RTF | ~0.4 | ≤0.15 | >0.20 |
| Matcha | mel L2 vs ORT | - | <5% | >10% |
| Paraformer | RTF | ~0.08 | ≤0.05 | >0.07 |
| Paraformer | 400ms chunk P50 | - | ≤80ms | >120ms |
| Paraformer | CER vs sherpa | - | 相对劣化 <2% | >5% |

1. 新增 `matcha_trt` TTS backend，在 `TTS_BACKEND=matcha_trt` 且 `LANGUAGE_MODE=zh_en` 时可替代当前 sherpa Matcha 路径，保持 `/tts` 与 `/tts/stream` API 不变。TTS backend 必须实现 `TTSBackend` 的 `name`、`capabilities`、`sample_rate`、`preload()`、`synthesize()`，并可选实现 `generate_streaming()`；这些是当前抽象层要求的接口（`app/tts_backend.py:28`, `app/tts_backend.py:31`, `app/tts_backend.py:37`, `app/tts_backend.py:43`, `app/tts_backend.py:52`, `app/tts_backend.py:57`, `app/tts_backend.py:88`）。
2. 新增 `paraformer_trt` ASR backend，在 `ASR_BACKEND=paraformer_trt` 且 `LANGUAGE_MODE=zh_en` 时提供真正流式识别，保持 `/asr` 与 `/asr/stream` API 不变。ASR backend 必须实现 `ASRBackend` 的 `preload()`、`transcribe()`、`create_stream()` 与 `ASRStream.accept_waveform()`/`get_partial()`/`finalize()`（`app/asr_backend.py:34`, `app/asr_backend.py:38`, `app/asr_backend.py:43`, `app/asr_backend.py:47`, `app/asr_backend.py:59`, `app/asr_backend.py:76`, `app/asr_backend.py:80`, `app/asr_backend.py:82`）。
3. 性能目标：Matcha TRT 冷启动后首包延迟 P50 <= 200 ms，离线合成 RTF <= 0.10；Paraformer TRT 每 400 ms 音频块处理耗时 P50 <= 80 ms，端点后最终结果延迟 P50 <= 300 ms。所有指标在 Jetson 目标机上用固定中文、英文、混合中英样本集记录到 benchmark 输出。
4. 兼容目标：`/health`、`/tts/capabilities`、`/asr/capabilities` 能正确报告新 backend 名称、能力和采样率，因为当前服务直接从 backend 读取这些字段（`app/main.py:169`, `app/main.py:180`, `app/main.py:190`, `app/main.py:203`）。

## 2. Matcha TTS TRT

架构：

- Python backend：新增 `app/backends/matcha_trt.py`，类名 `MatchaTRTBackend(TTSBackend)`。它负责文本前端、bucket 选择、TensorRT engine 生命周期、WAV/PCM 打包和元数据。现有 `tts_service` 是 backend-agnostic proxy，所有请求最终转发到当前 backend（`app/tts_service.py:23`, `app/tts_service.py:30`, `app/tts_service.py:38`, `app/tts_service.py:71`）。
- Native/runtime 层：新增或复用 C++/pybind11 模块，例如 `matcha_speech_engine.Pipeline`，仿照现有 Qwen3 TRT backend 在 `preload()` 中加载 engine 并常驻内存（`app/backends/qwen3_trt.py:45`, `app/backends/qwen3_trt.py:72`, `app/backends/qwen3_trt.py:91`, `app/backends/qwen3_trt.py:92`）。
- 模型结构：沿用 Matcha acoustic model + Vocos vocoder 两段式。RKNN 版本已经验证了 `文本 -> tokens -> Matcha -> mel -> Vocos -> ISTFT -> 音频` 的拆分（`rk3576/app/backends/rknn_matcha_tts.py:5`），并记录了 bucket 模型、固定帧长和采样率常量（`rk3576/app/backends/rknn_matcha_tts.py:9`, `rk3576/app/backends/rknn_matcha_tts.py:13`, `rk3576/app/backends/rknn_matcha_tts.py:31`）。
- **Vocos 输出格式**（`rkvoice-stream/models/tts/matcha/convert_vocos_16khz_rknn.py:4-10`）：Vocos engine 输出 `(mag, x, y)` STFT 三元组，不是 waveform。Python backend 持有 ISTFT 状态（`torch.istft` 或 `scipy.signal.istft`）在 CPU 完成重建。Benchmark 时延必须计入 ISTFT 耗时。
- TensorRT engine：建议产物放到 `/opt/models/matcha-icefall-zh-en/engines/`，至少包含 `matcha_s64_fp16.engine`、`matcha_s140_fp16.engine`、`vocos_600_fp16.engine`。路径支持环境变量覆盖，例如 `MATCHA_TRT_MODEL_DIR`、`MATCHA_TRT_ENGINE_DIR`、`MATCHA_VOCOS_ENGINE`。

文件：

- `app/backends/matcha_trt.py`：backend 实现、路径校验、engine 加载、`synthesize()`、`generate_streaming()`。
- `app/tts_backend.py`：在 factory 中新增 `backend_name == "matcha_trt"` 分支；当前 factory 已按 `TTS_BACKEND` 选择 sherpa 或 qwen3_trt（`app/tts_backend.py:98`, `app/tts_backend.py:115`, `app/tts_backend.py:117`, `app/tts_backend.py:120`）。
- `app/model_downloader.py`：扩展模型注册或校验逻辑，当前 zh_en 已包含 Matcha TTS 包（`app/model_downloader.py:24`, `app/model_downloader.py:25`, `app/model_downloader.py:26`），但 TRT engine 可以作为额外包或构建产物。
- `scripts/` 或 `trt/`：新增 ONNX export、polygraphy/trtexec build、engine 校验脚本；输出 manifest 记录 TensorRT 版本、输入 shape、precision、workspace。

步骤：

1. 导出 Matcha acoustic ONNX 与 Vocos ONNX，固定或 bucket 化 text length / mel frame shape；保留 CPU 文本前端，优先复用 RKNN Matcha 中的 lexicon/tokens 加载思路（`rk3576/app/backends/rknn_matcha_tts.py:92`, `rk3576/app/backends/rknn_matcha_tts.py:100`）。
   - **RandomNormalLike 外化 + probe-first ONNX surgery**（`rkvoice-stream/models/tts/matcha/fix_matcha_rknn.py:13-25` probe-first 5 步流程，`rkvoice-stream/models/tts/matcha/fix_matcha_rknn.py:64-107` `probe_original` 函数）：改图顺序：① probe 原模型在目标 shape 拿中间张量 → ② `onnxsim` 固化维度 → ③ 用 probed constant 替换 `RandomNormalLike`/`Range`/`Slice`。M1 manifest 已确认 Matcha 含 `RandomNormalLike`，此 surgery 路径为必选。
   - **Bucket 选择 + mel_frames 估算公式**（`rkvoice-stream/rkvoice_stream/backends/tts/matcha.py:8-10` bucket 表 `s64`/`s140`，`rkvoice-stream/rkvoice_stream/backends/tts/matcha.py:426-428`）：`est = int((11.9 * num_tokens + 51) * length_scale * 1.2 + 0.5)`，用于 output buffer 预分配 + bucket 选择逻辑（向上取整到最近 bucket）。
   - **int64 → int32 cast**（M1 实测）：Matcha ONNX 的 `x` / `x_length` 输入是 int64，TRT 对 int64 支持有限。Python backend `synthesize()` 入口必须把 token id 与长度 cast 成 int32（或在 ONNX surgery 阶段直接改 input dtype），避免 TRT engine build 失败或运行时 fallback。
2. 构建 TensorRT engines：先落地固定 bucket，后续再评估 dynamic shape。构建时必须对 ODE attention QK^T 与 LayerNorm/RMSNorm 层设置 FP32 precision，其他 encoder/vocoder 计算可优先 FP16；现有可借鉴脚本 `benchmark/build_cp_fp16_safe.py` 已启用 `OBEY_PRECISION_CONSTRAINTS` 并按 layer type 过滤可设置精度层（`benchmark/build_cp_fp16_safe.py:37`, `benchmark/build_cp_fp16_safe.py:46`），再对 attention/norm 层强制 FP32 输出（`benchmark/build_cp_fp16_safe.py:61`, `benchmark/build_cp_fp16_safe.py:75`, `benchmark/build_cp_fp16_safe.py:83`）。每个 engine 建立 golden test：同一输入的 mel/audio 长度、非静音比例、RTF、最大幅值范围必须通过；Matcha golden test 还必须包含 round-trip 对照：同一文本、TRT vs ORT baseline，mel L2 error < 5%，否则拒绝发布。
3. 实现 `MatchaTRTBackend.preload()`：校验 engine、lexicon、tokens、vocoder 文件；加载 native pipeline；设置 `_ready=True`。Qwen3 TRT backend 已有文件校验和 ready 状态模式（`app/backends/qwen3_trt.py:74`, `app/backends/qwen3_trt.py:81`, `app/backends/qwen3_trt.py:107`）。
4. 实现 `synthesize()`：返回 WAV bytes 和 `duration`、`inference_time`、`rtf`、`sample_rate` 元数据，与现有 Qwen3 TRT 返回结构对齐（`app/backends/qwen3_trt.py:128`, `app/backends/qwen3_trt.py:153`, `app/backends/qwen3_trt.py:161`）。
5. 实现 `generate_streaming()`：先支持按 vocoder chunk 切 PCM；API 层已经把首 4 字节采样率加到流前缀，并从 backend 迭代 chunks（`app/main.py:245`, `app/main.py:260`, `app/main.py:270`, `app/main.py:281`）。

## 3. Paraformer ASR TRT

架构：

- Python backend：新增 `app/backends/paraformer_trt.py`，类名 `ParaformerTRTBackend(ASRBackend)`；默认 16 kHz，能力至少包含 `OFFLINE` 与 `STREAMING`。
- Stream session：新增 `ParaformerTRTStream(ASRStream)`，每次 `accept_waveform()` 接收 float32 PCM，内部做 16 kHz resample、chunk 累积、TRT encoder/decoder 推理、endpoint 检测、partial 文本维护。当前 WebSocket 路径已经把 int16 bytes 转 float32，并在线程池中调用 `stream.accept_waveform()`（`app/main.py:447`, `app/main.py:553`, `app/main.py:555`, `app/main.py:558`）。
- Native/runtime 层：建议 `paraformer_speech_engine.Pipeline` 暴露 `create_stream()`、`accept_waveform()`、`get_result()`、`is_endpoint()`、`finalize()`，Python 只做协议适配。可参考现有 Qwen3 ASR 用 pybind11 TRT decoder、ORT encoder fallback 的加载方式（`app/backends/qwen3_asr.py:737`, `app/backends/qwen3_asr.py:740`, `app/backends/qwen3_asr.py:837`, `app/backends/qwen3_asr.py:841`）。
- 模型基础：当前 sherpa streaming backend 的 zh_en 模型目录是 `/opt/models/paraformer-streaming`，文件名是 `encoder.onnx`、`decoder.onnx`、`tokens.txt`（`app/backends/sherpa_asr.py:25`, `app/backends/sherpa_asr.py:27`, `app/backends/sherpa_asr.py:273`, `app/backends/sherpa_asr.py:274`, `app/backends/sherpa_asr.py:280`）。TRT 路径应从这些 ONNX 产物构建，不改变用户-facing API。

文件：

- `app/backends/paraformer_trt.py`：backend 和 stream session。
- `app/asr_backend.py`：在 factory 中新增 `backend_name == "paraformer_trt"` 分支；当前 factory 使用 `LANGUAGE_MODE` 或 `ASR_BACKEND` 自动选择（`app/asr_backend.py:90`, `app/asr_backend.py:99`, `app/asr_backend.py:104`, `app/asr_backend.py:106`, `app/asr_backend.py:109`）。
- `app/model_downloader.py`：zh_en 已声明 `paraformer-streaming` 模型包（`app/model_downloader.py:25`, `app/model_downloader.py:27`），新增 TRT engine 检查或下载条目。
- `tests/`：补充 websocket 协议测试、endpoint/reset 测试、短音频/空音频测试、golden transcript 误差测试。

步骤：

1. 盘点 Paraformer ONNX 输入输出和 sherpa 参数。当前 sherpa loader 用 `OnlineRecognizer.from_paraformer()`，启用了 endpoint detection，并设置了三条 trailing silence / utterance 规则（`app/backends/sherpa_asr.py:276`, `app/backends/sherpa_asr.py:277`, `app/backends/sherpa_asr.py:283`, `app/backends/sherpa_asr.py:284`, `app/backends/sherpa_asr.py:285`, `app/backends/sherpa_asr.py:286`）。
2. 构建 TRT engines：基于 M1 manifest 实测（`docs/plans/matcha-paraformer-trt-m1-manifest-2026-04-28.md`）：
   - **encoder 单独 dual-profile**：CIF predictor 的 `alphas`/`acoustic_embeds` token 维度是真动态，dual-profile `min=4, opt=20, max=40` tokens（M1 实测无硬上限，max 从原 80 收窄到 40 ≈ 32s 音频，避免 profile 浪费），超过切句
   - **decoder 单 profile**：M1 manifest 实测 decoder cache 是固定深度 window `16 × [512, 10]`（不是可变 KV cache），decoder 不需要 dual-profile，比 spec 原假设简化
   - 如 decoder shape 不适合 TRT，允许 encoder TRT + decoder CUDA/CPU fallback，backend 名称和 metrics 必须标注实际 provider
   - M1 已运行 `polygraphy inspect model` 拿到真实 input/output 名称（见 manifest），trtexec build 时直接引用 manifest 张量名，不要假定
   - trtexec skeleton（encoder dual-profile 示例）：

   ```bash
   trtexec --onnx=encoder.onnx \
     --minShapes=feats:1x4x560,feats_length:1 \
     --optShapes=feats:1x20x560,feats_length:1 \
     --maxShapes=feats:1x40x560,feats_length:1 \
     --saveEngine=paraformer_encoder.plan
   ```
3. 实现 `ParaformerTRTBackend.preload()`：校验 `tokens.txt` 和 engines，加载 tokenizer/symbol table，创建 native recognizer，执行 1 秒静音 warmup。服务启动已经预热 ASR backend，并会调用 `_asr_backend.preload()`（`app/main.py:104`, `app/main.py:106`, `app/main.py:108`）。
4. 实现 `ParaformerTRTStream.accept_waveform()` 与 `get_partial()`：对齐 sherpa stream 行为，`get_partial()` 返回 `(text, is_endpoint)`，endpoint 后清空上一段 partial（`app/backends/sherpa_asr.py:103`, `app/backends/sherpa_asr.py:113`, `app/backends/sherpa_asr.py:123`, `app/backends/sherpa_asr.py:130`, `app/backends/sherpa_asr.py:158`）。
5. 实现 `finalize()` 与 `force_endpoint()`：WebSocket 控制消息已有 `end_utterance` 分支直接调用 `stream.force_endpoint`，所以新 stream 必须实现该方法或在基类上补协议检测（`app/main.py:521`, `app/main.py:523`）。

## 4. 共享基础设施

- Backend factory：TTS 和 ASR 都通过环境变量选择 backend，新增 backend 只接入 factory，不改 API handler。TTS 当前分支在 `create_backend()` 中，ASR 当前分支在 `create_asr_backend()` 中（`app/tts_backend.py:98`, `app/asr_backend.py:90`）。
- 常驻 engine 与 warmup：TRT engine 必须在 `preload()` 加载并常驻，避免首请求编译或 I/O；Qwen3 TRT backend 已采用常驻 C++ pipeline（`app/backends/qwen3_trt.py:72`, `app/backends/qwen3_trt.py:87`, `app/backends/qwen3_trt.py:96`），服务启动也有 ASR/TTS warmup 路径（`app/main.py:112`, `app/main.py:145`）。
- 单线程 GPU 执行器：保持现有 TTS streaming 和 ASR streaming 单线程 executor，减少 CUDA per-thread context 抖动。当前代码明确为 TTS/ASR 各建一个单 worker executor（`app/main.py:47`, `app/main.py:53`, `app/main.py:55`, `app/main.py:60`, `app/main.py:67`, `app/main.py:76`）。
- 构建产物 manifest：所有 engine 目录包含 `manifest.json`，字段包括 source ONNX sha256、TensorRT version、JetPack/CUDA version、precision、input shapes、builder flags、benchmark summary。Python backend 启动时校验 manifest 的 compute capability 与当前设备。
- Benchmark：统一脚本输出 JSONL，字段包括 backend、provider、sample_id、duration_s、first_chunk_ms、inference_ms、rtf、gpu_mem_mb、text 或 transcript。Matcha 与 Paraformer 都要有 CPU/sherpa baseline 和 TRT 对比。
- 回退策略：默认仍保留 sherpa backend；`matcha_trt` 或 `paraformer_trt` 加载失败时 fail-fast，避免静默跑错 provider。开发阶段可加显式 env `*_ALLOW_FALLBACK=1`，但日志和 `/health` 必须报告真实 backend。

## 5. 风险

1. Shape 动态性风险：Matcha 文本长度、mel frames、Vocos frames 以及 Paraformer chunk/context shape 都可能导致 engine 数量膨胀。缓解：第一版只支持 2-3 个固定 bucket，超限文本切句；manifest 记录 bucket 上限，backend 选择最近 bucket。
2. 精度和音质回退：Matcha/Vocos FP16 可能带来爆音、静音或时长漂移，Matcha estimator 的 ODE 多步迭代尤其容易累积 FP16 误差。缓解：engine build 阶段必须对 ODE attention QK^T 与 LayerNorm/RMSNorm 层设置 FP32 precision，encoder 与 vocoder 允许 FP16；现有 `benchmark/build_cp_fp16_safe.py` 展示了按 layer type 安全设置 precision 的模式（`benchmark/build_cp_fp16_safe.py:46`, `benchmark/build_cp_fp16_safe.py:70`, `benchmark/build_cp_fp16_safe.py:83`）。golden audio 校验最大幅值、RMS、非静音比例、时长误差；同时加入 round-trip golden：同一文本、TRT vs ORT baseline，mel L2 error < 5%，超过则拒绝发布；保留 CPU/ORT 或 sherpa baseline 对照。
3. **FP16 mel 能量异常 + 局部平滑保险网**（`rkvoice-stream/rkvoice_stream/backends/tts/matcha.py:546-580`）：即使 BF16 + attention/norm FP32 也可能出现局部能量异常帧。建议加检测器：`window=5` 中值，阈值 `< 0.5` 或 `> 1.8` 触发邻帧混合（同 rkvoice 实现）。回归测试阶段作日志告警，量产前决定是否硬修正。
4. 端点检测不一致：Paraformer TRT 如果绕过 sherpa runtime，endpoint 行为可能和当前线上不同。缓解：复刻当前 sherpa endpoint 参数，并用 websocket reset/end_utterance 测试覆盖；当前参数可从 sherpa loader 直接对照（`app/backends/sherpa_asr.py:283`, `app/backends/sherpa_asr.py:284`, `app/backends/sherpa_asr.py:285`, `app/backends/sherpa_asr.py:286`）。
5. CUDA stream/thread 互斥问题：现有代码已经因为 TRT/CUDA Graph 和线程上下文加入单线程 executor；新增 TRT runtime 如果跨线程调用会重现冷上下文或 graph capture 错误。缓解：所有 streaming 调用继续走现有 executor，不在 backend 内自行开多 GPU worker。
6. 构建环境漂移：TensorRT engine 与 JetPack/CUDA/TensorRT 版本强绑定。缓解：engine 不跨版本复用；build 脚本在目标镜像内运行；manifest 启动校验不通过则报错。
7. 模型下载体积：新增 engine 包会增加 `/opt/models` 体积和首次启动耗时。缓解：TRT engine 作为 opt-in 包，仅当 `TTS_BACKEND=matcha_trt` 或 `ASR_BACKEND=paraformer_trt` 时下载；当前 downloader 已按 language mode 检查缺失模型（`app/model_downloader.py:91`, `app/model_downloader.py:97`, `app/model_downloader.py:118`）。

## 6. Milestones

1. M1 - 模型与构建规格冻结。验收：Matcha/Vocos/Paraformer ONNX 输入输出表、bucket 表、engine manifest schema、build 命令文档齐全；能在 Jetson 目标镜像内生成至少一个 Matcha bucket engine 和一个 Paraformer encoder engine。

   - **ONNX op 黑名单扫描**：M1 manifest 必须扫描 Matcha + Paraformer ONNX，报告所有命中节点。TRT 不友好 op 列表（`rkvoice-stream/models/tts/matcha/analyze_matcha_onnx.py:42`）：`Range`, `Slice`, `Scan`, `Loop`, `If`, `Where`, `GatherND`, `NonZero`。更新 M1 派发 prompt 模板以包含此扫描任务。

```bash
deepseek-fast <<'PROMPT'
Goal: freeze model/build specs only. Working dir: /home/recomputer/jetson-voice. Entry script: scripts/spec_freeze_matcha_paraformer_trt.sh, creating docs and manifests only. Keep the run to 25 bash steps or fewer; number each step in the raw log. Inspect Matcha, Vocos, and Paraformer ONNX with polygraphy/trtexec, record real input and output tensor names, bucket/profile ranges, TensorRT version, and model md5 values. Scan Matcha + Paraformer ONNX for TRT-unfriendly ops: Range, Slice, Scan, Loop, If, Where, GatherND, NonZero; record every hit node in the manifest. EVIDENCE must include md5 for ONNX and generated engine files, raw command log, before/after manifest diff, and blacklist scan output. Forbidden ops: no delete v3.4-slim engines, no docker-compose changes, no container down/up. Acceptance command: bash scripts/spec_freeze_matcha_paraformer_trt.sh --verify. Expected output: SPEC_FREEZE_OK with Matcha bucket, Vocos, and Paraformer encoder entries.
PROMPT
```

2. M2 - Matcha TRT 离线 TTS 可用。验收：`TTS_BACKEND=matcha_trt` 启动成功，`/tts` 返回可播放 WAV；`/tts/capabilities` 显示 `matcha_trt`、`basic_tts`、正确 sample rate；20 条 zh/en/zh-en 文本 RTF P50 <= 0.10。

   - **ODE step ablation (before engine compile)**：rkvoice-stream 实测 3-step FP16 ODE 导致 mel 能量崩溃，1-step Euler `dt=1.0` 又快又准（`rkvoice-stream/models/tts/matcha/split_matcha_rknn.py:5-9`, `rkvoice-stream/rkvoice_stream/backends/tts/matcha.py:47-51`, runtime: `rkvoice-stream/rkvoice_stream/backends/tts/matcha.py:381-384`）。M2 必须在编译 estimator engine 之前先做 ablation：`N=1` / `N=3` / `N=10` 三档，对比指标：mel L2 vs ORT baseline + 生成音频能量 RMS + TTFT。**决策规则**：mel L2 < 5% 下选最少步数，据此定 estimator 编译策略（fixed-step vs multi-profile）。更新 M2 派发 prompt 模板以包含此 ablation 任务。

```bash
deepseek-fast <<'PROMPT'
Goal: make offline Matcha TRT synthesize valid WAV through the existing /tts API. Working dir: /home/recomputer/jetson-voice. Entry script: scripts/run_matcha_trt_offline_acceptance.sh; do not call bare cmake or make. Use 25 bash steps or fewer and keep every command in the raw log. Before compiling the estimator engine, run ODE step ablation for N=1, N=3, and N=10, comparing mel L2 vs ORT baseline, generated audio RMS, and TTFT; choose the fewest steps with mel L2 < 5% and document fixed-step vs multi-profile strategy. Build or load one approved Matcha bucket and Vocos engine, start the service with TTS_BACKEND=matcha_trt, call /tts and /tts/capabilities, then run the 20-sample benchmark. EVIDENCE must include engine md5, WAV md5, raw service/benchmark logs, ODE ablation table, and before/after RTF summary versus sherpa/ORT. Forbidden ops: no delete v3.4-slim engines, no docker-compose changes, no container down/up. Acceptance command: bash scripts/run_matcha_trt_offline_acceptance.sh --verify. Expected output: MATCHA_OFFLINE_OK rtf_p50<=0.10 wav_valid=true.
PROMPT
```

3. M3 - Matcha TRT streaming 可用。验收：`/tts/stream` 首 4 字节采样率正确，后续 PCM chunks 连续可播放；首包延迟 P50 <= 200 ms；长文本切句或 bucket fallback 不崩溃。

```bash
deepseek-fast <<'PROMPT'
Goal: make Matcha TRT streaming produce continuous PCM chunks with TTFT at the new target. Working dir: /home/recomputer/jetson-voice. Entry script: scripts/run_matcha_trt_stream_acceptance.sh; use that wrapper for build, service launch, and benchmark actions. Limit the work to 25 bash steps, preserving a raw log with step numbers. Validate /tts/stream sample-rate prefix, chunk continuity, long-text splitting, bucket fallback, CUDA Graph cache hit behavior, and TTFT P50. EVIDENCE must include engine md5, first-chunk raw timing log, before/after TTFT table, and one output PCM md5. Forbidden ops: no delete v3.4-slim engines, no docker-compose changes, no container down/up. Acceptance command: bash scripts/run_matcha_trt_stream_acceptance.sh --verify. Expected output: MATCHA_STREAM_OK ttft_p50<=200ms chunks_valid=true.
PROMPT
```

4. M4 - Paraformer TRT streaming 可用。验收：`ASR_BACKEND=paraformer_trt` 启动成功，`/asr/stream` 能持续返回 partial 和 final；reset 与 end_utterance 控制消息工作；每 400 ms chunk 处理 P50 <= 80 ms。

```bash
deepseek-fast <<'PROMPT'
Goal: deliver Paraformer TRT streaming partial/final recognition over the existing WebSocket API. Working dir: /home/recomputer/jetson-voice. Entry script: scripts/run_paraformer_trt_stream_acceptance.sh; all build, service, and test commands must go through this wrapper or scripts it calls. Keep to 25 bash steps or fewer and save a raw command log. Verify real ONNX tensor names, CIF dynamic profile, engine md5, reset, end_utterance, partial updates, final text, and 400 ms chunk latency. EVIDENCE must include md5, raw websocket transcript log, before/after latency summary, and provider labels. Forbidden ops: no delete v3.4-slim engines, no docker-compose changes, no container down/up. Acceptance command: bash scripts/run_paraformer_trt_stream_acceptance.sh --verify. Expected output: PARAFORMER_STREAM_OK chunk_p50<=80ms partials=true finals=true.
PROMPT
```

5. M5 - Paraformer TRT offline/final path 可用。验收：`/asr` 对短音频返回文本；长音频 finalize 不丢尾音；固定 golden set 的字符错误率相对 sherpa baseline 不超过约定阈值。

```bash
deepseek-fast <<'PROMPT'
Goal: complete Paraformer TRT offline and finalization paths with CER parity against sherpa. Working dir: /home/recomputer/jetson-voice. Entry script: scripts/run_paraformer_trt_offline_acceptance.sh; no bare cmake or make commands are allowed. Use no more than 25 bash steps and keep a raw log. Run short-audio /asr, long-audio finalize, trailing-silence endpoint, and fixed golden-set CER comparisons. Label actual providers when decoder fallback is used. EVIDENCE must include engine md5, input audio md5, raw JSON responses, before/after CER table, and latency table. Forbidden ops: no delete v3.4-slim engines, no docker-compose changes, no container down/up. Acceptance command: bash scripts/run_paraformer_trt_offline_acceptance.sh --verify. Expected output: PARAFORMER_OFFLINE_OK cer_relative_degradation<2% final_tail_preserved=true.
PROMPT
```

6. M6 - 集成与回归。验收：新增 factory 分支、model downloader 条目、benchmark、tests 合并；默认 `LANGUAGE_MODE=zh_en` + 未设置 backend 仍走原 sherpa 路径；`matcha_trt` 与 `paraformer_trt` 加载失败时错误清晰且不伪装为其他 backend。

```bash
deepseek-fast <<'PROMPT'
Goal: finish integration and regression without changing the public API. Working dir: /home/recomputer/jetson-voice. Entry script: scripts/run_matcha_paraformer_trt_regression.sh; it must own build, unit, API, and benchmark checks. Stay within 25 bash steps and write a raw command log. Validate factory branches, downloader entries, health/capabilities, default sherpa behavior, explicit matcha_trt and paraformer_trt behavior, fail-fast errors, and benchmark JSONL schema. EVIDENCE must include md5 for touched engines or fixtures, raw test log, before/after backend matrix, and failure-mode logs. Forbidden ops: no delete v3.4-slim engines, no docker-compose changes, no container down/up. Acceptance command: bash scripts/run_matcha_paraformer_trt_regression.sh --verify. Expected output: TRT_INTEGRATION_OK default_backend=sherpa failfast=true tests_passed=true.
PROMPT
```

## 7. Performance Decisions

- CUDA Graph（primitive 1.2）：Matcha estimator 的 N=10 ODE steps 形状一致，必须使用 graph cache；否则每步 `enqueueV3` dispatch 约 12 ms，10 步会浪费约 120 ms。capture key 定义为 `(batch, mel_channels, T_mel)`，replay hook 放在每次 ODE step 调用处。Paraformer encoder 的 per-chunk shape 固定，每个 chunk size 使用一个 graph key。现有模式可复用 Talker graph cache 声明（`benchmark/cpp/tts_trt_engine.h:154`）和 CP decode capture/replay 逻辑（`benchmark/cpp/tts_trt_engine.cpp:1768`）。
- TRT CP Engine Pool（primitive 1.1）：Paraformer 并发按 N=2 pool 起步；Matcha estimator 单用户路径也不得低于 N=2，因为 TensorRT 10.3 Myelin 有 same-shape crash/state 污染风险。现有 pool 的目的和 per-slot mutex 已在 `TRTCPKVEnginePool` 注释中写明（`benchmark/cpp/tts_trt_engine.h:542`），默认 pool size 为 2 且支持 `CP_POOL_SIZE` 覆盖（`benchmark/cpp/tts_trt_engine.cpp:2294`, `benchmark/cpp/tts_trt_engine.cpp:2303`）。
- Prefill/Decode dual-profile（primitive 1.5）：Paraformer encoder 需要 offline-full 与 streaming-chunk 两类 profile，拆成 dual profile。Matcha estimator 每个 ODE step shape 相同，不需要 prefill/decode 拆分。现有 dual-profile context 结构可参考 `ctx_prefill_`/`ctx_decode_`（`benchmark/cpp/tts_trt_engine.h:186`）。
- GPU KV double buffer（primitive 1.3）：Matcha 与 Paraformer 都不是 autoregressive KV 解码路径，本规格不使用 GPU KV double buffer。不得为了复用框架引入无效 KV 状态。
- BF16/FP32 mixed precision（primitive 1.4）：Matcha estimator 的 ODE 多步误差会累积，attention QK^T 与 LayerNorm/RMSNorm 强制 FP32，encoder 与 vocoder FP16 OK。Paraformer encoder 使用 BF16，decoder 使用 FP16。现有 runtime 会探测 KV/logits dtype（`benchmark/cpp/tts_trt_engine.cpp:1317`），build 侧 layer-level precision 可参考 `benchmark/build_cp_fp16_safe.py` 的 attention/norm FP32 策略（`benchmark/build_cp_fp16_safe.py:61`, `benchmark/build_cp_fp16_safe.py:83`）。
- ORT Session Isolation（primitive 1.7）+ Warmup（primitive 1.10）：复用 `tts_trt_engine`/benchmark 现有隔离和 warmup 习惯。TTS ORTModels 使用自己的 `Ort::Env` 并创建 CUDA EP session options（`benchmark/cpp/tts_ort_models.h:74`, `benchmark/cpp/tts_ort_models.cpp:20`, `benchmark/cpp/tts_ort_models.cpp:38`）；ASR pipeline 也声明独立 ORT env（`benchmark/cpp/asr_pipeline.h:65`）。warmup 复用 CP pool warmup（`benchmark/cpp/tts_trt_engine.cpp:2352`）与 vocoder shape warmup（`benchmark/cpp/tts_trt_engine.cpp:2449`）的启动时预热模式。

## 8. Implementation Order

先做 Paraformer，再做 Matcha。Paraformer pipeline 更简单，能先验证 TRT build/test 工作流，并尽早解锁 V2V latency measurement baseline。

随后做 Matcha，因为它更难：ODE precision、FP16 mitigation、迭代 step 上的 CUDA Graph 都会放大构建和回归成本。等 Paraformer 的 engine manifest、benchmark JSONL、provider 标注、acceptance wrapper 都跑通后，Matcha 可以直接复用这些基础设施，降低首轮集成风险。
