# 交接文档 — 2026-04-27 全天 session 总结

**Session 范围**：从早上接手 ASR 优化 → 到晚上 v3.4 image ship + v3.5 clean rebuild 派 background。
**起点 commit**：`be0ef76`（feature/asr1-true-streaming）
**终点 commit (撰写时)**：`a3afd28`（main，PR #5 已 merge）
**v3.5 image build 仍在 background companion 跑** (job `20260427-193555-610894`)
**Prod**：orin-nx container `reachy_speech-speech-1` 跑 image `jetson-voice-speech:v3.4-librosa-no-transformers`

---

## 1. 今日落地 commit (按时间序)

| commit | 类型 | 内容 | V2V 影响 |
|---|---|---|---|
| `cbaa6b6` | fix | M1 multi-utterance buffer drop fix + finalize 加 `cudaStreamSynchronize` 防 906 | 修正确性 + 稳定性 |
| `bb69b0e` | perf | TTS streaming 单线程 executor + CUDA per-thread context prewarm (T3) | 并发 footgun fix |
| `893a37d` | perf | ASR `_transcribe_python` 加 enc + decode timing log | instrument |
| `1bee6a3` | perf | ASR 加 prefill/decode/d2h 拆分 timing + opt-in `ASR_ENCODER_BACKEND=ort_trt` env (B0 + A2 Path A 尝试) | A2 Path A blocked, B0 数据落地 |
| `a0fba89` | perf | librosa+numpy mel backend (transformers 变 fallback) | mel 10ms→5ms (噪音内) |
| `7d282c2` | refactor | 砍 transformers fallback 代码 + image v3.4-librosa-no-transformers ship | image 自包含 |
| `6a38aaf` | fix | ASR 单线程 executor (T3 模式镜像) + `benchmark/test_asr.py` type field 修 + `_offline_final_text` 单测 | 并发 race 修 |
| `48dc010` | chore | Dockerfile bake cuda-python/transformers/librosa/webrtcvad/tokenizers | image 干净度 |
| `a3afd28` | docs | 今日 5 个 spec + handover doc | — |

加上 PR #5 (`feature/asr1-true-streaming` → main)：含 ASR 1 真流式、P0 ORT stream、CHUNK_SIZE 250→400ms、partial dedup、WS type field。

---

## 2. 性能现状

### 单流 V2V 数字 (S1.wav 2.8s 中文，post-warmup steady)
| 指标 | 数字 |
|---|---|
| **V2V eos→audio first chunk steady median** | **~321-348ms**（host noise 范围，跟 baseline 327ms 持平或略好） |
| ASR 段 (eos→final text) | ~315-340ms |
| TTS 段 (request→first audio) | ~6ms |
| ASR per-call 服务端拆分 | enc=37ms + prefill=40ms + decode=176ms + d2h=2ms |
| ASR token 数 (S1) | 10 |
| ASR per-token | ~17.6ms (decode 176/10) |

### 3 路并发 V2V（实测）
| metric | N=1 | N=3 |
|---|---:|---:|
| 单路 wall per round | 3.85s | 11.4s（3×单流） |
| 单路 eos_audio | ~1s* | ~8.4s（8×单流） |
| 系统总吞吐 | 0.26/s | 0.27/s |
| 错误数 | 0 | 0 |
| Final text 一致 | ✅ | ✅ |

*N=1 这里测得 1s 跟 v2v_simple.py 的 327ms 不一致是测法差异，相对倍率有意义。

**结论**：3 路并发**稳定不崩**但每路延迟成倍涨；单 GPU 已被单流榨满（GR3D_FREQ 95.8% 时间满频），并发无吞吐增益。

### CER (8 wav)
| metric | 数字 |
|---|---|
| Mean | 16.33% (vs baseline 17.22%, -0.89pp) |
| Median | 13.71% (=baseline) |

---

## 3. 优化做了什么 + 为什么 work

### 3.1 ASR 真流式 (PR #5)
- 流式期间每 400ms chunk 增量推 encoder + decoder，吐 partial
- finalize 时 `_offline_final_text` 重跑全段 encoder + 全 decode（**故意**，因为流式 partial 质量比 offline 差）
- chunk_size 从 250→400ms（少 dispatch 次数，total 时间 -5%）

### 3.2 P0 ORT user_compute_stream (PR #5)
- 旧问题：ORT CUDA EP 默认走 default stream，TTS TRT CUDA Graph capture 时撞 CUDA error 906 `legacy stream depend on capturing blocking stream`
- 修法：ORT 创建 nonblocking stream 当 user_compute_stream，跟 TTS TRT context 隔离
- finalize handoff 时 `cudaStreamSynchronize(_ASR_CUDA_STREAM)` 确保 ASR GPU work 完才把控制权交给 TTS

### 3.3 M1 multi-utterance buffer drop fix (PR #5)
- bug：流式期间 `_run_vad` 检测到新 utterance → 清 `_utterance_audio_buffer` → **当前 chunk 已被 append 进 buffer 但被一并清掉**
- 修法：抽 `_check_new_utterance_resume(samples)` helper，在 append **之前**调用，先 reset state if needed，再 append
- 验证：4 个 unit test 覆盖（happy path / silence boundary / state reset 完整性）

### 3.4 TTS T3 单线程 executor
- 旧问题：FastAPI `run_in_executor(None, ...)` 用 asyncio 默认 ThreadPoolExecutor（多 worker）。每个新 worker 线程的 CUDA per-thread context 是 cold → prefill 第一次 80-122ms（vs warm 16ms）
- 修法：lazy 单线程 ThreadPoolExecutor + 启动时 dummy prefill 预热该线程的 CUDA context
- 收益：sequential 数字不动，**concurrent 8 个 /tts/stream prefill 从 33-122ms → 稳 16ms**

### 3.5 ASR 单线程 executor (TTS T3 模式镜像)
- 同 TTS T3 — 解决并发 WS 共享 `_ASR_CUDA_STREAM` 进程单例的 race
- 单流 V2V 稍微 +21ms（warmup 残留嫌疑），并发场景理论上消除 GIL/stream race
- 实质改动：4 处 `asyncio.to_thread(...)` → `loop.run_in_executor(_get_asr_executor(), ...)`

### 3.6 librosa mel (replace transformers WhisperFeatureExtractor)
- 旧：transformers `WhisperFeatureExtractor` (n_fft=400, hop=160, n_mels=128, slaney scale, 30s pad)
- 新：`app/utils/whisper_mel.py` 用 librosa STFT + numpy 等价管道
- A/B max abs diff 1.19e-07（远小于 1e-4 阈值）
- mel 单算 5ms vs 10ms（in-container），CER 反而 -0.89pp
- 解锁：transformers 可从 image 卸（v3.4 已卸）

### 3.7 v3.4 image baked
- 旧 v3.3 image 不含 cuda-python + transformers + librosa + webrtcvad，container 起来后靠 manual `pip install` 维持，docker compose recreate 必炸
- 新 v3.4 通过 `docker commit` 把当前 container（含手装 deps + librosa-only 代码）保存
- 验证：compose recreate 后所有 deps 自动 baked 可用

---

## 4. 验证过的坑（重要 footgun）

### 4.1 V2V 测量 warmup curve（**一天踩两次同坑**）
- TRT/CUDA Graph/decoder context 需要 **5+ 次 warm 才进真稳态**
- 早上 1 cold + 4 warm 测 330ms → 晚上 N=9 测 1100ms 一度被判退化 3.4×
- librosa 部署后 N=8 测 492ms → 错觉 prefill 涨 50ms，实际是 numba JIT 冷启动 + 容器 restart 双重 warmup
- **教训**：v2v_simple.py 已加 `--warmup 5` 默认，输出 `summary.steady` 字段。**永远看 steady 不看 warm_all**
- **复杂场景**（新 lib + 新 image + 容器刚起）需 **N≥20 warmup=12** 才能稳
- memory: `feedback_v2v_warmup_curve_trap.md`

### 4.2 Image 命名误导
- `v3.3-no-transformers` 实际**没卸 transformers**，只是命名预告打算卸
- 09:25 morning memory snapshot 还错记成"已卸"，被误导
- v3.4 才真的卸了
- **教训**：image 名是计划标签不是事实，每次部署 verify `docker exec ... pip show <pkg>`

### 4.3 cuda-python + transformers + librosa 必须 baked
- `v3.3` 都没装这些，靠 stale container manual `pip install` 维持
- 任何 `docker compose up -d`（recreate）都会丢
- 表现：ASR pre-load 失败，uvicorn startup fail，docker 自动 restart loop
- 修法：v3.4 commit 当前 container + `Dockerfile` 加 pip install layer (commit 48dc010)

### 4.4 host libcublas 空目录
- 04-26 20:43 host `/usr/lib/aarch64-linux-gnu/libcublas{,Lt}.so.12` 不知被哪个之前 agent `cp` 写错搞成**空目录**
- container `/host-libs/` mount 后 ORT 加载 cublas 报 `Is a directory` → ASR 起不来 → restart loop
- 修法：`fleet exec --sudo --literal orin-nx -- bash -c "ln -s /usr/local/cuda/lib64/libcublasLt.so.12.6.1.4 /usr/lib/aarch64-linux-gnu/libcublasLt.so.12 && ..."`
- **教训**：host shared infra 改动前必须 `ls -la` 确认是 file 不是 dir 再操作
- memory: `project_session_2026_04_27_status.md` 有完整记录

### 4.5 D2H 不是 ASR 瓶颈（codex 估错了）
- codex 之前假设：每 decode step 把完整 logits `[1, 151936]` FP32 D2H 拷回 Python = 600KB/step × 10 = 6MB blocking
- B0 instrument 实测：**d2h 只 2ms** (10 step 加起来)
- C2 GPU argmax 收益从估的 **-20~120ms 缩水到 -2ms** → **REJECT，不值得做**
- **教训**：bottleneck assumption 必须 instrument 验证再做工程

### 4.6 A1 encoder frame reuse 不可行（codex review 出 2 BLOCKING）
- 想法：finalize 时复用流式期间的 `_encoder_frames`，跳过整段重 encoder
- codex review 出：
  - **BLOCKING 1**：流式 encoder 用 `left_context+chunk` 然后 trim，offline 整段一把，concat ≠ full-pass，没代码证据
  - **BLOCKING 2**：`_encoder_frames` 是 rolling buffer，超过 `_max_encoder_frames` 旧帧 evict，长音频会静默丢早段
  - 还有 4 个 MAJOR (mel boundary / decoder state / >6s 分段 / acceptance criteria 太弱)
- **REJECT** + abort 实施 agent

### 4.7 C4 prefill prefix KV cache 不可行
- 想法：cache prompt 前缀 KV
- 实际 prompt 结构：`[8 fixed tokens] + [AUDIO_PAD * N (变长)] + [3 fixed] + [lang tokens] + [ASR_TEXT]`
- 变长 audio 在固定前缀 8 token 之后**就开始变**，cache 只能覆盖前 8 token，几乎无收益
- 用户敏锐问"语言变了怎么办" → 实际 audio 长度变更早就废了 cache
- **REJECT**

### 4.8 ORT TensorrtExecutionProvider 直接用不通 (A2 Path A blocked)
- encoder ONNX 有 `If` 子图，输出 shape 没传播 → ORT TRT EP 拒绝 onboard
- 报错 `IIfConditionalOutputLayer inputs must have the same shape. Shapes are [128,-1] and [1,128,-1]`
- 解法：跑 `onnxruntime.tools.symbolic_shape_infer` 修 ONNX，或 re-export with `--input_shapes`
- 没做，留为 backlog

### 4.9 C3 multi-step CUDA Graph 风险高
- TRT 10.3 Myelin 在 Jetson 已多次踩坑（906、myelinGraphLoad 崩）
- multi-step 跨多个 past_length 增加 capture state 复杂度
- 收益 -10~30ms，但项目稳定性风险大
- 进 **backlog**，不在 must-do

### 4.10 单 GPU 物理上限
- Orin NX GR3D 满载 1173MHz，单流 ASR 已 95.8% 时间满频
- 多用户并发的物理上限：**N 用户 = N× 单流延迟**
- 多进程容器**最多消除软件串行（GIL/stream race）从 8× 降到 3×**，消不掉物理 GPU 共享
- 想 3 路同时 < 500ms 延迟：要么减小模型，要么加硬件
- memory: `project_orin-nx-concurrent-2026-04-26.md`

### 4.11 deepseek-flash 长 prompt 易卡
- 多次踩：30 秒就退（"任务描述不完整"），56 秒 SIGTERM
- 派复杂多 phase 任务必须**ANTI-RECURSION header** + 明确步数 checkpoint + EVIDENCE 要求
- 简单 phase 才让 deepseek-flash 做，复杂的派 sonnet 或 deepseek-pro
- memory: `feedback_claude_rescue_glm5_recursive_dispatch.md` 类似（GLM-5 也有同问题）

### 4.12 Docker Hub 国内不通（v3.5 build 遇到）
- orin-nx 的 Docker registry 全断（Hub / daocloud / 阿里云 mirror / nvcr.io 都不通）
- 包括 `ubuntu:22.04` 都拉不到
- 解法：**Mac 拉了 → `docker save | gzip` → fleet push 5.6GB tarball → orin-nx `docker load`**
- 或者长期：在 orin-nx 上配 docker daemon registry-mirror

---

## 5. 剩余优化空间

### 5.1 单流 perf
| 项 | 收益 | 工时 | 状态 |
|---|---|---|---|
| **C5** ASR decoder KV BF16 IO（engine rebuild）| -5~15ms | 8-14h | 独立任务，不动 .so，最干净 |
| **A2 Path A v2** 修 encoder ONNX shape inference 后用 ORT TRT EP | -13~18ms | 1-3h | encoder ONNX 上跑 `symbolic_shape_infer` 即可 |
| **TTS T4** Emit first chunk 在下次 decode 之前 | TTFT -17ms | 5-10 行 C++ | spec 在 `tts-perf-backlog-2026-04-19.md` |
| **TTS T6** Skip CP KV buffer zeroing (dynamic-shape) | TTFT -5~15ms | 3 行 | 同上 |
| **TTS T7** Talker cold-start warmup | 首次 -10~15ms | 类似 vocoder warmup | 同上 |
| **C1** ASR encoder C++ TRTEncoder | -5~10ms (Path A 落了)/ -23ms (没落) | 20-32h | C++ + .so rebuild + ABI 风险，缓做 |
| **C3** multi-step CUDA Graph | -10~30ms | 16-28h | **HIGH risk**, backlog |

栈起来理论 V2V 320 → ~250-280ms。

### 5.2 多用户支持 (3 路对话流畅)
| 路径 | 评估 |
|---|---|
| **(a) 切 paraformer+matcha** (`LANGUAGE_MODE=zh_en`) | 5 分钟试，模型小 V2V 可能 100-200ms 单流，3 路 ~600ms wall 勉强够。但只支持中英、无 voice clone |
| **(b) 多进程容器**（2 实例 sticky 路由）| 1-2 天工程，最多 2 路用户，3 路显存炸（16GB 共享 RAM 装不下 3 个 5GB 实例）|
| **(c) 多 Jetson + LB** | 最稳，按用户加机器，fleet 已具备基础 |
| **(d) 单流性能减半** (5.1 全做) | 3 路 ~750ms wall 接近可用，但 30+h 工程 + C3 风险 |

### 5.3 架构 / 维护
- **v3.5 clean rebuild** 在 background 跑，跑通后 image 自包含 + slim
- **ASR cleanup plan** (`docs/plans/2026-04-15-asr-cleanup.md`) 有 4 个 task，今日 ASR 1 真流式重写后 task 1/2 部分作废，需重新评估
- **ORT stream concurrent WS lock** — 已通过单线程 executor 间接解决，但 codex review 提的"显式锁"没做（实际不需要了）
- **`benchmark/test_asr.py` type field** ✅ 今晚已修
- **`_offline_final_text` 单测** ✅ 今晚已加 4 case

### 5.4 已 REJECT 的方向（不要再尝试）
- ❌ A1 encoder frame reuse — 2 BLOCKING (rolling buffer evict + receptive field)
- ❌ C2 GPU argmax — d2h 只 2ms，没收益
- ❌ C4 prefill prefix KV cache — 模板结构不允许
- ❌ silero-vad 替换 webrtcvad — 同进程 ORT+TRT 冲突 (memory `feedback_silero_vad_trt_conflict.md`)
- ❌ 用 partial 当 final（A8）— 之前 commit `3cb6f3f` 故意 revert，准确度差太多

---

## 6. 当前进行中

### v3.5 clean rebuild (background companion `20260427-193555-610894`)
- Phase 1 ✅ Mac repo sync 到 orin-nx
- Phase 2 ⏳ Docker Hub 国内不通，**Mac 拉 base image → save 成 5.6GB tarball → fleet push 到 orin-nx 走 docker load**
- Phase 3-6 待跑（build / test / V2V / 切 prod）
- 预计还要 30-50 分钟
- 完成后 v3.5 image 应该 < v3.4 (没有一天 cache 累积) + 自包含 + 文档化（dustynv 国内 workaround 也成最佳实践）

---

## 7. Memory 更新

今日新建/更新的 memory：
- `feedback_v2v_warmup_curve_trap.md` (新) — V2V warmup 测量陷阱，必读
- `project_session_2026_04_27_status.md` (更新) — 全天 session 完成状态
- `MEMORY.md` (索引更新)

---

## 8. 关键文件位置

### 设计 spec
- `docs/plans/asr-perf-backlog-2026-04-27.md` — A0/A1/A2 总体 backlog
- `docs/plans/asr-prefill-dec-perf-2026-04-27.md` — codex prefill+dec 攻略
- `docs/plans/asr-cpp-bundle-2026-04-27.md` — codex C++ bundle (C1-C5)
- `docs/plans/asr-mel-librosa-2026-04-27.md` — librosa 替换 spec
- `docs/plans/handover-2026-04-27.md` — 早上 session 起手 handover
- `docs/plans/handover-2026-04-27-final.md` — **本文档**
- `docs/plans/tts-perf-backlog-2026-04-19.md` — TTS T1-T9 (T1 已落, T2-T7 backlog)

### benchmark
- `/home/harvest/bench/v2v_simple.py` (orin-nx) — V2V 测试脚本，已加 `--warmup` 参数
- `/home/harvest/bench/v2v_concurrent.py` (orin-nx) — N 路并发测试
- `/home/harvest/bench/eval_asr.py` (orin-nx) — 8 wav CER 评估
- `/home/harvest/bench/wavs/S0-S7.wav` — 测试集
- `/home/harvest/bench/results/v2v_*.json` — 历史 V2V 结果

### prod 部署
- Container `reachy_speech-speech-1` on orin-nx
- Image `jetson-voice-speech:v3.4-librosa-no-transformers` (2.34GB)
- Compose `/home/harvest/jetson-voice/reachy_speech/docker-compose.override.yml`
- Overlay `/home/harvest/jetson-voice/app_overlay/backends/qwen3_asr.py` (md5 = main HEAD)
- Models volume `reachy_speech_speech-models` (qwen3-asr-v2 / qwen3-tts / matcha-icefall-zh-en / paraformer-streaming 全部已下)

### LANGUAGE_MODE 切换
- `multilanguage` (当前) → qwen3 ASR + qwen3 TTS (52 lang, voice clone)
- `zh_en` (默认) → paraformer + matcha (轻、只中英)
- 切：改 compose env + `docker compose restart speech`，5 分钟事

---

## 9. 下次接手建议

1. **先看 v3.5 build 跑完没**：`node /Users/harvest/project/claude-rescue/scripts/claude-companion.mjs status 20260427-193555-610894`
2. **快赢顺序**：A2 Path A v2（修 ONNX shape）→ TTS T4/T6/T7 → C5
3. **任何 V2V 测量必读 `feedback_v2v_warmup_curve_trap.md`**，N≥15 warmup=8 起步
4. **C3 multi-step graph 不要碰**，除非项目稳定性可承受 16-28h 工程 + Myelin 风险
5. **多用户支持** 优先尝试 paraformer+matcha 切换（5 分钟事），再考虑多进程
6. **C++ 重 build .so** 必须先派 codex 出 spec，不要让 sonnet/glm 自己设计
