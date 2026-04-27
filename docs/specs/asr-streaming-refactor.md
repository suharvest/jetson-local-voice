# Qwen3StreamingASRStream — 完整重构 Spec

**Source**: codex read-only 分析 2026-04-20。`app/backends/qwen3_asr.py` 中 `Qwen3StreamingASRStream` 类有 5 个叠加 bug，需系统性重构。

## ⚠️ Prerequisite — 文件已坏

`_process_chunk` 从 L467 开始顶层缩进（脱出了 `Qwen3StreamingASRStream` 类），而 `_apply_rollback` 在 L554-L581 仍按类内缩进。**import 会挂**。任何重构前先修缩进，让文件能 import。

## 1. 架构设计

`Qwen3StreamingASRStream` 作为单次会话的流式编码器 + 易变 partial decoder：

- **`accept_waveform`**：保留 resample + buffer + 切块行为（L255-L284），**移除 speculative encoding**（L269-L278）。
- **`_process_chunk`**：每个完整 chunk **编码一次**，append 到 `_segments`（L500）；`_segments` 成为整句 encoder 输出的**完整存储**，不再是可变滑窗（当前 L497-L499 的 popleft 移除）。
- **Partial 解码**：每 append 后在**当前所有 chunks 拼接**上调 `_decode_window`，token 预算按音频时长动态推（替代 L46 的 fixed `STREAMING_MAX_TOKENS=4`）。
- **`finalize`**：flush tail 一次，然后**对全部 `_segments` 做一次最终解码**。不再返回 `_archive_text + _prev_text`（当前 L305-L318）。**不累积 `_archive_text`**。
- **Partial 输出**：仅用于显示（rollback + LocalAgreement 作用在 `_stable_text`），**绝不进入最终**。
- **`_build_prompt`**：`ASR_TEXT` anchor **无条件**追加（当前 L895-L901 仅在 `language` 非 None 时追加，导致英文 auto 模式挂）。

### 保留 / 丢弃

**保留**：`_sample_buf`、`_left_context`、`_segments`（但取消 maxlen）、`_prev_text`、`_stable_text`、`_eos_count`、timing counters
**丢弃**：`_spec_embd`、`_spec_audio_len`、`_spec_left_context`、`_archive_text` 累积路径

## 2. File:line 级 diff

**前置**：修 L467-L581 缩进（整块回到 `Qwen3StreamingASRStream` 类内部）。

### `__init__` (L226-L253)
- 保留 L226-L248（buffers + counters）
- 删 L250-L253（`_spec_embd`、`_spec_audio_len`、`_spec_left_context`）
- 更新 L232-L235 注释：`_segments` 保留整 session 所有 chunks

### `accept_waveform` (L255-L284)
- 保留 dtype / resample / concat / 切块 逻辑
- **删 L269-L278 的 speculative pre-encode**

### `prepare_finalize` (L289-L303)
- 保留公共方法（`app/main.py:L451-L454` 会调）
- 改为 no-op 或 cheap validation hook（`app/asr_backend.py:L51-L56` 标记此优化为可选）

### `_process_chunk` (L467-L534)
- 删 L473-L486 的 speculative reuse
- 总是用当前 left context 编码 `audio_chunk`，然后更新 `_left_context`（L490-L495 保留）
- **删 L497-L499 的 maxlen eviction**；L500 的 append 保留整 session 所有 chunks
- Partial 解码：拼接所有 segment embeddings；若超 decoder seq limit（L389-L404）用**不可变 trailing-window view**（不动 `_segments`）— ⚠️ ASSUMPTION，Jetson 实测延迟
- **L519-L534 endpoint 重置逻辑改**：`raw_text is None` 可 `_eos_count++`，但**不得** clear `_segments` / `_left_context` / `_prev_text`
- **L536-L544 输出处理改**：final 仅设 `_stable_text` 和 `_prev_text` = 全音频 decode；partial 用 `_local_agreement` 稳显示但**不 append `_archive_text`**
- `_apply_rollback` (L554-L562) display-only，超短假设跳过

### `_build_prompt` (L895-L901)
- `ASR_TEXT` **无条件**追加。若 `language` 提供则先注入 language token，再 ASR_TEXT；`None`/`auto` 也追 ASR_TEXT。

### `finalize` (L305-L318)
- 用**一次 fresh decode on all `_segments`** 替代 `_archive_text + _prev_text`
- 公共签名不变

### 解码循环
- `_decode_window:L388-L455` KV mapping（L423-L454）保持
- 离线路径 L798-L862 也喂修正后的 prompt

## 3. Bug-kill 矩阵

| Bug | 根因 | 由哪部分消除 |
|---|---|---|
| **A** 首字丢失 | finalize 返回 `_archive_text + _prev_text`(L305-L318)；partial 提前进 archive；prompt 缺 `ASR_TEXT` | finalize 一次全音频 decode + 无 archive + `ASR_TEXT` 无条件 |
| **B** stale spec embedding | L269-L278 create，L484-L486 else 没 clear，L477-L483 reuse 只比 length | **整套去 speculative encoding** |
| **C** 重复幻觉 | L311 + L537-L543 `_archive_text` 累进输出 | 无 archive + final 单次全音频 decode |
| **D** 英文挂 | L895-L901 `ASR_TEXT` 仅在 `if language` 内追加；streaming 传 `None`(L362-L365) | `ASR_TEXT` **无条件**追加 |
| **E** 早期 EOS | 两次 `None` 触发 L519-L534 window clear | EOS 只标 endpoint，不清 `_segments`/音频上下文 |

## 4. 风险

1. **Partial 延迟上升**：保留整段 `_segments` → 长句 partial 解码变慢。需 Jetson 实测 1.2s / 2.4s / 4.8s / 8s fixtures per-chunk wall time。
2. **Decoder seq overflow**：长音频 final prompt 可能超 TRT seq limit（L389-L404）。需测强制 ORT fallback。
3. **Partial 显示不稳**：去掉 archive 累积 → partials 纯 hypothesis。需测 monotonic display。
4. **Endpoint 语义变**：`is_final` from `app/main.py:L466-L473` 的客户端解读可能坏。测两短句 + 静默。
5. **Explicit language prompt 回归**：无条件 `ASR_TEXT` 不得重复。测 `zh` / `en` / `auto`。

## 5. 工期 — 分两 PR

### PR 1 (~0.5 day)
- 修缩进让文件 importable
- 去 speculative encoding
- 修 `_build_prompt`（`ASR_TEXT` 无条件）
- 加 state-machine 单测
- **低风险，直接杀 Bug B + D**

### PR 2 (~1–1.5 day)
- 重构 `_process_chunk` + `finalize`：`_segments` 全保留 + endpoint 不清 state + partial display-only + final 单次全音频 decode
- 动核心流式行为（L467-L581）
- 需设备侧验证

**总：1.5-2 工日**（+ Jetson 实测时间）

## 6. 验证方案

**不要**用 TTS→ASR round-trip（TTS 输出噪声）。用人录 WAV fixtures 或**直接向 `_run_encoder` / `_decode_window` mock 确定性输出**。

### 单测（patch _run_encoder + _decode_window）

- **Bug A**: 喂两 chunk "今天天气不错我们出去玩吧"，`finalize()` 必须返回**完整**字串
- **Bug B**: `accept_waveform` 半 chunk → 完整 chunk → 尾部长度碰巧等于 stale 半 chunk；断言 `_run_encoder` 被调真尾部，无 stale 重用
- **Bug C**: script 4 chunk 让 `_decode_window` 返 "今天"/"今天天气不错"/"今天天气不错我们出去"/"今天天气不错我们出去玩吧"；final 必须**恰一份**，无重复
- **Bug D**: `_build_prompt(audio_len, None)` 和 `(audio_len, "en")` 都以 `ASR_TEXT` 结尾；人录 "Hello world how are you today" WAV → `finalize()` 返完整
- **Bug E**: script 两次 `None` partial + 一次 valid final；断言 `_segments` 保留 + finalize 返 valid text
