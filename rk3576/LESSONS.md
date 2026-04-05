# RK3576 Qwen3-TTS 适配踩坑记录

> 2026-04-04 ~ 2026-04-05

## 1. RKLLM 转换踩坑

### 1.1 非标准 attention 不是问题（误判）

初始判断：talker 的 head_dim=128 而 hidden_size/num_heads=64，认为 RKLLM 不兼容。

**实际**：标准 Qwen3-0.6B 本身就是 head_dim=128，RKLLM 1.2.3 原生支持 Qwen3。这不是问题。

### 1.2 vocab_size 才是真正的阻塞

RKLLM `build()` 报 `index out of range in self`。

**根因**：talker vocab_size=3072（codec tokens），RKLLM 内部验证阶段用 tokenizer 编码测试文本，token ID 超出 3072 范围。最小可用 vocab_size=4096。

**修复**：把 embed_tokens 和 lm_head 从 [3072, 1024] zero-pad 到 [4096, 1024] 或 [151936, 1024]。

### 1.3 lm_head 用错了权重

原始 talker 有 `codec_head`（独立权重）而不是 `lm_head`。第一版错误地用了 `codec_embedding` 作为 lm_head。

**差异**：codec_head 和 codec_embedding max_diff=0.256，不是 tied weights。

**修复**：用 `talker.codec_head.weight` 作为 lm_head。

### 1.4 vocab_size 越大 decode 越快（反直觉）

| vocab | 文件 | decode |
|-------|------|--------|
| 4096 | 243 MB | 75.8 ms/步 |
| 8192 | 256 MB | 76.8 ms/步 |
| 16384 | 280 MB | 92.2 ms/步 |
| 32768 | 328 MB | 80.7 ms/步 |
| 65536 | 425 MB | 87.1 ms/步 |
| 151936 | 683 MB | **58.5 ms/步** |

**原因推测**：RKLLM 内部 NPU tile 对齐优化，151936 恰好对齐。

### 1.5 RKLLM keep_history 语义

- `keep_history=0`：保持 KV-cache（和直觉相反！）
- `keep_history=1`：清除 KV-cache
- 实际部署时 agent 测出的行为和文档不一致，需要实测验证

### 1.6 RKLLM 每次调用开销 ~143ms

`rkllm_run()` 每次调用有 ~143ms 固定开销（NPU 调度），实际 NPU 计算只要 ~6ms/token。这使得逐步调用模式（step-by-step）对短序列非常不利。

**影响**：code_predictor 15 步 × 143ms = 2.1s，比 RKNN 的 300ms 还慢 7 倍。

## 2. RKNN 转换踩坑

### 2.1 动态 shape 必须固定

RKNN 不支持任何动态维度。`rknn.load_onnx()` 会报错。需要在 ONNX 导出时去掉所有 `dynamic_axes`，或用 `input_size_list` 指定。

### 2.2 trace shape 必须匹配 RKNN shape

ONNX 导出时 `torch.onnx.export` 的 trace 会把 Reshape 节点的目标 shape bake 成常量。RKNN 的 `dynamic_input` 固定 shape 必须和 trace 时的 shape 一致，否则 Reshape 失败。

**教训**：想要 RKNN 固定 shape=32，就必须用 dummy_input shape=32 来 trace ONNX。

### 2.3 NPU 不支持的算子导致性能灾难

**Sin（SnakeBeta 激活）**：vocoder 29 个 Sin 算子全部 fallback CPU，导致 10.5s 延迟（目标 70ms）。
- 修复：7 阶 Taylor 多项式 + Clip(-π,π) 替换 Sin
- 效果：10.5s → 1.9s（5.4x 提速）

**Gather（embedding lookup）**：NPU 不支持，但单个 Gather 很快（<1ms），真正的开销是 NPU↔CPU 切换。
- 修复方向：用 one_hot @ embedding_table (MatMul) 替代

**Floor**：NPU 不支持。用做 range reduction 时遇到。
- 修复：改用 Clip 代替 Floor-based range reduction。

### 2.4 speaker_encoder 的 If 节点

RKNN 编译器遇到 If 节点会做 constant folding，把输出判定为常量导致失败。
- 修复：先用 `onnxsim.simplify()` 消除 If 节点。

### 2.5 RK3576 NPU 内存限制 ~180MB

超过 ~180MB 的 RKNN 模型会 NPU job timeout 或 fail to submit。talker_prefill 1.3GB 能加载但不能用双核。

### 2.6 RKNN toolkit 和 runtime 版本不匹配

模型用 toolkit 2.3.2 编译，设备 runtime 2.1.0。不影响加载但有警告，性能基本无差异。升级 runtime 到 2.3.2 后警告消除但速度不变。

## 3. ONNX 导出踩坑

### 3.1 export-onnx.py 的 bug（已修）

| Bug | 原因 | 修复 |
|-----|------|------|
| `talker.model.text_embed_tokens` | qwen-tts 包改名为 `text_embedding` | 改属性名 |
| `talker.model.embed_tokens` | 改名为 `codec_embedding` | 改属性名 |
| `talker.lm_head` | 改名为 `codec_head` | 改属性名 |
| `head_dim = D // num_attention_heads` | 实际 head_dim=128 不等于 1024/16=64 | 用 `config.head_dim` |
| `past_key_values` tuple 传入 | transformers 4.57+ 需要 `DynamicCache` | 用 `DynamicCache.from_legacy_cache()` |
| SDPA tracing 失败 | tokenizer decoder 用了 SDPA attention | 强制 `attn_implementation="eager"` |
| torch 2.11 TorchScript 回归 | `unordered_map::at` 错误 | 降到 torch 2.4 |
| `aten::__ior_` 不支持 | transformers masking_utils | monkey-patch `_is_torch_greater_or_equal_than_2_5 = True` |

### 3.2 torch 版本敏感

- torch 2.11：TorchScript ONNX exporter 有 regression，导出 talker 失败
- torch 2.4：可以导出，但只有 CPU 版 wheel（需要手动指定 cu121 index）
- torch 2.4+cu121：ONNX 导出 + CUDA 推理都 OK

## 4. FP16 精度踩坑

### 4.1 attention QK^T 溢出

Qwen3-TTS 的 head_dim=128，QK^T 中间值可能超过 FP16 max (65504)，导致 softmax 出 NaN。

- **bfloat16**：没问题（动态范围更大）
- **float16**：RTX 3060 上确认 NaN
- **RKNN FP16**：RK3576 NPU 不支持 bfloat16，有同样风险
- **RKLLM W4A16**：RKLLM 内部可能有处理，目前未出现 NaN

### 4.2 Paraformer encoder Pow(x,2) 溢出

50 层 encoder 在 FP16 下 Pow 算子溢出。
- 修复：替换为 Mul(x, x)

## 5. 远程设备管理踩坑

### 5.1 Agent 搞坏系统 torch

Agent 把 wsl2-local 的 torch 从 2.11+cu121 降级为 2.4+cpu，导致后续所有 GPU 操作失败。

**教训**：派 agent 到远程机器时，必须要求用 venv 隔离，禁止 `--break-system-packages`。

### 5.2 Docker iptables 搞坏 RK3576 网络

RK3576 内核 (6.1.99) 缺少 `iptable_raw` 模块。Docker 默认修改 iptables 规则时会破坏 Tailscale 网络栈，导致设备不可达。

**修复**：Docker daemon.json 设置 `"iptables": false`。

### 5.3 RKLLM runtime 重启后丢失

`librkllmrt.so` 放在 `/tmp` 会在重启后消失。应复制到 `/usr/lib/`。

## 6. 性能数据汇总

### talker decode（核心 AR 循环）

| 方案 | ms/步 | 备注 |
|------|-------|------|
| RKNN FP16 (p512) | 1306 | 内存带宽瓶颈 |
| RKNN INT8 (p32) | 256 | 最优 RKNN |
| **RKLLM W4A16** | **55** | 生产用 ✅ |
| RKLLM 纯计算 | ~6 | one-shot 模式实测 |

### vocoder（音频解码）

| 方案 | 延迟 | RTF | 备注 |
|------|------|-----|------|
| RKNN 原始 | 10,547ms | 1.76 | 29 Sin CPU fallback |
| Sin→polynomial | 1,942ms | 0.32 | FP16, ctx=50+chunk=25 |
| ctx25 FP16 | 1,183ms | 0.59 | |
| **ctx25 INT8** | **682ms** | **0.34** | 生产用 ✅ |
| ctx0 INT8 | 327ms | 0.16 | 质量降低 |

### code_predictor（15 步残差预测）

| 方案 | 总时间 | 备注 |
|------|--------|------|
| RKNN 15 次调用 | 300ms | 当前方案 |
| RKLLM one-shot | 87ms | 但无法切 lm_head |
| RKLLM step-by-step | 2,250ms | 143ms/次调用开销 |
| 展开单图 RKNN | 403ms | ❌ 更慢（权重复制15份，带宽更大） |
| INT8 顺序 15 次 | 166ms | ❌ 精度不可用（top-1 仅 10%） |
| **每次调用开销实测** | **0.9ms** | 之前估 16ms 是错的，瓶颈是纯计算 |
| **RKLLM v1.2.3 + A55 核绑定** | **120ms** | ✅ 突破！cache 复用 + 消除 CPU 缓存争用 |

### 完整 pipeline

| 配置 | 每帧 | RTF | 状态 |
|------|------|-----|------|
| RKLLM talker + RKNN cp × 15 + vocoder ctx25 int8 | 355ms + 682ms | 4.4 + vocoder | 当前已部署 |
| + 展开 code_predictor (16 码本) | 125ms + 682ms | 1.6 + vocoder | 计划中 |
| + 减到 4 码本 | 82ms + 682ms | 1.0 + vocoder | 计划中 |
