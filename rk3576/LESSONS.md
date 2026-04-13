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
| RKNN per-layer ONNX (W4A16) | 205ms | ❌ 75 次调用开销太大 |
| RKNN full transformer (W4A16) | 150ms | ❌ 精度崩溃（cosine 0.01），需 disable_rules 破坏正确性 |
| MatMul API approach | **69ms** | ✅ 目前最快 |

### code_predictor per-layer ONNX 实验 (2026-04-05)

**方案**：每层 transformer 导出为独立 ONNX，包含 RMSNorm + GQA attention + FFN。

**关键发现**：
1. **exSDPAttention 融合成功** — 所有 5 层的 MatMul+Softmax+MatMul 都被融合为 exSDPAttention（NPU FLOAT16）
2. **W4A16 精度问题** — Layer 2 的 cosine similarity 仅 0.76（其余层 0.98+），与 calibration 数据无关（W4A16 只量化权重）
3. **调用开销主导延迟** — 单层 2.4ms，75 次调用 = 180ms 纯开销
4. **全模型 RKNN 编译 bug** — `input_output_align_nd_expand` 规则在 5 层模型上崩溃；disable 后编译通过但输出完全错误（fp16 也一样）
5. **RMSNorm 的 rsqrt 分解** — 导出为 1/sqrt(x) = Div 算子，fallback CPU，增加 NPU↔CPU 切换开销

**数据**：
| 模型 | 大小 | 单次推理 | 15 步总时间 | 精度 (cosine) |
|------|------|---------|-----------|-------------|
| 单层 W4A16 ×5 | 46 MB | 2.4ms/层 | 205ms | 0.78 (cascaded) |
| 全模型 W4A16 | 44 MB | 8.5ms | 150ms | 0.01 ❌ |
| 全模型 FP16 | 161 MB | 18ms | ~290ms | 0.01 ❌ |

**结论**：per-layer ONNX 方案不可行。exSDPAttention 融合虽然成功，但 (1) 调用开销抵消了计算优化，(2) W4A16 在 Layer 2 精度不足，(3) 全模型编译需要 disable 影响正确性的规则。MatMul API (69ms) 仍是 code_predictor 最优方案。

### 完整 pipeline

| 配置 | 每帧 | RTF | 状态 |
|------|------|-----|------|
| RKLLM talker + RKNN cp × 15 + vocoder ctx25 int8 | 355ms + 682ms | 4.4 + vocoder | 当前已部署 |
| + 展开 code_predictor (16 码本) | 125ms + 682ms | 1.6 + vocoder | 计划中 |
| + 减到 4 码本 | 82ms + 682ms | 1.0 + vocoder | 计划中 |

## 7. Matcha TTS FP16 精度分析 (2026-04-13)

### 7.1 问题

Matcha TTS (3-step ODE 扩散模型) 在 RK3576 NPU FP16 下 round-trip 准确率只有 5/10，短句（"你好世界"、"欢迎使用"）完全乱码。

### 7.2 诊断路径

**Step 1: 定位 ODE 累积误差**
- 拆分模型为 encoder + estimator，ODE 累加在 CPU FP32
- 结果：5/10 → 5/10（CPU ODE 没改善，因为单模型里 ODE 不是主要问题）
- 实测帧 90+ mel 能量衰减到 ORT 的 2-10%

**Step 2: 减少 ODE 步数**
- 1-step ODE (dt=1.0, time_emb_step0)：7/10, 282ms
- 3-step 和 1-step 的 RKNN mel 之间 cosine=0.996（差异很小）
- 说明步数不是主因，单步 estimator 本身的 FP16 就有问题

**Step 3: accuracy_analysis() 逐层分析**
- GEGLU 激活 (ff/net.0/Mul_1): single_euc 0.17-0.18（最差单层 op）
- InstanceNorm (exNorm): single_euc 0.08-0.12
- 中间 Conv/MatMul: 累积 cosine 从 mid_blocks 开始降到 0.99985

**Step 4: 自定义 CPU 算子实验（关键！）**
- 替换 6 个 Snake 激活 (Mul+Sin+Pow+Mul+Add) + 13 个 InstanceNorm = 43 个 op 为 CPU FP32
- 结果：**cosine 0.926 → 0.926（零改善！）**
- 结论：Snake/InstanceNorm 是误差放大器，不是误差源

**Step 5: 确认 Conv/MatMul 是真正瓶颈**
- 43 个自定义 op 在 CPU FP32 跑，精度不变 → 剩下的 Conv/MatMul（90%+ 计算量）才是精度损失源
- Conv/MatMul 是 NPU 核心运算，无法通过自定义算子替换

### 7.3 自定义算子实战数据

| 发现 | 详情 |
|------|------|
| 数据类型 | 自定义 op 收到 **FP32** 数据（RKNN 自动 FP16→FP32 转换） |
| Buffer 复用 | 连续 CPU op 之间 **不共享 buffer**（每个 op 独立 malloc） |
| 性能影响 | 43 个 CPU op 导致 TTS 从 282ms → 399ms（+41%） |
| 精度影响 | cosine 0.926 → 0.926（**零改善**） |
| 注册方式 | ctypes 调 `rknn_register_custom_ops()`，rknnlite 不暴露此 API |
| context 获取 | `rknn_lite_obj.rknn_runtime.context` (uint64) |
| GC 风险 | 必须保持 ctypes 引用，否则回调指针失效 → segfault |

### 7.4 编译配置实验（全部无效）

| 配置 | 结果 |
|------|------|
| optimization_level 0/1/2/3 | 真机 **无差异**（simulator 也无差异） |
| float_dtype='bfloat16' | **编译失败** (RK3576 NPU 不支持) |
| float_dtype='tfloat32' | **编译失败** |
| enable_flash_attention=True | 真机 **无差异** |
| quantize_weight=True (w8a16) | **平台不支持** |
| compress_weight=True | **bit-identical** |
| op_target (指定 op 走 CPU) | **对 FP16 模型无效**（只在 INT8 混合量化场景有用） |
| ONNX Sin range reduction | 真机 **无差异** |
| mel 后处理平滑 (kernel=3) | **反而变差**（短句模糊） |
| mel adaptive 平滑 | **无改善** |

### 7.5 ORT CPU vs RKNN NPU 对比

| 模式 | TTS 延迟 | Round-trip 准确率 | 说明 |
|------|---------|------------------|------|
| RKNN 单模型 3-step | 535ms | 5/10 | 基线 |
| RKNN split + 3-step CPU ODE | 535ms | 5/10 | ODE 精度修复无效 |
| RKNN split + 1-step | 282ms | 7/10 | 减少累积次数 |
| RKNN + 43 自定义 CPU op | 399ms | 6/10 | 激活/Norm 不是瓶颈 |
| **ORT CPU FP32 (NEON)** | **443ms** | **9/10** | **最终方案** |
| ORT CPU 4 threads | 251ms | — | 纯 Matcha 延迟 |
| ORT CPU 1 thread | 590ms | — | 线程扩展 2.35x |

**关键洞察**：ORT 是动态 shape，短句（≤7 tokens）比 RKNN 更快（232ms vs 280ms），因为 RKNN 固定 600 帧。

### 7.6 结论

1. **RK3576 FP16 NPU 的精度瓶颈在 Conv/MatMul 硬件层面**，无法通过 graph surgery、自定义算子、编译选项解决
2. **accuracy_analysis 定位的 "worst op" 是误导**——它们是误差放大器，不是误差来源
3. 对迭代/扩散模型，**ORT CPU NEON FP32 是 RK3576 上的最优方案**
4. NPU 适合非迭代模型（encoder、vocoder），对 FP16 精度不敏感的场景
