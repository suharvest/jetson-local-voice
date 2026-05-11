# RKNN 框架从入门到放弃（再到没放弃）

> 我们花了三周时间，把语音 AI 塞进一块只有手掌大的 ARM 板子上。这是一篇血泪史。

---

## 前言：为什么要折腾 RKNN

我们团队在 Seeed Studio 做边缘 AI 相关的产品探索。公司一直在拓展 ARM 平台的 AI 能力——除了 Jetson 系列，RK3576/RK3588 也是重点关注的方向。正好拿到了一块 RK3576 的开发板，我们就想看看：在这颗只有 6 TOPS NPU 的芯片上，能不能跑通一个完整的语音交互系统——实时 TTS + ASR，延迟控制在 1s 以内。

Rockchip 官网说支持各种 AI 框架，文档页面摆了一堆框架的 logo。我们当时心想：6 TOPS，那不随便跑？

然后现实给了我们一记耳光，但我们最终还是跑起来了。

这篇文章记录我们踩过的每一个坑，走过的弯路，以及最终的解法。如果你也在折腾 RKNN，希望这篇文章能省你几周时间。

> **剧透**：我们中间尝试了很多看起来合理的方案——全图静态转换、matmul API 手写 transformer、per-layer ONNX 导出——最后发现大部分是弯路。真正跑通的路线是**子图拆分 + 混合推理**。这也是 RK 系列 NPU 架构真正的优势所在。

---

## 第一章：初识 RKNN——"6 TOPS，我可以！"

### 项目目标

在公司的 RK3576 测试设备上实现：
- TTS：中英文文字转语音，延迟 <500ms
- ASR：52 语言语音识别，RTF <1.0
- V2V（语音转语音）全链路延迟 <1s

模型选型：Qwen3-TTS（0.6B）+ Qwen3-ASR（RKNN encoder + RKLLM decoder）。

### RK 系列的架构优势

在聊踩坑之前，先说说为什么选 RK。

边缘 AI 芯片的 NPU 大致分两种设计思路：

- **全图编译型**（如 Hailo）：把整个模型编译成一张 NPU 指令流，数据不出芯片。CNN 时代这种方案很香——ResNet、YOLO 直接丢进去就跑。但到了 Transformer 时代，模型里有动态 shape、自回归循环、复杂控制流，全图编译就力不从心了。
- **子图混合型**（如 RK、Jetson）：允许把模型拆成多个子图，NPU 能跑的部分跑 NPU，不能跑的部分回 CPU。灵活性高，对 Transformer 友好。

RK 系列属于后者。而且 Rockchip 官方针对 LLM 场景专门做了 RKLLM 运行时，内部实现了 Flash Attention、SRAM 内算子融合、零拷贝 KV-cache——这些优化全在闭源的 `librknnrt.so` 里，但效果确实好。后面会详细讲。

这种子图灵活性，是我们最终能在 6 TOPS 的芯片上跑通语音 AI 的核心原因。

### 第一印象

RKNN 文档写得中规中矩。安装 rknn-toolkit2，写几行代码把 ONNX 转成 RKNN，push 到设备，`inference()`——

```
RuntimeError: NPU job timeout
```

……好，开始学习。

RKNN 的基本规则其实不复杂：

1. **只支持静态 shape**。所有维度必须在编译时固定。
2. **ONNX 先跑 onnxsim**。有 `If/Loop` 节点会直接报错。
3. **Simulator 通过 ≠ 设备能跑**。这条最坑，后面详说。

但这只是开始。

---

## 第二章：RKLLM 的蜜月期与分手

### 发现 RKLLM 快得离谱

我们最初的计划是把 Qwen3-TTS 的 talker（0.6B transformer，28 层）全部用 RKNN 跑。转换成功了，9 个子模型全部 build 完，推到设备上——

```
talker_decode: 1306ms/step，RTF=16.3
```

1306ms 一步。要生成 2 秒音频需要 25 步，总时间超过 30 秒。

彻底凉了。

然后我们发现了 RKLLM。它是 Rockchip 官方为 LLM 推理做的运行时，内部用了 Flash Attention 和 SRAM 内算子融合，把 KV-cache 和 attention 计算全部做在 NPU 内部缓存里，中间结果不走 DDR。

换成 RKLLM W4A16：

```
talker_decode: 55ms/step
```

快了 24 倍。RKLLM 不可替代，原因是这些优化全在闭源的 `librknnrt.so` 里，用 matmul API 手写根本追不上。

### 分手：RKLLM + RKNN 不能共存

我们后来要在同一个进程里跑：
- RKLLM（talker 自回归解码）
- RKNN（ASR encoder，Matcha TTS encoder）

结果：RKNN 在 RKLLM 运行后直接 hang 住，无报错。

调查了两天才搞清楚根因：**RKLLM 和 RKNN 默认共用 IOMMU domain 0**，RKLLM 初始化后占用了整个 domain，RKNN 再想用就卡死了。

GitHub Issue airockchip/rknn-llm#437 官方确认了这个问题，修复方法是：

```python
param.extend_param.base_domain_id = 1  # RKLLM 用 domain 1
# RKNN 固定在 domain 0，无需改动
```

一行配置，解决了两天的痛苦。

### 插曲：用 matmul API 手写一个开源 RKLLM？

RKLLM 这么快，但它是闭源的。我们动了一个念头：能不能用 RKNN 的 `rknn_matmul_api`（底层 NPU 矩阵乘法接口）自己手写一个开源的 transformer decoder？这样既不依赖闭源库，又能灵活控制架构。

我们真的做了。写了一整套 C 代码：RMSNorm、GQA attention、RoPE、SiLU FFN，每个 linear projection 用 matmul API 跑 NPU，CPU 做 norm 和 activation。还实现了 INT8 量化、context pool、双核并行（fork + 持久 context），甚至开源成了一个独立项目（[rknn-matmul-parallel](https://github.com/suharvest/rknn-matmul-parallel)）。

实测了一把：

| 方案 | ms/token | 说明 |
|------|---------|------|
| RKNN 整图 FP16 | 1306 | 纯 NPU，带宽瓶颈 |
| matmul API（单核） | 29 | 手写 transformer |
| matmul API（双核并行） | 21 | fork + 持久 context |
| **RKLLM W4A16** | **16** | 官方闭源运行时 |

**差距在底层**：matmul API 每次输出都走 DDR（片外内存），而 RKLLM 在 NPU 内部 SRAM（~384KB cbuf）里完成 QKV+attention+softmax 的全部计算，中间结果不出芯片。零拷贝 KV-cache、Flash Attention 分块加载——这些优化需要直接操作 NPU 内部寄存器，公开 API 做不到。

更要命的是：`rknn_matmul_run` 是**同步阻塞**的，单进程无法并发调度双核。我们用 fork 实现双核并行，也只有 1.37x 加速。

**结论**：AR decoder 这类任务，RKLLM 不可替代。官方的底层优化确实做得好，差距是架构级的，不是我们在 API 层面能追上的。但 matmul API 对于非 AR 任务（encoder、code predictor 等）仍然有价值——我们后来在 Qwen3-ASR 的 decoder 上用它做到了和 RKLLM 接近的 16ms/token。

---

## 第三章：精度是个什么鬼

### FP16 attention 溢出

Qwen3 的 attention head_dim=128，QK^T 的中间计算值在 FP16 下会超过最大值 65504，softmax 出 NaN。我们在 RTX 3060 上用 float16 推理时确认了这个问题（bfloat16 没问题，动态范围更大）。

问题是：RK3576 NPU **不支持 bfloat16**，只有 FP16 和 INT8。

这是 RK 系列目前最大的硬件短板之一。作为对比，我们在 Jetson 上用 TensorRT BF16 跑同样的模型完全没有精度问题——bfloat16 的动态范围和 FP32 一样，只是精度低一点，不会溢出。但 RK3576 没有这个选项。

所以在 RKNN 部署有 head_dim=128 的 attention 时，要么用 INT8 量化 attention 层，要么把精度敏感的部分拆到 CPU 跑 FP32。在 Jetson 上不存在的精度问题，到了 RK 上变成了主要工程障碍。

### INT4：别碰

librknnrt 2.3.2 的 INT4 NPU kernel 有 bug：正值 INT4 权重丢失约 50% 量级，负值正确。

社区 PR #412 分析是 nibble order（字节内高低 4 位顺序）问题，截至我们写这篇文章还没合并。

我们测了一下：
- INT8：cosine 0.99998，~35ms
- **INT4：cosine 0.06**，废了

直接禁用 INT4，用 INT8。

### librknnrt 2.3.2 在 RK3576 Linux 上 segfault

这个最离谱。2.3.2 runtime 在 RK3576 Linux 设备上的 `rknn_run` 有 regression，会 segfault。但 Android 设备不受影响——换句话说，官方自己测试的时候可能只测了 Android。

解法：runtime 降级到 2.3.0。toolkit 2.3.2 编译的模型可以在 2.3.0 runtime 上跑，会有 warning 但不影响结果。

### Vocoder INT8：magnitude 全零

Vocos vocoder 在 INT8 量化下，magnitude 输出全零。不是模型问题，是 RKNN INT8 量化对这个特定网络结构的处理有问题。

解法：vocoder 必须用 FP16。

---

## 第四章：静态图——RKNN 的铁律（以及我们走的弯路）

### 什么是"静态图"，为什么 RKNN 要求它

RKNN 要求模型的**每一个维度都在编译时确定**。这不是 RKNN 的缺陷，而是 NPU 硬件的根本限制——NPU 调度器需要提前规划数据搬运路线（DMA）、分配片上 SRAM，每一层的输入输出 buffer 大小都在编译时算好。如果维度是动态的，调度器没法工作。

这和 GPU 很不一样。GPU 有通用计算能力，可以在运行时动态分配显存；NPU 是固定功能单元（CNA+DPU+PPU），更像 ASIC。

### 哪些模型"天生静态"，哪些不行

| 类型 | 能否直接上 RKNN | 说明 |
|------|----------------|------|
| CNN 分类/检测 | 直接上 | 固定输入分辨率，所有维度编译时确定 |
| 固定长度 encoder | 直接上 | 比如 ASR encoder 固定 4s chunk |
| TTS vocoder (HiFi-GAN/Vocos) | 能，但需固定 mel 长度 | 用 bucket 策略 + padding + 裁剪 |
| Duration Predictor | **不能** | 输出长度取决于输入文本，编译时无法确定 |
| 自回归 decoder | **不能**（或代价极大） | KV-cache 长度每步变化，需要 RKLLM |
| Flow-matching ODE | 部分能 | estimator 能上 NPU，但累加循环需要在 CPU 做 |

**关键判断标准**：如果模型中有一个 op 的输出 shape 取决于**输入数据的值**（而不仅是输入的 shape），那这部分就不能直接上 RKNN。

### 静态化的核心技巧：bucket + padding + probe

对于"几乎静态"的模型（输入长度可变但范围有限），我们用 **bucket 策略**：

```python
# Matcha TTS 的 bucket 方案
# 短句 bucket：seq_len=80, x_len=64 → 输出 ~600 mel 帧（~9.6s 音频）
# 长句 bucket：seq_len=160, x_len=140 → 输出 ~1278 mel 帧（~20s 音频）
# 运行时按文本长度选 bucket，输出按实际帧数裁剪
```

但 bucket 有个隐蔽的坑：**onnxsim 会把固定输入算出的中间值 bake 成常量**。比如 Duration Predictor 在 onnxsim 时会把 "64 个 token" 的 duration 结果固化，导致运行时不管你传什么文本，输出的 mel 长度都一样。

解决方案是 **probe-first 策略**：

```
1. 先用 ORT 跑一遍原始（未简化）模型，记录所有动态 op 的输出值/shape
2. 再跑 onnxsim 固定 shape
3. 用预先记录的值替换被错误固化的节点
```

这就是我们在 Matcha 转换中用到的方法。先 probe，再 surgery，顺序不能反。

### 坦白说：我们在静态化上走了很多弯路

回头看，我们在"全图静态转换"这条路上花了太多时间：

- **Qwen3-TTS 全模型 RKNN**：9 个子模型全部转成功了，但 talker decode 1306ms/step，完全不可用
- **Matcha 全图 RKNN**：转成功了，能跑，但 FP16 精度在 ODE 累加中损失严重，mel 质量不达标
- **Piper VITS 全图 RKNN**：surgery 做完了，能跑，但固定 padding 污染了 Duration Predictor，输出音频和输入文本完全不相关
- **Code Predictor per-layer ONNX**：导出 5 层独立 RKNN，精度还行但 75 次调用开销太大（205ms）
- **Code Predictor 全图 RKNN**：编译器 bug（`input_output_align_nd_expand` 崩溃），disable 规则后编译通过但输出全错

**最终结论**：对于 Transformer 类模型，"全图静态化" 在 RK3576 上基本走不通。精度问题（FP16 + 无 BF16）、动态 shape、编译器 bug 三座大山挡在面前。

真正跑通的路线是**子图拆分 + 混合推理**——这也回到了 RK 架构本身的优势：灵活的子图定义能力。

### 不能静态化怎么办：拆模型

如果模型中确实有无法静态化的部分（这几乎是所有 Transformer 的常态），出路是**拆成子图**——计算密集且 shape 固定的部分跑 NPU，动态或精度敏感的部分跑 CPU。

下面两个模型的拆分案例说明了具体怎么做。

---

## 第五章：子图拆分——把模型切成两半

### 案例一：Matcha TTS 拆分（精度驱动）

Matcha TTS 用 flow-matching 架构，内部有一个 ODE（常微分方程）循环：

```
z₀ = noise                          # 初始噪声
z₁ = z₀ + dt * estimator(z₀, t₀)   # ODE 第 1 步
z₂ = z₁ + dt * estimator(z₁, t₁)   # ODE 第 2 步
z₃ = z₂ + dt * estimator(z₂, t₂)   # ODE 第 3 步
mel = denormalize(z₃)               # 最终 mel 谱
```

我们一开始把整个模型（3 步 ODE 展开）转成一张 RKNN 图。能跑，但 mel 质量很差——低频 bin（100-400Hz，语音最关键的 F0+F1 频段）偏差严重。

**根因**：RK3576 NPU 只支持 FP16。ODE 累加 `z = z + dt * v` 在 FP16 下精度不够，每步累积误差，3 步下来 cosine similarity 只有 0.918（需要 >0.95）。

**拆分方案**：

```
[NPU FP16] encoder：text → mu, mask, z₀
[NPU FP16] estimator：(z, mu, mask, time_emb) → velocity
[CPU FP32] ODE loop：z = z + dt * velocity（3 次调用 estimator）
```

拆分后，estimator 的 FP16 误差不会累积——每步的 velocity 都是独立计算的，累加在 CPU FP32 上做。

```python
# 拆分后的推理代码
mu, mask, z = encoder.inference(tokens)           # NPU
for step in range(3):
    v = estimator.inference(z, mu, mask, time_emb[step])  # NPU
    z = z + dt * v  # CPU FP32，不丢精度！
mel = z * sigma + mean  # CPU
```

文件产物：
- `matcha-encoder-fp16.rknn`（17.3MB）
- `matcha-estimator-fp16.rknn`（22.6MB）
- `time_emb_step{0,1,2}.npy`（预提取的时间 embedding）

### 案例二：Piper VITS 拆分（动态 shape 驱动）

Piper VITS 的拆分原因不同——不是精度问题，而是**动态 shape 无法静态化**。

VITS 架构的三个阶段：

```
Stage 1: Text Encoder + Duration Predictor + Length Regulator [动态 shape]
Stage 2: Flow (invertible 1x1 conv)                          [固定 shape]
Stage 3: HiFi-GAN Decoder                                    [固定 shape]
```

Stage 1 的 Length Regulator 根据 Duration Predictor 输出展开序列——输入 "你好" 和 "今天天气真不错" 产生的 mel 帧数完全不同。如果强行用固定 padding，Duration Predictor 会被 padding token 的 duration 预测污染，输出的音频和输入文本**完全不相关**（我们实测了，相关性接近 0）。

**拆分方案**：

```
[CPU ORT] encoder.onnx：tokens → z, y_mask    （动态 shape，ORT 处理）
[NPU RKNN] flow_decoder.rknn：z, y_mask → audio （固定 mel_len=256）
```

拆分点选在 Length Regulator 的输出 / Flow 的输入之间。自动检测方法：找 ONNX 图中第一个 `/flow/` 命名空间的节点，它的输入就是拆分点张量。

```python
# 拆分后的推理
z, y_mask = ort_session.run(None, {"input": phonemes, ...})  # CPU，动态长度
z_padded = pad_to(z, mel_len=256)                             # padding
audio = rknn.inference([z_padded, y_mask_padded])             # NPU，固定 shape
audio = trim_silence(audio)                                    # 裁掉 padding 产生的静音
```

性能：

| 阶段 | 耗时 | 运行位置 |
|------|------|---------|
| Encoder + DP + LR | 64-107ms（随文本变化） | CPU ORT |
| Flow + HiFi-GAN | 56-72ms | NPU RKNN |
| **合计** | **120-179ms，RTF=0.07** | |

### 拆分决策树

什么时候该拆，什么时候不该拆？

```
模型能直接转 RKNN 吗？
├── 能 → 直接跑，别拆
└── 不能 → 原因是什么？
    ├── 动态 shape → 在动态/静态边界拆（Piper 模式）
    ├── FP16 精度不够 → 把精度敏感的累积操作拆到 CPU（Matcha 模式）
    ├── 算子不支持 → 先试 graph surgery 替换
    │   ├── 能替换 → 替换后直接跑
    │   └── 替换后 CPU fallback crash → bake 常量或拆模型
    └── 维度超限 → 没办法，走 CPU（Kokoro 模式）
```

---

## 第六章：ONNX Graph Surgery——算子替换手册

### 为什么需要 surgery

RKNN 只吃静态图，且不支持一些常见 ONNX 算子。走 CPU fallback 的算子不仅慢（NPU↔CPU 上下文切换开销），还可能直接 crash（librknnrt 2.3.2 的 CPU fallback 内存管理有 bug）。

我们在 3 个 TTS 模型上分别做了 surgery，总结出一套通用的算子替换手册。

### 算子替换速查表

| 原始算子 | 替换方案 | 原因 | 适用模型 |
|---------|---------|------|---------|
| `Range` | `Constant`（probe 实际值后 bake） | 动态序列生成，NPU 不支持 | Matcha, Piper |
| `If/Loop/Sequence` | onnxsim 消除 | 控制流，NPU 不支持 | 通用 |
| `Ceil` | `Neg(Floor(Neg(x)))` | NPU 不支持 Ceil | Matcha |
| `Erf` | `x * Sigmoid(1.702 * x)` | NPU 不支持 Erf；用 Tanh 近似也行但 Sigmoid 版更简洁 | Piper |
| `Softplus` | `Log(1 + Exp(x))` | 数学等价分解 | Piper |
| `Sin`（SnakeBeta） | 7 阶 Taylor + Clip(-π,π) | CPU fallback 太慢（29 个 Sin → 10.5s） | Qwen3 vocoder |
| `Cos` | 6 阶 Taylor | 同 Sin | Qwen3 vocoder |
| `RandomNormalLike` | 固定噪声常量（seed=42） | NPU 不支持随机算子 | Matcha, Piper |
| `ScatterND` | ORT probe + bake 常量，或 `Slice+Concat` | CPU fallback 会 double free crash | Piper |
| `GatherND` | ORT probe + bake 常量 | 同 ScatterND | Piper |
| `CumSum` | `MatMul`（上三角全 1 矩阵） | CPU fallback crash | Piper |
| `NonZero` | `Clip(-5,5) + Where`（改写 spline 逻辑） | NPU 不支持 | Piper |
| `Pow(x, 2)` | `Mul(x, x)` | 50 层 encoder FP16 下 Pow 溢出 | Paraformer |

### Sin 算子替换详解

这是我们遇到的最戏剧性的案例。Qwen3-TTS vocoder 的 SnakeBeta 激活函数包含 29 个 Sin：

```
SnakeBeta(x) = x + (1/β) * sin²(β*x)
```

这 29 个 Sin 全部显示 `Unkown op target: 0`——CPU fallback。结果：

```
替换前：10,547ms（29 次 NPU→CPU→NPU 上下文切换）
Taylor 多项式替换后：1,942ms（5.4x 提速，所有 op 留在 NPU）
```

Taylor 近似用 Horner form 实现，只需 `Mul` 和 `Add`（都是 NPU 原生支持的）：

```python
# sin(x) ≈ x * (1 + x² * (-1/6 + x² * (1/120 + x² * (-1/5040))))
# 7 个 ONNX 节点替换 1 个 Sin，但全部跑在 NPU 上
```

重点：**rotary embedding 里的 Sin/Cos 不需要替换**——RKNN 编译器能识别 RoPE pattern 并用 NPU 内置的 exSDP 融合算子处理。我们在脚本里用路径名过滤：包含 "rotary" 的 Sin/Cos 跳过。

### Piper VITS 的完整 Surgery 流程（8 步）

这是我们做过最复杂的 surgery，把 Piper VITS 从 2755 节点缩到 721 节点：

**Step 0: 注入动态掩码输入（onnxsim 之前）**

这步最关键也最容易被忽略。onnxsim 会把 `input_lengths` 折叠成常量，导致 `sequence_mask` 变成全 1——padding 位置的 token 也会被当作真实输入，Duration Predictor 输出的时长全部错乱。

解法：在 onnxsim 之前，把 `x_mask`、`audio_length`、`cumulative_durations` 作为新输入注入模型，替换掉原来从 `input_lengths` 动态计算的子图。

```python
# 注入后的输入列表
inputs = ["input", "input_lengths", "scales",
          "x_mask",                    # (1, 1, seq_len) 新增
          "audio_length",              # (1,) 新增
          "cumulative_durations"]      # (1, seq_len+1) 新增
```

**Step 1-7:**
1. onnxsim 固定 shapes（2755 → 1889 节点）
2. `Range` → `Constant`（位置编码）
3. `Erf` → `x * Sigmoid(1.702*x)`（GELU 近似）
4. `Softplus` → `Log(1 + Exp(x))`
5. `RandomNormalLike` → 固定噪声（seed=42）
6. Spline coupling 改写：`NonZero+GatherND+ScatterND` → `Clip+Where`（消除 60 个节点）
7. 剩余 `ScatterND` → `Slice+Concat`；`CumSum` → `MatMul`（上三角矩阵）

最终：63MB ONNX → 35.5MB RKNN，**51ms 生成 1.5s 音频，RTF=0.034**。

### Matcha TTS 的 probe-first Surgery

Matcha 的 surgery 比 Piper 简洁（5 步），但有个重要的方法论创新：**probe-first**。

```
1. 先用 ORT 推理原始模型，捕获所有 Range/RandomNormalLike 的输出值
2. 再跑 onnxsim（此时 Range 已被折叠成错误的常量）
3. 用预先 probe 的正确值替换被错误折叠的节点
```

为什么要 probe-first？因为 onnxsim 折叠 Range 时用的是固定输入的值，但 Range 的 `start/limit/delta` 可能依赖运行时数据。先 probe 能拿到正确的输出形状和值，后续替换就不会出错。

```
surgery 步骤：
1. onnxsim 固定 seq_len=80
2. Range → pre-probed Constant（4 个）
3. Ceil → Neg(Floor(Neg(x)))（1 个）
4. Slice 动态 ends → probed Constant（1 个）
5. RandomNormalLike → pre-probed 固定噪声
```

### Kokoro 的"不可能任务"

我们也尝试了 Kokoro（82M）。Surgery 做完了（5039 → 1712 节点，所有不兼容算子消除），ORT 验证正常，但 RKNN build 失败：

```
REGTASK: The bit width of field value exceeds the limit,
         target: f2, limit: 0x1fff, value: 0x3318 (13080)
```

RK3576/3588 NPU 的寄存器位宽限制：**单层最大时间维度 8191**。Kokoro 的 ISTFTNet vocoder 内部时间维度是 13081（上采样后更达 65400），远超硬件限制。

这不是算子支持问题——99.4% 的算子 RKNN 都支持，但 50.2% 的算力集中在超限的 vocoder 后半段。拆模型收益有限，最终放弃，Kokoro 走 CPU onnxruntime。

**教训**：surgery 能解决算子兼容性问题，但解决不了硬件维度上限。这是物理限制。

---

## 第七章：自定义算子——当 Surgery 不够用的时候

Taylor 多项式替换 Sin 虽然管用，但毕竟是近似（max error ~1.5e-6）。如果对精度要求更高，或者算子的输出依赖输入数据无法 bake 常量，可以走 RKNN 自定义算子路线。

### RKNN Custom Op 注册机制

RKNN 允许注册自定义 CPU 算子，在 NPU 图执行时 callback 到 C 函数处理。注意 rknnlite Python API 没有暴露 `rknn_register_custom_ops()`，需要用 ctypes 直接调用 `librknnrt.so`：

```python
import ctypes

librknnrt = ctypes.CDLL("/usr/lib/librknnrt.so")
register_fn = librknnrt.rknn_register_custom_ops
ret = register_fn(ctx_value, ops_array, num_ops)
```

关键细节：返回的 ctypes callback 对象**必须保持引用不被 GC**，否则在 C 层调用时会 segfault。这个坑非常隐蔽——GC 发生在不确定的时间点，可能跑了 100 次推理才突然 crash。

### ARM NEON 加速的 Sin

我们实现了一个 NEON 向量化版本，4 个 float32 并行计算：

```c
// 7 阶 minimax 近似，max error ~1.5e-6
static inline float32x4_t vsinq_f32(float32x4_t x) {
    // range reduction: x -= 2*pi * round(x / 2*pi)
    float32x4_t n = vrndnq_f32(vmulq_f32(x, inv_2pi));
    x = vfmsq_f32(x, n, two_pi);
    // Horner form: x*(1 + x²*(c3 + x²*(c5 + x²*c7)))
    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t r = vfmaq_f32(c5, c7, x2);
    r = vfmaq_f32(c3, r, x2);
    r = vfmaq_f32(one, r, x2);
    return vmulq_f32(r, x);
}
```

同时支持 FP16 输入（`vcvt_f32_f16` → 计算 → `vcvt_f16_f32`），对 FP16 NPU 图透明。编译命令：

```bash
gcc -shared -fPIC -O2 -march=armv8-a+simd -o libcstops.so cst_sin_op.c -lm
```

我们最终注册了 5 个自定义算子：`CstSin`、`CstMul`、`CstPow`、`CstAdd`、`CstInstanceNorm`，覆盖了 vocoder 里所有 CPU fallback 的场景。

---

## 第八章：多模型共存——NPU 的独占式婚姻

### NPU 是个独占资源

RK3576 NPU 的内存大约 180MB。RKLLM 单个模型就能占 ~140MB，留给 RKNN 的空间很少。

我们曾经尝试同时加载 Matcha RKNN + RKLLM：Matcha 加载成功，但 `inference()` 调用时直接 SIGSEGV（exit code 139）。内存超限，推理时 NPU job 提交失败。

解法：TTS 和 ASR 分时复用 NPU，或者 TTS 改成不需要 RKLLM 的方案（Matcha TTS 就是这样做的）。

### NPU handle 泄漏

RKNPU 驱动 0.9.8 在进程异常退出时不清理 matmul context 的 DMA handle。多次 crash 后 handle table 满，连单个 `rknn_create_mem` 都会返回 EINVAL。

解法：`atexit` + signal handler 保证正常退出前调用 `rknn_matmul_destroy`。Docker `restart: unless-stopped` 配合反复 crash 会持续泄漏，最终需要 reboot。

### Docker iptables 的意外惊喜

在 RK3576 上跑 Docker 之后，设备 Tailscale 网络突然断了。原因：RK3576 内核（6.1.99）缺少 `iptable_raw` 模块，Docker 默认修改 iptables 规则时破坏了 Tailscale 的网络栈。

修复：`/etc/docker/daemon.json` 加 `"iptables": false`。

这种"Docker 启动完，网没了"的问题，第一次遇到真的很懵。

---

## 第九章：最终成绩单

经过三周的折腾，我们最终跑通了整个 TTS+ASR pipeline。

### Matcha TTS（中英文 TTS，拆分模式）

| 组件 | 延迟 | 运行位置 |
|------|------|---------|
| Matcha encoder RKNN | ~170ms | NPU FP16 |
| Matcha estimator RKNN × 1 step | ~170ms | NPU FP16 |
| ODE 累加 | <1ms | CPU FP32 |
| Vocos RKNN | ~33ms | NPU FP16 |
| ISTFT | ~25ms | CPU |
| **全流程** | **~430ms，RTF=0.054** | |

比实时快 18 倍。1 步 ODE（`MATCHA_ODE_STEPS=1`）是精度和速度的最优平衡。

### Piper VITS（多语言 TTS，混合推理模式）

| 组件 | 延迟 | 运行位置 |
|------|------|---------|
| Encoder + DP + LR | 64-107ms | CPU ORT（动态 shape） |
| Flow + HiFi-GAN | 56-72ms | NPU RKNN（固定 mel_len=256） |
| **合计** | **120-179ms，RTF=0.07** | |

已支持 17 种语言（en/zh/de/fr/es/it/ru/pt/nl/pl/ar/tr/vi/uk/sv/cs + ja CPU fallback），每种语言 encoder 26.8MB + flow_decoder 19.7MB。

### Qwen3-ASR（52 语言流式识别）

| 指标 | 值 |
|------|-----|
| **RTF** | **0.44** |
| Encoder（RKNN FP16） | 431ms / 4s chunk |
| Decoder（RKLLM W4A16） | ~16ms/token |

说完话到出文字约 0.5-0.6s（流式模式）。

### V2V 全链路延迟

估算约 0.8-1.1s，已接近设计目标。

---

## 总结：RK NPU 的优势与不足

### RK 做对了什么

1. **子图混合推理架构**——允许 NPU/CPU 灵活分工，Transformer 时代这比全图编译灵活太多
2. **RKLLM 运行时**——闭源但确实强，Flash Attention + SRAM 融合 + 零拷贝 KV-cache，W4A16 量化 16ms/token，API 层面追不上
3. **算子融合优化**——exSDP（attention 融合）、Conv+BN 融合，编译器自动完成
4. **matmul API 开放**——虽然比 RKLLM 慢，但给了开发者底层 NPU 计算的入口，非 AR 任务有用
5. **Custom Op 机制**——允许注册自定义 CPU 算子，在 NPU 图执行时回调 C 函数

### RK 目前的短板

1. **不支持 BFloat16**——这是最大的精度障碍。同样的模型在 Jetson 上用 TRT BF16 跑得好好的，到 RK 上精度就崩。Transformer attention 的 QK^T 在 FP16 下容易溢出，而 BF16 完全没这个问题。
2. **驱动 Bug 不少**——INT4 nibble order bug、2.3.2 segfault regression、CPU fallback double free、NPU handle 泄漏……社区 issue 里报了不少，修复周期偏长
3. **NPU 内存上限 ~180MB**——大模型需要拆分或用 RKLLM（RKLLM 有自己的内存管理）
4. **寄存器维度上限 8191**——vocoder 类模型时间维度经常超限
5. **文档偏弱**——很多关键细节（IOMMU domain、io_attr 结构体大小、B_layout 语义）需要翻源码或社区 issue 才能搞清楚

### 给新手的 checklist

1. **先别想全图静态转换**——Transformer 类模型大概率走不通，直接从子图拆分开始设计
2. **先跑 onnxsim**，消除控制流（`If/Loop/Sequence`）
3. **判断哪些部分能上 NPU**——计算密集 + shape 固定的部分上 NPU，动态或精度敏感的留 CPU
4. **检查 `Unkown op target: 0`**——这些 op 在真机上可能 crash（Simulator 不会！）
5. **必须真机验证**，Simulator 通过 ≠ 设备能跑
6. **RKLLM + RKNN 共存** → `base_domain_id=1`
7. **禁用 INT4**（librknnrt 2.3.2 有 nibble order bug）
8. **FP16 精度敏感操作拆到 CPU**（ODE 累加、attention QK^T 等）
9. **crash 后 reboot**（NPU handle 泄漏）
10. **Docker iptables 关掉**（RK3576 内核缺 iptable_raw 模块）

---

## 后记

折腾这些东西最大的感触有两个。

第一，**边缘 AI 不是"框架导入模型，推理"这么简单**。要深入理解 NPU 硬件限制、驱动 bug、内存管理，甚至得手写 C 代码做 NEON SIMD 优化。我们中间走了很多弯路——全图静态转换、matmul API 手写 transformer、per-layer ONNX 导出——最后发现大部分路线因为精度问题走不通。真正跑通的是子图拆分 + 混合推理。

第二，**RK 的生态其实在快速进步**。RKLLM 的性能确实让我们惊喜（W4A16 量化 16ms/token，这在 6 TOPS 的芯片上是很不错的）。子图灵活性、matmul API 开放、Custom Op 机制——这些设计选择说明 Rockchip 理解 Transformer 时代的需求。如果后续能补上 BFloat16 支持、修复驱动 bug、改善文档，RK 系列会是非常有竞争力的边缘 AI 平台。

跑通的那一刻也很爽——120ms 生成一句话的语音，17 种语言即时切换，52 语言流式识别，V2V 延迟接近 1 秒，在一块功耗不到 10W 的小板子上。这大概就是边缘 AI 的魅力吧。

---

## 相关项目

- **GitHub: [suharvest/rknn-matmul-parallel](https://github.com/suharvest/rknn-matmul-parallel)**
  开源的 RK3576/3588 NPU matmul 加速库，支持 FP16/INT8，双核并行，含 Qwen3-ASR 完整示例

- **主项目: suharvest/jetson-voice**
  ARM64 边缘设备低延迟语音交互系统（Jetson + RK3576），后续会拆出独立的 RKNN 部署项目

如果这篇文章帮到了你，欢迎给 rknn-matmul-parallel 点个 star。如果你也在 RK 上踩了别的坑，欢迎在评论区分享——让后来人少走一点弯路。
