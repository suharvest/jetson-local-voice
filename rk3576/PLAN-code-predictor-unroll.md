# Code Predictor 展开方案

## 目标

把 code_predictor 的 15 步串行 RKNN 调用（300ms）合并为 1 次调用（~70ms），使整条 pipeline 接近实时。

## 当前瓶颈

```
每帧 = talker_decode(55ms) + 15 × code_predictor(20ms) = 355ms
预算 = 80ms/帧 (12.5Hz codec)
```

20ms/次 里 ~3.6ms 是 NPU 计算，~16ms 是调度开销。15 次调度开销 = 240ms 浪费。

## 方案

### 核心思路

导出一个「展开版」ONNX，把 15 步 code_predictor 循环 bake 到一个图里：

```
输入:  last_hidden [1, 1024], primary_embed [1, 1024]
输出:  residual_codes [15], codec_sum [1, 1024]

图内部 (全部在 NPU):
  Step 0: transformer([hidden, primary]) → matmul(lm_head_0) → argmax → matmul(onehot, codebook_0) → embed_0
  Step 1: transformer([hidden, primary, embed_0]) → matmul(lm_head_1) → argmax → matmul(onehot, codebook_1) → embed_1
  ...
  Step 14: ... → embed_14
  
  codec_sum = primary_embed + sum(embed_0..14)
```

### NPU 兼容替换

| 原始 op | NPU 兼容替换 | 说明 |
|---------|-------------|------|
| Gather (embedding lookup) | one_hot @ embedding_table (MatMul) | embedding [2048, 1024] 作为常量权重 |
| argmax | ReduceMax + Equal + Cast | 或 TopK(k=1) |
| 15 个 lm_head | 15 个 MatMul (常量权重) | 每个 [1024, 2048] |
| 15 个 codebook | 15 个常量 embedding table | 每个 [2048, 1024] |

### context 策略

两个选择：

**A. 完整递增 context (正确但更慢)**
- Step j 的 attention 处理 j+2 个 token
- 总 attention token-steps: 2+3+...+17 = 142
- 但 RKNN 需要固定 shape → pad 到 17 tokens + attention mask
- 预估: ~100-150ms

**B. 固定 2-token sliding window (当前近似)**
- 每步只看 [accumulated_context, latest_embed]
- 15 × 2 token = 30 token-steps
- 和当前 RKNN 行为一致，不改变输出质量
- 预估: ~70ms

**推荐 B**：和当前已部署的行为一致，不引入质量变化。

## 实现步骤

### Step 1: PyTorch 展开模块

```python
class CodePredictorUnrolled(nn.Module):
    def __init__(self, transformer, lm_heads, codebook_embeds):
        super().__init__()
        self.transformer = transformer  # 5-layer Qwen3 transformer
        self.lm_heads = nn.ParameterList([nn.Parameter(h) for h in lm_heads])  # 15 × [2048, 1024]
        self.codebook_embeds = nn.ParameterList([nn.Parameter(e) for e in codebook_embeds])  # 15 × [2048, 1024]
    
    def _argmax_embed(self, logits, codebook):
        # logits: [1, 2048] → code: scalar → embed: [1, 1024]
        # NPU-friendly: softmax → hard attention → matmul
        code = logits.argmax(dim=-1)  # [1]
        one_hot = torch.zeros(1, 2048)
        one_hot.scatter_(1, code.unsqueeze(1), 1.0)
        embed = one_hot @ codebook  # [1, 1024]
        return code, embed
    
    def forward(self, last_hidden, primary_embed):
        # last_hidden: [1, 1024], primary_embed: [1, 1024]
        codes = []
        codec_sum = primary_embed.clone()
        accumulated = last_hidden  # running context
        
        for j in range(15):
            # 2-token context: [accumulated, latest]
            if j == 0:
                ctx = torch.stack([last_hidden, primary_embed], dim=0).unsqueeze(0)  # [1, 2, 1024]
            else:
                ctx = torch.stack([accumulated, embed], dim=0).unsqueeze(0)
            
            hidden = self.transformer(ctx)  # [1, 2, 1024]
            logits = hidden[:, -1, :] @ self.lm_heads[j].T  # [1, 2048]
            code, embed = self._argmax_embed(logits, self.codebook_embeds[j])
            codes.append(code)
            codec_sum = codec_sum + embed.squeeze(0)
            accumulated = hidden[:, -1, :].squeeze(0)  # update accumulated context
        
        return torch.stack(codes), codec_sum
```

**位置**: `rk3576/scripts/export_code_predictor_unrolled.py`

### Step 2: ONNX 导出

```python
torch.onnx.export(
    wrapper,
    (dummy_hidden, dummy_primary),
    "code_predictor_unrolled.onnx",
    input_names=["last_hidden", "primary_embed"],
    output_names=["residual_codes", "codec_sum"],
    opset_version=18,
    do_constant_folding=False,  # 保留 shape 节点
    dynamo=False,
)
```

固定 shape: 输入 [1, 1024] × 2，输出 [15] + [1, 1024]

### Step 3: ONNX 后处理

1. 检查是否有 Gather/Sin 等 NPU 不支持的 op
2. Gather → MatMul 替换（复用 `replace_sin_polynomial.py` 的图手术模式）
3. 验证：和 15 次分步调用的输出对比

### Step 4: RKNN 转换

```python
from rknn.api import RKNN
rknn = RKNN()
rknn.config(target_platform="rk3576", optimization_level=0)
rknn.load_onnx(model="code_predictor_unrolled_fixed.onnx")
rknn.build(do_quantization=False)  # 先 FP16，后续可 INT8
rknn.export_rknn("code_predictor_unrolled.rknn")
```

### Step 5: 集成到 tts_service.py

替换 `_run_code_predictor_loop()`:

```python
# 之前: 15 次 RKNN 调用
for j in range(15):
    logits = rknn_code_predictor.inference(ctx)
    code = argmax(logits)
    embed = codebook[j][code]
    ...

# 之后: 1 次 RKNN 调用
residual_codes, codec_sum = rknn_unrolled.inference([last_hidden, primary_embed])
```

### Step 6: Benchmark

在 cat-remote 上对比:
- 当前: 15 × RKNN = 300ms
- 展开: 1 × RKNN = ?ms (目标 <100ms)

## 风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| ONNX 图太大，RKNN 编译失败 | 阻塞 | 减少码本数 (15→7) 缩小图 |
| argmax 在 ONNX 里的梯度不连续 | 导出失败 | 用 straight-through estimator 或 ScatterND 替代 |
| 展开后精度和分步不一致 | 质量问题 | 验证对比，固定随机种子 |
| RKNN 不支持 ScatterND/TopK | 图手术 | 用 ReduceMax+Equal+MatMul 替代 |
| 模型超过 NPU 180MB 内存限制 | 加载失败 | INT8 量化，或分成两个图 (step 0-7, 8-14) |

## 预期结果

| 指标 | 当前 | 展开后 |
|------|------|--------|
| code_predictor 时间 | 300ms | ~70ms |
| 每帧总计 | 355ms | ~125ms |
| RTF | 4.4 | 1.6 |
| 配合 8 码本 | — | ~96ms (RTF≈1.2) |
| 配合 4 码本 | — | ~82ms (RTF≈1.0) |
