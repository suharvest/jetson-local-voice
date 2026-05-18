# TTS Pipeline Validation Toolkit

逐步验证 TTS 实现的正确性，定位到具体出错的步骤。

## 文件

| 文件 | 用途 |
|------|------|
| `tts_dump_reference.py` | 在 WSL2 上跑官方模型，dump 每个步骤的中间数据 |
| `tts_validate.py` | 对比参考数据 vs 目标实现，逐步 + E2E，出报告 |
| `tts_eval.py` | 轻量 E2E 评估（只需 /tts + /asr API） |
| `tts_ref/` | 官方参考音频（9 个 test case） |

## 流程

### Step 1: 生成参考数据（WSL2，只需一次）

```bash
ssh harve@100.73.210.80
source /tmp/qwen3-tts-env/bin/activate
cd /tmp/Qwen3-TTS
python /path/to/tts_dump_reference.py --output-dir /tmp/tts_ref_dump --seed 42 --cases 3
```

输出结构：
```
tts_ref_dump/
  0/                          # case "你好"
    token_ids.npy             # int64 text token IDs
    text_embeds.npy           # [1, N, 1024] float32
    prefill_logits.npy        # [1, N, 3072] float32
    prefill_hidden.npy        # [1, N, 1024] float32
    primary_codes.npy         # [n_frames] int32
    cp_codes.npy              # [n_frames, 15] int32
    codec_sums.npy            # [n_frames, 1024] float32
    audio.wav
    metadata.json
    frames/
      frame_0_input_embed.npy   # [1, 1, 1024]
      frame_0_talker_logits.npy # [1, 1, 3072]
      frame_0_talker_hidden.npy # [1, 1, 1024]
      frame_0_cp_step_logits.npy # [15, 2048]
      ...frame_1..4...
  1/                          # case "今天天气真不错"
  2/                          # case "今天天气真不错，我们一起去公园散步吧。"
```

### Step 2: 目标实现 dump 中间数据

你的 TTS 实现需要保存相同格式的 .npy 文件到一个目录。关键文件：

- **必须**: `token_ids.npy`, `primary_codes.npy`, `cp_codes.npy`, `audio.wav`
- **推荐**: `text_embeds.npy`, `prefill_logits.npy`, `prefill_hidden.npy`, `codec_sums.npy`
- **可选**: `frames/frame_*_*.npy`（前 5 帧的逐帧数据）

文件越多，定位问题越精确。缺失的文件会跳过比较。

### Step 3: 运行验证

```bash
# 逐步比较（需要目标实现的 .npy 文件）
python tts_validate.py \
    --ref-dir /tmp/tts_ref_dump \
    --target-dir /tmp/tts_target_dump

# E2E 比较（只需 API，不需要中间数据）
python tts_validate.py \
    --ref-dir /tmp/tts_ref_dump \
    --api-host localhost:8000

# 两者都跑
python tts_validate.py \
    --ref-dir /tmp/tts_ref_dump \
    --target-dir /tmp/tts_target_dump \
    --api-host localhost:8000

# 只跑某个 case
python tts_validate.py --ref-dir /tmp/tts_ref_dump --target-dir /tmp/tts_target_dump --case 0
```

### Step 4: 读报告

```
======================================================================
Case 0: "你好"
======================================================================

--- Step-by-step ---

✓ token_ids                PASS  MATCH (9 tokens)
✓ text_embeds              PASS  cos=1.0000  max_diff=2.3e-06
✓ prefill_logits           PASS  cos=0.9998  max_diff=3.1e-02
✗ prefill_hidden           FAIL  cos=0.8912  max_diff=1.2e+00  worst@(0,5,234)
✗ primary_codes            FAIL  DIVERGE (3/10 mismatch, first at idx 2)

--- E2E via API ---

✓ e2e_audio                PASS  dur=0.88s  sr=24000
✓ e2e_dur_ratio            PASS  ref=0.95s  tgt=0.88s  ratio=0.93
✗ e2e_cer                  FAIL  CER=35%  ref="你好"  asr="你好失败"

RESULT: FAIL — first divergence at: prefill_hidden
```

**关键**：报告的 "first divergence" 就是问题根因所在。上游步骤正确但该步骤出错 → 问题在该步骤的实现。

## Pipeline 步骤对照

| 步骤 | 检查文件 | 常见问题 |
|------|---------|---------|
| tokenization | `token_ids.npy` | chat template 格式、特殊 token |
| text embedding | `text_embeds.npy` | embed_tokens 权重精度、FP16 vs BF16 |
| talker prefill | `prefill_logits.npy`, `prefill_hidden.npy` | attention mask、position encoding、BF16 溢出 |
| primary code sampling | `primary_codes.npy` | temperature、top-k、suppress tokens |
| CP residual codes | `cp_codes.npy` | **并行 vs autoregressive**（最常见根因）、lm_head 选择 |
| codec sum | `codec_sums.npy` | embedding lookup 表错误、累加顺序 |
| vocoder | `audio.wav` | codec frame 排列、采样率 |

## 轻量 E2E 评估

不需要中间数据，只要 /tts 和 /asr API：

```bash
python tts_eval.py --host localhost:8000
```

输出 9 个 test case 的 CER + 时延表格。
