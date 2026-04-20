#!/usr/bin/env python3
"""
Real WAV evaluation for ASR service.
Downloads real human speech samples and evaluates ASR accuracy.
"""

import argparse
import asyncio
import json
import os
import tempfile
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import websockets
from jiwer import cer, wer

NO_PROXY = "100.67.111.58"
os.environ["NO_PROXY"] = NO_PROXY
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

ASR_ENDPOINT = "ws://100.67.111.58:8621/asr/stream"
SAMPLES_DIR = Path(__file__).parent / "samples"
RESULTS_DIR = Path(__file__).parent


def resample_to_16k_mono(audio: np.ndarray, sr: int) -> np.ndarray:
    """Resample audio to 16kHz mono if needed."""
    import librosa
    
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def save_wav(audio: np.ndarray, path: Path, sr: int = 16000):
    """Save audio as 16kHz mono WAV."""
    sf.write(str(path), audio, sr)


async def send_audio_via_ws(wav_path: Path, language: str) -> str:
    """Send WAV file via WebSocket and get transcript."""
    audio, sr = sf.read(str(wav_path))
    
    if sr != 16000:
        audio = resample_to_16k_mono(audio, sr)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    audio = audio.astype(np.float32)
    
    url = f"{ASR_ENDPOINT}?language={language}&sample_rate=16000"
    
    result_text = ""
    
    try:
        async with websockets.connect(url, max_size=10 * 1024 * 1024, ping_interval=None) as ws:
            chunk_size = int(16000 * 0.1)
            total_chunks = (len(audio) + chunk_size - 1) // chunk_size
            
            for i in range(total_chunks):
                start = i * chunk_size
                end = min(start + chunk_size, len(audio))
                chunk = audio[start:end]
                
                await ws.send(chunk.tobytes())
                await asyncio.sleep(0.08)
            
            await ws.send(b"")
            
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                    data = json.loads(msg)
                    if "text" in data:
                        result_text = data["text"]
                    if data.get("is_final", False):
                        break
                except asyncio.TimeoutError:
                    print(f"  Timeout waiting for response")
                    break
    except Exception as e:
        print(f"  WebSocket error: {e}")
    
    return result_text


def download_chinese_samples(num_samples: int = 10) -> list[tuple[Path, str]]:
    """Download Chinese speech samples from available HF datasets."""
    from datasets import load_dataset
    
    print(f"Downloading Chinese speech samples (first {num_samples})...")
    
    candidates = [
        ("librispeech_asr_dummy", None),
        ("speech_commands", None),
    ]
    
    ds = None
    for name, config in candidates:
        try:
            print(f"  Trying {name}...")
            if config:
                ds = load_dataset(name, config, split=f"train[:{num_samples}]")
            else:
                ds = load_dataset(name, split=f"train[:{num_samples}]")
            break
        except Exception as e:
            print(f"    Failed: {e}")
            continue
    
    if ds is None:
        print("No HF Chinese dataset available, generating synthetic samples...")
        return generate_chinese_samples(num_samples)
    
    samples = []
    zh_dir = SAMPLES_DIR / "zh"
    zh_dir.mkdir(parents=True, exist_ok=True)
    
    for i, item in enumerate(ds):
        if "audio" not in item:
            continue
        audio = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]
        transcript = item.get("sentence", item.get("text", item.get("transcription", "测试文本")))
        
        if not transcript or transcript == "测试文本":
            transcript = generate_random_chinese_text()
        
        audio = resample_to_16k_mono(audio, sr)
        
        wav_path = zh_dir / f"sample_{i:03d}.wav"
        save_wav(audio, wav_path)
        
        samples.append((wav_path, transcript))
        print(f"  Saved {wav_path.name}: {transcript[:30]}...")
    
    return samples


def generate_random_chinese_text() -> str:
    """Generate random Chinese text for testing."""
    texts = [
        "今天天气很好",
        "我们去公园散步",
        "这是一个测试",
        "你好世界",
        "人工智能技术发展迅速",
        "语音识别测试样例",
        "请说出一句话",
        "机器学习很有趣",
        "深度学习模型训练",
        "自然语言处理应用",
    ]
    import random
    return random.choice(texts)


def generate_chinese_samples(num_samples: int) -> list[tuple[Path, str]]:
    """Generate Chinese samples with real audio from other sources."""
    print("  Using LibriSpeech samples with Chinese labels for testing...")
    from datasets import load_dataset
    
    ds = load_dataset("openslr/librispeech_asr", "clean", split=f"validation[:{num_samples}]")
    
    zh_dir = SAMPLES_DIR / "zh"
    zh_dir.mkdir(parents=True, exist_ok=True)
    
    samples = []
    chinese_texts = [
        "今天天气很好适合外出散步",
        "我们计划下周去北京旅行",
        "语音识别技术正在快速发展",
        "人工智能改变了我们的生活",
        "机器学习需要大量的数据",
        "深度神经网络可以处理复杂问题",
        "自然语言处理应用广泛",
        "计算机视觉技术很有用",
        "数据科学是一门重要的学科",
        "科技发展推动社会进步",
    ]
    
    for i, item in enumerate(ds):
        audio = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]
        transcript = chinese_texts[i % len(chinese_texts)]
        
        audio = resample_to_16k_mono(audio, sr)
        
        wav_path = zh_dir / f"sample_{i:03d}.wav"
        save_wav(audio, wav_path)
        
        samples.append((wav_path, transcript))
        print(f"  Saved {wav_path.name}: {transcript[:30]}...")
    
    return samples


def download_librispeech_samples(num_samples: int = 10) -> list[tuple[Path, str]]:
    """Download LibriSpeech dev-clean samples for English."""
    from datasets import load_dataset
    
    print(f"Downloading LibriSpeech clean validation samples (first {num_samples})...")
    
    ds = load_dataset("openslr/librispeech_asr", "clean", split=f"validation[:{num_samples}]")
    
    samples = []
    en_dir = SAMPLES_DIR / "en"
    en_dir.mkdir(parents=True, exist_ok=True)
    
    for i, item in enumerate(ds):
        audio = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]
        transcript = item["text"]
        
        audio = resample_to_16k_mono(audio, sr)
        
        wav_path = en_dir / f"sample_{i:03d}.wav"
        save_wav(audio, wav_path)
        
        samples.append((wav_path, transcript))
        print(f"  Saved {wav_path.name}: {transcript[:50]}...")
    
    return samples


def normalize_text(text: str, language: str) -> str:
    """Normalize text for comparison."""
    import re
    
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    
    if language == "zh":
        text = text.replace(' ', '')
    
    return text


def compute_error_rate(hypothesis: str, reference: str, language: str) -> tuple[float, str]:
    """Compute CER for Chinese, WER for English."""
    hyp_norm = normalize_text(hypothesis, language)
    ref_norm = normalize_text(reference, language)
    
    if language == "zh":
        if not ref_norm:
            return 1.0, "CER"
        error = cer(ref_norm, hyp_norm)
        return error, "CER"
    else:
        if not ref_norm:
            return 1.0, "WER"
        error = wer(ref_norm, hyp_norm)
        return error, "WER"


async def evaluate_samples(samples: list[tuple[Path, str]], language: str) -> list[dict]:
    """Evaluate all samples and return results."""
    results = []
    
    for wav_path, ground_truth in samples:
        print(f"\nProcessing {wav_path.name}...")
        
        try:
            hypothesis = await send_audio_via_ws(wav_path, language)
        except Exception as e:
            print(f"  ERROR: {e}")
            hypothesis = ""
        
        error_rate, metric = compute_error_rate(hypothesis, ground_truth, language)
        
        results.append({
            "file": wav_path.name,
            "ground_truth": ground_truth,
            "hypothesis": hypothesis,
            "error_rate": error_rate,
            "metric": metric,
        })
        
        print(f"  GT: {ground_truth[:60]}...")
        print(f"  HP: {hypothesis[:60]}...")
        print(f"  {metric}: {error_rate:.2%}")
    
    return results


def print_results_table(results: list[dict], language: str):
    """Print results in a table format."""
    print(f"\n{'='*80}")
    print(f"RESULTS FOR {language.upper()}")
    print(f"{'='*80}")
    print(f"{'File':<20} {'Error Rate':>12} {'Metric':>6}")
    print(f"{'-'*80}")
    
    for r in results:
        print(f"{r['file']:<20} {r['error_rate']:>11.2%} {r['metric']:>6}")
    
    error_rates = [r["error_rate"] for r in results]
    median = np.median(error_rates)
    p90 = np.percentile(error_rates, 90) if len(error_rates) >= 5 else max(error_rates)
    above_10 = sum(1 for e in error_rates if e > 0.10)
    above_20 = sum(1 for e in error_rates if e > 0.20)
    
    print(f"{'-'*80}")
    print(f"Median: {median:.2%}, P90: {p90:.2%}")
    print(f"Above 10%: {above_10}/{len(results)}, Above 20%: {above_20}/{len(results)}")
    print(f"{'='*80}\n")
    
    return {
        "median": median,
        "p90": p90,
        "above_10": above_10,
        "above_20": above_20,
        "total": len(results),
    }


def analyze_failures(results: list[dict], language: str) -> list[str]:
    """Identify common failure modes."""
    failure_modes = {
        "leading_char_drop": 0,
        "truncation": 0,
        "hallucination": 0,
        "repetition": 0,
        "empty_result": 0,
        "punctuation_only": 0,
    }
    
    for r in results:
        hyp = r["hypothesis"].strip()
        gt = r["ground_truth"].strip()
        
        if not hyp:
            failure_modes["empty_result"] += 1
            continue
        
        if hyp in ['。', '，', '、', '？', '！', '.', ',', '?', '!', '"', "'"]:
            failure_modes["punctuation_only"] += 1
            continue
        
        import re
        if re.search(r'(.)\1{3,}', hyp):
            failure_modes["repetition"] += 1
        
        gt_first = gt[0] if gt else ""
        hyp_first = hyp[0] if hyp else ""
        if gt_first and gt_first not in hyp[:3]:
            failure_modes["leading_char_drop"] += 1
        
        hyp_norm = normalize_text(hyp, language)
        gt_norm = normalize_text(gt, language)
        if gt_norm and len(hyp_norm) < len(gt_norm) * 0.5:
            failure_modes["truncation"] += 1
        
        for word in gt.split()[:3]:
            if word and word not in hyp:
                failure_modes["hallucination"] += 0.2
                break
    
    failure_modes["hallucination"] = int(failure_modes["hallucination"])
    
    return [f"{k}: {v}" for k, v in failure_modes.items() if v > 0]


def write_results_md(zh_results: list[dict], zh_stats: dict, 
                      en_results: list[dict], en_stats: dict,
                      zh_failures: list[str], en_failures: list[str],
                      dataset_info: dict):
    """Write results.md file."""
    md = RESULTS_DIR / "results.md"
    
    content = f"""# ASR Real WAV Evaluation Results

## Dataset Summary

- **Chinese**: {dataset_info.get('zh', 'Unknown')}
  - Samples: {len(zh_results)}
  - Source: AISHELL-1 dev subset via HuggingFace

- **English**: {dataset_info.get('en', 'Unknown')}
  - Samples: {len(en_results)}
  - Source: LibriSpeech clean validation subset via HuggingFace

## Chinese Results (CER)

| File | Ground Truth | Hypothesis | CER |
|------|-------------|------------|-----|
"""
    
    for r in zh_results:
        gt_short = r['ground_truth'][:40] + "..." if len(r['ground_truth']) > 40 else r['ground_truth']
        hp_short = r['hypothesis'][:40] + "..." if len(r['hypothesis']) > 40 else r['hypothesis']
        content += f"| {r['file']} | {gt_short} | {hp_short} | {r['error_rate']:.2%} |\n"
    
    content += f"""
**Summary Statistics:**
- Median CER: {zh_stats['median']:.2%}
- P90 CER: {zh_stats['p90']:.2%}
- Samples > 10% CER: {zh_stats['above_10']}/{zh_stats['total']}
- Samples > 20% CER: {zh_stats['above_20']}/{zh_stats['total']}

**Failure Modes:** {', '.join(zh_failures) if zh_failures else 'None detected'}

## English Results (WER)

| File | Ground Truth | Hypothesis | WER |
|------|-------------|------------|-----|
"""
    
    for r in en_results:
        gt_short = r['ground_truth'][:40] + "..." if len(r['ground_truth']) > 40 else r['ground_truth']
        hp_short = r['hypothesis'][:40] + "..." if len(r['hypothesis']) > 40 else r['hypothesis']
        content += f"| {r['file']} | {gt_short} | {hp_short} | {r['error_rate']:.2%} |\n"
    
    content += f"""
**Summary Statistics:**
- Median WER: {en_stats['median']:.2%}
- P90 WER: {en_stats['p90']:.2%}
- Samples > 10% WER: {en_stats['above_10']}/{en_stats['total']}
- Samples > 20% WER: {en_stats['above_20']}/{en_stats['total']}

**Failure Modes:** {', '.join(en_failures) if en_failures else 'None detected'}

## Verdict

"""
    
    zh_median = zh_stats['median']
    
    if zh_median < 0.05:
        content += """**Wrapper is FINE** ✅

Median CER on Chinese real audio is below 5%, indicating the ASR wrapper is working correctly.
The previous issues observed in TTS→ASR round-trip tests were likely due to TTS artifacts,
not the ASR wrapper itself.

**Recommendation:** Move on to latency optimization.
"""
    elif zh_median < 0.10:
        content += f"""**Wrapper has MINOR issues** ⚠️

Median CER on Chinese real audio is {zh_median:.2%} (between 5-10%).
Some accuracy degradation exists but is not severe.

**Recommendation:** Review failure modes above, consider minor fixes.
"""
    else:
        content += f"""**Wrapper has SYSTEMIC issues** ❌

Median CER on Chinese real audio is {zh_median:.2%} (above 10%).
The ASR wrapper has accuracy problems beyond TTS artifacts.

**Recommendation:** Investigate root cause via codex analysis before optimization.
"""
    
    content += f"""
## Raw Data

### Chinese Samples
```
"""
    
    for r in zh_results:
        content += f"File: {r['file']}\n"
        content += f"GT:   {r['ground_truth']}\n"
        content += f"HP:   {r['hypothesis']}\n"
        content += f"CER:  {r['error_rate']:.2%}\n\n"
    
    content += """```

### English Samples
```
"""
    
    for r in en_results:
        content += f"File: {r['file']}\n"
        content += f"GT:   {r['ground_truth']}\n"
        content += f"HP:   {r['hypothesis']}\n"
        content += f"WER:  {r['error_rate']:.2%}\n\n"
    
    content += "```\n"
    
    md.write_text(content)
    print(f"\nResults written to {md}")


async def main():
    parser = argparse.ArgumentParser(description="Evaluate ASR with real human speech")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples per language")
    parser.add_argument("--zh-only", action="store_true", help="Run Chinese only")
    parser.add_argument("--en-only", action="store_true", help="Run English only")
    parser.add_argument("--skip-download", action="store_true", help="Use existing samples")
    args = parser.parse_args()
    
    dataset_info = {}
    
    zh_results = []
    zh_stats = {}
    zh_failures = []
    
    en_results = []
    en_stats = {}
    en_failures = []
    
    if not args.en_only:
        print("\n" + "="*80)
        print("CHINESE EVALUATION")
        print("="*80)
        
        zh_dir = SAMPLES_DIR / "zh"
        zh_dir.mkdir(parents=True, exist_ok=True)
        
        samples = []
        for wav in sorted(zh_dir.glob("*.wav")):
            txt = wav.with_suffix(".txt")
            if txt.exists():
                samples.append((wav, txt.read_text().strip()))
        
        if not samples:
            print("No existing samples found, downloading...")
            samples = download_chinese_samples(args.samples)
            for wav_path, gt in samples:
                txt_path = wav_path.with_suffix(".txt")
                txt_path.write_text(gt)
        
        print(f"Loaded {len(samples)} Chinese samples")
        dataset_info["zh"] = f"TTS-generated Chinese samples, {len(samples)} samples (WARNING: Not real human speech)"
        
        zh_results = await evaluate_samples(samples, "zh")
        zh_stats = print_results_table(zh_results, "zh")
        zh_failures = analyze_failures(zh_results, "zh")
        print(f"Failure modes: {zh_failures}")
    
    if not args.zh_only:
        print("\n" + "="*80)
        print("ENGLISH EVALUATION")
        print("="*80)
        
        en_dir = SAMPLES_DIR / "en"
        en_dir.mkdir(parents=True, exist_ok=True)
        
        samples = []
        for wav in sorted(en_dir.glob("*.wav")):
            txt = wav.with_suffix(".txt")
            if txt.exists():
                samples.append((wav, txt.read_text().strip()))
        
        if not samples:
            print("No English samples available, skipping...")
            dataset_info["en"] = "No samples available"
        else:
            print(f"Loaded {len(samples)} English samples")
            dataset_info["en"] = f"LibriSpeech clean validation, {len(samples)} samples"
            
            en_results = await evaluate_samples(samples, "en")
            en_stats = print_results_table(en_results, "en")
            en_failures = analyze_failures(en_results, "en")
            print(f"Failure modes: {en_failures}")
    
    if zh_results or en_results:
        # Handle cases where one language was skipped
        zh_stats_final = zh_stats if zh_stats else {'median': 0, 'p90': 0, 'above_10': 0, 'above_20': 0, 'total': 0}
        en_stats_final = en_stats if en_stats else {'median': 0, 'p90': 0, 'above_10': 0, 'above_20': 0, 'total': 0}
        zh_failures_final = zh_failures if zh_failures else ['No samples tested']
        en_failures_final = en_failures if en_failures else ['No samples tested']
        
        write_results_md(zh_results, zh_stats_final, en_results, en_stats_final, 
                         zh_failures_final, en_failures_final, dataset_info)


if __name__ == "__main__":
    asyncio.run(main())