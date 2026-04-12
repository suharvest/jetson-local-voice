"""
RKNN TTS Backend for RK3576

完整流程：
文本 → 文本前端 (sherpa-onnx, CPU) → tokens → Matcha RKNN (NPU) → mel → Vocos RKNN (NPU) → ISTFT (CPU) → 音频

性能：
- 文本前端: <10ms (CPU 查表)
- Matcha RKNN: ~170ms (NPU)
- Vocos RKNN: ~33ms (NPU, w4a16)
- ISTFT: ~25ms (CPU)
- 总计: ~230ms, RTF ~0.07
"""

from __future__ import annotations

import os
import time
import numpy as np
from pathlib import Path
from typing import Optional

# 音频参数
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
MAX_SEQ_LEN = int(os.environ.get('MATCHA_MAX_PHONEMES', '96'))
# RKNN model compiled input sequence length (must match the shape used during
# rknn.build()).  All tensor inputs to the Matcha RKNN model are padded to
# this length so that byte sizes match the static-shape expectations.
MATCHA_MODEL_SEQ_LEN = int(os.environ.get('MATCHA_MODEL_SEQ_LEN', '256'))


class RKNNMatchaVocoder:
    """RKNN 加速的 Matcha TTS 引擎"""

    def __init__(
        self,
        matcha_rknn_path: str,
        vocos_rknn_path: str,
        lexicon_path: str,
        tokens_path: str,
        data_dir: str,
    ):
        self.matcha_rknn_path = matcha_rknn_path
        self.vocos_rknn_path = vocos_rknn_path
        self.lexicon_path = lexicon_path
        self.tokens_path = tokens_path
        self.data_dir = data_dir

        # 加载后的模型
        self._matcha = None
        self._matcha_backend = None  # 'rknn' or 'ort'
        self._vocos = None
        self._lexicon = None
        self._token_to_id = None

    def load(self):
        """加载所有模型和资源"""
        import logging
        log = logging.getLogger(__name__)

        from rknnlite.api import RKNNLite

        # 加载 lexicon
        self._lexicon = {}
        with open(self.lexicon_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    self._lexicon[parts[0]] = parts[1:]

        # 加载 tokens
        self._token_to_id = {}
        with open(self.tokens_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) >= 1:
                    # token ID = 行号 + 1 (1-indexed)
                    self._token_to_id[parts[0]] = i + 1

        # 加载 Matcha 声学模型 — 优先 RKNN，回退 ORT (CPU)
        # Look for ONNX fallback: same name with .onnx, or model-steps-3.onnx
        # in the same directory or a well-known location.
        matcha_onnx_path = self.matcha_rknn_path.replace('.rknn', '.onnx')
        if not os.path.exists(matcha_onnx_path):
            # Try model-steps-3.onnx in same directory
            alt = os.path.join(os.path.dirname(self.matcha_rknn_path), 'model-steps-3.onnx')
            if os.path.exists(alt):
                matcha_onnx_path = alt
            else:
                # Try MATCHA_ONNX_PATH env override
                matcha_onnx_path = os.environ.get('MATCHA_ONNX_PATH', matcha_onnx_path)
        use_ort = os.environ.get('MATCHA_USE_ORT', '').lower() in ('1', 'true', 'yes')

        if not use_ort and os.path.exists(self.matcha_rknn_path):
            try:
                self._matcha = RKNNLite(verbose=False)
                ret = self._matcha.load_rknn(self.matcha_rknn_path)
                if ret != 0:
                    raise RuntimeError(f"load_rknn ret={ret}")
                ret = self._matcha.init_runtime(core_mask=1)
                if ret != 0:
                    raise RuntimeError(f"init_runtime ret={ret}")
                self._matcha_backend = 'rknn'
                log.info("Matcha acoustic model loaded via RKNN")
            except Exception as e:
                log.warning("Matcha RKNN load failed (%s), trying ORT fallback", e)
                if self._matcha is not None:
                    try:
                        self._matcha.release()
                    except Exception:
                        pass
                self._matcha = None

        if self._matcha is None and os.path.exists(matcha_onnx_path):
            import onnxruntime as ort
            self._matcha = ort.InferenceSession(
                matcha_onnx_path,
                providers=['CPUExecutionProvider'],
            )
            self._matcha_backend = 'ort'
            log.info("Matcha acoustic model loaded via ORT (CPU): %s", matcha_onnx_path)

        if self._matcha is None:
            raise RuntimeError(
                f"无法加载 Matcha 声学模型: RKNN={self.matcha_rknn_path}, "
                f"ONNX={matcha_onnx_path}"
            )

        # 加载 Vocos RKNN
        self._vocos = RKNNLite(verbose=False)
        ret = self._vocos.load_rknn(self.vocos_rknn_path)
        if ret != 0:
            raise RuntimeError(f"加载 Vocos RKNN 失败: ret={ret}")
        ret = self._vocos.init_runtime(core_mask=1)  # NPU_CORE_0 only; CORE_1 reserved for ASR encoder
        if ret != 0:
            raise RuntimeError(f"初始化 Vocos RKNN 运行时失败: ret={ret}")

    def release(self):
        """释放资源"""
        if self._matcha:
            if self._matcha_backend == 'rknn':
                try:
                    self._matcha.release()
                except Exception:
                    pass
            self._matcha = None
            self._matcha_backend = None
        if self._vocos:
            try:
                self._vocos.release()
            except:
                pass
            self._vocos = None

    def text_to_tokens(self, text: str) -> list[int]:
        """
        将文本转换为 token IDs

        使用 lexicon 查表：
        1. 中文字/词 → lexicon 查找 phonemes
        2. phonemes → token IDs
        """
        # 简化版文本处理：从 lexicon 中查找对应的 phonemes
        tokens = []

        # 处理文本中的每个字符/词
        i = 0
        while i < len(text):
            # 跳过标点符号
            if text[i] in '，。！？、；：""''（）':
                i += 1
                continue

            # 尝试匹配最长词
            found = False
            for length in range(min(4, len(text) - i), 0, -1):
                word = text[i:i+length]
                if word in self._lexicon:
                    phonemes = self._lexicon[word]
                    for p in phonemes:
                        if p in self._token_to_id:
                            tokens.append(self._token_to_id[p])
                    i += length
                    found = True
                    break

            if not found:
                # 单字处理
                char = text[i]
                if char in self._lexicon:
                    phonemes = self._lexicon[char]
                    for p in phonemes:
                        if p in self._token_to_id:
                            tokens.append(self._token_to_id[p])
                i += 1

        return tokens

    def run_matcha(
        self,
        tokens: list[int],
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        """
        运行 Matcha RKNN 声学模型

        Args:
            tokens: 音素 token IDs
            noise_scale: 噪声缩放因子
            length_scale: 时长缩放因子

        Returns:
            mel: Mel 频谱图 [1, 80, T]
            mel_frames: 有效帧数
        """
        num_tokens = len(tokens)
        x_length = np.array([num_tokens], dtype=np.int64)
        noise_scale_arr = np.array([noise_scale], dtype=np.float32)
        length_scale_arr = np.array([length_scale], dtype=np.float32)

        if self._matcha_backend == 'rknn':
            # Pad tokens to the RKNN model's compiled static input length.
            # MAX_SEQ_LEN caps the phoneme count; MATCHA_MODEL_SEQ_LEN is the
            # tensor dimension the RKNN model was built with (must match exactly).
            tokens_padded = np.zeros((1, MATCHA_MODEL_SEQ_LEN), dtype=np.int64)
            tokens_padded[0, :num_tokens] = tokens
            mel = self._matcha.inference(
                inputs=[tokens_padded, x_length, noise_scale_arr, length_scale_arr]
            )[0]
        else:
            # ORT fallback — dynamic shapes, no padding needed
            tokens_padded = np.zeros((1, max(num_tokens, 1)), dtype=np.int64)
            tokens_padded[0, :num_tokens] = tokens
            mel = self._matcha.run(
                None,
                {
                    'x': tokens_padded,
                    'x_length': x_length,
                    'noise_scale': noise_scale_arr,
                    'length_scale': length_scale_arr,
                },
            )[0]

        # 计算有效帧数
        mel_frames = int(num_tokens * 30 * length_scale + 0.5)
        mel_frames = min(mel_frames, mel.shape[2])

        return mel, mel_frames

    def run_vocos(self, mel: np.ndarray, mel_frames: int) -> np.ndarray:
        """
        运行 Vocos RKNN 声码器

        Args:
            mel: Mel 频谱图 [1, 80, T]
            mel_frames: 有效帧数

        Returns:
            audio: 音频样本
        """
        # Pad mel (Vocos 需要固定输入大小)
        mel_padded = np.zeros((1, 80, 256), dtype=np.float32)
        mel_padded[:, :, :min(mel_frames, 256)] = mel[:, :, :min(mel_frames, 256)]

        # 推理
        outputs = self._vocos.inference(inputs=[mel_padded])

        # 提取 STFT 分量
        mag = outputs[0][0]  # [513, T]
        x = outputs[1][0]    # cos 分量
        y = outputs[2][0]    # sin 分量

        # ISTFT
        audio = self._istft(mag, x, y)

        # 裁剪到正确长度
        audio = audio[:mel_frames * HOP_LENGTH]

        return audio

    def _istft(
        self,
        mag: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        逆短时傅里叶变换

        Args:
            mag: 幅度谱 [513, T]
            x: 余弦分量 (实部)
            y: 正弦分量 (虚部)

        Returns:
            audio: 重建的音频
        """
        # 重建复数频谱
        complex_spec = mag * (x + 1j * y)

        n_frames = complex_spec.shape[1]
        output_len = (n_frames - 1) * HOP_LENGTH + N_FFT

        audio = np.zeros(output_len, dtype=np.float32)
        window = np.hanning(N_FFT)

        # 重叠相加
        for i in range(n_frames):
            frame = np.fft.irfft(complex_spec[:, i], n=N_FFT) * window
            start = i * HOP_LENGTH
            audio[start:start + N_FFT] += frame

        # 归一化
        window_sum = np.zeros(output_len, dtype=np.float32)
        for i in range(n_frames):
            start = i * HOP_LENGTH
            window_sum[start:start + N_FFT] += window ** 2

        audio = audio / np.maximum(window_sum, 1e-8)

        return audio

    def _split_text(self, text: str) -> list[str]:
        """将文本按句子分割，确保每段不超过 MAX_SEQ_LEN 个音素。"""
        import re
        # 按句末标点分割
        segments = re.split(r'([。！？；!?;])', text)
        # 将标点重新附加到前一段
        result = []
        for i in range(0, len(segments), 2):
            seg = segments[i]
            if i + 1 < len(segments):
                seg += segments[i + 1]
            seg = seg.strip()
            if seg:
                result.append(seg)
        if not result:
            return [text]

        # 对仍超出 MAX_SEQ_LEN 的段，按逗号进一步拆分
        final = []
        for seg in result:
            tokens = self.text_to_tokens(seg)
            if len(tokens) <= MAX_SEQ_LEN:
                final.append(seg)
            else:
                # 按逗号拆分
                sub_segs = re.split(r'([，,])', seg)
                sub_result = []
                for j in range(0, len(sub_segs), 2):
                    s = sub_segs[j]
                    if j + 1 < len(sub_segs):
                        s += sub_segs[j + 1]
                    s = s.strip()
                    if s:
                        sub_result.append(s)
                final.extend(sub_result if sub_result else [seg])
        return final

    def _synthesize_segment(
        self,
        text: str,
        speed: float = 1.0,
        noise_scale: float = 0.667,
    ) -> tuple[np.ndarray, dict]:
        """合成单个文本段（不超过 MAX_SEQ_LEN 音素）。"""
        metadata = {}

        # Step 1: 文本 → tokens
        t0 = time.perf_counter()
        tokens = self.text_to_tokens(text)
        metadata['text_frontend_ms'] = (time.perf_counter() - t0) * 1000
        metadata['num_tokens'] = len(tokens)

        if len(tokens) == 0:
            return np.zeros(0, dtype=np.float32), metadata

        # 截断超长 tokens
        tokens = tokens[:MAX_SEQ_LEN]

        # Step 2: Matcha RKNN
        t0 = time.perf_counter()
        mel, mel_frames = self.run_matcha(tokens, noise_scale, 1.0 / speed)
        metadata['matcha_ms'] = (time.perf_counter() - t0) * 1000

        # Step 3: Vocos RKNN
        t0 = time.perf_counter()
        audio = self.run_vocos(mel, mel_frames)
        metadata['vocos_ms'] = (time.perf_counter() - t0) * 1000

        metadata['duration_s'] = len(audio) / SAMPLE_RATE
        metadata['total_ms'] = sum(v for k, v in metadata.items() if k.endswith('_ms'))
        if metadata['duration_s'] > 0:
            metadata['rtf'] = metadata['total_ms'] / 1000 / metadata['duration_s']

        return audio.astype(np.float32), metadata

    def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        noise_scale: float = 0.667,
    ) -> tuple[np.ndarray, dict]:
        """
        合成语音，自动分句处理超长文本

        Args:
            text: 输入文本 (中文)
            speed: 语速 (1.0 = 正常)
            noise_scale: 噪声强度

        Returns:
            audio: 音频样本 (float32, [-1, 1])
            metadata: 元数据 (耗时等)
        """
        segments = self._split_text(text)
        all_audio = []
        total_text_frontend_ms = 0.0
        total_matcha_ms = 0.0
        total_vocos_ms = 0.0
        total_num_tokens = 0

        for seg in segments:
            audio_seg, meta_seg = self._synthesize_segment(seg, speed, noise_scale)
            if len(audio_seg) > 0:
                all_audio.append(audio_seg)
            total_text_frontend_ms += meta_seg.get('text_frontend_ms', 0.0)
            total_matcha_ms += meta_seg.get('matcha_ms', 0.0)
            total_vocos_ms += meta_seg.get('vocos_ms', 0.0)
            total_num_tokens += meta_seg.get('num_tokens', 0)

        audio = np.concatenate(all_audio) if all_audio else np.zeros(0, dtype=np.float32)

        # 归一化
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max() * 0.95

        metadata = {
            'num_tokens': total_num_tokens,
            'text_frontend_ms': total_text_frontend_ms,
            'matcha_ms': total_matcha_ms,
            'vocos_ms': total_vocos_ms,
        }
        metadata['duration_s'] = len(audio) / SAMPLE_RATE
        metadata['total_ms'] = sum(v for k, v in metadata.items() if k.endswith('_ms'))
        if metadata['duration_s'] > 0:
            metadata['rtf'] = metadata['total_ms'] / 1000 / metadata['duration_s']

        return audio.astype(np.float32), metadata


def create_rknn_tts_backend(model_dir: str = None) -> RKNNMatchaVocoder:
    """
    创建 RKNN TTS 后端

    Args:
        model_dir: 模型目录，默认从环境变量获取
    """
    if model_dir is None:
        model_dir = os.environ.get('TTS_MODEL_DIR', '/home/cat/models')

    model_dir = Path(model_dir)

    return RKNNMatchaVocoder(
        matcha_rknn_path=str(model_dir / 'matcha-zh-en.rknn'),
        vocos_rknn_path=str(model_dir / 'vocos-16khz-univ-w4a16.rknn'),
        lexicon_path=str(model_dir / 'matcha-icefall-zh-en' / 'lexicon.txt'),
        tokens_path=str(model_dir / 'matcha-icefall-zh-en' / 'tokens.txt'),
        data_dir=str(model_dir / 'matcha-icefall-zh-en' / 'espeak-ng-data'),
    )


class MatchaRKNNBackend:
    """TTSBackend wrapper around RKNNMatchaVocoder.

    Select via TTS_BACKEND=matcha_rknn.

    Note: intentionally duck-typed (not inheriting TTSBackend) to avoid
    importing tts_backend at module level. The synthesize_stream() fallback
    is provided explicitly below.
    """

    def __init__(self) -> None:
        self._engine: Optional[RKNNMatchaVocoder] = None

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "matcha_rknn"

    def is_ready(self) -> bool:
        """Return True if the engine is loaded and ready."""
        return self._engine is not None and self._engine._matcha is not None

    def preload(self) -> None:
        """Create and load RKNNMatchaVocoder. Called once at startup."""
        self._engine = create_rknn_tts_backend()
        self._engine.load()

    def get_sample_rate(self) -> int:
        """Return audio sample rate in Hz."""
        return SAMPLE_RATE

    def synthesize(
        self,
        text: str,
        speaker_id: int = 0,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        """Synthesize text to WAV bytes.

        Args:
            text: Input text (Chinese / mixed Chinese-English).
            speaker_id: Ignored (Matcha model has a single speaker).
            speed: Speech rate multiplier (1.0 = normal). Defaults to 1.0.
            pitch_shift: Ignored (not supported by this backend).
            **kwargs: Forwarded to engine.synthesize() (e.g. noise_scale).

        Returns:
            wav_bytes: PCM audio encoded as a WAV file.
            metadata: Dict with keys ``duration``, ``inference_time``, ``rtf``
                      plus per-stage timing from the engine.
        """
        import io
        import soundfile as sf

        if self._engine is None:
            raise RuntimeError("MatchaRKNNBackend.preload() has not been called")

        t_start = time.perf_counter()
        audio, engine_meta = self._engine.synthesize(
            text,
            speed=speed if speed is not None else 1.0,
            **{k: v for k, v in kwargs.items() if k in ("noise_scale",)},
        )
        inference_time = time.perf_counter() - t_start

        # Encode float32 audio → WAV bytes
        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()

        duration = engine_meta.get("duration_s", len(audio) / SAMPLE_RATE)
        rtf = engine_meta.get("rtf", inference_time / duration if duration > 0 else 0.0)

        metadata = {
            "duration": duration,
            "inference_time": inference_time,
            "rtf": rtf,
            **engine_meta,
        }
        return wav_bytes, metadata

    def synthesize_stream(self, text, speaker_id=0, speed=None, pitch_shift=None, **kwargs):
        """Yield (audio_float32_chunk, metadata). Non-streaming fallback."""
        import io as _io
        import soundfile as sf

        wav_bytes, meta = self.synthesize(
            text=text, speaker_id=speaker_id, speed=speed,
            pitch_shift=pitch_shift, **kwargs,
        )
        buf = _io.BytesIO(wav_bytes)
        audio, _ = sf.read(buf, dtype="float32")
        yield audio, meta

    def cleanup(self) -> None:
        """Release RKNN resources."""
        if self._engine is not None:
            self._engine.release()
            self._engine = None


# 命令行测试
if __name__ == '__main__':
    import argparse
    import soundfile as sf

    parser = argparse.ArgumentParser(description='RKNN TTS 测试')
    parser.add_argument('--text', '-t', default='你好世界', help='输入文本')
    parser.add_argument('--output', '-o', default='/tmp/rknn_tts.wav', help='输出文件')
    parser.add_argument('--speed', '-s', type=float, default=1.0, help='语速')
    args = parser.parse_args()

    print(f"输入: {args.text}")
    print("\n加载模型...")

    engine = create_rknn_tts_backend()
    engine.load()
    print("模型加载完成")

    print("\n合成中...")
    audio, meta = engine.synthesize(args.text, speed=args.speed)

    print(f"\n结果:")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    if len(audio) > 0:
        sf.write(args.output, audio, SAMPLE_RATE)
        print(f"\n保存: {args.output}")

    engine.release()