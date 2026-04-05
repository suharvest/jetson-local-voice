"""Qwen3-TTS inference service for RK3576.

Pipeline:
  1. Tokenize text (transformers tokenizer)
  2. text_project (RKNN NPU) -> text embeddings [T, 1024]
  3. Build prefill embeddings (numpy element-wise add)
  4. talker prefill (RKLLM, mode=1) -> hidden states
  5. logits = hidden @ codec_head (numpy matmul)
  6. AR loop:
     a. sample primary token from logits
     b. code_predictor (RKNN NPU) -> residual codes
     c. build next embedding (numpy: sum 16 codebook lookups + text embed)
     d. talker decode (RKLLM, mode=1, keep_history=1) -> hidden
     e. logits = hidden @ codec_head
  7. vocoder (RKNN NPU, decoder_ctx25_int8) -> audio waveform
  8. Return WAV
"""

from __future__ import annotations

import io
import logging
import os
import time
from typing import Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────

HIDDEN_SIZE = 1024
NUM_CODE_GROUPS = 16
TALKER_VOCAB_SIZE = 3072
CODE_PREDICTOR_VOCAB_SIZE = 2048
SAMPLE_RATE = 12000  # 12 Hz codec * 1000 samples/frame

# Special token IDs (from Qwen3-TTS config)
TTS_BOS_TOKEN_ID = 151672
TTS_EOS_TOKEN_ID = 151673
TTS_PAD_TOKEN_ID = 151671

CODEC_BOS_ID = 2149
CODEC_EOS_TOKEN_ID = 2150
CODEC_PAD_ID = 2148
CODEC_NOTHINK_ID = 2155
CODEC_THINK_BOS_ID = 2156
CODEC_THINK_EOS_ID = 2157

# Vocoder params
VOCODER_CTX_FRAMES = 25  # context frames for streaming decoder
SAMPLES_PER_FRAME = 1000  # 12kHz sample rate / 12 Hz frame rate


class TTSService:
    """Full Qwen3-TTS pipeline on RK3576."""

    def __init__(self, model_dir: str):
        self._model_dir = model_dir
        self._ready = False

        self._tokenizer = None
        self._talker = None
        self._text_project = None
        self._codec_embed = None
        self._code_predictor = None
        self._code_predictor_embed = None
        self._vocoder = None
        self._codec_head_weight = None
        self._codebook_embeds = None

    def load(self):
        """Load all models. Call once at startup."""
        t0 = time.perf_counter()

        self._load_tokenizer()
        self._load_rknn_models()
        self._load_rkllm_talker()
        self._load_numpy_weights()

        elapsed = time.perf_counter() - t0
        logger.info("All models loaded in %.1fs", elapsed)
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def get_sample_rate(self) -> int:
        return SAMPLE_RATE

    # ── Model Loading ────────────────────────────────────────────

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        tok_path = os.path.join(self._model_dir, "tokenizer")
        logger.info("Loading tokenizer from %s", tok_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            tok_path, trust_remote_code=True
        )

    def _load_rknn_models(self):
        from rknnlite.api import RKNNLite

        def load_rknn(name: str) -> RKNNLite:
            path = os.path.join(self._model_dir, f"{name}.rknn")
            logger.info("Loading RKNN: %s", path)
            rknn = RKNNLite(verbose=False)
            ret = rknn.load_rknn(path)
            if ret != 0:
                raise RuntimeError(f"Failed to load RKNN {path}: ret={ret}")
            ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
            if ret != 0:
                raise RuntimeError(f"Failed to init RKNN runtime {name}: ret={ret}")
            return rknn

        self._text_project = load_rknn("text_project")
        self._codec_embed = load_rknn("codec_embed")
        self._code_predictor = load_rknn("code_predictor")
        self._code_predictor_embed = load_rknn("code_predictor_embed")

        # Vocoder: prefer tokenizer12hz_decode_stream (takes codes directly)
        # Fallback to decoder_ctx25_int8 (takes pre-computed embeddings)
        stream_path = os.path.join(self._model_dir, "tokenizer12hz_decode_stream.rknn")
        if os.path.exists(stream_path):
            self._vocoder = load_rknn("tokenizer12hz_decode_stream")
            self._vocoder_type = "stream"  # input: [1, 75, 16] int64
            logger.info("Using stream vocoder (direct codes input)")
        else:
            self._vocoder = load_rknn("decoder_ctx25_int8")
            self._vocoder_type = "noembed"  # input: [1, 512, 50] float32
            logger.info("Using noembed vocoder (pre-computed embeddings)")

    def _load_rkllm_talker(self):
        from rkllm_wrapper import RKLLMTalker

        rkllm_path = os.path.join(self._model_dir, "talker_fullvocab_fixed_w4a16_rk3576.rkllm")
        self._talker = RKLLMTalker(
            model_path=rkllm_path,
            rkllm_lib="/usr/lib/librkllmrt.so",
            rknn_lib="librknnrt.so",
            max_context_len=512,
            max_new_tokens=1,  # Step-by-step: 1 token per call
        )

    def _load_numpy_weights(self):
        logger.info("Loading numpy weights...")

        # codec_head: [3072, 1024] for logits = hidden @ codec_head.T
        ch_path = os.path.join(self._model_dir, "codec_head_weight.npy")
        self._codec_head_weight = np.load(ch_path).astype(np.float32)
        logger.info("codec_head: %s", self._codec_head_weight.shape)

        # codebook embeddings: 16 codebooks of [2048, 256]
        cb_dir = os.path.join(self._model_dir, "codebook_embeds")
        self._codebook_embeds = []
        for i in range(NUM_CODE_GROUPS):
            cb = np.load(os.path.join(cb_dir, f"codebook_{i}.npy")).astype(np.float32)
            self._codebook_embeds.append(cb)
        logger.info("Loaded %d codebook embeddings: %s each", len(self._codebook_embeds), self._codebook_embeds[0].shape)

        # output_proj weights for code_predictor logit computation
        self._output_proj_first = np.load(os.path.join(cb_dir, "output_proj_first_weight.npy")).astype(np.float32)
        self._output_proj_rest = np.load(os.path.join(cb_dir, "output_proj_rest_weight.npy")).astype(np.float32)

    # ── RKNN Helpers ─────────────────────────────────────────────

    def _run_text_project(self, token_ids: list[int]) -> np.ndarray:
        """Run text_project RKNN: [1, 128] int64 -> [1, 128, 1024] float32.
        Pads/truncates to 128 tokens. Returns [N, 1024] where N = len(token_ids).
        """
        n = len(token_ids)
        padded = np.zeros((1, 128), dtype=np.int64)
        use_n = min(n, 128)
        padded[0, :use_n] = token_ids[:use_n]
        outputs = self._text_project.inference(inputs=[padded])
        result = np.array(outputs[0])  # [1, 128, 1024]
        return result[0, :use_n]  # [N, 1024]

    def _run_codec_embed(self, codec_ids: list[int]) -> np.ndarray:
        """Run codec_embed RKNN: [1, 1] int64 -> [1, 1, 1024].
        Processes one token at a time. Returns [N, 1024].
        """
        results = []
        for cid in codec_ids:
            inp = np.array([[cid]], dtype=np.int64)
            out = self._codec_embed.inference(inputs=[inp])
            results.append(np.array(out[0])[0, 0])  # [1024]
        return np.stack(results)  # [N, 1024]

    def _run_code_predictor(self, context: np.ndarray) -> np.ndarray:
        """Run code_predictor RKNN.
        Args:
            context: [1, 2, 1024] float32 (fixed 2-token input for static RKNN model)
        Returns: logits [1, 1, 2048] float32
        """
        outputs = self._code_predictor.inference(inputs=[context])
        return np.array(outputs[0])  # [1, 1, 2048]

    def _run_code_predictor_embed(self, code_id: int, gen_step: int) -> np.ndarray:
        """Run code_predictor_embed RKNN.
        Returns: [1024] float32 embedding for residual code.
        """
        inp_id = np.array([[code_id]], dtype=np.int64)
        gs = np.array([gen_step], dtype=np.int64)
        outputs = self._code_predictor_embed.inference(inputs=[inp_id, gs])
        return np.array(outputs[0])[0, 0]  # [1024]

    def _run_vocoder_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Run vocoder RKNN (decoder_ctx25_int8) on pre-computed embeddings.
        Args:
            embeddings: [1, 512, T] float32
        Returns: audio samples float32
        """
        outputs = self._vocoder.inference(inputs=[embeddings])
        return np.array(outputs[0]).flatten()

    # ── Sampling ─────────────────────────────────────────────────

    @staticmethod
    def _sample_top_k(logits: np.ndarray, top_k: int = 5, temperature: float = 0.8) -> int:
        """Sample from logits with top-k and temperature."""
        if temperature <= 0 or top_k <= 1:
            return int(np.argmax(logits))

        logits = logits / temperature
        # Top-k filtering
        top_k_idx = np.argpartition(logits, -top_k)[-top_k:]
        top_k_logits = logits[top_k_idx]
        # Softmax
        top_k_logits -= top_k_logits.max()
        probs = np.exp(top_k_logits)
        probs /= probs.sum()
        chosen = np.random.choice(top_k_idx, p=probs)
        return int(chosen)

    # ── Main Synthesis ───────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        speaker_id: int = 0,
        speed: float = 1.0,
        temperature: float = 0.8,
        top_k: int = 5,
        max_new_tokens: int = 300,
    ) -> tuple[bytes, dict]:
        """Synthesize speech from text.

        Returns: (wav_bytes, metadata_dict)
        """
        t_start = time.perf_counter()

        # Step 1: Tokenize
        formatted_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = self._tokenizer.encode(formatted_text)
        logger.info("Tokenized: %d tokens", len(input_ids))

        # Step 2: Build prefill embeddings
        prefill_embeds, trailing_text, tts_pad_vec = self._build_prefill(input_ids)
        t_prefill_built = time.perf_counter()
        logger.info("Prefill embeddings: %s (%.0fms)", prefill_embeds.shape,
                     (t_prefill_built - t_start) * 1000)

        # Step 3: Talker prefill
        # keep_history=0: KV cache from this call persists for subsequent calls
        # (clear_kv_cache was called above, so this starts fresh)
        self._talker.clear_kv_cache()
        result = self._talker.run_embed(
            prefill_embeds,
            mode=1,  # GET_LAST_HIDDEN_LAYER -> returns hidden states
            keep_history=0,
        )
        hidden = result["hidden"]  # [n_prefill, 1024]
        last_hidden = hidden[-1:]  # [1, 1024]

        # Step 4: Compute initial logits
        logits = (last_hidden @ self._codec_head_weight.T)[0]  # [3072]
        t_prefill_done = time.perf_counter()
        logger.info("Prefill done: %.0fms", (t_prefill_done - t_prefill_built) * 1000)

        # Step 5: AR generate loop
        all_codes = []
        min_new_tokens = 2

        for step in range(max_new_tokens):
            # 5a: Sample primary code
            # Suppress tokens in [talker_vocab_size-1024, talker_vocab_size) for first tokens
            if step < min_new_tokens:
                logits[CODEC_EOS_TOKEN_ID] = -float("inf")

            primary_code = self._sample_top_k(logits, top_k=top_k, temperature=temperature)

            if primary_code == CODEC_EOS_TOKEN_ID:
                logger.info("EOS at step %d", step)
                break

            # 5b: Get primary embedding via codec_embed RKNN
            primary_embed = self._run_codec_embed([primary_code])  # [1, 1024]

            # 5c: Residual codes via code_predictor
            # The RKNN code_predictor has static shape [1, 2, 1024].
            # For each residual step, pass [last_hidden, latest_embed].
            frame_codes = [primary_code]
            codec_sum = primary_embed[0].copy()  # [1024]
            latest_embed = primary_embed[0]  # [1024]

            for j in range(NUM_CODE_GROUPS - 1):
                # Build 2-token input: [last_hidden, latest_embed]
                cp_input = np.stack([last_hidden[0], latest_embed])[np.newaxis, :, :]  # [1, 2, 1024]
                cp_logits = self._run_code_predictor(cp_input)
                # Output is [1, 1, 2048]
                cp_logits_last = cp_logits[0, 0]

                res_code = self._sample_top_k(
                    cp_logits_last[:CODE_PREDICTOR_VOCAB_SIZE],
                    top_k=1, temperature=0.0  # greedy for residual codes
                )
                frame_codes.append(res_code)

                # Get residual embedding for next iteration
                res_embed = self._run_code_predictor_embed(res_code, j)  # [1024]
                latest_embed = res_embed
                codec_sum += res_embed

            all_codes.append(frame_codes)

            # 5d: Build next talker input
            if step < len(trailing_text):
                txt_hidden = trailing_text[step]
            else:
                txt_hidden = tts_pad_vec

            next_embed = (codec_sum + txt_hidden)[np.newaxis, :]  # [1, 1024]

            # 5e: Talker decode step
            # keep_history=0: KV cache persists from previous calls
            result = self._talker.run_embed(
                next_embed,
                mode=1,  # GET_LAST_HIDDEN_LAYER -> returns hidden states
                keep_history=0,
            )
            last_hidden = result["hidden"][-1:]  # [1, 1024]
            logits = (last_hidden @ self._codec_head_weight.T)[0]  # [3072]

            if (step + 1) % 50 == 0:
                logger.info("Generated %d frames", step + 1)

        t_ar_done = time.perf_counter()
        n_frames = len(all_codes)
        ar_time = t_ar_done - t_prefill_done
        logger.info("AR loop: %d frames in %.1fs (%.1f frames/s)",
                     n_frames, ar_time, n_frames / ar_time if ar_time > 0 else 0)

        if n_frames == 0:
            # Return silence
            wav_bytes = self._make_wav(np.zeros(SAMPLE_RATE, dtype=np.float32))
            return wav_bytes, {"duration": 1.0, "inference_time": 0, "rtf": 0}

        # Step 6: Vocoder (codes -> embeddings -> audio)
        codes_array = np.array(all_codes, dtype=np.int64)  # [T, 16]
        audio = self._decode_audio(codes_array)
        t_vocoder_done = time.perf_counter()
        logger.info("Vocoder: %.1fs", t_vocoder_done - t_ar_done)

        # Apply speed adjustment
        if speed != 1.0 and speed > 0:
            # Simple speed via resampling
            n_orig = len(audio)
            n_new = int(n_orig / speed)
            indices = np.linspace(0, n_orig - 1, n_new)
            audio = np.interp(indices, np.arange(n_orig), audio).astype(np.float32)

        # Step 7: Make WAV
        duration = len(audio) / SAMPLE_RATE
        total_time = time.perf_counter() - t_start
        rtf = total_time / duration if duration > 0 else 0

        wav_bytes = self._make_wav(audio)

        meta = {
            "duration": round(duration, 3),
            "inference_time": round(total_time, 3),
            "rtf": round(rtf, 3),
            "frames": n_frames,
            "ar_time": round(ar_time, 3),
        }
        logger.info("TTS done: %.1fs audio in %.1fs (RTF=%.2f)", duration, total_time, rtf)
        return wav_bytes, meta

    # ── Prefill Construction ─────────────────────────────────────

    def _build_prefill(self, input_ids: list[int]):
        """Build prefill embeddings following sherpa-onnx layout.

        Prefill layout (8 positions):
          [role(3)] + [tts_pad+nothink, tts_pad+think_bos, tts_pad+think_eos,
                       tts_bos+pad] + [text[3]+codec_bos]

        Returns: (prefill_embeds [8, 1024], trailing_text list[ndarray], tts_pad_vec ndarray)
        """
        # Special embeddings
        special_ids = [TTS_BOS_TOKEN_ID, TTS_EOS_TOKEN_ID, TTS_PAD_TOKEN_ID]
        special_embed = self._run_text_project(special_ids)  # [3, 1024]
        tts_bos_embed = special_embed[0]  # [1024]
        tts_eos_embed = special_embed[1]  # [1024]
        tts_pad_embed = special_embed[2]  # [1024]

        # Role embeddings (first 3 tokens: <|im_start|> assistant \n)
        role_ids = input_ids[:3]
        role_embed = self._run_text_project(role_ids)  # [3, 1024]

        # Codec prefix embeddings
        codec_prefix_ids = [
            CODEC_NOTHINK_ID, CODEC_THINK_BOS_ID, CODEC_THINK_EOS_ID,
            CODEC_PAD_ID, CODEC_BOS_ID,
        ]
        codec_prefix_embed = self._run_codec_embed(codec_prefix_ids)  # [5, 1024]

        # Body text: tokens [3:-5] (between role and trailing special tokens)
        text_start = 3
        text_end = len(input_ids) - 5
        body_text_ids = input_ids[text_start:text_end] if text_end > text_start else []

        # Build prefill: 8 positions
        D = HIDDEN_SIZE
        prefill = np.zeros((8, D), dtype=np.float32)

        # [0..2]: role embeddings
        prefill[:3] = role_embed

        # [3..5]: tts_pad + codec_prefix[0..2] (nothink, think_bos, think_eos)
        prefill[3] = tts_pad_embed + codec_prefix_embed[0]  # tts_pad + nothink
        prefill[4] = tts_pad_embed + codec_prefix_embed[1]  # tts_pad + think_bos
        prefill[5] = tts_pad_embed + codec_prefix_embed[2]  # tts_pad + think_eos

        # [6]: tts_bos + codec_pad
        prefill[6] = tts_bos_embed + codec_prefix_embed[3]  # tts_bos + pad

        # [7]: text[first_body_token] + codec_bos
        if body_text_ids:
            first_body_embed = self._run_text_project([body_text_ids[0]])  # [1, 1024]
            prefill[7] = first_body_embed[0] + codec_prefix_embed[4]  # text[0] + codec_bos
        else:
            prefill[7] = tts_pad_embed + codec_prefix_embed[4]

        # Trailing text hidden: remaining body text tokens + tts_eos
        trailing = []
        if len(body_text_ids) > 1:
            trail_ids = body_text_ids[1:]
            trail_embed = self._run_text_project(trail_ids)  # [N, 1024]
            for i in range(len(trail_ids)):
                trailing.append(trail_embed[i])
        trailing.append(tts_eos_embed)

        return prefill, trailing, tts_pad_embed

    # ── Audio Decode ─────────────────────────────────────────────

    def _codes_to_embeddings(self, codes: np.ndarray) -> np.ndarray:
        """Convert discrete codec codes to continuous embeddings for the vocoder.

        Args:
            codes: [T, 16] int64 - T frames, 16 codebooks

        Returns: [512, T] float32 - vocoder input format

        Each frame's embedding is computed as:
            embed = sum(codebook_i[code_i] @ output_proj_i.T for i in range(16))
        where output_proj_first is used for codebook 0 and output_proj_rest for 1-15.
        """
        T = codes.shape[0]
        embeddings = np.zeros((T, 512), dtype=np.float32)

        for t in range(T):
            for cb_idx in range(NUM_CODE_GROUPS):
                code = int(codes[t, cb_idx])
                cb_embed = self._codebook_embeds[cb_idx][code]  # [256]
                if cb_idx == 0:
                    proj = cb_embed @ self._output_proj_first.T  # [256] @ [256, 512] -> [512]
                else:
                    proj = cb_embed @ self._output_proj_rest.T  # [256] @ [256, 512] -> [512]
                embeddings[t] += proj

        return embeddings.T  # [512, T]

    def _decode_audio(self, codes_array: np.ndarray) -> np.ndarray:
        """Decode codec codes to audio using vocoder.

        Args:
            codes_array: [T, 16] int64
        """
        total_frames = codes_array.shape[0]
        if total_frames == 0:
            return np.zeros(0, dtype=np.float32)

        if self._vocoder_type == "stream":
            return self._decode_audio_stream(codes_array)
        else:
            return self._decode_audio_noembed(codes_array)

    def _decode_audio_stream(self, codes_array: np.ndarray) -> np.ndarray:
        """Decode using tokenizer12hz_decode_stream.rknn (takes [1, 75, 16] int64 codes).

        Decodes in chunks of 25 frames with 50-frame context.
        """
        total_frames = codes_array.shape[0]
        chunk_size = 25
        ctx_size = 50
        total_T = ctx_size + chunk_size  # 75 (fixed input size)
        audio_chunks = []

        for start in range(0, total_frames, chunk_size):
            end = min(start + chunk_size, total_frames)
            actual_chunk = end - start
            ctx_start = max(0, start - ctx_size)

            # Get context + chunk codes
            chunk_codes = codes_array[ctx_start:end]  # [ctx+chunk, 16]

            # Pad to fixed size [75, 16]
            padded = np.zeros((total_T, 16), dtype=np.int64)
            padded[:chunk_codes.shape[0]] = chunk_codes

            # Run vocoder
            vocoder_input = padded[np.newaxis, :, :]  # [1, 75, 16]
            outputs = self._vocoder.inference(inputs=[vocoder_input])
            audio_raw = np.array(outputs[0]).flatten()

            # Extract only the chunk audio (skip context audio)
            ctx_frames_used = start - ctx_start
            ctx_samples = ctx_frames_used * SAMPLES_PER_FRAME
            chunk_samples = actual_chunk * SAMPLES_PER_FRAME

            audio_chunk = audio_raw[ctx_samples:ctx_samples + chunk_samples]
            audio_chunks.append(audio_chunk)

        return np.concatenate(audio_chunks).astype(np.float32)

    def _decode_audio_noembed(self, codes_array: np.ndarray) -> np.ndarray:
        """Decode using decoder_ctx25_int8.rknn (takes [1, 512, 50] float32 embeddings).

        Converts codes to embeddings first, then decodes in chunks of 25 frames
        with 25-frame context.
        """
        total_frames = codes_array.shape[0]

        # Convert codes to continuous embeddings
        all_embeddings = self._codes_to_embeddings(codes_array)  # [512, T]

        chunk_size = 25
        ctx_size = VOCODER_CTX_FRAMES  # 25
        total_T = ctx_size + chunk_size  # 50 (vocoder fixed input size)
        audio_chunks = []

        for start in range(0, total_frames, chunk_size):
            end = min(start + chunk_size, total_frames)
            actual_chunk = end - start
            ctx_start = max(0, start - ctx_size)

            # Get context + chunk embeddings
            chunk_emb = all_embeddings[:, ctx_start:end]  # [512, ctx+chunk]

            # Pad to fixed size [512, 50]
            padded = np.zeros((512, total_T), dtype=np.float32)
            padded[:, :chunk_emb.shape[1]] = chunk_emb

            # Run vocoder
            vocoder_input = padded[np.newaxis, :, :]  # [1, 512, 50]
            outputs = self._vocoder.inference(inputs=[vocoder_input])
            audio_raw = np.array(outputs[0]).flatten()

            # Extract only the chunk audio (skip context audio)
            ctx_frames_used = start - ctx_start
            ctx_samples = ctx_frames_used * SAMPLES_PER_FRAME
            chunk_samples = actual_chunk * SAMPLES_PER_FRAME

            audio_chunk = audio_raw[ctx_samples:ctx_samples + chunk_samples]
            audio_chunks.append(audio_chunk)

        return np.concatenate(audio_chunks).astype(np.float32)

    # ── WAV ──────────────────────────────────────────────────────

    @staticmethod
    def _make_wav(audio: np.ndarray) -> bytes:
        """Convert float32 audio to WAV bytes."""
        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return buf.getvalue()


# ── Module-level singleton ───────────────────────────────────────

_service: Optional[TTSService] = None


def preload():
    global _service
    model_dir = os.environ.get("MODEL_DIR", "/opt/tts/models")
    _service = TTSService(model_dir)
    _service.load()


def is_ready() -> bool:
    return _service is not None and _service.is_ready()


def synthesize(
    text: str,
    speaker_id: int = 0,
    speed: float = 1.0,
    pitch_shift: float = None,
) -> tuple[bytes, dict]:
    if _service is None:
        raise RuntimeError("TTS service not loaded")
    return _service.synthesize(text, speaker_id=speaker_id, speed=speed or 1.0)


def get_sample_rate() -> int:
    if _service is None:
        return SAMPLE_RATE
    return _service.get_sample_rate()
