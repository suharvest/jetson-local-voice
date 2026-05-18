// asr_pipeline.h — C++ ASR pipeline for Qwen3-ASR
// Encoder (ORT CUDA) + Unified TRT Decoder (prefill + step) + Embed lookup
// ORT prefill session kept as optional fallback
#pragma once

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tts_trt_engine.h"  // reuse TRTTalkerEngine for decoder step

struct ASRResult {
  std::vector<int64_t> text_ids;  // decoded token IDs (before EOS)
  int n_tokens = 0;
  double encoder_ms = 0;
  double prefill_ms = 0;
  double decode_ms = 0;      // total decode loop time
  double per_token_ms = 0;
  double total_ms = 0;
};

class ASRPipeline {
 public:
  // model_dir: directory containing encoder.onnx, embed_tokens.bin
  //            (decoder_prefill.onnx optional — kept as fallback)
  // engine_path: unified TRT engine for both prefill and decode step
  ASRPipeline(const std::string& model_dir,
              const std::string& engine_path,
              int device_id = 0);

  // Run full ASR: mel → encoder → prefill → decode → token IDs
  //   mel: float32 [128, T] or [1, 128, T] (flattened row-major)
  //   mel_len: T dimension
  //   prompt_ids: pre-built prompt token IDs
  //   audio_offset: index where AUDIO_PAD tokens start in prompt
  //   max_tokens: max decode steps
  ASRResult Transcribe(const float* mel, int mel_len,
                       const std::vector<int64_t>& prompt_ids,
                       int audio_offset,
                       int max_tokens = 200);

  // Check if ORT prefill is available (needed for correct results)
  bool has_prefill() const { return prefill_ != nullptr; }

  // Config
  int hidden_dim() const { return hidden_dim_; }
  int vocab_size() const { return vocab_size_; }
  int n_layers() const { return n_layers_; }

  // Enable CUDA event profiling for per-step timing breakdown
  void EnableProfiling(bool enable);
  void PrintProfilingStats();

  // Run encoder only: mel [1, 128, T] → EncoderOutput with audio_len
  struct EncoderOutput {
    std::vector<float> features;  // flattened [1, T', 1024]
    int audio_len;                // T'
  };
  EncoderOutput RunEncoder(const float* mel, int mel_len);

 private:
  // ORT sessions (own Env, isolated from TTS and sherpa)
  Ort::Env env_;
  Ort::MemoryInfo mem_info_;
  Ort::SessionOptions MakeSessionOptions(int device_id);

  std::unique_ptr<Ort::Session> encoder_;
  std::unique_ptr<Ort::Session> prefill_;

  // TRT decoder step engine (reuses TRTTalkerEngine)
  std::unique_ptr<TRTTalkerEngine> decoder_;

  // Embed tokens table: [vocab_size, hidden_dim] float32 (loaded from FP16 bin)
  std::vector<float> embed_table_;

  // Model dims
  int hidden_dim_ = 1024;
  int n_layers_ = 28;
  int n_heads_ = 8;
  int head_dim_ = 128;
  int vocab_size_ = 151936;
  int max_seq_ = 500;

  // EOS token IDs
  static constexpr int64_t EOS_1 = 151643;  // <|endoftext|>
  static constexpr int64_t EOS_2 = 151645;  // <|im_end|>

  // Embed lookup (CPU)
  const float* EmbedLookup(int64_t token_id) const;

  // Run prefill: input_ids, position_ids, audio_features, audio_offset
  //   → logits, KV cache
  struct PrefillOutput {
    std::vector<float> logits;  // [1, S, vocab_size]
    int seq_len;
    // KV: kv_data[2*i] = past_key_i, kv_data[2*i+1] = past_value_i
    std::vector<std::vector<float>> kv_data;
  };
  PrefillOutput RunPrefill(const std::vector<int64_t>& input_ids,
                           const float* audio_features, int audio_len,
                           int audio_offset);
};
