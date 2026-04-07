// tts_pipeline.h — Complete Qwen3-TTS inference pipeline
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "tts_ort_models.h"
#include "tts_trt_engine.h"

// Config loaded from config.json
struct TTSConfig {
  int hidden_dim = 1024;
  int num_hidden_layers = 28;
  int num_code_groups = 16;
  int vocab_size = 3072;
  int n_heads = 8;
  int head_dim = 128;
  int cp_vocab = 2048;

  int tts_bos_token_id;
  int tts_eos_token_id;
  int tts_pad_token_id;
  int codec_bos_id;
  int codec_eos_token_id;
  int codec_pad_id;
  int codec_nothink_id;
  int codec_think_bos_id;
  int codec_think_eos_id;
  int codec_think_id;

  // language -> codec_language_id mapping
  std::vector<std::pair<std::string, int>> lang_ids;

  int GetLangId(const std::string& lang) const;
};

struct SynthResult {
  std::vector<float> audio;  // 24kHz float32 PCM
  int sample_rate = 24000;
  int n_frames = 0;
  double prefill_ms = 0;
  double decode_ms_avg = 0;
  double cp_ms_avg = 0;
  double vocoder_ms = 0;
  double total_ms = 0;
  double rtf = 0;
};

// Streaming chunk delivered via callback
struct StreamChunk {
  std::vector<float> audio;  // new audio samples since last chunk
  int total_frames;          // total codec frames generated so far
  bool is_final;             // true on last chunk
};

// Callback type for streaming synthesis
using AudioChunkCallback = std::function<void(const StreamChunk&)>;

// Streaming configuration
struct StreamConfig {
  int first_chunk_frames = 10;   // smaller first chunk for low TTFA (~800ms)
  int chunk_frames = 25;         // subsequent chunks (~2s audio each)
  int max_frames = 200;
  int seed = 0;  // 0 = random (time-based), >0 = fixed
};

class TTSPipeline {
 public:
  TTSPipeline(const std::string& model_dir, const std::string& sherpa_dir,
              const std::string& talker_engine_path,
              const std::string& cp_engine_path, int device_id = 0);

  // Load config.json from sherpa_dir
  void LoadConfig(const std::string& sherpa_dir);

  // Standard TTS
  SynthResult Synthesize(const std::string& text, const std::string& lang,
                         int max_frames = 200, int seed = 0);

  // X-vector voice clone: provide pre-computed speaker embedding
  SynthResult SynthesizeWithSpeaker(const std::string& text,
                                    const std::string& lang,
                                    const std::vector<float>& speaker_embed,
                                    int max_frames = 200, int seed = 0);

  // Synthesize with pre-tokenized IDs (bypass tokenizer)
  SynthResult SynthesizeWithTokenIds(const std::string& text,
                                     const std::string& lang,
                                     const std::vector<int64_t>& token_ids,
                                     const std::vector<float>* speaker_embed,
                                     int max_frames = 200, int seed = 0);

  // Streaming TTS with callback per audio chunk
  void SynthesizeStreaming(const std::string& text, const std::string& lang,
                           const std::vector<int64_t>& token_ids,
                           const StreamConfig& config,
                           AudioChunkCallback callback);

  // Streaming with voice clone
  void SynthesizeStreamingWithSpeaker(const std::string& text,
                                      const std::string& lang,
                                      const std::vector<int64_t>& token_ids,
                                      const std::vector<float>& speaker_embed,
                                      const StreamConfig& config,
                                      AudioChunkCallback callback);

  // Extract speaker embedding from mel spectrogram
  std::vector<float> ExtractSpeakerEmbedding(const float* mel,
                                              int mel_frames);

  // Enable CUDA event profiling for per-step timing breakdown
  void EnableProfiling(bool enable);
  // Print profiling stats and reset counters
  void PrintProfilingStats();

 private:
  // Core streaming generation loop
  void GenerateStreaming(const std::string& text, const std::string& lang,
                         const float* speaker_embed,
                         const std::vector<int64_t>* token_ids,
                         const StreamConfig& config,
                         AudioChunkCallback callback);
  // Core generation loop
  SynthResult GenerateInternal(const std::string& text, const std::string& lang,
                               const float* speaker_embed,  // nullptr if none
                               const std::vector<int64_t>* token_ids,  // nullptr to use tokenizer
                               int max_frames, int seed);

  // Build prefill embedding sequence
  struct PrefillData {
    std::vector<float> embeds;  // [1, T, D] flattened
    int seq_len;
    std::vector<float> trailing_text;  // per-step text embeddings
    int n_trailing;
    std::vector<float> tts_pad_e;  // [D]
    std::vector<float> codec_pad_e;  // [D]
  };
  PrefillData BuildPrefill(const std::string& text, const std::string& lang,
                           const float* speaker_embed,
                           const std::vector<int64_t>* token_ids = nullptr);

  // Top-k sampling (convenience wrapper without repetition penalty)
  int Sample(const float* logits, int vocab_size, int k = 50, float temp = 0.9f,
             bool suppress_eos = false, int eos_id = -1);
  // Full sampling with repetition penalty (matches official Qwen3-TTS generate)
  // eos_bias: added to EOS logit before sampling (>0 encourages EOS)
  // suppress_range: apply [vocab-1024, vocab) suppress (talker only, NOT for CP)
  int SampleWithPenalty(const float* logits, int vocab_size,
                        const int* prev_tokens, int n_prev,
                        int k = 50, float temp = 0.9f,
                        bool suppress_eos = false, int eos_id = -1,
                        float eos_bias = 0.0f,
                        bool suppress_range = true);

  // Check if primary_history shows looping (same token N+ times in a row)
  static bool DetectRepetition(const std::vector<int>& history,
                               int min_repeat = 5);

  // Tokenize text (simple BPE via vocab.json + merges.txt)
  std::vector<int64_t> Tokenize(const std::string& text);

  // Vector math helpers
  static void VecAdd(float* dst, const float* a, const float* b, int n);
  static void VecCopy(float* dst, const float* src, int n);

  // Load cp_embed weights from ONNX and upload to GPU
  void LoadCPEmbedTable(const std::string& sherpa_dir);

  TTSConfig cfg_;
  std::unique_ptr<ORTModels> ort_;
  std::unique_ptr<TRTTalkerEngine> talker_;
  std::unique_ptr<TRTCPEngine> cp_;
  std::unique_ptr<TRTCPKVEngine> cp_kv_;           // optional KV-cache CP engine
  std::unique_ptr<TRTVocoderEngine> trt_vocoder_;  // optional TRT vocoder

  // CPU copy of cp_embed table: [n_layers][vocab][D]
  std::vector<float> cp_embed_table_;
  int cp_embed_n_layers_ = 0;
  int cp_embed_vocab_ = 0;
  const float* CPEmbedLookup(int layer, int token_id) const;

  // RNG for sampling — seeded per request for reproducibility
  std::mt19937 rng_{42};
};
