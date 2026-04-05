// tts_pipeline.h — Complete Qwen3-TTS inference pipeline
#pragma once

#include <cstdint>
#include <memory>
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

class TTSPipeline {
 public:
  TTSPipeline(const std::string& model_dir, const std::string& sherpa_dir,
              const std::string& talker_engine_path,
              const std::string& cp_engine_path, int device_id = 0);

  // Load config.json from sherpa_dir
  void LoadConfig(const std::string& sherpa_dir);

  // Standard TTS
  SynthResult Synthesize(const std::string& text, const std::string& lang,
                         int max_frames = 200, int seed = 42);

  // X-vector voice clone: provide pre-computed speaker embedding
  SynthResult SynthesizeWithSpeaker(const std::string& text,
                                    const std::string& lang,
                                    const std::vector<float>& speaker_embed,
                                    int max_frames = 200, int seed = 42);

  // Synthesize with pre-tokenized IDs (bypass tokenizer)
  SynthResult SynthesizeWithTokenIds(const std::string& text,
                                     const std::string& lang,
                                     const std::vector<int64_t>& token_ids,
                                     const std::vector<float>* speaker_embed,
                                     int max_frames = 200, int seed = 42);

  // Extract speaker embedding from mel spectrogram
  std::vector<float> ExtractSpeakerEmbedding(const float* mel,
                                              int mel_frames);

 private:
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

  // Top-k sampling
  int Sample(const float* logits, int vocab_size, int k = 50, float temp = 0.9f,
             bool suppress_eos = false, int eos_id = -1);

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

  // CPU copy of cp_embed table: [n_layers][vocab][D]
  std::vector<float> cp_embed_table_;
  int cp_embed_n_layers_ = 0;
  int cp_embed_vocab_ = 0;
  const float* CPEmbedLookup(int layer, int token_id) const;
};
