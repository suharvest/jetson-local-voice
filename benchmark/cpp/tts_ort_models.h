// tts_ort_models.h — ONNX Runtime C++ wrappers for cold-path models
#pragma once

#include <onnxruntime_cxx_api.h>

#include <string>
#include <vector>

// Flags to skip or defer loading ORT sessions when TRT engines cover them.
// Default-constructed (all false) → identical behaviour to original code.
struct ORTSkipFlags {
  bool skip_vocoder = false;           // TRT vocoder loaded; skip vocoder ORT
  bool skip_talker_prefill = false;    // TRT prefill loaded; skip talker_prefill ORT
  bool lazy_speaker_encoder = false;   // load speaker_encoder on first use
  bool lazy_tokenizer_encode = false;  // load tokenizer12hz_encode on first use
  bool skip_cp_embed = false;    // pre-extracted cp_embed_fp32.bin available
  bool skip_codec_embed = false; // pre-extracted codec_embed_fp32.bin available
};

// All cold-path models loaded once, run via ORT CUDA EP
class ORTModels {
 public:
  ORTModels(const std::string& model_dir, const std::string& sherpa_dir,
            ORTSkipFlags flags = {}, int device_id = 0);

  // text_project: input_ids [1, T] int64 → embeddings [1, T, D] float32
  std::vector<float> TextProject(const std::vector<int64_t>& input_ids);

  // codec_embed: token_ids [1, T] int64 → embeddings [1, T, D] float32
  std::vector<float> CodecEmbed(const std::vector<int64_t>& token_ids);

  // cp_embed: token_id [1,1] int64, layer_idx scalar int64
  //         → embedding [1, 1, D] float32
  std::vector<float> CPEmbed(int64_t token_id, int64_t layer_idx);

  // talker_prefill: inputs_embeds [1, T, D] float32
  //               → logits [1, T, vocab], last_hidden [1, T, D],
  //                 past_key_0..N-1, past_value_0..N-1
  struct PrefillResult {
    std::vector<float> logits;       // [1, T, vocab]
    std::vector<float> last_hidden;  // [1, T, D]
    // KV cache: kv_data[2*i] = past_key_i, kv_data[2*i+1] = past_value_i
    // Each: [1, n_heads, T, head_dim]
    std::vector<std::vector<float>> kv_data;
    int seq_len;
  };
  PrefillResult TalkerPrefill(const float* embeds, int seq_len, int hidden_dim);

  // vocoder: codes [1, T, 16] int64 → audio [N] float32
  std::vector<float> Vocoder(const int64_t* codes, int n_frames, int n_groups);

  // speaker_encoder: mel [1, T, 128] float32 → spk_embed [D] float32
  // Returns empty if speaker_encoder not loaded
  std::vector<float> SpeakerEncode(const float* mel, int mel_frames);

  // tokenizer_encode: audio [1, N] float32 → codes [1, T, 16] int64
  // Returns empty if not loaded
  std::vector<int64_t> TokenizerEncode(const float* audio, int num_samples);

  // Check if ORT talker_prefill session is loaded
  bool HasTalkerPrefill() const { return talker_prefill_ != nullptr; }

  // Lazy-load on demand (no-op if already loaded)
  void LoadSpeakerEncoder();
  void LoadTokenizerEncode();

  int hidden_dim() const { return hidden_dim_; }
  int n_layers() const { return n_layers_; }
  int n_heads() const { return n_heads_; }
  int head_dim() const { return head_dim_; }
  int vocab_size() const { return vocab_size_; }

 private:
  Ort::Env env_;
  Ort::SessionOptions MakeSessionOptions(int device_id);
  Ort::SessionOptions MakeCPUSessionOptions();

  // Stored for lazy loading
  std::string sherpa_dir_;
  int device_id_ = 0;

  std::unique_ptr<Ort::Session> text_project_;      // old combined model (fallback)
  std::unique_ptr<Ort::Session> text_projection_;   // new: projection-only ONNX
  std::vector<uint16_t> text_embed_table_;           // new: FP16 embedding table
  int text_embed_vocab_ = 0;
  int text_embed_dim_ = 0;
  bool has_split_text_embed_ = false;

  std::unique_ptr<Ort::Session> codec_embed_;
  std::unique_ptr<Ort::Session> cp_embed_;
  std::unique_ptr<Ort::Session> talker_prefill_;
  std::unique_ptr<Ort::Session> vocoder_;
  std::unique_ptr<Ort::Session> speaker_encoder_;
  std::unique_ptr<Ort::Session> tokenizer_encode_;

  Ort::MemoryInfo mem_info_;
  int hidden_dim_ = 1024;
  int n_layers_ = 28;
  int n_heads_ = 8;
  int head_dim_ = 128;
  int vocab_size_ = 3072;
};
