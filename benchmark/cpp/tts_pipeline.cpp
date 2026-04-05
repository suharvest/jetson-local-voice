// tts_pipeline.cpp — Full Qwen3-TTS generation pipeline
#include "tts_pipeline.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>

// Simple JSON parser for config.json (avoids external dependency)
#include "json_minimal.h"

// ---------------------------------------------------------------------------
// TTSConfig
// ---------------------------------------------------------------------------
int TTSConfig::GetLangId(const std::string& lang) const {
  for (auto& [name, id] : lang_ids) {
    if (name == lang) return id;
  }
  return -1;
}

// ---------------------------------------------------------------------------
// TTSPipeline
// ---------------------------------------------------------------------------
TTSPipeline::TTSPipeline(const std::string& model_dir,
                         const std::string& sherpa_dir,
                         const std::string& talker_engine_path,
                         const std::string& cp_engine_path, int device_id) {
  LoadConfig(sherpa_dir);

  ort_ = std::make_unique<ORTModels>(model_dir, sherpa_dir, device_id);

  std::cout << "Loading TRT engines..." << std::endl;
  talker_ = std::make_unique<TRTTalkerEngine>(
      talker_engine_path, cfg_.num_hidden_layers, cfg_.hidden_dim, cfg_.n_heads,
      cfg_.head_dim, cfg_.vocab_size);

  cp_ = std::make_unique<TRTCPEngine>(cp_engine_path, cfg_.hidden_dim,
                                      cfg_.cp_vocab);

  // Try to load cp_embed table on GPU for fast lookup
  LoadCPEmbedTable(sherpa_dir);

  std::cout << "Pipeline ready." << std::endl;
}

void TTSPipeline::LoadConfig(const std::string& sherpa_dir) {
  std::string path = sherpa_dir + "/config.json";
  auto j = JsonMinimal::Parse(path);

  cfg_.hidden_dim = j.GetInt("hidden_size", 1024);
  cfg_.num_hidden_layers = j.GetInt("num_hidden_layers", 28);
  cfg_.num_code_groups = j.GetInt("num_code_groups", 16);
  cfg_.vocab_size = j.GetInt("vocab_size", 3072);
  cfg_.n_heads = j.GetInt("num_key_value_heads", 8);  // GQA: KV uses fewer heads
  cfg_.head_dim = j.GetInt("head_dim", 128);

  cfg_.tts_bos_token_id = j.GetInt("tts_bos_token_id");
  cfg_.tts_eos_token_id = j.GetInt("tts_eos_token_id");
  cfg_.tts_pad_token_id = j.GetInt("tts_pad_token_id");
  cfg_.codec_bos_id = j.GetInt("codec_bos_id");
  cfg_.codec_eos_token_id = j.GetInt("codec_eos_token_id");
  cfg_.codec_pad_id = j.GetInt("codec_pad_id");
  cfg_.codec_nothink_id = j.GetInt("codec_nothink_id");
  cfg_.codec_think_bos_id = j.GetInt("codec_think_bos_id");
  cfg_.codec_think_eos_id = j.GetInt("codec_think_eos_id");
  cfg_.codec_think_id = j.GetInt("codec_think_id", cfg_.codec_nothink_id);

  // Parse codec_language_id map
  auto lang_map = j.GetObject("codec_language_id");
  for (auto& [k, v] : lang_map) {
    cfg_.lang_ids.push_back({k, std::stoi(v)});
  }

  std::cout << "Config: D=" << cfg_.hidden_dim << ", layers=" << cfg_.num_hidden_layers
            << ", vocab=" << cfg_.vocab_size << ", groups=" << cfg_.num_code_groups
            << std::endl;
}

// ---------------------------------------------------------------------------
// Vector helpers
// ---------------------------------------------------------------------------
void TTSPipeline::VecAdd(float* dst, const float* a, const float* b, int n) {
  for (int i = 0; i < n; ++i) dst[i] = a[i] + b[i];
}

void TTSPipeline::VecCopy(float* dst, const float* src, int n) {
  std::memcpy(dst, src, n * sizeof(float));
}

// ---------------------------------------------------------------------------
// Tokenize (placeholder — uses tokenizer from tokenizers-cpp or simple split)
// For now, loads pre-tokenized IDs. Real impl needs BPE tokenizer.
// ---------------------------------------------------------------------------
std::vector<int64_t> TTSPipeline::Tokenize(const std::string& text) {
  // TODO: Integrate tokenizers-cpp or sentencepiece
  // For now, this is a stub that should be replaced.
  // The caller should provide pre-tokenized IDs or we use a C++ BPE lib.
  std::cerr << "WARNING: Tokenize() stub called. Provide pre-tokenized IDs."
            << std::endl;
  return {};
}

// ---------------------------------------------------------------------------
// Sample
// ---------------------------------------------------------------------------
int TTSPipeline::Sample(const float* logits, int vocab_size, int k, float temp,
                        bool suppress_eos, int eos_id) {
  std::vector<double> l(vocab_size);
  for (int i = 0; i < vocab_size; ++i) l[i] = logits[i];

  if (suppress_eos && eos_id >= 0 && eos_id < vocab_size) {
    l[eos_id] = -1e9;
  }

  // Temperature
  if (temp > 1e-6) {
    for (auto& v : l) v /= temp;
  }

  // Top-k
  if (k > 0 && k < vocab_size) {
    std::vector<double> sorted(l);
    std::nth_element(sorted.begin(), sorted.end() - k, sorted.end());
    double threshold = sorted[sorted.size() - k];
    for (auto& v : l) {
      if (v < threshold) v = -1e9;
    }
  }

  // Softmax
  double max_val = *std::max_element(l.begin(), l.end());
  double sum = 0;
  for (auto& v : l) {
    v = std::exp(v - max_val);
    sum += v;
  }
  for (auto& v : l) v /= sum;

  // Sample
  static std::mt19937 rng;
  std::discrete_distribution<int> dist(l.begin(), l.end());
  return dist(rng);
}

// ---------------------------------------------------------------------------
// BuildPrefill
// ---------------------------------------------------------------------------
TTSPipeline::PrefillData TTSPipeline::BuildPrefill(
    const std::string& text, const std::string& lang,
    const float* speaker_embed,
    const std::vector<int64_t>* token_ids_ptr) {
  // Official Qwen3-TTS prefill layout (multi-language):
  //   [0..2]  = role_emb (text_proj([151644, 77091, 198]))
  //   [3]     = tts_pad + codec[THINK]
  //   [4]     = tts_pad + codec[THINK_BOS]
  //   [5]     = tts_pad + codec[lang_id]
  //   [6]     = tts_pad + codec[THINK_EOS]
  //   [6+s]   = speaker_embed             (only if voice clone, s=1)
  //   [7+s]   = tts_bos + codec[PAD]
  //   [8+s]   = body[0]  + codec[BOS]
  //   trailing = body[1:] + codec[PAD], then tts_eos + codec[PAD]

  int D = cfg_.hidden_dim;
  PrefillData pf;

  // role_emb = text_proj([<|im_start|>, assistant, \n])
  auto role_emb = ort_->TextProject({151644, 77091, 198});

  // special_emb = text_proj([TTS_BOS, TTS_EOS, TTS_PAD])
  auto special_emb = ort_->TextProject(
      {cfg_.tts_bos_token_id, cfg_.tts_eos_token_id, cfg_.tts_pad_token_id});
  const float* tts_bos_e = special_emb.data();
  const float* tts_eos_e = special_emb.data() + D;
  const float* tts_pad_e = special_emb.data() + 2 * D;

  // Codec prefix: [THINK, THINK_BOS, lang_id, THINK_EOS, PAD, BOS]
  int lang_id = cfg_.GetLangId(lang);
  std::vector<int64_t> codec_ids;
  if (lang_id >= 0) {
    codec_ids = {cfg_.codec_think_id, cfg_.codec_think_bos_id,
                 (int64_t)lang_id, cfg_.codec_think_eos_id,
                 cfg_.codec_pad_id, cfg_.codec_bos_id};
  } else {
    // Fallback: no language token (like old simplified version)
    codec_ids = {cfg_.codec_nothink_id, cfg_.codec_think_bos_id,
                 cfg_.codec_think_eos_id,
                 cfg_.codec_pad_id, cfg_.codec_bos_id};
  }
  auto codec_prefix = ort_->CodecEmbed(codec_ids);
  int n_codec = (int)codec_ids.size();  // 6 (with lang) or 5 (without)

  // codec_pad for decode loop
  auto codec_pad_v = ort_->CodecEmbed({cfg_.codec_pad_id});
  const float* codec_pad_e = codec_pad_v.data();

  pf.tts_pad_e.assign(tts_pad_e, tts_pad_e + D);
  pf.codec_pad_e.assign(codec_pad_e, codec_pad_e + D);

  // Text tokens
  std::vector<int64_t> text_ids;
  if (token_ids_ptr && !token_ids_ptr->empty()) {
    text_ids = *token_ids_ptr;
  } else {
    text_ids = Tokenize(text);
  }
  std::vector<float> body_emb;
  if (!text_ids.empty()) {
    body_emb = ort_->TextProject(text_ids);
  }

  // Assemble prefill:
  //   role(3) + tts_pad+codec[0..N-3](N-2 tokens) + speaker?(1) + tts_bos+codec[N-2] + body[0]+codec[N-1]
  // codec layout: [0..N-3] = think prefix, [N-2] = PAD, [N-1] = BOS
  int has_spk = speaker_embed ? 1 : 0;
  int n_prefix = 3 + (n_codec - 2) + has_spk + 1 + (body_emb.empty() ? 0 : 1);
  std::vector<float> prefill(n_prefix * D, 0.0f);
  int pos = 0;

  // [0..2] = role
  VecCopy(prefill.data() + pos * D, role_emb.data(), 3 * D);
  pos += 3;

  // [3..3+N-3] = tts_pad + codec[0..N-3] (think prefix tokens)
  for (int i = 0; i < n_codec - 2; ++i) {
    VecAdd(prefill.data() + pos * D, tts_pad_e, codec_prefix.data() + i * D, D);
    pos++;
  }

  // Optional speaker embed
  if (speaker_embed) {
    VecCopy(prefill.data() + pos * D, speaker_embed, D);
    pos++;
  }

  // tts_bos + codec[PAD] (second-to-last codec token)
  VecAdd(prefill.data() + pos * D, tts_bos_e, codec_prefix.data() + (n_codec - 2) * D, D);
  pos++;

  // body[0] + codec[BOS] (last codec token)
  if (!body_emb.empty()) {
    VecAdd(prefill.data() + pos * D, body_emb.data(), codec_prefix.data() + (n_codec - 1) * D, D);
    pos++;
  }

  pf.embeds = std::move(prefill);
  pf.seq_len = pos;

  // Trailing text: body[1:] + codec_pad, then tts_eos + codec_pad
  int n_text = body_emb.empty() ? 0 : (int)(body_emb.size() / D);
  int n_trailing = (n_text > 1 ? n_text - 1 : 0) + 1;  // body[1:] + eos
  pf.trailing_text.resize(n_trailing * D);
  pf.n_trailing = n_trailing;

  int t = 0;
  for (int i = 1; i < n_text; ++i, ++t) {
    VecAdd(pf.trailing_text.data() + t * D, body_emb.data() + i * D, codec_pad_e, D);
  }
  // Last: tts_eos + codec_pad
  VecAdd(pf.trailing_text.data() + t * D, tts_eos_e, codec_pad_e, D);

  return pf;
}

// ---------------------------------------------------------------------------
// GenerateInternal
// ---------------------------------------------------------------------------
SynthResult TTSPipeline::GenerateInternal(const std::string& text,
                                          const std::string& lang,
                                          const float* speaker_embed,
                                          const std::vector<int64_t>* token_ids,
                                          int max_frames, int seed) {
  using Clock = std::chrono::high_resolution_clock;
  SynthResult result;

  // Seed RNG
  // (Sample uses a static mt19937 — seed it here)
  // For reproducibility:
  srand(seed);

  int D = cfg_.hidden_dim;

  // Build prefill
  auto pf = BuildPrefill(text, lang, speaker_embed, token_ids);
  std::cout << "  Prefill: " << pf.seq_len << " tokens" << std::endl;

  // Run prefill (ORT CUDA)
  auto t0 = Clock::now();
  auto pf_result = ort_->TalkerPrefill(pf.embeds.data(), pf.seq_len, D);
  result.prefill_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

  // Seed TRT KV cache
  std::vector<const float*> kv_ptrs;
  for (auto& kv : pf_result.kv_data) {
    kv_ptrs.push_back(kv.data());
  }
  talker_->SeedKV(kv_ptrs.data(), (int)kv_ptrs.size(), pf.seq_len);

  // Logits and hidden from prefill (last position)
  std::vector<float> logits(cfg_.vocab_size);
  std::vector<float> last_hidden(D);
  // Extract last position from prefill output
  int last_pos = pf.seq_len - 1;
  VecCopy(logits.data(),
          pf_result.logits.data() + last_pos * cfg_.vocab_size,
          cfg_.vocab_size);
  VecCopy(last_hidden.data(),
          pf_result.last_hidden.data() + last_pos * D, D);

  // Decode loop
  std::vector<std::vector<int>> all_codes;
  std::vector<double> dt_times, ct_times;

  for (int step = 0; step < max_frames; ++step) {
    // Sample primary code
    int primary_code = Sample(logits.data(), cfg_.vocab_size, 50, 0.9f,
                              step < 2, cfg_.codec_eos_token_id);
    if (primary_code == cfg_.codec_eos_token_id) {
      std::cout << "  EOS at step " << step << std::endl;
      break;
    }

    // Code predictor: 15 residual codes (GPU-resident context)
    auto t_cp = Clock::now();
    auto primary_e = ort_->CodecEmbed({(int64_t)primary_code});

    // Initialize GPU context with [hidden, primary_emb]
    cp_->BeginFrame(last_hidden.data(), primary_e.data());

    std::vector<float> codec_sum(D);
    VecCopy(codec_sum.data(), primary_e.data(), D);

    std::vector<int> frame_codes = {primary_code};

    for (int j = 0; j < cfg_.num_code_groups - 1; ++j) {
      std::vector<float> cp_logits(cfg_.cp_vocab);
      cp_->PredictGPU(j, cp_logits.data());

      int rc = Sample(cp_logits.data(), cfg_.cp_vocab);
      frame_codes.push_back(rc);

      if (cp_->has_embed_table()) {
        // GPU→GPU: append embedding from pre-loaded table (no ORT call!)
        cp_->AppendEmbeddingFromTable(j, rc);
        // CPU table lookup for codec_sum (zero-cost pointer arithmetic)
        const float* re = CPEmbedLookup(j, rc);
        VecAdd(codec_sum.data(), codec_sum.data(), re, D);
      } else {
        auto re = ort_->CPEmbed(rc, j);
        cp_->AppendEmbedding(re.data());
        VecAdd(codec_sum.data(), codec_sum.data(), re.data(), D);
      }
    }
    ct_times.push_back(
        std::chrono::duration<double, std::milli>(Clock::now() - t_cp).count());
    all_codes.push_back(frame_codes);

    // Next talker input: codec_sum + text_embed
    std::vector<float> next_emb(D);
    if (step < pf.n_trailing) {
      VecAdd(next_emb.data(), codec_sum.data(),
             pf.trailing_text.data() + step * D, D);
    } else {
      // tts_pad + codec_pad
      std::vector<float> pad_sum(D);
      VecAdd(pad_sum.data(), pf.tts_pad_e.data(), pf.codec_pad_e.data(), D);
      VecAdd(next_emb.data(), codec_sum.data(), pad_sum.data(), D);
    }

    // Talker decode
    auto t_d = Clock::now();
    talker_->DecodeStep(next_emb.data(), logits.data(), last_hidden.data());
    dt_times.push_back(
        std::chrono::duration<double, std::milli>(Clock::now() - t_d).count());

    if ((step + 1) % 10 == 0) {
      std::cout << "  Frame " << step + 1 << std::endl;
    }
  }

  int n = (int)all_codes.size();
  if (n == 0) {
    std::cerr << "  No frames generated!" << std::endl;
    return result;
  }

  result.n_frames = n;
  double dur = n / 12.5;
  result.decode_ms_avg =
      std::accumulate(dt_times.begin(), dt_times.end(), 0.0) / n;
  result.cp_ms_avg =
      std::accumulate(ct_times.begin(), ct_times.end(), 0.0) / n;

  // Vocoder
  // Reshape codes to [1, n_frames, 16] — vocoder expects [1, T, 16]
  std::vector<int64_t> codes_t(n * cfg_.num_code_groups);
  for (int f = 0; f < n; ++f) {
    for (int g = 0; g < cfg_.num_code_groups; ++g) {
      codes_t[f * cfg_.num_code_groups + g] = all_codes[f][g];
    }
  }

  auto t_voc = Clock::now();
  result.audio = ort_->Vocoder(codes_t.data(), n, cfg_.num_code_groups);
  result.vocoder_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - t_voc).count();

  auto per_step = result.decode_ms_avg + result.cp_ms_avg;
  result.rtf = per_step / 80.0;  // 80ms per frame at 12.5 Hz
  result.total_ms = result.prefill_ms + n * per_step + result.vocoder_ms;

  std::cout << "\n  === TIMING (" << n << " frames, " << dur << "s audio) ==="
            << std::endl;
  std::cout << "  Prefill:     " << result.prefill_ms << " ms" << std::endl;
  std::cout << "  Talker/step: " << result.decode_ms_avg << " ms (TRT)"
            << std::endl;
  std::cout << "  CP/step:     " << result.cp_ms_avg << " ms (TRT)"
            << std::endl;
  std::cout << "  Per-step:    " << per_step << " ms  RTF=" << result.rtf
            << std::endl;
  std::cout << "  Vocoder:     " << result.vocoder_ms << " ms" << std::endl;
  std::cout << "  Total:       " << result.total_ms << " ms" << std::endl;

  return result;
}

void TTSPipeline::LoadCPEmbedTable(const std::string& sherpa_dir) {
  int n_layers = cfg_.num_code_groups - 1;  // 15
  int vocab = cfg_.cp_vocab;                // 2048
  int D = cfg_.hidden_dim;                  // 1024

  std::cout << "Pre-computing cp_embed table (" << n_layers << "×" << vocab
            << "×" << D << ")..." << std::endl;

  // Allocate flat table: [n_layers][vocab][D]
  std::vector<float> table(n_layers * vocab * D);

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int layer = 0; layer < n_layers; ++layer) {
    for (int tid = 0; tid < vocab; ++tid) {
      auto emb = ort_->CPEmbed(tid, layer);
      // emb may be [1,1,1,D] or [1,1,D] — take first D elements
      size_t offset = ((size_t)layer * vocab + tid) * D;
      std::memcpy(table.data() + offset, emb.data(), D * sizeof(float));
    }
    std::cout << "  Layer " << layer << " done" << std::endl;
  }
  auto dt = std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - t0).count();
  std::cout << "  cp_embed table computed in " << dt << "s" << std::endl;

  // Keep CPU copy for codec_sum accumulation
  cp_embed_table_ = std::move(table);
  cp_embed_n_layers_ = n_layers;
  cp_embed_vocab_ = vocab;

  // Upload to GPU for fast context append
  cp_->LoadEmbedTable(cp_embed_table_.data(), n_layers, vocab, D);
}

const float* TTSPipeline::CPEmbedLookup(int layer, int token_id) const {
  size_t offset = ((size_t)layer * cp_embed_vocab_ + token_id) * cfg_.hidden_dim;
  return cp_embed_table_.data() + offset;
}

SynthResult TTSPipeline::Synthesize(const std::string& text,
                                    const std::string& lang, int max_frames,
                                    int seed) {
  std::cout << "Synth: \"" << text << "\" (" << lang << ")" << std::endl;
  return GenerateInternal(text, lang, nullptr, nullptr, max_frames, seed);
}

SynthResult TTSPipeline::SynthesizeWithSpeaker(
    const std::string& text, const std::string& lang,
    const std::vector<float>& speaker_embed, int max_frames, int seed) {
  std::cout << "Synth (voice clone): \"" << text << "\" (" << lang << ")"
            << std::endl;
  return GenerateInternal(text, lang, speaker_embed.data(), nullptr, max_frames, seed);
}

SynthResult TTSPipeline::SynthesizeWithTokenIds(
    const std::string& text, const std::string& lang,
    const std::vector<int64_t>& token_ids,
    const std::vector<float>* speaker_embed, int max_frames, int seed) {
  std::cout << "Synth (token-ids): \"" << text << "\" (" << lang << ", "
            << token_ids.size() << " tokens)" << std::endl;
  return GenerateInternal(text, lang,
                          speaker_embed ? speaker_embed->data() : nullptr,
                          &token_ids, max_frames, seed);
}

std::vector<float> TTSPipeline::ExtractSpeakerEmbedding(const float* mel,
                                                         int mel_frames) {
  return ort_->SpeakerEncode(mel, mel_frames);
}

// ---------------------------------------------------------------------------
// Streaming synthesis
// ---------------------------------------------------------------------------
void TTSPipeline::SynthesizeStreaming(const std::string& text,
                                      const std::string& lang,
                                      const std::vector<int64_t>& token_ids,
                                      const StreamConfig& config,
                                      AudioChunkCallback callback) {
  std::cout << "Synth streaming (token-ids): \"" << text << "\" (" << lang
            << ", " << token_ids.size() << " tokens)" << std::endl;
  GenerateStreaming(text, lang, nullptr,
                    token_ids.empty() ? nullptr : &token_ids, config, callback);
}

void TTSPipeline::SynthesizeStreamingWithSpeaker(
    const std::string& text, const std::string& lang,
    const std::vector<int64_t>& token_ids,
    const std::vector<float>& speaker_embed, const StreamConfig& config,
    AudioChunkCallback callback) {
  std::cout << "Synth streaming (voice clone): \"" << text << "\" (" << lang
            << ")" << std::endl;
  GenerateStreaming(text, lang, speaker_embed.data(),
                    token_ids.empty() ? nullptr : &token_ids, config, callback);
}

void TTSPipeline::GenerateStreaming(const std::string& text,
                                     const std::string& lang,
                                     const float* speaker_embed,
                                     const std::vector<int64_t>* token_ids,
                                     const StreamConfig& config,
                                     AudioChunkCallback callback) {
  using Clock = std::chrono::high_resolution_clock;

  srand(config.seed);
  int D = cfg_.hidden_dim;

  // Build prefill
  auto pf = BuildPrefill(text, lang, speaker_embed, token_ids);
  std::cout << "  Prefill: " << pf.seq_len << " tokens (streaming)"
            << std::endl;

  // Run prefill
  auto t0 = Clock::now();
  auto pf_result = ort_->TalkerPrefill(pf.embeds.data(), pf.seq_len, D);
  double prefill_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  std::cout << "  Prefill: " << prefill_ms << " ms" << std::endl;

  // Seed TRT KV cache
  std::vector<const float*> kv_ptrs;
  for (auto& kv : pf_result.kv_data) {
    kv_ptrs.push_back(kv.data());
  }
  talker_->SeedKV(kv_ptrs.data(), (int)kv_ptrs.size(), pf.seq_len);

  // Logits and hidden from prefill (last position)
  std::vector<float> logits(cfg_.vocab_size);
  std::vector<float> last_hidden(D);
  int last_pos_pf = pf.seq_len - 1;
  VecCopy(logits.data(),
          pf_result.logits.data() + last_pos_pf * cfg_.vocab_size,
          cfg_.vocab_size);
  VecCopy(last_hidden.data(),
          pf_result.last_hidden.data() + last_pos_pf * D, D);

  // Decode loop with chunked vocoder
  std::vector<std::vector<int>> all_codes;
  size_t last_audio_pos = 0;  // track yielded audio position

  // Determine chunk boundaries: first chunk is smaller for low TTFA
  int next_chunk_at = config.first_chunk_frames;

  for (int step = 0; step < config.max_frames; ++step) {
    // Sample primary code
    int primary_code = Sample(logits.data(), cfg_.vocab_size, 50, 0.9f,
                              step < 2, cfg_.codec_eos_token_id);
    if (primary_code == cfg_.codec_eos_token_id) {
      std::cout << "  EOS at step " << step << std::endl;
      break;
    }

    // Code predictor: 15 residual codes
    auto primary_e = ort_->CodecEmbed({(int64_t)primary_code});
    cp_->BeginFrame(last_hidden.data(), primary_e.data());

    std::vector<float> codec_sum(D);
    VecCopy(codec_sum.data(), primary_e.data(), D);

    std::vector<int> frame_codes = {primary_code};

    for (int j = 0; j < cfg_.num_code_groups - 1; ++j) {
      std::vector<float> cp_logits(cfg_.cp_vocab);
      cp_->PredictGPU(j, cp_logits.data());

      int rc = Sample(cp_logits.data(), cfg_.cp_vocab);
      frame_codes.push_back(rc);

      if (cp_->has_embed_table()) {
        cp_->AppendEmbeddingFromTable(j, rc);
        const float* re = CPEmbedLookup(j, rc);
        VecAdd(codec_sum.data(), codec_sum.data(), re, D);
      } else {
        auto re = ort_->CPEmbed(rc, j);
        cp_->AppendEmbedding(re.data());
        VecAdd(codec_sum.data(), codec_sum.data(), re.data(), D);
      }
    }
    all_codes.push_back(frame_codes);

    // Next talker input
    std::vector<float> next_emb(D);
    if (step < pf.n_trailing) {
      VecAdd(next_emb.data(), codec_sum.data(),
             pf.trailing_text.data() + step * D, D);
    } else {
      std::vector<float> pad_sum(D);
      VecAdd(pad_sum.data(), pf.tts_pad_e.data(), pf.codec_pad_e.data(), D);
      VecAdd(next_emb.data(), codec_sum.data(), pad_sum.data(), D);
    }

    // Talker decode
    talker_->DecodeStep(next_emb.data(), logits.data(), last_hidden.data());

    // Check if we should emit a chunk
    int n = (int)all_codes.size();
    if (n >= next_chunk_at) {
      // Run vocoder on all accumulated codes
      std::vector<int64_t> codes_t(n * cfg_.num_code_groups);
      for (int f = 0; f < n; ++f) {
        for (int g = 0; g < cfg_.num_code_groups; ++g) {
          codes_t[f * cfg_.num_code_groups + g] = all_codes[f][g];
        }
      }

      auto full_audio = ort_->Vocoder(codes_t.data(), n, cfg_.num_code_groups);

      // Yield only new samples
      if (full_audio.size() > last_audio_pos) {
        StreamChunk chunk;
        chunk.audio.assign(full_audio.begin() + last_audio_pos,
                           full_audio.end());
        chunk.total_frames = n;
        chunk.is_final = false;
        last_audio_pos = full_audio.size();

        std::cout << "  Chunk: " << n << " frames, "
                  << chunk.audio.size() << " new samples" << std::endl;
        callback(chunk);
      }

      // After first chunk, use regular chunk size
      next_chunk_at = n + config.chunk_frames;
    }

    if ((step + 1) % 10 == 0 && n < next_chunk_at) {
      std::cout << "  Frame " << step + 1 << std::endl;
    }
  }

  // Final chunk: vocoder on all remaining codes
  int n = (int)all_codes.size();
  if (n > 0) {
    std::vector<int64_t> codes_t(n * cfg_.num_code_groups);
    for (int f = 0; f < n; ++f) {
      for (int g = 0; g < cfg_.num_code_groups; ++g) {
        codes_t[f * cfg_.num_code_groups + g] = all_codes[f][g];
      }
    }

    auto full_audio = ort_->Vocoder(codes_t.data(), n, cfg_.num_code_groups);

    StreamChunk chunk;
    if (full_audio.size() > last_audio_pos) {
      chunk.audio.assign(full_audio.begin() + last_audio_pos,
                         full_audio.end());
    }
    chunk.total_frames = n;
    chunk.is_final = true;

    std::cout << "  Final chunk: " << n << " frames, "
              << chunk.audio.size() << " new samples" << std::endl;
    callback(chunk);
  } else {
    // No frames at all — emit empty final chunk
    StreamChunk chunk;
    chunk.total_frames = 0;
    chunk.is_final = true;
    callback(chunk);
  }

  std::cout << "  Streaming complete: " << n << " total frames" << std::endl;
}
