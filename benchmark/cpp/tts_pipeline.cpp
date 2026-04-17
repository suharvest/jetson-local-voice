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

  std::cout << "Loading TRT engines..." << std::endl;
  talker_ = std::make_unique<TRTTalkerEngine>(
      talker_engine_path, cfg_.num_hidden_layers, cfg_.hidden_dim, cfg_.n_heads,
      cfg_.head_dim, cfg_.vocab_size);

  // Auto-detect engines: look in the same directory as the decode engine.
  bool trt_prefill_loaded = false;
  {
    std::string engine_dir = talker_engine_path;
    auto slash = engine_dir.rfind('/');
    if (slash != std::string::npos) {
      engine_dir = engine_dir.substr(0, slash);
    } else {
      engine_dir = ".";
    }
    // Try to load CP KV cache engine FIRST (cp_unified_bf16.engine).
    // Must be detected before deciding whether to load the old CP engine.
    std::string cp_kv_path = engine_dir + "/cp_unified_bf16.engine";
    std::ifstream cp_kv_test(cp_kv_path);
    if (cp_kv_test.good()) {
      cp_kv_test.close();
      std::cout << "  Found CP KV engine: " << cp_kv_path << std::endl;
      cp_kv_ = std::make_unique<TRTCPKVEngine>(
          cp_kv_path,
          5,                      // n_cp_layers
          cfg_.hidden_dim,        // 1024
          cfg_.n_heads,           // 8
          cfg_.head_dim,          // 128
          cfg_.cp_vocab,          // 2048
          cfg_.num_code_groups - 1  // 15 output groups
      );
    }

    // Load old CP engine only when no CP KV engine is available.
    if (!cp_kv_) {
      cp_ = std::make_unique<TRTCPEngine>(cp_engine_path, cfg_.hidden_dim,
                                          cfg_.cp_vocab);
      std::cout << "  No CP KV engine — loaded old context-copy CP" << std::endl;
    } else {
      std::cout << "  CP KV engine active — skipped old CP engine (saves ~332MB)" << std::endl;
    }

    // Load separate prefill engine only when the decode engine lacks dual profiles.
    // Dual-profile engines handle batch prefill via Profile 0 internally.
    if (!talker_->has_dual_profiles()) {
      // Prefer BF16 (no SIGSEGV), fall back to FP16, then iterative
      std::vector<std::string> prefill_candidates = {
          engine_dir + "/talker_prefill_bf16.engine",
          engine_dir + "/talker_prefill_fp16.engine",
      };
      for (const auto& prefill_path : prefill_candidates) {
        std::ifstream test(prefill_path);
        if (test.good()) {
          test.close();
          std::cout << "  Found prefill engine: " << prefill_path << std::endl;
          talker_->LoadPrefillEngine(prefill_path);
          trt_prefill_loaded = true;
          break;
        }
      }
      if (!trt_prefill_loaded) {
        std::cout << "  No prefill engine found — using iterative prefill fallback" << std::endl;
      }
    } else {
      trt_prefill_loaded = true;  // dual-profile handles prefill via Profile 0
      std::cout << "  Dual-profile decode engine — skipped separate prefill (saves ~861MB)" << std::endl;
    }

    // Try to load vocoder TRT engine
    std::string voc_engine_path = engine_dir + "/vocoder_fp16.engine";
    std::ifstream voc_test(voc_engine_path);
    if (voc_test.good()) {
      voc_test.close();
      std::cout << "  Found vocoder TRT engine: " << voc_engine_path << std::endl;
      trt_vocoder_ = std::make_unique<TRTVocoderEngine>(voc_engine_path, 200, 24000 * 20);
    } else {
      std::cout << "  No vocoder TRT engine found â using ORT fallback" << std::endl;
    }
  }

  // Build ORT skip flags based on which TRT engines were loaded and whether
  // pre-extracted binary tables are available. Create ORTModels afterwards.
  {
    ORTSkipFlags skip;
    skip.skip_vocoder = (trt_vocoder_ != nullptr);
    skip.skip_talker_prefill = trt_prefill_loaded;
    skip.lazy_speaker_encoder = true;   // load on first voice-clone use
    skip.lazy_tokenizer_encode = true;  // load on first voice-clone use
    // Skip ORT sessions when pre-extracted binary files are present
    skip.skip_cp_embed = std::ifstream(sherpa_dir + "/cp_embed_fp32.bin").good();
    skip.skip_codec_embed = std::ifstream(sherpa_dir + "/codec_embed_fp32.bin").good();
    ort_ = std::make_unique<ORTModels>(model_dir, sherpa_dir, skip, device_id);
  }

  // Try to load cp_embed table on GPU for fast lookup
  LoadCPEmbedTable(sherpa_dir);

  // Pre-compute codec_embed table for O(1) lookup in decode loop
  LoadCodecEmbedTable(sherpa_dir);

  // Start async vocoder worker thread (Scheme A overlap)
  voc_stop_ = false;
  vocoder_thread_ = std::thread(&TTSPipeline::VocoderWorkerLoop, this);

  std::cout << "Pipeline ready." << std::endl;
}

TTSPipeline::~TTSPipeline() {
  if (vocoder_thread_.joinable()) {
    {
      std::lock_guard<std::mutex> lk(voc_mutex_);
      voc_stop_ = true;
    }
    voc_cv_.notify_all();
    vocoder_thread_.join();
  }
}

void TTSPipeline::VocoderWorkerLoop() {
  while (true) {
    VocWork w;
    {
      std::unique_lock<std::mutex> lk(voc_mutex_);
      voc_cv_.wait(lk, [&] { return !voc_queue_.empty() || voc_stop_; });
      if (voc_stop_ && voc_queue_.empty()) return;
      w = std::move(voc_queue_.front());
      voc_queue_.pop();
    }
    std::vector<float> audio;
    try {
      if (trt_vocoder_) {
        audio =
            trt_vocoder_->Run(w.codes.data(), w.window_len, w.num_code_groups);
      } else {
        audio = ort_->Vocoder(w.codes.data(), w.window_len, w.num_code_groups);
      }
    } catch (const std::exception& ex) {
      std::cerr << "[voc-worker] ERROR: " << ex.what() << std::endl;
    }
    StreamChunk chunk;
    if (audio.size() > w.skip_samples) {
      chunk.audio.assign(audio.begin() + w.skip_samples, audio.end());
    }
    chunk.total_frames = w.total_frames;
    chunk.is_final = w.is_final;
    if (w.callback) w.callback(chunk);

    int rem = --voc_inflight_;
    if (rem == 0) {
      std::lock_guard<std::mutex> lk(voc_mutex_);
      voc_empty_cv_.notify_all();
    }
  }
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

  // Runtime CP codebook-count experiment knob.
  // Default = num_code_groups - 1 (all 15 residuals = full quality).
  cfg_.cp_active_groups = j.GetInt("cp_active_groups", cfg_.num_code_groups - 1);
  if (cfg_.cp_active_groups < 1) cfg_.cp_active_groups = 1;
  if (cfg_.cp_active_groups > cfg_.num_code_groups - 1)
    cfg_.cp_active_groups = cfg_.num_code_groups - 1;

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
            << ", cp_active_groups=" << cfg_.cp_active_groups
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
  // Sample() is used for CP (code predictor) — no suppress, no penalty, no bias
  return SampleWithPenalty(logits, vocab_size, nullptr, 0, k, temp,
                           suppress_eos, eos_id, 0.0f, /*suppress_range=*/false);
}

int TTSPipeline::SampleWithPenalty(const float* logits, int vocab_size,
                                    const int* prev_tokens, int n_prev,
                                    int k, float temp,
                                    bool suppress_eos, int eos_id,
                                    float eos_bias,
                                    bool suppress_range) {
  std::vector<double> l(vocab_size);
  for (int i = 0; i < vocab_size; ++i) l[i] = logits[i];

  // 1. Repetition penalty (before temperature) — official: 1.05
  //    Penalizes all previously generated tokens in the sequence.
  //    Positive logits: divide by penalty. Negative: multiply by penalty.
  if (prev_tokens && n_prev > 0) {
    const double rep_penalty = 1.05;
    for (int t = 0; t < n_prev; ++t) {
      int tok = prev_tokens[t];
      if (tok >= 0 && tok < vocab_size) {
        if (l[tok] < 0.0) l[tok] *= rep_penalty;
        else               l[tok] /= rep_penalty;
      }
    }
  }

  // 2. Suppress EOS for first 2 steps
  if (suppress_eos && eos_id >= 0 && eos_id < vocab_size) {
    l[eos_id] = -1e30;
  }

  // 3. Suppress non-codec tokens: [vocab_size-1024, vocab_size) except EOS
  //    Only for talker (primary codec), NOT for code predictor (residual codecs)
  if (suppress_range) {
    int suppress_start = vocab_size - 1024;
    for (int i = suppress_start; i < vocab_size; ++i) {
      if (i != eos_id) l[i] = -1e30;
    }
  }

  // 4. EOS bias: boost EOS logit to compensate systematic gap
  if (eos_bias != 0.0f && !suppress_eos && eos_id >= 0 && eos_id < vocab_size) {
    l[eos_id] += eos_bias;
  }

  // 5. Temperature
  if (temp > 1e-6) {
    for (auto& v : l) v /= temp;
  }

  // 6. Top-k
  if (k > 0 && k < vocab_size) {
    std::vector<double> sorted(l);
    std::nth_element(sorted.begin(), sorted.end() - k, sorted.end());
    double threshold = sorted[sorted.size() - k];
    for (auto& v : l) {
      if (v < threshold) v = -1e30;
    }
  }

  // 7. Softmax
  double max_val = *std::max_element(l.begin(), l.end());
  double sum = 0;
  for (auto& v : l) {
    v = std::exp(v - max_val);
    sum += v;
  }
  for (auto& v : l) v /= sum;

  // 8. Sample — rng_ is a member, seeded per-request in GenerateInternal
  std::discrete_distribution<int> dist(l.begin(), l.end());
  return dist(rng_);
}

bool TTSPipeline::DetectRepetition(const std::vector<int>& history,
                                    int min_repeat) {
  int n = (int)history.size();
  if (n < min_repeat) return false;
  // Check single-token repeat: last min_repeat tokens are identical
  int last = history[n - 1];
  int count = 0;
  for (int i = n - 1; i >= 0 && history[i] == last; --i) ++count;
  if (count >= min_repeat) return true;
  // Check 2-token pattern repeat: AB AB AB ...
  if (n >= min_repeat * 2) {
    int a = history[n - 2], b = history[n - 1];
    int pairs = 0;
    for (int i = n - 2; i >= 1; i -= 2) {
      if (history[i - 1] == a && history[i] == b) ++pairs;
      else break;
    }
    if (pairs >= min_repeat) return true;
  }
  return false;
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
  // Look up codec prefix embeddings from pre-loaded table (no ORT needed)
  int n_codec = (int)codec_ids.size();  // 6 (with lang) or 5 (without)
  std::vector<float> codec_prefix(n_codec * D);
  for (int i = 0; i < n_codec; ++i) {
    const float* e = CodecEmbedLookup((int)codec_ids[i]);
    VecCopy(codec_prefix.data() + i * D, e, D);
  }

  // codec_pad for decode loop
  const float* codec_pad_e = CodecEmbedLookup(cfg_.codec_pad_id);

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

  // Trailing text: body[1:], then tts_eos (NO codec_pad — matches official)
  int n_text = body_emb.empty() ? 0 : (int)(body_emb.size() / D);
  int n_trailing = (n_text > 1 ? n_text - 1 : 0) + 1;  // body[1:] + eos
  pf.trailing_text.resize(n_trailing * D);
  pf.n_trailing = n_trailing;

  int t = 0;
  for (int i = 1; i < n_text; ++i, ++t) {
    VecCopy(pf.trailing_text.data() + t * D, body_emb.data() + i * D, D);
  }
  // Last: tts_eos only (no codec_pad)
  VecCopy(pf.trailing_text.data() + t * D, tts_eos_e, D);

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

  // Seed RNG: 0 = random (time-based), >0 = fixed seed
  if (seed == 0) {
    rng_.seed(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  } else {
    rng_.seed(seed);
  }

  int D = cfg_.hidden_dim;

  // Build prefill
  auto pf = BuildPrefill(text, lang, speaker_embed, token_ids);
  std::cout << "  Prefill: " << pf.seq_len << " tokens" << std::endl;

  // Run prefill: try TRT unified engine first, fall back to ORT
  auto t0 = Clock::now();
  std::vector<float> logits(cfg_.vocab_size);
  std::vector<float> last_hidden(D);

  if (ort_->HasTalkerPrefill()) {
    // ORT prefill (loads ~1.7GB but reliable for any seq_len)
    auto pf_result = ort_->TalkerPrefill(pf.embeds.data(), pf.seq_len, D);
    result.prefill_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    // Seed TRT KV cache from ORT output
    std::vector<const float*> kv_ptrs;
    for (auto& kv : pf_result.kv_data) {
      kv_ptrs.push_back(kv.data());
    }
    talker_->SeedKV(kv_ptrs.data(), (int)kv_ptrs.size(), pf.seq_len);
    int last_pos = pf.seq_len - 1;
    VecCopy(logits.data(),
            pf_result.logits.data() + last_pos * cfg_.vocab_size,
            cfg_.vocab_size);
    VecCopy(last_hidden.data(),
            pf_result.last_hidden.data() + last_pos * D, D);
  } else {
    // TRT unified prefill (requires engine compiled with dynamic seq_len)
    auto pf_result = talker_->Prefill(pf.embeds.data(), pf.seq_len);
    result.prefill_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    // last_pos: engine may output last token only (logits.size()==vocab_size) or all tokens
    int last_pos = (int)pf_result.logits.size() / cfg_.vocab_size - 1;
    VecCopy(logits.data(),
            pf_result.logits.data() + last_pos * cfg_.vocab_size,
            cfg_.vocab_size);
    // last_hidden is always full sequence
    VecCopy(last_hidden.data(),
            pf_result.last_hidden.data() + (pf.seq_len - 1) * D, D);
  }



  // Decode loop
  std::vector<std::vector<int>> all_codes;
  std::vector<int> primary_history;  // for repetition penalty
  std::vector<double> dt_times, ct_times;

  // EOS bias: progressive boost, delayed until expected audio length
  // Each text token ≈ 3-4 frames at 12.5Hz. Start biasing after 3x trailing.
  const float kEosBiasBase = 5.0f;
  const float kEosBiasRate = 0.5f;   // per step after bias onset
  const float kEosBiasMax = 25.0f;
  int bias_onset = pf.n_trailing * 3;  // delay: let model generate audio first
  // Hard cap: max 10 frames per text token
  int text_based_max = std::max(50, pf.n_trailing * 10);
  int effective_max = std::min(max_frames, text_based_max);
  std::cerr << "  EOS params: n_trailing=" << pf.n_trailing
            << " bias_onset=" << bias_onset
            << " effective_max=" << effective_max << std::endl;

  for (int step = 0; step < effective_max; ++step) {
    // Compute progressive EOS bias (delayed start)
    float eos_bias = 0.0f;
    int steps_past_onset = step - bias_onset;
    if (steps_past_onset >= 0) {
      eos_bias = std::min(kEosBiasMax,
                          kEosBiasBase + steps_past_onset * kEosBiasRate);
    }

    // Sample primary code with repetition penalty + EOS bias
    int primary_code = SampleWithPenalty(
        logits.data(), cfg_.vocab_size,
        primary_history.data(), (int)primary_history.size(),
        50, 0.9f, step < 2, cfg_.codec_eos_token_id, eos_bias);
    if (primary_code == cfg_.codec_eos_token_id) {
      std::cout << "  EOS at step " << step << " (bias=" << eos_bias << ")" << std::endl;
      break;
    }

    // Repetition detection: force EOS if looping
    primary_history.push_back(primary_code);
    if (DetectRepetition(primary_history, 5)) {
      std::cout << "  Force EOS at step " << step << " (repetition detected)" << std::endl;
      break;
    }

    // Debug: print EOS logit and sampled token every 20 steps
    if (step % 20 == 0 || step < 3) {
      float eos_logit = logits[cfg_.codec_eos_token_id];
      float max_logit = *std::max_element(logits.data(), logits.data() + cfg_.vocab_size);
      int argmax = (int)(std::max_element(logits.data(), logits.data() + cfg_.vocab_size) - logits.data());
      std::cerr << "  [step " << step << "] sampled=" << primary_code
                << " eos_logit=" << eos_logit << " max_logit=" << max_logit
                << " argmax=" << argmax << " eos_bias=" << eos_bias << std::endl;
    }

    // Code predictor: 15 residual codes (GPU-resident context)
    auto t_cp = Clock::now();
    const float* primary_e_ptr = CodecEmbedLookup(primary_code);

    std::vector<float> codec_sum(D);
    VecCopy(codec_sum.data(), primary_e_ptr, D);

    std::vector<int> frame_codes = {primary_code};

    if (cp_kv_) {
      // --- Autoregressive CP (with optional CUDA Graph) ---
      // Always use RunFrameAutoregressive: CPU sampling + CUDA Graph for
      // enqueueV3 eliminates TRT dispatch overhead. RunFrameGPU (GPU sampling)
      // is 180ms due to TRT dispatch serialization — do not use.
      int n_groups_full = cfg_.num_code_groups - 1;
      int n_groups = cfg_.cp_active_groups;
      // Allocate full-size buffer so engine can zero-fill inactive slots
      std::vector<int> cp_codes(n_groups_full, 0);
      cp_kv_->RunFrameAutoregressive(
          last_hidden.data(), primary_e_ptr, cp_codes.data(),
          cp_embed_table_.data(), cp_embed_vocab_, n_groups);

      // Accumulate codec_sum from active residuals only; inactive slots
      // are zero (no embedding contribution, matching zero-fill semantics).
      for (int j = 0; j < n_groups_full; ++j) {
        int rc = cp_codes[j];
        frame_codes.push_back(rc);
        if (j >= n_groups) continue;  // inactive: code=0, skip embed add
        if (cp_embed_use_int8_) {
          float re_buf[1024];
          CPEmbedLookupINT8(j, rc, re_buf);
          VecAdd(codec_sum.data(), codec_sum.data(), re_buf, D);
        } else {
          const float* re = CPEmbedLookup(j, rc);
          VecAdd(codec_sum.data(), codec_sum.data(), re, D);
        }
      }
    } else {
      // --- Old context-copy CP engine path ---
      cp_->BeginFrame(last_hidden.data(), primary_e_ptr);

      for (int j = 0; j < cfg_.num_code_groups - 1; ++j) {
        std::vector<float> cp_logits(cfg_.cp_vocab);
        cp_->PredictGPU(j, cp_logits.data());

        int rc = Sample(cp_logits.data(), cfg_.cp_vocab);
        frame_codes.push_back(rc);

        if (cp_->has_embed_table()) {
          cp_->AppendEmbeddingFromTable(j, rc);
          if (cp_embed_use_int8_) {
            float re_buf[1024];
            CPEmbedLookupINT8(j, rc, re_buf);
            VecAdd(codec_sum.data(), codec_sum.data(), re_buf, D);
          } else {
            const float* re = CPEmbedLookup(j, rc);
            VecAdd(codec_sum.data(), codec_sum.data(), re, D);
          }
        } else {
          auto re = ort_->CPEmbed(rc, j);
          cp_->AppendEmbedding(re.data());
          VecAdd(codec_sum.data(), codec_sum.data(), re.data(), D);
        }
      }
    }
    ct_times.push_back(
        std::chrono::duration<double, std::milli>(Clock::now() - t_cp).count());
    all_codes.push_back(frame_codes);
    

    // Next talker input: codec_sum + trailing_text (or tts_pad after text)
    // Official: inputs_embeds = codec_sum + trailing_text[step] or + tts_pad_embed
    std::vector<float> next_emb(D);
    if (step < pf.n_trailing) {
      VecAdd(next_emb.data(), codec_sum.data(),
             pf.trailing_text.data() + step * D, D);
    } else {
      VecAdd(next_emb.data(), codec_sum.data(), pf.tts_pad_e.data(), D);
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

  // Dump primary codes for debugging
  std::cerr << "  PRIMARY_CODES=[";
  for (int f = 0; f < n; ++f) {
    if (f) std::cerr << ",";
    std::cerr << all_codes[f][0];
  }
  std::cerr << "]" << std::endl;

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
  if (trt_vocoder_) {
    result.audio = trt_vocoder_->Run(codes_t.data(), n, cfg_.num_code_groups);
  } else {
    result.audio = ort_->Vocoder(codes_t.data(), n, cfg_.num_code_groups);
  }
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
  size_t table_size = (size_t)n_layers * vocab * D;

  // --- Fast path: load pre-extracted binary (< 0.1s) ---
  std::string bin_path = sherpa_dir + "/cp_embed_fp32.bin";
  {
    std::ifstream f(bin_path, std::ios::binary | std::ios::ate);
    if (f.good()) {
      size_t fsize = f.tellg();
      size_t expected = table_size * sizeof(float);
      if (fsize == expected) {
        f.seekg(0);
        cp_embed_table_.resize(table_size);
        f.read(reinterpret_cast<char*>(cp_embed_table_.data()), expected);
        cp_embed_n_layers_ = n_layers;
        cp_embed_vocab_ = vocab;
        std::cout << "  cp_embed loaded from binary: " << bin_path
                  << " (" << expected / (1024*1024) << " MB)" << std::endl;
        // Upload to GPU — only for the active CP engine
        if (cp_kv_) {
          cp_kv_->LoadEmbedTable(cp_embed_table_.data(), n_layers, vocab, D);
        } else if (cp_) {
          cp_->LoadEmbedTable(cp_embed_table_.data(), n_layers, vocab, D);
        }
        // Try INT8 quantized version (replaces FP32 CPU copy if available)
        LoadCPEmbedTableINT8(sherpa_dir);
        return;
      } else {
        std::cerr << "  cp_embed binary size mismatch: " << fsize
                  << " vs expected " << expected << ", falling back to ORT" << std::endl;
      }
    }
  }

  // --- Slow path: compute via ORT (~15s) ---
  std::cout << "Pre-computing cp_embed table (" << n_layers << "×" << vocab
            << "×" << D << ")..." << std::endl;

  // Allocate flat table: [n_layers][vocab][D]
  std::vector<float> table(table_size);

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

  // Upload to GPU — only for the active CP engine
  if (cp_kv_) {
    cp_kv_->LoadEmbedTable(cp_embed_table_.data(), n_layers, vocab, D);
  } else if (cp_) {
    cp_->LoadEmbedTable(cp_embed_table_.data(), n_layers, vocab, D);
  }

  // Try INT8 quantized version (replaces FP32 CPU copy if available)
  LoadCPEmbedTableINT8(sherpa_dir);
}

const float* TTSPipeline::CPEmbedLookup(int layer, int token_id) const {
  size_t offset = ((size_t)layer * cp_embed_vocab_ + token_id) * cfg_.hidden_dim;
  return cp_embed_table_.data() + offset;
}

void TTSPipeline::LoadCPEmbedTableINT8(const std::string& sherpa_dir) {
  // Try to load pre-quantized INT8 files
  std::string int8_path = sherpa_dir + "/cp_embed_int8.bin";
  std::string scale_path = sherpa_dir + "/cp_embed_scales.bin";

  if (!std::ifstream(int8_path).good() || !std::ifstream(scale_path).good()) {
    std::cout << "  INT8 cp_embed not found, using FP32" << std::endl;
    return;
  }

  int n_layers = cfg_.num_code_groups - 1;  // 15
  int vocab = cfg_.cp_vocab;                 // 2048
  int D = cfg_.hidden_dim;                   // 1024

  // Load INT8 table: [n_layers * vocab * D] int8
  {
    std::ifstream f(int8_path, std::ios::binary | std::ios::ate);
    size_t expected = (size_t)n_layers * vocab * D;
    size_t fsize = f.tellg();
    if (fsize != expected) {
      std::cerr << "  INT8 cp_embed size mismatch: " << fsize << " vs " << expected << std::endl;
      return;
    }
    f.seekg(0);
    cp_embed_int8_table_.resize(expected);
    f.read(reinterpret_cast<char*>(cp_embed_int8_table_.data()), expected);
  }

  // Load scales: [n_layers * vocab] float32
  {
    std::ifstream f(scale_path, std::ios::binary | std::ios::ate);
    size_t expected = (size_t)n_layers * vocab * sizeof(float);
    size_t fsize = f.tellg();
    if (fsize != expected) {
      std::cerr << "  INT8 cp_embed scales size mismatch: " << fsize << " vs " << expected << std::endl;
      cp_embed_int8_table_.clear();
      return;
    }
    f.seekg(0);
    cp_embed_scales_.resize(n_layers * vocab);
    f.read(reinterpret_cast<char*>(cp_embed_scales_.data()), fsize);
  }

  cp_embed_use_int8_ = true;

  // Only free FP32 table if cp_kv_ is NOT in use.
  // cp_kv_->RunFrameAutoregressive takes cp_embed_table_.data() as a CPU pointer;
  // if cp_kv_ exists, we keep FP32 for its use. INT8 is used for direct lookups only.
  if (!cp_kv_) {
    cp_embed_table_.clear();
    cp_embed_table_.shrink_to_fit();
  }

  std::cout << "  INT8 cp_embed loaded: " << (n_layers * vocab * D) / (1024*1024)
            << "MB INT8 + " << (n_layers * vocab * 4) / 1024 << "KB scales"
            << " (saved ~" << (n_layers * vocab * D * 3) / (1024*1024) << "MB)" << std::endl;
}

void TTSPipeline::CPEmbedLookupINT8(int layer, int token_id, float* out) const {
  int D = cfg_.hidden_dim;
  size_t offset = ((size_t)layer * cp_embed_vocab_ + token_id) * D;
  size_t scale_idx = (size_t)layer * cp_embed_vocab_ + token_id;
  float scale = cp_embed_scales_[scale_idx];
  const int8_t* src = cp_embed_int8_table_.data() + offset;
  for (int i = 0; i < D; ++i) {
    out[i] = (float)src[i] * scale;
  }
}

void TTSPipeline::LoadCodecEmbedTable(const std::string& sherpa_dir) {
  int vocab = cfg_.vocab_size;  // 3072
  int D = cfg_.hidden_dim;      // 1024
  size_t table_size = (size_t)vocab * D;

  // --- Fast path: load pre-extracted binary ---
  std::string bin_path = sherpa_dir + "/codec_embed_fp32.bin";
  {
    std::ifstream f(bin_path, std::ios::binary | std::ios::ate);
    if (f.good()) {
      size_t fsize = f.tellg();
      size_t expected = table_size * sizeof(float);
      if (fsize == expected) {
        f.seekg(0);
        codec_embed_table_.resize(table_size);
        f.read(reinterpret_cast<char*>(codec_embed_table_.data()), expected);
        codec_embed_vocab_ = vocab;
        std::cout << "  codec_embed loaded from binary: " << bin_path
                  << " (" << expected / (1024*1024) << " MB)" << std::endl;
        return;
      } else {
        std::cerr << "  codec_embed binary size mismatch: " << fsize
                  << " vs expected " << expected << ", falling back to ORT" << std::endl;
      }
    }
  }

  // --- Slow path: single batch ORT call (fast enough, ~0.016s) ---
  std::cout << "Pre-computing codec_embed table (" << vocab << "x" << D
            << ")..." << std::endl;

  auto t0 = std::chrono::high_resolution_clock::now();
  // Batch: embed all tokens at once -> [1, vocab, D]
  std::vector<int64_t> all_ids(vocab);
  std::iota(all_ids.begin(), all_ids.end(), 0);
  codec_embed_table_ = ort_->CodecEmbed(all_ids);
  codec_embed_vocab_ = vocab;

  auto dt = std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - t0).count();
  std::cout << "  codec_embed table computed in " << dt << "s ("
            << vocab * D * 4 / (1024 * 1024) << " MB)" << std::endl;
}

const float* TTSPipeline::CodecEmbedLookup(int token_id) const {
  return codec_embed_table_.data() + (size_t)token_id * cfg_.hidden_dim;
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

void TTSPipeline::EnableProfiling(bool enable) {
  if (talker_) talker_->EnableProfiling(enable);
  if (cp_) cp_->EnableProfiling(enable);
  if (cp_kv_) cp_kv_->EnableProfiling(enable);
}

void TTSPipeline::EnableCudaGraph(bool enable) {
  if (talker_) talker_->EnableCudaGraph(enable);
  if (cp_kv_) cp_kv_->EnableCPCudaGraph(enable);
}

void TTSPipeline::PrintProfilingStats() {
  if (talker_ && talker_->stats().n_samples > 0) {
    auto& s = talker_->stats();
    std::cout << "\n  === TALKER DECODE PROFILING (" << s.n_samples
              << " steps) ===" << std::endl;
    std::cout << "  H2D:     avg=" << s.AvgH2D() << " ms, max=" << s.max_h2d
              << " ms" << std::endl;
    std::cout << "  Kernel:  avg=" << s.AvgKernel()
              << " ms, max=" << s.max_kernel << " ms" << std::endl;
    std::cout << "  D2H:     avg=" << s.AvgD2H() << " ms, max=" << s.max_d2h
              << " ms" << std::endl;
    std::cout << "  Total:   avg=" << s.AvgTotal()
              << " ms, max=" << s.max_total << " ms" << std::endl;
    std::cout << "  Overhead (bind+sync): avg=" << s.AvgOverhead() << " ms"
              << std::endl;
    talker_->ResetStats();
  }
  if (cp_ && cp_->stats().n_samples > 0) {
    auto& s = cp_->stats();
    std::cout << "\n  === CODE PREDICTOR PROFILING (" << s.n_samples
              << " steps) ===" << std::endl;
    std::cout << "  H2D:     avg=" << s.AvgH2D() << " ms, max=" << s.max_h2d
              << " ms" << std::endl;
    std::cout << "  Kernel:  avg=" << s.AvgKernel()
              << " ms, max=" << s.max_kernel << " ms" << std::endl;
    std::cout << "  D2H:     avg=" << s.AvgD2H() << " ms, max=" << s.max_d2h
              << " ms" << std::endl;
    std::cout << "  Total:   avg=" << s.AvgTotal()
              << " ms, max=" << s.max_total << " ms" << std::endl;
    std::cout << "  Overhead (bind+sync): avg=" << s.AvgOverhead() << " ms"
              << std::endl;
    cp_->ResetStats();
  }
  if (cp_kv_ && cp_kv_->stats().n_samples > 0) {
    auto& s = cp_kv_->stats();
    std::cout << "\n  === CP-KV PROFILING (" << s.n_samples << " frames) ==="
              << std::endl;
    std::cout << "  Kernel: avg=" << s.AvgKernel()
              << " ms, max=" << s.max_kernel << " ms" << std::endl;
    std::cout << "  D2H:    avg=" << s.AvgD2H() << " ms, max=" << s.max_d2h
              << " ms" << std::endl;
    std::cout << "  Total:  avg=" << s.AvgTotal()
              << " ms, max=" << s.max_total << " ms" << std::endl;
    cp_kv_->ResetStats();
  }
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

  if (config.seed == 0) {
    rng_.seed(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  } else {
    rng_.seed(config.seed);
  }
  int D = cfg_.hidden_dim;

  // Build prefill
  auto pf = BuildPrefill(text, lang, speaker_embed, token_ids);
  std::cout << "  Prefill: " << pf.seq_len << " tokens (streaming)"
            << std::endl;

  // Run prefill: ORT or TRT unified
  auto t0 = Clock::now();
  std::vector<float> logits(cfg_.vocab_size);
  std::vector<float> last_hidden(D);

  if (ort_->HasTalkerPrefill()) {
    auto pf_result = ort_->TalkerPrefill(pf.embeds.data(), pf.seq_len, D);
    double prefill_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    std::cout << "  Prefill: " << prefill_ms << " ms (ORT)" << std::endl;
    std::vector<const float*> kv_ptrs;
    for (auto& kv : pf_result.kv_data) kv_ptrs.push_back(kv.data());
    talker_->SeedKV(kv_ptrs.data(), (int)kv_ptrs.size(), pf.seq_len);
    int last_pos_pf = pf.seq_len - 1;
    VecCopy(logits.data(),
            pf_result.logits.data() + last_pos_pf * cfg_.vocab_size,
            cfg_.vocab_size);
    VecCopy(last_hidden.data(),
            pf_result.last_hidden.data() + last_pos_pf * D, D);
  } else {
    auto pf_result = talker_->Prefill(pf.embeds.data(), pf.seq_len);
    double prefill_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    std::cout << "  Prefill: " << prefill_ms << " ms (TRT unified)" << std::endl;
    // last_pos_pf: engine may output last token only (logits.size()==vocab_size) or all tokens
    int last_pos_pf = (int)pf_result.logits.size() / cfg_.vocab_size - 1;
    VecCopy(logits.data(),
            pf_result.logits.data() + last_pos_pf * cfg_.vocab_size,
            cfg_.vocab_size);
    // last_hidden is always full sequence
    VecCopy(last_hidden.data(),
            pf_result.last_hidden.data() + (pf.seq_len - 1) * D, D);
  }

  // Decode loop with chunked vocoder
  std::vector<std::vector<int>> all_codes;
  std::vector<int> primary_history;  // for repetition penalty

  // Sliding-window vocoder constants
  const int kVocContextFrames = 25;   // left context frames (official default)
  const int kSamplesPerFrame = 1920;  // 24000 Hz / 12.5 Hz codec rate

  int last_emitted_frames = 0;  // frame count at the last chunk emit

  // Determine chunk boundaries: first chunk is smaller for low TTFA
  int next_chunk_at = config.first_chunk_frames;

  // EOS bias: progressive boost, delayed until expected audio length
  const float kEosBiasBase = 5.0f;
  const float kEosBiasRate = 0.5f;
  const float kEosBiasMax = 25.0f;
  int bias_onset = pf.n_trailing * 3;
  int text_based_max = std::max(50, pf.n_trailing * 10);
  int effective_max = std::min(config.max_frames, text_based_max);
  std::cerr << "  EOS params: n_trailing=" << pf.n_trailing
            << " bias_onset=" << bias_onset
            << " effective_max=" << effective_max << std::endl;

  for (int step = 0; step < effective_max; ++step) {
    // Compute progressive EOS bias (delayed start)
    float eos_bias = 0.0f;
    int steps_past_onset = step - bias_onset;
    if (steps_past_onset >= 0) {
      eos_bias = std::min(kEosBiasMax,
                          kEosBiasBase + steps_past_onset * kEosBiasRate);
    }

    // Sample primary code with repetition penalty + EOS bias
    int primary_code = SampleWithPenalty(
        logits.data(), cfg_.vocab_size,
        primary_history.data(), (int)primary_history.size(),
        50, 0.9f, step < 2, cfg_.codec_eos_token_id, eos_bias);
    if (primary_code == cfg_.codec_eos_token_id) {
      std::cout << "  EOS at step " << step << " (bias=" << eos_bias << ")" << std::endl;
      break;
    }

    // Repetition detection: force EOS if looping
    primary_history.push_back(primary_code);
    if (DetectRepetition(primary_history, 5)) {
      std::cout << "  Force EOS at step " << step << " (repetition detected)" << std::endl;
      break;
    }

    // Code predictor: 15 residual codes
    const float* primary_e_ptr = CodecEmbedLookup(primary_code);

    std::vector<float> codec_sum(D);
    VecCopy(codec_sum.data(), primary_e_ptr, D);

    std::vector<int> frame_codes = {primary_code};

    if (cp_kv_) {
      // --- GPU-resident autoregressive CP (dual-context, no CPU sync) ---
      int n_groups_full = cfg_.num_code_groups - 1;
      int n_groups = cfg_.cp_active_groups;
      std::vector<int> cp_codes(n_groups_full, 0);
      cp_kv_->RunFrameAutoregressive(
          last_hidden.data(), primary_e_ptr, cp_codes.data(),
          cp_embed_table_.data(), cp_embed_vocab_, n_groups);

      for (int j = 0; j < n_groups_full; ++j) {
        int rc = cp_codes[j];
        frame_codes.push_back(rc);
        if (j >= n_groups) continue;  // inactive: code=0, skip embed add
        if (cp_embed_use_int8_) {
          float re_buf[1024];
          CPEmbedLookupINT8(j, rc, re_buf);
          VecAdd(codec_sum.data(), codec_sum.data(), re_buf, D);
        } else {
          const float* re = CPEmbedLookup(j, rc);
          VecAdd(codec_sum.data(), codec_sum.data(), re, D);
        }
      }
    } else {
      cp_->BeginFrame(last_hidden.data(), primary_e_ptr);

      for (int j = 0; j < cfg_.num_code_groups - 1; ++j) {
        std::vector<float> cp_logits(cfg_.cp_vocab);
        cp_->PredictGPU(j, cp_logits.data());

        int rc = Sample(cp_logits.data(), cfg_.cp_vocab);
        frame_codes.push_back(rc);

        if (cp_->has_embed_table()) {
          cp_->AppendEmbeddingFromTable(j, rc);
          if (cp_embed_use_int8_) {
            float re_buf[1024];
            CPEmbedLookupINT8(j, rc, re_buf);
            VecAdd(codec_sum.data(), codec_sum.data(), re_buf, D);
          } else {
            const float* re = CPEmbedLookup(j, rc);
            VecAdd(codec_sum.data(), codec_sum.data(), re, D);
          }
        } else {
          auto re = ort_->CPEmbed(rc, j);
          cp_->AppendEmbedding(re.data());
          VecAdd(codec_sum.data(), codec_sum.data(), re.data(), D);
        }
      }
    }
    all_codes.push_back(frame_codes);

    // Next talker input: codec_sum + trailing_text (or tts_pad after text)
    std::vector<float> next_emb(D);
    if (step < pf.n_trailing) {
      VecAdd(next_emb.data(), codec_sum.data(),
             pf.trailing_text.data() + step * D, D);
    } else {
      VecAdd(next_emb.data(), codec_sum.data(), pf.tts_pad_e.data(), D);
    }

    // Talker decode
    talker_->DecodeStep(next_emb.data(), logits.data(), last_hidden.data());

    // Check if we should emit a chunk
    int n = (int)all_codes.size();
    if (n >= next_chunk_at) {
      // Sliding-window vocoder: pass [window_start..n] where window_start
      // is at most kVocContextFrames before the new frames in this chunk.
      // Scheme A: enqueue to async worker instead of running synchronously.
      int prev_boundary = last_emitted_frames;  // start of new frames
      int window_start = std::max(0, prev_boundary - kVocContextFrames);
      int window_len = n - window_start;

      VocWork w;
      w.codes.resize((size_t)window_len * cfg_.num_code_groups);
      for (int f = 0; f < window_len; ++f) {
        for (int g = 0; g < cfg_.num_code_groups; ++g) {
          w.codes[(size_t)f * cfg_.num_code_groups + g] =
              all_codes[window_start + f][g];
        }
      }
      w.window_len = window_len;
      w.num_code_groups = cfg_.num_code_groups;
      w.skip_samples =
          (size_t)(prev_boundary - window_start) * kSamplesPerFrame;
      w.total_frames = n;
      w.is_final = false;
      w.callback = callback;

      ++voc_inflight_;
      {
        std::lock_guard<std::mutex> lk(voc_mutex_);
        voc_queue_.push(std::move(w));
      }
      voc_cv_.notify_one();

      last_emitted_frames = n;
      std::cout << "  Chunk enqueued: window=[" << window_start << ".." << n
                << "] for async vocode" << std::endl;

      // After first chunk, use regular chunk size
      next_chunk_at = n + config.chunk_frames;
    }

    if ((step + 1) % 10 == 0 && n < next_chunk_at) {
      std::cout << "  Frame " << step + 1 << std::endl;
    }
  }

  // Final chunk: sliding-window vocoder for remaining codes
  // Scheme A: enqueue final chunk to async worker, then wait for queue drain.
  int n = (int)all_codes.size();
  if (n > 0) {
    int prev_boundary = last_emitted_frames;
    int window_start = std::max(0, prev_boundary - kVocContextFrames);
    int window_len = n - window_start;

    VocWork w;
    w.codes.resize((size_t)window_len * cfg_.num_code_groups);
    for (int f = 0; f < window_len; ++f) {
      for (int g = 0; g < cfg_.num_code_groups; ++g) {
        w.codes[(size_t)f * cfg_.num_code_groups + g] =
            all_codes[window_start + f][g];
      }
    }
    w.window_len = window_len;
    w.num_code_groups = cfg_.num_code_groups;
    w.skip_samples =
        (size_t)(prev_boundary - window_start) * kSamplesPerFrame;
    w.total_frames = n;
    w.is_final = true;
    w.callback = callback;

    ++voc_inflight_;
    {
      std::lock_guard<std::mutex> lk(voc_mutex_);
      voc_queue_.push(std::move(w));
    }
    voc_cv_.notify_one();
    std::cout << "  Final chunk enqueued: window=[" << window_start << ".."
              << n << "] for async vocode" << std::endl;
  } else {
    // No frames at all — emit empty final chunk synchronously
    StreamChunk chunk;
    chunk.total_frames = 0;
    chunk.is_final = true;
    callback(chunk);
  }

  // Wait for vocoder worker to drain all outstanding work before returning.
  {
    std::unique_lock<std::mutex> lk(voc_mutex_);
    voc_empty_cv_.wait(lk, [&] { return voc_inflight_.load() == 0; });
  }

  std::cout << "  Streaming complete: " << n << " total frames" << std::endl;
}
