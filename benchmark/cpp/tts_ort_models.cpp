// tts_ort_models.cpp — ORT cold-path model implementations
#include "tts_ort_models.h"

#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

Ort::SessionOptions ORTModels::MakeCPUSessionOptions() {
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(4);
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
  // CPU-only: no CUDA EP to avoid GPU memory usage
  return opts;
}

Ort::SessionOptions ORTModels::MakeSessionOptions(int device_id) {
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(2);
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

  // Append CUDA EP
  OrtCUDAProviderOptions cuda_opts;
  cuda_opts.device_id = device_id;
  cuda_opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
  opts.AppendExecutionProvider_CUDA(cuda_opts);

  return opts;
}

ORTModels::ORTModels(const std::string& model_dir,
                     const std::string& sherpa_dir,
                     ORTSkipFlags flags,
                     int device_id)
    : env_(ORT_LOGGING_LEVEL_WARNING, "qwen3tts"),
      mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      sherpa_dir_(sherpa_dir),
      device_id_(device_id) {
  auto opts = MakeSessionOptions(device_id);

  std::cout << "Loading ORT models..." << std::endl;

  auto load = [&](const std::string& path) -> std::unique_ptr<Ort::Session> {
    if (!fs::exists(path)) {
      std::cerr << "  Skipped (not found): " << path << std::endl;
      return nullptr;
    }
    auto s = std::make_unique<Ort::Session>(env_, path.c_str(), opts);
    std::cout << "  Loaded: " << path << std::endl;
    return s;
  };

  // Try new split text_embed path first: FP16 binary + projection ONNX
  {
    std::string embed_path = sherpa_dir + "/text_embed_fp16.bin";
    std::string proj_path = sherpa_dir + "/text_projection_only.onnx";
    if (fs::exists(embed_path) && fs::exists(proj_path)) {
      // Load FP16 embedding table
      std::ifstream ef(embed_path, std::ios::binary | std::ios::ate);
      size_t fsize = ef.tellg();
      ef.seekg(0);
      text_embed_table_.resize(fsize / 2);
      ef.read(reinterpret_cast<char*>(text_embed_table_.data()), fsize);
      text_embed_vocab_ = 151936;  // Qwen3 vocab
      text_embed_dim_ = (int)(fsize / 2 / text_embed_vocab_);
      std::cout << "  Loaded text_embed FP16: " << embed_path
                << " (" << fsize / (1024*1024) << " MB)" << std::endl;
      // Load projection ONNX
      text_projection_ = std::make_unique<Ort::Session>(env_, proj_path.c_str(), opts);
      std::cout << "  Loaded text_projection: " << proj_path << std::endl;
      has_split_text_embed_ = true;
    } else {
      // Fallback to old combined text_project.onnx
      text_project_ = load(sherpa_dir + "/text_project.onnx");
    }
  }
  if (flags.skip_codec_embed) {
    std::cout << "  Skipped codec_embed ORT (pre-extracted binary available)" << std::endl;
  } else {
    codec_embed_ = load(sherpa_dir + "/codec_embed.onnx");
  }

  if (flags.skip_cp_embed) {
    std::cout << "  Skipped cp_embed ORT (pre-extracted binary available)" << std::endl;
  } else {
    cp_embed_ = load(sherpa_dir + "/code_predictor_embed.onnx");
  }

  // talker_prefill: skip if TRT prefill engine is active
  if (flags.skip_talker_prefill) {
    std::cout << "  Skipped talker_prefill ORT (TRT active)" << std::endl;
  } else {
    // Load talker_prefill with CPU-only to avoid GPU OOM
    std::string pfill_path = sherpa_dir + "/talker_prefill.onnx";
    if (fs::exists(pfill_path)) {
      auto cpu_opts = MakeCPUSessionOptions();
      talker_prefill_ = std::make_unique<Ort::Session>(env_, pfill_path.c_str(), cpu_opts);
      std::cout << "  Loaded (CPU): " << pfill_path << std::endl;
    } else {
      std::cerr << "  Skipped (not found): " << pfill_path << std::endl;
    }
    if (!talker_prefill_) {
      std::cout << "  talker_prefill.onnx not found — will use TRT unified prefill"
                << std::endl;
    }
  }

  // Vocoder: skip if TRT vocoder engine is active
  if (flags.skip_vocoder) {
    std::cout << "  Skipped vocoder ORT (TRT active)" << std::endl;
  } else {
    if (fs::exists(sherpa_dir + "/vocoder.onnx"))
      vocoder_ = load(sherpa_dir + "/vocoder.onnx");
    else if (fs::exists(sherpa_dir + "/tokenizer12hz_decode.onnx"))
      vocoder_ = load(sherpa_dir + "/tokenizer12hz_decode.onnx");
    else if (fs::exists(model_dir + "/vocoder.onnx"))
      vocoder_ = load(model_dir + "/vocoder.onnx");
    else
      vocoder_ = load(model_dir + "/tokenizer12hz_decode.onnx");
  }

  // Voice clone models: load eagerly or defer
  if (flags.lazy_speaker_encoder) {
    std::cout << "  speaker_encoder: deferred (lazy)" << std::endl;
  } else {
    speaker_encoder_ = load(sherpa_dir + "/speaker_encoder.onnx");
  }

  if (flags.lazy_tokenizer_encode) {
    std::cout << "  tokenizer_encode: deferred (lazy)" << std::endl;
  } else {
    tokenizer_encode_ = load(sherpa_dir + "/tokenizer12hz_encode.onnx");
  }

  std::cout << "ORT models loaded." << std::endl;
}

void ORTModels::LoadSpeakerEncoder() {
  if (speaker_encoder_) return;  // already loaded
  std::string path = sherpa_dir_ + "/speaker_encoder.onnx";
  if (!fs::exists(path)) {
    std::cerr << "  speaker_encoder.onnx not found: " << path << std::endl;
    return;
  }
  auto opts = MakeSessionOptions(device_id_);
  speaker_encoder_ = std::make_unique<Ort::Session>(env_, path.c_str(), opts);
  std::cout << "  Lazy-loaded: " << path << std::endl;
}

void ORTModels::LoadTokenizerEncode() {
  if (tokenizer_encode_) return;  // already loaded
  std::string path = sherpa_dir_ + "/tokenizer12hz_encode.onnx";
  if (!fs::exists(path)) {
    std::cerr << "  tokenizer12hz_encode.onnx not found: " << path << std::endl;
    return;
  }
  auto opts = MakeSessionOptions(device_id_);
  tokenizer_encode_ = std::make_unique<Ort::Session>(env_, path.c_str(), opts);
  std::cout << "  Lazy-loaded: " << path << std::endl;
}

// Helper: run session with single input, return first output as float vector
static std::vector<float> RunSingle(Ort::Session* sess,
                                    const char* input_name,
                                    Ort::Value& input_val) {
  const char* output_names[] = {sess->GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions()).get()};
  // We need stable pointers for output names
  auto out_name = sess->GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
  const char* out_name_ptr = out_name.get();

  auto results =
      sess->Run(Ort::RunOptions{nullptr}, &input_name, &input_val, 1,
                &out_name_ptr, 1);
  auto& tensor = results[0];
  auto info = tensor.GetTensorTypeAndShapeInfo();
  size_t count = info.GetElementCount();
  const float* data = tensor.GetTensorData<float>();
  return std::vector<float>(data, data + count);
}

// FP16→FP32 conversion helper
static inline float fp16_to_fp32(uint16_t h) {
  uint32_t sign = (h >> 15) & 1;
  uint32_t exp  = (h >> 10) & 0x1F;
  uint32_t mant =  h        & 0x3FF;
  uint32_t bits;
  if (exp == 0x1F)      bits = (sign << 31) | (0xFF << 23) | (mant << 13);
  else if (exp == 0)    bits = sign << 31;
  else                  bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

std::vector<float> ORTModels::TextProject(
    const std::vector<int64_t>& input_ids) {
  if (has_split_text_embed_) {
    // New path: FP16 embedding lookup + projection ONNX
    int T = (int)input_ids.size();
    int D = text_embed_dim_;
    // Gather: [T, D] float32 from FP16 table
    std::vector<float> embeds(T * D);
    for (int t = 0; t < T; ++t) {
      int idx = (int)input_ids[t];
      if (idx < 0 || idx >= text_embed_vocab_) idx = 0;
      const uint16_t* src = text_embed_table_.data() + (size_t)idx * D;
      for (int d = 0; d < D; ++d) {
        embeds[t * D + d] = fp16_to_fp32(src[d]);
      }
    }
    // Run projection: [1, T, D] → [1, T, D_out]
    int64_t shape[] = {1, (int64_t)T, (int64_t)D};
    auto val = Ort::Value::CreateTensor<float>(mem_info_, embeds.data(),
                                               embeds.size(), shape, 3);
    auto in_name = text_projection_->GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
    auto out_name = text_projection_->GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
    const char* in_ptr = in_name.get();
    const char* out_ptr = out_name.get();
    auto results = text_projection_->Run(Ort::RunOptions{nullptr}, &in_ptr, &val, 1, &out_ptr, 1);
    auto info = results[0].GetTensorTypeAndShapeInfo();
    size_t count = info.GetElementCount();
    const float* data = results[0].GetTensorData<float>();
    return std::vector<float>(data, data + count);
  }

  // Old path: combined text_project.onnx
  assert(text_project_);
  int64_t shape[] = {1, (int64_t)input_ids.size()};
  auto val = Ort::Value::CreateTensor<int64_t>(mem_info_, const_cast<int64_t*>(input_ids.data()),
                                                input_ids.size(), shape, 2);
  auto in_name = text_project_->GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
  auto out_name = text_project_->GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
  const char* in_ptr = in_name.get();
  const char* out_ptr = out_name.get();
  auto results = text_project_->Run(Ort::RunOptions{nullptr}, &in_ptr, &val, 1, &out_ptr, 1);
  auto info = results[0].GetTensorTypeAndShapeInfo();
  size_t count = info.GetElementCount();
  const float* data = results[0].GetTensorData<float>();
  return std::vector<float>(data, data + count);
}

std::vector<float> ORTModels::CodecEmbed(
    const std::vector<int64_t>& token_ids) {
  assert(codec_embed_);
  int64_t shape[] = {1, (int64_t)token_ids.size()};
  auto val = Ort::Value::CreateTensor<int64_t>(mem_info_, const_cast<int64_t*>(token_ids.data()),
                                                token_ids.size(), shape, 2);
  auto in_name = codec_embed_->GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
  auto out_name = codec_embed_->GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
  const char* in_ptr = in_name.get();
  const char* out_ptr = out_name.get();
  auto results = codec_embed_->Run(Ort::RunOptions{nullptr}, &in_ptr, &val, 1, &out_ptr, 1);
  auto info = results[0].GetTensorTypeAndShapeInfo();
  size_t count = info.GetElementCount();
  const float* data = results[0].GetTensorData<float>();
  return std::vector<float>(data, data + count);
}

std::vector<float> ORTModels::CPEmbed(int64_t token_id, int64_t layer_idx) {
  assert(cp_embed_);
  int64_t tid_shape[] = {1, 1};
  int64_t lid_shape[] = {1};

  auto tid_val = Ort::Value::CreateTensor<int64_t>(mem_info_, &token_id, 1, tid_shape, 2);
  auto lid_val = Ort::Value::CreateTensor<int64_t>(mem_info_, &layer_idx, 1, lid_shape, 1);

  auto in0 = cp_embed_->GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
  auto in1 = cp_embed_->GetInputNameAllocated(1, Ort::AllocatorWithDefaultOptions());
  auto out0 = cp_embed_->GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());

  const char* in_names[] = {in0.get(), in1.get()};
  const char* out_names[] = {out0.get()};
  Ort::Value inputs[] = {std::move(tid_val), std::move(lid_val)};

  auto results = cp_embed_->Run(Ort::RunOptions{nullptr}, in_names, inputs, 2,
                                 out_names, 1);
  auto info = results[0].GetTensorTypeAndShapeInfo();
  size_t count = info.GetElementCount();
  const float* data = results[0].GetTensorData<float>();
  return std::vector<float>(data, data + count);
}

ORTModels::PrefillResult ORTModels::TalkerPrefill(const float* embeds,
                                                   int seq_len,
                                                   int hidden_dim) {
  assert(talker_prefill_ &&
         "TalkerPrefill called but ORT session not loaded. "
         "Use TRTTalkerEngine::Prefill() with unified engine instead.");
  PrefillResult result;

  // Build inputs dynamically based on what the model expects
  size_t n_inputs = talker_prefill_->GetInputCount();
  std::vector<Ort::AllocatedStringPtr> in_name_ptrs;
  std::vector<const char*> in_names;
  for (size_t i = 0; i < n_inputs; ++i) {
    in_name_ptrs.push_back(
        talker_prefill_->GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions()));
    in_names.push_back(in_name_ptrs.back().get());
  }

  // Prepare tensors
  int64_t emb_shape[] = {1, seq_len, hidden_dim};
  size_t emb_count = 1 * seq_len * hidden_dim;
  // attention_mask: all ones [1, T]
  std::vector<int64_t> attn_mask(seq_len, 1);
  int64_t mask_shape[] = {1, (int64_t)seq_len};

  std::vector<Ort::Value> inputs;
  for (size_t i = 0; i < n_inputs; ++i) {
    std::string name = in_names[i];
    if (name == "inputs_embeds") {
      inputs.push_back(Ort::Value::CreateTensor<float>(
          mem_info_, const_cast<float*>(embeds), emb_count, emb_shape, 3));
    } else if (name == "attention_mask") {
      inputs.push_back(Ort::Value::CreateTensor<int64_t>(
          mem_info_, attn_mask.data(), seq_len, mask_shape, 2));
    } else {
      std::cerr << "[TTS] Unknown prefill input: " << name << std::endl;
      int64_t dummy = 0;
      int64_t dummy_shape[] = {1};
      inputs.push_back(Ort::Value::CreateTensor<int64_t>(
          mem_info_, &dummy, 1, dummy_shape, 1));
    }
  }

  // Get all output names
  size_t n_outputs = talker_prefill_->GetOutputCount();
  std::vector<Ort::AllocatedStringPtr> out_name_ptrs;
  std::vector<const char*> out_names;
  for (size_t i = 0; i < n_outputs; ++i) {
    out_name_ptrs.push_back(
        talker_prefill_->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions()));
    out_names.push_back(out_name_ptrs.back().get());
  }

  auto outputs = talker_prefill_->Run(Ort::RunOptions{nullptr}, in_names.data(),
                                       inputs.data(), n_inputs,
                                       out_names.data(), n_outputs);

  // Parse outputs: logits, last_hidden, past_key_0, past_value_0, ...
  for (size_t i = 0; i < n_outputs; ++i) {
    auto info = outputs[i].GetTensorTypeAndShapeInfo();
    size_t elem_count = info.GetElementCount();
    const float* data = outputs[i].GetTensorData<float>();
    std::string name = out_names[i];

    if (name == "logits") {
      result.logits.assign(data, data + elem_count);
    } else if (name == "last_hidden") {
      result.last_hidden.assign(data, data + elem_count);
    } else if (name.find("past_") == 0) {
      result.kv_data.emplace_back(data, data + elem_count);
    }
  }
  result.seq_len = seq_len;
  return result;
}

std::vector<float> ORTModels::Vocoder(const int64_t* codes, int n_frames,
                                      int n_groups) {
  if (!vocoder_) return {};

  int64_t shape[] = {1, n_frames, n_groups};
  size_t count = 1 * n_frames * n_groups;
  auto val = Ort::Value::CreateTensor<int64_t>(mem_info_, const_cast<int64_t*>(codes),
                                                count, shape, 3);

  auto in_name = vocoder_->GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
  auto out_name = vocoder_->GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
  const char* in_ptr = in_name.get();
  const char* out_ptr = out_name.get();

  auto results = vocoder_->Run(Ort::RunOptions{nullptr}, &in_ptr, &val, 1,
                                &out_ptr, 1);
  auto info = results[0].GetTensorTypeAndShapeInfo();
  size_t elem_count = info.GetElementCount();
  const float* data = results[0].GetTensorData<float>();
  return std::vector<float>(data, data + elem_count);
}

std::vector<float> ORTModels::SpeakerEncode(const float* mel, int mel_frames) {
  LoadSpeakerEncoder();
  if (!speaker_encoder_) return {};

  int64_t shape[] = {1, mel_frames, 128};
  size_t count = 1 * mel_frames * 128;
  auto val = Ort::Value::CreateTensor<float>(mem_info_, const_cast<float*>(mel),
                                             count, shape, 3);

  auto in_name = speaker_encoder_->GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
  auto out_name = speaker_encoder_->GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
  const char* in_ptr = in_name.get();
  const char* out_ptr = out_name.get();

  auto results = speaker_encoder_->Run(Ort::RunOptions{nullptr}, &in_ptr, &val,
                                        1, &out_ptr, 1);
  auto info = results[0].GetTensorTypeAndShapeInfo();
  size_t elem_count = info.GetElementCount();
  const float* data = results[0].GetTensorData<float>();
  return std::vector<float>(data, data + elem_count);
}

std::vector<int64_t> ORTModels::TokenizerEncode(const float* audio,
                                                 int num_samples) {
  LoadTokenizerEncode();
  if (!tokenizer_encode_) return {};

  int64_t shape[] = {1, num_samples};
  size_t count = num_samples;
  auto val = Ort::Value::CreateTensor<float>(mem_info_, const_cast<float*>(audio),
                                             count, shape, 2);

  auto in_name = tokenizer_encode_->GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
  auto out_name = tokenizer_encode_->GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
  const char* in_ptr = in_name.get();
  const char* out_ptr = out_name.get();

  auto results = tokenizer_encode_->Run(Ort::RunOptions{nullptr}, &in_ptr,
                                         &val, 1, &out_ptr, 1);
  auto info = results[0].GetTensorTypeAndShapeInfo();
  size_t elem_count = info.GetElementCount();
  const int64_t* data = results[0].GetTensorData<int64_t>();
  return std::vector<int64_t>(data, data + elem_count);
}
