// tts_ort_models.cpp — ORT cold-path model implementations
#include "tts_ort_models.h"

#include <cassert>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

Ort::SessionOptions ORTModels::MakeSessionOptions(int device_id) {
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(2);
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  // Append CUDA EP
  OrtCUDAProviderOptions cuda_opts;
  cuda_opts.device_id = device_id;
  cuda_opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
  opts.AppendExecutionProvider_CUDA(cuda_opts);

  return opts;
}

ORTModels::ORTModels(const std::string& model_dir,
                     const std::string& sherpa_dir, int device_id)
    : env_(ORT_LOGGING_LEVEL_WARNING, "qwen3tts"),
      mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
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

  text_project_ = load(sherpa_dir + "/text_project.onnx");
  codec_embed_ = load(sherpa_dir + "/codec_embed.onnx");
  cp_embed_ = load(sherpa_dir + "/code_predictor_embed.onnx");
  talker_prefill_ = load(sherpa_dir + "/talker_prefill.onnx");

  // Vocoder: try multiple names/paths
  if (fs::exists(sherpa_dir + "/vocoder.onnx"))
    vocoder_ = load(sherpa_dir + "/vocoder.onnx");
  else if (fs::exists(sherpa_dir + "/tokenizer12hz_decode.onnx"))
    vocoder_ = load(sherpa_dir + "/tokenizer12hz_decode.onnx");
  else if (fs::exists(model_dir + "/vocoder.onnx"))
    vocoder_ = load(model_dir + "/vocoder.onnx");
  else
    vocoder_ = load(model_dir + "/tokenizer12hz_decode.onnx");

  // Voice clone models (optional)
  speaker_encoder_ = load(sherpa_dir + "/speaker_encoder.onnx");
  tokenizer_encode_ = load(sherpa_dir + "/tokenizer12hz_encode.onnx");

  std::cout << "ORT models loaded." << std::endl;
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

std::vector<float> ORTModels::TextProject(
    const std::vector<int64_t>& input_ids) {
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
  assert(talker_prefill_);
  PrefillResult result;

  int64_t shape[] = {1, seq_len, hidden_dim};
  size_t count = 1 * seq_len * hidden_dim;
  auto val = Ort::Value::CreateTensor<float>(mem_info_, const_cast<float*>(embeds),
                                             count, shape, 3);

  // Get all output names
  size_t n_outputs = talker_prefill_->GetOutputCount();
  std::vector<Ort::AllocatedStringPtr> out_name_ptrs;
  std::vector<const char*> out_names;
  for (size_t i = 0; i < n_outputs; ++i) {
    out_name_ptrs.push_back(
        talker_prefill_->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions()));
    out_names.push_back(out_name_ptrs.back().get());
  }

  auto in_name = talker_prefill_->GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
  const char* in_ptr = in_name.get();

  auto outputs = talker_prefill_->Run(Ort::RunOptions{nullptr}, &in_ptr, &val,
                                       1, out_names.data(), n_outputs);

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
