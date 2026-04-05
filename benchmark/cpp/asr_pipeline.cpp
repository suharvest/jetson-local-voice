// asr_pipeline.cpp — C++ ASR pipeline implementation
#include "asr_pipeline.h"

#include <cassert>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Session options with CUDA EP
// ---------------------------------------------------------------------------
Ort::SessionOptions ASRPipeline::MakeSessionOptions(int device_id) {
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(2);
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  OrtCUDAProviderOptions cuda_opts;
  cuda_opts.device_id = device_id;
  cuda_opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
  opts.AppendExecutionProvider_CUDA(cuda_opts);

  return opts;
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
ASRPipeline::ASRPipeline(const std::string& model_dir,
                         const std::string& engine_path, int device_id)
    : env_(ORT_LOGGING_LEVEL_WARNING, "qwen3asr"),
      mem_info_(
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
  auto opts = MakeSessionOptions(device_id);

  std::cout << "[ASR] Loading ORT models from " << model_dir << std::endl;

  // Encoder
  std::string enc_path = model_dir + "/encoder.onnx";
  if (fs::exists(enc_path)) {
    encoder_ = std::make_unique<Ort::Session>(env_, enc_path.c_str(), opts);
    std::cout << "[ASR]   Encoder loaded: " << enc_path << std::endl;
  } else {
    std::cerr << "[ASR]   Encoder not found: " << enc_path << std::endl;
  }

  // Prefill
  for (auto& name : {"decoder_prefill.onnx", "decoder_init.onnx"}) {
    std::string pf_path = model_dir + "/" + name;
    if (fs::exists(pf_path)) {
      prefill_ = std::make_unique<Ort::Session>(env_, pf_path.c_str(), opts);
      std::cout << "[ASR]   Prefill loaded: " << pf_path << std::endl;
      break;
    }
  }

  // Embed tokens (FP16 binary → FP32)
  std::string emb_path = model_dir + "/embed_tokens.bin";
  if (fs::exists(emb_path)) {
    std::ifstream f(emb_path, std::ios::binary | std::ios::ate);
    size_t file_bytes = f.tellg();
    f.seekg(0);

    // FP16: 2 bytes per element
    size_t n_fp16 = file_bytes / 2;
    std::vector<uint16_t> fp16_data(n_fp16);
    f.read(reinterpret_cast<char*>(fp16_data.data()), file_bytes);

    // Convert FP16 → FP32
    embed_table_.resize(n_fp16);
    for (size_t i = 0; i < n_fp16; ++i) {
      // IEEE 754 half-precision to single-precision
      uint16_t h = fp16_data[i];
      uint32_t sign = (h >> 15) & 0x1;
      uint32_t exp = (h >> 10) & 0x1F;
      uint32_t frac = h & 0x3FF;
      uint32_t f32;
      if (exp == 0) {
        if (frac == 0) {
          f32 = sign << 31;
        } else {
          // Subnormal: normalize
          exp = 1;
          while (!(frac & 0x400)) {
            frac <<= 1;
            exp--;
          }
          frac &= 0x3FF;
          f32 = (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13);
        }
      } else if (exp == 31) {
        f32 = (sign << 31) | 0x7F800000 | (frac << 13);
      } else {
        f32 = (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13);
      }
      float val;
      std::memcpy(&val, &f32, 4);
      embed_table_[i] = val;
    }

    int detected_vocab = (int)(n_fp16 / hidden_dim_);
    std::cout << "[ASR]   Embed tokens loaded: " << detected_vocab << " x "
              << hidden_dim_ << " (" << file_bytes / 1024 / 1024 << " MB FP16)"
              << std::endl;
  }

  // TRT decoder engine
  if (fs::exists(engine_path)) {
    decoder_ = std::make_unique<TRTTalkerEngine>(engine_path, n_layers_,
                                                  hidden_dim_, n_heads_,
                                                  head_dim_, vocab_size_,
                                                  max_seq_);
    std::cout << "[ASR]   TRT decoder loaded: " << engine_path << std::endl;
  } else {
    std::cerr << "[ASR]   TRT engine not found: " << engine_path << std::endl;
  }

  std::cout << "[ASR] Pipeline ready." << std::endl;
}

// ---------------------------------------------------------------------------
// EmbedLookup
// ---------------------------------------------------------------------------
const float* ASRPipeline::EmbedLookup(int64_t token_id) const {
  assert(token_id >= 0 && token_id < vocab_size_);
  return embed_table_.data() + token_id * hidden_dim_;
}

// ---------------------------------------------------------------------------
// RunEncoder
// ---------------------------------------------------------------------------
ASRPipeline::EncoderOutput ASRPipeline::RunEncoder(const float* mel,
                                                    int mel_len) {
  assert(encoder_);
  EncoderOutput out;

  // mel shape: [1, 128, T]
  int64_t shape[] = {1, 128, (int64_t)mel_len};
  size_t count = 1 * 128 * mel_len;
  auto val = Ort::Value::CreateTensor<float>(
      mem_info_, const_cast<float*>(mel), count, shape, 3);

  auto in_name =
      encoder_->GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
  auto out_name =
      encoder_->GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
  const char* in_ptr = in_name.get();
  const char* out_ptr = out_name.get();

  auto results =
      encoder_->Run(Ort::RunOptions{nullptr}, &in_ptr, &val, 1, &out_ptr, 1);

  auto info = results[0].GetTensorTypeAndShapeInfo();
  auto shape_out = info.GetShape();
  size_t elem_count = info.GetElementCount();
  const float* data = results[0].GetTensorData<float>();

  out.features.assign(data, data + elem_count);
  out.audio_len = (int)shape_out[1];  // [1, T', 1024]

  return out;
}

// ---------------------------------------------------------------------------
// RunPrefill
// ---------------------------------------------------------------------------
ASRPipeline::PrefillOutput ASRPipeline::RunPrefill(
    const std::vector<int64_t>& input_ids, const float* audio_features,
    int audio_len, int audio_offset) {
  assert(prefill_);
  PrefillOutput out;

  int seq_len = (int)input_ids.size();

  // Build position_ids [1, S]: 0, 1, 2, ...
  std::vector<int64_t> position_ids(seq_len);
  for (int i = 0; i < seq_len; ++i) position_ids[i] = i;

  // Create tensors
  int64_t ids_shape[] = {1, (int64_t)seq_len};
  auto ids_val = Ort::Value::CreateTensor<int64_t>(
      mem_info_, const_cast<int64_t*>(input_ids.data()), seq_len, ids_shape, 2);

  int64_t pos_shape[] = {1, (int64_t)seq_len};
  auto pos_val = Ort::Value::CreateTensor<int64_t>(
      mem_info_, position_ids.data(), seq_len, pos_shape, 2);

  int64_t af_shape[] = {1, (int64_t)audio_len, (int64_t)hidden_dim_};
  size_t af_count = 1 * audio_len * hidden_dim_;
  auto af_val = Ort::Value::CreateTensor<float>(
      mem_info_, const_cast<float*>(audio_features), af_count, af_shape, 3);

  int64_t ao_val_data = audio_offset;
  int64_t ao_shape[] = {1};
  auto ao_val = Ort::Value::CreateTensor<int64_t>(
      mem_info_, &ao_val_data, 1, ao_shape, 1);

  // Gather input/output names
  size_t n_inputs = prefill_->GetInputCount();
  std::vector<Ort::AllocatedStringPtr> in_name_ptrs;
  std::vector<const char*> in_names;
  for (size_t i = 0; i < n_inputs; ++i) {
    in_name_ptrs.push_back(
        prefill_->GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions()));
    in_names.push_back(in_name_ptrs.back().get());
  }

  size_t n_outputs = prefill_->GetOutputCount();
  std::vector<Ort::AllocatedStringPtr> out_name_ptrs;
  std::vector<const char*> out_names;
  for (size_t i = 0; i < n_outputs; ++i) {
    out_name_ptrs.push_back(prefill_->GetOutputNameAllocated(
        i, Ort::AllocatorWithDefaultOptions()));
    out_names.push_back(out_name_ptrs.back().get());
  }

  // Map input names to tensors
  // Expected inputs: input_ids, position_ids, audio_features, audio_offset
  std::vector<Ort::Value> inputs;
  for (size_t i = 0; i < n_inputs; ++i) {
    std::string name = in_names[i];
    if (name == "input_ids") {
      inputs.push_back(std::move(ids_val));
    } else if (name == "position_ids") {
      inputs.push_back(std::move(pos_val));
    } else if (name == "audio_features") {
      inputs.push_back(std::move(af_val));
    } else if (name == "audio_offset") {
      inputs.push_back(std::move(ao_val));
    } else {
      std::cerr << "[ASR] Unknown prefill input: " << name << std::endl;
      // Push a dummy — this shouldn't happen
      int64_t dummy = 0;
      int64_t dummy_shape[] = {1};
      inputs.push_back(Ort::Value::CreateTensor<int64_t>(
          mem_info_, &dummy, 1, dummy_shape, 1));
    }
  }

  auto outputs = prefill_->Run(Ort::RunOptions{nullptr}, in_names.data(),
                                inputs.data(), n_inputs, out_names.data(),
                                n_outputs);

  // Parse outputs: logits, past_key_0..27, past_value_0..27
  out.seq_len = seq_len;
  for (size_t i = 0; i < n_outputs; ++i) {
    auto info = outputs[i].GetTensorTypeAndShapeInfo();
    size_t elem_count = info.GetElementCount();
    const float* data = outputs[i].GetTensorData<float>();
    std::string name = out_names[i];

    if (name == "logits") {
      out.logits.assign(data, data + elem_count);
    } else if (name.find("past_") == 0 || name.find("present_") == 0) {
      out.kv_data.emplace_back(data, data + elem_count);
    }
  }

  return out;
}

// ---------------------------------------------------------------------------
// Transcribe
// ---------------------------------------------------------------------------
ASRResult ASRPipeline::Transcribe(const float* mel, int mel_len,
                                   const std::vector<int64_t>& prompt_ids,
                                   int audio_offset, int max_tokens) {
  using Clock = std::chrono::high_resolution_clock;
  ASRResult result;

  // 1. Encoder
  auto t0 = Clock::now();
  auto enc = RunEncoder(mel, mel_len);
  result.encoder_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  std::cout << "[ASR] Encoder: " << mel_len << " mel → " << enc.audio_len
            << " features (" << result.encoder_ms << " ms)" << std::endl;

  // 2. Prefill
  t0 = Clock::now();
  auto pf = RunPrefill(prompt_ids, enc.features.data(), enc.audio_len,
                        audio_offset);
  result.prefill_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  std::cout << "[ASR] Prefill: seq_len=" << pf.seq_len << " ("
            << result.prefill_ms << " ms)" << std::endl;

  // 3. Seed TRT KV cache
  assert(decoder_);
  decoder_->Reset();
  std::vector<const float*> kv_ptrs;
  for (auto& kv : pf.kv_data) {
    kv_ptrs.push_back(kv.data());
  }
  decoder_->SeedKV(kv_ptrs.data(), (int)kv_ptrs.size(), pf.seq_len);

  // Get logits from last prefill position
  int last_pos = pf.seq_len - 1;
  const float* last_logits = pf.logits.data() + (size_t)last_pos * vocab_size_;

  // 4. Decode loop — greedy argmax
  t0 = Clock::now();
  std::vector<float> logits_buf(vocab_size_);
  std::vector<float> hidden_buf(hidden_dim_);  // TRT outputs this, we ignore it

  // First token from prefill logits
  int64_t next_token = 0;
  {
    float max_val = last_logits[0];
    for (int i = 1; i < vocab_size_; ++i) {
      if (last_logits[i] > max_val) {
        max_val = last_logits[i];
        next_token = i;
      }
    }
  }

  for (int step = 0; step < max_tokens; ++step) {
    // Check EOS
    if (next_token == EOS_1 || next_token == EOS_2) {
      std::cout << "[ASR] EOS at step " << step << std::endl;
      break;
    }
    result.text_ids.push_back(next_token);

    // Embed lookup → [1, 1, hidden_dim]
    const float* emb = EmbedLookup(next_token);

    // TRT decode step
    decoder_->DecodeStep(emb, logits_buf.data(), hidden_buf.data());

    // Argmax
    next_token = 0;
    float max_val = logits_buf[0];
    for (int i = 1; i < vocab_size_; ++i) {
      if (logits_buf[i] > max_val) {
        max_val = logits_buf[i];
        next_token = i;
      }
    }
  }

  result.decode_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  result.n_tokens = (int)result.text_ids.size();
  result.per_token_ms =
      result.n_tokens > 0 ? result.decode_ms / result.n_tokens : 0;
  result.total_ms = result.encoder_ms + result.prefill_ms + result.decode_ms;

  std::cout << "[ASR] Decode: " << result.n_tokens << " tokens in "
            << result.decode_ms << " ms (" << result.per_token_ms
            << " ms/tok)" << std::endl;
  std::cout << "[ASR] Total: " << result.total_ms << " ms" << std::endl;

  return result;
}

void ASRPipeline::EnableProfiling(bool enable) {
  if (decoder_) decoder_->EnableProfiling(enable);
}

void ASRPipeline::PrintProfilingStats() {
  if (decoder_ && decoder_->stats().n_samples > 0) {
    auto& s = decoder_->stats();
    std::cout << "\n  === ASR DECODER PROFILING (" << s.n_samples
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
    decoder_->ResetStats();
  }
}
