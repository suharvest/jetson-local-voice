// tts_trt_engine.cpp — TRT native engine implementations
#include "tts_trt_engine.h"
#include "cp_sample_kernel.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#define CHECK_CUDA(call)                                                  \
  do {                                                                    \
    cudaError_t err = (call);                                             \
    if (err != cudaSuccess) {                                             \
      std::cerr << "CUDA error " << cudaGetErrorString(err) << " at "    \
                << __FILE__ << ":" << __LINE__ << std::endl;              \
      std::abort();                                                       \
    }                                                                     \
  } while (0)

// ---------------------------------------------------------------------------
// Logger
// ---------------------------------------------------------------------------
void TRTLogger::log(Severity severity, const char* msg) noexcept {
  if (severity <= Severity::kWARNING) {
    std::cerr << "[TRT] " << msg << std::endl;
  }
}

// ---------------------------------------------------------------------------
// Helper: load engine from file
// ---------------------------------------------------------------------------
static std::vector<char> LoadEngineFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f.is_open()) {
    std::cerr << "Failed to open engine: " << path << std::endl;
    std::abort();
  }
  size_t size = f.tellg();
  f.seekg(0);
  std::vector<char> data(size);
  f.read(data.data(), size);
  return data;
}

// Helper: get element size for TRT data type
static size_t TrtDtypeSize(nvinfer1::DataType dtype) {
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kBF16:
      return 2;
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kINT64:
      return 8;
    default:
      return 4;
  }
}

// ===========================================================================
// TRTTalkerEngine
// ===========================================================================
TRTTalkerEngine::TRTTalkerEngine(const std::string& engine_path, int n_layers,
                                 int hidden_dim, int n_heads, int head_dim,
                                 int vocab_size, int max_seq)
    : n_layers_(n_layers),
      hidden_dim_(hidden_dim),
      n_heads_(n_heads),
      head_dim_(head_dim),
      vocab_size_(vocab_size),
      max_seq_(max_seq) {
  auto data = LoadEngineFile(engine_path);
  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
  assert(engine_ && "Failed to deserialize talker engine");
  CHECK_CUDA(cudaStreamCreate(&stream_));

  // Detect KV dtype from engine (TRT 10.x API uses tensor names)
  auto kv_dtype = engine_->getTensorDataType("past_key_0");
  kv_elem_bytes_ = TrtDtypeSize(kv_dtype);
  kv_is_bf16_ = (kv_dtype == nvinfer1::DataType::kBF16);

  // Check for dual-profile engine (Profile 0 = prefill, Profile 1 = decode).
  // A dual-profile engine has exactly 2 profiles and avoids the TRT 10.3
  // SIGSEGV bug when both seq_len and past_len are dynamic in the same profile.
  int n_profiles = engine_->getNbOptimizationProfiles();
  if (n_profiles >= 2) {
    has_dual_profiles_ = true;
    // Profile 0: prefill — seq_len dynamic, past_len=0
    ctx_prefill_.reset(engine_->createExecutionContext());
    ctx_prefill_->setOptimizationProfileAsync(0, nullptr);
    // Profile 1: decode — seq_len=1, past_len dynamic
    ctx_decode_.reset(engine_->createExecutionContext());
    ctx_decode_->setOptimizationProfileAsync(1, nullptr);
    // Keep context_ pointing to decode context for backward compat
    context_.reset();  // not used in dual-profile mode
    std::cout << "  TRTTalkerEngine: DUAL-PROFILE engine detected ("
              << n_profiles << " profiles)" << std::endl;
  } else {
    has_dual_profiles_ = false;
    context_.reset(engine_->createExecutionContext());
    std::cout << "  TRTTalkerEngine: single-profile engine" << std::endl;
  }

  AllocateBuffers();

  // Create CUDA events for profiling (always created, zero cost when unused)
  CHECK_CUDA(cudaEventCreate(&ev_start_));
  CHECK_CUDA(cudaEventCreate(&ev_h2d_done_));
  CHECK_CUDA(cudaEventCreate(&ev_kernel_done_));
  CHECK_CUDA(cudaEventCreate(&ev_d2h_done_));

  std::cout << "  TRTTalkerEngine loaded: " << n_layers_ << " layers, KV "
            << kv_elem_bytes_ << "B/elem (bf16=" << kv_is_bf16_
            << "), max_seq=" << max_seq_ << ", dual_profiles=" << has_dual_profiles_
            << std::endl;
}

TRTTalkerEngine::~TRTTalkerEngine() {
  // Free prefill engine buffers
  for (auto p : d_prefill_kv_)
    if (p) cudaFree(p);
  if (d_prefill_emb_) cudaFree(d_prefill_emb_);
  if (d_prefill_logits_) cudaFree(d_prefill_logits_);
  if (d_prefill_hidden_) cudaFree(d_prefill_hidden_);

  FreeBuffers();
  if (ev_start_) cudaEventDestroy(ev_start_);
  if (ev_h2d_done_) cudaEventDestroy(ev_h2d_done_);
  if (ev_kernel_done_) cudaEventDestroy(ev_kernel_done_);
  if (ev_d2h_done_) cudaEventDestroy(ev_d2h_done_);
  if (stream_) cudaStreamDestroy(stream_);
}

void TRTTalkerEngine::AllocateBuffers() {
  int n_kv = 2 * n_layers_;  // keys + values
  size_t kv_bytes = 1 * n_heads_ * max_seq_ * head_dim_ * kv_elem_bytes_;

  kv_a_.resize(n_kv);
  kv_b_.resize(n_kv);
  for (int i = 0; i < n_kv; ++i) {
    CHECK_CUDA(cudaMalloc(&kv_a_[i], kv_bytes));
    CHECK_CUDA(cudaMalloc(&kv_b_[i], kv_bytes));
    CHECK_CUDA(cudaMemset(kv_a_[i], 0, kv_bytes));
    CHECK_CUDA(cudaMemset(kv_b_[i], 0, kv_bytes));
  }

  // I/O buffers — detect dtype from engine (TRT 10.x)
  // Auto-detect embed tensor name: "inputs_embeds" (TTS) or "input_embeds" (ASR)
  std::string emb_tname = "inputs_embeds";
  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    std::string tn = engine_->getIOTensorName(i);
    if (tn == "input_embeds") { emb_tname = "input_embeds"; break; }
  }
  size_t emb_elem = TrtDtypeSize(engine_->getTensorDataType(emb_tname.c_str()));

  // Detect logits dtype and cache it for use in DecodeStep
  auto logits_dtype = engine_->getTensorDataType("logits");
  logits_elem_bytes_ = TrtDtypeSize(logits_dtype);
  logits_is_bf16_ = (logits_dtype == nvinfer1::DataType::kBF16);

  // Check if last_hidden output exists (TTS has it, ASR does not); detect its dtype
  has_last_hidden_ = false;
  hidden_elem_bytes_ = 4;
  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    std::string tn = engine_->getIOTensorName(i);
    if (tn == "last_hidden") {
      has_last_hidden_ = true;
      auto hidden_dtype = engine_->getTensorDataType("last_hidden");
      hidden_elem_bytes_ = TrtDtypeSize(hidden_dtype);
      hidden_is_bf16_ = (hidden_dtype == nvinfer1::DataType::kBF16);
      break;
    }
  }

  // Allocate max-size buffers for both prefill (seq_len=max_seq_) and decode (1)
  CHECK_CUDA(cudaMalloc(&d_emb_, 1 * max_seq_ * hidden_dim_ * emb_elem));
  CHECK_CUDA(cudaMalloc(&d_logits_, 1 * max_seq_ * vocab_size_ * logits_elem_bytes_));
  // Always allocate d_hidden_ (TTS uses it, ASR ignores but needs buffer for TRT binding)
  CHECK_CUDA(cudaMalloc(&d_hidden_, 1 * max_seq_ * hidden_dim_ * hidden_elem_bytes_));
  CHECK_CUDA(cudaMalloc(&d_position_id_, max_seq_ * sizeof(int64_t)));

  std::cout << "  TRTTalkerEngine dtypes: logits=" << logits_elem_bytes_
            << "B (bf16=" << logits_is_bf16_ << ") hidden=" << hidden_elem_bytes_
            << "B has_hidden=" << has_last_hidden_ << std::endl;
}

void TRTTalkerEngine::FreeBuffers() {
  for (auto p : kv_a_)
    if (p) cudaFree(p);
  for (auto p : kv_b_)
    if (p) cudaFree(p);
  if (d_emb_) cudaFree(d_emb_);
  if (d_logits_) cudaFree(d_logits_);
  if (d_hidden_) cudaFree(d_hidden_);
  if (d_position_id_) cudaFree(d_position_id_);
}

void TRTTalkerEngine::Reset() {
  seq_len_ = 0;
  parity_ = 0;
  // Clear GPU KV buffers to prevent stale state between requests
  if (!kv_a_.empty()) {
    size_t kv_bytes = (size_t)n_heads_ * max_seq_ * head_dim_ * kv_elem_bytes_;
    for (auto* p : kv_a_) cudaMemsetAsync(p, 0, kv_bytes, stream_);
    for (auto* p : kv_b_) cudaMemsetAsync(p, 0, kv_bytes, stream_);
    cudaStreamSynchronize(stream_);
  }
}

// ---------------------------------------------------------------------------
// LoadPrefillEngine: load separate batch-prefill engine.
// Allocates GPU output buffers for its KV cache (56 tensors for 28 layers).
// ---------------------------------------------------------------------------
void TRTTalkerEngine::LoadPrefillEngine(const std::string& prefill_engine_path) {
  std::cout << "  Loading prefill engine: " << prefill_engine_path << std::endl;

  auto data = LoadEngineFile(prefill_engine_path);
  prefill_runtime_.reset(nvinfer1::createInferRuntime(prefill_logger_));
  prefill_engine_.reset(
      prefill_runtime_->deserializeCudaEngine(data.data(), data.size()));
  if (!prefill_engine_) {
    std::cerr << "  ERROR: Failed to deserialize prefill engine!" << std::endl;
    return;
  }
  prefill_ctx_.reset(prefill_engine_->createExecutionContext());

  // Allocate GPU output buffers for KV cache from prefill engine.
  // Each KV tensor: [1, n_heads, max_seq, head_dim] in FP16.
  // We reuse kv_elem_bytes_ from the decode engine (both are FP16).
  int n_kv = 2 * n_layers_;
  size_t kv_bytes = 1 * n_heads_ * max_seq_ * head_dim_ * kv_elem_bytes_;
  d_prefill_kv_.resize(n_kv, nullptr);
  for (int i = 0; i < n_kv; ++i) {
    CHECK_CUDA(cudaMalloc(&d_prefill_kv_[i], kv_bytes));
  }

  // Input/output buffers for prefill engine (FP32 inputs, FP16/FP32 outputs)
  // d_prefill_emb_: inputs_embeds [1, max_seq, D] in float32
  CHECK_CUDA(cudaMalloc(&d_prefill_emb_,
                        1 * max_seq_ * hidden_dim_ * sizeof(float)));
  // d_prefill_logits_: logits [1, max_seq, vocab] in float32
  // (prefill engine output dtype — detect from engine)
  size_t logits_elem = TrtDtypeSize(
      prefill_engine_->getTensorDataType("logits"));
  CHECK_CUDA(cudaMalloc(&d_prefill_logits_,
                        1 * max_seq_ * vocab_size_ * logits_elem));
  // d_prefill_hidden_: last_hidden [1, max_seq, D] in float32
  size_t hidden_elem = 4;
  for (int i = 0; i < prefill_engine_->getNbIOTensors(); ++i) {
    std::string tn = prefill_engine_->getIOTensorName(i);
    if (tn == "last_hidden") {
      hidden_elem = TrtDtypeSize(
          prefill_engine_->getTensorDataType("last_hidden"));
      break;
    }
  }
  CHECK_CUDA(cudaMalloc(&d_prefill_hidden_,
                        1 * max_seq_ * hidden_dim_ * hidden_elem));

  std::cout << "  Prefill engine loaded: " << n_kv << " KV output buffers, "
            << "kv_bytes=" << kv_bytes << " each" << std::endl;
}

// ---------------------------------------------------------------------------
// RunPrefillEngine: batch prefill using the dedicated prefill engine.
// Inputs: inputs_embeds [1, seq_len, D] float32 on CPU.
// After execution:
//   - KV cache outputs (past_key_i, past_value_i) are D2D-copied into kv_a_.
//   - seq_len_ is set, parity_ = 0 (decode reads kv_a_).
//   - Returns logits and last_hidden on CPU.
// ---------------------------------------------------------------------------
TRTTalkerEngine::PrefillResult TRTTalkerEngine::RunPrefillEngine(
    const float* inputs_embeds, int seq_len) {
  assert(prefill_engine_ && prefill_ctx_);
  auto* pctx = prefill_ctx_.get();

  // H2D: copy inputs_embeds
  size_t emb_bytes = (size_t)seq_len * hidden_dim_ * sizeof(float);
  CHECK_CUDA(cudaMemcpyAsync(d_prefill_emb_, inputs_embeds, emb_bytes,
                             cudaMemcpyHostToDevice, stream_));

  // Bind inputs_embeds
  pctx->setInputShape("inputs_embeds", nvinfer1::Dims3{1, seq_len, hidden_dim_});
  pctx->setTensorAddress("inputs_embeds", d_prefill_emb_);

  // Bind output: logits
  pctx->setTensorAddress("logits", d_prefill_logits_);

  // Bind output: last_hidden (if present)
  bool has_hidden = false;
  for (int i = 0; i < prefill_engine_->getNbIOTensors(); ++i) {
    if (std::string(prefill_engine_->getIOTensorName(i)) == "last_hidden") {
      pctx->setTensorAddress("last_hidden", d_prefill_hidden_);
      has_hidden = true;
      break;
    }
  }

  // Bind KV output tensors: past_key_0..27, past_value_0..27
  // The prefill engine outputs them as "past_key_i" and "past_value_i".
  for (int i = 0; i < n_layers_; ++i) {
    std::string key_name = "past_key_" + std::to_string(i);
    std::string val_name = "past_value_" + std::to_string(i);
    pctx->setTensorAddress(key_name.c_str(), d_prefill_kv_[2 * i]);
    pctx->setTensorAddress(val_name.c_str(), d_prefill_kv_[2 * i + 1]);
  }

  // Execute prefill engine
  bool ok = pctx->enqueueV3(stream_);
  if (!ok) {
    std::cerr << "  ERROR: prefill engine enqueueV3 failed!" << std::endl;
    std::abort();
  }

  // D2D: copy KV outputs from prefill buffers into decode engine's kv_a_.
  // The decode engine reads kv_a_ when parity_=0.
  // KV shape from prefill output: [1, n_heads, seq_len, head_dim] in kv_elem_bytes_.
  size_t kv_bytes_seq = (size_t)n_heads_ * seq_len * head_dim_ * kv_elem_bytes_;
  for (int i = 0; i < 2 * n_layers_; ++i) {
    CHECK_CUDA(cudaMemcpyAsync(kv_a_[i], d_prefill_kv_[i], kv_bytes_seq,
                               cudaMemcpyDeviceToDevice, stream_));
  }

  // D2H: copy logits and last_hidden to CPU
  PrefillResult result;
  result.seq_len = seq_len;
  result.last_hidden.resize(seq_len * hidden_dim_);

  // Detect actual logits token count: prefill engine may output only last token (n=1)
  // or all tokens (n=seq_len). Check dim 1 of the logits output tensor.
  int logits_n_tokens = seq_len;
  {
    auto logits_dims = pctx->getTensorShape("logits");
    if (logits_dims.nbDims >= 2 && logits_dims.d[1] == 1) {
      logits_n_tokens = 1;  // engine outputs last token only
    }
  }
  result.logits.resize(logits_n_tokens * vocab_size_);

  // logits dtype may be FP16; copy as FP32-equivalent bytes.
  size_t logits_elem_bytes = TrtDtypeSize(
      prefill_engine_->getTensorDataType("logits"));
  if (logits_elem_bytes == sizeof(float)) {
    // FP32: direct copy
    CHECK_CUDA(cudaMemcpyAsync(result.logits.data(), d_prefill_logits_,
                               (size_t)logits_n_tokens * vocab_size_ * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_));
  } else {
    // FP16: copy raw bytes, then convert on CPU
    std::vector<uint16_t> logits_fp16(logits_n_tokens * vocab_size_);
    CHECK_CUDA(cudaMemcpyAsync(logits_fp16.data(), d_prefill_logits_,
                               (size_t)logits_n_tokens * vocab_size_ * sizeof(uint16_t),
                               cudaMemcpyDeviceToHost, stream_));
    // Will convert after sync below
    CHECK_CUDA(cudaEventRecord(ev_d2h_done_, stream_));
    CHECK_CUDA(cudaEventSynchronize(ev_d2h_done_));
    // FP16 -> FP32 conversion
    for (size_t j = 0; j < (size_t)logits_n_tokens * vocab_size_; ++j) {
      uint32_t sign = (logits_fp16[j] >> 15) & 1;
      uint32_t exp  = (logits_fp16[j] >> 10) & 0x1F;
      uint32_t mant = logits_fp16[j] & 0x3FF;
      if (exp == 0x1F) {
        // Inf or NaN
        uint32_t bits = (sign << 31) | (0xFF << 23) | (mant << 13);
        std::memcpy(&result.logits[j], &bits, 4);
      } else if (exp == 0) {
        // Subnormal -> 0
        result.logits[j] = 0.0f;
      } else {
        uint32_t bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        std::memcpy(&result.logits[j], &bits, 4);
      }
    }
    // Copy hidden separately
    size_t hidden_elem_bytes = TrtDtypeSize(
        prefill_engine_->getTensorDataType("last_hidden"));
    if (has_hidden && hidden_elem_bytes == sizeof(float)) {
      CHECK_CUDA(cudaMemcpy(result.last_hidden.data(), d_prefill_hidden_,
                            (size_t)seq_len * hidden_dim_ * sizeof(float),
                            cudaMemcpyDeviceToHost));
    } else if (has_hidden) {
      std::vector<uint16_t> hidden_fp16(seq_len * hidden_dim_);
      CHECK_CUDA(cudaMemcpy(hidden_fp16.data(), d_prefill_hidden_,
                            (size_t)seq_len * hidden_dim_ * sizeof(uint16_t),
                            cudaMemcpyDeviceToHost));
      for (size_t j = 0; j < (size_t)seq_len * hidden_dim_; ++j) {
        uint32_t sign = (hidden_fp16[j] >> 15) & 1;
        uint32_t exp  = (hidden_fp16[j] >> 10) & 0x1F;
        uint32_t mant = hidden_fp16[j] & 0x3FF;
        if (exp == 0x1F) {
          uint32_t bits = (sign << 31) | (0xFF << 23) | (mant << 13);
          std::memcpy(&result.last_hidden[j], &bits, 4);
        } else if (exp == 0) {
          result.last_hidden[j] = 0.0f;
        } else {
          uint32_t bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
          std::memcpy(&result.last_hidden[j], &bits, 4);
        }
      }
    }

    // Update state
    seq_len_ = seq_len;
    parity_ = 0;  // decode reads kv_a_

    // Debug
    {
      int last = (int)result.logits.size() / vocab_size_ - 1;
      float* lp = result.logits.data() + last * vocab_size_;
      float lmin = lp[0], lmax = lp[0]; int argmax = 0;
      for (int k = 1; k < vocab_size_; ++k) {
        if (lp[k] < lmin) lmin = lp[k];
        if (lp[k] > lmax) { lmax = lp[k]; argmax = k; }
      }
      std::cerr << "  DEBUG prefill (engine): logits_range=[" << lmin << ","
                << lmax << "] argmax=" << argmax << std::endl;
    }

    std::cout << "  TRT Prefill Engine (FP16): seq_len=" << seq_len
              << " logits=" << result.logits.size()
              << " hidden=" << result.last_hidden.size() << std::endl;
    return result;
  }

  // FP32 logits path: also copy hidden
  if (has_hidden) {
    CHECK_CUDA(cudaMemcpyAsync(result.last_hidden.data(), d_prefill_hidden_,
                               (size_t)seq_len * hidden_dim_ * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_));
  }

  CHECK_CUDA(cudaEventRecord(ev_d2h_done_, stream_));
  CHECK_CUDA(cudaEventSynchronize(ev_d2h_done_));

  // Update state
  seq_len_ = seq_len;
  parity_ = 0;  // decode reads kv_a_

  // Debug: print logits stats for last position (engine may output last token only)
  {
    int last = (int)result.logits.size() / vocab_size_ - 1;
    float* lp = result.logits.data() + last * vocab_size_;
    float lmin = lp[0], lmax = lp[0]; int argmax = 0;
    for (int k = 1; k < vocab_size_; ++k) {
      if (lp[k] < lmin) lmin = lp[k];
      if (lp[k] > lmax) { lmax = lp[k]; argmax = k; }
    }
    float hn = 0;
    float* hp = result.last_hidden.data() + last * hidden_dim_;
    for (int k = 0; k < hidden_dim_; ++k) hn += hp[k] * hp[k];
    hn = std::sqrt(hn);
    std::cerr << "  DEBUG prefill (engine): logits_range=[" << lmin << ","
              << lmax << "] argmax=" << argmax << " hidden_norm=" << hn
              << std::endl;
  }

  std::cout << "  TRT Prefill Engine (FP32): seq_len=" << seq_len
            << " logits=" << result.logits.size()
            << " hidden=" << result.last_hidden.size() << std::endl;
  return result;
}

void TRTTalkerEngine::SeedKV(const float* const* kv_ptrs, int n_kv,
                              int seq_len) {
  size_t elem_count = (size_t)1 * n_heads_ * seq_len * head_dim_;
  size_t bytes = elem_count * kv_elem_bytes_;
  for (int i = 0; i < n_kv && i < (int)kv_a_.size(); ++i) {
    if (kv_elem_bytes_ == 4) {
      // FP32: direct copy
      CHECK_CUDA(cudaMemcpyAsync(kv_a_[i], kv_ptrs[i], bytes,
                                 cudaMemcpyHostToDevice, stream_));
    } else if (kv_is_bf16_) {
      // BF16: FP32 -> BF16 conversion (truncate lower 16 bits of FP32 mantissa)
      std::vector<uint16_t> bf16_buf(elem_count);
      const float* src = kv_ptrs[i];
      for (size_t j = 0; j < elem_count; ++j) {
        uint32_t fbits;
        std::memcpy(&fbits, &src[j], 4);
        bf16_buf[j] = (uint16_t)(fbits >> 16);
      }
      CHECK_CUDA(cudaMemcpyAsync(kv_a_[i], bf16_buf.data(), bytes,
                                 cudaMemcpyHostToDevice, stream_));
      CHECK_CUDA(cudaStreamSynchronize(stream_));
    } else {
      // FP16: convert from FP32 CPU data to FP16 before uploading.
      std::vector<uint16_t> fp16_buf(elem_count);
      const float* src = kv_ptrs[i];
      for (size_t j = 0; j < elem_count; ++j) {
        float f = src[j];
        uint32_t fbits;
        std::memcpy(&fbits, &f, 4);
        uint32_t sign = (fbits >> 16) & 0x8000;
        int32_t exp = (int32_t)((fbits >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = (fbits & 0x7FFFFF) >> 13;
        uint16_t h;
        if (exp <= 0) {
          h = (uint16_t)sign;
        } else if (exp >= 31) {
          h = (uint16_t)(sign | 0x7C00);
        } else {
          h = (uint16_t)(sign | (exp << 10) | mant);
        }
        fp16_buf[j] = h;
      }
      CHECK_CUDA(cudaMemcpyAsync(kv_a_[i], fp16_buf.data(), bytes,
                                 cudaMemcpyHostToDevice, stream_));
      CHECK_CUDA(cudaStreamSynchronize(stream_));
    }
  }
  if (kv_elem_bytes_ == 4) {
    CHECK_CUDA(cudaStreamSynchronize(stream_));
  }
  seq_len_ = seq_len;
  parity_ = 0;
}

// ---------------------------------------------------------------------------
// Prefill: dispatch to prefill engine (if loaded) or fall back to iterative.
// ---------------------------------------------------------------------------
TRTTalkerEngine::PrefillResult TRTTalkerEngine::Prefill(
    const float* inputs_embeds, int seq_len) {
  assert(seq_len > 0 && seq_len <= max_seq_);
  Reset();

  // Use dedicated prefill engine if available (best quality, no iterative error)
  if (prefill_engine_) {
    return RunPrefillEngine(inputs_embeds, seq_len);
  }

  // Check if decode engine supports dynamic inputs_embeds seq_len (batch prefill).
  // Dual-profile engines always support batch prefill via Profile 0.
  // Single-profile engines: check if Profile 0 max seq_len > 1.
  bool supports_batch = has_dual_profiles_;
  if (!supports_batch) {
    std::string emb_tname = "inputs_embeds";
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
      std::string tn = engine_->getIOTensorName(i);
      if (tn == "input_embeds") { emb_tname = "input_embeds"; break; }
    }
    auto prof_max = engine_->getProfileShape(
        emb_tname.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
    if (prof_max.d[1] > 1) {
      supports_batch = true;
    }
  }

  if (!supports_batch || seq_len == 1) {
    // Iterative prefill: feed tokens one-by-one through the decode engine
    PrefillResult result;
    result.seq_len = seq_len;

    std::vector<float> logits(vocab_size_);
    std::vector<float> last_hidden(hidden_dim_);

    for (int t = 0; t < seq_len; ++t) {
      const float* token_emb = inputs_embeds + t * hidden_dim_;
      DecodeStep(token_emb, logits.data(), last_hidden.data());
    }

    result.logits.resize(seq_len * vocab_size_, 0.0f);
    result.last_hidden.resize(seq_len * hidden_dim_, 0.0f);

    std::memcpy(result.logits.data() + (seq_len - 1) * vocab_size_,
                logits.data(), vocab_size_ * sizeof(float));
    std::memcpy(result.last_hidden.data() + (seq_len - 1) * hidden_dim_,
                last_hidden.data(), hidden_dim_ * sizeof(float));

    // Debug: print logits stats for last position
    {
      int last_pos = seq_len - 1;
      float* lp = result.logits.data() + last_pos * vocab_size_;
      float lmin = lp[0], lmax = lp[0];
      int argmax = 0;
      for (int i = 1; i < vocab_size_; ++i) {
        if (lp[i] < lmin) lmin = lp[i];
        if (lp[i] > lmax) { lmax = lp[i]; argmax = i; }
      }
      float eos_logit = (2560 < vocab_size_) ? lp[2560] : 0;
      float hn = 0;
      float* hp = result.last_hidden.data() + last_pos * hidden_dim_;
      for (int i = 0; i < hidden_dim_; ++i) hn += hp[i] * hp[i];
      hn = std::sqrt(hn);
      std::cerr << "  DEBUG prefill (iterative): logits_range=[" << lmin << ","
                << lmax << "] argmax=" << argmax << " hidden_norm=" << hn
                << " eos_logit=" << eos_logit << std::endl;
    }
    std::cout << "  TRT Iterative Prefill: seq_len=" << seq_len
              << " logits=" << result.logits.size()
              << " hidden=" << result.last_hidden.size() << std::endl;
    return result;
  }

  // === Batch prefill path (decode engine with dynamic seq_len) ===
  // Dual-profile engine: use Profile 0 context (ctx_prefill_).
  // Single-profile engine: use context_ (which must support dynamic seq_len).
  auto* ctx = has_dual_profiles_ ? ctx_prefill_.get() : context_.get();

  if (first_step_) {
    kv_names_.resize(2 * n_layers_);
    new_kv_names_.resize(2 * n_layers_);
    for (int i = 0; i < n_layers_; ++i) {
      kv_names_[2 * i] = "past_key_" + std::to_string(i);
      kv_names_[2 * i + 1] = "past_value_" + std::to_string(i);
      new_kv_names_[2 * i] = "new_past_key_" + std::to_string(i);
      new_kv_names_[2 * i + 1] = "new_past_value_" + std::to_string(i);
    }
    emb_name_ = "inputs_embeds";
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
      std::string tn = engine_->getIOTensorName(i);
      if (tn == "input_embeds") { emb_name_ = "input_embeds"; break; }
    }
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
      std::string tn = engine_->getIOTensorName(i);
      if (tn == "position_ids") { has_position_ids_ = true; break; }
    }
    first_step_ = false;
  }

  size_t emb_bytes = 1 * seq_len * hidden_dim_ * sizeof(float);
  CHECK_CUDA(cudaMemcpyAsync(d_emb_, inputs_embeds, emb_bytes,
                             cudaMemcpyHostToDevice, stream_));

  ctx->setInputShape(emb_name_.c_str(),
                     nvinfer1::Dims3{1, seq_len, hidden_dim_});
  ctx->setTensorAddress(emb_name_.c_str(), d_emb_);

  auto& read = kv_a_;
  auto& write = kv_b_;
  nvinfer1::Dims4 kv_shape{1, n_heads_, 0, head_dim_};
  for (int i = 0; i < n_layers_; ++i) {
    ctx->setInputShape(kv_names_[2 * i].c_str(), kv_shape);
    ctx->setTensorAddress(kv_names_[2 * i].c_str(), read[2 * i]);
    ctx->setTensorAddress(new_kv_names_[2 * i].c_str(), write[2 * i]);
    ctx->setInputShape(kv_names_[2 * i + 1].c_str(), kv_shape);
    ctx->setTensorAddress(kv_names_[2 * i + 1].c_str(), read[2 * i + 1]);
    ctx->setTensorAddress(new_kv_names_[2 * i + 1].c_str(), write[2 * i + 1]);
  }

  if (has_position_ids_) {
    std::vector<int64_t> positions(seq_len);
    for (int i = 0; i < seq_len; ++i) positions[i] = i;
    CHECK_CUDA(cudaMemcpyAsync(d_position_id_, positions.data(),
                               seq_len * sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream_));
    ctx->setInputShape("position_ids", nvinfer1::Dims2{1, seq_len});
    ctx->setTensorAddress("position_ids", d_position_id_);
  }

  ctx->setTensorAddress("logits", d_logits_);
  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    std::string tn = engine_->getIOTensorName(i);
    if (tn == "last_hidden") {
      ctx->setTensorAddress("last_hidden", d_hidden_);
      break;
    }
  }

  ctx->enqueueV3(stream_);

  PrefillResult result;
  result.seq_len = seq_len;
  result.logits.resize(seq_len * vocab_size_);

  // Detect output dtypes from engine
  auto logits_dtype = engine_->getTensorDataType("logits");
  size_t logits_elem_bytes = TrtDtypeSize(logits_dtype);

  // Check if last_hidden output exists
  bool has_last_hidden = false;
  nvinfer1::DataType hidden_dtype = nvinfer1::DataType::kFLOAT;
  size_t hidden_elem_bytes = 4;
  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    if (std::string(engine_->getIOTensorName(i)) == "last_hidden") {
      has_last_hidden = true;
      hidden_dtype = engine_->getTensorDataType("last_hidden");
      hidden_elem_bytes = TrtDtypeSize(hidden_dtype);
      break;
    }
  }
  if (has_last_hidden) {
    result.last_hidden.resize(seq_len * hidden_dim_);
  }

  if (logits_elem_bytes == sizeof(float)) {
    // FP32 logits: direct copy
    CHECK_CUDA(cudaMemcpyAsync(result.logits.data(), d_logits_,
                               (size_t)seq_len * vocab_size_ * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_));
  } else {
    // FP16/BF16 logits: copy raw bytes then convert on CPU
    std::vector<uint16_t> logits_raw(seq_len * vocab_size_);
    CHECK_CUDA(cudaMemcpyAsync(logits_raw.data(), d_logits_,
                               (size_t)seq_len * vocab_size_ * logits_elem_bytes,
                               cudaMemcpyDeviceToHost, stream_));
    CHECK_CUDA(cudaEventRecord(ev_d2h_done_, stream_));
    CHECK_CUDA(cudaEventSynchronize(ev_d2h_done_));

    bool is_bf16 = (logits_dtype == nvinfer1::DataType::kBF16);
    for (size_t j = 0; j < (size_t)seq_len * vocab_size_; ++j) {
      if (is_bf16) {
        uint32_t bits = (uint32_t)logits_raw[j] << 16;
        std::memcpy(&result.logits[j], &bits, 4);
      } else {
        uint32_t sign = (logits_raw[j] >> 15) & 1;
        uint32_t exp  = (logits_raw[j] >> 10) & 0x1F;
        uint32_t mant =  logits_raw[j]        & 0x3FF;
        uint32_t bits;
        if (exp == 0x1F)      bits = (sign << 31) | (0xFF << 23) | (mant << 13);
        else if (exp == 0)    bits = sign << 31;
        else                  bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        std::memcpy(&result.logits[j], &bits, 4);
      }
    }
  }

  if (has_last_hidden) {
    if (hidden_elem_bytes == sizeof(float)) {
      CHECK_CUDA(cudaMemcpyAsync(result.last_hidden.data(), d_hidden_,
                                 (size_t)seq_len * hidden_dim_ * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream_));
    } else {
      std::vector<uint16_t> hidden_raw(seq_len * hidden_dim_);
      CHECK_CUDA(cudaMemcpyAsync(hidden_raw.data(), d_hidden_,
                                 (size_t)seq_len * hidden_dim_ * hidden_elem_bytes,
                                 cudaMemcpyDeviceToHost, stream_));
      CHECK_CUDA(cudaEventRecord(ev_d2h_done_, stream_));
      CHECK_CUDA(cudaEventSynchronize(ev_d2h_done_));
      bool is_bf16 = (hidden_dtype == nvinfer1::DataType::kBF16);
      for (size_t j = 0; j < (size_t)seq_len * hidden_dim_; ++j) {
        if (is_bf16) {
          uint32_t bits = (uint32_t)hidden_raw[j] << 16;
          std::memcpy(&result.last_hidden[j], &bits, 4);
        } else {
          uint32_t sign = (hidden_raw[j] >> 15) & 1;
          uint32_t exp  = (hidden_raw[j] >> 10) & 0x1F;
          uint32_t mant =  hidden_raw[j]        & 0x3FF;
          uint32_t bits;
          if (exp == 0x1F)      bits = (sign << 31) | (0xFF << 23) | (mant << 13);
          else if (exp == 0)    bits = sign << 31;
          else                  bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
          std::memcpy(&result.last_hidden[j], &bits, 4);
        }
      }
    }
  }

  if (logits_elem_bytes == sizeof(float)) {
    // Need to sync after the async copies above
    CHECK_CUDA(cudaEventRecord(ev_d2h_done_, stream_));
    CHECK_CUDA(cudaEventSynchronize(ev_d2h_done_));
  }

  seq_len_ = seq_len;
  parity_ = 1;

  // Debug: logits stats for last position
  {
    int last = seq_len - 1;
    float* lp = result.logits.data() + last * vocab_size_;
    float lmin = lp[0], lmax = lp[0]; int argmax = 0;
    for (int k = 1; k < vocab_size_; ++k) {
      if (lp[k] < lmin) lmin = lp[k];
      if (lp[k] > lmax) { lmax = lp[k]; argmax = k; }
    }
    std::cerr << "  DEBUG batch prefill (unified): logits_range=[" << lmin << ","
              << lmax << "] argmax=" << argmax << std::endl;
  }

  std::cout << "  TRT Batch Prefill (unified engine): seq_len=" << seq_len
            << " logits=" << result.logits.size()
            << " hidden=" << result.last_hidden.size() << std::endl;
  return result;
}


void TRTTalkerEngine::DecodeStep(const float* inputs_embeds, float* logits,
                                 float* last_hidden) {
  // Dual-profile engine: use Profile 1 context (ctx_decode_).
  // Single-profile engine: use context_.
  auto* ctx = has_dual_profiles_ ? ctx_decode_.get() : context_.get();
  auto& read = (parity_ == 0) ? kv_a_ : kv_b_;
  auto& write = (parity_ == 0) ? kv_b_ : kv_a_;

  // Record start event for profiling
  if (profiling_) {
    CHECK_CUDA(cudaEventRecord(ev_start_, stream_));
  }

  // Copy inputs_embeds to GPU (4KB)
  size_t emb_bytes = 1 * 1 * hidden_dim_ * sizeof(float);
  CHECK_CUDA(cudaMemcpyAsync(d_emb_, inputs_embeds, emb_bytes,
                             cudaMemcpyHostToDevice, stream_));

  // Initialize cached names on first call — auto-detect from engine.
  // For dual-profile engines, also initialize decode context (ctx_decode_)
  // since batch prefill may have already cleared first_step_.
  // Use decode_first_step_ to track whether ctx_decode_ is initialized.
  bool init_needed = first_step_ || decode_first_step_;
  if (init_needed) {
    if (first_step_) {
      // First time ever: populate KV name caches and detect tensor names
      kv_names_.resize(2 * n_layers_);
      new_kv_names_.resize(2 * n_layers_);
      for (int i = 0; i < n_layers_; ++i) {
        kv_names_[2 * i] = "past_key_" + std::to_string(i);
        kv_names_[2 * i + 1] = "past_value_" + std::to_string(i);
        new_kv_names_[2 * i] = "new_past_key_" + std::to_string(i);
        new_kv_names_[2 * i + 1] = "new_past_value_" + std::to_string(i);
      }
      emb_name_ = "inputs_embeds";
      for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        std::string tn = engine_->getIOTensorName(i);
        if (tn == "input_embeds") { emb_name_ = "input_embeds"; break; }
      }
      for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        std::string tn = engine_->getIOTensorName(i);
        if (tn == "position_ids") { has_position_ids_ = true; break; }
      }
      first_step_ = false;
    }
    // Bind static tensor addresses on decode context (done once per context)
    std::cout << "  TRT decode ctx init: emb_name=" << emb_name_
              << " has_position_ids=" << has_position_ids_ << std::endl;
    ctx->setInputShape(emb_name_.c_str(), nvinfer1::Dims3{1, 1, hidden_dim_});
    ctx->setTensorAddress(emb_name_.c_str(), d_emb_);
    ctx->setTensorAddress("logits", d_logits_);
    // "last_hidden" may not exist in ASR engine — bind only if present
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
      std::string tn = engine_->getIOTensorName(i);
      if (tn == "last_hidden") {
        ctx->setTensorAddress("last_hidden", d_hidden_);
        break;
      }
    }
    if (has_position_ids_) {
      ctx->setInputShape("position_ids", nvinfer1::Dims2{1, 1});
      ctx->setTensorAddress("position_ids", d_position_id_);
    }
    std::cout << "  TRT bind: seq_len=" << seq_len_ << std::endl;
    decode_first_step_ = false;
  }

  // Update position_ids if present
  if (has_position_ids_) {
    int64_t pos = seq_len_;
    CHECK_CUDA(cudaMemcpyAsync(d_position_id_, &pos, sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream_));
  }

  // Bind KV cache — only shapes and addresses change per step
  nvinfer1::Dims4 kv_shape{1, n_heads_, seq_len_, head_dim_};
  for (int i = 0; i < n_layers_; ++i) {
    ctx->setInputShape(kv_names_[2 * i].c_str(), kv_shape);
    ctx->setTensorAddress(kv_names_[2 * i].c_str(), read[2 * i]);
    ctx->setTensorAddress(new_kv_names_[2 * i].c_str(), write[2 * i]);

    ctx->setInputShape(kv_names_[2 * i + 1].c_str(), kv_shape);
    ctx->setTensorAddress(kv_names_[2 * i + 1].c_str(), read[2 * i + 1]);
    ctx->setTensorAddress(new_kv_names_[2 * i + 1].c_str(), write[2 * i + 1]);
  }

  // Record H2D done event (all input copies queued before kernel)
  if (profiling_) {
    CHECK_CUDA(cudaEventRecord(ev_h2d_done_, stream_));
  }

  // Execute TRT kernel
  ctx->enqueueV3(stream_);

  // Record kernel done event
  if (profiling_) {
    CHECK_CUDA(cudaEventRecord(ev_kernel_done_, stream_));
  }

  // Copy logits + hidden back to host, handling BF16/FP16 output
  if (logits_elem_bytes_ == sizeof(float)) {
    // FP32 output: direct copy
    size_t logits_bytes = (size_t)vocab_size_ * sizeof(float);
    CHECK_CUDA(cudaMemcpyAsync(logits, d_logits_, logits_bytes,
                               cudaMemcpyDeviceToHost, stream_));
  } else {
    // BF16 or FP16: copy raw 2-byte elements, then convert on CPU
    std::vector<uint16_t> logits_raw(vocab_size_);
    CHECK_CUDA(cudaMemcpyAsync(logits_raw.data(), d_logits_,
                               (size_t)vocab_size_ * logits_elem_bytes_,
                               cudaMemcpyDeviceToHost, stream_));
    CHECK_CUDA(cudaEventRecord(ev_d2h_done_, stream_));
    CHECK_CUDA(cudaEventSynchronize(ev_d2h_done_));
    for (int j = 0; j < vocab_size_; ++j) {
      uint32_t bits;
      if (logits_is_bf16_) {
        bits = (uint32_t)logits_raw[j] << 16;
      } else {
        // FP16 -> FP32
        uint32_t sign = (logits_raw[j] >> 15) & 1;
        uint32_t exp  = (logits_raw[j] >> 10) & 0x1F;
        uint32_t mant =  logits_raw[j]        & 0x3FF;
        if (exp == 0x1F)   bits = (sign << 31) | (0xFF << 23) | (mant << 13);
        else if (exp == 0) bits = sign << 31;
        else               bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
      }
      std::memcpy(&logits[j], &bits, 4);
    }
  }

  if (has_last_hidden_ && last_hidden != nullptr) {
    if (hidden_elem_bytes_ == sizeof(float)) {
      size_t hidden_bytes = (size_t)hidden_dim_ * sizeof(float);
      CHECK_CUDA(cudaMemcpyAsync(last_hidden, d_hidden_, hidden_bytes,
                                 cudaMemcpyDeviceToHost, stream_));
    } else {
      // BF16 or FP16: copy raw and convert
      std::vector<uint16_t> hidden_raw(hidden_dim_);
      CHECK_CUDA(cudaMemcpyAsync(hidden_raw.data(), d_hidden_,
                                 (size_t)hidden_dim_ * hidden_elem_bytes_,
                                 cudaMemcpyDeviceToHost, stream_));
      CHECK_CUDA(cudaEventRecord(ev_d2h_done_, stream_));
      CHECK_CUDA(cudaEventSynchronize(ev_d2h_done_));
      for (int j = 0; j < hidden_dim_; ++j) {
        uint32_t bits;
        if (hidden_is_bf16_) {
          bits = (uint32_t)hidden_raw[j] << 16;
        } else {
          uint32_t sign = (hidden_raw[j] >> 15) & 1;
          uint32_t exp  = (hidden_raw[j] >> 10) & 0x1F;
          uint32_t mant =  hidden_raw[j]        & 0x3FF;
          if (exp == 0x1F)   bits = (sign << 31) | (0xFF << 23) | (mant << 13);
          else if (exp == 0) bits = sign << 31;
          else               bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        }
        std::memcpy(&last_hidden[j], &bits, 4);
      }
    }
  }

  // Use event sync instead of full stream sync — avoids driver-level
  // bookkeeping overhead in cudaStreamSynchronize.
  CHECK_CUDA(cudaEventRecord(ev_d2h_done_, stream_));
  CHECK_CUDA(cudaEventSynchronize(ev_d2h_done_));

  // Collect profiling data
  if (profiling_) {
    StepTiming t;
    cudaEventElapsedTime(&t.h2d_ms, ev_start_, ev_h2d_done_);
    cudaEventElapsedTime(&t.kernel_ms, ev_h2d_done_, ev_kernel_done_);
    cudaEventElapsedTime(&t.d2h_ms, ev_kernel_done_, ev_d2h_done_);
    cudaEventElapsedTime(&t.total_ms, ev_start_, ev_d2h_done_);
    stats_.Add(t);
  }

  seq_len_ += 1;
  parity_ ^= 1;
}

// ===========================================================================
// TRTCPEngine
// ===========================================================================
TRTCPEngine::TRTCPEngine(const std::string& engine_path, int hidden_dim,
                         int cp_vocab, int max_ctx_len)
    : hidden_dim_(hidden_dim), cp_vocab_(cp_vocab) {
  auto data = LoadEngineFile(engine_path);
  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
  assert(engine_ && "Failed to deserialize CP engine");
  context_trt_.reset(engine_->createExecutionContext());
  CHECK_CUDA(cudaStreamCreate(&stream_));

  // Allocate max-size buffers
  size_t ctx_bytes = 1 * max_ctx_len * hidden_dim * sizeof(float);
  CHECK_CUDA(cudaMalloc(&d_ctx_, ctx_bytes));
  CHECK_CUDA(cudaMalloc(&d_gs_, sizeof(int64_t)));
  CHECK_CUDA(cudaMalloc(&d_out_, 1 * 1 * cp_vocab * sizeof(float)));

  // Create CUDA events for profiling
  CHECK_CUDA(cudaEventCreate(&ev_start_));
  CHECK_CUDA(cudaEventCreate(&ev_h2d_done_));
  CHECK_CUDA(cudaEventCreate(&ev_kernel_done_));
  CHECK_CUDA(cudaEventCreate(&ev_d2h_done_));

  std::cout << "  TRTCPEngine loaded: D=" << hidden_dim << ", vocab="
            << cp_vocab << std::endl;
}

TRTCPEngine::~TRTCPEngine() {
  if (d_ctx_) cudaFree(d_ctx_);
  if (d_gs_) cudaFree(d_gs_);
  if (d_out_) cudaFree(d_out_);
  if (ev_start_) cudaEventDestroy(ev_start_);
  if (ev_h2d_done_) cudaEventDestroy(ev_h2d_done_);
  if (ev_kernel_done_) cudaEventDestroy(ev_kernel_done_);
  if (ev_d2h_done_) cudaEventDestroy(ev_d2h_done_);
  if (stream_) cudaStreamDestroy(stream_);
}

void TRTCPEngine::Predict(const float* context, int ctx_len, int step,
                          float* logits_out) {
  auto* ctx = context_trt_.get();

  // Copy full context to GPU
  size_t ctx_bytes = 1 * ctx_len * hidden_dim_ * sizeof(float);
  CHECK_CUDA(
      cudaMemcpyAsync(d_ctx_, context, ctx_bytes, cudaMemcpyHostToDevice, stream_));

  int64_t gs = step;
  CHECK_CUDA(
      cudaMemcpyAsync(d_gs_, &gs, sizeof(int64_t), cudaMemcpyHostToDevice, stream_));

  ctx->setInputShape("context", nvinfer1::Dims3{1, ctx_len, hidden_dim_});
  ctx->setInputShape("gen_step", nvinfer1::Dims{1, {1}});
  ctx->setTensorAddress("context", d_ctx_);
  ctx->setTensorAddress("gen_step", d_gs_);
  ctx->setTensorAddress("logits", d_out_);

  ctx->enqueueV3(stream_);

  size_t out_bytes = 1 * 1 * cp_vocab_ * sizeof(float);
  CHECK_CUDA(cudaMemcpyAsync(logits_out, d_out_, out_bytes,
                             cudaMemcpyDeviceToHost, stream_));
  // Event-based sync
  CHECK_CUDA(cudaEventRecord(ev_d2h_done_, stream_));
  CHECK_CUDA(cudaEventSynchronize(ev_d2h_done_));
}

// --- GPU-resident context methods ---

void TRTCPEngine::BeginFrame(const float* hidden, const float* primary_emb) {
  // Copy hidden [D] + primary_emb [D] to GPU context at positions 0 and 1
  size_t d_bytes = hidden_dim_ * sizeof(float);
  CHECK_CUDA(cudaMemcpyAsync(d_ctx_, hidden, d_bytes,
                             cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA(cudaMemcpyAsync((char*)d_ctx_ + d_bytes, primary_emb, d_bytes,
                             cudaMemcpyHostToDevice, stream_));
  ctx_len_ = 2;
}

void TRTCPEngine::PredictGPU(int step, float* logits_out) {
  auto* ctx = context_trt_.get();

  // Record start event for profiling
  if (profiling_) {
    CHECK_CUDA(cudaEventRecord(ev_start_, stream_));
  }

  // gen_step — small H2D copy (8 bytes)
  int64_t gs = step;
  CHECK_CUDA(
      cudaMemcpyAsync(d_gs_, &gs, sizeof(int64_t), cudaMemcpyHostToDevice, stream_));

  // Bind — context is already on GPU
  ctx->setInputShape("context", nvinfer1::Dims3{1, ctx_len_, hidden_dim_});
  ctx->setInputShape("gen_step", nvinfer1::Dims{1, {1}});
  ctx->setTensorAddress("context", d_ctx_);
  ctx->setTensorAddress("gen_step", d_gs_);
  ctx->setTensorAddress("logits", d_out_);

  if (profiling_) {
    CHECK_CUDA(cudaEventRecord(ev_h2d_done_, stream_));
  }

  ctx->enqueueV3(stream_);

  if (profiling_) {
    CHECK_CUDA(cudaEventRecord(ev_kernel_done_, stream_));
  }

  // Only copy logits back (small: 2048*4 = 8KB)
  size_t out_bytes = 1 * 1 * cp_vocab_ * sizeof(float);
  CHECK_CUDA(cudaMemcpyAsync(logits_out, d_out_, out_bytes,
                             cudaMemcpyDeviceToHost, stream_));

  // Event-based sync instead of full stream sync
  CHECK_CUDA(cudaEventRecord(ev_d2h_done_, stream_));
  CHECK_CUDA(cudaEventSynchronize(ev_d2h_done_));


  // Collect profiling data
  if (profiling_) {
    StepTiming t;
    cudaEventElapsedTime(&t.h2d_ms, ev_start_, ev_h2d_done_);
    cudaEventElapsedTime(&t.kernel_ms, ev_h2d_done_, ev_kernel_done_);
    cudaEventElapsedTime(&t.d2h_ms, ev_kernel_done_, ev_d2h_done_);
    cudaEventElapsedTime(&t.total_ms, ev_start_, ev_d2h_done_);
    stats_.Add(t);
  }
}

void TRTCPEngine::AppendEmbedding(const float* emb) {
  // Append [D] floats at position ctx_len_ in GPU buffer
  size_t offset = ctx_len_ * hidden_dim_ * sizeof(float);
  size_t d_bytes = hidden_dim_ * sizeof(float);
  CHECK_CUDA(cudaMemcpyAsync((char*)d_ctx_ + offset, emb, d_bytes,
                             cudaMemcpyHostToDevice, stream_));
  ctx_len_++;
}

void TRTCPEngine::LoadEmbedTable(const float* table, int n_layers, int vocab,
                                  int dim) {
  embed_n_layers_ = n_layers;
  embed_vocab_ = vocab;
  embed_dim_ = dim;
  size_t bytes = (size_t)n_layers * vocab * dim * sizeof(float);
  CHECK_CUDA(cudaMalloc(&d_embed_table_, bytes));
  CHECK_CUDA(cudaMemcpy(d_embed_table_, table, bytes, cudaMemcpyHostToDevice));
  std::cout << "  CP embed table loaded on GPU: " << n_layers << "×" << vocab
            << "×" << dim << " (" << bytes / 1024 / 1024 << " MB)" << std::endl;
}

void TRTCPEngine::AppendEmbeddingFromTable(int layer_idx, int token_id) {
  // GPU→GPU copy: table[layer_idx][token_id] → ctx[ctx_len_]
  size_t src_offset =
      ((size_t)layer_idx * embed_vocab_ + token_id) * embed_dim_ * sizeof(float);
  size_t dst_offset = (size_t)ctx_len_ * hidden_dim_ * sizeof(float);
  size_t d_bytes = embed_dim_ * sizeof(float);
  CHECK_CUDA(cudaMemcpyAsync((char*)d_ctx_ + dst_offset,
                             (char*)d_embed_table_ + src_offset, d_bytes,
                             cudaMemcpyDeviceToDevice, stream_));
  ctx_len_++;
}

// ===========================================================================
// TRTCPKVEngine — CP engine with unified ONNX (cp_unified_bf16.engine)
// Runs one inference per codec frame: seq_len=2 [hidden, primary_emb],
// past_len=0 (stateless per frame). Outputs all 15 group logits at once.
// ===========================================================================
TRTCPKVEngine::TRTCPKVEngine(const std::string& engine_path, int n_cp_layers,
                              int hidden_dim, int n_heads, int head_dim,
                              int cp_vocab, int cp_out_groups, int /*max_past*/)
    : n_cp_layers_(n_cp_layers), hidden_dim_(hidden_dim), n_heads_(n_heads),
      head_dim_(head_dim), cp_vocab_(cp_vocab), cp_out_groups_(cp_out_groups) {
  auto data = LoadEngineFile(engine_path);
  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
  if (!engine_) {
    std::cerr << "[CPKV] Failed to deserialize engine: " << engine_path << std::endl;
    std::abort();
  }
  CHECK_CUDA(cudaStreamCreate(&stream_));

  // Check for dual-profile engine (Profile 0 = prefill, Profile 1 = decode)
  int n_profiles = engine_->getNbOptimizationProfiles();
  if (n_profiles >= 2) {
    has_dual_profiles_ = true;
    ctx_prefill_.reset(engine_->createExecutionContext());
    ctx_prefill_->setOptimizationProfileAsync(0, stream_);
    ctx_decode_.reset(engine_->createExecutionContext());
    ctx_decode_->setOptimizationProfileAsync(1, stream_);
    has_dual_ctx_ = true;
    context_.reset();  // not used
    std::cout << "  [CPKV] DUAL-PROFILE engine (" << n_profiles << " profiles)"
              << std::endl;
  } else {
    // Single-profile engine: create two contexts on the same profile.
    // Each context independently tracks shapes, so prefill (seq_len=2) and
    // decode (seq_len=1) don't interfere with each other's shape validation.
    has_dual_profiles_ = false;
    ctx_prefill_.reset(engine_->createExecutionContext());
    ctx_decode_.reset(engine_->createExecutionContext());
    has_dual_ctx_ = true;
    context_.reset(engine_->createExecutionContext());  // legacy fallback
    std::cout << "  [CPKV] Single-profile engine, dual-context mode" << std::endl;
  }

  // Detect single-head engine (has "gen_step" input, "logits" output)
  // vs unified engine (has "logits_all" output)
  {
    int n_io = engine_->getNbIOTensors();
    for (int i = 0; i < n_io; ++i) {
      const char* name = engine_->getIOTensorName(i);
      if (std::string(name) == "gen_step") {
        is_single_head_ = true;
        break;
      }
    }
  }

  // Detect KV output and logits dtypes from engine
  auto kv_dtype = engine_->getTensorDataType("new_past_key_0");
  kv_elem_bytes_ = TrtDtypeSize(kv_dtype);

  const char* logits_name = is_single_head_ ? "logits" : "logits_all";
  auto logits_dtype = engine_->getTensorDataType(logits_name);
  logits_elem_bytes_ = TrtDtypeSize(logits_dtype);
  logits_is_bf16_ = (logits_dtype == nvinfer1::DataType::kBF16);

  // Pre-cache tensor names to avoid string allocation per step
  cp_kv_names_.resize(2 * n_cp_layers);
  cp_new_kv_names_.resize(2 * n_cp_layers);
  for (int i = 0; i < n_cp_layers; ++i) {
    cp_kv_names_[2*i]     = "past_key_" + std::to_string(i);
    cp_kv_names_[2*i+1]   = "past_value_" + std::to_string(i);
    cp_new_kv_names_[2*i]   = "new_past_key_" + std::to_string(i);
    cp_new_kv_names_[2*i+1] = "new_past_value_" + std::to_string(i);
  }

  // Allocate embedding input: [1, 2, D] (seq_len=2 always)
  CHECK_CUDA(cudaMalloc(&d_embeds_, 2 * hidden_dim * sizeof(float)));
  // cache_position: [2] = {0, 1}
  CHECK_CUDA(cudaMalloc(&d_cache_pos_, 2 * sizeof(int64_t)));
  int64_t pos[2] = {0, 1};
  CHECK_CUDA(cudaMemcpy(d_cache_pos_, pos, 2 * sizeof(int64_t),
                        cudaMemcpyHostToDevice));
  // Small dummy buffer for zero-size past KV inputs (TRT needs non-null ptr)
  CHECK_CUDA(cudaMalloc(&d_kv_dummy_, 16));

  // KV output buffers for legacy parallel RunFrame
  size_t kv_out_size = (size_t)n_heads * 2 * head_dim * kv_elem_bytes_;
  d_kv_out_.resize(2 * n_cp_layers, nullptr);
  for (int i = 0; i < 2 * n_cp_layers; ++i) {
    CHECK_CUDA(cudaMalloc(&d_kv_out_[i], kv_out_size));
  }

  // KV double-buffer for autoregressive: max_past tokens
  max_past_ = 20;  // 2 prefill + up to 15 decode + margin
  size_t kv_buf_size = (size_t)n_heads * max_past_ * head_dim * kv_elem_bytes_;
  d_kv_a_.resize(2 * n_cp_layers, nullptr);
  d_kv_b_.resize(2 * n_cp_layers, nullptr);
  for (int i = 0; i < 2 * n_cp_layers; ++i) {
    CHECK_CUDA(cudaMalloc(&d_kv_a_[i], kv_buf_size));
    CHECK_CUDA(cudaMalloc(&d_kv_b_[i], kv_buf_size));
  }

  // Output logits buffer
  if (is_single_head_) {
    // Single-head: logits [1, cp_vocab]
    CHECK_CUDA(cudaMalloc(&d_logits_all_, (size_t)cp_vocab * logits_elem_bytes_));
    CHECK_CUDA(cudaMalloc(&d_logits_decode_, (size_t)cp_vocab * logits_elem_bytes_));
    // gen_step: scalar int64
    CHECK_CUDA(cudaMalloc(&d_gen_step_, sizeof(int64_t)));
  } else {
    // Unified: logits_all [cp_out_groups, cp_vocab]
    CHECK_CUDA(cudaMalloc(&d_logits_all_,
                          (size_t)cp_out_groups * cp_vocab * logits_elem_bytes_));
    CHECK_CUDA(cudaMalloc(&d_logits_decode_,
                          (size_t)cp_out_groups * cp_vocab * logits_elem_bytes_));
  }

  // GPU-resident autoregressive buffers
  CHECK_CUDA(cudaMalloc(&d_codes_out_, cp_out_groups * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_cache_pos_table_, max_past_ * sizeof(int64_t)));
  CHECK_CUDA(cudaMalloc(&d_cache_pos_single_, sizeof(int64_t)));
  // Pre-fill cache_pos_table with {0, 1, 2, ..., max_past-1}
  {
    std::vector<int64_t> pos_table(max_past_);
    for (int i = 0; i < max_past_; ++i) pos_table[i] = i;
    CHECK_CUDA(cudaMemcpy(d_cache_pos_table_, pos_table.data(),
                          max_past_ * sizeof(int64_t), cudaMemcpyHostToDevice));
  }

  // Profiling events
  CHECK_CUDA(cudaEventCreate(&ev_start_));
  CHECK_CUDA(cudaEventCreate(&ev_kernel_done_));
  CHECK_CUDA(cudaEventCreate(&ev_d2h_done_));

  std::cout << "  TRTCPKVEngine loaded: layers=" << n_cp_layers
            << " D=" << hidden_dim << " vocab=" << cp_vocab
            << " groups=" << cp_out_groups
            << " logits=" << (logits_is_bf16_ ? "bf16" : "fp32")
            << " single_head=" << is_single_head_
            << " dual_ctx=" << has_dual_ctx_
            << std::endl;
}

TRTCPKVEngine::~TRTCPKVEngine() {
  if (d_embeds_) cudaFree(d_embeds_);
  if (d_cache_pos_) cudaFree(d_cache_pos_);
  if (d_gen_step_) cudaFree(d_gen_step_);
  if (d_kv_dummy_) cudaFree(d_kv_dummy_);
  for (auto* p : d_kv_out_) if (p) cudaFree(p);
  for (auto* p : d_kv_a_) if (p) cudaFree(p);
  for (auto* p : d_kv_b_) if (p) cudaFree(p);
  if (d_logits_all_) cudaFree(d_logits_all_);
  if (d_logits_decode_) cudaFree(d_logits_decode_);
  if (d_embed_table_) cudaFree(d_embed_table_);
  if (d_codes_out_) cudaFree(d_codes_out_);
  if (d_cache_pos_table_) cudaFree(d_cache_pos_table_);
  if (d_cache_pos_single_) cudaFree(d_cache_pos_single_);
  if (ev_start_) cudaEventDestroy(ev_start_);
  if (ev_kernel_done_) cudaEventDestroy(ev_kernel_done_);
  if (ev_d2h_done_) cudaEventDestroy(ev_d2h_done_);
  if (stream_) cudaStreamDestroy(stream_);
}

void TRTCPKVEngine::RunFrameAutoregressive(
    const float* hidden, const float* primary_emb,
    int* codes_out,
    const float* embed_table, int embed_vocab) {
  // Autoregressive CP: 15 sequential steps.
  // If GPU embed table is available → GPU sampling (no CPU sync between steps).
  // Otherwise → CPU sampling (sync each step for D2H logits).
  auto* pctx = has_dual_ctx_ ? ctx_prefill_.get() : context_.get();
  auto* dctx = has_dual_ctx_ ? ctx_decode_.get()  : context_.get();
  int D = hidden_dim_;
  int n_groups = cp_out_groups_;  // 15
  size_t d_bytes = D * sizeof(float);
  const char* logits_out_name = is_single_head_ ? "logits" : "logits_all";

  // GPU sampling is slower due to TRT shape-change overhead when not syncing
  // between steps. CPU sampling with per-step sync is actually faster (~53ms
  // vs ~123ms) because the sync lets TRT finalize each step's execution plan.
  bool use_gpu_sample = false;  // disabled — CPU path is faster

  // RNG seed for GPU sampling
  unsigned long long rng_seed = std::chrono::high_resolution_clock::now()
      .time_since_epoch().count();

  // CPU sampling fallback (only used when no GPU embed table)
  static thread_local std::mt19937 cp_rng(
      std::chrono::high_resolution_clock::now().time_since_epoch().count());
  auto sample_topk_cpu = [&](const void* logits_buf, int group_idx) -> int {
    size_t offset = is_single_head_ ? 0 : (size_t)group_idx * cp_vocab_;
    std::vector<float> logits(cp_vocab_);
    if (logits_is_bf16_) {
      std::vector<uint16_t> raw(cp_vocab_);
      CHECK_CUDA(cudaMemcpyAsync(raw.data(), (char*)logits_buf + offset * 2,
                                  cp_vocab_ * 2, cudaMemcpyDeviceToHost, stream_));
      CHECK_CUDA(cudaStreamSynchronize(stream_));
      for (int v = 0; v < cp_vocab_; ++v) {
        uint32_t bits = (uint32_t)raw[v] << 16;
        std::memcpy(&logits[v], &bits, 4);
      }
    } else {
      CHECK_CUDA(cudaMemcpyAsync(logits.data(),
                                  (char*)logits_buf + offset * sizeof(float),
                                  cp_vocab_ * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream_));
      CHECK_CUDA(cudaStreamSynchronize(stream_));
    }
    std::vector<std::pair<float, int>> vals(cp_vocab_);
    for (int v = 0; v < cp_vocab_; ++v) vals[v] = {logits[v], v};
    std::partial_sort(vals.begin(), vals.begin() + 50, vals.end(),
                      [](auto& a, auto& b) { return a.first > b.first; });
    double max_v = vals[0].first;
    std::vector<double> probs(50);
    double sum = 0;
    for (int v = 0; v < 50; ++v) {
      probs[v] = std::exp((vals[v].first - max_v) / 0.9);
      sum += probs[v];
    }
    for (auto& p : probs) p /= sum;
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return vals[dist(cp_rng)].second;
  };

  // ---- Step 0: Prefill [hidden, primary_emb] (seq_len=2, past_len=0) ----
  CHECK_CUDA(cudaMemcpyAsync(d_embeds_, hidden, d_bytes,
                             cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA(cudaMemcpyAsync((char*)d_embeds_ + d_bytes, primary_emb, d_bytes,
                             cudaMemcpyHostToDevice, stream_));

  pctx->setInputShape("inputs_embeds", nvinfer1::Dims3{1, 2, D});
  pctx->setTensorAddress("inputs_embeds", d_embeds_);
  pctx->setInputShape("cache_position", nvinfer1::Dims{1, {2}});
  pctx->setTensorAddress("cache_position", d_cache_pos_table_);

  for (int i = 0; i < n_cp_layers_; ++i) {
    pctx->setInputShape(cp_kv_names_[2*i].c_str(),
                        nvinfer1::Dims4{1, n_heads_, 0, head_dim_});
    pctx->setInputShape(cp_kv_names_[2*i+1].c_str(),
                        nvinfer1::Dims4{1, n_heads_, 0, head_dim_});
    pctx->setTensorAddress(cp_kv_names_[2*i].c_str(), d_kv_dummy_);
    pctx->setTensorAddress(cp_kv_names_[2*i+1].c_str(), d_kv_dummy_);
    pctx->setTensorAddress(cp_new_kv_names_[2*i].c_str(), d_kv_b_[2*i]);
    pctx->setTensorAddress(cp_new_kv_names_[2*i+1].c_str(), d_kv_b_[2*i+1]);
  }
  pctx->setTensorAddress(logits_out_name, d_logits_all_);
  if (is_single_head_) {
    int64_t gs = 0;
    CHECK_CUDA(cudaMemcpyAsync(d_gen_step_, &gs, sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream_));
    pctx->setTensorAddress("gen_step", d_gen_step_);
  }

  bool ok = pctx->enqueueV3(stream_);
  if (!ok) {
    std::cerr << "[CPKV-AR] prefill enqueueV3 FAILED!" << std::endl;
    std::abort();
  }

  // Sample group 0
  if (use_gpu_sample) {
    const void* logits_ptr = is_single_head_
        ? d_logits_all_
        : d_logits_all_;  // offset 0 for group 0
    launchCPSampleAndEmbed(
        stream_, logits_ptr, logits_is_bf16_,
        reinterpret_cast<const float*>(d_embed_table_),
        reinterpret_cast<float*>(d_embeds_),  // write embed for next step
        d_codes_out_ + 0,
        reinterpret_cast<int64_t*>(d_cache_pos_),  // write next cache_pos = 2
        cp_vocab_, D, /*layer_idx=*/0,
        /*next_cache_pos=*/2,
        /*top_k=*/50, /*temperature=*/0.9f,
        rng_seed, rng_counter_++);
    // No sync — stream ordering ensures kernel completes before next enqueueV3.
  } else {
    codes_out[0] = sample_topk_cpu(d_logits_all_, 0);
  }

  // ---- Pre-bind decode context ----
  dctx->setInputShape("inputs_embeds", nvinfer1::Dims3{1, 1, D});
  dctx->setTensorAddress("inputs_embeds", d_embeds_);
  dctx->setInputShape("cache_position", nvinfer1::Dims{1, {1}});
  dctx->setTensorAddress("cache_position", d_cache_pos_);
  dctx->setTensorAddress(logits_out_name, d_logits_decode_);
  if (is_single_head_) {
    dctx->setTensorAddress("gen_step", d_gen_step_);
  }

  // ---- Steps 1-14: Decode ----
  int past_len = 2;
  auto* kv_read = &d_kv_b_;
  auto* kv_write = &d_kv_a_;

  for (int j = 1; j < n_groups; ++j) {
    if (use_gpu_sample) {
      // GPU path: embed already in d_embeds_ from previous kernel,
      // cache_pos already written by previous kernel.
      // Only need to upload gen_step for single-head engine.
      if (is_single_head_) {
        int64_t gs = j;
        CHECK_CUDA(cudaMemcpyAsync(d_gen_step_, &gs, sizeof(int64_t),
                                   cudaMemcpyHostToDevice, stream_));
      }
    } else {
      // CPU path: upload embed + cache_pos from host
      const float* emb = embed_table + ((size_t)(j - 1) * embed_vocab + codes_out[j - 1]) * D;
      CHECK_CUDA(cudaMemcpyAsync(d_embeds_, emb, d_bytes,
                                 cudaMemcpyHostToDevice, stream_));
      int64_t pos1 = past_len;
      CHECK_CUDA(cudaMemcpyAsync(d_cache_pos_, &pos1, sizeof(int64_t),
                                 cudaMemcpyHostToDevice, stream_));
      if (is_single_head_) {
        int64_t gs = j;
        CHECK_CUDA(cudaMemcpyAsync(d_gen_step_, &gs, sizeof(int64_t),
                                   cudaMemcpyHostToDevice, stream_));
      }
    }

    // Update KV shapes and pointers
    for (int i = 0; i < n_cp_layers_; ++i) {
      dctx->setInputShape(cp_kv_names_[2*i].c_str(),
                          nvinfer1::Dims4{1, n_heads_, past_len, head_dim_});
      dctx->setInputShape(cp_kv_names_[2*i+1].c_str(),
                          nvinfer1::Dims4{1, n_heads_, past_len, head_dim_});
      dctx->setTensorAddress(cp_kv_names_[2*i].c_str(), (*kv_read)[2*i]);
      dctx->setTensorAddress(cp_kv_names_[2*i+1].c_str(), (*kv_read)[2*i+1]);
      dctx->setTensorAddress(cp_new_kv_names_[2*i].c_str(), (*kv_write)[2*i]);
      dctx->setTensorAddress(cp_new_kv_names_[2*i+1].c_str(), (*kv_write)[2*i+1]);
    }

    ok = dctx->enqueueV3(stream_);
    if (!ok) {
      std::cerr << "[CPKV-AR] decode enqueueV3 FAILED! step=" << j
                << " past=" << past_len << std::endl;
      std::abort();
    }

    if (use_gpu_sample) {
      // GPU sample kernel: reads logits from GPU, writes embed + code + cache_pos
      const void* logits_ptr;
      if (is_single_head_) {
        logits_ptr = d_logits_decode_;  // [1, vocab]
      } else {
        logits_ptr = (const char*)d_logits_decode_ +
                     (size_t)j * cp_vocab_ * logits_elem_bytes_;
      }
      int64_t next_pos = past_len + 1;
      launchCPSampleAndEmbed(
          stream_, logits_ptr, logits_is_bf16_,
          reinterpret_cast<const float*>(d_embed_table_),
          reinterpret_cast<float*>(d_embeds_),
          d_codes_out_ + j,
          reinterpret_cast<int64_t*>(d_cache_pos_),
          cp_vocab_, D, /*layer_idx=*/j,
          next_pos,
          /*top_k=*/50, /*temperature=*/0.9f,
          rng_seed, rng_counter_++);
      // No sync — stream ordering ensures kernel completes before next enqueueV3.
      // setInputShape is CPU-side metadata, doesn't need GPU idle.
    } else {
      codes_out[j] = sample_topk_cpu(d_logits_decode_, j);
    }

    past_len++;
    std::swap(kv_read, kv_write);
  }

  // GPU path: copy codes from GPU to host
  if (use_gpu_sample) {
    CHECK_CUDA(cudaMemcpy(codes_out, d_codes_out_,
                          n_groups * sizeof(int), cudaMemcpyDeviceToHost));
  }
}

void TRTCPKVEngine::RunFrameGPU(const float* hidden, const float* primary_emb,
                                int* codes_out) {
  // GPU-resident autoregressive CP with dual-context optimization.
  // Two separate TRT execution contexts:
  //   ctx_prefill_: always seq_len=2, past_len=0 (step 0)
  //   ctx_decode_:  always seq_len=1, past_len varies 2→16 (steps 1-14)
  // This eliminates shape-change overhead: each context only sees its own
  // fixed seq_len, so TRT doesn't need to re-validate between prefill→decode.
  //
  // Between TRT steps, a CUDA kernel handles sampling and embedding lookup,
  // eliminating all CPU<->GPU synchronization points (single D2H at the end).

  if (!d_embed_table_) {
    std::cerr << "[CPKV-GPU] RunFrameGPU requires embed table on GPU. "
              << "Call LoadEmbedTable() first." << std::endl;
    std::abort();
  }

  auto* pctx = has_dual_ctx_ ? ctx_prefill_.get() : context_.get();
  auto* dctx = has_dual_ctx_ ? ctx_decode_.get()  : context_.get();
  int D = hidden_dim_;
  int n_groups = cp_out_groups_;  // 15
  const char* logits_out_name = is_single_head_ ? "logits" : "logits_all";

  // Use a fixed seed derived from time for this frame, sequence from counter
  unsigned long long rng_seed =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();

  if (profiling_) {
    CHECK_CUDA(cudaEventRecord(ev_start_, stream_));
  }

  // ---- Step 0: Prefill [hidden, primary_emb] (seq_len=2, past_len=0) ----
  size_t d_bytes = D * sizeof(float);
  CHECK_CUDA(cudaMemcpyAsync(d_embeds_, hidden, d_bytes,
                             cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA(cudaMemcpyAsync((char*)d_embeds_ + d_bytes, primary_emb, d_bytes,
                             cudaMemcpyHostToDevice, stream_));

  // Bind prefill context: seq_len=2, past_len=0
  pctx->setInputShape("inputs_embeds", nvinfer1::Dims3{1, 2, D});
  pctx->setTensorAddress("inputs_embeds", d_embeds_);
  pctx->setInputShape("cache_position", nvinfer1::Dims{1, {2}});
  pctx->setTensorAddress("cache_position", d_cache_pos_table_);

  for (int i = 0; i < n_cp_layers_; ++i) {
    pctx->setInputShape(cp_kv_names_[2*i].c_str(),
                        nvinfer1::Dims4{1, n_heads_, 0, head_dim_});
    pctx->setInputShape(cp_kv_names_[2*i+1].c_str(),
                        nvinfer1::Dims4{1, n_heads_, 0, head_dim_});
    pctx->setTensorAddress(cp_kv_names_[2*i].c_str(), d_kv_dummy_);
    pctx->setTensorAddress(cp_kv_names_[2*i+1].c_str(), d_kv_dummy_);
    pctx->setTensorAddress(cp_new_kv_names_[2*i].c_str(), d_kv_b_[2*i]);
    pctx->setTensorAddress(cp_new_kv_names_[2*i+1].c_str(), d_kv_b_[2*i+1]);
  }
  pctx->setTensorAddress(logits_out_name, d_logits_all_);
  if (is_single_head_) {
    int64_t gs = 0;
    CHECK_CUDA(cudaMemcpyAsync(d_gen_step_, &gs, sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream_));
    pctx->setTensorAddress("gen_step", d_gen_step_);
  }

  bool ok = pctx->enqueueV3(stream_);
  if (!ok) {
    std::cerr << "[CPKV-GPU] prefill enqueueV3 FAILED!" << std::endl;
    std::abort();
  }

  // Launch GPU sample kernel for group 0
  const void* logits_g0 = (const char*)d_logits_all_;
  launchCPSampleAndEmbed(
      stream_, logits_g0, logits_is_bf16_,
      reinterpret_cast<const float*>(d_embed_table_),
      reinterpret_cast<float*>(d_embeds_),
      d_codes_out_ + 0,
      d_cache_pos_single_,
      cp_vocab_, D, /*layer_idx=*/0,
      /*next_cache_pos=*/2,
      /*top_k=*/50, /*temperature=*/0.9f,
      rng_seed, rng_counter_++);

  // ---- Pre-bind decode context addresses that don't change between steps ----
  dctx->setInputShape("inputs_embeds", nvinfer1::Dims3{1, 1, D});
  dctx->setTensorAddress("inputs_embeds", d_embeds_);
  dctx->setInputShape("cache_position", nvinfer1::Dims{1, {1}});
  dctx->setTensorAddress("cache_position", d_cache_pos_single_);
  dctx->setTensorAddress(logits_out_name, d_logits_decode_);
  if (is_single_head_) {
    dctx->setTensorAddress("gen_step", d_gen_step_);
  }

  // ---- Steps 1-14: Decode (seq_len=1, past_len grows) ----
  // NOTE: No per-step sync — all 14 decode steps queued async on stream.
  // This path is 2x slower than RunFrameAutoregressive on single-profile
  // engines due to TRT's shape-change overhead without sync points.
  // Intended for future use with a dual-profile engine where seq_len is
  // fixed within each profile.
  auto* kv_read = &d_kv_b_;
  auto* kv_write = &d_kv_a_;

  for (int j = 1; j < n_groups; ++j) {
    int past_len = j + 1;  // step 0 produced 2 KV entries, each step adds 1

    // Only update the changing shapes (past_len) and KV buffer pointers
    for (int i = 0; i < n_cp_layers_; ++i) {
      dctx->setInputShape(cp_kv_names_[2*i].c_str(),
                          nvinfer1::Dims4{1, n_heads_, past_len, head_dim_});
      dctx->setInputShape(cp_kv_names_[2*i+1].c_str(),
                          nvinfer1::Dims4{1, n_heads_, past_len, head_dim_});
      dctx->setTensorAddress(cp_kv_names_[2*i].c_str(), (*kv_read)[2*i]);
      dctx->setTensorAddress(cp_kv_names_[2*i+1].c_str(), (*kv_read)[2*i+1]);
      dctx->setTensorAddress(cp_new_kv_names_[2*i].c_str(), (*kv_write)[2*i]);
      dctx->setTensorAddress(cp_new_kv_names_[2*i+1].c_str(), (*kv_write)[2*i+1]);
    }

    if (is_single_head_) {
      int64_t gs = j;
      CHECK_CUDA(cudaMemcpyAsync(d_gen_step_, &gs, sizeof(int64_t),
                                 cudaMemcpyHostToDevice, stream_));
    }

    ok = dctx->enqueueV3(stream_);
    if (!ok) {
      std::cerr << "[CPKV-GPU] decode enqueueV3 FAILED! step=" << j
                << " past=" << past_len << std::endl;
      std::abort();
    }

    // Launch GPU sample kernel for group j
    size_t logits_offset = (size_t)j * cp_vocab_ * logits_elem_bytes_;
    const void* logits_gj = (const char*)d_logits_decode_ + logits_offset;

    int64_t next_pos = past_len + 1;
    launchCPSampleAndEmbed(
        stream_, logits_gj, logits_is_bf16_,
        reinterpret_cast<const float*>(d_embed_table_),
        reinterpret_cast<float*>(d_embeds_),
        d_codes_out_ + j,
        d_cache_pos_single_,
        cp_vocab_, D, /*layer_idx=*/j,
        next_pos,
        /*top_k=*/50, /*temperature=*/0.9f,
        rng_seed, rng_counter_++);

    std::swap(kv_read, kv_write);
  }

  if (profiling_) {
    CHECK_CUDA(cudaEventRecord(ev_kernel_done_, stream_));
  }

  // ---- Single D2H sync: read all 15 sampled codes ----
  CHECK_CUDA(cudaMemcpyAsync(codes_out, d_codes_out_,
                             n_groups * sizeof(int),
                             cudaMemcpyDeviceToHost, stream_));

  CHECK_CUDA(cudaEventRecord(ev_d2h_done_, stream_));
  CHECK_CUDA(cudaEventSynchronize(ev_d2h_done_));

  if (profiling_) {
    StepTiming t;
    cudaEventElapsedTime(&t.kernel_ms, ev_start_, ev_kernel_done_);
    cudaEventElapsedTime(&t.d2h_ms, ev_kernel_done_, ev_d2h_done_);
    cudaEventElapsedTime(&t.total_ms, ev_start_, ev_d2h_done_);
    stats_.Add(t);
  }
}

void TRTCPKVEngine::RunFrame(const float* hidden, const float* primary_emb,
                              float* logits_out) {
  auto* ctx = has_dual_ctx_ ? ctx_prefill_.get() : context_.get();
  const char* logits_out_name = is_single_head_ ? "logits" : "logits_all";

  // H2D: copy [hidden, primary_emb] → d_embeds_ [1, 2, D]
  size_t d_bytes = hidden_dim_ * sizeof(float);
  CHECK_CUDA(cudaMemcpyAsync(d_embeds_, hidden, d_bytes,
                             cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA(cudaMemcpyAsync((char*)d_embeds_ + d_bytes, primary_emb, d_bytes,
                             cudaMemcpyHostToDevice, stream_));
  // d_cache_pos_ is pre-set to {0, 1} — no need to update per frame

  // Bind inputs
  ctx->setInputShape("inputs_embeds", nvinfer1::Dims3{1, 2, hidden_dim_});
  ctx->setTensorAddress("inputs_embeds", d_embeds_);

  ctx->setInputShape("cache_position", nvinfer1::Dims{1, {2}});
  ctx->setTensorAddress("cache_position", d_cache_pos_);

  // Past KV: past_len=0 (empty). TRT handles 0-size tensors if shapes are set.
  for (int i = 0; i < n_cp_layers_; ++i) {
    std::string kn = "past_key_" + std::to_string(i);
    std::string vn = "past_value_" + std::to_string(i);
    // past_len=0: set shape to [1, n_heads, 0, head_dim]
    ctx->setInputShape(kn.c_str(),
                       nvinfer1::Dims4{1, n_heads_, 0, head_dim_});
    ctx->setInputShape(vn.c_str(),
                       nvinfer1::Dims4{1, n_heads_, 0, head_dim_});
    // TRT requires a valid (non-null) pointer even for 0-size tensors.
    ctx->setTensorAddress(kn.c_str(), d_kv_dummy_);
    ctx->setTensorAddress(vn.c_str(), d_kv_dummy_);

    std::string new_kn = "new_past_key_" + std::to_string(i);
    std::string new_vn = "new_past_value_" + std::to_string(i);
    ctx->setTensorAddress(new_kn.c_str(), d_kv_out_[2 * i]);
    ctx->setTensorAddress(new_vn.c_str(), d_kv_out_[2 * i + 1]);
  }

  ctx->setTensorAddress(logits_out_name, d_logits_all_);
  if (is_single_head_) {
    int64_t gs = 0;
    CHECK_CUDA(cudaMemcpyAsync(d_gen_step_, &gs, sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream_));
    ctx->setTensorAddress("gen_step", d_gen_step_);
  }

  if (profiling_) {
    CHECK_CUDA(cudaEventRecord(ev_start_, stream_));
  }

  bool ok = ctx->enqueueV3(stream_);
  if (!ok) {
    std::cerr << "[CPKV] enqueueV3 FAILED!" << std::endl;
    std::abort();
  }

  if (profiling_) {
    CHECK_CUDA(cudaEventRecord(ev_kernel_done_, stream_));
  }

  // D2H: copy all 15 group logits at once
  size_t total_logits = (size_t)cp_out_groups_ * cp_vocab_;
  if (logits_is_bf16_) {
    std::vector<uint16_t> raw(total_logits);
    CHECK_CUDA(cudaMemcpyAsync(raw.data(), d_logits_all_,
                               total_logits * 2, cudaMemcpyDeviceToHost, stream_));
    CHECK_CUDA(cudaEventRecord(ev_d2h_done_, stream_));
    CHECK_CUDA(cudaEventSynchronize(ev_d2h_done_));
    for (size_t j = 0; j < total_logits; ++j) {
      uint32_t bits = (uint32_t)raw[j] << 16;
      std::memcpy(&logits_out[j], &bits, 4);
    }
  } else {
    CHECK_CUDA(cudaMemcpyAsync(logits_out, d_logits_all_,
                               total_logits * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_));
    CHECK_CUDA(cudaEventRecord(ev_d2h_done_, stream_));
    CHECK_CUDA(cudaEventSynchronize(ev_d2h_done_));
  }

  if (profiling_) {
    StepTiming t;
    cudaEventElapsedTime(&t.kernel_ms, ev_start_, ev_kernel_done_);
    cudaEventElapsedTime(&t.d2h_ms, ev_kernel_done_, ev_d2h_done_);
    t.total_ms = t.kernel_ms + t.d2h_ms;
    stats_.Add(t);
  }
}

void TRTCPKVEngine::LoadEmbedTable(const float* table, int n_layers, int vocab,
                                    int dim) {
  embed_n_layers_ = n_layers;
  embed_vocab_ = vocab;
  embed_dim_ = dim;
  size_t bytes = (size_t)n_layers * vocab * dim * sizeof(float);
  CHECK_CUDA(cudaMalloc(&d_embed_table_, bytes));
  CHECK_CUDA(cudaMemcpy(d_embed_table_, table, bytes, cudaMemcpyHostToDevice));
  std::cout << "  CPKV embed table loaded on GPU: " << n_layers << "×" << vocab
            << "×" << dim << " (" << bytes / 1024 / 1024 << " MB)" << std::endl;
}

// ===========================================================================
// TRTVocoderEngine
// ===========================================================================
TRTVocoderEngine::TRTVocoderEngine(const std::string& engine_path,
                                   int max_frames, int max_samples)
    : max_frames_(max_frames), max_samples_(max_samples) {
  auto data = LoadEngineFile(engine_path);
  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
  if (!engine_) {
    std::cerr << "[Vocoder] Failed to deserialize engine: " << engine_path
              << std::endl;
    std::abort();
  }
  context_.reset(engine_->createExecutionContext());
  CHECK_CUDA(cudaStreamCreate(&stream_));
  CHECK_CUDA(cudaEventCreate(&ev_done_));

  // Allocate GPU buffers
  CHECK_CUDA(cudaMalloc(&d_codes_, (size_t)max_frames * 16 * sizeof(int64_t)));
  CHECK_CUDA(cudaMalloc(&d_audio_values_, (size_t)max_samples * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_lengths_, sizeof(int64_t)));

  // Detect output tensor names
  int n_io = engine_->getNbIOTensors();
  for (int i = 0; i < n_io; ++i) {
    std::string name = engine_->getIOTensorName(i);
    auto mode = engine_->getTensorIOMode(name.c_str());
    if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
      if (audio_values_name_.empty() && name.find("length") == std::string::npos) {
        audio_values_name_ = name;
      } else if (lengths_name_.empty()) {
        lengths_name_ = name;
      }
    }
  }
  if (audio_values_name_.empty()) audio_values_name_ = "audio_values";
  if (lengths_name_.empty()) lengths_name_ = "lengths";

  std::cout << "  TRTVocoderEngine loaded: max_frames=" << max_frames
            << " max_samples=" << max_samples
            << " out=[" << audio_values_name_ << "," << lengths_name_ << "]"
            << std::endl;
}

TRTVocoderEngine::~TRTVocoderEngine() {
  if (d_codes_) cudaFree(d_codes_);
  if (d_audio_values_) cudaFree(d_audio_values_);
  if (d_lengths_) cudaFree(d_lengths_);
  if (ev_done_) cudaEventDestroy(ev_done_);
  if (stream_) cudaStreamDestroy(stream_);
}

std::vector<float> TRTVocoderEngine::Run(const int64_t* codes, int n_frames,
                                          int n_groups) {
  if (!context_) return {};
  if (n_frames <= 0 || n_frames > max_frames_) {
    std::cerr << "[Vocoder] Invalid n_frames=" << n_frames << std::endl;
    return {};
  }

  // H2D: copy codes [1, n_frames, 16]
  size_t codes_bytes = (size_t)n_frames * n_groups * sizeof(int64_t);
  CHECK_CUDA(cudaMemcpyAsync(d_codes_, codes, codes_bytes,
                             cudaMemcpyHostToDevice, stream_));

  // Set input shape: audio_codes [1, n_frames, 16]
  context_->setInputShape("audio_codes",
                          nvinfer1::Dims3{1, n_frames, n_groups});
  context_->setTensorAddress("audio_codes", d_codes_);
  context_->setTensorAddress(audio_values_name_.c_str(), d_audio_values_);
  context_->setTensorAddress(lengths_name_.c_str(), d_lengths_);

  bool ok = context_->enqueueV3(stream_);
  if (!ok) {
    std::cerr << "[Vocoder] enqueueV3 FAILED!" << std::endl;
    return {};
  }

  // Read lengths first (1 int64)
  int64_t valid_samples = 0;
  CHECK_CUDA(cudaMemcpyAsync(&valid_samples, d_lengths_, sizeof(int64_t),
                             cudaMemcpyDeviceToHost, stream_));
  CHECK_CUDA(cudaEventRecord(ev_done_, stream_));
  CHECK_CUDA(cudaEventSynchronize(ev_done_));

  // Clamp to safe range
  if (valid_samples <= 0 || valid_samples > max_samples_) {
    valid_samples = max_samples_;
  }

  // Copy audio output
  std::vector<float> audio(valid_samples);
  CHECK_CUDA(cudaMemcpyAsync(audio.data(), d_audio_values_,
                             valid_samples * sizeof(float),
                             cudaMemcpyDeviceToHost, stream_));
  CHECK_CUDA(cudaEventRecord(ev_done_, stream_));
  CHECK_CUDA(cudaEventSynchronize(ev_done_));

  return audio;
}

// ===========================================================================
// TRTASRPrefillEngine
// ===========================================================================
TRTASRPrefillEngine::TRTASRPrefillEngine(const std::string& engine_path,
                                         int n_layers, int hidden_dim,
                                         int n_heads, int head_dim,
                                         int vocab_size, int max_seq)
    : n_layers_(n_layers), hidden_dim_(hidden_dim), n_heads_(n_heads),
      head_dim_(head_dim), vocab_size_(vocab_size), max_seq_(max_seq) {
  auto data = LoadEngineFile(engine_path);
  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
  if (!engine_) {
    std::cerr << "[ASRPrefill] Failed to deserialize engine: " << engine_path
              << std::endl;
    return;
  }
  ctx_.reset(engine_->createExecutionContext());
  CHECK_CUDA(cudaStreamCreate(&stream_));
  CHECK_CUDA(cudaEventCreate(&ev_done_));

  // Detect KV dtype from engine
  auto kv_dtype = engine_->getTensorDataType("past_key_0");
  kv_elem_bytes_ = TrtDtypeSize(kv_dtype);

  // Allocate input buffers
  CHECK_CUDA(cudaMalloc(&d_input_ids_, (size_t)max_seq_ * sizeof(int64_t)));
  CHECK_CUDA(cudaMalloc(&d_position_ids_, (size_t)max_seq_ * sizeof(int64_t)));
  CHECK_CUDA(cudaMalloc(&d_audio_features_,
                        (size_t)max_seq_ * hidden_dim_ * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_audio_offset_, sizeof(int64_t)));

  // Detect logits dtype
  auto logits_dtype = engine_->getTensorDataType("logits");
  size_t logits_elem = TrtDtypeSize(logits_dtype);
  CHECK_CUDA(cudaMalloc(&d_logits_,
                        (size_t)max_seq_ * vocab_size_ * logits_elem));

  // Allocate KV output buffers: [1, n_heads, max_seq, head_dim] per tensor
  d_kv_.resize(2 * n_layers_, nullptr);
  size_t kv_bytes = (size_t)n_heads_ * max_seq_ * head_dim_ * kv_elem_bytes_;
  for (int i = 0; i < 2 * n_layers_; ++i) {
    CHECK_CUDA(cudaMalloc(&d_kv_[i], kv_bytes));
  }

  std::cout << "[ASRPrefill] Loaded: " << engine_path << " KV dtype="
            << kv_elem_bytes_ << "B max_seq=" << max_seq_ << std::endl;
}

TRTASRPrefillEngine::~TRTASRPrefillEngine() {
  if (d_input_ids_) cudaFree(d_input_ids_);
  if (d_position_ids_) cudaFree(d_position_ids_);
  if (d_audio_features_) cudaFree(d_audio_features_);
  if (d_audio_offset_) cudaFree(d_audio_offset_);
  if (d_logits_) cudaFree(d_logits_);
  for (auto p : d_kv_) if (p) cudaFree(p);
  if (ev_done_) cudaEventDestroy(ev_done_);
  if (stream_) cudaStreamDestroy(stream_);
}

TRTASRPrefillEngine::PrefillOutput TRTASRPrefillEngine::Run(
    const std::vector<int64_t>& input_ids,
    const std::vector<int64_t>& position_ids,
    const float* audio_features, int audio_len, int64_t audio_offset) {
  assert(engine_ && ctx_);
  int seq_len = (int)input_ids.size();
  assert(seq_len > 0 && seq_len <= max_seq_);
  assert(audio_len > 0 && audio_len <= max_seq_);

  // H2D: copy inputs
  CHECK_CUDA(cudaMemcpyAsync(d_input_ids_, input_ids.data(),
                             seq_len * sizeof(int64_t),
                             cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA(cudaMemcpyAsync(d_position_ids_, position_ids.data(),
                             seq_len * sizeof(int64_t),
                             cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA(cudaMemcpyAsync(d_audio_features_, audio_features,
                             (size_t)audio_len * hidden_dim_ * sizeof(float),
                             cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA(cudaMemcpyAsync(d_audio_offset_, &audio_offset, sizeof(int64_t),
                             cudaMemcpyHostToDevice, stream_));

  // Set dynamic shapes
  ctx_->setInputShape("input_ids", nvinfer1::Dims2{1, seq_len});
  ctx_->setInputShape("position_ids", nvinfer1::Dims2{1, seq_len});
  ctx_->setInputShape("audio_features", nvinfer1::Dims3{1, audio_len, hidden_dim_});
  ctx_->setInputShape("audio_offset", nvinfer1::Dims{1, {1}});

  // Bind inputs
  ctx_->setTensorAddress("input_ids", d_input_ids_);
  ctx_->setTensorAddress("position_ids", d_position_ids_);
  ctx_->setTensorAddress("audio_features", d_audio_features_);
  ctx_->setTensorAddress("audio_offset", d_audio_offset_);

  // Bind output: logits
  ctx_->setTensorAddress("logits", d_logits_);

  // Bind KV outputs
  for (int i = 0; i < n_layers_; ++i) {
    std::string kn = "past_key_" + std::to_string(i);
    std::string vn = "past_value_" + std::to_string(i);
    ctx_->setTensorAddress(kn.c_str(), d_kv_[2 * i]);
    ctx_->setTensorAddress(vn.c_str(), d_kv_[2 * i + 1]);
  }

  // Bind last_hidden if present (we allocate a temp buffer)
  void* d_hidden_tmp = nullptr;
  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    std::string tn = engine_->getIOTensorName(i);
    if (tn == "last_hidden") {
      CHECK_CUDA(cudaMalloc(&d_hidden_tmp,
                            (size_t)seq_len * hidden_dim_ * sizeof(float)));
      ctx_->setTensorAddress("last_hidden", d_hidden_tmp);
      break;
    }
  }

  // Execute
  bool ok = ctx_->enqueueV3(stream_);
  if (!ok) {
    std::cerr << "[ASRPrefill] enqueueV3 FAILED!" << std::endl;
    if (d_hidden_tmp) cudaFree(d_hidden_tmp);
    std::abort();
  }

  // Detect logits dtype and copy to CPU as FP32
  auto logits_dtype = engine_->getTensorDataType("logits");
  size_t logits_elem = TrtDtypeSize(logits_dtype);

  PrefillOutput out;
  out.seq_len = seq_len;
  out.logits.resize((size_t)seq_len * vocab_size_);

  if (logits_elem == sizeof(float)) {
    // FP32: direct copy
    CHECK_CUDA(cudaMemcpyAsync(out.logits.data(), d_logits_,
                               (size_t)seq_len * vocab_size_ * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_));
    CHECK_CUDA(cudaEventRecord(ev_done_, stream_));
    CHECK_CUDA(cudaEventSynchronize(ev_done_));
  } else {
    // BF16 or FP16: copy as raw uint16, then convert on CPU
    std::vector<uint16_t> raw((size_t)seq_len * vocab_size_);
    CHECK_CUDA(cudaMemcpyAsync(raw.data(), d_logits_,
                               (size_t)seq_len * vocab_size_ * 2,
                               cudaMemcpyDeviceToHost, stream_));
    CHECK_CUDA(cudaEventRecord(ev_done_, stream_));
    CHECK_CUDA(cudaEventSynchronize(ev_done_));

    bool is_bf16 = (logits_dtype == nvinfer1::DataType::kBF16);
    for (size_t j = 0; j < (size_t)seq_len * vocab_size_; ++j) {
      if (is_bf16) {
        // BF16 -> FP32: zero-extend 16-bit mantissa to 32-bit
        uint32_t bits = (uint32_t)raw[j] << 16;
        std::memcpy(&out.logits[j], &bits, 4);
      } else {
        // FP16 -> FP32
        uint32_t sign = (raw[j] >> 15) & 1;
        uint32_t exp  = (raw[j] >> 10) & 0x1F;
        uint32_t mant =  raw[j]        & 0x3FF;
        uint32_t bits;
        if (exp == 0x1F) {
          bits = (sign << 31) | (0xFF << 23) | (mant << 13);
        } else if (exp == 0) {
          bits = sign << 31;
        } else {
          bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        }
        std::memcpy(&out.logits[j], &bits, 4);
      }
    }
  }

  if (d_hidden_tmp) cudaFree(d_hidden_tmp);

  // Debug: last position logits stats
  {
    int last = seq_len - 1;
    const float* lp = out.logits.data() + (size_t)last * vocab_size_;
    float lmin = lp[0], lmax = lp[0]; int argmax = 0;
    for (int k = 1; k < vocab_size_; ++k) {
      if (lp[k] < lmin) lmin = lp[k];
      if (lp[k] > lmax) { lmax = lp[k]; argmax = k; }
    }
    std::cerr << "  [ASRPrefill] logits range=[" << lmin << "," << lmax
              << "] argmax=" << argmax << " seq_len=" << seq_len << std::endl;
  }

  return out;
}

void TRTASRPrefillEngine::SeedDecoder(TRTTalkerEngine* decoder, int seq_len) {
  // KV outputs are BF16 on GPU. decoder->SeedKV() accepts FP32 CPU arrays
  // and handles FP32->FP16 conversion internally. So we:
  //   1. D2H: copy BF16 KV to CPU as raw uint16
  //   2. BF16->FP32 convert on CPU
  //   3. Call SeedKV()

  size_t kv_elem = (size_t)n_heads_ * seq_len * head_dim_;
  size_t kv_bytes = kv_elem * kv_elem_bytes_;

  std::vector<std::vector<float>> kv_cpu(2 * n_layers_);
  std::vector<const float*> kv_ptrs(2 * n_layers_);

  for (int i = 0; i < 2 * n_layers_; ++i) {
    kv_cpu[i].resize(kv_elem);

    if (kv_elem_bytes_ == 4) {
      // FP32: direct copy
      CHECK_CUDA(cudaMemcpyAsync(kv_cpu[i].data(), d_kv_[i],
                                 kv_elem * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream_));
      CHECK_CUDA(cudaEventRecord(ev_done_, stream_));
      CHECK_CUDA(cudaEventSynchronize(ev_done_));
    } else {
      // BF16 or FP16
      std::vector<uint16_t> raw(kv_elem);
      CHECK_CUDA(cudaMemcpyAsync(raw.data(), d_kv_[i], kv_bytes,
                                 cudaMemcpyDeviceToHost, stream_));
      CHECK_CUDA(cudaEventRecord(ev_done_, stream_));
      CHECK_CUDA(cudaEventSynchronize(ev_done_));

      bool is_bf16 = (kv_elem_bytes_ == 2);
      for (size_t j = 0; j < kv_elem; ++j) {
        if (is_bf16) {
          uint32_t bits = (uint32_t)raw[j] << 16;
          std::memcpy(&kv_cpu[i][j], &bits, 4);
        } else {
          uint32_t sign = (raw[j] >> 15) & 1;
          uint32_t exp  = (raw[j] >> 10) & 0x1F;
          uint32_t mant =  raw[j]        & 0x3FF;
          uint32_t bits;
          if (exp == 0x1F) {
            bits = (sign << 31) | (0xFF << 23) | (mant << 13);
          } else if (exp == 0) {
            bits = sign << 31;
          } else {
            bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
          }
          std::memcpy(&kv_cpu[i][j], &bits, 4);
        }
      }
    }
    kv_ptrs[i] = kv_cpu[i].data();
  }

  // Seed the decoder engine (handles H2D and FP32->FP16 conversion internally)
  decoder->SeedKV(kv_ptrs.data(), 2 * n_layers_, seq_len);
  std::cout << "[ASRPrefill] Seeded decoder KV cache, seq_len=" << seq_len
            << std::endl;
}
