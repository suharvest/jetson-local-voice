// tts_trt_engine.cpp — TRT native engine implementations
#include "tts_trt_engine.h"

#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
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
  context_.reset(engine_->createExecutionContext());
  CHECK_CUDA(cudaStreamCreate(&stream_));

  // Detect KV dtype from engine (TRT 10.x API uses tensor names)
  auto kv_dtype = engine_->getTensorDataType("past_key_0");
  kv_elem_bytes_ = TrtDtypeSize(kv_dtype);

  AllocateBuffers();

  // Create CUDA events for profiling (always created, zero cost when unused)
  CHECK_CUDA(cudaEventCreate(&ev_start_));
  CHECK_CUDA(cudaEventCreate(&ev_h2d_done_));
  CHECK_CUDA(cudaEventCreate(&ev_kernel_done_));
  CHECK_CUDA(cudaEventCreate(&ev_d2h_done_));

  std::cout << "  TRTTalkerEngine loaded: " << n_layers_ << " layers, KV "
            << kv_elem_bytes_ << " bytes/elem, max_seq=" << max_seq_
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
  size_t logits_elem = TrtDtypeSize(engine_->getTensorDataType("logits"));

  // Check if last_hidden output exists (TTS has it, ASR does not)
  size_t hidden_elem = 4;
  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    std::string tn = engine_->getIOTensorName(i);
    if (tn == "last_hidden") {
      hidden_elem = TrtDtypeSize(engine_->getTensorDataType("last_hidden"));
      break;
    }
  }

  // Allocate max-size buffers for both prefill (seq_len=max_seq_) and decode (1)
  CHECK_CUDA(cudaMalloc(&d_emb_, 1 * max_seq_ * hidden_dim_ * emb_elem));
  CHECK_CUDA(cudaMalloc(&d_logits_, 1 * max_seq_ * vocab_size_ * logits_elem));
  // Always allocate d_hidden_ (TTS uses it, ASR ignores but needs buffer for TRT binding)
  CHECK_CUDA(cudaMalloc(&d_hidden_, 1 * max_seq_ * hidden_dim_ * hidden_elem));
  CHECK_CUDA(cudaMalloc(&d_position_id_, max_seq_ * sizeof(int64_t)));
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
  result.logits.resize(seq_len * vocab_size_);
  result.last_hidden.resize(seq_len * hidden_dim_);

  // logits dtype may be FP16; copy as FP32-equivalent bytes.
  // The prefill ONNX outputs logits with vocab=3072, but we only need the last position.
  // Copy all positions for compatibility with caller.
  size_t logits_elem_bytes = TrtDtypeSize(
      prefill_engine_->getTensorDataType("logits"));
  if (logits_elem_bytes == sizeof(float)) {
    // FP32: direct copy
    CHECK_CUDA(cudaMemcpyAsync(result.logits.data(), d_prefill_logits_,
                               (size_t)seq_len * vocab_size_ * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_));
  } else {
    // FP16: copy raw bytes, then convert on CPU
    std::vector<uint16_t> logits_fp16(seq_len * vocab_size_);
    CHECK_CUDA(cudaMemcpyAsync(logits_fp16.data(), d_prefill_logits_,
                               (size_t)seq_len * vocab_size_ * sizeof(uint16_t),
                               cudaMemcpyDeviceToHost, stream_));
    // Will convert after sync below
    CHECK_CUDA(cudaEventRecord(ev_d2h_done_, stream_));
    CHECK_CUDA(cudaEventSynchronize(ev_d2h_done_));
    // FP16 -> FP32 conversion
    for (size_t j = 0; j < (size_t)seq_len * vocab_size_; ++j) {
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
      int last = seq_len - 1;
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

  // Debug: print logits stats for last position
  {
    int last = seq_len - 1;
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
    } else {
      // FP16: convert from FP32 CPU data to FP16 before uploading.
      // kv_ptrs[i] is float* (FP32), GPU buffer expects FP16.
      std::vector<uint16_t> fp16_buf(elem_count);
      const float* src = kv_ptrs[i];
      for (size_t j = 0; j < elem_count; ++j) {
        // IEEE 754 FP32 -> FP16 conversion
        float f = src[j];
        uint32_t fbits;
        std::memcpy(&fbits, &f, 4);
        uint32_t sign = (fbits >> 16) & 0x8000;
        int32_t exp = (int32_t)((fbits >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = (fbits & 0x7FFFFF) >> 13;
        uint16_t h;
        if (exp <= 0) {
          h = (uint16_t)sign;  // subnormal -> 0
        } else if (exp >= 31) {
          h = (uint16_t)(sign | 0x7C00);  // overflow -> inf
        } else {
          h = (uint16_t)(sign | (exp << 10) | mant);
        }
        fp16_buf[j] = h;
      }
      CHECK_CUDA(cudaMemcpyAsync(kv_a_[i], fp16_buf.data(), bytes,
                                 cudaMemcpyHostToDevice, stream_));
      // Must sync before fp16_buf goes out of scope
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

  // Check if decode engine supports dynamic inputs_embeds seq_len (batch prefill)
  bool supports_batch = false;
  {
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
  auto* ctx = context_.get();

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
  result.last_hidden.resize(seq_len * hidden_dim_);

  size_t logits_bytes = seq_len * vocab_size_ * sizeof(float);
  size_t hidden_bytes = seq_len * hidden_dim_ * sizeof(float);
  CHECK_CUDA(cudaMemcpyAsync(result.logits.data(), d_logits_, logits_bytes,
                             cudaMemcpyDeviceToHost, stream_));
  CHECK_CUDA(cudaMemcpyAsync(result.last_hidden.data(), d_hidden_, hidden_bytes,
                             cudaMemcpyDeviceToHost, stream_));
  CHECK_CUDA(cudaEventRecord(ev_d2h_done_, stream_));
  CHECK_CUDA(cudaEventSynchronize(ev_d2h_done_));

  seq_len_ = seq_len;
  parity_ = 1;

  std::cout << "  TRT Batch Prefill (decode engine): seq_len=" << seq_len
            << " logits=" << result.logits.size()
            << " hidden=" << result.last_hidden.size() << std::endl;
  return result;
}


void TRTTalkerEngine::DecodeStep(const float* inputs_embeds, float* logits,
                                 float* last_hidden) {
  auto* ctx = context_.get();
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

  // Initialize cached names on first call — auto-detect from engine
  if (first_step_) {
    kv_names_.resize(2 * n_layers_);
    new_kv_names_.resize(2 * n_layers_);
    for (int i = 0; i < n_layers_; ++i) {
      kv_names_[2 * i] = "past_key_" + std::to_string(i);
      kv_names_[2 * i + 1] = "past_value_" + std::to_string(i);
      new_kv_names_[2 * i] = "new_past_key_" + std::to_string(i);
      new_kv_names_[2 * i + 1] = "new_past_value_" + std::to_string(i);
    }
    // Auto-detect embed tensor name: "inputs_embeds" (TTS) or "input_embeds" (ASR)
    emb_name_ = "inputs_embeds";
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
      std::string tn = engine_->getIOTensorName(i);
      if (tn == "input_embeds") { emb_name_ = "input_embeds"; break; }
    }
    std::cout << "  TRT bind: emb_name=" << emb_name_ << std::endl;
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
    // "position_ids" for ASR
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
      std::string tn = engine_->getIOTensorName(i);
      if (tn == "position_ids") {
        has_position_ids_ = true;
        ctx->setInputShape("position_ids", nvinfer1::Dims2{1, 1});
        ctx->setTensorAddress("position_ids", d_position_id_);
        break;
      }
    }
    std::cout << "  TRT bind: has_position_ids=" << has_position_ids_
              << " seq_len=" << seq_len_ << std::endl;
    first_step_ = false;
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

  // Copy logits + hidden back to host
  size_t logits_bytes = 1 * 1 * vocab_size_ * sizeof(float);
  size_t hidden_bytes = 1 * 1 * hidden_dim_ * sizeof(float);
  CHECK_CUDA(cudaMemcpyAsync(logits, d_logits_, logits_bytes,
                             cudaMemcpyDeviceToHost, stream_));
  CHECK_CUDA(cudaMemcpyAsync(last_hidden, d_hidden_, hidden_bytes,
                             cudaMemcpyDeviceToHost, stream_));

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
