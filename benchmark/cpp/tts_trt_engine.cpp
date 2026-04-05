// tts_trt_engine.cpp — TRT native engine implementations
#include "tts_trt_engine.h"

#include <cuda_runtime.h>

#include <cassert>
#include <cstring>
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
  std::cout << "  TRTTalkerEngine loaded: " << n_layers_ << " layers, KV "
            << kv_elem_bytes_ << " bytes/elem, max_seq=" << max_seq_
            << std::endl;
}

TRTTalkerEngine::~TRTTalkerEngine() {
  FreeBuffers();
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
  size_t emb_elem = TrtDtypeSize(engine_->getTensorDataType("inputs_embeds"));
  size_t logits_elem = TrtDtypeSize(engine_->getTensorDataType("logits"));
  size_t hidden_elem = TrtDtypeSize(engine_->getTensorDataType("last_hidden"));

  CHECK_CUDA(cudaMalloc(&d_emb_, 1 * 1 * hidden_dim_ * emb_elem));
  CHECK_CUDA(cudaMalloc(&d_logits_, 1 * 1 * vocab_size_ * logits_elem));
  CHECK_CUDA(cudaMalloc(&d_hidden_, 1 * 1 * hidden_dim_ * hidden_elem));
  CHECK_CUDA(cudaMalloc(&d_position_id_, sizeof(int64_t)));
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

void TRTTalkerEngine::SeedKV(const float* const* kv_ptrs, int n_kv,
                              int seq_len) {
  size_t elem_count = 1 * n_heads_ * seq_len * head_dim_;
  size_t bytes = elem_count * kv_elem_bytes_;
  for (int i = 0; i < n_kv && i < (int)kv_a_.size(); ++i) {
    if (kv_elem_bytes_ == 4) {
      // FP32: direct copy
      CHECK_CUDA(cudaMemcpyAsync(kv_a_[i], kv_ptrs[i], bytes,
                                 cudaMemcpyHostToDevice, stream_));
    } else {
      // FP16: need conversion from FP32 host data
      // For simplicity, copy as raw bytes (caller must provide correct dtype)
      CHECK_CUDA(cudaMemcpyAsync(kv_a_[i], kv_ptrs[i], bytes,
                                 cudaMemcpyHostToDevice, stream_));
    }
  }
  CHECK_CUDA(cudaStreamSynchronize(stream_));
  seq_len_ = seq_len;
  parity_ = 0;
}

void TRTTalkerEngine::DecodeStep(const float* inputs_embeds, float* logits,
                                 float* last_hidden) {
  auto* ctx = context_.get();
  auto& read = (parity_ == 0) ? kv_a_ : kv_b_;
  auto& write = (parity_ == 0) ? kv_b_ : kv_a_;

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
    std::string emb_name = "inputs_embeds";
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
      std::string tn = engine_->getIOTensorName(i);
      if (tn == "input_embeds") { emb_name = "input_embeds"; break; }
    }
    ctx->setInputShape(emb_name.c_str(), nvinfer1::Dims3{1, 1, hidden_dim_});
    ctx->setTensorAddress(emb_name.c_str(), d_emb_);
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

  // Execute
  ctx->enqueueV3(stream_);

  // Copy logits + hidden back to host
  size_t logits_bytes = 1 * 1 * vocab_size_ * sizeof(float);
  size_t hidden_bytes = 1 * 1 * hidden_dim_ * sizeof(float);
  CHECK_CUDA(cudaMemcpyAsync(logits, d_logits_, logits_bytes,
                             cudaMemcpyDeviceToHost, stream_));
  CHECK_CUDA(cudaMemcpyAsync(last_hidden, d_hidden_, hidden_bytes,
                             cudaMemcpyDeviceToHost, stream_));
  CHECK_CUDA(cudaStreamSynchronize(stream_));

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

  std::cout << "  TRTCPEngine loaded: D=" << hidden_dim << ", vocab="
            << cp_vocab << std::endl;
}

TRTCPEngine::~TRTCPEngine() {
  if (d_ctx_) cudaFree(d_ctx_);
  if (d_gs_) cudaFree(d_gs_);
  if (d_out_) cudaFree(d_out_);
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
  CHECK_CUDA(cudaStreamSynchronize(stream_));
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

  // gen_step
  int64_t gs = step;
  CHECK_CUDA(
      cudaMemcpyAsync(d_gs_, &gs, sizeof(int64_t), cudaMemcpyHostToDevice, stream_));

  // Bind — context is already on GPU
  ctx->setInputShape("context", nvinfer1::Dims3{1, ctx_len_, hidden_dim_});
  ctx->setInputShape("gen_step", nvinfer1::Dims{1, {1}});
  ctx->setTensorAddress("context", d_ctx_);
  ctx->setTensorAddress("gen_step", d_gs_);
  ctx->setTensorAddress("logits", d_out_);

  ctx->enqueueV3(stream_);

  // Only copy logits back (small: 2048*4 = 8KB)
  size_t out_bytes = 1 * 1 * cp_vocab_ * sizeof(float);
  CHECK_CUDA(cudaMemcpyAsync(logits_out, d_out_, out_bytes,
                             cudaMemcpyDeviceToHost, stream_));
  CHECK_CUDA(cudaStreamSynchronize(stream_));
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
