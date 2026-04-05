// tts_trt_engine.h — TensorRT native engine wrappers for Qwen3-TTS
// Talker decode: GPU-resident double-buffered KV cache
// Code predictor: BF16 engine with dynamic context length
#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>

// TRT logger
class TRTLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override;
};

// ---------------------------------------------------------------------------
// TRT Talker Decode Engine — GPU-resident KV cache
// ---------------------------------------------------------------------------
class TRTTalkerEngine {
 public:
  TRTTalkerEngine(const std::string& engine_path, int n_layers, int hidden_dim,
                  int n_heads, int head_dim, int vocab_size, int max_seq = 200);
  ~TRTTalkerEngine();

  // One-time: copy prefill KV output to GPU buffer A
  // kv_data: flat array of all 2*n_layers KV tensors, each [1, n_heads, seq_len, head_dim]
  void SeedKV(const float* const* kv_ptrs, int n_kv, int seq_len);

  // Single decode step. Only copies emb (4KB) in, logits+hidden (16KB) out.
  // KV cache stays on GPU via pointer swap.
  void DecodeStep(const float* inputs_embeds,  // [1, 1, hidden_dim]
                  float* logits,               // [1, 1, vocab_size]
                  float* last_hidden);         // [1, 1, hidden_dim]

  void Reset() {
    seq_len_ = 0;
    parity_ = 0;
  }

 private:
  void AllocateBuffers();
  void FreeBuffers();

  TRTLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  cudaStream_t stream_ = nullptr;

  int n_layers_;
  int hidden_dim_;
  int n_heads_;
  int head_dim_;
  int vocab_size_;
  int max_seq_;

  // Double-buffered KV cache on GPU: A[2*n_layers], B[2*n_layers]
  // Index: past_key_i = 2*i, past_value_i = 2*i+1
  std::vector<void*> kv_a_;
  std::vector<void*> kv_b_;
  size_t kv_elem_bytes_ = 0;  // per-element byte size (FP16=2, FP32=4)

  // Small fixed I/O buffers
  void* d_emb_ = nullptr;
  void* d_logits_ = nullptr;
  void* d_hidden_ = nullptr;

  int seq_len_ = 0;
  int parity_ = 0;  // 0 = read A write B, 1 = read B write A

  // Pre-cached tensor names to avoid snprintf per step
  std::vector<std::string> kv_names_;       // "past_key_0", "past_value_0", ...
  std::vector<std::string> new_kv_names_;   // "new_past_key_0", ...
  bool first_step_ = true;
};

// ---------------------------------------------------------------------------
// TRT Code Predictor Engine — BF16
// ---------------------------------------------------------------------------
class TRTCPEngine {
 public:
  TRTCPEngine(const std::string& engine_path, int hidden_dim, int cp_vocab,
              int max_ctx_len = 17);
  ~TRTCPEngine();

  // Run one CP step (copies full context each time)
  void Predict(const float* context, int ctx_len, int step, float* logits_out);

  // --- GPU-resident context API ---
  // Reset context, copy initial 2 vectors (hidden + primary_emb) to GPU
  void BeginFrame(const float* hidden, const float* primary_emb);

  // Run CP step using GPU-resident context, append new embedding after
  // new_emb: [1, 1, D] float32 on CPU — appended to GPU context for next step
  // Returns sampled logits on CPU
  void PredictGPU(int step, float* logits_out);

  // Append embedding to GPU context (for next step)
  void AppendEmbedding(const float* emb);

  int ctx_len() const { return ctx_len_; }

  // GPU embedding table for cp_embed (avoid ORT calls)
  // Call LoadEmbedTable once at init, then AppendEmbeddingFromTable per step
  void LoadEmbedTable(const float* table, int n_layers, int vocab, int dim);
  void AppendEmbeddingFromTable(int layer_idx, int token_id);
  bool has_embed_table() const { return d_embed_table_ != nullptr; }

 private:
  TRTLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_trt_;
  cudaStream_t stream_ = nullptr;

  int hidden_dim_;
  int cp_vocab_;
  int ctx_len_ = 0;  // current context length for GPU-resident mode

  void* d_ctx_ = nullptr;  // [1, max_ctx, D] on GPU
  void* d_gs_ = nullptr;
  void* d_out_ = nullptr;

  // Embedding table on GPU: [n_layers, vocab, D]
  void* d_embed_table_ = nullptr;
  int embed_n_layers_ = 0;
  int embed_vocab_ = 0;
  int embed_dim_ = 0;
};
