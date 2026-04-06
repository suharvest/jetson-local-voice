// tts_trt_engine.h — TensorRT native engine wrappers for Qwen3-TTS
// Talker decode: GPU-resident double-buffered KV cache
// Code predictor: BF16 engine with dynamic context length
#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>

// Per-step timing breakdown (CUDA events)
struct StepTiming {
  float h2d_ms = 0;      // Host-to-device copy
  float kernel_ms = 0;   // TRT enqueue (GPU kernel)
  float d2h_ms = 0;      // Device-to-host copy
  float total_ms = 0;    // Wall clock (for comparison)
};

// Aggregate profiling stats
struct ProfilingStats {
  int n_samples = 0;
  double sum_h2d = 0, sum_kernel = 0, sum_d2h = 0, sum_total = 0;
  double max_h2d = 0, max_kernel = 0, max_d2h = 0, max_total = 0;

  void Add(const StepTiming& t) {
    n_samples++;
    sum_h2d += t.h2d_ms;     sum_kernel += t.kernel_ms;
    sum_d2h += t.d2h_ms;     sum_total += t.total_ms;
    if (t.h2d_ms > max_h2d) max_h2d = t.h2d_ms;
    if (t.kernel_ms > max_kernel) max_kernel = t.kernel_ms;
    if (t.d2h_ms > max_d2h) max_d2h = t.d2h_ms;
    if (t.total_ms > max_total) max_total = t.total_ms;
  }

  double AvgH2D()    const { return n_samples ? sum_h2d / n_samples : 0; }
  double AvgKernel() const { return n_samples ? sum_kernel / n_samples : 0; }
  double AvgD2H()    const { return n_samples ? sum_d2h / n_samples : 0; }
  double AvgTotal()  const { return n_samples ? sum_total / n_samples : 0; }
  double AvgOverhead() const { return AvgTotal() - AvgH2D() - AvgKernel() - AvgD2H(); }

  void Reset() { *this = ProfilingStats{}; }
};

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

  // Load optional separate prefill engine (talker_prefill_fp16.engine).
  // When loaded, Prefill() uses batch execution instead of iterative decode.
  // The prefill engine takes inputs_embeds [1, T, D] and outputs
  // logits [1, T, vocab], last_hidden [1, T, D], past_key_0..27 [1, 8, T, 128].
  void LoadPrefillEngine(const std::string& prefill_engine_path);

  // One-time: copy prefill KV output to GPU buffer A
  // kv_data: flat array of all 2*n_layers KV tensors, each [1, n_heads, seq_len, head_dim]
  void SeedKV(const float* const* kv_ptrs, int n_kv, int seq_len);

  // Unified prefill: run TRT with inputs_embeds [1, S, D] and empty KV cache.
  // Returns logits [1, S, vocab_size] and last_hidden [1, S, D] on CPU.
  // KV cache is stored directly in GPU buffers (no SeedKV needed).
  struct PrefillResult {
    std::vector<float> logits;       // [1, S, vocab_size]
    std::vector<float> last_hidden;  // [1, S, D] (empty if engine has no last_hidden)
    int seq_len;
  };
  PrefillResult Prefill(const float* inputs_embeds, int seq_len);

  // Single decode step. Only copies emb (4KB) in, logits+hidden (16KB) out.
  // KV cache stays on GPU via pointer swap.
  void DecodeStep(const float* inputs_embeds,  // [1, 1, hidden_dim]
                  float* logits,               // [1, 1, vocab_size]
                  float* last_hidden);         // [1, 1, hidden_dim]

  void Reset() {
    seq_len_ = 0;
    parity_ = 0;
  }

  // Profiling control
  void EnableProfiling(bool enable) { profiling_ = enable; }
  bool profiling() const { return profiling_; }
  const ProfilingStats& stats() const { return stats_; }
  void ResetStats() { stats_.Reset(); }

 private:
  void AllocateBuffers();
  void FreeBuffers();

  // Run prefill using the dedicated prefill engine (batch, no KV inputs).
  // Copies KV outputs directly into kv_a_ via D2D cudaMemcpy.
  PrefillResult RunPrefillEngine(const float* inputs_embeds, int seq_len);

  TRTLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  cudaStream_t stream_ = nullptr;

  // Separate prefill engine (optional — loaded by LoadPrefillEngine)
  TRTLogger prefill_logger_;
  std::unique_ptr<nvinfer1::IRuntime> prefill_runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> prefill_engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> prefill_ctx_;

  // Temporary GPU buffers for prefill engine outputs (KV cache).
  // Shape: [1, n_heads, max_seq, head_dim] per tensor, 2*n_layers total.
  // After prefill, these are D2D-copied into kv_a_.
  std::vector<void*> d_prefill_kv_;   // size = 2 * n_layers
  void* d_prefill_emb_ = nullptr;     // [1, max_seq, hidden_dim]
  void* d_prefill_logits_ = nullptr;  // [1, max_seq, vocab_size]
  void* d_prefill_hidden_ = nullptr;  // [1, max_seq, hidden_dim]

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
  bool has_position_ids_ = false;
  void* d_position_id_ = nullptr;

  // CUDA event profiling
  bool profiling_ = false;
  ProfilingStats stats_;
  cudaEvent_t ev_start_ = nullptr;
  cudaEvent_t ev_h2d_done_ = nullptr;
  cudaEvent_t ev_kernel_done_ = nullptr;
  cudaEvent_t ev_d2h_done_ = nullptr;

  // Cached emb tensor name for binding (avoid re-detection)
  std::string emb_name_;
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

  // Profiling control
  void EnableProfiling(bool enable) { profiling_ = enable; }
  bool profiling() const { return profiling_; }
  const ProfilingStats& stats() const { return stats_; }
  void ResetStats() { stats_.Reset(); }

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

  // CUDA event profiling
  bool profiling_ = false;
  ProfilingStats stats_;
  cudaEvent_t ev_start_ = nullptr;
  cudaEvent_t ev_h2d_done_ = nullptr;
  cudaEvent_t ev_kernel_done_ = nullptr;
  cudaEvent_t ev_d2h_done_ = nullptr;
};
