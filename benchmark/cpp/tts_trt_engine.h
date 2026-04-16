// tts_trt_engine.h — TensorRT native engine wrappers for Qwen3-TTS
// Talker decode: GPU-resident double-buffered KV cache
// Code predictor: BF16 engine with dynamic context length
#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <unordered_map>
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

  void Reset();

  // Profiling control
  void EnableProfiling(bool enable) { profiling_ = enable; }
  bool profiling() const { return profiling_; }
  const ProfilingStats& stats() const { return stats_; }
  void ResetStats() { stats_.Reset(); }

  // CUDA Graph cache: captures decode kernel sequence per (kv_len, parity)
  // and replays cached graphs on subsequent requests.
  // First request: ~10ms capture overhead per step (cold).
  // Subsequent requests: ~0.5ms replay per step (cache hit).
  // Cache survives Reset() — addresses are fixed GPU allocations.
  void EnableCudaGraph(bool enable) {
    if (!enable && use_cuda_graph_) FreeCudaGraphs();
    use_cuda_graph_ = enable;
  }
  bool cuda_graph_enabled() const { return use_cuda_graph_; }
  bool cuda_graph_captured() const { return !graph_cache_.empty(); }

  // Returns true when the engine has dual optimization profiles (Profile 0 =
  // batch prefill, Profile 1 = autoregressive decode). When true, a separate
  // prefill engine is unnecessary and can be skipped to save ~861 MB of VRAM.
  bool has_dual_profiles() const { return has_dual_profiles_; }

 private:
  void AllocateBuffers();
  void FreeBuffers();
  void FreeCudaGraphs();

  // Run prefill using the dedicated prefill engine (batch, no KV inputs).
  // Copies KV outputs directly into kv_a_ via D2D cudaMemcpy.
  PrefillResult RunPrefillEngine(const float* inputs_embeds, int seq_len);

  TRTLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  cudaStream_t stream_ = nullptr;

  // Dual-profile contexts: created when engine has 2 optimization profiles.
  // ctx_prefill_: Profile 0 — seq_len dynamic, past_len=0 (batch prefill)
  // ctx_decode_:  Profile 1 — seq_len=1, past_len dynamic (autoregressive)
  // When present, Prefill() batch path uses ctx_prefill_, DecodeStep() uses ctx_decode_.
  // Single-profile engines continue to use context_ for both.
  std::unique_ptr<nvinfer1::IExecutionContext> ctx_prefill_;  // Profile 0
  std::unique_ptr<nvinfer1::IExecutionContext> ctx_decode_;   // Profile 1
  bool has_dual_profiles_ = false;

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
  size_t kv_elem_bytes_ = 0;      // per-element byte size (FP16=2, BF16=2, FP32=4)
  bool kv_is_bf16_ = false;       // true if KV cache uses BF16 (vs FP16)
  size_t logits_elem_bytes_ = 4;  // logits output element size (BF16=2, FP32=4)
  size_t hidden_elem_bytes_ = 4;  // last_hidden output element size
  bool has_last_hidden_ = false;  // whether engine has last_hidden output
  bool logits_is_bf16_ = false;   // true if logits output is BF16 (vs FP16)
  bool hidden_is_bf16_ = false;   // true if last_hidden output is BF16

  // Small fixed I/O buffers
  void* d_emb_ = nullptr;
  void* d_logits_ = nullptr;
  void* d_hidden_ = nullptr;

  int seq_len_ = 0;
  int parity_ = 0;  // 0 = read A write B, 1 = read B write A

  // Pre-cached tensor names to avoid snprintf per step
  std::vector<std::string> kv_names_;       // "past_key_0", "past_value_0", ...
  std::vector<std::string> new_kv_names_;   // "new_past_key_0", ...
  bool first_step_ = true;          // first call to Prefill() batch path or DecodeStep()
  bool decode_first_step_ = true;   // first call to DecodeStep() (for dual-profile init)
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

  // CUDA Graph cache: maps (kv_len, parity) to captured graph exec.
  // First time a (kv_len, parity) combo is seen: capture + instantiate (~10ms).
  // Same combo seen again (next TTS request): instant replay (~0.5ms).
  // Max ~200 seq_lens x 2 parities = 400 cached graphs.
  // Cache survives across Reset() calls since GPU buffer addresses are fixed.
  bool use_cuda_graph_ = false;
  struct GraphCacheKey {
    int kv_len;
    int parity;
    bool operator==(const GraphCacheKey& o) const {
      return kv_len == o.kv_len && parity == o.parity;
    }
  };
  struct GraphCacheHash {
    size_t operator()(const GraphCacheKey& k) const {
      return std::hash<int>()(k.kv_len * 2 + k.parity);
    }
  };
  std::unordered_map<GraphCacheKey, cudaGraphExec_t, GraphCacheHash> graph_cache_;
  // Temporary graph handle used during capture (freed after instantiate)
  cudaGraph_t capture_graph_ = nullptr;
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

// ---------------------------------------------------------------------------
// TRT Code Predictor Engine with KV Cache — BF16 unified ONNX
// Inputs: inputs_embeds [1,seq_len,1024], cache_position [seq_len],
//         past_key_i/past_value_i [1,8,past_len,128] (i=0..4)
// Outputs: logits_all [15,2048], new_past_key_i/new_past_value_i [1,8,new_len,128]
// ---------------------------------------------------------------------------
class TRTCPKVEngine {
 public:
  // engine_path: path to cp_unified_bf16.engine
  // n_cp_layers: number of CP transformer layers (5 for cp_unified)
  // hidden_dim: embedding dimension (1024)
  // n_heads: number of KV heads (8)
  // head_dim: attention head dimension (128)
  // cp_vocab: code vocabulary size per layer (2048)
  // cp_out_groups: number of output groups (15 = num_code_groups - 1)
  // max_past: max KV cache length (default 20 = max CP steps)
  TRTCPKVEngine(const std::string& engine_path, int n_cp_layers = 5,
                int hidden_dim = 1024, int n_heads = 8, int head_dim = 128,
                int cp_vocab = 2048, int cp_out_groups = 15, int max_past = 20);
  ~TRTCPKVEngine();

  // Run CP for one codec frame (autoregressive):
  //   Step 0: prefill [hidden, primary_emb] (seq_len=2) → logits[0] → sample code[0]
  //   Step 1-14: decode [embed(code[j-1])] (seq_len=1) → logits[j] → sample code[j]
  // codes_out: [cp_out_groups] int — sampled codes for each group
  // embed_table: CPU pointer to cp_embed table [n_layers][vocab][D]
  void RunFrameAutoregressive(const float* hidden, const float* primary_emb,
                              int* codes_out,
                              const float* embed_table, int embed_vocab);

  // GPU-resident autoregressive CP: all 15 steps run async on GPU.
  // Only 1 H2D at start + 1 D2H sync at the end to read 15 sampled codes.
  // Requires embed table loaded on GPU via LoadEmbedTable().
  void RunFrameGPU(const float* hidden, const float* primary_emb,
                   int* codes_out);

  // Legacy parallel RunFrame (kept for reference, not recommended)
  void RunFrame(const float* hidden, const float* primary_emb,
                float* logits_out);

  // GPU embedding table — same API as TRTCPEngine (not used in RunFrame,
  // but kept for potential future streaming use)
  void LoadEmbedTable(const float* table, int n_layers, int vocab, int dim);
  bool has_embed_table() const { return d_embed_table_ != nullptr; }

  // CUDA Graph for CP decode: captures enqueueV3 (TRT kernels) per parity.
  // Only 2 graphs needed (fixed shapes, fixed KV addresses per parity).
  // First frame captures (with 2 warmup steps), all subsequent frames replay.
  // Sample kernel runs outside the graph (per-step params change).
  void EnableCPCudaGraph(bool enable) {
    if (!enable && use_cuda_graph_cp_) FreeCPCudaGraphs();
    use_cuda_graph_cp_ = enable;
  }
  bool cp_cuda_graph_enabled() const { return use_cuda_graph_cp_; }
  bool cp_cuda_graph_captured() const {
    for (int j = 0; j < kMaxCPSteps; ++j) {
      for (int p = 0; p < 2; ++p) {
        if (!cp_graph_captured_[j][p]) return false;
      }
    }
    return true;
  }

  // Profiling
  void EnableProfiling(bool enable) { profiling_ = enable; }
  bool profiling() const { return profiling_; }
  const ProfilingStats& stats() const { return stats_; }
  void ResetStats() { stats_.Reset(); }

 private:
  TRTLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  cudaStream_t stream_ = nullptr;

  // Dual-context optimization: separate contexts for prefill (seq_len=2) and
  // decode (seq_len=1) to eliminate shape-change overhead within RunFrameGPU.
  // Each context independently tracks its shapes, so the decode context never
  // sees a seq_len change (only past_len increments by 1 each step).
  std::unique_ptr<nvinfer1::IExecutionContext> ctx_prefill_;
  std::unique_ptr<nvinfer1::IExecutionContext> ctx_decode_;
  bool has_dual_ctx_ = false;

  // Dual-profile engine support: when engine has 2 optimization profiles,
  // ctx_prefill_ uses Profile 0 (seq_len=2, past_len=0) and
  // ctx_decode_ uses Profile 1 (seq_len=1, past_len dynamic).
  bool has_dual_profiles_ = false;

  // Single-head engine: has "gen_step" input and outputs "logits" [1, vocab]
  // instead of "logits_all" [15, vocab]. Only computes 1 lm_head per step.
  bool is_single_head_ = false;

  int n_cp_layers_;   // 5
  int hidden_dim_;    // 1024
  int n_heads_;       // 8
  int head_dim_;      // 128
  int cp_vocab_;      // 2048
  int cp_out_groups_; // 15

  // Embedding input: [1, 2, D] (seq_len=2 for frame prefill)
  void* d_embeds_ = nullptr;
  void* d_cache_pos_ = nullptr;  // [2] int64 = {0, 1}
  void* d_gen_step_ = nullptr;   // scalar int64 for single-head engine

  // Small dummy buffer for zero-size past KV inputs (TRT needs non-null ptr)
  void* d_kv_dummy_ = nullptr;  // 16 bytes — just needs to be a valid GPU address

  // KV double-buffer for autoregressive decode:
  // d_kv_a_ and d_kv_b_ are ping-pong buffers
  // Each: 2 * n_cp_layers tensors of [1, n_heads, max_past, head_dim]
  std::vector<void*> d_kv_a_;   // read buffer (past KV input)
  std::vector<void*> d_kv_b_;   // write buffer (new KV output)
  int max_past_ = 20;

  // Separate logits buffer for decode context (avoids binding conflicts)
  void* d_logits_decode_ = nullptr;

  // Legacy: KV output buffers (used by parallel RunFrame)
  std::vector<void*> d_kv_out_;  // 2 * n_cp_layers output KV tensors

  size_t kv_elem_bytes_ = 2;  // BF16

  // Output: logits_all [cp_out_groups, cp_vocab]
  void* d_logits_all_ = nullptr;
  size_t logits_elem_bytes_ = 4;
  bool logits_is_bf16_ = false;

  // Embedding table on GPU (optional, not used by RunFrame)
  void* d_embed_table_ = nullptr;
  int embed_n_layers_ = 0;
  int embed_vocab_ = 0;
  int embed_dim_ = 0;

  // GPU-resident autoregressive members (RunFrameGPU)
  int* d_codes_out_ = nullptr;             // [cp_out_groups] int on GPU
  int64_t* d_cache_pos_table_ = nullptr;   // [max_past] int64 pre-filled {0,1,2,...}
  int64_t* d_cache_pos_single_ = nullptr;  // [1] int64 written by kernel
  unsigned long long rng_counter_ = 0;     // monotonic counter for cuRAND sequence

  // Pre-cached tensor names for fast binding
  std::vector<std::string> cp_kv_names_;     // "past_key_0", "past_value_0", ...
  std::vector<std::string> cp_new_kv_names_; // "new_past_key_0", ...

  // Lightweight event (kept for potential future use; currently unused
  // with fixed-shape decode that eliminates inter-step sync).
  cudaEvent_t ev_step_sync_ = nullptr;

  // CUDA Graph for CP decode steps: 14 steps × 2 parities = 28 graphs.
  // Each graph captures enqueueV3 (TRT kernels) + GPU sample kernel for
  // a specific (step_idx, parity) combination. This eliminates ALL per-step
  // CPU sync overhead since both TRT execution AND sampling are on GPU.
  //
  // The sample kernel parameters (layer_idx, code_out offset, next_cache_pos)
  // change per step and are baked into the graph at capture time.
  // With fixed shapes, we need 14 separate graphs (vs 2 in the simpler scheme)
  // because each step index j has different sample kernel parameters.
  //
  // Capture schedule:
  //   First frame: capture all 14 graphs (j=1..14, each with unique parity)
  //   Subsequent frames: replay cached graphs
  //
  // Note: parity = (j-1) % 2, so j=1,3,5... use parity 0, j=2,4,6... use parity 1.
  // Within one frame, we see all 14 unique (j, parity) combinations.
  //
  // Graph addresses are baked at capture time — KV buffer addresses are fixed
  // per parity (a_↔b_ or b_↔a_).
  bool use_cuda_graph_cp_ = false;
  static constexpr int kMaxCPSteps = 15;  // max decode steps (15 groups - 1 prefill)
  cudaGraph_t cp_graph_[kMaxCPSteps][2] = {};  // [step_idx][parity]
  cudaGraphExec_t cp_graph_exec_[kMaxCPSteps][2] = {};
  bool cp_graph_captured_[kMaxCPSteps][2] = {};
  void FreeCPCudaGraphs();

  // Profiling
  bool profiling_ = false;
  ProfilingStats stats_;
  cudaEvent_t ev_start_ = nullptr;
  cudaEvent_t ev_kernel_done_ = nullptr;
  cudaEvent_t ev_d2h_done_ = nullptr;
};

// ---------------------------------------------------------------------------
// TRT Vocoder Engine — FP16 engine for audio_codes -> audio_values
// ---------------------------------------------------------------------------
class TRTVocoderEngine {
 public:
  // engine_path: path to vocoder_fp16.engine
  // max_frames: maximum T dimension (audio_codes [1, T, 16])
  // max_samples: maximum output samples (default 192000)
  TRTVocoderEngine(const std::string& engine_path, int max_frames = 100,
                   int max_samples = 192000);
  ~TRTVocoderEngine();

  // Run vocoder: codes [1, T, 16] int64 → audio [valid_samples] float32
  // Returns only the valid samples (trimmed using lengths output).
  std::vector<float> Run(const int64_t* codes, int n_frames, int n_groups);

 private:
  TRTLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  cudaStream_t stream_ = nullptr;

  int max_frames_;
  int max_samples_;

  void* d_codes_ = nullptr;        // [1, max_frames, 16] int64
  void* d_audio_values_ = nullptr; // [max_samples] float32
  void* d_lengths_ = nullptr;      // [1] int64

  // Detect output names (may vary)
  std::string audio_values_name_;
  std::string lengths_name_;

  cudaEvent_t ev_done_ = nullptr;
};

class TRTASRPrefillEngine {
 public:
  // engine_path: path to asr_prefill_bf16.engine
  // max_seq: max(seq_len, audio_len) for buffer allocation
  TRTASRPrefillEngine(const std::string& engine_path, int n_layers,
                      int hidden_dim, int n_heads, int head_dim,
                      int vocab_size, int max_seq = 500);
  ~TRTASRPrefillEngine();

  // Run prefill.
  // Returns logits [seq_len, vocab_size] FP32 on CPU.
  // KV outputs are stored internally in d_kv_ GPU buffers.
  struct PrefillOutput {
    std::vector<float> logits;  // [seq_len * vocab_size] FP32
    int seq_len;
  };
  PrefillOutput Run(const std::vector<int64_t>& input_ids,
                    const std::vector<int64_t>& position_ids,
                    const float* audio_features, int audio_len,
                    int64_t audio_offset);

  // Seed a TRTTalkerEngine's KV cache with the last Run() results.
  // Handles BF16->FP32 conversion and calls decoder->SeedKV().
  void SeedDecoder(TRTTalkerEngine* decoder, int seq_len);

  bool loaded() const { return engine_ != nullptr; }

 private:
  TRTLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> ctx_;
  cudaStream_t stream_ = nullptr;

  int n_layers_;
  int hidden_dim_;
  int n_heads_;
  int head_dim_;
  int vocab_size_;
  int max_seq_;

  // GPU input buffers
  void* d_input_ids_ = nullptr;       // [max_seq] int64
  void* d_position_ids_ = nullptr;    // [max_seq] int64
  void* d_audio_features_ = nullptr;  // [max_seq * hidden_dim] fp32
  void* d_audio_offset_ = nullptr;    // [1] int64

  // GPU output buffers
  void* d_logits_ = nullptr;      // [max_seq * vocab_size] (BF16 or FP32)
  std::vector<void*> d_kv_;       // 2 * n_layers KV tensors on GPU

  size_t kv_elem_bytes_ = 2;      // BF16 = 2 bytes

  cudaEvent_t ev_done_ = nullptr;
};
