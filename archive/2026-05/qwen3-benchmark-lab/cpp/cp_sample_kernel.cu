// cp_sample_kernel.cu — Fused top-k sampling + embedding gather for CP
// Single block, 256 threads, ~8KB shared memory for vocab=2048
#include "cp_sample_kernel.h"

#include <cuda_bf16.h>
#include <curand_kernel.h>

#include <cfloat>
#include <cstdint>
#include <cstdio>

// Block size: 256 threads
static constexpr int kBlockSize = 256;
// Max vocab handled in shared memory
static constexpr int kMaxVocab = 2048;
// Top-k limit (compile-time max; runtime top_k <= this)
static constexpr int kMaxTopK = 64;

// Shared memory layout:
//   float s_logits[kMaxVocab];         // 8192 bytes
//   int   s_indices[kMaxTopK];         // 256 bytes
//   float s_topk_vals[kMaxTopK];       // 256 bytes
//   float s_probs[kMaxTopK];           // 256 bytes
//   float s_cdf[kMaxTopK];             // 256 bytes
//   int   s_sampled_code[1];           // 4 bytes
// Total: ~9220 bytes

__global__ void cpSampleAndEmbedKernel(
    const void* __restrict__ logits_ptr,
    bool logits_bf16,
    const float* __restrict__ embed_table,
    float* __restrict__ embed_out,
    int* __restrict__ code_out,
    int64_t* __restrict__ cache_pos_out,
    int vocab, int D, int layer_idx,
    int64_t next_cache_pos,
    int top_k, float temperature,
    unsigned long long seed, unsigned long long seq) {

  // Dynamic shared memory
  extern __shared__ char smem_raw[];
  float* s_logits = reinterpret_cast<float*>(smem_raw);
  int*   s_indices = reinterpret_cast<int*>(s_logits + kMaxVocab);
  float* s_topk_vals = reinterpret_cast<float*>(s_indices + kMaxTopK);
  float* s_probs = s_topk_vals + kMaxTopK;
  float* s_cdf = s_probs + kMaxTopK;
  int*   s_sampled_code = reinterpret_cast<int*>(s_cdf + kMaxTopK);

  int tid = threadIdx.x;

  // Step 1: Load logits into shared memory, converting BF16 -> FP32 if needed
  for (int i = tid; i < vocab; i += kBlockSize) {
    if (logits_bf16) {
      __nv_bfloat16 val = reinterpret_cast<const __nv_bfloat16*>(logits_ptr)[i];
      s_logits[i] = __bfloat162float(val);
    } else {
      s_logits[i] = reinterpret_cast<const float*>(logits_ptr)[i];
    }
  }
  __syncthreads();

  // Step 2: Find top-k values using iterative selection (k is small, e.g. 50)
  // Thread 0 does this serially — for vocab=2048 and k=50, this is fast enough.
  // Alternative: parallel reduction per iteration, but overhead dominates for small vocab.
  if (tid == 0) {
    // Initialize: mark nothing as selected
    // We use a simple approach: repeatedly find max and mark it
    // For vocab=2048, k=50: 50*2048 = 102K comparisons — trivial on GPU

    // Use a local "used" bitvector (2048 bits = 256 bytes = 64 ints)
    unsigned int used[kMaxVocab / 32];  // 64 ints for vocab=2048
    for (int i = 0; i < vocab / 32 + 1; i++) used[i] = 0;

    for (int k = 0; k < top_k; k++) {
      float best_val = -FLT_MAX;
      int best_idx = 0;
      for (int i = 0; i < vocab; i++) {
        if (!(used[i >> 5] & (1u << (i & 31)))) {
          if (s_logits[i] > best_val) {
            best_val = s_logits[i];
            best_idx = i;
          }
        }
      }
      s_topk_vals[k] = best_val;
      s_indices[k] = best_idx;
      used[best_idx >> 5] |= (1u << (best_idx & 31));
    }

    // Step 3: Apply temperature and softmax over top-k
    float max_val = s_topk_vals[0];  // Already sorted: first is largest
    float sum_exp = 0.0f;
    for (int k = 0; k < top_k; k++) {
      float e = expf((s_topk_vals[k] - max_val) / temperature);
      s_probs[k] = e;
      sum_exp += e;
    }
    float inv_sum = 1.0f / sum_exp;
    for (int k = 0; k < top_k; k++) {
      s_probs[k] *= inv_sum;
    }

    // Step 4: Build CDF and sample with cuRAND
    float running = 0.0f;
    for (int k = 0; k < top_k; k++) {
      running += s_probs[k];
      s_cdf[k] = running;
    }

    curandStatePhilox4_32_10_t state;
    curand_init(seed, seq, 0, &state);
    float u = curand_uniform(&state);

    int sampled = s_indices[top_k - 1];  // fallback to last
    for (int k = 0; k < top_k; k++) {
      if (u <= s_cdf[k]) {
        sampled = s_indices[k];
        break;
      }
    }
    s_sampled_code[0] = sampled;

    // Write code output
    code_out[0] = sampled;

    // Write cache position
    cache_pos_out[0] = next_cache_pos;
  }
  __syncthreads();

  // Step 5: Gather embedding — all 256 threads cooperate to copy D=1024 floats
  int sampled_code = s_sampled_code[0];
  const float* emb_src = embed_table +
      ((size_t)layer_idx * vocab + sampled_code) * D;
  for (int i = tid; i < D; i += kBlockSize) {
    embed_out[i] = emb_src[i];
  }
}

void launchCPSampleAndEmbed(
    cudaStream_t stream,
    const void* logits,
    bool logits_bf16,
    const float* embed_table,
    float* embed_out,
    int* code_out,
    int64_t* cache_pos_out,
    int vocab, int D, int layer_idx,
    int64_t next_cache_pos,
    int top_k, float temperature,
    unsigned long long seed, unsigned long long seq) {

  // Shared memory: logits + indices + topk_vals + probs + cdf + sampled_code
  size_t smem_bytes = kMaxVocab * sizeof(float)     // s_logits
                    + kMaxTopK * sizeof(int)         // s_indices
                    + kMaxTopK * sizeof(float)       // s_topk_vals
                    + kMaxTopK * sizeof(float)       // s_probs
                    + kMaxTopK * sizeof(float)       // s_cdf
                    + sizeof(int);                   // s_sampled_code

  cpSampleAndEmbedKernel<<<1, kBlockSize, smem_bytes, stream>>>(
      logits, logits_bf16, embed_table, embed_out,
      code_out, cache_pos_out,
      vocab, D, layer_idx, next_cache_pos,
      top_k, temperature, seed, seq);
}
