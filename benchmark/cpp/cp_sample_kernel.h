// cp_sample_kernel.h — GPU-resident top-k sampling + embedding gather for CP
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// Fused kernel: top-k sample from logits, gather embedding, write cache_pos.
// Runs 1 block of 256 threads. Uses ~8KB shared memory for 2048 FP32 logits.
//
// logits:        [vocab] BF16 or FP32 on GPU (one group's logits)
// logits_bf16:   true if logits are BF16 (else FP32)
// embed_table:   [n_layers * vocab * D] FP32 on GPU
// embed_out:     [D] FP32 — write sampled embedding here (input for next step)
// code_out:      [1] int — write sampled code index
// cache_pos_out: [1] int64 — write next cache position value
// vocab:         vocabulary size (2048)
// D:             embedding dimension (1024)
// layer_idx:     which layer of embed_table to use
// next_cache_pos: value to write into cache_pos_out
// top_k:         number of top candidates (50)
// temperature:   sampling temperature (0.9)
// seed, seq:     cuRAND Philox seed and sequence offset
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
    unsigned long long seed, unsigned long long seq);
