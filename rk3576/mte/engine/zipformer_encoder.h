/**
 * Zipformer Streaming Encoder Engine for RK3576 NPU
 *
 * Two modes:
 *   1. Non-streaming: processes all T frames at once (original).
 *   2. Streaming: processes 16-frame chunks with KV/conv cache state.
 *
 * Architecture: 5 encoder stacks x 2 layers = 10 layers total
 *   - hidden_dim=256, ffn_dim=768
 *   - Per layer: 3x FFN, 1x self-attention, 2x conv modules
 *   - Input: encoder_embed Conv2d+Linear [T,80] -> [T',256]
 *   - Output: encoder_proj [T',256] -> [T',512]
 *
 * Multi-scale downsample factors (relative to T_embed):
 *   [1, 2, 4, 8, 2]  =>  Stack 0: T, Stack 1: T/2, Stack 2: T/4, Stack 3: T/8, Stack 4: T/2
 *
 * Streaming state per stack (5 stacks, 2 layers each):
 *   cached_len:   [2] int64         — cumulative frame counter per layer
 *   cached_avg:   [2][256] float32  — running average for whiten
 *   cached_key:   [2][left_ctx][192] — attention K cache
 *   cached_val:   [2][left_ctx][96]  — attention V cache
 *   cached_val2:  [2][left_ctx][96]  — attention V2 cache
 *   cached_conv1: [2][256][30]       — depthwise conv1 history
 *   cached_conv2: [2][256][30]       — depthwise conv2 history
 */

#ifndef ZIPFORMER_ENCODER_H
#define ZIPFORMER_ENCODER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ZipformerEncoder ZipformerEncoder;
typedef struct ZipformerState ZipformerState;

/**
 * Initialize the Zipformer encoder engine.
 *
 * @param weight_dir  Path to weight directory containing stack{s}_layer{l}/ subdirs
 * @param max_T       Maximum number of input frames (for buffer allocation)
 * @return Engine handle, or NULL on failure
 */
ZipformerEncoder* zipformer_encoder_init(const char* weight_dir, int max_T);

/**
 * Run the full encoder (non-streaming).
 *
 * @param enc        Engine handle
 * @param features   Input: [T, 256] float32 (post-embed hidden states)
 * @param T          Number of input frames
 * @param feat_dim   Feature dimension (must be 256)
 * @param output     Output buffer: [T_out, 512] float32 (caller-allocated, at least T*512 floats)
 * @param out_T      Output: actual number of output frames (== (T+1)/2 due to output downsample)
 * @return 0 on success, negative on error
 */
int zipformer_encoder_run(ZipformerEncoder* enc,
                          const float* features, int T, int feat_dim,
                          float* output, int* out_T);

/* ─── Streaming API ─── */

/**
 * Create streaming state for chunk-by-chunk processing.
 * All caches are zero-initialized (first chunk).
 *
 * @param enc  Engine handle (needed for architecture params)
 * @return State handle, or NULL on failure
 */
ZipformerState* zipformer_state_create(const ZipformerEncoder* enc);

/**
 * Reset streaming state to initial (zero) values.
 * Call between utterances when reusing the state object.
 */
void zipformer_state_reset(ZipformerState* state);

/**
 * Destroy streaming state and free all resources.
 */
void zipformer_state_destroy(ZipformerState* state);

/**
 * Run one chunk through the streaming encoder.
 *
 * The caller must run encoder_embed externally to convert
 * [39, 80] fbank -> [16, 256] post-embed features.
 *
 * @param enc             Engine handle
 * @param state           Streaming state (updated in-place)
 * @param chunk_features  Input: [T_chunk, 256] float32 (post-embed, T_chunk=16)
 * @param T_chunk         Number of input frames in this chunk (typically 16)
 * @param feat_dim        Feature dimension (must be 256)
 * @param output          Output buffer: [out_T, 512] float32 (caller-allocated)
 * @param out_T           Output: actual number of output frames (== (T_chunk+1)/2 = 8)
 * @return 0 on success, negative on error
 */
int zipformer_encoder_run_chunk(ZipformerEncoder* enc,
                                ZipformerState* state,
                                const float* chunk_features, int T_chunk, int feat_dim,
                                float* output, int* out_T);

/**
 * Destroy engine and free all resources.
 */
void zipformer_encoder_destroy(ZipformerEncoder* enc);

/**
 * Set debug dump directory for intermediate tensor output.
 * Only effective if compiled with -DMTE_DEBUG_DUMP.
 * Pass NULL to disable dumping.
 */
void zipformer_encoder_set_debug_dump(const char* dir);

#ifdef __cplusplus
}
#endif

#endif /* ZIPFORMER_ENCODER_H */
