#ifndef CP_ENGINE_H
#define CP_ENGINE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CPEngine CPEngine;

/**
 * Initialize the code predictor engine.
 * Loads weights, creates rknn_matmul contexts, converts to native layout.
 *
 * @param weight_dir  Path to extracted weight directory (from extract_cp_weights.py)
 * @param num_npu_cores  Number of NPU cores (1 or 2 for RK3576)
 * @return Engine handle, or NULL on failure
 */
CPEngine* cp_engine_init(const char* weight_dir, int num_npu_cores);

/**
 * Run the 15-step code predictor.
 *
 * @param engine        Engine handle from cp_engine_init
 * @param last_hidden   Input hidden state [1024] float32 (from main model's last token)
 * @param primary_embed Primary embedding [1024] float32 (added to hidden at each step)
 * @param output_codes  Output: predicted codes [15] int32
 * @param output_codec_sum  Output: sum of codec embeddings [1024] float32
 * @return 0 on success, negative on error
 */
int cp_engine_run(CPEngine* engine,
                  const float* last_hidden,
                  const float* primary_embed,
                  int32_t* output_codes,
                  float* output_codec_sum);

/**
 * Destroy engine and free all resources.
 */
void cp_engine_destroy(CPEngine* engine);

#ifdef __cplusplus
}
#endif

#endif /* CP_ENGINE_H */
