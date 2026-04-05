/**
 * Code Predictor Engine for RK3576 NPU
 *
 * Uses rknn_matmul_api for matrix multiplications on the NPU,
 * with all scalar/attention ops on CPU (ARM NEON).
 *
 * Architecture: 5-layer Qwen3 transformer, 15 sequential decode steps,
 * each with a different lm_head and codebook.
 *
 * Modes:
 *   Default: FP16xFP16 mode (~232ms, bit-exact with PyTorch reference).
 *   -DCP_USE_W4A16: W4A16 mode (~88ms) using per-column INT4 quantization.
 *     The NPU computes A_fp16 * B_int4 -> C_fp16 (with implicit scale=1),
 *     then CPU applies per-column FP32 scales to the output.
 *     Requires pre-quantized weights from quantize_w4a16.py.
 *     Accuracy: cosine > 0.997 vs FP16 on each matmul, ~3.3x speedup.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <arm_neon.h>

#include "rknn_api.h"
#include "rknn_matmul_api.h"
#include "cp_engine.h"

/* ─── Constants ─── */
#define N_LAYERS        5
#define N_STEPS         15
#define HIDDEN_DIM      1024
#define NUM_Q_HEADS     16
#define NUM_KV_HEADS    8
#define HEAD_DIM        128
#define INTER_DIM       3072
#define VOCAB_SIZE      2048
#define MAX_SEQ_LEN     17   /* 1 base + 1 primary + 15 codes max */
#define RMS_EPS         1e-6f
#define ROPE_THETA      1000000.0f

/* Matmul indices within a layer */
enum { MM_Q=0, MM_K, MM_V, MM_O, MM_GATE, MM_UP, MM_DOWN, MM_COUNT };

/* Matmul dimensions: [K, N] for B matrix (input_dim, output_dim) */
static const int LAYER_DIMS[MM_COUNT][2] = {
    {1024, 2048},  /* q_proj:    hidden -> q */
    {1024, 1024},  /* k_proj:    hidden -> k */
    {1024, 1024},  /* v_proj:    hidden -> v */
    {2048, 1024},  /* o_proj:    attn_out -> hidden */
    {1024, 3072},  /* gate_proj: hidden -> ffn */
    {1024, 3072},  /* up_proj:   hidden -> ffn */
    {3072, 1024},  /* down_proj: ffn -> hidden */
};

static const char* PROJ_NAMES[MM_COUNT] = {
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
};

/* ─── Engine struct ─── */
struct CPEngine {
    /* NPU matmul contexts: 5 layers x 7 projections */
    rknn_matmul_ctx   layer_ctx[N_LAYERS][MM_COUNT];
    rknn_matmul_io_attr layer_io[N_LAYERS][MM_COUNT];
    rknn_tensor_mem*  layer_A[N_LAYERS][MM_COUNT];
    rknn_tensor_mem*  layer_B[N_LAYERS][MM_COUNT];
    rknn_tensor_mem*  layer_C[N_LAYERS][MM_COUNT];

    /* NPU matmul contexts: 15 lm_heads */
    rknn_matmul_ctx   lm_ctx[N_STEPS];
    rknn_matmul_io_attr lm_io[N_STEPS];
    rknn_tensor_mem*  lm_A[N_STEPS];
    rknn_tensor_mem*  lm_B[N_STEPS];
    rknn_tensor_mem*  lm_C[N_STEPS];

#if defined(CP_USE_W4A16) || defined(CP_USE_W8A16)
    /* Per-column FP32 scales for quantized weight dequantization */
    float* layer_scales[N_LAYERS][MM_COUNT];  /* each is N floats */
    float* lm_scales[N_STEPS];               /* each is VOCAB_SIZE floats */
#endif

    /* CPU weights */
    float input_norm[N_LAYERS][HIDDEN_DIM];
    float post_norm[N_LAYERS][HIDDEN_DIM];
    float q_norm[N_LAYERS][HEAD_DIM];
    float k_norm[N_LAYERS][HEAD_DIM];
    float final_norm[HIDDEN_DIM];

    /* Codec embeddings: [N_STEPS][VOCAB_SIZE][HIDDEN_DIM] */
    float* codec_embeds[N_STEPS];  /* each malloc'd as VOCAB_SIZE * HIDDEN_DIM floats */

    /* RoPE precomputed tables */
    float rope_cos[MAX_SEQ_LEN][HEAD_DIM / 2];
    float rope_sin[MAX_SEQ_LEN][HEAD_DIM / 2];

    /* Core mask for NPU */
    rknn_core_mask core_mask;
};

/* ─── File I/O helpers ─── */

static int load_bin_f32(const char* path, float* dst, int count) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); return -1; }
    size_t n = fread(dst, sizeof(float), count, f);
    fclose(f);
    if ((int)n != count) {
        fprintf(stderr, "ERROR: %s: expected %d floats, got %zu\n", path, count, n);
        return -1;
    }
    return 0;
}

static void* load_bin_raw(const char* path, size_t* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    void* buf = malloc(sz);
    if (!buf) { fclose(f); return NULL; }
    fread(buf, 1, sz, f);
    fclose(f);
    if (out_size) *out_size = (size_t)sz;
    return buf;
}

/* ─── CPU ops (NEON-optimized) ─── */

static void rms_norm(float* out, const float* x, const float* w, int dim, float eps) {
    float sum_sq = 0.0f;
    int i;
    for (i = 0; i <= dim - 4; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        sum_sq += vaddvq_f32(vmulq_f32(v, v));
    }
    for (; i < dim; i++) sum_sq += x[i] * x[i];
    float rms = 1.0f / sqrtf(sum_sq / dim + eps);
    float32x4_t scale = vdupq_n_f32(rms);
    for (i = 0; i <= dim - 4; i += 4) {
        float32x4_t xv = vld1q_f32(x + i);
        float32x4_t wv = vld1q_f32(w + i);
        vst1q_f32(out + i, vmulq_f32(vmulq_f32(xv, scale), wv));
    }
    for (; i < dim; i++) out[i] = x[i] * rms * w[i];
}

static void head_rms_norm(float* x, const float* w, int num_heads, int head_dim) {
    /* Per-head RMSNorm (Qwen3 Q/K norm), in-place */
    for (int h = 0; h < num_heads; h++) {
        float* hx = x + h * head_dim;
        float ss = 0.0f;
        for (int i = 0; i < head_dim; i += 4) {
            float32x4_t v = vld1q_f32(hx + i);
            ss += vaddvq_f32(vmulq_f32(v, v));
        }
        float s = 1.0f / sqrtf(ss / head_dim + 1e-6f);
        float32x4_t sv = vdupq_n_f32(s);
        for (int i = 0; i < head_dim; i += 4) {
            float32x4_t xv = vld1q_f32(hx + i);
            float32x4_t wv = vld1q_f32(w + i);
            vst1q_f32(hx + i, vmulq_f32(vmulq_f32(xv, sv), wv));
        }
    }
}

static void apply_rope(float* x, const float* cos_tab, const float* sin_tab,
                       int num_heads, int head_dim) {
    int half = head_dim / 2;
    for (int h = 0; h < num_heads; h++) {
        float* hx = x + h * head_dim;
        for (int i = 0; i < half; i += 4) {
            float32x4_t x0 = vld1q_f32(hx + i);
            float32x4_t x1 = vld1q_f32(hx + half + i);
            float32x4_t c  = vld1q_f32(cos_tab + i);
            float32x4_t s  = vld1q_f32(sin_tab + i);
            vst1q_f32(hx + i,        vmlsq_f32(vmulq_f32(x0, c), x1, s));
            vst1q_f32(hx + half + i, vmlaq_f32(vmulq_f32(x1, c), x0, s));
        }
    }
}

static void attention(float* out, const float* q, const float* k_cache, const float* v_cache,
                      int num_q_heads, int num_kv_heads, int head_dim, int seq_len) {
    float scale = 1.0f / sqrtf((float)head_dim);
    int groups = num_q_heads / num_kv_heads;

    for (int h = 0; h < num_q_heads; h++) {
        int kv_h = h / groups;
        const float* qh = q + h * head_dim;
        float scores[MAX_SEQ_LEN];
        float mx = -1e30f;

        for (int s = 0; s < seq_len; s++) {
            const float* ks = k_cache + kv_h * MAX_SEQ_LEN * head_dim + s * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d += 4) {
                float32x4_t qv = vld1q_f32(qh + d);
                float32x4_t kv = vld1q_f32(ks + d);
                dot += vaddvq_f32(vmulq_f32(qv, kv));
            }
            scores[s] = dot * scale;
            if (scores[s] > mx) mx = scores[s];
        }

        float sum = 0.0f;
        for (int s = 0; s < seq_len; s++) {
            scores[s] = expf(scores[s] - mx);
            sum += scores[s];
        }
        float inv_sum = 1.0f / sum;
        for (int s = 0; s < seq_len; s++) scores[s] *= inv_sum;

        float* oh = out + h * head_dim;
        memset(oh, 0, head_dim * sizeof(float));
        for (int s = 0; s < seq_len; s++) {
            const float* vs = v_cache + kv_h * MAX_SEQ_LEN * head_dim + s * head_dim;
            float sc = scores[s];
            for (int d = 0; d < head_dim; d += 4) {
                float32x4_t acc = vld1q_f32(oh + d);
                float32x4_t vv = vld1q_f32(vs + d);
                vst1q_f32(oh + d, vmlaq_n_f32(acc, vv, sc));
            }
        }
    }
}

static void silu_mul(float* out, const float* gate, const float* up, int dim) {
    for (int i = 0; i < dim; i++) {
        float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

static void vec_add(float* dst, const float* src, int n) {
    for (int i = 0; i <= n - 4; i += 4) {
        float32x4_t a = vld1q_f32(dst + i);
        float32x4_t b = vld1q_f32(src + i);
        vst1q_f32(dst + i, vaddq_f32(a, b));
    }
}

static int argmax_f32(const float* x, int n) {
    float mx = x[0];
    int mi = 0;
    for (int i = 1; i < n; i++) {
        if (x[i] > mx) { mx = x[i]; mi = i; }
    }
    return mi;
}

/* ─── FP16 <-> FP32 conversion helpers ─── */

static void fp16_to_fp32(float* dst, const uint16_t* src, int n) {
    /* Use NEON for bulk conversion */
    int i;
    for (i = 0; i <= n - 4; i += 4) {
        float16x4_t h = vld1_f16((const __fp16*)(src + i));
        float32x4_t f = vcvt_f32_f16(h);
        vst1q_f32(dst + i, f);
    }
    for (; i < n; i++) {
        __fp16 tmp;
        memcpy(&tmp, src + i, sizeof(__fp16));
        dst[i] = (float)tmp;
    }
}

static void fp32_to_fp16(uint16_t* dst, const float* src, int n) {
    int i;
    for (i = 0; i <= n - 4; i += 4) {
        float32x4_t f = vld1q_f32(src + i);
        float16x4_t h = vcvt_f16_f32(f);
        vst1_f16((__fp16*)(dst + i), h);
    }
    for (; i < n; i++) {
        __fp16 tmp = (__fp16)src[i];
        memcpy(dst + i, &tmp, sizeof(__fp16));
    }
}

/* ─── Matmul run helper: FP32 in -> NPU matmul (FP16) -> FP32 out ─── */

static void run_matmul(CPEngine* eng, rknn_matmul_ctx ctx,
                       rknn_tensor_mem* mem_a, rknn_tensor_mem* mem_c,
                       const float* input, int K, float* output, int N) {
    /* Convert FP32 input to FP16 into A buffer */
    fp32_to_fp16((uint16_t*)mem_a->virt_addr, input, K);
    /* Run NPU matmul */
    rknn_matmul_run(ctx);
    /* Convert FP16 output to FP32 */
    fp16_to_fp32(output, (const uint16_t*)mem_c->virt_addr, N);
}

#ifdef CP_USE_W4A16
/* W4A16 variant: FP16*INT4->FP16, then apply per-column scale on CPU. */
static void run_matmul_w4a16(CPEngine* eng, rknn_matmul_ctx ctx,
                             rknn_tensor_mem* mem_a, rknn_tensor_mem* mem_c,
                             const float* input, int K,
                             float* output, int N, const float* col_scales) {
    fp32_to_fp16((uint16_t*)mem_a->virt_addr, input, K);
    rknn_matmul_run(ctx);
    /* Convert FP16->FP32 and apply per-column scale in one pass (NEON) */
    const uint16_t* src = (const uint16_t*)mem_c->virt_addr;
    int i;
    for (i = 0; i <= N - 4; i += 4) {
        float16x4_t h = vld1_f16((const __fp16*)(src + i));
        float32x4_t f = vcvt_f32_f16(h);
        float32x4_t s = vld1q_f32(col_scales + i);
        vst1q_f32(output + i, vmulq_f32(f, s));
    }
    for (; i < N; i++) {
        __fp16 tmp;
        memcpy(&tmp, src + i, sizeof(__fp16));
        output[i] = (float)tmp * col_scales[i];
    }
}
#endif

#ifdef CP_USE_W8A16
/* W8A16 variant: FP16*INT8->FP32, then apply per-column scale on CPU. */
static void run_matmul_w8a16(CPEngine* eng, rknn_matmul_ctx ctx,
                             rknn_tensor_mem* mem_a, rknn_tensor_mem* mem_c,
                             const float* input, int K,
                             float* output, int N, const float* col_scales) {
    fp32_to_fp16((uint16_t*)mem_a->virt_addr, input, K);
    rknn_matmul_run(ctx);
    /* Output is FP32 directly from NPU, apply per-column scale */
    const float* src = (const float*)mem_c->virt_addr;
    int i;
    for (i = 0; i <= N - 4; i += 4) {
        float32x4_t f = vld1q_f32(src + i);
        float32x4_t s = vld1q_f32(col_scales + i);
        vst1q_f32(output + i, vmulq_f32(f, s));
    }
    for (; i < N; i++) {
        output[i] = src[i] * col_scales[i];
    }
}
#endif

/* ─── RoPE precomputation ─── */

static void precompute_rope(CPEngine* eng) {
    int half = HEAD_DIM / 2;
    for (int pos = 0; pos < MAX_SEQ_LEN; pos++) {
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(ROPE_THETA, (float)(2 * i) / HEAD_DIM);
            float angle = (float)pos * freq;
            eng->rope_cos[pos][i] = cosf(angle);
            eng->rope_sin[pos][i] = sinf(angle);
        }
    }
}

/* ─── Init ─── */

CPEngine* cp_engine_init(const char* weight_dir, int num_npu_cores) {
    CPEngine* eng = (CPEngine*)calloc(1, sizeof(CPEngine));
    if (!eng) return NULL;

    /* Note: rknn_matmul_set_core_mask is not supported on RK3576 matmul API.
     * The NPU auto-schedules across available cores. */
    eng->core_mask = RKNN_NPU_CORE_AUTO;

    char path[512];
    int ret;

#if defined(CP_USE_W8A16)
    printf("[cp_engine] Mode: W8A16 per-column INT8 (FP16xINT8->FP32, ~2x speedup)\n");
#elif defined(CP_USE_W4A16)
    printf("[cp_engine] Mode: W4A16 per-column INT4 (FP16xINT4->FP16, ~3x speedup)\n");
#else
    printf("[cp_engine] Mode: FP16xFP16 (bit-exact)\n");
#endif
    printf("[cp_engine] Weights: %s\n", weight_dir);

    /* ─── Create layer matmul contexts ─── */
    for (int l = 0; l < N_LAYERS; l++) {
        for (int m = 0; m < MM_COUNT; m++) {
            rknn_matmul_info info;
            memset(&info, 0, sizeof(info));
            info.M = 1;
            info.K = LAYER_DIMS[m][0];
            info.N = LAYER_DIMS[m][1];
#if defined(CP_USE_W8A16)
            info.type = RKNN_FLOAT16_MM_INT8_TO_FLOAT32;  /* W8A16 -> FP32 */
            info.B_layout = 1;  /* native layout (pre-converted) */
            info.B_quant_type = 0;  /* per-layer */
#elif defined(CP_USE_W4A16)
            info.type = RKNN_FLOAT16_MM_INT4_TO_FLOAT16;  /* W4A16 -> FP16 */
            info.B_layout = 1;  /* native layout (pre-converted) */
            info.B_quant_type = 0;  /* per-layer */
#else
            info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16;
            info.B_layout = 0;
#endif

            memset(&eng->layer_io[l][m], 0, sizeof(rknn_matmul_io_attr));
            ret = rknn_matmul_create(&eng->layer_ctx[l][m], &info, &eng->layer_io[l][m]);
            if (ret != 0) {
                fprintf(stderr, "[cp_engine] FATAL: layer %d %s matmul create failed: %d\n",
                        l, PROJ_NAMES[m], ret);
                goto fail;
            }

            /* Allocate I/O memory */
            eng->layer_A[l][m] = rknn_create_mem(eng->layer_ctx[l][m], eng->layer_io[l][m].A.size);
            eng->layer_B[l][m] = rknn_create_mem(eng->layer_ctx[l][m], eng->layer_io[l][m].B.size);
            eng->layer_C[l][m] = rknn_create_mem(eng->layer_ctx[l][m], eng->layer_io[l][m].C.size);

            if (!eng->layer_A[l][m] || !eng->layer_B[l][m] || !eng->layer_C[l][m]) {
                fprintf(stderr, "[cp_engine] FATAL: layer %d %s mem alloc failed\n", l, PROJ_NAMES[m]);
                goto fail;
            }

#if defined(CP_USE_W8A16)
            /* W8A16 mode: load INT8 weights + per-column scales */
            snprintf(path, sizeof(path), "%s/layer_%d/%s.int8.bin", weight_dir, l, PROJ_NAMES[m]);
            {
                size_t wsize;
                void* w_data = load_bin_raw(path, &wsize);
                if (!w_data) goto fail;
                ret = rknn_B_normal_layout_to_native_layout(w_data, eng->layer_B[l][m]->virt_addr,
                                                             info.K, info.N, &info);
                free(w_data);
                if (ret != 0) {
                    fprintf(stderr, "[cp_engine] FATAL: layer %d %s B layout conversion failed: %d\n",
                            l, PROJ_NAMES[m], ret);
                    goto fail;
                }
            }
            eng->layer_scales[l][m] = (float*)malloc(LAYER_DIMS[m][1] * sizeof(float));
            snprintf(path, sizeof(path), "%s/layer_%d/%s.scales.bin", weight_dir, l, PROJ_NAMES[m]);
            if (load_bin_f32(path, eng->layer_scales[l][m], LAYER_DIMS[m][1]) != 0) goto fail;
#elif defined(CP_USE_W4A16)
            /* W4A16 mode: load packed INT4 weights + per-column scales */
            snprintf(path, sizeof(path), "%s/layer_%d/%s.int4.bin", weight_dir, l, PROJ_NAMES[m]);
            {
                size_t wsize;
                void* w_data = load_bin_raw(path, &wsize);
                if (!w_data) goto fail;
                ret = rknn_B_normal_layout_to_native_layout(w_data, eng->layer_B[l][m]->virt_addr,
                                                             info.K, info.N, &info);
                free(w_data);
                if (ret != 0) {
                    fprintf(stderr, "[cp_engine] FATAL: layer %d %s B layout conversion failed: %d\n",
                            l, PROJ_NAMES[m], ret);
                    goto fail;
                }
            }
            eng->layer_scales[l][m] = (float*)malloc(LAYER_DIMS[m][1] * sizeof(float));
            snprintf(path, sizeof(path), "%s/layer_%d/%s.scales.bin", weight_dir, l, PROJ_NAMES[m]);
            if (load_bin_f32(path, eng->layer_scales[l][m], LAYER_DIMS[m][1]) != 0) goto fail;
#else
            /* FP16 mode: load and copy directly */
            snprintf(path, sizeof(path), "%s/layer_%d/%s.bin", weight_dir, l, PROJ_NAMES[m]);
            {
                size_t wsize;
                void* w_data = load_bin_raw(path, &wsize);
                if (!w_data) goto fail;
                memcpy(eng->layer_B[l][m]->virt_addr, w_data, wsize);
                free(w_data);
            }
#endif

            /* Bind I/O */
            rknn_matmul_set_io_mem(eng->layer_ctx[l][m], eng->layer_A[l][m], &eng->layer_io[l][m].A);
            rknn_matmul_set_io_mem(eng->layer_ctx[l][m], eng->layer_B[l][m], &eng->layer_io[l][m].B);
            rknn_matmul_set_io_mem(eng->layer_ctx[l][m], eng->layer_C[l][m], &eng->layer_io[l][m].C);
        }

        /* Load CPU norm weights */
        snprintf(path, sizeof(path), "%s/layer_%d/input_norm.bin", weight_dir, l);
        if (load_bin_f32(path, eng->input_norm[l], HIDDEN_DIM) != 0) goto fail;

        snprintf(path, sizeof(path), "%s/layer_%d/post_norm.bin", weight_dir, l);
        if (load_bin_f32(path, eng->post_norm[l], HIDDEN_DIM) != 0) goto fail;

        snprintf(path, sizeof(path), "%s/layer_%d/q_norm.bin", weight_dir, l);
        if (load_bin_f32(path, eng->q_norm[l], HEAD_DIM) != 0) goto fail;

        snprintf(path, sizeof(path), "%s/layer_%d/k_norm.bin", weight_dir, l);
        if (load_bin_f32(path, eng->k_norm[l], HEAD_DIM) != 0) goto fail;

        printf("[cp_engine] Layer %d loaded\n", l);
    }

    /* Final norm */
    snprintf(path, sizeof(path), "%s/final_norm.bin", weight_dir);
    if (load_bin_f32(path, eng->final_norm, HIDDEN_DIM) != 0) goto fail;

    /* ─── Create lm_head matmul contexts ─── */
    for (int s = 0; s < N_STEPS; s++) {
        rknn_matmul_info info;
        memset(&info, 0, sizeof(info));
        info.M = 1;
        info.K = HIDDEN_DIM;   /* 1024 */
        info.N = VOCAB_SIZE;   /* 2048 */
#if defined(CP_USE_W8A16)
        info.type = RKNN_FLOAT16_MM_INT8_TO_FLOAT32;
        info.B_layout = 1;
        info.B_quant_type = 0;
#elif defined(CP_USE_W4A16)
        info.type = RKNN_FLOAT16_MM_INT4_TO_FLOAT16;
        info.B_layout = 1;
        info.B_quant_type = 0;
#else
        info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16;
        info.B_layout = 0;
#endif

        memset(&eng->lm_io[s], 0, sizeof(rknn_matmul_io_attr));
        ret = rknn_matmul_create(&eng->lm_ctx[s], &info, &eng->lm_io[s]);
        if (ret != 0) {
            fprintf(stderr, "[cp_engine] FATAL: lm_head %d matmul create failed: %d\n", s, ret);
            goto fail;
        }

        eng->lm_A[s] = rknn_create_mem(eng->lm_ctx[s], eng->lm_io[s].A.size);
        eng->lm_B[s] = rknn_create_mem(eng->lm_ctx[s], eng->lm_io[s].B.size);
        eng->lm_C[s] = rknn_create_mem(eng->lm_ctx[s], eng->lm_io[s].C.size);

        if (!eng->lm_A[s] || !eng->lm_B[s] || !eng->lm_C[s]) {
            fprintf(stderr, "[cp_engine] FATAL: lm_head %d mem alloc failed\n", s);
            goto fail;
        }

#if defined(CP_USE_W8A16)
        snprintf(path, sizeof(path), "%s/lm_heads/lm_head_%d.int8.bin", weight_dir, s);
#elif defined(CP_USE_W4A16)
        snprintf(path, sizeof(path), "%s/lm_heads/lm_head_%d.int4.bin", weight_dir, s);
#endif
#if defined(CP_USE_W4A16) || defined(CP_USE_W8A16)
        {
            size_t wsize;
            void* w_data = load_bin_raw(path, &wsize);
            if (!w_data) goto fail;
            ret = rknn_B_normal_layout_to_native_layout(w_data, eng->lm_B[s]->virt_addr,
                                                         info.K, info.N, &info);
            free(w_data);
            if (ret != 0) {
                fprintf(stderr, "[cp_engine] FATAL: lm_head %d B layout conversion failed: %d\n", s, ret);
                goto fail;
            }
        }
        eng->lm_scales[s] = (float*)malloc(VOCAB_SIZE * sizeof(float));
        snprintf(path, sizeof(path), "%s/lm_heads/lm_head_%d.scales.bin", weight_dir, s);
        if (load_bin_f32(path, eng->lm_scales[s], VOCAB_SIZE) != 0) goto fail;
#else
        snprintf(path, sizeof(path), "%s/lm_heads/lm_head_%d.bin", weight_dir, s);
        {
            size_t wsize;
            void* w_data = load_bin_raw(path, &wsize);
            if (!w_data) goto fail;
            memcpy(eng->lm_B[s]->virt_addr, w_data, wsize);
            free(w_data);
        }
#endif

        rknn_matmul_set_io_mem(eng->lm_ctx[s], eng->lm_A[s], &eng->lm_io[s].A);
        rknn_matmul_set_io_mem(eng->lm_ctx[s], eng->lm_B[s], &eng->lm_io[s].B);
        rknn_matmul_set_io_mem(eng->lm_ctx[s], eng->lm_C[s], &eng->lm_io[s].C);

        printf("[cp_engine] lm_head %d loaded\n", s);
    }

    /* ─── Load codec embeddings ─── */
    for (int s = 0; s < N_STEPS; s++) {
        eng->codec_embeds[s] = (float*)malloc(VOCAB_SIZE * HIDDEN_DIM * sizeof(float));
        if (!eng->codec_embeds[s]) {
            fprintf(stderr, "[cp_engine] FATAL: codec embed %d malloc failed\n", s);
            goto fail;
        }
        snprintf(path, sizeof(path), "%s/codec_embeddings/codec_embed_%d.bin", weight_dir, s);
        if (load_bin_f32(path, eng->codec_embeds[s], VOCAB_SIZE * HIDDEN_DIM) != 0) goto fail;
    }

    /* ─── Precompute RoPE ─── */
    precompute_rope(eng);

    /* ─── Warmup ─── */
    printf("[cp_engine] Warming up NPU...\n");
    for (int w = 0; w < 2; w++) {
        for (int l = 0; l < N_LAYERS; l++)
            for (int m = 0; m < MM_COUNT; m++)
                rknn_matmul_run(eng->layer_ctx[l][m]);
        rknn_matmul_run(eng->lm_ctx[0]);
    }

    printf("[cp_engine] Init complete. %d layer contexts + %d lm_head contexts\n",
           N_LAYERS * MM_COUNT, N_STEPS);
    return eng;

fail:
    cp_engine_destroy(eng);
    return NULL;
}

/* ─── Run ─── */

int cp_engine_run(CPEngine* eng,
                  const float* last_hidden,
                  const float* primary_embed,
                  int32_t* output_codes,
                  float* output_codec_sum) {
    if (!eng) return -1;

    /* Working buffers on stack/heap */
    float hidden[HIDDEN_DIM];
    float normed[HIDDEN_DIM];
    float q_buf[NUM_Q_HEADS * HEAD_DIM];      /* 2048 */
    float k_buf[NUM_KV_HEADS * HEAD_DIM];     /* 1024 */
    float v_buf[NUM_KV_HEADS * HEAD_DIM];     /* 1024 */
    float attn_out[NUM_Q_HEADS * HEAD_DIM];   /* 2048 */
    float o_buf[HIDDEN_DIM];                  /* 1024 */
    float gate_buf[INTER_DIM];                /* 3072 */
    float up_buf[INTER_DIM];                  /* 3072 */
    float ffn_buf[INTER_DIM];                 /* 3072 */
    float down_buf[HIDDEN_DIM];               /* 1024 */
    float logits[VOCAB_SIZE];                 /* 2048 */

    /* KV cache: [N_LAYERS][num_kv_heads][MAX_SEQ_LEN][HEAD_DIM] */
    /* Allocate on heap to avoid stack overflow (5 * 8 * 17 * 128 * 4 = 348KB per cache) */
    static float k_cache[N_LAYERS][NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM];
    static float v_cache[N_LAYERS][NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM];

    /* Initialize output */
    memset(output_codec_sum, 0, HIDDEN_DIM * sizeof(float));

    /* Initial hidden state = last_hidden + primary_embed */
    for (int i = 0; i < HIDDEN_DIM; i++)
        hidden[i] = last_hidden[i] + primary_embed[i];

    /* === Step 0: Process the initial token through transformer, get first code === */
    /* For step 0, seq_len=1 (just this token). For step j>0, seq_len=j+1 but
       we use the KV cache approach: append new KV and attend to all. */

    for (int step = 0; step < N_STEPS; step++) {
        int seq_pos = step;  /* position in sequence */
        int seq_len = step + 1;  /* number of tokens to attend to */

        /* Run through 5 transformer layers */
        for (int l = 0; l < N_LAYERS; l++) {
            /* 1. Input RMSNorm */
            rms_norm(normed, hidden, eng->input_norm[l], HIDDEN_DIM, RMS_EPS);

            /* 2-4. Q, K, V projections via NPU */
#if defined(CP_USE_W8A16)
#define RUN_LAYER_MM(mm_idx, in, out) \
            run_matmul_w8a16(eng, eng->layer_ctx[l][mm_idx], \
                             eng->layer_A[l][mm_idx], eng->layer_C[l][mm_idx], \
                             in, LAYER_DIMS[mm_idx][0], out, LAYER_DIMS[mm_idx][1], \
                             eng->layer_scales[l][mm_idx])
#elif defined(CP_USE_W4A16)
#define RUN_LAYER_MM(mm_idx, in, out) \
            run_matmul_w4a16(eng, eng->layer_ctx[l][mm_idx], \
                             eng->layer_A[l][mm_idx], eng->layer_C[l][mm_idx], \
                             in, LAYER_DIMS[mm_idx][0], out, LAYER_DIMS[mm_idx][1], \
                             eng->layer_scales[l][mm_idx])
#else
#define RUN_LAYER_MM(mm_idx, in, out) \
            run_matmul(eng, eng->layer_ctx[l][mm_idx], \
                       eng->layer_A[l][mm_idx], eng->layer_C[l][mm_idx], \
                       in, LAYER_DIMS[mm_idx][0], out, LAYER_DIMS[mm_idx][1])
#endif
            RUN_LAYER_MM(MM_Q, normed, q_buf);
            RUN_LAYER_MM(MM_K, normed, k_buf);
            RUN_LAYER_MM(MM_V, normed, v_buf);

            /* 5-6. Q/K head norms */
            head_rms_norm(q_buf, eng->q_norm[l], NUM_Q_HEADS, HEAD_DIM);
            head_rms_norm(k_buf, eng->k_norm[l], NUM_KV_HEADS, HEAD_DIM);

            /* 7-8. Apply RoPE */
            apply_rope(q_buf, eng->rope_cos[seq_pos], eng->rope_sin[seq_pos],
                       NUM_Q_HEADS, HEAD_DIM);
            apply_rope(k_buf, eng->rope_cos[seq_pos], eng->rope_sin[seq_pos],
                       NUM_KV_HEADS, HEAD_DIM);

            /* Store K, V into cache at position seq_pos */
            for (int h = 0; h < NUM_KV_HEADS; h++) {
                memcpy(&k_cache[l][h * MAX_SEQ_LEN * HEAD_DIM + seq_pos * HEAD_DIM],
                       k_buf + h * HEAD_DIM, HEAD_DIM * sizeof(float));
                memcpy(&v_cache[l][h * MAX_SEQ_LEN * HEAD_DIM + seq_pos * HEAD_DIM],
                       v_buf + h * HEAD_DIM, HEAD_DIM * sizeof(float));
            }

            /* 9. Attention */
            attention(attn_out, q_buf, k_cache[l], v_cache[l],
                      NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, seq_len);

            /* 10. O projection via NPU */
            RUN_LAYER_MM(MM_O, attn_out, o_buf);

            /* 11. Residual add */
            vec_add(hidden, o_buf, HIDDEN_DIM);

            /* 12. Post-attention RMSNorm */
            rms_norm(normed, hidden, eng->post_norm[l], HIDDEN_DIM, RMS_EPS);

            /* 13-14. Gate and Up projections via NPU */
            RUN_LAYER_MM(MM_GATE, normed, gate_buf);
            RUN_LAYER_MM(MM_UP, normed, up_buf);

            /* 15. SiLU(gate) * up */
            silu_mul(ffn_buf, gate_buf, up_buf, INTER_DIM);

            /* 16. Down projection via NPU */
            RUN_LAYER_MM(MM_DOWN, ffn_buf, down_buf);
#undef RUN_LAYER_MM

            /* 17. Residual add */
            vec_add(hidden, down_buf, HIDDEN_DIM);
        }

        /* 18. Final norm + lm_head for this step */
        rms_norm(normed, hidden, eng->final_norm, HIDDEN_DIM, RMS_EPS);

#if defined(CP_USE_W8A16)
        run_matmul_w8a16(eng, eng->lm_ctx[step],
                         eng->lm_A[step], eng->lm_C[step],
                         normed, HIDDEN_DIM, logits, VOCAB_SIZE,
                         eng->lm_scales[step]);
#elif defined(CP_USE_W4A16)
        run_matmul_w4a16(eng, eng->lm_ctx[step],
                         eng->lm_A[step], eng->lm_C[step],
                         normed, HIDDEN_DIM, logits, VOCAB_SIZE,
                         eng->lm_scales[step]);
#else
        run_matmul(eng, eng->lm_ctx[step],
                   eng->lm_A[step], eng->lm_C[step],
                   normed, HIDDEN_DIM, logits, VOCAB_SIZE);
#endif

        /* 19. Argmax */
        int code = argmax_f32(logits, VOCAB_SIZE);
        output_codes[step] = code;

        /* 20. Lookup codec embedding and accumulate */
        const float* emb = eng->codec_embeds[step] + code * HIDDEN_DIM;
        vec_add(output_codec_sum, emb, HIDDEN_DIM);

        /* Prepare hidden for next step: add codec embedding to current hidden */
        vec_add(hidden, emb, HIDDEN_DIM);
    }

    return 0;
}

/* ─── Destroy ─── */

void cp_engine_destroy(CPEngine* eng) {
    if (!eng) return;

    for (int l = 0; l < N_LAYERS; l++) {
        for (int m = 0; m < MM_COUNT; m++) {
            if (eng->layer_C[l][m]) rknn_destroy_mem(eng->layer_ctx[l][m], eng->layer_C[l][m]);
            if (eng->layer_B[l][m]) rknn_destroy_mem(eng->layer_ctx[l][m], eng->layer_B[l][m]);
            if (eng->layer_A[l][m]) rknn_destroy_mem(eng->layer_ctx[l][m], eng->layer_A[l][m]);
            if (eng->layer_ctx[l][m]) rknn_matmul_destroy(eng->layer_ctx[l][m]);
#if defined(CP_USE_W4A16) || defined(CP_USE_W8A16)
            free(eng->layer_scales[l][m]);
#endif
        }
    }

    for (int s = 0; s < N_STEPS; s++) {
        if (eng->lm_C[s]) rknn_destroy_mem(eng->lm_ctx[s], eng->lm_C[s]);
        if (eng->lm_B[s]) rknn_destroy_mem(eng->lm_ctx[s], eng->lm_B[s]);
        if (eng->lm_A[s]) rknn_destroy_mem(eng->lm_ctx[s], eng->lm_A[s]);
        if (eng->lm_ctx[s]) rknn_matmul_destroy(eng->lm_ctx[s]);
#if defined(CP_USE_W4A16) || defined(CP_USE_W8A16)
        free(eng->lm_scales[s]);
#endif
    }

    for (int s = 0; s < N_STEPS; s++) {
        free(eng->codec_embeds[s]);
    }

    free(eng);
}
