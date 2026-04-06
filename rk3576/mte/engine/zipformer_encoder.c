/**
 * Zipformer Streaming Encoder Engine for RK3576 NPU
 *
 * Compile-time modes:
 *   Default:        W8A16 (FP16 x INT8 -> FP32) — quantized weights
 *   -DMTE_USE_FP16: W16A16 (FP16 x FP16 -> FP16) — full-precision weights
 *
 * Uses rknn_matmul_api for all linear projections,
 * with LayerNorm, SwooshR activation, depthwise conv, and attention on CPU.
 *
 * Non-streaming version: processes all T frames at once, no KV cache.
 *
 * Architecture (per layer — 0.5 scale is baked into weights at ONNX export):
 *   1. x += ff1(x)                  feed-forward module 1
 *   2. whiten + whiten_proj         BasicNorm with cumulative mean
 *   3-5. self_attn (no 1/sqrt(d))   multi-head self-attention (with pos bias)
 *   6. x += conv1(x)                conv module 1 (GLU + depthwise conv)
 *   7. x += ff2(x)                  feed-forward module 2
 *   8-9. V2 path                    reuse softmax scores with V2 values
 *  10. x += conv2(x)                conv module 2
 *  11. x += ff3(x)                  feed-forward module 3
 *  12-13. norm_final + bypass        RMS norm + learned bypass scaling
 *
 * Weight directory layout:
 *   weights/
 *     stack{s}_layer{l}/             s=0..4, l=0..1
 *       INT8 mode: ff{1,2,3}_{in,out}.int8.bin + .scales.bin
 *       FP16 mode: ff{1,2,3}_{in,out}.fp16.bin (no scales needed)
 *       Both:      ff{1,2,3}_{in,out}_bias.fp32.bin
 *       attn_{in,out,out2} — same pattern
 *       conv{1,2}_pw1.fp32.bin + _bias.fp32.bin  (pointwise conv1, [512,256])
 *       conv{1,2}_dw.fp32.bin + _bias.fp32.bin   (depthwise conv, [256,31])
 *       conv{1,2}_pw2.fp32.bin + _bias.fp32.bin  (pointwise conv2, [256,256])
 *       bypass_scale.fp32.bin
 *     encoder_proj.{int8,fp16}.bin + encoder_proj_bias.fp32.bin
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <arm_neon.h>

#include "rknn_api.h"
#include "rknn_matmul_api.h"
#include "zipformer_encoder.h"

/* ─── Debug intermediate tensor dumping ─── */
#ifdef MTE_DEBUG_DUMP
static const char* g_debug_dump_dir = NULL;

/**
 * Dump a float32 tensor to a .npy file for comparison with Python reference.
 * NumPy .npy format v1.0: 6-byte magic + 2-byte version + 2-byte header_len + header + data
 */
static void dump_npy(const char* name, const float* data, int ndim, const int* shape) {
    if (!g_debug_dump_dir) return;
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.npy", g_debug_dump_dir, name);
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[dump] failed to open %s\n", path); return; }

    /* Build header string */
    char header[256];
    int hlen = snprintf(header, sizeof(header),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (");
    for (int i = 0; i < ndim; i++) {
        hlen += snprintf(header + hlen, sizeof(header) - hlen,
            "%d%s", shape[i], (i < ndim - 1) ? ", " : "");
    }
    if (ndim == 1)
        hlen += snprintf(header + hlen, sizeof(header) - hlen, ",");
    hlen += snprintf(header + hlen, sizeof(header) - hlen, "), }");

    /* Pad header to align data to 64 bytes */
    int total_header = 10 + hlen + 1;  /* magic(6) + version(2) + header_len(2) + header + newline */
    int padding = 64 - (total_header % 64);
    if (padding == 64) padding = 0;
    for (int i = 0; i < padding; i++)
        header[hlen++] = ' ';
    header[hlen++] = '\n';
    header[hlen] = '\0';

    /* Write magic, version, header_len */
    unsigned char magic[10] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, 0, 0};
    uint16_t header_len = (uint16_t)hlen;
    magic[8] = header_len & 0xFF;
    magic[9] = (header_len >> 8) & 0xFF;
    fwrite(magic, 1, 10, f);
    fwrite(header, 1, hlen, f);

    /* Write data */
    int total = 1;
    for (int i = 0; i < ndim; i++) total *= shape[i];
    fwrite(data, sizeof(float), total, f);
    fclose(f);
}
#endif

/* ─── Constants ─── */
#define N_STACKS        5
#define N_LAYERS_PER_STACK  2
#define N_TOTAL_LAYERS  (N_STACKS * N_LAYERS_PER_STACK)  /* 10 */
#define HIDDEN_DIM      256
#define FFN_DIM         768
#define IN_PROJ_DIM     496     /* Q(192) + K(192) + V(96) + pos(16) = 496 */
#define KEY_DIM         192
#define VAL_DIM         96
#define POS_DIM         16
#define NUM_HEADS       4       /* Multi-head attention: 4 heads */
#define HEAD_DIM_K      (KEY_DIM / NUM_HEADS)   /* 48: key head dimension */
#define HEAD_DIM_V      (VAL_DIM / NUM_HEADS)   /* 24: value head dimension */
#define POS_HEAD_DIM    (POS_DIM / NUM_HEADS)   /* 4: positional head dimension */
#define POS_ENC_LEN     95      /* Positional encoding table length */
#define CONV_KERNEL     31
#define CONV_PAD        (CONV_KERNEL - 1)   /* 30, for causal conv */
#define PW1_OUT_DIM     512     /* pointwise_conv1 output (doubles, then GLU halves) */
#define ENCODER_OUT_DIM 512

/* ─── Streaming constants ─── */

/* Left context (KV cache size) per stack — from ONNX model:
 * cached_key_0: [2, 64, N, 192]
 * cached_key_1: [2, 32, N, 192]
 * cached_key_2: [2, 16, N, 192]
 * cached_key_3: [2,  8, N, 192]
 * cached_key_4: [2, 32, N, 192] */
static const int LEFT_CTX[N_STACKS] = {64, 32, 16, 8, 32};

/* Downsample factors relative to T_embed (NOT cascading).
 * Stack 0: T, Stack 1: T/2, Stack 2: T/4, Stack 3: T/8, Stack 4: T/2 */
static const int DS_FACTOR[N_STACKS] = {1, 2, 4, 8, 2};

/* Conv cache length per conv module = kernel_size - 1 = 30 */
#define CONV_CACHE_LEN  (CONV_KERNEL - 1)  /* 30 */

/* Matmul projection indices within a layer */
enum {
    MM_FF1_IN = 0, MM_FF1_OUT,
    MM_ATTN_IN, MM_ATTN_OUT, MM_ATTN_OUT2,
    MM_ATTN_WHITEN, MM_ATTN_WHITEN2, MM_ATTN_POS_BIAS,
    MM_FF2_IN, MM_FF2_OUT,
    MM_FF3_IN, MM_FF3_OUT,
    MM_PER_LAYER     /* = 12 */
};

/* Matmul dimensions [K, N] for each projection type */
static const int PROJ_DIMS[MM_PER_LAYER][2] = {
    {HIDDEN_DIM, FFN_DIM},      /* ff1_in_proj:     [256, 768]  */
    {FFN_DIM,    HIDDEN_DIM},   /* ff1_out_proj:    [768, 256]  */
    {HIDDEN_DIM, IN_PROJ_DIM},  /* attn_in_proj:    [256, 496]  */
    {VAL_DIM,    HIDDEN_DIM},   /* attn_out_proj:   [96,  256]  */
    {VAL_DIM,    HIDDEN_DIM},   /* attn_out_proj2:  [96,  256]  */
    {HIDDEN_DIM, HIDDEN_DIM},   /* attn_whiten:     [256, 256]  */
    {HIDDEN_DIM, VAL_DIM},      /* attn_whiten2:    [256, 96]   */
    {HIDDEN_DIM, POS_DIM},      /* attn_pos_bias:   [256, 16]   */
    {HIDDEN_DIM, FFN_DIM},      /* ff2_in_proj:     [256, 768]  */
    {FFN_DIM,    HIDDEN_DIM},   /* ff2_out_proj:    [768, 256]  */
    {HIDDEN_DIM, FFN_DIM},      /* ff3_in_proj:     [256, 768]  */
    {FFN_DIM,    HIDDEN_DIM},   /* ff3_out_proj:    [768, 256]  */
};

static const char* PROJ_NAMES[MM_PER_LAYER] = {
    "ff1_in",
    "ff1_out",
    "attn_in",
    "attn_out",
    "attn_out2",
    "attn_whiten",
    "attn_whiten2",
    "attn_pos_bias",
    "ff2_in",
    "ff2_out",
    "ff3_in",
    "ff3_out",
};

/* Has bias? whiten/whiten2/pos_bias are weight-only (no bias file) */
static const int PROJ_HAS_BIAS[MM_PER_LAYER] = {
    1, 1,  /* ff1_in, ff1_out */
    1, 1, 1,  /* attn_in, attn_out, attn_out2 */
    0, 0, 0,  /* attn_whiten, attn_whiten2, attn_pos_bias */
    1, 1,  /* ff2_in, ff2_out */
    1, 1,  /* ff3_in, ff3_out */
};

/* ─── Conv Module Weights ─── */
typedef struct {
    float* pw1_weight;  /* [PW1_OUT_DIM, HIDDEN_DIM] = [512, 256] */
    float* pw1_bias;    /* [PW1_OUT_DIM] = [512] */
    float* dw_weight;   /* [HIDDEN_DIM, CONV_KERNEL] = [256, 31] */
    float* dw_bias;     /* [HIDDEN_DIM] = [256] */
    float* pw2_weight;  /* [HIDDEN_DIM, HIDDEN_DIM] = [256, 256] */
    float* pw2_bias;    /* [HIDDEN_DIM] = [256] */
} ConvModule;

/* ─── Multi-scale temporal resolution per stack ─── */
/* Stack 0: T, Stack 1: T/2, Stack 2: T/4, Stack 3: T/8, Stack 4: T/16 */
/* Downsample factor from stack s to stack s+1 is 2x */
/* downsample_query[0..3] are for inter-stack transitions (before stacks 1-4) */
/* downsample_query[4] is for final output (stack0_out -> encoder_proj input) */
#define NUM_DOWNSAMPLE_QUERIES  5
#define NUM_OUT_COMBINERS       4

/* ─── Engine struct ─── */
struct ZipformerEncoder {
    int max_T;

    /* Per-layer NPU matmul contexts (10 layers x 9 projections) */
    rknn_matmul_ctx    layer_ctx[N_TOTAL_LAYERS][MM_PER_LAYER];
    rknn_matmul_io_attr layer_io[N_TOTAL_LAYERS][MM_PER_LAYER];
    rknn_tensor_mem*   layer_A[N_TOTAL_LAYERS][MM_PER_LAYER];
    rknn_tensor_mem*   layer_B[N_TOTAL_LAYERS][MM_PER_LAYER];
    rknn_tensor_mem*   layer_C[N_TOTAL_LAYERS][MM_PER_LAYER];
    float*             layer_scales[N_TOTAL_LAYERS][MM_PER_LAYER]; /* NULL in FP16 mode */
    float*             layer_bias[N_TOTAL_LAYERS][MM_PER_LAYER];

    /* Conv modules: 2 per layer = 20 total */
    ConvModule         conv[N_TOTAL_LAYERS][2];

    /* Per-layer bypass scale (scalar) */
    float              bypass_scale[N_TOTAL_LAYERS];

    /* Per-layer norm_final eps (learned constant, added to mean(x^2)) */
    float              norm_final_eps[N_TOTAL_LAYERS];

    /* Positional encoding table: [POS_ENC_LEN, HIDDEN_DIM] = [95, 256]
     * (used for non-streaming path / stack 0) */
    float*             pos_encoding_table;  /* [POS_ENC_LEN * HIDDEN_DIM] */

    /* Per-stack positional encoding tables for streaming.
     * Each stack uses a different slice of the global 9999-entry sinusoidal table.
     * Stack s: start = 5000 - T_stack[s] - LEFT_CTX[s], size = 2*T_stack[s] + LEFT_CTX[s] - 1
     * Stack 0: size=95, Stack 1: size=47, Stack 2: size=23, Stack 3: size=11, Stack 4: size=47 */
    float*             pos_enc_per_stack[N_STACKS]; /* per-stack tables */
    int                pos_enc_len_per_stack[N_STACKS]; /* sizes */

    /* Inter-stack multi-scale weights */
    float*             downsample_query[NUM_DOWNSAMPLE_QUERIES]; /* [HIDDEN_DIM] each */
    float              out_combiner_weight[NUM_OUT_COMBINERS];   /* scalar alpha */
    float              skip_weight;                               /* skip_modules_4 weight */

    /* Upsample learned offsets: [factor, HIDDEN_DIM] per upsample stage
     * upsample_offset[0] -> stack1 output upsample (factor=2)
     * upsample_offset[1] -> stack2 output upsample (factor=4)
     * upsample_offset[2] -> stack3 output upsample (factor=8)
     * upsample_offset[3] -> stack4 output upsample (factor=2) */
    float*             upsample_offset[4]; /* each is [factor * HIDDEN_DIM] */

    /* Encoder output projection: [256, 512] */
    rknn_matmul_ctx    proj_ctx;
    rknn_matmul_io_attr proj_io;
    rknn_tensor_mem*   proj_A;
    rknn_tensor_mem*   proj_B;
    rknn_tensor_mem*   proj_C;
    float*             proj_scales; /* NULL in FP16 mode */
    float*             proj_bias;

    /* Timing stats */
    double             time_npu_ms;
    double             time_cpu_ms;
};

/* ─── Streaming State ─── */

struct ZipformerState {
    /* Per-stack, per-layer (5 stacks x 2 layers):
     * Indices: [stack][layer] where stack=0..4, layer=0..1 */

    /* Cumulative frame counter (used for whiten running average) */
    int64_t cached_len[N_STACKS][N_LAYERS_PER_STACK];

    /* Running average for BasicNorm/whiten: [256] per layer */
    float cached_avg[N_STACKS][N_LAYERS_PER_STACK][HIDDEN_DIM];

    /* Attention KV cache:
     * Key:  [left_ctx][KEY_DIM]  where left_ctx = LEFT_CTX[stack]
     * Val:  [left_ctx][VAL_DIM]
     * Val2: [left_ctx][VAL_DIM]
     *
     * Allocated dynamically since left_ctx varies per stack. */
    float* cached_key[N_STACKS][N_LAYERS_PER_STACK];    /* [LEFT_CTX[s] * KEY_DIM] */
    float* cached_val[N_STACKS][N_LAYERS_PER_STACK];    /* [LEFT_CTX[s] * VAL_DIM] */
    float* cached_val2[N_STACKS][N_LAYERS_PER_STACK];   /* [LEFT_CTX[s] * VAL_DIM] */

    /* Conv module cache: [256][30] per conv module, 2 conv modules per layer
     * Layout: [HIDDEN_DIM][CONV_CACHE_LEN] = [256][30]
     * This stores the last 30 frames of GLU output for causal depthwise conv. */
    float cached_conv1[N_STACKS][N_LAYERS_PER_STACK][HIDDEN_DIM * CONV_CACHE_LEN];
    float cached_conv2[N_STACKS][N_LAYERS_PER_STACK][HIDDEN_DIM * CONV_CACHE_LEN];

    /* Out combiner state: previous stack output at each resolution.
     * Used by SimpleCombiner: forward(src_new, src_old).
     * In non-streaming first call src_old=0; in subsequent calls
     * it's the previous chunk's output at this stack resolution.
     *
     * Allocated dynamically based on T at each stack's resolution.
     * NULL means first chunk (src_old = 0). */
    float* prev_stack_out[N_STACKS];
    int    prev_stack_out_T[N_STACKS];
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

static float* load_bin_f32_alloc(const char* path, int count) {
    float* buf = (float*)malloc(count * sizeof(float));
    if (!buf) return NULL;
    if (load_bin_f32(path, buf, count) != 0) { free(buf); return NULL; }
    return buf;
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

/* ─── FP16 <-> FP32 conversion (NEON) ─── */

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

/* ─── Timing ─── */

static double time_diff_ms(struct timespec* start, struct timespec* end) {
    return (end->tv_sec - start->tv_sec) * 1000.0 +
           (end->tv_nsec - start->tv_nsec) / 1e6;
}

/* ─── CPU ops (NEON-optimized) ─── */

/**
 * SwooshR activation: x * sigmoid(x - 1)
 * This is the activation used in Zipformer's feed-forward modules.
 * Note: The ONNX streaming model does NOT include the -0.08 offset
 * that appears in some icefall source versions.
 */
static void swooshr(float* dst, const float* src, int n) {
    int i;
    for (i = 0; i <= n - 4; i += 4) {
        float32x4_t x = vld1q_f32(src + i);
        /* sigmoid(x - 1) = 1 / (1 + exp(-(x-1))) = 1 / (1 + exp(1-x)) */
        float sig0 = 1.0f / (1.0f + expf(1.0f - vgetq_lane_f32(x, 0)));
        float sig1 = 1.0f / (1.0f + expf(1.0f - vgetq_lane_f32(x, 1)));
        float sig2 = 1.0f / (1.0f + expf(1.0f - vgetq_lane_f32(x, 2)));
        float sig3 = 1.0f / (1.0f + expf(1.0f - vgetq_lane_f32(x, 3)));
        float32x4_t sig = {sig0, sig1, sig2, sig3};
        vst1q_f32(dst + i, vmulq_f32(x, sig));
    }
    for (; i < n; i++) {
        float x = src[i];
        float sig = 1.0f / (1.0f + expf(1.0f - x));
        dst[i] = x * sig;
    }
}

/**
 * SwooshL activation: x * sigmoid(x - 1)  (used in conv modules)
 * Note: In the ONNX streaming model, both FFN and conv activations
 * use the same x * sigmoid(x - 1) formula (constant = 1.0, no offset).
 */
static void swooshl(float* dst, const float* src, int n) {
    int i;
    for (i = 0; i < n; i++) {
        float x = src[i];
        float sig = 1.0f / (1.0f + expf(1.0f - x));
        dst[i] = x * sig;
    }
}

/**
 * Balancer-like scaling (Zipformer uses ScaledLinear which can have
 * parameter-specific scaling). For now we just pass through.
 */

/**
 * vec_add: dst[i] += src[i]
 */
static void vec_add(float* dst, const float* src, int n) {
    int i;
    for (i = 0; i <= n - 4; i += 4) {
        float32x4_t a = vld1q_f32(dst + i);
        float32x4_t b = vld1q_f32(src + i);
        vst1q_f32(dst + i, vaddq_f32(a, b));
    }
    for (; i < n; i++) dst[i] += src[i];
}

/**
 * vec_add_scaled: dst[i] += scale * src[i]
 */
static void vec_add_scaled(float* dst, const float* src, float scale, int n) {
    float32x4_t sv = vdupq_n_f32(scale);
    int i;
    for (i = 0; i <= n - 4; i += 4) {
        float32x4_t a = vld1q_f32(dst + i);
        float32x4_t b = vld1q_f32(src + i);
        vst1q_f32(dst + i, vmlaq_f32(a, b, sv));
    }
    for (; i < n; i++) dst[i] += scale * src[i];
}

/**
 * vec_scale: dst[i] *= scale
 */
static void vec_scale(float* dst, float scale, int n) {
    float32x4_t sv = vdupq_n_f32(scale);
    int i;
    for (i = 0; i <= n - 4; i += 4) {
        float32x4_t a = vld1q_f32(dst + i);
        vst1q_f32(dst + i, vmulq_f32(a, sv));
    }
    for (; i < n; i++) dst[i] *= scale;
}

/**
 * Softmax over a vector of length n, in-place.
 */
static void softmax(float* x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

/* ─── NPU Matmul Run ─── */

/* FP16 -> FP32 conversion (NEON) */
static void fp16_to_fp32(float* dst, const uint16_t* src, int n) {
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

#ifdef MTE_USE_FP16
/**
 * Run a FP16xFP16->FP16 matmul for M rows:
 *   input[M, K] -> FP16 -> NPU(FP16 x FP16 -> FP16) -> FP32 + bias -> output[M, N]
 *
 * No per-column scale needed in FP16 mode.
 */
static void run_matmul_fp16_rows(
    rknn_matmul_ctx ctx,
    rknn_tensor_mem* mem_a, rknn_tensor_mem* mem_c,
    const float* input, int M, int K,
    float* output, int N,
    const float* bias)
{
    for (int m = 0; m < M; m++) {
        /* Convert FP32 input row to FP16 */
        fp32_to_fp16((uint16_t*)mem_a->virt_addr, input + m * K, K);

        /* Run NPU */
        rknn_matmul_run(ctx);

        /* Output is FP16 — convert to FP32 and add bias */
        const uint16_t* src = (const uint16_t*)mem_c->virt_addr;
        float* dst = output + m * N;

        fp16_to_fp32(dst, src, N);

        if (bias) {
            int i;
            for (i = 0; i <= N - 4; i += 4) {
                float32x4_t f = vld1q_f32(dst + i);
                float32x4_t b = vld1q_f32(bias + i);
                vst1q_f32(dst + i, vaddq_f32(f, b));
            }
            for (; i < N; i++) {
                dst[i] += bias[i];
            }
        }
    }
}

#endif /* MTE_USE_FP16 */

/**
 * Unified matmul wrapper.
 * In INT8 mode: uses col_scales from enc->layer_scales; caller passes them.
 * In FP16 mode: col_scales is ignored (output is already correct).
 *
 * This function is used by all call sites (ff, attention, encoder_proj).
 */
static void run_matmul_w8a16_rows(
    rknn_matmul_ctx ctx,
    rknn_tensor_mem* mem_a, rknn_tensor_mem* mem_c,
    const float* input, int M, int K,
    float* output, int N,
    const float* col_scales, const float* bias)
{
#ifdef MTE_USE_FP16
    (void)col_scales;  /* unused in FP16 mode */
    run_matmul_fp16_rows(ctx, mem_a, mem_c, input, M, K, output, N, bias);
#else
    for (int m = 0; m < M; m++) {
        fp32_to_fp16((uint16_t*)mem_a->virt_addr, input + m * K, K);
        rknn_matmul_run(ctx);

        const float* src = (const float*)mem_c->virt_addr;
        float* dst = output + m * N;

        int i;
        for (i = 0; i <= N - 4; i += 4) {
            float32x4_t f = vld1q_f32(src + i);
            float32x4_t s = vld1q_f32(col_scales + i);
            float32x4_t r = vmulq_f32(f, s);
            if (bias) {
                float32x4_t b = vld1q_f32(bias + i);
                r = vaddq_f32(r, b);
            }
            vst1q_f32(dst + i, r);
        }
        for (; i < N; i++) {
            dst[i] = src[i] * col_scales[i] + (bias ? bias[i] : 0.0f);
        }
    }
#endif
}

/* Debug: print stats for a buffer */
static void debug_stats(const char* label, const float* buf, int n) {
    float mn = buf[0], mx = buf[0];
    double sum = 0;
    for (int i = 0; i < n; i++) {
        if (buf[i] < mn) mn = buf[i];
        if (buf[i] > mx) mx = buf[i];
        sum += buf[i];
    }
    printf("  [DEBUG] %-25s range=[%10.6f, %10.6f] mean=%10.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
           label, mn, mx, sum/n, buf[0], buf[1], buf[2], buf[3], buf[4]);
}

/* ─── Feed-Forward Module ─── */

/**
 * ff(x) = out_proj(swooshr(in_proj(x)))
 * x[T, HIDDEN_DIM] -> in_proj -> [T, FFN_DIM] -> swooshr -> out_proj -> [T, HIDDEN_DIM]
 */
static void run_ff_module(
    ZipformerEncoder* enc, int layer_idx, int ff_idx, /* ff_idx: 0,1,2 for ff1,ff2,ff3 */
    const float* x, int T, float* out)
{
    /* Map ff_idx to matmul indices */
    int mm_in, mm_out;
    switch (ff_idx) {
        case 0: mm_in = MM_FF1_IN;  mm_out = MM_FF1_OUT;  break;
        case 1: mm_in = MM_FF2_IN;  mm_out = MM_FF2_OUT;  break;
        case 2: mm_in = MM_FF3_IN;  mm_out = MM_FF3_OUT;  break;
        default: return;
    }

    /* Temp buffer for FFN hidden [T, FFN_DIM] */
    float* ffn_hidden = (float*)malloc(T * FFN_DIM * sizeof(float));

    /* in_proj: [T, 256] -> [T, 768] */
    run_matmul_w8a16_rows(
        enc->layer_ctx[layer_idx][mm_in],
        enc->layer_A[layer_idx][mm_in], enc->layer_C[layer_idx][mm_in],
        x, T, HIDDEN_DIM, ffn_hidden, FFN_DIM,
        enc->layer_scales[layer_idx][mm_in],
        enc->layer_bias[layer_idx][mm_in]);

#ifdef MTE_DEBUG_DUMP
    if (layer_idx == 0 && ff_idx == 0) {
        debug_stats("ff1_in_proj_out", ffn_hidden, T * FFN_DIM);
    }
#endif

    /* SwooshR activation in-place */
    swooshr(ffn_hidden, ffn_hidden, T * FFN_DIM);

#ifdef MTE_DEBUG_DUMP
    if (layer_idx == 0 && ff_idx == 0) {
        debug_stats("ff1_after_swooshr", ffn_hidden, T * FFN_DIM);
    }
#endif

    /* out_proj: [T, 768] -> [T, 256] */
    run_matmul_w8a16_rows(
        enc->layer_ctx[layer_idx][mm_out],
        enc->layer_A[layer_idx][mm_out], enc->layer_C[layer_idx][mm_out],
        ffn_hidden, T, FFN_DIM, out, HIDDEN_DIM,
        enc->layer_scales[layer_idx][mm_out],
        enc->layer_bias[layer_idx][mm_out]);

    free(ffn_hidden);
}

/* ─── Self-Attention Module (Multi-Head with Positional Bias) ─── */

/**
 * Non-streaming multi-head self-attention with positional bias.
 *
 * in_proj output [T, 496] splits into:
 *   Q[T, 192], K[T, 192], V[T, 96], pos[T, 16]
 *
 * Multi-head: 4 heads, HEAD_DIM_K=48, HEAD_DIM_V=24, POS_HEAD_DIM=4
 *
 * Positional bias:
 *   pos_enc [95, 256] @ W_pos [256, 16] -> [95, 16] -> reshape [95, 4, 4]
 *   pos_query [T, 4, 4] @ pos_enc_proj [95, 4, 4]^T -> [4, T, 95]
 *   Gather relative positions -> [4, T, T] pos_bias
 *   scores = Q@K^T + pos_bias -> softmax
 *
 * Returns: out_proj(attn_out) in out[T, 256]
 * Also stores softmax scores in *scores_out for V2 reuse (caller must free).
 * scores_out: [NUM_HEADS * T_q * T_kv] where T_kv = T for non-streaming
 */
static void run_self_attention(
    ZipformerEncoder* enc, int layer_idx,
    const float* x, int T, float* out,
    float** scores_out, int* out_T_kv)
{
    /* in_proj: [T, 256] -> [T, 496] */
    float* proj = (float*)malloc(T * IN_PROJ_DIM * sizeof(float));
    run_matmul_w8a16_rows(
        enc->layer_ctx[layer_idx][MM_ATTN_IN],
        enc->layer_A[layer_idx][MM_ATTN_IN], enc->layer_C[layer_idx][MM_ATTN_IN],
        x, T, HIDDEN_DIM, proj, IN_PROJ_DIM,
        enc->layer_scales[layer_idx][MM_ATTN_IN],
        enc->layer_bias[layer_idx][MM_ATTN_IN]);

    /* Split: Q[T, 192], K[T, 192], V[T, 96], pos_q[T, 16] */
    float* Q   = (float*)malloc(T * KEY_DIM * sizeof(float));
    float* K_  = (float*)malloc(T * KEY_DIM * sizeof(float));
    float* V   = (float*)malloc(T * VAL_DIM * sizeof(float));
    float* pos_q = (float*)malloc(T * POS_DIM * sizeof(float));

    for (int t = 0; t < T; t++) {
        const float* row = proj + t * IN_PROJ_DIM;
        memcpy(Q    + t * KEY_DIM, row,                                KEY_DIM * sizeof(float));
        memcpy(K_   + t * KEY_DIM, row + KEY_DIM,                     KEY_DIM * sizeof(float));
        memcpy(V    + t * VAL_DIM, row + 2 * KEY_DIM,                 VAL_DIM * sizeof(float));
        memcpy(pos_q + t * POS_DIM, row + 2 * KEY_DIM + VAL_DIM,     POS_DIM * sizeof(float));
    }
    free(proj);

    int T_kv = T;  /* non-streaming: no cache */
    *out_T_kv = T_kv;

    /* ── Positional bias computation ── */
    /* pos_enc [POS_ENC_LEN, 256] @ W_pos [256, 16] -> [POS_ENC_LEN, 16] */
    float* pos_proj = (float*)malloc(POS_ENC_LEN * POS_DIM * sizeof(float));
    run_matmul_w8a16_rows(
        enc->layer_ctx[layer_idx][MM_ATTN_POS_BIAS],
        enc->layer_A[layer_idx][MM_ATTN_POS_BIAS],
        enc->layer_C[layer_idx][MM_ATTN_POS_BIAS],
        enc->pos_encoding_table, POS_ENC_LEN, HIDDEN_DIM,
        pos_proj, POS_DIM,
        enc->layer_scales[layer_idx][MM_ATTN_POS_BIAS],
        enc->layer_bias[layer_idx][MM_ATTN_POS_BIAS]);

    /* Reshape pos_proj for multi-head: [S, 16] -> [S, 4, 4] -> transpose [4, 4, S]
     * pos_q for multi-head: [T, 16] -> [T, 4, 4] -> transpose [4, T, 4]
     * pos_q_mh @ pos_proj_mh^T -> [4, T, S]
     * Then gather relative position indices */
    int S = POS_ENC_LEN;

    /* Compute raw positional scores: [4, T, S]
     * pos_q_mh[h, t, d] = pos_q[t, h*4+d] (d=0..3)
     * pos_proj_mh[h, d, s] = pos_proj[s, h*4+d]
     * raw_pos[h, t, s] = sum_d pos_q_mh[h,t,d] * pos_proj_mh[h,d,s] */
    float* raw_pos = (float*)calloc(NUM_HEADS * T * S, sizeof(float));
    for (int h = 0; h < NUM_HEADS; h++) {
        for (int t = 0; t < T; t++) {
            for (int s = 0; s < S; s++) {
                float dot = 0.0f;
                for (int d = 0; d < POS_HEAD_DIM; d++) {
                    dot += pos_q[t * POS_DIM + h * POS_HEAD_DIM + d] *
                           pos_proj[s * POS_DIM + h * POS_HEAD_DIM + d];
                }
                raw_pos[h * T * S + t * S + s] = dot;
            }
        }
    }
    free(pos_proj);
    free(pos_q);

    /* Gather relative position indices:
     * For each (h, t_q, t_kv): index = (T-1-t_q) + t_kv
     * pos_bias[h, t_q, t_kv] = raw_pos[h, t_q, index] */
    float* pos_bias = (float*)malloc(NUM_HEADS * T * T_kv * sizeof(float));
    for (int h = 0; h < NUM_HEADS; h++) {
        for (int tq = 0; tq < T; tq++) {
            for (int tkv = 0; tkv < T_kv; tkv++) {
                int idx = (T - 1 - tq) + tkv;
                if (idx >= 0 && idx < S) {
                    pos_bias[h * T * T_kv + tq * T_kv + tkv] =
                        raw_pos[h * T * S + tq * S + idx];
                } else {
                    pos_bias[h * T * T_kv + tq * T_kv + tkv] = 0.0f;
                }
            }
        }
    }
    free(raw_pos);

    /* ── Multi-head Q @ K^T + pos_bias -> softmax ── */
    /* NOTE: No 1/sqrt(d_k) scaling — Zipformer bakes it into weights at export */
    float* scores = (float*)malloc(NUM_HEADS * T * T_kv * sizeof(float));

    for (int h = 0; h < NUM_HEADS; h++) {
        for (int tq = 0; tq < T; tq++) {
            for (int tkv = 0; tkv < T_kv; tkv++) {
                float dot = 0.0f;
                /* Q_mh[h, tq, d] = Q[tq, h*HEAD_DIM_K + d] */
                const float* q_row = Q + tq * KEY_DIM + h * HEAD_DIM_K;
                const float* k_row = K_ + tkv * KEY_DIM + h * HEAD_DIM_K;
                for (int d = 0; d < HEAD_DIM_K; d++) {
                    dot += q_row[d] * k_row[d];
                }
                scores[h * T * T_kv + tq * T_kv + tkv] =
                    dot + pos_bias[h * T * T_kv + tq * T_kv + tkv];
            }
            /* Softmax over tkv dimension */
            softmax(scores + h * T * T_kv + tq * T_kv, T_kv);
        }
    }
    free(pos_bias);
    free(Q);
    free(K_);

    /* ── Weighted sum: scores @ V -> attn_out ── */
    /* V_mh[h, tkv, d] = V[tkv, h*HEAD_DIM_V + d]
     * attn_out_mh[h, tq, d] = sum_tkv scores[h,tq,tkv] * V_mh[h,tkv,d]
     * attn_out[tq, h*HEAD_DIM_V + d] = attn_out_mh[h, tq, d] */
    float* attn_out = (float*)calloc(T * VAL_DIM, sizeof(float));
    for (int h = 0; h < NUM_HEADS; h++) {
        for (int tq = 0; tq < T; tq++) {
            float* ao = attn_out + tq * VAL_DIM + h * HEAD_DIM_V;
            for (int tkv = 0; tkv < T_kv; tkv++) {
                float sc = scores[h * T * T_kv + tq * T_kv + tkv];
                const float* v_row = V + tkv * VAL_DIM + h * HEAD_DIM_V;
                for (int d = 0; d < HEAD_DIM_V; d++) {
                    ao[d] += sc * v_row[d];
                }
            }
        }
    }
    free(V);

    /* out_proj: [T, 96] -> [T, 256] */
    run_matmul_w8a16_rows(
        enc->layer_ctx[layer_idx][MM_ATTN_OUT],
        enc->layer_A[layer_idx][MM_ATTN_OUT], enc->layer_C[layer_idx][MM_ATTN_OUT],
        attn_out, T, VAL_DIM, out, HIDDEN_DIM,
        enc->layer_scales[layer_idx][MM_ATTN_OUT],
        enc->layer_bias[layer_idx][MM_ATTN_OUT]);

    free(attn_out);

    /* Return softmax scores for V2 reuse */
    *scores_out = scores;
}

/* ─── Conv Module ─── */

/**
 * Conv module (Zipformer-style):
 *   1. pointwise_conv1: [T, 256] -> [T, 512] (with bias)
 *   2. GLU: split [T, 512] -> [T, 256] * sigmoid([T, 256])
 *   3. depthwise_conv1d: [T, 256] -> [T, 256] kernel=31 (with bias)
 *      Non-streaming: standard (non-causal) 1D conv with zero padding
 *   4. SwooshL activation
 *   5. pointwise_conv2: [T, 256] -> [T, 256] (with bias)
 */
static void run_conv_module(
    const ConvModule* cm, const float* x, int T, float* out)
{
    /* 1. Pointwise conv1: matmul [T, 256] x [256, 512] + bias */
    float* pw1_out = (float*)malloc(T * PW1_OUT_DIM * sizeof(float));
    for (int t = 0; t < T; t++) {
        const float* xt = x + t * HIDDEN_DIM;
        float* ot = pw1_out + t * PW1_OUT_DIM;
        /* Conv1d(in=256, out=512, kernel=1) is equivalent to a linear layer.
         * Weight shape in ONNX: [out_channels, in_channels, 1] = [512, 256, 1]
         * Stored as [512, 256] (kernel size 1 is implicit). */
        for (int o = 0; o < PW1_OUT_DIM; o++) {
            float sum = cm->pw1_bias[o];
            const float* w = cm->pw1_weight + o * HIDDEN_DIM;
            int d;
            for (d = 0; d <= HIDDEN_DIM - 4; d += 4) {
                float32x4_t xv = vld1q_f32(xt + d);
                float32x4_t wv = vld1q_f32(w + d);
                sum += vaddvq_f32(vmulq_f32(xv, wv));
            }
            for (; d < HIDDEN_DIM; d++) sum += xt[d] * w[d];
            ot[o] = sum;
        }
    }

    /* 2. GLU: first 256 * sigmoid(second 256) */
    float* glu_out = (float*)malloc(T * HIDDEN_DIM * sizeof(float));
    for (int t = 0; t < T; t++) {
        const float* a = pw1_out + t * PW1_OUT_DIM;
        const float* b = a + HIDDEN_DIM;
        float* g = glu_out + t * HIDDEN_DIM;
        for (int d = 0; d < HIDDEN_DIM; d++) {
            g[d] = a[d] * (1.0f / (1.0f + expf(-b[d])));
        }
    }
    free(pw1_out);

    /* 3. Depthwise conv1d: kernel=31, groups=256
     * Non-streaming: symmetric padding = (31-1)/2 = 15 on each side.
     * Weight: [channels=256, 1, kernel=31] stored as [256, 31]. */
    int pad = CONV_KERNEL / 2;  /* 15 for symmetric padding */
    float* dw_out = (float*)malloc(T * HIDDEN_DIM * sizeof(float));
    for (int c = 0; c < HIDDEN_DIM; c++) {
        const float* w = cm->dw_weight + c * CONV_KERNEL;
        float bias_val = cm->dw_bias[c];
        for (int t = 0; t < T; t++) {
            float sum = bias_val;
            for (int k = 0; k < CONV_KERNEL; k++) {
                int t_in = t + k - pad;
                if (t_in >= 0 && t_in < T) {
                    sum += glu_out[t_in * HIDDEN_DIM + c] * w[k];
                }
            }
            dw_out[t * HIDDEN_DIM + c] = sum;
        }
    }
    free(glu_out);

    /* 4. SwooshL activation */
    swooshl(dw_out, dw_out, T * HIDDEN_DIM);

    /* 5. Pointwise conv2: [T, 256] -> [T, 256]
     * Weight: [out=256, in=256, 1] = [256, 256] */
    for (int t = 0; t < T; t++) {
        const float* xt = dw_out + t * HIDDEN_DIM;
        float* ot = out + t * HIDDEN_DIM;
        for (int o = 0; o < HIDDEN_DIM; o++) {
            float sum = cm->pw2_bias[o];
            const float* w = cm->pw2_weight + o * HIDDEN_DIM;
            int d;
            for (d = 0; d <= HIDDEN_DIM - 4; d += 4) {
                float32x4_t xv = vld1q_f32(xt + d);
                float32x4_t wv = vld1q_f32(w + d);
                sum += vaddvq_f32(vmulq_f32(xv, wv));
            }
            for (; d < HIDDEN_DIM; d++) sum += xt[d] * w[d];
            ot[o] = sum;
        }
    }
    free(dw_out);
}

/* ─── Multi-scale operations ─── */

/**
 * Downsample (icefall SimpleDownsample) — Softmax attention pooling:
 *   For each group of `factor` adjacent frames:
 *     1. Compute attention score per frame: score[i] = dot(input[t*factor+i], query)
 *     2. Apply softmax over the group scores -> weights[i]
 *     3. Output = weighted sum: sum_i(weights[i] * input[t*factor+i])
 *
 * ONNX-confirmed algorithm (intermediates verified against streaming ONNX reference):
 *   - Reshape input (T, D) -> (new_T, factor, D) grouping adjacent frames
 *   - scores = matmul(groups, query) -> (new_T, factor, 1)
 *   - weights = softmax(scores, axis=1) -> (new_T, factor, 1)
 *   - output = sum(weights * groups, axis=1) -> (new_T, D)
 *
 * @param factor  Downsample factor (2, 4, or 8). Corresponds to DS_FACTOR[s].
 * @param new_T   Expected output length = T / factor (caller pre-computes).
 *                If T is not divisible by factor, last group uses available frames.
 *
 * new_T = (T + factor - 1) / factor  (ceiling division)
 *
 * Partial-group handling: if T % factor != 0, the last group has fewer frames;
 * softmax is applied only over those frames.
 */
static void downsample(const float* input, int T, int D,
                       const float* query, float* output, int factor) {
    /* Max factor supported = 8 (stack3 uses DS_FACTOR=8) */
    float scores[8];
    float weights[8];
    int new_T = (T + factor - 1) / factor;

    for (int t = 0; t < new_T; t++) {
        int base = t * factor;
        /* Number of actual frames in this group */
        int n_frames = (base + factor <= T) ? factor : (T - base);

        /* Step 1: compute dot-product scores score[i] = dot(input[base+i], query) */
        for (int i = 0; i < n_frames; i++) {
            const float* frame = input + (base + i) * D;
            float s = 0.0f;
            int d;
            for (d = 0; d <= D - 4; d += 4) {
                s += frame[d]   * query[d];
                s += frame[d+1] * query[d+1];
                s += frame[d+2] * query[d+2];
                s += frame[d+3] * query[d+3];
            }
            for (; d < D; d++) s += frame[d] * query[d];
            scores[i] = s;
        }

        /* Step 2: softmax over n_frames scores (numerically stable) */
        if (n_frames == 1) {
            weights[0] = 1.0f;
        } else {
            float max_s = scores[0];
            for (int i = 1; i < n_frames; i++)
                if (scores[i] > max_s) max_s = scores[i];
            float sum_e = 0.0f;
            for (int i = 0; i < n_frames; i++) {
                weights[i] = expf(scores[i] - max_s);
                sum_e += weights[i];
            }
            float inv_sum = 1.0f / sum_e;
            for (int i = 0; i < n_frames; i++) weights[i] *= inv_sum;
        }

        /* Step 3: weighted sum -> output[t] */
        float* out = output + t * D;
        memset(out, 0, D * sizeof(float));
        for (int i = 0; i < n_frames; i++) {
            const float* frame = input + (base + i) * D;
            float w = weights[i];
            int d;
            for (d = 0; d <= D - 4; d += 4) {
                out[d]   += w * frame[d];
                out[d+1] += w * frame[d+1];
                out[d+2] += w * frame[d+2];
                out[d+3] += w * frame[d+3];
            }
            for (; d < D; d++) out[d] += w * frame[d];
        }
    }
}

/**
 * Upsample: nearest-neighbor, doubles T by duplicating each frame.
 * output_T = T * 2
 */
static void upsample(const float* input, int T, int D, float* output) {
    for (int t = 0; t < T; t++) {
        memcpy(output + (t * 2) * D, input + t * D, D * sizeof(float));
        memcpy(output + (t * 2 + 1) * D, input + t * D, D * sizeof(float));
    }
}

/**
 * Out combiner: blends current stack output with downsampled previous output.
 * output[i] = alpha * output[i] + (1 - alpha) * downsampled[i]
 */
static void out_combiner(float* output, const float* downsampled,
                         float alpha, int T, int D) {
    float beta = 1.0f - alpha;
    for (int i = 0; i < T * D; i++) {
        output[i] = alpha * output[i] + beta * downsampled[i];
    }
}

/* ─── Init ─── */

static int load_conv_module(ConvModule* cm, const char* weight_dir,
                            int stack, int layer, int conv_idx) {
    char path[512];
    /* File naming: conv1_pw1.fp32.bin, conv1_dw.fp32.bin, conv2_pw2_bias.fp32.bin, etc. */
    const char* cn = (conv_idx == 0) ? "conv1" : "conv2";
    char dir[512];
    snprintf(dir, sizeof(dir), "%s/stack%d_layer%d", weight_dir, stack, layer);

    /* pointwise_conv1: [512, 256, 1] -> stored as [512, 256] */
    snprintf(path, sizeof(path), "%s/%s_pw1.fp32.bin", dir, cn);
    cm->pw1_weight = load_bin_f32_alloc(path, PW1_OUT_DIM * HIDDEN_DIM);
    if (!cm->pw1_weight) return -1;

    snprintf(path, sizeof(path), "%s/%s_pw1_bias.fp32.bin", dir, cn);
    cm->pw1_bias = load_bin_f32_alloc(path, PW1_OUT_DIM);
    if (!cm->pw1_bias) return -1;

    /* depthwise_conv: [256, 1, 31] -> stored as [256, 31] */
    snprintf(path, sizeof(path), "%s/%s_dw.fp32.bin", dir, cn);
    cm->dw_weight = load_bin_f32_alloc(path, HIDDEN_DIM * CONV_KERNEL);
    if (!cm->dw_weight) return -1;

    snprintf(path, sizeof(path), "%s/%s_dw_bias.fp32.bin", dir, cn);
    cm->dw_bias = load_bin_f32_alloc(path, HIDDEN_DIM);
    if (!cm->dw_bias) return -1;

    /* pointwise_conv2: [256, 256, 1] -> stored as [256, 256] */
    snprintf(path, sizeof(path), "%s/%s_pw2.fp32.bin", dir, cn);
    cm->pw2_weight = load_bin_f32_alloc(path, HIDDEN_DIM * HIDDEN_DIM);
    if (!cm->pw2_weight) return -1;

    snprintf(path, sizeof(path), "%s/%s_pw2_bias.fp32.bin", dir, cn);
    cm->pw2_bias = load_bin_f32_alloc(path, HIDDEN_DIM);
    if (!cm->pw2_bias) return -1;

    return 0;
}

static void free_conv_module(ConvModule* cm) {
    free(cm->pw1_weight); cm->pw1_weight = NULL;
    free(cm->pw1_bias);   cm->pw1_bias = NULL;
    free(cm->dw_weight);  cm->dw_weight = NULL;
    free(cm->dw_bias);    cm->dw_bias = NULL;
    free(cm->pw2_weight); cm->pw2_weight = NULL;
    free(cm->pw2_bias);   cm->pw2_bias = NULL;
}

ZipformerEncoder* zipformer_encoder_init(const char* weight_dir, int max_T) {
    ZipformerEncoder* enc = (ZipformerEncoder*)calloc(1, sizeof(ZipformerEncoder));
    if (!enc) return NULL;
    enc->max_T = max_T;

    char path[512];
    int ret;

#ifdef MTE_USE_FP16
    printf("[zipformer] Mode: FP16xFP16->FP16 (full-precision weights)\n");
#else
    printf("[zipformer] Mode: W8A16 (FP16xINT8->FP32)\n");
#endif
    printf("[zipformer] Weights: %s\n", weight_dir);
    printf("[zipformer] max_T: %d\n", max_T);

    /* ─── Create per-layer matmul contexts ─── */
    for (int s = 0; s < N_STACKS; s++) {
        for (int l = 0; l < N_LAYERS_PER_STACK; l++) {
            int li = s * N_LAYERS_PER_STACK + l;  /* flat layer index 0..9 */
            char layer_dir[512];
            snprintf(layer_dir, sizeof(layer_dir), "%s/stack%d_layer%d", weight_dir, s, l);

            for (int m = 0; m < MM_PER_LAYER; m++) {
                int K = PROJ_DIMS[m][0];
                int N = PROJ_DIMS[m][1];

                rknn_matmul_info info;
                memset(&info, 0, sizeof(info));
                info.M = 1;  /* process one row at a time */
                info.K = K;
                info.N = N;
#ifdef MTE_USE_FP16
                info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16;
                info.B_layout = 0;  /* normal layout for FP16 */
#else
                info.type = RKNN_FLOAT16_MM_INT8_TO_FLOAT32;
                info.B_layout = 1;  /* native layout for INT8 */
#endif
                info.B_quant_type = 0;

                memset(&enc->layer_io[li][m], 0, sizeof(rknn_matmul_io_attr));
                ret = rknn_matmul_create(&enc->layer_ctx[li][m], &info, &enc->layer_io[li][m]);
                if (ret != 0) {
                    fprintf(stderr, "[zipformer] FATAL: layer %d/%d %s matmul create failed: %d\n",
                            s, l, PROJ_NAMES[m], ret);
                    goto fail;
                }

                /* Allocate I/O memory */
                enc->layer_A[li][m] = rknn_create_mem(enc->layer_ctx[li][m], enc->layer_io[li][m].A.size);
                enc->layer_B[li][m] = rknn_create_mem(enc->layer_ctx[li][m], enc->layer_io[li][m].B.size);
                enc->layer_C[li][m] = rknn_create_mem(enc->layer_ctx[li][m], enc->layer_io[li][m].C.size);

                if (!enc->layer_A[li][m] || !enc->layer_B[li][m] || !enc->layer_C[li][m]) {
                    fprintf(stderr, "[zipformer] FATAL: layer %d/%d %s mem alloc failed\n", s, l, PROJ_NAMES[m]);
                    goto fail;
                }

#ifdef MTE_USE_FP16
                /* FP16 mode: load FP16 weights directly into B memory */
                snprintf(path, sizeof(path), "%s/%s.fp16.bin", layer_dir, PROJ_NAMES[m]);
                {
                    size_t wsize;
                    void* w_data = load_bin_raw(path, &wsize);
                    if (!w_data) goto fail;
                    /* FP16 weights: K*N*2 bytes, copy directly (B_layout=0, normal) */
                    memcpy(enc->layer_B[li][m]->virt_addr, w_data, K * N * sizeof(uint16_t));
                    free(w_data);
                }
                enc->layer_scales[li][m] = NULL; /* no scales in FP16 mode */
#else
                /* INT8 mode: load INT8 weights and convert to native layout */
                snprintf(path, sizeof(path), "%s/%s.int8.bin", layer_dir, PROJ_NAMES[m]);
                {
                    size_t wsize;
                    void* w_data = load_bin_raw(path, &wsize);
                    if (!w_data) goto fail;
                    ret = rknn_B_normal_layout_to_native_layout(
                        w_data, enc->layer_B[li][m]->virt_addr, K, N, &info);
                    free(w_data);
                    if (ret != 0) {
                        fprintf(stderr, "[zipformer] FATAL: layer %d/%d %s B layout failed: %d\n",
                                s, l, PROJ_NAMES[m], ret);
                        goto fail;
                    }
                }

                /* Load per-column scales [N] */
                enc->layer_scales[li][m] = (float*)malloc(N * sizeof(float));
                snprintf(path, sizeof(path), "%s/%s.scales.bin", layer_dir, PROJ_NAMES[m]);
                if (load_bin_f32(path, enc->layer_scales[li][m], N) != 0) goto fail;
#endif

                /* Load bias [N] (naming convention: {name}_bias.fp32.bin) */
                if (PROJ_HAS_BIAS[m]) {
                    enc->layer_bias[li][m] = (float*)malloc(N * sizeof(float));
                    snprintf(path, sizeof(path), "%s/%s_bias.fp32.bin", layer_dir, PROJ_NAMES[m]);
                    if (load_bin_f32(path, enc->layer_bias[li][m], N) != 0) {
                        /* Bias might not exist for some projections, zero-fill */
                        fprintf(stderr, "[zipformer] Warning: no bias for %s/%s, using zeros\n",
                                layer_dir, PROJ_NAMES[m]);
                        memset(enc->layer_bias[li][m], 0, N * sizeof(float));
                    }
                } else {
                    enc->layer_bias[li][m] = NULL;  /* No bias for whiten/pos projections */
                }

                /* Bind I/O */
                rknn_matmul_set_io_mem(enc->layer_ctx[li][m], enc->layer_A[li][m], &enc->layer_io[li][m].A);
                rknn_matmul_set_io_mem(enc->layer_ctx[li][m], enc->layer_B[li][m], &enc->layer_io[li][m].B);
                rknn_matmul_set_io_mem(enc->layer_ctx[li][m], enc->layer_C[li][m], &enc->layer_io[li][m].C);
            }

            /* Load conv module weights */
            if (load_conv_module(&enc->conv[li][0], weight_dir, s, l, 0) != 0) goto fail;
            if (load_conv_module(&enc->conv[li][1], weight_dir, s, l, 1) != 0) goto fail;

            /* Load bypass_scale */
            snprintf(path, sizeof(path), "%s/stack%d_layer%d/bypass_scale.fp32.bin", weight_dir, s, l);
            {
                float bs;
                if (load_bin_f32(path, &bs, 1) != 0) {
                    bs = 1.0f;  /* default */
                    fprintf(stderr, "[zipformer] Warning: no bypass_scale for stack%d_layer%d, using 1.0\n", s, l);
                }
                enc->bypass_scale[li] = bs;
            }

            printf("[zipformer] Stack %d Layer %d loaded (bypass_scale=%.4f)\n",
                   s, l, enc->bypass_scale[li]);
        }
    }

    /* ─── Encoder output projection: [256, 512] ─── */
    {
        rknn_matmul_info info;
        memset(&info, 0, sizeof(info));
        info.M = 1;
        info.K = HIDDEN_DIM;
        info.N = ENCODER_OUT_DIM;
#ifdef MTE_USE_FP16
        info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16;
        info.B_layout = 0;
#else
        info.type = RKNN_FLOAT16_MM_INT8_TO_FLOAT32;
        info.B_layout = 1;
#endif
        info.B_quant_type = 0;

        memset(&enc->proj_io, 0, sizeof(rknn_matmul_io_attr));
        ret = rknn_matmul_create(&enc->proj_ctx, &info, &enc->proj_io);
        if (ret != 0) {
            fprintf(stderr, "[zipformer] FATAL: encoder_proj matmul create failed: %d\n", ret);
            goto fail;
        }

        enc->proj_A = rknn_create_mem(enc->proj_ctx, enc->proj_io.A.size);
        enc->proj_B = rknn_create_mem(enc->proj_ctx, enc->proj_io.B.size);
        enc->proj_C = rknn_create_mem(enc->proj_ctx, enc->proj_io.C.size);
        if (!enc->proj_A || !enc->proj_B || !enc->proj_C) {
            fprintf(stderr, "[zipformer] FATAL: encoder_proj mem alloc failed\n");
            goto fail;
        }

#ifdef MTE_USE_FP16
        snprintf(path, sizeof(path), "%s/encoder_proj.fp16.bin", weight_dir);
        {
            size_t wsize;
            void* w_data = load_bin_raw(path, &wsize);
            if (!w_data) goto fail;
            memcpy(enc->proj_B->virt_addr, w_data, HIDDEN_DIM * ENCODER_OUT_DIM * sizeof(uint16_t));
            free(w_data);
        }
        enc->proj_scales = NULL;
#else
        snprintf(path, sizeof(path), "%s/encoder_proj.int8.bin", weight_dir);
        {
            size_t wsize;
            void* w_data = load_bin_raw(path, &wsize);
            if (!w_data) goto fail;
            ret = rknn_B_normal_layout_to_native_layout(
                w_data, enc->proj_B->virt_addr, HIDDEN_DIM, ENCODER_OUT_DIM, &info);
            free(w_data);
            if (ret != 0) {
                fprintf(stderr, "[zipformer] FATAL: encoder_proj B layout failed: %d\n", ret);
                goto fail;
            }
        }

        enc->proj_scales = (float*)malloc(ENCODER_OUT_DIM * sizeof(float));
        snprintf(path, sizeof(path), "%s/encoder_proj.scales.bin", weight_dir);
        if (load_bin_f32(path, enc->proj_scales, ENCODER_OUT_DIM) != 0) goto fail;
#endif

        enc->proj_bias = (float*)malloc(ENCODER_OUT_DIM * sizeof(float));
        snprintf(path, sizeof(path), "%s/encoder_proj_bias.fp32.bin", weight_dir);
        if (load_bin_f32(path, enc->proj_bias, ENCODER_OUT_DIM) != 0) {
            memset(enc->proj_bias, 0, ENCODER_OUT_DIM * sizeof(float));
        }

        rknn_matmul_set_io_mem(enc->proj_ctx, enc->proj_A, &enc->proj_io.A);
        rknn_matmul_set_io_mem(enc->proj_ctx, enc->proj_B, &enc->proj_io.B);
        rknn_matmul_set_io_mem(enc->proj_ctx, enc->proj_C, &enc->proj_io.C);

        printf("[zipformer] encoder_proj loaded\n");
    }

    /* ─── Load inter-stack multi-scale weights ─── */
    {
        /* Downsample queries: encoders_{1,2,3,4}_downsample_query and downsample_output_query */
        const char* ds_names[NUM_DOWNSAMPLE_QUERIES] = {
            "encoders_1_downsample_query",
            "encoders_2_downsample_query",
            "encoders_3_downsample_query",
            "encoders_4_downsample_query",
            "downsample_output_query",
        };
        for (int i = 0; i < NUM_DOWNSAMPLE_QUERIES; i++) {
            snprintf(path, sizeof(path), "%s/inter_stack/%s.fp32.bin", weight_dir, ds_names[i]);
            enc->downsample_query[i] = load_bin_f32_alloc(path, HIDDEN_DIM);
            if (!enc->downsample_query[i]) {
                fprintf(stderr, "[zipformer] FATAL: cannot load %s\n", path);
                goto fail;
            }
        }

        /* Out combiner weights: encoders_{1,2,3,4}_out_combiner_weight1 */
        for (int i = 0; i < NUM_OUT_COMBINERS; i++) {
            snprintf(path, sizeof(path), "%s/inter_stack/encoders_%d_out_combiner_weight1.fp32.bin",
                     weight_dir, i + 1);
            if (load_bin_f32(path, &enc->out_combiner_weight[i], 1) != 0) {
                fprintf(stderr, "[zipformer] FATAL: cannot load out_combiner_weight for stack %d\n", i + 1);
                goto fail;
            }
        }

        /* Skip module weight */
        snprintf(path, sizeof(path), "%s/inter_stack/skip_modules_4_weight1.fp32.bin", weight_dir);
        if (load_bin_f32(path, &enc->skip_weight, 1) != 0) {
            fprintf(stderr, "[zipformer] FATAL: cannot load skip_modules_4_weight1\n");
            goto fail;
        }

        /* Upsample learned offsets (SimpleUpsample from icefall)
         * stack1: factor=2 -> (2,256)   -> file: upsample_offset.fp32.bin
         * stack2: factor=4 -> (4,256)   -> file: upsample_1_offset.fp32.bin
         * stack3: factor=8 -> (8,256)   -> file: upsample_2_offset.fp32.bin
         * stack4: factor=2 -> (2,256)   -> file: upsample_3_offset.fp32.bin */
        {
            const char* up_files[4] = {
                "upsample_offset",
                "upsample_1_offset",
                "upsample_2_offset",
                "upsample_3_offset",
            };
            int up_factors[4] = {2, 4, 8, 2};  /* DS_FACTOR[1..4] */
            for (int i = 0; i < 4; i++) {
                int n_floats = up_factors[i] * HIDDEN_DIM;
                snprintf(path, sizeof(path), "%s/inter_stack/%s.fp32.bin", weight_dir, up_files[i]);
                enc->upsample_offset[i] = load_bin_f32_alloc(path, n_floats);
                if (!enc->upsample_offset[i]) {
                    fprintf(stderr, "[zipformer] FATAL: cannot load %s\n", path);
                    goto fail;
                }
            }
        }

        /* Load positional encoding table from .npy file.
         * The .npy file has shape [1, 95, 256] float32.
         * NumPy .npy format: 10-byte magic + header + data.
         * We skip the header and read raw float32 data. */
        {
            snprintf(path, sizeof(path), "%s/pos_encoding_table.npy", weight_dir);
            FILE* npy = fopen(path, "rb");
            if (!npy) {
                fprintf(stderr, "[zipformer] FATAL: cannot open %s\n", path);
                goto fail;
            }
            /* Read and skip .npy header:
             * Magic: \x93NUMPY (6 bytes)
             * Version: 2 bytes
             * Header len: 2 bytes (v1) or 4 bytes (v2)
             * Header: ASCII dict + padding to 64-byte alignment */
            unsigned char magic[8];
            fread(magic, 1, 8, npy);
            int header_len;
            if (magic[6] == 1) {
                /* Version 1.0: 2-byte header length */
                unsigned char hl[2];
                fread(hl, 1, 2, npy);
                header_len = hl[0] | (hl[1] << 8);
            } else {
                /* Version 2.0+: 4-byte header length */
                unsigned char hl[4];
                fread(hl, 1, 4, npy);
                header_len = hl[0] | (hl[1] << 8) | (hl[2] << 16) | (hl[3] << 24);
            }
            fseek(npy, header_len, SEEK_CUR);
            /* Now read the data: 1 * 95 * 256 = 24320 floats */
            int pos_count = POS_ENC_LEN * HIDDEN_DIM;
            enc->pos_encoding_table = (float*)malloc(pos_count * sizeof(float));
            size_t nread = fread(enc->pos_encoding_table, sizeof(float), pos_count, npy);
            fclose(npy);
            if ((int)nread != pos_count) {
                fprintf(stderr, "[zipformer] FATAL: pos_encoding_table: expected %d floats, got %zu\n",
                        pos_count, nread);
                goto fail;
            }
            printf("[zipformer] pos_encoding_table loaded: [%d, %d]\n", POS_ENC_LEN, HIDDEN_DIM);
        }

        /* Load per-stack positional encoding tables from .npy files.
         * Each file has shape [size, 256] where size = 2*T_stack[s] + LEFT_CTX[s] - 1.
         * Files: pos_enc_stack{0,1,2,3,4}.npy */
        {
            static const int pos_enc_sizes[N_STACKS] = {95, 47, 23, 11, 47};
            for (int s = 0; s < N_STACKS; s++) {
                enc->pos_enc_len_per_stack[s] = pos_enc_sizes[s];
                snprintf(path, sizeof(path), "%s/pos_enc_stack%d.npy", weight_dir, s);
                FILE* npy = fopen(path, "rb");
                if (!npy) {
                    fprintf(stderr, "[zipformer] FATAL: cannot open %s\n", path);
                    goto fail;
                }
                /* Skip .npy header */
                unsigned char mag[8];
                fread(mag, 1, 8, npy);
                int hlen;
                if (mag[6] == 1) {
                    unsigned char hl[2]; fread(hl, 1, 2, npy);
                    hlen = hl[0] | (hl[1] << 8);
                } else {
                    unsigned char hl[4]; fread(hl, 1, 4, npy);
                    hlen = hl[0] | (hl[1]<<8) | (hl[2]<<16) | (hl[3]<<24);
                }
                fseek(npy, hlen, SEEK_CUR);
                int n_floats = pos_enc_sizes[s] * HIDDEN_DIM;
                enc->pos_enc_per_stack[s] = (float*)malloc(n_floats * sizeof(float));
                size_t nr = fread(enc->pos_enc_per_stack[s], sizeof(float), n_floats, npy);
                fclose(npy);
                if ((int)nr != n_floats) {
                    fprintf(stderr, "[zipformer] FATAL: pos_enc_stack%d: got %zu/%d floats\n", s, nr, n_floats);
                    goto fail;
                }
                printf("[zipformer] pos_enc_stack%d loaded: [%d, %d]\n", s, pos_enc_sizes[s], HIDDEN_DIM);
            }
        }

        /* Hardcoded norm_final eps values (learned, extracted from ONNX model).
         * These are the constants added to mean(x^2) in the RMS norm.
         * Order: s0l0, s0l1, s1l0, s1l1, ..., s4l0, s4l1 */
        {
            static const float norm_eps_values[N_TOTAL_LAYERS] = {
                6.432362f, 8.898914f,   /* stack 0, layers 0-1 */
                5.620271f, 10.536471f,  /* stack 1, layers 0-1 */
                3.654023f, 5.342896f,   /* stack 2, layers 0-1 */
                4.219106f, 7.292570f,   /* stack 3, layers 0-1 */
                6.848550f, 4.731342f,   /* stack 4, layers 0-1 */
            };
            for (int i = 0; i < N_TOTAL_LAYERS; i++)
                enc->norm_final_eps[i] = norm_eps_values[i];
            printf("[zipformer] norm_final_eps loaded (hardcoded from ONNX)\n");
        }

        printf("[zipformer] Inter-stack weights loaded\n");
        printf("[zipformer]   out_combiner (raw, used directly): [%.4f, %.4f, %.4f, %.4f]\n",
               enc->out_combiner_weight[0], enc->out_combiner_weight[1],
               enc->out_combiner_weight[2], enc->out_combiner_weight[3]);
        printf("[zipformer]   skip_weight (raw, used directly)=%.4f\n",
               enc->skip_weight);
    }

    /* ─── Warmup NPU ─── */
    printf("[zipformer] Warming up NPU...\n");
    for (int w = 0; w < 2; w++) {
        for (int li = 0; li < N_TOTAL_LAYERS; li++)
            for (int m = 0; m < MM_PER_LAYER; m++)
                rknn_matmul_run(enc->layer_ctx[li][m]);
        rknn_matmul_run(enc->proj_ctx);
    }

    printf("[zipformer] Init complete. %d layer contexts + 1 proj context = %d total\n",
           N_TOTAL_LAYERS * MM_PER_LAYER, N_TOTAL_LAYERS * MM_PER_LAYER + 1);
    return enc;

fail:
    zipformer_encoder_destroy(enc);
    return NULL;
}

/* ─── Run a single encoder layer ─── */

/**
 * Correct operation order (from ONNX reference):
 *  1. x += ff1(x)                     -- NO 0.5 scale (baked into weights)
 *  2. whiten = cumulative_mean(x)
 *  3. x_attn_in = x + whiten_proj(whiten)
 *  4. in_proj(x_attn_in) -> multi-head attn with pos bias -> scores, attn_out1
 *  5. x = x_attn_in + out_proj(attn_out1)   -- residual from x_attn_in
 *  6. x += conv1(x)
 *  7. x += ff2(x)                     -- NO 0.5 scale
 *  8. v2 = whiten2_proj(x) -> V2 multi-head with reused scores -> attn_out2
 *  9. x += out_proj2(attn_out2)
 * 10. x += conv2(x)
 * 11. x += ff3(x)                     -- NO 0.5 scale
 * 12. x_normed = x * (mean(x^2) + eps)^(-0.5)   -- RMS norm with learned eps
 * 13. output = x_orig + bypass_scale * (x_normed - x_orig)
 */
static void run_layer(ZipformerEncoder* enc, int layer_idx,
                      float* x, int T) {
    float* residual_buf = (float*)malloc(T * HIDDEN_DIM * sizeof(float));
    int debug = (layer_idx == 0);  /* only debug first layer */

    /* Save input for bypass */
    float* x_orig = (float*)malloc(T * HIDDEN_DIM * sizeof(float));
    memcpy(x_orig, x, T * HIDDEN_DIM * sizeof(float));

    if (debug) {
        printf("[DEBUG] === Layer %d (T=%d) ===\n", layer_idx, T);
        debug_stats("input", x, T * HIDDEN_DIM);
    }

    /* 1. x += ff1(x)  — NO 0.5 scale (baked into weights at export) */
    run_ff_module(enc, layer_idx, 0, x, T, residual_buf);
    vec_add(x, residual_buf, T * HIDDEN_DIM);
    if (debug) debug_stats("after_ff1", x, T * HIDDEN_DIM);

    /* 2. Whiten: cumulative mean along time axis
     * cumsum[t] = sum(x[0..t]); whitened[t] = cumsum[t] / (t+1)
     * Then: x_attn_in = x + whiten_proj(whitened)
     * For non-streaming, cached_len=0, so indices are [1, 2, ..., T] */
    {
        float* whitened = (float*)malloc(T * HIDDEN_DIM * sizeof(float));
        float* cumsum = (float*)malloc(HIDDEN_DIM * sizeof(float));
        memset(cumsum, 0, HIDDEN_DIM * sizeof(float));

        for (int t = 0; t < T; t++) {
            for (int d = 0; d < HIDDEN_DIM; d++) {
                cumsum[d] += x[t * HIDDEN_DIM + d];
                whitened[t * HIDDEN_DIM + d] = cumsum[d] / (float)(t + 1);
            }
        }
        free(cumsum);

        /* 3. whiten_proj: [T, 256] -> [T, 256] */
        float* whiten_out = (float*)malloc(T * HIDDEN_DIM * sizeof(float));
        run_matmul_w8a16_rows(
            enc->layer_ctx[layer_idx][MM_ATTN_WHITEN],
            enc->layer_A[layer_idx][MM_ATTN_WHITEN],
            enc->layer_C[layer_idx][MM_ATTN_WHITEN],
            whitened, T, HIDDEN_DIM, whiten_out, HIDDEN_DIM,
            enc->layer_scales[layer_idx][MM_ATTN_WHITEN],
            enc->layer_bias[layer_idx][MM_ATTN_WHITEN]);
        free(whitened);

        /* x_attn_in = x + whiten_proj_out (stored in x for next step) */
        float* x_attn_in = (float*)malloc(T * HIDDEN_DIM * sizeof(float));
        for (int i = 0; i < T * HIDDEN_DIM; i++)
            x_attn_in[i] = x[i] + whiten_out[i];
        free(whiten_out);

        /* 4-5. Self-attention: returns out_proj(attn_out1) and softmax scores */
        float* attn_scores = NULL;
        int T_kv = 0;
        run_self_attention(enc, layer_idx, x_attn_in, T, residual_buf,
                           &attn_scores, &T_kv);

        /* x = x_attn_in + out_proj(attn_out1) */
        for (int i = 0; i < T * HIDDEN_DIM; i++)
            x[i] = x_attn_in[i] + residual_buf[i];
        free(x_attn_in);
        if (debug) debug_stats("after_attn", x, T * HIDDEN_DIM);

        /* 6. x += conv1(x) */
        run_conv_module(&enc->conv[layer_idx][0], x, T, residual_buf);
        vec_add(x, residual_buf, T * HIDDEN_DIM);
        if (debug) debug_stats("after_conv1", x, T * HIDDEN_DIM);

        /* 7. x += ff2(x)  — NO 0.5 scale */
        run_ff_module(enc, layer_idx, 1, x, T, residual_buf);
        vec_add(x, residual_buf, T * HIDDEN_DIM);
        if (debug) debug_stats("after_ff2", x, T * HIDDEN_DIM);

        /* 8-9. V2 path: whiten2_proj(x) -> reuse softmax scores -> out_proj2 */
        {
            /* whiten2_proj: [T, 256] -> [T, 96] */
            float* v2 = (float*)malloc(T * VAL_DIM * sizeof(float));
            run_matmul_w8a16_rows(
                enc->layer_ctx[layer_idx][MM_ATTN_WHITEN2],
                enc->layer_A[layer_idx][MM_ATTN_WHITEN2],
                enc->layer_C[layer_idx][MM_ATTN_WHITEN2],
                x, T, HIDDEN_DIM, v2, VAL_DIM,
                enc->layer_scales[layer_idx][MM_ATTN_WHITEN2],
                enc->layer_bias[layer_idx][MM_ATTN_WHITEN2]);

            /* Multi-head V2: reuse attn_scores
             * scores[NUM_HEADS, T, T_kv] @ V2[T_kv, VAL_DIM] -> attn_out2[T, VAL_DIM] */
            float* attn_out2 = (float*)calloc(T * VAL_DIM, sizeof(float));
            for (int h = 0; h < NUM_HEADS; h++) {
                for (int tq = 0; tq < T; tq++) {
                    float* ao = attn_out2 + tq * VAL_DIM + h * HEAD_DIM_V;
                    for (int tkv = 0; tkv < T_kv; tkv++) {
                        float sc = attn_scores[h * T * T_kv + tq * T_kv + tkv];
                        const float* v_row = v2 + tkv * VAL_DIM + h * HEAD_DIM_V;
                        for (int d = 0; d < HEAD_DIM_V; d++) {
                            ao[d] += sc * v_row[d];
                        }
                    }
                }
            }
            free(v2);

            /* out_proj2: [T, 96] -> [T, 256] */
            float* out2 = (float*)malloc(T * HIDDEN_DIM * sizeof(float));
            run_matmul_w8a16_rows(
                enc->layer_ctx[layer_idx][MM_ATTN_OUT2],
                enc->layer_A[layer_idx][MM_ATTN_OUT2],
                enc->layer_C[layer_idx][MM_ATTN_OUT2],
                attn_out2, T, VAL_DIM, out2, HIDDEN_DIM,
                enc->layer_scales[layer_idx][MM_ATTN_OUT2],
                enc->layer_bias[layer_idx][MM_ATTN_OUT2]);
            free(attn_out2);

            /* x += out_proj2(attn_out2) */
            vec_add(x, out2, T * HIDDEN_DIM);
            free(out2);
        }
        free(attn_scores);
    }
    if (debug) debug_stats("after_v2", x, T * HIDDEN_DIM);

    /* 10. x += conv2(x) */
    run_conv_module(&enc->conv[layer_idx][1], x, T, residual_buf);
    vec_add(x, residual_buf, T * HIDDEN_DIM);
    if (debug) debug_stats("after_conv2", x, T * HIDDEN_DIM);

    /* 11. x += ff3(x)  — NO 0.5 scale */
    run_ff_module(enc, layer_idx, 2, x, T, residual_buf);
    vec_add(x, residual_buf, T * HIDDEN_DIM);
    if (debug) debug_stats("after_ff3", x, T * HIDDEN_DIM);

    /* 12-13. Norm Final + Bypass:
     * x_normed = x * (mean(x^2) + eps_learned)^(-0.5)
     * output = x_orig + bypass_scale * (x_normed - x_orig) */
    {
        float bs = enc->bypass_scale[layer_idx];
        float eps = enc->norm_final_eps[layer_idx];
        for (int t = 0; t < T; t++) {
            float* xt = x + t * HIDDEN_DIM;
            const float* xo = x_orig + t * HIDDEN_DIM;

            /* Compute mean(x^2) */
            float sum_sq = 0.0f;
            for (int d = 0; d < HIDDEN_DIM; d++)
                sum_sq += xt[d] * xt[d];
            float mean_sq = sum_sq / HIDDEN_DIM;
            float rms_scale = 1.0f / sqrtf(mean_sq + eps);

            /* output = x_orig + bs * (x_normed - x_orig) */
            for (int d = 0; d < HIDDEN_DIM; d++) {
                float x_normed = xt[d] * rms_scale;
                xt[d] = xo[d] + bs * (x_normed - xo[d]);
            }
        }
    }
    if (debug) {
        debug_stats("after_bypass", x, T * HIDDEN_DIM);
        printf("[DEBUG] bypass_scale=%.6f, norm_eps=%.4f\n",
               enc->bypass_scale[layer_idx], enc->norm_final_eps[layer_idx]);
    }

    free(x_orig);
    free(residual_buf);
}

/* ─── Run (multi-scale) ─── */

int zipformer_encoder_run(ZipformerEncoder* enc,
                          const float* features, int T, int feat_dim,
                          float* output, int* out_T)
{
    if (!enc) return -1;
    if (feat_dim != HIDDEN_DIM) {
        fprintf(stderr, "[zipformer] ERROR: feat_dim=%d, expected %d\n", feat_dim, HIDDEN_DIM);
        return -1;
    }
    if (T > enc->max_T) {
        fprintf(stderr, "[zipformer] ERROR: T=%d exceeds max_T=%d\n", T, enc->max_T);
        return -1;
    }

    struct timespec t0, t1, t_stack_start;
    enc->time_npu_ms = 0;
    enc->time_cpu_ms = 0;

    clock_gettime(CLOCK_MONOTONIC, &t0);

    const int D = HIDDEN_DIM;

    /* Temporal resolution per stack:
     * Stack 0: T0 = T
     * Stack 1: T1 = (T0+1)/2
     * Stack 2: T2 = (T1+1)/2
     * Stack 3: T3 = (T2+1)/2
     * Stack 4: T4 = (T3+1)/2
     */
    int T_stack[N_STACKS];
    T_stack[0] = T;
    for (int s = 1; s < N_STACKS; s++)
        T_stack[s] = (T_stack[s-1] + 1) / 2;

    printf("[zipformer] Multi-scale T: [%d, %d, %d, %d, %d]\n",
           T_stack[0], T_stack[1], T_stack[2], T_stack[3], T_stack[4]);

    /* Allocate working buffers - one per stack resolution.
     * We need to keep stack0_out for the final output downsample. */
    float* x = (float*)malloc(T * D * sizeof(float));
    memcpy(x, features, T * D * sizeof(float));

    /* ═══ Stack 0 (layers 0-1): T_stack[0] frames ═══ */
    clock_gettime(CLOCK_MONOTONIC, &t_stack_start);
    run_layer(enc, 0, x, T_stack[0]);
    run_layer(enc, 1, x, T_stack[0]);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("[zipformer] Stack 0 done: T=%d (%.1fms)\n",
           T_stack[0], time_diff_ms(&t_stack_start, &t1));

    /* Save stack0_out for final output downsample */
    float* stack0_out = (float*)malloc(T_stack[0] * D * sizeof(float));
    memcpy(stack0_out, x, T_stack[0] * D * sizeof(float));

    /* ═══ Downsample stack0 -> stack1 resolution ═══ */
    float* x_ds = (float*)malloc(T_stack[1] * D * sizeof(float));
    downsample(x, T_stack[0], D, enc->downsample_query[0], x_ds, 2);

    /* Out combiner: SimpleCombiner from icefall Zipformer.
     * forward(src_new, src_old) = sigmoid(weight) * src_new + (1-sigmoid(weight)) * src_old
     * In non-streaming mode, src_old = 0, so output = sigmoid(weight) * src_new.
     * The stored weight1 is the RAW parameter (pre-sigmoid). */
    {
        float alpha = 1.0f / (1.0f + expf(-enc->out_combiner_weight[0]));
        for (int i = 0; i < T_stack[1] * D; i++)
            x_ds[i] *= alpha;
    }

    /* ═══ Stack 1 (layers 2-3): T_stack[1] frames ═══ */
    clock_gettime(CLOCK_MONOTONIC, &t_stack_start);
    run_layer(enc, 2, x_ds, T_stack[1]);
    run_layer(enc, 3, x_ds, T_stack[1]);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("[zipformer] Stack 1 done: T=%d (%.1fms)\n",
           T_stack[1], time_diff_ms(&t_stack_start, &t1));

    /* Save stack1_out for upsampling back later */
    float* stack1_out = (float*)malloc(T_stack[1] * D * sizeof(float));
    memcpy(stack1_out, x_ds, T_stack[1] * D * sizeof(float));

    /* ═══ Downsample stack1 -> stack2 resolution ═══ */
    float* x_ds2 = (float*)malloc(T_stack[2] * D * sizeof(float));
    downsample(x_ds, T_stack[1], D, enc->downsample_query[1], x_ds2, 2);
    {
        float alpha = 1.0f / (1.0f + expf(-enc->out_combiner_weight[1]));
        for (int i = 0; i < T_stack[2] * D; i++)
            x_ds2[i] *= alpha;
    }
    free(x_ds);

    /* ═══ Stack 2 (layers 4-5): T_stack[2] frames ═══ */
    clock_gettime(CLOCK_MONOTONIC, &t_stack_start);
    run_layer(enc, 4, x_ds2, T_stack[2]);
    run_layer(enc, 5, x_ds2, T_stack[2]);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("[zipformer] Stack 2 done: T=%d (%.1fms)\n",
           T_stack[2], time_diff_ms(&t_stack_start, &t1));

    /* Save stack2_out */
    float* stack2_out = (float*)malloc(T_stack[2] * D * sizeof(float));
    memcpy(stack2_out, x_ds2, T_stack[2] * D * sizeof(float));

    /* ═══ Downsample stack2 -> stack3 resolution ═══ */
    float* x_ds3 = (float*)malloc(T_stack[3] * D * sizeof(float));
    downsample(x_ds2, T_stack[2], D, enc->downsample_query[2], x_ds3, 2);
    {
        float alpha = 1.0f / (1.0f + expf(-enc->out_combiner_weight[2]));
        for (int i = 0; i < T_stack[3] * D; i++)
            x_ds3[i] *= alpha;
    }
    free(x_ds2);

    /* ═══ Stack 3 (layers 6-7): T_stack[3] frames ═══ */
    clock_gettime(CLOCK_MONOTONIC, &t_stack_start);
    run_layer(enc, 6, x_ds3, T_stack[3]);
    run_layer(enc, 7, x_ds3, T_stack[3]);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("[zipformer] Stack 3 done: T=%d (%.1fms)\n",
           T_stack[3], time_diff_ms(&t_stack_start, &t1));

    /* Save stack3_out */
    float* stack3_out = (float*)malloc(T_stack[3] * D * sizeof(float));
    memcpy(stack3_out, x_ds3, T_stack[3] * D * sizeof(float));

    /* ═══ Downsample stack3 -> stack4 resolution ═══ */
    float* x_ds4 = (float*)malloc(T_stack[4] * D * sizeof(float));
    downsample(x_ds3, T_stack[3], D, enc->downsample_query[3], x_ds4, 2);
    {
        float alpha = 1.0f / (1.0f + expf(-enc->out_combiner_weight[3]));
        for (int i = 0; i < T_stack[4] * D; i++)
            x_ds4[i] *= alpha;
    }
    free(x_ds3);

    /* ═══ Stack 4 (layers 8-9): T_stack[4] frames ═══ */
    clock_gettime(CLOCK_MONOTONIC, &t_stack_start);
    run_layer(enc, 8, x_ds4, T_stack[4]);
    run_layer(enc, 9, x_ds4, T_stack[4]);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("[zipformer] Stack 4 done: T=%d (%.1fms)\n",
           T_stack[4], time_diff_ms(&t_stack_start, &t1));

    /* ═══ Upsample path: stack4 -> stack3 -> stack2 -> stack1 -> stack0 ═══
     *
     * In icefall Zipformer, stacks at lower resolutions process their data,
     * then their outputs are upsampled back and combined with higher-resolution
     * stack outputs. The skip_modules control the blending.
     *
     * skip_modules_4: blends upsampled stack4 into stack3
     * No skip_modules for stacks 1-3 means direct addition after upsample.
     */
    {
        /* Upsample stack4 -> stack3 resolution */
        float* up4 = (float*)malloc(T_stack[3] * D * sizeof(float));
        int up4_T = T_stack[4] * 2;
        float* up4_full = (float*)malloc(up4_T * D * sizeof(float));
        upsample(x_ds4, T_stack[4], D, up4_full);
        /* Truncate to T_stack[3] in case of odd sizes */
        int copy_T = (up4_T < T_stack[3]) ? up4_T : T_stack[3];
        memcpy(up4, up4_full, copy_T * D * sizeof(float));
        if (copy_T < T_stack[3]) {
            /* Zero-fill remaining frames */
            memset(up4 + copy_T * D, 0, (T_stack[3] - copy_T) * D * sizeof(float));
        }
        free(up4_full);
        free(x_ds4);

        /* Blend: SimpleCombiner with skip_weight
         * forward(src1=stack3_out, src2=upsampled_stack4) =
         *   sigmoid(weight) * src1 + (1 - sigmoid(weight)) * src2 */
        float sw = 1.0f / (1.0f + expf(-enc->skip_weight));
        for (int i = 0; i < T_stack[3] * D; i++)
            stack3_out[i] = sw * stack3_out[i] + (1.0f - sw) * up4[i];
        free(up4);
        debug_stats("stack3_blended", stack3_out, T_stack[3] * D);

        /* Upsample stack3 -> stack2 res, add to stack2 */
        float* up3_full = (float*)malloc(T_stack[3] * 2 * D * sizeof(float));
        upsample(stack3_out, T_stack[3], D, up3_full);
        free(stack3_out);
        for (int i = 0; i < T_stack[2] * D; i++)
            stack2_out[i] += up3_full[i];
        free(up3_full);
        debug_stats("stack2_blended", stack2_out, T_stack[2] * D);

        /* Upsample stack2 -> stack1 res */
        float* up2_full = (float*)malloc(T_stack[2] * 2 * D * sizeof(float));
        upsample(stack2_out, T_stack[2], D, up2_full);
        free(stack2_out);
        for (int i = 0; i < T_stack[1] * D; i++)
            stack1_out[i] += up2_full[i];
        free(up2_full);
        debug_stats("stack1_blended", stack1_out, T_stack[1] * D);

        /* Upsample stack1 -> stack0 res */
        float* up1_full = (float*)malloc(T_stack[1] * 2 * D * sizeof(float));
        upsample(stack1_out, T_stack[1], D, up1_full);
        free(stack1_out);
        for (int i = 0; i < T_stack[0] * D; i++)
            stack0_out[i] += up1_full[i];
        free(up1_full);
    }

    debug_stats("stack0_final", stack0_out, T_stack[0] * D);

    /* ═══ Final output: downsample stack0_out → encoder_proj ═══ */
    int T_out = (T_stack[0] + 1) / 2;
    float* final_ds = (float*)malloc(T_out * D * sizeof(float));
    downsample(stack0_out, T_stack[0], D, enc->downsample_query[4], final_ds, 2);
    free(stack0_out);

    debug_stats("final_ds", final_ds, T_out * D);

    /* Encoder output projection: [T_out, 256] -> [T_out, 512] */
    run_matmul_w8a16_rows(
        enc->proj_ctx, enc->proj_A, enc->proj_C,
        final_ds, T_out, HIDDEN_DIM, output, ENCODER_OUT_DIM,
        enc->proj_scales, enc->proj_bias);

    *out_T = T_out;

    debug_stats("encoder_out", output, T_out * ENCODER_OUT_DIM);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double total_ms = time_diff_ms(&t0, &t1);
    printf("[zipformer] Total: %.1fms for T_in=%d -> T_out=%d (%.2fms/input_frame)\n",
           total_ms, T, T_out, total_ms / T);

    free(x);
    free(final_ds);
    return 0;
}

/* ─── Destroy ─── */

void zipformer_encoder_set_debug_dump(const char* dir) {
#ifdef MTE_DEBUG_DUMP
    g_debug_dump_dir = dir;
    if (dir) {
        printf("[zipformer] Debug dump enabled: %s\n", dir);
    }
#else
    (void)dir;
#endif
}

void zipformer_encoder_destroy(ZipformerEncoder* enc) {
    if (!enc) return;

    for (int li = 0; li < N_TOTAL_LAYERS; li++) {
        for (int m = 0; m < MM_PER_LAYER; m++) {
            if (enc->layer_C[li][m]) rknn_destroy_mem(enc->layer_ctx[li][m], enc->layer_C[li][m]);
            if (enc->layer_B[li][m]) rknn_destroy_mem(enc->layer_ctx[li][m], enc->layer_B[li][m]);
            if (enc->layer_A[li][m]) rknn_destroy_mem(enc->layer_ctx[li][m], enc->layer_A[li][m]);
            if (enc->layer_ctx[li][m]) rknn_matmul_destroy(enc->layer_ctx[li][m]);
            free(enc->layer_scales[li][m]);
            free(enc->layer_bias[li][m]);
        }
        free_conv_module(&enc->conv[li][0]);
        free_conv_module(&enc->conv[li][1]);
    }

    if (enc->proj_C) rknn_destroy_mem(enc->proj_ctx, enc->proj_C);
    if (enc->proj_B) rknn_destroy_mem(enc->proj_ctx, enc->proj_B);
    if (enc->proj_A) rknn_destroy_mem(enc->proj_ctx, enc->proj_A);
    if (enc->proj_ctx) rknn_matmul_destroy(enc->proj_ctx);
    free(enc->proj_scales);
    free(enc->proj_bias);

    /* Free inter-stack weights */
    for (int i = 0; i < NUM_DOWNSAMPLE_QUERIES; i++)
        free(enc->downsample_query[i]);

    for (int i = 0; i < 4; i++)
        free(enc->upsample_offset[i]);

    free(enc->pos_encoding_table);
    for (int s = 0; s < N_STACKS; s++)
        free(enc->pos_enc_per_stack[s]);
    free(enc);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Streaming Mode Implementation
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ─── State management ─── */

ZipformerState* zipformer_state_create(const ZipformerEncoder* enc) {
    (void)enc;  /* architecture params are compile-time constants */
    ZipformerState* state = (ZipformerState*)calloc(1, sizeof(ZipformerState));
    if (!state) return NULL;

    /* Allocate KV caches (variable size per stack due to left_ctx) */
    for (int s = 0; s < N_STACKS; s++) {
        int lctx = LEFT_CTX[s];
        for (int l = 0; l < N_LAYERS_PER_STACK; l++) {
            state->cached_key[s][l]  = (float*)calloc(lctx * KEY_DIM, sizeof(float));
            state->cached_val[s][l]  = (float*)calloc(lctx * VAL_DIM, sizeof(float));
            state->cached_val2[s][l] = (float*)calloc(lctx * VAL_DIM, sizeof(float));
            if (!state->cached_key[s][l] || !state->cached_val[s][l] ||
                !state->cached_val2[s][l]) {
                zipformer_state_destroy(state);
                return NULL;
            }
        }
    }
    /* cached_len, cached_avg, cached_conv1/2, prev_stack_out are zero from calloc */
    return state;
}

void zipformer_state_reset(ZipformerState* state) {
    if (!state) return;
    for (int s = 0; s < N_STACKS; s++) {
        int lctx = LEFT_CTX[s];
        for (int l = 0; l < N_LAYERS_PER_STACK; l++) {
            state->cached_len[s][l] = 0;
            memset(state->cached_avg[s][l], 0, HIDDEN_DIM * sizeof(float));
            memset(state->cached_key[s][l],  0, lctx * KEY_DIM * sizeof(float));
            memset(state->cached_val[s][l],  0, lctx * VAL_DIM * sizeof(float));
            memset(state->cached_val2[s][l], 0, lctx * VAL_DIM * sizeof(float));
            memset(state->cached_conv1[s][l], 0, HIDDEN_DIM * CONV_CACHE_LEN * sizeof(float));
            memset(state->cached_conv2[s][l], 0, HIDDEN_DIM * CONV_CACHE_LEN * sizeof(float));
        }
        free(state->prev_stack_out[s]);
        state->prev_stack_out[s] = NULL;
        state->prev_stack_out_T[s] = 0;
    }
}

void zipformer_state_destroy(ZipformerState* state) {
    if (!state) return;
    for (int s = 0; s < N_STACKS; s++) {
        for (int l = 0; l < N_LAYERS_PER_STACK; l++) {
            free(state->cached_key[s][l]);
            free(state->cached_val[s][l]);
            free(state->cached_val2[s][l]);
        }
        free(state->prev_stack_out[s]);
    }
    free(state);
}

/* ─── Streaming self-attention (Multi-Head with KV Cache + Positional Bias) ─── */

/**
 * Streaming multi-head self-attention with KV cache and positional bias.
 *
 * Current chunk: Q[T, 192], K_new[T, 192], V_new[T, 96], pos_q[T, 16]
 * Effective K = [cached_K ; K_new] of length left_ctx + T.
 * Effective V = [cached_V ; V_new] of length left_ctx + T.
 *
 * Returns: out_proj(attn_out1) in out[T, 256]
 * Also stores softmax scores in *scores_out and T_kv in *out_T_kv for V2 reuse.
 */
static void run_self_attention_streaming(
    ZipformerEncoder* enc, int stack_idx, int layer_idx_in_stack,
    const float* x, int T, float* out,
    ZipformerState* state,
    float** scores_out, int* out_T_kv)
{
    int s = stack_idx;
    int l = layer_idx_in_stack;
    int li = s * N_LAYERS_PER_STACK + l;  /* flat layer index */
    int lctx = LEFT_CTX[s];

    /* in_proj: [T, 256] -> [T, 496] */
    float* proj = (float*)malloc(T * IN_PROJ_DIM * sizeof(float));
    run_matmul_w8a16_rows(
        enc->layer_ctx[li][MM_ATTN_IN],
        enc->layer_A[li][MM_ATTN_IN], enc->layer_C[li][MM_ATTN_IN],
        x, T, HIDDEN_DIM, proj, IN_PROJ_DIM,
        enc->layer_scales[li][MM_ATTN_IN],
        enc->layer_bias[li][MM_ATTN_IN]);

    /* Split: Q[T, 192], K_new[T, 192], V_new[T, 96], pos_q[T, 16] */
    float* Q     = (float*)malloc(T * KEY_DIM * sizeof(float));
    float* K_new = (float*)malloc(T * KEY_DIM * sizeof(float));
    float* V_new = (float*)malloc(T * VAL_DIM * sizeof(float));
    float* pos_q = (float*)malloc(T * POS_DIM * sizeof(float));

    for (int t = 0; t < T; t++) {
        const float* row = proj + t * IN_PROJ_DIM;
        memcpy(Q     + t * KEY_DIM, row,                           KEY_DIM * sizeof(float));
        memcpy(K_new + t * KEY_DIM, row + KEY_DIM,                 KEY_DIM * sizeof(float));
        memcpy(V_new + t * VAL_DIM, row + 2 * KEY_DIM,             VAL_DIM * sizeof(float));
        memcpy(pos_q + t * POS_DIM, row + 2 * KEY_DIM + VAL_DIM,   POS_DIM * sizeof(float));
    }
#ifdef MTE_DEBUG_DUMP
    if (li == 0) {
        int sh_proj[] = {T, IN_PROJ_DIM};
        dump_npy("s0l0_attn_in_proj", proj, 2, sh_proj);
        int sh_q[] = {T, KEY_DIM};
        dump_npy("s0l0_Q", Q, 2, sh_q);
        int sh_v[] = {T, VAL_DIM};
        dump_npy("s0l0_V_new", V_new, 2, sh_v);
    }
#endif
    free(proj);

    /* Build effective K and V: [lctx + T, dim] */
    int T_kv = lctx + T;
    *out_T_kv = T_kv;

    float* K_eff = (float*)malloc(T_kv * KEY_DIM * sizeof(float));
    float* V_eff = (float*)malloc(T_kv * VAL_DIM * sizeof(float));

    memcpy(K_eff, state->cached_key[s][l], lctx * KEY_DIM * sizeof(float));
    memcpy(K_eff + lctx * KEY_DIM, K_new, T * KEY_DIM * sizeof(float));

    memcpy(V_eff, state->cached_val[s][l], lctx * VAL_DIM * sizeof(float));
    memcpy(V_eff + lctx * VAL_DIM, V_new, T * VAL_DIM * sizeof(float));

    /* ── Update K/V cache (keep most recent lctx frames) ── */
    if (T_kv > lctx) {
        int offset = T_kv - lctx;
        memcpy(state->cached_key[s][l], K_eff + offset * KEY_DIM, lctx * KEY_DIM * sizeof(float));
        memcpy(state->cached_val[s][l], V_eff + offset * VAL_DIM, lctx * VAL_DIM * sizeof(float));
    }

    free(K_new);
    free(V_new);

    /* ── Positional bias computation ──
     * Use per-stack positional encoding table (correct slice of global sinusoidal table).
     * Stack s uses global_table[5000-T-LEFT_CTX[s] : 4999+T], size = 2T + LEFT_CTX[s] - 1 */
    int S = enc->pos_enc_len_per_stack[s];
    const float* pos_table = enc->pos_enc_per_stack[s];
    float* pos_proj = (float*)malloc(S * POS_DIM * sizeof(float));
    run_matmul_w8a16_rows(
        enc->layer_ctx[li][MM_ATTN_POS_BIAS],
        enc->layer_A[li][MM_ATTN_POS_BIAS],
        enc->layer_C[li][MM_ATTN_POS_BIAS],
        pos_table, S, HIDDEN_DIM,
        pos_proj, POS_DIM,
        enc->layer_scales[li][MM_ATTN_POS_BIAS],
        enc->layer_bias[li][MM_ATTN_POS_BIAS]);

    /* Raw positional scores: [4, T, S] */
    float* raw_pos = (float*)calloc(NUM_HEADS * T * S, sizeof(float));
    for (int h = 0; h < NUM_HEADS; h++) {
        for (int t = 0; t < T; t++) {
            for (int si = 0; si < S; si++) {
                float dot = 0.0f;
                for (int d = 0; d < POS_HEAD_DIM; d++) {
                    dot += pos_q[t * POS_DIM + h * POS_HEAD_DIM + d] *
                           pos_proj[si * POS_DIM + h * POS_HEAD_DIM + d];
                }
                raw_pos[h * T * S + t * S + si] = dot;
            }
        }
    }
    free(pos_proj);
    free(pos_q);

    /* Gather relative position indices:
     * index = (T-1-t_q) + t_kv */
    float* pos_bias = (float*)malloc(NUM_HEADS * T * T_kv * sizeof(float));
    for (int h = 0; h < NUM_HEADS; h++) {
        for (int tq = 0; tq < T; tq++) {
            for (int tkv = 0; tkv < T_kv; tkv++) {
                int idx = (T - 1 - tq) + tkv;
                if (idx >= 0 && idx < S)
                    pos_bias[h * T * T_kv + tq * T_kv + tkv] =
                        raw_pos[h * T * S + tq * S + idx];
                else
                    pos_bias[h * T * T_kv + tq * T_kv + tkv] = 0.0f;
            }
        }
    }
    free(raw_pos);

    /* ── Multi-head attention: Q@K^T + pos_bias -> softmax ── */
    /* NOTE: No 1/sqrt(d_k) scaling — Zipformer bakes it into weights at export */
    float* scores = (float*)malloc(NUM_HEADS * T * T_kv * sizeof(float));

    for (int h = 0; h < NUM_HEADS; h++) {
        for (int tq = 0; tq < T; tq++) {
            const float* q_row = Q + tq * KEY_DIM + h * HEAD_DIM_K;
            for (int tkv = 0; tkv < T_kv; tkv++) {
                const float* k_row = K_eff + tkv * KEY_DIM + h * HEAD_DIM_K;
                float dot = 0.0f;
                for (int d = 0; d < HEAD_DIM_K; d++)
                    dot += q_row[d] * k_row[d];
                scores[h * T * T_kv + tq * T_kv + tkv] =
                    dot + pos_bias[h * T * T_kv + tq * T_kv + tkv];
            }
            softmax(scores + h * T * T_kv + tq * T_kv, T_kv);
        }
    }
    free(pos_bias);
    free(Q);
    free(K_eff);

    /* ── Weighted V: scores @ V_eff -> attn_out ── */
    float* attn_out = (float*)calloc(T * VAL_DIM, sizeof(float));
    for (int h = 0; h < NUM_HEADS; h++) {
        for (int tq = 0; tq < T; tq++) {
            float* ao = attn_out + tq * VAL_DIM + h * HEAD_DIM_V;
            for (int tkv = 0; tkv < T_kv; tkv++) {
                float sc = scores[h * T * T_kv + tq * T_kv + tkv];
                const float* v_row = V_eff + tkv * VAL_DIM + h * HEAD_DIM_V;
                for (int d = 0; d < HEAD_DIM_V; d++)
                    ao[d] += sc * v_row[d];
            }
        }
    }
    free(V_eff);

    /* out_proj: [T, 96] -> [T, 256] */
    run_matmul_w8a16_rows(
        enc->layer_ctx[li][MM_ATTN_OUT],
        enc->layer_A[li][MM_ATTN_OUT], enc->layer_C[li][MM_ATTN_OUT],
        attn_out, T, VAL_DIM, out, HIDDEN_DIM,
        enc->layer_scales[li][MM_ATTN_OUT],
        enc->layer_bias[li][MM_ATTN_OUT]);

    free(attn_out);

    /* Return softmax scores for V2 reuse */
    *scores_out = scores;
}

/* ─── Streaming conv module ─── */

/**
 * Streaming conv module with cache.
 *
 * Non-streaming uses symmetric padding (pad=15 on each side).
 * Streaming uses causal padding: prepend 30 cached frames, no right padding.
 * The cache stores the last 30 frames of GLU output.
 *
 * Flow:
 *   1. pointwise_conv1: [T, 256] -> [T, 512]
 *   2. GLU: [T, 512] -> [T, 256]
 *   3. Prepend conv cache [30, 256] to get [30+T, 256]
 *   4. Depthwise conv1d kernel=31, valid mode: [30+T, 256] -> [T, 256]
 *   5. Update cache: last 30 frames of [cache ; glu_out]
 *   6. SwooshL activation
 *   7. pointwise_conv2: [T, 256] -> [T, 256]
 */
static void run_conv_module_streaming(
    const ConvModule* cm, const float* x, int T, float* out,
    float* conv_cache /* [HIDDEN_DIM * CONV_CACHE_LEN], updated in-place */)
{
    /* 1. Pointwise conv1: [T, 256] -> [T, 512] */
    float* pw1_out = (float*)malloc(T * PW1_OUT_DIM * sizeof(float));
    for (int t = 0; t < T; t++) {
        const float* xt = x + t * HIDDEN_DIM;
        float* ot = pw1_out + t * PW1_OUT_DIM;
        for (int o = 0; o < PW1_OUT_DIM; o++) {
            float sum = cm->pw1_bias[o];
            const float* w = cm->pw1_weight + o * HIDDEN_DIM;
            int d;
            for (d = 0; d <= HIDDEN_DIM - 4; d += 4) {
                float32x4_t xv = vld1q_f32(xt + d);
                float32x4_t wv = vld1q_f32(w + d);
                sum += vaddvq_f32(vmulq_f32(xv, wv));
            }
            for (; d < HIDDEN_DIM; d++) sum += xt[d] * w[d];
            ot[o] = sum;
        }
    }

    /* 2. GLU: first 256 * sigmoid(second 256) */
    float* glu_out = (float*)malloc(T * HIDDEN_DIM * sizeof(float));
    for (int t = 0; t < T; t++) {
        const float* a = pw1_out + t * PW1_OUT_DIM;
        const float* b = a + HIDDEN_DIM;
        float* g = glu_out + t * HIDDEN_DIM;
        for (int d = 0; d < HIDDEN_DIM; d++) {
            g[d] = a[d] * (1.0f / (1.0f + expf(-b[d])));
        }
    }
    free(pw1_out);

    /* 3. Build padded input: [cache(30) ; glu_out(T)] = [30+T, 256]
     * Conv cache layout: [HIDDEN_DIM][CONV_CACHE_LEN] = channels-first
     * ONNX state: cached_conv1_s: [2, N, 256, 30] => per layer: [256, 30]
     * We need time-major for the conv: [30+T, 256]
     * Cache is stored as [256, 30] (channels-first), need to transpose to [30, 256] */
    int padded_T = CONV_CACHE_LEN + T;
    float* padded = (float*)malloc(padded_T * HIDDEN_DIM * sizeof(float));

    /* Transpose cache from [256, 30] to [30, 256] */
    for (int t = 0; t < CONV_CACHE_LEN; t++) {
        for (int c = 0; c < HIDDEN_DIM; c++) {
            padded[t * HIDDEN_DIM + c] = conv_cache[c * CONV_CACHE_LEN + t];
        }
    }
    /* Append GLU output */
    memcpy(padded + CONV_CACHE_LEN * HIDDEN_DIM, glu_out, T * HIDDEN_DIM * sizeof(float));

    /* 4. Depthwise conv1d: kernel=31, valid mode (no padding), groups=256
     * Input: [30+T, 256], output: [T, 256]
     * For each output frame t (0..T-1), the kernel window is padded[t..t+30] */
    float* dw_out = (float*)malloc(T * HIDDEN_DIM * sizeof(float));
    for (int c = 0; c < HIDDEN_DIM; c++) {
        const float* w = cm->dw_weight + c * CONV_KERNEL;
        float bias_val = cm->dw_bias[c];
        for (int t = 0; t < T; t++) {
            float sum = bias_val;
            for (int k = 0; k < CONV_KERNEL; k++) {
                sum += padded[(t + k) * HIDDEN_DIM + c] * w[k];
            }
            dw_out[t * HIDDEN_DIM + c] = sum;
        }
    }

    /* 5. Update cache: last 30 frames of [cache ; glu_out]
     * In channels-first layout: cache[c][t] = last 30 time steps of
     * the concatenation for channel c.
     * The last 30 frames of padded are padded[T..T+29] = padded[padded_T-30..padded_T-1]
     * which correspond to the last 30 frames of [old_cache ; glu_out]. */
    for (int c = 0; c < HIDDEN_DIM; c++) {
        for (int t = 0; t < CONV_CACHE_LEN; t++) {
            int src_t = T + t;  /* offset in padded: skip first T frames
                                 * (padded has 30+T frames, last 30 start at T) */
            conv_cache[c * CONV_CACHE_LEN + t] = padded[src_t * HIDDEN_DIM + c];
        }
    }

    free(padded);
    free(glu_out);

    /* 6. SwooshL activation */
    swooshl(dw_out, dw_out, T * HIDDEN_DIM);

    /* 7. Pointwise conv2: [T, 256] -> [T, 256] */
    for (int t = 0; t < T; t++) {
        const float* xt = dw_out + t * HIDDEN_DIM;
        float* ot = out + t * HIDDEN_DIM;
        for (int o = 0; o < HIDDEN_DIM; o++) {
            float sum = cm->pw2_bias[o];
            const float* w = cm->pw2_weight + o * HIDDEN_DIM;
            int d;
            for (d = 0; d <= HIDDEN_DIM - 4; d += 4) {
                float32x4_t xv = vld1q_f32(xt + d);
                float32x4_t wv = vld1q_f32(w + d);
                sum += vaddvq_f32(vmulq_f32(xv, wv));
            }
            for (; d < HIDDEN_DIM; d++) sum += xt[d] * w[d];
            ot[o] = sum;
        }
    }
    free(dw_out);
}

/* ─── Streaming layer ─── */

/**
 * Run a single encoder layer in streaming mode.
 *
 * Same operation order as non-streaming run_layer() but uses:
 * - KV cache for attention
 * - Conv cache for causal depthwise conv
 * - Running average cache for whiten (BasicNorm)
 *
 * Operation order (same as non-streaming):
 *  1. x += ff1(x)
 *  2. whiten + whiten_proj
 *  3-5. self_attn with KV cache + pos bias -> out_proj
 *  6. conv1 with conv cache
 *  7. x += ff2(x)
 *  8-9. V2 path: whiten2_proj + reuse scores + out_proj2
 * 10. conv2 with conv cache
 * 11. x += ff3(x)
 * 12-13. norm_final + bypass
 */
static void run_layer_streaming(ZipformerEncoder* enc,
                                int stack_idx, int layer_idx_in_stack,
                                float* x, int T,
                                ZipformerState* state) {
    int s = stack_idx;
    int l = layer_idx_in_stack;
    int li = s * N_LAYERS_PER_STACK + l;
    float* residual_buf = (float*)malloc(T * HIDDEN_DIM * sizeof(float));

    /* Save input for bypass */
    float* x_orig = (float*)malloc(T * HIDDEN_DIM * sizeof(float));
    memcpy(x_orig, x, T * HIDDEN_DIM * sizeof(float));

    /* 1. x += ff1(x) — NO 0.5 scale */
    run_ff_module(enc, li, 0, x, T, residual_buf);
#ifdef MTE_DEBUG_DUMP
    if (li == 0) {
        int sh2[] = {T, HIDDEN_DIM};
        dump_npy("s0l0_ff1_out", residual_buf, 2, sh2);
    }
#endif
    vec_add(x, residual_buf, T * HIDDEN_DIM);
#ifdef MTE_DEBUG_DUMP
    if (li == 0) {
        int sh2[] = {T, HIDDEN_DIM};
        dump_npy("s0l0_after_ff1", x, 2, sh2);
    }
#endif

    /* 2. Whiten: streaming cumulative mean with cached running average
     * For streaming, we maintain cached_avg and cached_len.
     * whitened[t] = cumsum / (cached_len + t + 1)
     * The cached_avg is the running sum / count from previous chunks. */
    {
        float* whitened = (float*)malloc(T * HIDDEN_DIM * sizeof(float));
        int64_t prev_len = state->cached_len[s][l];

        /* Compute streaming cumulative mean:
         * For each frame t, the "global" frame index is prev_len + t.
         * cumsum from prev chunk end: cached_avg * prev_len (the accumulated sum)
         * We add current frames cumulatively. */
        float cumsum[HIDDEN_DIM];
        /* Initialize cumsum with previous accumulated sum */
        for (int d = 0; d < HIDDEN_DIM; d++)
            cumsum[d] = state->cached_avg[s][l][d] * (float)prev_len;

        for (int t = 0; t < T; t++) {
            float count = (float)(prev_len + t + 1);
            for (int d = 0; d < HIDDEN_DIM; d++) {
                cumsum[d] += x[t * HIDDEN_DIM + d];
                whitened[t * HIDDEN_DIM + d] = cumsum[d] / count;
            }
        }
#ifdef MTE_DEBUG_DUMP
        if (li == 0) {
            int sh2[] = {T, HIDDEN_DIM};
            dump_npy("s0l0_whitened", whitened, 2, sh2);
        }
#endif

        /* Update cached_avg: running average = cumsum / (prev_len + T) */
        float new_count = (float)(prev_len + T);
        if (new_count > 0) {
            for (int d = 0; d < HIDDEN_DIM; d++)
                state->cached_avg[s][l][d] = cumsum[d] / new_count;
        }

        /* 3. whiten_proj: [T, 256] -> [T, 256] */
        float* whiten_out = (float*)malloc(T * HIDDEN_DIM * sizeof(float));
        run_matmul_w8a16_rows(
            enc->layer_ctx[li][MM_ATTN_WHITEN],
            enc->layer_A[li][MM_ATTN_WHITEN],
            enc->layer_C[li][MM_ATTN_WHITEN],
            whitened, T, HIDDEN_DIM, whiten_out, HIDDEN_DIM,
            enc->layer_scales[li][MM_ATTN_WHITEN],
            enc->layer_bias[li][MM_ATTN_WHITEN]);
        free(whitened);
#ifdef MTE_DEBUG_DUMP
        if (li == 0) {
            int sh2[] = {T, HIDDEN_DIM};
            dump_npy("s0l0_whiten_proj", whiten_out, 2, sh2);
        }
#endif

        /* x_attn_in = x + whiten_proj_out */
        float* x_attn_in = (float*)malloc(T * HIDDEN_DIM * sizeof(float));
        for (int i = 0; i < T * HIDDEN_DIM; i++)
            x_attn_in[i] = x[i] + whiten_out[i];
        free(whiten_out);
#ifdef MTE_DEBUG_DUMP
        if (li == 0) {
            int sh2[] = {T, HIDDEN_DIM};
            dump_npy("s0l0_x_attn_in", x_attn_in, 2, sh2);
        }
#endif

        /* 4-5. Streaming self-attention with KV cache */
        float* attn_scores = NULL;
        int T_kv = 0;
        run_self_attention_streaming(enc, s, l, x_attn_in, T, residual_buf,
                                     state, &attn_scores, &T_kv);
#ifdef MTE_DEBUG_DUMP
        if (li == 0) {
            int sh2[] = {T, HIDDEN_DIM};
            dump_npy("s0l0_attn_out_proj", residual_buf, 2, sh2);
            int sh3[] = {NUM_HEADS, T, T_kv};
            dump_npy("s0l0_attn_scores", attn_scores, 3, sh3);
        }
#endif

        /* x = x_attn_in + out_proj(attn_out1) */
        for (int i = 0; i < T * HIDDEN_DIM; i++)
            x[i] = x_attn_in[i] + residual_buf[i];
        free(x_attn_in);
#ifdef MTE_DEBUG_DUMP
        if (li == 0) {
            int sh2[] = {T, HIDDEN_DIM};
            dump_npy("s0l0_after_attn", x, 2, sh2);
        }
#endif

        /* 6. x += conv1(x) — streaming with conv cache */
        run_conv_module_streaming(&enc->conv[li][0], x, T, residual_buf,
                                  state->cached_conv1[s][l]);
        vec_add(x, residual_buf, T * HIDDEN_DIM);
#ifdef MTE_DEBUG_DUMP
        if (li == 0) {
            int sh2[] = {T, HIDDEN_DIM};
            dump_npy("s0l0_after_conv1", x, 2, sh2);
        }
#endif

        /* 7. x += ff2(x) — NO 0.5 scale */
        run_ff_module(enc, li, 1, x, T, residual_buf);
        vec_add(x, residual_buf, T * HIDDEN_DIM);
#ifdef MTE_DEBUG_DUMP
        if (li == 0) {
            int sh2[] = {T, HIDDEN_DIM};
            dump_npy("s0l0_after_ff2", x, 2, sh2);
        }
#endif

        /* 8-9. V2 path: whiten2_proj(x) -> reuse softmax scores -> out_proj2 */
        {
            /* whiten2_proj: [T, 256] -> [T, 96] */
            float* v2_new = (float*)malloc(T * VAL_DIM * sizeof(float));
            run_matmul_w8a16_rows(
                enc->layer_ctx[li][MM_ATTN_WHITEN2],
                enc->layer_A[li][MM_ATTN_WHITEN2],
                enc->layer_C[li][MM_ATTN_WHITEN2],
                x, T, HIDDEN_DIM, v2_new, VAL_DIM,
                enc->layer_scales[li][MM_ATTN_WHITEN2],
                enc->layer_bias[li][MM_ATTN_WHITEN2]);

            /* Build effective V2: [cached_val2 ; v2_new] */
            int lctx = LEFT_CTX[s];
            float* V2_eff = (float*)malloc(T_kv * VAL_DIM * sizeof(float));
            memcpy(V2_eff, state->cached_val2[s][l], lctx * VAL_DIM * sizeof(float));
            memcpy(V2_eff + lctx * VAL_DIM, v2_new, T * VAL_DIM * sizeof(float));

            /* Update V2 cache: keep most recent lctx frames */
            if (T_kv > lctx) {
                int offset = T_kv - lctx;
                memcpy(state->cached_val2[s][l], V2_eff + offset * VAL_DIM,
                       lctx * VAL_DIM * sizeof(float));
            }

            /* Multi-head: reuse attn_scores[NUM_HEADS, T, T_kv] @ V2_eff */
            float* attn_out2 = (float*)calloc(T * VAL_DIM, sizeof(float));
            for (int h = 0; h < NUM_HEADS; h++) {
                for (int tq = 0; tq < T; tq++) {
                    float* ao = attn_out2 + tq * VAL_DIM + h * HEAD_DIM_V;
                    for (int tkv = 0; tkv < T_kv; tkv++) {
                        float sc = attn_scores[h * T * T_kv + tq * T_kv + tkv];
                        const float* v_row = V2_eff + tkv * VAL_DIM + h * HEAD_DIM_V;
                        for (int d = 0; d < HEAD_DIM_V; d++)
                            ao[d] += sc * v_row[d];
                    }
                }
            }
            free(V2_eff);
            free(v2_new);

            /* out_proj2: [T, 96] -> [T, 256] */
            float* out2 = (float*)malloc(T * HIDDEN_DIM * sizeof(float));
            run_matmul_w8a16_rows(
                enc->layer_ctx[li][MM_ATTN_OUT2],
                enc->layer_A[li][MM_ATTN_OUT2],
                enc->layer_C[li][MM_ATTN_OUT2],
                attn_out2, T, VAL_DIM, out2, HIDDEN_DIM,
                enc->layer_scales[li][MM_ATTN_OUT2],
                enc->layer_bias[li][MM_ATTN_OUT2]);
            free(attn_out2);

            vec_add(x, out2, T * HIDDEN_DIM);
            free(out2);
        }
        free(attn_scores);
    }
#ifdef MTE_DEBUG_DUMP
    if (li == 0) {
        int sh2[] = {T, HIDDEN_DIM};
        dump_npy("s0l0_after_v2", x, 2, sh2);
    }
#endif

    /* 10. x += conv2(x) — streaming with conv cache */
    run_conv_module_streaming(&enc->conv[li][1], x, T, residual_buf,
                              state->cached_conv2[s][l]);
    vec_add(x, residual_buf, T * HIDDEN_DIM);
#ifdef MTE_DEBUG_DUMP
    if (li == 0) {
        int sh2[] = {T, HIDDEN_DIM};
        dump_npy("s0l0_after_conv2", x, 2, sh2);
    }
#endif

    /* 11. x += ff3(x) — NO 0.5 scale */
    run_ff_module(enc, li, 2, x, T, residual_buf);
    vec_add(x, residual_buf, T * HIDDEN_DIM);
#ifdef MTE_DEBUG_DUMP
    if (li == 0) {
        int sh2[] = {T, HIDDEN_DIM};
        dump_npy("s0l0_after_ff3", x, 2, sh2);
    }
#endif

    /* 12-13. Norm Final + Bypass:
     * x_normed = x * (mean(x^2) + eps_learned)^(-0.5)
     * output = x_orig + bypass_scale * (x_normed - x_orig) */
    {
        float bs = enc->bypass_scale[li];
        float eps = enc->norm_final_eps[li];
        for (int t = 0; t < T; t++) {
            float* xt = x + t * HIDDEN_DIM;
            const float* xo = x_orig + t * HIDDEN_DIM;

            float sum_sq = 0.0f;
            for (int d = 0; d < HIDDEN_DIM; d++)
                sum_sq += xt[d] * xt[d];
            float mean_sq = sum_sq / HIDDEN_DIM;
            float rms_scale = 1.0f / sqrtf(mean_sq + eps);

            for (int d = 0; d < HIDDEN_DIM; d++) {
                float x_normed = xt[d] * rms_scale;
                xt[d] = xo[d] + bs * (x_normed - xo[d]);
            }
        }
    }
#ifdef MTE_DEBUG_DUMP
    if (li == 0) {
        int sh2[] = {T, HIDDEN_DIM};
        dump_npy("s0l0_after_bypass", x, 2, sh2);
    }
#endif

    /* Update cached_len */
    state->cached_len[s][l] += T;

    free(x_orig);
    free(residual_buf);
}

/* ─── Streaming chunk run ─── */

/**
 * Upsample from src_T to dst_T = src_T * factor.
 * Implements icefall SimpleUpsample: for output frame t,
 *   dst[t] = src[t / factor] + offset[t % factor]
 * where offset has shape [factor, D].
 *
 * If offset is NULL, falls back to nearest-neighbor (no offset).
 */
static void upsample_to(const float* src, int src_T, int D,
                         float* dst, int dst_T,
                         const float* offset) {
    int factor = dst_T / src_T;
    for (int t = 0; t < dst_T; t++) {
        int src_t = t / factor;
        int frac  = t % factor;
        if (src_t >= src_T) src_t = src_T - 1;
        const float* src_row = src + src_t * D;
        float* dst_row = dst + t * D;
        if (offset) {
            const float* off_row = offset + frac * D;
            for (int d = 0; d < D; d++)
                dst_row[d] = src_row[d] + off_row[d];
        } else {
            memcpy(dst_row, src_row, D * sizeof(float));
        }
    }
}

int zipformer_encoder_run_chunk(ZipformerEncoder* enc,
                                ZipformerState* state,
                                const float* chunk_features, int T_chunk, int feat_dim,
                                float* output, int* out_T)
{
    if (!enc || !state) return -1;
    if (feat_dim != HIDDEN_DIM) {
        fprintf(stderr, "[zipformer-stream] ERROR: feat_dim=%d, expected %d\n",
                feat_dim, HIDDEN_DIM);
        return -1;
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    const int D = HIDDEN_DIM;
    const int T0 = T_chunk;  /* stack 0 resolution = full */

    /* Compute T at each stack's resolution.
     * DS_FACTOR = [1, 2, 4, 8, 2]
     * T_stack[s] = T_chunk / DS_FACTOR[s] */
    int T_stack[N_STACKS];
    for (int s = 0; s < N_STACKS; s++) {
        T_stack[s] = T_chunk / DS_FACTOR[s];
        if (T_stack[s] < 1) T_stack[s] = 1;
    }

    printf("[zipformer-stream] Chunk: T_chunk=%d, T_stack=[%d,%d,%d,%d,%d]\n",
           T_chunk, T_stack[0], T_stack[1], T_stack[2], T_stack[3], T_stack[4]);

    /* ═══════════════════════════════════════════════════════════════════
     * CORRECT multi-scale architecture (from ONNX graph tracing):
     *
     * The Zipformer processes stacks SEQUENTIALLY, not in parallel.
     * After each stack, its output is upsampled back to T0 resolution
     * and combined with the accumulated output via out_combiner.
     * The combined output is then downsampled for the next stack.
     *
     * Flow:
     *   1. Stack 0 (T0=16) -> stack0_out
     *   2. Downsample stack0_out (T0 -> T1=8) -> stack 1 input
     *   3. Stack 1 (T1=8) -> stack1_out
     *   4. Upsample stack1_out (T1 -> T0=16) -> upsampled1
     *   5. combined = alpha0 * stack0_out + (1-alpha0) * upsampled1
     *   6. Downsample combined (T0 -> T2=4) -> stack 2 input
     *   7. Stack 2 (T2=4) -> stack2_out
     *   8. Upsample stack2_out (T2 -> T0=16) -> upsampled2
     *   9. combined = alpha1 * combined + (1-alpha1) * upsampled2
     *  10. Downsample combined (T0 -> T3=2) -> stack 3 input
     *  11. Stack 3 (T3=2) -> stack3_out
     *  12. Upsample stack3_out (T3 -> T0=16) -> upsampled3
     *  13. combined = alpha2 * combined + (1-alpha2) * upsampled3
     *  14. skip_combined = skip * step5_combined + (1-skip) * combined
     *  15. Downsample skip_combined (T0 -> T4=8) -> stack 4 input
     *  16. Stack 4 (T4=8) -> stack4_out
     *  17. Upsample stack4_out (T4 -> T0=16) -> upsampled4
     *  18. combined = alpha3 * skip_combined + (1-alpha3) * upsampled4
     *  19. Downsample combined (T0 -> T_out=8) -> encoder_proj -> output
     * ═══════════════════════════════════════════════════════════════════ */

    /* Working buffer at stack 0 resolution */
    float* x = (float*)malloc(T0 * D * sizeof(float));
    memcpy(x, chunk_features, T0 * D * sizeof(float));

    /* ═══ Stack 0 (layers 0-1): T0 frames ═══ */
    run_layer_streaming(enc, 0, 0, x, T0, state);
    run_layer_streaming(enc, 0, 1, x, T0, state);

    /* combined = stack0_out (at T0 resolution) */
    float* combined = (float*)malloc(T0 * D * sizeof(float));
    memcpy(combined, x, T0 * D * sizeof(float));
    free(x);

    /* ═══ Stack 1 ═══ */
    {
        /* Downsample combined (T0=16) -> stack1 input (T1=T0/DS_FACTOR[1]=8), factor=2 */
        float* x_s1 = (float*)malloc(T_stack[1] * D * sizeof(float));
        downsample(combined, T0, D, enc->downsample_query[0], x_s1, DS_FACTOR[1]);

        /* Stack 1 layers */
        run_layer_streaming(enc, 1, 0, x_s1, T_stack[1], state);
        run_layer_streaming(enc, 1, 1, x_s1, T_stack[1], state);

        /* Upsample stack1_out (T1) -> T0 resolution (with learned offset) */
        float* up1 = (float*)malloc(T0 * D * sizeof(float));
        upsample_to(x_s1, T_stack[1], D, up1, T0, enc->upsample_offset[0]);
        free(x_s1);

        /* out_combiner_0: combined = w * combined + (1-w) * up1
         * NOTE: icefall SimpleCombiner uses raw weight directly (no sigmoid) */
        float w0 = enc->out_combiner_weight[0];
        for (int i = 0; i < T0 * D; i++)
            combined[i] = w0 * combined[i] + (1.0f - w0) * up1[i];
        free(up1);
    }

    /* Save combined after stack 0+1 for skip_modules.4 */
    float* combined_after_s01 = (float*)malloc(T0 * D * sizeof(float));
    memcpy(combined_after_s01, combined, T0 * D * sizeof(float));

    /* ═══ Stack 2 ═══ */
    {
        /* factor=4: T_stack[2] = T0/DS_FACTOR[2] = 16/4 = 4 */
        float* x_s2 = (float*)malloc(T_stack[2] * D * sizeof(float));
        downsample(combined, T0, D, enc->downsample_query[1], x_s2, DS_FACTOR[2]);

        run_layer_streaming(enc, 2, 0, x_s2, T_stack[2], state);
        run_layer_streaming(enc, 2, 1, x_s2, T_stack[2], state);

        float* up2 = (float*)malloc(T0 * D * sizeof(float));
        upsample_to(x_s2, T_stack[2], D, up2, T0, enc->upsample_offset[1]);
        free(x_s2);

        float w1 = enc->out_combiner_weight[1];
        for (int i = 0; i < T0 * D; i++)
            combined[i] = w1 * combined[i] + (1.0f - w1) * up2[i];
        free(up2);
    }

    /* ═══ Stack 3 ═══ */
    {
        /* factor=8: T_stack[3] = T0/DS_FACTOR[3] = 16/8 = 2 */
        float* x_s3 = (float*)malloc(T_stack[3] * D * sizeof(float));
        downsample(combined, T0, D, enc->downsample_query[2], x_s3, DS_FACTOR[3]);

        run_layer_streaming(enc, 3, 0, x_s3, T_stack[3], state);
        run_layer_streaming(enc, 3, 1, x_s3, T_stack[3], state);

        float* up3 = (float*)malloc(T0 * D * sizeof(float));
        upsample_to(x_s3, T_stack[3], D, up3, T0, enc->upsample_offset[2]);
        free(x_s3);

        float w2 = enc->out_combiner_weight[2];
        for (int i = 0; i < T0 * D; i++)
            combined[i] = w2 * combined[i] + (1.0f - w2) * up3[i];
        free(up3);
    }

    /* ═══ skip_modules.4: blend combined_after_s01 with combined_after_s0123
     * NOTE: icefall SimpleCombiner uses raw weight directly (no sigmoid) */
    {
        float sw = enc->skip_weight;
        for (int i = 0; i < T0 * D; i++)
            combined[i] = sw * combined_after_s01[i] + (1.0f - sw) * combined[i];
    }
    free(combined_after_s01);

    /* ═══ Stack 4 ═══ */
    {
        /* factor=2: T_stack[4] = T0/DS_FACTOR[4] = 16/2 = 8 */
        float* x_s4 = (float*)malloc(T_stack[4] * D * sizeof(float));
        downsample(combined, T0, D, enc->downsample_query[3], x_s4, DS_FACTOR[4]);

        run_layer_streaming(enc, 4, 0, x_s4, T_stack[4], state);
        run_layer_streaming(enc, 4, 1, x_s4, T_stack[4], state);

        float* up4 = (float*)malloc(T0 * D * sizeof(float));
        upsample_to(x_s4, T_stack[4], D, up4, T0, enc->upsample_offset[3]);
        free(x_s4);

        float w3 = enc->out_combiner_weight[3];
        for (int i = 0; i < T0 * D; i++)
            combined[i] = w3 * combined[i] + (1.0f - w3) * up4[i];
        free(up4);
    }

    /* ═══ Final output: downsample combined -> encoder_proj (factor=2) ═══ */
    int T_out_val = (T0 + 1) / 2;
    float* final_ds = (float*)malloc(T_out_val * D * sizeof(float));
    downsample(combined, T0, D, enc->downsample_query[4], final_ds, 2);
    free(combined);

    /* Encoder output projection: [T_out, 256] -> [T_out, 512] */
    run_matmul_w8a16_rows(
        enc->proj_ctx, enc->proj_A, enc->proj_C,
        final_ds, T_out_val, HIDDEN_DIM, output, ENCODER_OUT_DIM,
        enc->proj_scales, enc->proj_bias);

    *out_T = T_out_val;

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double total_ms = time_diff_ms(&t0, &t1);
    printf("[zipformer-stream] Chunk done: T_in=%d -> T_out=%d (%.1fms)\n",
           T_chunk, T_out_val, total_ms);

    free(final_ds);
    return 0;
}

/* ─── Test main ─── */

#ifdef ZIPFORMER_TEST_MAIN
int main(int argc, char** argv) {
    const char* weight_dir = "/tmp/zipformer_weights";
    int max_T = 64;
    int test_T = 8;

    if (argc >= 2) weight_dir = argv[1];
    if (argc >= 3) test_T = atoi(argv[2]);
    if (argc >= 4) max_T = atoi(argv[3]);
    if (max_T < test_T) max_T = test_T;

    printf("=== Zipformer Encoder Engine Test (Multi-Scale) ===\n");
    printf("Weight dir: %s\n", weight_dir);
    printf("Test T: %d, max_T: %d\n", test_T, max_T);
    printf("Architecture: %d stacks x %d layers = %d total (multi-scale)\n",
           N_STACKS, N_LAYERS_PER_STACK, N_TOTAL_LAYERS);
    {
        int ts[N_STACKS];
        ts[0] = test_T;
        for (int s = 1; s < N_STACKS; s++) ts[s] = (ts[s-1] + 1) / 2;
        printf("Multi-scale T: [%d, %d, %d, %d, %d] -> out T=%d\n",
               ts[0], ts[1], ts[2], ts[3], ts[4], (ts[0] + 1) / 2);
    }
    printf("Dims: hidden=%d, ffn=%d, attn_proj=%d, key=%d, val=%d, out=%d\n",
           HIDDEN_DIM, FFN_DIM, IN_PROJ_DIM, KEY_DIM, VAL_DIM, ENCODER_OUT_DIM);
    printf("Matmuls: %d per layer x %d layers + 1 proj = %d total\n",
           MM_PER_LAYER, N_TOTAL_LAYERS, N_TOTAL_LAYERS * MM_PER_LAYER + 1);

    /* Init engine */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    ZipformerEncoder* enc = zipformer_encoder_init(weight_dir, max_T);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (!enc) {
        fprintf(stderr, "FATAL: engine init failed\n");
        return 1;
    }
    printf("Init time: %.1fms\n", time_diff_ms(&t0, &t1));

    /* Generate random input [test_T, 256] */
    int feat_dim = HIDDEN_DIM;
    float* input = (float*)malloc(test_T * feat_dim * sizeof(float));
    unsigned int seed = 42;
    for (int i = 0; i < test_T * feat_dim; i++) {
        seed = seed * 1103515245 + 12345;
        input[i] = ((float)((seed >> 16) & 0x7FFF) / 32768.0f - 0.5f) * 0.1f;
    }

    /* Run inference (out_T may be smaller than test_T due to output downsample) */
    float* output = (float*)malloc(max_T * ENCODER_OUT_DIM * sizeof(float));
    int out_T = 0;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    int ret = zipformer_encoder_run(enc, input, test_T, feat_dim, output, &out_T);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    if (ret != 0) {
        fprintf(stderr, "FATAL: encoder run failed: %d\n", ret);
        zipformer_encoder_destroy(enc);
        return 1;
    }

    double run_ms = time_diff_ms(&t0, &t1);

    /* Output stats */
    printf("\n=== Output Stats ===\n");
    printf("Shape: [%d, %d]\n", out_T, ENCODER_OUT_DIM);

    float mn = output[0], mx = output[0];
    double sum = 0, sum_sq = 0;
    int nan_count = 0;
    for (int i = 0; i < out_T * ENCODER_OUT_DIM; i++) {
        if (isnan(output[i]) || isinf(output[i])) {
            nan_count++;
            continue;
        }
        if (output[i] < mn) mn = output[i];
        if (output[i] > mx) mx = output[i];
        sum += output[i];
        sum_sq += (double)output[i] * output[i];
    }
    int valid = out_T * ENCODER_OUT_DIM - nan_count;
    double mean = sum / valid;
    double std = sqrt(sum_sq / valid - mean * mean);

    printf("Range: [%.6f, %.6f]\n", mn, mx);
    printf("Mean: %.6f, Std: %.6f\n", mean, std);
    printf("NaN/Inf count: %d / %d\n", nan_count, out_T * ENCODER_OUT_DIM);

    /* Print first few values */
    printf("\nFirst 10 values:\n");
    for (int i = 0; i < 10 && i < out_T * ENCODER_OUT_DIM; i++) {
        printf("  [%d] = %.6f\n", i, output[i]);
    }

    printf("\nInference time: %.1fms for %d frames\n", run_ms, test_T);
    printf("Per-frame: %.2fms\n", run_ms / test_T);

    /* Warmup and benchmark */
    printf("\n=== Benchmark (5 runs) ===\n");
    for (int r = 0; r < 5; r++) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        zipformer_encoder_run(enc, input, test_T, feat_dim, output, &out_T);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        printf("  Run %d: %.1fms\n", r, time_diff_ms(&t0, &t1));
    }

    free(input);
    free(output);
    zipformer_encoder_destroy(enc);
    printf("\nDone.\n");
    return 0;
}
#endif /* ZIPFORMER_TEST_MAIN */
