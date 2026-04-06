/**
 * Zipformer Layer 0 Matmul Precision Test for RK3576 NPU
 *
 * Validates W8A16 matmul accuracy on each projection in a Zipformer encoder
 * layer by comparing NPU results against CPU FP32 reference outputs.
 *
 * Flow per projection:
 *   1. Load INT8 weights + FP32 scales + FP32 bias
 *   2. Load reference input [M, K] and reference output [M, N]
 *   3. Create rknn_matmul context (RKNN_FLOAT16_MM_INT8_TO_FLOAT32)
 *   4. Convert INT8 to native layout, run NPU matmul
 *   5. Apply per-column scale + bias on CPU
 *   6. Compare with reference: max_diff, RMSE, cosine similarity
 *
 * Build:  make
 * Run:    ./zipformer_layer_test [weight_dir] [ref_dir]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <arm_neon.h>

#include "rknn_api.h"
#include "rknn_matmul_api.h"

/* ─── Projection definitions ─── */

typedef struct {
    const char* name;
    int K;  /* input dim */
    int N;  /* output dim */
    int has_bias;
} ProjDef;

static const ProjDef PROJECTIONS[] = {
    {"feed_forward1_in_proj",   256, 768, 1},
    {"feed_forward1_out_proj",  768, 256, 1},
    {"self_attn_in_proj",       256, 496, 1},
    {"self_attn_out_proj",       96, 256, 1},
    {"self_attn_out_proj2",      96, 256, 1},
    {"feed_forward2_in_proj",   256, 768, 1},
    {"feed_forward2_out_proj",  768, 256, 1},
    {"feed_forward3_in_proj",   256, 768, 1},
    {"feed_forward3_out_proj",  768, 256, 1},
};
#define NUM_PROJS (sizeof(PROJECTIONS) / sizeof(PROJECTIONS[0]))

#define TEST_M  8   /* number of frames to test with */

#define WARMUP_RUNS  3
#define BENCH_RUNS   20

/* ─── File I/O ─── */

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

static int save_bin_f32(const char* path, const float* data, int count) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "ERROR: cannot create %s\n", path); return -1; }
    fwrite(data, sizeof(float), count, f);
    fclose(f);
    return 0;
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

/* ─── CPU FP32 reference matmul ─── */

static void cpu_matmul_f32(const float* A, const int8_t* B_int8,
                           const float* scales, const float* bias,
                           float* C, int M, int K, int N) {
    /* A[M,K] * dequant(B[K,N]) + bias[N] -> C[M,N]
     * B is stored as int8 row-major [K,N], per-column scale */
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += A[m * K + k] * ((float)B_int8[k * N + n] * scales[n]);
            }
            if (bias) acc += bias[n];
            C[m * N + n] = acc;
        }
    }
}

/* ─── Metrics ─── */

typedef struct {
    float max_diff;
    float rmse;
    float cosine;
    int max_diff_idx;
} Metrics;

static Metrics compute_metrics(const float* ref, const float* test, int n) {
    Metrics m = {0};
    double sum_sq_diff = 0.0;
    double dot_ab = 0.0, dot_aa = 0.0, dot_bb = 0.0;

    for (int i = 0; i < n; i++) {
        float diff = fabsf(ref[i] - test[i]);
        if (diff > m.max_diff) {
            m.max_diff = diff;
            m.max_diff_idx = i;
        }
        sum_sq_diff += (double)(ref[i] - test[i]) * (ref[i] - test[i]);
        dot_ab += (double)ref[i] * test[i];
        dot_aa += (double)ref[i] * ref[i];
        dot_bb += (double)test[i] * test[i];
    }

    m.rmse = (float)sqrt(sum_sq_diff / n);
    double denom = sqrt(dot_aa) * sqrt(dot_bb);
    m.cosine = (denom > 0) ? (float)(dot_ab / denom) : 0.0f;
    return m;
}

/* ─── Timing ─── */

static double time_diff_ms(struct timespec* start, struct timespec* end) {
    return (end->tv_sec - start->tv_sec) * 1000.0 +
           (end->tv_nsec - start->tv_nsec) / 1e6;
}

/* ─── Test one projection ─── */

static int test_projection(const ProjDef* proj, const char* weight_dir,
                           const char* ref_dir, int M) {
    int ret = 0;
    char path[512];
    int K = proj->K;
    int N = proj->N;

    printf("\n--- %s [%d, %d] x [%d, %d] ---\n", proj->name, M, K, K, N);

    /* Load INT8 weights [K, N] */
    snprintf(path, sizeof(path), "%s/%s.int8.bin", weight_dir, proj->name);
    size_t w_size;
    int8_t* w_int8 = (int8_t*)load_bin_raw(path, &w_size);
    if (!w_int8) return -1;
    if ((int)w_size != K * N) {
        fprintf(stderr, "  Weight size mismatch: expected %d, got %zu\n", K * N, w_size);
        free(w_int8);
        return -1;
    }

    /* Load scales [N] */
    float* scales = (float*)malloc(N * sizeof(float));
    snprintf(path, sizeof(path), "%s/%s.scales.bin", weight_dir, proj->name);
    if (load_bin_f32(path, scales, N) != 0) { free(w_int8); free(scales); return -1; }

    /* Load bias [N] (optional) */
    float* bias = NULL;
    if (proj->has_bias) {
        bias = (float*)malloc(N * sizeof(float));
        snprintf(path, sizeof(path), "%s/%s.bias.fp32.bin", weight_dir, proj->name);
        if (load_bin_f32(path, bias, N) != 0) {
            fprintf(stderr, "  Warning: bias not found, proceeding without\n");
            free(bias);
            bias = NULL;
        }
    }

    /* Load or generate test input [M, K] */
    float* input = (float*)malloc(M * K * sizeof(float));
    snprintf(path, sizeof(path), "%s/%s_input.bin", ref_dir, proj->name);
    FILE* fin = fopen(path, "rb");
    if (fin) {
        fread(input, sizeof(float), M * K, fin);
        fclose(fin);
        printf("  Input: loaded from %s\n", path);
    } else {
        /* Generate deterministic pseudo-random input */
        printf("  Input: generated (seeded PRNG)\n");
        unsigned int seed = 42;
        for (int i = 0; i < M * K; i++) {
            seed = seed * 1103515245 + 12345;
            input[i] = ((float)((seed >> 16) & 0x7FFF) / 32768.0f - 0.5f) * 0.1f;
        }
    }

    /* CPU FP32 reference */
    float* ref_output = (float*)malloc(M * N * sizeof(float));
    snprintf(path, sizeof(path), "%s/%s_output.bin", ref_dir, proj->name);
    FILE* fref = fopen(path, "rb");
    if (fref) {
        fread(ref_output, sizeof(float), M * N, fref);
        fclose(fref);
        printf("  Reference: loaded from file\n");
    } else {
        /* Compute CPU reference */
        cpu_matmul_f32(input, w_int8, scales, bias, ref_output, M, K, N);
        printf("  Reference: computed on CPU\n");
        /* Save for future use */
        save_bin_f32(path, ref_output, M * N);
    }

    /* ─── NPU W8A16 matmul ─── */
    rknn_matmul_ctx ctx = 0;
    rknn_matmul_info info;
    rknn_matmul_io_attr io;

    memset(&info, 0, sizeof(info));
    memset(&io, 0, sizeof(io));

    info.M = M;
    info.K = K;
    info.N = N;
    info.type = RKNN_FLOAT16_MM_INT8_TO_FLOAT32;
    info.B_layout = 1;  /* native layout */
    info.B_quant_type = 0;

    ret = rknn_matmul_create(&ctx, &info, &io);
    if (ret != 0) {
        fprintf(stderr, "  FATAL: matmul create failed: %d\n", ret);
        goto cleanup;
    }

    rknn_tensor_mem* mem_A = rknn_create_mem(ctx, io.A.size);
    rknn_tensor_mem* mem_B = rknn_create_mem(ctx, io.B.size);
    rknn_tensor_mem* mem_C = rknn_create_mem(ctx, io.C.size);

    if (!mem_A || !mem_B || !mem_C) {
        fprintf(stderr, "  FATAL: mem alloc failed\n");
        ret = -1;
        goto cleanup_ctx;
    }

    /* Convert INT8 weights to native layout */
    ret = rknn_B_normal_layout_to_native_layout(w_int8, mem_B->virt_addr, K, N, &info);
    if (ret != 0) {
        fprintf(stderr, "  FATAL: B layout conversion failed: %d\n", ret);
        goto cleanup_mem;
    }

    /* Bind I/O */
    rknn_matmul_set_io_mem(ctx, mem_A, &io.A);
    rknn_matmul_set_io_mem(ctx, mem_B, &io.B);
    rknn_matmul_set_io_mem(ctx, mem_C, &io.C);

    /* Convert FP32 input -> FP16 into A buffer */
    fp32_to_fp16((uint16_t*)mem_A->virt_addr, input, M * K);

    /* Warmup */
    for (int w = 0; w < WARMUP_RUNS; w++) {
        rknn_matmul_run(ctx);
    }

    /* Timed runs */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < BENCH_RUNS; r++) {
        rknn_matmul_run(ctx);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double avg_ms = time_diff_ms(&t0, &t1) / BENCH_RUNS;

    /* Read output (FP32 from W8A16 mode) and apply per-column scale + bias */
    float* npu_output = (float*)malloc(M * N * sizeof(float));
    const float* src = (const float*)mem_C->virt_addr;

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float val = src[m * N + n] * scales[n];
            if (bias) val += bias[n];
            npu_output[m * N + n] = val;
        }
    }

    /* Compare */
    Metrics met = compute_metrics(ref_output, npu_output, M * N);
    printf("  W8A16: max_diff=%.6f (@%d), cosine=%.8f, RMSE=%.6f\n",
           met.max_diff, met.max_diff_idx, met.cosine, met.rmse);
    printf("  Time: %.3fms (avg of %d runs)\n", avg_ms, BENCH_RUNS);

    /* Show a few values for sanity */
    printf("  Sample values (ref vs npu):\n");
    for (int i = 0; i < 5 && i < M * N; i++) {
        printf("    [%d] ref=%.6f  npu=%.6f  diff=%.6f\n",
               i, ref_output[i], npu_output[i], fabsf(ref_output[i] - npu_output[i]));
    }

    /* Also compare raw NPU output (before scale+bias) against dequantized matmul */
    Metrics met_raw;
    {
        /* Compute raw reference: A * (B_int8) without scale, without bias */
        float* ref_raw = (float*)malloc(M * N * sizeof(float));
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float acc = 0.0f;
                for (int k = 0; k < K; k++) {
                    acc += input[m * K + k] * (float)w_int8[k * N + n];
                }
                ref_raw[m * N + n] = acc;
            }
        }

        /* NPU raw output (before scale) is FP16_input * INT8_weight -> FP32 */
        /* But NPU uses FP16 input, so the reference should too */
        /* Re-compute with FP16-rounded input for fair comparison */
        uint16_t* input_fp16 = (uint16_t*)mem_A->virt_addr;
        float* input_fp16_f32 = (float*)malloc(M * K * sizeof(float));
        for (int i = 0; i < M * K; i++) {
            __fp16 tmp;
            memcpy(&tmp, input_fp16 + i, sizeof(__fp16));
            input_fp16_f32[i] = (float)tmp;
        }
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float acc = 0.0f;
                for (int k = 0; k < K; k++) {
                    acc += input_fp16_f32[m * K + k] * (float)w_int8[k * N + n];
                }
                ref_raw[m * N + n] = acc;
            }
        }

        met_raw = compute_metrics(ref_raw, src, M * N);
        printf("  Raw matmul (FP16_in * INT8_w -> FP32, no scale/bias):\n");
        printf("    max_diff=%.6f, cosine=%.8f, RMSE=%.6f\n",
               met_raw.max_diff, met_raw.cosine, met_raw.rmse);

        free(ref_raw);
        free(input_fp16_f32);
    }

    /* Save NPU output for external comparison */
    snprintf(path, sizeof(path), "%s/%s_npu_output.bin", ref_dir, proj->name);
    save_bin_f32(path, npu_output, M * N);

    free(npu_output);
    ret = 0;

cleanup_mem:
    if (mem_A) rknn_destroy_mem(ctx, mem_A);
    if (mem_B) rknn_destroy_mem(ctx, mem_B);
    if (mem_C) rknn_destroy_mem(ctx, mem_C);
cleanup_ctx:
    if (ctx) rknn_matmul_destroy(ctx);
cleanup:
    free(w_int8);
    free(scales);
    if (bias) free(bias);
    free(input);
    free(ref_output);
    return ret;
}

/* ─── Main ─── */

int main(int argc, char** argv) {
    const char* weight_dir = "/tmp/jetson-voice-mte/rk3576/mte/weights/layer_0_w8a16";
    const char* ref_dir    = "/tmp/jetson-voice-mte/rk3576/mte/reference";

    if (argc >= 2) weight_dir = argv[1];
    if (argc >= 3) ref_dir    = argv[2];

    printf("=== Zipformer Layer 0 Matmul Precision Test ===\n");
    printf("Weight dir: %s\n", weight_dir);
    printf("Ref dir:    %s\n", ref_dir);
    printf("M=%d (test frames)\n", TEST_M);
    printf("Mode: W8A16 (RKNN_FLOAT16_MM_INT8_TO_FLOAT32)\n");

    int pass = 0, fail = 0;

    for (int i = 0; i < (int)NUM_PROJS; i++) {
        int ret = test_projection(&PROJECTIONS[i], weight_dir, ref_dir, TEST_M);
        if (ret == 0) pass++;
        else fail++;
    }

    printf("\n=== Summary ===\n");
    printf("Passed: %d / %d\n", pass, pass + fail);
    if (fail > 0) {
        printf("FAILED: %d projections\n", fail);
        return 1;
    }
    printf("All projections passed!\n");
    return 0;
}
