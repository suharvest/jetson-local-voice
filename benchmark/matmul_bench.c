/**
 * RKNN MatMul API Benchmark — batch M>1 on RK3576 NPU
 *
 * Tests rknn_matmul_api with various M/K/N shapes to verify
 * that throughput scales with M (needed for ASR encoder).
 *
 * Build:
 *   gcc -O2 -march=armv8-a+fp16 -I/home/cat \
 *       -o matmul_bench matmul_bench.c \
 *       -L/home/cat/sherpa-onnx/build/lib -lrknnrt -lm
 *
 * Run:
 *   LD_LIBRARY_PATH=/home/cat/sherpa-onnx/build/lib ./matmul_bench
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#include "rknn_api.h"
#include "rknn_matmul_api.h"

#define WARMUP_RUNS  5
#define BENCH_RUNS   100

/* ─── FP16 helpers ─── */

static uint16_t fp32_to_fp16_val(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint16_t sign = (x >> 16) & 0x8000;
    int exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7C00;
    return sign | (exp << 10) | (mant >> 13);
}

/* ─── Timing ─── */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ─── Test shape definition ─── */

typedef struct {
    int M, K, N;
    const char* label;
} Shape;

static Shape shapes[] = {
    /* Zipformer-relevant shapes */
    {  16, 256, 256, "attn_proj_small"},
    {  16, 256, 768, "ffn_up_medium"  },
    {  16, 768, 256, "ffn_down_medium"},
    {  16, 256, 192, "q_proj"         },
    /* Scaling test */
    {   1, 512, 512, "scale_M1"       },
    {  16, 512, 512, "scale_M16"      },
    {  64, 512, 512, "scale_M64"      },
    { 128, 512, 512, "scale_M128"     },
    { 256, 512, 512, "scale_M256"     },
};
#define N_SHAPES (sizeof(shapes)/sizeof(shapes[0]))

/* ─── Run one benchmark ─── */

static int bench_one(int M, int K, int N, int mode_w8a16,
                     double* out_latency_ms, double* out_gflops) {
    int ret;
    rknn_matmul_info info;
    rknn_matmul_io_attr io_attr;
    rknn_matmul_ctx ctx = 0;

    memset(&info, 0, sizeof(info));
    info.M = M;
    info.K = K;
    info.N = N;

    if (mode_w8a16) {
        info.type = RKNN_FLOAT16_MM_INT8_TO_FLOAT32;
        info.B_layout = 1;
        info.B_quant_type = 0;
    } else {
        info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16;
        info.B_layout = 0;
    }

    memset(&io_attr, 0, sizeof(io_attr));
    ret = rknn_matmul_create(&ctx, &info, &io_attr);
    if (ret != 0) {
        fprintf(stderr, "  matmul_create failed: M=%d K=%d N=%d mode=%s ret=%d\n",
                M, K, N, mode_w8a16 ? "W8A16" : "FP16", ret);
        return -1;
    }

    /* Allocate I/O memory */
    rknn_tensor_mem* mem_a = rknn_create_mem(ctx, io_attr.A.size);
    rknn_tensor_mem* mem_b = rknn_create_mem(ctx, io_attr.B.size);
    rknn_tensor_mem* mem_c = rknn_create_mem(ctx, io_attr.C.size);

    if (!mem_a || !mem_b || !mem_c) {
        fprintf(stderr, "  mem alloc failed: M=%d K=%d N=%d\n", M, K, N);
        if (mem_a) rknn_destroy_mem(ctx, mem_a);
        if (mem_b) rknn_destroy_mem(ctx, mem_b);
        if (mem_c) rknn_destroy_mem(ctx, mem_c);
        rknn_matmul_destroy(ctx);
        return -1;
    }

    /* Fill A with random FP16 data */
    {
        uint16_t* a_ptr = (uint16_t*)mem_a->virt_addr;
        int a_count = M * K;
        for (int i = 0; i < a_count; i++) {
            float v = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
            a_ptr[i] = fp32_to_fp16_val(v);
        }
    }

    /* Fill B */
    if (mode_w8a16) {
        /* INT8 weights */
        int8_t* b_ptr = (int8_t*)mem_b->virt_addr;
        int b_count = K * N;
        for (int i = 0; i < b_count; i++) {
            b_ptr[i] = (int8_t)((rand() % 256) - 128);
        }
    } else {
        /* FP16 weights */
        uint16_t* b_ptr = (uint16_t*)mem_b->virt_addr;
        int b_count = K * N;
        for (int i = 0; i < b_count; i++) {
            float v = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
            b_ptr[i] = fp32_to_fp16_val(v);
        }
    }

    /* Bind I/O */
    rknn_matmul_set_io_mem(ctx, mem_a, &io_attr.A);
    rknn_matmul_set_io_mem(ctx, mem_b, &io_attr.B);
    rknn_matmul_set_io_mem(ctx, mem_c, &io_attr.C);

    /* Warmup */
    for (int i = 0; i < WARMUP_RUNS; i++) {
        rknn_matmul_run(ctx);
    }

    /* Benchmark */
    double t0 = now_ms();
    for (int i = 0; i < BENCH_RUNS; i++) {
        rknn_matmul_run(ctx);
    }
    double t1 = now_ms();

    double avg_ms = (t1 - t0) / BENCH_RUNS;
    /* FLOPS = 2*M*K*N per matmul (multiply-add) */
    double flops = 2.0 * M * K * N;
    double gflops = (flops / (avg_ms / 1000.0)) / 1e9;

    *out_latency_ms = avg_ms;
    *out_gflops = gflops;

    /* Cleanup */
    rknn_destroy_mem(ctx, mem_a);
    rknn_destroy_mem(ctx, mem_b);
    rknn_destroy_mem(ctx, mem_c);
    rknn_matmul_destroy(ctx);

    return 0;
}

int main(void) {
    srand(42);

    printf("RKNN MatMul Benchmark — RK3576 NPU\n");
    printf("Warmup: %d runs, Benchmark: %d runs\n\n", WARMUP_RUNS, BENCH_RUNS);

    printf("%-6s %-6s %-6s %-8s %-14s %-10s\n",
           "M", "K", "N", "Mode", "Latency(ms)", "GFLOPS");
    printf("--------------------------------------------------------------\n");

    for (int s = 0; s < (int)N_SHAPES; s++) {
        Shape* sh = &shapes[s];

        /* FP16 mode */
        {
            double lat, gf;
            int ok = bench_one(sh->M, sh->K, sh->N, 0, &lat, &gf);
            if (ok == 0) {
                printf("%-6d %-6d %-6d %-8s %-14.3f %-10.2f  [%s]\n",
                       sh->M, sh->K, sh->N, "FP16", lat, gf, sh->label);
            } else {
                printf("%-6d %-6d %-6d %-8s %-14s %-10s  [%s]\n",
                       sh->M, sh->K, sh->N, "FP16", "FAILED", "-", sh->label);
            }
        }

        /* W8A16 mode */
        {
            double lat, gf;
            int ok = bench_one(sh->M, sh->K, sh->N, 1, &lat, &gf);
            if (ok == 0) {
                printf("%-6d %-6d %-6d %-8s %-14.3f %-10.2f  [%s]\n",
                       sh->M, sh->K, sh->N, "W8A16", lat, gf, sh->label);
            } else {
                printf("%-6d %-6d %-6d %-8s %-14s %-10s  [%s]\n",
                       sh->M, sh->K, sh->N, "W8A16", "FAILED", "-", sh->label);
            }
        }

        printf("\n");
    }

    return 0;
}
