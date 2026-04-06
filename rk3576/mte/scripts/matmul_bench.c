/**
 * rknn_matmul_api Benchmark for RK3576
 *
 * Tests FP16xFP16->FP16 and FP16xINT8->FP32 (W8A16) modes
 * at various M values to measure batch matmul throughput.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "rknn_api.h"
#include "rknn_matmul_api.h"

#define WARMUP_RUNS  5
#define BENCH_RUNS   100

/* FP32 -> FP16 conversion (IEEE 754) */
static uint16_t f32_to_f16(float v) {
    union { float f; uint32_t u; } x;
    x.f = v;
    uint32_t s = (x.u >> 16) & 0x8000;
    int e = ((x.u >> 23) & 0xFF) - 127 + 15;
    uint32_t m = x.u & 0x7FFFFF;
    if (e <= 0) return s;
    if (e >= 31) return s | 0x7C00;
    return s | (e << 10) | (m >> 13);
}

typedef struct {
    int M, K, N;
    const char* label;
} Shape;

static Shape shapes[] = {
    {   1, 512, 512, "baseline"         },
    {  16, 512, 512, NULL               },
    {  64, 512, 512, NULL               },
    { 128, 512, 512, NULL               },
    { 256, 512, 512, NULL               },
    {  16, 256, 256, "Zipformer attn"   },
    {  16, 256, 768, "Zipformer FFN"    },
};
#define NUM_SHAPES (sizeof(shapes) / sizeof(shapes[0]))

static double time_diff_ms(struct timespec *start, struct timespec *end) {
    return (end->tv_sec - start->tv_sec) * 1000.0 +
           (end->tv_nsec - start->tv_nsec) / 1e6;
}

static int bench_fp16(Shape *s) {
    int ret;
    rknn_matmul_ctx ctx = 0;
    rknn_matmul_info info;
    rknn_matmul_io_attr io;

    memset(&info, 0, sizeof(info));
    memset(&io, 0, sizeof(io));

    info.M = s->M;
    info.K = s->K;
    info.N = s->N;
    info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16;
    info.B_layout = 0;

    ret = rknn_matmul_create(&ctx, &info, &io);
    if (ret != 0) {
        fprintf(stderr, "  FP16 create failed M=%d K=%d N=%d: %d\n", s->M, s->K, s->N, ret);
        return -1;
    }

    rknn_tensor_mem *mem_a = rknn_create_mem(ctx, io.A.size);
    rknn_tensor_mem *mem_b = rknn_create_mem(ctx, io.B.size);
    rknn_tensor_mem *mem_c = rknn_create_mem(ctx, io.C.size);

    if (!mem_a || !mem_b || !mem_c) {
        fprintf(stderr, "  FP16 alloc failed\n");
        rknn_matmul_destroy(ctx);
        return -1;
    }

    /* Fill A and B with random FP16 data */
    srand(42);
    uint16_t *a_ptr = (uint16_t *)mem_a->virt_addr;
    uint16_t *b_ptr = (uint16_t *)mem_b->virt_addr;
    for (int i = 0; i < s->M * s->K; i++)
        a_ptr[i] = f32_to_f16(((float)rand() / RAND_MAX - 0.5f) * 2.0f);
    for (int i = 0; i < s->K * s->N; i++)
        b_ptr[i] = f32_to_f16(((float)rand() / RAND_MAX - 0.5f) * 2.0f);

    rknn_matmul_set_io_mem(ctx, mem_a, &io.A);
    rknn_matmul_set_io_mem(ctx, mem_b, &io.B);
    rknn_matmul_set_io_mem(ctx, mem_c, &io.C);

    /* Warmup */
    for (int i = 0; i < WARMUP_RUNS; i++)
        rknn_matmul_run(ctx);

    /* Benchmark */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < BENCH_RUNS; i++)
        rknn_matmul_run(ctx);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double avg_ms = time_diff_ms(&t0, &t1) / BENCH_RUNS;
    double gflops = (2.0 * s->M * s->K * s->N) / (avg_ms * 1e6);

    if (s->label)
        printf("  M=%-4d K=%-4d N=%-4d: avg=%.3fms, GFLOPS=%.2f  (%s)\n",
               s->M, s->K, s->N, avg_ms, gflops, s->label);
    else
        printf("  M=%-4d K=%-4d N=%-4d: avg=%.3fms, GFLOPS=%.2f\n",
               s->M, s->K, s->N, avg_ms, gflops);

    rknn_destroy_mem(ctx, mem_a);
    rknn_destroy_mem(ctx, mem_b);
    rknn_destroy_mem(ctx, mem_c);
    rknn_matmul_destroy(ctx);
    return 0;
}

static int bench_w8a16(Shape *s) {
    int ret;
    rknn_matmul_ctx ctx = 0;
    rknn_matmul_info info;
    rknn_matmul_io_attr io;

    memset(&info, 0, sizeof(info));
    memset(&io, 0, sizeof(io));

    info.M = s->M;
    info.K = s->K;
    info.N = s->N;
    info.type = RKNN_FLOAT16_MM_INT8_TO_FLOAT32;
    info.B_layout = 1;  /* native layout */
    info.B_quant_type = 0;  /* per-layer */

    ret = rknn_matmul_create(&ctx, &info, &io);
    if (ret != 0) {
        fprintf(stderr, "  W8A16 create failed M=%d K=%d N=%d: %d\n", s->M, s->K, s->N, ret);
        return -1;
    }

    rknn_tensor_mem *mem_a = rknn_create_mem(ctx, io.A.size);
    rknn_tensor_mem *mem_b = rknn_create_mem(ctx, io.B.size);
    rknn_tensor_mem *mem_c = rknn_create_mem(ctx, io.C.size);

    if (!mem_a || !mem_b || !mem_c) {
        fprintf(stderr, "  W8A16 alloc failed\n");
        rknn_matmul_destroy(ctx);
        return -1;
    }

    /* Fill A with random FP16 */
    srand(42);
    uint16_t *a_ptr = (uint16_t *)mem_a->virt_addr;
    for (int i = 0; i < s->M * s->K; i++)
        a_ptr[i] = f32_to_f16(((float)rand() / RAND_MAX - 0.5f) * 2.0f);

    /* Fill B with random INT8 in normal layout, then convert to native */
    int8_t *b_normal = (int8_t *)malloc(s->K * s->N);
    if (!b_normal) {
        fprintf(stderr, "  W8A16 malloc b_normal failed\n");
        rknn_matmul_destroy(ctx);
        return -1;
    }
    for (int i = 0; i < s->K * s->N; i++)
        b_normal[i] = (int8_t)(rand() % 256 - 128);

    ret = rknn_B_normal_layout_to_native_layout(b_normal, mem_b->virt_addr,
                                                 info.K, info.N, &info);
    free(b_normal);
    if (ret != 0) {
        fprintf(stderr, "  W8A16 B layout convert failed: %d\n", ret);
        rknn_destroy_mem(ctx, mem_a);
        rknn_destroy_mem(ctx, mem_b);
        rknn_destroy_mem(ctx, mem_c);
        rknn_matmul_destroy(ctx);
        return -1;
    }

    rknn_matmul_set_io_mem(ctx, mem_a, &io.A);
    rknn_matmul_set_io_mem(ctx, mem_b, &io.B);
    rknn_matmul_set_io_mem(ctx, mem_c, &io.C);

    /* Warmup */
    for (int i = 0; i < WARMUP_RUNS; i++)
        rknn_matmul_run(ctx);

    /* Benchmark */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < BENCH_RUNS; i++)
        rknn_matmul_run(ctx);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double avg_ms = time_diff_ms(&t0, &t1) / BENCH_RUNS;
    double gflops = (2.0 * s->M * s->K * s->N) / (avg_ms * 1e6);

    if (s->label)
        printf("  M=%-4d K=%-4d N=%-4d: avg=%.3fms, GFLOPS=%.2f  (%s)\n",
               s->M, s->K, s->N, avg_ms, gflops, s->label);
    else
        printf("  M=%-4d K=%-4d N=%-4d: avg=%.3fms, GFLOPS=%.2f\n",
               s->M, s->K, s->N, avg_ms, gflops);

    rknn_destroy_mem(ctx, mem_a);
    rknn_destroy_mem(ctx, mem_b);
    rknn_destroy_mem(ctx, mem_c);
    rknn_matmul_destroy(ctx);
    return 0;
}

int main(void) {
    printf("=== rknn_matmul_api Benchmark ===\n\n");

    printf("Mode: FP16xFP16->FP16\n");
    for (int i = 0; i < (int)NUM_SHAPES; i++)
        bench_fp16(&shapes[i]);

    printf("\nMode: FP16xINT8->FP32 (W8A16)\n");
    for (int i = 0; i < (int)NUM_SHAPES; i++)
        bench_w8a16(&shapes[i]);

    printf("\nDone.\n");
    return 0;
}
