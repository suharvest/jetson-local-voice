#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(expr)                                                                            \
    do                                                                                              \
    {                                                                                               \
        cudaError_t _err = (expr);                                                                  \
        if (_err != cudaSuccess)                                                                    \
        {                                                                                           \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
            std::exit(1);                                                                           \
        }                                                                                           \
    } while (0)

#define CUBLAS_CHECK(expr)                                                                          \
    do                                                                                              \
    {                                                                                               \
        cublasStatus_t _err = (expr);                                                               \
        if (_err != CUBLAS_STATUS_SUCCESS)                                                          \
        {                                                                                           \
            std::fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, int(_err));        \
            std::exit(1);                                                                           \
        }                                                                                           \
    } while (0)

template <int TILE_N>
__global__ void bf16_gemv_kn_kernel(
    __nv_bfloat16 const* __restrict__ x,
    __nv_bfloat16 const* __restrict__ w_kn,
    __nv_bfloat16* __restrict__ y,
    int K,
    int N)
{
    int const n = blockIdx.x * TILE_N + threadIdx.x;
    if (threadIdx.x >= TILE_N || n >= N)
    {
        return;
    }
    float acc = 0.0f;
#pragma unroll 4
    for (int k = 0; k < K; ++k)
    {
        acc += __bfloat162float(x[k]) * __bfloat162float(w_kn[k * N + n]);
    }
    y[n] = __float2bfloat16_rn(acc);
}

template <int WarpsPerBlock>
__global__ void bf16_gemv_warp_nk_kernel(
    __nv_bfloat16 const* __restrict__ x,
    __nv_bfloat16 const* __restrict__ w_nk,
    __nv_bfloat16* __restrict__ y,
    int K,
    int N)
{
    int const warp = threadIdx.x / 32;
    int const lane = threadIdx.x % 32;
    int const n = blockIdx.x * WarpsPerBlock + warp;
    if (n >= N)
    {
        return;
    }

    float acc = 0.0f;
    __nv_bfloat16 const* w = w_nk + static_cast<size_t>(n) * K;
    for (int k = lane; k < K; k += 32)
    {
        acc += __bfloat162float(x[k]) * __bfloat162float(w[k]);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        acc += __shfl_down_sync(0xffffffffU, acc, offset);
    }
    if (lane == 0)
    {
        y[n] = __float2bfloat16_rn(acc);
    }
}

template <int TILE_N>
float bench_custom(__nv_bfloat16 const* x, __nv_bfloat16 const* w, __nv_bfloat16* y, int K, int N, int iters)
{
    dim3 block(TILE_N);
    dim3 grid((N + TILE_N - 1) / TILE_N);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < 20; ++i)
    {
        bf16_gemv_kn_kernel<TILE_N><<<grid, block>>>(x, w, y, K, N);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i)
    {
        bf16_gemv_kn_kernel<TILE_N><<<grid, block>>>(x, w, y, K, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / static_cast<float>(iters);
}

template <int WarpsPerBlock>
float bench_warp_nk(__nv_bfloat16 const* x, __nv_bfloat16 const* w_nk, __nv_bfloat16* y, int K, int N, int iters)
{
    dim3 block(WarpsPerBlock * 32);
    dim3 grid((N + WarpsPerBlock - 1) / WarpsPerBlock);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < 20; ++i)
    {
        bf16_gemv_warp_nk_kernel<WarpsPerBlock><<<grid, block>>>(x, w_nk, y, K, N);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i)
    {
        bf16_gemv_warp_nk_kernel<WarpsPerBlock><<<grid, block>>>(x, w_nk, y, K, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / static_cast<float>(iters);
}

float bench_cublas(cublasHandle_t handle, __nv_bfloat16 const* x, __nv_bfloat16 const* w_col,
    __nv_bfloat16* y, int K, int N, int iters)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < 20; ++i)
    {
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, 1, K, &alpha, w_col, CUDA_R_16BF, K,
            x, CUDA_R_16BF, K, &beta, y, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i)
    {
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, 1, K, &alpha, w_col, CUDA_R_16BF, K,
            x, CUDA_R_16BF, K, &beta, y, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / static_cast<float>(iters);
}

__global__ void silu_mul_kernel(
    __nv_bfloat16 const* __restrict__ gate,
    __nv_bfloat16 const* __restrict__ up,
    __nv_bfloat16* __restrict__ out,
    int N)
{
    for (int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); i < N;
         i += static_cast<int>(blockDim.x * gridDim.x))
    {
        float g = __bfloat162float(gate[i]);
        float u = __bfloat162float(up[i]);
        float sig = 1.0f / (1.0f + expf(-g));
        out[i] = __float2bfloat16_rn((g * sig) * u);
    }
}

float bench_cp_mlp_block(cublasHandle_t gateHandle, cublasHandle_t upHandle, cublasHandle_t downHandle,
    cudaStream_t gateStream, cudaStream_t upStream, cudaStream_t mainStream, __nv_bfloat16 const* x,
    __nv_bfloat16 const* wGate, __nv_bfloat16 const* wUp, __nv_bfloat16 const* wDown,
    __nv_bfloat16* gate, __nv_bfloat16* up, __nv_bfloat16* hidden, __nv_bfloat16* y, int iters)
{
    constexpr int K = 1024;
    constexpr int Ffn = 3072;
    constexpr int Out = 1024;
    float alpha = 1.0f;
    float beta = 0.0f;
    cudaEvent_t ready, gateDone, upDone, start, stop;
    CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&gateDone, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&upDone, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto run_once = [&]() {
        CUDA_CHECK(cudaStreamWaitEvent(gateStream, ready, 0));
        CUDA_CHECK(cudaStreamWaitEvent(upStream, ready, 0));
        CUBLAS_CHECK(cublasGemmEx(gateHandle, CUBLAS_OP_T, CUBLAS_OP_N, Ffn, 1, K, &alpha, wGate, CUDA_R_16BF, K,
            x, CUDA_R_16BF, K, &beta, gate, CUDA_R_16BF, Ffn, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        CUBLAS_CHECK(cublasGemmEx(upHandle, CUBLAS_OP_T, CUBLAS_OP_N, Ffn, 1, K, &alpha, wUp, CUDA_R_16BF, K,
            x, CUDA_R_16BF, K, &beta, up, CUDA_R_16BF, Ffn, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        CUDA_CHECK(cudaEventRecord(gateDone, gateStream));
        CUDA_CHECK(cudaEventRecord(upDone, upStream));
        CUDA_CHECK(cudaStreamWaitEvent(mainStream, gateDone, 0));
        CUDA_CHECK(cudaStreamWaitEvent(mainStream, upDone, 0));
        silu_mul_kernel<<<12, 256, 0, mainStream>>>(gate, up, hidden, Ffn);
        CUBLAS_CHECK(cublasGemmEx(downHandle, CUBLAS_OP_T, CUBLAS_OP_N, Out, 1, Ffn, &alpha, wDown, CUDA_R_16BF, Ffn,
            hidden, CUDA_R_16BF, Ffn, &beta, y, CUDA_R_16BF, Out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        CUDA_CHECK(cudaEventRecord(ready, mainStream));
    };

    CUDA_CHECK(cudaEventRecord(ready, mainStream));
    for (int i = 0; i < 20; ++i)
    {
        run_once();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start, mainStream));
    CUDA_CHECK(cudaEventRecord(ready, mainStream));
    for (int i = 0; i < iters; ++i)
    {
        run_once();
    }
    CUDA_CHECK(cudaEventRecord(stop, mainStream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(ready));
    CUDA_CHECK(cudaEventDestroy(gateDone));
    CUDA_CHECK(cudaEventDestroy(upDone));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / static_cast<float>(iters);
}

void run_shape(cublasHandle_t handle, int K, int N, int iters)
{
    std::vector<__nv_bfloat16> hx(K);
    std::vector<__nv_bfloat16> hw_kn(static_cast<size_t>(K) * N);
    std::vector<__nv_bfloat16> hw_nk(static_cast<size_t>(K) * N);
    std::vector<__nv_bfloat16> hw_col(static_cast<size_t>(K) * N);
    for (int k = 0; k < K; ++k)
    {
        hx[k] = __float2bfloat16_rn(static_cast<float>((k % 17) - 8) * 0.01f);
        for (int n = 0; n < N; ++n)
        {
            float v = static_cast<float>(((k * 131 + n * 17) % 31) - 15) * 0.003f;
            hw_kn[static_cast<size_t>(k) * N + n] = __float2bfloat16_rn(v);
            hw_nk[static_cast<size_t>(n) * K + k] = __float2bfloat16_rn(v);
            hw_col[static_cast<size_t>(n) * K + k] = __float2bfloat16_rn(v);
        }
    }

    __nv_bfloat16 *dx, *dw_kn, *dw_nk, *dw_col, *dy;
    CUDA_CHECK(cudaMalloc(&dx, K * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dw_kn, static_cast<size_t>(K) * N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dw_nk, static_cast<size_t>(K) * N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dw_col, static_cast<size_t>(K) * N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dy, N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw_kn, hw_kn.data(), static_cast<size_t>(K) * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw_nk, hw_nk.data(), static_cast<size_t>(K) * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw_col, hw_col.data(), static_cast<size_t>(K) * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    float t32 = bench_custom<32>(dx, dw_kn, dy, K, N, iters);
    float t64 = bench_custom<64>(dx, dw_kn, dy, K, N, iters);
    float t128 = bench_custom<128>(dx, dw_kn, dy, K, N, iters);
    float t256 = bench_custom<256>(dx, dw_kn, dy, K, N, iters);
    float twarp4 = bench_warp_nk<4>(dx, dw_nk, dy, K, N, iters);
    float twarp8 = bench_warp_nk<8>(dx, dw_nk, dy, K, N, iters);
    float twarp16 = bench_warp_nk<16>(dx, dw_nk, dy, K, N, iters);
    float tcublas = bench_cublas(handle, dx, dw_col, dy, K, N, iters);

    std::printf("shape K=%d N=%d custom32=%.4f custom64=%.4f custom128=%.4f custom256=%.4f warp4=%.4f warp8=%.4f warp16=%.4f cublas=%.4f ms\n",
        K, N, t32, t64, t128, t256, twarp4, twarp8, twarp16, tcublas);

    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dw_kn));
    CUDA_CHECK(cudaFree(dw_nk));
    CUDA_CHECK(cudaFree(dw_col));
    CUDA_CHECK(cudaFree(dy));
}

void run_mlp_block(int iters)
{
    constexpr int Hidden = 1024;
    constexpr int Ffn = 3072;
    std::vector<__nv_bfloat16> hx(Hidden);
    std::vector<__nv_bfloat16> hGate(static_cast<size_t>(Hidden) * Ffn);
    std::vector<__nv_bfloat16> hUp(static_cast<size_t>(Hidden) * Ffn);
    std::vector<__nv_bfloat16> hDown(static_cast<size_t>(Ffn) * Hidden);
    for (int k = 0; k < Hidden; ++k)
    {
        hx[k] = __float2bfloat16_rn(static_cast<float>((k % 17) - 8) * 0.01f);
        for (int n = 0; n < Ffn; ++n)
        {
            float g = static_cast<float>(((k * 131 + n * 17) % 31) - 15) * 0.003f;
            float u = static_cast<float>(((k * 97 + n * 29) % 37) - 18) * 0.002f;
            hGate[static_cast<size_t>(n) * Hidden + k] = __float2bfloat16_rn(g);
            hUp[static_cast<size_t>(n) * Hidden + k] = __float2bfloat16_rn(u);
        }
    }
    for (int k = 0; k < Ffn; ++k)
    {
        for (int n = 0; n < Hidden; ++n)
        {
            float v = static_cast<float>(((k * 53 + n * 11) % 41) - 20) * 0.002f;
            hDown[static_cast<size_t>(n) * Ffn + k] = __float2bfloat16_rn(v);
        }
    }

    __nv_bfloat16 *dx, *dwGate, *dwUp, *dwDown, *dGate, *dUp, *dHidden, *dy;
    CUDA_CHECK(cudaMalloc(&dx, Hidden * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dwGate, static_cast<size_t>(Hidden) * Ffn * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dwUp, static_cast<size_t>(Hidden) * Ffn * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dwDown, static_cast<size_t>(Ffn) * Hidden * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dGate, Ffn * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dUp, Ffn * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dHidden, Ffn * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dy, Hidden * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), Hidden * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dwGate, hGate.data(), static_cast<size_t>(Hidden) * Ffn * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dwUp, hUp.data(), static_cast<size_t>(Hidden) * Ffn * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dwDown, hDown.data(), static_cast<size_t>(Ffn) * Hidden * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    cudaStream_t gateStream, upStream, mainStream;
    CUDA_CHECK(cudaStreamCreate(&gateStream));
    CUDA_CHECK(cudaStreamCreate(&upStream));
    CUDA_CHECK(cudaStreamCreate(&mainStream));
    cublasHandle_t gateHandle, upHandle, downHandle;
    CUBLAS_CHECK(cublasCreate(&gateHandle));
    CUBLAS_CHECK(cublasCreate(&upHandle));
    CUBLAS_CHECK(cublasCreate(&downHandle));
    CUBLAS_CHECK(cublasSetStream(gateHandle, gateStream));
    CUBLAS_CHECK(cublasSetStream(upHandle, upStream));
    CUBLAS_CHECK(cublasSetStream(downHandle, mainStream));
    CUBLAS_CHECK(cublasSetMathMode(gateHandle, CUBLAS_DEFAULT_MATH));
    CUBLAS_CHECK(cublasSetMathMode(upHandle, CUBLAS_DEFAULT_MATH));
    CUBLAS_CHECK(cublasSetMathMode(downHandle, CUBLAS_DEFAULT_MATH));

    float t = bench_cp_mlp_block(gateHandle, upHandle, downHandle, gateStream, upStream, mainStream, dx, dwGate,
        dwUp, dwDown, dGate, dUp, dHidden, dy, iters);
    std::printf("cp_mlp_block K=1024 F=3072 N=1024 cublas_parallel_gate_up=%.4f ms\n", t);

    CUBLAS_CHECK(cublasDestroy(gateHandle));
    CUBLAS_CHECK(cublasDestroy(upHandle));
    CUBLAS_CHECK(cublasDestroy(downHandle));
    CUDA_CHECK(cudaStreamDestroy(gateStream));
    CUDA_CHECK(cudaStreamDestroy(upStream));
    CUDA_CHECK(cudaStreamDestroy(mainStream));
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dwGate));
    CUDA_CHECK(cudaFree(dwUp));
    CUDA_CHECK(cudaFree(dwDown));
    CUDA_CHECK(cudaFree(dGate));
    CUDA_CHECK(cudaFree(dUp));
    CUDA_CHECK(cudaFree(dHidden));
    CUDA_CHECK(cudaFree(dy));
}

int main(int argc, char** argv)
{
    int iters = argc > 1 ? std::atoi(argv[1]) : 500;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    // Qwen3-TTS CP decode M=1 GEMM-equivalent shapes.
    run_shape(handle, 1024, 4096, iters); // fused QKV
    run_shape(handle, 1024, 3072, iters); // MLP gate/up
    run_shape(handle, 3072, 1024, iters); // MLP down
    run_shape(handle, 1024, 2048, iters); // lm_head
    run_mlp_block(iters);

    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}
