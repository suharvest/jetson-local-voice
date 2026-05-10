#include "qwen3TtsCpMlpKernels.h"
#include "common/checkMacros.h"

namespace trt_edgellm::kernel
{
namespace
{

__global__ void siluMulKernel(
    __nv_bfloat16 const* __restrict__ gate,
    __nv_bfloat16 const* __restrict__ up,
    __nv_bfloat16* __restrict__ out,
    int elements)
{
    for (int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); i < elements;
         i += static_cast<int>(blockDim.x * gridDim.x))
    {
        float const g = __bfloat162float(gate[i]);
        float const u = __bfloat162float(up[i]);
        float const sigmoid = 1.0F / (1.0F + expf(-g));
        out[i] = __float2bfloat16_rn((g * sigmoid) * u);
    }
}

} // namespace

void qwen3_tts_cp_silu_mul(
    __nv_bfloat16 const* gate, __nv_bfloat16 const* up, __nv_bfloat16* out, int elements, cudaStream_t stream) noexcept
{
    if (gate == nullptr || up == nullptr || out == nullptr || elements <= 0)
    {
        return;
    }
    constexpr int kThreads = 256;
    int const blocks = std::min(16, (elements + kThreads - 1) / kThreads);
    siluMulKernel<<<blocks, kThreads, 0, stream>>>(gate, up, out, elements);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace trt_edgellm::kernel
