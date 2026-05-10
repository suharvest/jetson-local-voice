#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace trt_edgellm::kernel
{

void qwen3_tts_cp_silu_mul(
    __nv_bfloat16 const* gate, __nv_bfloat16 const* up, __nv_bfloat16* out, int elements, cudaStream_t stream) noexcept;

} // namespace trt_edgellm::kernel
