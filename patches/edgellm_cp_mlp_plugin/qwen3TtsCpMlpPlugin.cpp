#include "qwen3TtsCpMlpPlugin.h"
#include "kernels/qwen3TtsCpMlpKernels/qwen3TtsCpMlpKernels.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <mutex>

using namespace nvinfer1;

namespace trt_edgellm::plugins
{
namespace
{

constexpr char const* kPluginName{"Qwen3TtsCpMlpPlugin"};
constexpr char const* kPluginVersion{"1"};
constexpr int32_t kNbInputs{4};

int32_t lastDim(Dims const& dims)
{
    return dims.nbDims > 0 ? dims.d[dims.nbDims - 1] : -1;
}

bool supportedRank(Dims const& dims)
{
    return dims.nbDims == 2 || dims.nbDims == 3;
}

} // namespace

PluginFieldCollection Qwen3TtsCpMlpPluginCreator::mFieldCollection{};
std::vector<PluginField> Qwen3TtsCpMlpPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(Qwen3TtsCpMlpPluginCreator);

Qwen3TtsCpMlpPlugin::Qwen3TtsCpMlpPlugin(std::string const& name, int32_t hiddenSize, int32_t ffnSize)
    : mLayerName(name)
    , mHiddenSize(hiddenSize)
    , mFfnSize(ffnSize)
{
}

Qwen3TtsCpMlpPlugin::Qwen3TtsCpMlpPlugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name)
{
    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        std::string fieldName(fc->fields[i].name);
        if (fieldName == "hidden_size")
        {
            mHiddenSize = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "ffn_size")
        {
            mFfnSize = *static_cast<int32_t const*>(fc->fields[i].data);
        }
    }
}

Qwen3TtsCpMlpPlugin::~Qwen3TtsCpMlpPlugin()
{
    destroyResources();
}

void Qwen3TtsCpMlpPlugin::destroyResources() noexcept
{
    if (mGateHandle != nullptr)
    {
        cublasDestroy(mGateHandle);
        mGateHandle = nullptr;
    }
    if (mUpHandle != nullptr)
    {
        cublasDestroy(mUpHandle);
        mUpHandle = nullptr;
    }
    if (mDownHandle != nullptr)
    {
        cublasDestroy(mDownHandle);
        mDownHandle = nullptr;
    }
    if (mGateStream != nullptr)
    {
        cudaStreamDestroy(mGateStream);
        mGateStream = nullptr;
    }
    if (mUpStream != nullptr)
    {
        cudaStreamDestroy(mUpStream);
        mUpStream = nullptr;
    }
    if (mReadyEvent != nullptr)
    {
        cudaEventDestroy(mReadyEvent);
        mReadyEvent = nullptr;
    }
    if (mGateDoneEvent != nullptr)
    {
        cudaEventDestroy(mGateDoneEvent);
        mGateDoneEvent = nullptr;
    }
    if (mUpDoneEvent != nullptr)
    {
        cudaEventDestroy(mUpDoneEvent);
        mUpDoneEvent = nullptr;
    }
    mResourcesReady = false;
}

bool Qwen3TtsCpMlpPlugin::ensureResources() noexcept
{
    if (mResourcesReady)
    {
        return true;
    }
    if (cudaStreamCreateWithFlags(&mGateStream, cudaStreamNonBlocking) != cudaSuccess
        || cudaStreamCreateWithFlags(&mUpStream, cudaStreamNonBlocking) != cudaSuccess
        || cudaEventCreateWithFlags(&mReadyEvent, cudaEventDisableTiming) != cudaSuccess
        || cudaEventCreateWithFlags(&mGateDoneEvent, cudaEventDisableTiming) != cudaSuccess
        || cudaEventCreateWithFlags(&mUpDoneEvent, cudaEventDisableTiming) != cudaSuccess
        || cublasCreate(&mGateHandle) != CUBLAS_STATUS_SUCCESS || cublasCreate(&mUpHandle) != CUBLAS_STATUS_SUCCESS
        || cublasCreate(&mDownHandle) != CUBLAS_STATUS_SUCCESS)
    {
        destroyResources();
        return false;
    }
    cublasSetStream(mGateHandle, mGateStream);
    cublasSetStream(mUpHandle, mUpStream);
    mResourcesReady = true;
    return true;
}

IPluginCapability* Qwen3TtsCpMlpPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    if (type == PluginCapabilityType::kBUILD)
    {
        return static_cast<IPluginV3OneBuild*>(this);
    }
    if (type == PluginCapabilityType::kRUNTIME)
    {
        return static_cast<IPluginV3OneRuntime*>(this);
    }
    return static_cast<IPluginV3OneCore*>(this);
}

IPluginV3* Qwen3TtsCpMlpPlugin::clone() noexcept
{
    auto* plugin = new Qwen3TtsCpMlpPlugin(mLayerName, mHiddenSize, mFfnSize);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

char const* Qwen3TtsCpMlpPlugin::getPluginName() const noexcept
{
    return kPluginName;
}

char const* Qwen3TtsCpMlpPlugin::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

char const* Qwen3TtsCpMlpPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Qwen3TtsCpMlpPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

int32_t Qwen3TtsCpMlpPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t Qwen3TtsCpMlpPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    assert(nbOutputs == 1);
    assert(nbInputs == kNbInputs);
    outputTypes[0] = inputTypes[0];
    return 0;
}

int32_t Qwen3TtsCpMlpPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const*,
    int32_t, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    assert(nbInputs == kNbInputs);
    assert(nbOutputs == 1);
    outputs[0] = inputs[0];
    outputs[0].d[outputs[0].nbDims - 1] = exprBuilder.constant(mHiddenSize);
    return 0;
}

bool Qwen3TtsCpMlpPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    assert(nbInputs == kNbInputs && nbOutputs == 1);
    auto const& desc = inOut[pos].desc;
    if (desc.format != PluginFormat::kLINEAR || desc.type != DataType::kBF16)
    {
        return false;
    }
    if (pos == 0)
    {
        return supportedRank(desc.dims) && lastDim(desc.dims) == mHiddenSize;
    }
    if (pos == 1 || pos == 2)
    {
        return desc.dims.nbDims == 2 && desc.dims.d[0] == mHiddenSize && desc.dims.d[1] == mFfnSize;
    }
    if (pos == 3)
    {
        return desc.dims.nbDims == 2 && desc.dims.d[0] == mFfnSize && desc.dims.d[1] == mHiddenSize;
    }
    if (pos == 4)
    {
        return supportedRank(desc.dims) && lastDim(desc.dims) == mHiddenSize;
    }
    return false;
}

int32_t Qwen3TtsCpMlpPlugin::configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*,
    int32_t) noexcept
{
    return 0;
}

size_t Qwen3TtsCpMlpPlugin::getWorkspaceSize(
    DynamicPluginTensorDesc const* inputs, int32_t, DynamicPluginTensorDesc const*, int32_t) const noexcept
{
    int32_t m = 1;
    for (int32_t i = 0; i < inputs[0].desc.dims.nbDims - 1; ++i)
    {
        m *= std::max<int32_t>(1, static_cast<int32_t>(inputs[0].desc.dims.d[i]));
    }
    return static_cast<size_t>(m) * static_cast<size_t>(mFfnSize) * sizeof(__nv_bfloat16) * 3U;
}

int32_t Qwen3TtsCpMlpPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    if (!ensureResources() || workspace == nullptr)
    {
        return -1;
    }
    int32_t m = 1;
    for (int32_t i = 0; i < inputDesc[0].dims.nbDims - 1; ++i)
    {
        m *= inputDesc[0].dims.d[i];
    }

    auto const* x = static_cast<__nv_bfloat16 const*>(inputs[0]);
    auto const* wGate = static_cast<__nv_bfloat16 const*>(inputs[1]);
    auto const* wUp = static_cast<__nv_bfloat16 const*>(inputs[2]);
    auto const* wDown = static_cast<__nv_bfloat16 const*>(inputs[3]);
    auto* y = static_cast<__nv_bfloat16*>(outputs[0]);
    auto* gate = static_cast<__nv_bfloat16*>(workspace);
    auto* up = gate + static_cast<size_t>(m) * mFfnSize;
    auto* hidden = up + static_cast<size_t>(m) * mFfnSize;

    float alpha = 1.0F;
    float beta = 0.0F;
    cublasSetStream(mDownHandle, stream);

    auto status = cublasGemmEx(mDownHandle, CUBLAS_OP_N, CUBLAS_OP_N, mFfnSize, m, mHiddenSize, &alpha, wGate,
        CUDA_R_16BF, mFfnSize, x, CUDA_R_16BF, mHiddenSize, &beta, gate, CUDA_R_16BF, mFfnSize,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        return -1;
    }
    status = cublasGemmEx(mDownHandle, CUBLAS_OP_N, CUBLAS_OP_N, mFfnSize, m, mHiddenSize, &alpha, wUp, CUDA_R_16BF,
        mFfnSize, x, CUDA_R_16BF, mHiddenSize, &beta, up, CUDA_R_16BF, mFfnSize, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        return -1;
    }

    trt_edgellm::kernel::qwen3_tts_cp_silu_mul(gate, up, hidden, m * mFfnSize, stream);
    status = cublasGemmEx(mDownHandle, CUBLAS_OP_N, CUBLAS_OP_N, mHiddenSize, m, mFfnSize, &alpha, wDown,
        CUDA_R_16BF, mHiddenSize, hidden, CUDA_R_16BF, mFfnSize, &beta, y, CUDA_R_16BF, mHiddenSize,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    return status == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}

int32_t Qwen3TtsCpMlpPlugin::onShapeChange(PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) noexcept
{
    return 0;
}

IPluginV3* Qwen3TtsCpMlpPlugin::attachToContext(IPluginResourceContext*) noexcept
{
    return clone();
}

PluginFieldCollection const* Qwen3TtsCpMlpPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("hidden_size", &mHiddenSize, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("ffn_size", &mFfnSize, PluginFieldType::kINT32, 1);
    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

Qwen3TtsCpMlpPluginCreator::Qwen3TtsCpMlpPluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("hidden_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("ffn_size", nullptr, PluginFieldType::kINT32, 1));
    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* Qwen3TtsCpMlpPluginCreator::getPluginName() const noexcept
{
    return kPluginName;
}

char const* Qwen3TtsCpMlpPluginCreator::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

PluginFieldCollection const* Qwen3TtsCpMlpPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

char const* Qwen3TtsCpMlpPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Qwen3TtsCpMlpPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

IPluginV3* Qwen3TtsCpMlpPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase) noexcept
{
    auto* plugin = new Qwen3TtsCpMlpPlugin(std::string(name), fc);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

} // namespace trt_edgellm::plugins
