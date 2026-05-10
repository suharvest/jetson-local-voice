#pragma once

#include <NvInferRuntime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <string>
#include <vector>

namespace trt_edgellm::plugins
{

class Qwen3TtsCpMlpPlugin : public nvinfer1::IPluginV3,
                            public nvinfer1::IPluginV3OneCore,
                            public nvinfer1::IPluginV3OneBuild,
                            public nvinfer1::IPluginV3OneRuntime
{
public:
    Qwen3TtsCpMlpPlugin(std::string const& name, int32_t hiddenSize, int32_t ffnSize);
    Qwen3TtsCpMlpPlugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);
    Qwen3TtsCpMlpPlugin() = delete;
    Qwen3TtsCpMlpPlugin(Qwen3TtsCpMlpPlugin const&) = delete;
    ~Qwen3TtsCpMlpPlugin() override;

    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs,
        nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;
    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::DimsExprs const* shapeInputs, int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs,
        int32_t nbOutputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override;
    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs, nvinfer1::PluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;

private:
    bool ensureResources() noexcept;
    void destroyResources() noexcept;

    std::string mLayerName;
    std::string mNamespace;
    int32_t mHiddenSize{1024};
    int32_t mFfnSize{3072};

    mutable cublasHandle_t mGateHandle{};
    mutable cublasHandle_t mUpHandle{};
    mutable cublasHandle_t mDownHandle{};
    mutable cudaStream_t mGateStream{};
    mutable cudaStream_t mUpStream{};
    mutable cudaEvent_t mReadyEvent{};
    mutable cudaEvent_t mGateDoneEvent{};
    mutable cudaEvent_t mUpDoneEvent{};
    mutable bool mResourcesReady{false};

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
};

class Qwen3TtsCpMlpPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    Qwen3TtsCpMlpPluginCreator();
    ~Qwen3TtsCpMlpPluginCreator() override = default;
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;
    nvinfer1::IPluginV3* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase phase) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFieldCollection;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace trt_edgellm::plugins
