#pragma once

#include <vector>

#include <vulkan/vulkan.h>

namespace lsv {
class PipelineBuilder {
public:
    PipelineBuilder() { clear(); }

    void clear();
    VkPipeline build(VkDevice device);

    PipelineBuilder &setLayout(VkPipelineLayout layout);
    PipelineBuilder &setShaders(VkShaderModule vertexShader,
                                VkShaderModule fragmentShader);
    PipelineBuilder &setInputTopology(VkPrimitiveTopology topology);
    PipelineBuilder &setPrimitiveRestartEnabled(VkBool32 enabled = VK_FALSE);
    PipelineBuilder &setPolygonMode(VkPolygonMode mode);
    PipelineBuilder &
    setCullMode(VkCullModeFlags mode,
                VkFrontFace frontFace = VK_FRONT_FACE_CLOCKWISE);
    PipelineBuilder &setMultisampleDisabled();
    PipelineBuilder &setBlendingDisabled();
    PipelineBuilder &setColorAttachmentFormat(VkFormat format);
    PipelineBuilder &setDepthFormat(VkFormat format);
    PipelineBuilder &setDepthTestDisabled();

private:
    std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyState;
    VkPipelineRasterizationStateCreateInfo rasterizationState;
    VkPipelineMultisampleStateCreateInfo multisampleState;
    VkPipelineDepthStencilStateCreateInfo depthStencilState;
    VkPipelineColorBlendAttachmentState colorBlendState;
    VkPipelineDynamicStateCreateInfo dynamicState;
    VkPipelineLayout layout;
    VkPipelineRenderingCreateInfo renderingInfo;
    VkFormat colorAttachmentFormat;
};
} // namespace lsv