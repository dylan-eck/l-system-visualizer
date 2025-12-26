#include "PipelineBuilder.h"

namespace lsv {
void PipelineBuilder::clear() {
    inputAssemblyState = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    rasterizationState = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    multisampleState = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    depthStencilState = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    dynamicState = {.sType =
                        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    layout = {};
    renderingInfo = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    shaderStages.clear();
}

VkPipeline PipelineBuilder::build(VkDevice device) {
    VkPipelineViewportStateCreateInfo viewportState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1};

    VkPipelineColorBlendStateCreateInfo blendState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = VK_FALSE,
        .logicOp = VK_LOGIC_OP_COPY,
        .attachmentCount = 1,
        .pAttachments = &colorBlendState};

    VkPipelineVertexInputStateCreateInfo vertexInputState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};

    VkDynamicState state[] = {VK_DYNAMIC_STATE_VIEWPORT,
                              VK_DYNAMIC_STATE_SCISSOR};

    VkPipelineDynamicStateCreateInfo dynamicInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2,
        .pDynamicStates = state};

    VkGraphicsPipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &renderingInfo,
        .stageCount = static_cast<uint32_t>(shaderStages.size()),
        .pStages = shaderStages.data(),
        .pVertexInputState = &vertexInputState,
        .pInputAssemblyState = &inputAssemblyState,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizationState,
        .pMultisampleState = &multisampleState,
        .pDepthStencilState = &depthStencilState,
        .pColorBlendState = &blendState,
        .pDynamicState = &dynamicInfo,
        .layout = layout};

    VkPipeline pipeline;
    if (vkCreateGraphicsPipelines(device, nullptr, 1, &pipelineInfo, nullptr,
                                  &pipeline) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    } else {
        return pipeline;
    }
}

PipelineBuilder PipelineBuilder::setLayout(VkPipelineLayout layout) {
    this->layout = layout;
    return *this;
}

PipelineBuilder PipelineBuilder::setShaders(VkShaderModule vertexShader,
                                            VkShaderModule fragmentShader) {
    shaderStages.clear();

    VkPipelineShaderStageCreateInfo vertexInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = vertexShader,
        .pName = "vertMain"};

    VkPipelineShaderStageCreateInfo fragmentInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = vertexShader,
        .pName = "fragMain"};

    shaderStages.push_back(vertexInfo);
    shaderStages.push_back(fragmentInfo);

    return *this;
}

PipelineBuilder
PipelineBuilder::setInputTopology(VkPrimitiveTopology topology) {
    inputAssemblyState.topology = topology;
    inputAssemblyState.primitiveRestartEnable = VK_FALSE;

    return *this;
}

PipelineBuilder PipelineBuilder::setPolygonMode(VkPolygonMode polygonMode) {
    rasterizationState.polygonMode = polygonMode;
    rasterizationState.lineWidth = 1.0f;

    return *this;
}

PipelineBuilder PipelineBuilder::setCullMode(VkCullModeFlags cullMode,
                                             VkFrontFace frontFace) {
    rasterizationState.cullMode = cullMode;
    rasterizationState.frontFace = frontFace;

    return *this;
}

PipelineBuilder PipelineBuilder::setMultisampleDisabled() {
    multisampleState.sampleShadingEnable = VK_FALSE;
    multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampleState.pSampleMask = nullptr;
    multisampleState.alphaToCoverageEnable = VK_FALSE;
    multisampleState.alphaToOneEnable = VK_FALSE;

    return *this;
}

PipelineBuilder PipelineBuilder::setBlendingDisabled() {
    colorBlendState.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendState.blendEnable = VK_FALSE;

    return *this;
}

PipelineBuilder PipelineBuilder::setColorAttachmentFormat(VkFormat format) {
    colorAttachmentFormat = format;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &colorAttachmentFormat;

    return *this;
}

PipelineBuilder PipelineBuilder::setDepthFormat(VkFormat format) {
    renderingInfo.depthAttachmentFormat = format;

    return *this;
}

PipelineBuilder PipelineBuilder::setDepthTestDisabled() {
    depthStencilState.depthTestEnable = VK_FALSE;
    depthStencilState.depthWriteEnable = VK_FALSE;
    depthStencilState.depthCompareOp = VK_COMPARE_OP_NEVER;
    depthStencilState.depthBoundsTestEnable = VK_FALSE;
    depthStencilState.stencilTestEnable = VK_FALSE;
    depthStencilState.front = {};
    depthStencilState.back = {};
    depthStencilState.minDepthBounds = 0.0f;
    depthStencilState.maxDepthBounds = 1.0f;

    return *this;
}
} // namespace lsv