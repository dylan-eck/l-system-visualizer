#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <spdlog/spdlog.h>
#include <SDL2/SDL.h>
#include <SDL_vulkan.h>
#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h>
#include <VkBootstrap.h>

#include "Renderer.h"
#include "PipelineBuilder.h"

#define VK_CHECK(x)                                                            \
    do {                                                                       \
        VkResult err = x;                                                      \
        if (err) {                                                             \
            throw std::runtime_error(                                          \
                fmt::format("Vulkan error: {}", string_VkResult(err)));        \
        }                                                                      \
    } while (0)

namespace lsv {

void Renderer::init(RenderConfig config) {
    if (isInitialized) {
        return;
    }

    windowExtent = VkExtent2D{.width = config.width, .height = config.height};

    SDL_Init(SDL_INIT_VIDEO);
    SDL_WindowFlags windowFlags =
        (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    window = SDL_CreateWindow(config.applicationName, SDL_WINDOWPOS_UNDEFINED,
                              SDL_WINDOWPOS_UNDEFINED, windowExtent.width,
                              windowExtent.height, windowFlags);
    if (!window) {
        throw std::runtime_error("failed to create window");
    }

#ifndef NDEBUG
    bool useValidationLayers = true;
#else
    bool useValidationLayers = false;
#endif

    vkb::InstanceBuilder instanceBuilder;
    vkb::Instance vkbInst = instanceBuilder.set_app_name(config.applicationName)
                                .request_validation_layers(useValidationLayers)
                                .use_default_debug_messenger()
                                .require_api_version(1, 3, 0)
                                .build()
                                .value();

    instance = vkbInst.instance;
    debugMessenger = vkbInst.debug_messenger;

    SDL_Vulkan_CreateSurface(window, instance, &surface);
    if (surface == VK_NULL_HANDLE) {
        throw std::runtime_error("failed to create surface");
    }

    VkPhysicalDeviceVulkan11Features features11{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        .shaderDrawParameters = true};

    VkPhysicalDeviceVulkan13Features features13{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .dynamicRendering = true,
        .synchronization2 = true};

    vkb::PhysicalDeviceSelector deviceSelector{vkbInst};
    vkb::PhysicalDevice vkbPhysicalDevice =
        deviceSelector.set_minimum_version(1, 3)
            .set_required_features_11(features11)
            .set_required_features_13(features13)
            .set_surface(surface)
            .select()
            .value();

    vkb::DeviceBuilder deviceBuilder{vkbPhysicalDevice};
    vkb::Device vkbDevice = deviceBuilder.build().value();

    physicalDevice = vkbPhysicalDevice.physical_device;
    device = vkbDevice.device;
    graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    graphicsQueueFamily =
        vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    createSwapchain(windowExtent.width, windowExtent.height);

    initFrameDatas();

    buildPipelines();

    isInitialized = true;
}

void Renderer::cleanup() {
    if (!isInitialized) {
        return;
    }

    vkDeviceWaitIdle(device);

    vkDestroyPipeline(device, opaquePipeline, nullptr);
    vkDestroyPipelineLayout(device, opaquePipelineLayout, nullptr);

    destroyFrameDatas();

    destroySwapchain();
    vkDestroyDevice(device, nullptr);
    vkDestroySurfaceKHR(instance, surface, nullptr);

    vkb::destroy_debug_utils_messenger(instance, debugMessenger);
    vkDestroyInstance(instance, nullptr);
    SDL_DestroyWindow(window);
}

void Renderer::draw() {
    FrameData &currentFrame = getCurrentFrame();

    VK_CHECK(vkWaitForFences(device, 1, &currentFrame.commandsCompleteFence,
                             true, 1000000000));
    VK_CHECK(vkResetFences(device, 1, &currentFrame.commandsCompleteFence));

    uint32_t swapchainImageIndex;
    VkResult acquireResult = vkAcquireNextImageKHR(
        device, swapchain, 1000000000, currentFrame.imageAvailableSemaphore,
        nullptr, &swapchainImageIndex);
    if (acquireResult == VK_SUBOPTIMAL_KHR ||
        acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
        swapchainStale = true;
        return;
    }

    VkCommandBuffer cmd = getCurrentFrame().commandBuffer;
    VK_CHECK(vkResetCommandBuffer(cmd, 0));

    VkCommandBufferBeginInfo cmdBeginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    transitionImageLayout(cmd, swapchainImages[swapchainImageIndex],
                          VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    VkClearColorValue clearColor{0.0f, 0.0f, 0.0f, 1.0f};

    VkImageSubresourceRange subResourceRange =
        createSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT);

    vkCmdClearColorImage(cmd, swapchainImages[swapchainImageIndex],
                         VK_IMAGE_LAYOUT_GENERAL, &clearColor, 1,
                         &subResourceRange);

    transitionImageLayout(cmd, swapchainImages[swapchainImageIndex],
                          VK_IMAGE_LAYOUT_GENERAL,
                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    VkRenderingAttachmentInfo attachmentInfo{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = swapchainImageViews[swapchainImageIndex],
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    VkRenderingInfo renderingInfo{.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                                  .renderArea =
                                      VkRect2D{.extent = swapchainExtent},
                                  .colorAttachmentCount = 1,
                                  .pColorAttachments = &attachmentInfo,
                                  .layerCount = 1};

    vkCmdBeginRendering(cmd, &renderingInfo);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, opaquePipeline);

    VkViewport viewport{.x = 0,
                        .y = 0,
                        .width = (float)swapchainExtent.width,
                        .height = (float)swapchainExtent.height,
                        .minDepth = 0.0f,
                        .maxDepth = 1.0f};
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{.extent = swapchainExtent};

    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdDraw(cmd, 3, 1, 0, 0);

    vkCmdEndRendering(cmd);

    transitionImageLayout(cmd, swapchainImages[swapchainImageIndex],
                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                          VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkPipelineStageFlags waitStages =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkSubmitInfo submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &currentFrame.imageAvailableSemaphore,
        .pWaitDstStageMask = &waitStages,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &renderFinishedSemaphores[swapchainImageIndex]};

    VK_CHECK(vkQueueSubmit(graphicsQueue, 1, &submitInfo,
                           currentFrame.commandsCompleteFence));

    VkPresentInfoKHR presentInfo{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &renderFinishedSemaphores[swapchainImageIndex],
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = &swapchainImageIndex,
        .pResults = nullptr};

    VkResult presentResult = vkQueuePresentKHR(graphicsQueue, &presentInfo);
    if (presentResult == VK_SUBOPTIMAL_KHR ||
        presentResult == VK_ERROR_OUT_OF_DATE_KHR) {
        swapchainStale = true;
    }

    frameNumber++;
}

void Renderer::run() {
    SDL_Event e;
    bool shouldQuit = false;

    while (!shouldQuit) {
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) {
                shouldQuit = true;
            }
        }

        if (swapchainStale) {
            rebuildSwapchain();
        }

        draw();
    }
}

void Renderer::createSwapchain(uint32_t width, uint32_t height) {
    swapchainFormat = VK_FORMAT_B8G8R8A8_UNORM;

    vkb::SwapchainBuilder builder(physicalDevice, device, surface);

    vkb::Swapchain vkbSwapchain =
        builder
            .set_desired_format(VkSurfaceFormatKHR{
                .format = swapchainFormat,
                .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
            .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
            .set_desired_extent(width, height)
            .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
            .build()
            .value();

    swapchainExtent = vkbSwapchain.extent;

    swapchain = vkbSwapchain.swapchain;
    swapchainImages = vkbSwapchain.get_images().value();
    swapchainImageViews = vkbSwapchain.get_image_views().value();

    renderFinishedSemaphores.resize(swapchainImages.size());
    for (int i = 0; i < swapchainImages.size(); i++) {
        VkSemaphoreCreateInfo semaphoreInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};

        VK_CHECK(vkCreateSemaphore(device, &semaphoreInfo, nullptr,
                                   &renderFinishedSemaphores[i]));

        renderFinishedSemaphores.push_back(renderFinishedSemaphores[i]);
    }
}

void Renderer::rebuildSwapchain() {
    vkDeviceWaitIdle(device);

    destroyFrameDatas();

    destroySwapchain();

    int w, h;
    SDL_GetWindowSize(window, &w, &h);
    createSwapchain(w, h);

    initFrameDatas();

    swapchainStale = false;
}

void Renderer::destroySwapchain() {
    for (int i = 0; i < swapchainImages.size(); i++) {
        vkDestroyImageView(device, swapchainImageViews[i], nullptr);
        vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
    }

    vkDestroySwapchainKHR(device, swapchain, nullptr);
}

void Renderer::initFrameDatas() {
    VkCommandPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = graphicsQueueFamily};

    VkSemaphoreCreateInfo semaphoreInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };

    VkFenceCreateInfo fenceInfo{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                                .flags = VK_FENCE_CREATE_SIGNALED_BIT};

    for (int i = 0; i < FRAMES_IN_FLIGHT; i++) {
        VK_CHECK(vkCreateCommandPool(device, &poolInfo, nullptr,
                                     &frames[i].commandPool));

        VkCommandBufferAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = frames[i].commandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1};

        VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo,
                                          &frames[i].commandBuffer));

        VK_CHECK(vkCreateSemaphore(device, &semaphoreInfo, nullptr,
                                   &frames[i].imageAvailableSemaphore));

        VK_CHECK(vkCreateFence(device, &fenceInfo, nullptr,
                               &frames[i].commandsCompleteFence));
    }
}

void Renderer::destroyFrameDatas() {
    for (const auto &frame : frames) {
        vkDestroyCommandPool(device, frame.commandPool, nullptr);
        vkDestroySemaphore(device, frame.imageAvailableSemaphore, nullptr);
        vkDestroyFence(device, frame.commandsCompleteFence, nullptr);
    }
}

VkImageSubresourceRange
Renderer::createSubresourceRange(VkImageAspectFlags aspectFlags) {
    VkImageSubresourceRange subResourceRange{
        .aspectMask = aspectFlags,
        .baseMipLevel = 0,
        .levelCount = VK_REMAINING_MIP_LEVELS,
        .baseArrayLayer = 0,
        .layerCount = VK_REMAINING_ARRAY_LAYERS};

    return subResourceRange;
}

void Renderer::transitionImageLayout(VkCommandBuffer cmd, VkImage image,
                                     VkImageLayout oldLayout,
                                     VkImageLayout newLayout) {
    VkImageMemoryBarrier2 barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        .dstAccessMask =
            VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .subresourceRange = createSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT),
        .image = image,
    };

    VkDependencyInfo depInfo{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                             .imageMemoryBarrierCount = 1,
                             .pImageMemoryBarriers = &barrier};

    vkCmdPipelineBarrier2(cmd, &depInfo);
}

std::vector<char> Renderer::loadShader(const std::string &filePath) {
    std::ifstream file(filePath, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error(
            fmt::format("failed to open shader file: {}", filePath));
    }

    std::vector<char> buffer(file.tellg());

    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    file.close();

    return buffer;
}

void Renderer::buildPipelines() {
    std::vector<char> shader = loadShader("./build/shaders/triangle.spv");

    VkShaderModuleCreateInfo moduleInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = shader.size() * sizeof(char),
        .pCode = reinterpret_cast<const uint32_t *>(shader.data())};

    VkShaderModule module;
    vkCreateShaderModule(device, &moduleInfo, nullptr, &module);

    VkPipelineLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VK_CHECK(vkCreatePipelineLayout(device, &layoutInfo, nullptr,
                                    &opaquePipelineLayout));

    auto pipelineBuilder =
        PipelineBuilder()
            .setLayout(opaquePipelineLayout)
            .setShaders(module, module)
            .setInputTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
            .setPolygonMode(VK_POLYGON_MODE_FILL)
            .setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE)
            .setMultisampleDisabled()
            .setBlendingDisabled()
            .setDepthTestDisabled()
            .setColorAttachmentFormat(swapchainFormat)
            .setDepthFormat(VK_FORMAT_UNDEFINED);

    opaquePipeline = pipelineBuilder.build(device);

    if (opaquePipeline == VK_NULL_HANDLE) {
        throw std::runtime_error("failed to build opaque pipeline");
    }

    vkDestroyShaderModule(device, module, nullptr);
}
} // namespace lsv
