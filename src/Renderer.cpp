#include "imgui_internal.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <chrono>

#include <spdlog/spdlog.h>
#include <SDL2/SDL.h>
#include <SDL_vulkan.h>
#include <imgui.h>
#include <imgui_impl_sdl2.h>
#include <imgui_impl_vulkan.h>
#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h>
#include <VkBootstrap.h>
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

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
    mainDrawExtent = windowExtent;
    // TODO: auto set draw extent to a reasonable default based on window
    // extend;
    sceneDrawExtent = VkExtent2D{.width = 600, .height = 600};

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
        .shaderDrawParameters = VK_TRUE};

    VkPhysicalDeviceVulkan13Features features13{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .dynamicRendering = VK_TRUE,
        .synchronization2 = VK_TRUE};

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

    VmaAllocatorCreateInfo allocatorInfo{
        .physicalDevice = physicalDevice,
        .device = device,
        .instance = instance,
    };

    vmaCreateAllocator(&allocatorInfo, &allocator);

    createSwapchain(windowExtent.width, windowExtent.height);

    initImgui();

    initImmediateCommands();

    createDrawImages();

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

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    vkDestroyDescriptorPool(device, imguiDescriptorPool, nullptr);

    destroySwapchain();

    destroyDrawImages();

    vkDestroyFence(device, immediateCmdFence, nullptr);
    vkDestroyCommandPool(device, immediateCmdPool, nullptr);

    vmaDestroyAllocator(allocator);
    vkDestroyDevice(device, nullptr);

    vkDestroySurfaceKHR(instance, surface, nullptr);

    vkb::destroy_debug_utils_messenger(instance, debugMessenger);
    vkDestroyInstance(instance, nullptr);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

void Renderer::draw(ImDrawData *imGuiDrawData) {
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

    VkClearColorValue clearColor{1.0f, 0.0f, 1.0f, 1.0f};

    VkImageSubresourceRange subResourceRange =
        createSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT);

    // ### begin recording commands
    VkCommandBuffer cmd = getCurrentFrame().commandBuffer;
    VK_CHECK(vkResetCommandBuffer(cmd, 0));

    VkCommandBufferBeginInfo cmdBeginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    // ### render scene
    transitionImageLayout(cmd, sceneDrawImage.image, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_GENERAL);

    vkCmdClearColorImage(cmd, sceneDrawImage.image, VK_IMAGE_LAYOUT_GENERAL,
                         &clearColor, 1, &subResourceRange);

    transitionImageLayout(cmd, sceneDrawImage.image, VK_IMAGE_LAYOUT_GENERAL,
                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    VkRenderingAttachmentInfo sceneAttachmentInfo{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = sceneDrawImage.imageView,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    sceneDrawExtent = VkExtent2D{
        .width = sceneDrawImage.imageExtent.width,
        .height = sceneDrawImage.imageExtent.height,
    };

    VkRenderingInfo sceneRenderingInfo{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = VkRect2D{.extent = sceneDrawExtent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &sceneAttachmentInfo,
    };

    vkCmdBeginRendering(cmd, &sceneRenderingInfo);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, opaquePipeline);

    VkViewport viewport{.x = 0,
                        .y = 0,
                        .width = (float)sceneDrawExtent.width,
                        .height = (float)sceneDrawExtent.height,
                        .minDepth = 0.0f,
                        .maxDepth = 1.0f};
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{.extent = sceneDrawExtent};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdDraw(cmd, 3, 1, 0, 0);

    vkCmdEndRendering(cmd);

    transitionImageLayout(cmd, sceneDrawImage.image,
                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // ### render gui ###
    transitionImageLayout(cmd, mainDrawImage.image, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_GENERAL);

    vkCmdClearColorImage(cmd, mainDrawImage.image, VK_IMAGE_LAYOUT_GENERAL,
                         &clearColor, 1, &subResourceRange);

    transitionImageLayout(cmd, mainDrawImage.image, VK_IMAGE_LAYOUT_GENERAL,
                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    VkRenderingAttachmentInfo attachmentInfo{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = mainDrawImage.imageView,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    mainDrawExtent = VkExtent2D{
        .width = mainDrawImage.imageExtent.width,
        .height = mainDrawImage.imageExtent.height,
    };

    VkRenderingInfo renderingInfo{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = VkRect2D{.extent = mainDrawExtent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &attachmentInfo,
    };

    vkCmdBeginRendering(cmd, &renderingInfo);

    ImGui_ImplVulkan_RenderDrawData(imGuiDrawData, cmd);

    vkCmdEndRendering(cmd);

    // ### blit to swapchain and prepare for present ###
    transitionImageLayout(cmd, mainDrawImage.image,
                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    transitionImageLayout(cmd, swapchainImages[swapchainImageIndex],
                          VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    blitImageToImage(cmd, mainDrawImage.image,
                     swapchainImages[swapchainImageIndex], mainDrawExtent,
                     swapchainExtent);

    transitionImageLayout(cmd, swapchainImages[swapchainImageIndex],
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
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

    auto lastTime = std::chrono::high_resolution_clock::now();

    while (!shouldQuit) {
        auto now = std::chrono::high_resolution_clock::now();
        double delta =
            std::chrono::duration<double, std::milli>(now - lastTime).count();
        lastTime = now;

        while (SDL_PollEvent(&e) != 0) {
            ImGui_ImplSDL2_ProcessEvent(&e);
            if (e.type == SDL_QUIT) {
                shouldQuit = true;
            }
        }

        if (swapchainStale) {
            rebuildSwapchain();
        }

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        ImGuiID dockspaceID = ImGui::GetID("MainDockSpace");
        ImGuiDockNodeFlags dockspaceFlags =
            ImGuiDockNodeFlags_NoTabBar | ImGuiDockNodeFlags_NoWindowMenuButton;
        ImGui::DockSpaceOverViewport(dockspaceID, nullptr, dockspaceFlags);

        ImGui::Begin("info");
        ImGui::Text("cpu frame time: %2.0f ms (%4.0f fps)", delta,
                    1000 / delta);
        ImGui::End();

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::Begin("viewport");
        ImVec2 size;
        size.x = sceneDrawExtent.width;
        size.y = sceneDrawExtent.height;
        ImGui::Image((ImTextureID)sceneDrawSet, size);
        ImGui::End();
        ImGui::PopStyleVar();

        ImGui::Render();
        ImDrawData *drawData = ImGui::GetDrawData();

        draw(drawData);
    }
}

void Renderer::initImmediateCommands() {
    VkFenceCreateInfo fenceInfo{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
    };

    VK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &immediateCmdFence));

    VkCommandPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = graphicsQueueFamily,
    };

    VK_CHECK(
        vkCreateCommandPool(device, &poolInfo, nullptr, &immediateCmdPool));

    VkCommandBufferAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = immediateCmdPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo, &immediateCmdBuffer));
}

void Renderer::immediateSubmit(
    std::function<void(VkCommandBuffer cmd)> &&function) {

    VK_CHECK(vkResetFences(device, 1, &immediateCmdFence));
    VK_CHECK(vkResetCommandBuffer(immediateCmdBuffer, 0));

    VkCommandBuffer cmd = immediateCmdBuffer;

    VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

    function(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };

    VK_CHECK(vkQueueSubmit(graphicsQueue, 1, &submitInfo, immediateCmdFence));

    VK_CHECK(vkWaitForFences(device, 1, &immediateCmdFence, true, 9999999999));
}

void Renderer::initImgui() {
    std::vector<VkDescriptorPoolSize> poolSizes{VkDescriptorPoolSize{
        .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE}};

    uint32_t totalDescriptors =
        std::accumulate(poolSizes.begin(), poolSizes.end(), uint32_t{0},
                        [](uint32_t sum, const VkDescriptorPoolSize &p) {
                            return sum + p.descriptorCount;
                        });

    VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = totalDescriptors,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data()};

    VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr,
                                    &imguiDescriptorPool));

    VkFormat format = VK_FORMAT_R16G16B16A16_SFLOAT;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    ImGui_ImplSDL2_InitForVulkan(window);
    ImGui_ImplVulkan_InitInfo initInfo{
        .ApiVersion = VK_API_VERSION_1_3,
        .Instance = instance,
        .PhysicalDevice = physicalDevice,
        .Device = device,
        .QueueFamily = graphicsQueueFamily,
        .Queue = graphicsQueue,
        .DescriptorPool = imguiDescriptorPool,
        .DescriptorPoolSize = 0,
        .MinImageCount = 2,
        .ImageCount = static_cast<uint32_t>(swapchainImages.size()),
        .PipelineCache = VK_NULL_HANDLE,
        .PipelineInfoMain{.PipelineRenderingCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &format,
        }},
        .UseDynamicRendering = true};

    ImGui_ImplVulkan_Init(&initInfo);
}

void Renderer::createDrawImages() {

    mainDrawImage.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    mainDrawImage.imageExtent = VkExtent3D{
        .width = mainDrawExtent.width,
        .height = mainDrawExtent.height,
        .depth = 1,
    };

    sceneDrawImage.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    sceneDrawImage.imageExtent = VkExtent3D{
        .width = sceneDrawExtent.width,
        .height = sceneDrawExtent.height,
        .depth = 1,
    };

    VkImageCreateInfo mainImageInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = mainDrawImage.imageFormat,
        .extent = mainDrawImage.imageExtent,
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                 VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
    };

    VmaAllocationCreateInfo allocInfo{
        .usage = VMA_MEMORY_USAGE_GPU_ONLY,
        .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    };

    vmaCreateImage(allocator, &mainImageInfo, &allocInfo, &mainDrawImage.image,
                   &mainDrawImage.allocation, nullptr);

    VkImageViewCreateInfo viewInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = mainDrawImage.image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = mainDrawImage.imageFormat,
        .subresourceRange{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr,
                               &mainDrawImage.imageView));

    VkImageCreateInfo sceneImageInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = sceneDrawImage.imageFormat,
        .extent = sceneDrawImage.imageExtent,
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,

    };

    VkSamplerCreateInfo samplerInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.0f,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_ALWAYS,
        .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    };

    vmaCreateImage(allocator, &sceneImageInfo, &allocInfo,
                   &sceneDrawImage.image, &sceneDrawImage.allocation, nullptr);

    viewInfo.image = sceneDrawImage.image;
    viewInfo.format = sceneDrawImage.imageFormat;

    VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr,
                               &sceneDrawImage.imageView));

    VK_CHECK(vkCreateSampler(device, &samplerInfo, nullptr,
                             &sceneDrawImage.sampler));

    immediateSubmit([&](VkCommandBuffer cmd) {
        transitionImageLayout(cmd, sceneDrawImage.image,
                              VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });

    sceneDrawSet = ImGui_ImplVulkan_AddTexture(
        sceneDrawImage.sampler, sceneDrawImage.imageView,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void Renderer::destroyDrawImages() {
    vkDestroySampler(device, sceneDrawImage.sampler, nullptr);
    vkDestroyImageView(device, sceneDrawImage.imageView, nullptr);
    vmaDestroyImage(allocator, sceneDrawImage.image, sceneDrawImage.allocation);

    vkDestroyImageView(device, mainDrawImage.imageView, nullptr);
    vmaDestroyImage(allocator, mainDrawImage.image, mainDrawImage.allocation);
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
        .queueFamilyIndex = graphicsQueueFamily,
    };

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
            .commandBufferCount = 2};

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
        .layerCount = VK_REMAINING_ARRAY_LAYERS,
    };

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
        .image = image,
        .subresourceRange = createSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT),
    };

    VkDependencyInfo depInfo{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                             .imageMemoryBarrierCount = 1,
                             .pImageMemoryBarriers = &barrier};

    vkCmdPipelineBarrier2(cmd, &depInfo);
}

void Renderer::blitImageToImage(VkCommandBuffer cmd, VkImage src, VkImage dst,
                                VkExtent2D srcSize, VkExtent2D dstSize) {

    VkImageBlit2 blitRegion{
        .sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
        .srcSubresource{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
        .dstSubresource{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    blitRegion.srcOffsets[1].x = srcSize.width;
    blitRegion.srcOffsets[1].y = srcSize.height;
    blitRegion.srcOffsets[1].z = 1;

    blitRegion.dstOffsets[1].x = dstSize.width;
    blitRegion.dstOffsets[1].y = dstSize.height;
    blitRegion.dstOffsets[1].z = 1;

    VkBlitImageInfo2 blitInfo{
        .sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
        .srcImage = src,
        .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .dstImage = dst,
        .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .regionCount = 1,
        .pRegions = &blitRegion,
        .filter = VK_FILTER_LINEAR,
    };

    vkCmdBlitImage2(cmd, &blitInfo);
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
            .setColorAttachmentFormat(mainDrawImage.imageFormat)
            .setDepthFormat(VK_FORMAT_UNDEFINED);

    opaquePipeline = pipelineBuilder.build(device);

    if (opaquePipeline == VK_NULL_HANDLE) {
        throw std::runtime_error("failed to build opaque pipeline");
    }

    vkDestroyShaderModule(device, module, nullptr);
}
} // namespace lsv
