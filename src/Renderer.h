#pragma once

#include <functional>
#include <span>
#include <vector>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <imgui.h>

#include "RendererTypes.h"

namespace lsv {
constexpr unsigned int FRAMES_IN_FLIGHT = 2;

class Renderer {
public:
    void init(RenderConfig config = {});
    void cleanup();

    void draw(ImDrawData *imGuiDrawData);
    void run();

private:
    const char *applicationName;
    VkExtent2D windowExtent;
    int frameNumber{0};
    bool isInitialized{false};
    bool stopRendering{false};

    struct SDL_Window *window{nullptr};

    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VmaAllocator allocator;

    VkDescriptorPool imguiDescriptorPool;

    VkSurfaceKHR surface;
    VkPhysicalDevice physicalDevice;
    VkQueue graphicsQueue;
    uint32_t graphicsQueueFamily;
    VkDevice device;

    VkFence immediateCmdFence;
    VkCommandPool immediateCmdPool;
    VkCommandBuffer immediateCmdBuffer;

    VkSwapchainKHR swapchain;
    VkExtent2D swapchainExtent;
    VkFormat swapchainFormat;
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    bool swapchainStale = false;

    FrameData frames[FRAMES_IN_FLIGHT];
    FrameData &getCurrentFrame() {
        return frames[frameNumber % FRAMES_IN_FLIGHT];
    };

    std::array<float, 4> clearColor{1.0f, 0.0f, 1.0f, 1.0f};

    VkPipelineLayout meshPipelineLayout;
    VkPipeline meshPipeline;

    GPUMesh rectangle;

    AllocatedImage mainDrawImage;
    VkExtent2D mainDrawExtent;
    VkDescriptorSet imguiDescriptorSet;

    void initImmediateCommands();
    void immediateSubmit(std::function<void(VkCommandBuffer cmd)> &&function);

    void initImgui();

    void createDrawImage();
    void destroyDrawImage();

    void createSwapchain(uint32_t width, uint32_t height);
    void rebuildSwapchain();
    void destroySwapchain();

    void initFrameDatas();
    void destroyFrameDatas();

    VkImageSubresourceRange
    createSubresourceRange(VkImageAspectFlags aspectFlags);
    void transitionImageLayout(VkCommandBuffer cmd, VkImage image,
                               VkImageLayout oldLayout,
                               VkImageLayout newLayout);
    void blitImageToImage(VkCommandBuffer cmd, VkImage src, VkImage dst,
                          VkExtent2D srcSize, VkExtent2D dstSize);

    std::vector<char> loadShader(const std::string &filePath);
    void buildPipelines();
    void destroyPipelines();

    AllocatedBuffer createBuffer(size_t size, VkBufferUsageFlags usageFlags,
                                 VmaMemoryUsage memoryUsage);
    void destroyBuffer(AllocatedBuffer buffer);

    GPUMesh uploadMesh(std::span<Vertex> vertices, std::span<uint32_t> indices);
};
} // namespace lsv