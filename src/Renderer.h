#pragma once

#include <vector>

#include <vulkan/vulkan.h>

namespace lsv {
constexpr unsigned int FRAMES_IN_FLIGHT = 2;

struct RenderConfig {
    uint32_t width = 1280;
    uint32_t height = 720;
    const char *applicationName = "";
};

struct FrameData {
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;

    VkSemaphore imageAvailableSemaphore;
    VkFence commandsCompleteFence;
};

class Renderer {
public:
    void init(RenderConfig config = {});
    void cleanup();

    void draw();
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

    VkSurfaceKHR surface;
    VkPhysicalDevice physicalDevice;
    VkQueue graphicsQueue;
    uint32_t graphicsQueueFamily;
    VkDevice device;

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

    VkPipelineLayout opaquePipelineLayout;
    VkPipeline opaquePipeline;

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

    std::vector<char> loadShader(const std::string &filePath);
    void buildPipelines();
};
} // namespace lsv