#pragma once

#include <functional>
#include <span>
#include <vector>
#include <string>
#include <deque>

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
    std::string executablePath;
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

    std::array<float, 4> clearColor{0.0f, 0.0f, 0.0f, 1.0f};

    VkPipelineLayout pipelineLayout;
    VkPipeline linePipeline;
    VkPipeline meshPipeline;

    AllocatedImage mainDrawImage;
    VkExtent2D mainDrawExtent;
    VkDescriptorSet imguiDescriptorSet;

    glm::vec3 tmpAngle{0.0f, 0.0f, 45.0f};

    size_t frameTimeWindow = 10;
    std::deque<double> frameTimes;

    float cameraYaw = 0.0f;
    float cameraPitch = 0.0f;
    float cameraDistance = 3.0f;
    glm::vec3 cameraPosition{0.0f, 1.0f, -2.0f};

    GPUMesh axes;

    static constexpr size_t MAX_VERTEX_COUNT = 8192;
    std::array<Vertex, MAX_VERTEX_COUNT> vertices;
    std::array<uint32_t, MAX_VERTEX_COUNT> indices;
    size_t vertexCount = 0;
    size_t indexCount = 0;
    int lsStringLength = 0;

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
    void destroyMesh(GPUMesh mesh);

    void printMat4(glm::mat4 m);
    std::string vec3ToString(glm::vec3 v);

    void generateLSystem(glm::vec3 rotation);
};
} // namespace lsv