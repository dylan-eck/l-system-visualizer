#pragma once

#include <string>

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

namespace lsv {
struct RenderConfig {
    uint32_t width = 1280;
    uint32_t height = 720;
    const char *applicationName = "";
    std::string executablePath = "";
};

struct Vertex {
    glm::vec3 position;
    float uvX;
    glm::vec3 normal;
    float uvY;
    glm::vec4 color;
};

struct AllocatedImage {
    VkImage image;
    VkImageView imageView;
    VmaAllocation allocation;
    VkExtent3D imageExtent;
    VkFormat imageFormat;
    VkSampler sampler;
};

struct AllocatedBuffer {
    VkBuffer buffer;
    VmaAllocation allocation;
    VmaAllocationInfo allocationInfo;
};

struct FrameData {
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;

    VkSemaphore imageAvailableSemaphore;
    VkFence renderFinishedFence;

    AllocatedBuffer vertexBuffer;
    VkDeviceAddress vertexBufferAddress;
    AllocatedBuffer indexBuffer;
    uint32_t bufferGeneration = 0;
};

struct MeshData {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
};

struct GPUMesh {
    AllocatedBuffer vertices;
    AllocatedBuffer indices;
    VkDeviceAddress vertexBufferAddress;
};

struct GPUDrawPushConstants {
    glm::mat4 worldMatrix;
    VkDeviceAddress vertexBuffer;
};

struct Transformation {
    glm::vec3 translate = glm::vec3(0);
    glm::vec3 rotate = glm::vec3(0);
};
} // namespace lsv