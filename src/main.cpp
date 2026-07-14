#ifdef LSV_PLATFORM_WINDOWS // this is needed for building on windows
#include <SDL.h>
#endif

#include <stdexcept>
#include <filesystem>

#include <spdlog/spdlog.h>

#include "Renderer.h"

int main(int argc, char *argv[]) {
#ifndef NDEBUG
    spdlog::set_level(spdlog::level::debug);
#endif
    auto exe_path = std::filesystem::canonical(argv[0]).parent_path();
    SPDLOG_DEBUG("exe path: {}", exe_path.c_str());

    auto renderer = lsv::Renderer();
    lsv::RenderConfig config{.applicationName = "L System Visualizer",
                             .executablePath = exe_path.c_str()};

    try {
        renderer.init(config);
        renderer.run();
        renderer.cleanup();
    } catch (const std::runtime_error &e) {
        SPDLOG_CRITICAL(e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}