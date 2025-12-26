#ifdef LSV_PLATFORM_WINDOWS // this is needed for building on windows
#include <SDL.h>
#endif

#include <stdexcept>

#include <spdlog/spdlog.h>

#include "Renderer.h"

int main() {
#ifndef NDEBUG
    spdlog::set_level(spdlog::level::debug);
#endif

    auto renderer = lsv::Renderer();
    lsv::RenderConfig config{.applicationName = "L System Visualizer"};

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