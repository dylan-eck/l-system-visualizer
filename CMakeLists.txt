cmake_minimum_required(VERSION 3.10)
project(l-system-visualizer)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB_RECURSE SRC_FILES "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp"
     "${CMAKE_CURRENT_LIST_DIR}/src/*.h")

add_executable(${PROJECT_NAME} ${SRC_FILES})

if(APPLE)
  target_compile_definitions(${PROJECT_NAME} PRIVATE LSV_PLATFORM_APPLE)
elseif(WIN32)
  target_compile_definitions(${PROJECT_NAME} PRIVATE LSV_PLATFORM_WINDOWS)
endif()

find_package(SDL2 REQUIRED)
find_package(Vulkan REQUIRED)
find_program(SLANGC_EXECUTABLE slangc HINTS $ENV{VULKAN_SDK}/bin REQUIRED)
message("-- Found slangc: " ${SLANGC_EXECUTABLE})

set(SHADERS_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(ENTRY_POINTS -entry vertMain -entry fragMain)
set(SPIRV_SHADERS)

file(GLOB_RECURSE SHADER_SOURCES "${CMAKE_SOURCE_DIR}/shaders/*.slang")
foreach(SHADER ${SHADER_SOURCES})
  get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
  set(SPIRV "${CMAKE_BINARY_DIR}/shaders/${SHADER_NAME}.spv")

  add_custom_command(
    OUTPUT ${SPIRV}
    COMMAND
      ${SLANGC_EXECUTABLE} ${SHADER} -target spirv -profile spirv_1_4
      -emit-spirv-directly -fvk-use-entrypoint-name ${ENTRY_POINTS} -o ${SPIRV}
    DEPENDS ${SHADER}
    COMMENT "Compiling shader ${SHADER}"
    VERBATIM)

  list(APPEND SPIRV_SHADERS ${SPIRV})
endforeach()

add_custom_target(shaders ALL DEPENDS ${SPIRV_SHADERS})
add_dependencies(${PROJECT_NAME} shaders)

include(FetchContent)

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.16.0)
set(SPDLOG_BUILD_EXAMPLE
    OFF
    CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS
    OFF
    CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_BENCH
    OFF
    CACHE BOOL "" FORCE)
set(SPDLOG_INSTALL
    OFF
    CACHE BOOL "" FORCE)

target_compile_definitions(
  ${PROJECT_NAME}
  PRIVATE $<$<CONFIG:Debug>:SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG>)

FetchContent_Declare(
  vk-bootstrap
  GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap
  GIT_TAG v1.4.336)

FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG 396b33d0d011290f11915b6a2a2bf7737fb42dd7)

FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm
  GIT_TAG 1.0.2)

FetchContent_MakeAvailable(spdlog vk-bootstrap imgui glm)

add_library(
  imgui STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp ${imgui_SOURCE_DIR}/imgui_widgets.cpp)

target_sources(
  ${PROJECT_NAME} PRIVATE ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl2.cpp
                          ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ${imgui_SOURCE_DIR}
                                                   ${imgui_SOURCE_DIR}/backends)

target_link_libraries(
  ${PROJECT_NAME} PRIVATE imgui Vulkan::Vulkan SDL2::SDL2 spdlog::spdlog
                          vk-bootstrap::vk-bootstrap glm::glm)
