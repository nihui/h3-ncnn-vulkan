cmake_minimum_required(VERSION 3.18)
project(h3-ncnn-vulkan LANGUAGES C CXX)

include(GNUInstallDirs)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type" FORCE)
endif()

get_filename_component(H3_PROJECT_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

add_executable(h3-ncnn-vulkan
    main.cpp
    models.cpp
    snake.cpp
    lora.cpp
    safetensors.cpp
    memory_budget.cpp
    runtime.cpp
    allocation_layout.cpp
    shape_memory.cpp
    condition_tables.cpp
    modulation_builder.cpp
    reference_kv.cpp
    rough_graph.cpp
    tokenizer.cpp
    text_encoder.cpp
    vision.cpp
    image_io.cpp
    video_io.cpp
    webm_video.cpp
    matroska_input.cpp
    audio_io.cpp
    scheduler.cpp
    transformer.cpp
    vae.cpp
    vae_encoder.cpp
    vae_tiles.cpp
    audio_vae.cpp
    reference_input.cpp
    pipeline.cpp)
target_compile_features(h3-ncnn-vulkan PRIVATE cxx_std_17)
option(NCNN_DISABLE_RTTI "Build without C++ RTTI" ON)
option(NCNN_DISABLE_EXCEPTION "Build without C++ exceptions" ON)
set(h3_cxx_options "")
if(MSVC)
    if(NCNN_DISABLE_RTTI)
        list(APPEND h3_cxx_options /GR-)
    endif()
    if(NCNN_DISABLE_EXCEPTION)
        list(APPEND h3_cxx_options /EHs-c- /D_HAS_EXCEPTIONS=0)
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(NCNN_DISABLE_RTTI)
        list(APPEND h3_cxx_options -fno-rtti)
    endif()
    if(NCNN_DISABLE_EXCEPTION)
        list(APPEND h3_cxx_options -fno-exceptions)
    endif()
endif()
target_compile_options(h3-ncnn-vulkan PRIVATE ${h3_cxx_options})

include(deps_ncnn.cmake)
include(deps_codec.cmake)
find_package(Threads REQUIRED)
target_link_libraries(h3-ncnn-vulkan PRIVATE ncnn ${DEPS_CODEC_LIBRARIES} Threads::Threads)

if(WIN32)
    target_link_libraries(h3-ncnn-vulkan PRIVATE psapi)
endif()

install(TARGETS h3-ncnn-vulkan RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES "${H3_PROJECT_DIR}/README.md" "${H3_PROJECT_DIR}/LICENSE"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/h3-ncnn-vulkan)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/ncnn/LICENSE.txt"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/h3-ncnn-vulkan RENAME ncnn-LICENSE.txt OPTIONAL)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/ncnn/glslang/LICENSE.txt"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/h3-ncnn-vulkan RENAME glslang-LICENSE.txt OPTIONAL)
