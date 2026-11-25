option(USE_SYSTEM_NCNN "build with system libncnn" OFF)

if(ncnn_DIR)
    set(USE_SYSTEM_NCNN ON)
endif()

if(USE_SYSTEM_NCNN)
    find_package(ncnn CONFIG REQUIRED)
    if(NOT NCNN_VULKAN)
        message(FATAL_ERROR "h3-ncnn-vulkan requires ncnn built with NCNN_VULKAN=ON")
    endif()
endif()

if(NOT USE_SYSTEM_NCNN)
    # Build the ncnn submodule with disk weight paging support.
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ncnn/CMakeLists.txt"
        OR NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ncnn/glslang/CMakeLists.txt")
        message(FATAL_ERROR "Initialize the submodules with git submodule update --init --recursive")
    endif()

    option(NCNN_INSTALL_SDK "" OFF)
    option(GLSLANG_ENABLE_INSTALL "" ${NCNN_INSTALL_SDK})
    option(NCNN_VULKAN "" ON)
    option(NCNN_BATCH "" ON)
    option(NCNN_BUILD_BENCHMARK "" OFF)
    option(NCNN_BUILD_TESTS "" OFF)
    option(NCNN_BUILD_TOOLS "" OFF)
    option(NCNN_BUILD_PNNX "" OFF)
    option(NCNN_BUILD_EXAMPLES "" OFF)
    if(NOT NCNN_VULKAN OR NOT NCNN_BATCH)
        message(FATAL_ERROR "h3-ncnn-vulkan requires NCNN_VULKAN and NCNN_BATCH")
    endif()

    add_subdirectory(ncnn)
endif()
