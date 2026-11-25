option(USE_SYSTEM_JPEG "build with system libjpeg" OFF)
option(USE_SYSTEM_ZLIB "build with system zlib" OFF)
option(USE_SYSTEM_PNG "build with system libpng" OFF)
option(USE_SYSTEM_WEBP "build with system libwebp" OFF)

if(USE_SYSTEM_PNG)
    set(USE_SYSTEM_ZLIB ON)
endif()

set(system_codecs "")
set(system_headers "")
if(USE_SYSTEM_JPEG)
    find_package(JPEG REQUIRED)
    list(APPEND system_codecs jpeg)
    list(APPEND system_headers jconfig.h jerror.h jmorecfg.h jpeglib.h)
endif()

if(USE_SYSTEM_PNG)
    find_package(PNG REQUIRED)
    list(APPEND system_codecs png)
    list(APPEND system_headers libpng libpng16 png.h pngconf.h pnglibconf.h)
endif()

if(USE_SYSTEM_WEBP)
    find_package(WebP REQUIRED)
    list(APPEND system_codecs webp)
    list(APPEND system_headers webp)
endif()

if(USE_SYSTEM_ZLIB)
    find_package(ZLIB REQUIRED)
    list(APPEND system_codecs zlib)
    list(APPEND system_headers zconf.h zlib.h)
endif()

list(JOIN system_codecs "-" system_codec_suffix)
set(H3_MEDIA_PREFIX "" CACHE PATH "Use an existing static media codec installation")
if(H3_MEDIA_PREFIX)
    set(codec_prefix "${H3_MEDIA_PREFIX}")
else()
    set(H3_CODEC_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/deps-install" CACHE PATH "Bundled codec install prefix")
    set(codec_prefix "${H3_CODEC_INSTALL_PREFIX}")
    if(system_codec_suffix)
        set(codec_prefix "${codec_prefix}/system-${system_codec_suffix}")
    endif()
endif()

set(codec_libdir "${codec_prefix}/lib")
set(archive_prefix "${codec_libdir}/${CMAKE_STATIC_LIBRARY_PREFIX}")
set(archive_suffix "${CMAKE_STATIC_LIBRARY_SUFFIX}")
set(vpx_library "${archive_prefix}vpx${archive_suffix}")
set(opus_library "${archive_prefix}opus${archive_suffix}")
set(webm_library "${archive_prefix}webm${archive_suffix}")

set(bundled_codecs libvpx opus libwebm)
if(USE_SYSTEM_JPEG)
    set(jpeg_library JPEG::JPEG)
else()
    set(jpeg_library "${archive_prefix}jpeg${archive_suffix}")
    list(APPEND bundled_codecs libjpeg-turbo)
endif()

if(USE_SYSTEM_PNG)
    set(png_library PNG::PNG)
else()
    set(png_library "${archive_prefix}png${archive_suffix}")
    list(APPEND bundled_codecs libpng)
endif()

if(USE_SYSTEM_WEBP)
    set(webp_libraries WebP::WebP)
else()
    set(webp_libraries "${archive_prefix}webpdecoder${archive_suffix}" "${archive_prefix}sharpyuv${archive_suffix}")
    list(APPEND bundled_codecs libwebp)
endif()

if(USE_SYSTEM_ZLIB)
    set(zlib_library ZLIB::ZLIB)
else()
    if(MSVC)
        set(zlib_library "${archive_prefix}zlibstatic${archive_suffix}")
    else()
        set(zlib_library "${archive_prefix}z${archive_suffix}")
    endif()
    list(APPEND bundled_codecs zlib-ng)
endif()

set(DEPS_CODEC_LIBRARIES ${vpx_library} ${opus_library} ${webm_library}
    ${jpeg_library} ${png_library} ${webp_libraries} ${zlib_library})

if(H3_MEDIA_PREFIX)
    foreach(library IN LISTS DEPS_CODEC_LIBRARIES)
        if(NOT TARGET "${library}" AND NOT EXISTS "${library}")
            message(FATAL_ERROR "Missing static codec archive ${library}")
        endif()
    endforeach()
endif()

if(NOT H3_MEDIA_PREFIX)
    include(ExternalProject)
    foreach(codec IN LISTS bundled_codecs)
        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${codec}/CMakeLists.txt"
            AND NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${codec}/configure")
            message(FATAL_ERROR "Initialize the submodules with git submodule update --init --recursive")
        endif()
    endforeach()

    set(H3_CODEC_JOBS 2 CACHE STRING "Parallel jobs per bundled codec build")
    set(codec_cmake_args
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_INSTALL_LIBDIR=lib
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        -DBUILD_SHARED_LIBS=OFF)
    if(CMAKE_TOOLCHAIN_FILE)
        list(APPEND codec_cmake_args "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}")
    elseif(CMAKE_CROSSCOMPILING)
        list(APPEND codec_cmake_args
            "-DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}"
            "-DCMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}")
    endif()
    if(APPLE)
        list(APPEND codec_cmake_args
            "-DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}"
            "-DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}")
    endif()
    set(codec_build_command "${CMAKE_COMMAND}" --build <BINARY_DIR> --config $<CONFIG>
        --parallel ${H3_CODEC_JOBS} --target install)

    # Build libvpx for VP8/VP9.
    find_program(H3_MAKE NAMES gmake make REQUIRED)
    find_program(H3_SH NAMES sh REQUIRED)
    set(vpx_configure "${CMAKE_CURRENT_SOURCE_DIR}/libvpx/configure")
    set(vpx_prefix "${codec_prefix}")
    set(vpx_target "")
    if(MINGW)
        set(vpx_target --target=x86_64-win64-gcc)
        if(CMAKE_HOST_WIN32)
            find_program(H3_CYGPATH NAMES cygpath REQUIRED)
            execute_process(COMMAND "${H3_CYGPATH}" -u "${vpx_configure}"
                OUTPUT_VARIABLE vpx_configure OUTPUT_STRIP_TRAILING_WHITESPACE)
            execute_process(COMMAND "${H3_CYGPATH}" -u "${vpx_prefix}"
                OUTPUT_VARIABLE vpx_prefix OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(NOT vpx_configure OR NOT vpx_prefix)
                message(FATAL_ERROR "Could not convert the libvpx paths for the MSYS2 shell")
            endif()
        endif()
    elseif(APPLE AND CMAKE_OSX_ARCHITECTURES MATCHES "^(x86_64|arm64)$")
        set(vpx_target --target=${CMAKE_OSX_ARCHITECTURES}-darwin20-gcc)
    endif()
    ExternalProject_Add(deps-libvpx
        SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libvpx"
        BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/libvpx"
        INSTALL_DIR "${codec_prefix}"
        CONFIGURE_COMMAND "${CMAKE_COMMAND}" -E env
            "CC=${CMAKE_C_COMPILER}" "CXX=${CMAKE_CXX_COMPILER}" "AR=${CMAKE_AR}"
            "${H3_SH}" "${vpx_configure}" "--prefix=${vpx_prefix}" "--libdir=${vpx_prefix}/lib" ${vpx_target}
            --enable-pic --disable-shared --disable-examples --disable-tools --disable-docs --disable-unit-tests
        BUILD_COMMAND "${H3_MAKE}" -j ${H3_CODEC_JOBS} install
        BUILD_BYPRODUCTS "${vpx_library}"
        INSTALL_COMMAND "")

    # Build libopus and libwebm for WebM and Matroska.
    ExternalProject_Add(deps-opus
        SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/opus"
        BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/opus"
        INSTALL_DIR "${codec_prefix}"
        CMAKE_ARGS ${codec_cmake_args}
            -DOPUS_BUILD_TESTING=OFF
            -DOPUS_BUILD_PROGRAMS=OFF
        BUILD_COMMAND ${codec_build_command}
        BUILD_BYPRODUCTS "${opus_library}"
        INSTALL_COMMAND "")

    string(JOIN " " webm_cxx_flags "${CMAKE_CXX_FLAGS}" ${h3_cxx_options})
    ExternalProject_Add(deps-libwebm
        SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libwebm"
        BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/libwebm"
        INSTALL_DIR "${codec_prefix}"
        CMAKE_ARGS ${codec_cmake_args}
            "-DCMAKE_CXX_FLAGS:STRING=${webm_cxx_flags}"
            -DENABLE_WEBMTS=OFF
            -DENABLE_WEBMINFO=OFF
            -DENABLE_SAMPLE_PROGRAMS=OFF
            -DENABLE_TESTS=OFF
        BUILD_COMMAND ${codec_build_command}
        BUILD_BYPRODUCTS "${webm_library}"
        INSTALL_COMMAND "")

    if(NOT USE_SYSTEM_JPEG)
        ExternalProject_Add(deps-libjpeg-turbo
            SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libjpeg-turbo"
            BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/libjpeg-turbo"
            INSTALL_DIR "${codec_prefix}"
            CMAKE_ARGS ${codec_cmake_args}
                -DENABLE_SHARED=OFF
                -DENABLE_STATIC=ON
                -DWITH_TURBOJPEG=OFF
                -DWITH_TOOLS=OFF
                -DWITH_TESTS=OFF
            BUILD_COMMAND ${codec_build_command}
            BUILD_BYPRODUCTS "${jpeg_library}"
            INSTALL_COMMAND "")
    endif()

    if(NOT USE_SYSTEM_ZLIB)
        ExternalProject_Add(deps-zlib-ng
            SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/zlib-ng"
            BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/zlib-ng"
            INSTALL_DIR "${codec_prefix}"
            CMAKE_ARGS ${codec_cmake_args}
                -DZLIB_COMPAT=ON
                -DBUILD_TESTING=OFF
            BUILD_COMMAND ${codec_build_command}
            BUILD_BYPRODUCTS "${zlib_library}"
            INSTALL_COMMAND "")
    endif()

    if(NOT USE_SYSTEM_PNG)
        if(USE_SYSTEM_ZLIB)
            set(png_zlib_args -DZLIB_INCLUDE_DIR=${ZLIB_INCLUDE_DIRS} -DZLIB_LIBRARY=${ZLIB_LIBRARIES})
            set(png_dependencies "")
        else()
            set(png_zlib_args "-DZLIB_INCLUDE_DIR=${codec_prefix}/include" "-DZLIB_LIBRARY=${zlib_library}")
            set(png_dependencies deps-zlib-ng)
        endif()
        ExternalProject_Add(deps-libpng
            SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libpng"
            BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/libpng"
            INSTALL_DIR "${codec_prefix}"
            CMAKE_ARGS ${codec_cmake_args}
                -DPNG_SHARED=OFF
                -DPNG_STATIC=ON
                -DPNG_TESTS=OFF
                -DPNG_TOOLS=OFF
                ${png_zlib_args}
            DEPENDS ${png_dependencies}
            BUILD_COMMAND ${codec_build_command}
            BUILD_BYPRODUCTS "${png_library}"
            INSTALL_COMMAND "")
    endif()

    if(NOT USE_SYSTEM_WEBP)
        ExternalProject_Add(deps-libwebp
            SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libwebp"
            BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/libwebp"
            INSTALL_DIR "${codec_prefix}"
            CMAKE_ARGS ${codec_cmake_args}
                -DWEBP_BUILD_CWEBP=OFF
                -DWEBP_BUILD_DWEBP=OFF
                -DWEBP_BUILD_GIF2WEBP=OFF
                -DWEBP_BUILD_IMG2WEBP=OFF
                -DWEBP_BUILD_VWEBP=OFF
                -DWEBP_BUILD_ANIM_UTILS=OFF
                -DWEBP_BUILD_WEBPINFO=OFF
                -DWEBP_BUILD_WEBPMUX=OFF
                -DWEBP_BUILD_EXTRAS=OFF
            BUILD_COMMAND ${codec_build_command}
            BUILD_BYPRODUCTS ${webp_libraries}
            INSTALL_COMMAND "")
    endif()

    foreach(codec IN LISTS bundled_codecs)
        add_dependencies(h3-ncnn-vulkan deps-${codec})
        foreach(notice LICENSE LICENSE.md LICENSE.txt LICENSE.TXT LICENSE.MIT COPYING COPYING.md PATENTS PATENTS.TXT NOTICE README.ijg)
            set(source "${CMAKE_CURRENT_SOURCE_DIR}/${codec}/${notice}")
            if(EXISTS "${source}")
                configure_file("${source}" "${codec_prefix}/licenses/${codec}-${notice}" COPYONLY)
            endif()
        endforeach()
    endforeach()
endif()

set(codec_includes "${codec_prefix}/include")
if(H3_MEDIA_PREFIX AND system_codec_suffix)
    # Keep bundled headers for the libraries linked from this prefix.
    set(codec_includes "${CMAKE_CURRENT_BINARY_DIR}/codec-include/system-${system_codec_suffix}")
    set(exclude_headers "")
    foreach(header IN LISTS system_headers)
        list(APPEND exclude_headers PATTERN "${header}" EXCLUDE)
    endforeach()
    file(COPY "${codec_prefix}/include/" DESTINATION "${codec_includes}" ${exclude_headers})
endif()

target_include_directories(h3-ncnn-vulkan SYSTEM PRIVATE "${codec_includes}" "${codec_includes}/webm")
install(DIRECTORY "${codec_prefix}/licenses/"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/h3-ncnn-vulkan OPTIONAL)
