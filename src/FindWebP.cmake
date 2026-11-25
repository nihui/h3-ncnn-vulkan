# SPDX-License-Identifier: Apache-2.0

find_path(WebP_INCLUDE_DIR NAMES webp/decode.h)
find_library(WebP_LIBRARY NAMES webp)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(WebP REQUIRED_VARS WebP_LIBRARY WebP_INCLUDE_DIR)

if(WebP_FOUND AND NOT TARGET WebP::WebP)
    add_library(WebP::WebP UNKNOWN IMPORTED)
    set_target_properties(WebP::WebP PROPERTIES
        IMPORTED_LOCATION "${WebP_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${WebP_INCLUDE_DIR}")
endif()

mark_as_advanced(WebP_INCLUDE_DIR WebP_LIBRARY)
