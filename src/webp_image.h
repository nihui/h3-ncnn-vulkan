// SPDX-License-Identifier: Apache-2.0
// Adapted from the WEBP decoder in nihui/h3image-ncnn-vulkan.
#pragma once

// webp image decoder with libwebp
#include "webp/decode.h"
#include <stdio.h>
#include <stdlib.h>

static unsigned char* webp_load(const unsigned char* buffer, int len, int* w, int* h, int* c)
{
    unsigned char* pixeldata = 0;

    WebPDecoderConfig config;
    WebPInitDecoderConfig(&config);

    if (WebPGetFeatures(buffer, len, &config.input) != VP8_STATUS_OK)
        return NULL;

    int width = config.input.width;
    int height = config.input.height;
    int channels = config.input.has_alpha ? 4 : 3;

    pixeldata = (unsigned char*)malloc((size_t)width * height * channels);
    if (!pixeldata)
        return NULL;

    config.output.colorspace = channels == 4 ? MODE_RGBA : MODE_RGB;

    config.output.u.RGBA.stride = width * channels;
    config.output.u.RGBA.size = (size_t)width * height * channels;
    config.output.u.RGBA.rgba = pixeldata;
    config.output.is_external_memory = 1;

    if (WebPDecode(buffer, len, &config) != VP8_STATUS_OK)
    {
        free(pixeldata);
        return NULL;
    }

    *w = width;
    *h = height;
    *c = channels;

    return pixeldata;
}
