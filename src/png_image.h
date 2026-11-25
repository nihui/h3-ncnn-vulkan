// SPDX-License-Identifier: Apache-2.0
// Adapted from the PNG decoder in nihui/h3image-ncnn-vulkan.
#pragma once

// png image decoder with libpng
#include "png.h"
#include <limits>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>

struct png_read_context
{
    const unsigned char* indata;
    unsigned int inlen;
    unsigned int current;
};

static void png_read_fn(png_structp png_ptr, png_bytep dst, png_size_t nread)
{
    struct png_read_context* prc = (struct png_read_context*)(png_get_io_ptr(png_ptr));
    if (prc->current > prc->inlen || nread > prc->inlen - prc->current)
        png_error(png_ptr, "truncated PNG data");

    memcpy(dst, prc->indata + prc->current, nread);
    prc->current += nread;
}

static unsigned char* png_load(const unsigned char* buffer, int len, int* w, int* h, int* c)
{
    if (len < 8 || png_sig_cmp(buffer, 0, 8))
        return NULL;

    unsigned char* volatile pixeldata = 0;
    png_bytepp volatile row_pointers = NULL;

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
        return NULL;

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return NULL;
    }

    if (setjmp(png_jmpbuf(png_ptr)))
    {
        free(row_pointers);
        free(pixeldata);
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return NULL;
    }

    struct png_read_context prc = { buffer, (unsigned int)len, 0 };
    png_set_read_fn(png_ptr, &prc, png_read_fn);

    png_read_info(png_ptr, info_ptr);

    png_set_interlace_handling(png_ptr);
    int width = png_get_image_width(png_ptr, info_ptr);
    int height = png_get_image_height(png_ptr, info_ptr);

    png_byte color_type = png_get_color_type(png_ptr, info_ptr);
    png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);

    // always output RGB(A) 8bit
    if (bit_depth == 16)
        png_set_strip_16(png_ptr);

    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png_ptr);

    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png_ptr);

    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png_ptr);

    int has_alpha = 0;
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
        has_alpha = 1;
    if (color_type == PNG_COLOR_TYPE_RGBA || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        has_alpha = 1;

    if (has_alpha)
        png_set_tRNS_to_alpha(png_ptr);

    png_read_update_info(png_ptr, info_ptr);

    int channels = has_alpha ? 4 : 3;

    if (width <= 0 || height <= 0 || width > 32768 || height > 32768 || (size_t)width > (std::numeric_limits<size_t>::max)() / height / channels
        || (size_t)height > (std::numeric_limits<size_t>::max)() / sizeof(png_bytep))
        png_error(png_ptr, "PNG dimensions exceed supported storage");
    pixeldata = (unsigned char*)malloc((size_t)width * height * channels);
    row_pointers = (png_bytepp)malloc((size_t)height * sizeof(png_bytep));
    if (!pixeldata || !row_pointers)
        png_error(png_ptr, "cannot allocate PNG image storage");
    for (int i = 0; i < height; i++)
        row_pointers[i] = pixeldata + (size_t)i * width * channels;

    png_read_image(png_ptr, row_pointers);

    png_read_end(png_ptr, info_ptr);

    free(row_pointers);
    row_pointers = NULL;

    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);

    *w = width;
    *h = height;
    *c = channels;

    return pixeldata;
}
