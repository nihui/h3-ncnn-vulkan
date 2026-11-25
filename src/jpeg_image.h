// SPDX-License-Identifier: Apache-2.0
// Adapted from the JPEG decoder in nihui/h3image-ncnn-vulkan.
#pragma once

// jpeg image decoder with libjpeg
#include "jpeglib.h"
#include <limits>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>

struct safe_jpeg_error_mgr
{
    struct jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
};

static void safe_jpeg_error_exit(j_common_ptr cinfo)
{
    struct safe_jpeg_error_mgr* myerr = (struct safe_jpeg_error_mgr*)(cinfo->err);
    longjmp(myerr->setjmp_buffer, 1);
}

static void safe_jpeg_emit_message(j_common_ptr, int) { }

static void safe_jpeg_output_message(j_common_ptr) { }

static unsigned char* jpeg_load(const unsigned char* buffer, int len, int* w, int* h, int* c)
{
    unsigned char* volatile pixeldata = 0;

    struct jpeg_decompress_struct cinfo {};
    struct safe_jpeg_error_mgr jerr {};

    cinfo.err = jpeg_std_error(&jerr.pub);
    cinfo.err->error_exit = safe_jpeg_error_exit;
    cinfo.err->emit_message = safe_jpeg_emit_message;
    cinfo.err->output_message = safe_jpeg_output_message;

    if (setjmp(jerr.setjmp_buffer))
    {
        jpeg_destroy_decompress(&cinfo);
        free(pixeldata);
        return NULL;
    }

    jpeg_create_decompress(&cinfo);

    jpeg_mem_src(&cinfo, (unsigned char*)buffer, len);

    if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK)
    {
        jpeg_destroy_decompress(&cinfo);
        return NULL;
    }

    // always output RGB 8bit
    cinfo.out_color_space = JCS_RGB;
    cinfo.output_components = 3;
    cinfo.data_precision = 8;

    int width = cinfo.image_width;
    int height = cinfo.image_height;

    if (width <= 0 || height <= 0 || width > 32768 || height > 32768 || (size_t)width > (std::numeric_limits<size_t>::max)() / height / 3)
    {
        jpeg_destroy_decompress(&cinfo);
        return NULL;
    }
    jpeg_start_decompress(&cinfo);
    pixeldata = (unsigned char*)malloc((size_t)width * height * 3);
    if (!pixeldata)
    {
        jpeg_destroy_decompress(&cinfo);
        return NULL;
    }

    while (cinfo.output_scanline < cinfo.output_height)
    {
        unsigned char* ptr = (unsigned char*)pixeldata + (size_t)cinfo.output_scanline * width * 3;
        jpeg_read_scanlines(&cinfo, &ptr, 1);
    }

    jpeg_finish_decompress(&cinfo);

    jpeg_destroy_decompress(&cinfo);

    *w = width;
    *h = height;
    *c = 3;

    return pixeldata;
}
