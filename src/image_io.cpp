// SPDX-License-Identifier: Apache-2.0
#include "image_io.h"
#include "jpeg_image.h"
#include "png_image.h"
#include "runtime.h"
#include "webp_image.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <vector>

namespace h3
{
namespace
{
bool image_size(const uint8_t* bytes, size_t size, int& width, int& height)
{
    if (size >= 24 && !png_sig_cmp(bytes, 0, 8))
    {
        width = static_cast<int>(png_get_uint_32(bytes + 16));
        height = static_cast<int>(png_get_uint_32(bytes + 20));
        return width > 0 && height > 0;
    }
    if (size >= 12 && WebPGetInfo(bytes, size, &width, &height))
        return width > 0 && height > 0;
    if (size < 2 || bytes[0] != 0xff || bytes[1] != 0xd8)
        return false;

    jpeg_decompress_struct decoder {};
    safe_jpeg_error_mgr error {};
    decoder.err = jpeg_std_error(&error.pub);
    decoder.err->error_exit = safe_jpeg_error_exit;
    decoder.err->emit_message = safe_jpeg_emit_message;
    decoder.err->output_message = safe_jpeg_output_message;
    volatile bool created = false;
    if (setjmp(error.setjmp_buffer))
    {
        if (created)
            jpeg_destroy_decompress(&decoder);
        return false;
    }
    jpeg_create_decompress(&decoder);
    created = true;
    jpeg_mem_src(&decoder, bytes, static_cast<unsigned long>(size));
    const bool ok = jpeg_read_header(&decoder, TRUE) == JPEG_HEADER_OK;
    width = static_cast<int>(decoder.image_width);
    height = static_cast<int>(decoder.image_height);
    jpeg_destroy_decompress(&decoder);
    return ok && width > 0 && height > 0;
}

double sinc(double x)
{
    const double angle = 3.14159265358979323846 * x;
    return x == 0.0 ? 1.0 : std::sin(angle) / angle;
}

double lanczos(double x)
{
    return x < 3.0 ? sinc(x) * sinc(x / 3.0) : 0.0;
}

double bicubic(double x)
{
    if (x < 1.0)
        return ((1.5 * x - 2.5) * x) * x + 1.0;
    if (x < 2.0)
        return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
    return 0.0;
}

// Match PIL's separable 8-bit resampling, including the intermediate rounding.
constexpr int kResizePrecision = 22;

struct Weights
{
    int start;
    std::vector<int> coefficients;
};

std::vector<Weights> make_weights(int source, int target, int radius, double (*filter)(double))
{
    const double scale = (double)source / target;
    const double filter_scale = std::max(1.0, scale);
    const double support = radius * filter_scale;
    std::vector<Weights> result(target);
    for (int i = 0; i < target; i++)
    {
        const double center = (i + 0.5) * scale;
        const int first = std::max(0, (int)(center - support + 0.5));
        const int last = std::min(source, (int)(center + support + 0.5));
        std::vector<double> values(last - first);
        double sum = 0;
        for (int j = first; j < last; j++)
        {
            const double value = filter(std::abs((j - center + 0.5) / filter_scale));
            values[j - first] = value;
            sum += value;
        }
        result[i].start = first;
        for (double value : values)
            result[i].coefficients.push_back((int)std::lround(value / sum * (1 << kResizePrecision)));
    }
    return result;
}

ncnn::Mat resize_axis(
    std::string& error, const ncnn::Mat& image, int length, bool horizontal, int radius, double (*filter)(double))
{
    const int source = horizontal ? image.w : image.h;
    if (source == length)
        return image;
    const auto weights = make_weights(source, length, radius, filter);
    ncnn::Mat out(horizontal ? length : image.w, horizontal ? image.h : length, (size_t)3, 3);
    if (out.empty())
    {
        error = "cannot allocate resized image";
        return {};
    }
    const auto* src = static_cast<const unsigned char*>(image.data);
    auto* dst = static_cast<unsigned char*>(out.data);
    for (int y = 0; y < out.h; y++)
        for (int x = 0; x < out.w; x++)
        {
            const auto& w = weights[horizontal ? x : y];
            const int step = horizontal ? 3 : 3 * image.w;
            const size_t offset = ((size_t)(horizontal ? y : w.start) * image.w + (horizontal ? w.start : x)) * 3;
            for (int c = 0; c < 3; c++)
            {
                int sum = 1 << (kResizePrecision - 1);
                for (size_t i = 0; i < w.coefficients.size(); i++)
                    sum += src[offset + i * step + c] * w.coefficients[i];
                dst[((size_t)y * out.w + x) * 3 + c] = (unsigned char)std::clamp(sum >> kResizePrecision, 0, 255);
            }
        }
    return out;
}

ncnn::Mat resize(
    std::string& error, const ncnn::Mat& image, int width, int height, int radius, double (*filter)(double))
{
    if (image.empty() || image.elemsize != 3 || image.elempack != 3 || width < 1 || height < 1 || width > 32768
        || height > 32768)
    {
        error = "invalid image resize dimensions";
        return {};
    }
    ncnn::Mat horizontal = resize_axis(error, image, width, true, radius, filter);
    if (horizontal.empty())
        return {};
    return resize_axis(error, horizontal, height, false, radius, filter);
}
}

ncnn::Mat load_rgb_image(std::string& error, const std::string& path, uint64_t ram_budget)
{
    std::ifstream file(std::filesystem::u8path(path), std::ios::binary | std::ios::ate);
    const auto size = file.tellg();
    if (size < 8 || size > std::numeric_limits<int>::max())
    {
        error = "cannot open reference image: " + path;
        return {};
    }
    if (!check_ram_headroom(error, ram_budget, (uint64_t)size, "reference image input"))
        return {};
    file.seekg(0);
    std::vector<uint8_t> data((size_t)size);
    if (!file.read(reinterpret_cast<char*>(data.data()), size))
    {
        error = "cannot read reference image: " + path;
        return {};
    }
    int expected_w = 0, expected_h = 0;
    if (!image_size(data.data(), data.size(), expected_w, expected_h) || expected_w > 32768 || expected_h > 32768)
    {
        error = "cannot read PNG, JPEG or WebP image dimensions: " + path;
        return {};
    }
    // Decoder planes, decoded RGBA and the packed RGB destination can overlap.
    const uint64_t pixels = uint64_t(expected_w) * expected_h;
    if (!check_ram_headroom(error, ram_budget, pixels * (4 + 4 + 3), "reference image decode"))
        return {};
    int w = 0, h = 0, channels = 0;
    unsigned char* image = nullptr;
    const uint8_t* bytes = data.data();
    if (png_sig_cmp(bytes, 0, 8) == 0)
        image = png_load(bytes, (int)size, &w, &h, &channels);
    else if (bytes[0] == 0xff && bytes[1] == 0xd8)
        image = jpeg_load(bytes, (int)size, &w, &h, &channels);
    else if (size >= 12 && std::memcmp(bytes, "RIFF", 4) == 0 && std::memcmp(bytes + 8, "WEBP", 4) == 0)
        image = webp_load(bytes, (int)size, &w, &h, &channels);
    std::unique_ptr<unsigned char, decltype(&std::free)> decoded(image, std::free);
    if (!decoded || w <= 0 || h <= 0 || w > 32768 || h > 32768 || (channels != 3 && channels != 4))
    {
        error = "cannot decode PNG, JPEG or WebP reference image: " + path;
        return {};
    }
    // Keep the packed RGB byte layout used by the video and vision encoders.
    ncnn::Mat result(w, h, (size_t)3, 3);
    if (result.empty())
    {
        error = "cannot allocate reference RGB pixels";
        return {};
    }
    if (channels == 3)
        std::memcpy(result.data, decoded.get(), (size_t)w * h * 3);
    else
        for (int y = 0; y < h; y++)
        {
            auto* row = result.row<unsigned char>(y);
            const auto* src = decoded.get() + (size_t)y * w * 4;
            for (int x = 0; x < w; x++)
                std::memcpy(row + 3 * x, src + 4 * x, 3);
        }
    return result;
}

ncnn::Mat resize_rgb_lanczos(std::string& error, const ncnn::Mat& image, int width, int height)
{
    return resize(error, image, width, height, 3, lanczos);
}

ncnn::Mat resize_rgb_bicubic(std::string& error, const ncnn::Mat& image, int width, int height)
{
    return resize(error, image, width, height, 2, bicubic);
}

ncnn::Mat cover_rgb_image(std::string& error, const ncnn::Mat& image, int width, int height)
{
    const double scale = std::max((double)width / image.w, (double)height / image.h);
    const int scaled_width = std::max(width, (int)std::nearbyint(image.w * scale));
    const int scaled_height = std::max(height, (int)std::nearbyint(image.h * scale));
    const ncnn::Mat scaled = resize_rgb_lanczos(error, image, scaled_width, scaled_height);
    if (scaled.empty() || (scaled_width == width && scaled_height == height))
        return scaled;

    ncnn::Mat crop(width, height, (size_t)3, 3);
    if (crop.empty())
    {
        error = "cannot allocate cropped image";
        return {};
    }
    const int left = (scaled_width - width) / 2;
    const int top = (scaled_height - height) / 2;
    for (int y = 0; y < height; y++)
        std::memcpy(crop.row<unsigned char>(y), scaled.row<unsigned char>(y + top) + left * 3, (size_t)width * 3);
    return crop;
}
}
