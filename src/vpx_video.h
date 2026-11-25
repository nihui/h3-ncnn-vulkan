// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "h3_config.h"
#include "video_io.h"

#include <vpx/vp8cx.h>
#include <vpx/vp8dx.h>
#include <vpx/vpx_decoder.h>
#include <vpx/vpx_encoder.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>

namespace h3
{
inline void fill_vpx_image(vpx_image_t& img, const uint8_t* rgb, int width, int height, bool lossless)
{
    for (int y = 0; y < height; ++y)
    {
        const uint8_t* row = rgb + static_cast<size_t>(y) * width * 3;
        uint8_t* plane_y = img.planes[0] + y * img.stride[0];
        if (lossless)
        {
            uint8_t* plane_u = img.planes[1] + y * img.stride[1];
            uint8_t* plane_v = img.planes[2] + y * img.stride[2];
            for (int x = 0; x < width; x++)
            {
                plane_y[x] = row[3 * x + 1];
                plane_u[x] = row[3 * x + 2];
                plane_v[x] = row[3 * x];
            }
        }
        else
            for (int x = 0; x < width; x++)
                plane_y[x] = static_cast<uint8_t>(std::clamp(
                    (47 * row[3 * x] + 157 * row[3 * x + 1] + 16 * row[3 * x + 2] + 128) / 256 + 16, 16, 235));
    }
    if (lossless)
        return;
    for (int y = 0; y < height; y += 2)
        for (int x = 0; x < width; x += 2)
        {
            int r = 0, g = 0, b = 0;
            for (int dy = 0; dy < 2; dy++)
                for (int dx = 0; dx < 2; dx++)
                {
                    const auto pos = (static_cast<size_t>(y + dy) * width + x + dx) * 3;
                    r += rgb[pos];
                    g += rgb[pos + 1];
                    b += rgb[pos + 2];
                }
            img.planes[1][(y / 2) * img.stride[1] + x / 2]
                = static_cast<uint8_t>(std::clamp((128 * 1024 - 26 * r - 87 * g + 112 * b + 512) / 1024, 16, 240));
            img.planes[2][(y / 2) * img.stride[2] + x / 2]
                = static_cast<uint8_t>(std::clamp((128 * 1024 + 112 * r - 102 * g - 10 * b + 512) / 1024, 16, 240));
        }
}

class VpxVideoEncoder
{
public:
    ~VpxVideoEncoder()
    {
        if (initialized_)
            vpx_codec_destroy(&codec_);
    }

    int open(int width, int height, bool lossless)
    {
        auto* iface = vpx_codec_vp9_cx();
        vpx_codec_enc_cfg_t config {};
        if (vpx_codec_enc_config_default(iface, &config, 0) != VPX_CODEC_OK)
            return VideoEncodingFailed;
        config.g_w = width;
        config.g_h = height;
        config.g_profile = lossless ? 1 : 0;
        config.g_timebase = { 1, kVideoFps };
        config.g_threads = 4;
        config.g_lag_in_frames = 0;
        config.kf_min_dist = 0;
        config.kf_max_dist = 48;
        config.rc_end_usage = VPX_Q;
        if (lossless)
            config.rc_min_quantizer = config.rc_max_quantizer = 0;
        if (vpx_codec_enc_init(&codec_, iface, &config, 0) != VPX_CODEC_OK)
            return VideoEncodingFailed;
        initialized_ = true;
        if (vpx_codec_control(&codec_, VP8E_SET_CPUUSED, 5) != VPX_CODEC_OK
            || vpx_codec_control(&codec_, VP9E_SET_ROW_MT, 1u) != VPX_CODEC_OK
            || vpx_codec_control(&codec_, VP8E_SET_ENABLEAUTOALTREF, 0u) != VPX_CODEC_OK
            || vpx_codec_control(&codec_, VP9E_SET_COLOR_SPACE, lossless ? VPX_CS_SRGB : VPX_CS_BT_709) != VPX_CODEC_OK
            || vpx_codec_control(&codec_, VP9E_SET_COLOR_RANGE, lossless ? 1 : 0) != VPX_CODEC_OK)
            return VideoEncodingFailed;
        const auto quality = lossless ? vpx_codec_control(&codec_, VP9E_SET_LOSSLESS, 1u)
                                      : vpx_codec_control(&codec_, VP8E_SET_CQ_LEVEL, 30u);
        if (quality != VPX_CODEC_OK)
            return VideoEncodingFailed;
        image_.reset(vpx_img_alloc(nullptr, lossless ? VPX_IMG_FMT_I444 : VPX_IMG_FMT_I420, width, height, 32));
        if (!image_)
            return VideoAllocationFailed;
        lossless_ = lossless;
        return VideoOk;
    }

    template <class Write> int encode(const ncnn::Mat& rgb, int pts, const Write& write)
    {
        if (rgb.dims != 2 || rgb.w != static_cast<int>(image_->d_w) || rgb.h != static_cast<int>(image_->d_h)
            || rgb.elemsize != 3 || rgb.elempack != 3 || rgb.empty())
            return VideoInvalidFrame;
        fill_vpx_image(*image_, static_cast<const uint8_t*>(rgb.data), rgb.w, rgb.h, lossless_);
        if (vpx_codec_encode(&codec_, image_.get(), pts, 1, 0, VPX_DL_GOOD_QUALITY) != VPX_CODEC_OK)
            return VideoEncodingFailed;

        vpx_codec_iter_t it = nullptr;
        int emitted = 0;
        while (const auto* packet = vpx_codec_get_cx_data(&codec_, &it))
        {
            if (packet->kind != VPX_CODEC_CX_FRAME_PKT)
                continue;
            const auto& frame = packet->data.frame;
            if (frame.pts != pts)
                return VideoEncodingFailed;
            const int status
                = write(static_cast<const uint8_t*>(frame.buf), frame.sz, (frame.flags & VPX_FRAME_IS_KEY) != 0);
            if (status != VideoOk)
                return status;
            emitted++;
        }
        return emitted == 1 ? VideoOk : VideoEncodingFailed;
    }

private:
    vpx_codec_ctx_t codec_ {};
    std::unique_ptr<vpx_image_t, decltype(&vpx_img_free)> image_ { nullptr, vpx_img_free };
    bool initialized_ = false;
    bool lossless_ = false;
};

struct VpxVideoDecoder
{
    ~VpxVideoDecoder()
    {
        if (initialized)
            vpx_codec_destroy(&codec);
    }

    bool open(const char* id)
    {
        auto* interface = std::strcmp(id, "V_VP9") == 0 ? vpx_codec_vp9_dx() : vpx_codec_vp8_dx();
        const vpx_codec_dec_cfg_t config = { 4, 0, 0 };
        initialized = vpx_codec_dec_init(&codec, interface, &config, 0) == VPX_CODEC_OK;
        return initialized;
    }

    vpx_codec_ctx_t codec {};
    bool initialized = false;
};

inline ncnn::Mat vpx_rgb_frame(std::string& error, const vpx_image_t& image)
{
    if (!image.d_w || !image.d_h || image.d_w > 32768 || image.d_h > 32768 || !(image.fmt & VPX_IMG_FMT_PLANAR)
        || image.fmt == VPX_IMG_FMT_NV12 || image.x_chroma_shift > 1 || image.y_chroma_shift > 1
        || (image.cs == VPX_CS_SRGB && (image.x_chroma_shift || image.y_chroma_shift))
        || ((image.fmt & VPX_IMG_FMT_HIGHBITDEPTH) && (image.bit_depth < 8 || image.bit_depth > 12)))
    {
        error = "unsupported VPx image format";
        return {};
    }

    ncnn::Mat rgb((int)image.d_w, (int)image.d_h, (size_t)3, 3);
    if (rgb.empty())
    {
        error = "cannot allocate decoded RGB frame";
        return {};
    }

    const bool wide = image.fmt & VPX_IMG_FMT_HIGHBITDEPTH;
    const int bits = wide ? image.bit_depth : 8;
    const auto sample = [&](int plane, unsigned int y, unsigned int x) -> int
    {
        const auto* row = image.planes[plane] + (ptrdiff_t)y * image.stride[plane];
        return wide ? reinterpret_cast<const uint16_t*>(row)[x] : row[x];
    };
    const bool full = image.range == VPX_CR_FULL_RANGE;
    const int levels = (1 << bits) - 1;
    const double unit = 1 << (bits - 8);
    const double luma_scale = full ? 255.0 / levels : 255.0 / (219 * unit);
    const double chroma_scale = full ? 255.0 / levels : 255.0 / (224 * unit);
    const double luma_offset = full ? 0 : 16 * unit;
    const double chroma_offset = 128 * unit;
    const double kr = image.cs == VPX_CS_BT_709 ? 0.2126 : image.cs == VPX_CS_BT_2020 ? 0.2627 : 0.299;
    const double kb = image.cs == VPX_CS_BT_709 ? 0.0722 : image.cs == VPX_CS_BT_2020 ? 0.0593 : 0.114;
    const double kg = 1 - kr - kb;
    const auto to_byte = [](double value) -> uint8_t { return (uint8_t)std::clamp(std::lround(value), 0l, 255l); };
    for (unsigned int y = 0; y < image.d_h; y++)
    {
        auto* row = rgb.row<uint8_t>(y);
        for (unsigned int x = 0; x < image.d_w; x++)
        {
            if (image.cs == VPX_CS_SRGB)
            {
                row[3 * x] = to_byte(255.0 * sample(2, y, x) / levels);
                row[3 * x + 1] = to_byte(255.0 * sample(0, y, x) / levels);
                row[3 * x + 2] = to_byte(255.0 * sample(1, y, x) / levels);
            }
            else
            {
                const double v
                    = (sample(2, y >> image.y_chroma_shift, x >> image.x_chroma_shift) - chroma_offset) * chroma_scale;
                const double u
                    = (sample(1, y >> image.y_chroma_shift, x >> image.x_chroma_shift) - chroma_offset) * chroma_scale;
                const double luma = (sample(0, y, x) - luma_offset) * luma_scale;
                row[3 * x] = to_byte(luma + 2 * (1 - kr) * v);
                row[3 * x + 1] = to_byte(luma - (2 * kr * (1 - kr) * v + 2 * kb * (1 - kb) * u) / kg);
                row[3 * x + 2] = to_byte(luma + 2 * (1 - kb) * u);
            }
        }
    }
    return rgb;
}
}
