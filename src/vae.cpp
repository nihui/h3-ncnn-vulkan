// SPDX-License-Identifier: Apache-2.0
#include "vae.h"
#include "vae_tiles.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace h3
{
namespace
{
ncnn::Mat decode_video_tile(
    std::string& error, const ncnn::Net& net, const ncnn::Mat& video, int start, const TileSpan& x, const TileSpan& y)
{
    const int frames = kVaeClipLatents + kVaeTailLatents;
    const int rows = frames * x.size * y.size;
    ncnn::Mat input(kVideoLatentChannels, rows), cosine(48, rows + 5, 1), sine(48, rows + 5, 1);
    if (input.empty() || cosine.empty() || sine.empty())
    {
        error = "cannot allocate video decoder inputs";
        return {};
    }

    float frequency[8];
    for (int i = 0; i < 8; i++)
        frequency[i] = 1.f / std::pow(100.f, (float)i / 8);
    for (int row = 0; row < rows + 5; row++)
    {
        float* cos_row = cosine.channel(0).row(row);
        float* sin_row = sine.channel(0).row(row);
        float pos[3] = {};
        if (row < rows)
        {
            const int t = row / (x.size * y.size), h = (row / x.size) % y.size, w = row % x.size;
            pos[0] = 2.f * ((t + 0.5f) / frames) - 1.f;
            pos[1] = 2.f * ((h + 0.5f) / y.size) - 1.f;
            pos[2] = 2.f * ((w + 0.5f) / x.size) - 1.f;
            float* values = input.row(row);
            for (int c = 0; c < kVideoLatentChannels; c++)
                values[c] = video.channel(c).depth(start + t).row(y.start + h)[x.start + w];
        }
        for (int axis = 0; axis < 3; axis++)
            for (int f = 0; f < 8; f++)
            {
                const int j = axis * 8 + f;
                const float angle = (6.28318530717958647692f * pos[axis]) * frequency[f];
                cos_row[j] = cos_row[j + 24] = std::cos(angle);
                sin_row[j] = sin_row[j + 24] = std::sin(angle);
            }
    }

    std::vector<ncnn::Mat> outputs;
    if (!run_model(error, net, { { "in0", input }, { "in1", cosine }, { "in2", sine } }, { "out0" }, outputs))
        return {};
    const ncnn::Mat& patches = outputs[0];
    if (patches.dims != 2 || patches.h != rows || patches.w != 3 * kVaeTemporalScale * kSpatialScale * kSpatialScale)
    {
        error = "unexpected video decoder output shape";
        return {};
    }

    ncnn::Mat clip(x.size * kSpatialScale, y.size * kSpatialScale, frames * kVaeTemporalScale, 3);
    if (clip.empty())
    {
        error = "cannot allocate decoded video tile";
        return {};
    }
    for (int t = 0, row = 0; t < frames; t++)
        for (int h = 0; h < y.size; h++)
            for (int w = 0; w < x.size; w++, row++)
            {
                const float* values = patches.row(row);
                for (int c = 0; c < 3; c++)
                    for (int pt = 0; pt < kVaeTemporalScale; pt++)
                        for (int py = 0; py < kSpatialScale; py++, values += kSpatialScale)
                        {
                            float* dest = clip.channel(c).depth(t * kVaeTemporalScale + pt).row(h * kSpatialScale + py);
                            std::memcpy(dest + w * kSpatialScale, values, kSpatialScale * sizeof(float));
                        }
            }
    return clip;
}
}

bool decode_video(std::string& error, H3ModelSet& models, ncnn::Mat& video, const WriteFrame& write_frame)
{
    const auto* net = models.prepare(error, Component::VideoDecoder);
    if (!net)
        return false;
    for (int c = 0; c < kVideoLatentChannels; c++)
    {
        float* channel = video.channel(c);
        for (int i = 0; i < video.w * video.h * video.d; i++)
            channel[i] = channel[i] * kVideoLatentStd[c] + kVideoLatentMean[c];
    }
    const auto horizontal = split_tiles(video.w), vertical = split_tiles(video.h);
    const int width = video.w * kSpatialScale, height = video.h * kSpatialScale;
    ncnn::Mat previous;
    ncnn::Mat frame(width, height, (size_t)3, 3);
    if (frame.empty())
    {
        error = "cannot allocate video frame";
        return false;
    }

    // Strip the head padding from both the main clip and the overlapping tail.
    const int tail_start = kVaeClipLatents * kVaeTemporalScale + kVaeHeadPadding;
    const int clip_length = (kVaeClipLatents + kVaeTailLatents) * kVaeTemporalScale;
    const int chunks = (video.d - kVaeTailLatents) / kVaeClipLatents;
    for (int i = 0; i < chunks; i++)
    {
        std::vector<ncnn::Mat> tiles;
        for (const auto& y : vertical)
            for (const auto& x : horizontal)
            {
                ncnn::Mat tile = decode_video_tile(error, *net, video, i * kVaeClipLatents, x, y);
                if (tile.empty())
                    return false;
                tiles.push_back(tile);
            }
        ncnn::Mat clip(width, height, clip_length, 3);
        if (clip.empty())
        {
            error = "cannot allocate decoded video clip";
            return false;
        }
        copy_tiles(clip, horizontal, vertical, tiles, kSpatialScale);
        const int count = kVaeClipFrames + (i + 1 == chunks ? kVaeTailFrames : 0);
        for (int t = 0; t < count; t++)
        {
            const int offset = t < kVaeClipFrames ? t + kVaeHeadPadding : tail_start + t - kVaeClipFrames;
            for (int c = 0; c < 3; c++)
            {
                const float* current = clip.channel(c).depth(offset);
                const float* old = i > 0 && t < kVaeTailFrames
                    ? static_cast<const float*>(previous.channel(c).depth(tail_start + t))
                    : nullptr;
                const float weight = (float)t / kVaeTailFrames;
                auto* dest = static_cast<unsigned char*>(frame.data);
                for (int p = 0; p < width * height; p++)
                {
                    const float value = old ? old[p] * (1.f - weight) + current[p] * weight : current[p];
                    const float rgb = value * kPixelStd[c] + kPixelMean[c];
                    if (!std::isfinite(rgb))
                    {
                        error = "non-finite video decoder output";
                        return false;
                    }
                    dest[p * 3 + c] = (unsigned char)std::lround(std::clamp(rgb, 0.f, 1.f) * 255.f);
                }
            }
            if (!write_frame(error, frame))
                return false;
        }
        previous = clip;
        std::fprintf(stderr, "video VAE %d/%d\n", i + 1, chunks);
    }
    return true;
}
}
