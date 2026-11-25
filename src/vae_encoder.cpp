// SPDX-License-Identifier: Apache-2.0
#include "vae_encoder.h"
#include "scheduler.h"
#include "vae_tiles.h"

#include <algorithm>
#include <cmath>

namespace h3
{
namespace
{
float round_half(float value)
{
    if (value == 0.f || !std::isfinite(value))
        return value;
    const int exp = std::max(-14, std::ilogb(value)) - 10;
    return std::ldexp(std::nearbyint(std::ldexp(value, -exp)), exp);
}

ncnn::Mat encode_video_clip(std::string& error, const ncnn::Net& net, const std::vector<ncnn::Mat>& frames, int start,
    int frame_count, int clip_frames)
{
    const int width = frames[0].w / kSpatialScale, height = frames[0].h / kSpatialScale;
    const int latent_frames = (clip_frames + kVaeTemporalScale - 1) / kVaeTemporalScale;
    const auto horizontal = split_tiles(width), vertical = split_tiles(height);
    std::vector<ncnn::Mat> tiles;
    for (const auto& y : vertical)
        for (const auto& x : horizontal)
        {
            const int sx = x.start * kSpatialScale, sy = y.start * kSpatialScale;
            const int w = x.size * kSpatialScale, h = y.size * kSpatialScale;
            ncnn::Mat input = clip_frames == 1 ? ncnn::Mat(w, h, 3) : ncnn::Mat(w, h, clip_frames, 3);
            if (input.empty())
            {
                error = "cannot allocate video encoder input";
                return {};
            }
            for (int t = 0; t < clip_frames; t++)
            {
                const ncnn::Mat& frame = frames[std::min(start + t, frame_count - 1)];
                for (int c = 0; c < 3; c++)
                {
                    ncnn::Mat channel = input.channel(c).depth(t);
                    for (int yy = 0; yy < input.h; yy++)
                    {
                        const unsigned char* src = frame.row<unsigned char>(sy + yy) + sx * 3;
                        float* dest = channel.row(yy);
                        for (int xx = 0; xx < input.w; xx++)
                            dest[xx] = (src[xx * 3 + c] / 255.f - kPixelMean[c]) / kPixelStd[c];
                    }
                }
            }
            std::vector<ncnn::Mat> outputs;
            if (!run_model(error, net, { { "in0", input } }, { "out0" }, outputs))
                return {};
            const ncnn::Mat& tile = outputs[0];
            if (tile.w != x.size || tile.h != y.size || tile.d != latent_frames || tile.c != 2 * kVideoLatentChannels)
            {
                error = "unexpected video encoder output shape";
                return {};
            }
            tiles.push_back(tile);
        }

    ncnn::Mat posterior(width, height, latent_frames, 2 * kVideoLatentChannels);
    if (posterior.empty())
    {
        error = "cannot allocate video conditioning moments";
        return {};
    }
    copy_tiles(posterior, horizontal, vertical, tiles, 1);
    return posterior;
}
}

ncnn::Mat encode_video(std::string& error, H3ModelSet& models, const std::vector<ncnn::Mat>& frames)
{
    const bool spatial = frames.size() == 1;
    const auto* net = models.prepare(error, spatial ? Component::VideoEncoder : Component::TemporalVideoEncoder);
    if (!net)
        return {};

    const int available = (int)frames.size();
    const int aligned = std::max(1, (available - kVaeTailFrames) / kVaeClipFrames) * kVaeClipFrames + kVaeTailFrames;
    const int frame_count = spatial ? 1 : std::min(available, aligned);
    const int chunks = spatial ? 1 : (frame_count + kVaeClipFrames - 1) / kVaeClipFrames;
    const int width = frames[0].w / kSpatialScale, height = frames[0].h / kSpatialScale;
    const int latent_frames = spatial ? 1 : (chunks - 1) * kVaeClipLatents + kVaeTailLatents;
    const int channel_size = width * height * latent_frames;
    ncnn::Mat latent(width, height, latent_frames, kVideoLatentChannels);
    if (latent.empty())
    {
        error = "cannot allocate video conditioning latents";
        return {};
    }

    GaussianNoise noise(42);
    std::vector<float> samples((size_t)channel_size * kVideoLatentChannels);
    noise.fill(samples.data(), samples.size());
    for (int chunk = 0; chunk < chunks; chunk++)
    {
        const ncnn::Mat posterior
            = encode_video_clip(error, *net, frames, chunk * kVaeClipFrames, frame_count, spatial ? 1 : kVaeClipFrames);
        if (posterior.empty())
            return {};
        const int start = chunk * kVaeClipLatents * width * height;
        const int count = std::min(channel_size - start, posterior.d * width * height);
        for (int c = 0; c < kVideoLatentChannels; c++)
        {
            const float* mean = posterior.channel(c);
            const float* logvar = posterior.channel(c + kVideoLatentChannels);
            float* output = latent.channel(c);
            for (int i = 0; i < count; i++)
            {
                const float value = mean[i]
                    + std::exp(0.5f * std::clamp(logvar[i], -30.f, 20.f))
                        * samples[(size_t)c * channel_size + start + i];
                output[start + i] = (round_half(value) - kVideoLatentMean[c]) / kVideoLatentStd[c];
            }
        }
    }
    return latent;
}
}
