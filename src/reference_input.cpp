// SPDX-License-Identifier: Apache-2.0
#include "reference_input.h"
#include "audio_io.h"
#include "image_io.h"
#include "runtime.h"
#include "tokenizer.h"
#include "transformer.h"
#include "video_io.h"
#include "vision.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <utility>

namespace h3
{
namespace
{
int aligned_size(double size)
{
    return std::max(32, (int)std::nearbyint(size / 32) * 32);
}

void resolve_canvas(int source_width, int source_height, int& width, int& height, int short_edge = 768)
{
    const double scale = std::min((double)short_edge / std::min(source_width, source_height),
        std::sqrt((768.0 * 1344.0) / ((double)source_width * source_height)));
    width = aligned_size(source_width * scale);
    height = aligned_size(source_height * scale);
}

float vision_pixel(unsigned char value)
{
    const float normalized = (value / 255.f - 0.5f) / 0.5f;
    uint32_t bits;
    std::memcpy(&bits, &normalized, sizeof(bits));
    bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
    float pixel;
    std::memcpy(&pixel, &bits, sizeof(pixel));
    return pixel;
}

std::pair<int, int> vision_size(int w, int h, int frames, bool video)
{
    const double area = (double)w * h;
    const double min_pixels = video ? 4096.0 : 65536.0;
    const double max_pixels = video ? 25165824.0 : 16777216.0;
    const int temporal = video ? (frames + 1) / 2 * 2 : 1;
    if (area * temporal < min_pixels)
    {
        const double scale = std::sqrt(min_pixels / (area * frames));
        return { (int)std::ceil(w * scale / 32) * 32, (int)std::ceil(h * scale / 32) * 32 };
    }
    if (area * temporal > max_pixels)
    {
        const double scale = std::sqrt(max_pixels / (area * frames));
        return { (int)std::floor(w * scale / 32) * 32, (int)std::floor(h * scale / 32) * 32 };
    }
    return { w, h };
}
}

bool prepare_conditions(std::string& error, const GenerateRequest& request, int& width, int& height, int frames,
    uint64_t ram_budget, std::vector<Condition>& conditions, std::map<std::string, ncnn::Mat>* keyframes)
{
    if (request.workflow == Workflow::KeyframesToVideo)
    {
        for (const auto& path : { request.first_image, request.last_image })
        {
            if (path.empty())
                continue;
            Condition condition;
            condition.last_frame = request.first_image.empty() || !conditions.empty();
            if (keyframes)
            {
                auto previous = keyframes->find(path);
                if (previous != keyframes->end())
                {
                    condition.frames.push_back(previous->second);
                    conditions.push_back(std::move(condition));
                    continue;
                }
            }
            ncnn::Mat image = load_rgb_image(error, path, ram_budget);
            if (image.empty())
                return false;
            if (width == 0)
            {
                if (image.w > 4 * image.h || image.h > 4 * image.w)
                {
                    error = "keyframe aspect ratio must be within 1:4 and 4:1";
                    return false;
                }
                resolve_canvas(image.w, image.h, width, height);
            }
            const bool cover = keyframes || !conditions.empty();
            int resize_width = width, resize_height = height;
            if (cover)
            {
                const double scale = std::max((double)width / image.w, (double)height / image.h);
                resize_width = std::max(width, (int)std::nearbyint(image.w * scale));
                resize_height = std::max(height, (int)std::nearbyint(image.h * scale));
            }
            const uint64_t scratch
                = 3 * (uint64_t(resize_width) * (resize_height + image.h) + uint64_t(width) * height);
            if (!check_ram_headroom(error, ram_budget, scratch, "keyframe resize"))
                return false;
            ncnn::Mat resized = cover ? cover_rgb_image(error, image, width, height)
                                      : resize_rgb_lanczos(error, image, width, height);
            if (resized.empty())
                return false;
            if (keyframes)
                keyframes->emplace(path, resized);
            condition.frames.push_back(resized);
            conditions.push_back(std::move(condition));
        }
    }
    if (!request.references.empty())
    {
        for (const auto& reference : request.references)
        {
            Condition condition;
            condition.kind = reference.kind;
            if (reference.kind == ReferenceKind::Image)
            {
                const ncnn::Mat image = load_rgb_image(error, reference.path, ram_budget);
                if (image.empty())
                    return false;
                if (image.w > 4 * image.h || image.h > 4 * image.w)
                {
                    error = "reference image aspect ratio must be within 1:4 and 4:1";
                    return false;
                }
                const double scale = (double)request.reference_short_edge / std::min(image.w, image.h);
                const int w = aligned_size(image.w * scale), h = aligned_size(image.h * scale);
                if (!check_ram_headroom(error, ram_budget, 3 * uint64_t(w) * (h + image.h), "reference image resize"))
                    return false;
                ncnn::Mat resized = resize_rgb_lanczos(error, image, w, h);
                if (resized.empty())
                    return false;
                condition.frames.push_back(resized);
            }
            else
            {
                const bool video = reference.kind == ReferenceKind::Video;
                if (!load_audio_samples(
                        error, reference.path, frames * kAudioSampleRate / kVideoFps, !video, condition.samples))
                    return false;
                if (video)
                {
                    const auto prepare = [&](std::string& message, const ncnn::Mat& source) -> ncnn::Mat
                    {
                        if (source.w > 4 * source.h || source.h > 4 * source.w)
                        {
                            message = "reference video aspect ratio must be within 1:4 and 4:1";
                            return {};
                        }
                        int w, h;
                        resolve_canvas(source.w, source.h, w, h, request.reference_video_short_edge);
                        if (!check_ram_headroom(
                                message, ram_budget, 3 * uint64_t(w) * (h + source.h), "reference video resize"))
                            return {};
                        return resize_rgb_lanczos(message, source, w, h);
                    };
                    if (!load_video_frames(error, reference.path, frames, ram_budget, prepare, condition.frames))
                        return false;
                }
            }
            conditions.push_back(std::move(condition));
        }
    }

    if (width == 0)
        resolve_canvas(16, 9, width, height);
    return true;
}

PromptInput prepare_prompt(
    std::string& error, const std::string& model, const std::string& prompt, const std::vector<Condition>& conditions)
{
    const auto tokenizer = BpeTokenizer::LoadFromModel(error, model);
    if (!error.empty())
        return {};
    PromptInput result;
    auto& ids = result.ids;
    auto& tags = result.tags;
    const auto append_prompt = [&](const std::string& text)
    {
        const auto tokens = tokenizer.encode(error, text);
        if (!error.empty())
            return false;
        ids.insert(ids.end(), tokens.begin(), tokens.end());
        tags.insert(tags.end(), tokens.size(), kTextTag);
        return true;
    };
    const int start_token = tokenizer.token_id("<|vision_start|>");
    const int end_token = tokenizer.token_id("<|vision_end|>");
    if (start_token < 0 || end_token < 0 || tokenizer.token_id("<|image_pad|>") != kImagePadToken
        || tokenizer.token_id("<|video_pad|>") != kVideoPadToken)
    {
        error = "incompatible MiniMax-H3 tokenizer vocabulary";
        return {};
    }

    int image_count = 0, video_count = 0, audio_count = 0;
    for (const auto& condition : conditions)
    {
        if (!condition.samples[0].empty()
            && !append_prompt("<Audio " + std::to_string(++audio_count) + ">: "))
            return {};
        if (condition.frames.empty())
            continue;

        const bool video = condition.kind == ReferenceKind::Video;
        const auto label = video ? "<Video " + std::to_string(++video_count) : "<Picture " + std::to_string(++image_count);
        if (!append_prompt(label + ">: "))
            return {};
        const int stride = kVideoFps / 2;
        const int sampled = video ? ((int)condition.frames.size() + stride - 1) / stride : 1;
        const auto& first = condition.frames[0];
        const auto size = vision_size(first.w, first.h, sampled, video);
        const int patches = size.first / 16 * (size.second / 16);
        for (int t = 0; t < sampled; t += 2)
        {
            const int next = std::min(t + 1, sampled - 1);
            if (video)
            {
                // Timestamps are means of two sample times on the 2 fps grid.
                char timestamp[48];
                std::snprintf(timestamp, sizeof(timestamp), "<%.1f seconds>", (t + next) / 4.0);
                if (!append_prompt(timestamp))
                    return {};
            }
            result.blocks.push_back(
                { { condition.frames[t * stride], condition.frames[next * stride] }, size.first, size.second });
            ids.push_back(start_token);
            ids.insert(ids.end(), patches / 4, video ? kVideoPadToken : kImagePadToken);
            ids.push_back(end_token);
            tags.insert(tags.end(), patches / 4 + 2, kVideoTag);
        }
    }

    if (!append_prompt(prompt))
        return {};
    if (ids.empty() || ids.size() > 32768)
    {
        error = "the conditioning sequence must contain from 1 to 32768 tokens";
        return {};
    }
    return result;
}

ncnn::Mat prepare_vision_patches(std::string& error, const VisionBlock& block)
{
    const int w = block.width, h = block.height;
    ncnn::Mat pixels(1536, (w / 16) * (h / 16));
    if (pixels.empty())
    {
        error = "cannot allocate vision patches";
        return {};
    }
    std::array<ncnn::Mat, 2> frames;
    frames[0] = resize_rgb_bicubic(error, block.frames[0], w, h);
    if (frames[0].empty())
        return {};
    frames[1] = block.frames[0].data == block.frames[1].data ? frames[0]
                                                            : resize_rgb_bicubic(error, block.frames[1], w, h);
    if (frames[1].empty())
        return {};

    float values[256];
    for (int i = 0; i < 256; i++)
        values[i] = vision_pixel((unsigned char)i);

    int row = 0;
    for (int by = 0; by < h; by += 32)
        for (int bx = 0; bx < w; bx += 32)
            for (int dy = 0; dy < 32; dy += 16)
                for (int dx = 0; dx < 32; dx += 16, row++)
                {
                    float* out = pixels.row(row);
                    for (int c = 0; c < 3; c++)
                        for (const auto& image : frames)
                            for (int y = 0; y < 16; y++)
                            {
                                const auto* src = image.row<unsigned char>(by + dy + y) + (bx + dx) * 3;
                                for (int x = 0; x < 16; x++)
                                    *out++ = values[src[x * 3 + c]];
                            }
                }
    return pixels;
}
}
