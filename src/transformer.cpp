// SPDX-License-Identifier: Apache-2.0
#include "transformer.h"
#include "condition_tables.h"
#include "h3_config.h"
#include "reference_input.h"
#include "reference_kv.h"
#include "scheduler.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace h3
{
Transformer::Transformer(
    H3ModelSet& models, Component component, const TransformerLayout& layout, ModulationBank& bank)
    : models_(models)
    , component_(component)
    , bank_(bank)
    , layout_(layout)
{
    const int count = (int)layout.positions.size();
    cos_.create(96, count, 1);
    sin_.create(96, count, 1);
    float frequency[16];
    for (int j = 0; j < 16; j++)
        frequency[j] = 1.f / std::pow(10000.f, (float)j / 16);
    for (int i = 0; i < count; i++)
    {
        float* cosine = cos_.channel(0).row(i);
        float* sine = sin_.channel(0).row(i);
        for (int axis = 0; axis < 3; axis++)
            for (int j = 0; j < 16; j++)
            {
                const float angle = (float)layout.positions[i][axis] * frequency[j];
                const int offset = axis * 16 + j;
                cosine[offset] = cosine[offset + 48] = std::cos(angle);
                sine[offset] = sine[offset + 48] = std::sin(angle);
            }
    }
}

bool Transformer::predict(std::string& error, const ncnn::Mat& video, const ncnn::Mat& audio, const ncnn::Mat& text,
    int step, float video_time, float audio_time, ncnn::Mat& video_velocity, ncnn::Mat& audio_velocity)
{
    const int count = (int)layout_.tags.size();
    const int audio_start = layout_.text_rows + layout_.video_rows;
    std::vector<float> row_times(count, video_time);
    std::fill_n(
        row_times.begin() + layout_.text_rows, layout_.condition_video_rows, std::max(video_time, kConditionTime));
    std::fill(row_times.begin() + audio_start, row_times.end(), audio_time);
    std::fill_n(row_times.begin() + audio_start, layout_.condition_audio_rows, 1.f);
    const auto times = modulation_times(
        video_time, audio_time, layout_.condition_video_rows != 0, layout_.condition_audio_rows != 0);
    const int time_count = (int)times.size();
    ncnn::Mat selector(time_count, count), adaln(time_count * 3, count);
    if (selector.empty() || adaln.empty() || cos_.empty() || sin_.empty())
    {
        error = "cannot allocate transformer inputs";
        return false;
    }
    selector.fill(0.f);
    adaln.fill(0.f);
    for (int i = 0; i < count; i++)
    {
        const int index = (int)(std::lower_bound(times.begin(), times.end(), row_times[i]) - times.begin());
        selector.row(i)[index] = 1.f;
        adaln.row(i)[index * 3 + layout_.tags[i]] = 1.f;
    }

    ncnn::Mat active_video = video;
    ncnn::Mat active_audio = audio;
    ncnn::Mat cosine = cos_;
    ncnn::Mat sine = sin_;
    auto* cache = models_.reference_cache();
    bool reuse = false;
    int padding = 0;
    if (cache)
    {
        const int cached_video = cache->includes_references() ? layout_.condition_video_rows : 0;
        const int cached_audio = cache->includes_references() ? layout_.condition_audio_rows : 0;
        const int first_video = layout_.text_rows + cached_video;
        const int first_audio = audio_start + cached_audio;
        const int video_rows = video.h - cached_video;
        const int audio_rows = audio.h - cached_audio;
        if (!cache->begin(error, models_.device(), step, first_video, video_rows, cached_audio, audio_rows))
            return false;
        reuse = cache->reuse();
        if (reuse)
        {
            active_video = video.row_range(cached_video, video_rows);
            active_audio = audio.row_range(cached_audio, audio_rows);
            const int active_rows = video_rows + audio_rows;
            const int queries = ReferenceKVCache::query_rows(active_rows);
            padding = queries - active_rows;
            if (padding)
            {
                ncnn::Mat padded_audio(audio.w, audio_rows + padding);
                if (padded_audio.empty())
                {
                    error = "cannot pad rough-mode query rows";
                    return false;
                }
                std::memcpy(padded_audio.data, active_audio.data, (size_t)audio.w * audio_rows * sizeof(float));
                std::fill_n(padded_audio.row(audio_rows), audio.w * padding, 0.f);
                active_audio = padded_audio;
            }
            const auto gather = [&](const ncnn::Mat& full, float fill = 0.f)
            {
                ncnn::Mat value(full.w, queries);
                if (!value.empty())
                {
                    const size_t stride = full.w * sizeof(float);
                    const auto* src = static_cast<const char*>(full.data);
                    std::memcpy(value.data, src + first_video * stride, video_rows * stride);
                    std::memcpy(value.row(video_rows), src + first_audio * stride, audio_rows * stride);
                    if (padding)
                        std::fill_n(value.row(active_rows), full.w * padding, fill);
                }
                return value;
            };
            adaln = gather(adaln);
            selector = gather(selector);
            cosine = gather(cos_, 1.f).reshape(96, queries, 1);
            sine = gather(sin_).reshape(96, queries, 1);
            if (adaln.empty() || selector.empty() || cosine.empty() || sine.empty())
            {
                error = "cannot allocate rough-mode selectors and positions";
                return false;
            }
        }
    }
    std::vector<ModelInput> inputs = { { "video", active_video, true }, { "audio", active_audio, true },
        { "adaln_selector", adaln }, { "time_selector", selector }, { "cos", cosine }, { "sin", sine } };
    if (!reuse)
        inputs.push_back({ "text", text });
    if (!bank_.bind(error, step, inputs))
        return false;
    const auto* net = models_.prepare(error, component_);
    std::vector<ncnn::Mat> outputs;
    if (!net || !run_model(error, *net, inputs, { "video_velocity", "audio_velocity" }, outputs, reuse)
        || (cache && !cache->finish(error)))
        return false;
    video_velocity = outputs[0];
    audio_velocity = outputs[1];
    if (video_velocity.w != active_video.w || video_velocity.h != active_video.h || audio_velocity.w != active_audio.w
        || audio_velocity.h != active_audio.h || video_velocity.dims != 2 || audio_velocity.dims != 2)
    {
        error = "unexpected transformer output shape";
        return false;
    }
    if (padding)
    {
        audio_velocity = audio_velocity.row_range(0, audio_velocity.h - padding).clone();
        if (audio_velocity.empty())
        {
            error = "cannot allocate unpadded audio velocity";
            return false;
        }
    }
    return true;
}

namespace
{
double frame_span(int frame)
{
    return (5.0 / 3.0) * (frame % kVaeClipLatents == 0 ? 1.0 : 4.0);
}

std::vector<double> spatial_axis(int dim, double area_root)
{
    const double ratio = dim / area_root;
    const double start = (1.0 - ratio) / 2.0;
    std::vector<double> axis(dim / 2);
    const double step = ((start + ratio) - start) / axis.size();
    for (int i = 0; i < (int)axis.size(); i++)
        axis[i] = (start + i * step) * 32.0;
    return axis;
}

void add_frame(TransformerLayout& layout, double time, const std::vector<double>& ys, const std::vector<double>& xs)
{
    for (double y : ys)
        for (double x : xs)
        {
            layout.positions.push_back({ time, y, x });
            layout.tags.push_back(kVideoTag);
        }
}

void add_video_positions(
    TransformerLayout& layout, double time, int frames, const std::vector<double>& ys, const std::vector<double>& xs)
{
    double elapsed = 0.0;
    for (int t = 0; t < frames; t++)
    {
        add_frame(layout, time + elapsed, ys, xs);
        elapsed += frame_span(t);
    }
}

void add_audio_positions(TransformerLayout& layout, double time, int frames, const std::vector<double>& xs)
{
    for (int c = 0; c < 2; c++)
        for (int t = 0; t < frames; t++)
        {
            layout.positions.push_back({ time + t, 0.0, c == 0 ? xs.front() : xs.back() });
            layout.tags.push_back(kAudioTag);
        }
}
}

TransformerLayout make_transformer_layout(std::vector<int> tags, const std::vector<Condition>& conditions, int width,
    int height, int frames, int audio_frames, bool reference)
{
    TransformerLayout layout, audio_layout;
    layout.tags = std::move(tags);
    layout.text_rows = (int)layout.tags.size();
    for (int i = 0; i < layout.text_rows; i++)
        layout.positions.push_back({ (double)i, 0.0, 0.0 });
    const double area = std::sqrt((double)width * height);
    const auto xs = spatial_axis(width, area), ys = spatial_axis(height, area);

    double origin = layout.text_rows;
    double keyframe_origin = origin;
    if (!reference)
        for (const auto& condition : conditions)
            keyframe_origin += condition.audio.h / 2;
    double end = keyframe_origin;
    for (int t = 0; t < frames; t++)
        end += frame_span(t);
    end -= 5.0 / 3.0;

    for (const auto& condition : conditions)
    {
        const int sound = condition.audio.h / 2;
        if (condition.video.empty())
        {
            add_audio_positions(audio_layout, origin, sound, xs);
            origin += sound;
            continue;
        }
        const auto& video = condition.video;
        const double condition_area = std::sqrt((double)video.w * video.h);
        const auto cx = spatial_axis(video.w, condition_area);
        const auto cy = spatial_axis(video.h, condition_area);
        const double start = reference ? origin : keyframe_origin;
        add_video_positions(layout, condition.last_frame ? end : start, video.d, cy, cx);
        add_audio_positions(audio_layout, origin, sound, cx);
        if (reference)
        {
            double span = 1.0;
            if (condition.kind == ReferenceKind::Video)
            {
                span = 0.0;
                for (int t = 0; t < video.d; t++)
                    span += frame_span(t);
            }
            origin += std::max(span, (double)sound);
        }
    }
    layout.condition_video_rows = (int)layout.positions.size() - layout.text_rows;
    add_video_positions(layout, origin, frames, ys, xs);
    layout.video_rows = (int)layout.positions.size() - layout.text_rows;

    layout.condition_audio_rows = (int)audio_layout.positions.size();
    layout.positions.insert(layout.positions.end(), audio_layout.positions.begin(), audio_layout.positions.end());
    layout.tags.insert(layout.tags.end(), audio_layout.tags.begin(), audio_layout.tags.end());
    add_audio_positions(layout, origin, audio_frames, xs);
    return layout;
}

void patchify_video(const ncnn::Mat& video, ncnn::Mat& rows, int start)
{
    for (int t = 0, row = start; t < video.d; t++)
        for (int y = 0; y < video.h; y += 2)
            for (int x = 0; x < video.w; x += 2, row++)
            {
                float* out = rows.row(row);
                for (int c = 0; c < kVideoLatentChannels; c++)
                {
                    const ncnn::Mat frame = video.channel(c).depth(t);
                    *out++ = frame.row(y)[x];
                    *out++ = frame.row(y)[x + 1];
                    *out++ = frame.row(y + 1)[x];
                    *out++ = frame.row(y + 1)[x + 1];
                }
            }
}

ncnn::Mat unpatchify_video(const ncnn::Mat& rows, int start, int width, int height, int frames)
{
    ncnn::Mat video(width, height, frames, kVideoLatentChannels);
    if (video.empty())
        return {};
    for (int t = 0, row = start; t < frames; t++)
        for (int y = 0; y < height; y += 2)
            for (int x = 0; x < width; x += 2, row++)
            {
                const float* input = rows.row(row);
                for (int c = 0; c < kVideoLatentChannels; c++)
                {
                    ncnn::Mat frame = video.channel(c).depth(t);
                    frame.row(y)[x] = *input++;
                    frame.row(y)[x + 1] = *input++;
                    frame.row(y + 1)[x] = *input++;
                    frame.row(y + 1)[x + 1] = *input++;
                }
            }
    return video;
}
}
