// SPDX-License-Identifier: Apache-2.0
#include "pipeline.h"
#include "audio_vae.h"
#include "condition_tables.h"
#include "lora.h"
#include "reference_input.h"
#include "reference_kv.h"
#include "runtime.h"
#include "scheduler.h"
#include "text_encoder.h"
#include "transformer.h"
#include "vae.h"
#include "vae_encoder.h"
#include "video_io.h"

#include <gpu.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>

namespace h3
{
namespace
{
using Clock = std::chrono::steady_clock;

struct Clip
{
    MemoryWorkload workload;
    PromptInput prompt;
    std::vector<Condition> conditions;
    std::vector<VisionResult> vision;
    ncnn::Mat grids, text, video, audio;
    std::array<ncnn::Mat, 2> waveform;
    int bank = 0;
    int audio_start = 0;
};

struct Bank
{
    ModulationBank data;
    std::vector<std::vector<float>> schedule;
    int preset = -1;
    bool needs_build = false;
};

bool encode_vision(std::string& error, H3ModelSet& models, std::vector<Clip>& clips)
{
    if (std::none_of(clips.begin(), clips.end(), [](const Clip& clip) { return !clip.prompt.blocks.empty(); }))
        return true;
    VisionEncoder encoder(error, models);
    if (!error.empty())
        return false;

    std::map<const void*, VisionResult> features;
    for (auto& clip : clips)
    {
        auto& blocks = clip.prompt.blocks;
        if (blocks.empty())
            continue;
        clip.grids.create(3, (int)blocks.size());
        if (clip.grids.empty())
        {
            error = "cannot allocate vision grids";
            return false;
        }
        for (size_t i = 0; i < blocks.size(); i++)
        {
            const auto& block = blocks[i];
            const void* key = clips.size() > 1 ? block.frames[0].data : nullptr;
            const int h = block.height / 16, w = block.width / 16;
            float* grid = clip.grids.row((int)i);
            grid[0] = 1.f;
            grid[1] = h;
            grid[2] = w;
            const auto cached = features.find(key);
            if (cached != features.end())
            {
                clip.vision.push_back(cached->second);
                continue;
            }

            const auto pixels = prepare_vision_patches(error, block);
            if (pixels.empty())
                return false;
            auto result = encoder.encode(error, pixels, h, w);
            if (!error.empty())
                return false;
            if (key)
                features.emplace(key, result);
            clip.vision.push_back(std::move(result));
        }
        blocks.clear();
    }
    return true;
}

bool encode_text(std::string& error, H3ModelSet& models, std::vector<Clip>& clips)
{
    TextEncoder encoder(models);
    for (auto& clip : clips)
    {
        const auto& ids = clip.prompt.ids;
        ncnn::Mat input((int)ids.size());
        if (input.empty())
        {
            error = "cannot allocate prompt tokens";
            return false;
        }
        for (int i = 0; i < input.w; i++)
            input[i] = (float)ids[i];
        clip.text = encoder.encode(error, input, clip.vision, clip.grids);
        if (clip.text.empty())
            return false;
        clip.vision = {};
        clip.grids.release();
        clip.prompt.ids = {};
    }
    return true;
}

bool encode_references(std::string& error, H3ModelSet& models, std::vector<Clip>& clips)
{
    struct EncodedFrame
    {
        ncnn::Mat rgb, latent;
    };
    for (ReferenceKind kind : { ReferenceKind::Image, ReferenceKind::Video })
    {
        std::map<const void*, EncodedFrame> encoded;
        for (auto& clip : clips)
            for (auto& condition : clip.conditions)
            {
                if (condition.kind != kind)
                    continue;
                const void* key = clips.size() > 1 ? condition.frames.front().data : nullptr;
                const auto cached = encoded.find(key);
                if (cached != encoded.end())
                    condition.video = cached->second.latent;
                else
                {
                    condition.video = encode_video(error, models, condition.frames);
                    if (condition.video.empty())
                        return false;
                    if (key)
                        encoded.emplace(key, EncodedFrame { condition.frames.front(), condition.video });
                }
                condition.frames.clear();
            }
    }
    for (auto& clip : clips)
        for (auto& condition : clip.conditions)
        {
            if (condition.samples[0].empty())
                continue;
            condition.audio = encode_audio(error, models, condition.samples);
            if (condition.audio.empty())
                return false;
            condition.samples = {};
        }
    return true;
}

void fill_video_noise(GaussianNoise& noise, ncnn::Mat& video, bool condition)
{
    const size_t channel_size = (size_t)video.w * video.h * video.d;
    std::vector<float> values(channel_size * video.c);
    noise.fill(values.data(), values.size());
    for (int c = 0; c < video.c; c++)
    {
        float* channel = video.channel(c);
        const float* samples = values.data() + c * channel_size;
        if (condition)
            for (size_t i = 0; i < channel_size; i++)
                channel[i] = kConditionTime * channel[i] + (1.f - kConditionTime) * samples[i];
        else
            std::copy_n(samples, channel_size, channel);
    }
}

double elapsed_seconds(Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

bool video_ok(int status, std::string& error)
{
    if (status == VideoOk)
        return true;
    error = video_error_message(status);
    return false;
}

int aligned_frames(std::string& error, int requested)
{
    if (requested > kMaxClipFrames)
    {
        error = "a segment supports up to " + std::to_string(kMaxClipFrames)
            + " frames. Use intermediate FL2VA keyframes for longer videos";
        return 0;
    }
    if (requested <= kVaeTailFrames)
    {
        error = "aligned frame count must include at least one VAE clip (22 frames)";
        return 0;
    }
    const int align = kVaeTailFrames + kVaeClipFrames - requested % kVaeClipFrames;
    return requested + align % kVaeClipFrames;
}

bool sample_clip(std::string& error, H3ModelSet& models, Component component, ModulationBank& bank,
    const std::vector<float>& video_sigmas, const std::vector<float>& audio_sigmas,
    uint64_t seed, bool shared_keyframes, Clip& clip)
{
    const auto& workload = clip.workload;
    const int width = workload.width, height = workload.height;
    const int latent_frames = video_latent_frames(workload.frames);
    const int audio_frames = workload.audio_frames;
    const bool reference = workload.ref2va;
    auto& conditions = clip.conditions;
    const int latent_w = width / kSpatialScale, latent_h = height / kSpatialScale;
    auto layout = make_transformer_layout(
        std::move(clip.prompt.tags), conditions, latent_w, latent_h, latent_frames, audio_frames, reference);
    ncnn::Mat video(4 * kVideoLatentChannels, layout.video_rows),
        audio(kAudioLatentChannels, layout.condition_audio_rows + 2 * audio_frames);
    ncnn::Mat initial(latent_w, latent_h, latent_frames, kVideoLatentChannels);
    if (video.empty() || audio.empty() || initial.empty())
    {
        error = "cannot allocate sampling latents";
        return false;
    }

    GaussianNoise noise(seed);
    int video_start = 0, audio_start = 0;
    for (auto& condition : conditions)
    {
        if (!condition.video.empty())
        {
            ncnn::Mat noisy = shared_keyframes ? condition.video.clone() : condition.video;
            if (noisy.empty())
            {
                error = "cannot allocate a keyframe latent";
                return false;
            }
            fill_video_noise(noise, noisy, true);
            patchify_video(noisy, video, video_start);
            video_start += condition.video.w * condition.video.h * condition.video.d / 4;
        }
        if (!condition.audio.empty())
        {
            std::memcpy(audio.row(audio_start), condition.audio.data,
                (size_t)condition.audio.w * condition.audio.h * sizeof(float));
            audio_start += condition.audio.h;
        }
    }
    conditions.clear();
    fill_video_noise(noise, initial, false);
    patchify_video(initial, video, video_start);
    initial.release();
    noise.fill(audio.row(audio_start), (size_t)audio.w * (audio.h - audio_start));

    std::fprintf(stderr, "packed rows %d text + %d video + %d audio\n", layout.text_rows, video.h, audio.h);
    Clock::time_point denoise_start;
    {
        Transformer transformer(models, component, layout, bank);
        denoise_start = Clock::now();
        for (int i = 0; i + 1 < (int)video_sigmas.size(); i++)
        {
            const auto step = Clock::now();
            ncnn::Mat video_velocity, audio_velocity;
            const float video_time = 1.f - video_sigmas[i], audio_time = 1.f - audio_sigmas[i];
            if (!transformer.predict(error, video, audio, clip.text, i, video_time, audio_time,
                    video_velocity, audio_velocity))
                return false;

            const size_t count = (size_t)(video.h - video_start) * video.w;
            euler_step(video.row(video_start), video_velocity.row(video_velocity.h - (video.h - video_start)), count,
                video_sigmas[i], video_sigmas[i + 1]);
            const int generated_audio = audio.h - audio_start;
            euler_step(audio.row(audio_start), audio_velocity.row(audio_velocity.h - generated_audio),
                (size_t)audio.w * generated_audio, audio_sigmas[i], audio_sigmas[i + 1]);
            std::fprintf(stderr, "DiT %d/%d %.2f s\n", i + 1, (int)video_sigmas.size() - 1, elapsed_seconds(step));
        }
    }
    layout = {};
    clip.text.release();
    if (auto* cache = models.reference_cache())
        cache->clear();
    std::fprintf(stderr, "DiT total %.2f s\n", elapsed_seconds(denoise_start));

    clip.audio = audio;
    clip.audio_start = audio_start;
    clip.video = unpatchify_video(video, video_start, latent_w, latent_h, latent_frames);
    if (clip.video.empty())
    {
        error = "cannot allocate video decoder latents";
        return false;
    }
    return true;
}
}

bool generate(std::string& error, const RuntimeConfig& config, const GenerateRequest& request)
{
    const auto start = Clock::now();
    const bool bake_only = config.bake_modulations >= 0;
    const auto output_format = video_output_format(request.output);
    if (!bake_only && output_format == VideoFormat::Unknown)
        return video_ok(VideoUnsupportedFormat, error);

    const bool joined = !request.segments.empty();
    const size_t count = joined ? request.segments.size() : 1;
    std::vector<Clip> clips(count);
    int64_t total_frames = 1;
    for (size_t i = 0; i < count; i++)
    {
        const int requested = joined ? request.segments[i].frames : request.frames;
        const int frames = bake_only ? 0 : aligned_frames(error, requested);
        if (!bake_only && !frames)
            return false;
        clips[i].workload.frames = frames;
        total_frames += frames - 1;
        if (total_frames > std::numeric_limits<int>::max())
        {
            error = "the joined video has too many frames";
            return false;
        }
    }
    constexpr float default_video_shift = 12.f;
    float video_shift = default_video_shift;
    constexpr float audio_shift = 3.f;
    const bool reference = request.workflow == Workflow::ReferenceToVideo;
    std::unique_ptr<Lora> lora;
    int steps = request.steps;
    int pi_steps = 0;
    if (!request.loras.empty())
    {
        lora = std::make_unique<Lora>(error, request.loras);
        if (!error.empty() || !lora->configure(error, config.model, reference))
            return false;
        if (!request.steps_explicit && lora->default_steps())
            steps = lora->default_steps();
        video_shift = lora->video_shift(video_shift, reference);
        pi_steps = lora->pi_steps();
        if (pi_steps < 0)
        {
            if (!request.steps_explicit || (steps != 3 && steps != 5))
            {
                error = "use the original pi-H3 adapter name or specify -l 3 or -l 5";
                return false;
            }
            pi_steps = steps - 1;
        }
    }
    const auto schedule_for = [&](float shift)
    {
        if (pi_steps)
            return pi_sigma_schedule(steps, shift, pi_steps);
        if (lora && lora->simple_schedule())
            return simple_sigma_schedule(steps, shift);
        return sigma_schedule(steps, shift);
    };
    const auto video_sigmas = schedule_for(video_shift);
    const auto audio_sigmas = schedule_for(audio_shift);
    const uint64_t kv_limit = config.rough_mode && steps > 2 ? uint64_t(config.kv_cache_mib) * kMiB : 0;
    H3ModelSet models(error, config.model, config.gpu, config.rough_mode, kv_limit);
    if (!error.empty())
        return false;
    models.set_lora(lora.get());
    const auto* device = models.device();

    const char* workflow = "t2va";
    if (request.workflow == Workflow::KeyframesToVideo)
        workflow = "fl2va";
    else if (reference)
        workflow = "ref2va";

    if (!bake_only)
    {
        const auto& prompt = joined ? request.segments.front().prompt : request.prompt;
        if (std::any_of(request.segments.begin(), request.segments.end(),
                [&](const KeyframeSegment& segment) { return segment.prompt != prompt; }))
        {
            for (size_t i = 0; i < count; i++)
                std::fprintf(stderr, "prompt[%zu] = %s\n", i + 1, request.segments[i].prompt.c_str());
        }
        else
            std::fprintf(stderr, "prompt = %s\n", prompt.c_str());
        std::fprintf(stderr, "output-path = %s\n", request.output.c_str());
    }
    std::fprintf(stderr, "model = %s\ngpu-id = %d\nworkflow = %s\nsteps = %d\n",
        config.model.c_str(), device->info.device_index(), workflow, steps);
    if (!bake_only)
        std::fprintf(stderr, "seed = %llu\nrough-mode = %d\n", (unsigned long long)request.seed, config.rough_mode);
    if (joined)
        std::fprintf(stderr, "FL2VA %zu segments, %lld output frames at %d fps\n", count, (long long)total_frames, kVideoFps);
    if (lora)
        std::fprintf(stderr, "LoRA sampling %s, %d DiT forwards, video shift %.4g, audio shift %.4g\n",
            pi_steps                      ? "pi-H3"
                : lora->simple_schedule() ? "simple Euler"
                                          : "H3 Euler",
            steps - 1, video_shift, audio_shift);
    const auto memory = query_memory(error, device);
    if (!error.empty())
        return false;

    std::array<Bank, 4> banks;
    int width = request.width, height = request.height;
    std::map<std::string, ncnn::Mat> keyframes;
    GenerateRequest clip_request = request;
    clip_request.segments.clear();
    for (size_t i = 0; i < count; i++)
    {
        auto& clip = clips[i];
        auto& workload = clip.workload;
        auto& conditions = clip.conditions;
        auto& prompt = clip.prompt;
        const int frames = workload.frames;
        workload.lora = lora.get();
        workload.base_ram = memory.process_resident;
        workload.ref2va = reference;
        workload.steps = steps;
        workload.rough_mode = config.rough_mode;
        workload.kv_cache_mib = config.kv_cache_mib;
        workload.lossless = output_format == VideoFormat::Matroska;
        workload.pi_modulations = pi_steps != 0;
        bool has_video = bake_only && (config.bake_modulations & 1);
        bool has_audio = bake_only && (config.bake_modulations & 2);
        if (!bake_only)
        {
            if (joined)
            {
                const auto& segment = request.segments[i];
                clip_request.first_image = segment.first_image;
                clip_request.last_image = segment.last_image;
                clip_request.prompt = segment.prompt;
            }
            const uint64_t preprocess_limit = config.ram_budget && !config.estimate_memory
                ? std::min(config.ram_budget, memory.ram_available)
                : memory.ram_available;
            if (!prepare_conditions(error, clip_request, width, height, frames, preprocess_limit,
                    conditions, joined ? &keyframes : nullptr))
                return false;
            std::fprintf(stderr, "canvas %dx%d, %d frames at %d fps\n", width, height, frames, kVideoFps);

            workload.width = width;
            workload.height = height;
            workload.frames = frames;
            workload.audio_frames = audio_latent_frames(frames);
            for (const auto& condition : conditions)
            {
                const auto& media = condition.frames;
                workload.references.push_back({ media.empty() ? 0 : media[0].w, media.empty() ? 0 : media[0].h,
                    (int)media.size(), (int)condition.samples[0].size() });
                has_video |= !media.empty();
                has_audio |= !condition.samples[0].empty();
            }

            prompt = prepare_prompt(error, config.model, clip_request.prompt, conditions);
            if (!error.empty())
                return false;
            workload.text_rows = (int)prompt.ids.size();
            for (const auto& block : prompt.blocks)
            {
                const int patches = block.width / 16 * (block.height / 16);
                workload.vision_patches += patches;
                workload.max_vision_patches = std::max(workload.max_vision_patches, patches);
                workload.resized_vision_bytes = std::max(workload.resized_vision_bytes,
                    uint64_t(block.width) * block.height * 3 * (block.frames[0].data == block.frames[1].data ? 1 : 2));
            }
        }
        clip.bank = int(has_video) + 2 * int(has_audio);
        auto& bank = banks[clip.bank];
        if (bank.schedule.empty())
        {
            for (int step = 0; step + 1 < steps; step++)
                bank.schedule.push_back(modulation_times(1.f - video_sigmas[step], 1.f - audio_sigmas[step], has_video, has_audio));

            const bool base_schedule = !pi_steps && video_shift == default_video_shift
                && (!lora || (!lora->simple_schedule() && lora->modulation_key().empty()));
            const bool pi_schedule = (pi_steps == 2 || pi_steps == 4) && steps == pi_steps + 1
                && video_shift == default_video_shift && !has_audio;
            bank.preset = base_schedule || pi_schedule ? clip.bank : -1;
            if (!bank.data.prepare(error, models, reference, bank.schedule, bank.preset,
                    bake_only && bank.preset >= 0, &bank.needs_build))
                return false;
            workload.prepare_modulations = bank.needs_build;
        }
        for (const auto& times : bank.schedule)
            workload.modulation_rows = std::max(workload.modulation_rows, (int)times.size());
    }
    keyframes.clear();

    const auto preprocessing_peak = process_peak_ram();
    std::vector<MemoryWorkload> workloads;
    workloads.reserve(clips.size());
    for (auto& clip : clips)
    {
        clip.workload.preprocessing_peak = preprocessing_peak;
        workloads.push_back(clip.workload);
    }
    auto plan = plan_memory(
        error, config.model, device, workloads, config.ram_budget, config.vram_budget, config.estimate_memory);
    if (!error.empty())
        return false;
    print_memory_plan(plan, config.estimate_memory);
    if (!plan.feasible)
    {
        if (!config.estimate_memory || config.ram_budget || config.vram_budget)
        {
            error = plan.failure;
            return false;
        }
        std::fprintf(stderr, "Selected plan exceeds currently available memory. %s\n", plan.failure.c_str());
    }
    models.set_memory_plan(std::move(plan));
    if (config.estimate_memory)
        return true;
    if (!bake_only)
    {
        if (!encode_vision(error, models, clips) || !encode_text(error, models, clips)
            || !encode_references(error, models, clips))
            return false;
        models.release();
        std::fprintf(stderr, "conditioning %.2f s\n", elapsed_seconds(start));
    }

    for (auto& bank : banks)
    {
        if (!bank.needs_build)
            continue;
        const auto bank_start = Clock::now();
        if (!bank.data.prepare(error, models, reference, bank.schedule, bank.preset, bake_only && bank.preset >= 0))
            return false;
        std::fprintf(stderr, "modulation bank %.2f s\n", elapsed_seconds(bank_start));
    }
    if (bake_only)
        return true;

    const auto component = reference ? Component::ReferenceTransformerCore : Component::TransformerCore;
    const auto load_start = Clock::now();
    if (!models.prepare(error, component))
        return false;
    std::fprintf(stderr, "DiT load %.2f s\n", elapsed_seconds(load_start));
    for (size_t i = 0; i < count; i++)
    {
        auto& clip = clips[i];
        if (joined)
            std::fprintf(stderr, "FL2VA segment %zu/%zu\n", i + 1, count);
        if (!sample_clip(error, models, component, banks[clip.bank].data, video_sigmas, audio_sigmas,
                request.seed, joined, clip))
            return false;
    }
    models.release();

    const auto decode_start = Clock::now();
    for (auto& clip : clips)
    {
        clip.waveform = decode_audio(error, models, clip.audio, clip.audio_start, clip.workload.audio_frames);
        clip.audio.release();
        if (clip.waveform[0].empty() || clip.waveform[1].empty())
            return false;
    }

    MediaWriter muxer;
    if (!video_ok(muxer.open(request.output, width, height, (int)total_frames), error))
        return false;
    for (size_t i = 0; i < count; i++)
    {
        auto& clip = clips[i];
        if (!video_ok(muxer.append_audio(clip.waveform, clip.workload.frames), error))
            return false;
        clip.waveform = {};
        bool skip_first = i != 0;
        if (!decode_video(error, models, clip.video, [&](std::string& frame_error, const ncnn::Mat& frame)
                {
                    if (skip_first)
                    {
                        skip_first = false;
                        return true;
                    }
                    return video_ok(muxer.write_video(frame), frame_error);
                }))
            return false;
        clip.video.release();
    }
    if (!video_ok(muxer.finish(), error))
        return false;
    std::fprintf(stderr, "VAE and media output %.2f s, total %.2f s\n", elapsed_seconds(decode_start), elapsed_seconds(start));
    return true;
}
}
