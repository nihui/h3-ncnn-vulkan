// SPDX-License-Identifier: Apache-2.0
#include "scheduler.h"
#include "h3_config.h"

#include <algorithm>
#include <cmath>

namespace h3
{
int video_latent_frames(int frames)
{
    return (frames - kVaeTailFrames) / kVaeClipFrames * kVaeClipLatents + kVaeTailLatents;
}

int audio_latent_frames(int frames)
{
    constexpr int denominator = kVideoFps * kAudioHop;
    return (frames * kAudioSampleRate + denominator / 2) / denominator;
}

GaussianNoise::GaussianNoise(uint64_t seed)
    : generator_(static_cast<uint32_t>(seed))
{
}

void GaussianNoise::fill(float* values, size_t count)
{
    // Match the reference CPU generator's uniform draws and 8-pair Box-Muller layout.
    const auto uniform = [this]() { return (generator_() & 0xffffffu) * (1.f / 16777216.f); };
    const auto transform = [](float* data)
    {
        for (int i = 0; i < 8; i++)
        {
            const float radius = std::sqrt(-2.f * std::log(1.f - data[i]));
            const float angle = 6.28318530717958647692f * data[i + 8];
            data[i] = radius * std::cos(angle);
            data[i + 8] = radius * std::sin(angle);
        }
    };
    // H3's smallest latent tensor has at least 24 values.
    for (size_t i = 0; i < count; i++)
        values[i] = uniform();
    for (size_t i = 0; i + 16 <= count; i += 16)
        transform(values + i);
    if (count % 16 != 0)
    {
        float* tail = values + count - 16;
        for (int i = 0; i < 16; i++)
            tail[i] = uniform();
        transform(tail);
    }
}

std::vector<float> modulation_times(float video, float audio, bool reference_video, bool reference_audio)
{
    std::vector<float> times = { video, audio };
    if (reference_video)
        times.push_back(std::max(video, kConditionTime));
    if (reference_audio)
        times.push_back(1.f);
    std::sort(times.begin(), times.end());
    times.erase(std::unique(times.begin(), times.end()), times.end());
    return times;
}

std::vector<float> sigma_schedule(int steps, float shift)
{
    std::vector<float> sigmas(steps);
    const float step = -1.f / (steps - 1);
    for (int i = 0; i < steps; i++)
    {
        const float base = i < steps / 2 ? std::fma(step, (float)i, 1.f) : -step * (steps - 1 - i);
        sigmas[i] = shift * base / (1.f + (shift - 1.f) * base);
    }
    return sigmas;
}

std::vector<float> simple_sigma_schedule(int steps, float shift)
{
    constexpr int training_steps = 1000;
    std::vector<float> sigmas(steps);
    for (int i = 0; i < steps - 1; i++)
    {
        const float base = (training_steps - (i * training_steps) / (steps - 1)) / float(training_steps);
        sigmas[i] = shift * base / (1.f + (shift - 1.f) * base);
    }
    return sigmas;
}

std::vector<float> pi_sigma_schedule(int steps, float shift, int adapter_steps)
{
    const float maximum = adapter_steps == 2 ? 0.93f : 0.96f;
    const float stride = maximum / (steps - 2 + 0.6f);
    std::vector<float> sigmas(steps);
    for (int i = 0; i < steps - 1; i++)
    {
        const float base = std::fma(-stride, float(i), maximum);
        sigmas[i] = shift * base / (1.f + (shift - 1.f) * base);
    }
    return sigmas;
}

void euler_step(float* samples, const float* velocity, size_t count, float sigma, float sigma_next)
{
    const float ratio = sigma_next / sigma;
    const float sigma_from_timestep = 1.f - (1.f - sigma);
    for (size_t i = 0; i < count; i++)
    {
        const float denoised = samples[i] + sigma_from_timestep * velocity[i];
        samples[i] = ratio * samples[i] + (1.f - ratio) * denoised;
    }
}
}
