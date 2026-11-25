// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace h3
{
int video_latent_frames(int frames);
int audio_latent_frames(int frames);

class GaussianNoise
{
public:
    explicit GaussianNoise(uint64_t seed);
    void fill(float* values, size_t count);

private:
    std::mt19937 generator_;
};

std::vector<float> modulation_times(float video, float audio, bool reference_video, bool reference_audio);
std::vector<float> sigma_schedule(int steps, float shift);
std::vector<float> simple_sigma_schedule(int steps, float shift);
std::vector<float> pi_sigma_schedule(int steps, float shift, int adapter_steps);
void euler_step(float* samples, const float* velocity, size_t count, float sigma, float sigma_next);
}
