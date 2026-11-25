// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace h3
{
std::vector<float> resample_audio(const std::vector<float>& source, int source_rate, int target_rate, size_t limit);

bool load_audio_samples(std::string& error, const std::string& path, int max_samples, bool required,
    std::array<std::vector<float>, 2>& samples);
}
