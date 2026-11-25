// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "models.h"

#include <array>

namespace h3
{
ncnn::Mat encode_audio(std::string& error, H3ModelSet& models, const std::array<std::vector<float>, 2>& audio);
std::array<ncnn::Mat, 2> decode_audio(
    std::string& error, H3ModelSet& models, const ncnn::Mat& rows, int start, int length);
}
