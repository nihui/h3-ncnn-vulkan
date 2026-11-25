// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "h3_config.h"
#include "models.h"

#include <iosfwd>

namespace h3
{
constexpr int table_width(int i)
{
    return kDiTWidth * (i == kDiTBlocks ? 2 : 6 * 3);
}

ncnn::Mat modulation_time_features(const std::vector<float>& times);
bool build_modulations(std::string& error, H3ModelSet& models, bool reference,
    const std::vector<std::vector<float>>& schedule, const std::vector<ncnn::Mat>& features,
    const std::vector<const char*>& names, std::ostream& out);
}
