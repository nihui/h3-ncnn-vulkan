// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "models.h"

namespace h3
{
ncnn::Mat encode_video(std::string& error, H3ModelSet& models, const std::vector<ncnn::Mat>& frames);
}
