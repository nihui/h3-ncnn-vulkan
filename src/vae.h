// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "models.h"

#include <functional>

namespace h3
{
using WriteFrame = std::function<bool(std::string&, const ncnn::Mat&)>;

bool decode_video(std::string& error, H3ModelSet& models, ncnn::Mat& video, const WriteFrame& write_frame);
}
