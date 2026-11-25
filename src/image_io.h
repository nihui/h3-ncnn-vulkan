// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <mat.h>
#include <cstdint>
#include <string>

namespace h3
{
ncnn::Mat load_rgb_image(std::string& error, const std::string& path, uint64_t ram_budget);
ncnn::Mat resize_rgb_lanczos(std::string& error, const ncnn::Mat& image, int width, int height);
ncnn::Mat resize_rgb_bicubic(std::string& error, const ncnn::Mat& image, int width, int height);
ncnn::Mat cover_rgb_image(std::string& error, const ncnn::Mat& image, int width, int height);
}
