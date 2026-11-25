// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "models.h"
#include <string>
namespace h3
{
constexpr int kImagePadToken = 151655;
constexpr int kVideoPadToken = 151656;

inline bool is_vision_pad(int id)
{
    return id == kImagePadToken || id == kVideoPadToken;
}

struct VisionResult
{
    ncnn::Mat merged;
    std::vector<ncnn::Mat> deepstack;
};
class VisionEncoder
{
public:
    explicit VisionEncoder(std::string& error, H3ModelSet& models);
    VisionResult encode(std::string& error, const ncnn::Mat& pixels, int grid_h, int grid_w);

private:
    H3ModelSet& models_;
    ncnn::Mat positions_;
};
ncnn::Mat multimodal_positions(std::string& error, const ncnn::Mat& ids, const ncnn::Mat& grids);
}
