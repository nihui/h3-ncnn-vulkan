// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "vision.h"
#include <string>
namespace h3
{
class TextEncoder
{
public:
    explicit TextEncoder(H3ModelSet& models)
        : models_(models)
    {
    }
    ncnn::Mat encode(std::string& error, const ncnn::Mat& token_ids,
        const std::vector<VisionResult>& vision, const ncnn::Mat& grids);

private:
    H3ModelSet& models_;
};
}
