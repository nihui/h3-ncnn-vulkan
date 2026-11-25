// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "h3_config.h"

#include <mat.h>

#include <vector>

namespace h3
{
constexpr float kPixelMean[] = { 0.485f, 0.456f, 0.406f };
constexpr float kPixelStd[] = { 0.229f, 0.224f, 0.225f };

constexpr float kVideoLatentMean[] = {
    0.858090341f,
    -0.960659146f,
    1.06616402f,
    -0.509032547f,
    -0.272758186f,
    -1.36754143f,
    -0.255325496f,
    -0.269075543f,
    -0.537684083f,
    -0.0464097299f,
    0.665737033f,
    0.196901277f,
    -0.546060801f,
    -0.403534204f,
    -0.236830249f,
    0.259284526f,
    -0.301339447f,
    0.211341992f,
    -1.12068486f,
    0.358193338f,
    -0.0422514379f,
    0.260482997f,
    0.228640929f,
    0.705603182f,
};
constexpr float kVideoLatentStd[] = {
    1.22237742f,
    1.27672637f,
    1.68317747f,
    1.75494552f,
    1.56362164f,
    2.19414353f,
    0.965313792f,
    1.0569886f,
    0.841948926f,
    0.772995293f,
    1.89559376f,
    0.946841836f,
    0.799680948f,
    0.449889004f,
    0.719739974f,
    0.693629324f,
    2.96109509f,
    2.76941991f,
    3.04961848f,
    2.10880542f,
    3.27622628f,
    3.1627357f,
    2.2816813f,
    2.61278439f,
};

struct TileSpan
{
    int start;
    int size;
    int overlap;
};

std::vector<TileSpan> split_tiles(int size);
void copy_tiles(ncnn::Mat& dst, const std::vector<TileSpan>& horizontal, const std::vector<TileSpan>& vertical,
    const std::vector<ncnn::Mat>& tiles, int scale);
}
