// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "pipeline.h"

#include <mat.h>

#include <array>
#include <map>

namespace h3
{
struct Condition
{
    ReferenceKind kind = ReferenceKind::Image;
    std::vector<ncnn::Mat> frames;
    std::array<std::vector<float>, 2> samples;
    ncnn::Mat video;
    ncnn::Mat audio;
    bool last_frame = false;
};

struct VisionBlock
{
    std::array<ncnn::Mat, 2> frames;
    int width = 0;
    int height = 0;
};

struct PromptInput
{
    std::vector<int> ids, tags;
    std::vector<VisionBlock> blocks;
};

bool prepare_conditions(std::string& error, const GenerateRequest& request, int& width, int& height, int frames,
    uint64_t ram_budget, std::vector<Condition>& conditions, std::map<std::string, ncnn::Mat>* keyframes);
PromptInput prepare_prompt(
    std::string& error, const std::string& model, const std::string& prompt, const std::vector<Condition>& conditions);
ncnn::Mat prepare_vision_patches(std::string& error, const VisionBlock& block);
}
