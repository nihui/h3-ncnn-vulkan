// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "models.h"

#include <array>

namespace h3
{
constexpr int kVideoTag = 0;
constexpr int kTextTag = 1;
constexpr int kAudioTag = 2;
class ModulationBank;
struct Condition;

struct TransformerLayout
{
    std::vector<std::array<double, 3>> positions;
    std::vector<int> tags;
    int text_rows = 0;
    int condition_video_rows = 0;
    int video_rows = 0;
    int condition_audio_rows = 0;
};

TransformerLayout make_transformer_layout(std::vector<int> tags, const std::vector<Condition>& conditions, int width,
    int height, int frames, int audio_frames, bool reference);
void patchify_video(const ncnn::Mat& video, ncnn::Mat& rows, int start);
ncnn::Mat unpatchify_video(const ncnn::Mat& rows, int start, int width, int height, int frames);

class Transformer
{
public:
    Transformer(H3ModelSet& models, Component component, const TransformerLayout& layout, ModulationBank& bank);
    bool predict(std::string& error, const ncnn::Mat& video, const ncnn::Mat& audio, const ncnn::Mat& text, int step,
        float video_time, float audio_time, ncnn::Mat& video_velocity, ncnn::Mat& audio_velocity);

private:
    H3ModelSet& models_;
    Component component_;
    ModulationBank& bank_;
    const TransformerLayout& layout_;
    ncnn::Mat cos_;
    ncnn::Mat sin_;
};
}
