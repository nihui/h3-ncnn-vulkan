// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "h3_config.h"
#include "lora.h"
#include "runtime.h"

#include <cstdint>
#include <string>
#include <vector>

namespace h3
{
enum class Workflow
{
    TextToVideo,
    KeyframesToVideo,
    ReferenceToVideo
};

enum class ReferenceKind
{
    Image,
    Video,
    Audio
};

struct Reference
{
    ReferenceKind kind;
    std::string path;
};

struct KeyframeSegment
{
    std::string first_image, last_image;
    std::string prompt;
    int frames;
};

struct GenerateRequest
{
    Workflow workflow = Workflow::TextToVideo;
    std::string prompt;
    std::string first_image;
    std::string last_image;
    std::vector<Reference> references;
    std::vector<KeyframeSegment> segments;
    std::string output = "output.webm";
    std::vector<LoraInput> loras;
    int width = 0;
    int height = 0;
    int frames = kDefaultFrames;
    int steps = 50;
    bool steps_explicit = false;
    int reference_short_edge = 2048;
    int reference_video_short_edge = 768;
    uint64_t seed = 42;
};

bool generate(std::string& error, const RuntimeConfig& config, const GenerateRequest& request);
}
