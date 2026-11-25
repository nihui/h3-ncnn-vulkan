// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace h3
{
constexpr uint64_t kMiB = 1024 * 1024;
constexpr float kConditionTime = 0.999f;
constexpr int kDiTBlocks = 50;
constexpr int kDiTWidth = 5376;
constexpr int kVideoLatentChannels = 24;
constexpr int kAudioLatentChannels = 32;
constexpr int kSpatialScale = 16;
constexpr int kVaeTile = 16;
constexpr int kVaeTileOverlap = 4;
constexpr int kVaeTileStride = kVaeTile - kVaeTileOverlap;
constexpr int kAudioSampleRate = 32000;
constexpr int kAudioHop = 800;
constexpr int kVideoFps = 24;
constexpr int kDefaultFrames = 124;
constexpr int kMaxClipFrames = 345;

// The video VAE keeps five latent frames per 17-frame clip. The last
// clip contributes a five-frame tail, represented by two latent frames.
constexpr int kVaeClipFrames = 17;
constexpr int kVaeClipLatents = 5;
constexpr int kVaeTailFrames = 5;
constexpr int kVaeTailLatents = 2;
constexpr int kVaeTemporalScale = 4;
constexpr int kVaeHeadPadding = kVaeClipLatents * kVaeTemporalScale - kVaeClipFrames;
}
