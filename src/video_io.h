// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <mat.h>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace h3
{
enum class VideoFormat
{
    Unknown,
    WebM,
    Matroska
};

enum VideoStatus
{
    VideoOk = 0,
    VideoUnsupportedFormat = -1,
    VideoInvalidDimensions = -2,
    VideoInvalidAudio = -3,
    VideoInvalidFrame = -4,
    VideoClosed = -5,
    VideoIncomplete = -6,
    VideoAllocationFailed = -7,
    VideoFileError = -8,
    VideoEncodingFailed = -9,
    VideoAudioEncodingFailed = -10,
    VideoMuxingFailed = -11
};

VideoFormat video_output_format(const std::string& path);
const char* video_error_message(int status);

using PrepareVideoFrame = std::function<ncnn::Mat(std::string&, const ncnn::Mat&)>;
bool load_video_frames(std::string& error, const std::string& path, int max_frames, uint64_t ram_budget,
    const PrepareVideoFrame& prepare, std::vector<ncnn::Mat>& frames);

class WebmVideoWriter;

class MediaWriter
{
public:
    MediaWriter();
    ~MediaWriter();

    int open(const std::string& path, int width, int height, int frames);
    int append_audio(const std::array<ncnn::Mat, 2>& audio, int clip_frames);
    int write_video(const ncnn::Mat& rgb);
    int finish();

private:
    std::unique_ptr<WebmVideoWriter> writer_;
};
}
