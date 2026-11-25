// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "vpx_video.h"

#include <mkvmuxer/mkvmuxer.h>
#include <opus/opus.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

namespace h3
{
class WebmVideoWriter
{
public:
    ~WebmVideoWriter();

    int open(const std::filesystem::path& path, bool lossless, int width, int height, int frames);
    int append_audio(const std::array<ncnn::Mat, 2>& audio, int clip_frames);
    int write_video(const ncnn::Mat& rgb);
    int finish();

private:
    class OutputFile : public mkvmuxer::IMkvWriter
    {
    public:
        mkvmuxer::int32 Write(const void* data, mkvmuxer::uint32 size) override;
        mkvmuxer::int64 Position() const override;
        mkvmuxer::int32 Position(mkvmuxer::int64 offset) override;
        bool Seekable() const override { return true; }
        void ElementStartNotify(mkvmuxer::uint64, mkvmuxer::int64) override {}

        std::ofstream stream;
        mkvmuxer::int64 position = 0;
    };

    int open_audio(mkvmuxer::AudioTrack& track);
    int write_pcm(size_t length);
    int write_opus(size_t length);
    int write_audio(uint64_t stop);

    std::filesystem::path path_;
    std::filesystem::path temporary_;
    OutputFile file_;
    mkvmuxer::Segment mux_;
    VpxVideoEncoder encoder_;
    std::unique_ptr<OpusEncoder, decltype(&opus_encoder_destroy)> opus_ {nullptr, opus_encoder_destroy};
    std::array<std::vector<float>, 2> audio_;
    uint64_t video_track_ = 0;
    uint64_t audio_track_ = 0;
    size_t audio_pos_ = 0;
    size_t audio_offset_ = 0;
    int audio_frames_ = 0;
    int opus_delay_ = 0;
    int rate_ = 0;
    int frames_ = 0;
    int written_ = 0;
    bool lossless_ = false;
};
}
