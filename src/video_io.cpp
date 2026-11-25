// SPDX-License-Identifier: Apache-2.0
#include "video_io.h"
#include "h3_config.h"
#include "matroska_input.h"
#include "runtime.h"
#include "vpx_video.h"
#include "webm_video.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <new>
#include <string_view>

namespace h3
{
bool load_video_frames(std::string& error, const std::string& path, int max_frames, uint64_t ram_budget,
    const PrepareVideoFrame& prepare, std::vector<ncnn::Mat>& frames)
{
    frames.clear();
    if (max_frames <= 0)
        return media_error(error, "invalid video reference length");

    MatroskaInput input(path);
    if (!input.open(error, path))
        return false;
    const auto* track = input.track(mkvparser::Track::kVideo);
    const std::string_view codec = track && track->GetCodecId() ? track->GetCodecId() : "";
    if (codec != "V_VP8" && codec != "V_VP9")
        return media_error(error, "reference video requires a WebM or MKV VP8/VP9 track: " + path);
    VpxVideoDecoder decoder;
    if (!decoder.open(track->GetCodecId()))
        return media_error(error, "cannot initialize VPx decoder");

    ncnn::Mat previous;
    int start = 0;
    unsigned int source_width = 0, source_height = 0;
    long long first_timestamp = 0;
    long long last_interval = 0;
    long long last_timestamp = 0;
    if (!input.read(error, *track,
            [&](const std::vector<uint8_t>& packet, const mkvparser::Block& block, const mkvparser::Cluster* cluster,
                int i)
            {
                if (vpx_codec_decode(&decoder.codec, packet.data(), (unsigned int)packet.size(), nullptr, 0)
                    != VPX_CODEC_OK)
                    return media_error(error, "cannot decode VPx packet");
                const long long base = block.GetTime(cluster);
                const auto duration = track->GetDefaultDuration();
                if (base < 0
                    || duration > (uint64_t)std::numeric_limits<long long>::max() / std::max(1, block.GetFrameCount())
                    || base > std::numeric_limits<long long>::max() - (long long)(duration * i))
                    return media_error(error, "invalid video timestamp");
                const auto timestamp = base + (long long)(duration * i);
                vpx_codec_iter_t iterator = nullptr;
                while (const auto* image = vpx_codec_get_frame(&decoder.codec, &iterator))
                {
                    if (previous.empty())
                        first_timestamp = last_timestamp = timestamp;
                    if (timestamp > last_timestamp)
                        last_interval = timestamp - last_timestamp;
                    last_timestamp = timestamp;
                    const auto seconds = (long double)(timestamp - first_timestamp) / kNanoseconds;
                    const int end = (int)std::lround(std::clamp<long double>(seconds * kVideoFps, start, max_frames));
                    if (!previous.empty())
                    {
                        frames.insert(frames.end(), end - start, previous);
                        start = end;
                    }
                    if (start == max_frames)
                        return false;
                    if (source_width && (image->d_w != source_width || image->d_h != source_height))
                        return media_error(error, "reference video changes dimensions");
                    source_width = image->d_w;
                    source_height = image->d_h;
                    if (!check_ram_headroom(
                            error, ram_budget, uint64_t(source_width) * source_height * 3, "reference video RGB frame"))
                        return false;
                    auto rgb = vpx_rgb_frame(error, *image);
                    if (rgb.empty())
                        return false;
                    previous = prepare(error, rgb);
                    if (previous.empty())
                        return false;
                }
                return true;
            }))
        return false;
    if (!previous.empty() && start < max_frames)
    {
        const long long interval = track->GetDefaultDuration() ? (long long)track->GetDefaultDuration() : last_interval;
        const long double duration = (long double)(last_timestamp - first_timestamp) + interval;
        const int end = (int)std::lround(
            std::clamp<long double>(duration * kVideoFps / kNanoseconds, std::max(1, start), max_frames));
        frames.insert(frames.end(), end - start, previous);
    }
    return frames.size() >= 13 || media_error(error, "a reference video needs at least 13 frames at 24 fps");
}

VideoFormat video_output_format(const std::string& path)
{
    auto ext = std::filesystem::u8path(path).extension().string();
    for (char& c : ext)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (ext == ".webm")
        return VideoFormat::WebM;
    if (ext == ".mkv")
        return VideoFormat::Matroska;
    return VideoFormat::Unknown;
}

const char* video_error_message(int status)
{
    switch (status)
    {
    case VideoOk:
        return "success";
    case VideoUnsupportedFormat:
        return "output must end in .webm for lossy or .mkv for lossless video";
    case VideoInvalidDimensions:
        return "invalid video output dimensions or frame count";
    case VideoInvalidAudio:
        return "expected a finite stereo FP32 waveform";
    case VideoInvalidFrame:
        return "unexpected RGB frame size or count";
    case VideoClosed:
        return "video output is closed";
    case VideoIncomplete:
        return "incomplete video from VAE";
    case VideoAllocationFailed:
        return "cannot allocate video output buffers";
    case VideoFileError:
        return "cannot create, write or publish the video file";
    case VideoEncodingFailed:
        return "VP9 encoding failed";
    case VideoAudioEncodingFailed:
        return "Opus encoding failed";
    case VideoMuxingFailed:
        return "cannot write the video container";
    default:
        return "video output failed";
    }
}

MediaWriter::MediaWriter() = default;
MediaWriter::~MediaWriter() = default;

int MediaWriter::open(const std::string& path, int width, int height, int frames)
{
    const auto format = video_output_format(path);
    if (format == VideoFormat::Unknown)
        return VideoUnsupportedFormat;
    std::unique_ptr<WebmVideoWriter> writer(new (std::nothrow) WebmVideoWriter);
    if (!writer)
        return VideoAllocationFailed;
    const int status = writer->open(
        std::filesystem::u8path(path), format == VideoFormat::Matroska, width, height, frames);
    if (status == VideoOk)
        writer_ = std::move(writer);
    return status;
}

int MediaWriter::append_audio(const std::array<ncnn::Mat, 2>& audio, int clip_frames)
{
    return writer_ ? writer_->append_audio(audio, clip_frames) : VideoClosed;
}

int MediaWriter::write_video(const ncnn::Mat& rgb)
{
    if (!writer_)
        return VideoClosed;
    const int status = writer_->write_video(rgb);
    if (status != VideoOk && status != VideoInvalidFrame)
        writer_.reset();
    return status;
}

int MediaWriter::finish()
{
    if (!writer_)
        return VideoClosed;
    const int status = writer_->finish();
    if (status != VideoIncomplete)
        writer_.reset();
    return status;
}
}
