// SPDX-License-Identifier: Apache-2.0
#include "webm_video.h"
#include "audio_io.h"
#include "media_codec.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>

namespace h3
{
mkvmuxer::int32 WebmVideoWriter::OutputFile::Write(const void* data, mkvmuxer::uint32 size)
{
    stream.write(static_cast<const char*>(data), size);
    position += stream ? size : 0;
    return stream ? 0 : -1;
}

mkvmuxer::int64 WebmVideoWriter::OutputFile::Position() const
{
    return stream ? position : -1;
}

mkvmuxer::int32 WebmVideoWriter::OutputFile::Position(mkvmuxer::int64 offset)
{
    stream.seekp(offset);
    if (stream)
        position = offset;
    return stream ? 0 : -1;
}

WebmVideoWriter::~WebmVideoWriter()
{
    if (file_.stream.is_open())
        file_.stream.close();
    if (!temporary_.empty())
    {
        std::error_code ignored;
        std::filesystem::remove(temporary_ / "video", ignored);
        std::filesystem::remove(temporary_, ignored);
    }
}

int WebmVideoWriter::open(const std::filesystem::path& path, bool lossless, int width, int height, int frames)
{
    if (width <= 0 || height <= 0 || width % 2 || height % 2 || frames <= 0)
        return VideoInvalidDimensions;
    path_ = path;
    lossless_ = lossless;
    frames_ = frames;
    rate_ = lossless ? kAudioSampleRate : kOpusRate;

    std::error_code ec;
    if (!path_.parent_path().empty())
        std::filesystem::create_directories(path_.parent_path(), ec);
    if (ec)
        return VideoFileError;
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    for (int attempt = 0; attempt < 8; ++attempt)
    {
        auto directory = path_;
        directory += ".tmp." + std::to_string(nonce) + "." + std::to_string(attempt);
        if (std::filesystem::create_directory(directory, ec))
        {
            temporary_ = std::move(directory);
            break;
        }
        if (ec)
            return VideoFileError;
    }
    if (temporary_.empty())
        return VideoFileError;
    file_.stream.open(temporary_ / "video", std::ios::binary | std::ios::trunc);
    if (!file_.stream)
        return VideoFileError;

    const int status = encoder_.open(width, height, lossless_);
    if (status != VideoOk)
        return status;
    if (!mux_.Init(&file_))
        return VideoMuxingFailed;
    mux_.set_mode(mkvmuxer::Segment::kFile);
    mux_.OutputCues(true);
    const auto scale = lossless_ ? 125000u : 1000000u;
    mux_.GetSegmentInfo()->set_timecode_scale(scale);
    mux_.GetSegmentInfo()->set_writing_app("h3-ncnn-vulkan");
    mux_.set_duration((double)frames_ * kNanoseconds / kVideoFps / scale);
    video_track_ = mux_.AddVideoTrack(width, height, 0);
    audio_track_ = mux_.AddAudioTrack(rate_, 2, 0);
    if (!video_track_ || !audio_track_)
        return VideoMuxingFailed;
    mux_.GetTrackByNumber(video_track_)->set_codec_id("V_VP9");
    mux_.GetTrackByNumber(video_track_)->set_default_duration(media_ns(1, kVideoFps));
    auto& track = *static_cast<mkvmuxer::AudioTrack*>(mux_.GetTrackByNumber(audio_track_));
    return open_audio(track);
}

int WebmVideoWriter::open_audio(mkvmuxer::AudioTrack& track)
{
    if (lossless_)
    {
        track.set_codec_id("A_PCM/FLOAT/IEEE");
        track.set_bit_depth(32);
        return VideoOk;
    }

    int result = 0;
    opus_.reset(opus_encoder_create(kOpusRate, 2, OPUS_APPLICATION_AUDIO, &result));
    if (result != OPUS_OK || !opus_)
        return VideoAudioEncodingFailed;
    if (opus_encoder_ctl(opus_.get(), OPUS_SET_BITRATE(128000)) != OPUS_OK
        || opus_encoder_ctl(opus_.get(), OPUS_GET_LOOKAHEAD(&opus_delay_)) != OPUS_OK)
        return VideoAudioEncodingFailed;
    uint8_t head[19] = {'O', 'p', 'u', 's', 'H', 'e', 'a', 'd', 1, 2};
    write_le(head + 10, opus_delay_, 2);
    write_le(head + 12, kOpusRate, 4);
    track.set_codec_id("A_OPUS");
    if (!track.SetCodecPrivate(head, sizeof(head)))
        return VideoMuxingFailed;
    track.set_codec_delay(media_ns(opus_delay_, kOpusRate));
    track.set_seek_pre_roll(80000000);
    return VideoOk;
}

int WebmVideoWriter::append_audio(const std::array<ncnn::Mat, 2>& pcm, int clip_frames)
{
    // Each segment after the first shares its first frame with the previous segment.
    const bool skip_first = written_ != 0;
    if (audio_frames_ != written_ || clip_frames <= skip_first || clip_frames - skip_first > frames_ - written_)
        return VideoInvalidFrame;
    if (pcm[0].empty() || pcm[1].empty() || pcm[0].w != pcm[1].w)
        return VideoInvalidAudio;

    const int end = written_ + (clip_frames - skip_first);
    const bool joined = skip_first || end != frames_;
    const auto samples_between = [&](int rate) -> size_t
    {
        return (uint64_t(end) * rate / kVideoFps) - (uint64_t(written_) * rate / kVideoFps);
    };
    const size_t samples = samples_between(kAudioSampleRate);
    const size_t skip = skip_first ? kAudioSampleRate / kVideoFps : 0;
    if (static_cast<size_t>(pcm[0].w) <= skip)
        return VideoInvalidAudio;
    const auto available = std::min(samples, static_cast<size_t>(pcm[0].w) - skip);
    for (const auto& channel : pcm)
    {
        if (channel.dims != 1 || channel.elemsize != sizeof(float) || channel.elempack != 1)
            return VideoInvalidAudio;
        const auto* wave = static_cast<const float*>(channel.data) + skip;
        if (!std::all_of(wave, wave + available, [](float sample) { return std::isfinite(sample); }))
            return VideoInvalidAudio;
    }

    const size_t consumed = audio_pos_ - audio_offset_;
    for (int c = 0; c < 2; c++)
    {
        const auto* wave = static_cast<const float*>(pcm[c].data) + skip;
        std::vector<float> channel(wave, wave + available);
        if (joined)
        {
            const auto fade = std::min<size_t>(kAudioSampleRate / 200, available / 2);
            for (size_t i = 0; i < fade; i++)
            {
                const float gain = float(i) / fade;
                if (skip_first)
                    channel[i] *= gain;
                if (end != frames_ || available < samples)
                    channel[available - 1 - i] *= gain;
            }
            channel.resize(samples);
        }
        if (!lossless_)
        {
            channel = resample_audio(channel, kAudioSampleRate, rate_, std::numeric_limits<size_t>::max());
            if (joined)
                channel.resize(samples_between(rate_));
        }
        auto& queued = audio_[c];
        queued.erase(queued.begin(), queued.begin() + consumed);
        queued.insert(queued.end(), channel.begin(), channel.end());
    }
    audio_offset_ = audio_pos_;
    audio_frames_ = end;
    return write_audio(media_ns(written_, kVideoFps));
}

int WebmVideoWriter::write_pcm(size_t length)
{
    const auto count = std::min<size_t>(rate_ / 50, length - audio_pos_);
    std::array<uint8_t, kAudioSampleRate / 50 * 8> packet;
    for (size_t i = 0; i < count; i++)
        for (int c = 0; c < 2; c++)
        {
            uint32_t bits;
            std::memcpy(&bits, &audio_[c][audio_pos_ - audio_offset_ + i], sizeof(bits));
            write_le(packet.data() + i * 8 + c * 4, bits, 4);
        }
    if (!mux_.AddFrame(packet.data(), count * 8, audio_track_, media_ns(audio_pos_, rate_), true))
        return VideoMuxingFailed;
    audio_pos_ += count;
    return VideoOk;
}

int WebmVideoWriter::write_opus(size_t length)
{
    constexpr int samples_per_packet = kOpusRate / 50;
    std::array<float, samples_per_packet * 2> pcm {};
    for (size_t i = 0; i < samples_per_packet && audio_pos_ + i < audio_offset_ + audio_[0].size(); i++)
        for (int c = 0; c < 2; c++)
            pcm[i * 2 + c] = audio_[c][audio_pos_ - audio_offset_ + i];
    std::array<uint8_t, 4000> packet;
    const int bytes = opus_encode_float(opus_.get(), pcm.data(), samples_per_packet, packet.data(), packet.size());
    if (bytes <= 0)
        return VideoAudioEncodingFailed;
    const auto timestamp = media_ns(audio_pos_, rate_);
    audio_pos_ += samples_per_packet;
    const bool saved = audio_pos_ > length
        ? mux_.AddFrameWithDiscardPadding(packet.data(), bytes, media_ns(audio_pos_ - length, rate_),
            audio_track_, timestamp, true)
        : mux_.AddFrame(packet.data(), bytes, audio_track_, timestamp, true);
    return saved ? VideoOk : VideoMuxingFailed;
}

int WebmVideoWriter::write_audio(uint64_t stop)
{
    const size_t end = audio_offset_ + audio_[0].size();
    // Carry a partial packet across joins. The Opus delay is applied once for the final packet.
    const size_t length = audio_frames_ == frames_ ? end + opus_delay_ : end - end % (rate_ / 50);
    while (audio_pos_ < length && media_ns(audio_pos_, rate_) < stop)
    {
        const int status = lossless_ ? write_pcm(length) : write_opus(length);
        if (status != VideoOk)
            return status;
    }
    return VideoOk;
}

int WebmVideoWriter::write_video(const ncnn::Mat& rgb)
{
    if (written_ >= audio_frames_)
        return VideoInvalidFrame;
    const int status = encoder_.encode(rgb, written_, [&](const uint8_t* bytes, size_t size, bool key) {
        return mux_.AddFrame(bytes, size, video_track_, media_ns(written_, kVideoFps), key) ? VideoOk : VideoMuxingFailed;
    });
    if (status != VideoOk)
        return status;
    return write_audio(media_ns(++written_, kVideoFps));
}

int WebmVideoWriter::finish()
{
    if (written_ != frames_)
        return VideoIncomplete;
    const int status = write_audio(std::numeric_limits<uint64_t>::max());
    if (status != VideoOk)
        return status;
    if (!mux_.Finalize())
        return VideoMuxingFailed;
    file_.stream.close();
    if (!file_.stream)
        return VideoFileError;
    std::error_code ec;
    std::filesystem::rename(temporary_ / "video", path_, ec);
    return ec ? VideoFileError : VideoOk;
}
}
