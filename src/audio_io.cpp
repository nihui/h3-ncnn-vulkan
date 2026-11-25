// SPDX-License-Identifier: Apache-2.0
#include "audio_io.h"
#include "h3_config.h"
#include "matroska_input.h"

#include <opus/opus.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string_view>

namespace h3
{
std::vector<float> resample_audio(const std::vector<float>& source, int source_rate, int target_rate, size_t limit)
{
    const auto length = source.size();
    if (!length || !limit)
        return {};
    if (source_rate == target_rate)
        return std::vector<float>(source.begin(), source.begin() + std::min(length, limit));

    constexpr int radius = 24;
    constexpr int phases = 256;
    constexpr double pi = 3.14159265358979323846;
    const double cutoff = 0.94 * std::min(1.0, (double)target_rate / source_rate);
    std::array<std::array<float, 2 * radius + 1>, phases + 1> table {};
    for (int phase = 0; phase <= phases; phase++)
    {
        auto& coefficients = table[phase];
        double total = 0;
        for (int k = -radius; k <= radius; k++)
        {
            const double x = k - (double)phase / phases;
            const double angle = cutoff * pi * x;
            const double window = 0.5 + 0.5 * std::cos(pi * x / (radius + 1));
            const double value = (angle == 0 ? 1.0 : std::sin(angle) / angle) * window;
            coefficients[k + radius] = (float)value;
            total += value;
        }
        for (auto& coefficient : coefficients)
            coefficient /= (float)total;
    }

    const uint64_t samples = (uint64_t(length) * target_rate + source_rate / 2) / source_rate;
    std::vector<float> result(std::min<uint64_t>(samples, limit));
    for (size_t i = 0; i < result.size(); i++)
    {
        const uint64_t position = uint64_t(i) * source_rate;
        const int64_t center = position / target_rate;
        const int phase = (position % target_rate * phases + target_rate / 2) / target_rate;
        const auto& coefficients = table[phase];
        double value = 0;
        for (int k = -radius; k <= radius; k++)
            value += source[std::clamp<int64_t>(center + k, 0, length - 1)] * coefficients[k + radius];
        result[i] = (float)value;
    }
    return result;
}

namespace
{
using Stereo = std::array<std::vector<float>, 2>;

struct PcmFormat
{
    int rate = 0;
    int channels = 0;
    int bits = 0;
    bool floating = false;
    bool unsigned_bytes = false;

    bool valid() const
    {
        return rate >= 4000 && rate <= 384000 && (channels == 1 || channels == 2)
            && (floating ? (bits == 32 || bits == 64) : (bits == 8 || bits == 16 || bits == 24 || bits == 32));
    }

    int stride() const
    {
        return channels * (bits / 8);
    }

    bool append(std::string& error, const uint8_t* bytes, size_t length, size_t limit, Stereo& samples) const
    {
        if (length % stride())
            return media_error(error, "invalid PCM packet alignment");
        const auto count = std::min(length / stride(), limit - samples[0].size());
        for (size_t i = 0; i < count; i++)
            for (int c = 0; c < channels; c++)
            {
                const auto* sample = bytes + i * stride() + c * (bits / 8);
                float value;
                if (floating && bits == 64)
                {
                    const uint64_t raw = (uint64_t(read_le(sample + 4, 4)) << 32) | read_le(sample, 4);
                    double converted;
                    std::memcpy(&converted, &raw, sizeof(raw));
                    value = (float)converted;
                }
                else
                {
                    const uint32_t raw = read_le(sample, bits / 8);
                    if (floating)
                        std::memcpy(&value, &raw, sizeof(raw));
                    else
                    {
                        const int64_t range = int64_t(1) << bits;
                        const int64_t midpoint = range / 2;
                        const int64_t signed_value
                            = unsigned_bytes ? (int64_t)raw - midpoint : (raw < midpoint ? raw : (int64_t)raw - range);
                        value = (float)((double)signed_value / midpoint);
                    }
                }
                if (!std::isfinite(value))
                    return media_error(error, "non-finite PCM sample");
                samples[c].push_back(value);
            }
        return true;
    }
};

size_t source_samples(int cap, int rate)
{
    return (uint64_t(cap) * rate + kAudioSampleRate - 1) / kAudioSampleRate;
}

void finish_audio(Stereo& output, Stereo& source, int channels, int rate, size_t limit)
{
    for (int c = 0; c < channels; c++)
        output[c] = resample_audio(source[c], rate, kAudioSampleRate, limit);
    if (channels == 1)
        output[1] = output[0];
}

bool wav_audio(std::string& error, MediaInputFile& file, int limit, Stereo& samples)
{
    std::array<uint8_t, 40> buffer {};
    if (file.Read(0, 12, buffer.data()) != 0 || std::memcmp(buffer.data() + 8, "WAVE", 4))
        return media_error(error, "invalid PCM WAV header");
    PcmFormat format;
    uint32_t data_size = 0;
    long long data_offset = -1;
    const auto riff_end = 8ll + read_le(buffer.data() + 4, 4);
    if (riff_end > file.size)
        return media_error(error, "truncated PCM WAV file");
    for (long long offset = 12; offset + 8 <= riff_end;)
    {
        if (file.Read(offset, 8, buffer.data()) != 0)
            return media_error(error, "cannot read WAV chunk");
        const auto length = read_le(buffer.data() + 4, 4);
        offset += 8;
        if (length > riff_end - offset)
            return media_error(error, "truncated WAV chunk");
        if (!std::memcmp(buffer.data(), "fmt ", 4))
        {
            if (length < 16 || file.Read(offset, std::min<size_t>(length, buffer.size()), buffer.data()) != 0)
                return media_error(error, "cannot read WAV audio format");
            auto type = read_le(buffer.data(), 2);
            if (type == 0xfffe && length >= 40 && read_le(buffer.data() + 16, 2) >= 22)
            {
                constexpr uint8_t guid_tail[] = { 0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71 };
                if (std::memcmp(buffer.data() + 28, guid_tail, sizeof(guid_tail)) == 0)
                    type = read_le(buffer.data() + 24, 4);
            }
            format = { (int)read_le(buffer.data() + 4, 4), (int)read_le(buffer.data() + 2, 2),
                (int)read_le(buffer.data() + 14, 2), type == 3, read_le(buffer.data() + 14, 2) == 8 };
            if ((type != 1 && type != 3) || !format.valid() || format.stride() != (int)read_le(buffer.data() + 12, 2))
                return media_error(error, "WAV reference requires mono or stereo PCM");
        }
        else if (!std::memcmp(buffer.data(), "data", 4) && data_offset < 0)
        {
            data_size = length;
            data_offset = offset;
        }
        if (data_offset >= 0 && format.valid())
            break;
        offset += length + (length & 1u);
    }
    if (data_offset < 0 || !format.valid() || data_size % format.stride())
        return media_error(error, "WAV reference has invalid PCM data");
    const auto count = std::min<size_t>(source_samples(limit, format.rate), data_size / format.stride());
    if (!count)
        return media_error(error, "WAV reference has no audio");
    std::vector<uint8_t> pcm(count * format.stride());
    Stereo source;
    if (file.Read(data_offset, (long)pcm.size(), pcm.data()) != 0)
        return media_error(error, "cannot read WAV samples");
    if (!format.append(error, pcm.data(), pcm.size(), count, source))
        return false;
    finish_audio(samples, source, format.channels, format.rate, limit);
    return true;
}

bool matroska_opus(
    std::string& error, MatroskaInput& input, const mkvparser::Track& track, int channels, int cap, Stereo& output)
{
    size_t private_size = 0;
    const auto* head = track.GetCodecPrivate(private_size);
    if (private_size < 19 || !head || std::memcmp(head, "OpusHead", 8) || head[8] > 15 || head[9] != channels
        || head[18] != 0)
        return media_error(error, "unsupported mono or stereo Opus header");

    int status = 0;
    std::unique_ptr<OpusDecoder, decltype(&opus_decoder_destroy)> decoder(
        opus_decoder_create(kOpusRate, channels, &status), opus_decoder_destroy);
    if (!decoder || status != OPUS_OK
        || opus_decoder_ctl(decoder.get(), OPUS_SET_GAIN((int16_t)read_le(head + 16, 2))) != OPUS_OK)
        return media_error(error, "cannot initialize Opus decoder");

    int skip = read_le(head + 10, 2);
    const auto limit = source_samples(cap, kOpusRate);
    Stereo source;
    std::array<float, 5760 * 2> decoded;
    if (!input.read(error, track,
            [&](const std::vector<uint8_t>& packet, const mkvparser::Block& block, const mkvparser::Cluster*, int i)
            {
                const int count = opus_decode_float(
                    decoder.get(), packet.data(), (opus_int32)packet.size(), decoded.data(), 5760, 0);
                if (count < 0)
                    return media_error(error, "cannot decode Opus packet");
                const int lead = std::min(count, skip);
                skip -= lead;
                const auto padding = block.GetDiscardPadding();
                const long double trim = (long double)padding * kOpusRate / kNanoseconds;
                int first = lead, end = count;
                if (trim < 0 && i == 0)
                    first = std::max(first, (int)std::lround(std::min<long double>(-trim, count)));
                if (trim > 0 && i + 1 == block.GetFrameCount())
                    end -= (int)std::lround(std::min<long double>(trim, count));
                if (end > first)
                    end = first + (int)std::min<size_t>(end - first, limit - source[0].size());
                for (int n = first; n < end; n++)
                    for (int c = 0; c < channels; c++)
                        source[c].push_back(decoded[n * channels + c]);
                return source[0].size() < limit;
            }))
        return false;
    finish_audio(output, source, channels, kOpusRate, cap);
    return true;
}
}

bool load_audio_samples(std::string& error, const std::string& path, int max_samples, bool required, Stereo& samples)
{
    samples = {};
    if (max_samples <= 0)
        return media_error(error, "invalid audio reference length");

    MediaInputFile file(path);
    uint8_t magic[4];
    if (file.Read(0, sizeof(magic), magic) != 0)
        return media_error(error, "cannot open reference audio: " + path);
    if (!std::memcmp(magic, "RIFF", 4))
        return wav_audio(error, file, max_samples, samples);

    MatroskaInput input(path);
    if (!input.open(error, path))
        return false;
    const auto* track = static_cast<const mkvparser::AudioTrack*>(input.track(mkvparser::Track::kAudio));
    if (!track)
        return !required || media_error(error, "no audio track in " + path);

    const auto channels = track->GetChannels();
    if (channels != 1 && channels != 2)
        return media_error(error, "reference audio must be mono or stereo");
    const std::string_view codec = track->GetCodecId() ? track->GetCodecId() : "";
    if (codec == "A_OPUS")
    {
        if (!matroska_opus(error, input, *track, (int)channels, max_samples, samples))
            return false;
    }
    else
    {
        const double rate = track->GetSamplingRate();
        const auto bits = track->GetBitDepth();
        if ((codec != "A_PCM/FLOAT/IEEE" && codec != "A_PCM/INT/LIT") || !std::isfinite(rate) || rate < 4000
            || rate > 384000 || bits < 8 || bits > 64)
            return media_error(error, "reference audio requires an Opus or little-endian PCM track: " + path);
        const PcmFormat format
            = { (int)std::lround(rate), (int)channels, (int)bits, codec == "A_PCM/FLOAT/IEEE", bits == 8 };
        if (!format.valid())
            return media_error(error, "unsupported PCM audio format");
        const auto limit = source_samples(max_samples, format.rate);
        Stereo source;
        if (!input.read(error, *track,
                [&](const std::vector<uint8_t>& packet, const mkvparser::Block&, const mkvparser::Cluster*, int) {
                    return format.append(error, packet.data(), packet.size(), limit, source)
                        && source[0].size() < limit;
                }))
            return false;
        finish_audio(samples, source, format.channels, format.rate, max_samples);
    }
    return !samples[0].empty() || !required || media_error(error, "no decoded audio in " + path);
}
}
