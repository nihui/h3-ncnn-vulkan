// SPDX-License-Identifier: Apache-2.0
#include "audio_vae.h"
#include "h3_config.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace h3
{
namespace
{
constexpr float kAudioLatentMean[] = {
    -0.0202116875f,
    0.387646648f,
    -0.043982798f,
    -0.285915149f,
    0.0817968621f,
    -0.357826414f,
    0.04062381f,
    -0.015525345f,
    -0.223362482f,
    0.182100684f,
    0.294177878f,
    -0.079011676f,
    -0.0568150728f,
    -0.369902822f,
    -0.316163156f,
    0.590595138f,
    -0.0521395681f,
    0.0136731603f,
    -0.0369164786f,
    0.0973266065f,
    -0.339466233f,
    -0.306856775f,
    -0.245045989f,
    -0.0346985245f,
    0.0286803218f,
    -0.212177793f,
    -0.167826317f,
    0.322128789f,
    -0.122305585f,
    0.435660493f,
    -0.0502599202f,
    0.397925838f,
};
constexpr float kAudioLatentStd[] = {
    1.68955242f,
    2.76263727f,
    1.79453443f,
    1.68016818f,
    1.63902265f,
    2.77882983f,
    1.76590901f,
    1.61997576f,
    2.63365256f,
    1.85393567f,
    2.50564979f,
    1.81101924f,
    1.95796578f,
    1.66854982f,
    1.49224693f,
    3.2986702f,
    1.94918045f,
    1.87200033f,
    1.83340801f,
    1.64880704f,
    1.61769577f,
    1.91314492f,
    1.56952454f,
    1.69436599f,
    1.83184208f,
    1.55406374f,
    1.93449303f,
    1.59919822f,
    1.71804599f,
    1.63072192f,
    1.86612261f,
    1.56137682f,
};
}

ncnn::Mat encode_audio(std::string& error, H3ModelSet& models, const std::array<std::vector<float>, 2>& audio)
{
    const auto* net = models.prepare(error, Component::AudioEncoder);
    if (!net)
        return {};
    const int sample_count = (int)audio[0].size();
    const int length = (sample_count + kAudioHop - 1) / kAudioHop;
    ncnn::Mat mask(length, length, 1), rows(kAudioLatentChannels, 2 * length);
    if (mask.empty() || rows.empty())
    {
        error = "cannot allocate audio conditioning latents";
        return {};
    }
    mask.fill(0.f);
    for (int t = 0; t < length; t++)
        std::fill(
            mask.channel(0).row(t) + t + 1, mask.channel(0).row(t) + length, -std::numeric_limits<float>::infinity());

    for (int stereo = 0; stereo < 2; stereo++)
    {
        ncnn::Mat input(length * kAudioHop, 1, 1);
        if (input.empty())
        {
            error = "cannot allocate audio encoder input";
            return {};
        }
        input.fill(0.f);
        std::copy(audio[stereo].begin(), audio[stereo].end(), static_cast<float*>(input.data));
        std::vector<ncnn::Mat> outputs;
        if (!run_model(error, *net, { { "in0", input }, { "in1", mask } }, { "out0" }, outputs))
            return {};
        const ncnn::Mat& moments = outputs[0];
        if (moments.w != length || moments.h != 1 || moments.c != 2 * kAudioLatentChannels)
        {
            error = "unexpected audio encoder output shape";
            return {};
        }
        for (int t = 0; t < length; t++)
        {
            float* output = rows.row(stereo * length + t);
            for (int c = 0; c < kAudioLatentChannels; c++)
                output[c] = (moments.channel(c)[t] - kAudioLatentMean[c]) / kAudioLatentStd[c];
        }
    }
    return rows;
}

std::array<ncnn::Mat, 2> decode_audio(
    std::string& error, H3ModelSet& models, const ncnn::Mat& rows, int start, int length)
{
    std::array<ncnn::Mat, 2> audio;
    const auto* net = models.prepare(error, Component::AudioDecoder);
    if (!net)
        return audio;
    for (int stereo = 0; stereo < 2; stereo++)
    {
        ncnn::Mat input(length, 1, kAudioLatentChannels);
        if (input.empty())
        {
            error = "cannot allocate audio decoder inputs";
            return {};
        }
        for (int c = 0; c < kAudioLatentChannels; c++)
        {
            float* channel = input.channel(c);
            for (int t = 0; t < length; t++)
                channel[t] = rows.row(start + stereo * length + t)[c] * kAudioLatentStd[c] + kAudioLatentMean[c];
        }

        std::vector<ncnn::Mat> outputs;
        if (!run_model(error, *net, { { "in0", input } }, { "out0" }, outputs))
            return {};
        const ncnn::Mat& waveform = outputs[0];
        if (waveform.w != length * kAudioHop || waveform.h != 1 || waveform.c != 1 || waveform.d != 1)
        {
            error = "unexpected audio decoder output shape";
            return {};
        }
        for (int i = 0; i < waveform.w; i++)
            if (!std::isfinite(waveform[i]))
            {
                error = "non-finite audio decoder output";
                return {};
            }
        audio[stereo] = waveform.reshape(waveform.w);
    }
    return audio;
}
}
