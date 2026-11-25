// SPDX-License-Identifier: Apache-2.0
#include "text_encoder.h"
#include "vision.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
namespace h3
{
namespace
{
ncnn::Mat lookup_embeddings(
    std::string& error, const std::string& path, const ncnn::Mat& ids, const std::vector<int>& replaced)
{
    // Read only requested rows. The table never becomes a full FP32 allocation.
    const int vocabulary = 151936, width = 5120;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
    {
        error = "cannot open text embedding table";
        return {};
    }
    const auto bytes = file.tellg();
    file.seekg(0);
    uint32_t tag = 0;
    file.read(reinterpret_cast<char*>(&tag), sizeof(tag));
    const bool bf16 = tag == 0x01348b83;
    if (!file || (tag != 0 && !bf16))
    {
        error = "unsupported text embedding storage tag";
        return {};
    }
    const size_t row_bytes = width * (bf16 ? 2u : 4u);
    if (bytes != static_cast<std::streamoff>(sizeof(tag) + vocabulary * row_bytes))
    {
        error = "invalid text embedding table size";
        return {};
    }
    ncnn::Mat hidden(width, ids.w, bf16 ? static_cast<size_t>(2) : static_cast<size_t>(4));
    if (hidden.empty())
    {
        error = "cannot allocate text embedding rows";
        return {};
    }
    std::vector<bool> skip(ids.w, false);
    for (int row : replaced)
        skip[row] = true;
    for (int i = 0; i < ids.w; i++)
    {
        if (skip[i])
            continue;
        file.seekg(sizeof(tag) + static_cast<size_t>(ids[i]) * row_bytes);
        void* destination = static_cast<unsigned char*>(hidden.data) + static_cast<size_t>(i) * row_bytes;
        file.read(static_cast<char*>(destination), row_bytes);
        if (!file)
        {
            error = "cannot read text embedding row";
            return {};
        }
    }
    return hidden;
}
}
ncnn::Mat TextEncoder::encode(std::string& error, const ncnn::Mat& ids,
    const std::vector<VisionResult>& vision, const ncnn::Mat& grids)
{
    if (ids.dims != 1 || ids.w <= 0)
    {
        error = "token_ids must have shape (tokens,)";
        return {};
    }
    const int count = ids.w;
    for (int i = 0; i < count; i++)
        if (!std::isfinite(ids[i]) || ids[i] < 0 || ids[i] >= 151936 || ids[i] != (int)ids[i])
        {
            error = "token out of range";
            return {};
        }
    ncnn::Mat positions;
    std::vector<int> visual_indices;
    if (!vision.empty())
    {
        int visual_rows = 0;
        for (const auto& block : vision)
            visual_rows += block.merged.h;
        for (int i = 0; i < count; i++)
            if (is_vision_pad((int)ids[i]))
                visual_indices.push_back(i);
        if ((int)visual_indices.size() != visual_rows)
        {
            error = "vision token count mismatch";
            return {};
        }
        positions = multimodal_positions(error, ids, grids);
        if (!error.empty())
            return {};
    }
    ncnn::Mat hidden
        = lookup_embeddings(error, models_.directory() + "/text_encoder/embedding.bin", ids, visual_indices);
    if (!error.empty())
        return {};

    size_t visual_row = 0;
    for (const auto& block : vision)
    {
        ncnn::Mat merged = block.merged;
        if (hidden.elembits() == 16)
        {
            ncnn::Mat native;
            ncnn::cast_float32_to_bfloat16(merged, native);
            if (native.empty())
            {
                error = "cannot allocate BF16 vision features";
                return {};
            }
            merged = native;
        }
        for (int i = 0; i < merged.h; i++, visual_row++)
            std::memcpy(static_cast<unsigned char*>(hidden.data) + visual_indices[visual_row] * size_t(5120) * hidden.elemsize,
                merged.row<unsigned char>(i), 5120 * hidden.elemsize);
    }
    if (!positions.empty() && (positions.dims != 2 || positions.w != count || positions.h != 3))
    {
        error = "positions must have shape (3,tokens)";
        return {};
    }
    const int width = 64;
    ncnn::Mat cos(width, count, 1), sin(width, count, 1), mask(count, count);
    if (cos.empty() || sin.empty() || mask.empty())
    {
        error = "cannot allocate text rotary tables and mask";
        return {};
    }
    cos.fill(0.f);
    sin.fill(0.f);
    mask.fill(0.f);
    for (int i = 0; i < count; i++)
    {
        for (int j = 0; j < 64; j++)
        {
            const int axis = (j < 60 && j % 3 == 1) ? 1 : (j < 60 && j % 3 == 2) ? 2 : 0;
            const float position = positions.empty() ? (float)i : positions.row(axis)[i];
            const float angle = position / std::pow(5000000.f, (float)(2 * j) / 128);
            cos.channel(0).row(i)[j] = std::cos(angle);
            sin.channel(0).row(i)[j] = std::sin(angle);
        }
        for (int j = i + 1; j < count; j++)
            mask.row(i)[j] = -std::numeric_limits<float>::infinity();
    }
    std::vector<ncnn::Mat> additions(3);
    for (int layer = 0; layer < 3; layer++)
    {
        additions[layer].create(5120, count);
        if (additions[layer].empty())
        {
            error = "cannot allocate visual conditioning for text encoder";
            return {};
        }
        additions[layer].fill(0.f);
        size_t row = 0;
        for (const auto& block : vision)
            for (int j = 0; j < block.merged.h; j++, row++)
                std::memcpy(additions[layer].row(visual_indices[row]), block.deepstack[layer].row(j), 5120 * sizeof(float));
    }
    auto* net = models_.prepare(error, Component::Text);
    if (!net)
        return {};
    std::vector<ncnn::Mat> outputs;
    if (!run_model(error, *net,
            { { "embeddings", hidden }, { "cos", cos }, { "sin", sin }, { "mask", mask },
                { "deepstack0", additions[0] }, { "deepstack1", additions[1] }, { "deepstack2", additions[2] } },
            { "out0" }, outputs))
        return {};
    return outputs[0];
}
}
