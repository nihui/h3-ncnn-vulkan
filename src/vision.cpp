// SPDX-License-Identifier: Apache-2.0
#include "vision.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
namespace h3
{
namespace
{
ncnn::Mat load_positions(std::string& error, const H3ModelSet& models)
{
    const std::string path = models.directory() + "/vision/vision_pos_embed.f32";
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    constexpr int width = 1152, height = 48 * 48;
    if (!file || file.tellg() != static_cast<std::streamoff>(width * height * sizeof(float)))
    {
        error = "invalid vision position embeddings: " + path;
        return {};
    }
    file.seekg(0);
    ncnn::Mat positions(width, height);
    if (positions.empty())
    {
        error = "cannot allocate vision position embeddings";
        return {};
    }
    for (int i = 0; i < height; i++)
        file.read(reinterpret_cast<char*>(positions.row(i)), width * sizeof(float));
    if (!file)
    {
        error = "cannot read vision position embeddings: " + path;
        return {};
    }
    return positions;
}
}
VisionEncoder::VisionEncoder(std::string& error, H3ModelSet& models)
    : models_(models)
    , positions_(load_positions(error, models))
{
}
VisionResult VisionEncoder::encode(std::string& error, const ncnn::Mat& pixels, int h, int w)
{
    if (h <= 0 || w <= 0 || h % 2 || w % 2 || pixels.dims != 2 || pixels.h != h * w || pixels.w != 1536)
    {
        error = "vision pixels must have shape (H*W,1536), with even patch-grid H,W";
        return {};
    }
    ncnn::Mat position_embeds(1152, h * w);
    const int cache_width = 36;
    ncnn::Mat cos(cache_width, h * w, 1), sin(cache_width, h * w, 1);
    if (cos.empty() || sin.empty() || position_embeds.empty())
    {
        error = "cannot allocate vision position tables";
        return {};
    }
    for (int by = 0; by < h / 2; by++)
        for (int bx = 0; bx < w / 2; bx++)
            for (int dy = 0; dy < 2; dy++)
                for (int dx = 0; dx < 2; dx++)
                {
                    const int y = by * 2 + dy, x = bx * 2 + dx, index = ((by * (w / 2) + bx) * 2 + dy) * 2 + dx;
                    const float step_y = 47.f / (h - 1), step_x = 47.f / (w - 1);
                    const float gy = y < h / 2 ? (float)y * step_y : std::fma(-(float)(h - y - 1), step_y, 47.f);
                    const float gx = x < w / 2 ? (float)x * step_x : std::fma(-(float)(w - x - 1), step_x, 47.f);
                    const int y0 = (int)gy, x0 = (int)gx, y1 = std::min(y0 + 1, 47), x1 = std::min(x0 + 1, 47);
                    const float fy = gy - y0, fx = gx - x0,
                                weights[4] = { (1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx };
                    float* out = position_embeds.row(index);
                    for (int d = 0; d < 1152; d++)
                    {
                        float value = positions_.row(y0 * 48 + x0)[d] * weights[0];
                        value += positions_.row(y0 * 48 + x1)[d] * weights[1];
                        value += positions_.row(y1 * 48 + x0)[d] * weights[2];
                        value += positions_.row(y1 * 48 + x1)[d] * weights[3];
                        out[d] = value;
                    }
                    for (int axis = 0; axis < 2; axis++)
                        for (int j = 0; j < 18; j++)
                        {
                            const float inverse = 1.f / std::pow(10000.f, (float)(2 * j) / 36);
                            const float angle = (float)(axis == 0 ? y : x) * inverse;
                            const int offset = axis * 18 + j;
                            cos.channel(0).row(index)[offset] = std::cos(angle);
                            sin.channel(0).row(index)[offset] = std::sin(angle);
                        }
                }
    auto* net = models_.prepare(error, Component::Vision);
    if (!net)
        return {};
    std::vector<ncnn::Mat> values;
    if (!run_model(error, *net, { { "in0", pixels }, { "positions", position_embeds }, { "cos", cos }, { "sin", sin } },
            { "out0", "deepstack0", "deepstack1", "deepstack2" }, values))
        return {};
    for (const auto& value : values)
        if (value.dims != 2 || value.h != h * w / 4 || value.w != 5120)
        {
            error = "unexpected vision encoder output shape";
            return {};
        }
    return { values[0], { values[1], values[2], values[3] } };
}

ncnn::Mat multimodal_positions(std::string& error, const ncnn::Mat& ids, const ncnn::Mat& grids)
{
    const int count = ids.w;
    int position = 0, grid_index = 0;
    ncnn::Mat result(count, 3);
    if (result.empty())
    {
        error = "cannot allocate multimodal text positions";
        return {};
    }
    result.fill(0.f);
    for (int i = 0; i < count;)
    {
        const int token = (int)ids[i];
        if (!is_vision_pad(token))
        {
            for (int axis = 0; axis < 3; axis++)
                result.row(axis)[i] = (float)position;
            i++;
            position++;
            continue;
        }
        if (grid_index >= grids.h)
        {
            error = "vision token run has no matching vision grid";
            return {};
        }
        const float* grid = grids.row(grid_index++);
        const int t = grid[0], h = (int)grid[1] / 2, w = (int)grid[2] / 2;
        if (i + t * h * w > count)
        {
            error = "vision token run exceeds sequence length";
            return {};
        }
        for (int j = i; j < i + t * h * w; j++)
            if ((int)ids[j] != token)
            {
                error = "vision tokens must form a complete contiguous grid";
                return {};
            }
        for (int tt = 0; tt < t; tt++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    const int index = i + (tt * h + y) * w + x;
                    result.row(0)[index] = position + tt;
                    result.row(1)[index] = position + y;
                    result.row(2)[index] = position + x;
                }
        i += t * h * w;
        position += std::max(h, w);
    }
    if (grid_index != grids.h)
    {
        error = "unused vision grid";
        return {};
    }
    return result;
}
}
