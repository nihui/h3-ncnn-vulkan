// SPDX-License-Identifier: Apache-2.0
#include "vae_tiles.h"

#include <algorithm>
#include <cstring>

namespace h3
{
std::vector<TileSpan> split_tiles(int size)
{
    if (size <= kVaeTile)
        return { { 0, size, 0 } };

    const int stride = kVaeTile - kVaeTileOverlap;
    const int n = 1 + (size - kVaeTile + stride - 1) / stride;
    const int slack = kVaeTile + (n - 1) * stride - size;
    std::vector<TileSpan> tiles(n);
    int start = 0;
    for (int i = 0; i < n; i++)
    {
        const int overlap = i + 1 < n ? kVaeTileOverlap + slack / (n - 1) + (i < slack % (n - 1)) : 0;
        tiles[i] = { start, kVaeTile, overlap };
        start += kVaeTile - overlap;
    }
    return tiles;
}

void copy_tiles(ncnn::Mat& dst, const std::vector<TileSpan>& horizontal, const std::vector<TileSpan>& vertical,
    const std::vector<ncnn::Mat>& tiles, int scale)
{
    const int columns = (int)horizontal.size();
    for (int ty = 0; ty < (int)vertical.size(); ty++)
        for (int tx = 0; tx < columns; tx++)
        {
            const size_t index = (size_t)ty * columns + tx;
            const ncnn::Mat& tile = tiles[index];
            const int width = tile.w - horizontal[tx].overlap * scale;
            const int height = tile.h - vertical[ty].overlap * scale;
            const int start_x = horizontal[tx].start * scale;
            const int start_y = vertical[ty].start * scale;
            const int blend_x = tx == 0 ? 0 : horizontal[tx - 1].overlap * scale;
            const int blend_y = ty == 0 ? 0 : vertical[ty - 1].overlap * scale;

            for (int c = 0; c < tile.c; c++)
                for (int t = 0; t < tile.d; t++)
                {
                    const ncnn::Mat current = tile.channel(c).depth(t);
                    const ncnn::Mat top = ty > 0 ? tiles[index - columns].channel(c).depth(t) : ncnn::Mat();
                    const ncnn::Mat left = tx > 0 ? tiles[index - 1].channel(c).depth(t) : ncnn::Mat();
                    ncnn::Mat output = dst.channel(c).depth(t);
                    for (int y = 0; y < height; y++)
                    {
                        float* row = output.row(start_y + y) + start_x;
                        const float* source = current.row(y);
                        std::memcpy(row, source, width * sizeof(float));
                        if (y < blend_y)
                        {
                            const float weight = (float)y / blend_y;
                            const float* neighbor = top.row(top.h - blend_y + y);
                            for (int x = 0; x < width; x++)
                                row[x] = neighbor[x] * (1.f - weight) + row[x] * weight;
                        }
                        for (int x = 0; x < std::min(width, blend_x); x++)
                        {
                            const float weight = (float)x / blend_x;
                            row[x] = left.row(y)[left.w - blend_x + x] * (1.f - weight) + row[x] * weight;
                        }
                    }
                }
        }
}
}
