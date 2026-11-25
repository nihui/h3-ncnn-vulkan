// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "h3_config.h"

#include <gpu.h>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace ncnn
{
class Layer;
class Net;
}

namespace h3
{
class ReferenceKVCache
{
public:
    ReferenceKVCache(int mode, uint64_t limit);
    bool register_layer(ncnn::Net& net);
    bool begin(std::string& error, const ncnn::VulkanDevice* device, int step, int prefix_rows, int active_video,
        int cached_audio, int active_audio);
    void clear();
    bool finish(std::string& error);
    bool reuse() const { return reuse_; }
    bool includes_references() const { return mode_ >= 2; }
    static int query_rows(int active) { return (active + 3) & ~3; }

    int forward(int index, const std::vector<ncnn::VkMat>& inputs, std::vector<ncnn::VkMat>& outputs,
        ncnn::Layer& crop, ncnn::Layer& concat, ncnn::VkCompute& command, const ncnn::Option& options);

private:
    struct Entry
    {
        std::array<ncnn::VkMat, 2> prefix;
        std::array<ncnn::VkMat, 2> reference_audio;
    };

    int mode_;
    uint64_t limit_;
    uint64_t bytes_ = 0;
    // Destroy the tensors before their allocator.
    std::unique_ptr<ncnn::VkBlobAllocator> allocator_;
    std::array<Entry, kDiTBlocks> entries_;
    std::array<int, 4> shape_ {};
    int refreshed_ = -1;
    int visited_ = 0;
    bool ready_ = false;
    bool reuse_ = false;
};
}
