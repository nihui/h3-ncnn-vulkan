// SPDX-License-Identifier: Apache-2.0
#include "reference_kv.h"

#include <command.h>
#include <layer.h>
#include <net.h>
#include <paramdict.h>

#include <cstdio>

namespace h3
{
namespace
{
constexpr int kHeadWidth = 128;
constexpr int kHeads = 56;

class CacheLayer : public ncnn::Layer
{
public:
    explicit CacheLayer(ReferenceKVCache* cache)
        : cache_(cache)
    {
        one_blob_only = false;
        support_vulkan = true;
        support_vulkan_packing = true;
    }

    int load_param(const ncnn::ParamDict& params) override
    {
        index_ = params.get(0, -1);
        return index_ >= 0 && index_ < kDiTBlocks ? 0 : -1;
    }

    int create_pipeline(const ncnn::Option& options) override
    {
        crop_.reset(ncnn::create_layer_vulkan("Crop"));
        concat_.reset(ncnn::create_layer_vulkan("Concat"));
        if (!crop_ || !concat_)
            return -1;
        crop_->vkdev = vkdev;
        concat_->vkdev = vkdev;
        ncnn::ParamDict crop_params;
        crop_params.set(3, kHeadWidth);
        crop_params.set(4, -233);
        crop_params.set(5, kHeads);
        ncnn::ParamDict concat_params;
        concat_params.set(0, 1);
        if (crop_->load_param(crop_params) || concat_->load_param(concat_params)
            || crop_->create_pipeline(options) || concat_->create_pipeline(options))
            return -1;
        return 0;
    }

    int destroy_pipeline(const ncnn::Option& options) override
    {
        if (crop_)
            crop_->destroy_pipeline(options);
        if (concat_)
            concat_->destroy_pipeline(options);
        crop_.reset();
        concat_.reset();
        return 0;
    }

    int forward(const std::vector<ncnn::VkMat>& in, std::vector<ncnn::VkMat>& out, ncnn::VkCompute& cmd,
        const ncnn::Option& opt) const override
    {
        return cache_->forward(index_, in, out, *crop_, *concat_, cmd, opt);
    }

private:
    ReferenceKVCache* cache_;
    int index_ = -1;
    std::unique_ptr<ncnn::Layer> crop_;
    std::unique_ptr<ncnn::Layer> concat_;
};

ncnn::Layer* create_cache_layer(void* cache)
{
    return new CacheLayer(static_cast<ReferenceKVCache*>(cache));
}

bool slice_rows(ncnn::Layer& crop, const ncnn::VkMat& source, int start, int count, ncnn::VkMat& target,
    ncnn::VkCompute& command, const ncnn::Option& options)
{
    ncnn::ParamDict params;
    params.set(1, start);
    params.set(3, kHeadWidth);
    params.set(4, count);
    params.set(5, kHeads);
    return crop.load_param(params) == 0 && crop.forward(source, target, command, options) == 0 && !target.empty();
}
}

ReferenceKVCache::ReferenceKVCache(int mode, uint64_t limit)
    : mode_(mode)
    , limit_(limit)
{
}

bool ReferenceKVCache::register_layer(ncnn::Net& net)
{
    return net.register_custom_layer("H3ReferenceKV", create_cache_layer, nullptr, this) == 0;
}

bool ReferenceKVCache::begin(std::string& error, const ncnn::VulkanDevice* device, int step, int prefix_rows,
    int active_video, int cached_audio, int active_audio)
{
    const int refresh_step = mode_ == 3 ? 0 : step - step % 2;
    reuse_ = step != refresh_step;
    visited_ = 0;
    const std::array<int, 4> shape = { prefix_rows, active_video, cached_audio, active_audio };
    if (step < 0 || prefix_rows <= 0 || active_video <= 0 || active_audio <= 0 || cached_audio < 0
        || (reuse_ && (!ready_ || refreshed_ != refresh_step || shape != shape_)))
    {
        error = "rough K/V cache shape or refresh step mismatch";
        return false;
    }
    if (!reuse_)
    {
        const uint64_t payload = uint64_t(prefix_rows + cached_audio) * kDiTBlocks * (2 * kHeads * kHeadWidth * 2);
        if (payload > limit_)
        {
            error = "rough K/V needs at least " + std::to_string((payload + kMiB - 1) / kMiB)
                + " MiB. Increase --kv-cache-mib or use --rough-mode 0";
            return false;
        }
        entries_ = {};
        bytes_ = 0;
        ready_ = false;
        if (!allocator_)
            allocator_ = std::make_unique<ncnn::VkBlobAllocator>(device);
        shape_ = shape;
        refreshed_ = step;
    }
    const int active_rows = active_video + active_audio;
    const int total = prefix_rows + active_rows + cached_audio;
    std::fprintf(stderr, "rough K/V step %d %s, computed rows %d/%d\n", step + 1, reuse_ ? "reuse" : "refresh",
        reuse_ ? active_rows : total, total);
    return true;
}

void ReferenceKVCache::clear()
{
    entries_ = {};
    allocator_.reset();
    bytes_ = 0;
    ready_ = false;
}

bool ReferenceKVCache::finish(std::string& error)
{
    if (visited_ != kDiTBlocks)
    {
        error = "rough K/V graph did not visit all transformer blocks";
        return false;
    }
    if (!reuse_)
    {
        ready_ = true;
        std::fprintf(stderr, "rough K/V reserved %.1f MiB of tensor buffers\n", (double)bytes_ / kMiB);
    }
    return true;
}

int ReferenceKVCache::forward(int index, const std::vector<ncnn::VkMat>& inputs, std::vector<ncnn::VkMat>& outputs,
    ncnn::Layer& crop, ncnn::Layer& concat, ncnn::VkCompute& command, const ncnn::Option& options)
{
    if (index != visited_++ || index < 0 || index >= kDiTBlocks || !allocator_ || inputs.size() != 2
        || outputs.size() != 2)
        return -1;
    const auto [prefix_rows, active_video, cached_audio, active_audio_rows] = shape_;
    const int active_rows = active_video + active_audio_rows;
    const int total = prefix_rows + active_rows + cached_audio;

    auto& entry = entries_[index];
    for (int i = 0; i < 2; i++)
    {
        ncnn::VkMat value = inputs[i];
        if (value.dims != 3 || value.w != kHeadWidth || value.h != (reuse_ ? query_rows(active_rows) : total)
            || value.c * value.elempack != kHeads || value.elembits() != 16)
            return -1;

        auto& prefix = entry.prefix[i];
        auto& audio = entry.reference_audio[i];
        if (!reuse_)
        {
            auto store = options;
            store.blob_vkallocator = allocator_.get();
            if (!slice_rows(crop, value, 0, prefix_rows, prefix, command, store)
                || (cached_audio && !slice_rows(crop, value, prefix_rows + active_video,
                                           cached_audio, audio, command, store)))
                return -100;
            const uint64_t added = prefix.buffer_capacity() + (cached_audio ? audio.buffer_capacity() : 0);
            bytes_ += added;
            if (bytes_ > limit_)
            {
                std::fprintf(stderr, "rough K/V buffers exceeded --kv-cache-mib after allocation\n");
                return -100;
            }
            outputs[i] = inputs[i];
            continue;
        }

        if (prefix.empty() || (cached_audio && audio.empty()))
            return -1;
        if (value.elempack != prefix.elempack)
        {
            ncnn::VkMat packed;
            concat.vkdev->convert_packing(value, packed, prefix.elempack, command, options);
            if (packed.empty())
                return -100;
            value = packed;
        }
        std::vector<ncnn::VkMat> parts = { prefix, value };
        if (cached_audio)
        {
            ncnn::VkMat video, active_audio;
            if (!slice_rows(crop, value, 0, active_video, video, command, options)
                || !slice_rows(crop, value, active_video, active_audio_rows, active_audio, command, options))
                return -100;
            parts = { prefix, video, audio, active_audio };
        }
        else if (value.h != active_rows)
        {
            ncnn::VkMat active;
            if (!slice_rows(crop, value, 0, active_rows, active, command, options))
                return -100;
            parts[1] = active;
        }
        std::vector<ncnn::VkMat> joined(1);
        if (concat.forward(parts, joined, command, options) || joined[0].empty())
            return -100;
        outputs[i] = joined[0];
    }
    return 0;
}
}
