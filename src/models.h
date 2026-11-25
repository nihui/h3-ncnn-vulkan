// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <mat.h>
#include "memory_budget.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ncnn
{
class Net;
class VulkanDevice;
}

namespace h3
{
class ReferenceKVCache;
class Lora;

class H3ModelSet
{
public:
    H3ModelSet(std::string& error, const std::string& directory, int gpu, int rough_mode = 0, uint64_t kv_cache_limit = 0);
    ~H3ModelSet();

    const std::string& directory() const
    {
        return directory_;
    }
    const ncnn::VulkanDevice* device() const;
    ReferenceKVCache* reference_cache() const { return cache_.get(); }
    ncnn::Net* prepare(std::string& error, Component component);
    void release();
    void set_lora(const Lora* lora) { lora_ = lora; }
    const Lora* lora() const { return lora_; }
    void set_memory_plan(MemoryPlan plan) { memory_plan_ = std::move(plan); }

private:
    std::string directory_;
    int gpu_;
    const Lora* lora_ = nullptr;
    MemoryPlan memory_plan_ {};
    Component loaded_ = Component::Vision;
    std::unique_ptr<ReferenceKVCache> cache_;
    std::unique_ptr<ncnn::Net> net_;
};

struct ModelInput
{
    const char* name;
    ncnn::Mat value;
    bool fp32 = false;
};

bool run_model(std::string& error, const ncnn::Net& net, const std::vector<ModelInput>& inputs,
    const std::vector<const char*>& names, std::vector<ncnn::Mat>& outputs, bool active_only = false);
ncnn::Mat unpack_float32(std::string& error, const ncnn::Mat& value);
}
