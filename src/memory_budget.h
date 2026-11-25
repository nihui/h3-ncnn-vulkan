// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "h3_config.h"
#include "shape_memory.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace ncnn
{
class VulkanDevice;
}

namespace h3
{
class Lora;
enum class Component
{
    Vision,
    Text,
    VideoEncoder,
    TemporalVideoEncoder,
    VideoDecoder,
    AudioEncoder,
    AudioDecoder,
    TransformerCore,
    ReferenceTransformerCore,
    Modulations,
    ReferenceModulations,
    Count
};

struct StageMemory
{
    uint64_t ram = 0;
    uint64_t vram = 0;
    uint64_t vram_weights = 0;
    uint64_t vram_workspace = 0;
    uint64_t host_weights = 0;
    uint64_t host_visible = 0;
    uint64_t disk_weights = 0;
    uint64_t load_ram = 0;
    uint64_t forward_ram = 0;
    bool paged = false;
};

struct MemoryPlan
{
    struct Stage
    {
        StageMemory minimum;
        StageMemory resident;
        StageMemory selected;
    };

    std::array<Stage, static_cast<size_t>(Component::Count)> stages;
    uint64_t preprocessing_ram = 0;
    uint64_t ram_budget = 0;
    uint64_t vram_budget = 0;
    uint64_t kv_bytes = 0;
    bool feasible = true;
    std::string failure;
};

struct MemoryReference
{
    int width = 0;
    int height = 0;
    int frames = 0;
    int samples = 0;
};

struct MemoryWorkload
{
    int width = 0;
    int height = 0;
    int frames = 0;
    int audio_frames = 0;
    int text_rows = 0;
    int vision_patches = 0;
    int max_vision_patches = 0;
    uint64_t resized_vision_bytes = 0;
    uint64_t preprocessing_peak = 0;
    uint64_t base_ram = 0;
    int steps = 0;
    const Lora* lora = nullptr;
    int rough_mode = 0;
    int kv_cache_mib = 0;
    int modulation_rows = 0;
    bool prepare_modulations = false;
    bool pi_modulations = false;
    bool ref2va = false;
    bool lossless = false;
    std::vector<MemoryReference> references;
};

const char* model_path(Component component);
bool model_bf16(Component component);
bool model_subgroup_ops(Component component, const ncnn::VulkanDevice* device);
const char* component_name(Component component);
AllocationLayout query_allocation_layout(const ncnn::VulkanDevice* device, bool bf16);
MemoryPlan plan_memory(std::string& error, const std::string& model, const ncnn::VulkanDevice* device,
    const std::vector<MemoryWorkload>& workloads, uint64_t ram_budget, uint64_t vram_budget, bool estimate);
void print_memory_plan(const MemoryPlan& plan, bool estimate);
}
