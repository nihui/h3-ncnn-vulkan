// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>

namespace ncnn
{
class VulkanDevice;
}

namespace h3
{
struct RuntimeConfig
{
    std::string model = "models/h3";
    int gpu = -1;
    int bake_modulations = -1;
    int rough_mode = 0;
    int kv_cache_mib = 6144;
    uint64_t ram_budget = 0;
    uint64_t vram_budget = 0;
    bool estimate_memory = false;
};

struct MemorySnapshot
{
    uint64_t ram_available = 0;
    uint64_t process_resident = 0;
    uint64_t process_anonymous = 0;
    uint64_t vram_available = 0;
    uint64_t device_usage = 0;
    uint64_t device_budget = 0;
    uint64_t host_visible_available = UINT64_MAX;
    bool shared_heap = false;
};

MemorySnapshot query_memory(std::string& error, const ncnn::VulkanDevice* device);
uint64_t process_peak_ram();
bool check_ram_headroom(std::string& error, uint64_t budget, uint64_t bytes, const char* stage);
}
