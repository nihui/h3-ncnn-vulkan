// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <set>
#include <vector>
namespace h3
{
// Scalar dimensions in ncnn's w,h,d,c order. Packing is a storage property.
struct MemoryShape
{
    std::vector<uint64_t> dimensions;
    uint32_t scalar_bytes = 4;
    uint32_t packing = 1;
    uint64_t batch = 1;
    uint64_t elements(std::string& error) const;
    uint64_t storage_bytes(std::string& error) const;
    MemoryShape packed(bool enable = true) const;
};
struct MemoryNode
{
    std::string type, name;
    std::vector<std::string> inputs, outputs;
    std::map<int, std::string> parameters;
    std::set<int> quoted_parameters;
    int64_t integer(std::string& error, int key, int64_t fallback = 0) const;
};
struct MemoryGraph
{
    std::vector<MemoryNode> nodes;
    std::map<std::string, size_t> producers;
    std::map<std::string, uint64_t> loading_source_mapping;
};
std::shared_ptr<const MemoryGraph> read_memory_graph(std::string& error, const std::string& path);
std::shared_ptr<const MemoryGraph> parse_memory_graph(std::string& error, const std::string& text);
struct WeightSpan
{
    uint64_t offset;
    uint64_t elements;
    uint64_t bytes;
    uint32_t tag;
};
using WeightSpans = std::map<std::string, std::vector<WeightSpan>>;
void inspect_weight_storage(std::string& error, const std::string& path, MemoryGraph& graph, WeightSpans* spans = nullptr);
struct AllocationLayout
{
    bool gpu = false;
    bool bf16 = false;
    bool discrete = true;
    bool subgroup_reductions = false;
    bool flash_attention = false;
    uint32_t flash_output_limit = 0;
    uint32_t cpu_threads = 1;
    uint32_t cpu_vector_width = 1;
    uint64_t cpu_l2_cache_bytes = 0;
    bool cpu_attention_tiled = false;
    std::function<bool(std::string&, const std::string&)> cpu_bf16_support;
    std::function<std::array<uint64_t, 3>(std::string&, uint64_t, uint64_t, uint64_t)> cpu_gemm_tile_shape;
    std::function<std::array<uint64_t, 3>(std::string&, uint64_t, uint64_t, uint64_t, bool)> cpu_convolution_tile_shape;
    uint64_t buffer_alignment = 16;
    uint64_t memory_map_alignment = 16;
    uint64_t memory_requirement_alignment = 16;
    bool winograd23 = true;
    bool winograd43 = true;
    std::function<std::array<uint64_t, 3>(std::string&, uint64_t, uint64_t, uint64_t)> gemm_weight_shape;
    // Queries VkMemoryRequirements without allocating/binding device memory.
    // The bool distinguishes transfer-only staging from storage buffers.
    std::function<uint64_t(std::string&, uint64_t, bool)> memory_requirement;
    uint64_t allocation_bytes(std::string& error, uint64_t size, bool staging = false) const;
};
// Same first-fit splitting/coalescing and block sizing as VkBlobAllocator.
// It stores offsets and byte counts only, never tensor payloads.
class AllocationPool
{
public:
    AllocationPool(std::string& error, uint64_t block_size, const AllocationLayout& layout);
    size_t allocate(std::string& error, uint64_t bytes);
    void release(std::string& error, size_t id);
    uint64_t allocated_bytes() const;
    uint64_t live_bytes() const;
    uint64_t peak_live_bytes() const;

private:
    struct Range
    {
        uint64_t offset, bytes;
    };
    struct Allocation
    {
        size_t block;
        Range range;
        bool active;
    };
    AllocationLayout layout_;
    uint64_t block_size_, allocated_ = 0, live_ = 0, peak_live_ = 0;
    std::vector<std::vector<Range>> free_;
    std::vector<Allocation> allocations_;
};
struct GraphMemoryRequest
{
    // CPU inputs are uploaded lazily, at their first consumer.
    std::map<std::string, MemoryShape> inputs;
    std::set<std::string> raw_inputs;
    uint64_t kv_full_rows = 0;
    uint64_t kv_active_video = 0;
    uint64_t kv_active_audio = 0;
    bool kv_has_cached_audio = false;
    // Already-accounted caches/live host inputs bypass their producers/uploads.
    std::map<std::string, MemoryShape> device_inputs;
    std::map<std::string, std::string> aliases;
    std::vector<std::string> outputs;
    bool raw_outputs = false;
    bool preserve_host_inputs = true;
    // The tiled VAE host executor calls each node separately on the GPU. Its
    // outer host graph keeps full images and never overwrites an input in place.
    bool external_operators = false;
};
struct GraphMemoryUsage
{
    uint64_t device_pool = 0;
    uint64_t device_live_peak = 0;
    uint64_t staging = 0;
    uint64_t host_live_peak = 0;
    uint64_t host_transfer_scratch = 0;
    uint64_t host_outputs = 0;
    uint64_t max_weight_group = 0;
    std::string peak_node;
    size_t visited_nodes = 0;
    std::map<std::string, MemoryShape> output_shapes;
    std::vector<std::string> execution_order;
    std::map<std::string, std::vector<size_t>> weight_tensor_selection;
    std::map<std::string, std::vector<size_t>> weight_early_tensors;
    std::map<std::string, uint64_t> host_node_peaks;
};
GraphMemoryUsage plan_graph_memory(
    std::string& error, const MemoryGraph& graph, const GraphMemoryRequest& request, const AllocationLayout& layout);
struct WeightAllocation
{
    std::string node;
    std::vector<uint64_t> tensors;
    uint64_t host_loading_scratch = 0;
    uint64_t source_mapping_scratch = 0;
};
// The actual packed/uploaded tensors, including both enabled Winograd transforms.
std::vector<WeightAllocation> model_weight_allocations(
    std::string& error, const MemoryGraph& graph, const AllocationLayout& layout);
struct WeightMemoryUsage
{
    uint64_t device_resident = 0;
    uint64_t host_resident = 0;
    uint64_t packed_payload = 0;
    uint64_t disk_payload = 0;
    uint64_t loading_host_peak = 0;
    uint64_t loading_host_scratch = 0;
    uint64_t loading_staging_peak = 0;
    uint64_t streaming_device_pool = 0;
    uint64_t streaming_host_staging = 0;
    uint64_t streaming_host_source = 0;
    std::map<std::string, std::vector<int>> placements;
};
WeightMemoryUsage plan_weight_memory(std::string& error, const std::vector<WeightAllocation>& weights,
    const std::vector<GraphMemoryUsage>& calls, const AllocationLayout& layout, uint64_t device_limit,
    uint64_t host_limit, bool fully_resident = false);
uint64_t aligned_memory_bytes(std::string& error, uint64_t value, uint64_t alignment);
}
