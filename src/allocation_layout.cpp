// SPDX-License-Identifier: Apache-2.0
#include "memory_budget.h"
#include <algorithm>
#include <gpu.h>
#include <map>
#include <numeric>

namespace h3
{
AllocationLayout query_allocation_layout(const ncnn::VulkanDevice* device, bool bf16)
{
    AllocationLayout result;
    result.gpu = true;
    result.bf16 = bf16;
    result.winograd23 = false;
    result.winograd43 = false;
    const auto& info = device->info;
    result.discrete = info.type() == 0;
    result.buffer_alignment = info.buffer_offset_alignment();
    if (info.type() == 1)
    {
        result.buffer_alignment
            = std::lcm(result.buffer_alignment, static_cast<uint64_t>(info.memory_map_alignment()));
        result.buffer_alignment
            = std::lcm(result.buffer_alignment, static_cast<uint64_t>(info.non_coherent_atom_size()));
    }
    if (info.support_VK_KHR_robustness2() || info.support_VK_EXT_robustness2())
        result.buffer_alignment = std::lcm(result.buffer_alignment,
            static_cast<uint64_t>(info.queryRobustness2Properties().robustStorageBufferAccessSizeAlignment));
    result.memory_map_alignment = std::max<uint64_t>(info.memory_map_alignment(), info.non_coherent_atom_size());
    const uint32_t reduction = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
    result.subgroup_reductions = info.subgroup_size() >= 4 && info.subgroup_size() <= 128
        && (info.support_subgroup_ops() & reduction) == reduction;
    const bool cooperative = bf16 && info.support_bf16_cooperative_matrix() && info.vendor_id() != 0x5143;
    result.flash_attention = bf16;
    result.flash_output_limit = 32 * 8; // ncnn's non-cooperative FA tile N.
    if (cooperative)
    {
        const uint32_t required = reduction | VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
        result.flash_attention = (info.support_subgroup_ops() & required) == required;
        int m = 0, n = 0, k = 0, subgroup = 0;
        info.get_optimal_cooperative_matrix_mnk(1024, 1024, 1024, VK_COMPONENT_TYPE_BFLOAT16_KHR,
            VK_COMPONENT_TYPE_FLOAT32_KHR, VK_SCOPE_SUBGROUP_KHR, m, n, k, subgroup);
        if (!m || n % 8 || k % 8 || n != k || subgroup < n)
            result.flash_attention = false;
        else
        {
            result.flash_output_limit = static_cast<uint32_t>(n * 8);
            int sm = 2, wm = 2, pn = std::min(4, subgroup / n);
            for (;;)
            {
                const uint64_t rows = static_cast<uint64_t>(sm * wm * m);
                const uint64_t shared = 16 * (rows + static_cast<uint64_t>(pn * n)) * (k / 8)
                    + 4 * rows * static_cast<uint64_t>((pn + 1) * n + 3);
                const uint32_t invocations = static_cast<uint32_t>(subgroup * wm);
                if (shared <= info.max_shared_memory_size() && invocations <= info.max_workgroup_invocations()
                    && invocations <= info.max_workgroup_size_x()
                    && static_cast<uint32_t>(wm) <= info.max_compute_workgroup_subgroups())
                    break;
                if (pn > 1)
                    pn /= 2;
                else if (wm > 1)
                    wm = 1;
                else if (sm > 1)
                    sm = 1;
                else
                {
                    result.flash_attention = false;
                    break;
                }
            }
        }
    }
    if (info.vendor_id() == 0x5143)
        result.flash_attention = false;
    result.gemm_weight_shape
        = [device, cooperative](std::string& resource_error, uint64_t raw_m, uint64_t raw_n, uint64_t raw_k)
    {
        if (!cooperative)
            return std::array<uint64_t, 3> { raw_m, raw_n, raw_k };
        const auto& kernel_info = device->info;
        const int m0 = static_cast<int>(raw_m ? raw_m : 1024), n0 = static_cast<int>(raw_n ? raw_n : 1024),
                  k0 = static_cast<int>(raw_k ? raw_k : 1024);
        int m = 0, n = 0, k = 0, subgroup = 0;
        VkComponentTypeKHR accumulator;
        kernel_info.get_optimal_cooperative_matrix_mnk(m0, n0, k0, VK_COMPONENT_TYPE_BFLOAT16_KHR,
            VK_COMPONENT_TYPE_FLOAT32_KHR, VK_SCOPE_SUBGROUP_KHR, m, n, k, subgroup, accumulator);
        if (!m || (kernel_info.support_VK_KHR_cooperative_matrix() && ((m != 4 && m % 8) || n % 8 || k % 8)))
            return std::array<uint64_t, 3> { raw_m, raw_n, raw_k };
        int sm = std::min((m0 + m - 1) / m, 2), sn = std::min((n0 + n - 1) / n, 2),
            sk = std::min((k0 + k - 1) / k, 2);
        int wm = std::min((m0 + m * sm - 1) / (m * sm), 2), wn = std::min((n0 + n * sn - 1) / (n * sn), 2);
        for (;;)
        {
            const uint64_t a = 16ull * m * (k / 8), b = 16ull * n * (k / 8), c = 16ull * m * (n / 8);
            const uint64_t shared = a * wm * sm * sk + b * wn * sn * sk + c * wm * wn * sm * sn;
            const uint32_t invocations = static_cast<uint32_t>(subgroup * wm * wn);
            if (shared <= kernel_info.max_shared_memory_size()
                && invocations <= kernel_info.max_workgroup_invocations()
                && invocations <= kernel_info.max_workgroup_size_x()
                && static_cast<uint32_t>(wm * wn) <= kernel_info.max_compute_workgroup_subgroups())
                break;
            if (sk > 1)
                sk = 1;
            else if (wn > 1)
                wn = 1;
            else if (wm > 1)
                wm = 1;
            else if (sn > 1)
                sn = 1;
            else if (sm > 1)
                sm = 1;
            else
                return std::array<uint64_t, 3> { raw_m, raw_n, raw_k };
        }
        return std::array<uint64_t, 3> {
            raw_m ? aligned_memory_bytes(resource_error, raw_m, static_cast<uint64_t>(m * sm * wm)) : 0,
            raw_n ? aligned_memory_bytes(resource_error, raw_n, static_cast<uint64_t>(n * sn * wn)) : 0,
            raw_k ? aligned_memory_bytes(resource_error, raw_k, static_cast<uint64_t>(k)) : 0
        };
    };
    auto cache = std::make_shared<std::map<std::pair<uint64_t, bool>, uint64_t>>();
    result.memory_requirement
        = [device, cache](std::string& resource_error, uint64_t bytes, bool staging) -> uint64_t
    {
        const auto key = std::make_pair(bytes, staging);
        auto known = cache->find(key);
        if (known != cache->end())
            return known->second;
        VkBufferCreateInfo buffer_info {};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = bytes;
        buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (!staging)
            buffer_info.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkBuffer buffer = VK_NULL_HANDLE;
        if (ncnn::vkCreateBuffer(device->vkdevice(), &buffer_info, nullptr, &buffer) != VK_SUCCESS)
        {
            resource_error = "device cannot represent a calculated resource buffer";
            return {};
        }
        VkMemoryRequirements requirements;
        ncnn::vkGetBufferMemoryRequirements(device->vkdevice(), buffer, &requirements);
        ncnn::vkDestroyBuffer(device->vkdevice(), buffer, nullptr);
        (*cache)[key] = requirements.size;
        return static_cast<uint64_t>(requirements.size);
    };
    return result;
}
}
