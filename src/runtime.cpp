// SPDX-License-Identifier: Apache-2.0
#include "runtime.h"
#include "filesystem_utils.h"
#include "h3_config.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <gpu.h>
#include <sstream>
#include <vector>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <psapi.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif

namespace h3
{
static uint64_t available_in_memory_limit(uint64_t available, uint64_t limit, uint64_t used, uint64_t reclaimable)
{
    const uint64_t working = used - std::min(used, reclaimable);
    return std::min(available, limit > working ? limit - working : 0);
}
namespace
{
uint64_t file_number(const std::filesystem::path& path)
{
    std::ifstream file(path);
    std::string value;
    if (!(file >> value) || value == "max")
        return 0;
    std::string error;
    const auto number = parse_u64(error, value);
    return !error.empty() ? 0 : number;
}
#if defined(__linux__)
uint64_t cgroup_available(uint64_t available)
{
    struct Group
    {
        std::string path;
        bool v2;
    };
    std::vector<Group> groups;
    std::ifstream membership("/proc/self/cgroup");
    std::string line;
    while (std::getline(membership, line))
    {
        const auto first = line.find(':'), last = line.find(':', first + 1);
        if (first == std::string::npos || last == std::string::npos)
            continue;
        const auto controllers = "," + line.substr(first + 1, last - first - 1) + ",";
        if (controllers == ",," || controllers.find(",memory,") != std::string::npos)
            groups.push_back({ line.substr(last + 1), controllers == ",," });
    }
    std::ifstream mounts("/proc/self/mountinfo");
    while (std::getline(mounts, line))
    {
        const auto split = line.find(" - ");
        if (split == std::string::npos)
            continue;
        std::istringstream before(line.substr(0, split)), after(line.substr(split + 3));
        std::string id, parent, device, root, mount, type, source, options;
        before >> id >> parent >> device >> root >> mount;
        after >> type >> source >> options;
        for (const auto& group : groups)
        {
            if (group.v2 ? type != "cgroup2"
                         : (type != "cgroup" || ("," + options + ",").find(",memory,") == std::string::npos))
                continue;
            if (root != "/" && group.path != root && group.path.find(root + "/") != 0)
                continue;
            std::string relative = root == "/" ? group.path : group.path.substr(root.size());
            while (!relative.empty() && relative.front() == '/')
                relative.erase(0, 1);
            std::filesystem::path current = std::filesystem::path(mount) / relative;
            const auto mount_path = std::filesystem::path(mount);
            while (true)
            {
                const uint64_t limit = file_number(current / (group.v2 ? "memory.max" : "memory.limit_in_bytes"));
                const uint64_t used = file_number(current / (group.v2 ? "memory.current" : "memory.usage_in_bytes"));
                uint64_t reclaimable = 0;
                std::ifstream statistics(current / "memory.stat");
                std::string key;
                uint64_t value = 0;
                while (statistics >> key >> value)
                    if (key == "inactive_file" || (!group.v2 && key == "total_inactive_file"))
                        reclaimable = std::max(reclaimable, value);
                if (limit && limit < (1ull << 62))
                    available = available_in_memory_limit(available, limit, used, reclaimable);
                if (current == mount_path || current == current.root_path())
                    break;
                current = current.parent_path();
            }
        }
    }
    return available;
}
#endif
#if defined(__linux__)
std::pair<uint64_t, uint64_t> linux_process_memory()
{
    // VmRSS/statm use summary counters that can disagree substantially with
    // mapped Vulkan host memory. smaps_rollup walks the mappings themselves.
    const auto read = [](const char* path)
    {
        std::ifstream file(path);
        std::string line;
        uint64_t resident = 0, anonymous = 0;
        while (std::getline(file, line))
        {
            std::istringstream row(line);
            std::string key;
            uint64_t kib = 0;
            if (row >> key >> kib)
            {
                if (key == "Rss:")
                    resident += kib * 1024;
                if (key == "Anonymous:")
                    anonymous += kib * 1024;
            }
        }
        return std::make_pair(resident, anonymous);
    };
    auto usage = read("/proc/self/smaps_rollup");
    if (!usage.first)
        usage = read("/proc/self/smaps");
    if (usage.first)
        return usage;
    // Older/restricted procfs may expose only the approximate summaries.
    uint64_t size = 0, resident = 0;
    std::ifstream statm("/proc/self/statm");
    statm >> size >> resident;
    usage.first = resident * static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line))
        if (line.compare(0, 8, "RssAnon:") == 0)
        {
            std::istringstream row(line.substr(8));
            uint64_t kib = 0;
            row >> kib;
            usage.second = kib * 1024;
        }
    return usage;
}
#endif

uint64_t available_ram(std::string& error)
{
#if defined(_WIN32)
    MEMORYSTATUSEX status {};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status))
        return status.ullAvailPhys;
#elif defined(__APPLE__)
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t stats {};
    vm_size_t page = 0;
    const mach_port_t host = mach_host_self();
    const bool ok = host_page_size(host, &page) == KERN_SUCCESS
        && host_statistics64(host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&stats), &count) == KERN_SUCCESS;
    mach_port_deallocate(mach_task_self(), host);
    if (ok)
        return static_cast<uint64_t>(stats.free_count + stats.inactive_count / 2) * page;
#elif defined(__linux__)
    std::ifstream file("/proc/meminfo");
    std::string key, unit;
    uint64_t value;
    while (file >> key >> value >> unit)
        if (key == "MemAvailable:")
            return cgroup_available(value * 1024);
#else
    const long pages = sysconf(_SC_AVPHYS_PAGES), page = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page > 0)
        return static_cast<uint64_t>(pages) * page;
#endif
    {
        error = "cannot query available system RAM";
        return {};
    }
}
}
static uint64_t process_resident_memory()
{
#if defined(__linux__)
    return linux_process_memory().first;
#else
    std::string error;
    return query_memory(error, nullptr).process_resident;
#endif
}
uint64_t process_peak_ram()
{
#if defined(__linux__)
    std::ifstream file("/proc/self/status");
    std::string line;
    while (std::getline(file, line))
        if (line.compare(0, 6, "VmHWM:") == 0)
        {
            uint64_t kib = 0;
            std::istringstream(line.substr(6)) >> kib;
            return kib * 1024;
        }
#endif
    return process_resident_memory();
}
bool check_ram_headroom(std::string& error, uint64_t budget, uint64_t bytes, const char* stage)
{
    const uint64_t resident = process_resident_memory();
    if (resident <= budget && bytes <= budget - resident)
        return true;
    error = std::string(stage) + " exceeds the RAM budget; reduce reference size or increase --ram-budget";
    return false;
}
MemorySnapshot query_memory(std::string& error, const ncnn::VulkanDevice* device)
{
    MemorySnapshot result;
    result.ram_available = available_ram(error);
    if (!error.empty())
        return {};

#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS process {};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &process, sizeof(process)))
    {
        result.process_resident = process.WorkingSetSize;
        result.process_anonymous = process.WorkingSetSize;
        result.ram_available += process.WorkingSetSize;
    }
#elif defined(__linux__)
    const auto usage = linux_process_memory();
    result.process_resident = usage.first;
    result.process_anonymous = usage.second;
    // MemAvailable already includes reclaimable file/model pages. Count owned
    // anonymous data once, without crediting clean file pages a second time.
    result.ram_available += usage.second;
#elif defined(__APPLE__)
    mach_task_basic_info_data_t task {};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&task), &count) == KERN_SUCCESS)
    {
        result.process_resident = task.resident_size;
        result.process_anonymous = task.resident_size;
        result.ram_available += task.resident_size;
    }
#endif
#if NCNN_VULKAN
    if (device)
    {
        const auto* vkdev = device;
        const auto& info = vkdev->info;
        const auto& properties = info.physicalDeviceMemoryProperties();
        uint32_t heap = 0;
        for (uint32_t i = 0; i < properties.memoryHeapCount; i++)
            if ((properties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                && (!(properties.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                    || properties.memoryHeaps[i].size > properties.memoryHeaps[heap].size))
                heap = i;
        result.shared_heap = true;
        for (uint32_t i = 0; i < properties.memoryTypeCount; i++)
            if ((properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
                && !(properties.memoryHeaps[properties.memoryTypes[i].heapIndex].flags
                    & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT))
                result.shared_heap = false;
        if (!result.shared_heap)
        {
            result.host_visible_available = 0;
            for (uint32_t type = 0; type < properties.memoryTypeCount; type++)
                if ((properties.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
                    && !(properties.memoryHeaps[properties.memoryTypes[type].heapIndex].flags
                        & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT))
                    result.host_visible_available = std::max(result.host_visible_available,
                        static_cast<uint64_t>(properties.memoryHeaps[properties.memoryTypes[type].heapIndex].size));
        }
        result.vram_available = static_cast<uint64_t>(vkdev->get_heap_budget()) * kMiB;
        if (info.support_VK_EXT_memory_budget() && ncnn::vkGetPhysicalDeviceMemoryProperties2KHR)
        {
            VkPhysicalDeviceMemoryBudgetPropertiesEXT budget {};
            budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
            VkPhysicalDeviceMemoryProperties2KHR props {};
            props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2_KHR;
            props.pNext = &budget;
            ncnn::vkGetPhysicalDeviceMemoryProperties2KHR(info.physicalDevice(), &props);
            uint64_t host_visible = 0;
            for (uint32_t i = 0; i < properties.memoryHeapCount; i++)
            {
                bool visible = false;
                for (uint32_t type = 0; type < properties.memoryTypeCount; type++)
                    if (properties.memoryTypes[type].heapIndex == i
                        && (properties.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
                        visible = true;
                if (visible && !(properties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT))
                    host_visible = std::max(host_visible,
                        budget.heapBudget[i] > budget.heapUsage[i] ? budget.heapBudget[i] - budget.heapUsage[i] : 0);
            }
            if (!result.shared_heap)
                result.host_visible_available = host_visible;
            result.device_usage = budget.heapUsage[heap];
            result.device_budget = budget.heapBudget[heap];
            result.vram_available = budget.heapBudget[heap] > budget.heapUsage[heap]
                ? budget.heapBudget[heap] - budget.heapUsage[heap]
                : 0;
        }
    }
#else
    (void)device;
#endif
    return result;
}
}
