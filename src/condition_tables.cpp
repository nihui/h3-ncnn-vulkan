// SPDX-License-Identifier: Apache-2.0
#include "condition_tables.h"
#include "artifact_hash.h"
#include "filesystem_utils.h"
#include "h3_config.h"
#include "lora.h"
#include "modulation_builder.h"

#include <gpu.h>
#include <net.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <map>
#include <sstream>
#include <string_view>

namespace h3
{
namespace
{
using BankFields = std::map<std::string_view, std::string>;

std::string json_string(std::string_view value)
{
    std::string out = "\"";
    for (unsigned char c : value)
    {
        switch (c)
        {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\b':
            out += "\\b";
            break;
        case '\f':
            out += "\\f";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (c < 0x20)
            {
                constexpr char hex[] = "0123456789abcdef";
                out += "\\u00";
                out += hex[c >> 4];
                out += hex[c & 15];
            }
            else
                out += static_cast<char>(c);
        }
    }
    out += '"';
    return out;
}

template <class T> std::string integer_array(const std::vector<T>& values)
{
    std::string out = "[";
    for (T value : values)
    {
        if (out.size() > 1)
            out += ',';
        out += std::to_string(value);
    }
    out += ']';
    return out;
}

std::string bank_header(const char* magic, const BankFields& fields)
{
    // Existing banks compare the header byte for byte. Preserve compact JSON and key order.
    std::string header = std::string(magic) + "\n{";
    for (auto it = fields.begin(); it != fields.end(); ++it)
    {
        if (it != fields.begin())
            header += ',';
        header += '"';
        header += it->first;
        header += "\":";
        header += it->second;
    }
    header += "}\n";
    return header;
}

std::string model_file_key(std::string& error, const std::filesystem::path& path, bool portable)
{
    const auto size = fsutil::file_size(error, path);
    if (!error.empty())
        return {};

    // Small graphs are read in full. Sample large weight files without loading them.
    std::ifstream file(path, std::ios::binary);
    std::vector<char> bytes(std::min<uint64_t>(size, 256 * 1024));
    if (!file || bytes.empty())
    {
        error = "cannot read model fingerprint from " + path.string();
        return {};
    }
    uint64_t hash = kHashStart;
    const auto last = size - bytes.size();
    const int samples = portable ? 32 : 3;
    for (int i = 0; i < samples; i++)
    {
        const auto offset = last * i / (samples - 1);
        file.seekg(static_cast<std::streamoff>(offset));
        if (!file.read(bytes.data(), bytes.size()))
        {
            error = "cannot read model fingerprint from " + path.string();
            return {};
        }
        hash = hash_bytes(hash, bytes.data(), bytes.size());
        if (!last)
            break;
    }
    std::string key = "[" + std::to_string(size) + ",";
    if (!portable)
    {
        const auto modified = fsutil::last_write_time(error, path);
        if (!error.empty())
            return {};
        key += std::to_string(modified.time_since_epoch().count()) + ",";
    }
    return key + std::to_string(hash) + "]";
}

BankFields bank_model_key(std::string& error, const std::filesystem::path& dit, bool portable)
{
    const std::pair<const char*, const char*> files[] = {
        { "core_param", "transformer-core.ncnn.param" },
        { "core_weights", "transformer-core.ncnn.bin" },
        { "producer_param", "fixed-build/modulations.ncnn.param" },
        { "producer_weights", "fixed-build/modulations.ncnn.bin" },
    };
    BankFields key;
    for (const auto& [name, path] : files)
    {
        key[name] = model_file_key(error, dit / path, portable);
        if (!error.empty())
            return {};
    }
    return key;
}

struct TemporaryDirectory
{
    std::filesystem::path path;

    ~TemporaryDirectory()
    {
        if (!path.empty())
        {
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
        }
    }
};
}

bool ModulationBank::prepare(std::string& error, H3ModelSet& models, bool reference,
    const std::vector<std::vector<float>>& schedule, int preset, bool write_portable, bool* needs_bake)
{
    namespace fs = std::filesystem;
    file_.close();
    file_.clear();
    counts_.clear();
    offsets_.clear();
    const auto& gpu = models.device()->info;
    if (!gpu.support_bf16_packed() && !gpu.support_bf16_storage())
    {
        error = "Vulkan BF16 buffers are required for fixed modulations";
        return false;
    }
    const uint32_t endian = 1;
    const bool little_endian = *(const unsigned char*)&endian == 1;
    const fs::path dit = fs::path(models.directory()) / "transformer" / (reference ? "ref2va" : "fl2va");
    for (const auto& times : schedule)
        counts_.push_back((int)times.size());

    std::vector<const char*> names;
    for (int i = 0; i < kModulationTables; i++)
    {
        if (i == kDiTBlocks)
            names_[i] = "final_raw";
        else
        {
            char name[24];
            std::snprintf(name, sizeof(name), "adaln_%02d_raw", i);
            names_[i] = name;
        }
        names.push_back(names_[i].c_str());
    }

    const auto* lora = models.lora();
    const bool pi = lora && lora->pi_steps() != 0;
    fs::path path;
    std::string header;
    if (preset >= 0)
    {
        constexpr const char* modalities[] = { "none", "video", "audio", "both" };
        const auto preset_name
            = pi ? "pi-h3-" + std::to_string(schedule.size()) + "step" : "steps" + std::to_string(schedule.size() + 1);
        path = dit / ("fixed-modulations-bf16-" + preset_name + "-" + modalities[preset] + ".bin");
        if (write_portable || fsutil::is_regular_file(error, path))
        {
            auto key = bank_model_key(error, dit, true);
            if (!error.empty())
                return false;
            // Presets bind to the scheduler, not platform-specific sin/cos results.
            // Bump the portable format when the preset scheduler or table layout changes.
            key.insert({ { "schedule", json_string(pi ? "pi-h3-euler-v1" : "h3-euler-v1") },
                { "workflow", json_string(reference ? "ref2va" : "fl2va") },
                { "steps", std::to_string(schedule.size() + 1) }, { "references", std::to_string(preset) },
                { "times", integer_array(counts_) }, { "little_endian", little_endian ? "true" : "false" } });
            if (pi)
                key["lora"] = json_string(lora->modulation_key());
            header = bank_header("H3BF16MOD2", key);
            if (open(path, header))
            {
                if (needs_bake)
                    *needs_bake = false;
                std::fprintf(stderr, "using portable BF16 modulation bank %s\n", path.string().c_str());
                return true;
            }
        }
        if (!error.empty())
            return false;
    }

    std::vector<ncnn::Mat> features;
    features.reserve(schedule.size());
    std::vector<uint64_t> frequency_hashes;
    for (const auto& times : schedule)
    {
        features.push_back(modulation_time_features(times));
        if (features.back().empty())
        {
            error = "cannot allocate modulation time features";
            return false;
        }
        if (!write_portable)
            frequency_hashes.push_back(
                hash_bytes(kHashStart, features.back().data, times.size() * 256 * sizeof(float)));
    }

    if (!write_portable)
    {
        auto key = bank_model_key(error, dit, false);
        if (!error.empty())
            return false;
        key.insert({ { "times", integer_array(counts_) }, { "frequencies", integer_array(frequency_hashes) },
            { "ncnn", json_string(NCNN_VERSION_STRING) }, { "vendor", std::to_string(gpu.vendor_id()) },
            { "device", std::to_string(gpu.device_id()) }, { "driver", std::to_string(gpu.driver_version()) },
            { "bf16_storage", gpu.support_bf16_storage() ? "true" : "false" },
            { "little_endian", little_endian ? "true" : "false" } });
        if (lora && !lora->modulation_key().empty())
            key["lora"] = json_string(lora->modulation_key());
        header = bank_header("H3BF16MOD1", key);
        std::ostringstream id;
        id << std::hex << hash_bytes(kHashStart, header.data(), header.size());
        path = dit / "fixed-build" / "banks" / (id.str() + ".bin");
        if (open(path, header))
        {
            if (needs_bake)
                *needs_bake = false;
            std::fprintf(stderr, "using BF16 modulation bank %s\n", path.string().c_str());
            return true;
        }
    }

    if (needs_bake)
    {
        *needs_bake = true;
        return true;
    }
    std::error_code ec;
    const auto directory = path.parent_path();
    fs::create_directories(directory, ec);
    if (ec)
    {
        error = "cannot create modulation cache directory: " + ec.message();
        return false;
    }
    TemporaryDirectory temporary;
    const auto serial = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto temp_dir = directory / (".bake-" + std::to_string(serial));
    if (!fs::create_directory(temp_dir, ec))
    {
        error = "cannot create temporary modulation bank: " + ec.message();
        return false;
    }
    temporary.path = temp_dir;
    const auto temp = temporary.path / "bank.bin";
    std::ofstream out(temp, std::ios::binary);
    if (!out.write(header.data(), header.size()))
    {
        error = "cannot start writing modulation bank";
        return false;
    }
    if (!build_modulations(error, models, reference, schedule, features, names, out))
        return false;
    out.close();
    if (!out)
    {
        error = "cannot finish modulation bank";
        return false;
    }
    fs::rename(temp, path, ec);
    if (!open(path, header))
    {
        error = "cannot publish modulation bank " + path.string();
        return false;
    }
    std::fprintf(stderr, "baked %zu BF16 modulation steps in %s\n", schedule.size(), path.string().c_str());
    return true;
}

bool ModulationBank::open(const std::filesystem::path& path, const std::string& header)
{
    std::streamoff size = header.size();
    offsets_.clear();
    for (int count : counts_)
    {
        offsets_.push_back(size);
        size += sizeof(uint64_t);
        for (int i = 0; i < kModulationTables; i++)
            size += (std::streamoff)count * table_width(i) * sizeof(uint16_t);
    }
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() != size)
        return false;
    input.seekg(0);
    std::string prefix(header.size(), '\0');
    if (!input.read(prefix.data(), prefix.size()) || prefix != header)
        return false;
    std::vector<char> chunk(64 * 1024);
    for (size_t step = 0; step < offsets_.size(); step++)
    {
        uint64_t stored_hash;
        if (!input.read(reinterpret_cast<char*>(&stored_hash), sizeof(stored_hash)))
            return false;
        uint64_t hash = kHashStart;
        const auto end = step + 1 < offsets_.size() ? offsets_[step + 1] : size;
        auto remaining = end - offsets_[step] - (std::streamoff)sizeof(stored_hash);
        while (remaining > 0)
        {
            const auto bytes = std::min(remaining, (std::streamoff)chunk.size());
            if (!input.read(chunk.data(), bytes))
                return false;
            hash = hash_bytes(hash, chunk.data(), bytes);
            remaining -= bytes;
        }
        if (hash != stored_hash)
            return false;
    }
    file_ = std::move(input);
    return true;
}

bool ModulationBank::bind(std::string& error, int step, std::vector<ModelInput>& inputs)
{
    if (step < 0 || step >= (int)offsets_.size())
    {
        error = "modulation bank step out of range";
        return false;
    }
    uint64_t stored_hash = 0;
    uint64_t hash = kHashStart;
    file_.seekg(offsets_[step]);
    file_.read(reinterpret_cast<char*>(&stored_hash), sizeof(stored_hash));
    for (int i = 0; i < kModulationTables; i++)
    {
        const bool final = i == kDiTBlocks;
        const int width = kDiTWidth * (final ? 2 : 6);
        const int rows = counts_[step] * (final ? 1 : 3);
        ncnn::Mat value(width, rows, (size_t)sizeof(uint16_t));
        if (value.empty())
        {
            error = "cannot allocate modulation bank input";
            return false;
        }
        const auto bytes = (std::streamsize)width * rows * sizeof(uint16_t);
        if (!file_.read(static_cast<char*>(value.data), bytes))
        {
            error = "cannot read modulation bank step";
            return false;
        }
        hash = hash_bytes(hash, value.data, bytes);
        inputs.push_back({ names_[i].c_str(), value });
    }
    if (hash != stored_hash)
    {
        error = "modulation bank checksum mismatch";
        return false;
    }
    return true;
}
}
