// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace h3
{
inline long long parse_i64(std::string& error, const std::string& text, size_t* consumed = nullptr)
{
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text.c_str(), &end, 10);
    if (consumed)
        *consumed = static_cast<size_t>(end - text.c_str());
    if (errno || end == text.c_str() || *end)
    {
        error = "invalid integer: " + text;
        return 0;
    }
    return value;
}
inline unsigned long long parse_u64(std::string& error, const std::string& text, size_t* consumed = nullptr)
{
    char* end = nullptr;
    errno = 0;
    const auto first = text.find_first_not_of(" \t\r\n");
    const auto value = std::strtoull(text.c_str(), &end, 10);
    if (consumed)
        *consumed = static_cast<size_t>(end - text.c_str());
    if (errno || first == std::string::npos || text[first] == '-' || end == text.c_str() || *end)
    {
        error = "invalid unsigned integer: " + text;
        return 0;
    }
    return value;
}
inline int parse_i32(std::string& error, const std::string& text)
{
    const auto value = parse_i64(error, text);
    if (value < INT_MIN || value > INT_MAX)
    {
        error = "integer out of range: " + text;
        return 0;
    }
    return static_cast<int>(value);
}

inline bool parse_float(const std::string& text, float& value, bool complete = true)
{
    char* end = nullptr;
    errno = 0;
    value = std::strtof(text.c_str(), &end);
    return errno == 0 && end != text.c_str() && (!complete || end == text.c_str() + text.size()) && std::isfinite(value);
}

template <class T, class Allocator> T& checked_at(std::string& error, std::vector<T, Allocator>& values, size_t index)
{
    if (index < values.size())
        return values[index];
    error = "tensor/graph index out of range";
    static thread_local T empty {};
    return empty;
}
template <class T, class Allocator>
const T& checked_at(std::string& error, const std::vector<T, Allocator>& values, size_t index)
{
    return checked_at(error, const_cast<std::vector<T, Allocator>&>(values), index);
}
template <class Map, class Key, class Mapped = typename Map::mapped_type>
auto& checked_at(std::string& error, Map& values, const Key& key)
{
    auto found = values.find(key);
    if (found != values.end())
        return found->second;
    error = "missing tensor/graph entry";
    static thread_local typename Map::mapped_type empty {};
    return empty;
}
template <class Map, class Key, class Mapped = typename Map::mapped_type>
const auto& checked_at(std::string& error, const Map& values, const Key& key)
{
    return checked_at(error, const_cast<Map&>(values), key);
}

namespace fsutil
{
namespace fs = std::filesystem;
inline void report(std::string& error, const char* operation, const fs::path& path, const std::error_code& code)
{
    if (code)
        error = std::string(operation) + " " + path.string() + ": " + code.message();
}
inline uint64_t file_size(std::string& error, const fs::path& path)
{
    std::error_code code;
    const auto result = fs::file_size(path, code);
    report(error, "cannot read file size", path, code);
    return code ? 0 : result;
}
inline auto last_write_time(std::string& error, const fs::path& path)
{
    std::error_code code;
    const auto result = fs::last_write_time(path, code);
    report(error, "cannot read modification time", path, code);
    return result;
}
inline bool exists(std::string& error, const fs::path& path)
{
    std::error_code code;
    const bool result = fs::exists(path, code);
    report(error, "cannot inspect", path, code);
    return result;
}
inline bool is_regular_file(std::string& error, const fs::path& path)
{
    std::error_code code;
    const bool result = fs::is_regular_file(path, code);
    if (code == std::errc::no_such_file_or_directory)
        code.clear();
    report(error, "cannot inspect", path, code);
    return result;
}
inline bool create_directories(std::string& error, const fs::path& path)
{
    std::error_code code;
    const bool result = fs::create_directories(path, code);
    report(error, "cannot create directory", path, code);
    return result;
}
inline fs::path temp_directory_path(std::string& error)
{
    std::error_code code;
    auto result = fs::temp_directory_path(code);
    report(error, "cannot find temporary directory", {}, code);
    return result;
}
inline fs::path weakly_canonical(std::string& error, const fs::path& path)
{
    std::error_code code;
    auto result = fs::weakly_canonical(path, code);
    report(error, "cannot resolve", path, code);
    return result;
}
inline bool remove(std::string& error, const fs::path& path)
{
    std::error_code code;
    const bool result = fs::remove(path, code);
    report(error, "cannot remove", path, code);
    return result;
}
inline void rename(std::string& error, const fs::path& from, const fs::path& to)
{
    std::error_code code;
    fs::rename(from, to, code);
    report(error, "cannot rename", from, code);
}
}
}
