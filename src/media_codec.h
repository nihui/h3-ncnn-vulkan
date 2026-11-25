// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>

namespace h3
{
constexpr int kOpusRate = 48000;
constexpr int64_t kNanoseconds = 1000000000;

inline uint32_t read_le(const uint8_t* bytes, int count)
{
    uint32_t value = 0;
    for (int i = 0; i < count; i++)
        value |= uint32_t(bytes[i]) << (i * 8);
    return value;
}

inline void write_le(uint8_t* bytes, uint32_t value, int count)
{
    for (int i = 0; i < count; i++)
        bytes[i] = uint8_t(value >> (i * 8));
}

inline uint64_t media_ns(uint64_t index, int rate)
{
    return index / rate * kNanoseconds + index % rate * kNanoseconds / rate;
}

inline bool media_error(std::string& error, const std::string& reason)
{
    error = reason;
    return false;
}
}
