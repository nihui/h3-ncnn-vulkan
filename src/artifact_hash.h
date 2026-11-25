// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>

namespace h3
{
constexpr uint64_t kHashStart = 14695981039346656037ull;

inline uint64_t hash_bytes(uint64_t hash, const void* data, size_t bytes)
{
    const auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < bytes; i++)
        hash = (hash ^ p[i]) * 1099511628211ull;
    return hash;
}
}
