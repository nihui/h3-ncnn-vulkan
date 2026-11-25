// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace h3
{
struct SafeTensor
{
    std::vector<int> shape;
    uint64_t offset = 0;
    uint64_t elements = 0;
    enum Type
    {
        F32,
        F16,
        BF16
    } type;
};

class SafeTensors
{
public:
    SafeTensors(std::string& error, const std::string& path);
    const SafeTensor* tensor(std::string& error, const std::string& name, const std::vector<int>& shape) const;
    bool read(const SafeTensor& tensor, uint64_t offset, size_t count, float* values) const;
    bool read_bf16(const SafeTensor& tensor, uint64_t offset, size_t count, uint16_t* values) const;
    std::string content_key(std::string& error) const;

    std::map<std::string, SafeTensor> tensors;
    std::map<std::string, std::string> metadata;

private:
    std::string path_;
    mutable std::ifstream stream_;
};
}
