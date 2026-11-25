// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "h3_config.h"
#include "models.h"

#include <array>
#include <filesystem>
#include <fstream>

namespace h3
{
constexpr int kModulationTables = kDiTBlocks + 1;

class ModulationBank
{
public:
    bool prepare(std::string& error, H3ModelSet& models, bool reference,
        const std::vector<std::vector<float>>& schedule, int preset, bool write_portable = false,
        bool* needs_bake = nullptr);
    bool bind(std::string& error, int step, std::vector<ModelInput>& inputs);

private:
    bool open(const std::filesystem::path& path, const std::string& header);

    std::ifstream file_;
    std::vector<std::streamoff> offsets_;
    std::vector<int> counts_;
    std::array<std::string, kModulationTables> names_;
};
}
