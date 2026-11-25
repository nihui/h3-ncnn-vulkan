// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "memory_budget.h"
#include <mat.h>

#include <memory>
#include <string>
#include <vector>

namespace ncnn
{
class Net;
}

namespace h3
{
struct LoraInput
{
    std::string path;
    float strength = 1.f;
};

class Lora
{
public:
    Lora(std::string& error, const std::vector<LoraInput>& inputs);
    ~Lora();

    bool configure(std::string& error, const std::string& model, bool reference);
    int default_steps() const;
    bool simple_schedule() const;
    int pi_steps() const;
    float video_shift(float base, bool reference) const;
    uint64_t loading_workspace(Component component) const;
    bool live(Component component) const;
    void add_live_graph(std::string& error, Component component, MemoryGraph& graph) const;
    const std::string& modulation_key() const;

    bool load_model(std::string& error, ncnn::Net& net, Component component, const std::string& path) const;
    bool make_modulations(std::string& error, const std::vector<float>& times, std::vector<ncnn::Mat>& values) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
