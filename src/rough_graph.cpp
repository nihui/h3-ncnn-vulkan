// SPDX-License-Identifier: Apache-2.0
#include "rough_graph.h"
#include "h3_config.h"

#include <sstream>

namespace h3
{
namespace
{
bool add_fanout(std::string& error, MemoryGraph& result, std::vector<MemoryNode>& nodes)
{
    struct Buffer
    {
        size_t producer;
        std::vector<std::pair<size_t, size_t>> consumers;
    };
    std::map<std::string, Buffer> buffers;
    for (size_t i = 0; i < nodes.size(); i++)
    {
        const auto& node = nodes[i];
        for (size_t j = 0; j < node.inputs.size(); j++)
        {
            const auto it = buffers.find(node.inputs[j]);
            if (it == buffers.end())
            {
                error = "undefined input in rough graph: " + node.inputs[j];
                return false;
            }
            it->second.consumers.emplace_back(i, j);
        }
        for (const auto& name : node.outputs)
        {
            if (!buffers.emplace(name, Buffer { i, {} }).second)
            {
                error = "duplicate output in rough graph: " + name;
                return false;
            }
        }
    }

    size_t index = 0;
    std::vector<std::vector<MemoryNode>> splits(nodes.size());
    for (const auto& [name, buffer] : buffers)
    {
        if (buffer.consumers.size() < 2)
            continue;
        const std::string prefix = "rough_split_" + std::to_string(index++);
        MemoryNode split { "Split", prefix, { name }, {}, {}, {} };
        for (const auto& [row, edge] : buffer.consumers)
        {
            const std::string branch = prefix + "_" + std::to_string(split.outputs.size());
            nodes[row].inputs[edge] = branch;
            split.outputs.push_back(branch);
        }
        splits[buffer.producer].push_back(std::move(split));
    }

    const auto append = [&](MemoryNode&& node) {
        for (const auto& output : node.outputs)
            result.producers.emplace(output, result.nodes.size());
        result.nodes.push_back(std::move(node));
    };
    for (size_t i = 0; i < nodes.size(); i++)
    {
        append(std::move(nodes[i]));
        for (auto& split : splits[i])
            append(std::move(split));
    }
    return true;
}
}

MemoryGraph make_rough_graph(std::string& error, const MemoryGraph& core)
{
    std::vector<MemoryNode> nodes;
    std::map<std::string, std::string> aliases;
    int attentions = 0;
    bool block_pending = false, packed = false, video_cropped = false, audio_cropped = false;
    for (auto node : core.nodes)
    {
        for (auto& input : node.inputs)
        {
            const auto it = aliases.find(input);
            if (it != aliases.end())
                input = it->second;
        }
        if (node.type == "Split")
        {
            if (node.inputs.size() != 1)
            {
                error = "unsupported Split in transformer core";
                return {};
            }
            for (const auto& output : node.outputs)
                aliases.emplace(output, node.inputs[0]);
            continue;
        }

        if (node.type == "Gemm" && node.name.size() > 8 && node.name.compare(0, 6, "adaln_") == 0
            && (node.name.compare(8, std::string::npos, "_expanded") == 0
                || node.name.compare(8, std::string::npos, "_attention_expanded") == 0))
        {
            if (block_pending)
            {
                error = "missing attention in transformer block";
                return {};
            }
            block_pending = true;
        }
        else if (node.type == "SDPA" && block_pending)
        {
            if (node.inputs.size() != 3 || node.outputs.size() != 1)
            {
                error = "unsupported attention layout in " + node.name;
                return {};
            }
            const std::string prefix = "rough_kv_" + std::to_string(attentions);
            const std::string key = prefix + "_key", value = prefix + "_value";
            nodes.push_back({ "H3ReferenceKV", prefix, { node.inputs[1], node.inputs[2] }, { key, value },
                { { 0, std::to_string(attentions++) } }, {} });
            node.inputs[1] = key;
            node.inputs[2] = value;
            block_pending = false;
        }

        if (node.name == "pack_modalities")
        {
            if (node.type != "Concat" || node.inputs.size() != 3 || node.outputs != std::vector<std::string> { "hidden_0" })
            {
                error = "unsupported transformer modality packing";
                return {};
            }
            nodes.push_back({ "Concat", "pack_active", { node.inputs[1], node.inputs[2] }, { "active_modalities" },
                { { 0, "0" } }, {} });
            packed = true;
        }
        else if (node.name == "video_rows" || node.name == "audio_rows")
        {
            if (node.type != "Crop" || node.inputs.size() != 3 || node.outputs.size() != 1)
            {
                error = "unsupported transformer modality crop";
                return {};
            }
            node.quoted_parameters = { 19, 20, 21 };
            if (node.name == "video_rows")
            {
                node.inputs = { node.inputs[0], "video", "audio" };
                node.parameters = { { 19, "-(0h,+(1h,2h))" }, { 20, "-(0h,2h)" }, { 21, "0" } };
                video_cropped = true;
            }
            else
            {
                node.inputs = { node.inputs[0], "audio" };
                node.parameters = { { 19, "-(0h,1h)" }, { 20, "0h" }, { 21, "0" } };
                audio_cropped = true;
            }
        }
        nodes.push_back(std::move(node));
    }
    if (attentions != kDiTBlocks || block_pending || !packed || !video_cropped || !audio_cropped)
    {
        error = "transformer graph does not support rough mode";
        return {};
    }
    MemoryGraph result;
    result.loading_source_mapping = core.loading_source_mapping;
    if (!add_fanout(error, result, nodes))
        return {};
    return result;
}

std::string rough_graph_param(const MemoryGraph& graph)
{
    std::ostringstream out;
    out << "7767517\n" << graph.nodes.size() << ' ' << graph.producers.size() << '\n';
    for (const auto& node : graph.nodes)
    {
        out << node.type << ' ' << node.name << ' ' << node.inputs.size() << ' ' << node.outputs.size();
        for (const auto& input : node.inputs)
            out << ' ' << input;
        for (const auto& output : node.outputs)
            out << ' ' << output;
        for (const auto& [id, value] : node.parameters)
        {
            out << ' ' << id << '=';
            if (node.quoted_parameters.count(id))
                out << '"' << value << '"';
            else
                out << value;
        }
        out << '\n';
    }
    return out.str();
}
}
