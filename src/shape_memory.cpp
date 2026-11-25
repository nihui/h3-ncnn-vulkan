// SPDX-License-Identifier: Apache-2.0
#include "shape_memory.h"
#include "filesystem_utils.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <utility>
namespace h3
{
namespace
{
constexpr uint64_t MiB = 1024 * 1024;
uint64_t add(std::string& error, uint64_t a, uint64_t b)
{
    if (a > std::numeric_limits<uint64_t>::max() - b)
    {
        error = "memory shape addition overflow";
        return {};
    }
    return a + b;
}
uint64_t multiply(std::string& error, uint64_t a, uint64_t b)
{
    if (b && a > std::numeric_limits<uint64_t>::max() / b)
    {
        error = "memory shape multiplication overflow";
        return {};
    }
    return a * b;
}
uint64_t product(std::string& error, const std::vector<uint64_t>& values)
{
    uint64_t result = 1;
    for (uint64_t value : values)
    {
        result = multiply(error, result, value);
        if (!error.empty())
            return {};
    }

    return result;
}
std::vector<std::string> expressions(std::string& error, const std::string& text)
{
    std::vector<std::string> result;
    int depth = 0;
    size_t start = 0;
    for (size_t i = 0; i < text.size(); i++)
    {
        if (text[i] == '(')
            depth++;
        if (text[i] == ')')
            depth--;
        if (text[i] == ',' && !depth)
        {
            result.push_back(text.substr(start, i - start));
            start = i + 1;
        }
        if (depth < 0)
        {
            error = "invalid shape expression";
            return {};
        }
    }
    if (depth)
    {
        error = "invalid shape expression";
        return {};
    }
    result.push_back(text.substr(start));
    return result;
}
int64_t expression(std::string& error, const std::string& text, const std::vector<MemoryShape>& inputs)
{
    const auto open = text.find('(');
    if (open != std::string::npos)
    {
        if (text.back() != ')')
        {
            error = "invalid shape expression: " + text;
            return {};
        }
        const auto args = expressions(error, text.substr(open + 1, text.size() - open - 2));
        if (!error.empty())
            return {};

        if (args.size() != 2)
        {
            error = "unsupported shape expression: " + text;
            return {};
        }
        const int64_t a = expression(error, args[0], inputs), b = expression(error, args[1], inputs);
        if (!error.empty())
            return {};

        const std::string op = text.substr(0, open);
        if (a < 0 || b < 0)
        {
            error = "negative shape expression operand";
            return {};
        }
        if (op == "-")
            return a - b;
        if (op == "+")
            return static_cast<int64_t>(add(error, static_cast<uint64_t>(a), static_cast<uint64_t>(b)));
        if (op == "*")
            return static_cast<int64_t>(multiply(error, static_cast<uint64_t>(a), static_cast<uint64_t>(b)));
        if (op == "//" && b)
            return a / b;
        {
            error = "unsupported shape expression: " + text;
            return {};
        }
    }
    const char axis = text.empty() ? 0 : text.back();
    if (axis == 'w' || axis == 'h' || axis == 'd' || axis == 'c' || axis == 'n')
    {
        const size_t index = parse_u64(error, text.substr(0, text.size() - 1));
        if (!error.empty())
            return {};

        const auto& input = checked_at(error, inputs, index);
        if (!error.empty())
            return {};

        if (axis == 'n')
            return static_cast<int64_t>(input.batch);
        const auto& dims = input.dimensions;
        if (axis == 'w')
            return static_cast<int64_t>(checked_at(error, dims, 0));
        if (axis == 'h')
            return static_cast<int64_t>(dims.size() > 1 ? dims[1] : 1);
        if (axis == 'd')
            return static_cast<int64_t>(dims.size() == 4 ? dims[2] : 1);
        return static_cast<int64_t>(dims.size() > 2 ? dims.back() : 1);
    }
    size_t consumed = 0;
    const int64_t value = parse_i64(error, text, &consumed);
    if (!error.empty())
        return {};

    if (consumed != text.size())
    {
        error = "unsupported shape expression: " + text;
        return {};
    }
    return value;
}
MemoryShape shape(std::vector<uint64_t> dimensions, uint32_t bytes, bool packed = true)
{
    MemoryShape result { std::move(dimensions), bytes, 1 };
    return result.packed(packed);
}
size_t axis_index(std::string& error, const MemoryShape& value, int64_t axis)
{
    const int64_t rank = static_cast<int64_t>(value.dimensions.size());
    if (axis < 0)
        axis += rank;
    if (axis < 0 || axis >= rank)
    {
        error = "invalid memory shape axis";
        return {};
    }
    return static_cast<size_t>(rank - axis - 1);
}
struct Value
{
    MemoryShape shape;
    size_t storage;
};
struct Storage
{
    size_t allocation = 0;
    uint64_t bytes = 0, references = 0;
    bool external = false;
};
class GraphReplay
{
public:
    GraphReplay(
        std::string& error, const MemoryGraph& graph, const GraphMemoryRequest& request, const AllocationLayout& layout)
        : graph_(graph)
        , request_(request)
        , layout_(layout)
        , pool_(error, 16 * MiB, layout)
    {
    }
    GraphMemoryUsage run(std::string& error)
    {
        uint64_t retained_outputs = 0, final_outputs = 0, cpu_conversion = 0;
        if (request_.external_operators)
            for (size_t i = 0; i < graph_.nodes.size(); i++)
                if (graph_.nodes[i].type != "Input")
                {
                    execute(error, i);
                    if (!error.empty())
                        return {};
                }

        for (const auto& name : request_.outputs)
        {
            const Value output = evaluate(error, name);
            if (!error.empty())
                return {};

            usage_.output_shapes[name] = output.shape;
            const uint64_t native = output.shape.storage_bytes(error);
            if (!error.empty())
                return {};

            uint64_t host = native;
            if (!request_.raw_outputs)
            {
                host = multiply(error, output.shape.elements(error), 4);
                if (!error.empty())
                    return {};
            }

            if (layout_.gpu)
            {
                // record_download converts directly into HOST_CACHED staging. The
                // command retains it until its CPU copy/cast is complete.
                usage_.staging = add(error, usage_.staging, layout_.allocation_bytes(error, native, true));
                if (!error.empty())
                    return {};

                uint64_t conversion_peak = host;
                if (!request_.raw_outputs && output.shape.scalar_bytes == 2)
                {
                    conversion_peak = add(error, host, native);
                    if (!error.empty())
                        return {};
                }

                if (output.shape.packing != 1)
                {
                    conversion_peak = std::max(conversion_peak, multiply(error, host, 2));
                    if (!error.empty())
                        return {};
                }

                usage_.host_transfer_scratch
                    = std::max(usage_.host_transfer_scratch, add(error, retained_outputs, conversion_peak));
                if (!error.empty())
                    return {};
            }
            if (!layout_.gpu
                && ((!request_.raw_outputs && output.shape.scalar_bytes == 2) || output.shape.packing != 1))
            {
                cpu_conversion = std::max(cpu_conversion, add(error, final_outputs, host));
                if (!error.empty())
                    return {};
            }

            retained_outputs = add(error, retained_outputs, host);
            if (!error.empty())
                return {};

            if (!request_.raw_outputs && output.shape.packing != 1)
                if (!layout_.gpu
                    && ((!request_.raw_outputs && output.shape.scalar_bytes == 2) || output.shape.packing != 1))
                {
                    cpu_conversion = std::max(cpu_conversion, add(error, final_outputs, host));
                    if (!error.empty())
                        return {};
                }

            retained_outputs = add(error, retained_outputs, host);
            if (!error.empty())
                return {};

            final_outputs = add(error, final_outputs, host);
            if (!error.empty())
                return {};
        }
        usage_.host_outputs = final_outputs;
        usage_.device_pool = layout_.gpu ? pool_.allocated_bytes() : 0;
        usage_.device_live_peak = layout_.gpu ? pool_.peak_live_bytes() : 0;
        usage_.host_live_peak = layout_.gpu ? 0 : std::max(host_peak_, add(error, host_live_, cpu_conversion));
        if (!error.empty())
            return {};

        usage_.visited_nodes = visited_.size();
        return usage_;
    }

private:
    Value allocate(std::string& error, const MemoryShape& value, const std::string& node)
    {
        const uint64_t bytes = value.storage_bytes(error);
        if (!error.empty())
            return {};

        const uint64_t before = pool_.allocated_bytes();
        const size_t allocation = layout_.gpu ? pool_.allocate(error, bytes) : 0;
        if (!error.empty())
            return {};

        storage_.push_back({ allocation, bytes, 1, false });
        if (layout_.gpu && pool_.allocated_bytes() != before)
            usage_.peak_node = node;
        if (!layout_.gpu)
        {
            // Mat refcount, the documented NCNN_MALLOC_OVERREAD and alignment.
            host_live_ = add(error, host_live_, add(error, aligned_memory_bytes(error, add(error, bytes, 4), 64), 64));
            if (!error.empty())
                return {};

            if (host_live_ > host_peak_)
            {
                host_peak_ = host_live_;
                usage_.peak_node = node;
            }
        }
        return { value, storage_.size() - 1 };
    }
    Value alias(std::string& error, Value value)
    {
        checked_at(error, storage_, value.storage).references++;
        if (!error.empty())
            return {};

        return value;
    }
    void release(std::string& error, const Value& value)
    {
        auto& item = checked_at(error, storage_, value.storage);
        if (!error.empty())
            return;

        if (!item.references)
        {
            error = "memory graph released an inactive tensor";
            return;
        }
        if (--item.references || item.external)
            return;
        if (layout_.gpu)
        {
            pool_.release(error, item.allocation);
        }
        else
            host_live_ -= add(error, aligned_memory_bytes(error, add(error, item.bytes, 4), 64), 64);
    }
    Value evaluate(std::string& error, const std::string& name)
    {
        auto ready = values_.find(name);
        if (ready != values_.end())
            return ready->second;
        auto alias_input = request_.aliases.find(name);
        if (alias_input != request_.aliases.end())
        {
            const Value value = alias(error, evaluate(error, alias_input->second));
            if (!error.empty())
                return {};

            values_[name] = value;
            return value;
        }
        auto device = request_.device_inputs.find(name);
        if (device != request_.device_inputs.end())
        {
            storage_.push_back({ 0, 0, 2, true });
            const Value value { device->second, storage_.size() - 1 };
            values_[name] = value;
            return value;
        }
        auto input = request_.inputs.find(name);
        if (input != request_.inputs.end())
        {
            MemoryShape value = input->second;
            if (layout_.gpu)
            {
                value.scalar_bytes = layout_.bf16 && !request_.raw_inputs.count(name) ? 2 : 4;
                value = value.packed();
                const uint64_t uploaded = value.storage_bytes(error);
                if (!error.empty())
                    return {};

                usage_.staging = add(error, usage_.staging, layout_.allocation_bytes(error, uploaded, true));
                if (!error.empty())
                    return {};

                if (layout_.bf16 && input->second.scalar_bytes == 4 && layout_.discrete)
                    usage_.host_transfer_scratch = std::max(usage_.host_transfer_scratch, uploaded);
            }
            else
            {
                value.packing = 1;
                if (request_.preserve_host_inputs)
                {
                    storage_.push_back({ 0, 0, 2, true });
                    const Value tensor { value, storage_.size() - 1 };
                    values_[name] = tensor;
                    return tensor;
                }
            }
            const Value tensor = allocate(error, value, "upload " + name);
            if (!error.empty())
                return {};

            if (!layout_.gpu)
            {
                checked_at(error, storage_, tensor.storage).references++;
                if (!error.empty())
                    return {};
            }

            values_[name] = tensor;
            return tensor;
        }
        auto producer = graph_.producers.find(name);
        if (producer == graph_.producers.end())
        {
            error = "memory graph has no producer for " + name;
            return {};
        }
        const auto& node = checked_at(error, graph_.nodes, producer->second);
        if (!error.empty())
            return {};

        if (node.type == "Input")
        {
            error = "memory graph is missing input shape: " + name;
            return {};
        }
        execute(error, producer->second);
        if (!error.empty())
            return {};

        return checked_at(error, values_, name);
    }
    std::vector<MemoryShape> outputs(
        std::string& error, const MemoryNode& node, const std::vector<MemoryShape>& input) const
    {
        const auto get = [&](int key, int64_t fallback = 0) { return node.integer(error, key, fallback); };
        const MemoryShape& first = input.empty() ? MemoryShape {} : input[0];
        const auto make = [&](std::vector<uint64_t> dimensions, bool packing = true)
        { return shape(std::move(dimensions), first.scalar_bytes, layout_.gpu && packing); };
        if (node.type == "Gemm")
        {
            const bool constant_a = get(4) != 0, constant_b = get(5) != 0;

            const auto& a_shape = first.dimensions;
            const auto& b_shape = constant_b ? a_shape : checked_at(error, input, constant_a ? 0 : 1).dimensions;
            if (!error.empty())
                return {};

            const uint64_t ka = constant_a ? static_cast<uint64_t>(get(9)) : checked_at(error, a_shape, get(2) ? 1 : 0);
            if (!error.empty())
                return {};

            const uint64_t kb = constant_b ? static_cast<uint64_t>(get(9)) : checked_at(error, b_shape, get(3) ? 0 : 1);
            if (!error.empty())
                return {};

            if (!ka || ka != kb || (get(9) && ka != static_cast<uint64_t>(get(9))))
            {
                error = "Gemm input/weight K dimensions differ: " + node.name;
                return {};
            }
            const uint64_t m = constant_a ? static_cast<uint64_t>(get(7)) : checked_at(error, a_shape, get(2) ? 0 : 1);
            if (!error.empty())
                return {};

            const uint64_t n = constant_b ? static_cast<uint64_t>(get(8)) : checked_at(error, b_shape, get(3) ? 1 : 0);
            if (!error.empty())
                return {};

            if (get(11) || get(14))
            {
                error = "unsupported H3 Gemm output layout: " + node.name;
                return {};
            }
            auto result = make({ n, m });
            result.scalar_bytes = layout_.bf16 && !(get(31) & 4) ? 2 : 4;
            return { result };
        }
        if (node.type == "Reshape")
        {
            std::vector<int64_t> dimensions;
            auto formula = node.parameters.find(6);
            if (formula != node.parameters.end())
                for (const auto& term : expressions(error, formula->second))
                {
                    dimensions.push_back(expression(error, term, input));
                    if (!error.empty())
                        return {};
                }
            else
            {
                dimensions.push_back(get(0, -233));
                if (node.parameters.count(1))
                    dimensions.push_back(get(1));
                if (node.parameters.count(11))
                    dimensions.push_back(get(11));
                if (node.parameters.count(2))
                    dimensions.push_back(get(2));
            }

            const int64_t input_axis = get(12, 233), output_axis = get(13, 233);
            const bool explicit_batch = input_axis != 233 || output_axis != 233;
            auto logical_input = first.dimensions;
            const auto axis_in_whdc_order = [](int64_t axis, size_t rank) {
                const auto count = static_cast<int64_t>(rank);
                return count - 1 - (axis < 0 ? axis + count : axis);
            };
            if (input_axis != 233)
            {
                const int64_t axis = axis_in_whdc_order(input_axis, logical_input.size() + 1);
                if (axis < 0 || axis > static_cast<int64_t>(logical_input.size()))
                {
                    error = "invalid reshape input batch axis at " + node.name;
                    return {};
                }
                logical_input.insert(logical_input.begin() + axis, first.batch);
            }
            else if (explicit_batch && first.batch != 1)
            {
                error = "reshape requires an input batch axis at " + node.name;
                return {};
            }

            const uint64_t input_total = first.elements(error);
            if (!error.empty())
                return {};
            const uint64_t total = explicit_batch ? input_total : input_total / first.batch;
            uint64_t known = 1;
            int inferred = -1;
            for (size_t i = 0; i < dimensions.size(); i++)
            {
                if (!dimensions[i])
                {
                    if (explicit_batch)
                    {
                        const size_t outer_index = dimensions.size() - 1 - i;
                        if (outer_index >= logical_input.size())
                        {
                            error = "invalid reshape dimension reference at " + node.name;
                            return {};
                        }
                        dimensions[i] = static_cast<int64_t>(logical_input[logical_input.size() - 1 - outer_index]);
                    }
                    else
                    {
                        dimensions[i] = static_cast<int64_t>(checked_at(error, logical_input, i));
                        if (!error.empty())
                            return {};
                    }
                }

                if (dimensions[i] == -1 && inferred == -1)
                    inferred = static_cast<int>(i);
                else if (dimensions[i] > 0)
                    known = multiply(error, known, static_cast<uint64_t>(dimensions[i]));
                else
                {
                    error = "invalid reshape dimension at " + node.name;
                    return {};
                }
                if (!error.empty())
                    return {};
            }
            if (inferred != -1 && total % known == 0)
            {
                dimensions[static_cast<size_t>(inferred)] = static_cast<int64_t>(total / known);
                known = total;
            }
            if (known != total || (inferred != -1 && dimensions[static_cast<size_t>(inferred)] <= 0))
            {
                error = "memory reshape changes the number of elements: " + node.name;
                return {};
            }

            uint64_t batch = explicit_batch ? 1 : first.batch;
            if (output_axis != 233)
            {
                const int64_t axis = axis_in_whdc_order(output_axis, dimensions.size());
                if (axis < 0 || axis >= static_cast<int64_t>(dimensions.size()))
                {
                    error = "invalid reshape output batch axis at " + node.name;
                    return {};
                }
                batch = static_cast<uint64_t>(dimensions[axis]);
                dimensions.erase(dimensions.begin() + axis);
            }
            auto result = make(std::vector<uint64_t>(dimensions.begin(), dimensions.end()));
            result.batch = batch;
            return { result };
        }
        if (node.type == "Permute")
        {
            // In ncnn each group moves one axis to the outermost position, then
            // permutes the remaining inner axes in the same order as rank 3.
            const int64_t order = get(0);
            const size_t rank = first.dimensions.size();
            auto dimensions = first.dimensions;
            if (rank == 2 && order == 1)
                std::swap(dimensions[0], dimensions[1]);
            else if ((rank == 3 || rank == 4) && order >= 0 && order < (rank == 3 ? 6 : 24))
            {
                static constexpr int inner[6][3] = {{0, 1, 2}, {1, 0, 2}, {0, 2, 1},
                    {2, 0, 1}, {1, 2, 0}, {2, 1, 0}};
                if (rank == 4)
                {
                    const size_t outer = 3 - order / 6;
                    dimensions.erase(dimensions.begin() + outer);
                    dimensions.push_back(first.dimensions[outer]);
                }
                const auto remaining = dimensions;
                for (int i = 0; i < 3; i++)
                    dimensions[i] = remaining[inner[order % 6][i]];
            }
            else if (order)
            {
                error = "unsupported permutation at " + node.name;
                return {};
            }
            return { make(std::move(dimensions)) };
        }
        if (node.type == "Slice")
        {
            const size_t axis = axis_index(error, first, get(1));
            if (!error.empty())
                return {};

            auto terms = expressions(error, checked_at(error, node.parameters, -23300));
            if (!error.empty())
                return {};

            if (parse_u64(error, checked_at(error, terms, 0)) != node.outputs.size())
            {
                error = "memory slice count differs from graph";
                return {};
            }
            std::vector<MemoryShape> result;
            uint64_t offset = 0;
            for (size_t i = 0; i < node.outputs.size(); i++)
            {
                const int64_t term = parse_i64(error, terms[i + 1]);
                if (!error.empty())
                    return {};

                const uint64_t length = term == -233 ? (first.dimensions[axis] - offset) / (node.outputs.size() - i)
                                                     : static_cast<uint64_t>(term);
                auto dimensions = first.dimensions;
                dimensions[axis] = length;
                result.push_back(make(std::move(dimensions)));
                offset = add(error, offset, length);
                if (!error.empty())
                    return {};
            }
            if (offset != first.dimensions[axis])
            {
                error = "memory slice does not cover its input: " + node.name;
                return {};
            }
            return result;
        }
        if (node.type == "Concat")
        {
            auto dimensions = first.dimensions;
            const size_t axis = axis_index(error, first, get(0));
            if (!error.empty())
                return {};

            dimensions[axis] = 0;
            for (const auto& value : input)
            {
                dimensions[axis] = add(error, dimensions[axis], checked_at(error, value.dimensions, axis));
                if (!error.empty())
                    return {};
            }

            return { make(std::move(dimensions)) };
        }
        if (node.type == "Crop")
        {
            auto dimensions = first.dimensions;
            const bool dynamic = node.parameters.count(19);
            const auto list = [&](int key) { return expressions(error, checked_at(error, node.parameters, key)); };
            const auto starts = list(dynamic ? 19 : -23309);
            const auto ends = list(dynamic ? 20 : -23310);
            const auto axes = list(dynamic ? 21 : -23311);
            const size_t offset = dynamic ? 0 : 1;
            if (!error.empty() || starts.size() != ends.size() || starts.size() != axes.size() || starts.size() <= offset)
            {
                error = "unsupported crop format at " + node.name;
                return {};
            }
            for (size_t i = offset; i < starts.size(); i++)
            {
                const size_t axis = axis_index(error, first, parse_i64(error, axes[i]));
                const int64_t extent = checked_at(error, first.dimensions, axis);
                int64_t begin = expression(error, starts[i], input), end = expression(error, ends[i], input);
                begin = begin < 0 ? begin + extent : begin;
                end = end == -233 ? extent : end < 0 ? end + extent : end;
                if (!error.empty() || begin < 0 || end <= begin || end > extent)
                {
                    error = "invalid crop bounds at " + node.name;
                    return {};
                }
                dimensions[axis] = end - begin;
            }
            return { make(std::move(dimensions)) };
        }
        if (node.type == "BinaryOp")
        {
            auto dimensions = first.dimensions;
            if (input.size() > 1)
            {
                if (input[1].dimensions.size() != dimensions.size())
                {
                    error = "unsupported mixed-rank H3 broadcast: " + node.name;
                    return {};
                }
                for (size_t i = 0; i < dimensions.size(); i++)
                {
                    const uint64_t other = input[1].dimensions[i];
                    if (dimensions[i] != other && dimensions[i] != 1 && other != 1)
                    {
                        error = "incompatible H3 broadcast: " + node.name;
                        return {};
                    }
                    dimensions[i] = std::max(dimensions[i], other);
                }
            }
            return { make(std::move(dimensions)) };
        }
        if (node.type == "ResourceKV")
        {
            auto a = first, b = checked_at(error, input, 1);
            if (!error.empty())
                return {};

            if (get(0))
            {
                checked_at(error, a.dimensions, 1) = checked_at(error, b.dimensions, 1) = static_cast<uint64_t>(get(1));
                if (!error.empty())
                    return {};
            }

            return { a, b };
        }
        if (node.type == "ResourceImageRows")
            return { first };
        if (node.type == "SDPA")
        {
            auto result = make({ checked_at(error, input, 2).dimensions[0], checked_at(error, first.dimensions, 1),
                                   checked_at(error, first.dimensions, 2) },
                false);
            if (!error.empty())
                return {};

            if (!layout_.gpu)
                result.scalar_bytes = 4; // Native CPU SDPA accumulates/returns FP32 even with BF16 Q/K/V.
            return { result };
        }
        if (node.type == "Convolution" || node.type == "Convolution3D" || node.type == "ConvolutionDepthWise"
            || node.type == "Deconvolution" || node.type == "DeconvolutionDepthWise")
        {
            const bool transposed = node.type.find("Deconvolution") == 0;
            const int rank = node.type == "Convolution3D" ? 3 : 2;
            auto dimensions = first.dimensions;
            if ((int)dimensions.size() != rank + 1)
            {
                error = "unsupported convolution rank at " + node.name;
                return {};
            }
            const int64_t kernel[] = { get(1), get(11, get(1)), get(21, get(1)) };
            const int64_t stride[] = { get(3, 1), get(13, get(3, 1)), get(23, get(3, 1)) };
            const int64_t dilation[] = { get(2, 1), get(12, get(2, 1)), get(22, get(2, 1)) };
            const int64_t left[] = { get(4), get(14, get(4)), get(24, get(4)) };
            const int64_t right[] = { get(15, left[0]), get(16, left[1]), get(17, left[2]) };
            const int64_t extra[] = { get(18), get(19, get(18)), 0 };
            for (int axis = 0; axis < rank; axis++)
            {
                const int64_t extent = dimensions[axis], field = (kernel[axis] - 1) * dilation[axis] + 1;
                if (stride[axis] < 1 || field < 1 || left[axis] < 0 || right[axis] < 0
                    || (!transposed && extent + left[axis] + right[axis] < field))
                {
                    error = "unsupported convolution geometry at " + node.name;
                    return {};
                }
                const int64_t size = transposed
                    ? (extent - 1) * stride[axis] + field + extra[axis] - left[axis] - right[axis]
                    : (extent + left[axis] + right[axis] - field) / stride[axis] + 1;
                if (size <= 0)
                {
                    error = "invalid convolution output at " + node.name;
                    return {};
                }
                dimensions[axis] = size;
            }
            dimensions.back() = get(0);
            return { make(std::move(dimensions)) };
        }
        if (node.type == "Reduction")
        {
            auto dimensions = first.dimensions;
            std::vector<int> reduced(dimensions.size(), get(1, 1) != 0);
            if (!get(1, 1))
            {
                const auto axes = expressions(error, checked_at(error, node.parameters, -23303));
                for (size_t i = 1; i < axes.size(); i++)
                    checked_at(error, reduced, axis_index(error, first, parse_i64(error, axes[i]))) = true;
            }
            std::vector<uint64_t> result;
            for (size_t i = 0; i < dimensions.size(); i++)
                if (!reduced[i] || get(4))
                    result.push_back(reduced[i] ? 1 : dimensions[i]);
            if (result.empty())
                result.push_back(1);
            return { make(std::move(result), false) };
        }
        if (node.type == "Padding")
        {
            auto dimensions = first.dimensions;
            checked_at(error, dimensions, 0) = add(error, dimensions[0], static_cast<uint64_t>(get(2) + get(3)));
            if (!error.empty())
                return {};

            checked_at(error, dimensions, 1) = add(error, dimensions[1], static_cast<uint64_t>(get(0) + get(1)));
            if (!error.empty())
                return {};

            if (dimensions.size() >= 3)
                dimensions[2] += get(7) + get(8);
            return { make(std::move(dimensions)) };
        }
        if (node.type == "MemoryData")
        {
            std::vector<uint64_t> dimensions;
            for (int key : { 0, 1, 11, 2 })
                if (get(key))
                    dimensions.push_back(get(key));
            return { shape(std::move(dimensions), layout_.bf16 ? 2 : 4, layout_.gpu) };
        }
        if (node.type == "Packing")
        {
            auto result = first;
            const auto pack = get(0, 1);
            result.packing = pack > 0 && result.dimensions.back() % pack == 0 ? pack : 1;
            if (get(3))
                result.scalar_bytes = get(3) == 1 ? 4 : 2;
            return { result };
        }
        if (node.type == "RMSNorm" || node.type == "LayerNorm" || node.type == "GroupNorm" || node.type == "Swish"
            || node.type == "GELU" || node.type == "RotaryEmbed" || node.type == "UnaryOp"
            || node.type == "Snake")
            return { first };
        {
            error = "no allocation recipe for H3 operator: " + node.type + " (" + node.name + ")";
            return {};
        }
    }
    void group_norm_scratch(std::string& error, const MemoryNode& node, const MemoryShape& input)
    {
        if (!layout_.gpu)
            return;

        // ncnn invokes GroupNorm on each batch view. Workspace is freed after
        // recording that frame, so the allocations are reused by the next one.
        const uint64_t groups = static_cast<uint64_t>(node.integer(error, 0, 1));
        if (!error.empty())
            return;

        const auto mean = allocate(error, shape({ groups }, 4, false), node.name + " mean");
        if (!error.empty())
            return;
        const auto variance = allocate(error, shape({ groups }, 4, false), node.name + " variance");
        if (!error.empty())
            return;

        const uint64_t channels = input.dimensions.back();
        if (!layout_.subgroup_reductions)
        {
            const uint64_t area = input.elements(error) / input.batch / channels;
            if (!error.empty())
                return;
            const auto buffer = [&](uint64_t width) {
                return allocate(error, shape({ width, 1, channels }, 4, input.packing == 4), node.name + " reduction");
            };
            const auto reduce = [&](uint64_t width, Value sums) {
                while (width > 4 && error.empty())
                {
                    width = (width + 3) / 4;
                    const auto next = buffer(width);
                    if (!error.empty())
                        return;
                    release(error, sums);
                    sums = next;
                }
                if (error.empty())
                    release(error, sums);
            };

            const uint64_t width = (area + 3) / 4;
            reduce(width, buffer(width));
            if (!error.empty())
                return;
            const auto squares = buffer(area);
            if (!error.empty())
                return;
            reduce(area, alias(error, squares));
            release(error, squares);
            if (!error.empty())
                return;
        }
        const auto coefficients = allocate(error, shape({ channels * 2 }, input.scalar_bytes, false),
            node.name + " coefficients");
        if (!error.empty())
            return;
        release(error, coefficients);
        release(error, variance);
        release(error, mean);
    }
    void normalization_scratch(
        std::string& error, const MemoryNode& node, const MemoryShape& input, std::vector<Value>& scratch)
    {
        const bool layer_norm = node.type == "LayerNorm";
        const uint64_t affine = static_cast<uint64_t>(node.integer(error, 0));
        if (!error.empty())
            return;

        if (!affine)
            return;
        const uint64_t groups = input.elements(error) / affine;
        if (!error.empty())
            return;

        const auto temp
            = [&](const MemoryShape& s) { scratch.push_back(allocate(error, s, node.name + " workspace")); };
        temp(shape({ groups }, 4, false));
        if (layer_norm)
            temp(shape({ groups }, 4, false));
        if (layout_.gpu && !layout_.subgroup_reductions)
        {
            temp(input);
            uint64_t length = affine;

            while (length > 1)
            {
                length = (length + 3) / 4;
                temp(shape({ length, groups }, 4, false));
                // The previous reduction is replaced by the new one. Square and
                // mean/variance persist until the normalization has finished.
                if (scratch.size() > static_cast<size_t>(layer_norm ? 5 : 4))
                {
                    release(error, scratch[scratch.size() - 2]);
                    if (!error.empty())
                        return;

                    scratch.erase(scratch.end() - 2);
                }
            }
        }
        temp(shape({ multiply(error, groups, layer_norm ? 2 : 1) }, 4, false));
        if (!error.empty())
            return;
    }
    void execute(std::string& error, size_t index)
    {
        if (!visited_.insert(index).second)
            return;
        const auto& node = graph_.nodes[index];
        std::vector<Value> inputs;
        std::vector<MemoryShape> input_shapes;
        for (const auto& name : node.inputs)
        {
            inputs.push_back(evaluate(error, name));
            if (!error.empty())
                return;

            input_shapes.push_back(inputs.back().shape);
        }
        if (node.type != "Reshape" && node.type != "GroupNorm" && node.type != "Split"
            && std::any_of(input_shapes.begin(), input_shapes.end(), [](const MemoryShape& shape) { return shape.batch > 1; }))
        {
            error = "no batched allocation recipe for " + node.name;
            return;
        }
        std::vector<Value> layout_copies;
        usage_.execution_order.push_back(node.name);
        if (!layout_.gpu && !request_.external_operators && node.type != "ResourceKV"
            && node.type != "ResourceImageRows")
        {
            const bool supported = layout_.cpu_bf16_support ? layout_.cpu_bf16_support(error, node.type) : true;
            if (!error.empty())
                return;

            const uint32_t scalar = layout_.bf16 && supported ? 2 : 4;
            for (size_t i = 0; i < inputs.size(); i++)
                if (input_shapes[i].scalar_bytes != scalar)
                {
                    input_shapes[i].scalar_bytes = scalar;
                    const auto converted = allocate(error, input_shapes[i], node.name + " storage conversion");
                    if (!error.empty())
                        return;

                    layout_copies.push_back(converted);
                    inputs[i] = converted;
                }
        }
        if (layout_.gpu)
            for (size_t i = 0; i < input_shapes.size(); i++)
            {
                auto desired = input_shapes[i].packed(node.type != "SDPA");
                if (layout_.bf16 && (node.integer(error, 31) & 4) && node.type != "Packing")
                    desired.scalar_bytes = 4;
                if (desired.packing != input_shapes[i].packing || desired.scalar_bytes != input_shapes[i].scalar_bytes)
                {
                    const auto converted = allocate(error, desired, node.name + " packing conversion");
                    if (!error.empty())
                        return;

                    layout_copies.push_back(converted);
                    inputs[i] = converted;
                    input_shapes[i] = desired;
                }
            }
        std::vector<Value> result;
        if (node.type == "H3ReferenceKV")
        {
            for (size_t i = 0; i < 2; i++)
            {
                if (!request_.kv_full_rows)
                {
                    result.push_back(alias(error, inputs[i]));
                    continue;
                }
                const uint64_t active = request_.kv_active_video + request_.kv_active_audio;
                std::vector<Value> slices;
                auto shape = input_shapes[i];
                if (request_.kv_has_cached_audio)
                {
                    shape.dimensions[1] = request_.kv_active_video;
                    slices.push_back(allocate(error, shape, node.name + " generated video K/V"));
                    shape.dimensions[1] = request_.kv_active_audio;
                    slices.push_back(allocate(error, shape, node.name + " generated audio K/V"));
                }
                else if (active != shape.dimensions[1])
                {
                    shape.dimensions[1] = active;
                    slices.push_back(allocate(error, shape, node.name + " cropped padding K/V"));
                }
                shape.dimensions[1] = request_.kv_full_rows;
                result.push_back(allocate(error, shape, node.name + " concatenated K/V"));
                for (const auto& slice : slices)
                    release(error, slice);
            }
        }
        else if (node.type == "ResourceKV")
        {
            for (size_t i = 0; i < 2; i++)
            {
                Value prefix;
                const bool host_transfer = layout_.gpu && node.integer(error, 2);
                if (!error.empty())
                    return;

                if (host_transfer)
                {
                    auto prefix_shape = checked_at(error, input_shapes, i);
                    if (!error.empty())
                        return;

                    checked_at(error, prefix_shape.dimensions, 1) = static_cast<uint64_t>(node.integer(error, 3));
                    if (!error.empty())
                        return;

                    prefix = allocate(error, prefix_shape, node.name + " prefix transfer");
                    if (!error.empty())
                        return;

                    usage_.staging = add(error, usage_.staging,
                        layout_.allocation_bytes(error, prefix_shape.storage_bytes(error), true));
                    if (!error.empty())
                        return;
                }
                if (node.integer(error, 0))
                {
                    result.push_back(allocate(error, checked_at(error, outputs(error, node, input_shapes), i),
                        node.name + " concatenated prefix"));
                    if (!error.empty())
                        return;
                }
                else
                    result.push_back(alias(error, checked_at(error, inputs, i)));
                if (host_transfer)
                {
                    release(error, prefix);
                    if (!error.empty())
                        return;
                }
            }
        }
        else if (node.type == "Split")
        {
            for (size_t i = 0; i < node.outputs.size(); i++)
            {
                result.push_back(alias(error, checked_at(error, inputs, 0)));
                if (!error.empty())
                    return;
            }
        }
        else
        {
            const auto out_shapes = node.type == "Noop"
                ? std::vector<MemoryShape> { checked_at(error, input_shapes, 0) }
                : outputs(error, node, input_shapes);
            if (!error.empty())
                return;

            const bool inplace = !request_.external_operators
                && (node.type == "RMSNorm" || node.type == "LayerNorm" || node.type == "GroupNorm"
                    || node.type == "Swish" || node.type == "GELU" || node.type == "Noop" || node.type == "UnaryOp"
                    || node.type == "Snake"
                    || node.type == "ResourceImageRows" || (node.type == "BinaryOp" && node.integer(error, 1)));
            if (!error.empty())
                return;

            std::vector<Value> gemm_scratch;
            if (layout_.gpu && node.type == "Gemm" && node.integer(error, 6) && node.integer(error, 10, -1) != -1)
            {
                const uint64_t m = out_shapes[0].dimensions[1], n = out_shapes[0].dimensions[0];
                const int64_t type = node.integer(error, 10);
                if (!error.empty())
                    return;

                const uint64_t count = type == 0 ? 1
                    : type == 1 || type == 2     ? m
                    : type == 4                  ? n
                                                 : multiply(error, m, n);
                if (!error.empty())
                    return;

                usage_.weight_early_tensors[node.name] = { static_cast<size_t>(node.integer(error, 4) != 0)
                    + static_cast<size_t>(node.integer(error, 5) != 0) };
                if (!error.empty())
                    return;

                if (count % 4 == 0)
                {
                    gemm_scratch.push_back(
                        allocate(error, shape({ count }, layout_.bf16 ? 2 : 4, false), node.name + " bias unpack"));
                    if (!error.empty())
                        return;
                }
            }
            std::vector<Value> attention_scratch;
            if (!request_.external_operators && layout_.gpu && node.type == "SDPA")
            {
                const auto& q = input_shapes[0];
                const auto& k = input_shapes[1];
                const auto& v = input_shapes[2];
                const bool flash = layout_.bf16 && layout_.flash_attention && q.dimensions[0] % 8 == 0
                    && v.dimensions[0] % 8 == 0 && v.dimensions[0] <= layout_.flash_output_limit;
                if (!flash)
                {
                    attention_scratch.push_back(allocate(error,
                        shape({ k.dimensions[1], q.dimensions[1], q.dimensions[2] }, q.scalar_bytes, false),
                        node.name + " attention scores"));
                    if (!error.empty())
                        return;

                    const auto maximum = allocate(error,
                        shape({ q.dimensions[1], q.dimensions[2] }, q.scalar_bytes, false), node.name + " softmax max");
                    if (!error.empty())
                        return;

                    const auto total = allocate(error,
                        shape({ q.dimensions[1], q.dimensions[2] }, q.scalar_bytes, false), node.name + " softmax sum");
                    if (!error.empty())
                        return;

                    release(error, maximum);
                    if (!error.empty())
                        return;

                    release(error, total);
                    if (!error.empty())
                        return;
                }
            }
            std::vector<Value> convolution_scratch;
            if (!request_.external_operators && node.type == "Convolution")
            {
                const auto& in = input_shapes[0];
                const auto& out = out_shapes[0];
                const bool eligible = node.integer(error, 1) == 3 && node.integer(error, 11, 3) == 3
                    && node.integer(error, 3, 1) == 1 && node.integer(error, 13, 1) == 1 && in.dimensions[2] >= 16
                    && out.dimensions[2] >= 16;
                if (!error.empty())
                    return;

                if (layout_.gpu && eligible && (layout_.winograd23 || layout_.winograd43))
                {
                    bool use43 = layout_.winograd43;
                    const uint64_t w = in.dimensions[0], h = in.dimensions[1];
                    if (layout_.winograd23
                        && ((w <= 18 && h <= 18) || (layout_.discrete && w >= 23 && w <= 24 && h >= 23 && h <= 24)))
                        use43 = false;
                    const uint64_t tile = use43 ? 4 : 2, coefficients = use43 ? 36 : 16;
                    const size_t selected = use43 || !layout_.winograd43 ? 0 : 1;
                    usage_.weight_tensor_selection[node.name] = { selected };
                    if (node.integer(error, 5))
                        usage_.weight_tensor_selection[node.name].push_back(
                            static_cast<size_t>(layout_.winograd23) + static_cast<size_t>(layout_.winograd43));
                    const uint64_t blocks
                        = multiply(error, (out.dimensions[0] + tile - 1) / tile, (out.dimensions[1] + tile - 1) / tile);
                    if (!error.empty())
                        return;

                    convolution_scratch.push_back(allocate(error,
                        shape({ blocks, 1, multiply(error, in.dimensions[2], coefficients) }, in.scalar_bytes),
                        node.name + " Winograd input"));
                    if (!error.empty())
                        return;

                    convolution_scratch.push_back(allocate(error,
                        shape({ blocks, 1, multiply(error, out.dimensions[2], coefficients) }, out.scalar_bytes),
                        node.name + " Winograd output"));
                    if (!error.empty())
                        return;
                }
            }
            if (layout_.gpu && (node.type.find("Convolution") != std::string::npos
                    || node.type.find("Deconvolution") != std::string::npos))
            {
                const auto get = [&](int key, int64_t fallback = 0) { return node.integer(error, key, fallback); };
                const bool transposed = node.type.find("Deconvolution") == 0;
                auto padded = transposed ? out_shapes[0] : input_shapes[0];
                const auto horizontal = get(4) + get(15, get(4));
                const auto vertical = get(14, get(4)) + get(16, get(14, get(4)));
                const auto depth = padded.dimensions.size() == 4 ? get(24, get(4)) + get(17, get(24, get(4))) : 0;
                if (horizontal || vertical || depth)
                {
                    padded.dimensions[0] += horizontal;
                    padded.dimensions[1] += vertical;
                    if (depth)
                        padded.dimensions[2] += depth;
                    convolution_scratch.push_back(allocate(error, padded, node.name + " border"));
                }
            }
            for (const auto& out : out_shapes)
            {
                bool view = false;
                if (node.type == "Reshape")
                {
                    const auto& in = checked_at(error, input_shapes, 0);
                    if (!error.empty())
                        return;

                    if (in.batch == 1 && out.batch == 1 && in.packing == out.packing)
                    {
                        const auto in_bytes = in.storage_bytes(error), out_bytes = out.storage_bytes(error);
                        const bool explicit_batch = node.integer(error, 12, 233) != 233 || node.integer(error, 13, 233) != 233;
                        if (explicit_batch)
                            view = in_bytes == out_bytes && in_bytes == in.elements(error) * in.scalar_bytes
                                && (in.packing == 1 || in.dimensions.back() == out.dimensions.back());
                        else
                            view = in.dimensions.size() == out.dimensions.size()
                                && in.dimensions.back() == out.dimensions.back() && in_bytes == out_bytes;
                    }
                    if (!error.empty())
                        return;
                }
                if ((inplace && checked_at(error, storage_, checked_at(error, inputs, 0).storage).references == 1)
                    || view)
                {
                    auto value = alias(error, checked_at(error, inputs, 0));
                    if (!error.empty())
                        return;

                    value.shape = out;
                    result.push_back(value);
                }
                else
                    result.push_back(allocate(error, out, node.name));
            }
            std::vector<Value> scratch;
            if (!request_.external_operators)
            {
                if (node.type == "GroupNorm")
                    group_norm_scratch(error, node, checked_at(error, input_shapes, 0));
                else if (node.type == "RMSNorm" || node.type == "LayerNorm")
                    normalization_scratch(error, node, checked_at(error, input_shapes, 0), scratch);
                if (!error.empty())
                    return;
            }

            if (!request_.external_operators && !layout_.gpu && node.type == "Convolution")
            {
                const auto& in = checked_at(error, input_shapes, 0);
                if (!error.empty())
                    return;

                const auto& out = checked_at(error, out_shapes, 0);
                if (!error.empty())
                    return;

                const uint64_t channels = checked_at(error, in.dimensions, 2),
                               outputs = checked_at(error, out.dimensions, 2);
                if (!error.empty())
                    return;

                const bool winograd = node.integer(error, 1) == 3 && node.integer(error, 11, 3) == 3
                    && node.integer(error, 3, 1) == 1 && node.integer(error, 13, 1) == 1
                    && (channels > 8 || outputs > 8);
                if (!error.empty())
                    return;

                const uint64_t coefficients = winograd ? (channels <= 32 && outputs <= 32 ? 64 : 36) : 1;
                const uint64_t pixels_per_tile = coefficients == 64 ? 6 : 4;
                const uint64_t rows = winograd ? ((out.dimensions[0] + pixels_per_tile - 1) / pixels_per_tile)
                        * ((out.dimensions[1] + pixels_per_tile - 1) / pixels_per_tile)
                                               : out.dimensions[0] * out.dimensions[1];
                const uint64_t k = winograd ? channels
                                            : channels
                        * static_cast<uint64_t>(
                            node.integer(error, 1) * node.integer(error, 11, node.integer(error, 1)));
                if (!error.empty())
                    return;

                const auto tile = layout_.cpu_convolution_tile_shape
                    ? layout_.cpu_convolution_tile_shape(error, outputs, rows, k, winograd)
                    : std::array<uint64_t, 3> { outputs, rows, k };
                const uint64_t nk = (k + tile[2] - 1) / tile[2], nn = (rows + tile[1] - 1) / tile[1];
                scratch.push_back(allocate(error, shape({ tile[2] * tile[1] * coefficients, nk, nn }, 4, false),
                    node.name + " convolution packed input"));
                if (!error.empty())
                    return;

                if (winograd)
                {
                    const uint64_t concurrent = layout_.cpu_threads < nn * nk ? layout_.cpu_threads : 1;
                    scratch.push_back(
                        allocate(error, shape({ tile[1] * coefficients * tile[2], 1, concurrent }, 4, false),
                            node.name + " Winograd input tiles"));
                    if (!error.empty())
                        return;
                }
                if (winograd || k > tile[2])
                {
                    scratch.push_back(
                        allocate(error, shape({ tile[1] * coefficients * tile[0], 1, layout_.cpu_threads }, 4, false),
                            node.name + " convolution accumulator tiles"));
                    if (!error.empty())
                        return;
                }
            }
            if (!request_.external_operators && !layout_.gpu && node.type == "Gemm")
            {
                const uint64_t m = checked_at(error, out_shapes[0].dimensions, 1),
                               n = checked_at(error, out_shapes[0].dimensions, 0);
                if (!error.empty())
                    return;

                const uint64_t k = static_cast<uint64_t>(node.integer(error, 9))
                    ? static_cast<uint64_t>(node.integer(error, 9))
                    : checked_at(error, checked_at(error, input_shapes, 0).dimensions, node.integer(error, 2) ? 1 : 0);
                if (!error.empty())
                    return;

                const auto tile = layout_.cpu_gemm_tile_shape ? layout_.cpu_gemm_tile_shape(error, m, n, k)
                                                              : std::array<uint64_t, 3> { m, n, k };
                const uint64_t nn_m = (m + tile[0] - 1) / tile[0], nn_n = (n + tile[1] - 1) / tile[1];
                const uint64_t nn_k = (k + tile[2] - 1) / tile[2];
                const uint32_t scalar = layout_.bf16 ? 2 : 4;
                if (!node.integer(error, 4))
                {
                    scratch.push_back(allocate(error,
                        shape({ tile[0] * tile[2], nn_k, layout_.cpu_threads > nn_m ? nn_m : layout_.cpu_threads },
                            scalar, false),
                        node.name + " packed A"));
                    if (!error.empty())
                        return;
                }

                if (!node.integer(error, 5))
                {
                    scratch.push_back(allocate(
                        error, shape({ tile[1] * tile[2], nn_k, nn_n }, scalar, false), node.name + " packed B"));
                    if (!error.empty())
                        return;
                }

                if (layout_.bf16 || k > tile[2] || node.integer(error, 10) == 3)
                {
                    scratch.push_back(allocate(error, shape({ tile[0] * tile[1], 1, layout_.cpu_threads }, 4, false),
                        node.name + " accumulators"));
                    if (!error.empty())
                        return;
                }
            }
            if (!request_.external_operators && node.type == "SDPA")
            {
                const auto& q = input_shapes[0];
                const auto& k = input_shapes[1];
                const auto& v = input_shapes[2];
                const bool flash = layout_.gpu && layout_.bf16 && layout_.flash_attention && q.dimensions[0] % 8 == 0
                    && v.dimensions[0] % 8 == 0 && v.dimensions[0] <= layout_.flash_output_limit;
                if (!layout_.gpu && layout_.cpu_attention_tiled && q.dimensions[1] > 1)
                {
                    const uint64_t lane = std::max<uint32_t>(layout_.cpu_vector_width, 2), tm = lane;
                    const uint64_t h = q.dimensions[0], d = v.dimensions[0], rows = k.dimensions[1],
                                   groups = k.dimensions[2];
                    const uint64_t blocks = (q.dimensions[1] + tm - 1) / tm;
                    const uint64_t reuse_threshold = lane >= 8 ? 3 : 4;
                    const bool fp32_value = q.scalar_bytes == 2
                        && blocks * q.dimensions[2] / groups >= reuse_threshold + (d < 32 ? 2 : 0);
                    const uint64_t value_scalar = fp32_value ? 4 : q.scalar_bytes;
                    const uint64_t mask_scalar = input_shapes.size() > 3 ? input_shapes[3].scalar_bytes : 0;
                    const uint64_t fixed = tm * (h * q.scalar_bytes + d * 4);
                    const uint64_t per_token = h * k.scalar_bytes + d * value_scalar + tm * (4 + mask_scalar);
                    uint64_t tn
                        = layout_.cpu_l2_cache_bytes > fixed ? (layout_.cpu_l2_cache_bytes - fixed) / per_token : 0;
                    tn = std::max(lane, std::min(tn, rows) / lane * lane);
                    const uint64_t count = (rows + tn - 1) / tn;
                    tn = aligned_memory_bytes(error, (rows + count - 1) / count, lane);
                    if (!error.empty())
                        return;

                    const auto temp = [&](std::vector<uint64_t> dims, uint32_t scalar)
                    {
                        scratch.push_back(allocate(
                            error, shape(std::move(dims), scalar, false), node.name + " CPU attention workspace"));
                    };
                    temp({ h, aligned_memory_bytes(error, rows, lane), groups }, k.scalar_bytes);
                    if (!error.empty())
                        return;

                    if (q.scalar_bytes == 4)
                    {
                        temp({ d, aligned_memory_bytes(error, rows, lane), groups }, 4);
                        if (!error.empty())
                            return;
                    }

                    temp({ d * tn, (rows + tn - 1) / tn, groups }, static_cast<uint32_t>(value_scalar));
                    const uint64_t workspace = tm * (tn + d + 1) + (tm * h * q.scalar_bytes + 3) / 4;
                    temp({ workspace, 1, layout_.cpu_threads }, 4);
                    if (mask_scalar)
                        temp({ rows * tm, blocks,
                                 input_shapes[3].dimensions.size() > 2 ? input_shapes[3].dimensions[2] : 1 },
                            static_cast<uint32_t>(mask_scalar));
                }
                else if (!layout_.gpu && !flash)
                {
                    // The generic CPU fallback holds the full score tensor for all heads.
                    const uint64_t heads = q.dimensions[2];
                    scratch.push_back(allocate(error,
                        shape({ k.dimensions[1], q.dimensions[1], heads }, layout_.gpu ? q.scalar_bytes : 4, false),
                        node.name + " attention scores"));
                    if (!error.empty())
                        return;

                    scratch.push_back(allocate(
                        error, shape({ q.dimensions[1], heads }, q.scalar_bytes, false), node.name + " softmax max"));
                    if (!error.empty())
                        return;

                    scratch.push_back(allocate(
                        error, shape({ q.dimensions[1], heads }, q.scalar_bytes, false), node.name + " softmax sum"));
                    if (!error.empty())
                        return;
                }
            }
            for (const auto& value : scratch)
            {
                release(error, value);
                if (!error.empty())
                    return;
            }

            for (const auto& value : attention_scratch)
            {
                release(error, value);
                if (!error.empty())
                    return;
            }

            for (const auto& value : convolution_scratch)
            {
                release(error, value);
                if (!error.empty())
                    return;
            }

            for (const auto& value : gemm_scratch)
            {
                release(error, value);
                if (!error.empty())
                    return;
            }
        }
        for (size_t i = 0; i < node.outputs.size(); i++)
        {
            values_[node.outputs[i]] = checked_at(error, result, i);
            if (!error.empty())
                return;

            usage_.output_shapes[node.outputs[i]] = checked_at(error, result, i).shape;
            if (!error.empty())
                return;
        }
        usage_.host_node_peaks[node.name] = host_live_;
        for (const auto& name : node.inputs)
        {
            auto value = values_.find(name);
            if (value != values_.end())
            {
                release(error, value->second);
                if (!error.empty())
                    return;

                values_.erase(value);
            }
        }
        for (const auto& value : layout_copies)
        {
            release(error, value);
            if (!error.empty())
                return;
        }
    }
    const MemoryGraph& graph_;
    const GraphMemoryRequest& request_;
    const AllocationLayout& layout_;
    AllocationPool pool_;
    std::map<std::string, Value> values_;
    std::vector<Storage> storage_;
    std::set<size_t> visited_;
    GraphMemoryUsage usage_;
    uint64_t host_live_ = 0, host_peak_ = 0;
};
}
uint64_t aligned_memory_bytes(std::string& error, uint64_t value, uint64_t alignment)
{
    if (!alignment)
    {
        error = "zero allocation alignment";
        return {};
    }
    return multiply(error, add(error, value, alignment - 1) / alignment, alignment);
}
uint64_t MemoryShape::elements(std::string& error) const
{
    const auto max_extent = static_cast<uint64_t>(std::numeric_limits<int>::max());
    if (!batch || batch > max_extent || dimensions.empty() || dimensions.size() > 4
        || std::any_of(dimensions.begin(), dimensions.end(),
            [max_extent](uint64_t extent) { return !extent || extent > max_extent; }))
    {
        error = "invalid tensor memory shape";
        return {};
    }
    return multiply(error, product(error, dimensions), batch);
}
uint64_t MemoryShape::storage_bytes(std::string& error) const
{
    if (!packing || dimensions.empty() || dimensions.back() % packing)
    {
        error = "invalid tensor packing";
        return {};
    }
    const uint64_t count = elements(error);
    if (!error.empty())
        return {};

    const uint64_t sample = count / batch;
    uint64_t bytes = 0;
    if (dimensions.size() < 3)
        bytes = multiply(error, sample, scalar_bytes);
    else
    {
        const uint64_t channels = dimensions.back() / packing;
        const uint64_t plane_bytes = multiply(error, sample / channels, scalar_bytes);
        bytes = multiply(error, aligned_memory_bytes(error, plane_bytes, 16), channels);
    }
    // ncnn pads the step between samples for batch views used as Vulkan bindings.
    return batch == 1 ? bytes : multiply(error, aligned_memory_bytes(error, bytes, 4096), batch);
}
MemoryShape MemoryShape::packed(bool enable) const
{
    MemoryShape result = *this;
    result.packing = enable && !dimensions.empty() && dimensions.back() % 4 == 0 ? 4 : 1;
    return result;
}
int64_t MemoryNode::integer(std::string& error, int key, int64_t fallback) const
{
    auto value = parameters.find(key);
    return value == parameters.end() ? fallback : parse_i64(error, value->second);
}
std::shared_ptr<const MemoryGraph> parse_memory_graph(std::string& error, const std::string& text)
{
    auto graph = std::make_shared<MemoryGraph>();
    std::istringstream file(text);
    std::string line;
    std::getline(file, line);
    if (line != "7767517")
    {
        error = "unsupported model graph magic for resource planning";
        return {};
    }
    std::getline(file, line);
    while (std::getline(file, line))
    {
        std::istringstream row(line);
        MemoryNode node;
        int inputs = 0, outputs = 0;
        if (!(row >> node.type >> node.name >> inputs >> outputs))
            continue;
        if (inputs < 0 || outputs < 1)
        {
            error = "invalid resource graph node";
            return {};
        }
        node.inputs.resize(static_cast<size_t>(inputs));
        node.outputs.resize(static_cast<size_t>(outputs));
        for (auto& value : node.inputs)
            row >> value;
        for (auto& value : node.outputs)
            row >> value;
        std::string field;
        while (row >> field)
        {
            const auto equal = field.find('=');
            if (equal == std::string::npos)
            {
                error = "invalid resource graph parameter";
                return {};
            }
            const int id = parse_i32(error, field.substr(0, equal));
            if (!error.empty())
                return {};
            std::string value = field.substr(equal + 1);
            if (value.size() >= 2 && value.front() == '"' && value.back() == '"')
            {
                node.quoted_parameters.insert(id);
                value = value.substr(1, value.size() - 2);
            }
            node.parameters[id] = std::move(value);
        }
        for (const auto& value : node.outputs)
            if (!graph->producers.emplace(value, graph->nodes.size()).second)
            {
                error = "duplicate resource graph output: " + value;
                return {};
            }
        graph->nodes.push_back(std::move(node));
    }
    return graph;
}
std::shared_ptr<const MemoryGraph> read_memory_graph(std::string& error, const std::string& path)
{
    std::ifstream file(path);
    if (!file)
    {
        error = "cannot read resource graph: " + path;
        return {};
    }
    std::ostringstream contents;
    contents << file.rdbuf();
    return parse_memory_graph(error, contents.str());
}

void inspect_weight_storage(std::string& error, const std::string& path, MemoryGraph& graph, WeightSpans* spans)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        error = "cannot read model weight storage metadata: " + path;
        return;
    }
    const uint64_t total = fsutil::file_size(error, path);
    if (!error.empty())
        return;

    uint64_t offset = 0;
    for (const auto& node : graph.nodes)
    {
        uint64_t source_mapping = 0;
        const auto consume = [&](uint64_t elements, bool tagged) -> void
        {
            uint64_t payload = multiply(error, elements, 4);
            if (!error.empty())
                return;

            uint32_t tag = 0;
            if (tagged)
            {
                unsigned char bytes[4];
                if (offset > total || total - offset < 4)
                {
                    error = "truncated model weight storage header: " + node.name;
                    return;
                }
                file.seekg(static_cast<std::streamoff>(offset));
                file.read(reinterpret_cast<char*>(bytes), 4);
                if (!file)
                {
                    error = "cannot read model weight storage header: " + node.name;
                    return;
                }
                offset += 4;
                tag = static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8)
                    | (static_cast<uint32_t>(bytes[2]) << 16) | (static_cast<uint32_t>(bytes[3]) << 24);
                if (tag == 0x01306B47u || tag == 0x01348B83u)
                {
                    payload = aligned_memory_bytes(error, multiply(error, elements, 2), 4);
                    if (!error.empty())
                        return;

                    // ModelBin expands 16-bit storage to an FP32 source Mat.
                    // Its mapped 16-bit input can coexist with that Mat and the
                    // kernel's packed/upload buffers during the current layer.
                    source_mapping = add(error, source_mapping, payload);
                    if (!error.empty())
                        return;
                }
                else if (tag != 0 && tag != 0x0002C056u)
                {
                    error = "unsupported H3 weight storage tag at " + node.name;
                    return;
                }
            }
            if (offset > total || payload > total - offset)
            {
                error = "model tensor storage exceeds the weight file: " + node.name;
                return;
            }
            if (spans)
                (*spans)[node.name].push_back({ offset, elements, payload, tag });
            offset += payload;
        };
        const auto get
            = [&](int key, int64_t fallback = 0) { return static_cast<uint64_t>(node.integer(error, key, fallback)); };
        if (node.type == "Gemm")
        {
            if (get(4))
            {
                consume(multiply(error, get(7), get(9)), true);
                if (!error.empty())
                    return;
            }

            if (get(5))
            {
                consume(multiply(error, get(8), get(9)), true);
                if (!error.empty())
                    return;
            }

            if (get(6) && node.integer(error, 10, -1) != -1)
            {
                const auto broadcast = node.integer(error, 10);
                if (!error.empty())
                    return;

                consume(broadcast == 0                     ? 1
                        : broadcast == 1 || broadcast == 2 ? get(7)
                        : broadcast == 4                   ? get(8)
                                                           : multiply(error, get(7), get(8)),
                    true);
                if (!error.empty())
                    return;
            }
        }
        else if (node.type == "Convolution" || node.type == "Convolution3D" || node.type == "ConvolutionDepthWise"
            || node.type == "Deconvolution" || node.type == "DeconvolutionDepthWise")
        {
            consume(get(6), true);

            if (get(5))
            {
                consume(get(0), false);
            }
        }
        else if (node.type == "RMSNorm" && get(2, 1))
        {
            consume(get(0), false);
        }
        else if (node.type == "LayerNorm" && get(2, 1))
        {
            consume(get(0), false);

            consume(get(0), false);
        }
        else if (node.type == "GroupNorm" && get(3, 1))
        {
            consume(get(1), false);

            consume(get(1), false);
        }
        else if (node.type == "Snake")
        {
            consume(multiply(error, get(0), 2), false);
        }
        else if (node.type == "MemoryData")
        {
            uint64_t elements = get(0);

            for (int key : { 1, 2, 11 })
                if (get(key))
                {
                    elements = multiply(error, elements, get(key));
                    if (!error.empty())
                        return;
                }

            if (elements)
            {
                consume(elements, get(21, 1) == 0);
            }
        }
        graph.loading_source_mapping[node.name] = source_mapping;
    }
    if (offset != total)
    {
        error = "model graph/storage tensor inventory does not cover the weight file: " + path;
        return;
    }
}
uint64_t AllocationLayout::allocation_bytes(std::string& error, uint64_t size, bool staging) const
{
    if (memory_requirement)
        return memory_requirement(error, size, staging);
    return aligned_memory_bytes(error, size, memory_requirement_alignment);
}
AllocationPool::AllocationPool(std::string& error, uint64_t block_size, const AllocationLayout& layout)
    : layout_(layout)
    , block_size_(aligned_memory_bytes(error, block_size, layout.buffer_alignment))
{
}
size_t AllocationPool::allocate(std::string& error, uint64_t bytes)
{
    bytes = aligned_memory_bytes(error, bytes, layout_.buffer_alignment);
    if (!error.empty())
        return {};

    for (size_t i = 0; i < free_.size(); i++)
        for (auto range = free_[i].begin(); range != free_[i].end(); ++range)
            if (range->bytes >= bytes)
            {
                const Range chosen { range->offset, bytes };
                range->offset += bytes;
                range->bytes -= bytes;
                if (!range->bytes)
                    free_[i].erase(range);
                allocations_.push_back({ i, chosen, true });
                live_ = add(error, live_, bytes);
                if (!error.empty())
                    return {};

                peak_live_ = std::max(peak_live_, live_);
                return allocations_.size() - 1;
            }
    const uint64_t capacity = std::max(block_size_, bytes);
    allocated_ = add(error, allocated_, layout_.allocation_bytes(error, capacity));
    if (!error.empty())
        return {};

    free_.push_back({});
    if (capacity > bytes)
        free_.back().push_back({ bytes, capacity - bytes });
    allocations_.push_back({ free_.size() - 1, { 0, bytes }, true });
    live_ = add(error, live_, bytes);
    if (!error.empty())
        return {};

    peak_live_ = std::max(peak_live_, live_);
    return allocations_.size() - 1;
}
void AllocationPool::release(std::string& error, size_t id)
{
    auto& value = checked_at(error, allocations_, id);
    if (!error.empty())
        return;

    if (!value.active)
    {
        error = "memory pool allocation released twice";
        return;
    }
    value.active = false;
    live_ -= value.range.bytes;
    auto& ranges = checked_at(error, free_, value.block);
    if (!error.empty())
        return;

    ranges.push_back(value.range);
    std::sort(ranges.begin(), ranges.end(), [](const Range& a, const Range& b) { return a.offset < b.offset; });

    for (size_t i = 1; i < ranges.size();)
        if (ranges[i - 1].offset + ranges[i - 1].bytes == ranges[i].offset)
        {
            ranges[i - 1].bytes += ranges[i].bytes;
            ranges.erase(ranges.begin() + static_cast<std::ptrdiff_t>(i));
        }
        else
            i++;
}
uint64_t AllocationPool::allocated_bytes() const
{
    return allocated_;
}
uint64_t AllocationPool::live_bytes() const
{
    return live_;
}
uint64_t AllocationPool::peak_live_bytes() const
{
    return peak_live_;
}
GraphMemoryUsage plan_graph_memory(
    std::string& error, const MemoryGraph& graph, const GraphMemoryRequest& request, const AllocationLayout& layout)
{
    return GraphReplay(error, graph, request, layout).run(error);
}
std::vector<WeightAllocation> model_weight_allocations(
    std::string& error, const MemoryGraph& graph, const AllocationLayout& layout)
{
    std::vector<WeightAllocation> result;
    for (const auto& node : graph.nodes)
    {
        WeightAllocation value;
        value.node = node.name;
        auto source_mapping = graph.loading_source_mapping.find(node.name);
        if (source_mapping != graph.loading_source_mapping.end())
            value.source_mapping_scratch = source_mapping->second;
        const auto get = [&](int key, int64_t fallback = 0) { return node.integer(error, key, fallback); };
        const uint64_t scalar = layout.bf16 && !(get(31) & 4) ? 2 : 4;
        if (node.type == "Gemm")
        {
            const uint64_t m = static_cast<uint64_t>(get(7)), n = static_cast<uint64_t>(get(8)),
                           k = static_cast<uint64_t>(get(9));

            const auto packed = layout.gemm_weight_shape && scalar == 2 ? layout.gemm_weight_shape(error, m, n, k)
                                                         : std::array<uint64_t, 3> { m, n, k };
            uint64_t source = 0, converted = 0;
            if (get(4))
            {
                source = multiply(error, multiply(error, m, k), 4);
                if (!error.empty())
                    return {};

                converted = multiply(error, multiply(error, packed[0], packed[2]), 4);
                if (!error.empty())
                    return {};

                value.tensors.push_back(multiply(error, multiply(error, packed[0], packed[2]), scalar));
                if (!error.empty())
                    return {};
            }
            if (get(5))
            {
                source = std::max(source, multiply(error, multiply(error, n, k), 4));
                if (!error.empty())
                    return {};

                converted = std::max(converted, multiply(error, multiply(error, packed[1], packed[2]), 4));
                if (!error.empty())
                    return {};

                value.tensors.push_back(multiply(error, multiply(error, packed[1], packed[2]), scalar));
                if (!error.empty())
                    return {};
            }
            if (get(6) && get(10) != -1)
            {
                const uint64_t count = get(10) == 0 ? 1
                    : get(10) == 1 || get(10) == 2  ? m
                    : get(10) == 4                  ? n
                                                    : multiply(error, m, n);
                if (!error.empty())
                    return {};

                value.tensors.push_back(multiply(error, count, layout.gpu ? scalar : 4));
                if (!error.empty())
                    return {};
            }
            value.host_loading_scratch = layout.gpu ? add(error, source, converted) : source;
            if (!error.empty())
                return {};
        }
        else if (node.type == "RMSNorm" && get(2, 1))
        {
            value.tensors.push_back(multiply(error, static_cast<uint64_t>(get(0)), layout.gpu ? scalar : 4));
            if (!error.empty())
                return {};
        }
        else if (node.type == "LayerNorm" && get(2, 1))
        {
            value.tensors = { multiply(error, static_cast<uint64_t>(get(0)), layout.gpu ? scalar : 4),
                multiply(error, static_cast<uint64_t>(get(0)), layout.gpu ? scalar : 4) };
            if (!error.empty())
                return {};
        }
        else if (node.type == "GroupNorm" && get(3, 1))
        {
            value.tensors = { multiply(error, static_cast<uint64_t>(get(1)), layout.gpu ? scalar : 4),
                multiply(error, static_cast<uint64_t>(get(1)), layout.gpu ? scalar : 4) };
            if (!error.empty())
                return {};
        }
        else if (node.type == "Convolution")
        {
            const uint64_t count = static_cast<uint64_t>(get(6)), output = static_cast<uint64_t>(get(0));

            const uint64_t kernel = static_cast<uint64_t>(get(1) * get(11, get(1)));

            const uint64_t input = count / kernel / output;
            if (!layout.gpu && layout.cpu_convolution_tile_shape)
            {
                const bool winograd = kernel == 9 && get(3, 1) == 1 && get(13, 1) == 1 && (input > 8 || output > 8);

                const uint64_t coefficients = winograd ? (input <= 32 && output <= 32 ? 64 : 36) : 1;
                const uint64_t k = winograd ? input : input * kernel;
                const auto tile = layout.cpu_convolution_tile_shape(error, output, 0, k, winograd);
                const uint64_t packed = multiply(error,
                    multiply(error,
                        multiply(error, aligned_memory_bytes(error, output, tile[0]),
                            aligned_memory_bytes(error, k, tile[2])),
                        coefficients),
                    4);
                if (!error.empty())
                    return {};

                value.tensors.push_back(packed);
                value.host_loading_scratch = multiply(error, count, 4);
                if (!error.empty())
                    return {};

                if (winograd)
                {
                    value.host_loading_scratch = add(error, value.host_loading_scratch,
                        multiply(error,
                            multiply(error, multiply(error, multiply(error, tile[0], tile[2]), coefficients),
                                layout.cpu_threads),
                            4));
                    if (!error.empty())
                        return {};
                }
                else if (kernel != 1)
                {
                    value.host_loading_scratch = add(error, value.host_loading_scratch, multiply(error, count, 4));
                    if (!error.empty())
                        return {};
                }

                if (get(5))
                {
                    value.tensors.push_back(multiply(error, output, 4));
                    if (!error.empty())
                        return {};
                }

                result.push_back(std::move(value));
                continue;
            }
            const bool winograd = layout.gpu && kernel == 9 && get(3, 1) == 1 && get(13, 1) == 1 && input >= 16
                && output >= 16 && (layout.winograd23 || layout.winograd43);

            uint64_t packed = 0;
            if (winograd)
            {
                if (layout.winograd43)
                {
                    value.tensors.push_back(
                        multiply(error, multiply(error, multiply(error, input, output), 36), scalar));
                    if (!error.empty())
                        return {};
                }

                if (layout.winograd23)
                {
                    value.tensors.push_back(
                        multiply(error, multiply(error, multiply(error, input, output), 16), scalar));
                    if (!error.empty())
                        return {};
                }

                for (uint64_t bytes : value.tensors)
                {
                    packed = add(error, packed, bytes);
                    if (!error.empty())
                        return {};
                }
            }
            else
            {
                value.tensors.push_back(multiply(error, count, scalar));
                if (!error.empty())
                    return {};

                packed = multiply(error, count, 4);
                if (!error.empty())
                    return {};
            }
            if (winograd)
            {
                // Each transform temporarily keeps source, transformed FP32 and
                // packed FP32. The earlier packed transform survives the next one.
                const uint64_t unit = multiply(error, multiply(error, input, output), 4);
                if (!error.empty())
                    return {};

                packed = layout.winograd43 ? multiply(error, unit, 72) : multiply(error, unit, 32);
                if (!error.empty())
                    return {};

                if (layout.winograd43 && layout.winograd23)
                {
                    packed = std::max(packed, multiply(error, unit, 36 + 32));
                    if (!error.empty())
                        return {};
                }
            }
            value.host_loading_scratch = add(error, multiply(error, count, 4), packed);
            if (!error.empty())
                return {};

            if (get(5))
            {
                value.tensors.push_back(multiply(error, output, scalar));
                if (!error.empty())
                    return {};
            }
        }
        else if (node.type == "Convolution3D" || node.type == "ConvolutionDepthWise"
            || node.type == "Deconvolution" || node.type == "DeconvolutionDepthWise")
        {
            const uint64_t count = get(6), output = get(0);
            uint64_t packed = count;
            if (node.type == "Convolution3D")
                packed = count / output * aligned_memory_bytes(error, output, 4);
            value.tensors.push_back(multiply(error, packed, scalar));
            value.host_loading_scratch = 4 * (count + packed);
            if (get(5))
                value.tensors.push_back(multiply(error, output, scalar));
        }
        else if (node.type == "Snake")
        {
            value.tensors.push_back(multiply(error, get(0), 2 * sizeof(float)));
            if (!error.empty())
                return {};
        }
        else if (node.type == "MemoryData")
        {
            uint64_t count = static_cast<uint64_t>(get(0));

            for (int key : { 1, 2, 11 })
                if (get(key))
                {
                    count = multiply(error, count, static_cast<uint64_t>(get(key)));
                    if (!error.empty())
                        return {};
                }

            value.tensors.push_back(multiply(error, count, scalar));
            if (!error.empty())
                return {};
        }
        if (!value.tensors.empty())
        {
            result.push_back(std::move(value));
        }
    }
    return result;
}
namespace
{
class ResidentTier
{
public:
    ResidentTier(uint64_t limit, const AllocationLayout& layout)
        : limit_(limit)
        , layout_(layout)
    {
    }
    bool allocate(std::string& error, uint64_t bytes)
    {
        bytes = aligned_memory_bytes(error, bytes, layout_.buffer_alignment);
        if (!error.empty())
            return {};

        if (bytes <= free_)
        {
            free_ -= bytes;
            return true;
        }
        if (allocated_ >= limit_ || bytes > limit_ - allocated_)
            return false;
        const uint64_t remaining = limit_ - allocated_;
        uint64_t capacity = std::min(std::max(bytes, 8 * MiB), remaining);
        uint64_t physical = layout_.allocation_bytes(error, capacity);
        if (!error.empty())
            return {};

        if (physical > remaining)
        {
            const uint64_t excess = physical - remaining;
            if (excess >= capacity)
                return false;
            capacity -= excess;
            capacity = capacity / layout_.buffer_alignment * layout_.buffer_alignment;
            if (capacity < bytes)
                return false;
            physical = layout_.allocation_bytes(error, capacity);
            if (!error.empty())
                return {};
        }
        if (physical > remaining)
            return false;
        allocated_ = add(error, allocated_, physical);
        if (!error.empty())
            return {};

        free_ = capacity - bytes;
        return true;
    }
    uint64_t allocated() const
    {
        return allocated_;
    }

private:
    uint64_t limit_, allocated_ = 0, free_ = 0;
    const AllocationLayout& layout_;
};
struct StreamRange
{
    size_t block;
    uint64_t offset, bytes;
};
struct StreamBlock
{
    uint64_t capacity;
    std::vector<std::pair<uint64_t, uint64_t>> free;
};
struct StreamState
{
    std::vector<StreamBlock> blocks;
    std::vector<StreamRange> current;
    void release(std::string& error, const StreamRange& value)
    {
        auto& ranges = checked_at(error, blocks, value.block).free;
        if (!error.empty())
            return;

        ranges.emplace_back(value.offset, value.bytes);
        std::sort(ranges.begin(), ranges.end());

        for (size_t i = 1; i < ranges.size();)
            if (ranges[i - 1].first + ranges[i - 1].second == ranges[i].first)
            {
                ranges[i - 1].second += ranges[i].second;
                ranges.erase(ranges.begin() + static_cast<std::ptrdiff_t>(i));
            }
            else
                i++;
    }
    StreamRange allocate(std::string& error, uint64_t bytes, const AllocationLayout& layout)
    {
        bytes = aligned_memory_bytes(error, bytes, std::max(layout.buffer_alignment, layout.memory_map_alignment));
        if (!error.empty())
            return {};

        for (size_t block = 0; block < blocks.size(); block++)
            for (auto range = blocks[block].free.begin(); range != blocks[block].free.end(); ++range)
                if (range->second >= bytes)
                {
                    const StreamRange result { block, range->first, bytes };
                    range->first += bytes;
                    range->second -= bytes;
                    if (!range->second)
                        blocks[block].free.erase(range);
                    return result;
                }
        const uint64_t capacity = std::max<uint64_t>(16 * MiB, bytes);
        blocks.push_back({ capacity, {} });
        if (capacity > bytes)
            blocks.back().free.emplace_back(bytes, capacity - bytes);
        return { blocks.size() - 1, 0, bytes };
    }
    std::string key() const
    {
        std::string result;
        const auto append
            = [&](uint64_t value) { result.append(reinterpret_cast<const char*>(&value), sizeof(value)); };
        append(blocks.size());

        for (const auto& block : blocks)
        {
            append(block.capacity);

            append(block.free.size());

            for (const auto& range : block.free)
            {
                append(range.first);

                append(range.second);
            }
        }
        append(current.size());

        for (const auto& range : current)
        {
            append(range.block);

            append(range.offset);

            append(range.bytes);
        }
        return result;
    }
    uint64_t allocated(std::string& error, const AllocationLayout& layout) const
    {
        uint64_t result = 0;
        for (const auto& block : blocks)
        {
            result = add(error, result, layout.allocation_bytes(error, block.capacity));
            if (!error.empty())
                return {};
        }

        return result;
    }
};
}
WeightMemoryUsage plan_weight_memory(std::string& error, const std::vector<WeightAllocation>& weights,
    const std::vector<GraphMemoryUsage>& calls, const AllocationLayout& layout, uint64_t device_limit,
    uint64_t host_limit, bool fully_resident)
{
    WeightMemoryUsage result;
    ResidentTier device(device_limit, layout), host(host_limit, layout);
    AllocationPool resident(error, 8 * MiB, layout);
    std::map<std::string, const WeightAllocation*> by_name;
    uint64_t upload_payload = 0, upload_memory = 0;
    for (const auto& group : weights)
    {
        by_name[group.node] = &group;
        auto& placements = result.placements[group.node];
        result.loading_host_scratch = std::max(
            result.loading_host_scratch, add(error, group.host_loading_scratch, group.source_mapping_scratch));
        if (!error.empty())
            return {};

        result.loading_host_peak = std::max(result.loading_host_peak,
            add(error, host.allocated(),
                add(error, group.host_loading_scratch, add(error, group.source_mapping_scratch, upload_memory))));
        if (!error.empty())
            return {};

        uint64_t group_cpu_cast = 0;
        for (uint64_t bytes : group.tensors)
        {
            result.packed_payload = add(error, result.packed_payload, bytes);
            if (!error.empty())
                return {};

            int location = 3;
            if (fully_resident)
            {
                resident.allocate(error, bytes);
                if (!error.empty())
                    return {};

                location = layout.gpu ? 0 : 1;
                if (layout.gpu)
                {
                    upload_payload = add(error, upload_payload, bytes);
                    if (!error.empty())
                        return {};

                    upload_memory = add(error, upload_memory, layout.allocation_bytes(error, bytes, true));
                    if (!error.empty())
                        return {};

                    result.loading_staging_peak = std::max(result.loading_staging_peak, upload_memory);
                }
            }
            else if (layout.gpu && device.allocate(error, bytes))
            {
                location = 0;
                upload_payload = add(error, upload_payload, bytes);
                if (!error.empty())
                    return {};

                upload_memory = add(error, upload_memory, layout.allocation_bytes(error, bytes, true));
                if (!error.empty())
                    return {};

                result.loading_staging_peak = std::max(result.loading_staging_peak, upload_memory);
            }
            else if (host.allocate(error, bytes))
                location = 1;
            else
                result.disk_payload = add(error, result.disk_payload, bytes);
            placements.push_back(location);
            group_cpu_cast = std::max(group_cpu_cast, bytes);
        }
        // Packing is FP32, followed by a storage cast and the outstanding upload
        // batch. Large native matrices dominate the tiny affine/bias tensors.
        const uint64_t packed_fp32 = layout.bf16 ? multiply(error, group_cpu_cast, 2) : group_cpu_cast;
        if (!error.empty())
            return {};

        result.loading_host_peak = std::max(result.loading_host_peak,
            add(error, host.allocated(),
                add(error, group.source_mapping_scratch,
                    add(error, packed_fp32, add(error, layout.bf16 ? group_cpu_cast : 0, upload_memory)))));
        if (!error.empty())
            return {};

        // This is ncnn's actual VkTransfer submission watermark, not a budget reserve.
        if (upload_payload > 256 * MiB)
            upload_payload = upload_memory = 0;
    }
    result.device_resident = device.allocated();

    result.host_resident = host.allocated();

    if (fully_resident)
    {
        if (layout.gpu)
            result.device_resident = resident.allocated_bytes();
        else
        {
            // CPU packed Mat tensors use individual allocations, not VkWeightAllocator blocks.
            uint64_t prefix = 0;
            result.loading_host_peak = 0;
            for (const auto& group : weights)
            {
                for (uint64_t bytes : group.tensors)
                {
                    prefix = add(error, prefix, add(error, aligned_memory_bytes(error, add(error, bytes, 4), 64), 64));
                    if (!error.empty())
                        return {};
                }

                result.loading_host_peak = std::max(result.loading_host_peak,
                    add(error, prefix, add(error, group.host_loading_scratch, group.source_mapping_scratch)));
                if (!error.empty())
                    return {};
            }
            result.host_resident = prefix;
            result.loading_host_peak = std::max(result.loading_host_peak, prefix);
        }
        return result;
    }
    for (const auto& call : calls)
    {
        std::map<std::string, StreamState> states;
        states.emplace("", StreamState {});
        for (const auto& name : call.execution_order)
        {
            auto found = by_name.find(name);
            if (found == by_name.end())
                continue;
            const auto& group = *found->second;
            const auto& placements = checked_at(error, result.placements, name);
            if (!error.empty())
                return {};

            std::vector<size_t> selected;
            auto specific = call.weight_tensor_selection.find(name);
            if (specific != call.weight_tensor_selection.end())
                selected = specific->second;
            else
                for (size_t i = 0; i < group.tensors.size(); i++)
                    selected.push_back(i);
            std::vector<uint64_t> current, early;
            const auto early_found = call.weight_early_tensors.find(name);
            uint64_t staging = 0, source = 0;
            for (size_t i : selected)
            {
                if (checked_at(error, placements, i) == 0)
                    continue;
                const uint64_t bytes = checked_at(error, group.tensors, i);
                if (!error.empty())
                    return {};

                // A disk-backed tensor's source pages coexist with its upload
                // staging during resolve_weights(). Resident host tensors are
                // already included in the host weight cache.
                if (checked_at(error, placements, i) == 3)
                {
                    source = add(error, source, bytes);
                    if (!error.empty())
                        return {};
                }

                if (early_found != call.weight_early_tensors.end()
                    && std::find(early_found->second.begin(), early_found->second.end(), i)
                        != early_found->second.end())
                    early.push_back(bytes);
                else
                    current.push_back(bytes);
                staging = add(error, staging, layout.allocation_bytes(error, bytes, true));
                if (!error.empty())
                    return {};
            }
            if (current.empty())
                current.swap(early);
            if (current.empty())
                continue;
            // An extractor may submit/reset before a layer at the native dispatch
            // watermark. Enumerating both legal fence positions gives a byte bound
            // valid for either kernel dispatch path, including its first-fit fragmentation.
            std::map<std::string, StreamState> next;
            for (const auto& state : states)
                for (bool fence_before : { false, true })
                {
                    StreamState prepared = state.second;
                    const auto preceding = prepared.current;
                    if (fence_before)
                        for (const auto& range : preceding)
                        {
                            prepared.release(error, range);
                        }

                    prepared.current.clear();

                    std::vector<StreamRange> early_buffers;
                    for (uint64_t bytes : early)
                    {
                        early_buffers.push_back(prepared.allocate(error, bytes, layout));
                        if (!error.empty())
                            return {};
                    }

                    for (bool fence_between : { false, true })
                    {
                        StreamState candidate = prepared;
                        if (fence_between)
                        {
                            for (const auto& range : early_buffers)
                            {
                                candidate.release(error, range);
                            }

                            if (!fence_before)
                                for (const auto& range : preceding)
                                {
                                    candidate.release(error, range);
                                }
                        }
                        for (uint64_t bytes : current)
                        {
                            candidate.current.push_back(candidate.allocate(error, bytes, layout));
                            if (!error.empty())
                                return {};
                        }

                        if (!fence_between)
                        {
                            for (const auto& range : early_buffers)
                            {
                                candidate.release(error, range);
                            }

                            if (!fence_before)
                                for (const auto& range : preceding)
                                {
                                    candidate.release(error, range);
                                }
                        }
                        result.streaming_device_pool
                            = std::max(result.streaming_device_pool, candidate.allocated(error, layout));
                        if (!error.empty())
                            return {};

                        const auto key = candidate.key();

                        next.emplace(key, std::move(candidate));
                    }
                }
            states = std::move(next);
            result.streaming_host_staging = std::max(result.streaming_host_staging, staging);
            result.streaming_host_source = std::max(result.streaming_host_source, source);
        }
    }
    return result;
}
}
