// SPDX-License-Identifier: Apache-2.0
#include "lora.h"
#include "filesystem_utils.h"
#include "h3_config.h"
#include "safetensors.h"
#include "shape_memory.h"

#include <datareader.h>
#include <layer.h>
#include <modelbin.h>
#include <net.h>
#include <paramdict.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <string_view>

namespace h3
{
namespace
{
constexpr int kMergeRows = 256;
constexpr uint32_t kBf16Tag = 0x01348b83;
constexpr uint32_t kF16Tag = 0x01306b47;

float round_bf16(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16) & 1u);
    return ncnn::bfloat16_to_float32(static_cast<uint16_t>(bits >> 16));
}

bool remove_prefix(std::string_view& key, std::string_view prefix)
{
    if (key.substr(0, prefix.size()) != prefix)
        return false;
    key.remove_prefix(prefix.size());
    return true;
}

int block_index(std::string_view& key, int blocks)
{
    const size_t dot = key.find('.');
    if (dot == std::string_view::npos)
        return -1;
    int block;
    const auto result = std::from_chars(key.data(), key.data() + dot, block);
    if (result.ec != std::errc() || result.ptr != key.data() + dot || block < 0 || block >= blocks)
        return -1;
    key.remove_prefix(dot + 1);
    return block;
}

std::string gemm_name(int segment, int slot = 0)
{
    return "s" + std::to_string(segment) + "_gemm_" + std::to_string(slot);
}

std::string modulation_weight(int block)
{
    return block == kDiTBlocks ? "norm_out.linear.weight"
                               : "transformer_blocks." + std::to_string(block) + ".adaln_proj.linear.weight";
}

std::string modulation_bias(int block)
{
    return block == kDiTBlocks ? "norm_out.linear.bias"
                               : "transformer_blocks." + std::to_string(block) + ".adaln_proj.folded_bias";
}

int modulation_width(int block)
{
    return kDiTWidth * (block == kDiTBlocks ? 2 : 18);
}

struct Target
{
    bool modulation = false;
    int segment = -1;
    int slot = 0;
    int projections = 1;
    bool swap_gate = false;
};

Target target_for(std::string_view key)
{
    remove_prefix(key, "diffusion_model.");
    if (key == "audio_patch_proj")
        return { false, 1 };
    if (key == "final_layer.audio_out")
        return { false, 58 };
    if (key == "final_layer.adaln_proj.linear")
        return { true, 51 };

    int block = -1;
    int segment = -1;
    const bool diffusers = remove_prefix(key, "transformer_blocks.");
    if (diffusers || remove_prefix(key, "blocks."))
    {
        block = block_index(key, kDiTBlocks);
        if (block == -1)
            return {};
        segment = block + 6;
    }
    else if (remove_prefix(key, "token_refiner.blocks."))
    {
        const int refiner = block_index(key, 2);
        if (refiner == -1)
            return {};
        segment = refiner + 3;
    }
    else
        return {};

    // ComfyUI stores one contiguous Q/K/V matrix, and puts the SwiGLU gate
    // first. The ncnn graph uses separate Q/K/V projections and value first.
    if (key == "attn.qkv_proj")
        return { false, segment, 0, 3 };
    if (key == "attn.to_q")
        return { false, segment, 0 };
    if (key == "attn.to_k")
        return { false, segment, 1 };
    if (key == "attn.to_v")
        return { false, segment, 2 };
    if (key == "attn.out_proj" || key == "attn.to_out.0")
        return { false, segment, 3 };
    if (key == "mlp.fc1" || key == "ff.net.0.proj")
        return { false, segment, 4, 1, !diffusers };
    if (key == "mlp.fc2" || key == "ff.net.2")
        return { false, segment, 5 };
    if (block != -1 && key == "adaln_proj.linear")
        return { true, block + 1 };
    return {};
}

struct Patch
{
    WeightSpan span;
    const SafeTensor* down;
    const SafeTensor* up = nullptr;
    int rows = 0;
    int cols = 0;
    int source_row = 0;
    bool swap_gate = false;
    float scale = 1.f;
    std::string layer {};
    const SafeTensors* file = nullptr;
    bool cast_bf16 = false;
};

using Patches = std::map<uint64_t, std::vector<Patch>>;
using LivePatches = std::map<uint64_t, Patch>;

class WeightReader : public ncnn::DataReader
{
public:
    WeightReader(const std::string& path, const Patches& patches, const LivePatches& live)
        : base_(path, std::ios::binary)
        , patches_(patches)
        , patch_(patches_.begin())
        , live_(live)
        , next_live_(live_.begin())
    {
    }

    size_t read(void* data, size_t bytes) const override
    {
        // Insert the two tagged LoRA matrices immediately after the base
        // layer's weights. Position tracks the original model stream.
        if (next_live_ != live_.end() && position_ == next_live_->first)
            return read_live(data, bytes);
        if (!base_.read(static_cast<char*>(data), bytes))
            return 0;
        if (patch_ != patches_.end() && patch_->first < position_ + bytes)
        {
            if (patch_->first != position_ || patch_->second.front().span.bytes != bytes)
                return 0;
            if (!merge(patch_->second, data))
                return 0;
            ++patch_;
        }
        position_ += bytes;
        return bytes;
    }

    bool complete() const
    {
        return patch_ == patches_.end() && next_live_ == live_.end();
    }

private:
    size_t read_live(void* data, size_t bytes) const
    {
        const auto& patch = next_live_->second;
        if ((live_part_ & 1) == 0)
        {
            if (bytes != sizeof(kBf16Tag))
                return 0;
            std::memcpy(data, &kBf16Tag, sizeof(kBf16Tag));
        }
        else
        {
            const size_t rank = patch.down->shape[0];
            const bool down = live_part_ == 1;
            const size_t count = rank * (down ? patch.cols : patch.rows);
            const size_t padded = (count + 1) & ~size_t(1);
            if (bytes != padded * sizeof(uint16_t))
                return 0;
            auto* dst = static_cast<uint16_t*>(data);
            bool ok;
            if (down)
                ok = patch.file->read_bf16(*patch.down, 0, count, dst);
            else if (patch.swap_gate)
            {
                const size_t half = count / 2;
                const size_t start = size_t(patch.source_row) * rank;
                ok = patch.file->read_bf16(*patch.up, start + half, half, dst)
                    && patch.file->read_bf16(*patch.up, start, half, dst + half);
            }
            else
                ok = patch.file->read_bf16(*patch.up, size_t(patch.source_row) * rank, count, dst);
            if (!ok)
                return 0;
            if (padded != count)
                dst[count] = 0;
        }
        if (++live_part_ == 4)
        {
            live_part_ = 0;
            ++next_live_;
        }
        return bytes;
    }

    static bool read_adapter(const Patch& patch, const SafeTensor& tensor, size_t offset, size_t count, float* values)
    {
        if (!patch.file->read(tensor, offset, count, values))
            return false;
        if (patch.cast_bf16)
            for (size_t i = 0; i < count; i++)
                values[i] = round_bf16(values[i]);
        return true;
    }

    static bool read_up(const Patch& patch, int row, int count, float* values)
    {
        const int rank = patch.down->shape[0];
        const int source = patch.source_row + (patch.swap_gate ? (row + patch.rows / 2) % patch.rows : row);
        return read_adapter(patch, *patch.up, size_t(source) * rank, size_t(count) * rank, values);
    }

    bool merge(const std::vector<Patch>& patches, void* data) const
    {
        const auto& shape = patches.front();
        const Patch* replacement = nullptr;
        std::vector<const Patch*> updates;
        int rank = 0;
        bool swap_gate = false;
        for (const auto& patch : patches)
        {
            if (patch.up)
            {
                updates.push_back(&patch);
                rank += patch.down->shape[0];
                swap_gate |= patch.swap_gate;
            }
            else
                replacement = &patch;
        }

        std::vector<float> base_values;
        if (replacement)
        {
            base_values.resize(shape.span.elements);
            if (!read_adapter(*replacement, *replacement->down, 0, base_values.size(), base_values.data()))
                return false;
        }

        const bool bf16 = shape.span.tag == kBf16Tag;
        const bool half = shape.span.tag == kF16Tag;
        const float scale = updates.size() == 1 ? updates.front()->scale : 1.f;
        const auto update = [&](size_t offset, size_t count, const float* delta)
        {
            auto* wide = static_cast<float*>(data);
            auto* narrow = static_cast<uint16_t*>(data);
            for (size_t i = 0; i < count; i++)
            {
                const size_t index = offset + i;
                const float base = replacement ? base_values[index]
                    : bf16                     ? ncnn::bfloat16_to_float32(narrow[index])
                    : half                     ? ncnn::float16_to_float32(narrow[index])
                                               : wide[index];
                float value = base;
                if (delta)
                    value = scale == 1.f ? base + delta[i] : std::fma(scale, delta[i], base);
                if (!std::isfinite(value))
                    return false;
                if (bf16)
                    narrow[index] = ncnn::float32_to_bfloat16(value);
                else if (half)
                    narrow[index] = ncnn::float32_to_float16(value);
                else
                    wide[index] = value;
            }
            return true;
        };
        if (updates.empty())
            return update(0, shape.span.elements, nullptr);

        ncnn::Mat down(shape.cols, rank);
        if (down.empty())
            return false;
        int down_row = 0;
        for (const auto* patch : updates)
        {
            const int count = patch->down->shape[0];
            if (!read_adapter(*patch, *patch->down, 0, size_t(count) * shape.cols, down.row(down_row)))
                return false;
            down_row += count;
        }

        std::unique_ptr<ncnn::Layer> gemm(ncnn::create_layer_cpu("Gemm"));
        ncnn::ParamDict pd;
        pd.set(5, 1); // Constant B is the down projection.
        pd.set(8, shape.cols);
        pd.set(9, rank);
        if (!gemm || gemm->load_param(pd) || gemm->load_model(ncnn::ModelBinFromMatArray(&down)))
            return false;

        ncnn::Option opt;
        opt.num_threads = std::min(opt.num_threads, 8);
        opt.use_packing_layout = false;
        opt.use_fp16_packed = false;
        opt.use_fp16_storage = false;
        opt.use_fp16_arithmetic = false;
        if (gemm->create_pipeline(opt))
            return false;

        // Concatenate the ranks so all updates go through one FP32 GEMM and a single conversion to the stored type.
        std::vector<float> part(updates.size() > 1 ? size_t(rank) * kMergeRows : 0);
        bool valid = true;
        for (int row = 0; row < shape.rows;)
        {
            const int half_rows = shape.rows / 2;
            const int count = std::min(kMergeRows, (swap_gate && row < half_rows ? half_rows : shape.rows) - row);
            ncnn::Mat up(rank, count);
            if (up.empty())
            {
                valid = false;
                break;
            }
            if (updates.size() == 1)
                valid = read_up(*updates.front(), row, count, up);
            else
            {
                int col = 0;
                for (const auto* patch : updates)
                {
                    const int width = patch->down->shape[0];
                    if (!read_up(*patch, row, count, part.data()))
                    {
                        valid = false;
                        break;
                    }
                    for (int y = 0; y < count; y++)
                        for (int x = 0; x < width; x++)
                            up.row(y)[col + x] = part[size_t(y) * width + x] * patch->scale;
                    col += width;
                }
            }
            std::vector<ncnn::Mat> result(1);
            if (!valid || gemm->forward({ up }, result, opt) || result[0].w != shape.cols || result[0].h != count
                || !update(size_t(row) * shape.cols, size_t(count) * shape.cols, result[0]))
            {
                valid = false;
                break;
            }
            row += count;
        }
        gemm->destroy_pipeline(opt);
        return valid;
    }

    mutable std::ifstream base_;
    const Patches& patches_;
    mutable Patches::const_iterator patch_;
    const LivePatches& live_;
    mutable LivePatches::const_iterator next_live_;
    mutable int live_part_ = 0;
    mutable uint64_t position_ = 0;
};

struct Adapter
{
    Adapter(std::string& error, const std::string& path, float strength)
        : file(error, path)
        , name(std::filesystem::path(path).filename().string())
        , parent(std::filesystem::path(path).parent_path().filename().string())
        , strength(strength)
    {
        if (!error.empty())
            return;
        pi = file.tensors.count("norm_out.linear.weight") != 0;
        for (const auto& entry : file.tensors)
            if (entry.first.find("dwd_") == 0)
            {
                error = "pi-H3 DWD requires a pixel refiner model";
                return;
            }
        pika = name.find("pika_h3_flex_turbo_") == 0;
        turbo = file.tensors.count("blocks.0.adaln_proj.linear.lora_A.weight") != 0
            || file.tensors.count("diffusion_model.blocks.0.adaln_proj.linear.lora_A.weight") != 0;
        if (pi)
        {
            const auto label = parent + "/" + name;
            pi_nfe = label.find("2step") != std::string::npos ? 2 : label.find("4step") != std::string::npos ? 4 : -1;
        }
    }

    SafeTensors file;
    std::string name, parent;
    float strength;
    bool pi = false;
    bool pika = false;
    bool turbo = false;
    int pi_nfe = 0;
};
}

struct Lora::Impl
{
    Impl(std::string& error, const std::vector<LoraInput>& inputs)
    {
        adapters.reserve(inputs.size());
        for (const auto& input : inputs)
        {
            const auto& adapter = adapters.emplace_back(error, input.path, input.strength);
            if (!error.empty())
                return;
            if (adapter.pi)
            {
                if (pi)
                {
                    error = "multiple pi-H3 adapters replace the same model weights";
                    return;
                }
                pi = &adapter;
                preset = &adapter;
            }
            else if (!preset && (adapter.pika || adapter.turbo))
                preset = &adapter;
        }
    }

    std::vector<Adapter> adapters;
    const Adapter* pi = nullptr;
    const Adapter* preset = nullptr;
    std::unique_ptr<SafeTensors> pi_base;
    std::string bank_key;
    std::map<Component, Patches> patches;
    std::map<Component, LivePatches> live;
    std::map<Component, uint64_t> workspace;
};

Lora::Lora(std::string& error, const std::vector<LoraInput>& inputs)
    : impl_(std::make_unique<Impl>(error, inputs))
{
}

Lora::~Lora() = default;

int Lora::default_steps() const
{
    const auto* adapter = impl_->preset;
    if (!adapter)
        return 0;
    if (adapter->pi)
        return adapter->pi_nfe > 0 ? adapter->pi_nfe + 1 : 0;
    if (adapter->pika)
        return adapter->name.find("_2step_") != std::string::npos ? 3
            : adapter->name.find("_8step_") != std::string::npos  ? 9
                                                                 : 5;
    return adapter->name.find("v4") != std::string::npos ? 7 : 5;
}

bool Lora::simple_schedule() const
{
    return impl_->preset && impl_->preset->pika;
}
int Lora::pi_steps() const
{
    return impl_->pi ? impl_->pi->pi_nfe : 0;
}
float Lora::video_shift(float base, bool reference) const
{
    return simple_schedule() && !reference ? 6.f : base;
}

const std::string& Lora::modulation_key() const
{
    return impl_->bank_key;
}

uint64_t Lora::loading_workspace(Component component) const
{
    const auto it = impl_->workspace.find(component);
    return it == impl_->workspace.end() ? 0 : it->second;
}

bool Lora::configure(std::string& error, const std::string& model, bool reference)
{
    const auto fail = [&](std::string message) {
        error = std::move(message);
        return false;
    };
    auto& lora = *impl_;
    if (lora.pi && (reference || lora.pi->strength != 1.f))
        return fail("pi-H3 adapters require the FL2VA fixed build and strength 1");

    const Component core = reference ? Component::ReferenceTransformerCore : Component::TransformerCore;
    const Component producer = reference ? Component::ReferenceModulations : Component::Modulations;
    std::map<Component, WeightSpans> layouts;
    std::map<Component, std::shared_ptr<const MemoryGraph>> graphs;
    const auto node = [&](Component component, const std::string& layer) -> const MemoryNode*
    {
        auto graph = graphs.find(component);
        if (graph == graphs.end())
        {
            const auto path = model + "/" + model_path(component);
            const auto param = read_memory_graph(error, path + ".ncnn.param");
            if (!param)
                return nullptr;
            MemoryGraph inventory = *param;
            inspect_weight_storage(error, path + ".ncnn.bin", inventory, &layouts[component]);
            if (!error.empty())
                return nullptr;
            graph = graphs.emplace(component, param).first;
        }
        const auto& nodes = graph->second->nodes;
        const auto it = std::find_if(nodes.begin(), nodes.end(), [&](const auto& value) { return value.name == layer; });
        if (it != nodes.end())
            return &*it;
        error = "LoRA target layer missing from ncnn graph: " + layer;
        return nullptr;
    };
    for (const auto& adapter : lora.adapters)
    {
        const auto insert = [&](Component component, const std::string& layer, size_t index, Patch patch)
        {
            const auto& layout = layouts.find(component)->second;
            const auto target = layout.find(layer);
            if (target == layout.end() || index >= target->second.size())
                return fail("missing LoRA weight target: " + layer);
            const auto& spans = target->second;
            patch.span = spans[index];
            const uint64_t elements = uint64_t(patch.rows) * patch.cols;
            if (patch.span.elements != elements)
                return fail("incompatible LoRA weight target: " + layer);
            if (adapter.strength == 0.f)
                return true;
            patch.file = &adapter.file;
            patch.cast_bf16 = adapter.pi;
            if (adapter.pika && patch.up)
            {
                patch.layer = layer;
                const auto& last = spans.back();
                return lora.live[component].emplace(last.offset + last.bytes, std::move(patch)).second
                    || fail("duplicate LoRA weight target: " + layer);
            }
            auto& group = lora.patches[component][patch.span.offset];
            for (const auto& previous : group)
                if (previous.file == patch.file || (!previous.up && !patch.up))
                    return fail("duplicate LoRA weight target: " + layer);
            group.push_back(patch);
            return true;
        };

        std::set<std::string> consumed;
        if (adapter.pi)
        {
            lora.pi_base = std::make_unique<SafeTensors>(error, model + "/transformer/pi-h3.safetensors");
            if (!error.empty() || !lora.pi_base->tensor(error, "time_embedder.table", { 1025, 8 }))
                return false;
            for (int block = 0; block <= kDiTBlocks; block++)
            {
                const auto& bias = block == kDiTBlocks ? adapter.file : *lora.pi_base;
                const int width = modulation_width(block);
                if (!adapter.file.tensor(error, modulation_weight(block), { width, 8 })
                    || !bias.tensor(error, modulation_bias(block), { width }))
                    return false;
                consumed.insert(modulation_weight(block));
            }
            consumed.insert(modulation_bias(kDiTBlocks));
            const auto replace = [&](const std::string& name, int count, int width, const std::string& layer, int index)
            {
                if (!node(core, layer))
                    return false;
                const auto shape = width == 1 ? std::vector<int> { count } : std::vector<int> { count, width };
                const auto* tensor = adapter.file.tensor(error, name, shape);
                if (!tensor || !insert(core, layer, index, { {}, tensor, nullptr, count, width }))
                    return false;
                consumed.insert(name);
                return true;
            };
            if (!replace("norm_out.norm.weight", kDiTWidth, 1, "s56_rmsn_0", 0))
                return false;
            for (const auto& head : { std::string("proj_out"), std::string("audio_proj_out") })
            {
                const bool audio = head == "audio_proj_out";
                const int rows = audio ? 32 : 96;
                const auto layer = gemm_name(audio ? 58 : 57);
                if (!replace(head + ".weight", rows, kDiTWidth, layer, 0)
                    || !replace(head + ".bias", rows, 1, layer, 1))
                    return false;
            }
        }

        size_t pairs = 0;
        bool modifies_modulations = false;
        for (const auto& entry : adapter.file.tensors)
        {
            const std::string_view name(entry.first);
            constexpr std::string_view a_suffix = ".lora_A.weight";
            constexpr std::string_view down_suffix = ".lora_down.weight";
            const bool a_format = name.size() >= a_suffix.size() && name.substr(name.size() - a_suffix.size()) == a_suffix;
            const auto suffix = a_format ? a_suffix : down_suffix;
            if (name.size() <= suffix.size() || name.substr(name.size() - suffix.size()) != suffix)
                continue;
            const std::string prefix(name.substr(0, name.size() - suffix.size()));
            const std::string up_key = prefix + (a_format ? ".lora_B.weight" : ".lora_up.weight");
            const auto target = target_for(prefix);
            if (target.segment < 0)
                return fail("unsupported H3 LoRA target: " + prefix);
            const Component component = target.modulation ? producer : core;
            const auto layer = gemm_name(target.segment, target.slot);
            const auto* base = node(component, layer);
            if (!base)
                return false;
            const int cols = base->integer(error, 9);
            const int rows = base->integer(error, 8);
            const auto& down = entry.second;
            if (base->type != "Gemm" || base->integer(error, 4) != 0 || base->integer(error, 5) != 1
                || base->integer(error, 3) != 1 || !error.empty() || cols <= 0 || rows <= 0 || down.shape.size() != 2
                || down.shape[1] != cols || (target.swap_gate && rows % 2 != 0))
                return fail("incompatible H3 LoRA projection " + prefix + " at " + layer);
            const int rank = down.shape[0];
            if (rank > std::min(cols, rows * target.projections))
                return fail("invalid LoRA rank for " + prefix);
            const auto* up = adapter.file.tensor(error, up_key, { rows * target.projections, rank });
            if (!up)
                return false;
            float scale = adapter.strength;
            const auto alpha = adapter.file.tensors.find(prefix + ".alpha");
            if (alpha != adapter.file.tensors.end())
            {
                float value = 0.f;
                if (alpha->second.elements != 1 || !adapter.file.read(alpha->second, 0, 1, &value))
                    return fail("invalid LoRA alpha for " + prefix);
                scale *= value / rank;
                consumed.insert(alpha->first);
            }
            else if (const auto it = adapter.file.metadata.find("ss_network_alpha"); it != adapter.file.metadata.end())
            {
                if (!it->second.empty() && it->second != "None")
                {
                    float value = 0.f;
                    if (!parse_float(it->second, value, false))
                        return fail("invalid LoRA alpha for " + prefix);
                    scale *= value / rank;
                }
            }
            if (!std::isfinite(scale))
                return fail("invalid LoRA scale for " + prefix);
            if (target.modulation && scale != 0.f)
            {
                if (lora.pi)
                    return fail("pi-H3 replaces the modulation projections targeted by " + adapter.name);
                modifies_modulations = true;
            }
            for (int projection = 0; projection < target.projections; projection++)
            {
                const auto part = gemm_name(target.segment, target.slot + projection);
                const auto* projection_node = node(component, part);
                if (!projection_node)
                    return false;
                if (projection_node->integer(error, 8) != rows || projection_node->integer(error, 9) != cols)
                    return fail("incompatible Q/K/V projection in " + part);
                if (!insert(component, part, 0, { {}, &down, up, rows, cols, rows * projection, target.swap_gate, scale }))
                    return false;
            }
            consumed.insert(entry.first);
            consumed.insert(up_key);
            pairs++;
        }
        if (!pairs)
            return fail("no H3 LoRA tensors found");
        for (const auto& entry : adapter.file.tensors)
            if (!consumed.count(entry.first))
                return fail("unsupported or unpaired adapter tensor: " + entry.first);

        if (modifies_modulations)
        {
            char scale[32];
            std::snprintf(scale, sizeof(scale), "%a", adapter.strength);
            const auto key = adapter.file.content_key(error);
            if (!error.empty())
                return false;
            lora.bank_key += (lora.bank_key.empty() ? "lora-v1:" : ":") + key + ":" + scale;
        }
        std::fprintf(stderr, "LoRA %s, %zu projection pairs, strength %.4g\n", adapter.name.c_str(), pairs, adapter.strength);
    }

    if (lora.pi)
    {
        // pi-H3 requires strength 1. The other adapters leave its modulation tables intact.
        const auto adapter_key = lora.pi->file.content_key(error);
        if (!error.empty())
            return false;
        const auto base_key = lora.pi_base->content_key(error);
        if (!error.empty())
            return false;
        lora.bank_key = "pi-h3-v2:" + adapter_key + ":0x1p+0:" + base_key;
    }

    for (const auto& [component, groups] : lora.patches)
        for (const auto& [offset, group] : groups)
        {
            uint64_t rank = 0, replacement_bytes = 0;
            for (const auto& patch : group)
                if (patch.up)
                    rank += patch.down->shape[0];
                else
                    replacement_bytes = 4 * patch.span.elements;
            if (rank > std::numeric_limits<int>::max())
                return fail("combined LoRA rank is too large");
            // Include FP32 source, CPU Gemm packing and the current output batch.
            const uint64_t cols = group.front().cols;
            const uint64_t workspace = replacement_bytes + (rank == 0 ? 0
                : 4 * (3 * rank * cols + 3 * rank * kMergeRows + 3 * uint64_t(kMergeRows) * cols));
            auto& maximum = lora.workspace[component];
            maximum = std::max(maximum, workspace);
        }
    return true;
}

bool Lora::live(Component component) const
{
    return impl_->live.count(component) != 0;
}

void Lora::add_live_graph(std::string& error, Component component, MemoryGraph& graph) const
{
    const auto it = impl_->live.find(component);
    if (it == impl_->live.end())
        return;
    std::map<std::string, const Patch*> targets;
    for (const auto& entry : it->second)
        targets.emplace(entry.second.layer, &entry.second);

    std::vector<MemoryNode> nodes;
    nodes.reserve(graph.nodes.size() + targets.size() * 5);
    for (auto& node : graph.nodes)
    {
        const auto target = targets.find(node.name);
        if (target == targets.end())
        {
            nodes.push_back(std::move(node));
            continue;
        }
        if (node.type != "Gemm" || node.inputs.size() != 1 || node.outputs.size() != 1)
        {
            error = "incompatible runtime LoRA target " + node.name;
            return;
        }
        const auto& patch = *target->second;
        const int rank = patch.down->shape[0];
        const std::string prefix = node.name + "_lora";
        const std::string input = node.inputs[0], output = node.outputs[0];
        const std::string base = prefix + "_base", delta = prefix + "_delta";
        nodes.push_back(
            { "Split", prefix + "_split", { input }, { prefix + "_base_input", prefix + "_low_input" }, {}, {} });
        node.inputs[0] = prefix + "_base_input";
        node.outputs[0] = base;
        nodes.push_back(std::move(node));
        const auto projection
            = [&](const std::string& name, const std::string& from, const std::string& to, int rows, int cols)
        {
            nodes.push_back({ "Gemm", name, { from }, { to },
                { { 3, "1" }, { 5, "1" }, { 6, "1" }, { 8, std::to_string(rows) }, { 9, std::to_string(cols) },
                    { 10, "-1" } },
                {} });
            graph.loading_source_mapping[name] = (uint64_t(rows) * cols * 2 + 3) & ~uint64_t(3);
        };
        projection(prefix + "_down", prefix + "_low_input", prefix + "_rank", rank, patch.cols);
        projection(prefix + "_up", prefix + "_rank", delta, patch.rows, rank);
        std::string update = delta;
        if (patch.scale != 1.f)
        {
            std::ostringstream strength;
            strength << std::setprecision(std::numeric_limits<float>::max_digits10) << patch.scale;
            update = prefix + "_scaled";
            nodes.push_back({ "BinaryOp", prefix + "_scale", { delta }, { update },
                { { 0, "2" }, { 1, "1" }, { 2, strength.str() } }, {} });
        }
        nodes.push_back({ "BinaryOp", prefix + "_sum", { base, update }, { output }, { { 0, "0" } }, {} });
        targets.erase(target);
    }
    if (!targets.empty())
    {
        error = "runtime LoRA target missing from the DiT graph: " + targets.begin()->first;
        return;
    }
    graph.nodes = std::move(nodes);
    graph.producers.clear();
    for (size_t i = 0; i < graph.nodes.size(); i++)
        for (const auto& output : graph.nodes[i].outputs)
            graph.producers.emplace(output, i);
}

bool Lora::load_model(std::string& error, ncnn::Net& net, Component component, const std::string& path) const
{
    const auto merged = impl_->patches.find(component);
    const auto live = impl_->live.find(component);
    if (merged == impl_->patches.end() && live == impl_->live.end())
        return net.load_model(path.c_str()) == 0;
    const Patches empty;
    const Patches& patches = merged == impl_->patches.end() ? empty : merged->second;
    const LivePatches empty_live;
    const LivePatches& injected = live == impl_->live.end() ? empty_live : live->second;
    const WeightReader reader(path, patches, injected);
    if (!patches.empty())
        std::fprintf(stderr, "merging %zu LoRA tensors while loading %s\n", patches.size(), model_path(component));
    if (!injected.empty())
        std::fprintf(stderr, "loading %zu runtime LoRA projections for %s\n", injected.size(), model_path(component));
    if (net.load_model(reader) == 0 && reader.complete())
        return true;
    error = "cannot load LoRA tensors for " + path;
    return false;
}

bool Lora::make_modulations(std::string& error, const std::vector<float>& times, std::vector<ncnn::Mat>& values) const
{
    const auto& base = *impl_->pi_base;
    const auto& file = impl_->pi->file;
    std::vector<float> table(1025 * 8);
    const auto* table_tensor = base.tensor(error, "time_embedder.table", { 1025, 8 });
    if (!table_tensor)
        return false;
    if (!base.read(*table_tensor, 0, table.size(), table.data()))
    {
        error = "cannot read pi-H3 timestep table";
        return false;
    }
    // LakonLab casts the pruned checkpoint and replacement weights to the
    // transformer's dtype before evaluating the timestep path.
    for (float& value : table)
        value = round_bf16(value);
    std::vector<float> embeddings(times.size() * 8);
    for (size_t row = 0; row < times.size(); row++)
    {
        const float coord = std::clamp(times[row], 0.f, 1.f) * 1024;
        const int lo = int(coord), hi = std::min(lo + 1, 1024);
        for (int j = 0; j < 8; j++)
        {
            const float first = table[lo * 8 + j], last = table[hi * 8 + j];
            const float v = std::fma(coord - lo, last - first, first);
            embeddings[row * 8 + j] = round_bf16(v);
        }
    }
    for (int block = 0; block <= kDiTBlocks; block++)
    {
        const auto& bias_file = block == kDiTBlocks ? file : base;
        const int width = modulation_width(block);
        std::vector<float> weight(size_t(width) * 8), bias(width);
        const auto* weight_tensor = file.tensor(error, modulation_weight(block), { width, 8 });
        if (!weight_tensor)
            return false;
        const auto* bias_tensor = bias_file.tensor(error, modulation_bias(block), { width });
        if (!bias_tensor)
            return false;
        if (!file.read(*weight_tensor, 0, weight.size(), weight.data())
            || !bias_file.read(*bias_tensor, 0, bias.size(), bias.data()))
        {
            error = "cannot read pi-H3 modulation projection";
            return false;
        }
        ncnn::Mat output(width, int(times.size()));
        if (output.empty())
        {
            error = "cannot allocate pi-H3 modulation table";
            return false;
        }
        for (float& value : weight)
            value = round_bf16(value);
        for (float& value : bias)
            value = round_bf16(value);
        for (size_t row = 0; row < times.size(); row++)
        {
            float* dst = output.row(int(row));
            for (int i = 0; i < width; i++)
            {
                float sum = bias[i];
                for (int j = 0; j < 8; j++)
                {
                    const float w = weight[i * 8 + j];
                    sum = std::fma(w, embeddings[row * 8 + j], sum);
                }
                dst[i] = round_bf16(sum);
            }
        }
        values.push_back(std::move(output));
    }
    return true;
}
}
