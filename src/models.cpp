// SPDX-License-Identifier: Apache-2.0
#include "models.h"
#include "runtime.h"
#include "lora.h"
#include "reference_kv.h"
#include "rough_graph.h"
#include "snake.h"

#include <command.h>
#include <gpu.h>
#include <layer.h>
#include <net.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>

namespace h3
{
const char* model_path(Component component)
{
    switch (component)
    {
    case Component::Vision:
        return "vision/vision_encoder";
    case Component::Text:
        return "text_encoder/text_encoder";
    case Component::VideoEncoder:
        return "vae/encoder_spatial";
    case Component::TemporalVideoEncoder:
        return "vae/encoder_temporal";
    case Component::VideoDecoder:
        return "vae/decoder";
    case Component::AudioEncoder:
        return "audio_vae/encoder";
    case Component::AudioDecoder:
        return "audio_vae/decoder";
    case Component::TransformerCore:
        return "transformer/fl2va/transformer-core";
    case Component::ReferenceTransformerCore:
        return "transformer/ref2va/transformer-core";
    case Component::Modulations:
        return "transformer/fl2va/fixed-build/modulations";
    case Component::ReferenceModulations:
        return "transformer/ref2va/fixed-build/modulations";
    case Component::Count:
        break;
    }
    return "";
}

bool model_bf16(Component component)
{
    return component == Component::Text || component == Component::TransformerCore
        || component == Component::ReferenceTransformerCore || component == Component::Modulations
        || component == Component::ReferenceModulations;
}

bool model_subgroup_ops(Component component, const ncnn::VulkanDevice* device)
{
    // On NVIDIA the multi-workgroup reduction avoids long serial GroupNorm reductions on the encoder tiles.
    return component != Component::TemporalVideoEncoder || device->info.vendor_id() != 0x10de;
}

namespace
{
struct VulkanWorkspace
{
    explicit VulkanWorkspace(const ncnn::VulkanDevice* device)
        : blob(device)
        , staging(device)
    {
    }

    ncnn::VkBlobAllocator blob;
    ncnn::VkStagingAllocator staging;
};
}

H3ModelSet::H3ModelSet(std::string& error, const std::string& directory, int gpu, int rough_mode, uint64_t kv_cache_limit)
    : directory_(directory)
    , gpu_(-1)
{
    if (ncnn::create_gpu_instance() != 0)
    {
        error = "failed to initialize Vulkan";
        return;
    }
    if (gpu == -1)
        gpu = ncnn::get_default_gpu_index();
    if (gpu < 0 || gpu >= ncnn::get_gpu_count())
    {
        ncnn::destroy_gpu_instance();
        error = "requested Vulkan device is unavailable";
        return;
    }
    gpu_ = gpu;
    if (kv_cache_limit)
        cache_ = std::make_unique<ReferenceKVCache>(rough_mode, kv_cache_limit);
}

H3ModelSet::~H3ModelSet()
{
    release();
    cache_.reset();
    if (gpu_ >= 0)
        ncnn::destroy_gpu_instance();
}

const ncnn::VulkanDevice* H3ModelSet::device() const
{
    return ncnn::get_gpu_device(gpu_);
}

ncnn::Net* H3ModelSet::prepare(std::string& error, Component component)
{
    if (net_ && loaded_ == component)
        return net_.get();
    release();

    const bool bf16 = model_bf16(component);
    const auto& plan = memory_plan_.stages[static_cast<size_t>(component)].selected;
    if (plan.vram)
    {
        const auto available = query_memory(error, device());
        if (!error.empty())
            return nullptr;
        if (available.vram_available < plan.vram || available.ram_available < plan.ram
            || available.host_visible_available < plan.host_visible
            || (available.shared_heap && available.ram_available < plan.ram + plan.vram))
        {
            error = std::string("available memory fell below the planned peak for ") + component_name(component);
            return nullptr;
        }
    }
    const std::string path = directory_ + "/" + model_path(component);
    std::fprintf(stderr, "loading %s\n", model_path(component));
    auto net = std::make_unique<ncnn::Net>();
    net->set_vulkan_device(gpu_);
    if ((component == Component::AudioEncoder || component == Component::AudioDecoder) && !register_snake_layer(*net))
    {
        error = "cannot register the audio VAE activations";
        return nullptr;
    }
    net->opt.use_vulkan_compute = true;
    net->opt.use_subgroup_ops = net->opt.use_subgroup_ops && model_subgroup_ops(component, device());
    net->opt.use_fp16_packed = false;
    net->opt.use_fp16_storage = false;
    net->opt.use_fp16_arithmetic = false;
    net->opt.use_bf16_packed = bf16;
    net->opt.use_bf16_storage = bf16;
    net->opt.use_winograd_convolution = false;
    net->opt.use_local_pool_allocator = false;
    if (plan.paged)
    {
        net->opt.use_weights_in_disk = true;
        net->opt.weight_device_memory_limit = static_cast<size_t>(plan.vram_weights);
        net->opt.weight_host_memory_limit = static_cast<size_t>(plan.host_weights);
        if (plan.disk_weights)
        {
            std::error_code ec;
            const char* override_directory = std::getenv("H3_WEIGHT_TMPDIR");
            const auto temporary = override_directory ? std::filesystem::path(override_directory)
                                                     : std::filesystem::temp_directory_path(ec);
            const auto disk = std::filesystem::space(temporary, ec);
            if (ec || disk.available < plan.disk_weights)
            {
                error = std::string("insufficient temporary disk space for paged ") + component_name(component)
                    + "; set H3_WEIGHT_TMPDIR to a disk-backed directory with sufficient free space";
                return nullptr;
            }
            net->set_weight_file_directory(temporary.string().c_str());
        }
    }
    const bool rough = cache_ && (component == Component::TransformerCore || component == Component::ReferenceTransformerCore);
    const std::string param = path + ".ncnn.param";
    int load_result;
    if (rough || (lora_ && lora_->live(component)))
    {
        const auto core = read_memory_graph(error, param);
        if (!core)
            return nullptr;
        auto graph = rough ? make_rough_graph(error, *core) : *core;
        if (!error.empty())
            return nullptr;
        if (lora_)
            lora_->add_live_graph(error, component, graph);
        if (!error.empty())
            return nullptr;
        if (rough && !cache_->register_layer(*net))
        {
            error = "cannot register the transformer K/V cache layer";
            return nullptr;
        }
        load_result = net->load_param_mem(rough_graph_param(graph).c_str());
    }
    else
        load_result = net->load_param(param.c_str());
    if (load_result != 0)
    {
        error = "cannot load " + param;
        return nullptr;
    }
    for (const ncnn::Layer* layer : net->layers())
    {
        if (layer->type != "Input" && !layer->support_vulkan)
        {
            error = "Vulkan implementation required for " + layer->type + " " + layer->name;
            return nullptr;
        }
    }
    if (lora_ ? !lora_->load_model(error, *net, component, path + ".ncnn.bin")
              : net->load_model((path + ".ncnn.bin").c_str()) != 0)
    {
        if (error.empty())
            error = "cannot load " + path + ".ncnn.bin";
        return nullptr;
    }

    if (plan.vram)
    {
        const auto available = query_memory(error, device());
        if (!error.empty())
            return nullptr;
        if (available.vram_available < plan.vram_workspace)
        {
            error = std::string("available VRAM fell below the workspace required by ") + component_name(component);
            return nullptr;
        }
    }

    loaded_ = component;
    net_ = std::move(net);
    return net_.get();
}

void H3ModelSet::release()
{
    net_.reset();
    if (cache_ && (loaded_ == Component::TransformerCore || loaded_ == Component::ReferenceTransformerCore))
        cache_->clear();
}

ncnn::Mat unpack_float32(std::string& error, const ncnn::Mat& value)
{
    ncnn::Mat output;
    if (value.elembits() == 16)
        ncnn::cast_bfloat16_to_float32(value, output);
    else
        output = value;
    if (output.elempack != 1)
    {
        ncnn::Mat unpacked;
        ncnn::convert_packing(output, unpacked, 1);
        output = unpacked;
    }
    if (output.empty() || output.elembits() != 32 || output.elempack != 1)
    {
        error = "expected unpacked FP32 output";
        return {};
    }
    return output;
}

bool run_model(std::string& error, const ncnn::Net& net, const std::vector<ModelInput>& inputs,
    const std::vector<const char*>& names, std::vector<ncnn::Mat>& outputs, bool active_only)
{
    const auto* device = net.vulkan_device();
    VulkanWorkspace workspace(device);
    ncnn::Option raw_upload = net.opt;
    raw_upload.blob_vkallocator = &workspace.blob;
    raw_upload.staging_vkallocator = &workspace.staging;
    raw_upload.use_bf16_packed = false;
    raw_upload.use_bf16_storage = false;
    ncnn::Extractor extractor = net.create_extractor();
    extractor.set_blob_vkallocator(&workspace.blob);
    extractor.set_workspace_vkallocator(&workspace.blob);
    extractor.set_staging_vkallocator(&workspace.staging);

    for (const auto& input : inputs)
    {
        int status;
        if (input.fp32)
        {
            ncnn::VkMat uploaded;
            ncnn::VkCompute command(device);
            command.record_upload(input.value, uploaded, raw_upload);
            status = command.submit_and_wait();
            if (status == 0)
                status = extractor.input(input.name, uploaded);
        }
        else
        {
            status = extractor.input(input.name, input.value);
        }
        if (status != 0)
        {
            error = std::string("cannot bind model input ") + input.name;
            return false;
        }
    }

    if (active_only)
    {
        ncnn::VkMat active;
        ncnn::VkCompute command(device);
        if (extractor.extract("active_modalities", active, command) || active.empty() || command.submit_and_wait()
            || extractor.input("hidden_0", active))
        {
            error = "cannot bind generated video and audio rows for rough mode";
            return false;
        }
    }

    outputs.clear();
    for (const char* name : names)
    {
        ncnn::Mat value;
        if (extractor.extract(name, value) != 0)
        {
            error = std::string("cannot extract model output ") + name;
            return false;
        }
        outputs.push_back(unpack_float32(error, value));
        if (outputs.back().empty())
            return false;
    }
    return true;
}
}
