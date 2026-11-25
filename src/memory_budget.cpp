// SPDX-License-Identifier: Apache-2.0
#include "memory_budget.h"
#include "runtime.h"
#include "h3_config.h"
#include "lora.h"
#include "rough_graph.h"
#include "scheduler.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <numeric>

namespace h3
{
namespace
{
using Shape = std::vector<uint64_t>;

uint64_t mib(uint64_t bytes)
{
    return (bytes + kMiB - 1) / kMiB;
}

uint64_t tile_count(uint64_t extent)
{
    return 1 + (std::max<uint64_t>(extent, kVaeTile) - kVaeTile + kVaeTileStride - 1) / kVaeTileStride;
}

uint64_t audio_rows(int samples)
{
    return 2 * ((static_cast<uint64_t>(samples) + kAudioHop - 1) / kAudioHop);
}

GraphMemoryRequest call(std::initializer_list<std::pair<const char*, Shape>> inputs,
    std::initializer_list<const char*> outputs = { "out0" })
{
    GraphMemoryRequest request;
    for (const auto& it : inputs)
        request.inputs.emplace(it.first, MemoryShape { it.second });
    request.outputs.assign(outputs.begin(), outputs.end());
    return request;
}

struct StageResources
{
    std::vector<GraphMemoryUsage> graphs;
    uint64_t ram = 0;
    uint64_t device = 0;
    uint64_t staging = 0;
    uint64_t load_base = 0;
    uint64_t load_extra = 0;
};

StageMemory placement(const WeightMemoryUsage& usage, const StageResources& resources, bool paged, bool unified)
{
    StageMemory result;
    result.load_ram = resources.load_base + resources.load_extra + usage.loading_host_peak;
    result.forward_ram = resources.ram + usage.host_resident + usage.streaming_host_staging + usage.streaming_host_source;
    result.ram = std::max(result.load_ram, result.forward_ram);
    result.vram_weights = usage.device_resident;
    result.host_weights = usage.host_resident;
    result.host_visible = std::max(usage.loading_staging_peak,
        resources.staging + usage.host_resident + usage.streaming_host_staging);
    result.vram_workspace = resources.device + usage.streaming_device_pool;
    if (unified)
        result.vram_workspace += std::max(usage.loading_staging_peak, resources.staging + usage.streaming_host_staging);
    result.vram = result.vram_weights + result.vram_workspace;
    result.paged = paged;
    result.disk_weights = usage.disk_payload;
    return result;
}

constexpr size_t kComponents = static_cast<size_t>(Component::Count);
using StageRam = std::array<uint64_t, kComponents>;

struct StageRequest
{
    std::vector<GraphMemoryRequest> graphs;
    uint64_t retained;
    uint64_t extra_host;
    uint64_t extra_device;
    bool rough;
};

int reference_latent_frames(int frames)
{
    if (frames <= 1)
        return frames;
    const int aligned = std::max(1, (frames - kVaeTailFrames) / kVaeClipFrames) * kVaeClipFrames + kVaeTailFrames;
    const int chunks = (std::min(frames, aligned) + kVaeClipFrames - 1) / kVaeClipFrames;
    return (chunks - 1) * kVaeClipLatents + kVaeTailLatents;
}

std::pair<StageRam, StageRam> retained_clip_memory(const MemoryWorkload& w)
{
    uint64_t rgb = 0, input_audio = 0, image_latents = 0, audio_latents = 0;
    for (const auto& ref : w.references)
    {
        rgb += uint64_t(ref.width) * ref.height * ref.frames * 3;
        input_audio += uint64_t(ref.samples) * 2 * sizeof(float);
        image_latents += uint64_t(ref.width / 16) * (ref.height / 16) * reference_latent_frames(ref.frames)
            * kVideoLatentChannels * sizeof(float);
        audio_latents += audio_rows(ref.samples) * kAudioLatentChannels * sizeof(float);
    }
    const uint64_t vision = uint64_t(w.vision_patches / 4) * 5120 * 4 * sizeof(float);
    const uint64_t text = uint64_t(w.text_rows) * 5120 * sizeof(float);
    const uint64_t video = uint64_t(w.width / 16) * (w.height / 16) * video_latent_frames(w.frames)
        * kVideoLatentChannels * sizeof(float);
    const uint64_t audio = uint64_t(w.audio_frames) * 2 * kAudioLatentChannels * sizeof(float) + audio_latents;
    const uint64_t pcm = uint64_t(w.audio_frames) * kAudioHop * 2 * sizeof(float);
    const uint64_t raw = rgb + input_audio;
    const uint64_t conditioned = text + image_latents + audio_latents;

    StageRam completed {}, upcoming {};
    const auto set = [&](Component component, uint64_t before, uint64_t after)
    {
        completed[static_cast<size_t>(component)] = before;
        upcoming[static_cast<size_t>(component)] = after;
    };
    set(Component::Vision, raw + vision, raw);
    set(Component::Text, raw + text, raw + vision);
    for (auto component : {Component::VideoEncoder, Component::TemporalVideoEncoder})
        set(component, raw + text + image_latents, raw + text);
    set(Component::AudioEncoder, conditioned, text + image_latents + input_audio);
    for (auto component : {Component::Modulations, Component::ReferenceModulations})
        set(component, conditioned, conditioned);
    for (auto component : {Component::TransformerCore, Component::ReferenceTransformerCore})
        set(component, video + audio, conditioned);
    set(Component::AudioDecoder, video + pcm, video + audio);
    set(Component::VideoDecoder, 0, video + pcm);
    return {completed, upcoming};
}

std::vector<StageRam> retained_other_clips(const std::vector<MemoryWorkload>& workloads)
{
    std::vector<StageRam> result(workloads.size());
    if (workloads.size() <= 1)
        return result;

    std::vector<StageRam> upcoming(workloads.size());
    StageRam sum {};
    for (size_t i = 0; i < workloads.size(); i++)
    {
        result[i] = sum;
        const auto [completed, pending] = retained_clip_memory(workloads[i]);
        upcoming[i] = pending;
        for (size_t c = 0; c < kComponents; c++)
            sum[c] += completed[c];
    }
    sum = {};
    for (size_t i = workloads.size(); i-- > 0;)
        for (size_t c = 0; c < kComponents; c++)
        {
            result[i][c] += sum[c];
            sum[c] += upcoming[i][c];
        }
    return result;
}

bool fits(const StageMemory& stage, const MemoryPlan& plan, const MemorySnapshot& available)
{
    return stage.ram <= plan.ram_budget && stage.vram <= plan.vram_budget
        && stage.host_visible <= available.host_visible_available
        && (!available.shared_heap || stage.ram + stage.vram <= plan.ram_budget);
}
}

const char* component_name(Component component)
{
    switch (component)
    {
    case Component::Vision: return "Vision";
    case Component::Text: return "Text encoder";
    case Component::VideoEncoder: return "Image VAE encode";
    case Component::TemporalVideoEncoder: return "Video VAE encode";
    case Component::VideoDecoder: return "Video VAE decode + media";
    case Component::AudioEncoder: return "Audio VAE encode";
    case Component::AudioDecoder: return "Audio VAE decode";
    case Component::TransformerCore:
    case Component::ReferenceTransformerCore: return "DiT core";
    case Component::Modulations:
    case Component::ReferenceModulations: return "Modulation producer";
    case Component::Count: break;
    }
    return "";
}

MemoryPlan plan_memory(std::string& error, const std::string& model, const ncnn::VulkanDevice* device,
    const std::vector<MemoryWorkload>& workloads, uint64_t ram_budget, uint64_t vram_budget, bool estimate)
{
    const auto available = query_memory(error, device);
    if (!error.empty())
        return {};

    MemoryPlan result;
    result.ram_budget = ram_budget ? (estimate ? ram_budget : std::min(ram_budget, available.ram_available))
                                   : available.ram_available;
    result.vram_budget = vram_budget ? (estimate ? vram_budget : std::min(vram_budget, available.vram_available))
                                     : available.vram_available;
    const auto retained = retained_other_clips(workloads);
    std::array<std::vector<StageRequest>, kComponents> requests;
    const std::array<AllocationLayout, 2> layouts = {
        query_allocation_layout(device, false), query_allocation_layout(device, true)
    };

    for (size_t i = 0; i < workloads.size(); i++)
    {
        const auto& w = workloads[i];
        const auto& other = retained[i];
        result.preprocessing_ram = std::max(result.preprocessing_ram, w.preprocessing_peak);
        if (result.preprocessing_ram > result.ram_budget)
        {
            result.feasible = false;
            if (result.failure.empty())
                result.failure = "reference preprocessing exceeds the RAM budget";
        }
        const auto stage = [&](Component component, std::vector<GraphMemoryRequest> graphs, uint64_t current,
                               uint64_t extra_host = 0, uint64_t extra_device = 0, bool rough = false)
        {
            const auto index = static_cast<size_t>(component);
            requests[index].push_back({std::move(graphs), w.base_ram + other[index] + current, extra_host, extra_device, rough});
        };
        const uint64_t text = w.text_rows, width = w.width, height = w.height;
        const uint64_t h = height / kSpatialScale, x = width / kSpatialScale;
        const uint64_t latent_frames = w.frames ? video_latent_frames(w.frames) : 0;
        const uint64_t generated_video = x * h * latent_frames / 4;
        const uint64_t generated_audio = 2 * w.audio_frames;
        uint64_t reference_video = 0, reference_audio = 0, rgb = 0, input_audio = 0;
        std::array<std::vector<GraphMemoryRequest>, 2> encoding;
        std::vector<GraphMemoryRequest> audio_encoding;
        for (const auto& ref : w.references)
        {
            const uint64_t rw = ref.width, rh = ref.height;
            if (ref.frames)
            {
                const bool spatial = ref.frames == 1;
                const uint64_t depth = reference_latent_frames(ref.frames);
                reference_video += (rw / 32) * (rh / 32) * depth;
                rgb += 3 * rw * rh * ref.frames;
                const uint64_t tw = std::min(rw, uint64_t(kVaeTile * kSpatialScale));
                const uint64_t th = std::min(rh, uint64_t(kVaeTile * kSpatialScale));
                auto request = call({ { "in0", spatial ? Shape { tw, th, 3 } : Shape { tw, th, kVaeClipFrames, 3 } } });
                encoding[!spatial].push_back(std::move(request));
            }
            if (ref.samples)
            {
                const uint64_t length = audio_rows(ref.samples) / 2;
                reference_audio += length * 2;
                input_audio += uint64_t(ref.samples) * 2 * sizeof(float);
                audio_encoding.push_back(call({ { "in0", {length * kAudioHop, 1, 1} }, { "in1", {length, length, 1} } }));
            }
        }
        const uint64_t image_latents = reference_video * 4 * kVideoLatentChannels * sizeof(float);
        const uint64_t audio_latents = reference_audio * kAudioLatentChannels * sizeof(float);
        const uint64_t raw = rgb + input_audio;
        const uint64_t vision = w.vision_patches, largest_vision = w.max_vision_patches;
        const uint64_t vision_output = (vision / 4) * 5120 * 4 * sizeof(float);
        const uint64_t vision_host = w.resized_vision_bytes + vision_output;

        if (largest_vision)
            stage(Component::Vision,
                { call({ { "in0", {1536, largest_vision} }, { "positions", {1152, largest_vision} },
                        { "cos", {36, largest_vision, 1} }, { "sin", {36, largest_vision, 1} } },
                    { "out0", "deepstack0", "deepstack1", "deepstack2" }) },
                raw + vision_host + 48 * 48 * 1152 * 4);

        if (text)
            stage(Component::Text,
                { call({ { "embeddings", {5120, text} }, { "mask", {text, text} }, { "cos", {64, text, 1} },
                        { "sin", {64, text, 1} }, { "deepstack0", {5120, text} }, { "deepstack1", {5120, text} },
                        { "deepstack2", {5120, text} } }) },
                raw + vision_output);
        const uint64_t text_ram = text * 5120 * sizeof(float);
        for (int i = 0; i < 2; i++)
            if (!encoding[i].empty())
                stage(i == 0 ? Component::VideoEncoder : Component::TemporalVideoEncoder,
                    std::move(encoding[i]), raw + image_latents + text_ram);
        if (!audio_encoding.empty())
            stage(Component::AudioEncoder, std::move(audio_encoding), input_audio + image_latents + audio_latents + text_ram);

        const auto producer = w.ref2va ? Component::ReferenceModulations : Component::Modulations;
        if (w.prepare_modulations && w.pi_modulations)
        {
            constexpr uint64_t width = kDiTWidth * 18;
            const uint64_t count = w.modulation_rows;
            const uint64_t tables = count * kDiTWidth * (18 * kDiTBlocks + 2);
            const uint64_t frequencies = uint64_t(w.steps - 1) * count * 256;
            const uint64_t latents = ((reference_video + generated_video) * 4 * kVideoLatentChannels
                + (reference_audio + generated_audio) * kAudioLatentChannels) * sizeof(float);
            const uint64_t peak = w.base_ram + other[static_cast<size_t>(producer)] + text_ram + latents
                + (tables + frequencies + width * 9 + 1025 * 8 + count * 8) * sizeof(float);
            result.preprocessing_ram = std::max(result.preprocessing_ram, peak);
            if (peak > result.ram_budget)
            {
                result.feasible = false;
                if (result.failure.empty())
                    result.failure = "pi-H3 modulation tables exceed the RAM budget";
            }
        }
        else if (w.prepare_modulations)
        {
            const uint64_t count = w.modulation_rows;
            auto request = call({ { "time_features", {256, count} } }, {});
            request.raw_inputs.insert("time_features");
            for (int i = 0; i < kDiTBlocks; i++)
            {
                char name[32];
                std::snprintf(name, sizeof(name), "adaln_%02d_raw", i);
                request.outputs.emplace_back(name);
            }
            request.outputs.emplace_back("final_raw");
            stage(producer, {std::move(request)}, uint64_t(w.steps - 1) * count * 256 * 4
                + text_ram + image_latents + audio_latents);
        }

        if (w.steps == 0 || !text)
            continue;

        const auto transformer = w.ref2va ? Component::ReferenceTransformerCore : Component::TransformerCore;
        const uint64_t video = reference_video + generated_video;
        const uint64_t audio = reference_audio + generated_audio;
        const uint64_t rows = text + video + audio;
        const uint64_t times = w.modulation_rows;
        constexpr uint64_t position_stride = sizeof(double) * 3 + sizeof(int);
        const uint64_t packed = (video * 4 * kVideoLatentChannels + audio * kAudioLatentChannels) * sizeof(float);
        const uint64_t layout_bytes = rows * position_stride;
        result.preprocessing_ram = std::max(result.preprocessing_ram,
            w.base_ram + other[static_cast<size_t>(transformer)] + image_latents + audio_latents + text_ram + 3 * packed + 2 * layout_bytes);
        if (result.preprocessing_ram > result.ram_budget)
        {
            result.feasible = false;
            if (result.failure.empty())
                result.failure = "latent packing and reference preprocessing exceed the RAM budget";
        }

        uint64_t kv_pool = 0, prefix = 0, audio_prefix = 0, kv_bytes = 0;
        const bool rough = w.rough_mode && w.steps > 2;
        if (rough)
        {
            prefix = text + (w.rough_mode >= 2 ? reference_video : 0);
            audio_prefix = w.rough_mode >= 2 ? reference_audio : 0;
            const auto& layout = layouts[1];
            AllocationPool cache(error, 16 * kMiB, layout);
            for (int block = 0; block < kDiTBlocks; block++)
                for (int operand = 0; operand < 2; operand++)
                    for (uint64_t count : {prefix, audio_prefix})
                    {
                        if (!count)
                            continue;
                        const uint64_t bytes = MemoryShape { {128, count, 56}, 2 }.storage_bytes(error);
                        kv_bytes += aligned_memory_bytes(error, bytes, layout.buffer_alignment);
                        cache.allocate(error, bytes);
                    }
            kv_pool = cache.allocated_bytes();
            if (!error.empty())
                return {};
            result.kv_bytes = std::max(result.kv_bytes, kv_bytes);
            if (kv_bytes > uint64_t(w.kv_cache_mib) * kMiB)
            {
                result.feasible = false;
                if (result.failure.empty())
                    result.failure = "rough K/V requires --kv-cache-mib " + std::to_string(mib(kv_bytes));
            }
        }

        std::vector<GraphMemoryRequest> dit_calls;
        auto dit = call({ { "video", {4 * kVideoLatentChannels, video} }, { "audio", {kAudioLatentChannels, audio} },
            { "text", {5120, text} }, { "adaln_selector", {times * 3, rows} }, { "time_selector", {times, rows} },
            { "cos", {96, rows, 1} }, { "sin", {96, rows, 1} } }, { "video_velocity", "audio_velocity" });
        dit.raw_inputs = { "video", "audio" };
        for (int i = 0; i < kDiTBlocks; i++)
        {
            char name[32];
            std::snprintf(name, sizeof(name), "adaln_%02d_raw", i);
            dit.inputs.emplace(name, MemoryShape { {kDiTWidth * 6, times * 3}, 2 });
        }
        dit.inputs.emplace("final_raw", MemoryShape { {kDiTWidth * 2, times}, 2 });

        if (rough)
        {
            auto reuse = dit;
            const uint64_t active_video = video - (w.rough_mode >= 2 ? reference_video : 0);
            const uint64_t active_audio = audio - audio_prefix;
            const uint64_t active = active_video + active_audio;
            const uint64_t queries = (active + 3) / 4 * 4;
            reuse.kv_full_rows = rows;
            reuse.kv_active_video = active_video;
            reuse.kv_active_audio = active_audio;
            reuse.kv_has_cached_audio = audio_prefix != 0;
            reuse.inputs.erase("text");
            reuse.inputs.find("video")->second.dimensions[1] = active_video;
            reuse.inputs.find("audio")->second.dimensions[1] = queries - active_video;
            for (const char* name : {"cos", "sin", "adaln_selector", "time_selector"})
                reuse.inputs.find(name)->second.dimensions[1] = queries;
            reuse.aliases.emplace("hidden_0", "active_modalities");
            dit_calls.push_back(std::move(reuse));
        }
        dit_calls.push_back(std::move(dit));

        const uint64_t rough_host = rough ? rows * (96 * 2 + times * 4) * sizeof(float) : 0;
        stage(transformer, std::move(dit_calls), packed + layout_bytes + text_ram + image_latents + audio_latents,
            packed + rough_host, kv_pool, rough);

        const uint64_t audio_frames = w.audio_frames;
        const uint64_t samples = audio_frames * kAudioHop;
        const uint64_t video_latent_bytes = generated_video * 4 * kVideoLatentChannels * sizeof(float);
        stage(Component::AudioDecoder, { call({ {"in0", {audio_frames, 1, kAudioLatentChannels}} }) },
            video_latent_bytes + audio * kAudioLatentChannels * sizeof(float) + samples * sizeof(float));

        const uint64_t tile_w = std::min(x, uint64_t(kVaeTile)), tile_h = std::min(h, uint64_t(kVaeTile));
        const uint64_t patch_rows = tile_w * tile_h * (kVaeClipLatents + kVaeTailLatents);
        const uint64_t clip_frames = (kVaeClipLatents + kVaeTailLatents) * kVaeTemporalScale;
        const uint64_t clip = 3 * width * height * clip_frames * 4;
        const uint64_t tiles = 3 * tile_w * tile_h * kSpatialScale * kSpatialScale * clip_frames * 4 * tile_count(x) * tile_count(h);
        const uint64_t decoder_inputs = video_latent_bytes + samples * 8;
        const uint64_t pixels = width * height;
        const uint64_t sampling = w.lossless ? 6 : 3;
        // libvpx's VP9 frame pool has 8 references, 3 frame references and the
        // current reconstruction. The encoder's border is 160 pixels per side.
        constexpr uint64_t vp9_frame_buffers = 8 + 3 + 1;
        constexpr uint64_t vp9_border = 160;
        const uint64_t luma_stride = (width + 2 * vp9_border + 31) / 32 * 32;
        const uint64_t luma_rows = (height + 7) / 8 * 8 + 2 * vp9_border;
        const uint64_t frame_pool = vp9_frame_buffers * luma_stride * luma_rows * sampling / 2;
        const uint64_t encoder_image = pixels * (6 + sampling) / 2 + frame_pool;
        const uint64_t encoder_audio = samples * (w.lossless ? 8 : 12);
        stage(Component::VideoDecoder,
            { call({ {"in0", {kVideoLatentChannels, patch_rows}}, {"in1", {48, patch_rows + 5, 1}},
                     {"in2", {48, patch_rows + 5, 1}} }) },
            decoder_inputs + 2 * clip + tiles + encoder_image + encoder_audio);
    }

    const auto* lora = workloads.empty() ? nullptr : workloads.front().lora;
    const auto place_stage = [&](Component component, const std::vector<StageRequest>& requests)
    {
        auto layout = layouts[model_bf16(component)];
        layout.subgroup_reductions = layout.subgroup_reductions && model_subgroup_ops(component, device);
        const std::string prefix = model + "/" + model_path(component);
        auto graph = read_memory_graph(error, prefix + ".ncnn.param");
        if (!error.empty())
            return false;
        // The inventory records source storage by walking tensor headers, not by loading model weights.
        MemoryGraph inventory = *graph;
        inspect_weight_storage(error, prefix + ".ncnn.bin", inventory);
        if (!error.empty())
        {
            error = std::string(component_name(component)) + ": " + error;
            return false;
        }
        if (requests.front().rough)
        {
            inventory = make_rough_graph(error, inventory);
            if (!error.empty())
                return false;
        }
        if (lora)
            lora->add_live_graph(error, component, inventory);
        if (!error.empty())
            return false;
        auto weights = model_weight_allocations(error, inventory, layout);
        if (!error.empty())
            return false;

        StageResources resources;
        resources.load_extra = lora ? lora->loading_workspace(component) : 0;
        for (const auto& job : requests)
        {
            resources.load_base = std::max(resources.load_base, job.retained);
            for (const auto& request : job.graphs)
            {
                const auto usage = plan_graph_memory(error, inventory, request, layout);
                if (!error.empty())
                {
                    error = std::string(component_name(component)) + ": " + error;
                    return false;
                }
                uint64_t inputs = 0;
                for (const auto& input : request.inputs)
                    inputs += input.second.storage_bytes(error);

                resources.ram = std::max(resources.ram,
                    job.retained + job.extra_host + inputs + usage.host_transfer_scratch + usage.staging);
                resources.device = std::max(resources.device, job.extra_device + usage.device_pool);
                resources.staging = std::max(resources.staging, usage.staging);
                resources.graphs.push_back(usage);
            }
        }

        const auto make = [&](uint64_t dev, uint64_t host, bool resident)
        {
            return placement(plan_weight_memory(error, weights, resources.graphs, layout, dev, host, resident),
                resources, !resident, available.shared_heap);
        };
        auto& candidate = result.stages[static_cast<size_t>(component)];
        candidate.minimum = make(0, 0, false);
        candidate.resident = make(0, 0, true);
        if (!error.empty())
            return false;
        candidate.selected = candidate.resident;

        if (!fits(candidate.resident, result, available))
        {
            candidate.selected = candidate.minimum;
            if (!fits(candidate.minimum, result, available))
            {
                result.feasible = false;
                if (result.failure.empty())
                    result.failure = std::string(component_name(component)) + " needs at least "
                        + std::to_string(mib(candidate.minimum.ram)) + " MiB RAM and "
                        + std::to_string(mib(candidate.minimum.vram)) + " MiB VRAM with paged weights under this plan";
                return true;
            }

            uint64_t device_limit = result.vram_budget - candidate.minimum.vram_workspace;
            uint64_t host_limit = available.shared_heap ? 0 : std::min(result.ram_budget - candidate.minimum.ram,
                available.host_visible_available - candidate.minimum.host_visible);
            // The actual cache and streaming pool can change after each placement.
            for (;;)
            {
                const auto plan = make(device_limit, host_limit, false);
                if (!error.empty())
                    return false;
                if (fits(plan, result, available))
                {
                    candidate.selected = plan;
                    break;
                }
                const uint64_t shared = available.shared_heap && plan.ram + plan.vram > result.ram_budget
                    ? plan.ram + plan.vram - result.ram_budget : 0;
                const uint64_t device_deficit = std::max(shared, plan.vram > result.vram_budget ? plan.vram - result.vram_budget : 0);
                const uint64_t host_deficit = std::max(plan.ram > result.ram_budget ? plan.ram - result.ram_budget : 0,
                    plan.host_visible > available.host_visible_available ? plan.host_visible - available.host_visible_available : 0);
                const uint64_t next_device = device_limit - std::min(device_limit, device_deficit);
                const uint64_t next_host = host_limit - std::min(host_limit, host_deficit);
                if (device_limit == next_device && host_limit == next_host)
                    break;
                device_limit = next_device;
                host_limit = next_host;
            }
        }
        return true;
    };

    for (size_t i = 0; i < requests.size(); i++)
        if (!requests[i].empty() && !place_stage(static_cast<Component>(i), requests[i]))
            return {};

    return result;
}

void print_memory_plan(const MemoryPlan& plan, bool estimate)
{
    uint64_t min_ram = plan.preprocessing_ram, min_vram = 0, resident_ram = plan.preprocessing_ram, resident_vram = 0;
    uint64_t selected_ram = plan.preprocessing_ram, selected_vram = 0;
    for (size_t i = 0; i < plan.stages.size(); i++)
    {
        const auto& stage = plan.stages[i];
        if (!stage.selected.vram)
            continue;
        min_ram = std::max(min_ram, stage.minimum.ram);
        min_vram = std::max(min_vram, stage.minimum.vram);
        resident_ram = std::max(resident_ram, stage.resident.ram);
        resident_vram = std::max(resident_vram, stage.resident.vram);
        selected_ram = std::max(selected_ram, stage.selected.ram);
        selected_vram = std::max(selected_vram, stage.selected.vram);
        std::fprintf(stderr, "  %-24s RAM %llu MiB, VRAM %llu MiB (%llu workspace, %llu weights), %s, disk %llu MiB\n",
            component_name(static_cast<Component>(i)), (unsigned long long)mib(stage.selected.ram),
            (unsigned long long)mib(stage.selected.vram), (unsigned long long)mib(stage.selected.vram_workspace),
            (unsigned long long)mib(stage.selected.vram_weights), stage.selected.paged ? "paged" : "resident",
            (unsigned long long)mib(stage.selected.disk_weights));
    }
    std::fprintf(stderr, "Memory budget RAM %llu MiB, VRAM %llu MiB\n"
        "Estimated selected peak RAM %llu MiB, VRAM %llu MiB\n",
        (unsigned long long)mib(plan.ram_budget), (unsigned long long)mib(plan.vram_budget),
        (unsigned long long)mib(selected_ram), (unsigned long long)mib(selected_vram));
    if (estimate)
        std::fprintf(stderr, "Estimated minimum with disk paging RAM %llu MiB, VRAM %llu MiB\n"
            "Estimated resident peak RAM %llu MiB, VRAM %llu MiB\n",
            (unsigned long long)mib(min_ram), (unsigned long long)mib(min_vram),
            (unsigned long long)mib(resident_ram), (unsigned long long)mib(resident_vram));
}
}
