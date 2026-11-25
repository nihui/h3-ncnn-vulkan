// SPDX-License-Identifier: Apache-2.0
#include "pipeline.h"
#include "filesystem_utils.h"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <limits>
#include <string_view>

namespace
{
void print_usage()
{
    std::fprintf(stderr,
        "Usage: h3-ncnn-vulkan [options]\n\n"
        "  -h                 show this help\n"
        "  -p TEXT            prompt (one for the video or one per FL2VA segment)\n"
        "  -o PATH            lossy .webm or lossless .mkv (default: output.webm)\n"
        "  -i PATH            FL2VA keyframe or Ref2VA image reference (repeat in order)\n"
        "  -s WIDTH,HEIGHT    output dimensions, multiples of 32\n"
        "  -f N               frames at 24 fps, rounded up (repeat per FL2VA segment)\n"
        "  -l N               scheduler points, including zero (default: 50 or adapter preset)\n"
        "  -r N               sampling seed (default: 42)\n"
        "  -m DIR             ncnn model directory (default: models/h3)\n"
        "  -g N               Vulkan device index (default: automatic)\n"
        "  -t TYPE            t2va, fl2va, ref2va (default: t2va)\n"
        "\n"
        "  --last-image PATH                 optional final FL2VA keyframe\n"
        "  --reference-video PATH            WebM/MKV video reference, including its audio stream\n"
        "  --reference-audio PATH            WAV or WebM/MKV audio reference for fl2va or ref2va\n"
        "  --reference-short-edge N          ref2va image short edge (default: 2048)\n"
        "  --reference-video-short-edge N    ref2va video short edge (default: 768)\n"
        "  --lora PATH                       H3 LoRA .safetensors file (repeat to combine)\n"
        "  --lora-scale VALUE                adapter strengths in --lora order (default: 1)\n"
        "  --bake-modulations TYPE           build a bank only: none, video, audio, both\n"
        "  --rough-mode 0|1|2|3              exact, alternate text, alternate full, freeze full (default: 0)\n"
        "  --kv-cache-mib N                  Vulkan K/V cache ceiling (default: 6144)\n"
        "  --ram-budget MiB                  process RAM budget (0=auto, default)\n"
        "  --vram-budget MiB                 Vulkan memory budget (0=auto, default)\n"
        "  --estimate-memory                 calculate the memory plan without running inference\n");
}

template <class T> T number(std::string& error, std::string_view value, std::string_view name)
{
    T result = 0;
    const auto status = std::from_chars(value.data(), value.data() + value.size(), result);
    if (status.ec != std::errc() || status.ptr != value.data() + value.size())
        error = "invalid value for " + std::string(name);
    return result;
}

bool parse(std::string& error, int argc, char** argv, h3::RuntimeConfig& config, h3::GenerateRequest& request)
{
    const auto fail = [&](std::string message) {
        error = std::move(message);
        return false;
    };
    std::string workflow = "t2va";
    std::vector<std::string> images, prompts;
    std::vector<int> frame_counts;
    std::vector<float> lora_scales;
    for (int i = 1; i < argc; i++)
    {
        const std::string_view option = argv[i];
        if (option == "--estimate-memory")
        {
            config.estimate_memory = true;
            continue;
        }
        if (i + 1 == argc)
            return fail("missing value for " + std::string(option));
        const std::string_view value = argv[++i];
        if (option == "--workflow" || option == "-t")
            workflow = value;
        else if (option == "--image" || option == "-i")
        {
            if (value.empty())
                return fail("-i expects a path");
            images.emplace_back(value);
            request.references.push_back({ h3::ReferenceKind::Image, std::string(value) });
        }
        else if (option == "--reference" || option == "--reference-video" || option == "--reference-audio")
        {
            const auto kind = option == "--reference-video" ? h3::ReferenceKind::Video
                : option == "--reference-audio"             ? h3::ReferenceKind::Audio
                                                            : h3::ReferenceKind::Image;
            request.references.push_back({ kind, std::string(value) });
        }
        else if (option == "--last-image")
        {
            if (!request.last_image.empty() || value.empty())
                return fail("--last-image expects one path");
            request.last_image = value;
        }
        else if (option == "--reference-short-edge")
            request.reference_short_edge = number<int>(error, value, option);
        else if (option == "--reference-video-short-edge")
            request.reference_video_short_edge = number<int>(error, value, option);
        else if (option == "--prompt" || option == "-p")
            prompts.emplace_back(value);
        else if (option == "--width" || option == "-w")
            request.width = number<int>(error, value, option);
        else if (option == "--height")
            request.height = number<int>(error, value, option);
        else if (option == "--ram-budget" || option == "--vram-budget")
        {
            const uint64_t mib = number<uint64_t>(error, value, option);
            if (mib > std::numeric_limits<uint64_t>::max() / (1024 * 1024))
                return fail("budget is out of range");
            (option == "--ram-budget" ? config.ram_budget : config.vram_budget) = mib * 1024 * 1024;
        }
        else if (option == "--frames" || option == "-f")
            frame_counts.push_back(number<int>(error, value, option));
        else if (option == "--steps" || option == "-l" || option == "-s")
        {
            const auto comma = option == "-s" ? value.find(',') : std::string_view::npos;
            if (comma != std::string_view::npos)
            {
                request.width = number<int>(error, value.substr(0, comma), option);
                request.height = number<int>(error, value.substr(comma + 1), option);
                if (error.empty() && (request.width <= 0 || request.height <= 0))
                    return fail("-s WIDTH,HEIGHT requires positive dimensions");
            }
            else
            {
                request.steps = number<int>(error, value, option);
                request.steps_explicit = true;
            }
        }
        else if (option == "--lora")
        {
            if (value.empty())
                return fail("--lora expects an adapter path");
            request.loras.push_back({std::string(value)});
        }
        else if (option == "--lora-scale")
        {
            float scale = 0.f;
            if (!h3::parse_float(std::string(value), scale))
                return fail("invalid --lora-scale value");
            lora_scales.push_back(scale);
        }
        else if (option == "--bake-modulations")
        {
            if (value == "none")
                config.bake_modulations = 0;
            else if (value == "video")
                config.bake_modulations = 1;
            else if (value == "audio")
                config.bake_modulations = 2;
            else if (value == "both")
                config.bake_modulations = 3;
            else
                return fail("invalid --bake-modulations type");
        }
        else if (option == "--rough-mode")
            config.rough_mode = number<int>(error, value, option);
        else if (option == "--kv-cache-mib")
            config.kv_cache_mib = number<int>(error, value, option);
        else if (option == "--seed" || option == "-r")
            request.seed = number<uint64_t>(error, value, option);
        else if (option == "--gpu" || option == "-g")
            config.gpu = number<int>(error, value, option);
        else if (option == "--model" || option == "-m")
            config.model = value;
        else if (option == "--output" || option == "-o")
            request.output = value;
        else
            return fail("unknown option " + std::string(option));
        if (!error.empty())
            return false;
    }

    if (workflow == "fl2va")
    {
        request.workflow = h3::Workflow::KeyframesToVideo;
        auto& references = request.references;
        const auto audios = std::count_if(references.begin(), references.end(), [](const auto& ref) {
            return ref.kind == h3::ReferenceKind::Audio;
        });
        if (audios > 1 || references.size() != images.size() + audios)
            return fail("fl2va accepts keyframes and one --reference-audio");
        if (images.empty() && request.last_image.empty() && audios == 0 && config.bake_modulations < 0)
            return fail("fl2va requires -i, --last-image or --reference-audio");
        if (!images.empty())
        {
            request.first_image = images.front();
            if (!request.last_image.empty())
                images.push_back(request.last_image);
            if (images.size() >= 2)
                request.last_image = images.back();
            if (images.size() > 2)
            {
                const auto count = images.size() - 1;
                if (frame_counts.size() != count)
                    return fail("fl2va requires one -f per pair of adjacent keyframes");
                if (prompts.size() != 1 && prompts.size() != count)
                    return fail("fl2va requires one shared -p or one -p per segment");
                for (size_t i = 0; i < count; i++)
                    request.segments.push_back({images[i], images[i + 1], prompts.size() == 1 ? prompts.front() : prompts[i],
                        frame_counts[i]});
            }
        }
        references.erase(std::remove_if(references.begin(), references.end(), [](const auto& ref) {
            return ref.kind == h3::ReferenceKind::Image;
        }), references.end());
    }
    else if (workflow == "ref2va")
    {
        request.workflow = h3::Workflow::ReferenceToVideo;
        if (!request.last_image.empty() || (request.references.empty() && config.bake_modulations < 0))
            return fail("ref2va requires at least one image, video or audio reference");
        int images = 0, videos = 0, audios = 0;
        for (const auto& reference : request.references)
        {
            images += reference.kind == h3::ReferenceKind::Image;
            videos += reference.kind == h3::ReferenceKind::Video;
            audios += reference.kind == h3::ReferenceKind::Audio;
        }
        if (images > 9 || videos > 3 || audios > 3)
            return fail("ref2va accepts up to 9 images, 3 videos and 3 audio files");
    }
    else if (workflow != "t2va")
        return fail("unknown workflow " + workflow);
    else if (!images.empty() || !request.references.empty() || !request.last_image.empty())
        return fail("use -t fl2va or -t ref2va to pass an image");

    if (lora_scales.size() > request.loras.size())
        return fail("too many --lora-scale values");
    for (size_t i = 0; i < lora_scales.size(); i++)
        request.loras[i].strength = lora_scales[i];

    if (request.segments.empty() && (prompts.size() > 1 || frame_counts.size() > 1))
        return fail("repeat -p and -f only for FL2VA with intermediate keyframes");
    if ((config.bake_modulations < 0 && prompts.empty()) || config.model.empty() || request.output.empty())
        return fail("prompt, model directory and output path must be nonempty");
    for (const auto& prompt : prompts)
        if (prompt.find_first_not_of(" \t\r\n") == std::string::npos)
            return fail("prompt must be nonempty");
    if (!prompts.empty())
        request.prompt = std::move(prompts.front());
    if (!frame_counts.empty())
        request.frames = frame_counts.front();
    if ((request.width == 0) != (request.height == 0) || request.width < 0 || request.height < 0 || request.width > 4096
        || request.height > 4096 || request.width % 32 || request.height % 32)
        return fail("width and height must both be positive multiples of 32 up to 4096, or omitted");
    for (const auto& reference : request.references)
        if (reference.path.empty())
            return fail("reference path must be nonempty");
    if (request.frames < 1 || request.steps < 2 || request.steps > 1000
        || request.reference_short_edge < 32 || request.reference_short_edge > 4096
        || request.reference_video_short_edge < 32 || request.reference_video_short_edge > 768 || config.gpu < -1)
        return fail("invalid frame count, step count, reference short edge or GPU index");
    if (config.rough_mode < 0 || config.rough_mode > 3 || config.kv_cache_mib <= 0)
        return fail("invalid rough mode or K/V cache ceiling");
    if (config.bake_modulations >= 0 && (config.rough_mode || !images.empty()
            || !request.references.empty() || !request.last_image.empty()))
        return fail("modulation baking takes a workflow and condition type without media inputs");
    return true;
}
}

int main(int argc, char** argv)
{
    if (argc == 1 || (argc == 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")))
    {
        print_usage();
        return argc == 1 ? 1 : 0;
    }
    h3::RuntimeConfig config;
    h3::GenerateRequest request;
    std::string error;
    if (!parse(error, argc, argv, config, request) || !h3::generate(error, config, request))
    {
        std::fprintf(stderr, "%s\n", error.empty() ? "generation failed" : error.c_str());
        return 1;
    }
    if (config.bake_modulations < 0 && !config.estimate_memory)
        std::printf("%s\n", request.output.c_str());
    return 0;
}
