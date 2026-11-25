// SPDX-License-Identifier: Apache-2.0
#include "snake.h"

#include <command.h>
#include <gpu.h>
#include <layer.h>
#include <modelbin.h>
#include <net.h>
#include <pipeline.h>

#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace h3
{
namespace
{
std::vector<uint32_t> compile_snake_shader(bool pack4)
{
    const std::string source = std::string("#version 450\n#define sample_type ") + (pack4 ? "vec4\n" : "float\n")
        + R"glsl(
layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
layout(binding = 0) buffer Samples { sample_type samples[]; };
layout(binding = 1) readonly buffer Coefficients { sample_type coefficients[]; };
layout(push_constant) uniform parameter
{
    int w;
    int h;
    int c;
    int cstep;
} p;
void main()
{
    const int x = int(gl_GlobalInvocationID.x);
    const int y = int(gl_GlobalInvocationID.y);
    const int c = int(gl_GlobalInvocationID.z);
    if (x >= p.w || y >= p.h || c >= p.c)
        return;

    const int i = c * p.cstep + y * p.w + x;
    const sample_type value = samples[i];
    precise sample_type phase = value * coefficients[c];
    const sample_type wave = sin(phase);
    precise sample_type square = wave * wave;
    precise sample_type periodic = square * coefficients[p.c + c];
    precise sample_type result = value + periodic;
    samples[i] = result;
}
)glsl";
    ncnn::Option opt;
    opt.use_fp16_packed = opt.use_fp16_storage = opt.use_fp16_arithmetic = false;
    opt.use_bf16_packed = opt.use_bf16_storage = false;
    std::vector<uint32_t> spirv;
    if (ncnn::compile_spirv_module(source.c_str(), opt, spirv))
        return {};
    return spirv;
}

std::unique_ptr<ncnn::Pipeline> create_snake_pipeline(const ncnn::VulkanDevice* device, bool pack4)
{
    static const auto scalar = compile_snake_shader(false);
    static const auto packed = compile_snake_shader(true);
    const auto& spirv = pack4 ? packed : scalar;
    if (spirv.empty())
        return {};

    auto pipeline = std::make_unique<ncnn::Pipeline>(device);
    pipeline->set_local_size_xyz(128, 1, 1);
    if (pipeline->create(spirv.data(), spirv.size() * sizeof(uint32_t), {})
        || pipeline->shader_info().push_constant_count != 4 || pipeline->shader_info().binding_count != 2)
        return {};
    return pipeline;
}

class Snake : public ncnn::Layer
{
public:
    Snake()
    {
        one_blob_only = true;
        support_inplace = true;
        support_vulkan = true;
        support_vulkan_packing = true;
    }

    int load_param(const ncnn::ParamDict& params) override
    {
        channels_ = params.get(0, 0);
        return channels_ > 0 && channels_ <= std::numeric_limits<int>::max() / 2 ? 0 : -1;
    }

    int load_model(const ncnn::ModelBin& weights) override
    {
        // The exporter stores the frequency and amplitude vectors in FP32.
        coefficients_ = weights.load(2 * channels_, 1);
        return coefficients_.empty() ? -100 : 0;
    }

    int create_pipeline(const ncnn::Option&) override
    {
        pack1_ = create_snake_pipeline(vkdev, false);
        if (!pack1_)
            return -1;
        if (channels_ % 4 == 0)
        {
            pack4_ = create_snake_pipeline(vkdev, true);
            if (!pack4_)
                return -1;
        }
        return 0;
    }

    int destroy_pipeline(const ncnn::Option&) override
    {
        pack4_.reset();
        pack1_.reset();
        return 0;
    }

    int upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt) override
    {
        cmd.record_upload(coefficients_, coefficients_gpu_, opt);
        if (coefficients_gpu_.empty())
            return -100;
        if (coefficients_gpu_.elembits() != 32)
            return -1;
        if (opt.lightmode)
            coefficients_.release();
        return 0;
    }

    int forward_inplace(ncnn::VkMat& audio, ncnn::VkCompute& cmd, const ncnn::Option&) const override
    {
        if (audio.dims != 3 || audio.elembits() != 32 || audio.c * audio.elempack != channels_
            || (audio.elempack != 1 && audio.elempack != 4))
            return -1;

        const ncnn::Pipeline* pipeline = audio.elempack == 4 ? pack4_.get() : pack1_.get();
        if (!pipeline)
            return -1;

        const std::vector<ncnn::VkMat> bindings = { audio, coefficients_gpu_ };
        std::vector<ncnn::vk_constant_type> constants(4);
        constants[0].i = audio.w;
        constants[1].i = audio.h;
        constants[2].i = audio.c;
        constants[3].i = static_cast<int>(audio.cstep);
        cmd.record_pipeline(pipeline, bindings, constants, audio);
        return 0;
    }

private:
    int channels_ = 0;
    ncnn::Mat coefficients_;
    ncnn::VkMat coefficients_gpu_;
    std::unique_ptr<ncnn::Pipeline> pack1_;
    std::unique_ptr<ncnn::Pipeline> pack4_;
};

ncnn::Layer* create_snake(void*)
{
    return new Snake;
}
}

bool register_snake_layer(ncnn::Net& net)
{
    return net.register_custom_layer("Snake", create_snake) == 0;
}
}
