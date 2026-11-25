// SPDX-License-Identifier: Apache-2.0
#include "modulation_builder.h"
#include "artifact_hash.h"
#include "condition_tables.h"
#include "lora.h"

#include <net.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <ostream>

namespace h3
{
ncnn::Mat modulation_time_features(const std::vector<float>& times)
{
    ncnn::Mat features(256, (int)times.size());
    if (features.empty())
        return features;
    for (int j = 0; j < 128; j++)
    {
        const float frequency = std::exp(-std::log(10000.f) * j / 128);
        for (int i = 0; i < features.h; i++)
        {
            const float angle = times[i] * frequency;
            features.row(i)[j] = std::cos(angle);
            features.row(i)[128 + j] = std::sin(angle);
        }
    }
    return features;
}

bool build_modulations(std::string& error, H3ModelSet& models, bool reference,
    const std::vector<std::vector<float>>& schedule, const std::vector<ncnn::Mat>& features,
    const std::vector<const char*>& names, std::ostream& out)
{
    const auto* lora = models.lora();
    const bool pi = lora && lora->pi_steps() != 0;
    const auto* producer
        = pi ? nullptr : models.prepare(error, reference ? Component::ReferenceModulations : Component::Modulations);
    if (!pi && !producer)
        return false;
    for (size_t step = 0; step < features.size(); step++)
    {
        std::vector<ncnn::Mat> values;
        if (pi ? !lora->make_modulations(error, schedule[step], values)
               : !run_model(error, *producer, { { "time_features", features[step], true } }, names, values))
            return false;
        uint64_t hash = kHashStart;
        for (int i = 0; i < kModulationTables; i++)
        {
            auto& value = values[i];
            if (value.dims != 2 || value.w != table_width(i) || value.h != (int)schedule[step].size())
            {
                error = "unexpected BF16 modulation table shape";
                return false;
            }
            const int count = value.w * value.h;
            const auto* p = static_cast<const float*>(value.data);
            for (int j = 0; j < count; j++)
            {
                uint32_t bits = 0;
                std::memcpy(&bits, p + j, sizeof(bits));
                if ((bits & 0xffff) != 0 || !std::isfinite(p[j]))
                {
                    error = "producer did not return finite BF16 modulation values";
                    return false;
                }
            }
            ncnn::Mat bf16;
            if (producer)
                ncnn::cast_float32_to_bfloat16(value, bf16, producer->opt);
            else
                ncnn::cast_float32_to_bfloat16(value, bf16);
            if (bf16.empty())
            {
                error = "cannot allocate BF16 modulation table";
                return false;
            }
            hash = hash_bytes(hash, bf16.data, (size_t)count * sizeof(uint16_t));
            value = bf16;
        }
        out.write(reinterpret_cast<const char*>(&hash), sizeof(hash));
        for (const auto& value : values)
            out.write(static_cast<const char*>(value.data), (std::streamsize)value.w * value.h * sizeof(uint16_t));
        if (!out)
        {
            error = "cannot write modulation bank step";
            return false;
        }
    }
    return true;
}
}
