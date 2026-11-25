// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "media_codec.h"

#include <mkvparser/mkvparser.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

namespace h3
{
struct MediaInputFile : mkvparser::IMkvReader
{
    explicit MediaInputFile(const std::string& path)
        : stream(std::filesystem::u8path(path), std::ios::binary | std::ios::ate)
        , size(stream.tellg())
    {
    }

    int Read(long long pos, long length, unsigned char* destination) override
    {
        if (pos < 0 || length < 0 || pos > size || length > size - pos)
            return -1;
        stream.seekg(pos);
        return stream.read(reinterpret_cast<char*>(destination), length) ? 0 : -1;
    }

    int Length(long long* total, long long* available) override
    {
        if (total)
            *total = size;
        if (available)
            *available = size;
        return size < 0 ? -1 : 0;
    }

    std::ifstream stream;
    const long long size;
};

struct MatroskaInput
{
    explicit MatroskaInput(const std::string& path)
        : reader(path)
    {
    }

    bool open(std::string& error, const std::string& path);
    const mkvparser::Track* track(long type) const;

    template <class Consume> bool read(std::string& error, const mkvparser::Track& track, Consume consume)
    {
        std::vector<uint8_t> packet;
        const mkvparser::BlockEntry* entry = nullptr;
        long status = track.GetFirst(entry);
        while (status == 0 && entry && !entry->EOS())
        {
            const auto* block = entry->GetBlock();
            if (!block)
                return media_error(error, "missing media block");
            for (int i = 0; i < block->GetFrameCount(); i++)
            {
                const auto& frame = block->GetFrame(i);
                constexpr long kMaxPacket = 64 * 1024 * 1024;
                if (frame.len <= 0 || frame.len > kMaxPacket)
                    return media_error(error, "invalid media packet size");
                packet.resize((size_t)frame.len);
                if (frame.Read(&reader, packet.data()) != 0)
                    return media_error(error, "cannot read media packet");
                if (!consume(packet, *block, entry->GetCluster(), i))
                    return error.empty();
            }
            const mkvparser::BlockEntry* next = nullptr;
            status = track.GetNext(entry, next);
            entry = next;
        }
        if (status < 0)
            return media_error(error, "cannot read media blocks");
        return true;
    }

    MediaInputFile reader;
    std::unique_ptr<mkvparser::Segment> segment;
};
}
