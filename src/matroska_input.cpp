// SPDX-License-Identifier: Apache-2.0
#include "matroska_input.h"

#include <cstring>

namespace h3
{
bool MatroskaInput::open(std::string& error, const std::string& path)
{
    long long position = 0;
    mkvparser::EBMLHeader header;
    if (reader.size < 0 || header.Parse(&reader, position) != 0 || !header.m_docType
        || (std::strcmp(header.m_docType, "webm") && std::strcmp(header.m_docType, "matroska")))
        return media_error(error, "cannot open WebM or Matroska reference: " + path);

    mkvparser::Segment* opened = nullptr;
    if (mkvparser::Segment::CreateInstance(&reader, position, opened) != 0 || !opened)
        return media_error(error, "cannot read media segment: " + path);
    segment.reset(opened);
    if (segment->Load() != 0 || !segment->GetTracks())
        return media_error(error, "cannot read media tracks: " + path);
    return true;
}

const mkvparser::Track* MatroskaInput::track(long type) const
{
    const auto* tracks = segment->GetTracks();
    for (unsigned long i = 0; i < tracks->GetTracksCount(); i++)
    {
        const auto* candidate = tracks->GetTrackByIndex(i);
        if (candidate && candidate->GetType() == type)
            return candidate;
    }
    return nullptr;
}
}
