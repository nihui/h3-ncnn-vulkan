// SPDX-License-Identifier: Apache-2.0
#include "safetensors.h"
#include "artifact_hash.h"

#include <mat.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <limits>
#include <sstream>
#include <string_view>
#include <utility>

namespace h3
{
namespace
{
class HeaderParser
{
public:
    HeaderParser(std::string& error, const std::string& header, const std::string& path)
        : p_(header.data())
        , end_(p_ + header.size())
        , error_(error)
        , path_(path)
    {
        if (header.compare(0, 3, "\xef\xbb\xbf") == 0)
            p_ += 3;
    }

    bool parse(std::map<std::string, SafeTensor>& tensors, std::map<std::string, std::string>& metadata,
        std::vector<std::pair<uint64_t, uint64_t>>& ranges, uint64_t data_offset, uint64_t payload_size)
    {
        if (!object([&](std::string name) {
                if (name == "__metadata__")
                    return object([&](std::string key) {
                        std::string value;
                        if (!string(value))
                            return false;
                        metadata.insert_or_assign(std::move(key), std::move(value));
                        return true;
                    });

                SafeTensor value;
                if (!tensor(name, payload_size, ranges, value))
                    return false;
                value.offset += data_offset;
                return tensors.emplace(std::move(name), std::move(value)).second || fail();
            }))
            return false;
        skip_space();
        return p_ == end_ || fail();
    }

private:
    bool fail(std::string message)
    {
        error_ = std::move(message);
        return false;
    }

    bool fail()
    {
        return fail("invalid safetensors JSON header in " + path_);
    }

    char peek() const { return p_ == end_ ? '\0' : *p_; }
    static bool digit(char c) { return c >= '0' && c <= '9'; }

    void skip_space()
    {
        while (peek() == ' ' || peek() == '\t' || peek() == '\r' || peek() == '\n')
            ++p_;
    }

    bool consume(char c)
    {
        skip_space();
        if (peek() != c)
            return false;
        ++p_;
        return true;
    }

    template<class Read>
    bool list(char begin, char end, const Read& read)
    {
        if (!consume(begin))
            return fail();
        if (consume(end))
            return true;
        do
        {
            if (!read())
                return false;
            if (consume(end))
                return true;
        } while (consume(','));
        return fail();
    }

    template<class Read>
    bool object(const Read& read)
    {
        return list('{', '}', [&] {
            std::string key;
            if (!string(key))
                return false;
            if (!consume(':'))
                return fail();
            return read(std::move(key));
        });
    }

    bool hex4(unsigned& value)
    {
        if (end_ - p_ < 4)
            return fail();
        value = 0;
        for (int i = 0; i < 4; i++)
        {
            const char c = *p_++;
            value <<= 4;
            if (digit(c))
                value |= c - '0';
            else if (c >= 'a' && c <= 'f')
                value |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F')
                value |= c - 'A' + 10;
            else
                return fail();
        }
        return true;
    }

    bool unicode(std::string& out)
    {
        unsigned value;
        if (!hex4(value))
            return false;
        if (value >= 0xd800 && value <= 0xdbff)
        {
            if (end_ - p_ < 2 || p_[0] != '\\' || p_[1] != 'u')
                return fail();
            p_ += 2;
            unsigned low;
            if (!hex4(low))
                return false;
            if (low < 0xdc00 || low > 0xdfff)
                return fail();
            value = 0x10000 + ((value - 0xd800) << 10) + (low - 0xdc00);
        }
        else if (value >= 0xdc00 && value <= 0xdfff)
            return fail();

        if (value <= 0x7f)
            out.push_back(static_cast<char>(value));
        else
        {
            constexpr unsigned prefixes[] = { 0, 0xc0, 0xe0, 0xf0 };
            const int tails = value <= 0x7ff ? 1 : value <= 0xffff ? 2 : 3;
            out.push_back(static_cast<char>(prefixes[tails] | (value >> (6 * tails))));
            for (int i = tails - 1; i >= 0; --i)
                out.push_back(static_cast<char>(0x80 | ((value >> (6 * i)) & 0x3f)));
        }
        return true;
    }

    bool utf8(std::string& out, unsigned lead)
    {
        if (lead < 0xc2 || lead > 0xf4)
            return fail();
        const auto start = p_ - 1;
        const int tails = lead < 0xe0 ? 1 : lead < 0xf0 ? 2 : 3;
        if (end_ - p_ < tails)
            return fail();
        const auto first = static_cast<unsigned char>(*p_);
        if ((lead == 0xe0 && first < 0xa0) || (lead == 0xed && first >= 0xa0)
            || (lead == 0xf0 && first < 0x90) || (lead == 0xf4 && first >= 0x90))
            return fail();
        for (int i = 0; i < tails; ++i)
            if ((static_cast<unsigned char>(*p_++) & 0xc0) != 0x80)
                return fail();
        out.append(start, p_);
        return true;
    }

    bool string(std::string& out)
    {
        if (!consume('"'))
            return fail();
        out.clear();
        while (p_ != end_)
        {
            const auto c = static_cast<unsigned char>(*p_++);
            if (c == '"')
                return true;
            if (c < 0x20)
                return fail();
            if (c >= 0x80)
            {
                if (!utf8(out, c))
                    return false;
            }
            else if (c != '\\')
                out.push_back(static_cast<char>(c));
            else
            {
                if (p_ == end_)
                    return fail();
                switch (*p_++)
                {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u':
                    if (!unicode(out))
                        return false;
                    break;
                default: return fail();
                }
            }
        }
        return fail();
    }

    bool integer(uint64_t& value)
    {
        skip_space();
        const auto parsed = std::from_chars(p_, end_, value);
        if (parsed.ec != std::errc() || (*p_ == '0' && parsed.ptr - p_ > 1))
            return fail();
        p_ = parsed.ptr;
        return true;
    }

    bool digits()
    {
        if (!digit(peek()))
            return fail();
        do
            ++p_;
        while (digit(peek()));
        return true;
    }

    bool skip_value(int depth = 0)
    {
        if (depth == 64)
            return fail();
        skip_space();
        switch (peek())
        {
        case '{': return object([&](const std::string&) { return skip_value(depth + 1); });
        case '[': return list('[', ']', [&] { return skip_value(depth + 1); });
        case '"':
        {
            std::string value;
            return string(value);
        }
        case 't':
        case 'f':
        case 'n':
        {
            const std::string_view literal = peek() == 't' ? "true" : peek() == 'f' ? "false" : "null";
            if (std::string_view(p_, end_ - p_).substr(0, literal.size()) != literal)
                return fail();
            p_ += literal.size();
            return true;
        }
        default:
            if (peek() == '-')
                ++p_;
            if (peek() == '0')
                ++p_;
            else if (!digits())
                return false;
            if (peek() == '.')
            {
                ++p_;
                if (!digits())
                    return false;
            }
            if (peek() == 'e' || peek() == 'E')
            {
                ++p_;
                if (peek() == '+' || peek() == '-')
                    ++p_;
                return digits();
            }
            return true;
        }
    }

    bool tensor(const std::string& name, uint64_t payload_size,
        std::vector<std::pair<uint64_t, uint64_t>>& ranges, SafeTensor& value)
    {
        std::string dtype;
        std::vector<uint64_t> offsets;
        bool has_shape = false;
        if (!object([&](const std::string& field) {
                if (field == "dtype")
                    return string(dtype);
                if (field == "shape")
                {
                    has_shape = true;
                    value.shape.clear();
                    return list('[', ']', [&] {
                        uint64_t dim;
                        if (!integer(dim))
                            return false;
                        if (dim == 0 || dim > std::numeric_limits<int>::max())
                            return fail("invalid safetensors shape for " + name);
                        value.shape.push_back(static_cast<int>(dim));
                        return true;
                    });
                }
                if (field == "data_offsets")
                {
                    offsets.clear();
                    return list('[', ']', [&] {
                        uint64_t offset;
                        if (!integer(offset))
                            return false;
                        offsets.push_back(offset);
                        return true;
                    });
                }
                return skip_value();
            }))
            return false;

        if (dtype == "F32")
            value.type = SafeTensor::F32;
        else if (dtype == "F16")
            value.type = SafeTensor::F16;
        else if (dtype == "BF16")
            value.type = SafeTensor::BF16;
        else
            return fail("unsupported safetensors dtype " + dtype + " in " + name);

        if (!has_shape)
            return fail();
        value.elements = 1;
        for (int dim : value.shape)
        {
            if (value.elements > payload_size / dim)
                return fail("invalid safetensors shape for " + name);
            value.elements *= dim;
        }
        const uint64_t stride = value.type == SafeTensor::F32 ? 4 : 2;
        if (offsets.size() != 2 || offsets[0] > offsets[1] || offsets[1] > payload_size
            || (offsets[1] - offsets[0]) % stride || (offsets[1] - offsets[0]) / stride != value.elements)
            return fail("invalid safetensors offsets for " + name);
        value.offset = offsets[0];
        ranges.emplace_back(offsets[0], offsets[1]);
        return true;
    }

    const char* p_;
    const char* end_;
    std::string& error_;
    const std::string& path_;
};
}

SafeTensors::SafeTensors(std::string& error, const std::string& path)
    : path_(path)
    , stream_(path, std::ios::binary | std::ios::ate)
{
    const auto size = stream_.tellg();
    std::array<unsigned char, 8> prefix {};
    stream_.seekg(0);
    if (!stream_.read(reinterpret_cast<char*>(prefix.data()), prefix.size()))
    {
        error = "cannot read safetensors file " + path;
        return;
    }
    uint64_t header_size = 0;
    for (size_t i = 0; i < prefix.size(); i++)
        header_size |= uint64_t(prefix[i]) << (i * 8);
    if (header_size > 100000000 || header_size + prefix.size() > uint64_t(size))
    {
        error = "invalid safetensors header in " + path;
        return;
    }
    const uint64_t data_offset = prefix.size() + header_size;
    std::string header(header_size, '\0');
    if (!stream_.read(header.data(), header.size()))
    {
        error = "truncated safetensors header in " + path;
        return;
    }

    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    if (!HeaderParser(error, header, path).parse(tensors, metadata, ranges, data_offset, uint64_t(size) - data_offset))
        return;
    std::sort(ranges.begin(), ranges.end());
    uint64_t expected = 0;
    for (const auto& range : ranges)
    {
        if (range.first != expected)
        {
            error = "invalid safetensors tensor layout in " + path;
            return;
        }
        expected = range.second;
    }
    if (data_offset + expected != uint64_t(size) || tensors.empty())
        error = "invalid safetensors payload in " + path;
}

const SafeTensor* SafeTensors::tensor(std::string& error, const std::string& name, const std::vector<int>& shape) const
{
    const auto it = tensors.find(name);
    if (it != tensors.end() && it->second.shape == shape)
        return &it->second;
    error = "missing or incompatible tensor " + name + " in " + path_;
    return nullptr;
}

bool SafeTensors::read(const SafeTensor& t, uint64_t offset, size_t count, float* values) const
{
    if (offset > t.elements || count > t.elements - offset)
        return false;
    const size_t stride = t.type == SafeTensor::F32 ? 4 : 2;
    stream_.seekg(t.offset + offset * stride);
    if (stride == 4)
        return bool(stream_.read(reinterpret_cast<char*>(values), count * stride));

    std::array<uint16_t, 4096> packed;
    for (size_t i = 0; i < count;)
    {
        const size_t n = std::min(packed.size(), count - i);
        if (!stream_.read(reinterpret_cast<char*>(packed.data()), n * stride))
            return false;
        for (size_t j = 0; j < n; j++)
            values[i + j] = t.type == SafeTensor::F16 ? ncnn::float16_to_float32(packed[j])
                                                      : ncnn::bfloat16_to_float32(packed[j]);
        i += n;
    }
    return true;
}

bool SafeTensors::read_bf16(const SafeTensor& t, uint64_t offset, size_t count, uint16_t* values) const
{
    if (offset > t.elements || count > t.elements - offset)
        return false;
    if (t.type == SafeTensor::BF16)
    {
        stream_.seekg(t.offset + offset * sizeof(uint16_t));
        return bool(stream_.read(reinterpret_cast<char*>(values), count * sizeof(uint16_t)));
    }

    std::array<float, 4096> chunk;
    for (size_t i = 0; i < count;)
    {
        const size_t n = std::min(chunk.size(), count - i);
        if (!read(t, offset + i, n, chunk.data()))
            return false;
        for (size_t j = 0; j < n; j++)
            values[i + j] = ncnn::float32_to_bfloat16(chunk[j]);
        i += n;
    }
    return true;
}

std::string SafeTensors::content_key(std::string& error) const
{
    std::array<unsigned char, 65536> bytes;
    uint64_t hash = kHashStart;
    stream_.seekg(0);
    while (stream_.read(reinterpret_cast<char*>(bytes.data()), bytes.size()) || stream_.gcount())
        hash = hash_bytes(hash, bytes.data(), static_cast<size_t>(stream_.gcount()));
    if (!stream_.eof() || stream_.bad())
    {
        error = "cannot hash safetensors file " + path_;
        return {};
    }
    stream_.clear();
    std::ostringstream key;
    key << std::hex << hash;
    return key.str();
}
}
