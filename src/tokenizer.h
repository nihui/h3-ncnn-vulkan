// SPDX-License-Identifier: Apache-2.0
// Byte-level BPE tokenizer adapted from h3image-ncnn-vulkan and qwenimage-ncnn-vulkan.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace h3
{
class BpeTokenizer
{
public:
    static BpeTokenizer LoadFromModel(std::string& error, const std::string& model);
    int token_id(const std::string& token) const;
    std::vector<int> encode(std::string& error, const std::string& text) const;

private:
    static bool IsAsciiSpace(unsigned char c);
    static bool IsUnicodeSpace(uint32_t cp);
    static bool NextUtf8(const std::string& s, size_t& i, uint32_t& cp, size_t& cp_len);
    static std::vector<std::string> Utf8Chars(const std::string& s);
    static std::vector<std::string> PretokenizeByteLevel(const std::string& text);
    static std::string PairKey(const std::string& a, const std::string& b);

    void InitByteMaps();
    std::string ByteEncode(const std::string& text) const;
    std::vector<std::string> BpeForPiece(const std::string& piece) const;

    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<std::string, int> merges_rank_;
    std::vector<std::pair<std::string, int>> additional_tokens_;
    std::array<uint32_t, 256> byte_encoder_ {};
};
}
