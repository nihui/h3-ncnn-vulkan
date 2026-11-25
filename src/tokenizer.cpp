// SPDX-License-Identifier: Apache-2.0
// Byte-level BPE tokenizer adapted from h3image-ncnn-vulkan and qwenimage-ncnn-vulkan.

#include "tokenizer.h"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <limits>

namespace h3
{
bool BpeTokenizer::IsAsciiSpace(unsigned char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

bool BpeTokenizer::IsUnicodeSpace(uint32_t cp)
{
    if (cp <= 0x7F)
        return IsAsciiSpace(static_cast<unsigned char>(cp));
    switch (cp)
    {
    case 0x00A0:
    case 0x1680:
    case 0x2000:
    case 0x2001:
    case 0x2002:
    case 0x2003:
    case 0x2004:
    case 0x2005:
    case 0x2006:
    case 0x2007:
    case 0x2008:
    case 0x2009:
    case 0x200A:
    case 0x2028:
    case 0x2029:
    case 0x202F:
    case 0x205F:
    case 0x3000:
        return true;
    default:
        return false;
    }
}

bool BpeTokenizer::NextUtf8(const std::string& s, size_t& i, uint32_t& cp, size_t& cp_len)
{
    if (i >= s.size())
        return false;
    unsigned char c0 = static_cast<unsigned char>(s[i]);
    if (c0 < 0x80)
    {
        cp = c0;
        cp_len = 1;
        ++i;
        return true;
    }
    else if ((c0 >> 5) == 0x6)
    {
        if (i + 1 >= s.size())
            return false;
        unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
        if ((c1 & 0xC0) != 0x80)
            return false;
        cp = ((c0 & 0x1F) << 6) | (c1 & 0x3F);
        cp_len = 2;
        i += 2;
        return true;
    }
    else if ((c0 >> 4) == 0xE)
    {
        if (i + 2 >= s.size())
            return false;
        unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
        unsigned char c2 = static_cast<unsigned char>(s[i + 2]);
        if (((c1 & 0xC0) != 0x80) || ((c2 & 0xC0) != 0x80))
            return false;
        cp = ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
        cp_len = 3;
        i += 3;
        return true;
    }
    else if ((c0 >> 3) == 0x1E)
    {
        if (i + 3 >= s.size())
            return false;
        unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
        unsigned char c2 = static_cast<unsigned char>(s[i + 2]);
        unsigned char c3 = static_cast<unsigned char>(s[i + 3]);
        if (((c1 & 0xC0) != 0x80) || ((c2 & 0xC0) != 0x80) || ((c3 & 0xC0) != 0x80))
            return false;
        cp = ((c0 & 0x07) << 18) | ((c1 & 0x3F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);
        cp_len = 4;
        i += 4;
        return true;
    }
    return false;
}

std::vector<std::string> BpeTokenizer::Utf8Chars(const std::string& s)
{
    std::vector<std::string> chars;
    chars.reserve(s.size());
    size_t i = 0;
    while (i < s.size())
    {
        size_t start = i;
        uint32_t cp;
        size_t cp_len;
        if (!NextUtf8(s, i, cp, cp_len))
            break;
        chars.emplace_back(s.substr(start, i - start));
    }
    return chars;
}

void BpeTokenizer::InitByteMaps()
{
    // Logic based on GPT-2 byte encoder (map printable chars to themselves, others to 256+)
    auto is_printable = [](int b)
    {
        return (b >= '!' && b <= '~') // '!' to '~'
            || (b >= 161 && b <= 172) // '¡' to '¬'
            || (b >= 174 && b <= 255); // '®' to 'ÿ'
    };

    int n = 0;
    for (int b = 0; b < 256; ++b)
    {
        if (is_printable(b))
        {
            byte_encoder_[b] = static_cast<uint32_t>(b);
        }
        else
        {
            byte_encoder_[b] = static_cast<uint32_t>(256 + n);
            n++;
        }
    }
}

std::string BpeTokenizer::ByteEncode(const std::string& text) const
{
    std::string out;
    out.reserve(text.size() * 2);

    for (unsigned char b : text)
    {
        uint32_t cp = byte_encoder_[b];
        // Append cp as UTF-8
        if (cp < 0x80)
        {
            out += static_cast<char>(cp);
        }
        else if (cp < 0x800)
        {
            out += static_cast<char>((cp >> 6) | 0xC0);
            out += static_cast<char>((cp & 0x3F) | 0x80);
        }
        else if (cp < 0x10000)
        {
            out += static_cast<char>((cp >> 12) | 0xE0);
            out += static_cast<char>(((cp >> 6) & 0x3F) | 0x80);
            out += static_cast<char>((cp & 0x3F) | 0x80);
        }
        else
        {
            out += static_cast<char>((cp >> 18) | 0xF0);
            out += static_cast<char>(((cp >> 12) & 0x3F) | 0x80);
            out += static_cast<char>(((cp >> 6) & 0x3F) | 0x80);
            out += static_cast<char>((cp & 0x3F) | 0x80);
        }
    }
    return out;
}

std::vector<std::string> BpeTokenizer::PretokenizeByteLevel(const std::string& text)
{
    auto is_letter = [](uint32_t cp)
    {
        return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z') || (cp >= 0x00c0 && cp <= 0x02ff)
            || (cp >= 0x0370 && cp <= 0x052f) || (cp >= 0x1e00 && cp <= 0x1eff) || (cp >= 0x3040 && cp <= 0x30ff)
            || (cp >= 0x3400 && cp <= 0x4dbf) || (cp >= 0x4e00 && cp <= 0x9fff) || (cp >= 0xa000 && cp <= 0xa4cf)
            || (cp >= 0xac00 && cp <= 0xd7af) || (cp >= 0xf900 && cp <= 0xfaff) || (cp >= 0x10000 && cp <= 0x1efff);
    };
    auto is_number = [](uint32_t cp)
    {
        return (cp >= '0' && cp <= '9') || (cp >= 0x0660 && cp <= 0x0669) || (cp >= 0x06f0 && cp <= 0x06f9)
            || (cp >= 0x0966 && cp <= 0x096f) || (cp >= 0xff10 && cp <= 0xff19);
    };
    auto is_newline = [](uint32_t cp) { return cp == '\r' || cp == '\n'; };
    auto lower_ascii = [](char c) { return c >= 'A' && c <= 'Z' ? char(c - 'A' + 'a') : c; };
    auto next_cp = [&](size_t p, uint32_t& cp, size_t& end)
    {
        size_t q = p;
        if (!NextUtf8(text, q, cp, end))
            return false;
        end = q;
        return true;
    };
    auto append_range = [&](std::vector<std::string>& out, size_t a, size_t b)
    {
        if (b > a)
            out.emplace_back(text.substr(a, b - a));
    };

    std::vector<std::string> out;
    size_t pos = 0;
    while (pos < text.size())
    {
        // GPT-2's contraction alternative has priority over the letter rule.
        bool contraction = false;
        if (text[pos] == '\'')
        {
            static const char* suffixes[] = { "'s", "'t", "'re", "'ve", "'m", "'ll", "'d" };
            for (const char* suffix : suffixes)
            {
                const size_t n = std::strlen(suffix);
                if (pos + n > text.size())
                    continue;
                bool match = true;
                for (size_t i = 0; i < n; i++)
                    if (lower_ascii(text[pos + i]) != lower_ascii(suffix[i]))
                        match = false;
                if (match)
                {
                    append_range(out, pos, pos + n);
                    pos += n;
                    contraction = true;
                    break;
                }
            }
        }
        if (contraction)
            continue;

        uint32_t cp = 0;
        size_t next = pos;
        if (!next_cp(pos, cp, next))
            break;

        // [^... ]?\p{L}+
        uint32_t qcp = cp;
        size_t qnext = next;
        if (!is_newline(cp) && !is_letter(cp) && !is_number(cp))
        {
            size_t after = qnext;
            uint32_t next_letter = 0;
            if (next_cp(qnext, next_letter, after) && is_letter(next_letter))
            {
                qcp = next_letter;
                qnext = after;
            }
        }
        if (is_letter(qcp))
        {
            size_t end = qnext;
            while (end < text.size())
            {
                uint32_t c = 0;
                size_t e = end;
                if (!next_cp(end, c, e) || !is_letter(c))
                    break;
                end = e;
            }
            append_range(out, pos, end);
            pos = end;
            continue;
        }

        // \p{N}
        if (is_number(cp))
        {
            append_range(out, pos, next);
            pos = next;
            continue;
        }

        // ?[^\s\p{L}\p{N}]+[\r\n]*
        size_t punct_begin = pos;
        size_t punct = pos;
        if (cp == ' ')
        {
            uint32_t after_space = 0;
            size_t after = next;
            if (next_cp(next, after_space, after) && !IsUnicodeSpace(after_space) && !is_letter(after_space)
                && !is_number(after_space))
                punct = next;
        }
        uint32_t pc = 0;
        size_t pe = punct;
        if (next_cp(punct, pc, pe) && !IsUnicodeSpace(pc) && !is_letter(pc) && !is_number(pc) && !is_newline(pc))
        {
            size_t end = pe;
            while (end < text.size())
            {
                uint32_t c = 0;
                size_t e = end;
                if (!next_cp(end, c, e) || IsUnicodeSpace(c) || is_letter(c) || is_number(c) || is_newline(c))
                    break;
                end = e;
            }
            while (end < text.size())
            {
                uint32_t c = 0;
                size_t e = end;
                if (!next_cp(end, c, e) || !is_newline(c))
                    break;
                end = e;
            }
            append_range(out, punct_begin, end);
            pos = end;
            continue;
        }

        // \s*[\r\n]+, or a whitespace run.
        if (IsUnicodeSpace(cp))
        {
            size_t end = pos;
            bool newline = false;
            while (end < text.size())
            {
                uint32_t c = 0;
                size_t e = end;
                if (!next_cp(end, c, e) || !IsUnicodeSpace(c))
                    break;
                newline = newline || is_newline(c);
                end = e;
            }
            append_range(out, pos, end);
            pos = end;
            continue;
        }

        append_range(out, pos, next);
        pos = next;
    }
    return out;
}

// ---------------- BPE core ----------------

std::string BpeTokenizer::PairKey(const std::string& a, const std::string& b)
{
    std::string key;
    key.reserve(a.size() + 1 + b.size());
    key.append(a);

    key.push_back('\t');
    key.append(b);

    return key;
}

std::vector<std::string> BpeTokenizer::BpeForPiece(const std::string& piece) const
{
    std::vector<std::string> symbols = Utf8Chars(piece);
    if (symbols.size() <= 1)
        return symbols;

    while (symbols.size() >= 2)
    {
        int best_rank = std::numeric_limits<int>::max();
        int best_i = -1;

        for (int i = 0; i + 1 < static_cast<int>(symbols.size()); ++i)
        {
            std::string key = PairKey(symbols[i], symbols[i + 1]);

            auto it = merges_rank_.find(key);
            if (it != merges_rank_.end())
            {
                int r = it->second;
                if (r < best_rank)
                {
                    best_rank = r;
                    best_i = i;
                }
            }
        }
        if (best_i < 0)
            break;

        symbols[best_i] += symbols[best_i + 1];
        symbols.erase(symbols.begin() + best_i + 1);
    }
    return symbols;
}

BpeTokenizer BpeTokenizer::LoadFromModel(std::string& error, const std::string& model)
{
    const std::string processor = model + "/processor/";
    const auto open = [&](const char* name) {
        std::ifstream file(processor + name, std::ios::binary);
        if (!file)
            error = "cannot open " + processor + name;
        return file;
    };
    const auto read_line = [](std::ifstream& file, std::string& line) {
        if (!std::getline(file, line))
            return false;
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        return true;
    };

    BpeTokenizer tokenizer;
    constexpr int vocabulary = 151936;
    tokenizer.token_to_id_.reserve(vocabulary);
    auto vocab = open("vocab.txt");
    if (!vocab)
        return {};
    std::string line;
    int id = 0;
    while (read_line(vocab, line))
    {
        // A vocabulary line's position is its embedding row.
        if (line.empty() || id >= vocabulary || !tokenizer.token_to_id_.emplace(line, id).second)
        {
            error = "invalid tokenizer token in " + processor + "vocab.txt";
            return {};
        }
        id++;
    }
    if (!vocab.eof() || id < 151669)
    {
        error = "incomplete tokenizer vocabulary in " + processor + "vocab.txt";
        return {};
    }

    auto merges = open("merges.txt");
    if (!merges)
        return {};
    tokenizer.merges_rank_.reserve(vocabulary);
    int rank = 0;
    while (read_line(merges, line))
    {
        if (rank == 0 && line.compare(0, 9, "#version:") == 0)
            continue;
        const size_t space = line.find(' ');
        if (space == std::string::npos || !space || space + 1 == line.size()
            || line.find(' ', space + 1) != std::string::npos || rank >= vocabulary)
        {
            error = "invalid tokenizer merge in " + processor + "merges.txt";
            return {};
        }
        const std::string first = line.substr(0, space), second = line.substr(space + 1);
        if (!tokenizer.token_to_id_.count(first) || !tokenizer.token_to_id_.count(second))
        {
            error = "unknown tokenizer merge token in " + processor + "merges.txt";
            return {};
        }
        tokenizer.merges_rank_.emplace(PairKey(first, second), rank++);
    }
    if (!merges.eof() || !rank)
    {
        error = "incomplete tokenizer merges in " + processor + "merges.txt";
        return {};
    }

    auto added = open("added_tokens.txt");
    if (!added)
        return {};
    while (read_line(added, line))
    {
        const auto token = tokenizer.token_to_id_.find(line);
        if (token == tokenizer.token_to_id_.end())
        {
            error = "unknown token in " + processor + "added_tokens.txt";
            return {};
        }
        tokenizer.additional_tokens_.emplace_back(token->first, token->second);
    }
    if (!added.eof() || tokenizer.additional_tokens_.empty())
    {
        error = "incomplete added tokens in " + processor + "added_tokens.txt";
        return {};
    }

    std::stable_sort(tokenizer.additional_tokens_.begin(), tokenizer.additional_tokens_.end(),
        [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
    tokenizer.InitByteMaps();
    return tokenizer;
}

int BpeTokenizer::token_id(const std::string& token) const
{
    const auto it = token_to_id_.find(token);
    return it == token_to_id_.end() ? -1 : it->second;
}

std::vector<int> BpeTokenizer::encode(std::string& error, const std::string& text) const
{
    std::vector<int> ids;
    ids.reserve(text.size() / 2 + 8);
    const auto append_text = [&](size_t start, size_t length)
    {
        for (const auto& piece : PretokenizeByteLevel(text.substr(start, length)))
            for (const auto& token : BpeForPiece(ByteEncode(piece)))
            {
                const int id = token_id(token);
                if (id < 0)
                {
                    error = "token missing from MiniMax-H3 vocabulary";
                    return false;
                }
                ids.push_back(id);
            }
        return true;
    };

    size_t start = 0;
    for (size_t i = 0; i < text.size();)
    {
        const auto match = std::find_if(additional_tokens_.begin(), additional_tokens_.end(),
            [&](const auto& item) { return text.compare(i, item.first.size(), item.first) == 0; });
        if (match == additional_tokens_.end())
        {
            i++;
            continue;
        }
        if (!append_text(start, i - start))
            return {};
        ids.push_back(match->second);
        i += match->first.size();
        start = i;
    }
    if (!append_text(start, text.size() - start))
        return {};
    return ids;
}
}
