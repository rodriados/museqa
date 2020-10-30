/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Biological sequence encoder and compresser.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#include <cctype>
#include <string>
#include <vector>

#include "buffer.hpp"
#include "encoder.hpp"
#include "exception.hpp"
#include "environment.h"

#if defined(__museqa_compiler_gcc)
  #pragma GCC push_options
  #pragma GCC optimize ("unroll-loops")
#endif

namespace museqa
{
    /**
     * The translate table used for creating new encoded units.
     * @since 0.1.1
     */
    static constexpr encoder::unit encode_table[] = {
        0x00, 0x14, 0x01, 0x06, 0x08, 0x0E, 0x03, 0x09, 0x0A, 0x15, 0x0C, 0x0B, 0x0D
    ,   0x05, 0x17, 0x0F, 0x07, 0x04, 0x10, 0x02, 0x17, 0x13, 0x11, 0x17, 0x12, 0x16
    };

    /**
     * The translate table used for decoding an encoded unit.
     * @since 0.1.1
     */
    static constexpr char decode_table[] = {
        'A', 'C', 'T', 'G', 'R', 'N', 'D', 'Q', 'E', 'H', 'I', 'L', 'K'
    ,   'M', 'F', 'P', 'S', 'W', 'Y', 'V', 'B', 'J', 'Z', 'X', '*', '-'
    };

    /**
     * Encodes a single sequence character to a unit.
     * @param value The character to encode.
     * @return The corresponding encoded unit.
     */
    encoder::unit encoder::encode(char value) noexcept
    {
        const char upper = toupper(value);
        return ('A' <= upper && upper <= 'Z') ? encode_table[upper - 'A'] : encoder::end;
    }

    /**
     * Decodes a single encoded unit into a character.
     * @param value The encoded unit to be decoded.
     * @return The corresponding decoded character.
     */
    char encoder::decode(encoder::unit value)
    {
        enforce(value < sizeof(decode_table), "cannot decode invalid unit: '%hhd'", value);
        return decode_table[value];
    }

    /**
     * Compresses a character string into a buffer of encoded blocks.
     * @param ptr The pointer to string to encode.
     * @param size The size of given string.
     * @return The buffer of enconded blocks.
     */
    encoder::buffer encoder::encode(const char *ptr, size_t size)
    {
        static constexpr uint8_t shift[] = {1, 6, 11, 17, 22, 27};
        std::vector<encoder::block> encoded;

        for(size_t i = 0, n = 0; n < size; ++i) {
            encoder::block current = 0x01;

            for(uint8_t j = 0; j < encoder::block_size; ++j, ++n) {
                const auto value = (n < size) ? encoder::encode(ptr[n]) : encoder::end;
                current |= value << shift[j];
            }

            encoded.push_back(current);
        }

        return encoder::buffer::copy(encoded);
    }

    /**
     * Decodes an encoded block buffer to a human-friendly string.
     * @param tgt The target buffer to decode.
     * @return The decoded string.
     */
    std::string encoder::decode(const encoder::buffer& tgt)
    {
        const size_t length = tgt.size() * encoder::block_size;
        std::string decoded (length, decode_table[encoder::end]);

        for(size_t i = 0, n = 0; n < length; ++i) {
            const encoder::block& block = tgt[i];

            for(uint8_t j = 0; j < encoder::block_size; ++j, ++n)
                decoded[n] = decode_table[access(block, j)];
        }

        return decoded;
    }

    /**
     * Formats an encoded block buffer to be printed.
     * @param tgt The target block buffer to be formatted.
     * @return The formatted buffer.
     */
    auto fmt::formatter<encoder::buffer>::parse(const encoder::buffer& tgt) -> return_type
    {
        return adapt(encoder::decode(tgt));
    }
}

#if defined(__museqa_compiler_gcc)
  #pragma GCC pop_options
#endif
