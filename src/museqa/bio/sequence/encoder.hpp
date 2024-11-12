/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file The functions for encoding and decoding biological sequences.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2023-present Rodrigo Siqueira
 */
#pragma once

#include <string>
#include <cstdint>

#include <museqa/environment.h>
#include <museqa/bio/alphabet.hpp>
#include <museqa/bio/sequence/block.hpp>
#include <museqa/bio/sequence/buffer.hpp>
#include <museqa/memory/allocator.hpp>
#include <museqa/memory/pointer/shared.hpp>

#include <museqa/thirdparty/fmtlib.h>

MUSEQA_BEGIN_NAMESPACE

namespace bio::sequence
{
    /**
     * The number of total available bits by sequence block.
     * @since 1.0
     */
    MUSEQA_CONSTEXPR static size_t block_bits = 8 * sizeof(block_t);

    /**
     * The total number of symbols that can be compacted into a single sequence
     * block. This correlates directly to the amount of memory saved by each block
     * of compacted sequence symbols.
     * @since 1.0
     */
  #ifndef MUSEQA_AVOID_COMPACTED_SEQUENCES
    MUSEQA_CONSTEXPR static size_t symbols_by_block = block_bits / alphabet::symbol_bits;
  #else
    MUSEQA_CONSTEXPR static size_t symbols_by_block = 1;
  #endif

    static_assert(symbols_by_block > 0, "at least 1 symbol must fit within a sequence block");

    /**
     * Encodes a biological sequence from its human-readable representation into
     * the internal compressed sequence buffer format.
     * @param sequence The symbol characters of the sequence to be encoded.
     * @param length The number of symbols within the target sequence.
     * @param allocator The memory allocator to use for encoded sequence.
     * @return The compressed sequence buffer.
     */
    MUSEQA_INLINE buffer_t encode(
        const char *sequence
      , const size_t length
      , const memory::allocator_t& allocator = factory::memory::allocator<block_t>()
    ) {
        const bool has_padding = length % symbols_by_block > 0;
        const auto block_count = has_padding + length / symbols_by_block;

        auto encoded = factory::memory::pointer::shared<block_t>(block_count, allocator);

        for (size_t i = 0, n = 0; i < block_count; ++i) {
            block_t& current_block = encoded[i] = block_t(0);

            for (size_t j = 0; j < symbols_by_block; ++j, ++n) {
                const block_t symbol = n < length
                    ? alphabet::encode(sequence[n])
                    : alphabet::end;
                current_block |= symbol << (alphabet::symbol_bits * j);
            }
        }

        return buffer_t(encoded, length);
    }

    /**
     * Encodes a biological sequence from a human-readable string representation
     * into the internal compressed sequence buffer format.
     * @param sequence The sequence string to be encoded.
     * @param allocator The memory allocator to use for encoded sequence.
     * @return The compressed sequence buffer.
     */
    MUSEQA_INLINE buffer_t encode(
        const std::string& sequence
      , const memory::allocator_t& allocator = factory::memory::allocator<block_t>()
    ) {
        const auto end  = alphabet::decode(alphabet::end);
        const auto size = sequence.find_last_not_of(end) + 1;
        return encode(
            sequence.data()
          , size, allocator
        );
    }

    /**
     * Decodes a biological sequence from its internal compressed buffer representation
     * to its human-readable string representation.
     * @param sequence The sequence to be decoded.
     * @return The sequence's decoded string.
     */
    MUSEQA_INLINE std::string decode(const buffer_t& sequence)
    {
        const auto end = alphabet::decode(alphabet::end);
        const auto mask = ~(~0u << alphabet::symbol_bits);

        const size_t length = sequence.length();
        auto decoded = std::string(length, end);

        for (size_t i = 0, n = 0; n < length; ++i) {
            auto current_block = sequence[i];

            for (size_t j = 0; j < symbols_by_block && n < length; ++j, ++n) {
                decoded[n] = alphabet::decode(current_block & mask);
                current_block >>= alphabet::symbol_bits;
            }
        }

        return decoded;
    }
}

MUSEQA_END_NAMESPACE

#ifndef MUSEQA_AVOID_FMTLIB

/**
 * Implements a string formatter for a biological sequence buffer.
 * @since 1.0
 */
template <>
struct fmt::formatter<MUSEQA_NAMESPACE::bio::sequence::buffer_t>
{
    typedef MUSEQA_NAMESPACE::bio::sequence::buffer_t target_t;

    /**
     * Evaluates the formatter's parsing context.
     * @tparam C The parsing context type.
     * @param ctx The parsing context instance.
     * @return The processed and evaluated parsing context.
     */
    template <typename C>
    MUSEQA_CONSTEXPR auto parse(C& ctx) const -> decltype(ctx.begin())
    {
        return ctx.begin();
    }

    /**
     * Formats the sequence buffer into a printable string.
     * @tparam F The formatting context type.
     * @param sequence The sequence buffer to be formatted into a string.
     * @param ctx The formatting context instance.
     * @return The formatting context instance.
     */
    template <typename F>
    auto format(const target_t& sequence, F& ctx) const -> decltype(ctx.out())
    {
        return fmt::format_to(
            ctx.out(), "{}"
          , MUSEQA_NAMESPACE::bio::sequence::decode(sequence)
        );
    }
};

#endif
