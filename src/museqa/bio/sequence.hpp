/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file The representation of a biological sequence and surronding functions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2023-present Rodrigo Siqueira
 */
#pragma once

#include <string>
#include <cstdint>

#include <museqa/environment.h>
#include <museqa/utility.hpp>
#include <museqa/bio/alphabet.hpp>
#include <museqa/memory/buffer.hpp>

#include <museqa/thirdparty/fmtlib.h>

MUSEQA_PUSH_GCC_OPTION_BEGIN(optimize ("unroll-loops"))

MUSEQA_BEGIN_NAMESPACE

namespace bio::sequence
{
    /**
     * The underlying type of a block of a biological sequence, which stores one
     * or more compressed symbols within it.
     * @since 1.0
     */
    using block_t = uint16_t;

    /**
     * Indicates the total number of bits that a sequence block can possibly store.
     * This metric influences the number of symbols to store in a block.
     * @since 1.0
     */
    inline constexpr size_t block_bits = 8 * sizeof(block_t);

    /**
     * Sets the number of sequence symbols to be compressed and stored within a
     * single sequence block.
     * @since 1.0
     */
    inline constexpr size_t symbols_by_block = block_bits / alphabet::symbol_bits;

    /**
     * Compressed buffer for a biological sequence. The symbols of a sequence are
     * compressed and stored within memory blocks that easily accessible and decompressible.
     * @since 1.0
     */
    class buffer_t : protected memory::buffer_t<block_t>
    {
        static_assert(symbols_by_block > 0, "at least 1 symbol must fit within a block");

        private:
            typedef memory::buffer_t<block_t> underlying_t;

        protected:
            size_t m_length = 0;

        public:
            __host__ __device__ inline buffer_t() noexcept = default;
            __host__ __device__ inline buffer_t(const buffer_t&) __devicesafe__ = default;
            __host__ __device__ inline buffer_t(buffer_t&&) __devicesafe__ = default;

            __host__ __device__ inline buffer_t& operator=(const buffer_t&) __devicesafe__ = default;
            __host__ __device__ inline buffer_t& operator=(buffer_t&&) __devicesafe__ = default;

            /**
             * Informs the number of symbols within the current sequence buffer.
             * @return The length of the sequence.
             */
            __host__ __device__ inline size_t length() const noexcept
            {
                return m_length;
            }

        protected:
            /**
             * Initializes a new sequence buffer from an existing buffer.
             * @param buffer The buffer to initialize sequence buffer from.
             * @param length The number of symbols contained by the sequence.
             */
            __host__ __device__ inline buffer_t(const underlying_t& buffer, size_t length) __devicesafe__
              : underlying_t (buffer)
              , m_length (length)
            {}

        friend sequence::buffer_t encode(const char *, size_t);
        friend sequence::buffer_t encode(const std::string&);
        friend std::string decode(const sequence::buffer_t&);
    };

    /**
     * Encodes a biological sequence from its character representation into the
     * internal compressed sequence buffer format.
     * @param buffer The symbol characters of the sequence to be encoded.
     * @param length The number of symbols within the target sequence.
     * @return The compressed sequence buffer.
     */
    inline sequence::buffer_t encode(const char *buffer, size_t length)
    {
        const bool has_padding = length % symbols_by_block > 0;
        const auto block_count = has_padding + length / symbols_by_block;

        auto encoded = factory::memory::buffer<block_t>(block_count);

        for (size_t i = 0, n = 0; i < block_count; ++i) {
            auto current_block = block_t (0);

            for (size_t j = 0; j < symbols_by_block; ++j, ++n) {
                const auto symbol = n < length
                    ? alphabet::encode(buffer[n])
                    : alphabet::end;
                current_block |= symbol << (alphabet::symbol_bits * j);
            }

            encoded[i] = current_block;
        }

        return sequence::buffer_t (encoded, length);
    }

    /**
     * Encodes a biological sequence from its string representation into the internal
     * compressed sequence buffer format.
     * @param buffer The sequence string to be encoded.
     * @return The compressed sequence buffer.
     */
    inline sequence::buffer_t encode(const std::string& buffer)
    {
        return sequence::encode(
            buffer.data()
          , buffer.size()
        );
    }

    /**
     * Decodes a biological sequence from its internal compressed buffer representation
     * to its string representation.
     * @param buffer The sequence to be decoded.
     * @return The sequence's decoded string.
     */
    inline std::string decode(const sequence::buffer_t& buffer)
    {
        constexpr auto end = alphabet::decode(alphabet::end);
        constexpr auto mask = ~(~0u << alphabet::symbol_bits);

        const size_t length = buffer.length();
        auto decoded = std::string(length, end);

        for (size_t i = 0, n = 0; n < length; ++i) {
            auto current_block = buffer[i];

            for (size_t j = 0; j < symbols_by_block; ++j, ++n) {
                if (n >= length) break;
                decoded[n] = alphabet::decode(current_block & mask);
                current_block >>= alphabet::symbol_bits;
            }
        }

        return decoded;
    }
}

namespace bio
{
    /**
     * The buffer type for a biological sequence. Internally, sequence symbols are
     * stored compressed within blocks that are easily accessible and decompressible.
     * @since 1.0
     */
    using sequence_t = sequence::buffer_t;
}

MUSEQA_END_NAMESPACE

MUSEQA_PUSH_GCC_OPTION_END(optimize ("unroll-loops"))

#if !defined(MUSEQA_AVOID_FMTLIB)

/**
 * Implements a string formatter for a biological sequence buffer.
 * @since 1.0
 */
template <>
struct fmt::formatter<MUSEQA_NAMESPACE::bio::sequence_t>
{
    typedef MUSEQA_NAMESPACE::bio::sequence_t target_t;

    /**
     * Evaluates the formatter's parsing context.
     * @tparam C The parsing context type.
     * @param ctx The parsing context instance.
     * @return The processed and evaluated parsing context.
     */
    template <typename C>
    constexpr auto parse(C& ctx) const -> decltype(ctx.begin())
    {
        return ctx.begin();
    }

    /**
     * Formats the sequence buffer into a printable string.
     * @tparam F The formatting context type.
     * @param buffer The sequence buffer to be formatted into a string.
     * @param ctx The formatting context instance.
     * @return The formatting context instance.
     */
    template <typename F>
    auto format(const target_t& buffer, F& ctx) const -> decltype(ctx.out())
    {
        return fmt::format_to(
            ctx.out(), "{}"
          , MUSEQA_NAMESPACE::bio::sequence::decode(buffer)
        );
    }
};

#endif
