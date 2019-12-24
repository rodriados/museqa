/** 
 * Multiple Sequence Alignment sequence encoder header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#include <string>
#include <cstdint>

#include <utils.hpp>
#include <buffer.hpp>
#include <format.hpp>

namespace msa
{
    namespace encoder
    {
        /**
         * The encoder character or unit type.
         * @since 0.1.1
         */
        using unit = uint8_t;

        /**
         * The encoder sequence block type.
         * @since 0.1.1
         */
        using block = uint16_t;

        /**
         * Aliases a block buffer into a more readable name.
         * @since 0.1.1
         */
        using buffer = msa::buffer<block>;

        /**
         * Defining the unit symbol for final of sequence.
         * @since 0.1.1
         */
        enum : unit { end = 0x18 };

        /**
         * Indicates the number of units held by each block.
         * @since 0.1.1
         */
        enum : size_t { block_size = 8 * sizeof(block) / 5 };

        /**
         * Accesses a specific offset within a block.
         * @param tgt The target block.
         * @param offset The requested offset.
         * @return The stored value in given offset.
         */
        __host__ __device__ inline unit access(block tgt, uint8_t offset) noexcept
        {
            static constexpr uint8_t shift[] = {1, 6, 11, 17, 22, 27};
            return (tgt >> shift[offset]) & 0x1F;
        }

        /**
         * Accesses a specific offset on a block buffer.
         * @param tgt The target buffer to search on.
         * @param offset The requested offset.
         * @return The stored value in given offset.
         */
        __host__ __device__ inline unit access(const buffer& tgt, ptrdiff_t offset)
        {
            return access(tgt[offset / block_size], offset % block_size);
        }

        extern unit encode(char) noexcept;
        extern buffer encode(const char *, size_t);

        extern char decode(unit);
        extern std::string decode(const buffer&);
    };

    namespace fmt
    {
        /**
         * Formats an encoded block buffer to be printed.
         * @since 0.1.1
         */
        template <>
        struct formatter<encoder::buffer> : public adapter<std::string>
        {
            auto parse(const encoder::buffer&) -> return_type;
        };
    }
}
