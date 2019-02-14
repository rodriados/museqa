/** 
 * Multiple Sequence Alignment sequence encoder header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef ENCODER_HPP_INCLUDED
#define ENCODER_HPP_INCLUDED

#include <string>
#include <ostream>

#include "utils.hpp"
#include "buffer.hpp"

namespace encoder
{
    /**
     * The encoded sequence block type.
     * @since 0.1.1
     */
    using EncodedBlock = uint16_t;

    /**
     * The final of sequence element.
     * @since 0.1.1
     */
    enum : uint8_t { end = 0x18 };

    /**
     * Indicates the number of elements held by each block.
     * @since 0.1.1
     */
    enum : int { batchSize = 8 * sizeof(EncodedBlock) / 5 };

    /**
     * Accesses a specific offset on a block.
     * @param block The target block.
     * @param offset The requested offset.
     * @return The stored value in given offset.
     */
    __host__ __device__ inline uint8_t access(EncodedBlock block, uint8_t offset)
    {
        constexpr uint8_t shift[6] = {1, 6, 11, 17, 22, 27};
        return (block >> shift[offset]) & 0x1F;
    }

    /**
     * Accesses a specific offset on a block buffer.
     * @param buffer The target buffer to search on.
     * @param offset The requested offset.
     * @return The stored value in given offset.
     */
    __host__ __device__ inline uint8_t access(const BaseBuffer<EncodedBlock>& buffer, uint8_t offset)
    {
        return access(buffer[offset / batchSize], offset % batchSize);
    }

    extern char decode(uint8_t);
    extern std::string decode(const BaseBuffer<EncodedBlock>&);

    extern Buffer<EncodedBlock> encode(const char *, size_t);
};

extern std::ostream& operator<<(std::ostream&, const BaseBuffer<encoder::EncodedBlock>&);

#endif