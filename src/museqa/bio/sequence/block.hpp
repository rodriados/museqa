/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file The definition of a biological sequence block of symbols.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2023-present Rodrigo Siqueira
 */
#pragma once

#include <cstdint>
#include <museqa/environment.h>

MUSEQA_BEGIN_NAMESPACE

namespace bio::sequence
{
    /**#@+
     * The underlying type of a block of a biological sequence, which stores one
     * or more compressed symbols within it. The length of the block can be overridden
     * in compile-time by defining a compilation macro.
     * @since 1.0
     */
  #if !defined(MUSEQA_BLOCK_LENGTH)
    // By default, the sequence block length is set to 16. This is the minimum length
    // needed for a block to be able to store more than one symbol.
    #define MUSEQA_BLOCK_LENGTH 16
    using block_t = uint16_t;

  #elif MUSEQA_BLOCK_LENGTH == 8
    using block_t = uint8_t;

  #elif MUSEQA_BLOCK_LENGTH == 16
    using block_t = uint16_t;

  #elif MUSEQA_BLOCK_LENGTH == 32
    using block_t = uint32_t;

  #elif MUSEQA_BLOCK_LENGTH == 64
    using block_t = uint64_t;

  #else
    // A sequence block cannot have a length that is not a power of 2 bits. This
    // is so because there is no scalar type that has a non-power of 2 bit length.
    // Therefore, a compilation error must be issued in such scenarios.
    #error The length of a sequence block must be a power of 2 bits.

  #endif
    /**#@-*/
}

MUSEQA_END_NAMESPACE
