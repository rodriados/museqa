/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements OEIS's sequences generator functions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020-present Rodrigo Siqueira
 */
#pragma once

/**
 * This file contains functions that implement sequences whose formulas can be found
 * in the On-Line Encyclopedia of Integer Sequences (OEIS). As the OEIS has been
 * of great help when trying to deduce these formulas, we name our functions as
 * the corresponding sequence's index in the OEIS, so they can be easily found and
 * their source implicitly referenced whenever we use them.
 * @link http://oeis.org
 */

#include <cmath>
#include <cstdint>

namespace museqa
{
    namespace oeis
    {
        /**
         * The triangular numbers sequence. This sequence's elements are equal to
         * the sum of all integers from zero to the given index.
         * @param n The requested sequence element's index.
         * @return The requested sequence element.
         * @link http://oeis.org/A000217
         */
        __host__ __device__ inline auto a000217(int32_t n) noexcept -> int32_t
        {
            return (n * (n + 1)) / 2;
        }

        /**
         * The integer reverse of the triangular numbers sequence, with offset zero.
         * Every integer n appears exactly n times, sequentially, on this sequence.
         * @param n The requested sequence element's index.
         * @return The requested sequence element.
         * @link https://oeis.org/A002024
         */    
        __host__ __device__ inline auto a002024(int32_t n) noexcept -> int32_t
        {
            return floor(.5f + sqrt(n * 2));
        }
    }
}
