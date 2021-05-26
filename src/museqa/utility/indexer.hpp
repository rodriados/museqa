/** 
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file An integral sequence indexer type implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2021-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/utility.hpp>

namespace museqa
{
    namespace utility
    {
        /**
         * Represents an indexed sequence type.
         * @tparam I The sequence of the type's indeces.
         * @since 1.0
         */
        template <size_t ...I>
        struct indexer
        {
            using type = indexer;
        };

        /**
         * Generates the indexed sequence type.
         * @tparam L The length of the sequence type to generate.
         * @since 1.0
         */
        template <size_t L>
        struct indexer<L>
        {
            /**
             * Concatenates two type index sequences into a single one.
             * @tparam I The first index sequence to merge.
             * @tparam J The second index sequence to merge.
             * @return The concatenated index sequence.
             */
            template <size_t ...I, size_t ...J>
            __host__ __device__ static constexpr auto concat(indexer<I...>, indexer<J...>) noexcept
            -> typename indexer<I..., sizeof...(I) + J...>::type;

          private:
            using low = typename indexer<L / 2>::type;
            using high = typename indexer<L - L / 2>::type;

          public:
            using type = decltype(concat(low {}, high {}));
        };

        /**
         * The 0-base type for the sequence indexer generator. This type is the
         * recursion base when generating a sequence index. Thus, as a representant
         * of a sequence containing zero elements, its produced sequence is empty.
         * @since 1.0
         */
        template <>
        struct indexer<0>
        {
            using type = indexer<>;
        };

        /**
         * A generator for a 1-length sequence index. This type is also a recursion
         * base when generating a sequence index. Similarly to the 0-length sequence,
         * the produced sequence of length 1 must only contain one element: zero.
         * @since 1.0
         */
        template <>
        struct indexer<1>
        {
            using type = indexer<0>;
        };
    }
}
