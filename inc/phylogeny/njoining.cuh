/**
 * Multiple Sequence Alignment neighbor-joining header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019 Rodrigo Siqueira
 */
#pragma once

#ifndef PG_NJOINING_CUH_INCLUDED
#define PG_NJOINING_CUH_INCLUDED

#include <limits>

#include "buffer.hpp"
#include "pairwise.cuh"

#include "phylogeny/tree.cuh"
#include "phylogeny/phylogeny.cuh"

namespace phylogeny
{
    /**
     * References a candidate sequence pair for joining.
     * @since 0.1.1
     */
    struct JoinablePair
    {
        OTURef ref[2] = {};                                 /// The pair's sequences reference.
        Score score = std::numeric_limits<Score>::max();    /// The candidate pair obtained score.
    };

    /**
     * Aliases the type used for caching matrix's lines element sums.
     * @since 0.1.1
     */
    using SumCache = Buffer<Score>;

    /**
     * Represents a general neighbor-joining algorithm.
     * @since 0.1.1
     */
    struct NJoining : public Algorithm
    {
        Tree tree;          /// The resulting joined tree.
        uint16_t nodes;     /// The number of nodes being used for algorithm.

        virtual Tree run(const Configuration&) = 0;

        JoinablePair synchronize(const JoinablePair&) const;
    };

    namespace njoining
    {
        extern Algorithm *hybrid();
        extern Algorithm *sequential();
    }
};

#endif