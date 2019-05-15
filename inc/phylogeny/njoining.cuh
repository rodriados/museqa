/**
 * Multiple Sequence Alignment neighbor-joining header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019 Rodrigo Siqueira
 */
#pragma once

#ifndef PG_NJOINING_CUH_INCLUDED
#define PG_NJOINING_CUH_INCLUDED

#include "pairwise.cuh"

#include "phylogeny/tree.cuh"
#include "phylogeny/matrix.cuh"
#include "phylogeny/phylogeny.cuh"

namespace phylogeny
{
    /**
     * Represents a general neighbor-joining algorithm.
     * @since 0.1.1
     */
    struct NJoining : public Algorithm
    {
        uint16_t active;        /// Indicates the number of active cluster nodes.

        virtual Tree run(const Configuration&) = 0;

        Pair reduce(const Pair&) const;
    };

    namespace njoining
    {
        extern Algorithm *hybrid();
    }
};

#endif