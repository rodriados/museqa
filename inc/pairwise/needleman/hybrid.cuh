/**
 * Multiple Sequence Alignment hybrid needleman header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef PW_NEEDLEMAN_HYBRID_CUH_INCLUDED
#define PW_NEEDLEMAN_HYBRID_CUH_INCLUDED

#include "buffer.hpp"
#include "pairwise/pairwise.cuh"
#include "pairwise/needleman.cuh"

namespace pairwise
{
    namespace needleman
    {
        /**
         * The hybrid needleman algorithm object. This algorithm uses hybrid
         * parallelism to run the Needleman-Wunsch algorithm.
         * @since 0.1.1
         */
        struct Hybrid : public Needleman
        {
            Buffer<Score> run(const Configuration&) override;
        };

        /**
         * Instantiates a new hybrid needleman instance.
         * @return The new algorithm instance.
         */
        inline Algorithm *hybrid()
        {
            return new Hybrid;
        }
    };
};

#endif