/**
 * Multiple Sequence Alignment parallel needleman header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef PW_NEEDLEMAN_PARALLEL_HPP_INCLUDED
#define PW_NEEDLEMAN_PARALLEL_HPP_INCLUDED

#include "buffer.hpp"
#include "pairwise/pairwise.cuh"
#include "pairwise/needleman.cuh"

namespace pairwise
{
    namespace needleman
    {
        /**
         * The parallel needleman algorithm object. This algorithm uses simple
         * parallelism to run the Needleman-Wunsch algorithm.
         * @since 0.1.1
         */
        struct Parallel : public Needleman
        {
            Buffer<Score> run(const Configuration&) override;
        };

        /**
         * Instantiates a new parallel needleman instance.
         * @return The new algorithm instance.
         */
        inline Algorithm *parallel()
        {
            return new Parallel;
        }
    };
};

#endif