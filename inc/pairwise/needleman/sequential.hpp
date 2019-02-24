/**
 * Multiple Sequence Alignment sequential needleman header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef PW_NEEDLEMAN_SEQUENTIAL_HPP_INCLUDED
#define PW_NEEDLEMAN_SEQUENTIAL_HPP_INCLUDED

#include "buffer.hpp"
#include "pairwise/pairwise.cuh"
#include "pairwise/needleman.cuh"

namespace pairwise
{
    namespace needleman
    {
        /**
         * The sequential needleman algorithm object. This algorithm uses no
         * parallelism, besides pairs distribution to run the Needleman-Wunsch
         * algorithm.
         * @since 0.1.1
         */
        struct Sequential : public Needleman
        {
            Buffer<Score> run(const Configuration&) override;
        };

        /**
         * Instantiates a new sequential needleman instance.
         * @return The new algorithm instance.
         */
        inline Algorithm *sequential()
        {
            return new Sequential;
        }
    };
};

#endif