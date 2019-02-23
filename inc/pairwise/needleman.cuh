/**
 * Multiple Sequence Alignment needleman header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef PW_NEEDLEMAN_CUH_INCLUDED
#define PW_NEEDLEMAN_CUH_INCLUDED

#include "buffer.hpp"
#include "pairwise/pairwise.cuh"

namespace pairwise
{
    namespace needleman
    {
        /**
         * Represents a general needleman algorithm.
         * @since 0.1.1
         */
        class Needleman : public Algorithm
        {
            protected:
                Buffer<Score> score;            /// The algorithm result.

            public:
                virtual Buffer<Score> run(const Configuration&) = 0;

            protected:
                Buffer<Pair> scatter();
                Buffer<Score> gather();

        };

        extern Algorithm *factory();
    };
};

#endif