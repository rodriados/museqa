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
         * Module general functions.
         */
        extern Buffer<Pair> generate(size_t);
        
        /*
         * Module distribution functions.
         */
        extern Buffer<Pair> scatter(Buffer<Pair>&);
        extern Buffer<Score> gather(Buffer<Score>&);

        /*
         * Module algorithms.
         */
        //extern Buffer<Score> sequential(const Configuration&);
        //extern Buffer<Score> parallel(const Configuration&);
        //extern Buffer<Score> distributed(cosnt Configuration&);
        extern Buffer<Score> hybrid(const Configuration&);

        /**
         * Calls the default algorithm for the current module.
         * @param config The module configuration parameters.
         * @return The algorithm execution result.
         */
        inline Buffer<Score> run(const Configuration& config)
        {
            return hybrid(config);
        }
    };
};

#endif