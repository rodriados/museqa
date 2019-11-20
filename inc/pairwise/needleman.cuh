/**
 * Multiple Sequence Alignment needleman header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef PW_NEEDLEMAN_CUH_INCLUDED
#define PW_NEEDLEMAN_CUH_INCLUDED

#include <buffer.hpp>
#include <pointer.hpp>

#include <pairwise/pairwise.cuh>

namespace pairwise
{
    namespace needleman
    {
        /**
         * Represents a general needleman algorithm.
         * @since 0.1.1
         */
        struct algorithm : public pairwise::algorithm
        {
            virtual auto scatter() -> buffer<pair>;
            virtual auto gather(buffer<score>&) const -> buffer<score>;            
            virtual auto run(const configuration&) -> buffer<score> = 0;
        };

        /*
         * The list of available needleman algorithm implementations.
         */
        extern pairwise::algorithm *hybrid();
        extern pairwise::algorithm *sequential();
    }
}

#endif