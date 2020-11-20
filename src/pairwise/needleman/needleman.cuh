/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements the pairwise module's needleman algorithm.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include "buffer.hpp"
#include "pairwise/pairwise.cuh"

namespace museqa
{
    namespace pairwise
    {
        namespace needleman
        {
            /**
             * Represents a general needleman algorithm for solving the heuristic's
             * pairwise alignment step.
             * @since 0.1.1
             */
            struct algorithm : public pairwise::algorithm
            {
                auto generate(size_t) const -> buffer<pair> override;

                virtual auto gather(buffer<score>&) const -> buffer<score>;            
                virtual auto run(const context&) const -> distance_matrix = 0;
            };

            /*
             * The list of all available needleman algorithm implementations.
             */
            extern auto hybrid() -> pairwise::algorithm *;
            extern auto sequential() -> pairwise::algorithm *;
        }
    }
}
