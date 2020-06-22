/**
 * Multiple Sequence Alignment needleman header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2020 Rodrigo Siqueira
 */
#pragma once

#include <buffer.hpp>
#include <pointer.hpp>

#include <pairwise/pairwise.cuh>

namespace msa
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
                virtual auto scatter(buffer<pair>&) const -> buffer<pair>;
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
