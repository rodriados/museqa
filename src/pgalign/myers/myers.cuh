/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements the profile-aligner module's myers-miller algorithm.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020-present Rodrigo Siqueira
 */
#pragma once

#include "pgalign/pgalign.cuh"
#include "pgalign/alignment.cuh"

namespace museqa
{
    namespace pgalign
    {
        namespace myers
        {
            /**
             * Represents a general k-dim needleman algorithm for solving the heuristic's
             * profile-aligner step.
             * @since 0.1.1
             */
            struct algorithm : public pgalign::algorithm
            {
                virtual auto run(const context&) const -> alignment = 0;
            };

            /*
             * The list of all available k-dim needleman-wunsch algorithm implementations.
             */
            extern auto sequential() -> pgalign::algorithm *;
        }
    }
}
