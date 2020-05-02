/**
 * Multiple Sequence Alignment neighbor-joining header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019-2020 Rodrigo Siqueira
 */
#pragma once

#include <limits>
#include <vector>

#include <pairwise.cuh>
#include <reflection.hpp>

#include <phylogeny/phylogeny.cuh>

namespace msa
{
    namespace phylogeny
    {
        namespace njoining
        {
            /**
             * Represents a joinable OTU pair candidate.
             * @since 0.1.1
             */
            struct joinpair : public reflector
            {
                oturef ref[2] = {undefined, undefined};             /// The OTU pair references.
                score distance = std::numeric_limits<score>::max(); /// The distance between the OTU pair.
                using reflex = decltype(reflect(ref, distance));
            };

            /**
             * Represents a general neighbor-joining algorithm for solving the step
             * of building a phylogenetic tree.
             * @since 0.1.1
             */
            struct algorithm : public phylogeny::algorithm
            {
                tree phylotree;

                virtual auto reduce(joinpair&) -> joinpair;
                virtual auto run(const context&) -> tree = 0;
            };

            /*
             * The list of all available needleman algorithm implementations.
             */
            extern phylogeny::algorithm *sequential();
        }
    }
}