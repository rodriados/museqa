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

#include <phylogeny/tree.cuh> 
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
                std::vector<otu> rootless;      /// The list of rootless nodes in tree.

                virtual auto join(const joinpair&) -> void = 0;
                virtual auto run(const context&) -> tree = 0;

                virtual auto reduce(joinpair&) -> joinpair;
            };

            /*
             * The list of all available needleman algorithm implementations.
             */
            extern phylogeny::algorithm *sequential();
        }
    }
}