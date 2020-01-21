/**
 * Multiple Sequence Alignment neighbor-joining header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019 Rodrigo Siqueira
 */
#pragma once

#ifndef PG_NJOINING_CUH_INCLUDED
#define PG_NJOINING_CUH_INCLUDED

#include <limits>
#include <vector>

#include <phylogeny/tree.cuh>
#include <phylogeny/phylogeny.cuh>

namespace phylogeny
{
    namespace njoining
    {
        /**
         * Represents a candidate pair of OTUs to join.
         * @since 0.1.1
         */
        struct candidate
        {
            oturef otu[2] = {undefined, undefined};             /// The candidate's OTUs to join.
            score distance = std::numeric_limits<score>::max(); /// The candidate's score.
        }

        /**
         * Represents a general neighbor-joining algorithm.
         * @since 0.1.1
         */
        struct algorithm : public phylogeny::algorithm
        {
            std::vector<clade> clades;                      /// The list of tree's clades.

            virtual auto leaves(size_t) -> std::vector<clade>&;
            virtual auto reduce(const candidate&) const -> candidate;
            virtual auto run(const configuration&) -> tree = 0;
        };

        /*
         * The list of available neighbor-joining algorithm implementations.
         */
        //extern phylogeny::algorithm *hybrid();
        extern phylogeny::algorithm *sequential();
    }
}

#endif