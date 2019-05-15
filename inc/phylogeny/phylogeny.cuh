/**
 * Multiple Sequence Alignment phylogeny header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019 Rodrigo Siqueira
 */
#pragma once

#ifndef PG_PHYLOGENY_CUH_INCLUDED
#define PG_PHYLOGENY_CUH_INCLUDED

#include <string>

#include "utils.hpp"
#include "pairwise.cuh"
 
#include "phylogeny/tree.cuh"

namespace phylogeny
{
    /**
     * Manages and encapsulates all configurable aspects of the pairwise module.
     * @since 0.1.1
     */
    struct Configuration
    {
        const Pairwise& pw;       /// The pairwise module instance.
        std::string algorithm;      /// The chosen phylogeny algorithm.
    };

    /**
     * Represents a phylogeny module algorithm.
     * @since 0.1.1
     */
    struct Algorithm
    {
        Algorithm() = default;
        Algorithm(const Algorithm&) = default;
        Algorithm(Algorithm&&) = default;

        virtual ~Algorithm() = default;

        Algorithm& operator=(const Algorithm&) = default;
        Algorithm& operator=(Algorithm&&) = default;

        virtual Tree run(const Configuration&) = 0;
    };

    /**
     * Functor responsible for instantiating an algorithm.
     * @see Phylogeny::run
     * @since 0.1.1
     */
    using Factory = Functor<Algorithm *()>;

    /**
     * Manages all data and execution of the pgenetic module.
     * @since 0.1.1
     */
    class Phylogeny final : public Tree
    {
        public:
            Phylogeny() = default;
            Phylogeny(const Phylogeny&) = default;
            Phylogeny(Phylogeny&&) = default;

            Phylogeny& operator=(const Phylogeny&) = default;
            Phylogeny& operator=(Phylogeny&&) = default;

            using Tree::operator=;

            void run(const Configuration&);
    };
};

#endif