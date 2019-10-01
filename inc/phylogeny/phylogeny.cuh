/**
 * Multiple Sequence Alignment phylogeny header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019 Rodrigo Siqueira
 */
#pragma once

#ifndef PG_PHYLOGENY_CUH_INCLUDED
#define PG_PHYLOGENY_CUH_INCLUDED

#include <string>
#include <utility>

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
        const Pairwise& pw;         /// The pairwise module instance.
        std::string algorithm;      /// The chosen phylogeny algorithm.
    };

    /**
     * Represents a phylogeny module algorithm.
     * @since 0.1.1
     */
    struct Algorithm
    {
        inline Algorithm() noexcept = default;
        inline Algorithm(const Algorithm&) noexcept = default;
        inline Algorithm(Algorithm&&) noexcept = default;

        virtual ~Algorithm() = default;

        inline Algorithm& operator=(const Algorithm&) noexcept = default;
        inline Algorithm& operator=(Algorithm&&) noexcept = default;

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
            inline Phylogeny() = default;
            inline Phylogeny(const Phylogeny&) = default;
            inline Phylogeny(Phylogeny&&) = default;

            inline Phylogeny& operator=(const Phylogeny&) = default;
            inline Phylogeny& operator=(Phylogeny&&) = default;

            using Tree::operator=;

            void run(const Configuration&);
    };
};

#endif