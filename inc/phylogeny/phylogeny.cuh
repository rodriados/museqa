/**
 * Multiple Sequence Alignment phylogeny header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019 Rodrigo Siqueira
 */
#pragma once

#ifndef PG_PHYLOGENY_CUH_INCLUDED
#define PG_PHYLOGENY_CUH_INCLUDED

#include <string>

#include <utils.hpp>
#include <pairwise.cuh>

#include <phylogeny/tree.cuh>
#include <phylogeny/matrix.cuh>

namespace phylogeny
{
    /**
     * Manages and encapsulates all configurable aspects of the phylogeny module.
     * @since 0.1.1
     */
    struct configuration
    {
        const pairwise::manager& pw;        /// The pairwise module manager instance.
        std::string algorithm;              /// The selected phylogeny algorithm.
    };

    /**
     * Represents a phylogeny module algorithm.
     * @since 0.1.1
     */
    struct algorithm
    {
        matrix<score> mat;                  /// The alignments' distance matrix.

        inline algorithm() noexcept = default;
        inline algorithm(const algorithm&) noexcept = default;
        inline algorithm(algorithm&&) noexcept = default;

        virtual ~algorithm() = default;

        inline algorithm& operator=(const algorithm&) = default;
        inline algorithm& operator=(algorithm&&) = default;

        virtual auto populate(const pairwise::manager&) -> matrix<score>&;
        virtual auto run(const configuration&) -> tree = 0;
    };

    /**
     * Functor responsible for instantiating an algorithm.
     * @see phylogeny::manager::run
     * @since 0.1.1
     */
    using factory = functor<algorithm *()>;

    /**
     * Manages all data and execution of the phylogeny module.
     * @since 0.1.1
     */
    class manager final : public tree
    {
        protected:
            using underlying_type = tree;       /// The manager's underlying type.

        public:
            inline manager() noexcept = default;
            inline manager(const manager&) noexcept = default;
            inline manager(manager&&) noexcept = default;

            inline manager& operator=(const manager&) = default;
            inline manager& operator=(manager&&) = default;

            using underlying_type::operator=;

            static auto run(const configuration&) -> manager;
    };

    /**
     * Creates a module's configuration instance.
     * @param pw The pairwise module manager instance reference.
     * @param algorithm The chosen phylogeny algorithm.
     * @return The module's configuration instance.
     */
    inline configuration configure(
            const pairwise::manager& pw
        ,   const std::string& algorithm = {}
        )
    {
        return {pw, algorithm};
    }
}

#endif