/**
 * Multiple Sequence Alignment phylogeny header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019-2020 Rodrigo Siqueira
 */
#pragma once

#include <string>
#include <vector>
#include <cstdint>

#include <cuda.cuh>
#include <utils.hpp>
#include <buffer.hpp>
#include <matrix.hpp>
#include <pairwise.cuh>

#include <phylogeny/tree.cuh>

namespace msa
{
    namespace phylogeny
    {
        /**
         * Definition of an operational taxonomic unit (OTU). This representation
         * effectively transforms an OTU into a phylogenetic tree node.
         * @since 0.1.1
         */
        using otu = detail::phylogeny::node;

        /**
         * Represents the reference for an OTU.
         * @since 0.1.1
         */
        using oturef = detail::phylogeny::noderef;

        /**
         * Definition of an undefined OTU reference. This reference represents an
         * unknown or undefined OTU.
         * @since 0.1.1
         */
        enum : oturef { undefined = detail::phylogeny::undefined };

        /**
         * Manages and encapsulates all configurable aspects of the phylogeny module.
         * @since 0.1.1
         */
        struct configuration
        {
            const pairwise::manager& pw;            /// The pairwise module manager instance.
            std::string algorithm;                  /// The selected phylogeny algorithm.
        };

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

                /**
                 * Initializes a new phylogeny manager from a tree instance.
                 * @param tree The tree to create manager from.
                 */
                inline manager(const underlying_type& tree) noexcept
                :   underlying_type {tree}
                {}

                inline manager& operator=(const manager&) = default;
                inline manager& operator=(manager&&) = default;

                using underlying_type::operator=;

                static auto run(const configuration&) -> manager;

            private:
                using underlying_type::connect;
        };

        /**
         * Represents a common phylogeny algorithm context.
         * @since 0.1.1
         */
        struct context
        {
            const pairwise::manager& dmatrix;
            const size_t nsequences;
        };

        /**
         * Represents a phylogeny module algorithm.
         * @since 0.1.1
         */
        struct algorithm
        {
            matrix<score> dmatrix;          /// The nodes' distance matrix.

            inline algorithm() noexcept = default;
            inline algorithm(const algorithm&) noexcept = default;
            inline algorithm(algorithm&&) noexcept = default;

            virtual ~algorithm() = default;

            inline algorithm& operator=(const algorithm&) = default;
            inline algorithm& operator=(algorithm&&) = default;

            virtual auto inflate(const pairwise::manager&) -> matrix<score>&;
            virtual auto run(const context&) -> tree = 0;

            static auto retrieve(const std::string&) -> const functor<algorithm *()>&;
            static auto list() noexcept -> const std::vector<std::string>&;
        };

        /**
         * Functor responsible for instantiating an algorithm.
         * @see phylogeny::manager::run
         * @since 0.1.1
         */
        using factory = functor<algorithm *()>;

        /**
         * Creates a module's configuration instance.
         * @param pw The pairwise module manager instance reference.
         * @param algorithm The chosen phylogeny algorithm.
         * @return The module's configuration instance.
         */
        inline configuration configure(
                const pairwise::manager& pw
            ,   const std::string& algorithm = "default"
            ) noexcept
        {
            return {pw, algorithm};
        }
    }
}
