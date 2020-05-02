/**
 * Multiple Sequence Alignment phylogeny header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019-2020 Rodrigo Siqueira
 */
#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <utility>

#include <utils.hpp>
#include <buffer.hpp>
#include <dendogram.hpp>
#include <symmatrix.hpp>

#include <cuda.cuh>
#include <pairwise.cuh>

namespace msa
{
    namespace phylogeny
    {
        /**
         * Definition of the reference for a operational taxonomic unit (OTU). As
         * sequences themselves are also considered OTUs, and thus independent nodes
         * in our phylogenetic tree, we must guarantee that our OTU references are
         * able to address at least all sequence references.
         * @since 0.1.1
         */
        using oturef = typename std::conditional<
                (sizeof(uint_least32_t) < sizeof(seqref))
            ,   typename std::make_unsigned<seqref>::type
            ,   uint_least32_t
            >::type;

        /**
         * As the only relevant aspect of a phylogenetic tree during its construction
         * is its topology, we can assume an OTU's relationship to its neighbors
         * is its only relevant piece of information. Thus, at this stage, we consider
         * OTUs are simply references in our OTU addressing space.
         * @since 0.1.1
         */
        using otu = oturef;

        /**
         * Represents a phylogenetic tree. Our phylogenetic tree will be treated
         * as a dendogram, and each node in this dendogram is effectively an OTU.
         * The nodes in this tree are stored contiguously in memory, as the number
         * of total nodes is known at instantiation-time. Rather unconventionally,
         * though, we do not hold any physical memory pointers in our tree's nodes,
         * so that we don't need to worry about them if we ever need to transfer
         * the tree around the cluster to different machines. Furthermore, all of
         * the tree's leaves occupy the lowest references on its addressing space.
         * @since 0.1.1
         */
        using tree = dendogram<otu, score, oturef>;

        /**
         * Definition of an undefined OTU reference. It is very unlikely that you'll
         * ever need to fill up our whole pseudo-addressing-space with distinct
         * OTUs references. For that reason, we use the highest available reference
         * to represent an unset or undefined node.
         * @since 0.1.1
         */
        enum : oturef { undefined = tree::undefined };

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
        };

        /**
         * Represents a common phylogeny algorithm context.
         * @since 0.1.1
         */
        struct context
        {
            const pairwise::manager& matrix;
            const size_t nsequences;
        };

        /**
         * Represents a phylogeny module algorithm.
         * @since 0.1.1
         */
        struct algorithm
        {
            symmatrix<score> distances;          /// The nodes' distance matrix.

            inline algorithm() noexcept = default;
            inline algorithm(const algorithm&) noexcept = default;
            inline algorithm(algorithm&&) noexcept = default;

            virtual ~algorithm() = default;

            inline algorithm& operator=(const algorithm&) = default;
            inline algorithm& operator=(algorithm&&) = default;

            virtual auto inflate(const pairwise::manager&) -> symmatrix<score>&;
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
