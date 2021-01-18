/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements the phylogeny module's functionality.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019-present Rodrigo Siqueira
 */
#pragma once

#include <limits>
#include <string>
#include <vector>
#include <cstdint>
#include <utility>

#include "utils.hpp"
#include "functor.hpp"
#include "pairwise.cuh"
#include "phylogeny/tree.cuh"

namespace museqa
{
    namespace phylogeny
    {
        /**
         * Definition of the reference for an operational taxonomic unit (OTU).
         * As sequences by themselves are also considered OTUs, and thus leaf nodes
         * on our phylogenetic tree, we must guarantee that our OTU references are
         * able to individually identify all sequence references.
         * @since 0.1.1
         */
        using oturef = typename std::conditional<
                utils::lt(sizeof(uint_least32_t), sizeof(pairwise::seqref))
            ,   typename std::make_unsigned<pairwise::seqref>::type
            ,   uint_least32_t
            >::type;

        /**
         * As the only relevant aspect of a phylogenetic tree during its construction
         * is its topology, we can assume an OTU's relationship to its neighbors
         * is its only relevant piece of information. Thus, at this stage, we consider
         * OTUs to simply be references on our OTU addressing space, at a certain
         * distance to one of its neighbors on a higher tree level.
         * @since 0.1.1
         */
        struct otu
        {
            using distance_type = score;        /// The distance type between OTUs.

            static constexpr auto farthest = std::numeric_limits<distance_type>::max();

            oturef id;                          /// The OTU's reference id.
            distance_type distance = farthest;  /// The distance from this OTU to its parent.
            uint_least32_t level = 0;           /// The OTU's level on the phylogenetic tree.
        };

        /**
         * Aliases a generic binary tree of OTUs into a phylogenetic tree, which
         * is the module's final result type.
         * @since 0.1.1
         */
        using phylotree = tree<otu, oturef>;

        /**
         * We use the highest available reference in our pseudo-addressing-space
         * to represent an unknown or undefined node of the phylogenetic tree. It
         * is very unlikely that you'll ever need to fill up our whole addressing
         * space with distinct OTUs references. And if you do, well, you'll have
         * issues with this approach.
         * @since 0.1.1
         */
        enum : oturef { undefined = phylotree::undefined };

        /**
         * Represents a common phylogeny algorithm context.
         * @since 0.1.1
         */
        struct context
        {
            const pairwise::distance_matrix matrix;
            const size_t count;
        };

        /**
         * Functor responsible for instantiating an algorithm.
         * @see phylogeny::run
         * @since 0.1.1
         */
        using factory = functor<struct algorithm *()>;

        /**
         * Represents a phylogeny module algorithm.
         * @since 0.1.1
         */
        struct algorithm
        {
            inline algorithm() noexcept = default;
            inline algorithm(const algorithm&) noexcept = default;
            inline algorithm(algorithm&&) noexcept = default;

            virtual ~algorithm() = default;

            inline algorithm& operator=(const algorithm&) = default;
            inline algorithm& operator=(algorithm&&) = default;

            virtual auto run(const context&) const -> phylotree = 0;

            static auto has(const std::string&) -> bool;
            static auto make(const std::string&) -> const factory&;
            static auto list() noexcept -> const std::vector<std::string>&;
        };

        /**
         * Runs the module when not on a pipeline.
         * @param matrix The distance matrix between sequences.
         * @param count The total number of sequences to align.
         * @param algorithm The chosen phylogeny algorithm.
         * @return The chosen algorithm's resulting phylogenetic tree.
         */
        inline phylotree run(
                const pairwise::distance_matrix& matrix
            ,   const size_t count
            ,   const std::string& algorithm = "default"
            )
        {
            auto lambda = phylogeny::algorithm::make(algorithm);
            
            const phylogeny::algorithm *worker = lambda ();
            auto result = worker->run({matrix, count});
            
            delete worker;
            return result;
        }
    }
}
