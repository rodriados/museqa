/**
 * Multiple Sequence Alignment phylogeny tree header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019 Rodrigo Siqueira
 */
#pragma once

#ifndef PG_TREE_CUH_INCLUDED
#define PG_TREE_CUH_INCLUDED

#include <limits>
#include <utility>
#include <cstdint>

#include <pairwise.cuh>

namespace phylogeny
{
    /**
     * Represents a reference for an operational taxonomic unit (OTU).
     * @since 0.1.1
     */
    using oturef = typename std::conditional<
            (sizeof(seqref) > sizeof(uint_least32_t))
        ,   typename std::make_unsigned<seqref>::type
        ,   uint_least32_t
        >::type;

    /*
     * Defining the value of an unknown OTU reference. This value will be used whenever
     * we have an unset or undefined reference to an OTU.
     */
    enum : oturef { undefined = ~0 };

    /**
     * Represents a clade, that is, a recursive node of a tree. A clade is a group
     * of organisms believed to have evolved from a common ancestor.
     * @since 0.1.1
     */
    struct clade
    {
        oturef otu = undefined;                             /// The clade's OTU identifier.
        oturef parent = undefined;                          /// The clade's parent OTU reference.
        oturef children[2] = {undefined, undefined};        /// The clade's children OTU reference.
        score distance = std::numeric_limits<score>::max(); /// The distance to clade's parent node.
        size_t height = 0;                                  /// The clade node's height in tree.
    };

    /**
     * Represents a pseudo-phylogenetic tree, containing the hierarchy for future
     * progressive alignment of input sequences.
     * @since 0.1.1
     */
    class tree : protected buffer<clade>
    {
        protected:
            using element_type = clade;                 /// The tree's element type.
            using underlying_buffer = buffer<clade>;    /// The tree's underlying buffer.

        protected:
            clade root = {};                            /// The tree's root node.

        // Nodes must be stored in a buffer and then sorted according to their respective heights.
        // That means, the first nodes in buffer must have height 1, followed by those with height 2
        // and thus successively.

        public:
            inline tree() noexcept = default;
            inline tree(const tree&) noexcept = default;
            inline tree(tree&&) noexcept = default;

            inline tree& operator=(const tree&) = default;
            inline tree& operator=(tree&&) = default;
    };
}

#endif