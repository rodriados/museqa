/**
 * Multiple Sequence Alignment phylogeny tree header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019 Rodrigo Siqueira
 */
#pragma once

#ifndef PG_TREE_CUH_INCLUDED
#define PG_TREE_CUH_INCLUDED

#include "buffer.hpp"
#include "pairwise.cuh"

namespace phylogeny
{
    /**
     * An OTU, a operational taxonomic unit, can be either a single sequence or
     * an alignment of two or more sequences.
     * @since 0.1.1
     */
    using OTURef = typename std::make_unsigned<SequenceRef>::type;

    /*
     * Defines the value of an undefined node reference. This value will be used
     * whenever we have an unset or unknown reference.
     */
    enum : OTURef { undefined = 0xffff };

    /**
     * Stores information about a branch connection in the phylogenetic tree.
     * @since 0.1.1
     */
    struct Branch
    {
        OTURef id = undefined;
        Score distance = -1;
    };

    /**
     * Represents a tree node, containing a pair of child OTU references.
     * @since 0.1.1
     */
    struct Node
    {
        Branch parent;          /// The current node's parent connection.
        Branch child[2];        /// The current node's children connections.
        bool isLeaf = false;    /// Is the node currently a leaf?
    };

    /**
     * Represents a pseudo-phylogenetic tree, containing the hierarchy for
     * future progressive alignment of input OTUs.
     * @since 0.1.1
     */
    class Tree : protected Buffer<Node>
    {
        protected:
            uint16_t root;      /// The current tree root node index.

        public:
            inline Tree() noexcept = default;
            inline Tree(const Tree&) noexcept = default;
            inline Tree(Tree&&) noexcept = default;

            /**
             * Creates a new tree by allocating all nodes it will ever need.
             * @param count The number of input OTUs.
             */
            inline Tree(size_t count)
            :   Buffer<Node> {count * 2}
            ,   root {static_cast<uint16_t>(count)}
            {}

            inline Tree& operator=(const Tree&) = default;
            inline Tree& operator=(Tree&&) = default;

            /**
             * Joins two nodes together and sets the current tree root as their parents.
             * @param x The identifier of first OTU to join.
             * @param y The identifier of second OTU to join.
             */
            inline void join(uint16_t x, uint16_t y) noexcept
            {
                this->ptr[root].child[0].id = x;
                this->ptr[root].child[1].id = y;
                this->ptr[x].parent.id = root;
                this->ptr[y].parent.id = root++;
            }

            using Buffer<Node>::operator[];
            using Buffer<Node>::getSize;
    };
};

#endif