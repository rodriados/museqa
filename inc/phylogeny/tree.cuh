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
    /*
     * Defines the value of an undefined node reference.
     */
    enum : uint16_t { undefined = 0xffff };

    /**
     * Represents a tree node, containing a pair of child OTU references.
     * @since 0.1.1
     */
    struct Node
    {
        Pair child {{undefined, undefined}};    /// The nodes that lead to the current one.
        uint16_t parent = undefined;            /// The parent node the current one leads to.
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
            inline Tree(const Tree&) = default;
            inline Tree(Tree&&) = default;

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
                this->ptr[root].child.id[0] = x;
                this->ptr[root].child.id[1] = y;
                this->ptr[x].parent = root;
                this->ptr[y].parent = root++;
            }

            using Buffer<Node>::operator[];
            using Buffer<Node>::getSize;
    };
};

#endif