/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements the neighbor-joining auxiliary star-tree data structure.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2021-present Rodrigo Siqueira
 */
#pragma once

#include <cstdint>
#include <utility>

#include "utils.hpp"

#include "phylogeny/tree.cuh"
#include "phylogeny/phylogeny.cuh"

namespace museqa
{
    namespace phylogeny
    {
        namespace njoining
        {
            /**
             * Represents a star-tree. A star-tree is a special kind of tree in
             * which the nodes initial disposition is such that all leaf nodes are
             * connected to a common parent node.
             * @since 0.1.1
             */
            class star : public phylogeny::phylotree
            {
                protected:
                    using underlying_tree = phylotree;
                    using distance_type = typename node_type::distance_type;

                protected:
                    /**
                     * Gathers all information needed to perform a join operation
                     * between two nodes to a new common parent node.
                     * @since 0.1.1
                     */
                    struct joinable
                    {
                        reference_type id;          /// The reference to OTU to join.
                        distance_type distance;     /// The distance from OTU to its new parent.
                    };

                public:
                    inline star() noexcept = default;
                    inline star(const star&) noexcept = default;
                    inline star(star&&) noexcept = default;

                    inline star& operator=(const star&) = default;
                    inline star& operator=(star&&) = default;

                    /**
                     * Joins a pair of OTUs into a common parent node.
                     * @param parent The parent node reference to join children OTUs to.
                     * @param one The joining information for the parent's first child.
                     * @param two The joining information for the parent's second child.
                     */
                    inline void join(reference_type parent, const joinable& one, const joinable& two)
                    {
                        auto& father = m_buffer[parent];

                        const auto& lchild = connect(father, one);
                        const auto& rchild = connect(father, two);

                        father.child[0] = lchild.id;
                        father.child[1] = rchild.id;

                        father.level = utils::max(lchild.level, rchild.level) + 1;
                    }

                    /**
                     * Creates a new star-tree with given number of nodes as leaves.
                     * @param leaves The number of leaf nodes in tree.
                     * @return The newly created star-tree instance.
                     */
                    static inline star make(uint32_t leaves) noexcept
                    {
                        return star {underlying_tree::make(leaves)};
                    }

                    /**
                     * Creates a new star-tree with given number of nodes as leaves.
                     * @param allocator The allocator to be used to create new dendogram.
                     * @param leaves The number of leaf nodes in tree.
                     * @return The newly created star-tree instance.
                     */
                    static inline star make(const museqa::allocator& allocator, uint32_t leaves)
                    {
                        return star {underlying_tree::make(allocator, leaves)};
                    }

                protected:
                    /**
                     * Connects a node to its new parent node on the tree.
                     * @param parent The node to act as the new child's parent.
                     * @param info The child node's joining information.
                     * @return The reference to the connected child.
                     */
                    inline const node_type& connect(const node_type& parent, const joinable& info)
                    {
                        auto& child = m_buffer[info.id];

                        child.distance = info.distance;
                        child.parent = parent.id;

                        return child;
                    }

                private:
                    /**
                     * Instantiates a new star-tree from an underlying tree.
                     * @param tree The underlying tree's instance.
                     */
                    inline star(underlying_tree&& tree)
                    :   underlying_tree {std::forward<decltype(tree)>(tree)}
                    {
                        for(size_t i = 0, n = this->m_buffer.size(); i < n; ++i)
                            this->m_buffer[i].id = (oturef) i;
                    }
            };
        }
    }
}
