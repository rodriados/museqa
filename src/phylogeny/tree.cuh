/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements the phylogeny module's tree data structure.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020-present Rodrigo Siqueira
 */
#pragma once

#include <cstdint>
#include <utility>

#include "tree.hpp"
#include "utils.hpp"
#include "buffer.hpp"
#include "allocator.hpp"

namespace museqa
{
    namespace phylogeny
    {
        /**
         * Represents a phylogenetic tree. Our phylogenetic tree will be treated
         * as a dendogram, and each node in this dendogram must be represented by
         * an unique reference in the given reference type's addressing space. The
         * nodes on this tree are stored contiguously in memory, as the number of
         * total nodes is known at instantiation-time. Rather unconventionally,
         * though, we do not hold any effective memory pointers in our tree's nodes,
         * so that we don't need to worry about them if we ever need to transfer
         * the tree around the cluster to different machines. Furthermore, all of
         * the tree's leaves occupy the lowest references on its addressing space.
         * @tparam T The tree's nodes' contents type.
         * @tparam R The tree's nodes' reference type.
         * @since 0.1.1
         */
        template <typename T, typename R>
        class tree : protected museqa::tree<T, R>
        {
            static_assert(std::is_integral<R>::value, "tree node reference must be an integral type");

            protected:
                using underlying_tree = museqa::tree<T, R>;
                using node_type = typename underlying_tree::node;
                using buffer_type = buffer<node_type>;

            public:
                using reference_type = typename underlying_tree::reference_type;

            public:
                static constexpr reference_type undefined = node_type::undefined;

            protected:
                buffer_type m_buffer;           /// The buffer of all nodes on the tree.
                uint32_t m_leaves = 0;          /// The total number of leaf-nodes on the tree.

            public:
                inline tree() noexcept = default;
                inline tree(const tree&) noexcept = default;
                inline tree(tree&&) noexcept = default;

                inline tree& operator=(const tree&) = default;
                inline tree& operator=(tree&) = default;

                /**
                 * Retrieves a const-qualified node from the tree by it's reference value.
                 * @param ref The node reference value to be retrieved.
                 * @return The requested const-qualified node.
                 */
                __host__ __device__ inline const node_type& operator[](reference_type ref) const
                {
                    return m_buffer[ref];
                }

                /**
                 * Gives access to the tree's root node.
                 * @return The tree's root node.
                 */
                __host__ __device__ inline const node_type& root() const noexcept
                {
                    return operator[](underlying_tree::root());
                }

                /**
                 * Retrieves a buffer containing all of the tree's leaf-nodes.
                 * @return The list of leaf-nodes on the tree.
                 */
                inline auto leaves() const noexcept -> const buffer_slice<node_type>
                {
                    return buffer_slice<node_type> {m_buffer, 0, m_leaves};
                }

            protected:
                /**
                 * Creates a new tree with given number of nodes as leaves.
                 * @param leaves The number of leaf nodes in tree.
                 * @return The newly created tree instance.
                 */
                static inline tree make(uint32_t leaves) noexcept
                {
                    return tree {buffer_type::make((leaves * 2) - 1), leaves};
                }

                /**
                 * Creates a new tree with given number of nodes as leaves.
                 * @param allocator The allocator to be used to create new dendogram.
                 * @param leaves The number of leaf nodes in tree.
                 * @return The newly created tree instance.
                 */
                static inline tree make(const museqa::allocator& allocator, uint32_t leaves)
                {
                    return tree {buffer_type::make(allocator, (leaves * 2) - 1), leaves};
                }

            private:
                /**
                 * Builds a new tree from an underlying tree nodes buffer. It is
                 * assumed that the given tree is empty, without relevant hierarchy.
                 * @param raw The tree's underlying buffer instance.
                 * @param leaves The number of leaf nodes in tree.
                 */
                inline tree(buffer_type&& raw, uint32_t leaves)
                :   underlying_tree {static_cast<reference_type>(raw.size() - 1)}
                ,   m_buffer {std::forward<decltype(raw)>(raw)}
                ,   m_leaves {leaves}
                {}
        };
    }
}
