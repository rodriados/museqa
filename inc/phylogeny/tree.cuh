/**
 * Multiple Sequence Alignment phylogenetic tree header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019-2020 Rodrigo Siqueira
 */
#pragma once

#include <limits>
#include <cstdint>
#include <utility>

#include <buffer.hpp>
#include <pairwise.cuh>

namespace msa
{
    namespace detail
    {
        namespace phylogeny
        {
            /**
             * Represents the reference for a phylogenic tree node. As sequences
             * themselves are also considered nodes in our phylogenetic tree, we
             * need node references to be at least as big as sequence references.
             * @since 0.1.1
             */
            using noderef = typename std::conditional<
                    (sizeof(seqref) > sizeof(uint_least32_t))
                ,   std::make_unsigned<seqref>::type
                ,   uint_least32_t
                >::type;

            /**
             * Definition of an undefined tree node reference. It is very unlikely
             * that you'll ever need to fill up our whole pseudo-address-space with
             * node references. For that reason, we use the highest available reference
             * to represent an unset or undefined node.
             * @since 0.1.1
             */
            enum : noderef { undefined = noderef(~0) };

            /**
             * Definition of a phylogenic tree node. Rather unconventional, we do
             * not hold direct pointers in our tree's nodes. This is done so we
             * don't ever need to worry about pointers if we end up having to send
             * a node, or the whole tree through MPI to another slave MPI node.
             * @since 0.1.1
             */
            struct node
            {
                noderef ref = undefined;                            /// The current node reference.
                noderef parent = undefined;                         /// The parent node's reference.
                score distance = std::numeric_limits<score>::max(); /// The distance to parent node.
                uint32_t height = 0;                                /// The node's height in tree.
            };
        }
    }

    namespace phylogeny
    {
        /**
         * Represents a pseudo-phylogenetic tree. Each node in this tree is effectively
         * an operational taxonomic unit (OTU), which in turn can represent a single
         * sequence or an alignment of multiple sequences. Nodes in this tree are
         * stored contiguously in memory, as the number of total nodes is known
         * at instantiation time. Furthermore, this tree addresses its lowest nodes,
         * its leaves, to sequences as do our node address space and every node
         * is accessible via their respective node reference.
         * @since 0.1.1
         */
        class tree : protected buffer<detail::phylogeny::node>
        {
            public:
                using node_type = detail::phylogeny::node;      /// The tree's node type.
                using underlying_buffer = buffer<node_type>;    /// The tree's underlying buffer type.

            protected:
                using ref_type = detail::phylogeny::noderef;    /// The tree's nodes' reference.

            protected:
                size_t m_leaves = 0;                            /// The number of leaves in tree.

            public:
                inline tree() noexcept = default;
                inline tree(const tree&) noexcept = default;
                inline tree(tree&&) noexcept = default;

                /**
                 * Initializes a new tree with the given number of leaves. As our
                 * tree requires that all non-leaf nodes have at least two children,
                 * we know the maximum number of nodes at instantiation time.
                 * @param leaves The number of leaves in tree.
                 */
                inline tree(const size_t& leaves) noexcept
                :   underlying_buffer {underlying_buffer::make(leaves * 2 - 1)}
                ,   m_leaves {leaves}
                {
                    for(size_t i = 0; i < this->m_size; ++i)
                        this->m_ptr[i].ref = i;
                }

                inline tree& operator=(const tree&) = default;
                inline tree& operator=(tree&&) = default;

                using underlying_buffer::operator=;
                using underlying_buffer::operator[];

                /**
                 * Connects a node to its direct ancestor. The new created link
                 * may also update the ancestor's height.
                 * @param node The base node reference.
                 * @param ancestor The ancestor node reference.
                 * @param distance The distance between ancestor and base nodes.
                 * @return The updated ancestor node height.
                 */
                inline uint32_t connect(ref_type node, ref_type ancestor, score distance)
                {
                    node_type& current = operator[](node);
                    current.parent = ancestor;
                    current.distance = distance;

                    node_type& parent = operator[](ancestor);
                    return parent.height = utils::max(parent.height, current.height + 1);
                }

                /**
                 * Informs the number of leaves in the current tree.
                 * @return The number of leaves in tree.
                 */
                inline size_t leaves() const noexcept
                {
                    return m_leaves;
                }

                using underlying_buffer::size;

                using underlying_buffer::begin;
                using underlying_buffer::end;
                using underlying_buffer::raw;
        };
    }
}
