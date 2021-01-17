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
#include "allocator.hpp"
#include "binarytree.hpp"

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
         * OTUs to simply be references on our OTU addressing space.
         * @since 0.1.1
         */
        using otu = oturef;

        /**
         * Represents an OTU node in a phylogenetic tree. The OTU node must also
         * keep track of its height on the tree, and its distance to its parent.
         * @since 0.1.1
         */
        struct otunode
        {
            using otu_type = otu;                   /// The node's contents type.
            using distance_type = score;            /// The type of the distance between nodes.

            static constexpr distance_type farthest = std::numeric_limits<distance_type>::max();

            otu_type id;                            /// The node's id on the tree.
            distance_type distance = farthest;      /// The distance from this node to its parent.

            uint32_t level = 0;                     /// The node's level or height in dendogram.
        };

        /**
         * Represents a phylogenetic tree. Our phylogenetic tree will be treated
         * as a dendogram, and each node in this dendogram is effectively an OTU.
         * The nodes on this tree are stored contiguously in memory, as the number
         * of total nodes is known at instantiation-time. Rather unconventionally,
         * though, we do not hold any physical memory pointers in our tree's nodes,
         * so that we don't need to worry about them if we ever need to transfer
         * the tree around the cluster to different machines. Furthermore, all of
         * the tree's leaves occupy the lowest references on its addressing space.
         * @since 0.1.1
         */
        class tree : public binarytree<otunode, oturef>
        {
            protected:
                using underlying_tree = binarytree<otunode, oturef>;

            protected:
                using distance_type = otunode::distance_type;
                using node_type = typename underlying_tree::node;
                using reference_type = typename underlying_tree::reference_type;
                using buffer_type = buffer<node_type>;

            public:
                static constexpr reference_type undefined = node::undefined;

            protected:
                /**
                 * Gathers all information needed to perform a join operation between
                 * two nodes to a new resulting parent node.
                 * @since 0.1.1
                 */
                struct joininfo
                {
                    reference_type node;        /// The child node's reference.
                    distance_type distance;     /// The node's distance to its new parent.
                };

            protected:
                buffer_type m_buffer;           /// The buffer of all nodes in tree.
                uint32_t m_leaves = 0;          /// The total number of leaves in tree.

            public:
                inline tree() noexcept = default;
                inline tree(const tree&) noexcept = default;
                inline tree(tree&&) noexcept = default;

                inline tree& operator=(const tree&) = default;
                inline tree& operator=(tree&&) = default;

                /**
                 * Retrieves a node from the tree by it's reference value.
                 * @param ref The node reference value to be retrieved.
                 * @return The requested node.
                 */
                inline const node_type& operator[](reference_type ref) const
                {
                    return m_buffer[ref];
                }

                /**
                 * Joins a pair of OTUs into a common parent node.
                 * @param parent The parent node reference to join children OTUs to.
                 * @param fst The joining information for the parent's first child.
                 * @param snd The joining information for the parent's second child.
                 */
                inline void join(reference_type parent, const joininfo& fst, const joininfo& snd)
                {
                    auto &father = m_buffer[parent];
                    const auto rheight = branch(father, fst, 0);
                    const auto lheight = branch(father, snd, 1);

                    father.level = utils::max(rheight, lheight) + 1;
                }

                /**
                 * Gives access to the tree's root node.
                 * @return The tree's root node.
                 */
                inline const node_type& root() const noexcept
                {
                    return operator[](underlying_tree::root());
                }

                /**
                 * Informs the number of leaves in the tree.
                 * @return The total amount of leaf nodes in the tree.
                 */
                inline uint32_t leaves() const noexcept
                {
                    return m_leaves;
                }

                /**
                 * Creates a new tree with given number of nodes as leaves.
                 * @param leaves The number of leaf nodes in tree.
                 * @return The newly created tree instance.
                 */
                static inline tree make(uint32_t leaves) noexcept
                {
                    return tree {buffer_type::make((leaves << 1) - 1), leaves};
                }

                /**
                 * Creates a new tree with given number of nodes as leaves.
                 * @param allocator The allocator to be used to create new dendogram.
                 * @param leaves The number of leaf nodes in tree.
                 * @return The newly created tree instance.
                 */
                static inline tree make(const museqa::allocator& allocator, uint32_t leaves) noexcept
                {
                    return tree {buffer_type::make(allocator, (leaves << 1) - 1), leaves};
                }

            protected:
                /**
                 * Updates a node's branch according to the given joining action
                 * being performed. The given node instance will be used as parent.
                 * @param parent The node to act as the new branch's parent.
                 * @param info The current joining action information.
                 * @param relation The new branch's relation id.
                 * @return The branch's current height.
                 */
                inline auto branch(node_type& parent, const joininfo& info, int relation) -> uint32_t
                {
                    auto& child = m_buffer[info.node];

                    child.parent = parent.id;
                    child.distance = info.distance;
                    parent.child[relation] = info.node;

                    return child.level;
                }

            private:
                /**
                 * Builds a new tree from an underlying tree nodes buffer. It is
                 * assumed that the given tree is empty, without relevant hierarchy.
                 * @param raw The tree's underlying buffer instance.
                 * @param leaves The number of leaf nodes in tree.
                 */
                inline tree(buffer_type&& raw, uint32_t leaves)
                :   underlying_tree {static_cast<oturef>(raw.size() - 1)}
                ,   m_buffer {std::forward<decltype(raw)>(raw)}
                ,   m_leaves {leaves}
                {
                    for(size_t i = 0; i < m_buffer.size(); ++i)
                        m_buffer[i].id = (otu) i;
                }
        };

        /**
         * We use the highest available reference in our pseudo-addressing-space
         * to represent an unknown or undefined node of the phylogenetic tree. It
         * is very unlikely that you'll ever need to fill up our whole addressing
         * space with distinct OTUs references. And if you do, well, you'll have
         * issues with this approach.
         * @since 0.1.1
         */
        enum : oturef { undefined = tree::undefined };

        /**
         * Represents a common phylogeny algorithm context.
         * @since 0.1.1
         */
        struct context
        {
            const pairwise::distance_matrix matrix;
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

            virtual auto run(const context&) const -> tree = 0;

            static auto has(const std::string&) -> bool;
            static auto make(const std::string&) -> const factory&;
            static auto list() noexcept -> const std::vector<std::string>&;
        };

        /**
         * Runs the module when not on a pipeline.
         * @param dmat The distance matrix between sequences.
         * @param algorithm The chosen phylogeny algorithm.
         * @return The chosen algorithm's resulting phylogenetic tree.
         */
        inline tree run(
                const pairwise::distance_matrix& dmat
            ,   const std::string& algorithm = "default"
            )
        {
            auto lambda = phylogeny::algorithm::make(algorithm);
            
            const phylogeny::algorithm *worker = lambda ();
            auto result = worker->run({dmat});
            
            delete worker;
            return result;
        }
    }
}
