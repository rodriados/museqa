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

#include <io.hpp>
#include <cuda.cuh>
#include <utils.hpp>
#include <database.hpp>
#include <pairwise.cuh>
#include <pipeline.hpp>
#include <allocator.hpp>
#include <dendogram.hpp>

namespace msa
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
        class tree : protected dendogram<otu, score, oturef>
        {
            protected:
                using distance_type = score;
                using reference_type = oturef;
                using underlying_type = dendogram<otu, score, oturef>;

            public:
                static constexpr reference_type undefined = node_type::undefined;

            protected:
                /**
                 * Gathers all information needed to join a node to a new parent.
                 * @since 0.1.1
                 */
                struct joininfo
                {
                    reference_type node;        /// The child node's reference.
                    distance_type distance;     /// The node's distance to its new parent.
                };

            public:
                inline tree() noexcept = default;
                inline tree(const tree&) noexcept = default;
                inline tree(tree&&) noexcept = default;

                inline tree& operator=(const tree&) = default;
                inline tree& operator=(tree&&) = default;

                using underlying_type::operator[];

                /**
                 * Joins a pair of OTUs into a common parent.
                 * @param parent The parent node reference to join children OTUs to.
                 * @param fst The joining information for the parent's first child.
                 * @param snd The joining information for the parent's second child.
                 */
                inline void join(reference_type parent, const joininfo& fst, const joininfo& snd)
                {
                    auto& father = underlying_type::operator[](parent);

                    const auto rheight = branch(father, fst, 0);
                    const auto lheight = branch(father, snd, 1);
                    father.level = utils::max(rheight, lheight) + 1;
                }

                using underlying_type::leaves;

                /**
                 * Creates a new tree with given number of nodes as leaves.
                 * @param leaves The number of leaf nodes in tree.
                 * @return The newly created tree instance.
                 */
                static inline tree make(uint32_t leaves) noexcept
                {
                    return tree {underlying_type::make(leaves)};
                }

                /**
                 * Creates a new tree with given number of nodes as leaves.
                 * @param allocator The allocator to be used to create new dendogram.
                 * @param leaves The number of leaf nodes in tree.
                 * @return The newly created tree instance.
                 */
                static inline tree make(const msa::allocator& allocator, uint32_t leaves) noexcept
                {
                    return tree {underlying_type::make(allocator, leaves)};
                }

            protected:
                /**
                 * Builds a new tree from an underlying dendogram. It is assumed
                 * that the given dendogram is empty, without relevant hierarchy.
                 * @param raw The tree's underlying dendogram instance.
                 */
                inline tree(underlying_type&& raw)
                : underlying_type {std::forward<decltype(raw)>(raw)}
                {
                    for(size_t i = 0; i < this->m_size; ++i)
                        this->m_ptr[i].contents = (otu) i;
                }

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
                    auto& child = underlying_type::operator[](info.node);

                    child.parent = parent.contents;
                    child.distance = info.distance;
                    parent.child[relation] = info.node;

                    return child.level;
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
            const size_t total;
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
         * Defines the module's conduit. This conduit is composed of the sequences
         * being aligned and the phylogenetic tree to guide their alignment.
         * @since 0.1.1
         */
        struct conduit : public pipeline::conduit
        {
            const pointer<msa::database> db;        /// The loaded sequences' database.
            const tree phylotree;                   /// The sequences' alignment guiding tree.
            const size_t total;                     /// The total number of sequences.

            inline conduit() noexcept = delete;
            inline conduit(const conduit&) = default;
            inline conduit(conduit&&) = default;

            /**
             * Instantiates a new conduit.
             * @param db The sequence database to transfer to the next module.
             * @param ptree The alignment guiding tree to transfer to the next module.
             */
            inline conduit(const pointer<msa::database>& db, const tree& ptree) noexcept
            :   db {db}
            ,   phylotree {ptree}
            ,   total {db->count()}
            {}

            inline conduit& operator=(const conduit&) = delete;
            inline conduit& operator=(conduit&&) = delete;
        };

        /**
         * Defines the module's pipeline manager. This object will be the one responsible
         * for checking and managing the module's execution when on a pipeline.
         * @since 0.1.1
         */
        struct module : public pipeline::module
        {
            using previous = pairwise::module;      /// Indicates the expected previous module.
            using conduit = phylogeny::conduit;     /// The module's conduit type.

            using pipe = pointer<pipeline::conduit>;
            
            /**
             * Returns an string identifying the module's name.
             * @return The module's name.
             */
            inline auto name() const -> const char * override
            {
                return "phylogeny";
            }

            auto run(const io::service&, const pipe&) const -> pipe override;
            auto check(const io::service&) const -> bool override;
        };

        /**
         * Runs the module when not on a pipeline.
         * @param dmat The distance matrix between sequences.
         * @param total The total number of sequences being aligned.
         * @param algorithm The chosen phylogeny algorithm.
         * @return The chosen algorithm's resulting phylogenetic tree.
         */
        inline tree run(
                const pairwise::distance_matrix& dmat
            ,   const size_t total
            ,   const std::string& algorithm = "default"
            )
        {
            auto lambda = phylogeny::algorithm::make(algorithm);
            
            const phylogeny::algorithm *worker = lambda ();
            auto result = worker->run({dmat, total});
            
            delete worker;
            return result;
        }
    }
}
