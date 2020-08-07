/**
 * Multiple Sequence Alignment sequential neighbor-joining file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019-2020 Rodrigo Siqueira
 */
#include <cstdint>
#include <utility>

#include <node.hpp>
#include <oeis.hpp>
#include <utils.hpp>
#include <buffer.hpp>
#include <matrix.hpp>
#include <pairwise.cuh>
#include <exception.hpp>
#include <environment.h>

#include <phylogeny/matrix.cuh>
#include <phylogeny/phylogeny.cuh>
#include <phylogeny/algorithm/njoining.cuh>

namespace
{
    using namespace msa;
    using namespace phylogeny;

    /**
     * The algorithm's distance type. 
     * @since 0.1.1
     */
    using distance_type = pairwise::score;

    /**
     * The type for mapping an OTU to its coordinates on the matrix.
     * @since 0.1.1
     */
    using map_type = buffer<oturef>;

    /**
     * Defines a cache for the matrix's columns and row sums.
     * @since 0.1.1
     */
    using cache_type = buffer<distance_type>;

    /**
     * The neighbor-joining algorithm's star tree data structures.
     * @since 0.1.1
     */
    template <typename M>
    struct startree
    {
        M matrix;           /// The algorithm's distance matrix.
        map_type map;       /// The OTU references map to matrix indeces.
        cache_type cache;   /// The cache of lines and columns total sums.
        size_t count;       /// The number of OTUs yet to be joined.
    };

    /**
     * Defines the algorithm's required base matrix type.
     * @since 0.1.1
     */
    using base_matrix = msa::matrix<distance_type>;

    /**
     * The point type required by the algorithm's matrices.
     * @since 0.1.1
     */
    using pair_type = typename base_matrix::point_type;

    /**
     * Builds a cache for the sum of all elements from a matrix's columns and rows.
     * @tparam M The target matrix type from which the sums must be calculated.
     * @param star The algorithm's star tree to initialize the sum cache of.
     */
    template <typename M>
    static void cache_init(startree<M>& star)
    {
        for(size_t i = 0; i < star.count - 1; ++i) {
            for(size_t j = i + 1; j < star.count; ++j) {
                const auto current = star.matrix[{i, j}];
                star.cache[i] += current;
                star.cache[j] += current;
            }
        }
    }

    /**
     * Initialize a new star tree, and builds all data structures needed for a fast
     * neighbor-joining execution.
     * @tparam M The distance matrix type to be used.
     * @param matrix The pairwise module's distance matrix.
     * @param count The total number of OTUs to be aligned.
     * @return The initialized star tree.
     */
    template <typename M>
    static auto initialize(const pairwise::distance_matrix& matrix, size_t count) -> startree<M>
    {
        startree<M> star;

        star.count = count;
        star.matrix = M {matrix};
        star.map = map_type::make(count);

        onlyslaves star.cache = cache_type::make(count);
        onlyslaves cache_init(star);

        for(size_t i = 0; i < count; ++i)
            star.map[i] = (otu) i;

        return star;
    }

    /**
     * Calculates the Q-value for the given OTU pair.
     * @tparam M The algorithm's matrix type.
     * @param star The OTUs' star tree data structures.
     * @param pair The target pair to get the Q-value of.
     * @return The given pair's Q-value.
     */
    template <typename M>
    inline distance_type calculate_q(const startree<M>& star, const pair_type& pair)
    {
        return (star.count - 2) * star.matrix[pair] - star.cache[pair.x] - star.cache[pair.y];
    }

    /**
     * Finds the best joinable pair on the given partition.
     * @tparam M The algorithm's matrix type.
     * @param star The OTUs' star tree data structures.
     * @param partition The local range at which a candidate must be found.
     * @return The best joinable pair candidate found on the given partition.
     */
    template <typename M>
    static njoining::joinable pick_joinable(const startree<M>& star, const range<size_t>& partition)
    {
        njoining::joinable chosen;
        size_t i = oeis::a002024(partition.offset + 1);
        size_t j = partition.offset - utils::nchoose(i);

        // Let's iterate over the partition by calculating each partition element's
        // Q-value and picking the one with the lowest value. The point with the
        // lowest Q-value is then selected to be returned.
        for(size_t c = 0; c < partition.total; ++i, j = 0)
            for( ; c < partition.total && j < i; ++c, ++j) {
                const auto distance = calculate_q(star, {i, j});

                if(distance < chosen.distance) {
                    chosen.distance = distance;
                    chosen.ref[0] = i;
                    chosen.ref[1] = j;
                }
            }

        /// Now that we have our partition's best candidate, we must calculate its
        /// deltas, as the other nodes will not figure it out by themselves.
        const pair_type pair = {chosen.ref[0], chosen.ref[1]};
        const auto pairsum = star.cache[pair.x] - star.cache[pair.y];

        chosen.delta[0] = (.5 * star.matrix[pair]) + (pairsum / (2 * (star.count - 2)));
        chosen.delta[1] = star.matrix[pair] - chosen.delta[0];

        return chosen;
    }

    /**
     * Swaps the given pair of OTUs and removes one of them from the star tree.
     * @tparam M The algorithm's matrix type.
     * @param star The OTU's star tree data structures.
     * @param keep The OTU to be swapped but kept in the star tree.
     * @param remove The OTU to be swapped and removed from the star tree.
     */
    template <typename M>
    static void swap_remove(startree<M>& star, oturef keep, oturef remove)
    {
        onlyslaves utils::swap(star.cache[keep], star.cache[remove]);
        utils::swap(star.map[keep], star.map[remove]);
        star.matrix.swap(keep, remove);
        star.matrix.remove(remove);

        for(size_t i = remove; i < star.count - 1; ++i) {
            onlyslaves star.cache[i] = star.cache[i + 1];
            star.map[i] = star.map[i + 1];
        }
    }

    /**
     * Updates the star tree's cache structures by removing an OTU.
     * @tparam M The algorithm's matrix type.
     * @param star The OTU's star tree data structures.
     * @param x The OTU to be removed from the star tree's caches and matrix.
     */
    template <typename M>
    static void update_cache(startree<M>& star, oturef x)
    {
        if(std::is_same<phylogeny::symmatrix<false>, M>::value) {
            swap_remove(star, x, 0);
        } else {
            swap_remove(star, x, star.count - 1);
        }
    }

    /**
     * Joins an OTU pair into a new parent OTU.
     * @tparam M The algorithm's matrix type.
     * @param phylotree The phylogenetic tree being constructed.
     * @param parent The parent OTU into which the pair will be joined.
     * @param star The OTU's star tree data structures.
     * @param join The OTU pair to join.
     */
    template <typename M>
    static void join_pair(tree& phylotree, oturef parent, startree<M>& star, const njoining::joinable& join)
    {
        const auto x = join.ref[0];
        const auto y = join.ref[1];

        distance_type new_sum = 0;

        // As updating the star tree is a computationally expensive task, we optimize
        // it by reusing one of the joined OTU's column and row on the matrix to
        // store the new OTU's distances.
        phylotree.join(parent, {star.map[x], join.delta[0]}, {star.map[y], join.delta[1]});

        // Let's calculate the distances between the OTU being created and the others
        // which have not been affected by the current joining operation.
        for(size_t i = 0; i < star.count; ++i) {
            const auto previous = star.matrix[{i, x}] + star.matrix[{i, y}];
            const auto current = .5 * (previous - star.matrix[{x, y}]);

            star.matrix[{i, x}] = star.matrix[{x, i}] = current;
            onlyslaves star.cache[i] += current - previous;
            new_sum += current;
        }

        // Finally, let's take advantage from our data structures' layouts and always remove
        // the cheapest column from our star tree's distance matrix.
        onlyslaves star.cache[x] = new_sum;
        star.map[x] = parent;

        update_cache(star, y);
        --star.count;
    }

    /**
     * The sequential neighbor-joining algorithm object. This algorithm uses no
     * GPU parallelism whatsoever.
     * @tparam M The matrix type to use within the algorithm.
     * @since 0.1.1
     */
    template <typename M>
    struct sequential : public njoining::algorithm
    {
        static_assert(std::is_base_of<base_matrix, M>::value, "the given type is not a valid matrix");

        /**
         * Builds the pseudo-phylogenetic tree from the given distance matrix.
         * @param matrix The distance matrix to build tree from.
         * @param count The total number of leaves in tree.
         * @return The calculated phylogenetic tree.
         */
        auto build_tree(startree<M>& star) const -> tree
        {
            oturef parent = (otu) star.count;
            auto phylotree = tree::make(star.count);

            // We must keep joining OTU pairs until there are only three OTUs left
            // in our star tree, so all the other OTUs have been joined.
            while(star.count > 2) {
                range<size_t> partition;
                njoining::joinable vote;

                const size_t total = utils::nchoose(star.count);

                // Let's split the total amount of work to be done between our compute
                // nodes. Each node must pick its local best joinable candidate.
                #if !defined(__msa_runtime_cython)
                    onlyslaves partition = utils::partition(total, node::count - 1, node::rank - 1);
                #else
                    partition = range<size_t> {0, total};
                #endif

                // After finding each compute node's local best joinable candidate,
                // we must gather the votes and find the best one globally.
                onlyslaves vote = pick_joinable(star, partition);
                           vote = this->reduce(vote);

                // At last, we join the selected pair, rebuild our distance matrix
                // with the newly created OTU, recalculate our sum cache with the
                // new OTU and update our OTU map to reflect the changes.
                join_pair(phylotree, parent++, star, vote);
            }

            return phylotree;
        }

        /**
         * Executes the sequential neighbor-joining algorithm for the phylogeny
         * step. This method is responsible for coordinating the execution.
         * @return The module's result value.
         */
        auto run(const context& ctx) const -> tree override
        {
            auto star = initialize<M>(ctx.matrix, ctx.total);
            auto result = build_tree(star);

            return result;
        }
    };
}

namespace msa
{
    /**
     * Instantiates a new sequential neighbor-joining instance using a simple matrix.
     * @return The new algorithm instance.
     */
    extern auto phylogeny::njoining::sequential_mat() -> phylogeny::algorithm *
    {
        return new ::sequential<phylogeny::matrix<false>>;
    }

    /**
     * Instantiates a new sequential neighbor-joining instance using a symmatrix.
     * @return The new algorithm instance.
     */
    extern auto phylogeny::njoining::sequential_sym() -> phylogeny::algorithm *
    {
        return new ::sequential<phylogeny::symmatrix<false>>;
    }
}
