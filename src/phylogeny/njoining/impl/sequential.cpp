/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Sequential implementation for the phylogeny module's neighbor-joining algorithm.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019-present Rodrigo Siqueira
 */
#include <cstdint>
#include <utility>

#include "node.hpp"
#include "oeis.hpp"
#include "utils.hpp"
#include "buffer.hpp"
#include "matrix.hpp"
#include "pairwise.cuh"
#include "exception.hpp"
#include "environment.h"

#include "phylogeny/matrix.cuh"
#include "phylogeny/phylogeny.cuh"
#include "phylogeny/njoining/star.cuh"
#include "phylogeny/njoining/njoining.cuh"

namespace
{
    using namespace museqa;
    using namespace phylogeny;

    /**
     * The algorithm's distance type. 
     * @since 0.1.1
     */
    using distance_type = otu::distance_type;

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
     * The point type required by the algorithm's matrices.
     * @since 0.1.1
     */
    using pair_type = typename museqa::matrix<distance_type>::point_type;

    /**
     * The matrix type for distances between OTUs.
     * @tparam T The distance matrix's spatial transformation.
     * @since 0.1.1
     */
    template <typename T>
    using distance_matrix = phylogeny::matrix<false, T>;

    /**
     * The neighbor-joining algorithm's data structures' state.
     * @tparam T The distance matrix's spatial transformation.
     * @since 0.1.1
     */
    template <typename T>
    struct state
    {
        map_type map;                   /// The OTU references map to matrix indeces.
        cache_type cache;               /// The cache of lines and columns total sums.
        distance_matrix<T> matrix;      /// The algorithm's distance matrix.
        size_t count;                   /// The number of OTUs yet to be joined.
    };

    /**
     * Builds a cache for the sum of all elements from a matrix's columns and rows.
     * @tparam T The algorithm's distance matrix's spatial transformation.
     * @param state The algorithm's state instance to initialize the sum cache of.
     */
    template <typename T>
    static void cache_init(state<T>& state)
    {
        for(size_t i = 0; i < state.count - 1; ++i) {
            state.cache[i] = (distance_type) 0;
        }

        for(size_t i = 0; i < state.count - 1; ++i) {
            for(size_t j = i + 1; j < state.count; ++j) {
                const auto current = state.matrix[{i, j}];
                state.cache[i] += current;
                state.cache[j] += current;
            }
        }
    }

    /**
     * Initialize a new algorithm state instance, and builds all data structures
     * needed for a fast neighbor-joining execution.
     * @tparam T The algorithm's distance matrix's spatial transformation.
     * @param matrix The pairwise module's distance matrix.
     * @param count The total number of OTUs to be aligned.
     * @return The initialized algorithm state instance.
     */
    template <typename T>
    static auto initialize(const pairwise::distance_matrix& matrix, size_t count) -> state<T>
    {
        state<T> state;

        state.matrix = distance_matrix<T> {matrix};
        state.map = map_type::make(count);
        state.count = count;

        onlyslaves state.cache = cache_type::make(count);
        onlyslaves cache_init(state);

        for(size_t i = 0; i < count; ++i)
            state.map[i] = (oturef) i;

        return state;
    }

    /**
     * Calculates the Q-value for the given OTU pair.
     * @tparam T The algorithm's distance matrix's spatial transformation.
     * @param state The algorithm's state data structures.
     * @param pair The target pair to get the Q-value of.
     * @return The given pair's Q-value.
     */
    template <typename T>
    inline distance_type q_transform(const state<T>& state, const pair_type& pair)
    {
        return (state.count - 2) * state.matrix[pair] - state.cache[pair.x] - state.cache[pair.y];
    }

    /**
     * Raises a candidate OTU pair into the local best joinable OTU pair.
     * @tparam T The algorithm's distance matrix's spatial transformation.
     * @param state The algorithm's state data structures.
     * @param chosen The chosen candidate as the local best OTU pair.
     * @return The fully-joinable OTU pair.
     */
    template <typename T>
    static njoining::joinable raise_candidate(const state<T>& state, const njoining::candidate& chosen)
    {
        const pair_type pair = {chosen.ref[0], chosen.ref[1]};
        const auto pairsum = state.cache[pair.x] - state.cache[pair.y];

        const distance_type dx = (.5 * state.matrix[pair]) + (pairsum / (2 * (state.count - 2)));
        const distance_type dy = state.matrix[pair] - dx;

        return {chosen, dx, dy};
    }

    /**
     * Finds the best joinable pair on the given partition.
     * @tparam T The algorithm's distance matrix's spatial transformation.
     * @param state The algorithm's state data structures.
     * @param partition The local range at which a candidate must be found.
     * @return The best joinable pair candidate found on the given partition.
     */
    template <typename T>
    static njoining::joinable pick_joinable(const state<T>& state, const range<size_t>& partition)
    {
        njoining::candidate chosen;
        size_t i = oeis::a002024(partition.offset + 1);
        size_t j = partition.offset - utils::nchoose(i);

        // Let's iterate over the partition by calculating each partition element's
        // Q-value and picking the one with the lowest value. The point with the
        // lowest Q-value is then selected to be returned.
        for(size_t c = 0; c < partition.total; ++i, j = 0)
            for( ; c < partition.total && j < i; ++c, ++j) {
                const auto distance = q_transform(state, {i, j});

                if(distance > chosen.distance)
                    chosen = {oturef(i), oturef(j), distance};
            }

        /// Now that we have our partition's best candidate, we must calculate its
        /// deltas, as the other nodes will not figure it out by themselves.
        return raise_candidate(state, chosen);
    }

    /**
     * Swaps the given pair of OTUs and removes one of them from the star tree.
     * @tparam T The algorithm's distance matrix's spatial transformation.
     * @param state The algorithm's state data structures.
     * @param keep The OTU to be swapped but kept in the star tree.
     * @param remove The OTU to be swapped and removed from the star tree.
     */
    template <typename T>
    static void swap_remove(state<T>& state, oturef keep, oturef remove)
    {
        onlyslaves {
            utils::swap(state.cache[keep], state.cache[remove]);
            state.matrix.swap(keep, remove);
            state.matrix.remove(remove);
        }

        ptrdiff_t shift = (remove == 0);
        utils::swap(state.map[keep], state.map[remove]);
        state.map = map_type {state.map.offset(shift), state.map.size() - 1};
        onlyslaves state.cache = cache_type {state.cache.offset(shift), state.cache.size() - 1};
    }

    /**
     * Updates the star tree's cache structures by removing an OTU.
     * @tparam T The algorithm's distance matrix's spatial transformation.
     * @param state The algorithm's state data structures.
     * @param x The OTU to be removed from the star tree's caches and matrix.
     */
    template <typename T>
    static void update_cache(state<T>& state, oturef x)
    {
        if(std::is_same<transform::symmetric, T>::value) {
            swap_remove(state, x, 0);
        } else {
            swap_remove(state, x, state.count - 1);
        }
    }

    /**
     * Joins an OTU pair into a new parent OTU.
     * @tparam T The algorithm's distance matrix's spatial transformation.
     * @param tree The phylogenetic tree being constructed.
     * @param parent The parent OTU into which the pair will be joined.
     * @param state The algorithm's state data structures.
     * @param join The OTU pair to join.
     */
    template <typename T>
    static void join_pair(njoining::star& tree, oturef parent, state<T>& state, const njoining::joinable& join)
    {
        const auto x = join.ref[0];
        const auto y = join.ref[1];

        distance_type new_sum = 0;

        // As updating the star tree is a computationally expensive task, we optimize
        // it by reusing one of the joined OTU's column and row on the matrix to
        // store the new OTU's distances.
        tree.join(parent, {state.map[x], join.delta[0]}, {state.map[y], join.delta[1]});

        // Let's calculate the distances between the OTU being created and the others
        // which have not been affected by the current joining operation.
        onlyslaves for(size_t i = 0; i < state.count; ++i) {
            const auto previous = state.matrix[{i, x}] + state.matrix[{i, y}];
            const auto current = .5 * (previous - state.matrix[{x, y}]);

            state.matrix[{i, x}] = state.matrix[{x, i}] = current;
            state.cache[i] += current - previous;
            new_sum += current;
        }

        // Finally, let's take advantage from our data structures' layouts and always
        // remove the cheapest column from our star tree's distance matrix.
        onlyslaves state.cache[x] = new_sum;
        state.map[x] = parent;

        update_cache(state, y);
        --state.count;
    }

    /**
     * The sequential neighbor-joining algorithm object. This algorithm uses no
     * GPU parallelism whatsoever.
     * @tparam T The algorithm's distance matrix's spatial transformation.
     * @since 0.1.1
     */
    template <typename T>
    struct sequential : public njoining::algorithm
    {
        /**
         * Builds the pseudo-phylogenetic tree from the given distance matrix.
         * @param state The algorithm's state data structures.
         * @return The calculated phylogenetic tree.
         */
        auto build_tree(state<T>& state) const -> njoining::star
        {
            oturef parent = (oturef) state.count;
            auto tree = njoining::star::make(state.count);

            // We must keep joining OTU pairs until there are only three OTUs left
            // in our star tree, so all the other OTUs have been joined.
            while(state.count > 3) {
                range<size_t> partition;
                njoining::joinable vote;

                onlyslaves if(state.count > static_cast<size_t>(node::rank)) {
                    const size_t total = utils::nchoose(state.count);

                    // Let's split the total amount of work to be done between our compute
                    // nodes. Each node must pick its local best joinable candidate.
                    #if !defined(__museqa_runtime_cython)
                        const auto workers = utils::min<size_t>(node::count - 1, state.count - 1);
                        onlyslaves partition = utils::partition(total, workers, node::rank - 1);
                    #else
                        partition = range<size_t> {0, total};
                    #endif

                    // After finding each compute node's local best joinable candidate,
                    // we must gather the votes and find the best one globally.
                    vote = pick_joinable(state, partition);
                }

                vote = this->reduce(vote);

                // At last, we join the selected pair, rebuild our distance matrix
                // with the newly created OTU, recalculate our sum cache with the
                // new OTU and update our OTU map to reflect the changes.
                join_pair(tree, parent++, state, vote);
            }

            return tree;
        }

        /**
         * Executes the sequential neighbor-joining algorithm for the phylogeny
         * step. This method is responsible for coordinating the execution.
         * @return The module's result value.
         */
        auto run(const context& ctx) const -> guidetree override
        {
            if(ctx.count < 2)
                return guidetree {};

            auto state = initialize<T>(ctx.matrix, ctx.count);
            auto result = build_tree(state);

            return result;
        }
    };
}

namespace museqa
{
    /**
     * Instantiates a new sequential neighbor-joining instance using a simple matrix.
     * @return The new algorithm instance.
     */
    extern auto phylogeny::njoining::sequential_linear() -> phylogeny::algorithm *
    {
        return new ::sequential<transform::linear<2>>;
    }

    /**
     * Instantiates a new sequential neighbor-joining instance using a symmatrix.
     * @return The new algorithm instance.
     */
    extern auto phylogeny::njoining::sequential_symmetric() -> phylogeny::algorithm *
    {
        return new ::sequential<transform::symmetric>;
    }
}
