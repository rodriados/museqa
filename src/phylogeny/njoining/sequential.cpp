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
     * Defines a cache for the matrix's columns and row sums.
     * @since 0.1.1
     */
    using cache_type = buffer<distance_type>;

    /**
     * Defines the algorithm's required base matrix type.
     * @since 0.1.1
     */
    using base_matrix = msa::matrix<distance_type>;

    /**
     * The point type required by the algorithm's matrices.
     * @since 0.1.1
     */
    using point_type = typename base_matrix::point_type;

    /**
     * Builds a cache for the sum of all elements from a matrix's columns and rows.
     * @tparam M The target matrix type from which the sums must be calculated.
     * @param matrix The target matrix to have its elements summed up.
     * @param total The total number of elements on the matrix's rows and columns.
     * @return The built cache instance.
     */
    template <typename M>
    static cache_type init_cache(const M& matrix, size_t total)
    {
        auto cache = cache_type::make(total);

        for(size_t i = 0; i < total - 1; ++i) {
            for(size_t j = i + 1; j < total; ++j) {
                const auto current = matrix[{i, j}];
                cache[i] += current;
                cache[j] += current;
            }
        }

        return cache;
    }

    /**
     * Updates the algorithm's data structures to the newly joined OTU pair. This
     * function assumes a linear distance matrix is being used.
     * @param matrix The linear distance matrix's instance.
     * @param cache The distance matrix's column and row sums.
     * @param map The matrix's columns OTU reference map.
     * @param pair The pair of OTUs being currently joined.
     * @param count The current number of OTU's represented on matrix.
     * @return The new OTU's chosen column index on matrix.
     */
    static size_t update_matrix(
            phylogeny::matrix<false>& matrix
        ,   cache_type& cache
        ,   buffer<oturef>& map
        ,   const point_type& pair
        ,   size_t count
        )
    {
        distance_type sum = 0;
        const auto otudist = matrix[pair];

        // As updating the distance matrix is an expensive task, we will optimize
        // it by reusing one of the joined OTU's column and row on the matrix to
        // store the new OTU's distances. As we're currently dealing with a linear
        // matrix, we will reuse the space of the x-OTU, which is at a lower index,
        // and remove the y-OTU completely from the matrix.

        // Let's calculate the distances between the OTU being created and the others
        // which have not been affected by the current joining operation.
        for(size_t i = 0; i < count; ++i) {
            const auto previous = matrix[{i, pair.x}] + matrix[{i, pair.y}];
            const auto current = .5 * (previous - otudist);

            matrix[{i, pair.x}] = matrix[{pair.x, i}] = current;
            onlyslaves cache[i] += current - previous;
            sum += current;
        }

        // Taking advantage of our linear matrix, we will move the y-OTU to the
        // matrix's last column, which is the cheapest to be removed.
        matrix.swap(pair.y, count - 1);
        matrix.remove(count - 1);

        // Finally, at last, we commit the changes to the matrix by adjusting the
        // matrix's OTUs map and the columns' sum cache.
        utils::swap(map[pair.y], map[count - 1]);
        onlyslaves utils::swap(cache[pair.y], cache[count - 1]);
        onlyslaves cache[pair.x] = sum;

        return pair.x;
    }

    /**
     * Updates the algorithm's data structures to the newly joined OTU pair. This
     * function assumes a symmetric distance matrix is being used.
     * @param matrix The symmetric distance matrix's instance.
     * @param cache The distance matrix's column and row sums.
     * @param map The matrix's columns OTU reference map.
     * @param pair The pair of OTUs being currently joined.
     * @param count The current number of OTU's represented on matrix.
     * @return The new OTU's chosen column index on matrix.
     */
    static size_t update_matrix(
            phylogeny::symmatrix<false>& matrix
        ,   cache_type& cache
        ,   buffer<oturef>& map
        ,   const point_type& pair
        ,   size_t count
        )
    {
        distance_type sum = 0;
        const auto otudist = matrix[pair];

        // As updating the distance matrix is an expensive task, we will optimize
        // it by reusing one of the joined OTU's column and row on the matrix to
        // store the new OTU's distances. As we're currently dealing with a linear
        // matrix, we will reuse the space of the x-OTU, which is at a lower index,
        // and remove the y-OTU completely from the matrix.

        // Let's calculate the distances between the OTU being created and the others
        // which have not been affected by the current joining operation.
        for(size_t i = 0; i < count; ++i) {
            const auto previous = matrix[{i, pair.x}] + matrix[{i, pair.y}];
            const auto current = .5 * (previous - otudist);

            matrix[{i, pair.x}] = matrix[{pair.x, i}] = current;
            onlyslaves cache[i] += current - previous;
            sum += current;
        }

        // Taking advantage of our linear matrix, we will move the y-OTU to the
        // matrix's last column, which is the cheapest to be removed.
        matrix.swap(0, pair.y);
        matrix.remove(0);

        // Finally, at last, we commit the changes to the matrix by adjusting the
        // matrix's OTUs map and the columns' sum cache.
        utils::swap(map[0], map[pair.y]);
        onlyslaves utils::swap(cache[0], cache[pair.y]);
        onlyslaves cache[pair.x] = sum;

        for(size_t i = 0; i < count - 1; ++i) {
            map[i] = map[i + 1];
            onlyslaves cache[i] = cache[i + 1];
        }

        return pair.x;
    }

    /**
     * Calculates the Q-value of the given point.
     * @tparam M The algorithm's matrix type.
     * @param matrix The distance matrix's instance.
     * @param sum The distance matrix's column and row sums.
     * @param point The target point.
     * @param count The total number of OTUs in matrix.
     * @return The point's Q-value.
     */
    template <typename M>
    inline distance_type find_q(const M& matrix, const cache_type& sum, const point_type& point, size_t count)
    {
        return (count - 2) * matrix[point] - sum[point.x] - sum[point.y];
    }

    /**
     * Finds the best joinable pair on the given partition.
     * @tparam M The algorithm's matrix type.
     * @param matrix The distance matrix's instance.
     * @param sum The distance matrix's column and row sums.
     * @param partition The local range at which a candidate must be found.
     * @param count The total number of OTUs in matrix.
     * @return The best joinable pair candidate found on the given partition.
     */
    template <typename M>
    static njoining::joinable pick_joinable(
            const M& matrix
        ,   const cache_type& sum
        ,   const range<size_t>& partition
        ,   size_t count
        )
    {
        njoining::joinable chosen;
        size_t i = oeis::a002024(partition.offset + 1);
        size_t j = partition.offset - utils::nchoose(i);

        // Let's iterate over the partition by calculating each partition element's
        // Q-value and picking the one with the lowest value. The point with the
        // lowest Q-value is then selected to be returned.
        for(size_t c = 0; c < partition.total; ++i, j = 0)
            for(; c < partition.total && j < i; ++c, ++j) {
                const score distance = find_q(matrix, sum, {i, j}, count);

                if(distance < chosen.distance) {
                    chosen.distance = distance;
                    chosen.ref[0] = i;
                    chosen.ref[1] = j;
                }
            }

        /// Now that we have our partition's best candidate, we must calculate its
        /// deltas, as the other nodes cannot figure it out by themselves.
        const point_type pt = {chosen.ref[0], chosen.ref[1]};
        chosen.delta[0] = (.5 * matrix[pt]) + ((sum[pt.x] - sum[pt.y]) / (2 * (count - 2)));
        chosen.delta[1] = matrix[pt] - chosen.delta[0];

        return chosen;
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
        using matrix_type = M;
        static_assert(std::is_base_of<base_matrix, M>::value, "the given type is not a valid matrix");

        /**
         * Builds the pseudo-phylogenetic tree from the given distance matrix.
         * @param matrix The distance matrix to build tree from.
         * @param count The total number of leaves in tree.
         * @return The calculated phylogenetic tree.
         */
        auto build_tree(matrix_type& matrix, size_t count) const -> tree
        {
            cache_type cache;
            oturef parent = (otu) count;

            auto phylotree = tree::make(count);
            auto map = buffer<oturef>::make(count);

            // As an initialization step, we must first create our sum cache with 
            // the initial sums of the original distance matrix.
            onlyslaves cache = init_cache(matrix, count);

            // Also, we must initialize our OTU reference map, so have flexibility
            // when dealing with changes to our distance matrix.
            for(size_t i = 0; i < count; ++i)
                map[i] = (otu) i;

            // We must keep joining OTU pairs until there are only two OTUs left
            // in our distance matrix, so all the others have been joined.
            for(size_t i = count; i > 2; --i, ++parent) {
                range<size_t> partition;
                njoining::joinable vote;

                const size_t total = utils::nchoose(i);

                // Let's split the total amount of work to be done between our compute
                // nodes. Each node must pick its local best joinable candidate.
                #if !defined(__msa_runtime_cython)
                    onlyslaves partition = utils::partition(total, node::count - 1, node::rank - 1);
                #else
                    partition = range<size_t> {0, total};
                #endif

                // After finding each compute node's local best joinable candidate,
                // we must gather the votes and find the best one globally.
                onlyslaves vote = pick_joinable(matrix, cache, partition, i);
                           vote = this->reduce(vote);

                const auto& ref = vote.ref;
                const auto& delta = vote.delta;

                // At last, we join the selected pair, rebuild our distance matrix
                // with the newly created OTU, recalculate our sum cache with the
                // new OTU and update our OTU map to reflect the changes.
                phylotree.join(parent, {map[ref[0]], delta[0]}, {map[ref[1]], delta[1]});
                const auto col = update_matrix(matrix, cache, map, {ref[0], ref[1]}, i);

                map[col] = parent;
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
            auto matrix = matrix_type {ctx.matrix};
            auto result = build_tree(matrix, ctx.total);

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
    extern auto phylogeny::njoining::sequential_matrix() -> phylogeny::algorithm *
    {
        return new ::sequential<phylogeny::matrix<false>>;
    }

    /**
     * Instantiates a new sequential neighbor-joining instance using a symmatrix.
     * @return The new algorithm instance.
     */
    extern auto phylogeny::njoining::sequential_symmatrix() -> phylogeny::algorithm *
    {
        return new ::sequential<phylogeny::symmatrix<false>>;
    }
}
