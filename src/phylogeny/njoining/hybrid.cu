/**
 * Multiple Sequence Alignment hybrid neighbor-joining file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019-2020 Rodrigo Siqueira
 */
#include <limits>
#include <cstdint>
#include <utility>

#include <cuda.cuh>
#include <node.hpp>
#include <oeis.hpp>
#include <utils.hpp>
#include <buffer.hpp>
#include <matrix.hpp>
#include <pairwise.cuh>
#include <exception.hpp>
#include <transform.hpp>
#include <environment.h>

#include <phylogeny/matrix.cuh>
#include <phylogeny/phylogeny.cuh>
#include <phylogeny/algorithm/njoining.cuh>

namespace
{
    using namespace msa;
    using namespace phylogeny;

    /*
     * Algorithm configuration parameters. These values interfere directly into
     * the algorithm's execution, thus, they shall be modified with caution.
     */
    enum : size_t { reduce_factor = 2 };

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
     * The point type required by the algorithm's matrices.
     * @since 0.1.1
     */
    using pair_type = typename msa::matrix<distance_type>::point_type;

    /**
     * The neighbor-joining algorithm's star tree data structures.
     * @tparam T The star tree's matrix spatial transformation type.
     * @since 0.1.1
     */
    template <typename T>
    struct startree
    {
        phylogeny::matrix<true, T> matrix;  /// The algorithm's distance matrix.
        cache_type cache;                   /// The cache of lines and columns total sums.
        map_type map;                       /// The OTU references map to matrix indeces.
        size_t count;                       /// The number of OTUs yet to be joined.
    };

    /**
     * The reduceable interface. This interface's implementations shall be forced
     * to have a method to join two elements at the given offsets together.
     * @tparam T The type of elements to be reduced.
     * @since 0.1.1
     */
    template <typename T>
    struct reduceable
    {
        __device__ static inline void join(volatile T *, size_t, size_t);
    };

    /**
     * Calculates the highest number which is a power of 2 and is smaller than or
     * equal to the given input.
     * @param x The target input number.
     * @return The resulting power of 2.
     */
    static inline uint32_t floor_power2(uint32_t x) noexcept
    {
      #if !defined(__msa_compiler_gnuc)
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        x |= x >> 32;
        return x ^ (x >> 1);
      #else
        return 0x80000000 >> __builtin_clz(x);
      #endif
    }

    /**
     * Performs a reduce inside a single warp.
     * @tparam N The number of total elements being reduced.
     * @tparam T The type of elements being reduced.
     * @param data The data array being reduced into a single value.
     * @param offset The current thread's offset to process.
     */
    template <typename R, typename T>
    __device__ inline void reduce(volatile T *data, size_t count, size_t offset)
    {
        static_assert(std::is_base_of<reduceable<T>, R>::value, "invalid reduceable type");

        switch(count) {
            case 1024: if(offset < 512) { R::join(data, offset, offset + 512); } __syncthreads();
            case  512: if(offset < 256) { R::join(data, offset, offset + 256); } __syncthreads();
            case  256: if(offset < 128) { R::join(data, offset, offset + 128); } __syncthreads();
            case  128: if(offset <  64) { R::join(data, offset, offset +  64); } __syncthreads();
            case   64: if(offset <  32) { R::join(data, offset, offset +  32); }
            case   32: if(offset <  16) { R::join(data, offset, offset +  16); }
            case   16: if(offset <   8) { R::join(data, offset, offset +   8); }
            case    8: if(offset <   4) { R::join(data, offset, offset +   4); }
            case    4: if(offset <   2) { R::join(data, offset, offset +   2); }
            case    2: if(offset <   1) { R::join(data, offset, offset +   1); }
        }
    }

    /**
     * Fills the star tree's distances sum cache on device memory.
     * @tparam T The star tree's matrix spatial transformation type.
     * @param star The star tree's object instance.
     */
    template <typename T>
    __global__ void fill_cache(startree<T> star)
    {
        extern __shared__ distance_type sums[];

        // Implements the reduction operation for filling the star tree's matrix
        // columns and lines cache. As we are interested on building a cache with
        // the total sum of the matrix's lines and columns, this operation shall
        // simply accumulate by summing two data offsets into one.
        using sum = struct : reduceable<distance_type> {
            __device__ static inline void join(volatile distance_type *data, size_t dest, size_t src) {
                data[dest] += data[src];
            }
        };

        // For each of the star tree's matrix's columns and lines, we must iterate
        // over their elements and sum them all together in order to fill our cache.
        for(int32_t i = blockIdx.x; i < star.count; i += gridDim.x) {
            sums[threadIdx.x] = 0;

            // As we cannot spawn a thread to every single element of our cache
            // or distance matrix, we must "manually" sum every exceeding element
            // so that these elements are included when we reduce our shared array.
            for(int32_t j = threadIdx.x; j < star.count; j += blockDim.x)
                sums[threadIdx.x] += star.matrix[{i, j}];

            __syncthreads();

            // Now that out shared array is ready to be reduced, and all exceeding
            // elements have been summed, we can perform our reduce operation.
            reduce<sum>(sums, blockDim.x, threadIdx.x);

            if(threadIdx.x == 0)
                star.cache[i] = sums[0];
        }
    }

    /**
     * Builds a cache for the sum of all elements from a matrix's columns and rows.
     * @tparam T The star tree's matrix spatial transformation type.
     * @param star The algorithm's star tree to initialize the sum cache of.
     */
    template <typename T>
    static void cache_init(startree<T>& star)
    {
        using namespace cuda::device;
        const auto height = (int32_t) star.matrix.dimension()[0];
        const auto width  = (int32_t) star.matrix.dimension()[1];

        // The number of threads spawned by each block to initialize our cache will
        // be a power of 2 roughly equal to half the width of our matrix. We force
        // such specific number in order to take the most out of our reduce kernel.
        const auto blocks  = max_blocks(height);
        const auto threads = floor_power2(max_threads(width / reduce_factor));

        fill_cache<<<blocks, threads, sizeof(distance_type) * threads>>>(star);
    }

    /**
     * Initialize a new star tree, and builds all data structures needed for a fast
     * neighbor-joining execution.
     * @tparam T The star tree's matrix spatial transformation type.
     * @param matrix The pairwise module's distance matrix.
     * @param count The total number of OTUs to be aligned.
     * @return The initialized star tree.
     */
    template <typename T>
    static auto initialize(const pairwise::distance_matrix& matrix, size_t count) -> startree<T>
    {
        startree<T> star;
        auto hmat = phylogeny::matrix<false, T> {matrix};

        star.count = count;
        star.matrix = hmat.to_device();
        star.map = map_type::make(count);

        onlyslaves star.cache = cache_type::make(cuda::allocator::device, star.count);
        onlyslaves cache_init(star);

        for(size_t i = 0; i < count; ++i)
            star.map[i] = (otu) i;

        return star;
    }

    /**
     * Calculates the Q-value for the given OTU pair.
     * @tparam T The star tree's matrix spatial transformation type.
     * @param star The OTUs' star tree data structures.
     * @param pair The target pair to get the Q-value of.
     * @return The given pair's Q-value.
     */
    template <typename T>
    __device__ inline distance_type q_transform(const startree<T>& star, const pair_type& pair)
    {
        return (star.count - 2) * star.matrix[pair] - star.cache[pair.x] - star.cache[pair.y];
    }

    /**
     * Raises a candidate OTU pair into the local best joinable OTU pair.
     * @tparam T The star tree's matrix spatial transformation type.
     * @param star The OTUs' star tree data structures.
     * @param chosen The chosen candidate as the local best OTU pair.
     * @return The fully-joinable OTU pair.
     */
    template <typename T>
    __device__ inline njoining::joinable raise_candidate(
            const startree<T>& star
        ,   const njoining::candidate& chosen
        )
    {
        const pair_type pair = {chosen.ref[0], chosen.ref[1]};
        const auto pairsum = star.cache[pair.x] - star.cache[pair.y];

        const distance_type dx = (.5 * star.matrix[pair]) + (pairsum / (2 * (star.count - 2)));
        const distance_type dy = star.matrix[pair] - dx;

        return {chosen, dx, dy};
    }

    /**
     * Finds the local best OTU candidates to be joined.
     * @tparam T The star tree's matrix spatial transformation type.
     * @param result The kernel's final result buffer.
     * @param star The OTUs' star tree data structures.
     * @param partition The local range at which a candidate must be found.
     */
    template <typename T>
    __global__ void find_candidates(
            buffer<njoining::joinable> result
        ,   const startree<T> star
        ,   const range<size_t> partition
        )
    {
        extern __shared__ njoining::candidate list[];
        new (&list[threadIdx.x]) njoining::candidate {};

        // Implements a reduction operation for finding the local best OTU pair
        // to be joined next. As we apply the Q-transformation for every pair on
        // the distance matrix, we must find the one with the smallest Q-value.
        using min = struct : reduceable<njoining::candidate> {
            __device__ static inline void join(volatile njoining::candidate *data, size_t dest, size_t src) {
                if(data[src].distance < data[dest].distance) data[dest] = data[src];
            }
        };

        // As we cannot spawn a thread for every single pair we must calculate,
        // we have to process the excess before reducing our shared array.
        for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < partition.total; i += gridDim.x * blockDim.x) {
            const size_t x = oeis::a002024(partition.offset + i + 1);
            const size_t y = (partition.offset + i) - utils::nchoose(x);
            const auto distance = q_transform(star, {x, y});

            if(distance < list[threadIdx.x].distance)
                list[threadIdx.x] = njoining::candidate {x, y, distance};
        }

        __syncthreads();

        // Reduces the shared list of candidates to find the absolute local best
        // on the current device. The list has already been reduced to a smaller
        // amount due to the operation performed above.
        reduce<min>(list, blockDim.x, threadIdx.x);

        if(threadIdx.x == 0)
            result[blockIdx.x] = raise_candidate(star, list[0]);
    }

    /**
     * Finds the best joinable pair on the given partition.
     * @tparam T The star tree's matrix spatial transformation type.
     * @param star The OTUs' star tree data structures.
     * @param partition The local range at which a candidate must be found.
     * @return The best joinable pair candidate found on the given partition.
     */
    template <typename T>
    static njoining::joinable pick_joinable(const startree<T>& star, const range<size_t>& partition)
    {
        using namespace cuda::device;

        // The number of threads spawned by each block to find the local best joinable
        // OTU will be a power of 2 roughly equal to half the partition size. Also,
        // we only spawn new blocks if all of its threads will be used.
        const size_t total = partition.total / reduce_factor;
        const auto threads = floor_power2(max_threads(total));
        const auto blocks  = max_blocks(total / threads);

        auto result = buffer<njoining::joinable>::make(blocks);
        auto chosen = buffer<njoining::joinable>::make(cuda::allocator::device, blocks);
        size_t smallest = 0;

        find_candidates<<<blocks, threads, sizeof(njoining::candidate) * threads>>>(chosen, star, partition);
        cuda::memory::copy(result.raw(), chosen.raw(), blocks);

        // Now that we reduced the total number of candidates, we can finally apply
        // a small reduction to find the absolute best on this node's partition.
        for(size_t i = 1; i < blocks; ++i)
            if(result[i].distance < result[smallest].distance)
                smallest = i;

        return result[smallest];
    }

    /**
     * Swaps the given pair of OTUs and removes one of them from the star tree.
     * @tparam T The star tree's matrix spatial transformation type.
     * @param star The OTU's star tree data structures.
     * @param keep The OTU to be swapped but kept in the star tree.
     * @param remove The OTU to be swapped and removed from the star tree.
     */
    template <typename T>
    static void swap_remove(startree<T>& star, oturef keep, oturef remove)
    {
        onlyslaves {
            star.matrix.swap(keep, remove);
            star.matrix.remove(remove);
        }

        ptrdiff_t shift = (remove == 0);
        utils::swap(star.map[keep], star.map[remove]);
        star.map = map_type {star.map.offset(shift), star.map.size() - 1};
        onlyslaves star.cache = cache_type {star.cache.offset(shift), star.cache.size() - 1};
    }

    /**
     * Updates the star tree's cache structures by removing an OTU.
     * @tparam T The star tree's matrix spatial transformation type.
     * @param star The OTU's star tree data structures.
     * @param x The OTU to be removed from the star tree's caches and matrix.
     */
    template <typename T>
    static void update_cache(startree<T>& star, oturef x)
    {
        if(std::is_same<transform::symmetric, T>::value) {
            swap_remove(star, x, 0);
        } else {
            swap_remove(star, x, star.count - 1);
        }
    }

    /**
     * Rebuils the star tree's matrix and cache while joining neighboring OTUs.
     * @tparam T The star tree's matrix spatial transformation type.
     * @param star The OTU's star tree data structures.
     * @param pair The pair of OTUs being currently joined.
     */
    template <typename T>
    __global__ void rebuild(startree<T> star, const pair_type pair)
    {
        __shared__ distance_type new_sum;
        
        if(threadIdx.x == 0)
            new_sum = 0;

        __syncthreads();

        // Calculate the distances from the new OTU, being created from the join
        // of the two given OTUs, to all other unmodified OTUs.
        for(size_t i = threadIdx.x; i < star.count; i += blockDim.x) {
            const auto previous = star.matrix[{i, pair.x}] + star.matrix[{i, pair.y}];
            const auto current = .5 * (previous - star.matrix[pair]);

            star.matrix[{i, pair.x}] = star.matrix[{pair.x, i}] = current;
            star.cache[i] += current - previous;

            atomicAdd(&new_sum, current);
        }

        __syncthreads();

        // Updates the new OTU's cache to reflect its total distances sum and removes
        // one of the older OTUs from the star tree's sum cache.
        if(threadIdx.x == 0) {
            std::is_same<transform::symmetric, T>::value
                ? utils::swap(star.cache[pair.y], star.cache[0])
                : utils::swap(star.cache[pair.y], star.cache[star.count - 1]);
            star.cache[pair.x] = new_sum;
        }
    }

    /**
     * Joins an OTU pair into a new parent OTU.
     * @tparam T The star tree's matrix spatial transformation type.
     * @param phylotree The phylogenetic tree being constructed.
     * @param parent The parent OTU into which the pair will be joined.
     * @param star The OTU's star tree data structures.
     * @param join The OTU pair to join.
     */
    template <typename T>
    static void join_pair(tree& phylotree, oturef parent, startree<T>& star, const njoining::joinable& join)
    {
        using namespace cuda::device;

        const auto x = join.ref[0];
        const auto y = join.ref[1];

        // As updating the star tree is a computationally expensive task, we optimize
        // it by reusing one of the joined OTU's column and row on the matrix to
        // store the new OTU's distances.
        phylotree.join(parent, {star.map[x], join.delta[0]}, {star.map[y], join.delta[1]});

        // Let's calculate the distances between the OTU being created and the others
        // which have not been affected by the current joining operation.
        onlyslaves rebuild<<<1, max_threads(star.count)>>>(star, {x, y});

        // Finally, let's take advantage from our data structures' layouts and always remove
        // the cheapest column from our star tree's distance matrix.
        star.map[x] = parent;

        update_cache(star, y);
        --star.count;
    }

    /**
     * The hybrid neighbor-joining algorithm object. This algorithm uses hybrid
     * parallelism to run the Neighbor-Joining algorithm.
     * @tparam T The matrix spatial transformation to use within the algorithm.
     * @since 0.1.1
     */
    template <typename T>
    struct hybrid : public njoining::algorithm
    {
        /**
         * Builds the pseudo-phylogenetic tree from the given distance matrix.
         * @param matrix The distance matrix to build tree from.
         * @param count The total number of leaves in tree.
         * @return The calculated phylogenetic tree.
         */
        auto build_tree(startree<T>& star) const -> tree
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
            if (ctx.total < 2)
                return tree {};

            auto star = initialize<T>(ctx.matrix, ctx.total);
            auto result = build_tree(star);

            return result;
        }
    };
}

namespace msa
{
    /**
     * Instantiates a new hybrid neighbor-joining instance using a simple matrix.
     * @return The new algorithm instance.
     */
    extern auto phylogeny::njoining::hybrid_linear() -> phylogeny::algorithm *
    {
        return new ::hybrid<transform::linear<2>>;
    }

    /**
     * Instantiates a new hybrid neighbor-joining instance using a symmatrix.
     * @return The new algorithm instance.
     */
    extern auto phylogeny::njoining::hybrid_symmetric() -> phylogeny::algorithm *
    {
        return new ::hybrid<transform::symmetric>;
    }
}
