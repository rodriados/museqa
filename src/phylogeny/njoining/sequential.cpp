/**
 * Multiple Sequence Alignment sequential neighbor-joining file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019-2020 Rodrigo Siqueira
 */
#include <cstdint>

#include <msa.hpp>
#include <buffer.hpp>
#include <symmatrix.hpp>

#include <phylogeny/phylogeny.cuh>
#include <phylogeny/njoining.cuh>

namespace
{
    using namespace msa;
    using namespace phylogeny;

    /**
     * Creates a cache of the sum of the matrix's lines and columns. This cache
     * will be used whenever we need the sum of a whole line or column of the matrix.
     * @param matrix The distance matrix to build cache from.
     * @param count The total number of elements.
     * @return The newly built line sum cache.
     */
    static buffer<score> sum_cache(const symmatrix<score>& matrix, size_t count)
    {
        auto cache = buffer<score>::make(count);

        for(size_t i = 0; i < count - 1; ++i)
            for(size_t j = i + 1; j < count; ++j) {
                score current = matrix[{i, j}];
                cache[i] += current;
                cache[j] += current;
            }

        return cache;
    }

    /**
     * The sequential needleman algorithm object. This algorithm uses no
     * parallelism whatsoever.
     * @since 0.1.1
     */
    struct sequential : public njoining::algorithm
    {
        /**
         * Builds the pseudo-phylogenetic tree from the given distance matrix
         * @param matrix The distance matrix to build tree from.
         * @param count The total number of leaves in tree.
         * @return The calculated phylogenetic tree.
         */
        auto build_tree(symmatrix<score>& matrix, size_t count) -> tree
        {
            auto phylotree = tree::make(count);
            auto linecache = sum_cache(matrix, count);


            for(size_t i = count; i > 3; --i) {
                //njoining::joinpair chosen = reduce_q(dmatrix, linecache, mapping, i);
                //oturef parent = join_nodes(chosen, dmatrix, linecache, tree);
            }

            return phylotree;
        }

        /**
         * Executes the sequential neighbor-joining algorithm for the phylogeny
         * step. This method is responsible for coordinating the execution.
         * @return The module's result value.
         */
        auto run(const context& ctx) -> tree override
        {
            /*symmatrix<score>& matrix = this->inflate(ctx.matrix);

            auto phylotree = build_tree(matrix, ctx.nsequences);

            return phylotree;*/
        }
    };
}

namespace msa
{
    /**
     * Instantiates a new sequential neighbor-joining instance.
     * @return The new algorithm instance.
     */
    extern auto phylogeny::njoining::sequential() -> phylogeny::algorithm *
    {
        return new ::sequential;
    }
}
