/**
 * Multiple Sequence Alignment sequential neighbor-joining file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019 Rodrigo Siqueira
 */
#include <vector>
#include <cstdint>

#include <msa.hpp>
#include <buffer.hpp>

/*#include <phylogeny/tree.cuh>
#include <phylogeny/phylogeny.cuh>
#include <phylogeny/njoining.cuh>
*/
//using namespace phylogeny;

//namespace
//{
    /**
     * The sequential neighbor-joining algorithm object. This algorithm uses no
     * parallelism at all.
     * @since 0.1.1
     */
  //  struct sequential : public njoining::algorithm
    //{
        /**
         * Executes the sequential neighbor-joining algorithm for the phylogeny
         * step. This method executes it completely sequentially.
         * @param config The module's configuration.
         * @return The module's result value.
         */
      /*  auto run(const configuration& config) -> tree override
        {
            auto& mat = this->populate(config.pw);
            auto& nodes = this->leaves(config.pw.count());

            return tree {build_tree(nodes, mat)};
        }
    };
}
*/
/**
 * Instantiates a new sequential neighbor-joining instance.
 * @return The new algorithm instance.
 */
/*extern phylogeny::algorithm *njoining::sequential()
{
    return new ::sequential;
}


namespace
{*/
    /**
     * Initializes the cache of matrix's line sums.
     * @param matrix The matrix from which sums will be taken from.
     * @return The cache containing the corresponding sums.
     */
    /*SumCache initCache(const PhyloMatrix& matrix)
    {
        const size_t count = matrix.getCount();
        SumCache cache {Pointer<Score> {new Score[count]()}, count};

        for(size_t i = 0; i < count - 1; ++i)
            for(size_t j = i + 1; j < count; ++j) {
                auto current = matrix(i, j);
                cache[i] += current;
                cache[j] += current;
            }

        return cache;
    }*/

    /**
     * Finds a pair of OTUs to join.
     * @param matrix The distance matrix to look OTUs up.
     * @param cache The buffer with the current line sums of the matrix.
     * @return A candidate element to join in tree next.
     */
    /*JoinablePair findToJoin(const PhyloMatrix& matrix, const SumCache& cache)
    {
        const size_t count = matrix.getCount();
        const auto off = matrix.getOffset();
        JoinablePair selected = {};

        for(size_t i = 0; i < count - 1; ++i)
            for(size_t j = i + 1; j < count; ++j) {
                Score q = (count - 2) * matrix(i, j) - cache[off[i]] - cache[off[j]];

                if(selected.score > q) {
                    selected.ref[0] = i;
                    selected.ref[1] = j;
                    selected.score = q;
                }
            }

        return selected;
    }

    void rebuildMatrix(PhyloMatrix& matrix, SumCache& cache, const JoinablePair& selected)
    {
        const size_t count = matrix.getCount();
        const auto off = matrix.getOffset();
        const auto dval = matrix(selected.ref[0], selected.ref[1]);
        Score linesum = 0;

        for(size_t i = 0; i < count; ++i) {
            if(i != selected.ref[1]) {
                cache[off[i]] -= matrix(selected.ref[0], i) + matrix(selected.ref[1], i);
                auto value = matrix(selected.ref[0], i)
                    = matrix(selected.ref[0], i) + matrix(selected.ref[1], i) - dval;
                cache[off[i]] += value;
                linesum += value;
            }
        }

        cache[off[selected.ref[0]]] = linesum;
        matrix.removeOffset(selected.ref[1]);
    }*/

    /**
     * The sequential neighbor-joining algorithm object. This object executes the
     * sequential version of the Neighbor-Joining algorithm.
     * @since 0.1.1
     */
    //struct Sequential : public NJoining
    //{
        /**
         * Builds the pseudo-phylogenetic tree from the given matrix and sums.
         * @param matrix The distance matrix between OTUs to be joined.
         * @param cache The buffer with the current line sums of the matrix.
         * @return The constructed tree from given matrix.
         */
        /*Tree buildTree(PhyloMatrix& matrix, SumCache& cache)
        {
            while(matrix.getCount() > 3) {
                const auto off = matrix.getOffset();
                auto selected = findToJoin(matrix, cache);
                this->tree.join(off[selected.ref[0]], off[selected.ref[1]]);
                rebuildMatrix(matrix, cache, selected);
            }

            return this->tree;
        }*/

        /**
         * Executes the neighbor-joining algorithm sequentially for the
         * phylogeny step.
         * @param config The module's configuration.
         * @return The module's result value.
         */
        /*Tree run(const Configuration& config) override
        {
            this->tree = Tree {config.pw.getCount()};
            this->nodes = 1;

            auto matrix = PhyloMatrix::fromPairwise(config.pw);
            auto lineCache = initCache(matrix);

            onlymaster msa::task("phylogeny", "joining %llu sequences", config.pw.getCount());
            return buildTree(matrix, lineCache);
        }
    };*/
//}

/**
 * Instantiates a new sequential neighbor-joining instance.
 * @return The new algorithm instance.
 */
/*extern Algorithm *njoining::sequential()
{
    return new Sequential;
}*/