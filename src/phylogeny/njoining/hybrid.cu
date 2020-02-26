/**
 * Multiple Sequence Alignment hybrid neighbor-joining file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019 Rodrigo Siqueira
 */
/*#include "msa.hpp"
#include "cuda.cuh"
#include "pointer.hpp"
#include "pairwise.cuh"
#include "cartesian.hpp"
#include "exception.hpp"

#include "phylogeny/tree.cuh"
#include "phylogeny/matrix.cuh"
#include "phylogeny/njoining.cuh"
#include "phylogeny/phylogeny.cuh"

using namespace phylogeny;

namespace
{
    JoinablePair findLocalPair(const ShrinkableMatrix<Score>& mat)
    {
        
    }*/

    /**
     * The hybrid neighbor-joining algorithm object. This algorithm uses
     * hybrid parallelism to run the Neighbor-Joining algorithm.
     * @since 0.1.1
     */
    //struct Hybrid : public NJoining
    //{
        /**
         * Executes the hybrid neighbor-joining algorithm for the phylogeny step.
         * This method is responsible for distributing and gathering workload
         * from different cluster nodes.
         * @param config The module's configuration.
         * @return The module's result value.
         */
        /*Tree run(const Configuration& config) override
        {
            auto mat = TriangularMatrix<Score>::fromPairwise(config.pw).toDevice();
            this->tree = Tree {config.pw.getCount()};

            //auto pair = this->synchronize(selectMinimum(mat));

            while(mat.getCount() > 3) {
                //this->tree.join(pair.id[0], pair.id[1]);
                //pair = rebuildAndNext(mat, pair);
            }

            return this->tree;
        }
    };
};*/

/**
 * Instantiates a new hybrid neighbor-joining instance.
 * @return The new algorithm instance.
 */
/*extern Algorithm *njoining::hybrid()
{
    return new Hybrid;
}*/