/**
 * Multiple Sequence Alignment parallel phylogeny pair-reducer file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019 Rodrigo Siqueira
 */
#include "mpi.hpp"
#include "pairwise.cuh"

#include "phylogeny/njoining.cuh"
#include "phylogeny/phylogeny.cuh"

using namespace phylogeny;

Pair NJoining::reduce(const Pair& target) const
{
    //auto fmpi = mpi::op::create<pairCompare>(true);
    return target;
}