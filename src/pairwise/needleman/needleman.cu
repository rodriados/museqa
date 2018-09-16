/**
 * Multiple Sequence Alignment needleman file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include "msa.hpp"
#include "device.cuh"
#include "pairwise/pairwise.hpp"
#include "pairwise/needleman.cuh"

namespace needleman = pairwise::needleman;

/**
 * Executes the needleman algorithm for the pairwise step. This method
 * is responsible for distributing and gathering workload from different
 * cluster nodes.
 */
void pairwise::Needleman::run()
{
    onlyslaves needleman::exec<<<1,1>>>({this->table});
    cudacall(cudaThreadSynchronize());
}

/** 
 * Performs the needleman sequence aligment algorithm in parallel.
 * @param in The input data requested by the algorithm.
 * @param out The output data produced by the algorithm.
 */
__launch_bounds__(pw_threads_per_block)
__global__ void needleman::exec(needleman::Input in /*, Buffer<Score> out*/)
{
    printf("from device = %d\n", in.table[4][4]);
}
