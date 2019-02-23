/**
 * Multiple Sequence Alignment needleman factory file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#include "pairwise/pairwise.cuh"
#include "pairwise/needleman.cuh"
#include "pairwise/needleman/hybrid.cuh"

/**
 * Calls the default algorithm for the current module.
 * @return The algorithm instance.
 */
pairwise::Algorithm *pairwise::needleman::factory()
{
    return pairwise::needleman::hybrid();
}
