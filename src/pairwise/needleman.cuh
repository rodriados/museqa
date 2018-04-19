/** @file needleman.cuh
 * @brief Parallel Multiple Sequence Alignment needleman header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _NEEDLEMAN_CUH
#define _NEEDLEMAN_CUH

#include "pairwise.hpp"

namespace pairwise
{
    extern __global__
    void needleman(char *, position_t *, workpair_t *, score_t *);
}

#endif