/** @file distribute.hpp
 * @brief Parallel Multiple Sequence Alignment pairwise distribute header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _DISTRIBUTE_HPP
#define _DISTRIBUTE_HPP

#include "fasta.hpp"

namespace pairwise
{
    extern void sync(const fasta_t&);
    extern void scatter();
    extern void clean();
}

#endif