/**
 * Multiple Sequence Alignment pairwise needleman algorithm main file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <vector>

#include "buffer.hpp"

#include "pairwise/pairwise.cuh"
#include "pairwise/needleman.cuh"

using namespace pairwise;

/**
 * Generates all working pairs for a given number of elements.
 * @param num The total number of elements.
 * @return The generated pairs.
 */
Buffer<Pair> needleman::generate(size_t num)
{
    Buffer<Pair> pairs {(num * (num - 1)) / 2};

    for(size_t i = 0, c = 0; i < num - 1; ++i)
        for(size_t j = i + 1; j < num; ++j, ++c)
            pairs[c] = {static_cast<uint16_t>(i), static_cast<uint16_t>(j)};

    return pairs;
}
