/**
 * Multiple Sequence Alignment pairwise file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#include <map>
#include <string>

#include "msa.hpp"
#include "buffer.hpp"
#include "exception.hpp"

#include "pairwise/pairwise.cuh"
#include "pairwise/needleman.cuh"
#include "pairwise/needleman/hybrid.cuh"
#include "pairwise/needleman/sequential.hpp"

/*
 * Keeps the list of available algorithms and their respective factories.
 */
static const std::map<std::string, pairwise::Factory> dispatcher = {
    {"needleman",               pairwise::needleman::factory}
,   {"needleman-sequential",    pairwise::needleman::sequential}
,   {"needleman-distributed",   pairwise::needleman::sequential}
,   {"needleman-hybrid",        pairwise::needleman::hybrid}
};

/**
 * Generates all working pairs for a given number of elements.
 * @param num The total number of elements.
 * @return The generated pairs.
 */
Buffer<pairwise::Pair> pairwise::Algorithm::generate(size_t num)
{
    pair = Buffer<pairwise::Pair> {(num * (num - 1)) / 2};

    for(size_t i = 0, c = 0; i < num - 1; ++i)
        for(size_t j = i + 1; j < num; ++j, ++c)
            pair[c] = {static_cast<uint16_t>(i), static_cast<uint16_t>(j)};

    return pair;
}

/**
 * Aligns every sequence in given database pairwise, thus calculating a similarity
 * score for every different permutation of sequence pairs.
 * @param config The module's configuration.
 */
void pairwise::Pairwise::run(const pairwise::Configuration& config)
{
    const auto& pair = dispatcher.find(config.algorithm);

    if(pair == dispatcher.end())
        throw Exception("unknown pairwise algorithm:", config.algorithm);

    onlymaster info("chosen pairwise algorithm:" s_bold, config.algorithm, s_reset);

    pairwise::Algorithm *algorithm = pair->second();
    *this = algorithm->run(config);

    delete algorithm;
}
