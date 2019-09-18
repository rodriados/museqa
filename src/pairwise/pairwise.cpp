/**
 * Multiple Sequence Alignment pairwise module file.
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

/*
 * Keeps the list of available algorithms and their respective factories.
 */
static const std::map<std::string, pairwise::Factory> dispatcher = {
    {"needleman",               pairwise::needleman::hybrid}
,   {"needleman-hybrid",        pairwise::needleman::hybrid}
,   {"needleman-sequential",    pairwise::needleman::sequential}
,   {"needleman-distributed",   pairwise::needleman::sequential}
};

/**
 * Generates all working pairs for a given number of elements.
 * @param num The total number of elements.
 * @return The generated pairs.
 */
Buffer<pairwise::Pair> pairwise::Algorithm::generate(size_t num)
{
    const size_t combinations = num * (num - 1) >> 1;
    pair = Buffer<pairwise::Pair> {combinations};

    for(size_t i = 0, c = 0; i < num - 1; ++i)
        for(size_t j = i + 1; j < num; ++j, ++c)
            pair[c] = {static_cast<SequenceRef>(i), static_cast<SequenceRef>(j)};

    return pair;
}

/**
 * Aligns every sequence in given database pairwise, thus calculating a similarity
 * score for every different permutation of sequence pairs.
 * @param config The module's configuration.
 */
void pairwise::Pairwise::run(const pairwise::Configuration& config)
{
    const auto& selection = dispatcher.find(config.algorithm);

    enforce(selection != dispatcher.end(), "unknown pairwise algorithm: %s", config.algorithm.c_str());
    onlymaster msa::info("chosen pairwise algorithm: %s", config.algorithm.c_str());

    pairwise::Algorithm *algorithm = selection->second();

    *this = algorithm->run(config);
    count = config.db.getCount();

    delete algorithm;
}
