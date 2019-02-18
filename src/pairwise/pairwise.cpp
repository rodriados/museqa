/**
 * Multiple Sequence Alignment pairwise file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#include <map>
#include <string>

#include "msa.hpp"
#include "exception.hpp"

#include "pairwise/pairwise.cuh"
#include "pairwise/needleman.cuh"

/*
 * Keeps the list of available algorithms and their respective factories.
 */
static const std::map<std::string, pairwise::Algorithm> dispatcher = {
    {"needleman",           pairwise::needleman::run}
,   {"needleman-hybrid",    pairwise::needleman::hybrid}
};

/**
 * Aligns every sequence in given database pairwise, thus calculating a similarity
 * score for every different permutation of sequence pairs. The algorithm is chosen
 * according to command line arguments.
 * @param config The module's configuration.
 */
void pairwise::Pairwise::run(const pairwise::Configuration& config)
{
    const auto& pair = dispatcher.find(config.algorithm);

    if(pair == dispatcher.end())
        throw Exception("unknown pairwise algorithm:", config.algorithm);

    onlymaster info("chosen pairwise algorithm:" s_bold, config.algorithm, s_reset);

    score = pair->second(config);
}
