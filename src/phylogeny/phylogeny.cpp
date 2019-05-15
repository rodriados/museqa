/**
 * Multiple Sequence Alignment phylogeny module file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019 Rodrigo Siqueira
 */
#include <map>

#include "msa.hpp"
#include "exception.hpp"

#include "phylogeny/phylogeny.cuh"
#include "phylogeny/njoining.cuh"

/*
 * Keeps the list of available algorithms and their respective factories.
 */
static const std::map<std::string, phylogeny::Factory> dispatcher = {
    {"njoining",               phylogeny::njoining::hybrid}
,   {"njoining-hybrid",        phylogeny::njoining::hybrid}
};

/**
 * Executes a phylogeny algorithm, transforming the distance matrix between
 * the sequences into a pseudo-phylogenitic tree.
 * @param config The module's configuration.
 */
void phylogeny::Phylogeny::run(const phylogeny::Configuration& config)
{
    const auto& selection = dispatcher.find(config.algorithm);

    if(selection == dispatcher.end())
        throw Exception("unknown phylogeny algorithm:", config.algorithm);

    onlymaster info("chosen phylogeny algorithm:" s_bold, config.algorithm, s_reset);

    phylogeny::Algorithm *algorithm = selection->second();

    *this = algorithm->run(config);

    delete algorithm;
}