/**
 * Multiple Sequence Alignment pairwise file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <cstdint>

#include "msa.hpp"
#include "fasta.hpp"
#include "device.cuh"

#include "pairwise/pairwise.hpp"
#include "pairwise/needleman.cuh"

/**
 * Manages the module processing, and produces the pairwise
 * module's results.
 * @param fasta The fasta file to be processed.
 */
pairwise::Pairwise::Pairwise(const Fasta& fasta)
:   list(fasta)
{
    pairwise::Algorithm *algorithm;

    algorithm = new Needleman(this->list, this->score);

    algorithm->scatter();
    onlyslaves algorithm->run();  
    algorithm->gather();

    delete algorithm;
}

/**
 * Generates all workpairs to be processed. This method runs
 * only on the master node.
 */
void pairwise::Algorithm::generate()
{
    for(uint16_t i = 0, n = this->list.getCount(); i < n; ++i)
        for(uint16_t j = i + 1; j < n; ++j)
            this->pair.push_back({i, j});

    info("generated %d sequence pairs", this->pair.size());
}
