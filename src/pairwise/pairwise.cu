/**
 * Multiple Sequence Alignment pairwise file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <cstdint>

#include "msa.hpp"
#include "fasta.hpp"
#include "pairwise/pairwise.cuh"

namespace pw = pairwise;

/**
 * Sets up the module and makes it ready to process.
 * @param fasta The fasta file to be processed.
 */
pw::Pairwise::Pairwise(const Fasta& fasta)
:   list(fasta)
{
    uint16_t size = this->list.getCount();
    this->score = new Score [size * size];
}

/**
 * Erases all data collected from processing.
 */
pw::Pairwise::~Pairwise() noexcept
{
    delete[] this->score;
}

/**
 * Manages the module processing, and produces the pairwise
 * module's results.
 */
void pw::Pairwise::process()
{
    /*__onlyslaves {
        this->distribute();
        this->loadblosum();
        this->run();
    }

    this->gather();*/
}
