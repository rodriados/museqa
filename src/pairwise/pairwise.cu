/**
 * Multiple Sequence Alignment pairwise file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <cstdint>

#include "msa.hpp"
#include "fasta.hpp"
#include "pairwise/pairwise.hpp"
#include "pairwise/needleman.cuh"

/**
 * Sets up the module and makes it ready to process.
 * @param fasta The fasta file to be processed.
 */
pairwise::Pairwise::Pairwise(const Fasta& fasta)
:   list(fasta)
{
    uint16_t listCount = this->getCount();
    this->count = (listCount - 1) * listCount / 2; 

    onlyslaves {
        uint32_t div = this->count / (cluster::size - 1);
        uint32_t mod = this->count % (cluster::size - 1);

        this->count = div + (mod > cluster::rank - 1);
    }

    this->score = new pairwise::Score[this->count];
}

/**
 * Erases all data collected from processing.
 */
pairwise::Pairwise::~Pairwise() noexcept
{
    delete[] this->score;
}

/**
 * Manages the module processing, and produces the pairwise
 * module's results.
 * @param fasta The fasta file to be processed.
 */
pairwise::Pairwise pairwise::Pairwise::run(const Fasta& fasta)
{
    pairwise::Needleman needleman(fasta);

    onlymaster {
        needleman.generate();
    }
        
    needleman.scatter();

    onlyslaves {
        needleman.loadblosum();
        needleman.run();
    }

    needleman.gather();

    return pairwise::Pairwise();//Pairwise::Pairwise(needleman);
}
