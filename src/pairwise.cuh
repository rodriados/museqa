/** @file pairwise.cuh
 * @brief Parallel Multiple Sequence Alignment pairwise header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _PAIRWISE_CUH
#define _PAIRWISE_CUH

#include <cstdint>
#include <vector>

#include "fasta.hpp"

/** @struct position_t
 * @brief Informs how to access a sequence from a continuous char-pointer.
 * @var offset Indicates the sequence offset to the pointer.
 * @var length Indicates how big is the sequence.
 */
typedef struct {
    uint32_t offset;
    uint32_t length;
} position_t;

/** @struct workpair_t
 * @brief Indicates a pair of sequences to be aligned.
 * @var seq The pair of sequences to align.
 */
typedef struct {
    uint16_t seq[2];
} workpair_t;

/** @struct score_t
 * @brief Stores score information about a sequence pair.
 * @var cached The cached score value for a sequence pair.
 * @var matches The number of matches in the pair.
 * @var mismatches The number of mismatches in the pair.
 * @var gaps The number of gaps in the pair.
 */
typedef struct {
    int32_t cached;
    uint16_t matches;
    uint16_t mismatches;
    uint16_t gaps;
} score_t;

/** @class needleman_t
 * @brief Groups all data required for needleman execution.
 * @var seqchar The pointer to character sequences.
 * @var nseq The number of sequences loaded from file.
 * @var npair The number of working pairs received to process.
 * @var seq The sequences' positions.
 * @var pair The working pairs to process.
 */
class needleman_t
{
public:
    char *seqchar;
    int8_t *table;
    uint16_t nseq;
    uint32_t npair;
    position_t *seq;
    workpair_t *pair;

public:
    void alloc(std::vector<uint32_t>&);
    void free();
};

/** @class pairwise_t
 * @brief Stores data and structures needed for executing pairwise algorithm.
 * @var seqchar The pointer to character sequences.
 * @var nseq The number of sequences loaded from file.
 * @var npair The number of working pairs received to process.
 * @var seq The sequences' positions.
 * @var pair The working pairs to process.
 * @var score The score of each working pair processed.
 */
class pairwise_t
{
public:
    char *seqchar;
    uint16_t nseq;
    uint32_t npair;
    position_t *seq;
    workpair_t *pair;
    score_t *score;

public:
    pairwise_t();
    ~pairwise_t();

    void load(const fasta_t *);
    void pairwise();

private:
    void scatter();
    bool filter(bool[], std::vector<uint32_t>&);
    void blosum(needleman_t&);
    void run(needleman_t&);
};

namespace pairwise
{
#ifdef __CUDACC__
    extern __global__ void needleman(needleman_t, score_t *);
#endif
}

#endif