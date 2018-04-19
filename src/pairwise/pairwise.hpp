/** @file pairwise.hpp
 * @brief Parallel Multiple Sequence Alignment pairwise header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _PAIRWISE_HPP
#define _PAIRWISE_HPP

/** @struct position_t
 * @brief Informs how to access a sequence from a continuous char-pointer.
 * @var offset Indicates the sequence offset to the pointer.
 * @var length Indicates how big is the sequence.
 */
typedef struct {
    unsigned offset;
    unsigned length;
} position_t;

/** @struct workpair_t
 * @brief Indicates a pair of sequences to be aligned.
 * @var seq The pair of sequences to align.
 */
typedef struct {
    short seq[2];
} workpair_t;

/** @struct score_t
 * @brief Stores score information about a sequence pair.
 * @var cached The cached score value for a sequence pair.
 * @var matches The number of matches in the pair.
 * @var mismatches The number of mismatches in the pair.
 * @var gaps The number of gaps in the pair.
 */
typedef struct {
    short cached;
    unsigned short matches;
    unsigned short mismatches;
    unsigned short gaps;
} score_t;

/** @struct pairwise_t
 * @brief Stores data and structures needed for executing pairwise algorithm.
 * @var data The pointer to character sequences.
 * @var nseq The number of sequences loaded from file.
 * @var npair The number of working pairs received to process.
 * @var seq The sequences' positions.
 * @var pair The working pairs to process.
 */
typedef struct {
    char *data;
    short nseq;
    unsigned npair;
    position_t *seq;
    workpair_t *pair;
} pairwise_t;

namespace pairwise
{
    extern void prepare();
    extern score_t *pairwise();
    extern void clean();
}

#endif