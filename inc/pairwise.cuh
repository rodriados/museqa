/**
 * Multiple Sequence Alignment pairwise module interface header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef PAIRWISE_CUH_INCLUDED
#define PAIRWISE_CUH_INCLUDED

#include <pairwise/pairwise.cuh>

/**
 * The score of a sequence pair alignment. Represents the score of an alignment of
 * a pair of sequences.
 * @since 0.1.1
 */
typedef pairwise::score score;

/**
 * Represents a reference for a sequence. This type is simply an index identification
 * for a sequence in the sequence database.
 * @since 0.1.1
 */
typedef pairwise::seqref seqref;

/**
 * A pair of sequence identifiers. This is the union pair object the pairwise module
 * processes. The sequence references can be accessed either by their respective
 * names or by their indeces.
 * @since 0.1.1
 */
typedef pairwise::pair pair;

#endif
