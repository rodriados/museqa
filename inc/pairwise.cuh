/**
 * Multiple Sequence Alignment pairwise module interface header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef PAIRWISE_CUH_INCLUDED
#define PAIRWISE_CUH_INCLUDED

#include "pairwise/pairwise.cuh"

/**
 * A pair of sequence identifiers.
 * @since 0.1.1
 */
typedef pairwise::Pair Pair;

/**
 * The module execution manager.
 * @since 0.1.1
 */
typedef pairwise::Pairwise Pairwise;

/**
 * The score of a sequence pair alignment.
 * @since 0.1.1
 */
typedef pairwise::Score Score;

/**
 * Represents a reference for a sequence.
 * @since 0.1.1
 */
typedef pairwise::SequenceRef SequenceRef;

#endif
