/**
 * Multiple Sequence Alignment pairwise interface header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef PAIRWISE_HPP_INCLUDED
#define PAIRWISE_HPP_INCLUDED

#include "pairwise/pairwise.cuh"

/**
 * The score of a sequence pair alignment.
 * @since 0.1.1
 */
typedef pairwise::Score Score;

/**
 * The module execution manager.
 * @since 0.1.1
 */
typedef pairwise::Pairwise Pairwise;

#endif
