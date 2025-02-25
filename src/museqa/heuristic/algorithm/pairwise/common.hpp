/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Common type definitions for the pairwise-alignment heuristic step.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <cstdint>

#include <museqa/environment.h>
#include <museqa/geometry/point.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace heuristic::algorithm::pairwise
{
    /**
     * The type of a score for a pairwise alignment between two sequences.
     * @since 1.0
     */
    using score_t = float;

    /**
     * The index-representation type for a collection of sequences.
     * @since 1.0
     */
    using seqref_t = int_least32_t;

    /**
     * Represents the reference to the alignment between two sequences.
     * @since 1.0
     */
    using pair_t = geometry::point_t<2, seqref_t>;
}

MUSEQA_END_NAMESPACE
