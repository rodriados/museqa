/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Score for sequence alignments and symbol matches or mismatches.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/environment.h>

MUSEQA_BEGIN_NAMESPACE

namespace bio::scoring
{
    /**
     * Alignment score between two biological sequences. This is the type from which
     * all other scoring related types derive from. The size and precision of the
     * score type relates directly to the memory requirements and quality of the
     * algorithms implemented upon its dependency.
     * @since 1.0
     */
  #ifndef MUSEQA_ENABLE_SCORE_AS_DOUBLE
    using score_t = float;
  #else
    using score_t = double;
  #endif
}

MUSEQA_END_NAMESPACE
