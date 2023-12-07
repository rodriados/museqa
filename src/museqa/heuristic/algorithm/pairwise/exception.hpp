/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file A base class for all pairwise algorithm-related exception types.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/environment.h>
#include <museqa/heuristic/exception.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace heuristic::algorithm::pairwise
{
    /**
     * Represents a generic pairwise-related exception. If needed, related exceptions
     * should inherit from this object in order to be caught and correctly treated.
     * @since 1.0
     */
    struct exception_t : public heuristic::exception_t
    {
        using heuristic::exception_t::exception_t;
    };
}

MUSEQA_END_NAMESPACE
