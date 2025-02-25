/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file A base class for all heuristic-related exception types.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/environment.h>
#include <museqa/exception.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace heuristic
{
    /**
     * Represents a generic heuristic-related exception. If needed, related exceptions
     * should inherit from this object in order to be caught and correctly treated.
     * @since 1.0
     */
    struct exception_t : public museqa::exception_t
    {
        using museqa::exception_t::exception_t;
    };
}

MUSEQA_END_NAMESPACE
