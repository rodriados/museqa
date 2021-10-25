/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file A base class for all memory pointer exception types.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2021-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/environment.h>
#include <museqa/memory/exception.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace memory::pointer
{
    /**
     * Represents a generic exception related to memory pointers. As usual, exceptions
     * should inherit from this object in order to be caught and correctly treated.
     * @since 1.0
     */
    struct exception : public museqa::memory::exception
    {
        using museqa::memory::exception::exception;
    };
}

MUSEQA_END_NAMESPACE
