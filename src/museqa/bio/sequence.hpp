/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file The representation of a biological sequence and surronding functions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2023-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/environment.h>
#include <museqa/bio/sequence/buffer.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace bio
{
    /**
     * The buffer type for a biological sequence. Internally, sequence symbols are
     * stored compressed within blocks that are easily accessible and decompressible.
     * @since 1.0
     */
    using sequence_t = sequence::buffer_t;
}

MUSEQA_END_NAMESPACE
