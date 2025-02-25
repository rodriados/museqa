/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file The types for managing attributes of a biological sequence.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#include <string>

#include <museqa/environment.h>
#include <museqa/memory/buffer.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace bio::sequence::attribute
{
    /**
     * A key-value pair to represent a sequence attribute. Attributes are optional
     * and may not be available in every sequence format file or parser.
     * @since 1.0
     */
    struct data_t
    {
        std::string key;
        std::string value;
    };

    /**
     * The structure responsible for grouping up all attributes of a sequence. As
     * sequence description is the only required attribute, it should be elevated
     * to a special position, making its access easier.
     * @since 1.0
     */
    using bag_t = memory::buffer_t<data_t>;
}

MUSEQA_END_NAMESPACE
