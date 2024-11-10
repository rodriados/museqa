/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file The types for managing attributes of a biological sequence.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#include <map>
#include <string>

#include <museqa/environment.h>

MUSEQA_BEGIN_NAMESPACE

namespace bio::sequence
{
    /**
     * Enumerates all attributes that might be extracted from a sequence input file.
     * These attributes are not guaranteed to be known for every sequence nor to
     * parsed by every format reader when available.
     * @since 1.0
     */
    enum class attribute_t : uint8_t
    {
        description = 0x01
      , name
      , accession
      , locus
      , country
      , patent
      , database
      , entry
      , chain
      , application
      , number
      , quality
      , type
    };

    /**
     * The data structure responsible for holding all attributes of a sequence.
     * As sequence description is the only attribute required, it should be elevated
     * to a special position, making its access easier.
     * @since 1.0
     */
    using attribute_bag_t = std::map<attribute_t, std::string>;
}

MUSEQA_END_NAMESPACE
