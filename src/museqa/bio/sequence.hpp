/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file The representation of a biological sequence and surronding functions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2023-present Rodrigo Siqueira
 */
#pragma once

#include <map>
#include <string>

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

    namespace sequence
    {
        /**
         * Enumerates all attributes that might be extracted from a sequence source
         * file. These attributes may not be available to every sequence.
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
         * The structure containing all data available of a sequence, joining its
         * symbols' buffer to all optionally available attributes.
         * @since 1.0
         */
        struct data_t
        {
            sequence_t buffer;
            std::map<attribute_t, std::string> attributes;
        };
    }
}

MUSEQA_END_NAMESPACE
